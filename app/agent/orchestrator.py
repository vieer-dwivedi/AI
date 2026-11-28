"""The tool-use loop with policy checks (Step 9).

Approval-gated calls are parked and reported; Step 10 adds persistence and resume.
"""
import json
import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from app.agent import prompts
from app.agent.policy import Decision, evaluate_with_reason
from app.agent.schemas import CaseState, ResolutionPlan
from app.config import settings
from app.llm import as_text, get_model
from app.observability import tracing
from app.observability.cost import track_usage
from app.observability.tracing import Tracer
from app.tools import carrier_server, comms_server, erp_server, kb_server

READ_TOOLS = [
    erp_server.get_order, erp_server.get_customer, erp_server.check_inventory,
    erp_server.calculate_penalty, carrier_server.get_shipment_status,
    carrier_server.get_freight_quote, kb_server.search_contracts, kb_server.search_playbook,
]
WRITE_TOOLS = [
    comms_server.draft_customer_email, comms_server.add_case_note, comms_server.update_order_status,
    comms_server.send_email, comms_server.create_reship_order,
    comms_server.book_expedited_freight, comms_server.issue_credit_note,
]
REGISTRY = {f.__name__: f for f in READ_TOOLS + WRITE_TOOLS}
READ_NAMES = {f.__name__ for f in READ_TOOLS}
CASE_ID_TOOLS = {"draft_customer_email", "add_case_note"}
EXCEPTION_TYPES = ["delay", "damage", "stockout", "customs", "dispute", "other"]
MAX_TOOL_RESULT_CHARS = 2000

SUBMIT = StructuredTool.from_function(
    func=lambda **kwargs: "submitted",
    name="submit_resolution",
    description="Submit the final resolution plan. Call exactly once, when the investigation is done.",
    args_schema=ResolutionPlan,
)


@dataclass
class CaseResult:
    case: CaseState
    plan: ResolutionPlan | None = None
    pending: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    messages: list = field(default_factory=list)
    stop_reason: str = ""


def triage(email: dict, case: CaseState | None = None, tracer: Tracer | None = None) -> str:
    """One cheap call to label the exception type. Falls back to 'other'. Its tokens count toward the case."""
    try:
        t0 = time.perf_counter()
        ai = get_model("triage", temperature=0).invoke(prompts.triage_prompt(email))
        if case is not None:
            usage = track_usage(case, ai)
            if tracer:
                tracer.event("llm_call", phase="triage", duration_s=round(time.perf_counter() - t0, 2), **usage)
        reply = as_text(ai.content).lower()
        for t in EXCEPTION_TYPES:
            if t in reply:
                return t
    except Exception:
        pass
    return "other"


def _budget_exceeded(case: CaseState) -> str | None:
    if case.cost_usd >= settings.max_cost_usd_per_case:
        return f"cost budget exceeded (${case.cost_usd:.2f})"
    if case.tokens_in + case.tokens_out >= settings.max_tokens_per_case:
        return f"token budget exceeded ({case.tokens_in + case.tokens_out} tokens)"
    return None


def _execute(name: str, args: dict):
    try:
        return REGISTRY[name](**args)
    except TypeError as exc:
        return {"error": "BAD_ARGUMENTS", "detail": str(exc)}
    except Exception as exc:
        return {"error": "TOOL_FAILED", "detail": f"{type(exc).__name__}: {exc}"}


def execute_tool(name: str, args: dict):
    """Run a registered tool (used for approved actions). Policy was already satisfied by a human."""
    if name not in REGISTRY or name in READ_NAMES:
        return {"error": "UNKNOWN_WRITE_TOOL", "tool": name}
    return _execute(name, args)


def _truncate(text: str) -> str:
    return text if len(text) <= MAX_TOOL_RESULT_CHARS else text[:MAX_TOOL_RESULT_CHARS] + "...[truncated]"


def _handle_call(call: dict, case: CaseState, result: CaseResult, log, tracer: Tracer) -> str:
    name, args = call["name"], dict(call["args"])

    if name == "submit_resolution":
        args["case_id"] = case.case_id
        try:
            result.plan = ResolutionPlan.model_validate(args)
        except ValidationError as exc:
            log(f"  x plan rejected ({exc.error_count()} validation errors)")
            tracer.event("plan_rejected", step=case.steps, errors=exc.error_count())
            return f"PLAN_REJECTED. Fix these problems and call submit_resolution again: {exc}"
        log("  ok plan accepted")
        tracer.event("plan_submitted", step=case.steps, confidence=result.plan.confidence,
                     recommended=result.plan.options[result.plan.recommended_option].name)
        return "Plan accepted."

    if name in CASE_ID_TOOLS:
        args["case_id"] = case.case_id

    if name in READ_NAMES:
        decision, reason = Decision.ALLOW, "read-only"
    else:
        decision, reason = evaluate_with_reason(name, args, case)   # unknown tools are denied here
    log(f"  -> {name}({args}) [{decision.value}: {reason}]")

    if decision == Decision.DENY:
        out = {"error": "DENIED", "reason": reason}
    elif decision == Decision.NEEDS_APPROVAL:
        result.pending.append({"tool": name, "args": args, "reason": reason})
        out = {"status": "PENDING_APPROVAL",
               "detail": "A human must approve this action. Do not retry it. Mention it in your plan."}
    else:
        with tracing.span("tool.call", tool=name):
            out = _execute(name, args)
        if name == "get_customer" and isinstance(out, dict) and "tier" in out:
            case.customer_tier = out["tier"]          # the policy needs the tier for send_email

    result.trace.append({"step": case.steps, "tool": name, "args": args,
                         "decision": decision.value, "result": out})
    text = _truncate(json.dumps(out, default=str, separators=(",", ":")))
    tracer.event("tool_call", step=case.steps, tool=name, args=args, decision=decision.value,
                 reason=reason, result=text[:800])
    return text


def run_case(email: dict, case_id: str = "CASE-1", log=print) -> CaseResult:
    with tracing.span("agent.loop", case_id=case_id):
        return _run_case(email, case_id, log)


def _run_case(email: dict, case_id: str, log) -> CaseResult:
    tracer = Tracer(case_id, reset=True)
    tracer.event("case_start", subject=email.get("subject", ""))
    case = CaseState(case_id=case_id)
    result = CaseResult(case=case)
    exception_type = triage(email, case, tracer)
    log(f"Triage: {exception_type}")

    model = get_model(temperature=0).bind_tools(READ_TOOLS + WRITE_TOOLS + [SUBMIT])
    messages = [SystemMessage(prompts.SYSTEM_PROMPT),
                HumanMessage(prompts.first_message(email, case_id, exception_type))]
    nudges = 0

    for step in range(settings.max_steps):
        reason = _budget_exceeded(case)
        if reason:                                   # stop and escalate to a human
            result.stop_reason = reason
            log(f"  ! {reason}, escalating")
            tracer.event("budget_exceeded", step=step + 1, reason=reason)
            break
        case.steps = step + 1
        log(f"[step {case.steps}]")
        t0 = time.perf_counter()
        with tracing.span("llm.call", step=case.steps):
            ai = model.invoke(messages)
        took = time.perf_counter() - t0
        log(f"  (model took {took:.0f}s)")
        usage = track_usage(case, ai)
        tracer.event("llm_call", phase="agent", step=case.steps, duration_s=round(took, 2),
                     tool_calls=[c["name"] for c in ai.tool_calls], **usage)
        messages.append(ai)

        if not ai.tool_calls:
            if nudges < 2:                           # model talked instead of finishing
                nudges += 1
                messages.append(HumanMessage("Call submit_resolution now with your plan."))
                continue
            result.stop_reason = "model stopped without a plan"
            break

        for call in ai.tool_calls:
            out = _handle_call(call, case, result, log, tracer)
            messages.append(ToolMessage(content=out, tool_call_id=call["id"]))

        if result.plan:
            result.stop_reason = "plan submitted"
            break
    else:
        result.stop_reason = "step budget exhausted"

    result.messages = messages
    tracer.event("case_end", stop_reason=result.stop_reason, steps=case.steps, tokens_in=case.tokens_in,
                 tokens_out=case.tokens_out, cost_usd=round(case.cost_usd, 6))
    return result
