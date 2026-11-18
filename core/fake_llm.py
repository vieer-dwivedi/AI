# =====================================================================
# core/fake_llm.py
# In simple words: a pretend "LLM" that needs no internet and no API key.
# It is NOT random and NOT a canned story: it behaves like a tiny
# rule-based engineer. Each turn it looks at what the tools already
# returned (alarms, log analysis, deployments...) and decides the next
# tool call, exactly like a real model would in the tool-calling loop.
# This lets us test the WHOLE pipeline (agents, tools, memory, graph, API)
# offline. Switch LLM_MODEL in .env to a real model for real reasoning.
#
# How it knows which agent it is playing: each agent's system prompt
# contains a marker like [ROLE:triage] / [ROLE:diagnosis] / [ROLE:remediation].
# =====================================================================
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _text(content: Any) -> str:
    """Message content can be a string or a list of blocks - make it a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def _call(name: str, **args: Any) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}


def _ai(calls: list[dict] | None = None, text: str = "") -> AIMessage:
    return AIMessage(content=text, tool_calls=calls or [])


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class _View:
    """A readable summary of the conversation so far."""

    def __init__(self, messages: list[BaseMessage]):
        self.system = "\n".join(_text(m.content) for m in messages if isinstance(m, SystemMessage))
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        self.task = _text(humans[0].content) if humans else ""
        # "key: value" lines of the task text (incident_id, service, log_group, ...)
        self.fields: dict[str, str] = dict(re.findall(r"(?m)^([a-z_]+):[ \t]*(.+)$", self.task))
        self.calls: list[tuple[str, dict]] = []
        self.results: dict[str, list[str]] = {}
        id_to_name: dict[str, str] = {}
        for m in messages:
            if isinstance(m, AIMessage):
                for c in m.tool_calls:
                    self.calls.append((c["name"], c["args"]))
                    id_to_name[c["id"]] = c["name"]
            elif isinstance(m, ToolMessage):
                name = m.name or id_to_name.get(m.tool_call_id, "")
                self.results.setdefault(name, []).append(_text(m.content))

    def called(self, name: str, **match: Any) -> bool:
        return any(n == name and all(a.get(k) == v for k, v in match.items())
                   for n, a in self.calls)

    def json(self, name: str, idx: int = -1) -> dict:
        try:
            return json.loads(self.results.get(name, [""])[idx])
        except (ValueError, IndexError):
            return {}

    def all_json(self, name: str) -> list[dict]:
        out = []
        for r in self.results.get(name, []):
            try:
                out.append(json.loads(r))
            except ValueError:
                pass
        return out


# ---------------------------------------------------------------------
# TRIAGE: look at alarms + logs, then hand in an impact assessment.
# ---------------------------------------------------------------------
def _triage(v: _View) -> AIMessage:
    f = v.fields
    if not v.called("analyze_logs"):
        calls = [_call("describe_cloudwatch_alarms", state="ALARM")]
        if f.get("log_group"):
            calls.append(_call("analyze_logs", log_group=f["log_group"], minutes=60))
        return _ai(calls, "Checking the firing alarms and the service logs first.")

    if not v.called("submit_triage"):
        logs = v.json("analyze_logs")
        total = logs.get("total_events", 0) or 1
        lc = logs.get("level_counts", {})
        errors = lc.get("ERROR", 0) + lc.get("FATAL", 0) + lc.get("CRITICAL", 0)
        rate = round(100 * errors / total, 1)
        service = f.get("service", "")
        revenue = any(k in service for k in ("checkout", "payment", "order"))
        impact = "partial_outage" if rate >= 15 else ("degraded" if errors else "none")
        top = (logs.get("top_error_messages") or [{}])[0].get("pattern", "no errors found")
        summary = (f"{service}: {errors} error lines out of {total} ({rate}%), mainly '{top}'. "
                   f"Alarm {f.get('alarm', 'n/a')} is firing.")
        return _ai([_call("submit_triage", incident_id=f.get("incident_id", ""),
                          user_impact=impact, summary=summary, affected_service=service,
                          environment=f.get("environment", "prod"),
                          error_rate_pct=rate, revenue_path=revenue)])
    return _ai(text="Triage done: severity recorded.")


# ---------------------------------------------------------------------
# DIAGNOSIS: logs -> suspected cause -> verify with the right tools ->
#            find what changed -> hand in root cause.
# ---------------------------------------------------------------------
_CAUSES = {
    "out_of_memory": ("memory_exhaustion", "ECS container OOMKilled exit code 137 out of memory"),
    "db_connection_exhaustion": ("connection_exhaustion",
                                 "postgres connection slots exhausted too many connections"),
    "disk_full": ("disk_exhaustion", "disk full no space left on device log growth"),
    "null_pointer_exception": ("bad_deployment", "NullPointerException after deployment rollback"),
    "throttling": ("throttling", "lambda throttling TooManyRequestsException"),
    "certificate_problem": ("dependency_failure", "tls certificate expired"),
    "dns_failure": ("dependency_failure", "dns resolution failure"),
}


def _diagnosis(v: _View) -> AIMessage:
    f = v.fields
    iid, svc = f.get("incident_id", ""), f.get("service", "")
    lg, cluster = f.get("log_group", ""), f.get("ecs_cluster", "prod-cluster")
    db, inst = f.get("db_instance", ""), f.get("instance_id", "")

    # Phase A: first look - logs, recent changes, alarms, similar past incidents.
    if not v.called("analyze_logs"):
        calls = [_call("list_recent_deployments", service=svc),
                 _call("describe_cloudwatch_alarms", state="ALARM"),
                 _call("search_past_incidents", query=f.get("title", svc))]
        if lg:
            calls.insert(0, _call("analyze_logs", log_group=lg, minutes=60))
        return _ai(calls, "Starting with logs, recent changes and past incidents.")

    logs = v.json("analyze_logs")
    cause = next((s for s in logs.get("signatures", []) if s.get("kind") == "cause"), None)
    cname = cause["name"] if cause else None
    category, rq = _CAUSES.get(cname, ("other", f"{svc} errors troubleshooting"))

    # The alert did not name the database? Discover it from the account's log groups.
    if cname == "db_connection_exhaustion" and not db:
        if not v.called("list_log_groups"):
            return _ai([_call("list_log_groups")], "Which database is involved? Listing log groups.")
        m = re.search(r"/aws/rds/instance/([\w-]+)/postgresql", " ".join(v.results.get("list_log_groups", [])))
        db = m.group(1) if m else ""

    # Phase B: verify the suspected cause with the matching tools + runbook.
    if not v.called("search_runbooks"):
        calls = [_call("search_runbooks", query=rq)]
        if cname == "out_of_memory":
            calls += [_call("get_cloudwatch_metrics", namespace="AWS/ECS", metric_name="MemoryUtilization",
                            dimensions_json=json.dumps({"ClusterName": cluster, "ServiceName": svc}),
                            minutes=60),
                      _call("describe_ecs_service", cluster=cluster, service=svc)]
        elif cname == "db_connection_exhaustion" and db:
            calls += [_call("describe_rds_instance", db_instance_id=db),
                      _call("get_cloudwatch_metrics", namespace="AWS/RDS", metric_name="DatabaseConnections",
                            dimensions_json=json.dumps({"DBInstanceIdentifier": db}), minutes=60),
                      _call("get_cloudwatch_logs", log_group=f"/aws/rds/instance/{db}/postgresql",
                            filter_pattern="connection authorized", minutes=60, max_events=10)]
        elif cname == "disk_full":
            if inst:
                calls.append(_call("describe_ec2_instance", instance_id=inst))
            calls.append(_call("get_cloudwatch_logs", log_group=lg, filter_pattern="DEBUG",
                               minutes=60, max_events=3))
        elif cname == "null_pointer_exception":
            calls += [_call("get_cloudwatch_logs", log_group=lg, filter_pattern="NullPointerException",
                            minutes=60, max_events=3),
                      _call("describe_ecs_service", cluster=cluster, service=svc)]
        return _ai(calls, f"Suspected cause: {cname or 'unknown'}. Verifying.")

    # Phase C: the DB case - find WHICH client opened the connections, check ITS deployments.
    app = None
    if cname == "db_connection_exhaustion":
        m = re.search(r"application_name=([\w.-]+)", " ".join(v.results.get("get_cloudwatch_logs", [])))
        app = m.group(1) if m else None
        if app and not v.called("list_recent_deployments", service=app):
            return _ai([_call("list_recent_deployments", service=app)],
                       f"Connections come from '{app}'. Checking its recent changes.")

    # Phase D: collect the facts and hand in the diagnosis.
    if not v.called("submit_diagnosis"):
        first_err = logs.get("first_error_time")
        deps = [d for r in v.all_json("list_recent_deployments") for d in r.get("deployments", [])]
        before = [d for d in deps if not first_err or _ts(d["createdAt"]) <= _ts(first_err)]
        dep = max(before, key=lambda d: _ts(d["createdAt"])) if before else None
        trigger = (f"deployment {dep['deploymentId']} of {dep['service']} at {dep['createdAt']}: "
                   f"{dep['description']}") if dep else "no recent change found"

        ev = [f"Logs {lg}: {logs.get('total_events', 0)} events, first error {first_err}, "
              f"spike start {logs.get('error_spike_start')}."]
        if cause:
            ev.append(f"Signature '{cname}' matched {cause['count']}x: {cause['hint']}")
        for e in (logs.get("top_error_messages") or [])[:2]:
            ev.append(f"Top error x{e['count']}: {e['pattern']}")
        for mj in v.all_json("get_cloudwatch_metrics"):
            if "max" in mj:
                ev.append(f"{mj['metric']}: max {mj['max']}, latest {mj['latest']}, trend {mj['trend']}, "
                          f"peak {mj['peak_time']}.")
        for sj in v.all_json("describe_ecs_service"):
            ev.append(f"ECS {sj.get('service')}: desired {sj.get('desired')}, running {sj.get('running')}.")
        if dep:
            ev.append(f"Change before the incident: {trigger}")
        sim = [s for r in v.all_json("search_past_incidents") for s in r.get("similar_incidents", [])
               if s.get("relevance", 0) >= 0.3]
        if sim:
            ev.append(f"Similar past incident {sim[0]['id']}: {sim[0]['title']} (fix: {sim[0]['resolution']})")
        rb = (v.json("search_runbooks").get("results") or [{}])[0].get("file", "")

        why = {
            "memory_exhaustion": f"{svc} tasks are being OOM-killed (heap/container memory exhausted), "
                                 f"starting right after the {trigger}.",
            "connection_exhaustion": f"Database {db} ran out of connection slots. '{app or 'a new client'}' "
                                     f"opens connections without pooling ({trigger}), starving {svc}.",
            "disk_exhaustion": f"The disk on {inst or svc} is full: log output exploded after the {trigger}.",
            "bad_deployment": f"The latest release of {svc} throws NullPointerException on requests, "
                              f"starting right after the {trigger}.",
        }.get(category, f"Could not determine a root cause for {svc}; manual investigation needed.")
        conf = 0.9 if (cause and dep) else 0.6 if cause else 0.2
        return _ai([
            _call("record_finding", incident_id=iid, agent="diagnosis", finding=f"Trigger: {trigger}"),
            _call("submit_diagnosis", incident_id=iid, root_cause=why, category=category, confidence=conf,
                  evidence="\n".join(ev), affected_service=svc, trigger=trigger, runbook=rb)])
    return _ai(text="Diagnosis done: root cause recorded.")


# ---------------------------------------------------------------------
# REMEDIATION: read the diagnosis, look up the runbook, PROPOSE fixes.
# ---------------------------------------------------------------------
def _remediation(v: _View) -> AIMessage:
    f = v.fields
    iid, svc, cat = f.get("incident_id", ""), f.get("service", ""), f.get("category", "other")
    cluster, inst = f.get("ecs_cluster", "prod-cluster"), f.get("instance_id", "")
    db = f.get("db_instance") or (re.search(r"Database ([\w-]+) ran out", f.get("root_cause", "")) or [None, "db"])[1]

    if not v.called("search_runbooks"):
        calls = [_call("search_runbooks", query=f.get("root_cause", svc)[:200])]
        if cat in ("memory_exhaustion", "bad_deployment"):
            calls.append(_call("describe_ecs_service", cluster=cluster, service=svc))
        if cat == "disk_exhaustion" and inst:
            calls.append(_call("describe_ec2_instance", instance_id=inst))
        return _ai(calls, "Reading the runbook and current service state.")

    if not v.called("propose_remediation"):
        def prop(action: str, target: str, why: str, risk: str, steps: str) -> dict:
            return _call("propose_remediation", incident_id=iid, action_type=action, target=target,
                         rationale=why, risk=risk, steps=steps)

        ecs = v.json("describe_ecs_service")
        deps = ecs.get("deployments", [])
        prev_td = deps[1]["task_definition"] if len(deps) > 1 else "<previous-task-definition>"
        tgt = f"ecs:{cluster}/{svc}"
        vol = (re.search(r"vol-[0-9a-f]+", " ".join(v.results.get("describe_ec2_instance", []))) or [None])[0]
        calls: list[dict] = []
        if cat in ("memory_exhaustion", "bad_deployment"):
            calls.append(prop("rollback_deployment", tgt,
                              f"Roll back to the last good version: the failures began with the latest release. "
                              f"{f.get('trigger', '')}", "medium",
                              f"aws ecs update-service --cluster {cluster} --service {svc} "
                              f"--task-definition {prev_td}"))
        if cat == "memory_exhaustion":
            calls.append(prop("increase_memory", tgt,
                              "Stop-gap while the release is fixed: more task memory reduces OOM kills.",
                              "low", "Register a task definition with higher memory, then update-service."))
        if cat == "connection_exhaustion":
            m = re.search(r"of ([\w-]+) at", f.get("trigger", ""))
            job = m.group(1) if m else "reporting-job"
            calls += [
                prop("stop_job", f"ecs:{cluster}/{job}", f"{job} is exhausting the DB connection slots.",
                     "low", f"aws ecs update-service --cluster {cluster} --service {job} --desired-count 0"),
                prop("terminate_idle_connections", f"rds:{db}",
                     "Free the slots held by idle sessions of the job.", "medium",
                     "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                     "WHERE application_name='" + job + "' AND state='idle';"),
                prop("add_connection_pooling", f"app:{job}",
                     "Permanent fix: use a pool (or RDS Proxy) so the job reuses connections.",
                     "low", "Add a connection pool / RDS Proxy before re-enabling the job.")]
        if cat == "disk_exhaustion":
            calls += [
                prop("reduce_log_level", f"host:{inst or svc}", "DEBUG logging filled the disk.", "low",
                     "Set log_level=INFO in the app config and reload the service."),
                prop("truncate_logs", f"host:{inst or svc}", "Free space now; archive first.", "medium",
                     "Archive then truncate the oversized app.log; configure logrotate."),
                prop("extend_volume", f"ebs:{vol or 'volume'}", "Headroom so it does not recur.", "low",
                     f"aws ec2 modify-volume --volume-id {vol or '<volume-id>'} --size <new-size-gb>")]
        if not calls:
            calls.append(prop("other", f"service:{svc}", "No automatic fix is known; escalate to a human.",
                              "low", "Page the owning team."))
        return _ai(calls, "Proposing fixes for human approval (nothing is executed).")
    return _ai(text="Remediation proposals recorded; awaiting human approval.")


_ROLES = {"triage": _triage, "diagnosis": _diagnosis, "remediation": _remediation}


class FakeIncidentModel(BaseChatModel):
    """Offline, rule-based stand-in for a chat model that supports tool calling."""

    @property
    def _llm_type(self) -> str:
        return "fake-incident-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeIncidentModel":
        # Real models need to be told which tools exist. Ours already knows.
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        view = _View(messages)
        m = re.search(r"\[ROLE:(\w+)\]", view.system)
        step = _ROLES.get(m.group(1) if m else "")
        msg = step(view) if step else _ai(text="OK")
        return ChatResult(generations=[ChatGeneration(message=msg)])
