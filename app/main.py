"""FastAPI entrypoint: /exceptions, /cases, /approvals (Step 10).

Run: uv run uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from app.agent import memory
from app.agent.orchestrator import execute_tool, run_case
from app.observability import tracing
from app.observability.tracing import Tracer


@asynccontextmanager
async def lifespan(_app: FastAPI):
    memory.fail_stale_cases()                      # cases cut off by a restart would otherwise show 'running' forever
    yield


app = FastAPI(title="DispatchAgent", lifespan=lifespan)


class EmailIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    subject: str
    body: str


class ApprovalIn(BaseModel):
    approve: bool
    comment: str = ""


class ReviewIn(BaseModel):
    comment: str = ""


def _run(case_id: str, email: dict) -> None:
    """Background job: run the agent, then persist the outcome."""
    try:
        result = run_case(email, case_id=case_id, log=lambda _msg: None)
        memory.save_result(result, email)
    except Exception as exc:
        memory.fail_case(case_id, f"{type(exc).__name__}: {exc}")


@app.post("/exceptions", status_code=202)
def submit_exception(email: EmailIn, tasks: BackgroundTasks):
    case_id = memory.new_case_id()
    payload = email.model_dump()
    memory.create_case(case_id, payload)
    tasks.add_task(_run, case_id, payload)
    return {"case_id": case_id, "status": "running"}


@app.get("/cases")
def list_cases():
    return memory.list_cases()


@app.get("/cases/{case_id}")
def get_case(case_id: str):
    case = memory.get_case(case_id)
    if case is None:
        raise HTTPException(404, "case not found")
    return case


@app.get("/cases/{case_id}/trace")
def get_trace(case_id: str):
    return tracing.read_trace(case_id)


@app.post("/cases/{case_id}/review")
def mark_reviewed(case_id: str, body: ReviewIn):
    case = memory.get_case(case_id)
    if case is None:
        raise HTTPException(404, "case not found")
    if case["status"] not in ("needs_review", "needs_human"):
        raise HTTPException(409, f"case is {case['status']}, nothing to review")
    memory.set_status(case_id, "reviewed")
    Tracer(case_id).event("reviewed", comment=body.comment)
    return {"case_id": case_id, "status": "reviewed"}


@app.get("/approvals")
def pending_approvals():
    return memory.list_approvals(status="pending")


@app.post("/approvals/{approval_id}")
def decide_approval(approval_id: int, body: ApprovalIn):
    approval = memory.get_approval(approval_id)
    if approval is None:
        raise HTTPException(404, "approval not found")
    if not memory.decide(approval_id, body.approve, body.comment):
        raise HTTPException(409, f"approval already {approval['status']}")

    result = None
    tracer = Tracer(approval["case_id"])
    tracer.event("approval", approval_id=approval_id, tool=approval["tool_name"],
                 decision="approved" if body.approve else "rejected", comment=body.comment)
    if body.approve:                                   # a human said yes: run the parked action
        result = execute_tool(approval["tool_name"], approval["args"])
        memory.log_tool_call(approval["case_id"], approval["tool_name"], approval["args"], result)
        tracer.event("tool_call", tool=approval["tool_name"], args=approval["args"],
                     decision="approved_by_human", result=str(result)[:800])

    if memory.pending_count(approval["case_id"]) == 0:
        memory.set_status(approval["case_id"], "closed")
    return {"approval_id": approval_id, "decision": "approved" if body.approve else "rejected",
            "result": result, "case_status": memory.get_case(approval["case_id"])["status"]}
