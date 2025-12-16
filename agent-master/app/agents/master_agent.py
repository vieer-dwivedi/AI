import json
import re
import uuid
import time
from app.llm_client import LLMClient
from app.queue_client import push_job
from app.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT
from app.metrics.agent_metrics import (
    AGENT_REQUESTS,
    AGENT_ERRORS,
    AGENT_STAGE_DURATION,
    AGENT_TASKS_QUEUED
)
from app.logging_config import get_logger

logger = get_logger("master-agent")


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("LLM did not return JSON")
    return json.loads(match.group())


class MasterAgent:
    def __init__(self):
        self.llm = LLMClient()

    def run(self, user_instruction: str) -> dict:
        AGENT_REQUESTS.inc()
        workflow_id = str(uuid.uuid4())

        logger.info("Agent run started", extra={"workflow_id": workflow_id})

        start = time.time()
        try:
            prompt = f"""
{PLANNER_SYSTEM_PROMPT}

User request:
{user_instruction}
"""

            raw = self.llm.generate(prompt)
            plan = extract_json(raw)

        except Exception:
            AGENT_ERRORS.labels(stage="planning").inc()
            logger.exception("Planning failed")
            raise
        finally:
            AGENT_STAGE_DURATION.labels(stage="planning").observe(
                time.time() - start
            )

        # ---------- Clarification ----------
        if plan.get("needs_clarification"):
            logger.info("Clarification required")
            return {
                "workflow_id": workflow_id,
                "needs_clarification": True,
                "questions": plan["questions"]
            }

        # ---------- Task Queueing ----------
        tasks = plan["tasks"]
        total = len(tasks)

        for idx, task in enumerate(tasks, start=1):
            message = {
                "workflow_id": workflow_id,
                "task_index": idx,
                "total_tasks": total,
                **task
            }

            push_job(message)
            AGENT_TASKS_QUEUED.inc()

            logger.info(
                f"Task queued {idx}/{total}",
                extra={"workflow_id": workflow_id}
            )

        return {
            "workflow_id": workflow_id,
            "project_name": plan["project_name"],
            "total_tasks": total,
            "status": "queued"
        }
