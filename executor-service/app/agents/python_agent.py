import logging
from app.agents.base_agent import BaseAgent
from app.execution.filesystem import ensure_workdir, write_file
from app.execution.runner import run_command
from app.execution.validation import validate_success
from app.llm.client import LLMClient
from app.llm.fix_prompt import build_fix_prompt
from app.config import BASE_WORKDIR, MAX_LLM_RETRIES

logger = logging.getLogger("python-agent")

class PythonAgent(BaseAgent):
    def __init__(self):
        self.llm = LLMClient()

    def execute(self, task):
        workdir = ensure_workdir(BASE_WORKDIR, task["workflow_id"])
        content = task["file"]["content"]

        cmd = task["context"].get("run_instructions","")
        cmd = cmd.split("Command:",1)[-1].strip() if "Command:" in cmd else None

        for _ in range(MAX_LLM_RETRIES):
            write_file(workdir, task["file"]["path"], content)
            if not cmd:
                return True

            result = run_command(cmd, workdir)
            if validate_success(result):
                return True

            content = self.llm.generate(
                build_fix_prompt(task, result.stdout, result.stderr, result.returncode)
            )
            task["file"]["content"] = content

        return False
