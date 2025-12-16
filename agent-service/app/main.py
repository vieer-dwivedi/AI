from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.agents.master_agent import MasterAgent
from app.logging_config import get_logger

logger = get_logger("api")

app = FastAPI()
agent = MasterAgent()


class Request(BaseModel):
    instruction: str


@app.post("/run")
def run_agent(req: Request):
    logger.info("API request received")
    return agent.run(req.instruction)


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
