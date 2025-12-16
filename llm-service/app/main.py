# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from llm import GeminiLLM

from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import time

app = FastAPI(title="LLM Service", version="1.0.0")

llm = GeminiLLM()

# ---------- Prometheus Metrics ----------
REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total LLM generate requests"
)

REQUEST_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "LLM request latency"
)

# ---------- Schemas ----------
class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    output: str

# ---------- Routes ----------

@app.get("/ping")
def ping():
    """
    Lightweight health check.
    Used by k8s, docker, load balancers.
    """
    return {"status": "ok", "service": "llm"}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    REQUEST_COUNT.inc()
    start = time.time()

    output = llm.generate(req.prompt)

    REQUEST_LATENCY.observe(time.time() - start)

    return GenerateResponse(output=output)

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
