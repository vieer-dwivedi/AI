from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="LangGraph Agent API")

app.include_router(router)