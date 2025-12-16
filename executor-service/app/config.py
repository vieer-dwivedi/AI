import os

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_QUEUE_NAME = os.getenv("AZURE_QUEUE_NAME")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "2"))
MAX_LLM_RETRIES = int(os.getenv("MAX_LLM_RETRIES", "5"))
BASE_WORKDIR = os.getenv("BASE_WORKDIR", "workdir")
LOG_DIR = os.getenv("LOG_DIR", "logs")

if not AZURE_STORAGE_CONNECTION_STRING:
    raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")
if not AZURE_QUEUE_NAME:
    raise RuntimeError("AZURE_QUEUE_NAME not set")
if not LLM_BASE_URL:
    raise RuntimeError("LLM_BASE_URL not set")
