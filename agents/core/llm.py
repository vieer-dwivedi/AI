import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()  # needs GROQ_API_KEY, MODEL_NAME and MODEL_PROVIDER=groq in .env

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "groq")
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "2048"))
MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "2"))

if not MODEL_NAME:
    raise EnvironmentError("MODEL_NAME is not set in the .env file")

llm = init_chat_model(
    MODEL_NAME,
    model_provider=MODEL_PROVIDER,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    max_retries=MAX_RETRIES,
    reasoning_effort="low",  # fewer hidden reasoning tokens = faster and cheaper
)
