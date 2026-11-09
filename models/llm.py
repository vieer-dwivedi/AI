import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model="google_genai:gemini-3.5-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    # temperature=0.7,
    timeout=30,
    max_retries=2
)