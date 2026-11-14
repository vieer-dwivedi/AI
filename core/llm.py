import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
model = init_chat_model(
        model=os.getenv("MODEL"),
        api_key=os.getenv("GOOGLE_API_KEY"),
        timeout=60,
        max_retries= 3
    )