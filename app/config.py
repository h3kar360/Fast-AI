import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment or .env file.")

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)