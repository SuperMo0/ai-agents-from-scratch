import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

client = OpenAI()
tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
RESULTS_DIR = Path(__file__).parent / "results"
SUB_CALL_MODEL = "gpt-4o-mini"
