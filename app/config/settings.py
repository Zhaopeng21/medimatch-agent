import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
