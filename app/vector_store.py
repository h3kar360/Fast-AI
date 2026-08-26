import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy.ext.asyncio import create_async_engine
from dotenv import load_dotenv

load_dotenv()

PGVECTOR_DATABASE_URL = os.getenv("PGVECTOR_DATABASE_URL")

async_engine = create_async_engine(PGVECTOR_DATABASE_URL)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="pdf_docs",
    connection=async_engine
)

def get_vector_store() -> PGVector:
    return vector_store