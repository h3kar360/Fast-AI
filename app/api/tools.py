from app.api import crud
from app.vector_store import get_vector_store
from langchain.tools import tool

@tool
async def search_documents(query: str) -> str:
    """Search the document database for relevant information based on a user query."""
    vector_store = get_vector_store()

    docs = await crud.langchain_retrieve(vector_store, query)

    if not docs:
        return "Documents not found"

    formatted_context = []

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        formatted_context.append(f"--- Document Chunk {i} (Source: {source}, Page: {page}) ---\n{doc.page_content}")

    return "\n\n".join(formatted_context)