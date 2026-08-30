"""
Document retrieval tool — connects to the Chroma vector store built by
rag/ingestor.py and exposes a search tool the agent can call.
"""

from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
)


@tool
def search_documents(query: str) -> str:
    """Search internal documents for relevant information.

    Use this when the user asks a question that might be answered by
    the ingested documents/notes, rather than general knowledge or weather.

    Args:
        query: The search query

    Returns:
        Relevant document excerpts
    """
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content for doc in results)
