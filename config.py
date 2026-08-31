"""
Centralized settings — equivalent to application.yml / application.properties
in a Spring Boot project. Import from here instead of hardcoding values
across multiple files.
"""
# config.py — add at the very top, before anything else
from dotenv import load_dotenv
load_dotenv()

# Model names
CHAT_MODEL = "qwen2.5-coder:7b"
EMBEDDING_MODEL = "nomic-embed-text"

# Model behavior
CHAT_TEMPERATURE = 0.2

# Vector store
CHROMA_PERSIST_DIR = "./rag/chroma_db"

# Logging
LOGGER_NAME = "promptops"

ACTIVE_SYSTEM_PROMPT = "system_v1"


import os

# Observability (Langfuse)
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")