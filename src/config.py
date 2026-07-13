"""Application configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "Qwen/Qwen2.5-7B-Instruct",
)
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_rag")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
