import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Central configuration class.
    All environment variables and app settings are accessed from here.
    No other file should call os.getenv() directly.
    """

    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # ---------------------------------------------------------
    # EMBEDDINGS
    # ---------------------------------------------------------
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN", "")

    # ---------------------------------------------------------
    # WEB SEARCH
    # ---------------------------------------------------------
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # ---------------------------------------------------------
    # LANGSMITH
    # ---------------------------------------------------------
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "lexai-law-agent")

    # ---------------------------------------------------------
    # RAG SETTINGS
    # ---------------------------------------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    RETRIEVAL_TOP_K: int = 4
    CONFIDENCE_THRESHOLD: float = 0.75

    # ---------------------------------------------------------
    # FASTAPI SETTINGS
    # ---------------------------------------------------------
    APP_TITLE: str = "LexAI — Agentic Law Assistant"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "An agentic AI system that helps citizens understand "
        "constitutional rights and applicable laws."
    )
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ---------------------------------------------------------
    # FILE UPLOAD SETTINGS
    # ---------------------------------------------------------
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_FILE_TYPES: list = ["application/pdf"]


# Single instance used across the entire app
config = Config()