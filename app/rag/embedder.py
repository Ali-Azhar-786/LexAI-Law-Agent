import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.core.config import config

# ---------------------------------------------------------
# Initialize embedding model once at module level
# This avoids reloading the model on every function call
# ---------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def create_vector_store(chunks: list[Document]) -> FAISS:
    """
    Takes a list of document chunks and creates a FAISS
    vector store by embedding each chunk.

    Args:
        chunks: List of Document chunks from chunker.py

    Returns:
        FAISS vector store loaded in memory.
    """

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


def save_vector_store(vector_store: FAISS, session_id: str) -> str:
    """
    Saves the FAISS vector store to disk for the given session.

    Args:
        vector_store: The FAISS vector store to save.
        session_id: Unique session identifier used as folder name.

    Returns:
        Path where the vector store was saved.
    """

    save_path = os.path.join("faiss_index", session_id)
    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    return save_path


def load_vector_store(session_id: str) -> FAISS:
    """
    Loads a previously saved FAISS vector store from disk.

    Args:
        session_id: Session ID used when the store was saved.

    Returns:
        FAISS vector store loaded from disk.

    Raises:
        FileNotFoundError: If no vector store exists for this session.
    """

    load_path = os.path.join("faiss_index", session_id)

    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"No vector store found for session: {session_id}"
        )

    vector_store = FAISS.load_local(
        load_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store