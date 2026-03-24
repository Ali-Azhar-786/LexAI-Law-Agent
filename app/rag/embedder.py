import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.core.config import config

# ---------------------------------------------------------
# Initialize embedding model once at module level
# ---------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 64,            # embed 64 chunks at once instead of 1
    },
)


def create_vector_store(chunks: list[Document]) -> FAISS:
    """
    Takes a list of document chunks and creates a FAISS
    vector store by embedding each chunk in batches.

    Args:
        chunks: List of Document chunks from chunker.py

    Returns:
        FAISS vector store loaded in memory.
    """

    # Process in batches to avoid memory issues on large documents
    batch_size = 100
    vector_store = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]

        if vector_store is None:
            # Create vector store from first batch
            vector_store = FAISS.from_documents(
                documents=batch,
                embedding=embeddings,
            )
        else:
            # Add subsequent batches to existing store
            vector_store.add_documents(batch)

    return vector_store


def save_vector_store(vector_store: FAISS, session_id: str) -> str:
    """
    Saves the FAISS vector store to disk for the given session.
    """
    save_path = os.path.join("faiss_index", session_id)
    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    return save_path


def load_vector_store(session_id: str) -> FAISS:
    """
    Loads a previously saved FAISS vector store from disk.
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