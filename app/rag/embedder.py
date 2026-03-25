from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.core.config import config

# ---------------------------------------------------------
# Embedding model — initialized once at module level
# ---------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 64,
    },
)

# ---------------------------------------------------------
# Qdrant client — connects to running Qdrant container
# ---------------------------------------------------------
qdrant_client = QdrantClient(
    host=config.QDRANT_HOST,
    port=config.QDRANT_PORT,
)

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIM = 384


def get_collection_name(session_id: str) -> str:
    """
    Returns the Qdrant collection name for a given session.
    Each session gets its own isolated collection.
    """
    return f"{config.QDRANT_COLLECTION_PREFIX}_{session_id}"


def create_vector_store(
    chunks: list[Document],
    session_id: str,
) -> QdrantVectorStore:
    """
    Creates a Qdrant collection for the session and
    indexes all document chunks into it.

    Args:
        chunks: List of Document chunks from chunker.py
        session_id: Unique session identifier.

    Returns:
        QdrantVectorStore instance ready for retrieval.
    """

    collection_name = get_collection_name(session_id)

    # Delete existing collection for this session if it exists
    existing = [
        c.name for c in qdrant_client.get_collections().collections
    ]
    if collection_name in existing:
        qdrant_client.delete_collection(collection_name)

    # Create fresh collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )

    # Index chunks in batches
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        url=f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}",
    )

    print(f"[QDRANT] Indexed {len(chunks)} chunks "
          f"into collection: {collection_name}")

    return vector_store


def load_vector_store(session_id: str) -> QdrantVectorStore:
    """
    Loads an existing Qdrant collection for the given session.

    Args:
        session_id: Unique session identifier.

    Returns:
        QdrantVectorStore instance ready for retrieval.

    Raises:
        FileNotFoundError: If no collection exists for this session.
    """

    collection_name = get_collection_name(session_id)

    existing = [
        c.name for c in qdrant_client.get_collections().collections
    ]

    if collection_name not in existing:
        raise FileNotFoundError(
            f"No Qdrant collection found for session: {session_id}"
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    print(f"[QDRANT] Loaded collection: {collection_name}")
    return vector_store