from qdrant_client import QdrantClient
from app.graph.state import AgentState
from app.core.config import config

qdrant_client = QdrantClient(
    host=config.QDRANT_HOST,
    port=config.QDRANT_PORT,
)


def router_node(state: AgentState) -> AgentState:
    """
    Routes to RAG or WEB based on whether a Qdrant
    collection exists for this session.
    """

    doc_available = (
        state.get("uploaded_doc_path") is not None and
        state.get("doc_validated") is True
    )

    vector_store_exists = False

    if doc_available:
        collection_name = (
            f"{config.QDRANT_COLLECTION_PREFIX}_{state['session_id']}"
        )
        try:
            existing = [
                c.name for c in
                qdrant_client.get_collections().collections
            ]
            vector_store_exists = collection_name in existing
        except Exception:
            vector_store_exists = False

        print(f"[ROUTER] Collection: {collection_name}")
        print(f"[ROUTER] Exists: {vector_store_exists}")

    source_mode = "RAG" if (
        doc_available and vector_store_exists
    ) else "WEB"

    print(f"[ROUTER] Source mode: {source_mode}")

    return {**state, "source_mode": source_mode}


def route_after_router(state: AgentState) -> str:
    if state["source_mode"] == "RAG":
        return "rag_node"
    return "web_search_node"