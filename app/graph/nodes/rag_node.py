from app.graph.state import AgentState
from app.rag.embedder import load_vector_store
from app.rag.retriever import (
    retrieve_with_scores,
    filter_by_confidence,
    extract_citations,
)
from app.core.config import config


def rag_node(state: AgentState) -> AgentState:
    """
    Retrieves relevant document chunks from the FAISS
    vector store for each sub-question.
    """

    try:
        vector_store = load_vector_store(state["session_id"])
        print(f"[RAG] Vector store loaded for session: {state['session_id']}")
    except FileNotFoundError:
        print(f"[RAG] No vector store found for session: {state['session_id']}")
        print(f"[RAG] Falling back to web search")
        return {
            **state,
            "retrieved_chunks": [],
            "retrieval_scores": [],
            "has_confident_retrieval": False,
            "citations": [],
            "source_mode": "WEB",
        }

    all_chunks = []
    all_scores = []

    # Retrieve for each sub-question
    for question in state.get("sub_questions", [state["user_query"]]):
        print(f"[RAG] Retrieving for sub-question: {question}")
        docs, scores = retrieve_with_scores(
            vector_store=vector_store,
            query=question,
        )
        print(f"[RAG] Retrieved {len(docs)} chunks with scores: {scores}")
        all_chunks.extend(docs)
        all_scores.extend(scores)

    # Filter by confidence threshold
    confident_docs, confident_scores = filter_by_confidence(
        documents=all_chunks,
        scores=all_scores,
    )

    print(f"[RAG] After filtering: {len(confident_docs)} confident chunks")
    has_confident = len(confident_docs) > 0

    # If no confident chunks found fall through to web search
    if not has_confident:
        print(f"[RAG] No confident chunks found. Falling back to web search.")
        return {
            **state,
            "retrieved_chunks": [doc.page_content for doc in all_chunks],
            "retrieval_scores": all_scores,
            "has_confident_retrieval": False,
            "citations": [],
            "source_mode": "BOTH",
        }

    citations = extract_citations(confident_docs)
    print(f"[RAG] Citations found: {citations}")

    return {
        **state,
        "retrieved_chunks": [doc.page_content for doc in confident_docs],
        "retrieval_scores": confident_scores,
        "has_confident_retrieval": True,
        "citations": citations,
    }