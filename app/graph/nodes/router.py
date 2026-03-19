from app.graph.state import AgentState


def router_node(state: AgentState) -> AgentState:
    """
    Determines the source mode for retrieval based on
    whether a validated document exists.

    source_mode values:
    - "RAG"  → document uploaded and validated
    - "WEB"  → no document, use web search only
    - "BOTH" → document exists but may need web supplement

    Args:
        state: Current AgentState.

    Returns:
        Updated state with source_mode set.
    """

    doc_available = (
        state.get("uploaded_doc_path") is not None and
        state.get("doc_validated") is True
    )

    if doc_available:
        source_mode = "RAG"
    else:
        source_mode = "WEB"

    return {
        **state,
        "source_mode": source_mode,
    }


def route_after_router(state: AgentState) -> str:
    """
    Conditional edge function used by LangGraph to determine
    which node to go to after the router node.

    Returns:
        "rag_node" or "web_search_node"
    """

    if state["source_mode"] == "RAG":
        return "rag_node"
    return "web_search_node"