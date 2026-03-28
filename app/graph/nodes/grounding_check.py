def route_after_grounding(state: AgentState) -> str:
    """
    Conditional edge after grounding check.

    HIGH confidence          → stakes_assessor
    LOW + RAG path           → web_search_fallback (only if not already done)
    LOW + already fell back  → stakes_assessor (accept web result as-is)
    LOW + WEB path           → fallback_node
    """

    confidence = state.get("confidence", "LOW")
    source_mode = state.get("source_mode", "WEB")
    already_fell_back = state.get("rag_fallback_to_web", False)
    fallback_triggered = state.get("fallback_triggered", False)

    if fallback_triggered:
        return "fallback_node"

    if confidence == "HIGH":
        return "stakes_assessor"

    # LOW confidence on RAG path — try web once
    if confidence == "LOW" and source_mode == "RAG" and not already_fell_back:
        return "web_search_fallback"

    # LOW confidence but already tried web fallback
    # Accept the web result and move to stakes assessor
    if confidence == "LOW" and already_fell_back:
        return "stakes_assessor"

    # LOW confidence on pure WEB path — honest fallback
    return "fallback_node"