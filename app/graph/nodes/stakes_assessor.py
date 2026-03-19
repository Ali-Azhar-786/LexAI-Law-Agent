from app.graph.state import AgentState

HIGH_STAKES_MATTER_TYPES = {
    "criminal", "custody", "deportation",
    "property", "constitutional",
}

HIGH_STAKES_KEYWORDS = [
    "arrested", "charged", "imprisoned", "jail", "prison",
    "custody", "child", "divorce", "deportation", "eviction",
    "fired", "terminated", "sued", "lawsuit", "court",
    "warrant", "sentence", "penalty", "fine", "compensation",
    "rights violated", "illegal", "unlawful",
]


def stakes_assessor_node(state: AgentState) -> AgentState:
    """
    Determines whether the user's situation is high stakes
    based on matter type and keywords in the query.

    Args:
        state: Current AgentState.

    Returns:
        Updated state with is_high_stakes set.
    """

    matter_type = state.get("matter_type", "").lower()
    query = state.get("user_query", "").lower()

    # Check matter type
    type_is_high_stakes = any(
        t in matter_type
        for t in HIGH_STAKES_MATTER_TYPES
    )

    # Check query keywords
    keyword_is_high_stakes = any(
        keyword in query
        for keyword in HIGH_STAKES_KEYWORDS
    )

    is_high_stakes = type_is_high_stakes or keyword_is_high_stakes

    return {
        **state,
        "is_high_stakes": is_high_stakes,
    }


def route_after_stakes(state: AgentState) -> str:
    """
    Conditional edge after stakes assessor.

    Returns:
        "hitl_node" if high stakes
        "memory_update_node" if not high stakes
    """

    if state.get("is_high_stakes"):
        return "hitl_node"

    return "memory_update_node"