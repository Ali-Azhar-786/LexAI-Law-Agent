from app.graph.state import AgentState

# Matter types that are inherently high stakes
HIGH_STAKES_MATTER_TYPES = {
    "criminal",
    "custody",
    "deportation",
}

# Keywords that signal the user is PERSONALLY involved
# and facing immediate consequences — not just asking academically
PERSONAL_URGENCY_KEYWORDS = [
    "i have been arrested",
    "i was arrested",
    "i am arrested",
    "i have been charged",
    "i was charged",
    "i am accused",
    "i have been accused",
    "they are suing me",
    "i am being sued",
    "i received a court notice",
    "i got a court notice",
    "i am in custody",
    "i have been detained",
    "i was detained",
    "my child was taken",
    "i am facing deportation",
    "i have been fired",
    "my property was seized",
    "i need a lawyer",
    "i need legal help urgently",
    "what should i do now",
    "what do i do now",
    "i am in trouble",
]


def stakes_assessor_node(state: AgentState) -> AgentState:
    """
    Determines whether the situation is genuinely high stakes.

    Requires BOTH a high stakes matter type AND personal
    urgency signals in the query.

    Simple informational queries like "what is the punishment
    for theft" are NOT high stakes — the user is asking
    academically, not because they are personally accused.

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

    # Check for personal urgency — user must be personally
    # involved and facing immediate consequences
    personally_involved = any(
        keyword in query
        for keyword in PERSONAL_URGENCY_KEYWORDS
    )

    # BOTH conditions must be true for escalation
    # This prevents academic queries from triggering HITL
    is_high_stakes = type_is_high_stakes and personally_involved

    print(f"[STAKES] Matter type high stakes: {type_is_high_stakes}")
    print(f"[STAKES] Personally involved: {personally_involved}")
    print(f"[STAKES] Final decision: {is_high_stakes}")

    return {
        **state,
        "is_high_stakes": is_high_stakes,
    }


def route_after_stakes(state: AgentState) -> str:
    """
    Conditional edge after stakes assessor.
    """
    if state.get("is_high_stakes"):
        return "hitl_node"
    return "memory_update_node"