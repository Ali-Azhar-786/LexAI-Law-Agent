from app.graph.state import AgentState


FALLBACK_MESSAGE = """I was unable to find a reliable and fully verified answer 
to your question.

This may be because:
- The specific legal provision you need is not covered in the uploaded document
- Web search did not return sufficiently detailed legal information
- This area of law is highly specialized or jurisdiction-specific
- The answer depends on case-specific facts that require professional assessment

**What I recommend:**
1. Consult a licensed lawyer in {jurisdiction} who specializes in {matter_type} law
2. Visit your country's official government legal portal for authoritative information
3. Contact a legal aid organization if cost is a concern

Would you like me to help you prepare a list of specific questions to bring 
to a lawyer consultation?"""


def fallback_node(state: AgentState) -> AgentState:
    """
    Generates an honest fallback response when no reliable
    answer could be found or confidence is too low.

    Args:
        state: Current AgentState.

    Returns:
        Updated state with final_response set to honest fallback.
    """

    message = FALLBACK_MESSAGE.format(
        jurisdiction=state.get("jurisdiction", "your jurisdiction"),
        matter_type=state.get("matter_type", "your legal matter"),
    )

    return {
        **state,
        "fallback_triggered": True,
        "final_response": message,
        "confidence": "LOW",
    }