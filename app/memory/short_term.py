from typing import List


def add_to_stm(stm: List[dict], exchange: dict) -> List[dict]:
    """
    Adds a new exchange to short term memory.
    Keeps only the last 10 exchanges to avoid
    context window overflow.

    Args:
        stm: Current STM list from state.
        exchange: Dict with query, response, metadata.

    Returns:
        Updated STM list.
    """

    stm.append(exchange)

    # Keep only last 10 exchanges
    if len(stm) > 10:
        stm = stm[-10:]

    return stm


def get_stm_context(stm: List[dict]) -> str:
    """
    Formats STM as a readable conversation history string
    for inclusion in prompts.

    Args:
        stm: Current STM list from state.

    Returns:
        Formatted conversation history string.
    """

    if not stm:
        return ""

    lines = ["Previous conversation in this session:"]
    for exchange in stm[-3:]:     # only last 3 for brevity
        lines.append(f"User: {exchange.get('query', '')}")
        lines.append(
            f"Agent: {exchange.get('response', '')[:200]}..."
        )

    return "\n".join(lines)