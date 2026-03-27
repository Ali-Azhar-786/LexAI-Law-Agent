import re
from app.graph.state import AgentState

HIGH_STAKES_MATTER_TYPES = {
    "criminal",
    "custody",
    "deportation",
}

# Regex patterns that detect personal involvement
# More robust than exact string matching
PERSONAL_URGENCY_PATTERNS = [
    r"i\s+(have\s+been|was|am|got)\s+(arrested|detained|charged|accused)",
    r"i\s+am\s+(being\s+)?(arrested|sued|charged|accused|detained)",
    r"they\s+are\s+suing\s+me",
    r"i\s+(received|got)\s+a?\s+court\s+(notice|summon|order)",
    r"i\s+am\s+(in\s+custody|facing\s+deportation|in\s+jail|in\s+prison)",
    r"my\s+child\s+was\s+taken",
    r"my\s+property\s+was\s+(seized|taken|confiscated)",
    r"i\s+(have\s+been|was)\s+fired\s+(illegally|wrongfully|unfairly)",
    r"i\s+need\s+(urgent|immediate)\s+legal\s+help",
    r"what\s+should\s+i\s+do\s+(now|immediately|urgently)",
    r"i\s+am\s+in\s+(serious\s+)?trouble",
    r"i\s+(have\s+)?just\s+(been\s+)?(arrested|charged|accused)",
]


def is_personally_involved(query: str) -> bool:
    """
    Uses regex to detect whether the user is personally
    involved in a legal situation requiring urgent help.
    More robust than exact string matching.
    """
    query_lower = query.lower().strip()
    for pattern in PERSONAL_URGENCY_PATTERNS:
        if re.search(pattern, query_lower):
            print(f"[STAKES] Matched urgency pattern: {pattern}")
            return True
    return False


def stakes_assessor_node(state: AgentState) -> AgentState:
    """
    Determines whether the situation is genuinely high stakes.
    Requires BOTH high stakes matter type AND personal urgency.
    """

    matter_type = state.get("matter_type", "").lower()
    query = state.get("user_query", "")

    type_is_high_stakes = any(
        t in matter_type
        for t in HIGH_STAKES_MATTER_TYPES
    )

    personally_involved = is_personally_involved(query)

    is_high_stakes = type_is_high_stakes and personally_involved

    print(f"[STAKES] Query: {query}")
    print(f"[STAKES] Matter type: {matter_type}")
    print(f"[STAKES] Type high stakes: {type_is_high_stakes}")
    print(f"[STAKES] Personally involved: {personally_involved}")
    print(f"[STAKES] Final decision: {is_high_stakes}")

    return {
        **state,
        "is_high_stakes": is_high_stakes,
    }


def route_after_stakes(state: AgentState) -> str:
    if state.get("is_high_stakes"):
        return "hitl_node"
    return "memory_update_node"