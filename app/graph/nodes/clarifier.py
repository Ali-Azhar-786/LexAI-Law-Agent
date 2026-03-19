from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.graph.state import AgentState
from app.core.config import config

# ---------------------------------------------------------
# Initialize LLM
# ---------------------------------------------------------
llm = ChatGroq(
    model=config.GROQ_MODEL,
    api_key=config.GROQ_API_KEY,
    temperature=0,
)

# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------
clarifier_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a legal assistant intake specialist.
Your job is to extract three pieces of information from the user's query:
1. jurisdiction — the country and state/province if mentioned
2. user_role — the user's role in the legal matter
   (e.g. tenant, employee, accused, plaintiff, defendant, citizen, student)
3. matter_type — the type of legal matter
   (e.g. civil, criminal, labor, family, property, constitutional, consumer)

If any of these cannot be determined from the query, return "unspecified" for that field.

Respond ONLY with a JSON object in this exact format:
{{"jurisdiction": "...", "user_role": "...", "matter_type": "..."}}

No explanation. No extra text. Only the JSON object."""
    ),
    (
        "human",
        "User query: {user_query}"
    ),
])

clarifier_chain = clarifier_prompt | llm


def clarifier_node(state: AgentState) -> AgentState:
    """
    Extracts jurisdiction, user_role, and matter_type from
    the user query. If these were already provided by the
    frontend form they are preserved as-is.

    Args:
        state: Current AgentState.

    Returns:
        Updated state with jurisdiction, user_role, matter_type.
    """

    import json
    import re

    # If context was already provided by user skip LLM extraction
    if (
        state.get("jurisdiction") and
        state.get("user_role") and
        state.get("matter_type") and
        state["jurisdiction"] != "" and
        state["user_role"] != "" and
        state["matter_type"] != ""
    ):
        return state

    # Extract context from query using LLM
    response = clarifier_chain.invoke({
        "user_query": state["user_query"]
    })

    raw = response.content.strip()

    # Parse JSON response
    try:
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                **state,
                "jurisdiction": data.get("jurisdiction", "unspecified"),
                "user_role": data.get("user_role", "unspecified"),
                "matter_type": data.get("matter_type", "unspecified"),
            }
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback if parsing fails
    return {
        **state,
        "jurisdiction": state.get("jurisdiction") or "unspecified",
        "user_role": state.get("user_role") or "unspecified",
        "matter_type": state.get("matter_type") or "unspecified",
    }