import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.graph.state import AgentState
from app.core.config import config

llm = ChatGroq(
    model=config.GROQ_MODEL,
    api_key=config.GROQ_API_KEY,
    temperature=0,
)

decomposer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a legal query analyst.
Your job is to break down a complex legal question into
focused sub-questions, each targeting one specific legal aspect.

Rules:
- Generate between 1 and 4 sub-questions
- Each sub-question must be self-contained and specific
- Do not repeat the same question in different words
- If the query is already simple and specific return it as a
  single sub-question

Respond ONLY with a JSON object in this exact format:
{{"sub_questions": ["question 1", "question 2", "question 3"]}}

No explanation. No extra text. Only the JSON object."""
    ),
    (
        "human",
        "User query: {user_query}\nJurisdiction: {jurisdiction}\nMatter type: {matter_type}"
    ),
])

decomposer_chain = decomposer_prompt | llm


def decomposer_node(state: AgentState) -> AgentState:
    """
    Breaks the user query into focused sub-questions.

    Args:
        state: Current AgentState.

    Returns:
        Updated state with sub_questions list populated.
    """

    response = decomposer_chain.invoke({
        "user_query": state["user_query"],
        "jurisdiction": state.get("jurisdiction", "unspecified"),
        "matter_type": state.get("matter_type", "unspecified"),
    })

    raw = response.content.strip()

    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            sub_questions = data.get("sub_questions", [])

            # Fallback — if empty use original query
            if not sub_questions:
                sub_questions = [state["user_query"]]

            return {
                **state,
                "sub_questions": sub_questions,
            }

    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback — use original query as single sub-question
    return {
        **state,
        "sub_questions": [state["user_query"]],
    }