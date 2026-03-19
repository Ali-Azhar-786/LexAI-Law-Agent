from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.graph.state import AgentState
from app.core.config import config

llm = ChatGroq(
    model=config.GROQ_MODEL,
    api_key=config.GROQ_API_KEY,
    temperature=0,
)

lawyer_questions_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a legal assistant helping a citizen prepare
for a consultation with a licensed lawyer.

Generate a list of 5 to 7 specific, practical questions the user
should ask their lawyer based on their situation.

Respond ONLY with a JSON object in this exact format:
{{"questions": ["question 1", "question 2", "question 3"]}}

No explanation. No extra text. Only the JSON object."""
    ),
    (
        "human",
        """User situation:
Query: {user_query}
Jurisdiction: {jurisdiction}
Matter type: {matter_type}
User role: {user_role}"""
    ),
])

lawyer_questions_chain = lawyer_questions_prompt | llm

ESCALATION_TEMPLATE = """⚠️ **Important Notice — Legal Consultation Recommended**

Based on your query, this appears to be a **{matter_type}** matter 
that carries significant legal consequences. While I have provided 
general information above, this situation strongly warrants advice 
from a licensed lawyer in **{jurisdiction}**.

**Why you need a lawyer:**
- {matter_type} matters can have serious and lasting consequences
- Laws vary significantly by jurisdiction and specific circumstances
- A lawyer can review the specific facts of your case
- Procedural deadlines may apply that you are unaware of

**I have prepared specific questions for your lawyer consultation below.**"""


def hitl_node(state: AgentState) -> AgentState:
    """
    Appends an escalation message and generates targeted
    lawyer consultation questions for high-stakes situations.

    This is the Human-in-the-Loop checkpoint — the agent
    explicitly hands off to a human professional.

    Args:
        state: Current AgentState.

    Returns:
        Updated state with escalation_message and
        lawyer_questions populated.
    """

    import json
    import re

    # Build escalation message
    escalation_message = ESCALATION_TEMPLATE.format(
        matter_type=state.get("matter_type", "legal"),
        jurisdiction=state.get("jurisdiction", "your jurisdiction"),
    )

    # Generate lawyer questions
    try:
        response = lawyer_questions_chain.invoke({
            "user_query": state["user_query"],
            "jurisdiction": state.get("jurisdiction", "unspecified"),
            "matter_type": state.get("matter_type", "unspecified"),
            "user_role": state.get("user_role", "unspecified"),
        })

        raw = response.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)

        if match:
            data = json.loads(match.group())
            lawyer_questions = data.get("questions", [])
        else:
            lawyer_questions = []

    except Exception:
        lawyer_questions = []

    # Append escalation to existing response
    current_response = state.get("final_response") or state.get(
        "generated_answer", ""
    )

    final_response = (
        f"{current_response}\n\n{escalation_message}"
        if current_response
        else escalation_message
    )

    return {
        **state,
        "escalation_message": escalation_message,
        "lawyer_questions": lawyer_questions,
        "final_response": final_response,
    }