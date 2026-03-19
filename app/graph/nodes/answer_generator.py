from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.graph.state import AgentState
from app.core.config import config

llm = ChatGroq(
    model=config.GROQ_MODEL,
    api_key=config.GROQ_API_KEY,
    temperature=0,
)

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert legal assistant helping a citizen understand their rights.

Your job is to answer the user's legal question using ONLY the provided context.

Rules:
- Base your answer strictly on the provided context
- Cite specific articles, sections, or clauses when available
- Use plain language that a non-lawyer can understand
- Be thorough but concise
- If the context is insufficient to fully answer the question
  state clearly which parts you could not find information for
- Do NOT make up laws or cite non-existent articles
- If you use any knowledge beyond the provided context
  explicitly label it as: [General Knowledge - Verify Independently]

Jurisdiction: {jurisdiction}
User Role: {user_role}
Matter Type: {matter_type}"""
    ),
    (
        "human",
        """Question: {user_query}

Context:
{context}

Please provide a clear, cited answer."""
    ),
])

answer_chain = answer_prompt | llm


def build_context(state: AgentState) -> str:
    """
    Builds the context string from retrieved chunks
    or web search results depending on source mode.
    """

    source_mode = state.get("source_mode", "WEB")
    context_parts = []

    if source_mode in ("RAG", "BOTH"):
        chunks = state.get("retrieved_chunks", [])
        if chunks:
            context_parts.append(
                "--- From Uploaded Document ---\n" +
                "\n\n".join(chunks)
            )

    if source_mode in ("WEB", "BOTH"):
        web_results = state.get("web_search_results", [])
        if web_results:
            context_parts.append(
                "--- From Web Search ---\n" +
                "\n\n".join(web_results)
            )

    return "\n\n".join(context_parts) if context_parts else ""


def answer_generator_node(state: AgentState) -> AgentState:
    """
    Generates a grounded legal answer using retrieved context.

    Args:
        state: Current AgentState.

    Returns:
        Updated state with generated_answer populated.
    """

    context = build_context(state)

    # If no context at all trigger fallback
    if not context.strip():
        return {
            **state,
            "generated_answer": "",
            "fallback_triggered": True,
        }

    response = answer_chain.invoke({
        "user_query": state["user_query"],
        "context": context,
        "jurisdiction": state.get("jurisdiction", "unspecified"),
        "user_role": state.get("user_role", "unspecified"),
        "matter_type": state.get("matter_type", "unspecified"),
    })

    return {
        **state,
        "generated_answer": response.content.strip(),
        "fallback_triggered": False,
    }