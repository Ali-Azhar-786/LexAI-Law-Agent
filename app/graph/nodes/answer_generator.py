from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.graph.state import AgentState
from app.core.config import config

llm = ChatGroq(
    model=config.GROQ_MODEL,
    api_key=config.GROQ_API_KEY,
    temperature=0,
)

rag_answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert legal assistant helping a citizen understand their rights.

You are answering based on an uploaded legal document provided by the user.

STRICT RULES:
- Use ONLY the provided document context to answer
- Do NOT use your general knowledge
- Cite specific Articles, Sections, or Clauses from the document
- If the exact answer is not in the context say clearly that
  the document does not contain this information
- Use plain language a non-lawyer can understand

Jurisdiction: {jurisdiction}
User Role: {user_role}
Matter Type: {matter_type}"""
    ),
    (
        "human",
        """Question: {user_query}

Document Context:
{context}

Provide a clear answer citing specific articles from the document."""
    ),
])

web_answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert legal assistant helping a citizen understand their rights.

You are answering based on web search results.

RULES:
- Base your answer on the provided web search results
- Cite the source URLs when referencing specific information
- Use plain language a non-lawyer can understand
- If information is insufficient say so clearly

Jurisdiction: {jurisdiction}
User Role: {user_role}
Matter Type: {matter_type}"""
    ),
    (
        "human",
        """Question: {user_query}

Web Search Results:
{context}

Provide a clear answer based on the search results."""
    ),
])

# Prompt used when RAG fell back to web
rag_fallback_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert legal assistant helping a citizen understand their rights.

The user uploaded a legal document but the specific answer to their 
question was not found in that document. You are now answering 
based on web search results.

RULES:
- Start your answer with this exact line:
  "📄 Note: The uploaded document does not contain specific 
   information about this query. The following answer is based 
   on web search results."
- Then provide a clear answer based on the web search results
- Cite the source URLs when referencing specific information
- Use plain language a non-lawyer can understand

Jurisdiction: {jurisdiction}
User Role: {user_role}
Matter Type: {matter_type}"""
    ),
    (
        "human",
        """Question: {user_query}

Web Search Results:
{context}

Provide a clear answer based on the search results."""
    ),
])

rag_chain = rag_answer_prompt | llm
web_chain = web_answer_prompt | llm
fallback_chain = rag_fallback_prompt | llm


def build_context(state: AgentState) -> str:
    """
    Builds context string based on source mode.
    """

    source_mode = state.get("source_mode", "WEB")

    if source_mode == "RAG":
        chunks = state.get("retrieved_chunks", [])
        return "\n\n---\n\n".join(chunks) if chunks else ""

    # WEB or BOTH — use web results
    web_results = state.get("web_search_results", [])
    return "\n\n".join(web_results) if web_results else ""


def answer_generator_node(state: AgentState) -> AgentState:
    """
    Generates a grounded legal answer.
    Selects the correct prompt based on source mode
    and whether RAG fell back to web.
    """

    context = build_context(state)
    source_mode = state.get("source_mode", "WEB")
    rag_fallback = state.get("rag_fallback_to_web", False)

    print(f"[ANSWER] Source mode: {source_mode}")
    print(f"[ANSWER] RAG fallback to web: {rag_fallback}")
    print(f"[ANSWER] Context length: {len(context)} chars")

    if not context.strip():
        return {
            **state,
            "generated_answer": "",
            "fallback_triggered": True,
        }

    # Select correct chain
    if rag_fallback:
        chain = fallback_chain
    elif source_mode == "RAG":
        chain = rag_chain
    else:
        chain = web_chain

    response = chain.invoke({
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
        "parametric_knowledge_used": False,
    }