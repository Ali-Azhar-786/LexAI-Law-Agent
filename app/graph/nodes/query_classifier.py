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

classifier_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query classifier for a legal assistant AI system.

Your job is to determine whether a user query is related to legal 
matters or not.

A query IS legal if it involves:
- Laws, acts, statutes, regulations, or legal codes
- Constitutional rights or provisions
- Criminal, civil, family, labor, property, or consumer law
- Courts, judges, lawyers, legal procedures
- Arrests, charges, bail, trials, sentences
- Contracts, agreements, legal obligations
- Rights and duties of citizens
- Government policies with legal implications
- Legal definitions or interpretations

A query is NOT legal if it involves:
- General knowledge questions unrelated to law
- Technology, science, mathematics, history
- How to build software, models, or systems
- Cooking, sports, entertainment, art
- Nonsensical or meaningless queries
- Personal opinions or general advice unrelated to law

Respond ONLY with a JSON object in this exact format:
{{"is_legal": true or false, "reason": "one sentence explanation"}}

No extra text. Only the JSON object."""
    ),
    (
        "human",
        "Query: {user_query}"
    ),
])

classifier_chain = classifier_prompt | llm

NON_LEGAL_RESPONSE = """I appreciate you reaching out, but I am a specialized 
legal assistant and can only help with legal matters.

Your query **"{query}"** does not appear to be related to law or legal issues.

**I can help you with questions about:**
- Your constitutional rights and fundamental rights
- Criminal law — arrests, charges, bail, trials
- Civil law — contracts, disputes, property
- Labor law — employment rights, termination, wages
- Family law — divorce, custody, inheritance
- Tenancy law — landlord-tenant disputes
- Consumer protection rights
- Court procedures and legal processes

Please feel free to ask a legal question and I will do my best to help you."""


def query_classifier_node(state: AgentState) -> AgentState:
    """
    Classifies whether the user query is legal in nature.
    Sets is_legal_query flag in state.
    Non-legal queries get a polite rejection response immediately.
    """

    user_query = state.get("user_query", "")
    print(f"[CLASSIFIER] Classifying query: {user_query}")

    try:
        response = classifier_chain.invoke({
            "user_query": user_query
        })

        raw = response.content.strip()
        match = re.search(r'\{.*?\}', raw, re.DOTALL)

        if match:
            data = json.loads(match.group())
            is_legal = data.get("is_legal", False)
            reason = data.get("reason", "")

            print(f"[CLASSIFIER] Is legal: {is_legal}")
            print(f"[CLASSIFIER] Reason: {reason}")

            if not is_legal:
                rejection = NON_LEGAL_RESPONSE.format(
                    query=user_query
                )
                return {
                    **state,
                    "is_legal_query": False,
                    "final_response": rejection,
                    "confidence": "HIGH",
                    "fallback_triggered": False,
                }

            return {
                **state,
                "is_legal_query": True,
            }

    except Exception as e:
        print(f"[CLASSIFIER] Error: {e} — defaulting to legal")

    # If classification fails default to treating as legal
    # Better to answer a non-legal query than reject a legal one
    return {
        **state,
        "is_legal_query": True,
    }


def route_after_classifier(state: AgentState) -> str:
    """
    Conditional edge after query classifier.
    Routes non-legal queries directly to memory_update_node
    which will return the rejection response.
    Routes legal queries to clarifier.
    """

    if state.get("is_legal_query", True):
        print("[CLASSIFIER] → clarifier (legal query)")
        return "clarifier"

    print("[CLASSIFIER] → memory_update_node (non-legal query)")
    return "memory_update_node"