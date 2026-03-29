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

grounding_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict legal fact-checker.

Your job is to verify whether every claim in the generated answer
is directly supported by the provided context.

Respond ONLY with a JSON object in this exact format:
{{
  "grounding_result": "GROUNDED" or "UNGROUNDED",
  "confidence": "HIGH" or "LOW",
  "unsupported_claims": ["claim 1", "claim 2"]
}}

Rules:
- grounding_result is GROUNDED only if ALL major claims are supported
- confidence is HIGH if grounding_result is GROUNDED and context is rich
- confidence is LOW if any claims are unsupported or context is thin
- unsupported_claims lists any claims not found in the context
- If all claims are supported unsupported_claims should be an empty list

No explanation. No extra text. Only the JSON object."""
    ),
    (
        "human",
        """Context:
{context}

Generated Answer:
{generated_answer}

Verify the answer against the context."""
    ),
])

grounding_chain = grounding_prompt | llm


def grounding_check_node(state: AgentState) -> AgentState:
    """
    Verifies that the generated answer is fully supported
    by the retrieved context. Sets confidence level.
    """

    if state.get("fallback_triggered"):
        return {
            **state,
            "grounding_result": "UNGROUNDED",
            "confidence": "LOW",
            "unsupported_claims": [],
        }

    context_parts = (
        state.get("retrieved_chunks", []) +
        state.get("web_search_results", [])
    )
    context = "\n\n".join(context_parts)

    if not context.strip() or not state.get("generated_answer"):
        return {
            **state,
            "grounding_result": "UNGROUNDED",
            "confidence": "LOW",
            "unsupported_claims": [],
        }

    response = grounding_chain.invoke({
        "context": context,
        "generated_answer": state["generated_answer"],
    })

    raw = response.content.strip()

    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                **state,
                "grounding_result": data.get(
                    "grounding_result", "UNGROUNDED"
                ),
                "confidence": data.get("confidence", "LOW"),
                "unsupported_claims": data.get(
                    "unsupported_claims", []
                ),
            }
    except (json.JSONDecodeError, AttributeError):
        pass

    return {
        **state,
        "grounding_result": "UNGROUNDED",
        "confidence": "LOW",
        "unsupported_claims": [],
    }


def route_after_grounding(state: AgentState) -> str:
    """
    Conditional edge after grounding check.

    HIGH confidence                      → stakes_assessor
    LOW + RAG + not yet fell back        → web_search_fallback
    LOW + already fell back              → stakes_assessor
    LOW + WEB path                       → fallback_node
    fallback explicitly triggered        → fallback_node
    """

    confidence = state.get("confidence", "LOW")
    source_mode = state.get("source_mode", "WEB")
    already_fell_back = state.get("rag_fallback_to_web", False)
    fallback_triggered = state.get("fallback_triggered", False)

    # ✅ Print statements at the TOP before any return
    print(f"[GROUNDING ROUTE] Confidence     : {confidence}")
    print(f"[GROUNDING ROUTE] Source mode    : {source_mode}")
    print(f"[GROUNDING ROUTE] Already fell   : {already_fell_back}")
    print(f"[GROUNDING ROUTE] Fallback flag  : {fallback_triggered}")

    if fallback_triggered:
        print("[GROUNDING ROUTE] → fallback_node (explicitly triggered)")
        return "fallback_node"

    if confidence == "HIGH":
        print("[GROUNDING ROUTE] → stakes_assessor")
        return "stakes_assessor"

    if source_mode == "RAG" and not already_fell_back:
        print("[GROUNDING ROUTE] → web_search_fallback (first attempt)")
        return "web_search_fallback"

    if already_fell_back:
        print("[GROUNDING ROUTE] → stakes_assessor (after web fallback)")
        return "stakes_assessor"

    print("[GROUNDING ROUTE] → fallback_node (web path low confidence)")
    return "fallback_node"