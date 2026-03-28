from langgraph.graph import StateGraph, START, END
from app.graph.state import AgentState
from app.graph.nodes.clarifier import clarifier_node
from app.graph.nodes.validator import validator_node
from app.graph.nodes.decomposer import decomposer_node
from app.graph.nodes.router import router_node, route_after_router
from app.graph.nodes.rag_node import rag_node
from app.graph.nodes.web_search_node import (
    web_search_node,
    web_search_fallback_node,
)
from app.graph.nodes.freshness_check import freshness_check_node
from app.graph.nodes.answer_generator import answer_generator_node
from app.graph.nodes.grounding_check import (
    grounding_check_node,
    route_after_grounding,
)
from app.graph.nodes.fallback_node import fallback_node
from app.graph.nodes.stakes_assessor import (
    stakes_assessor_node,
    route_after_stakes,
)
from app.graph.nodes.hitl_node import hitl_node
from app.memory.long_term import save_to_ltm, load_from_ltm
from app.memory.short_term import add_to_stm


def memory_update_node(state: AgentState) -> AgentState:
    """
    Final node. Assembles final response, updates STM,
    saves key information to Mem0 LTM.
    """

    if state.get("final_response"):
        final_response = state["final_response"]
    else:
        parts = []

        if state.get("generated_answer"):
            parts.append(state["generated_answer"])

        citations = state.get("citations", [])
        if citations:
            cited = ", ".join(citations)
            parts.append(f"\n📌 **Referenced:** {cited}")

        if state.get("staleness_warning"):
            parts.append(
                "\n⚠️ **Document Freshness Warning:** "
                "Your uploaded document may be outdated. "
                f"{state.get('amendment_summary', '')}"
            )

        if state.get("parametric_knowledge_used"):
            parts.append(
                "\n🔍 **Note:** Parts of this answer are based on "
                "general training knowledge. Please verify against "
                "official sources."
            )

        final_response = "\n".join(parts)

    stm = state.get("stm", [])
    stm = add_to_stm(stm, {
        "query": state.get("user_query"),
        "response": final_response[:500],
        "jurisdiction": state.get("jurisdiction"),
        "matter_type": state.get("matter_type"),
        "confidence": state.get("confidence"),
    })

    ltm_data = {
        "jurisdiction": state.get("jurisdiction"),
        "last_matter_type": state.get("matter_type"),
        "last_user_role": state.get("user_role"),
        "last_doc_path": state.get("uploaded_doc_path"),
        "last_doc_date": state.get("doc_date"),
    }
    save_to_ltm(user_id=state["session_id"], data=ltm_data)
    ltm_profile = load_from_ltm(state["session_id"])

    return {
        **state,
        "final_response": final_response,
        "stm": stm,
        "ltm_profile": ltm_profile,
    }


def build_graph() -> StateGraph:
    """
    Assembles and compiles the full LangGraph agent.

    Graph flow:
    START
      → clarifier
      → validator
      → decomposer
      → router ─────────────────────────┐
          │ (RAG)                        │ (WEB)
          ↓                              ↓
        rag_node                  web_search_node
          │                              │
          └──────────┬───────────────────┘
                     ↓
              freshness_check
                     ↓
            answer_generator  ←──────────────────┐
                     ↓                            │
            grounding_check                       │
             │         │          │               │
          (HIGH)  (web_fallback) (LOW)            │
             ↓         ↓          ↓               │
      stakes_assessor  │      fallback_node        │
       │          │    │                           │
    (high)     (low)   └── web_search_fallback ───┘
       ↓          ↓         (only runs once)
    hitl_node  memory_update_node
       ↓
    memory_update_node
       ↓
      END
    """

    graph = StateGraph(AgentState)

    # ---------------------------------------------------------
    # Register all nodes
    # ---------------------------------------------------------
    graph.add_node("clarifier", clarifier_node)
    graph.add_node("validator", validator_node)
    graph.add_node("decomposer", decomposer_node)
    graph.add_node("router", router_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("web_search_node", web_search_node)
    graph.add_node("freshness_check", freshness_check_node)
    graph.add_node("answer_generator", answer_generator_node)
    graph.add_node("grounding_check", grounding_check_node)
    graph.add_node("web_search_fallback", web_search_fallback_node)  # ✅ registered
    graph.add_node("fallback_node", fallback_node)
    graph.add_node("stakes_assessor", stakes_assessor_node)
    graph.add_node("hitl_node", hitl_node)
    graph.add_node("memory_update_node", memory_update_node)

    # ---------------------------------------------------------
    # Linear flow — start to router
    # ---------------------------------------------------------
    graph.add_edge(START, "clarifier")
    graph.add_edge("clarifier", "validator")
    graph.add_edge("validator", "decomposer")
    graph.add_edge("decomposer", "router")

    # ---------------------------------------------------------
    # Conditional edge after router
    # ---------------------------------------------------------
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "rag_node": "rag_node",
            "web_search_node": "web_search_node",
        },
    )

    # ---------------------------------------------------------
    # Both retrieval paths converge at freshness_check
    # ---------------------------------------------------------
    graph.add_edge("rag_node", "freshness_check")
    graph.add_edge("web_search_node", "freshness_check")
    graph.add_edge("freshness_check", "answer_generator")
    graph.add_edge("answer_generator", "grounding_check")

    # ---------------------------------------------------------
    # Conditional edge after grounding check
    # ---------------------------------------------------------
    graph.add_conditional_edges(
        "grounding_check",
        route_after_grounding,
        {
            "stakes_assessor": "stakes_assessor",
            "web_search_fallback": "web_search_fallback",
            "fallback_node": "fallback_node",
        },
    )

    # ---------------------------------------------------------
    # Web search fallback — goes to answer_generator
    # then directly to stakes_assessor to avoid infinite loop
    # ---------------------------------------------------------
    graph.add_edge("web_search_fallback", "answer_generator")

    # ---------------------------------------------------------
    # Conditional edge after stakes assessor
    # ---------------------------------------------------------
    graph.add_conditional_edges(
        "stakes_assessor",
        route_after_stakes,
        {
            "hitl_node": "hitl_node",
            "memory_update_node": "memory_update_node",
        },
    )

    # ---------------------------------------------------------
    # All paths converge at memory_update_node
    # ---------------------------------------------------------
    graph.add_edge("hitl_node", "memory_update_node")
    graph.add_edge("fallback_node", "memory_update_node")
    graph.add_edge("memory_update_node", END)

    return graph.compile()


app_graph = build_graph()