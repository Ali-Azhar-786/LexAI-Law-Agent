from langgraph.graph import StateGraph, START, END
from app.graph.state import AgentState
from app.graph.nodes.clarifier import clarifier_node
from app.graph.nodes.validator import validator_node
from app.graph.nodes.decomposer import decomposer_node
from app.graph.nodes.router import router_node, route_after_router
from app.graph.nodes.rag_node import rag_node
from app.graph.nodes.web_search_node import web_search_node
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


def memory_update_node(state: AgentState) -> AgentState:
    """
    Final node in every execution path.
    Assembles the final response and updates STM.
    LTM persistence is handled separately via checkpointer.

    Args:
        state: Current AgentState.

    Returns:
        Updated state with final_response assembled.
    """

    # If final_response already set by fallback or hitl use it
    if state.get("final_response"):
        final_response = state["final_response"]
    else:
        # Assemble final response from generated answer
        parts = []

        # Main answer
        if state.get("generated_answer"):
            parts.append(state["generated_answer"])

        # Citations
        citations = state.get("citations", [])
        if citations:
            cited = ", ".join(citations)
            parts.append(f"\n📌 **Referenced:** {cited}")

        # Staleness warning
        if state.get("staleness_warning"):
            parts.append(
                "\n⚠️ **Document Freshness Warning:** "
                "Your uploaded document may be outdated. "
                f"Recent amendments may exist: "
                f"{state.get('amendment_summary', '')}"
            )

        # Parametric knowledge flag
        if state.get("parametric_knowledge_used"):
            parts.append(
                "\n🔍 **Note:** Parts of this answer are based on "
                "general training knowledge. "
                "Please verify against official sources."
            )

        final_response = "\n".join(parts)

    # Update STM with current exchange
    stm = state.get("stm", [])
    stm.append({
        "query": state.get("user_query"),
        "response": final_response,
        "jurisdiction": state.get("jurisdiction"),
        "matter_type": state.get("matter_type"),
        "confidence": state.get("confidence"),
    })

    # Update LTM profile
    ltm_profile = state.get("ltm_profile", {})
    ltm_profile["jurisdiction"] = state.get("jurisdiction")
    ltm_profile["last_matter_type"] = state.get("matter_type")
    ltm_profile["last_user_role"] = state.get("user_role")
    if state.get("uploaded_doc_path"):
        ltm_profile["last_doc_path"] = state.get("uploaded_doc_path")
        ltm_profile["last_doc_date"] = state.get("doc_date")

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
      → router ──────────────────────┐
          │ (RAG)                     │ (WEB)
          ↓                           ↓
        rag_node              web_search_node
          │                           │
          └──────────┬────────────────┘
                     ↓
              freshness_check
                     ↓
            answer_generator
                     ↓
            grounding_check
             │              │
          (HIGH)          (LOW)
             ↓              ↓
      stakes_assessor   fallback_node
       │          │          │
    (high)    (low)          │
       ↓         ↓           │
    hitl_node   memory ←─────┘
       ↓
    memory_update
       ↓
      END

    Returns:
        Compiled LangGraph StateGraph.
    """

    # ---------------------------------------------------------
    # Initialize graph with state schema
    # ---------------------------------------------------------
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
    graph.add_node("fallback_node", fallback_node)
    graph.add_node("stakes_assessor", stakes_assessor_node)
    graph.add_node("hitl_node", hitl_node)
    graph.add_node("memory_update_node", memory_update_node)

    # ---------------------------------------------------------
    # Define edges — linear flow
    # ---------------------------------------------------------
    graph.add_edge(START, "clarifier")
    graph.add_edge("clarifier", "validator")
    graph.add_edge("validator", "decomposer")
    graph.add_edge("decomposer", "router")

    # ---------------------------------------------------------
    # Conditional edge after router
    # Routes to rag_node or web_search_node
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

    # ---------------------------------------------------------
    # Linear flow through generation and grounding
    # ---------------------------------------------------------
    graph.add_edge("freshness_check", "answer_generator")
    graph.add_edge("answer_generator", "grounding_check")

    # ---------------------------------------------------------
    # Conditional edge after grounding check
    # Routes to stakes_assessor or fallback_node
    # ---------------------------------------------------------
    graph.add_conditional_edges(
        "grounding_check",
        route_after_grounding,
        {
            "stakes_assessor": "stakes_assessor",
            "fallback_node": "fallback_node",
        },
    )

    # ---------------------------------------------------------
    # Conditional edge after stakes assessor
    # Routes to hitl_node or memory_update_node
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
    # Both hitl and fallback converge at memory_update_node
    # ---------------------------------------------------------
    graph.add_edge("hitl_node", "memory_update_node")
    graph.add_edge("fallback_node", "memory_update_node")

    # ---------------------------------------------------------
    # End
    # ---------------------------------------------------------
    graph.add_edge("memory_update_node", END)

    # ---------------------------------------------------------
    # Compile and return
    # ---------------------------------------------------------
    return graph.compile()


# ---------------------------------------------------------
# Single compiled graph instance
# imported by FastAPI routes
# ---------------------------------------------------------
app_graph = build_graph()