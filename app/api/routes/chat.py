import uuid
from fastapi import APIRouter, HTTPException
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.graph.graph_builder import app_graph

router = APIRouter()


def build_initial_state(request: ChatRequest) -> dict:
    """
    Builds the initial AgentState from a ChatRequest.
    Sets all required fields with safe defaults.

    Args:
        request: Incoming ChatRequest from frontend.

    Returns:
        Dictionary matching AgentState TypedDict structure.
    """

    return {
        # User input
        "user_query": request.user_query,
        "uploaded_doc_path": request.uploaded_doc_path,

        # Clarification — use provided values if available
        "jurisdiction": request.jurisdiction or "",
        "user_role": request.user_role or "",
        "matter_type": request.matter_type or "",

        # Document metadata — populated by validator node
        "doc_date": None,
        "doc_validated": False,

        # Query processing — populated by decomposer node
        "sub_questions": [],
        "source_mode": "",

        # Retrieval — populated by rag/web nodes
        "retrieved_chunks": [],
        "retrieval_scores": [],
        "has_confident_retrieval": False,
        "web_search_results": [],

        # Freshness — populated by freshness check node
        "staleness_warning": False,
        "amendment_summary": None,

        # Generation — populated by answer generator node
        "generated_answer": "",
        "citations": [],

        # Grounding — populated by grounding check node
        "grounding_result": "",
        "confidence": "",
        "unsupported_claims": [],

        # Stakes and HITL
        "is_high_stakes": False,
        "escalation_message": None,
        "lawyer_questions": None,

        # Fallback
        "fallback_triggered": False,
        "parametric_knowledge_used": False,
        "rag_fallback_to_web": False,

        # Memory
        "session_id": request.session_id,
        "stm": [],
        "ltm_profile": {},

        # Final output
        "final_response": "",
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Receives a user query, runs it
    through the full LangGraph agent, and returns
    a structured response.

    Args:
        request: ChatRequest with query and optional context.

    Returns:
        ChatResponse with answer, citations, and metadata.
    """

    try:
        # Build initial state
        initial_state = build_initial_state(request)

        # Run the full agent graph
        result = app_graph.invoke(initial_state)

        # Build and return response
        return ChatResponse(
            session_id=request.session_id,
            answer=result.get("final_response", ""),
            citations=result.get("citations", []),
            staleness_warning=result.get("staleness_warning", False),
            amendment_note=result.get("amendment_summary"),
            confidence=result.get("confidence", "LOW"),
            escalation_advice=result.get("escalation_message"),
            lawyer_questions=result.get("lawyer_questions"),
            fallback_triggered=result.get("fallback_triggered", False),
            parametric_knowledge_used=result.get(
                "parametric_knowledge_used", False
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}",
        )