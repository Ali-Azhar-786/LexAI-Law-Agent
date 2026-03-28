from typing import TypedDict, Optional, List


class AgentState(TypedDict):
    """
    Central state object passed between all LangGraph nodes.
    Every node reads from and writes to this state.
    """

    # ---------------------------------------------------------
    # USER INPUT
    # ---------------------------------------------------------
    user_query: str
    uploaded_doc_path: Optional[str]

    # ---------------------------------------------------------
    # CLARIFICATION CONTEXT
    # Collected by Context Clarifier node
    # ---------------------------------------------------------
    jurisdiction: str
    user_role: str
    matter_type: str

    # ---------------------------------------------------------
    # DOCUMENT METADATA
    # Populated by Document Validator node
    # ---------------------------------------------------------
    doc_date: Optional[str]
    doc_validated: bool

    # ---------------------------------------------------------
    # QUERY PROCESSING
    # Populated by Query Decomposer node
    # ---------------------------------------------------------
    sub_questions: List[str]
    source_mode: str            # "RAG" | "WEB" | "BOTH"

    # ---------------------------------------------------------
    # RETRIEVAL RESULTS
    # Populated by RAG Node and Web Search Node
    # ---------------------------------------------------------
    retrieved_chunks: List[str]
    retrieval_scores: List[float]
    has_confident_retrieval: bool
    web_search_results: List[str]


    # ---------------------------------------------------------
    # FRESHNESS CHECK
    # Populated by Freshness Check node
    # ---------------------------------------------------------
    staleness_warning: bool
    amendment_summary: Optional[str]

    # ---------------------------------------------------------
    # ANSWER GENERATION
    # Populated by Answer Generator node
    # ---------------------------------------------------------
    generated_answer: str
    citations: List[str]

    # ---------------------------------------------------------
    # GROUNDING AND CONFIDENCE
    # Populated by Grounding Check node
    # ---------------------------------------------------------
    grounding_result: str           # "GROUNDED" | "UNGROUNDED"
    confidence: str                 # "HIGH" | "LOW"
    unsupported_claims: List[str]

    # ---------------------------------------------------------
    # STAKES AND HITL
    # Populated by Stakes Assessor and HITL nodes
    # ---------------------------------------------------------
    is_high_stakes: bool
    escalation_message: Optional[str]
    lawyer_questions: Optional[List[str]]

    # ---------------------------------------------------------
    # FALLBACK
    # Populated by Fallback node
    # ---------------------------------------------------------
    fallback_triggered: bool
    parametric_knowledge_used: bool

    # ---------------------------------------------------------
    # RAG FALLBACK FLAG
    # Set when RAG path falls back to web search
    # ---------------------------------------------------------
    rag_fallback_to_web: bool

    # ---------------------------------------------------------
    # MEMORY
    # Managed by Memory Update node
    # ---------------------------------------------------------
    session_id: str
    stm: List[dict]
    ltm_profile: dict

    # ---------------------------------------------------------
    # FINAL OUTPUT
    # ---------------------------------------------------------
    final_response: str