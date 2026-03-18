from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    """
    Request body for the /chat endpoint.
    Sent by Streamlit frontend to FastAPI backend.
    """
    session_id: str                         # Unique session identifier
    user_query: str                         # User's legal question
    jurisdiction: Optional[str] = ""        # Country/state
    user_role: Optional[str] = ""           # tenant, employee, accused, etc.
    matter_type: Optional[str] = ""         # civil, criminal, labor, etc.
    uploaded_doc_path: Optional[str] = None # Path to uploaded doc if any


class ChatResponse(BaseModel):
    """
    Response body returned by the /chat endpoint.
    Received by Streamlit frontend from FastAPI backend.
    """
    session_id: str
    answer: str                                     # Main legal explanation
    citations: List[str] = []                       # Article references
    staleness_warning: bool = False                 # Doc may be outdated
    amendment_note: Optional[str] = None            # Recent amendments found
    confidence: str = "HIGH"                        # HIGH or LOW
    escalation_advice: Optional[str] = None         # Lawyer recommendation
    lawyer_questions: Optional[List[str]] = None    # Questions for lawyer
    fallback_triggered: bool = False                # Honest fallback used
    parametric_knowledge_used: bool = False         # Model memory used


class DocumentUploadResponse(BaseModel):
    """
    Response body returned by the /upload endpoint.
    """
    success: bool
    message: str
    doc_path: Optional[str] = None
    doc_date: Optional[str] = None