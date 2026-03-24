import streamlit as st
import requests
import uuid
import os

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
API_BASE_URL = "http://127.0.0.1:8000/api/v1"

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="LexAI — Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Session state initialization
# Persists across reruns within the same browser session
# ---------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_doc_path" not in st.session_state:
    st.session_state.uploaded_doc_path = None

if "doc_info" not in st.session_state:
    st.session_state.doc_info = None


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def upload_document(file, session_id: str) -> dict:
    """
    Sends PDF file to FastAPI /upload endpoint.
    Returns response dict with success status and doc metadata.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/upload",
            params={"session_id": session_id},
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            timeout=60,
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "message": "Could not connect to backend. "
                       "Make sure the FastAPI server is running.",
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


def send_chat_message(
    query: str,
    session_id: str,
    jurisdiction: str,
    user_role: str,
    matter_type: str,
    doc_path: str = None,
) -> dict:
    """
    Sends chat request to FastAPI /chat endpoint.
    Returns response dict with answer and metadata.
    """
    try:
        payload = {
            "session_id": session_id,
            "user_query": query,
            "jurisdiction": jurisdiction,
            "user_role": user_role,
            "matter_type": matter_type,
            "uploaded_doc_path": doc_path,
        }
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=120,
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "error": "Could not connect to backend. "
                     "Make sure the FastAPI server is running."
        }
    except Exception as e:
        return {"error": str(e)}


def render_response(response: dict):
    """
    Renders a structured ChatResponse in the Streamlit UI
    with appropriate formatting and color coding.
    """

    # Main answer
    st.markdown("### ⚖️ Legal Guidance")
    st.markdown(response.get("answer", "No answer received."))

    # Citations
    citations = response.get("citations", [])
    if citations:
        st.markdown("---")
        st.markdown("### 📌 Referenced Articles")
        for citation in citations:
            st.markdown(f"- {citation}")

    # Confidence indicator
    confidence = response.get("confidence", "LOW")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if confidence == "HIGH":
            st.success("✅ Confidence: HIGH")
        else:
            st.warning("⚠️ Confidence: LOW")

    with col2:
        if response.get("fallback_triggered"):
            st.error("❌ Fallback Triggered")
        else:
            st.success("✅ Answer Found")

    with col3:
        if response.get("staleness_warning"):
            st.warning("⚠️ Document May Be Outdated")
        else:
            st.success("✅ Document Fresh")

    # Staleness warning
    if response.get("staleness_warning"):
        st.markdown("---")
        st.warning(
            "⚠️ **Document Freshness Warning**\n\n"
            "Your uploaded document may not reflect the most recent "
            "legal amendments. Please verify critical information "
            "against the latest official sources."
        )
        if response.get("amendment_note"):
            with st.expander("View Recent Amendment Details"):
                st.markdown(response["amendment_note"])

    # Lawyer questions
    lawyer_questions = response.get("lawyer_questions")
    if lawyer_questions:
        st.markdown("---")
        st.markdown("### 👨‍⚖️ Questions to Ask Your Lawyer")
        for i, question in enumerate(lawyer_questions, 1):
            st.markdown(f"**{i}.** {question}")

    # Parametric knowledge warning
    if response.get("parametric_knowledge_used"):
        st.markdown("---")
        st.info(
            "🔍 **Note:** Parts of this answer are based on general "
            "AI training knowledge. Please verify against official "
            "legal sources before taking action."
        )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/scales-emoji.png", width=80)
    st.title("LexAI")
    st.caption("Your Agentic Legal Assistant")
    st.markdown("---")

    # Session info
    st.markdown("### 🔑 Session")
    st.code(st.session_state.session_id[:16] + "...", language=None)

    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.uploaded_doc_path = None
        st.session_state.doc_info = None
        st.rerun()

    st.markdown("---")

    # Context inputs
    st.markdown("### 🌍 Your Context")
    st.caption(
        "Providing context helps the agent give more accurate answers. "
        "You can also leave these blank and describe your situation "
        "in your question."
    )

    jurisdiction = st.text_input(
        "Jurisdiction (Country/State)",
        placeholder="e.g. Pakistan, India - Maharashtra",
        key="jurisdiction",
    )

    user_role = st.selectbox(
        "Your Role",
        options=[
            "",
            "citizen",
            "tenant",
            "employee",
            "employer",
            "accused",
            "plaintiff",
            "defendant",
            "student",
            "other",
        ],
        key="user_role",
    )

    matter_type = st.selectbox(
        "Matter Type",
        options=[
            "",
            "civil",
            "criminal",
            "labor",
            "family",
            "property",
            "constitutional",
            "consumer",
            "other",
        ],
        key="matter_type",
    )

    st.markdown("---")

    # Document upload
    st.markdown("### 📄 Upload Legal Document")
    st.caption(
        "Upload a PDF of your country's constitution, "
        "civil code, or relevant law for grounded answers."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="file_uploader",
    )

    if uploaded_file is not None:
        if st.button(
            "📤 Process Document",
            use_container_width=True,
        ):
            with st.spinner("Processing document..."):
                result = upload_document(
                    file=uploaded_file,
                    session_id=st.session_state.session_id,
                )

            if result.get("success"):
                st.session_state.uploaded_doc_path = result.get(
                    "doc_path"
                )
                st.session_state.doc_info = result
                st.success(result.get("message", "Document processed."))
            else:
                st.error(result.get("message", "Upload failed."))

    # Show document status
    if st.session_state.doc_info:
        st.markdown("**📋 Document Status**")
        st.success("✅ Document Ready")
        doc_date = st.session_state.doc_info.get("doc_date", "Unknown")
        st.caption(f"Document date: {doc_date}")

    st.markdown("---")
    st.caption(
        "⚠️ LexAI provides general legal information only. "
        "It is not a substitute for professional legal advice. "
        "Always consult a licensed lawyer for your specific situation."
    )


# ---------------------------------------------------------
# Main content area
# ---------------------------------------------------------
st.title("⚖️ LexAI — Agentic Law Assistant")
st.caption(
    "Ask any question about your legal rights and applicable laws. "
    "Upload a legal document for grounded answers, "
    "or ask freely and the agent will search the web."
)

st.markdown("---")

# ---------------------------------------------------------
# Chat history display
# ---------------------------------------------------------
if st.session_state.chat_history:
    for exchange in st.session_state.chat_history:
        # User message
        with st.chat_message("user"):
            st.markdown(exchange["query"])

        # Assistant response
        with st.chat_message("assistant"):
            render_response(exchange["response"])
else:
    st.info(
        "👋 Welcome to LexAI. "
        "Type your legal question below to get started. "
        "You can also upload a legal document in the sidebar "
        "for more accurate, document-grounded answers."
    )

# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------
user_query = st.chat_input(
    "Ask your legal question here...",
)

if user_query:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # Send to backend and display response
    with st.chat_message("assistant"):
        with st.spinner("Consulting legal sources..."):
            response = send_chat_message(
                query=user_query,
                session_id=st.session_state.session_id,
                jurisdiction=st.session_state.get("jurisdiction", ""),
                user_role=st.session_state.get("user_role", ""),
                matter_type=st.session_state.get("matter_type", ""),
                doc_path=st.session_state.uploaded_doc_path,
            )

        # Handle backend error
        if "error" in response:
            st.error(response["error"])
        else:
            render_response(response)

            # Save to chat history
            st.session_state.chat_history.append({
                "query": user_query,
                "response": response,
            })