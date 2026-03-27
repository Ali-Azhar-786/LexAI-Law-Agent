import os
import re
from app.graph.state import AgentState
from app.rag.document_loader import load_pdf, extract_doc_date
from app.rag.chunker import chunk_documents
from app.rag.embedder import create_vector_store

LEGAL_SIGNALS = [
    r"\barticle\s+\d+",
    r"\bsection\s+\d+",
    r"\bclause\s+\d+",
    r"\bwhereas\b",
    r"\bhereinafter\b",
    r"\bconstitution\b",
    r"\blegislation\b",
    r"\bstatute\b",
    r"\bact\s+of\b",
    r"\bprovision\b",
    r"\bfundamental rights\b",
    r"\bjurisdiction\b",
]


def validate_legal_document(text: str) -> bool:
    """
    Checks whether the document text contains enough
    legal signals to be considered a valid legal document.
    """
    text_lower = text.lower()
    matches = sum(
        1 for pattern in LEGAL_SIGNALS
        if re.search(pattern, text_lower)
    )
    return matches >= 3


def validator_node(state: AgentState) -> AgentState:
    """
    Validates the uploaded document, extracts its date,
    chunks it, and indexes it into Qdrant.
    """

    # No document uploaded — skip
    if not state.get("uploaded_doc_path"):
        print("[VALIDATOR] No document path in state — skipping")
        return {
            **state,
            "doc_validated": False,
            "doc_date": None,
        }

    file_path = state["uploaded_doc_path"]

    # Check file exists on disk
    if not os.path.exists(file_path):
        print(f"[VALIDATOR] File not found on disk: {file_path}")
        return {
            **state,
            "doc_validated": False,
            "doc_date": None,
        }

    try:
        print(f"[VALIDATOR] Loading PDF: {file_path}")
        documents = load_pdf(file_path)

        if not documents:
            print("[VALIDATOR] No pages extracted from PDF")
            return {
                **state,
                "doc_validated": False,
                "doc_date": None,
            }

        print(f"[VALIDATOR] Loaded {len(documents)} pages")

        # Validate using first two pages
        sample_text = " ".join(
            doc.page_content for doc in documents[:2]
        )
        is_valid = validate_legal_document(sample_text)
        print(f"[VALIDATOR] Legal document check: {is_valid}")

        if not is_valid:
            print("[VALIDATOR] Document failed legal signal check")
            return {
                **state,
                "doc_validated": False,
                "doc_date": None,
            }

        # Extract document date
        doc_date = extract_doc_date(file_path)
        print(f"[VALIDATOR] Document date: {doc_date}")

        # Chunk documents
        chunks = chunk_documents(documents)
        print(f"[VALIDATOR] Created {len(chunks)} chunks")

        # ✅ FIXED: create_vector_store now takes session_id
        # and handles Qdrant indexing internally
        # No separate save_vector_store call needed
        create_vector_store(chunks, state["session_id"])
        print(f"[VALIDATOR] Indexed into Qdrant for "
              f"session: {state['session_id']}")

        return {
            **state,
            "doc_validated": True,
            "doc_date": doc_date,
        }

    except Exception as e:
        # ✅ Now prints the REAL error instead of silently failing
        print(f"[VALIDATOR] ERROR: {type(e).__name__}: {e}")
        return {
            **state,
            "doc_validated": False,
            "doc_date": None,
        }