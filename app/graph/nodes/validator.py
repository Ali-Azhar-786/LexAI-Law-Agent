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
    r"\bpreamble\b",
    r"\brepublic\b",
    r"\bparliament\b",
    r"\bgovernment\b",
    r"\bcourt\b",
    r"\bjudiciary\b",
    r"\blegislature\b",
    r"\bexecutive\b",
]


def validate_legal_document(text: str) -> bool:
    """
    Checks whether document text contains legal signals.
    Requires at least 2 matches from the signal list.
    Threshold lowered from 3 to 2 for robustness.
    """
    text_lower = text.lower()
    matches = sum(
        1 for pattern in LEGAL_SIGNALS
        if re.search(pattern, text_lower)
    )
    print(f"[VALIDATOR] Legal signals found: {matches}")
    return matches >= 2


def validator_node(state: AgentState) -> AgentState:
    """
    Validates uploaded document and indexes into Qdrant.
    """

    if not state.get("uploaded_doc_path"):
        print("[VALIDATOR] No document path in state — skipping")
        return {
            **state,
            "doc_validated": False,
            "doc_date": None,
        }

    file_path = state["uploaded_doc_path"]

    if not os.path.exists(file_path):
        print(f"[VALIDATOR] File not found: {file_path}")
        return {
            **state,
            "doc_validated": False,
            "doc_date": None,
        }

    try:
        print(f"[VALIDATOR] Loading PDF: {file_path}")
        documents = load_pdf(file_path)

        if not documents:
            print("[VALIDATOR] No pages extracted")
            return {
                **state,
                "doc_validated": False,
                "doc_date": None,
            }

        print(f"[VALIDATOR] Loaded {len(documents)} pages")

        # Check first 5 pages instead of 2
        # Table of contents pages may not have enough signals
        pages_to_check = min(5, len(documents))
        sample_text = " ".join(
            doc.page_content for doc in documents[:pages_to_check]
        )

        is_valid = validate_legal_document(sample_text)
        print(f"[VALIDATOR] Validation result: {is_valid}")

        if not is_valid:
            print("[VALIDATOR] Failed legal signal check")
            return {
                **state,
                "doc_validated": False,
                "doc_date": None,
            }

        doc_date = extract_doc_date(file_path)
        print(f"[VALIDATOR] Document date: {doc_date}")

        chunks = chunk_documents(documents)
        print(f"[VALIDATOR] Created {len(chunks)} chunks")

        create_vector_store(chunks, state["session_id"])
        print(f"[VALIDATOR] Indexed into Qdrant: {state['session_id']}")

        return {
            **state,
            "doc_validated": True,
            "doc_date": doc_date,
        }

    except Exception as e:
        print(f"[VALIDATOR] ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "doc_validated": False,
            "doc_date": None,
        }