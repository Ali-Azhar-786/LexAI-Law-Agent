import os
import re
from pypdf import PdfReader
from langchain_core.documents import Document
from app.graph.state import AgentState
from app.rag.document_loader import load_pdf, extract_doc_date
from app.rag.chunker import chunk_documents
from app.rag.embedder import create_vector_store, save_vector_store

# Legal document signals — presence of these suggests a valid legal document
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

    Args:
        text: Raw text extracted from the document.

    Returns:
        True if document appears to be a legal document.
    """

    text_lower = text.lower()
    matches = sum(
        1 for pattern in LEGAL_SIGNALS
        if re.search(pattern, text_lower)
    )

    # Require at least 3 legal signals
    return matches >= 3


def validator_node(state: AgentState) -> AgentState:
    """
    Validates the uploaded document, extracts its date,
    chunks it, embeds it, and saves to FAISS vector store.

    If no document is uploaded this node is skipped.

    Args:
        state: Current AgentState.

    Returns:
        Updated state with doc_validated, doc_date set.
        Vector store saved to disk under session_id.
    """

    # No document uploaded — skip validation
    if not state.get("uploaded_doc_path"):
        return {
            **state,
            "doc_validated": False,
            "doc_date": None,
        }

    file_path = state["uploaded_doc_path"]

    # Check file exists
    if not os.path.exists(file_path):
        return {
            **state,
            "doc_validated": False,
            "doc_date": None,
        }

    try:
        # Load document pages
        documents = load_pdf(file_path)

        if not documents:
            return {
                **state,
                "doc_validated": False,
                "doc_date": None,
            }

        # Combine first two pages for validation check
        sample_text = " ".join(
            doc.page_content for doc in documents[:2]
        )

        # Validate it looks like a legal document
        is_valid = validate_legal_document(sample_text)

        if not is_valid:
            return {
                **state,
                "doc_validated": False,
                "doc_date": None,
            }

        # Extract document date
        doc_date = extract_doc_date(file_path)

        # Chunk and embed
        chunks = chunk_documents(documents)
        vector_store = create_vector_store(chunks)

        # Save vector store under session_id
        save_vector_store(vector_store, state["session_id"])

        return {
            **state,
            "doc_validated": True,
            "doc_date": doc_date,
        }

    except Exception as e:
        print(f"Document validation error: {e}")
        return {
            **state,
            "doc_validated": False,
            "doc_date": None,
        }