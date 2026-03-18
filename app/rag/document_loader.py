import os
from pypdf import PdfReader
from langchain_core.documents import Document
from app.core.config import config


def load_pdf(file_path: str) -> list[Document]:
    """
    Loads a PDF file and converts each page into a LangChain Document.
    Extracts text and attaches page number and source path as metadata.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of LangChain Document objects, one per page.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
    """

    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Validate file type
    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"File must be a PDF: {file_path}")

    reader = PdfReader(file_path)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        # Skip empty pages
        if not text or not text.strip():
            continue

        # Clean text — remove null bytes and normalize whitespace
        text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
        text = " ".join(text.split())

        doc = Document(
            page_content=text,
            metadata={
                "source": file_path,
                "page": page_num + 1,
                "total_pages": len(reader.pages),
            },
        )
        documents.append(doc)

    return documents


def extract_doc_date(file_path: str) -> str:
    """
    Attempts to extract the version date or last amended date
    from the first two pages of a legal document.
    Returns the extracted date string or "Unknown" if not found.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Date string if found, otherwise "Unknown".
    """
    import re

    reader = PdfReader(file_path)

    # Only check first two pages — date is usually in preamble
    pages_to_check = min(2, len(reader.pages))
    text = ""

    for i in range(pages_to_check):
        page_text = reader.pages[i].extract_text()
        if page_text:
            text += page_text

    # Common patterns in legal documents
    date_patterns = [
        r"as amended (?:up to|through|on)[:\s]+([A-Za-z0-9,\s]+\d{4})",
        r"(?:last\s+)?updated[:\s]+([A-Za-z0-9,\s]+\d{4})",
        r"(?:version|edition)[:\s]+([A-Za-z0-9,\s]+\d{4})",
        r"(?:enacted|passed|adopted)[:\s]+([A-Za-z0-9,\s]+\d{4})",
        r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b",
        r"\b((?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return "Unknown"