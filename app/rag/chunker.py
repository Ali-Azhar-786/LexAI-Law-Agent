from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import config


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits a list of LangChain Documents into smaller chunks
    suitable for embedding and retrieval.

    Uses RecursiveCharacterTextSplitter which tries to split
    on natural boundaries (paragraphs, sentences) before
    falling back to character splits.

    Args:
        documents: List of LangChain Document objects from document_loader.

    Returns:
        List of smaller Document chunks with preserved metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    return chunks