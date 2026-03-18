from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.core.config import config


def retrieve_with_scores(
    vector_store: FAISS,
    query: str,
    top_k: int = None,
) -> tuple[list[Document], list[float]]:
    """
    Retrieves the most relevant document chunks for a query
    along with their similarity scores.

    Args:
        vector_store: The FAISS vector store to search.
        query: The search query string.
        top_k: Number of results to return. Defaults to config value.

    Returns:
        Tuple of (documents list, scores list) sorted by relevance.
    """

    if top_k is None:
        top_k = config.RETRIEVAL_TOP_K

    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    documents = [doc for doc, score in results]
    scores = [float(score) for doc, score in results]

    return documents, scores


def filter_by_confidence(
    documents: list[Document],
    scores: list[float],
    threshold: float = None,
) -> tuple[list[Document], list[float]]:
    """
    Filters retrieved chunks to only keep those above
    the confidence threshold.

    Args:
        documents: List of retrieved documents.
        scores: Corresponding similarity scores.
        threshold: Minimum score to keep. Defaults to config value.

    Returns:
        Tuple of (filtered documents, filtered scores).
    """

    if threshold is None:
        threshold = config.CONFIDENCE_THRESHOLD

    filtered = [
        (doc, score)
        for doc, score in zip(documents, scores)
        if score >= threshold
    ]

    if not filtered:
        return [], []

    filtered_docs, filtered_scores = zip(*filtered)
    return list(filtered_docs), list(filtered_scores)


def extract_citations(documents: list[Document]) -> list[str]:
    """
    Extracts article and section references from retrieved chunks
    to use as citations in the final response.

    Args:
        documents: List of retrieved Document chunks.

    Returns:
        List of citation strings found in the chunks.
    """

    import re
    citations = []

    patterns = [
        r"Article\s+\d+[A-Za-z]?(?:\(\d+\))?",
        r"Section\s+\d+[A-Za-z]?(?:\(\d+\))?",
        r"Clause\s+\d+[A-Za-z]?(?:\(\d+\))?",
        r"Part\s+[IVXLC]+",
        r"Schedule\s+\d+",
    ]

    for doc in documents:
        text = doc.page_content
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)

    # Deduplicate while preserving order
    seen = set()
    unique_citations = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique_citations.append(c)

    return unique_citations