from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from app.core.config import config


def retrieve_with_scores(
    vector_store: QdrantVectorStore,
    query: str,
    top_k: int = None,
) -> tuple[list[Document], list[float]]:
    """
    Retrieves most relevant chunks with relevance scores.
    Qdrant with COSINE distance returns proper similarity
    scores between 0 and 1. Higher is more relevant.
    """

    if top_k is None:
        top_k = config.RETRIEVAL_TOP_K

    results = vector_store.similarity_search_with_relevance_scores(
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
    Filters chunks to only those above confidence threshold.
    With Qdrant COSINE distance, higher score = more similar.
    Threshold of 0.30 keeps anything reasonably relevant.
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
    Extracts article and section references from chunks.
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
        for pattern in patterns:
            matches = re.findall(
                pattern, doc.page_content, re.IGNORECASE
            )
            citations.extend(matches)

    seen = set()
    unique_citations = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique_citations.append(c)

    return unique_citations