from datetime import datetime
from app.graph.state import AgentState
from app.tools.web_search import search_web, format_search_results


def freshness_check_node(state: AgentState) -> AgentState:
    """
    Checks whether the uploaded document may be outdated
    by searching for recent amendments to the retrieved articles.

    Only runs on the RAG path.

    Args:
        state: Current AgentState.

    Returns:
        Updated state with staleness_warning and
        amendment_summary populated.
    """

    # Only relevant when a document was used
    if state.get("source_mode") == "WEB":
        return {
            **state,
            "staleness_warning": False,
            "amendment_summary": None,
        }

    doc_date = state.get("doc_date", "Unknown")
    jurisdiction = state.get("jurisdiction", "")
    citations = state.get("citations", [])
    current_year = datetime.now().year

    # Build amendment search query
    if citations:
        articles_str = ", ".join(citations[:3])
        query = (
            f"recent amendments to {articles_str} "
            f"{jurisdiction} law {current_year}"
        )
    else:
        query = (
            f"recent changes to {jurisdiction} "
            f"constitution law {current_year}"
        )

    results = search_web(query=query, max_results=3)

    if not results:
        return {
            **state,
            "staleness_warning": False,
            "amendment_summary": None,
        }

    # Check if results mention amendments after document date
    combined_text = " ".join(
        r.get("content", "") for r in results
    ).lower()

    amendment_keywords = [
        "amended", "amendment", "updated", "revised",
        "new law", "recent change", "modification",
        str(current_year), str(current_year - 1),
    ]

    found_amendments = any(
        keyword in combined_text
        for keyword in amendment_keywords
    )

    if found_amendments:
        amendment_summary = format_search_results(results[:2])
        return {
            **state,
            "staleness_warning": True,
            "amendment_summary": amendment_summary,
        }

    return {
        **state,
        "staleness_warning": False,
        "amendment_summary": None,
    }