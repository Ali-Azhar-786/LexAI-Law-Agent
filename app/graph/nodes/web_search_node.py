from app.graph.state import AgentState
from app.tools.web_search import (
    search_web,
    format_search_results,
    assess_result_quality,
)


def web_search_node(state: AgentState) -> AgentState:
    """
    Performs web search for each sub-question using Tavily.
    Combines results into a formatted context string.

    Args:
        state: Current AgentState.

    Returns:
        Updated state with web_search_results populated.
    """

    sub_questions = state.get(
        "sub_questions", [state["user_query"]]
    )
    jurisdiction = state.get("jurisdiction", "")

    all_results = []

    for question in sub_questions:
        # Add jurisdiction to query for more relevant results
        query = f"{question} {jurisdiction} law".strip()
        results = search_web(query=query)
        all_results.extend(results)

    # Assess overall quality of results
    results_satisfactory = assess_result_quality(all_results)

    # Format results into a single context string
    formatted = format_search_results(all_results)

    return {
        **state,
        "web_search_results": [formatted] if formatted else [],
        "has_confident_retrieval": results_satisfactory,
    }