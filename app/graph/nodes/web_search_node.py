from app.graph.state import AgentState
from app.tools.web_search import (
    search_web,
    format_search_results,
    assess_result_quality,
)


def web_search_node(state: AgentState) -> AgentState:
    """
    Performs web search for each sub-question using Tavily.
    Primary web search node used when no document is uploaded.
    """

    sub_questions = state.get(
        "sub_questions", [state["user_query"]]
    )
    jurisdiction = state.get("jurisdiction", "")

    all_results = []

    for question in sub_questions:
        query = f"{question} {jurisdiction} law".strip()
        results = search_web(query=query)
        all_results.extend(results)

    results_satisfactory = assess_result_quality(all_results)
    formatted = format_search_results(all_results)

    return {
        **state,
        "web_search_results": [formatted] if formatted else [],
        "has_confident_retrieval": results_satisfactory,
    }


def web_search_fallback_node(state: AgentState) -> AgentState:
    """
    Fallback web search triggered when RAG retrieval
    did not produce a confident grounded answer.

    Prepends a transparent note explaining that the
    uploaded document did not contain the answer,
    then searches the web and regenerates the answer.

    Args:
        state: Current AgentState.

    Returns:
        Updated state with web results added and
        source_mode switched to BOTH so answer generator
        uses web results while acknowledging the document.
    """

    print("[WEB FALLBACK] RAG confidence low — falling back to web search")

    sub_questions = state.get(
        "sub_questions", [state["user_query"]]
    )
    jurisdiction = state.get("jurisdiction", "")

    all_results = []

    for question in sub_questions:
        query = f"{question} {jurisdiction} law".strip()
        results = search_web(query=query)
        all_results.extend(results)

    results_satisfactory = assess_result_quality(all_results)
    formatted = format_search_results(all_results)

    print(f"[WEB FALLBACK] Web results satisfactory: {results_satisfactory}")

    # Switch source mode to BOTH so answer generator
    # uses web results and shows the document note
    return {
        **state,
        "web_search_results": [formatted] if formatted else [],
        "has_confident_retrieval": results_satisfactory,
        "source_mode": "BOTH",
        # Reset generated answer so it gets regenerated
        # with web context this time
        "generated_answer": "",
        "grounding_result": "",
        "confidence": "",
        # Add a flag so answer generator prepends the note
        "rag_fallback_to_web": True,
    }