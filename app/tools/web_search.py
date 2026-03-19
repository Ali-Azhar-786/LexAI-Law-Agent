from tavily import TavilyClient
from app.core.config import config

# ---------------------------------------------------------
# Initialize Tavily client once at module level
# ---------------------------------------------------------
client = TavilyClient(api_key=config.TAVILY_API_KEY)


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Performs a web search using Tavily and returns
    a list of results with title, url, and content.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, url, content.
    """

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
        )

        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            })

        return results

    except Exception as e:
        print(f"Web search error: {e}")
        return []


def format_search_results(results: list[dict]) -> str:
    """
    Formats raw search results into a clean string
    that can be passed to the LLM as context.

    Args:
        results: List of result dicts from search_web().

    Returns:
        Formatted string of all search results.
    """

    if not results:
        return ""

    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(
            f"[Result {i}]\n"
            f"Title: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Content: {result['content']}\n"
        )

    return "\n".join(formatted)


def assess_result_quality(results: list[dict]) -> bool:
    """
    Assesses whether the search results are good enough
    to use for answer generation.

    A result set is considered satisfactory if:
    - At least 2 results were returned
    - At least one result has substantial content (over 100 chars)

    Args:
        results: List of result dicts from search_web().

    Returns:
        True if results are satisfactory, False otherwise.
    """

    if len(results) < 2:
        return False

    substantial = [
        r for r in results
        if len(r.get("content", "")) > 100
    ]

    return len(substantial) >= 1