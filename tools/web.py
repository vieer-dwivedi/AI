import requests

USER_AGENT = "SmolLM-Agent/1.0"
TIMEOUT = 10


def web_search(query: str, max_results: int = 5) -> str:
    """
    Perform a simple web search using DuckDuckGo Instant Answer API.
    No API key required.
    """

    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
        }

        headers = {
            "User-Agent": USER_AGENT
        }

        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()

        data = response.json()

        results = []

        # Abstract answer
        if data.get("AbstractText"):
            results.append(data["AbstractText"])

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(topic["Text"])

        if not results:
            return "No relevant results found."

        return "\n".join(results)

    except Exception as e:
        return f"Web search error: {e}"
