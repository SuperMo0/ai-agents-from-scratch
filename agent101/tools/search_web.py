from typing import Any

from clients import tavily_client
from tools.tool import Tool

SEARCH_WEB_SCHEMA: dict[str, Any] = dict(
    name="search_web",
    description="Search the web for a query and return a summary of top results with their sources.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."}
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


def search_web(query: str) -> str:
    response = tavily_client.search(query=query, search_depth="basic", max_results=3)
    results = response.get("results", [])
    if not results:
        return f"No results found for '{query}'."
    return "\n".join(
        f"- {r.get('title', '')}: {r.get('content', '')} (source: {r.get('url', '')})"
        for r in results
    )


search_web_tool = Tool(fn=search_web, **SEARCH_WEB_SCHEMA)
