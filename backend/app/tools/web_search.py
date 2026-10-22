"""
NEXA AI - Web Search Tool
Searches the internet for information using a search API or fallback.
"""

from __future__ import annotations
from typing import Any, Dict, List
import asyncio
import logging
import aiohttp

from app.tools.registry import BaseTool
from app.config import settings

logger = logging.getLogger("nexa.tools.web_search")


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the internet for current information on any topic. "
        "Returns a list of relevant results with titles, snippets, and URLs. "
        "Use this when you need up-to-date information or facts you're unsure about."
    )
    timeout = 10
    permission = "allowed"

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "").strip()
        num_results = min(max(input_data.get("num_results", 5), 1), 10)

        if not query:
            return {"error": "Empty search query", "results": []}

        # Try real search API if configured
        if settings.search_api_key and settings.search_api_endpoint:
            return await self._real_search(query, num_results)

        # Fallback: simulated search response
        logger.info(f"Web search (simulated): '{query}' -> {num_results} results")
        return {
            "query": query,
            "results": [
                {
                    "title": f"Search result for: {query}",
                    "snippet": (
                        f"This is a simulated search result. To enable real web search, "
                        f"configure SEARCH_API_KEY and SEARCH_API_ENDPOINT in your .env file."
                    ),
                    "url": f"https://example.com/search?q={query.replace(' ', '+')}",
                }
            ],
            "note": "Simulated results - configure search API for real results",
        }

    async def _real_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Call a real search API (Bing-compatible endpoint)."""
        headers = {
            "Ocp-Apim-Subscription-Key": settings.search_api_key,
        }
        params = {
            "q": query,
            "count": num_results,
            "mkt": "en-US",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    settings.search_api_endpoint,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        return {
                            "error": f"Search API returned status {resp.status}",
                            "results": [],
                        }
                    data = await resp.json()
                    results = []
                    for item in data.get("webPages", {}).get("value", [])[:num_results]:
                        results.append({
                            "title": item.get("name", ""),
                            "snippet": item.get("snippet", ""),
                            "url": item.get("url", ""),
                        })
                    return {"query": query, "results": results}

        except aiohttp.ClientError as e:
            logger.error(f"Search API error: {e}")
            return {"error": f"Network error: {str(e)}", "results": []}
