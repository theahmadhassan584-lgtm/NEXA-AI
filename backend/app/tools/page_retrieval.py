"""
NEXA AI - Page Retrieval Tool
Fetches and extracts text content from a given URL.
Enforces HTTPS-only and domain blocklist.
"""

from __future__ import annotations
from typing import Any, Dict
from urllib.parse import urlparse
import logging
import re

import aiohttp
from bs4 import BeautifulSoup

from app.tools.registry import BaseTool
from app.config import settings

logger = logging.getLogger("nexa.tools.page_retrieval")

# Maximum content length to return (characters)
MAX_CONTENT_LENGTH = 15_000


class PageRetrievalTool(BaseTool):
    name = "page_retrieval"
    description = (
        "Retrieve and read the text content from a web page URL. "
        "Returns the main text content extracted from the page. "
        "Only HTTPS URLs are allowed. Use this to read articles, documentation, "
        "or any publicly accessible web page."
    )
    timeout = 15
    permission = "allowed"

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The HTTPS URL to retrieve content from",
                },
                "extract_mode": {
                    "type": "string",
                    "enum": ["text", "html", "markdown"],
                    "description": "Output format: 'text' (plain text), 'html' (raw HTML), 'markdown' (simplified)",
                    "default": "text",
                },
            },
            "required": ["url"],
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        url = input_data.get("url", "").strip()
        mode = input_data.get("extract_mode", "text")

        # Validate URL
        validation = self._validate_url(url)
        if validation:
            return {"error": validation}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=12),
                    headers={
                        "User-Agent": "NEXA-AI-Agent/1.0 (Page Retrieval Tool)"
                    },
                    allow_redirects=True,
                    max_redirects=3,
                ) as resp:
                    if resp.status != 200:
                        return {
                            "error": f"HTTP {resp.status}: Failed to retrieve {url}",
                            "status_code": resp.status,
                        }

                    content_type = resp.headers.get("Content-Type", "")
                    if "text/html" not in content_type and "text/plain" not in content_type:
                        return {
                            "error": f"Unsupported content type: {content_type}",
                            "url": url,
                        }

                    raw_html = await resp.text(encoding="utf-8", errors="replace")

            # Extract content
            if mode == "html":
                content = raw_html[:MAX_CONTENT_LENGTH]
            else:
                content = self._extract_text(raw_html)

            return {
                "url": url,
                "content": content,
                "length": len(content),
                "truncated": len(content) >= MAX_CONTENT_LENGTH,
            }

        except aiohttp.ClientError as e:
            logger.error(f"Page retrieval failed for {url}: {e}")
            return {"error": f"Network error: {str(e)}", "url": url}

    def _validate_url(self, url: str) -> str | None:
        """Validate URL. Returns error string or None if valid."""
        if not url:
            return "No URL provided"

        parsed = urlparse(url)

        if parsed.scheme != "https":
            return "Only HTTPS URLs are allowed for security"

        if not parsed.netloc:
            return "Invalid URL: no domain found"

        # Check blocklist
        domain = parsed.netloc.lower()
        for blocked in settings.blocked_domains_list:
            if blocked in domain:
                return f"Domain '{domain}' is blocked by security policy"

        return None

    def _extract_text(self, html: str) -> str:
        """Extract readable text from HTML using BeautifulSoup."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script, style, nav, footer, header elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        # Truncate if too long
        if len(text) > MAX_CONTENT_LENGTH:
            text = text[:MAX_CONTENT_LENGTH] + "\n\n[Content truncated...]"

        return text
