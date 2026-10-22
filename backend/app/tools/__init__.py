"""NEXA AI - Tools Package Initialization"""

from app.tools.registry import BaseTool, ToolRegistry, registry
from app.tools.web_search import WebSearchTool
from app.tools.page_retrieval import PageRetrievalTool
from app.tools.document_qa import DocumentQATool
from app.tools.data_extraction import DataExtractionTool
from app.tools.file_ops import FileSystemOpsTool
from app.tools.mcp_connector import MCPConnectorTool


def register_all_tools() -> ToolRegistry:
    """Register all tool implementations in the global registry."""
    from app.config import settings

    tools = [
        (WebSearchTool(), settings.enable_web_search),
        (PageRetrievalTool(), settings.enable_page_retrieval),
        (DocumentQATool(), settings.enable_document_qa),
        (DataExtractionTool(), settings.enable_data_extraction),
        (FileSystemOpsTool(), settings.enable_file_ops),
        (MCPConnectorTool(), settings.enable_mcp_connectors),
    ]

    for tool, enabled in tools:
        tool.enabled = enabled
        registry.register(tool)

    return registry


__all__ = [
    "BaseTool",
    "ToolRegistry",
    "registry",
    "register_all_tools",
    "WebSearchTool",
    "PageRetrievalTool",
    "DocumentQATool",
    "DataExtractionTool",
    "FileSystemOpsTool",
    "MCPConnectorTool",
]
