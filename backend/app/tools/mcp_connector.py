"""
NEXA AI - MCP Connector Tool (Stub)
Model Context Protocol integration for external services.
Provides a stub interface; real connectors require credentials.
"""

from __future__ import annotations
from typing import Any, Dict
import logging

from app.tools.registry import BaseTool

logger = logging.getLogger("nexa.tools.mcp")


class MCPConnectorTool(BaseTool):
    name = "mcp_connector"
    description = (
        "Connect to external services via the Model Context Protocol (MCP). "
        "Supports Google Drive, Slack, and other MCP-compatible services. "
        "NOTE: This tool requires MCP credentials to be configured. "
        "Without credentials, it operates in simulation mode."
    )
    timeout = 15
    permission = "allowed"
    enabled = False  # Disabled by default until credentials are configured

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "enum": ["google_drive", "slack", "custom"],
                    "description": "The MCP service to interact with",
                },
                "action": {
                    "type": "string",
                    "description": "Action to perform (e.g., 'list_files', 'send_message', 'read_document')",
                },
                "parameters": {
                    "type": "object",
                    "description": "Service-specific parameters for the action",
                },
            },
            "required": ["service", "action"],
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        service = input_data.get("service", "")
        action = input_data.get("action", "")
        params = input_data.get("parameters", {})

        logger.info(f"MCP connector called: service={service}, action={action}")

        # Check if credentials are available
        from app.config import settings

        if service == "google_drive" and not settings.mcp_google_drive_credentials:
            return {
                "error": "Google Drive MCP credentials not configured",
                "hint": "Set MCP_GOOGLE_DRIVE_CREDENTIALS in your .env file",
                "simulated": True,
                "result": self._simulate_google_drive(action, params),
            }

        if service == "slack" and not settings.mcp_slack_token:
            return {
                "error": "Slack MCP token not configured",
                "hint": "Set MCP_SLACK_TOKEN in your .env file",
                "simulated": True,
                "result": self._simulate_slack(action, params),
            }

        if service == "custom":
            return {
                "error": "Custom MCP connectors require implementation",
                "hint": "Extend MCPConnectorTool with your custom service logic",
            }

        return {
            "error": f"Unknown MCP service: {service}",
            "available_services": ["google_drive", "slack", "custom"],
        }

    def _simulate_google_drive(self, action: str, params: Dict) -> Dict[str, Any]:
        """Simulated Google Drive responses for demonstration."""
        if action == "list_files":
            return {
                "files": [
                    {"name": "Document.pdf", "id": "sim_001", "type": "pdf"},
                    {"name": "Spreadsheet.xlsx", "id": "sim_002", "type": "xlsx"},
                    {"name": "Notes.txt", "id": "sim_003", "type": "txt"},
                ],
                "note": "Simulated response - configure credentials for real data",
            }
        elif action == "read_document":
            return {
                "content": "This is simulated document content from Google Drive.",
                "note": "Simulated response",
            }
        return {"action": action, "note": "Simulated - action not implemented"}

    def _simulate_slack(self, action: str, params: Dict) -> Dict[str, Any]:
        """Simulated Slack responses for demonstration."""
        if action == "send_message":
            return {
                "sent": True,
                "channel": params.get("channel", "#general"),
                "note": "Simulated - message not actually sent",
            }
        elif action == "list_channels":
            return {
                "channels": ["#general", "#engineering", "#random"],
                "note": "Simulated response",
            }
        return {"action": action, "note": "Simulated - action not implemented"}
