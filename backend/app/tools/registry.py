"""
NEXA AI - Tool Registry & Base Tool
Modular tool system with registration, permissions, timeouts, and error handling.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
import asyncio
import time
import logging

from app.models import ToolDefinition, ToolExecutionResponse

logger = logging.getLogger("nexa.tools")


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    name: str = ""
    description: str = ""
    timeout: int = 10  # seconds
    permission: str = "allowed"  # "allowed", "confirmation_required", "blocked"
    enabled: bool = True

    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Return JSON Schema for this tool's input."""
        ...

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with validated input. Returns result dict."""
        ...

    def get_definition(self) -> ToolDefinition:
        """Return the tool definition for Claude's tool list."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.get_input_schema(),
            timeout=self.timeout,
            permission=self.permission,
            enabled=self.enabled,
        )

    def to_claude_tool(self) -> Dict[str, Any]:
        """Format as Claude API tool parameter."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_input_schema(),
        }

    async def safe_execute(self, input_data: Dict[str, Any]) -> ToolExecutionResponse:
        """Execute with timeout and error handling."""
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self.execute(input_data),
                timeout=self.timeout,
            )
            duration = int((time.monotonic() - start) * 1000)
            logger.info(f"Tool '{self.name}' completed in {duration}ms")
            return ToolExecutionResponse(
                tool_name=self.name,
                success=True,
                data=result,
                duration_ms=duration,
            )
        except asyncio.TimeoutError:
            duration = int((time.monotonic() - start) * 1000)
            msg = f"Tool '{self.name}' timed out after {self.timeout}s"
            logger.warning(msg)
            return ToolExecutionResponse(
                tool_name=self.name,
                success=False,
                message=msg,
                duration_ms=duration,
            )
        except Exception as e:
            duration = int((time.monotonic() - start) * 1000)
            msg = f"Tool '{self.name}' error: {type(e).__name__}: {str(e)}"
            logger.error(msg, exc_info=True)
            return ToolExecutionResponse(
                tool_name=self.name,
                success=False,
                message=msg,
                duration_ms=duration,
            )


class ToolRegistry:
    """
    Central registry for all agent tools.
    Manages registration, lookup, enable/disable, and serialization to Claude format.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} (enabled={tool.enabled})")

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_all(self) -> List[BaseTool]:
        return list(self._tools.values())

    def list_enabled(self) -> List[BaseTool]:
        return [t for t in self._tools.values() if t.enabled]

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def list_enabled_names(self) -> List[str]:
        return [t.name for t in self._tools.values() if t.enabled]

    def enable(self, name: str) -> bool:
        tool = self._tools.get(name)
        if tool:
            tool.enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        tool = self._tools.get(name)
        if tool:
            tool.enabled = False
            return True
        return False

    def get_claude_tools(self, override_names: Optional[List[str]] = None) -> List[Dict]:
        """Get tool definitions formatted for Claude API."""
        if override_names is not None:
            tools = [t for t in self._tools.values() if t.name in override_names and t.enabled]
        else:
            tools = self.list_enabled()
        return [t.to_claude_tool() for t in tools]

    def get_definitions(self) -> List[ToolDefinition]:
        return [t.get_definition() for t in self._tools.values()]

    async def execute_tool(self, name: str, input_data: Dict[str, Any]) -> ToolExecutionResponse:
        """Look up and execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return ToolExecutionResponse(
                tool_name=name,
                success=False,
                message=f"Unknown tool: {name}",
            )
        if not tool.enabled:
            return ToolExecutionResponse(
                tool_name=name,
                success=False,
                message=f"Tool '{name}' is currently disabled",
            )
        return await tool.safe_execute(input_data)


# Global registry instance
registry = ToolRegistry()
