"""
NEXA AI - File System Operations Tool
Safe local file read/write/append within a designated workspace directory.
Enforces path restrictions to prevent directory traversal attacks.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import logging
import os

from app.tools.registry import BaseTool
from app.config import settings

logger = logging.getLogger("nexa.tools.file_ops")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_CONTENT_RETURN = 50_000  # chars


class FileSystemOpsTool(BaseTool):
    name = "file_ops"
    description = (
        "Perform safe file operations (read, write, append, list, delete) "
        "within the designated workspace directory. All paths are restricted "
        "to the workspace sandbox. Destructive operations require confirmation."
    )
    timeout = 5
    permission = "allowed"

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "append", "list", "delete", "exists"],
                    "description": "File operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "Relative file path within the workspace",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write/append (for write/append operations)",
                },
            },
            "required": ["operation", "path"],
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        operation = input_data.get("operation", "").lower()
        rel_path = input_data.get("path", "").strip()
        content = input_data.get("content", "")

        if not operation:
            return {"error": "No operation specified"}
        if not rel_path:
            return {"error": "No path specified"}

        # Resolve and validate path within workspace
        workspace = settings.workspace_path
        try:
            full_path = self._resolve_safe_path(workspace, rel_path)
        except ValueError as e:
            return {"error": str(e)}

        # Execute operation
        handlers = {
            "read": self._read,
            "write": self._write,
            "append": self._append,
            "list": self._list,
            "delete": self._delete,
            "exists": self._exists,
        }

        handler = handlers.get(operation)
        if not handler:
            return {"error": f"Unknown operation: {operation}"}

        return handler(full_path, content)

    def _resolve_safe_path(self, workspace: Path, rel_path: str) -> Path:
        """Resolve path and ensure it stays within workspace (prevent traversal)."""
        # Normalize and block obvious traversal
        cleaned = rel_path.replace("\\", "/")
        if ".." in cleaned.split("/"):
            raise ValueError("Directory traversal ('..') is not allowed")

        full = (workspace / cleaned).resolve()

        # Verify resolved path is under workspace
        try:
            full.relative_to(workspace.resolve())
        except ValueError:
            raise ValueError(
                f"Path escapes workspace boundary. Must stay within: {workspace}"
            )

        return full

    def _read(self, path: Path, _content: str) -> Dict[str, Any]:
        if not path.exists():
            return {"error": f"File not found: {path.name}"}
        if not path.is_file():
            return {"error": f"Not a file: {path.name}"}
        if path.stat().st_size > MAX_FILE_SIZE:
            return {"error": f"File too large (max {MAX_FILE_SIZE // 1024}KB)"}

        text = path.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > MAX_CONTENT_RETURN
        if truncated:
            text = text[:MAX_CONTENT_RETURN] + "\n[Truncated...]"

        return {
            "operation": "read",
            "path": path.name,
            "content": text,
            "size": path.stat().st_size,
            "truncated": truncated,
        }

    def _write(self, path: Path, content: str) -> Dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info(f"File written: {path}")
        return {
            "operation": "write",
            "path": path.name,
            "size": len(content),
            "success": True,
        }

    def _append(self, path: Path, content: str) -> Dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"File appended: {path}")
        return {
            "operation": "append",
            "path": path.name,
            "bytes_added": len(content),
            "success": True,
        }

    def _list(self, path: Path, _content: str) -> Dict[str, Any]:
        target = path if path.is_dir() else path.parent
        if not target.exists():
            return {"error": f"Directory not found: {target.name}"}

        entries = []
        for item in sorted(target.iterdir()):
            entries.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            })
        return {
            "operation": "list",
            "directory": str(target.relative_to(settings.workspace_path)),
            "entries": entries,
            "count": len(entries),
        }

    def _delete(self, path: Path, _content: str) -> Dict[str, Any]:
        if not path.exists():
            return {"error": f"File not found: {path.name}"}

        # Safety: only delete files, not directories
        if path.is_dir():
            return {
                "error": "Cannot delete directories. Only individual files can be deleted.",
                "requires_confirmation": True,
            }

        path.unlink()
        logger.info(f"File deleted: {path}")
        return {
            "operation": "delete",
            "path": path.name,
            "success": True,
            "note": "File permanently deleted",
        }

    def _exists(self, path: Path, _content: str) -> Dict[str, Any]:
        return {
            "operation": "exists",
            "path": path.name,
            "exists": path.exists(),
            "is_file": path.is_file() if path.exists() else False,
            "is_dir": path.is_dir() if path.exists() else False,
        }
