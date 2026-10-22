"""
NEXA AI - Backend Tests
Comprehensive tests for API endpoints, agent logic, and tools.
"""

import os
import json
import shutil
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Set test environment before imports
os.environ["ANTHROPIC_API_KEY"] = "test-key-for-testing"
os.environ["DATABASE_URL"] = "sqlite:///./test_nexa.db"
os.environ["FILE_OPS_WORKSPACE"] = "./test_workspace"

from app.main import app
from app.tools.registry import ToolRegistry, BaseTool
from app.tools.web_search import WebSearchTool
from app.tools.page_retrieval import PageRetrievalTool
from app.tools.document_qa import DocumentQATool
from app.tools.data_extraction import DataExtractionTool
from app.tools.file_ops import FileSystemOpsTool
from app.tools.mcp_connector import MCPConnectorTool
from app.models import ChatRequest, MessageRole, Language


client: TestClient


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="session", autouse=True)
def manage_app_lifespan():
    """Run startup/shutdown once and safely clean the test database."""
    global client
    from app.database import engine

    test_db = Path("./test_nexa.db")
    test_ws = Path("./test_workspace")

    engine.dispose()
    if test_db.exists():
        try:
            test_db.unlink()
        except PermissionError as exc:
            raise RuntimeError(
                "Cannot reset test_nexa.db. Stop any process using this "
                "test database, then rerun the tests."
            ) from exc

    with TestClient(app) as test_client:
        client = test_client
        yield

    engine.dispose()
    if test_db.exists():
        try:
            test_db.unlink()
        except PermissionError as exc:
            raise RuntimeError(
                "Cannot remove test_nexa.db after tests; a connection "
                "may still be open."
            ) from exc

    if test_ws.exists():
        shutil.rmtree(test_ws, ignore_errors=True)


@pytest.fixture(autouse=True)
def cleanup():
    """Clean temporary workspace after each test without deleting the DB."""
    yield
    test_ws = Path("./test_workspace")
    if test_ws.exists():
        shutil.rmtree(test_ws, ignore_errors=True)


@pytest.fixture
def test_workspace(tmp_path):
    """Create a temporary workspace for file ops tests."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


# ============================================================
# Health & Status Tests
# ============================================================

class TestHealthEndpoints:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "timestamp" in data

    def test_agent_status(self):
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "api_key_configured" in data
        assert "model" in data
        assert "tools_available" in data
        assert "tools_enabled" in data
        assert isinstance(data["tools_available"], list)


# ============================================================
# Chat Endpoint Tests
# ============================================================

class TestChatEndpoint:
    @patch("app.agent.AgentLoop.run")
    def test_chat_basic(self, mock_run):
        """Test basic chat with mocked Claude response."""
        mock_run.return_value = {
            "reply": "Hello! I'm NEXA AI. How can I help you?",
            "tool_calls": [],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        response = client.post("/chat", json={
            "message": "Hello",
            "history": [],
        })
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert "session_id" in data

    def test_chat_empty_message(self):
        """Test that empty messages are rejected."""
        response = client.post("/chat", json={
            "message": "",
            "history": [],
        })
        assert response.status_code == 422  # Validation error

    def test_chat_very_long_message(self):
        """Test message length validation."""
        response = client.post("/chat", json={
            "message": "x" * 50_000,  # Over 32k limit
            "history": [],
        })
        assert response.status_code == 422


# ============================================================
# Tool Registry Tests
# ============================================================

class TestToolRegistry:
    def test_register_and_lookup(self):
        reg = ToolRegistry()
        tool = WebSearchTool()
        reg.register(tool)
        assert reg.get("web_search") is tool
        assert "web_search" in reg.list_names()

    def test_enable_disable(self):
        reg = ToolRegistry()
        tool = WebSearchTool()
        reg.register(tool)

        reg.disable("web_search")
        assert "web_search" not in reg.list_enabled_names()

        reg.enable("web_search")
        assert "web_search" in reg.list_enabled_names()

    def test_unknown_tool(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        result = await reg.execute_tool("nonexistent", {})
        assert not result.success
        assert "Unknown tool" in result.message

    @pytest.mark.asyncio
    async def test_execute_disabled_tool(self):
        reg = ToolRegistry()
        tool = WebSearchTool()
        tool.enabled = False
        reg.register(tool)
        result = await reg.execute_tool("web_search", {"query": "test"})
        assert not result.success
        assert "disabled" in result.message

    def test_claude_tools_format(self):
        reg = ToolRegistry()
        tool = WebSearchTool()
        reg.register(tool)
        claude_tools = reg.get_claude_tools()
        assert len(claude_tools) == 1
        assert claude_tools[0]["name"] == "web_search"
        assert "input_schema" in claude_tools[0]
        assert "description" in claude_tools[0]


# ============================================================
# Individual Tool Tests
# ============================================================

class TestWebSearchTool:
    @pytest.mark.asyncio
    async def test_empty_query(self):
        tool = WebSearchTool()
        result = await tool.execute({"query": ""})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_simulated_search(self):
        tool = WebSearchTool()
        result = await tool.execute({"query": "test query"})
        assert "results" in result
        assert "note" in result  # Simulated mode

    @pytest.mark.asyncio
    async def test_num_results_limit(self):
        tool = WebSearchTool()
        result = await tool.execute({"query": "test", "num_results": 20})
        # Should cap at 10
        assert "results" in result


class TestPageRetrievalTool:
    @pytest.mark.asyncio
    async def test_empty_url(self):
        tool = PageRetrievalTool()
        result = await tool.execute({"url": ""})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_http_url_rejected(self):
        tool = PageRetrievalTool()
        result = await tool.execute({"url": "http://example.com"})
        assert "error" in result
        assert "HTTPS" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_url(self):
        tool = PageRetrievalTool()
        result = await tool.execute({"url": "not-a-url"})
        assert "error" in result


class TestDocumentQATool:
    @pytest.mark.asyncio
    async def test_no_file(self):
        tool = DocumentQATool()
        result = await tool.execute({"file_path": ""})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_missing_file(self):
        tool = DocumentQATool()
        result = await tool.execute({"file_path": "/nonexistent/file.txt"})
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_unsupported_format(self):
        tool = DocumentQATool()
        # Create a temp file with unsupported extension
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            result = await tool.execute({"file_path": temp_path})
            assert "error" in result
            assert "Unsupported" in result["error"]
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_txt_extraction(self):
        tool = DocumentQATool()
        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write("Hello, this is test content for NEXA AI.")
            temp_path = f.name

        try:
            result = await tool.execute({
                "file_path": temp_path,
                "question": "What is this about?"
            })
            assert "content" in result
            assert "NEXA AI" in result["content"]
            assert result["question"] == "What is this about?"
        finally:
            os.unlink(temp_path)


class TestDataExtractionTool:
    @pytest.mark.asyncio
    async def test_no_input(self):
        tool = DataExtractionTool()
        result = await tool.execute({})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_csv_inline(self):
        tool = DataExtractionTool()
        csv_data = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago"
        result = await tool.execute({
            "raw_data": csv_data,
            "format": "csv",
            "operation": "summary",
        })
        assert result["format"] == "csv"
        assert result["column_count"] == 3
        assert "columns" in result

    @pytest.mark.asyncio
    async def test_json_inline(self):
        tool = DataExtractionTool()
        json_data = json.dumps({"name": "NEXA", "version": 1, "features": ["chat", "tools"]})
        result = await tool.execute({
            "raw_data": json_data,
            "format": "json",
        })
        assert result["format"] == "json"
        assert result["type"] == "object"


class TestFileOpsTool:
    @pytest.mark.asyncio
    async def test_no_operation(self):
        tool = FileSystemOpsTool()
        result = await tool.execute({"path": "test.txt"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_no_path(self):
        tool = FileSystemOpsTool()
        result = await tool.execute({"operation": "read"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_directory_traversal_blocked(self):
        tool = FileSystemOpsTool()
        result = await tool.execute({
            "operation": "read",
            "path": "../../../etc/passwd",
        })
        assert "error" in result
        assert "traversal" in result["error"].lower() or "escape" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_write_and_read(self):
        tool = FileSystemOpsTool()
        # Write
        result = await tool.execute({
            "operation": "write",
            "path": "test_file.txt",
            "content": "Hello NEXA AI!",
        })
        assert result.get("success") is True

        # Read back
        result = await tool.execute({
            "operation": "read",
            "path": "test_file.txt",
        })
        assert result.get("content") == "Hello NEXA AI!"

    @pytest.mark.asyncio
    async def test_exists_check(self):
        tool = FileSystemOpsTool()
        result = await tool.execute({
            "operation": "exists",
            "path": "nonexistent_file.txt",
        })
        assert result.get("exists") is False


class TestMCPConnectorTool:
    @pytest.mark.asyncio
    async def test_google_drive_simulated(self):
        tool = MCPConnectorTool()
        result = await tool.execute({
            "service": "google_drive",
            "action": "list_files",
        })
        assert "simulated" in result or "error" in result

    @pytest.mark.asyncio
    async def test_unknown_service(self):
        tool = MCPConnectorTool()
        result = await tool.execute({
            "service": "unknown_service",
            "action": "test",
        })
        assert "error" in result


# ============================================================
# Tool Endpoint Tests
# ============================================================

class TestToolEndpoints:
    def test_list_tools(self):
        response = client.get("/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)
        # Should have at least the core tools
        tool_names = [t["name"] for t in data["tools"]]
        assert "web_search" in tool_names
        assert "file_ops" in tool_names


# ============================================================
# File Upload Tests
# ============================================================

class TestFileUpload:
    def test_upload_txt(self):
        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"Hello World", "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert data["filename"] == "test.txt"
        assert data["size_bytes"] == 11

    def test_upload_unsupported_type(self):
        response = client.post(
            "/upload",
            files={"file": ("malware.exe", b"evil", "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_upload_too_large(self):
        # Create content larger than limit
        large_content = b"x" * (11 * 1024 * 1024)  # 11 MB
        response = client.post(
            "/upload",
            files={"file": ("big.txt", large_content, "text/plain")},
        )
        assert response.status_code == 413


# ============================================================
# Session Tests
# ============================================================

class TestSessions:
    def test_list_sessions(self):
        response = client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data

    def test_get_nonexistent_session(self):
        response = client.get("/sessions/nonexistent-id/messages")
        assert response.status_code == 404


# ============================================================
# Security Tests
# ============================================================

class TestSecurity:
    def test_control_characters_sanitized(self):
        """Test that control characters in messages are stripped."""
        msg = ChatRequest(
            message="Hello\x00\x01\x02World",
            history=[],
        )
        assert "\x00" not in msg.message
        assert "HelloWorld" == msg.message

    def test_xss_in_message(self):
        """Test that XSS attempts don't cause issues."""
        # The backend should pass through safely (frontend must escape)
        msg = ChatRequest(
            message='<script>alert("xss")</script>',
            history=[],
        )
        assert "<script>" in msg.message  # Backend stores as-is; frontend escapes


# ============================================================
# Model Validation Tests
# ============================================================

class TestModels:
    def test_chat_request_validation(self):
        req = ChatRequest(
            message="Test message",
            language=Language.ENGLISH,
            history=[],
        )
        assert req.message == "Test message"
        assert req.language == Language.ENGLISH

    def test_chat_request_auto_language(self):
        req = ChatRequest(message="Test", history=[])
        assert req.language == Language.AUTO
