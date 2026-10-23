"""
NEXA AI - Pydantic Models
Data models for API requests, responses, tool schemas, and database entities.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import uuid
import re


# ============================================================
# Enums
# ============================================================

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ToolStatus(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"


class Language(str, Enum):
    ENGLISH = "en"
    ROMAN_URDU = "ur"
    AUTO = "auto"


# ============================================================
# Chat Models
# ============================================================

class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None
    tool_use_id: Optional[str] = None
    tool_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=32_000,
        description="User message text"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier; omit to create a new session"
    )
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages for context"
    )
    language: Language = Field(
        default=Language.AUTO,
        description="Preferred response language"
    )
    enabled_tools: Optional[List[str]] = Field(
        default=None,
        description="Override which tools are active for this request"
    )

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """Strip control characters and excessive whitespace."""
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)
        return v.strip()


class ToolCallInfo(BaseModel):
    """Information about a tool invocation within a response."""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Optional[Any] = None
    duration_ms: Optional[int] = None
    success: bool = True
    error: Optional[str] = None


class ChatResponse(BaseModel):
    """Response body from the /chat endpoint."""
    session_id: str
    reply: str
    tool_calls: List[ToolCallInfo] = Field(default_factory=list)
    model: str = ""
    usage: Optional[Dict[str, int]] = None
    language_detected: Optional[str] = None


class StreamEvent(BaseModel):
    """A single SSE event sent during streaming."""
    event: str  # "text_delta", "tool_start", "tool_end", "done", "error"
    data: Any


# ============================================================
# Tool Models
# ============================================================

class ToolDefinition(BaseModel):
    """Schema describing a registered tool for Claude."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    timeout: int = 10
    permission: str = "allowed"
    enabled: bool = True


class ToolExecutionRequest(BaseModel):
    """Request to manually invoke a tool (admin/debug)."""
    tool_name: str
    input_data: Dict[str, Any]
    session_id: Optional[str] = None


class ToolExecutionResponse(BaseModel):
    """Response from a tool invocation."""
    tool_name: str
    success: bool
    data: Any = None
    message: Optional[str] = None
    duration_ms: int = 0


# ============================================================
# File Upload Models
# ============================================================

class FileUploadResponse(BaseModel):
    """Response after uploading a document."""
    file_id: str
    filename: str
    size_bytes: int
    content_type: str
    status: str = "uploaded"


class DocumentQARequest(BaseModel):
    """Request for document-based question answering."""
    file_id: str
    question: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[str] = None


# ============================================================
# Session / Memory Models
# ============================================================

class SessionInfo(BaseModel):
    """Information about a chat session."""
    session_id: str
    created_at: datetime
    last_active: datetime
    message_count: int
    language: Language = Language.AUTO


class MemoryEntry(BaseModel):
    """A discrete memory item in the long-term store."""
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key: str
    value: str
    category: str = "general"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None


# ============================================================
# Health / Status Models
# ============================================================

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentStatus(BaseModel):
    api_key_configured: bool
    model: str
    tools_available: List[str]
    tools_enabled: List[str]
    max_steps: int
    database_connected: bool


# ============================================================
# Error Models
# ============================================================

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
