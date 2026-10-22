"""
NEXA AI - FastAPI Server Entry Point
Main application with all routes, middleware, and startup hooks.
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import os
import uuid

from fastapi import (
    FastAPI, HTTPException, UploadFile, File, Form, Query, Depends, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    ChatRequest, ChatResponse, ToolCallInfo,
    HealthResponse, AgentStatus, ErrorResponse,
    FileUploadResponse, ToolExecutionRequest, ToolExecutionResponse,
    SessionInfo, StreamEvent,
)
from app.database import (
    init_db, get_db, create_conversation, add_message,
    get_conversation_messages, log_audit, ConversationRecord,
)
from app.tools import register_all_tools, registry
from app.agent import AgentLoop

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nexa.server")

# ============================================================
# Upload directory
# ============================================================

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# Lifespan (startup / shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=" * 60)
    logger.info("  NEXA AI Agent Server Starting")
    logger.info("=" * 60)

    init_db()
    logger.info("Database initialized")

    register_all_tools()
    logger.info(f"Tools registered: {registry.list_names()}")
    logger.info(f"Tools enabled: {registry.list_enabled_names()}")

    api_configured = settings.validate_api_key()
    logger.info(f"Anthropic API key configured: {api_configured}")

    if not api_configured:
        logger.warning(
            "⚠ ANTHROPIC_API_KEY not set! Chat endpoints will return errors. "
            "Set it in your .env file."
        )

    logger.info(f"Server ready on {settings.backend_host}:{settings.backend_port}")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("NEXA AI Server shutting down")


# ============================================================
# App instance
# ============================================================

app = FastAPI(
    title="NEXA AI Agent API",
    description="Production-grade Claude-powered AI Agent with modular tool system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Rate Limiting (simple in-memory)
# ============================================================

_rate_store: Dict[str, List[float]] = {}

def check_rate_limit(request: Request):
    """Simple per-IP rate limiting."""
    import time
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60  # 1 minute

    if client_ip not in _rate_store:
        _rate_store[client_ip] = []

    # Clean old entries
    _rate_store[client_ip] = [t for t in _rate_store[client_ip] if now - t < window]

    if len(_rate_store[client_ip]) >= settings.rate_limit_rpm:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({settings.rate_limit_rpm} requests/minute)"
        )

    _rate_store[client_ip].append(now)


# ============================================================
# Health & Status Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@app.get("/status", response_model=AgentStatus, tags=["System"])
async def agent_status():
    """Get agent configuration and status."""
    return AgentStatus(
        api_key_configured=settings.validate_api_key(),
        model=settings.agent_model,
        tools_available=registry.list_names(),
        tools_enabled=registry.list_enabled_names(),
        max_steps=settings.agent_max_steps,
        database_connected=True,
    )


# ============================================================
# Chat Endpoints
# ============================================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    req: Request = None,
    db: Session = Depends(get_db),
):
    """
    Send a message to the NEXA AI agent and receive a response.
    Supports multi-turn conversations with tool invocation.
    """
    # Rate limit
    if req:
        check_rate_limit(req)

    # Validate API key
    if not settings.validate_api_key():
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not configured. Set it in .env to use chat."
        )

    # Session management
    session_id = request.session_id
    if not session_id:
        conv = create_conversation(db, language=request.language.value)
        session_id = conv.id
    else:
        conv = db.query(ConversationRecord).filter_by(id=session_id).first()
        if not conv:
            conv = create_conversation(db, language=request.language.value)
            session_id = conv.id

    # Store user message
    add_message(db, session_id, "user", request.message)

    # Build messages for Claude
    claude_messages = []
    for msg in request.history:
        claude_messages.append({
            "role": msg.role.value if hasattr(msg.role, 'value') else msg.role,
            "content": msg.content,
        })
    claude_messages.append({"role": "user", "content": request.message})

    # Run agent loop
    try:
        agent = AgentLoop()
        result = await agent.run(claude_messages, request.enabled_tools)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Agent loop error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Store assistant response
    add_message(db, session_id, "assistant", result["reply"])

    # Audit log
    log_audit(
        db,
        action="chat",
        conversation_id=session_id,
        input_summary=request.message[:200],
        output_summary=result["reply"][:200],
        success=True,
    )

    # Log tool calls
    for tc in result.get("tool_calls", []):
        log_audit(
            db,
            action="tool_call",
            conversation_id=session_id,
            tool_name=tc.tool_name,
            input_summary=json.dumps(tc.tool_input, default=str)[:200],
            output_summary=str(tc.tool_output)[:200] if tc.tool_output else None,
            success=tc.success,
            error_message=tc.error,
            duration_ms=tc.duration_ms,
        )

    return ChatResponse(
        session_id=session_id,
        reply=result["reply"],
        tool_calls=result.get("tool_calls", []),
        model=result.get("model", settings.agent_model),
        usage=result.get("usage"),
    )


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    req: Request = None,
    db: Session = Depends(get_db),
):
    """
    Streaming version of chat. Returns Server-Sent Events (SSE).
    """
    if req:
        check_rate_limit(req)

    if not settings.validate_api_key():
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not configured."
        )

    # Session
    session_id = request.session_id
    if not session_id:
        conv = create_conversation(db, language=request.language.value)
        session_id = conv.id

    add_message(db, session_id, "user", request.message)

    # Build messages
    claude_messages = []
    for msg in request.history:
        claude_messages.append({
            "role": msg.role.value if hasattr(msg.role, 'value') else msg.role,
            "content": msg.content,
        })
    claude_messages.append({"role": "user", "content": request.message})

    async def event_generator():
        # Send session ID first
        yield f"data: {json.dumps({'event': 'session', 'data': {'session_id': session_id}})}\n\n"

        try:
            agent = AgentLoop()
            full_text = []
            async for event in agent.run_stream(claude_messages, request.enabled_tools):
                payload = json.dumps({
                    "event": event.event,
                    "data": event.data,
                }, default=str)
                yield f"data: {payload}\n\n"

                if event.event == "text_delta":
                    full_text.append(event.data.get("text", ""))

            # Store complete response
            complete_text = "".join(full_text)
            if complete_text:
                add_message(db, session_id, "assistant", complete_text)

        except Exception as e:
            error_payload = json.dumps({
                "event": "error",
                "data": {"error": str(e)},
            })
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# File Upload Endpoint
# ============================================================

@app.post("/upload", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(
    file: UploadFile = File(...),
    req: Request = None,
):
    """Upload a document for DocumentQA processing."""
    if req:
        check_rate_limit(req)

    # Validate file size
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB."
        )

    # Validate extension
    allowed_ext = {".pdf", ".txt", ".md", ".csv", ".json", ".docx"}
    ext = Path(file.filename or "unknown").suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_ext)}"
        )

    # Save file
    file_id = str(uuid.uuid4())
    safe_name = f"{file_id}{ext}"
    file_path = UPLOAD_DIR / safe_name
    file_path.write_bytes(content)

    logger.info(f"File uploaded: {file.filename} -> {safe_name} ({len(content)} bytes)")

    return FileUploadResponse(
        file_id=file_id,
        filename=file.filename or "unknown",
        size_bytes=len(content),
        content_type=file.content_type or "application/octet-stream",
    )


@app.get("/upload/{file_id}", tags=["Files"])
async def get_upload_info(file_id: str):
    """Get info about an uploaded file."""
    # Search for file
    for f in UPLOAD_DIR.iterdir():
        if f.stem == file_id:
            return {
                "file_id": file_id,
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "exists": True,
            }
    raise HTTPException(status_code=404, detail="File not found")


# ============================================================
# Tool Management Endpoints
# ============================================================

@app.get("/tools", tags=["Tools"])
async def list_tools():
    """List all registered tools and their status."""
    return {
        "tools": [t.get_definition().model_dump() for t in registry.list_all()]
    }


@app.post("/tools/toggle/{tool_name}", tags=["Tools"])
async def toggle_tool(tool_name: str, enabled: bool = Query(...)):
    """Enable or disable a specific tool."""
    tool = registry.get(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

    if enabled:
        registry.enable(tool_name)
    else:
        registry.disable(tool_name)

    return {"tool": tool_name, "enabled": enabled}


@app.post("/tools/execute", response_model=ToolExecutionResponse, tags=["Tools"])
async def execute_tool(
    request: ToolExecutionRequest,
    db: Session = Depends(get_db),
):
    """Manually execute a tool (for debugging/admin)."""
    result = await registry.execute_tool(request.tool_name, request.input_data)

    log_audit(
        db,
        action="manual_tool_execute",
        conversation_id=request.session_id,
        tool_name=request.tool_name,
        input_summary=json.dumps(request.input_data)[:200],
        output_summary=str(result.data)[:200] if result.data else None,
        success=result.success,
        error_message=result.message if not result.success else None,
        duration_ms=result.duration_ms,
    )

    return result


# ============================================================
# Session Management Endpoints
# ============================================================

@app.get("/sessions", tags=["Sessions"])
async def list_sessions(
    limit: int = Query(default=20, le=100),
    db: Session = Depends(get_db),
):
    """List recent chat sessions."""
    conversations = (
        db.query(ConversationRecord)
        .order_by(ConversationRecord.last_active.desc())
        .limit(limit)
        .all()
    )
    return {
        "sessions": [
            SessionInfo(
                session_id=c.id,
                created_at=c.created_at,
                last_active=c.last_active,
                message_count=c.message_count or 0,
                language=c.language or "auto",
            ).model_dump()
            for c in conversations
        ]
    }


@app.get("/sessions/{session_id}/messages", tags=["Sessions"])
async def get_session_messages(
    session_id: str,
    db: Session = Depends(get_db),
):
    """Get all messages for a session."""
    messages = get_conversation_messages(db, session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found or empty")

    return {
        "session_id": session_id,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                "tool_name": m.tool_name,
            }
            for m in messages
        ],
    }


# ============================================================
# Error Handlers
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            code=str(exc.status_code),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if os.getenv("DEBUG") else None,
        ).model_dump(),
    )


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
        log_level="info",
    )
