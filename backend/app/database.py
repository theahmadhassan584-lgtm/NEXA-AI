"""
NEXA AI - Database Layer
SQLAlchemy models and session management for conversation persistence.
Supports SQLite (dev) and PostgreSQL (prod) via DATABASE_URL.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, List
import json
import uuid

from sqlalchemy import (
    create_engine, Column, String, Text, DateTime, Integer, Boolean, JSON
)
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from app.config import settings

# ============================================================
# Engine & Session
# ============================================================

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Session:
    """FastAPI dependency: yield a DB session and close after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================
# ORM Models
# ============================================================

def utcnow_naive() -> datetime:
    """Return naive UTC for existing database DateTime columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ConversationRecord(Base):
    """Stores a full conversation session."""
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=utcnow_naive)
    last_active = Column(DateTime, default=utcnow_naive, onupdate=utcnow_naive)
    language = Column(String(10), default="auto")
    message_count = Column(Integer, default=0)
    metadata_ = Column("metadata", JSON, nullable=True)


class MessageRecord(Base):
    """Individual messages within a conversation."""
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, tool, system
    content = Column(Text, nullable=False)
    tool_use_id = Column(String(100), nullable=True)
    tool_name = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=utcnow_naive)
    metadata_ = Column("metadata", JSON, nullable=True)


class MemoryRecord(Base):
    """Long-term memory store (separate from conversation context)."""
    __tablename__ = "memories"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String(255), nullable=False, index=True)
    value = Column(Text, nullable=False)
    category = Column(String(50), default="general")
    created_at = Column(DateTime, default=utcnow_naive)
    expires_at = Column(DateTime, nullable=True)


class AuditLogRecord(Base):
    """Structured audit log of agent actions."""
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), nullable=True, index=True)
    action = Column(String(100), nullable=False)
    tool_name = Column(String(100), nullable=True)
    input_summary = Column(Text, nullable=True)
    output_summary = Column(Text, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=utcnow_naive)


# ============================================================
# Database Operations
# ============================================================

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def create_conversation(db: Session, language: str = "auto") -> ConversationRecord:
    conv = ConversationRecord(language=language)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def add_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
    tool_use_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> MessageRecord:
    msg = MessageRecord(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        metadata_=metadata,
    )
    db.add(msg)

    # Update conversation stats
    conv = db.query(ConversationRecord).filter_by(id=conversation_id).first()
    if conv:
        conv.message_count = (conv.message_count or 0) + 1
        conv.last_active = utcnow_naive()

    db.commit()
    db.refresh(msg)
    return msg


def get_conversation_messages(
    db: Session, conversation_id: str
) -> List[MessageRecord]:
    return (
        db.query(MessageRecord)
        .filter_by(conversation_id=conversation_id)
        .order_by(MessageRecord.timestamp.asc())
        .all()
    )


def log_audit(
    db: Session,
    action: str,
    conversation_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    input_summary: Optional[str] = None,
    output_summary: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    duration_ms: Optional[int] = None,
):
    entry = AuditLogRecord(
        conversation_id=conversation_id,
        action=action,
        tool_name=tool_name,
        input_summary=input_summary[:500] if input_summary else None,
        output_summary=output_summary[:500] if output_summary else None,
        success=success,
        error_message=error_message,
        duration_ms=duration_ms,
    )
    db.add(entry)
    db.commit()
