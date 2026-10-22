"""
NEXA AI - Application Configuration
Loads all settings from environment variables with safe defaults.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Anthropic / Claude ---
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude access"
    )
    agent_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model identifier"
    )
    agent_max_steps: int = Field(
        default=15,
        description="Maximum agent loop iterations per request"
    )
    agent_step_timeout: int = Field(
        default=30,
        description="Timeout in seconds for each agent step"
    )

    # --- Server ---
    backend_host: str = Field(default="0.0.0.0")
    backend_port: int = Field(default=8000)
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173"
    )

    # --- Database ---
    database_url: str = Field(default="sqlite:///./nexa_dev.db")

    # --- Tool toggles ---
    enable_web_search: bool = Field(default=True)
    enable_page_retrieval: bool = Field(default=True)
    enable_document_qa: bool = Field(default=True)
    enable_data_extraction: bool = Field(default=True)
    enable_file_ops: bool = Field(default=True)
    enable_mcp_connectors: bool = Field(default=False)

    # --- File operations ---
    file_ops_workspace: str = Field(default="./workspace")
    max_upload_size_mb: int = Field(default=10)

    # --- Security ---
    blocked_domains: str = Field(default="")
    rate_limit_rpm: int = Field(default=30)

    # --- Search API ---
    search_api_key: str = Field(default="")
    search_api_endpoint: str = Field(default="")

    # --- MCP ---
    mcp_google_drive_credentials: str = Field(default="")
    mcp_slack_token: str = Field(default="")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def blocked_domains_list(self) -> List[str]:
        if not self.blocked_domains:
            return []
        return [d.strip() for d in self.blocked_domains.split(",") if d.strip()]

    @property
    def workspace_path(self) -> Path:
        p = Path(self.file_ops_workspace).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def validate_api_key(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key != "your_claude_api_key_here")


# Singleton settings instance
settings = Settings()
