"""
NEXA AI - Document QA Tool
Handles uploaded documents (PDF, TXT, DOCX) for summarization and Q&A.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import logging

from app.tools.registry import BaseTool

logger = logging.getLogger("nexa.tools.document_qa")

# Supported extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json", ".docx"}
MAX_TEXT_LENGTH = 50_000  # chars


class DocumentQATool(BaseTool):
    name = "document_qa"
    description = (
        "Process uploaded documents (PDF, TXT, DOCX, MD, CSV, JSON) to extract "
        "text content for summarization or question answering. "
        "Provide the file path of an uploaded document and optionally a question."
    )
    timeout = 30
    permission = "allowed"

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the uploaded document file",
                },
                "question": {
                    "type": "string",
                    "description": "Optional question to answer about the document",
                },
                "operation": {
                    "type": "string",
                    "enum": ["extract", "summarize"],
                    "description": "Operation: 'extract' (return text) or 'summarize' (brief summary)",
                    "default": "extract",
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        file_path = input_data.get("file_path", "").strip()
        question = input_data.get("question", "")
        operation = input_data.get("operation", "extract")

        if not file_path:
            return {"error": "No file path provided"}

        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return {
                "error": f"Unsupported file format: {ext}",
                "supported": list(SUPPORTED_EXTENSIONS),
            }

        # Extract text based on file type
        try:
            text = await self._extract_text(path, ext)
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            return {"error": f"Failed to extract text: {str(e)}"}

        if not text.strip():
            return {"error": "No text content could be extracted from the document"}

        # Truncate if needed
        truncated = len(text) > MAX_TEXT_LENGTH
        if truncated:
            text = text[:MAX_TEXT_LENGTH] + "\n\n[Content truncated...]"

        result: Dict[str, Any] = {
            "file": path.name,
            "format": ext,
            "content": text,
            "length": len(text),
            "truncated": truncated,
        }

        if question:
            result["question"] = question
            result["note"] = (
                "Document text has been extracted. The AI will use this content "
                "to answer the question."
            )

        return result

    async def _extract_text(self, path: Path, ext: str) -> str:
        """Extract text from various file formats."""
        if ext in {".txt", ".md", ".csv"}:
            return path.read_text(encoding="utf-8", errors="replace")

        elif ext == ".json":
            import json
            data = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(data, indent=2, ensure_ascii=False)

        elif ext == ".pdf":
            return await self._extract_pdf(path)

        elif ext == ".docx":
            return await self._extract_docx(path)

        return ""

    async def _extract_pdf(self, path: Path) -> str:
        """Extract text from PDF using PyPDF2."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append(f"--- Page {i + 1} ---\n{text}")
            return "\n\n".join(pages)
        except ImportError:
            return "[PDF support requires PyPDF2. Install with: pip install PyPDF2]"
        except Exception as e:
            raise RuntimeError(f"PDF extraction error: {e}")

    async def _extract_docx(self, path: Path) -> str:
        """Extract text from DOCX using python-docx."""
        try:
            from docx import Document
            doc = Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            return "[DOCX support requires python-docx. Install with: pip install python-docx]"
        except Exception as e:
            raise RuntimeError(f"DOCX extraction error: {e}")
