"""
NEXA AI - Data Extraction Tool
Processes structured data (CSV, JSON) to generate summaries and analysis.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json
import csv
import io
import logging

from app.tools.registry import BaseTool

logger = logging.getLogger("nexa.tools.data_extraction")

MAX_ROWS_PREVIEW = 100
MAX_CELL_LENGTH = 500


class DataExtractionTool(BaseTool):
    name = "data_extraction"
    description = (
        "Analyze structured data from CSV or JSON files/content. "
        "Returns a summary including column names, row counts, data types, "
        "sample rows, and basic statistics. Use this for data exploration and reporting."
    )
    timeout = 20
    permission = "allowed"

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to a CSV or JSON data file",
                },
                "raw_data": {
                    "type": "string",
                    "description": "Raw CSV or JSON string (alternative to file_path)",
                },
                "format": {
                    "type": "string",
                    "enum": ["csv", "json", "auto"],
                    "description": "Data format",
                    "default": "auto",
                },
                "operation": {
                    "type": "string",
                    "enum": ["summary", "preview", "stats"],
                    "description": (
                        "'summary' = overview, 'preview' = first N rows, "
                        "'stats' = basic statistics"
                    ),
                    "default": "summary",
                },
            },
            "required": [],
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        file_path = input_data.get("file_path", "").strip()
        raw_data = input_data.get("raw_data", "").strip()
        fmt = input_data.get("format", "auto")
        operation = input_data.get("operation", "summary")

        # Get data content
        content = ""
        source = "inline"
        if file_path:
            path = Path(file_path)
            if not path.exists():
                return {"error": f"File not found: {file_path}"}
            content = path.read_text(encoding="utf-8", errors="replace")
            source = path.name
            if fmt == "auto":
                fmt = "json" if path.suffix.lower() == ".json" else "csv"
        elif raw_data:
            content = raw_data
            if fmt == "auto":
                fmt = "json" if content.lstrip().startswith(("{", "[")) else "csv"
        else:
            return {"error": "Provide either 'file_path' or 'raw_data'"}

        try:
            if fmt == "json":
                return self._analyze_json(content, source, operation)
            else:
                return self._analyze_csv(content, source, operation)
        except Exception as e:
            logger.error(f"Data extraction error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def _analyze_csv(self, content: str, source: str, operation: str) -> Dict[str, Any]:
        reader = csv.DictReader(io.StringIO(content))
        rows = []
        for i, row in enumerate(reader):
            if i >= MAX_ROWS_PREVIEW:
                break
            rows.append({k: v[:MAX_CELL_LENGTH] if v else "" for k, v in row.items()})

        if not rows:
            return {"error": "No data rows found in CSV"}

        columns = list(rows[0].keys())
        result: Dict[str, Any] = {
            "source": source,
            "format": "csv",
            "columns": columns,
            "column_count": len(columns),
            "row_count": len(rows),
            "note": f"Showing up to {MAX_ROWS_PREVIEW} rows",
        }

        if operation == "preview":
            result["rows"] = rows[:10]
        elif operation == "stats":
            result["column_stats"] = self._compute_column_stats(rows, columns)
        else:  # summary
            result["sample_rows"] = rows[:5]
            result["column_stats"] = self._compute_column_stats(rows, columns)

        return result

    def _analyze_json(self, content: str, source: str, operation: str) -> Dict[str, Any]:
        data = json.loads(content)
        result: Dict[str, Any] = {
            "source": source,
            "format": "json",
        }

        if isinstance(data, list):
            result["type"] = "array"
            result["length"] = len(data)
            if data and isinstance(data[0], dict):
                result["keys"] = list(data[0].keys())
                result["sample"] = data[:5]
            else:
                result["sample"] = data[:10]
        elif isinstance(data, dict):
            result["type"] = "object"
            result["keys"] = list(data.keys())
            result["key_count"] = len(data)
            # Summarize each key's value type
            result["structure"] = {
                k: type(v).__name__ for k, v in data.items()
            }
        else:
            result["type"] = type(data).__name__
            result["value"] = str(data)[:1000]

        return result

    def _compute_column_stats(self, rows: list, columns: list) -> Dict[str, Any]:
        stats = {}
        for col in columns:
            values = [r.get(col, "") for r in rows if r.get(col)]
            numeric = []
            for v in values:
                try:
                    numeric.append(float(v))
                except (ValueError, TypeError):
                    pass

            col_stat: Dict[str, Any] = {
                "non_empty": len(values),
                "empty": len(rows) - len(values),
                "unique": len(set(values)),
            }
            if numeric:
                col_stat["numeric_count"] = len(numeric)
                col_stat["min"] = min(numeric)
                col_stat["max"] = max(numeric)
                col_stat["avg"] = round(sum(numeric) / len(numeric), 2)
            else:
                col_stat["type_hint"] = "text"
                if values:
                    col_stat["sample_values"] = list(set(values))[:5]

            stats[col] = col_stat
        return stats
