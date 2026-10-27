"""Readers for ISOTOPIA text input and YANDF-style .act output files.

This is a first Python port of the file-handling layer, not a complete
replacement for ISOTOPIA's Fortran calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


def _parse_scalar(text: str) -> Any:
    """Parse a simple numeric value, otherwise preserve it as text."""
    value = text.strip()
    try:
        # Fortran input files commonly use values such as 16. and 1.0E-3.
        return float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return value


def read_input(path: str | Path) -> dict[str, Any]:
    """Read simple keyword/value lines from an ISOTOPIA .inp file.

    Comments and blank lines are ignored. Keywords are case-insensitive.
    Both `project` and `projectile` are accepted as aliases.
    """
    result: dict[str, Any] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(None, 1)
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected a keyword and value, got {raw_line!r}"
                )
            key, value = fields
            key = key.lower()
            if key == "project":
                key = "projectile"
            result[key] = _parse_scalar(value)
    return result


@dataclass
class ActFile:
    """A parsed isotope-specific .act file.

    `metadata` stores header values as strings; `columns` maps stable,
    Python-friendly column names to lists of floats. Original header text
    is retained in `header_lines` so no source information is discarded.
    """
    path: Path
    metadata: dict[str, str]
    columns: dict[str, list[float]]
    header_lines: list[str]

    def __len__(self) -> int:
        return len(self.columns.get("time_s", []))

    def column(self, name: str) -> list[float]:
        """Return a data column or raise a useful error."""
        try:
            return self.columns[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown column {name!r}. Available columns: {', '.join(self.columns)}"
            ) from exc


# The names below follow the seven columns written by prodout.f90.
_ACT_COLUMNS = (
    "time_s",
    "activity_bq",
    "specific_activity_bq_g",
    "production_rate_bq_per_mA_h",
    "isotope_inventory",
    "isotope_fraction",
    "time_h",
)


def read_act(path: str | Path) -> ActFile:
    """Read an ISOTOPIA .act file's metadata and seven-column data block."""
    path = Path(path)
    header_lines: list[str] = []
    rows: list[list[float]] = []
    metadata: dict[str, str] = {}

    # Preserve all comment/header lines and extract common flat metadata fields.
    key_value = re.compile(r"^\s*#\s*([^:#][^:]*?):\s*(.*?)\s*$")
    with path.open(encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            stripped = raw_line.strip()
            if stripped.startswith("#"):
                header_lines.append(raw_line.rstrip("\n"))
                match = key_value.match(raw_line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    # Keep the first occurrence of repeated generic YAML keys
                    # such as "title" or "type"; parameter labels remain unique.
                    metadata.setdefault(key, value)
                continue
            if not stripped:
                continue
            fields = stripped.split()
            try:
                values = [float(field.replace("D", "E").replace("d", "e")) for field in fields]
            except ValueError:
                # Non-numeric lines outside the comment block are ignored.
                continue
            if len(values) == 7:
                rows.append(values)

    if not rows:
        raise ValueError(f"No seven-column numeric data block found in {path}")

    columns = {name: [row[i] for row in rows] for i, name in enumerate(_ACT_COLUMNS)}
    return ActFile(path=path, metadata=metadata, columns=columns, header_lines=header_lines)
