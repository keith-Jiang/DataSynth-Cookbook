"""JSONL I/O and progress utilities."""
import json
from pathlib import Path
from typing import List, Dict, Any, Iterator


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read all lines from a JSONL file. Returns empty list if file missing."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, data: List[Dict[str, Any]], ensure_ascii: bool = False) -> None:
    """Write data to a JSONL file (overwrites)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")


def append_jsonl(path: str, record: Dict[str, Any], ensure_ascii: bool = False) -> None:
    """Append a single record to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Lazily iterate over a large JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
