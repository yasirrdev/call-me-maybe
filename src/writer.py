import json
from pathlib import Path
from typing import List
from src.models import OutputEntry


def write_results(results: List[OutputEntry], path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.model_dump() for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
