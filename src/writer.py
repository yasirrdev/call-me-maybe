import json
from pathlib import Path
from src.models import OutputEntry


def write_results(results: list[OutputEntry], path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.model_dump() for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
