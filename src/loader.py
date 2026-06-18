import json
from pathlib import Path
from typing import List
from src.models import FuncitonDef


def load_tests(path: Path) -> List[str]:

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Tests file must be a JSON array.")
        return [str(item) for item in data]
    except FileNotFoundError:
        raise FileNotFoundError(f"Test file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tests file: {e}")


def load_functions(path: Path) -> List[FuncitonDef]:

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Functions file must be a JSON array.")
        return [FuncitonDef.model_validate(item) for item in data]
    except FileNotFoundError:
        raise FileNotFoundError(f"Test file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tests file: {e}")
