import json
from pathlib import Path

from src.models import FunctionDef


def load_tests(path: Path) -> list[str]:
    """Load and validate natural language prompts from a JSON file.

    Args:
        path: Path to the tests JSON file.

    Returns:
        A list of non-empty prompt strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or has an invalid shape.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Tests file must be a JSON array.")
        prompts: list[str] = []
        for item in data:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict) and "prompt" in item:
                text = str(item["prompt"])
            else:
                raise ValueError(
                    "Each test entry must be a string or an "
                    "object with a 'prompt' key."
                )
            if not text.strip():
                raise ValueError("Prompt cannot be empty.")
            prompts.append(text.strip())
        return prompts
    except FileNotFoundError:
        raise FileNotFoundError(f"Test file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tests file: {e}")


def load_functions(path: Path) -> list[FunctionDef]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the function definitions JSON file.

    Returns:
        A list of validated FunctionDef instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or fails validation.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Functions file must be a JSON array.")
        return [FunctionDef.model_validate(item) for item in data]
    except FileNotFoundError:
        raise FileNotFoundError(f"Functions file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in functions file: {e}")
