import json

from src.models import FunctionDef


def build_prompt(query: str, functions: list[FunctionDef]) -> str:
    """Build the full prompt sent to the model for a single request.

    Args:
        query: The natural language user request.
        functions: Available function definitions.

    Returns:
        The prompt text, ending right before the JSON the model
        should generate.
    """
    schema = json.dumps([f.model_dump() for f in functions])
    return (
        f"Available Functions: {schema}\n"
        f"User: {query}\n"
        f"JSON: "
    )
