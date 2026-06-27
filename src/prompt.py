import json

from src.models import FunctionDef


def build_prompt(query: str, functions: list[FunctionDef]) -> str:
    fn_list = json.dumps(
        [f.model_dump() for f in functions],
        indent=2,
    )
    return (
        "You are a function calling assistant.\n"
        "Given a user request, select the correct function "
        "and arguments.\n"
        "Available functions:\n"
        f"{fn_list}\n\n"
        f"User request: {query}\n\n"
        "Function name:"
    )


def build_argument_prompt(param_name: str) -> str:
    return f'\n{param_name} = "'
