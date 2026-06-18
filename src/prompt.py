import json
from src.models import FuncitonDef


def build_prompt(query: str, functions: list[FuncitonDef]) -> str:

    fn_list = json.dumps(
        [f.model_dump() for f in functions],
        indent=2
    )

    return (
        f"You are a function calling assistant.\n"
        f"Given a user request, select the correct function and arguments.\n"
        f"Respond ONLY with a JSON object with keys 'fn_name' and 'args'.\n\n"
        f"Available functions:\n{fn_list}\n\n"
        f"User request: {query}\n\n"
        f"Response:"
    )
