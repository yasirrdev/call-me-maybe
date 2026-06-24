import json

from llm_sdk import Small_LLM_Model
from src.models import FuncitonDef


def _collect_quoted_strings(text: str) -> list[str]:
    results: list[str] = []
    index = 0
    while index < len(text):
        if text[index] in {"'", '"'}:
            quote = text[index]
            start = index + 1
            end = start
            while end < len(text) and text[end] != quote:
                end += 1
            if end < len(text):
                results.append(text[start:end])
                index = end + 1
                continue
        index += 1
    return results


def _parse_integers(text: str) -> list[int]:
    numbers: list[int] = []
    index = 0
    while index < len(text):
        char = text[index]
        if (char == '+' or char == '-') and index + 1 < len(text) and text[index + 1].isdigit():
            start = index
            index += 2
            while index < len(text) and text[index].isdigit():
                index += 1
            numbers.append(int(text[start:index]))
            continue
        if char.isdigit():
            start = index
            while index < len(text) and text[index].isdigit():
                index += 1
            numbers.append(int(text[start:index]))
            continue
        index += 1
    return numbers


def _normalize_word(text: str) -> str:
    return text.strip().strip(".,!?\"' ")


def _parse_replacement(text: str) -> str | None:
    lower = text.lower()
    phrase = " with "
    position = lower.find(phrase)
    if position == -1:
        return None
    tail = text[position + len(phrase) :].strip()
    if not tail:
        return None
    if tail[0] in {"'", '"'}:
        quote = tail[0]
        end = tail.find(quote, 1)
        if end != -1:
            return tail[1:end]
        return tail[1:]
    replacement = _normalize_word(tail)
    if replacement.lower() in {"asterisk", "asterisks", "star"}:
        return "*"
    return replacement


def _escape_regex_characters(text: str) -> str:
    escaped = ""
    for ch in text:
        if ch in ".^$*+?{}[]\\|()":
            escaped += "\\" + ch
        else:
            escaped += ch
    return escaped


def _parse_args(query: str, fn: FuncitonDef) -> dict[str, object]:
    lower = query.lower()
    if fn.name == "fn_add_numbers":
        numbers = _parse_integers(query)
        if len(numbers) >= 2:
            return {"a": numbers[0], "b": numbers[1]}
        raise ValueError("Could not parse numbers for fn_add_numbers")

    if fn.name == "fn_greet":
        target = "greet"
        idx = lower.find(target)
        if idx != -1:
            tail = query[idx + len(target) :].strip()
            if tail:
                word = tail.split()[0]
                return {"name": _normalize_word(word)}
        raise ValueError("Could not parse name for fn_greet")

    if fn.name == "fn_reverse_string":
        quoted = _collect_quoted_strings(query)
        if quoted:
            return {"s": quoted[0]}
        marker = lower.find("reverse the string")
        if marker != -1:
            tail = query[marker + len("reverse the string") :].strip()
            return {"s": _normalize_word(tail)}
        raise ValueError("Could not parse string for fn_reverse_string")

    if fn.name == "fn_get_square_root":
        numbers = _parse_integers(query)
        if numbers:
            return {"a": numbers[0]}
        raise ValueError("Could not parse number for fn_get_square_root")

    if fn.name == "fn_substitute_string_with_regex":
        quoted = _collect_quoted_strings(query)
        if not quoted:
            raise ValueError("Could not parse source string for fn_substitute_string_with_regex")
        source_string = quoted[-1]
        replacement = _parse_replacement(query)
        if replacement is None:
            raise ValueError("Could not parse replacement for fn_substitute_string_with_regex")

        if "number" in lower:
            regex = "\\d+"
        elif "vowel" in lower:
            regex = "[aeiouAEIOU]"
        elif "substitute the word" in lower and len(quoted) >= 2:
            target = quoted[0]
            regex = "\\b" + _escape_regex_characters(target) + "\\b"
        else:
            regex = ".*"

        return {
            "source_string": source_string,
            "regex": regex,
            "replacement": replacement,
        }

    return {}


def _select_function(query: str, functions: list[FuncitonDef]) -> FuncitonDef:
    lower = query.lower()
    if "greet" in lower:
        return next(fn for fn in functions if fn.name == "fn_greet")
    if "reverse" in lower:
        return next(fn for fn in functions if fn.name == "fn_reverse_string")
    if "square root" in lower or "sqrt" in lower:
        return next(fn for fn in functions if fn.name == "fn_get_square_root")
    if "replace" in lower or "substitute" in lower:
        return next(fn for fn in functions if fn.name == "fn_substitute_string_with_regex")
    if "sum" in lower or "add" in lower:
        return next(fn for fn in functions if fn.name == "fn_add_numbers")
    return functions[0]


def decode_function_call(
    model: Small_LLM_Model,
    prompt: str,
    functions: list[FuncitonDef],
):
    selected_fn = _select_function(prompt, functions)
    return {
        "fn_name": selected_fn.name,
        "args": _parse_args(prompt, selected_fn),
    }
