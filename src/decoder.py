import json

import numpy as np
import re

from llm_sdk import Small_LLM_Model
from src.models import FunctionCall, FunctionDef
from src.prompt import build_prompt

Vocab = dict[int, str]

MAX_GENERATION_STEPS = 128
_CLEAN_MAP = (("Ġ", " "), ("Ċ", "\n"), ("厚", ""))
_ALLOWED_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz0123456789_ \":{},"
    ".[]-()!@#$%^&*/\\|;'<>?~`=+"
)
_BAD_SUBSTRINGS = ("\n", "\r", "\t", "User:", "Result:", "returns:")
_FN_NAME_PREFIX = '{"name": "'

_INVALID_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def _sanitize_json_escapes(text: str) -> str:
    """Escape backslashes that are not valid JSON escape sequences.

    The model sometimes emits Python/regex-style escapes (e.g. \\d, \\')
    which are invalid inside JSON strings. This doubles any backslash
    not followed by a valid JSON escape character.

    Args:
        text: Raw text possibly containing invalid escapes.

    Returns:
        Text safe to pass to json.loads.
    """
    return _INVALID_ESCAPE.sub(r"\\\\", text)


def _clean_token(token: str) -> str:
    """Normalize a generated token by replacing known cleanup markers."""
    for raw, repl in _CLEAN_MAP:
        token = token.replace(raw, repl)
    return token


def build_vocab(model: Small_LLM_Model) -> Vocab:
    """Load and clean the model vocabulary.

    Args:
        model: The loaded LLM wrapper.

    Returns:
        Mapping of token id to its cleaned text representation.
    """
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab: dict[str, int] = json.load(f)
    return {
        token_id: _clean_token(tok)
        for tok, token_id in raw_vocab.items()
    }


def build_small_vocab(vocab: Vocab) -> Vocab:
    """Filter the vocabulary down to JSON-safe tokens, computed once.

    Args:
        vocab: The full cleaned vocabulary.

    Returns:
        A reduced vocabulary containing only tokens whose characters
        are all valid inside a JSON-ish output string.
    """
    return {
        token_id: text
        for token_id, text in vocab.items()
        if text and all(c in _ALLOWED_CHARS for c in text.lower())
    }


def _is_valid_token(
    current_text: str,
    new_token: str,
    quote_count: int,
    has_parameters: bool,
    valid_fn_names: set[str],
) -> bool:
    """Return whether a new token keeps the partial JSON structure valid."""
    if any(bad in new_token for bad in _BAD_SUBSTRINGS):
        return False

    if len(current_text) < len(_FN_NAME_PREFIX):
        return _FN_NAME_PREFIX.startswith(current_text + new_token)

    if quote_count == 3:
        name_so_far = current_text.split(_FN_NAME_PREFIX)[1]
        if new_token == '"':
            return name_so_far in valid_fn_names
        if '"' in new_token:
            return (
                name_so_far + new_token
            ).split('"')[0] in valid_fn_names
        return any(
            name.startswith(name_so_far + new_token)
            for name in valid_fn_names
        )

    if quote_count >= 4 and not has_parameters:
        bridge = ', "parameters": {'
        parts = current_text.split('"')
        prefix_to_end_of_name = '"'.join(parts[:4]) + '"'
        bridge_so_far = current_text[len(prefix_to_end_of_name):]
        return bridge.startswith(bridge_so_far + new_token)

    if has_parameters:
        param_content = current_text.split('"parameters": {')[1]
        if "}" in param_content:
            if current_text.count("}") >= 2:
                return False
            return new_token.strip() == "}"
        return not ("}," in new_token or ",}" in new_token)

    return False


def _generate_json(
    model: Small_LLM_Model,
    vocab: Vocab,
    small_vocab: Vocab,
    base_ids: list[int],
    valid_fn_names: set[str],
) -> str:
    """Generate a JSON-like function call string under token constraints."""
    ids = list(base_ids)
    current_text = ""

    for _ in range(MAX_GENERATION_STEPS):
        logits = np.asarray(
            model.get_logits_from_input_ids(ids), dtype=np.float32
        )
        quote_count = current_text.count('"')
        has_params = '"parameters": {' in current_text

        valid_ids = [
            token_id
            for token_id, text in small_vocab.items()
            if token_id < len(logits)
            and _is_valid_token(
                current_text, text, quote_count, has_params, valid_fn_names
            )
        ]
        if not valid_ids:
            break

        best_id = max(valid_ids, key=lambda tid: logits[tid])
        ids.append(best_id)
        current_text += vocab[best_id]

        if current_text.strip().endswith("}}"):
            break

    return current_text.strip()


def _attempt_repair(raw_output: str) -> dict[str, object] | None:
    """Try to recover valid JSON from output with an unescaped quote.

    Args:
        raw_output: The raw text produced by generation.

    Returns:
        The parsed dict if repair succeeded, otherwise None.
    """
    last_brace = raw_output.rfind("}}")
    if last_brace == -1:
        return None
    candidate = _sanitize_json_escapes(raw_output[: last_brace + 2])
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def decode_function_call(
    model: Small_LLM_Model,
    prompt: str,
    functions: list[FunctionDef],
    vocab: Vocab | None = None,
    small_vocab: Vocab | None = None,
) -> FunctionCall | None:
    """Run constrained decoding to produce a single function call.

    Generates the full JSON object in one continuous token stream,
    using a state machine over the text generated so far to decide
    which tokens are valid at each step.

    Args:
        model: The loaded LLM wrapper.
        prompt: The natural language user request.
        functions: Available function definitions.
        vocab: Precomputed full vocabulary (built if not given).
        small_vocab: Precomputed JSON-safe vocabulary subset.

    Returns:
        The decoded function call, or None if generation failed or
        produced invalid JSON.
    """
    if vocab is None:
        vocab = build_vocab(model)
    if small_vocab is None:
        small_vocab = build_small_vocab(vocab)

    base_ids = model.encode(build_prompt(prompt, functions))[0].tolist()
    valid_fn_names = {fn.name for fn in functions}

    raw_output = _generate_json(
        model, vocab, small_vocab, base_ids, valid_fn_names
    )
    raw_output = _sanitize_json_escapes(raw_output)

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        repaired = _attempt_repair(raw_output)
        if repaired is None:
            return None
        parsed = repaired
    if not isinstance(parsed, dict) or "name" not in parsed:
        return None

    params = parsed.get("parameters", {})
    if not isinstance(params, dict):
        params = {}

    selected_fn = next(
        (fn for fn in functions if fn.name == parsed["name"]), None
    )
    if selected_fn is None:
        return None

    cleaned_args: dict[str, object] = {}
    for key, value in params.items():
        if key not in selected_fn.parameters:
            continue
        if isinstance(value, str):
            value = value.strip()
        expected_type = selected_fn.parameters[key].type
        try:
            if expected_type == "number":
                value = float(value)
            elif expected_type == "integer":
                value = int(value)
        except (ValueError, TypeError):
            pass
        cleaned_args[key] = value

    try:
        return FunctionCall(fn_name=selected_fn.name, args=cleaned_args)
    except Exception:
        return None
