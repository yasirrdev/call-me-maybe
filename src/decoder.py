import json
import math
from typing import Any
import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FuncitonDef


NEG_INF: float = -math.inf


def _load_vocab(model: Small_LLM_Model) -> dict[str, int]:

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)
    return vocab


def _get_valid_token_ids(
        vocab: dict[str, int],
        allowed_chars: str) -> list[int]:

    valid: list[int] = []
    for token_str, token_id in vocab.items():
        if token_str and token_str[0] in allowed_chars:
            valid.append(token_id)
    return valid


def _mask_logits(
        logits: list[float],
        valid_ids: list[int]) -> list[float]:

    masked = [NEG_INF] * len(logits)
    for tid in valid_ids:
        if tid < len(masked):
            masked[tid] = logits[tid]
    return masked


def _argmax(logits: list[float]) -> int:
    return int(np.argmax(logits))


def _build_json_schema(fn_def: FuncitonDef) -> dict[str, Any]:
    return {
        "fn_name": fn_def.name,
        "args": {k: None for k in fn_def.parameters}
    }


def _select_function(
        model: Small_LLM_Model,
        vocab: dict[str, int],
        input_ids: list[int],
        functions: list[FuncitonDef]) -> FuncitonDef:

    fn_names = [fn.name for fn in functions]
    id_to_token = {v: k for k, v in vocab.items()}

    generated = ""
    current_ids = list(input_ids)

    for _ in range(64):
        logits = model.get_logits_from_input_ids(current_ids)

        candidates = [n for n in fn_names if n.startswith(generated)]
        if not candidates:
            break

        if len(candidates) == 1 and generated == candidates[0]:
            break

        allowed_chars = {c[len(generated)] for c in candidates}
        valid_ids = _get_valid_token_ids(vocab, "".join(allowed_chars))

        if not valid_ids:
            break

        masked = _mask_logits(logits, valid_ids)
        next_id = _argmax(masked)
        token_str = id_to_token.get(next_id, "")
        generated += token_str
        current_ids.append(next_id)

        if any(generated == n for n in fn_names):
            break

    for fn in functions:
        if fn.name == generated:
            return fn

    best = max(functions, key=lambda f: len(
        [1 for a, b in zip(f.name, generated) if a == b]
    ))

    return best


def _coerce_value(raw: str, type_hint: str) -> Any:

    type_hint = type_hint.lower()
    if type_hint == "number":
        return float(raw)
    if type_hint == "boolean":
        return raw.lower() == "true"
    return raw


def _decode_argument(
        model: Small_LLM_Model,
        vocab: dict[str, int],
        input_ids: list[int],
        param_type: str,
        max_tokens: int = 32) -> str:

    id_to_token = {v: k for k, v in vocab.items()}

    if param_type == "number":
        allowed = "0123456789.-"
    elif param_type == "boolean":
        allowed = "tf"
    else:
        allowed = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789 .,!?-_"
        )
        stop_chars = {'"'}

    generated = ""
    current_ids = list(input_ids)

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(current_ids)
        current_allowed = (allowed if not generated
                           else allowed + "".join(stop_chars))
        valid_ids = _get_valid_token_ids(vocab, current_allowed)

        if not valid_ids:
            break

        masked = _mask_logits(logits, valid_ids)
        next_id = _argmax(masked)
        token_str = id_to_token.get(next_id, "")

        if any(c in token_str for c in stop_chars):
            break

        generated += token_str
        current_ids.append(next_id)

        if not generated:
            continue

    return generated.strip()


def decode_function_call(
        model: Small_LLM_Model,
        prompt: str,
        functions: list[FuncitonDef]) -> dict[str, Any] | None:

    try:
        vocab = _load_vocab(model)
        input_tensor = model.encode(prompt)
        input_ids: list[int] = input_tensor[0].tolist()

        selected_fn = _select_function(
            model, vocab, input_ids, functions
        )

        args: dict[str, Any] = {}
        for param_name, param_def in selected_fn.parameters.items():
            raw = _decode_argument(
                model, vocab, input_ids, param_def.type
            )
            args[param_name] = _coerce_value(raw, param_def.type)
        return {"fn_name": selected_fn.name, "args": args}

    except Exception as e:
        raise RuntimeError(f"Decoding failed: {e}") from e
