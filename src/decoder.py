"""Restricted decoding for structured JSON function-call output."""

import json
import math
import re
from typing import Any

import numpy as np

from llm_sdk import Small_LLM_Model
from src.models import FuncitonDef


NEG_INF: float = -math.inf


def _load_vocab(model: Small_LLM_Model) -> dict[str, int]:
    """Load vocabulary mapping from token string to token id."""
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)
    return vocab


def _clean_vocab(vocab: dict[str, int]) -> list[tuple[int, str]]:
    """Precompute decoded (id, text) pairs once, replacing BPE markers."""
    result: list[tuple[int, str]] = []
    for token_str, token_id in vocab.items():
        clean = token_str.replace("Ġ", " ").replace("Ċ", "\n")
        if clean:
            result.append((token_id, clean))
    return result


def _mask_logits(logits: list[float], valid_ids: list[int]) -> list[float]:
    """Set all logits to -inf except valid token ids."""
    masked = [NEG_INF] * len(logits)
    for tid in valid_ids:
        if tid < len(masked):
            masked[tid] = logits[tid]
    return masked


def _argmax(logits: list[float]) -> int:
    """Return index of maximum value."""
    return int(np.argmax(logits))


def _select_function(
        model: Small_LLM_Model,
        clean_vocab: list[tuple[int, str]],
        input_ids: list[int],
        functions: list[FuncitonDef]) -> FuncitonDef:
    """Select the function whose name best matches generated tokens."""
    fn_names = [fn.name for fn in functions]
    id_to_clean = dict(clean_vocab)

    generated = ""
    current_ids = list(input_ids)

    for i in range(64):
        candidates = [n for n in fn_names if n.startswith(generated)]
        if not candidates:
            break
        if len(candidates) == 1 and generated == candidates[0]:
            break

        logits = model.get_logits_from_input_ids(current_ids)

        valid_ids = [
            tid for tid, clean in clean_vocab
            if any((generated + clean).startswith(c) for c in candidates)
            or any(c.startswith(generated + clean) for c in candidates)
        ]

        if not valid_ids:
            break

        masked = _mask_logits(logits, valid_ids)
        next_id = _argmax(masked)
        clean = id_to_clean.get(next_id, "")
        generated += clean
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
    """Coerce a raw string value to the expected type."""
    type_hint = type_hint.lower()
    if type_hint == "number":
        return float(raw) if raw else 0.0
    if type_hint == "boolean":
        return raw.lower().startswith("true")
    return raw


def _decode_argument(
        model: Small_LLM_Model,
        clean_vocab: list[tuple[int, str]],
        input_ids: list[int],
        param_type: str,
        max_tokens: int = 16) -> str:
    """Generate a single argument value using restricted decoding."""
    id_to_clean = dict(clean_vocab)

    if param_type == "number":
        allowed_re = re.compile(r'^[\s\-0-9\.]+$')
    elif param_type == "boolean":
        allowed_re = re.compile(r'^[\sA-Za-z]+$')
    else:
        allowed_re = re.compile(r'^[^"\n,\}\]\:\(\)]+$')

    generated = ""
    current_ids = list(input_ids)

    forbidden_chars = ('"', '\n', ',', '}', ']', ':', '(', ')', ';')

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(current_ids)

        valid_ids = []
        for tid, clean in clean_vocab:
            if any(ch in clean for ch in forbidden_chars):
                continue
            candidate = generated + clean
            if allowed_re.fullmatch(candidate):
                valid_ids.append(tid)

        if not valid_ids:
            break

        masked = _mask_logits(logits, valid_ids)
        next_id = _argmax(masked)
        clean = id_to_clean.get(next_id, "")

        if any(ch in clean for ch in forbidden_chars):
            break

        if param_type != "string" and clean == " " and generated:
            break

        if param_type == "number":
            if re.fullmatch(r'^\s*-?\d+(?:\.\d+)?\s*$', generated):
                break

        generated += clean
        current_ids.append(next_id)

    return generated.strip()


def decode_function_call(
        model: Small_LLM_Model,
        prompt: str,
        functions: list[FuncitonDef]) -> dict[str, Any] | None:
    """Run restricted decoding to produce a function call from a prompt."""
    try:
        vocab = _load_vocab(model)
        clean_vocab = _clean_vocab(vocab)
        input_tensor = model.encode(prompt)
        input_ids: list[int] = input_tensor[0].tolist()

        selected_fn = _select_function(
            model, clean_vocab, input_ids, functions
        )
        args: dict[str, Any] = {}

        for param_name, param_schema in selected_fn.parameters.items():
            param_type = param_schema.type.lower()

            hint_text = f' "{param_name}": '
            hint_ids_tensor = model.encode(hint_text)
            hint_ids: list[int] = hint_ids_tensor[0].tolist()

            start_ids = list(input_ids) + hint_ids

            raw_value = _decode_argument(
                model, clean_vocab, start_ids, param_type, max_tokens=64
            )

            coerced = _coerce_value(raw_value, param_type)
            args[param_name] = coerced

        return {"fn_name": selected_fn.name, "args": args}

    except Exception as e:
        raise RuntimeError(f"Decoding failed: {e}") from e
