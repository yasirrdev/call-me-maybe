import json
from typing import Callable

import numpy as np
import sys

from llm_sdk import Small_LLM_Model
from src.models import FunctionCall, FunctionDef
from src.prompt import build_argument_prompt, build_prompt

Vocab = dict[int, str]

MAX_FN_NAME_TOKENS = 24
MAX_ARG_TOKENS = 8

_CLEAN_MAP = (("Ġ", " "), ("Ċ", "\n"))
_NUMBER_CHARS = set("0123456789.")
_NUMBER_START_CHARS = set("0123456789.-")
_BOOL_OPTIONS = ("true", "false")
_FORBIDDEN_STRING_CHARS = {'"', "\n"}


def _clean_token(token: str) -> str:
    for raw, repl in _CLEAN_MAP:
        token = token.replace(raw, repl)
    return token


def build_vocab(model: Small_LLM_Model) -> Vocab:
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab: dict[str, int] = json.load(f)
    return {
        token_id: _clean_token(tok)
        for tok, token_id in raw_vocab.items()
    }


def _encode_ids(model: Small_LLM_Model, text: str) -> list[int]:
    return model.encode(text)[0].tolist()


def _masked_argmax(logits: list[float], valid_ids: list[int]) -> int:
    logits_arr = np.asarray(logits, dtype=np.float32)
    mask = np.full(logits_arr.shape, -np.inf, dtype=np.float32)
    mask[valid_ids] = logits_arr[valid_ids]
    return int(np.argmax(mask))


def _step(
    model: Small_LLM_Model,
    ids: list[int],
    is_valid: Callable[[str], bool],
    vocab: Vocab,
) -> tuple[int, str] | None:
    logits = model.get_logits_from_input_ids(ids)
    valid_ids = [
        token_id
        for token_id, token_text in vocab.items()
        if token_text and token_id < len(logits) and is_valid(token_text)
    ]
    if not valid_ids:
        return None
    next_id = _masked_argmax(logits, valid_ids)
    return next_id, vocab[next_id]


def _generate_function_name(
    model: Small_LLM_Model,
    vocab: Vocab,
    base_ids: list[int],
    names: list[str],
) -> str:
    ids = list(base_ids)
    generated = ""
    remaining = list(names)
    for _ in range(MAX_FN_NAME_TOKENS):
        if remaining == [generated]:
            break

        def is_valid(token_text: str, _gen: str = generated) -> bool:
            prefix = _gen + token_text
            return any(name.startswith(prefix) for name in remaining)

        result = _step(model, ids, is_valid, vocab)
        if result is None:
            break
        next_id, token_text = result
        generated += token_text
        ids.append(next_id)
        remaining = [n for n in remaining if n.startswith(generated)]
    return generated


def _is_valid_number_char(text: str, is_first: bool) -> bool:
    allowed = _NUMBER_START_CHARS if is_first else _NUMBER_CHARS
    stripped = text.strip()
    return bool(stripped) and all(c in allowed for c in stripped)


def _generate_number(
    model: Small_LLM_Model,
    vocab: Vocab,
    base_ids: list[int],
) -> float:
    ids = list(base_ids)
    generated = ""
    for step in range(MAX_ARG_TOKENS):
        is_first = step == 0

        def is_valid(token_text: str, _first: bool = is_first) -> bool:
            return _is_valid_number_char(token_text, _first)

        result = _step(model, ids, is_valid, vocab)
        if result is None:
            break
        next_id, token_text = result
        generated += token_text.strip()
        ids.append(next_id)
    if not generated:
        raise ValueError("Model failed to generate a valid number")
    return float(generated)


def _generate_boolean(
    model: Small_LLM_Model,
    vocab: Vocab,
    base_ids: list[int],
) -> bool:
    ids = list(base_ids)
    generated = ""
    remaining = list(_BOOL_OPTIONS)
    for _ in range(MAX_ARG_TOKENS):
        if remaining == [generated]:
            break

        def is_valid(token_text: str, _gen: str = generated) -> bool:
            cleaned = token_text.strip().lower()
            if not cleaned:
                return False
            prefix = _gen + cleaned
            return any(opt.startswith(prefix) for opt in remaining)

        result = _step(model, ids, is_valid, vocab)
        if result is None:
            break
        next_id, token_text = result
        generated += token_text.strip().lower()
        ids.append(next_id)
        remaining = [o for o in remaining if o.startswith(generated)]
    if generated not in _BOOL_OPTIONS:
        raise ValueError("Model failed to generate a valid boolean")
    return generated == "true"


def _generate_string(
    model: Small_LLM_Model,
    vocab: Vocab,
    base_ids: list[int],
) -> str:
    ids = list(base_ids)
    generated = ""

    def is_valid(token_text: str) -> bool:
        return "\n" not in token_text

    for _ in range(MAX_ARG_TOKENS):
        result = _step(model, ids, is_valid, vocab)
        if result is None:
            break
        next_id, token_text = result
        if '"' in token_text:
            break
        generated += token_text
        ids.append(next_id)
    return generated.strip()


def _generate_argument(
    model: Small_LLM_Model,
    vocab: Vocab,
    base_ids: list[int],
    arg_type: str,
) -> object:
    if arg_type == "number":
        return _generate_number(model, vocab, base_ids)
    if arg_type == "boolean":
        return _generate_boolean(model, vocab, base_ids)
    return _generate_string(model, vocab, base_ids)


def decode_function_call(
    model: Small_LLM_Model,
    prompt: str,
    functions: list[FunctionDef],
    vocab: Vocab | None = None,
) -> FunctionCall | None:
    if vocab is None:
        vocab = build_vocab(model)

    base_ids = _encode_ids(model, build_prompt(prompt, functions))
    names = [fn.name for fn in functions]
    fn_name = _generate_function_name(model, vocab, base_ids, names)

    selected_fn = next(
        (fn for fn in functions if fn.name == fn_name), None
    )
    if selected_fn is None:
        print(
            f"DEBUG fn_name generated: {fn_name!r} for prompt: {prompt!r}",
            file=sys.stderr,
        )
        return None

    args: dict[str, object] = {}
    current_ids = base_ids + _encode_ids(model, fn_name)
    for param_name, param in selected_fn.parameters.items():
        current_ids += _encode_ids(
            model, build_argument_prompt(param_name)
        )
        try:
            value = _generate_argument(
                model, vocab, current_ids, param.type
            )
        except ValueError as e:
            print(
                f"DEBUG arg fail param={param_name!r} "
                f"type={param.type!r} error={e} prompt={prompt!r}",
                file=sys.stderr,
            )
            return None
        args[param_name] = value
        current_ids += _encode_ids(model, str(value))

    return FunctionCall(fn_name=fn_name, args=args)
