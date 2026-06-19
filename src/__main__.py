
"""Entry point for call me maybe: function calling with restricted decoding."""

import argparse
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.decoder import decode_function_call
from src.loader import load_functions, load_tests
from src.models import OutputEntry
from src.prompt import build_prompt
from src.writer import write_results

DEFAULT_INPUT = Path("data/input")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Function calling with restricted decoding."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to input directory or file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the function calling pipeline."""
    args = parse_args()

    input_path = args.input
    if input_path.is_dir():
        tests_path = input_path / "function_calling_tests.json"
        fns_path = input_path / "function_definitions.json"
    else:
        tests_path = input_path
        fns_path = input_path.parent / "function_definitions.json"

    try:
        prompts = load_tests(tests_path)
        functions = load_functions(fns_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading input files: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        model = Small_LLM_Model()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    for prompt in prompts:
        try:
            full_prompt = build_prompt(prompt, functions)
            call = decode_function_call(model, full_prompt, functions)
            if call is None:
                print(
                    f"Warning: no result for prompt: {prompt!r}",
                    file=sys.stderr,
                )
                continue
            results.append(
                OutputEntry(
                    prompt=prompt,
                    fn_name=call["fn_name"],
                    args=call["args"],
                )
            )
        except Exception as e:
            print(
                f"Error processing prompt {prompt!r}: {e}",
                file=sys.stderr,
            )

    try:
        write_results(results, args.output)
        print(f"Results written to {args.output}")
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
