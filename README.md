*This project has been created as part of the 42 curriculum by ybel-maa.*

# Call Me Maybe

## Description

Call Me Maybe is a function-calling system built around the Qwen3-0.6B language model.

The goal of the project is to transform natural language requests into structured function calls composed of:

- Function name
- Typed arguments

The implementation uses a constrained decoding approach that limits the model's available tokens during generation in order to improve reliability and output consistency.

---

## Features

- Function selection using Qwen3-0.6B
- Token-level constrained decoding
- Pydantic schema validation
- JSON input/output support
- Error handling for malformed files
- Type-safe Python implementation

---

## Project Structure

```text
.
├── src/
│   ├── decoder.py
│   ├── loader.py
│   ├── models.py
│   ├── prompt.py
│   ├── writer.py
│   └── __main__.py
├── llm_sdk/
├── data/
│   └── input/
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yasirrdev/Call-Me-Maybe.git
cd Call-Me-Maybe
uv sync
```

---

## Usage

```bash
uv run python -m src
```

Custom paths:

```bash
uv run python -m src \
    --input data/input \
    --output data/output/function_calling_results.json
```

---

## Algorithm Explanation

### Function Selection

The user request and available function definitions are included in a prompt sent to the model.

The decoder progressively generates a function name by restricting valid tokens to those that match existing function names.

At each generation step:

1. Model logits are obtained.
2. Invalid token candidates are masked.
3. The highest-scoring valid token is selected.
4. Generation continues until a valid function name is produced.

### Argument Generation

After selecting a function, each parameter is generated independently.

Different token restrictions are applied according to parameter type:

- number
- boolean
- string

The generated value is then converted to the expected Python type.

---

## Design Decisions

### Pydantic

All function definitions and output structures use Pydantic models.

### Modular Architecture

The project separates:

- Input loading
- Prompt generation
- Decoding
- Validation
- Output writing

### Token Restrictions

Instead of relying entirely on prompting, generation is guided through token filtering based on expected outputs.

---

## Performance Analysis

The implementation prioritizes correctness over generation creativity.

Expected characteristics:

- Fast execution
- Low memory usage
- Improved function selection consistency
- Better structured output than unrestricted generation

---

## Challenges Encountered

- Vocabulary handling
- Restricted generation
- Error management
- Type validation

---

## Testing Strategy

The implementation was validated through:

- Input validation
- Decoder validation
- Type conversion tests
- Edge-case testing

---

## Example

Input:

```text
What is the sum of 40 and 2?
```

Output:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "fn_name": "fn_add_numbers",
  "args": {
    "a": 40,
    "b": 2
  }
}
```

---

## Makefile Commands

```bash
make install
make run
make debug
make clean
make lint
make lint-strict
```

---

## Resources

### Documentation

- Python Documentation
- Pydantic Documentation
- NumPy Documentation
- Qwen Documentation

### AI Usage

AI tools were used for:

- Understanding constrained decoding concepts
- Reviewing implementation ideas
- Exploring testing strategies
- Improving project documentation

All implementation details were manually reviewed and integrated into the final solution.
