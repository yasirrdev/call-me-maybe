from typing import Any

from pydantic import BaseModel, Field, field_validator


def _validate_non_empty(s: str) -> str:
    """Ensure a string is non-empty after stripping whitespace."""
    if not isinstance(s, str) or not s.strip():
        raise ValueError("String cannot be empty or just whitespace.")
    return s.strip()


class FunctionParam(BaseModel):
    """Type descriptor for a single function parameter or return value."""

    type: str


class FunctionDef(BaseModel):
    """A function definition loaded from function_definitions.json."""

    name: str
    description: str = Field(..., min_length=1)
    parameters: dict[str, FunctionParam] = Field(default_factory=dict)
    returns: FunctionParam

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v = _validate_non_empty(v)
        if not v.startswith("fn_"):
            raise ValueError(f"Function name must start with 'fn_': {v}")
        return v


class FunctionCall(BaseModel):
    """A decoded function call: selected function name plus arguments."""

    fn_name: str
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fn_name")
    @classmethod
    def _validate_fn_name(cls, v: str) -> str:
        v = _validate_non_empty(v)
        if v.startswith("error_"):
            return v
        if not v.startswith("fn_"):
            raise ValueError(f"Function name must start with 'fn_': {v}")
        return v


class OutputEntry(BaseModel):
    """A single entry in the final results JSON output."""

    prompt: str
    fn_name: str
    args: dict[str, Any]
