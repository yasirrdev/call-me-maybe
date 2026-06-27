from typing import Any

from pydantic import BaseModel


class FunctionParam(BaseModel):
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, FunctionParam]
    returns: FunctionParam


class FunctionCall(BaseModel):
    fn_name: str
    args: dict[str, Any]


class OutputEntry(BaseModel):
    prompt: str
    fn_name: str
    args: dict[str, Any]
