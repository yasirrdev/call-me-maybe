from typing import Any, Dict
from pydantic import BaseModel


class FunctionParam(BaseModel):
    type: str


class FuncitonDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, FunctionParam]
    returns: FunctionParam


class FunctionCall(BaseModel):
    fn_name: str
    args: Dict[str, Any]


class OutputEntry(BaseModel):
    prompt: str
    fn_name: str
    args: Dict[str, Any]
