"""
Request/response schemas for the /ask endpoint.
Equivalent to Java DTOs (@RequestBody / @ResponseBody classes).
"""

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
