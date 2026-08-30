"""
HTTP routes for /health and /ask. Equivalent to a @RestController —
this layer stays thin: parse request, call the service, shape the
response. No business logic here.
"""

import logging
import time

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from config import LOGGER_NAME
from schemas.ask import AskRequest, AskResponse
from services.agent_service import agent

logger = logging.getLogger(LOGGER_NAME)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    start = time.perf_counter()

    try:
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=request.question)]
        })
        answer_text = result["messages"][-1].content

        latency_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"question_len={len(request.question)} "
            f"latency_ms={latency_ms:.2f}"
        )

        return AskResponse(
            answer=answer_text,
            latency_ms=round(latency_ms, 2),
            input_tokens=0,
            output_tokens=0,
        )

    except Exception:
        logger.exception("Agent invocation failed")
        raise HTTPException(
            status_code=502,
            detail="Failed to generate an answer from the agent.",
        ) from None
