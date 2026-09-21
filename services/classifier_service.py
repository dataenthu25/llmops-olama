"""
Tool-selection classifier: decides whether a question needs no_tool,
search_documents, or get_current_weather.

Two backends, in priority order:
  1. Jev (TypeSafe AI) — used if TYPESAFE_API_KEY is configured. A
     purpose-built "System One" decision model: no text generation,
     returns a typed, calibrated answer in one parallel pass. Since
     this is exactly the kind of narrow routing decision Jev is built
     for, it's a reasonable candidate to replace the local fine-tune
     for this task specifically.
  2. Local LoRA fine-tune (Phase 4.5) — used as a fallback if Jev is
     not configured, or if a Jev call fails for any reason. This is
     the same defensive-fallback pattern used elsewhere in this project
     (e.g. try_parse_fallback_tool_call in agent_service.py): never let
     a new dependency become a hard single point of failure.

Both paths return the same three labels, so callers (agent_service.py)
don't need to know or care which backend actually answered.
"""

import logging
from typing import Literal

from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LOGGER_NAME, TYPESAFE_API_KEY

logger = logging.getLogger(LOGGER_NAME)

VALID_LABELS = {"no_tool_needed", "use_search_documents", "use_get_current_weather"}

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./models/lora-tool-selector-final"


# ─────────────────────────────────────────────────────────────
# Backend 1: Jev (optional — only set up if an API key is present)
# ─────────────────────────────────────────────────────────────

_jev_available = False

if TYPESAFE_API_KEY:
    try:
        import jev

        class ToolDecision(BaseModel):
            label: Literal["no_tool_needed", "use_search_documents", "use_get_current_weather"]

        @jev.fn
        def _classify_with_jev_impl(question: str) -> ToolDecision:
            """Decide what, if anything, is needed to answer this question:
            {{ question }}

            - no_tool_needed: general knowledge, math, or reasoning questions
              that don't require external data
            - use_search_documents: questions about this project's own
              documentation, notes, or internal content
            - use_get_current_weather: questions asking about current
              weather in a specific location
            """
            return _classify_with_jev_impl.state()

        _jev_available = True
        logger.info("Jev classifier backend configured and available.")
    except Exception:
        logger.exception("Failed to set up Jev backend; will use local classifier only.")
        _jev_available = False
else:
    logger.info("TYPESAFE_API_KEY not set; using local LoRA classifier only.")


def _classify_with_jev(question: str):
    """Returns a label from Jev, or None if the call fails for any reason
    (network issue, API error, unexpected response) — callers should
    treat None as 'fall back to the local classifier', not as an error."""
    try:
        decision = _classify_with_jev_impl(question)
        logger.debug(f"jev classifier: question={question!r} -> label={decision.label}")
        return decision.label
    except Exception:
        logger.exception(f"Jev classification failed for question={question!r}, falling back")
        return None


# ─────────────────────────────────────────────────────────────
# Backend 2: Local LoRA fine-tune (Phase 4.5) — always loaded, used as
# the default when Jev isn't configured, and as the fallback if it fails.
# ─────────────────────────────────────────────────────────────

logger.info("Loading local tool-selection classifier...")
_base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
_local_model = PeftModel.from_pretrained(_base_model, ADAPTER_PATH)
_local_model.eval()
logger.info("Local tool-selection classifier loaded.")


def _classify_with_local_model(question: str) -> str:
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    inputs = _tokenizer(prompt, return_tensors="pt")
    outputs = _local_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    result = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    label = result[len(prompt):].strip()

    if label not in VALID_LABELS:
        logger.debug(f"local classifier produced unrecognized label={label!r}, defaulting to use_search_documents")
        return "use_search_documents"

    logger.debug(f"local classifier: question={question!r} -> label={label}")
    return label


# ─────────────────────────────────────────────────────────────
# Public entry point — this is what agent_service.py calls
# ─────────────────────────────────────────────────────────────

def classify(question: str) -> str:
    if _jev_available:
        label = _classify_with_jev(question)
        if label is not None:
            return label
        # Jev failed for this call — fall through to local model below

    return _classify_with_local_model(question)