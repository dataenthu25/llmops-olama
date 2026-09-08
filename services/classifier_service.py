"""
Loads the fine-tuned LoRA tool-selection classifier once at import time
and exposes a single classify() function.

This model was fine-tuned (see /finetune_data/generate_dataset.py and
the Colab training notebook) specifically to fix the Phase 4 known
limitation: qwen2.5-coder:7b consistently over-triggers search_documents
on simple math/reasoning questions, and prompting alone couldn't fix it.

This is a narrow, fast pre-router — not a replacement for the main agent.
It only answers one question: "does this need a tool, and if so, which
one?" The actual tool execution and answer synthesis still happens in
agent_service.py.
"""

import logging

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./models/lora-tool-selector-final"

VALID_LABELS = {"no_tool_needed", "use_search_documents", "use_get_current_weather"}

# Loaded once at import time — reused across all requests, same pattern
# as the main chat model in agent_service.py.
logger.info("Loading tool-selection classifier...")
_base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
_classifier_model = PeftModel.from_pretrained(_base_model, ADAPTER_PATH)
_classifier_model.eval()
logger.info("Tool-selection classifier loaded.")


def classify(question: str) -> str:
    """Returns one of: 'no_tool_needed', 'use_search_documents',
    'use_get_current_weather'. Falls back to a safe default if the
    model produces something unexpected, since this is a narrow
    fine-tune and shouldn't be trusted blindly on malformed output."""
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    inputs = _tokenizer(prompt, return_tensors="pt")
    outputs = _classifier_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    result = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    label = result[len(prompt):].strip()

    if label not in VALID_LABELS:
        logger.debug(f"classifier produced unrecognized label={label!r}, falling back to full agent")
        return "use_search_documents"  # safe fallback: let the full agent decide

    logger.debug(f"classifier: question={question!r} -> label={label}")
    return label