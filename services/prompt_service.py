from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text().strip()