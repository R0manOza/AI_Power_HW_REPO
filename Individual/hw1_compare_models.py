"""
Homework 1 (Lab 1) — Model Comparison + Cost Logging

Implements the actions from `homework1/Lab-1/homework/hw1-individual.md`:
 - load GEMINI_API_KEY from .env (python-dotenv)
 - call at least two different Gemini models
 - print response text + token counts + latency
 - compute a paid-tier cost equivalent (reference pricing)
 - save a small JSON log for later inclusion in README
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

# Load the repo-root `.env` regardless of where you run the script from.
_HERE = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_DOTENV_PATH = os.path.join(_REPO_ROOT, ".env")
load_dotenv(dotenv_path=_DOTENV_PATH)

try:
    import google.genai as genai
except ImportError as e:
    raise SystemExit(
        "ERROR: google-genai package not installed. Install with:\n"
        "  pip install google-genai python-dotenv\n"
        f"\nOriginal import error: {e}"
    )


# Two different models (suggested by the course template)
MODEL_A = "gemini-3-flash-preview"
MODEL_B = "gemini-3.1-flash-lite-preview"

# Something that requires real thinking, not just trivia.
# Keep it short to avoid unnecessarily large token counts.
PROMPT = (
    "You are troubleshooting a beginner-friendly app for car maintenance.\n"
    "Ask for an engine-bay photo and then guide the user to locate and identify:\n"
    "1) the spark plug,\n"
    "2) the battery,\n"
    "3) the oil fill cap.\n\n"
    "Requirements:\n"
    "- Ask 2-3 clarifying questions first.\n"
    "- Then list the next steps as a short checklist.\n"
    "- Use safety disclaimers briefly (1-2 sentences).\n"
    "- Output must be in plain text."
)


# Paid-tier reference pricing (USD per 1M tokens) from the course guide.
# The guide lists gemini-3.1-flash; we reuse the same rates for -lite-preview.
PRICING_USD_PER_1M = {
    "gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
    "gemini-3.1-flash": {"input": 0.10, "output": 0.40},
    "gemini-3.1-flash-lite-preview": {"input": 0.10, "output": 0.40},
}


@dataclass
class CallResult:
    call_index: int
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cost_paid_tier_reference: float
    response_preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "cost_paid_tier_reference": self.cost_paid_tier_reference,
            "response_preview": self.response_preview,
        }


def compute_paid_tier_cost_reference(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING_USD_PER_1M.get(model)
    if not rates:
        # If unsupported, return 0 but keep the field present.
        return 0.0
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    return input_cost + output_cost


def call_model(client: Any, model: str, prompt: str, call_index: int) -> CallResult:
    print("\n" + "=" * 80)
    print(f"CALL {call_index}: Model = {model}")
    print("=" * 80)

    start_time = time.perf_counter()
    response = client.models.generate_content(model=model, contents=prompt)
    latency_ms = (time.perf_counter() - start_time) * 1000

    usage = response.usage_metadata
    input_tokens = int(usage.prompt_token_count)
    output_tokens = int(usage.candidates_token_count)
    total_tokens = int(usage.total_token_count)

    cost = compute_paid_tier_cost_reference(model, input_tokens, output_tokens)

    print("\nRESPONSE TEXT:")
    print("-" * 80)
    print(response.text)
    print("-" * 80)

    print("TOKEN USAGE:")
    print(f"  Input tokens:  {input_tokens}")
    print(f"  Output tokens: {output_tokens}")
    print(f"  Total tokens:  {total_tokens}")

    print("\nLATENCY:")
    print(f"  Latency (ms): {latency_ms:.0f}")

    print("\nCOST ESTIMATE (paid tier reference):")
    print(f"  ${cost:.6f}")

    preview = (response.text or "").replace("\r\n", "\n").strip()[:300]

    return CallResult(
        call_index=call_index,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        cost_paid_tier_reference=cost,
        response_preview=preview,
    )


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GEMINI_API_KEY not found. Check your .env file.")

    client = genai.Client(api_key=api_key)

    print("Running Homework 1 model comparison...")
    print(f"Prompt:\n{PROMPT}\n")

    results: list[CallResult] = []
    results.append(call_model(client, MODEL_A, PROMPT, 1))
    results.append(call_model(client, MODEL_B, PROMPT, 2))

    payload = {
        "timestamp": datetime.now().isoformat(),
        "prompt": PROMPT,
        "results": [r.to_dict() for r in results],
    }

    out_path = os.path.join(os.path.dirname(__file__), "results_hw1_models.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\nSaved results to:")
    print(f"  {out_path}")


if __name__ == "__main__":
    main()

