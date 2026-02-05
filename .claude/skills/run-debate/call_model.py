#!/usr/bin/env python3
"""
call_model.py - Wrapper for calling Gemini and OpenAI APIs with token cost tracking.

Usage:
    python3 call_model.py <provider> <model> <input_file> <output_file> [--search-grounding] [--system <file>]

Examples:
    python3 call_model.py gemini gemini-2.5-pro prompt.txt output.md --search-grounding
    python3 call_model.py openai gpt-5.2-pro prompt.txt output.md
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

# Pricing per 1M tokens (USD) — update as models change
PRICING = {
    # Gemini (https://ai.google.dev/gemini-api/docs/pricing)
    "gemini-2.5-pro":   {"input": 1.25, "output": 10.00, "thinking": 3.50},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60,  "thinking": 0.70},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40,  "thinking": 0.00},
    # OpenAI (https://openai.com/api/pricing/)
    "gpt-5.2-pro":  {"input": 1.75, "output": 14.00, "thinking": 0.00},
    "gpt-5.2":      {"input": 1.00, "output": 4.00,  "thinking": 0.00},
    "gpt-5.1":      {"input": 0.80, "output": 3.20,  "thinking": 0.00},
    "gpt-5":        {"input": 0.60, "output": 2.40,  "thinking": 0.00},
    "gpt-4.1":      {"input": 2.00, "output": 8.00,  "thinking": 0.00},
    "gpt-4o":       {"input": 2.50, "output": 10.00, "thinking": 0.00},
}


def calc_cost(model: str, input_tokens: int, output_tokens: int, thinking_tokens: int = 0) -> float:
    """Calculate USD cost from token counts."""
    rates = PRICING.get(model, {"input": 0, "output": 0, "thinking": 0})
    cost = (
        input_tokens * rates["input"] / 1_000_000
        + output_tokens * rates["output"] / 1_000_000
        + thinking_tokens * rates.get("thinking", 0) / 1_000_000
    )
    return round(cost, 6)


def call_gemini(model: str, prompt: str, system_prompt: Optional[str] = None, search_grounding: bool = False) -> tuple:
    """Call Gemini API. Returns (text, usage_dict)."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    config_kwargs = {"temperature": 1.0, "max_output_tokens": 16384}
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt
    if search_grounding:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    config = types.GenerateContentConfig(**config_kwargs)

    start = time.time()
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    elapsed = round(time.time() - start, 2)

    # Extract usage
    u = response.usage_metadata
    input_tokens = getattr(u, "prompt_token_count", 0) or 0
    output_tokens = getattr(u, "candidates_token_count", 0) or 0
    thinking_tokens = getattr(u, "thoughts_token_count", 0) or 0
    total_tokens = getattr(u, "total_token_count", 0) or 0

    usage = {
        "provider": "gemini",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
        "cost_usd": calc_cost(model, input_tokens, output_tokens, thinking_tokens),
        "latency_seconds": elapsed,
    }

    return response.text or "", usage


def call_openai(model: str, prompt: str, system_prompt: Optional[str] = None) -> tuple:
    """Call OpenAI API. Returns (text, usage_dict)."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Get one at https://platform.openai.com/api-keys", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        max_completion_tokens=16384,
    )
    elapsed = round(time.time() - start, 2)

    # Extract usage
    u = response.usage
    input_tokens = getattr(u, "prompt_tokens", 0) or 0
    output_tokens = getattr(u, "completion_tokens", 0) or 0
    reasoning_tokens = 0
    if hasattr(u, "completion_tokens_details") and u.completion_tokens_details:
        reasoning_tokens = getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0
    total_tokens = getattr(u, "total_tokens", 0) or 0

    usage = {
        "provider": "openai",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "cost_usd": calc_cost(model, input_tokens, output_tokens, reasoning_tokens),
        "latency_seconds": elapsed,
    }

    return response.choices[0].message.content or "", usage


def main():
    parser = argparse.ArgumentParser(description="Call Gemini or OpenAI APIs with cost tracking")
    parser.add_argument("provider", choices=["gemini", "openai"], help="API provider")
    parser.add_argument("model", help="Model name (e.g., gemini-2.5-pro, gpt-5.2-pro)")
    parser.add_argument("input_file", help="Path to prompt file")
    parser.add_argument("output_file", help="Path to write response")
    parser.add_argument("--search-grounding", action="store_true", help="Enable Google Search grounding (Gemini only)")
    parser.add_argument("--system", help="Path to system prompt file")

    args = parser.parse_args()

    # Read prompt
    with open(args.input_file, "r") as f:
        prompt = f.read().strip()

    # Read system prompt if provided
    system_prompt = None
    if args.system:
        with open(args.system, "r") as f:
            system_prompt = f.read().strip()

    # Call provider
    if args.provider == "gemini":
        text, usage = call_gemini(args.model, prompt, system_prompt, args.search_grounding)
    else:
        text, usage = call_openai(args.model, prompt, system_prompt)

    # Write response
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(text)

    # Write cost report (JSON sidecar)
    cost_file = args.output_file.rsplit(".", 1)[0] + "_cost.json"
    with open(cost_file, "w") as f:
        json.dump(usage, f, indent=2)

    # Print summary
    cost_str = f"${usage['cost_usd']:.4f}" if usage["cost_usd"] > 0 else "$?.???? (model not in pricing table)"
    print(f"OK: {len(text)} chars → {args.output_file}")
    print(f"    Model:    {usage['model']}")
    print(f"    Tokens:   {usage['input_tokens']:,} in / {usage['output_tokens']:,} out / {usage['thinking_tokens']:,} thinking")
    print(f"    Cost:     {cost_str}")
    print(f"    Latency:  {usage['latency_seconds']}s")
    print(f"    Report:   {cost_file}")


if __name__ == "__main__":
    main()
