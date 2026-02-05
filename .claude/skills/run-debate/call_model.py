#!/usr/bin/env python3
"""
call_model.py - Wrapper for calling Gemini and OpenAI APIs with token cost tracking.

Usage:
    python3 call_model.py <provider> <model> <input_file> <output_file> [--search-grounding] [--search-rounds N] [--system <file>]

Examples:
    python3 call_model.py gemini gemini-3-pro-preview prompt.txt output.md --search-grounding
    python3 call_model.py gemini gemini-3-pro-preview prompt.txt output.md --search-grounding --search-rounds 5
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
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00, "thinking": 3.50},
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


def _single_gemini_call(client, model, prompt, config):
    """Make a single Gemini API call. Returns (text, usage_metadata, grounding_metadata, elapsed)."""
    start = time.time()
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    elapsed = round(time.time() - start, 2)

    u = response.usage_metadata
    gm = None
    if response.candidates:
        gm = response.candidates[0].grounding_metadata

    return response.text or "", u, gm, elapsed


def call_gemini(model: str, prompt: str, system_prompt: Optional[str] = None,
                search_grounding: bool = False, search_rounds: int = 1) -> tuple:
    """Call Gemini API with optional multi-round search. Returns (text, usage_dict)."""
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

    # Aggregate across rounds
    all_text_parts = []
    total_input = 0
    total_output = 0
    total_thinking = 0
    total_all = 0
    total_elapsed = 0.0
    all_search_queries = []
    all_web_sources = []
    seen_uris = set()

    for round_num in range(1, search_rounds + 1):
        if round_num == 1:
            round_prompt = prompt
        else:
            # Build a follow-up prompt that references previous findings
            previous_queries = ", ".join(f'"{q}"' for q in all_search_queries) if all_search_queries else "none yet"
            previous_sources = ", ".join(s["title"] for s in all_web_sources[:20]) if all_web_sources else "none yet"
            round_prompt = (
                f"{prompt}\n\n"
                f"---\n"
                f"IMPORTANT: This is search round {round_num} of {search_rounds}. "
                f"Previous rounds already searched for: {previous_queries}\n"
                f"Previous sources already found: {previous_sources}\n\n"
                f"You MUST search for NEW and DIFFERENT information that was NOT covered above. "
                f"Use different search queries. Look for:\n"
                f"- Alternative data sources and perspectives\n"
                f"- More recent or niche information\n"
                f"- Contradictory evidence or minority viewpoints\n"
                f"- Deeper quantitative data, statistics, or primary sources\n"
                f"- Regional, industry-specific, or expert analysis not yet covered\n"
                f"DO NOT repeat information already found. Focus entirely on NEW findings."
            )

        if search_rounds > 1:
            print(f"  [Round {round_num}/{search_rounds}] Searching...", flush=True)

        text, u, gm, elapsed = _single_gemini_call(client, model, round_prompt, config)

        # Accumulate usage
        total_input += getattr(u, "prompt_token_count", 0) or 0
        total_output += getattr(u, "candidates_token_count", 0) or 0
        total_thinking += getattr(u, "thoughts_token_count", 0) or 0
        total_all += getattr(u, "total_token_count", 0) or 0
        total_elapsed += elapsed

        # Accumulate text
        if round_num == 1:
            all_text_parts.append(text)
        else:
            all_text_parts.append(f"\n\n---\n## Additional Research (Round {round_num})\n\n{text}")

        # Accumulate search grounding
        if gm:
            if hasattr(gm, "web_search_queries") and gm.web_search_queries:
                all_search_queries.extend(gm.web_search_queries)
            if hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
                for chunk in gm.grounding_chunks:
                    if hasattr(chunk, "web") and chunk.web:
                        uri = getattr(chunk.web, "uri", "")
                        if uri not in seen_uris:
                            seen_uris.add(uri)
                            all_web_sources.append({
                                "title": getattr(chunk.web, "title", ""),
                                "uri": uri,
                            })

        if search_rounds > 1:
            print(f"  [Round {round_num}/{search_rounds}] +{len(text)} chars, "
                  f"+{len(gm.web_search_queries) if gm and hasattr(gm, 'web_search_queries') and gm.web_search_queries else 0} searches, "
                  f"running total: {len(all_search_queries)} searches / {len(all_web_sources)} sources",
                  flush=True)

    combined_text = "".join(all_text_parts)

    usage = {
        "provider": "gemini",
        "model": model,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "thinking_tokens": total_thinking,
        "total_tokens": total_all,
        "cost_usd": calc_cost(model, total_input, total_output, total_thinking),
        "latency_seconds": round(total_elapsed, 2),
        "search_rounds": search_rounds,
        "search_queries": all_search_queries,
        "web_sources": all_web_sources,
        "num_searches": len(all_search_queries),
        "num_webpages": len(all_web_sources),
    }

    return combined_text, usage


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
        "search_rounds": 0,
        "search_queries": [],
        "web_sources": [],
        "num_searches": 0,
        "num_webpages": 0,
    }

    return response.choices[0].message.content or "", usage


def main():
    parser = argparse.ArgumentParser(description="Call Gemini or OpenAI APIs with cost tracking")
    parser.add_argument("provider", choices=["gemini", "openai"], help="API provider")
    parser.add_argument("model", help="Model name (e.g., gemini-3-pro-preview, gpt-5.2-pro)")
    parser.add_argument("input_file", help="Path to prompt file")
    parser.add_argument("output_file", help="Path to write response")
    parser.add_argument("--search-grounding", action="store_true", help="Enable Google Search grounding (Gemini only)")
    parser.add_argument("--search-rounds", type=int, default=1,
                        help="Number of search rounds (each round searches for NEW info). Default: 1. Use 3-5 for deep research.")
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
        text, usage = call_gemini(args.model, prompt, system_prompt, args.search_grounding, args.search_rounds)
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
    if usage.get("num_searches", 0) > 0 or usage.get("num_webpages", 0) > 0:
        rounds_str = f" across {usage['search_rounds']} rounds" if usage.get("search_rounds", 1) > 1 else ""
        print(f"    Searches: {usage['num_searches']} queries / {usage['num_webpages']} webpages cited{rounds_str}")
    print(f"    Report:   {cost_file}")


if __name__ == "__main__":
    main()
