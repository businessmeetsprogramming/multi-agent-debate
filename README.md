# Multi-Agent Debate Pipeline

A structured 5-agent debate system that uses **Gemini**, **GPT**, and **Claude** to deeply research any topic through adversarial analysis.

## How It Works

```
Phase 1: Research (parallel)         Phase 2: Debate (parallel)        Phase 3: Synthesis
┌─────────────────────┐              ┌──────────────────────┐          ┌──────────────────┐
│ Gemini 3 Pro      │              │ Gemini 3 Pro       │          │ Claude Opus 4.5  │
│ + Google Search      │──┐       ┌──│ FOR (85-95%)         │──┐       │                  │
│ (market data, stats) │  │       │  │ (high conviction)    │  │       │ Calibrated       │
└─────────────────────┘  │       │  └──────────────────────┘  ├──────▶│ Synthesis        │
                         ├───────┤                             │       │                  │
┌─────────────────────┐  │       │  ┌──────────────────────┐  │       │ + Cost Report    │
│ GPT-5.2 Pro         │  │       │  │ GPT-5.2 Pro          │  │       └──────────────────┘
│ (frameworks,        │──┘       └──│ AGAINST (15-35%)     │──┘
│  contrarian views)  │              │ (skeptical)          │
└─────────────────────┘              └──────────────────────┘
```

**Key design:** The FOR agent argues at high confidence (85-95%) while the AGAINST agent argues at low confidence (15-35%). This asymmetry creates productive tension that the synthesis agent calibrates between.

## Features

- **Multi-model**: Gemini 3 Pro (with Google Search grounding) + GPT-5.2 Pro + Claude
- **Token cost tracking**: Every API call logs input/output/thinking tokens and calculates USD cost
- **Parallel execution**: Research and debate phases run agents simultaneously
- **Claude Code skill**: Invoke with `/run-debate <your question>` in Claude Code

## Quick Start

### 1. Install dependencies

```bash
bash .claude/skills/run-debate/setup.sh
```

### 2. Set API keys

```bash
export GEMINI_API_KEY="your-gemini-key"   # https://aistudio.google.com/apikey
export OPENAI_API_KEY="your-openai-key"   # https://platform.openai.com/api-keys
```

### 3. Run a debate

In Claude Code (from this repo):
```
/run-debate Should the US invest heavily in nuclear energy to meet AI datacenter demand?
```

Or call the model wrapper directly:
```bash
python3 .claude/skills/run-debate/call_model.py gemini gemini-3-pro-preview prompt.txt output.md --search-grounding
python3 .claude/skills/run-debate/call_model.py openai gpt-5.2-pro prompt.txt output.md
```

## Output Structure

Each debate creates a numbered directory in `analyses/`:

```
analyses/01-topic-name/
├── prompt_research_gemini.txt     # Research prompt (Gemini)
├── prompt_research_gpt.txt        # Research prompt (GPT)
├── prompt_for.txt                 # FOR debate prompt
├── prompt_against.txt             # AGAINST debate prompt
├── 01_research_gemini.md          # Gemini research output
├── 01_research_gemini_cost.json   # Token usage & cost
├── 02_research_gpt.md             # GPT research output
├── 02_research_gpt_cost.json      # Token usage & cost
├── 03_debate_for.md               # FOR argument
├── 03_debate_for_cost.json        # Token usage & cost
├── 04_debate_against.md           # AGAINST argument
├── 04_debate_against_cost.json    # Token usage & cost
└── SYNTHESIS.md                   # Final calibrated synthesis + cost report
```

## Cost Tracking

Every API call produces a `_cost.json` sidecar file:

```json
{
  "provider": "gemini",
  "model": "gemini-2.5-pro",
  "input_tokens": 1250,
  "output_tokens": 4800,
  "thinking_tokens": 0,
  "total_tokens": 6050,
  "cost_usd": 0.0496,
  "latency_seconds": 12.34
}
```

The synthesis includes a total cost table across all 4 API calls.

## Supported Models

| Provider | Model | Input $/1M | Output $/1M |
|----------|-------|-----------|-------------|
| Gemini | gemini-3-pro-preview | $2.00 | $12.00 |
| Gemini | gemini-2.5-pro | $1.25 | $10.00 |
| Gemini | gemini-2.5-flash | $0.15 | $0.60 |
| OpenAI | gpt-5.2-pro | $1.75 | $14.00 |
| OpenAI | gpt-5.2 | $1.00 | $4.00 |
| OpenAI | gpt-4.1 | $2.00 | $8.00 |

## License

MIT
