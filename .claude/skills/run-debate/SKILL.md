---
name: run-debate
description: Run a multi-agent debate pipeline on a research question. Uses Gemini for web research with search grounding, GPT for counter-analysis, and Claude for synthesis. Use when asked to debate, analyze, or deeply research a topic.
argument-hint: [research question]
disable-model-invocation: false
allowed-tools: Bash(python3 *), Bash(mkdir *), Bash(chmod *), Read, Write, Glob
---

# Multi-Agent Debate Pipeline

Run a structured 5-agent debate on: **$ARGUMENTS**

## Setup Check

Before running, verify dependencies are installed:
```bash
python3 -c "import google.generativeai; import openai; print('OK: dependencies installed')" 2>/dev/null || pip install -r .claude/skills/run-debate/requirements.txt
```

Verify API keys are set:
```bash
python3 -c "import os; assert os.environ.get('GEMINI_API_KEY'), 'GEMINI_API_KEY not set'; assert os.environ.get('OPENAI_API_KEY'), 'OPENAI_API_KEY not set'; print('OK: API keys found')"
```

If keys are missing, stop and tell the user to set them.

## Directory Setup

Create the output directory. Find the next available number in `analyses/`:
```bash
next_num=$(ls analyses/ | grep -oE '^[0-9]+' | sort -n | tail -1 | awk '{printf "%02d", $1+1}')
```
Create: `analyses/${next_num}-<slugified-topic>/`

All agent outputs go into this directory. Use this path for all file writes below.

## Pipeline (5 Agents)

IMPORTANT: Run agents in parallel where indicated. The CALL_MODEL script is at:
`.claude/skills/run-debate/call_model.py`

### Phase 1: Research (2 agents in parallel)

Write two prompt files, then run both simultaneously:

**Agent 1 — Gemini Research (with search grounding):**
Write a prompt to `{output_dir}/prompt_research_gemini.txt` that asks for comprehensive web research on the topic. Include:
- Current market data, statistics, and trends
- Key players and recent developments
- Quantitative data points with sources

```bash
python3 .claude/skills/run-debate/call_model.py gemini gemini-2.5-pro {output_dir}/prompt_research_gemini.txt {output_dir}/01_research_gemini.md --search-grounding
```

**Agent 2 — GPT Research:**
Write a prompt to `{output_dir}/prompt_research_gpt.txt` that asks for analytical research from a different angle. Include:
- Historical context and precedents
- Structural analysis and frameworks
- Contrarian perspectives and edge cases

```bash
python3 .claude/skills/run-debate/call_model.py openai gpt-5.2-pro {output_dir}/prompt_research_gpt.txt {output_dir}/02_research_gpt.md
```

Run both research agents in PARALLEL (two Bash calls in one message).

### Phase 2: Debate (2 agents in parallel)

After research completes, read both research outputs. Then write two debate prompts that INCLUDE the research findings.

**Agent 3 — FOR (Gemini, high confidence):**
Write a prompt to `{output_dir}/prompt_for.txt` that:
- Includes a summary of both research outputs as context
- Assigns the role: "You are arguing FOR this proposition with HIGH CONFIDENCE (85-95%)"
- Requires: thesis statement, 5+ supporting arguments with evidence, quantitative projections, timeline, acknowledgment of 2-3 weaknesses

```bash
python3 .claude/skills/run-debate/call_model.py gemini gemini-2.5-pro {output_dir}/prompt_for.txt {output_dir}/03_debate_for.md
```

**Agent 4 — AGAINST (GPT, low confidence):**
Write a prompt to `{output_dir}/prompt_against.txt` that:
- Includes a summary of both research outputs as context
- Assigns the role: "You are arguing AGAINST with LOW CONFIDENCE in the proposition (15-35%)"
- Requires: counter-thesis, 5+ counterarguments with evidence, risk analysis, failure modes, acknowledgment of 2-3 strengths of the FOR position

```bash
python3 .claude/skills/run-debate/call_model.py openai gpt-5.2-pro {output_dir}/prompt_against.txt {output_dir}/04_debate_against.md
```

Run both debate agents in PARALLEL (two Bash calls in one message).

### Phase 3: Synthesis (Claude — you do this yourself)

Read all four outputs:
- `01_research_gemini.md`
- `02_research_gpt.md`
- `03_debate_for.md`
- `04_debate_against.md`

Also read all four cost reports:
- `01_research_gemini_cost.json`
- `02_research_gpt_cost.json`
- `03_debate_for_cost.json`
- `04_debate_against_cost.json`

Then write `{output_dir}/SYNTHESIS.md` with this structure:

```markdown
# Synthesis: [Topic]

**Date:** [today's date]
**Models used:** Gemini 2.5 Pro (research + FOR), GPT-5.2 Pro (research + AGAINST), Claude (synthesis)

## Executive Summary
[2-3 paragraph calibrated summary]

## Key Findings
[Bulleted list of the most important facts and data points from research]

## FOR Position Summary
[Summary of the strongest FOR arguments, with confidence assessment]

## AGAINST Position Summary
[Summary of the strongest AGAINST arguments, with confidence assessment]

## Calibrated Assessment
[Your synthesis that weighs both sides. Include:]
- Overall confidence level (0-100%) with justification
- Where the FOR side is strongest
- Where the AGAINST side is strongest
- Key uncertainties that could swing the conclusion
- What additional information would change the assessment

## Actionable Recommendations
[Numbered list of specific, actionable next steps]

## Sources and Data Quality
[Assessment of the quality and recency of data used]

## Cost Report
[Include a table summarizing token usage and costs for each agent call]

| Agent | Model | Input Tokens | Output Tokens | Thinking Tokens | Cost (USD) | Latency |
|-------|-------|-------------|---------------|-----------------|------------|---------|
| Research (Gemini) | ... | ... | ... | ... | ... | ... |
| Research (GPT) | ... | ... | ... | ... | ... | ... |
| Debate FOR (Gemini) | ... | ... | ... | ... | ... | ... |
| Debate AGAINST (GPT) | ... | ... | ... | ... | ... | ... |
| **Total** | | | | | **$X.XXXX** | |

*Note: Claude synthesis cost is not tracked here as it runs within the Claude Code session.*
```

## Final Output

After writing SYNTHESIS.md, print a summary to the user:
- The output directory path
- The 5 files created
- The overall confidence level from the synthesis
- Top 3 actionable recommendations
- **Total token cost across all 4 API calls**
