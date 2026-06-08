# Polymath — Multi-Agent Research & Briefing Studio

Given a research topic, Polymath produces a **sourced markdown research report** with
inline citations and an **auto-generated PowerPoint deck**. It coordinates specialized
agents (Planner, Search, Reader, Critic, Writer) through a LangGraph state machine, with
a Chroma vector store as shared working memory and tools exposed over MCP.

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full design and build phases.

## Status

- [x] **Week 1 — Single-agent baseline.** `scripts/week1_baseline.py`: one LLM + web
      search + page fetch (function-calling), producing a cited 1-page markdown summary.
- [x] **Week 2 — Structured claim extraction (Reader).** `agents/reader.py` extracts
      `Claim` JSON from page text, Pydantic-validates with ≤2 retries. Acceptance eval:
      100% first-try valid across 5 topics (`eval/run_eval.py`).
- [x] **Week 3 — Memory + Critic.** `memory/vector_store.py` (Chroma) stores every
      claim; `agents/critic.py` reads accumulated claims and decides continue/stop with
      gap-targeting subtasks; `scripts/week3_research.py` loops until stop or 3 iterations.
      Acceptance: Critic spotted the omitted aspect in 5/5 cases (`eval/run_critic_eval.py`).
- [ ] Week 4 — LangGraph orchestration
- [ ] Week 5 — Slide deck + MCP
- [ ] Week 6 — Streamlit UI + deploy

## Setup

```bash
uv sync                      # create .venv and install deps
cp .env.example .env         # then fill in OPENROUTER_API_KEY and TAVILY_API_KEY
```

## Week 1 usage

```bash
uv run python scripts/week1_baseline.py "current state of solid-state batteries"
```

Writes a cited markdown summary to `outputs/`.

## Week 2 usage

```bash
# Extract structured claims for one topic:
uv run python scripts/week2_reader.py "solid-state batteries" --pages 3

# Run the acceptance eval (validation rates across 5 topics):
uv run python eval/run_eval.py --topics 5 --pages 2
```

## Week 3 usage

```bash
# Iterative research with memory + Critic (loops until stop or 3 iterations):
uv run python scripts/week3_research.py "remote work" --max-iterations 3 --pages 2

# Run the Critic acceptance eval (omitted-aspect detection over 5 cases):
uv run python eval/run_critic_eval.py
```
