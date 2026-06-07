# Polymath — Multi-Agent Research & Briefing Studio

Given a research topic, Polymath produces a **sourced markdown research report** with
inline citations and an **auto-generated PowerPoint deck**. It coordinates specialized
agents (Planner, Search, Reader, Critic, Writer) through a LangGraph state machine, with
a Chroma vector store as shared working memory and tools exposed over MCP.

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full design and build phases.

## Status

- [x] **Week 1 — Single-agent baseline.** `scripts/week1_baseline.py`: one LLM + web
      search + page fetch (function-calling), producing a cited 1-page markdown summary.
- [ ] Week 2 — Structured claim extraction (Reader)
- [ ] Week 3 — Memory + Critic
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
