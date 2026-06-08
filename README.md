---
title: Polymath
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
---

# 🧠 Polymath — Multi-Agent Research & Briefing Studio

Give Polymath a research topic; it returns a **sourced markdown report** with inline
citations and an **auto-generated PowerPoint deck**. Specialized agents (Planner,
Search, Reader, Critic, Writer) are coordinated by a **LangGraph** state machine, share
a **Chroma** vector store as working memory, and call their tools over **MCP**.

> 🎥 _Demo video: (add link after recording)_
> 🖼️ _Screenshot: `docs/screenshot.png` (add after first deploy)_

## Architecture

```
        topic
          │
          ▼
     ┌─────────┐   decomposes topic → subtasks
     │ Planner │
     └────┬────┘
          ▼
     ┌─────────┐   web_search (via MCP) → URLs
     │ Search  │
     └────┬────┘
          ▼
     ┌─────────┐   page_fetch (via MCP) → trafilatura → Reader extracts Claims
     │ Reader  │ ──────────────► Chroma vector store (working memory)
     └────┬────┘                          │
          ▼                               │ all claims so far
     ┌─────────┐  ◄───────────────────────┘
     │ Critic  │   gaps/contradictions?  continue (new subtasks) ──┐ loop ≤ 3×
     └────┬────┘                          stop                      │
          │  ────────────────────────────────────────── back to Search
          ▼
     ┌─────────┐   synthesizes from Claims
     │ Writer  │
     └────┬────┘
     ┌────┴───────────┐
     ▼                ▼
  report.md       deck.pptx
```

Tools (`web_search`, `page_fetch`, `claim_extract`) are exposed by a local **MCP server**
(`mcp_server/server.py`) and called through an MCP client over stdio.

## Status — all six phases complete

| Phase | What | Acceptance |
|---|---|---|
| 1 | Single-agent baseline (LLM + tools → cited summary) | ≥5 cited claims, no hallucinated URLs |
| 2 | Structured claim extraction (Reader + Pydantic, retry) | 100% first-try valid / 5 topics |
| 3 | Chroma memory + Critic continue/stop loop | omitted aspect found 5/5 cases |
| 4 | LangGraph orchestration (Planner…Writer, conditional edges) | runs end-to-end w/ per-node trace |
| 5 | PPTX deck + tools over MCP | workflow over MCP, valid PPTX |
| 6 | Streamlit UI + deploy | enter topic → download both artifacts |

## Tech stack

Python 3.11+ · `uv` · LangGraph · OpenRouter (free-tier models, routed in
`models/router.py`) · Pydantic v2 · Chroma (+ ONNX all-MiniLM-L6-v2) · Tavily ·
trafilatura · MCP · python-pptx · Streamlit · pytest.

## Quickstart

```bash
uv sync                      # create .venv and install deps
cp .env.example .env         # fill in OPENROUTER_API_KEY and TAVILY_API_KEY

# Web app (recommended):
uv run streamlit run app.py

# Or the full pipeline from the CLI (search+fetch over MCP) → .md + .pptx in outputs/:
uv run python -m polymath.graph.workflow --topic "current state of solid-state batteries"
```

Get free keys at [openrouter.ai](https://openrouter.ai) and [tavily.com](https://tavily.com).

## Tests

```bash
uv run pytest          # full suite (offline; no API keys needed)
```

Acceptance evals (these make live API calls):

```bash
uv run python eval/run_eval.py --topics 5 --pages 2   # Week 2: extraction validity
uv run python eval/run_critic_eval.py                 # Week 3: Critic gap detection
```

## Deploy to Hugging Face Spaces

This repo doubles as a Streamlit Space (config is the YAML frontmatter at the top of
this file). To deploy:

1. Create a new **Streamlit** Space on Hugging Face.
2. Push this repo to the Space remote (`git push hf main`).
3. In **Settings → Variables and secrets**, add `OPENROUTER_API_KEY` and
   `TAVILY_API_KEY` as **secrets**.
4. The Space installs from `requirements.txt`. The first run downloads the ~80 MB ONNX
   embedding model once (cached afterward).

The app reads keys from environment variables (via `config.py`), so the secrets are
picked up automatically.

## Earlier-phase scripts (still runnable)

```bash
uv run python scripts/week1_baseline.py "..."                       # Week 1
uv run python scripts/week2_reader.py "..." --pages 3               # Week 2
uv run python scripts/week3_research.py "..." --max-iterations 3    # Week 3
```

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full design.
