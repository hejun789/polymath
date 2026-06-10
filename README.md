---
title: Polymath
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# 🧠 Polymath — Multi-Agent Research & Briefing Studio

Give Polymath a research topic; it returns a **sourced markdown report** with inline
citations and an **auto-generated PowerPoint deck**. Specialized agents (Planner,
Search, Reader, Critic, Writer) are coordinated by a **LangGraph** state machine, share
a **Chroma** vector store as working memory, and call their tools over **MCP**.

**▶️ Live demo:** https://huggingface.co/spaces/hejun123/polymath

<img width="1919" height="911" alt="Screenshot 2026-06-09 111553" src="https://github.com/user-attachments/assets/7ef9b245-de51-449c-8274-0a1780c76e10" />
![Polymath app](docs/screenshot.png)
<img width="1919" height="911" alt="Screenshot 2026-06-09 111442" src="https://github.com/user-attachments/assets/07fa7913-d205-4a06-8e2f-f273cfc3907f" />
<img width="1919" height="910" alt="Screenshot 2026-06-09 111458" src="https://github.com/user-attachments/assets/4fe57ca1-3c69-4f04-af0c-f8e2ee18a6d2" />
<img width="1915" height="909" alt="Screenshot 2026-06-09 111247" src="https://github.com/user-attachments/assets/84def3d0-2193-40fb-8fe8-c0e82b5a0226" />
<img width="1917" height="855" alt="Screenshot 2026-06-09 120843" src="https://github.com/user-attachments/assets/24bae07b-443c-4e11-9347-b738f1f0ced7" />
<img width="1269" height="918" alt="Screenshot 2026-06-09 120929" src="https://github.com/user-attachments/assets/24875f3b-2f3b-46b6-a927-2f0b905c00cf" />


> 🎥 _Demo video: (add link after recording)_

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

This repo doubles as a **Docker** Space (config is the YAML frontmatter at the top of
this file; the build uses the `Dockerfile`, which runs the Streamlit app). To deploy:

1. Create a new Space on Hugging Face → SDK **Docker** → **Streamlit** template →
   **CPU Basic (Free)**.
2. Push this repo to the Space remote (`git push hf main --force`).
3. In **Settings → Variables and secrets**, add `OPENROUTER_API_KEY` and
   `TAVILY_API_KEY` as **secrets**.
4. The Space builds from the `Dockerfile` (installs `requirements.txt`). The first run
   downloads the ~80 MB ONNX embedding model once (cached afterward).

The app reads keys from environment variables (via `config.py`), so the secrets are
picked up automatically.

## Earlier-phase scripts (still runnable)

```bash
uv run python scripts/week1_baseline.py "..."                       # Week 1
uv run python scripts/week2_reader.py "..." --pages 3               # Week 2
uv run python scripts/week3_research.py "..." --max-iterations 3    # Week 3
```

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full design.
