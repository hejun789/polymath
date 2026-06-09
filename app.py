"""Polymath — Streamlit UI.

Enter a topic, watch the agent graph run live (planner -> search -> reader ->
critic -> writer), then download the cited markdown report and the .pptx deck.

Run locally:
    uv run streamlit run app.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from polymath.config import settings  # noqa: E402
from polymath.graph.state import GraphState  # noqa: E402
from polymath.graph.workflow import make_direct_workflow  # noqa: E402
from polymath.outputs.slides import deck_to_bytes  # noqa: E402

NODE_LABELS = {
    "planner": "🧭 Planner — decomposing the topic",
    "search": "🔎 Search — finding sources",
    "reader": "📖 Reader — extracting claims",
    "critic": "🧐 Critic — checking for gaps",
    "writer": "✍️ Writer — synthesizing the report",
}

st.set_page_config(page_title="Polymath", page_icon="🧠", layout="centered")
st.title("🧠 Polymath")
st.caption("Multi-agent research studio — topic in, sourced report + slide deck out.")


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "topic"


async def _run(topic: str, max_iterations: int, status) -> dict:
    """Stream the graph, updating the status panel per node; build the deck."""
    graph, writer = make_direct_workflow()
    seen = 0
    final: dict = {}
    async for state in graph.astream(
        GraphState(topic=topic, max_iterations=max_iterations), stream_mode="values"
    ):
        final = state
        trace = state.get("trace", [])
        for entry in trace[seen:]:
            node = entry.get("node", "")
            status.write(NODE_LABELS.get(node, node))
        seen = len(trace)

    status.write("🎞️ Building slide deck…")
    deck = await writer.build_deck(topic=topic, claims=final["claims"])
    return {"report": final["report"], "claims": final["claims"], "deck": deck}


def main() -> None:
    if not settings.openrouter_api_key or not settings.tavily_api_key:
        st.error(
            "Missing API keys. Set OPENROUTER_API_KEY and TAVILY_API_KEY "
            "(Space secrets, or a local .env)."
        )
        return

    topic = st.text_input("Research topic", placeholder="e.g. current state of solid-state batteries")
    max_iterations = st.slider("Max research iterations", 1, 3, 2)

    if st.button("Run research", type="primary", disabled=not topic):
        try:
            with st.status("Running the research pipeline…", expanded=True) as status:
                result = asyncio.run(_run(topic, max_iterations, status))
                status.update(label="Done ✅", state="complete")
        except Exception as exc:  # noqa: BLE001 - surface any runtime error to the user
            st.session_state.pop("result", None)
            st.error(f"Run failed: {exc}")
            return
        # Persist so the result survives the rerun a download_button click causes.
        st.session_state["result"] = result
        st.session_state["topic"] = topic

    # Render persisted results (stays put across download-button reruns).
    result = st.session_state.get("result")
    if result:
        stem = _slug(st.session_state.get("topic", "topic"))
        st.success(f"Gathered {len(result['claims'])} claims · {len(result['deck'].slides)} slides.")

        col1, col2 = st.columns(2)
        col1.download_button(
            "⬇️ Download report (.md)",
            data=result["report"],
            file_name=f"{stem}.md",
            mime="text/markdown",
        )
        col2.download_button(
            "⬇️ Download deck (.pptx)",
            data=deck_to_bytes(result["deck"]),
            file_name=f"{stem}.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )

        st.markdown("---")
        # Escape '$' so Streamlit doesn't render dollar amounts as LaTeX math.
        st.markdown(result["report"].replace("$", "\\$"))


# `streamlit run app.py` executes this file with __name__ == "__main__".
if __name__ == "__main__":
    main()
