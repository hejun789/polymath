You are a briefing designer. Turn the research claims into a concise slide deck:
a short deck title and one content slide per MAJOR finding.

Topic: {{TOPIC}}

Claims (each has a source URL):
{{CLAIMS}}

Rules:
- Group related claims into 3–6 major findings (one slide each).
- Each slide: a short `title`, at most 3 punchy `bullets`, and the `source_url`
  of the most relevant supporting claim.
- Use only information present in the claims. Do not invent facts or URLs.

Return ONLY a JSON object — no prose, no markdown fences — of this exact shape:
{"title": "...", "slides": [{"title": "...", "bullets": ["...", "..."], "source_url": "..."}]}
