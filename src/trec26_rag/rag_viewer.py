from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from .generator import AnswerGenerationRequest
from .rag_output import RagResponse


def build_rag_viewer_data(
    answer_requests: list[AnswerGenerationRequest],
    responses: list[RagResponse],
    validation_report: dict[str, Any],
    citation_diagnostics: dict[str, Any],
    proxy_metrics: dict[str, Any],
) -> dict[str, Any]:
    request_by_topic = {request.topic.id: request for request in answer_requests}
    diagnostics_by_topic = citation_diagnostics.get("per_topic", {})
    rows: list[dict[str, Any]] = []
    for response in responses:
        request = request_by_topic.get(response.topic.id)
        evidence = request.evidence if request else []
        rows.append(
            {
                "topic_id": response.topic.id,
                "title": response.topic.title,
                "narrative": response.topic.narrative,
                "answer": [
                    {"text": sentence.text, "citations": sentence.citations}
                    for sentence in response.answer
                ],
                "answer_text": " ".join(sentence.text for sentence in response.answer),
                "references": response.references,
                "evidence": [
                    {
                        "docid": document.docid,
                        "text": document.text,
                    }
                    for document in evidence
                ],
                "diagnostics": diagnostics_by_topic.get(response.topic.id, {}),
            }
        )
    return {
        "summary": {
            "valid": validation_report.get("valid", False),
            "metrics": validation_report.get("metrics", {}),
            "proxy_metrics": proxy_metrics,
            "citation_summary": citation_diagnostics.get("summary", {}),
            "errors": validation_report.get("errors", []),
            "warnings": validation_report.get("warnings", []),
        },
        "topics": rows,
    }


def write_rag_viewer_html(
    path: str | Path,
    answer_requests: list[AnswerGenerationRequest],
    responses: list[RagResponse],
    validation_report: dict[str, Any],
    citation_diagnostics: dict[str, Any],
    proxy_metrics: dict[str, Any],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_rag_viewer_data(
        answer_requests=answer_requests,
        responses=responses,
        validation_report=validation_report,
        citation_diagnostics=citation_diagnostics,
        proxy_metrics=proxy_metrics,
    )
    output_path.write_text(render_rag_viewer_html(data), encoding="utf-8")
    return output_path


def render_rag_viewer_html(data: dict[str, Any]) -> str:
    json_payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    title = "TREC RAG Run Viewer"
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --ink: #18211f;
      --muted: #65736f;
      --paper: #fbf7ec;
      --card: rgba(255, 255, 255, 0.82);
      --line: rgba(24, 33, 31, 0.14);
      --accent: #c65d2e;
      --accent-strong: #8f351f;
      --good: #2f7d4b;
      --warn: #a76812;
      --bad: #a63232;
      --shadow: 0 24px 70px rgba(45, 34, 19, 0.16);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(198, 93, 46, 0.20), transparent 32rem),
        radial-gradient(circle at bottom right, rgba(47, 125, 75, 0.18), transparent 30rem),
        linear-gradient(135deg, #fcf3df 0%, #edf3ea 100%);
      min-height: 100vh;
    }
    header {
      padding: 34px clamp(20px, 4vw, 56px) 18px;
    }
    h1 {
      margin: 0 0 8px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2.1rem, 5vw, 4.8rem);
      line-height: 0.95;
      letter-spacing: -0.055em;
    }
    .subtitle {
      color: var(--muted);
      max-width: 820px;
      font-size: 1.03rem;
      line-height: 1.5;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      padding: 0 clamp(20px, 4vw, 56px) 20px;
    }
    .metric {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 15px 16px;
      box-shadow: 0 10px 28px rgba(45, 34, 19, 0.08);
      backdrop-filter: blur(14px);
    }
    .metric .label {
      color: var(--muted);
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.10em;
    }
    .metric .value {
      margin-top: 7px;
      font-size: 1.5rem;
      font-weight: 760;
    }
    main {
      display: grid;
      grid-template-columns: minmax(260px, 360px) minmax(0, 1fr);
      gap: 18px;
      padding: 0 clamp(20px, 4vw, 56px) 40px;
    }
    aside, section.panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
      overflow: hidden;
    }
    .search {
      width: calc(100% - 28px);
      margin: 14px;
      padding: 12px 13px;
      border: 1px solid var(--line);
      border-radius: 16px;
      color: var(--ink);
      background: rgba(255,255,255,0.78);
      font: inherit;
    }
    .topic-list {
      max-height: 72vh;
      overflow: auto;
      padding: 0 10px 12px;
    }
    .topic-button {
      display: block;
      width: 100%;
      border: 0;
      border-radius: 18px;
      margin: 6px 0;
      padding: 13px 14px;
      text-align: left;
      color: var(--ink);
      background: transparent;
      cursor: pointer;
      font: inherit;
    }
    .topic-button:hover, .topic-button.active {
      background: rgba(198, 93, 46, 0.13);
    }
    .topic-button strong {
      display: block;
      font-size: 0.95rem;
      margin-bottom: 4px;
    }
    .topic-button span {
      color: var(--muted);
      font-size: 0.82rem;
    }
    .content {
      padding: clamp(18px, 3vw, 34px);
    }
    .eyebrow {
      color: var(--accent-strong);
      font-size: 0.78rem;
      font-weight: 760;
      letter-spacing: 0.11em;
      text-transform: uppercase;
    }
    h2 {
      margin: 8px 0 10px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(1.7rem, 3.2vw, 3rem);
      letter-spacing: -0.035em;
    }
    .narrative {
      color: var(--muted);
      line-height: 1.58;
      margin-bottom: 22px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(260px, 0.85fr);
      gap: 16px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 16px;
      background: rgba(255,255,255,0.58);
    }
    .card h3 {
      margin: 0 0 12px;
      font-size: 0.86rem;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      color: var(--muted);
    }
    .sentence {
      padding: 13px 0;
      border-top: 1px solid var(--line);
      line-height: 1.55;
    }
    .sentence:first-of-type { border-top: 0; }
    .cite {
      display: inline-block;
      margin-left: 6px;
      padding: 2px 7px;
      border-radius: 999px;
      background: rgba(47, 125, 75, 0.12);
      color: var(--good);
      font-size: 0.78rem;
      font-weight: 700;
    }
    .doc {
      padding: 12px 0;
      border-top: 1px solid var(--line);
    }
    .doc:first-of-type { border-top: 0; }
    .doc-id {
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.82rem;
      color: var(--accent-strong);
      word-break: break-all;
    }
    .doc-text {
      color: var(--muted);
      margin-top: 7px;
      line-height: 1.45;
      max-height: 8.7rem;
      overflow: auto;
    }
    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .pill {
      border-radius: 999px;
      padding: 7px 10px;
      background: rgba(24,33,31,0.06);
      font-size: 0.84rem;
    }
    .empty {
      color: var(--muted);
      font-style: italic;
    }
    @media (max-width: 920px) {
      main, .grid { grid-template-columns: 1fr; }
      .topic-list { max-height: 260px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>RAG Run Viewer</h1>
    <p class="subtitle">Per-topic inspection for generated answers, citations, evidence, and validation diagnostics. This file is a self-contained snapshot of one experiment run.</p>
  </header>
  <div class="metrics" id="metrics"></div>
  <main>
    <aside>
      <input class="search" id="search" placeholder="Filter topics...">
      <div class="topic-list" id="topic-list"></div>
    </aside>
    <section class="panel">
      <div class="content" id="content"></div>
    </section>
  </main>
  <script id="rag-viewer-data" type="application/json">__JSON_PAYLOAD__</script>
  <script>
    const DATA = JSON.parse(document.getElementById('rag-viewer-data').textContent);
    const topics = DATA.topics || [];
    let selectedTopicId = topics[0]?.topic_id || null;

    const fmt = (value) => {{
      if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3);
      if (value === undefined || value === null || value === '') return 'n/a';
      return String(value);
    }};
    const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({{
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    }}[char]));
    const truncate = (value, max = 160) => {{
      const text = String(value || '');
      return text.length > max ? `${{text.slice(0, max)}}...` : text;
    }};

    function renderMetrics() {{
      const summary = DATA.summary || {{}};
      const metrics = summary.metrics || {{}};
      const proxy = summary.proxy_metrics || {{}};
      const cards = [
        ['Valid output', summary.valid ? 'yes' : 'no'],
        ['Topics', metrics.requested_topics ?? metrics.rag_topic_count],
        ['Citation coverage', proxy.rag_proxy_citation_coverage_mean],
        ['Citation density', proxy.rag_proxy_citation_density_mean],
        ['Validation errors', metrics.rag_validation_error_count],
        ['Response rate', proxy.rag_proxy_response_rate],
      ];
      document.getElementById('metrics').innerHTML = cards.map(([label, value]) => `
        <div class="metric"><div class="label">${{escapeHtml(label)}}</div><div class="value">${{escapeHtml(fmt(value))}}</div></div>
      `).join('');
    }}

    function renderTopicList() {{
      const query = document.getElementById('search').value.toLowerCase();
      const filtered = topics.filter((topic) =>
        `${{topic.topic_id}} ${{topic.title}} ${{topic.narrative}}`.toLowerCase().includes(query)
      );
      document.getElementById('topic-list').innerHTML = filtered.map((topic) => `
        <button class="topic-button ${{topic.topic_id === selectedTopicId ? 'active' : ''}}" data-topic-id="${{escapeHtml(topic.topic_id)}}">
          <strong>${{escapeHtml(topic.title || topic.topic_id)}}</strong>
          <span>${{escapeHtml(topic.topic_id)}} · ${{escapeHtml(truncate(topic.answer_text, 96))}}</span>
        </button>
      `).join('') || '<p class="empty" style="padding: 0 14px;">No matching topics.</p>';
      document.querySelectorAll('.topic-button').forEach((button) => {{
        button.addEventListener('click', () => {{
          selectedTopicId = button.dataset.topicId;
          renderTopicList();
          renderContent();
        }});
      }});
    }}

    function renderContent() {{
      const topic = topics.find((item) => item.topic_id === selectedTopicId);
      if (!topic) {{
        document.getElementById('content').innerHTML = '<p class="empty">No topic selected.</p>';
        return;
      }}
      const diagnostics = topic.diagnostics || {{}};
      const validator = diagnostics.validator || {{}};
      const references = topic.references || [];
      const evidenceByDocId = Object.fromEntries((topic.evidence || []).map((doc) => [doc.docid, doc]));
      const answerHtml = (topic.answer || []).map((sentence, index) => `
        <div class="sentence">
          <strong>${{index + 1}}.</strong> ${{escapeHtml(sentence.text)}}
          ${(sentence.citations || []).map((citation) => `<span class="cite">ref ${{citation}}</span>`).join('')}
        </div>
      `).join('') || '<p class="empty">No answer sentences.</p>';
      const referenceHtml = references.map((docid, index) => {{
        const evidence = evidenceByDocId[docid];
        return `
          <div class="doc">
            <div class="doc-id">[${{index}}] ${{escapeHtml(docid)}}</div>
            <div class="doc-text">${{escapeHtml(evidence?.text || 'No evidence text captured for this reference.')}}</div>
          </div>
        `;
      }}).join('') || '<p class="empty">No references.</p>';
      document.getElementById('content').innerHTML = `
        <div class="eyebrow">${{escapeHtml(topic.topic_id)}}</div>
        <h2>${{escapeHtml(topic.title)}}</h2>
        <p class="narrative">${{escapeHtml(topic.narrative)}}</p>
        <div class="grid">
          <div class="card">
            <h3>Answer</h3>
            ${{answerHtml}}
          </div>
          <div>
            <div class="card" style="margin-bottom: 16px;">
              <h3>Citation Diagnostics</h3>
              <div class="pill-row">
                <span class="pill">coverage: ${{escapeHtml(fmt(diagnostics.citation_coverage))}}</span>
                <span class="pill">density: ${{escapeHtml(fmt(diagnostics.citation_density_per_sentence))}}</span>
                <span class="pill">uncited refs: ${{escapeHtml(fmt(diagnostics.uncited_reference_count))}}</span>
                <span class="pill">invalid citations: ${{escapeHtml(fmt(validator.invalid_citation_count))}}</span>
                <span class="pill">answer words: ${{escapeHtml(fmt(diagnostics.answer_word_count))}}</span>
              </div>
            </div>
            <div class="card">
              <h3>References & Evidence</h3>
              ${{referenceHtml}}
            </div>
          </div>
        </div>
      `;
    }}

    document.getElementById('search').addEventListener('input', renderTopicList);
    renderMetrics();
    renderTopicList();
    renderContent();
  </script>
</body>
</html>
"""
    return template.replace("__TITLE__", escape(title)).replace("__JSON_PAYLOAD__", json_payload)
