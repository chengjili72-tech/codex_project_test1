"""Standalone browser HTML rendering for Megatron-LM visualizations."""

from __future__ import annotations

import html
import json

from megatron_viz.ir.config import MegatronConfig
from megatron_viz.reports.diagnostics import diagnostics, pipeline_layer_ranges, shape_examples


def _json_for_browser(config: MegatronConfig) -> str:
    return json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True).replace("<", "\\u003c")


def _diagnostic_class(level: str) -> str:
    return {"OK": "ok", "WARN": "warn", "ERROR": "error"}.get(level, "info")


def _layer_cards(config: MegatronConfig) -> str:
    model = config.model
    attention_name = "MLA / Multi-Latent Attention" if config.raw_args.get("multi_latent_attention") else "Self Attention"
    mlp_name = "MoE Experts" if model.num_experts else "MLP"
    cards = [
        ("Embedding", [f"vocab={model.vocab_size or '-'}", f"hidden={model.hidden_size or '-'}"]),
        (
            f"Decoder Layer × {model.num_layers or '-'}",
            [attention_name, mlp_name, f"norm={model.normalization or '-'}", f"activation={model.activation or '-'}"],
        ),
        ("LM Head", ["tied embeddings" if not model.untie_embeddings_and_output_weights else "untied embeddings"]),
    ]
    parts = []
    for title, items in cards:
        list_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in items)
        parts.append(f"<article class='card'><h3>{html.escape(title)}</h3><ul>{list_items}</ul></article>")
    return "\n".join(parts)


def _pipeline_svg(config: MegatronConfig) -> str:
    ranges = pipeline_layer_ranges(config)
    width = max(900, len(ranges) * 230)
    height = 190
    box_w = 190
    gap = 35
    y = 54
    elements = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Pipeline parallel layout'>",
        "<defs><marker id='arrow' markerWidth='10' markerHeight='10' refX='9' refY='3' orient='auto'><path d='M0,0 L0,6 L9,3 z' fill='#2563eb'/></marker></defs>",
    ]
    for idx, (stage, start, end) in enumerate(ranges):
        x = 25 + idx * (box_w + gap)
        label = f"Layers {start}-{end}" if end >= start else "No layers"
        if idx == 0:
            label = "Embedding + " + label
        if idx == len(ranges) - 1:
            label += " + Head"
        elements.extend(
            [
                f"<rect x='{x}' y='{y}' width='{box_w}' height='82' rx='16' class='stage-box'/>",
                f"<text x='{x + 18}' y='{y + 31}' class='stage-title'>PP Stage {stage}</text>",
                f"<text x='{x + 18}' y='{y + 58}' class='stage-label'>{html.escape(label)}</text>",
            ]
        )
        if idx < len(ranges) - 1:
            line_x1 = x + box_w
            line_x2 = x + box_w + gap - 9
            elements.append(f"<line x1='{line_x1}' y1='{y + 41}' x2='{line_x2}' y2='{y + 41}' class='arrow'/>")
    elements.append("</svg>")
    return "\n".join(elements)


def render_html(config: MegatronConfig) -> str:
    """Render a standalone HTML page for browser visualization."""

    model = config.model
    parallel = config.parallelism
    diag_items = "\n".join(
        f"<li class='{_diagnostic_class(item.level)}'><strong>{html.escape(item.level)}</strong> {html.escape(item.message)}</li>"
        for item in diagnostics(config)
    )
    shape_items = "\n".join(f"<li><code>{html.escape(item)}</code></li>" for item in shape_examples(config))
    unknown_items = "\n".join(
        f"<li><code>{html.escape(key)}</code>: {html.escape(str(config.unknown_args[key]))}</li>"
        for key in sorted(config.unknown_args)
    )
    data_json = _json_for_browser(config)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Megatron-LM Model Visualization</title>
  <style>
    :root {{ color-scheme: light; --bg:#f6f8fb; --panel:#ffffff; --ink:#172033; --muted:#64748b; --blue:#2563eb; --green:#16a34a; --amber:#d97706; --red:#dc2626; --line:#dbe4f0; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:32px 40px; background:linear-gradient(135deg, #172033 0%, #1d4ed8 100%); color:white; }}
    header h1 {{ margin:0 0 8px; font-size:32px; }}
    header p {{ margin:0; color:#dbeafe; }}
    main {{ padding:28px 40px 56px; max-width:1280px; margin:0 auto; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:16px; margin-top:-54px; }}
    .metric, section, .card {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; box-shadow:0 12px 30px rgba(15,23,42,.08); }}
    .metric {{ padding:18px; }}
    .metric span {{ display:block; color:var(--muted); font-size:13px; }}
    .metric strong {{ display:block; font-size:26px; margin-top:6px; }}
    section {{ margin-top:22px; padding:24px; }}
    section h2 {{ margin:0 0 16px; font-size:21px; }}
    .cards {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap:16px; }}
    .card {{ padding:18px; box-shadow:none; }}
    .card h3 {{ margin:0 0 10px; }}
    .card ul, .diag, .shapes, .unknown {{ margin:0; padding-left:20px; }}
    .pipeline {{ overflow-x:auto; }}
    .stage-box {{ fill:#eff6ff; stroke:#2563eb; stroke-width:2; }}
    .stage-title {{ font-size:15px; font-weight:700; fill:#1e3a8a; }}
    .stage-label {{ font-size:12px; fill:#334155; }}
    .arrow {{ stroke:#2563eb; stroke-width:2.5; marker-end:url(#arrow); }}
    .diag li {{ margin:8px 0; }}
    .diag .ok strong {{ color:var(--green); }} .diag .warn strong {{ color:var(--amber); }} .diag .error strong {{ color:var(--red); }}
    details {{ margin-top:16px; }}
    summary {{ cursor:pointer; font-weight:700; color:#1d4ed8; }}
    pre {{ overflow:auto; background:#0f172a; color:#e2e8f0; border-radius:14px; padding:18px; }}
    button {{ border:0; border-radius:999px; padding:10px 14px; background:#2563eb; color:white; font-weight:700; cursor:pointer; }}
  </style>
</head>
<body>
  <header>
    <h1>Megatron-LM 模型结构可视化</h1>
    <p>从启动脚本静态解析生成；可离线在浏览器打开此 HTML。</p>
  </header>
  <main>
    <div class="grid">
      <div class="metric"><span>Layers</span><strong>{html.escape(str(model.num_layers or '-'))}</strong></div>
      <div class="metric"><span>Hidden</span><strong>{html.escape(str(model.hidden_size or '-'))}</strong></div>
      <div class="metric"><span>Attention Heads</span><strong>{html.escape(str(model.num_attention_heads or '-'))}</strong></div>
      <div class="metric"><span>TP / PP / EP / CP</span><strong>{parallel.tensor_model_parallel_size} / {parallel.pipeline_model_parallel_size} / {parallel.expert_model_parallel_size} / {parallel.context_parallel_size}</strong></div>
      <div class="metric"><span>DP</span><strong>{html.escape(str(parallel.data_parallel_size or '-'))}</strong></div>
    </div>

    <section>
      <h2>逻辑模型结构</h2>
      <div class="cards">{_layer_cards(config)}</div>
    </section>

    <section>
      <h2>Pipeline 可视化</h2>
      <div class="pipeline">{_pipeline_svg(config)}</div>
    </section>

    <section>
      <h2>Tensor 示例形状</h2>
      <ul class="shapes">{shape_items}</ul>
    </section>

    <section>
      <h2>诊断</h2>
      <ul class="diag">{diag_items}</ul>
    </section>

    <section>
      <h2>原始解析数据</h2>
      <button type="button" onclick="navigator.clipboard && navigator.clipboard.writeText(JSON.stringify(window.MEGATRON_CONFIG, null, 2))">复制 JSON</button>
      <details>
        <summary>查看 normalized JSON</summary>
        <pre id="json"></pre>
      </details>
      <details>
        <summary>已解析但暂未建模的参数</summary>
        <ul class="unknown">{unknown_items}</ul>
      </details>
    </section>
  </main>
  <script id="config-data" type="application/json">{data_json}</script>
  <script>
    window.MEGATRON_CONFIG = JSON.parse(document.getElementById('config-data').textContent);
    document.getElementById('json').textContent = JSON.stringify(window.MEGATRON_CONFIG, null, 2);
  </script>
</body>
</html>
"""
