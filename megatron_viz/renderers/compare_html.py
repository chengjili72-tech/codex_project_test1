"""HTML renderer for side-by-side Megatron config comparison."""

from __future__ import annotations

import html
import json

from megatron_viz.ir.config import MegatronConfig
from megatron_viz.renderers.html import _pipeline_svg
from megatron_viz.reports.comparison import ConfigDiff, compare_configs


def _fmt(value: object) -> str:
    return "-" if value is None else str(value)


def _summary_cards(left: MegatronConfig, right: MegatronConfig, diffs: list[ConfigDiff]) -> str:
    changed = sum(1 for diff in diffs if diff.changed)
    return f"""
    <div class='grid'>
      <div class='metric'><span>Changed key params</span><strong>{changed}</strong></div>
      <div class='metric'><span>Left TP/PP/EP/CP</span><strong>{left.parallelism.tensor_model_parallel_size}/{left.parallelism.pipeline_model_parallel_size}/{left.parallelism.expert_model_parallel_size}/{left.parallelism.context_parallel_size}</strong></div>
      <div class='metric'><span>Right TP/PP/EP/CP</span><strong>{right.parallelism.tensor_model_parallel_size}/{right.parallelism.pipeline_model_parallel_size}/{right.parallelism.expert_model_parallel_size}/{right.parallelism.context_parallel_size}</strong></div>
      <div class='metric'><span>Left layers / hidden</span><strong>{_fmt(left.model.num_layers)} / {_fmt(left.model.hidden_size)}</strong></div>
      <div class='metric'><span>Right layers / hidden</span><strong>{_fmt(right.model.num_layers)} / {_fmt(right.model.hidden_size)}</strong></div>
    </div>
    """


def _diff_table(diffs: list[ConfigDiff]) -> str:
    rows = []
    for diff in diffs:
        css = "changed" if diff.changed else "same"
        badge = "不同" if diff.changed else "相同"
        rows.append(
            "<tr class='{css}'><td><code>{path}</code></td><td>{left}</td><td>{right}</td><td><span>{badge}</span></td></tr>".format(
                css=css,
                path=html.escape(diff.path),
                left=html.escape(_fmt(diff.left)),
                right=html.escape(_fmt(diff.right)),
                badge=badge,
            )
        )
    return "\n".join(rows)


def render_compare_html(left: MegatronConfig, right: MegatronConfig, *, left_name: str = "left", right_name: str = "right") -> str:
    """Render a standalone browser page that highlights key-parameter differences."""

    diffs = compare_configs(left, right)
    payload = json.dumps({"left": left.to_dict(), "right": right.to_dict()}, ensure_ascii=False, sort_keys=True).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Megatron-LM Config Compare</title>
  <style>
    :root {{ --bg:#f8fafc; --panel:#fff; --ink:#172033; --muted:#64748b; --blue:#2563eb; --red:#dc2626; --green:#16a34a; --amber:#fef3c7; --line:#dbe4f0; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:32px 40px; color:white; background:linear-gradient(135deg,#111827,#7c3aed); }}
    main {{ max-width:1320px; margin:0 auto; padding:28px 40px 56px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:16px; margin-top:-54px; }}
    .metric, section {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; box-shadow:0 12px 30px rgba(15,23,42,.08); }}
    .metric {{ padding:18px; }} .metric span {{ color:var(--muted); font-size:13px; display:block; }} .metric strong {{ display:block; font-size:24px; margin-top:6px; }}
    section {{ margin-top:22px; padding:24px; }} h1, h2 {{ margin-top:0; }}
    .panes {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:18px; }}
    .pipeline {{ overflow-x:auto; border:1px solid var(--line); border-radius:14px; padding:8px; }}
    .stage-box {{ fill:#eff6ff; stroke:#2563eb; stroke-width:2; }} .stage-title {{ font-size:15px; font-weight:700; fill:#1e3a8a; }} .stage-label {{ font-size:12px; fill:#334155; }} .arrow {{ stroke:#2563eb; stroke-width:2.5; marker-end:url(#arrow); }}
    table {{ width:100%; border-collapse:collapse; }} th, td {{ border-bottom:1px solid var(--line); padding:10px 12px; text-align:left; vertical-align:top; }}
    tr.changed {{ background:var(--amber); }} tr.changed td:first-child {{ border-left:5px solid var(--red); }} tr.same {{ color:#475569; }}
    tr.changed span {{ color:white; background:var(--red); border-radius:999px; padding:3px 9px; font-weight:700; }} tr.same span {{ color:white; background:var(--green); border-radius:999px; padding:3px 9px; }}
    pre {{ overflow:auto; background:#0f172a; color:#e2e8f0; border-radius:14px; padding:18px; }} summary {{ cursor:pointer; font-weight:700; color:#6d28d9; }}
  </style>
</head>
<body>
  <header><h1>Megatron-LM 配置对比</h1><p>对关键模型、并行和训练参数进行可视化差异标注。</p></header>
  <main>
    {_summary_cards(left, right, diffs)}
    <section>
      <h2>Pipeline 对比</h2>
      <div class='panes'>
        <div><h3>{html.escape(left_name)}</h3><div class='pipeline'>{_pipeline_svg(left)}</div></div>
        <div><h3>{html.escape(right_name)}</h3><div class='pipeline'>{_pipeline_svg(right)}</div></div>
      </div>
    </section>
    <section>
      <h2>关键参数差异</h2>
      <table><thead><tr><th>参数</th><th>{html.escape(left_name)}</th><th>{html.escape(right_name)}</th><th>状态</th></tr></thead><tbody>{_diff_table(diffs)}</tbody></table>
    </section>
    <section>
      <h2>Normalized JSON</h2>
      <details><summary>查看对比输入 JSON</summary><pre id='json'></pre></details>
    </section>
  </main>
  <script id='compare-data' type='application/json'>{payload}</script>
  <script>document.getElementById('json').textContent = JSON.stringify(JSON.parse(document.getElementById('compare-data').textContent), null, 2);</script>
</body>
</html>
"""
