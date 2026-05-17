"""Markdown and Mermaid rendering for normalized Megatron configs."""

from __future__ import annotations

import json

from megatron_viz.ir.config import MegatronConfig
from megatron_viz.reports.diagnostics import diagnostics, pipeline_layer_ranges, shape_examples


def _value(value: object) -> str:
    return "-" if value is None else str(value)


def render_mermaid(config: MegatronConfig) -> str:
    """Render a pipeline-oriented Mermaid graph."""

    lines = ["flowchart LR", "  E[Embedding]", "  H[Final Norm + LM Head]"]
    ranges = pipeline_layer_ranges(config)
    previous = "E"
    for stage, start, end in ranges:
        label = f"PP{stage}: Layers {start}-{end}" if end >= start else f"PP{stage}: no decoder layers"
        node = f"PP{stage}"
        lines.append(f"  {node}[{label}]")
        lines.append(f"  {previous} --> {node}")
        previous = node
    lines.append(f"  {previous} --> H")
    lines.append("  classDef parallel fill:#eef6ff,stroke:#2b6cb0,stroke-width:1px;")
    for stage, _, _ in ranges:
        lines.append(f"  class PP{stage} parallel;")
    return "\n".join(lines)


def render_markdown(config: MegatronConfig) -> str:
    """Render a human-readable report with Mermaid and diagnostics."""

    model = config.model
    parallel = config.parallelism
    training = config.training
    lines = [
        "# Megatron-LM Model Visualization Report",
        "",
        "## Model Summary",
        "",
        f"- Model type: `{model.model_type}`",
        f"- Layers: `{_value(model.num_layers)}`",
        f"- Hidden size: `{_value(model.hidden_size)}`",
        f"- FFN hidden size: `{_value(model.ffn_hidden_size)}`",
        f"- Attention heads: `{_value(model.num_attention_heads)}`",
        f"- Query groups: `{_value(model.num_query_groups)}`",
        f"- Sequence length: `{_value(model.seq_length)}`",
        f"- Vocabulary size: `{_value(model.vocab_size)}`",
        f"- Activation: `{_value(model.activation)}`",
        f"- Normalization: `{_value(model.normalization)}`",
        "",
        "## Parallelism Summary",
        "",
        f"- World size: `{_value(parallel.world_size)}`",
        f"- Tensor parallel size: `{parallel.tensor_model_parallel_size}`",
        f"- Pipeline parallel size: `{parallel.pipeline_model_parallel_size}`",
        f"- Context parallel size: `{parallel.context_parallel_size}`",
        f"- Data parallel size: `{_value(parallel.data_parallel_size)}`",
        f"- Expert parallel size: `{parallel.expert_model_parallel_size}`",
        f"- Virtual pipeline size: `{_value(parallel.virtual_pipeline_model_parallel_size)}`",
        f"- Sequence parallel: `{parallel.sequence_parallel}`",
        "",
        "## Training Hints",
        "",
        f"- Micro batch size: `{_value(training.micro_batch_size)}`",
        f"- Global batch size: `{_value(training.global_batch_size)}`",
        f"- Precision: `{'bf16' if training.bf16 else 'fp16' if training.fp16 else 'unspecified'}`",
        f"- Recompute: `{_value(training.recompute_granularity)}` / `{_value(training.recompute_method)}`",
        "",
        "## Pipeline Layout",
        "",
    ]
    for stage, start, end in pipeline_layer_ranges(config):
        layer_text = f"Layers {start}-{end}" if end >= start else "No decoder layers"
        prefix = "Embedding + " if stage == 0 else ""
        suffix = " + Final Norm + LM Head" if stage == parallel.pipeline_model_parallel_size - 1 else ""
        lines.append(f"- PP stage {stage}: {prefix}{layer_text}{suffix}")

    lines.extend([
        "",
        "## Mermaid Diagram",
        "",
        "```mermaid",
        render_mermaid(config),
        "```",
        "",
        "## Tensor Shape Examples",
        "",
    ])
    for example in shape_examples(config):
        lines.append(f"- `{example}`")

    lines.extend(["", "## Diagnostics", ""])
    for item in diagnostics(config):
        lines.append(f"- **{item.level}**: {item.message}")

    if config.unknown_args:
        lines.extend(["", "## Parsed But Not Yet Modeled", ""])
        for key in sorted(config.unknown_args):
            lines.append(f"- `{key}`: `{config.unknown_args[key]}`")

    lines.extend([
        "",
        "## Normalized JSON",
        "",
        "```json",
        json.dumps(config.to_dict(), indent=2, sort_keys=True),
        "```",
        "",
    ])
    return "\n".join(lines)
