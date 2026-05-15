"""Diagnostics and simple structural derivations for Megatron configs."""

from __future__ import annotations

from dataclasses import dataclass

from megatron_viz.ir.config import MegatronConfig


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A user-facing validation or information item."""

    level: str
    message: str


def pipeline_layer_ranges(config: MegatronConfig) -> list[tuple[int, int, int]]:
    """Return inclusive layer ranges per pipeline stage.

    The MVP uses an even contiguous split and assigns any remainder to earlier
    stages so the rendered plan is explicit even for imperfect splits.
    """

    layers = config.model.num_layers or 0
    pp = max(config.parallelism.pipeline_model_parallel_size, 1)
    if layers <= 0:
        return [(stage, 0, -1) for stage in range(pp)]
    base, remainder = divmod(layers, pp)
    ranges: list[tuple[int, int, int]] = []
    start = 0
    for stage in range(pp):
        count = base + (1 if stage < remainder else 0)
        end = start + count - 1
        ranges.append((stage, start, end))
        start = end + 1
    return ranges


def shape_examples(config: MegatronConfig) -> list[str]:
    """Produce lightweight tensor-shape examples for future per-layer tracing work."""

    micro_batch = config.training.micro_batch_size or "micro_batch"
    seq = config.model.seq_length or "seq"
    hidden = config.model.hidden_size or "hidden"
    heads = config.model.num_attention_heads or "heads"
    tp = config.parallelism.tensor_model_parallel_size
    cp = config.parallelism.context_parallel_size
    head_dim = None
    if isinstance(hidden, int) and isinstance(heads, int) and heads:
        head_dim = hidden // heads
    local_seq = seq // cp if isinstance(seq, int) and cp else f"{seq}/{cp}"
    local_heads = heads // tp if isinstance(heads, int) and tp else f"{heads}/{tp}"
    examples = [
        f"tokens: [{micro_batch}, {seq}]",
        f"embedding output: [{micro_batch}, {seq}, {hidden}]",
        f"per-CP-rank hidden states: [{micro_batch}, {local_seq}, {hidden}]",
        f"per-TP-rank attention heads: {local_heads}",
    ]
    if head_dim is not None:
        examples.append(f"attention head dim: {head_dim}")
    return examples


def diagnostics(config: MegatronConfig) -> list[Diagnostic]:
    """Validate a normalized config and return report diagnostics."""

    result: list[Diagnostic] = []
    model = config.model
    parallel = config.parallelism

    model_parallel = (
        parallel.tensor_model_parallel_size
        * parallel.pipeline_model_parallel_size
        * parallel.context_parallel_size
    )
    if parallel.world_size is None:
        result.append(Diagnostic("WARN", "world_size is missing; data parallel size cannot be inferred."))
    elif parallel.world_size % model_parallel == 0:
        result.append(
            Diagnostic(
                "OK",
                f"world_size={parallel.world_size} is divisible by TP*PP*CP={model_parallel}; inferred DP={parallel.data_parallel_size}.",
            )
        )
    else:
        result.append(
            Diagnostic(
                "ERROR",
                f"world_size={parallel.world_size} is not divisible by TP*PP*CP={model_parallel}.",
            )
        )

    if model.num_attention_heads is None:
        result.append(Diagnostic("WARN", "num_attention_heads is missing; TP head split cannot be validated."))
    elif model.num_attention_heads % parallel.tensor_model_parallel_size == 0:
        result.append(
            Diagnostic(
                "OK",
                f"num_attention_heads={model.num_attention_heads} is divisible by TP={parallel.tensor_model_parallel_size}.",
            )
        )
    else:
        result.append(
            Diagnostic(
                "ERROR",
                f"num_attention_heads={model.num_attention_heads} is not divisible by TP={parallel.tensor_model_parallel_size}.",
            )
        )

    if model.hidden_size and model.num_attention_heads:
        if model.hidden_size % model.num_attention_heads == 0:
            result.append(
                Diagnostic(
                    "OK",
                    f"hidden_size={model.hidden_size} is divisible by num_attention_heads={model.num_attention_heads}.",
                )
            )
        else:
            result.append(
                Diagnostic(
                    "ERROR",
                    f"hidden_size={model.hidden_size} is not divisible by num_attention_heads={model.num_attention_heads}.",
                )
            )

    if model.num_layers is None:
        result.append(Diagnostic("WARN", "num_layers is missing; pipeline stage ranges cannot be validated."))
    elif model.num_layers % parallel.pipeline_model_parallel_size == 0:
        result.append(
            Diagnostic(
                "OK",
                f"num_layers={model.num_layers} is evenly split across PP={parallel.pipeline_model_parallel_size}.",
            )
        )
    else:
        result.append(
            Diagnostic(
                "WARN",
                f"num_layers={model.num_layers} is not evenly split across PP={parallel.pipeline_model_parallel_size}; earlier stages receive one extra layer in the report.",
            )
        )

    if parallel.virtual_pipeline_model_parallel_size and parallel.pipeline_model_parallel_size <= 1:
        result.append(Diagnostic("WARN", "virtual pipeline parallelism is set while PP is 1."))
    if model.num_experts and model.num_experts % parallel.expert_model_parallel_size != 0:
        result.append(
            Diagnostic(
                "WARN",
                f"num_experts={model.num_experts} is not divisible by EP={parallel.expert_model_parallel_size}.",
            )
        )
    elif model.num_experts:
        result.append(Diagnostic("OK", f"MoE enabled with {model.num_experts} experts."))

    return result
