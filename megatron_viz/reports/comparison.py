"""Compare normalized Megatron configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from megatron_viz.ir.config import MegatronConfig

KEY_PARAMETER_PATHS = [
    "model.num_layers",
    "model.hidden_size",
    "model.ffn_hidden_size",
    "model.num_attention_heads",
    "model.num_query_groups",
    "model.seq_length",
    "model.max_position_embeddings",
    "model.vocab_size",
    "model.normalization",
    "model.activation",
    "model.num_experts",
    "model.moe_router_topk",
    "model.moe_ffn_hidden_size",
    "parallelism.world_size",
    "parallelism.tensor_model_parallel_size",
    "parallelism.pipeline_model_parallel_size",
    "parallelism.context_parallel_size",
    "parallelism.expert_model_parallel_size",
    "parallelism.expert_tensor_parallel_size",
    "parallelism.virtual_pipeline_model_parallel_size",
    "parallelism.sequence_parallel",
    "parallelism.data_parallel_size",
    "training.micro_batch_size",
    "training.global_batch_size",
    "training.fp16",
    "training.bf16",
    "training.fp8_format",
    "training.recompute_granularity",
    "training.recompute_method",
    "training.transformer_impl",
]


@dataclass(frozen=True, slots=True)
class ConfigDiff:
    """One field-level config comparison result."""

    path: str
    left: Any
    right: Any
    changed: bool
    important: bool = True


def _get_path(config: MegatronConfig, path: str) -> Any:
    current: Any = config
    for part in path.split("."):
        current = getattr(current, part)
    return current


def compare_configs(left: MegatronConfig, right: MegatronConfig) -> list[ConfigDiff]:
    """Compare key model, parallelism, and training parameters."""

    return [
        ConfigDiff(path=path, left=_get_path(left, path), right=_get_path(right, path), changed=_get_path(left, path) != _get_path(right, path))
        for path in KEY_PARAMETER_PATHS
    ]


def changed_diffs(left: MegatronConfig, right: MegatronConfig) -> list[ConfigDiff]:
    """Return changed key-parameter diffs only."""

    return [diff for diff in compare_configs(left, right) if diff.changed]
