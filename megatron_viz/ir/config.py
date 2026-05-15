"""Normalized configuration objects for Megatron-LM visualization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    """Logical transformer model settings that affect structure rendering."""

    model_type: str = "gpt"
    num_layers: int | None = None
    hidden_size: int | None = None
    ffn_hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_query_groups: int | None = None
    seq_length: int | None = None
    max_position_embeddings: int | None = None
    vocab_size: int | None = None
    position_embedding_type: str | None = None
    normalization: str | None = None
    activation: str | None = None
    untie_embeddings_and_output_weights: bool = False
    num_experts: int | None = None
    moe_router_topk: int | None = None
    moe_ffn_hidden_size: int | None = None


@dataclass(slots=True)
class ParallelConfig:
    """Megatron parallelism settings."""

    world_size: int | None = None
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    sequence_parallel: bool = False
    data_parallel_size: int | None = None


@dataclass(slots=True)
class TrainingConfig:
    """Training settings that influence generated reports and future shape samples."""

    micro_batch_size: int | None = None
    global_batch_size: int | None = None
    fp16: bool = False
    bf16: bool = False
    fp8_format: str | None = None
    recompute_granularity: str | None = None
    recompute_method: str | None = None
    transformer_impl: str | None = None


@dataclass(slots=True)
class MegatronConfig:
    """Complete normalized config plus parser metadata."""

    model: ModelConfig = field(default_factory=ModelConfig)
    parallelism: ParallelConfig = field(default_factory=ParallelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    raw_args: dict[str, Any] = field(default_factory=dict)
    unknown_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)
