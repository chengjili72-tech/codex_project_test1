"""Convert parsed CLI args into a normalized Megatron visualizer config."""

from __future__ import annotations

from typing import Any

from megatron_viz.ir.config import MegatronConfig, ModelConfig, ParallelConfig, TrainingConfig

_MODEL_FIELDS = {
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "seq_length",
    "max_position_embeddings",
    "vocab_size",
    "position_embedding_type",
    "normalization",
    "untie_embeddings_and_output_weights",
    "num_experts",
    "moe_router_topk",
    "moe_ffn_hidden_size",
}
_PARALLEL_FIELDS = {
    "world_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "sequence_parallel",
}
_TRAINING_FIELDS = {
    "micro_batch_size",
    "global_batch_size",
    "fp16",
    "bf16",
    "fp8_format",
    "recompute_granularity",
    "recompute_method",
    "transformer_impl",
}


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    return int(value)


def _bool(value: Any) -> bool:
    return bool(value)


def normalize_args(args: dict[str, Any]) -> MegatronConfig:
    """Normalize parser output and infer derived values such as DP size."""

    model = ModelConfig(
        num_layers=_int_or_none(args.get("num_layers")),
        hidden_size=_int_or_none(args.get("hidden_size")),
        ffn_hidden_size=_int_or_none(args.get("ffn_hidden_size")),
        num_attention_heads=_int_or_none(args.get("num_attention_heads")),
        num_query_groups=_int_or_none(args.get("num_query_groups")),
        seq_length=_int_or_none(args.get("seq_length")),
        max_position_embeddings=_int_or_none(args.get("max_position_embeddings")),
        vocab_size=_int_or_none(args.get("vocab_size")),
        position_embedding_type=args.get("position_embedding_type"),
        normalization=args.get("normalization"),
        activation="SwiGLU" if args.get("swiglu") else args.get("activation"),
        untie_embeddings_and_output_weights=_bool(args.get("untie_embeddings_and_output_weights")),
        num_experts=_int_or_none(args.get("num_experts")),
        moe_router_topk=_int_or_none(args.get("moe_router_topk")),
        moe_ffn_hidden_size=_int_or_none(args.get("moe_ffn_hidden_size")),
    )
    if model.ffn_hidden_size is None and model.hidden_size is not None:
        model.ffn_hidden_size = model.hidden_size * 4

    parallelism = ParallelConfig(
        world_size=_int_or_none(args.get("world_size")),
        tensor_model_parallel_size=_int_or_none(args.get("tensor_model_parallel_size")) or 1,
        pipeline_model_parallel_size=_int_or_none(args.get("pipeline_model_parallel_size")) or 1,
        context_parallel_size=_int_or_none(args.get("context_parallel_size")) or 1,
        expert_model_parallel_size=_int_or_none(args.get("expert_model_parallel_size")) or 1,
        expert_tensor_parallel_size=_int_or_none(args.get("expert_tensor_parallel_size")) or 1,
        virtual_pipeline_model_parallel_size=_int_or_none(args.get("virtual_pipeline_model_parallel_size")),
        sequence_parallel=_bool(args.get("sequence_parallel")),
    )
    model_parallel = (
        parallelism.tensor_model_parallel_size
        * parallelism.pipeline_model_parallel_size
        * parallelism.context_parallel_size
    )
    if parallelism.world_size and model_parallel > 0 and parallelism.world_size % model_parallel == 0:
        parallelism.data_parallel_size = parallelism.world_size // model_parallel

    training = TrainingConfig(
        micro_batch_size=_int_or_none(args.get("micro_batch_size")),
        global_batch_size=_int_or_none(args.get("global_batch_size")),
        fp16=_bool(args.get("fp16")),
        bf16=_bool(args.get("bf16")),
        fp8_format=args.get("fp8_format"),
        recompute_granularity=args.get("recompute_granularity"),
        recompute_method=args.get("recompute_method"),
        transformer_impl=args.get("transformer_impl"),
    )

    known = _MODEL_FIELDS | _PARALLEL_FIELDS | _TRAINING_FIELDS | {"swiglu", "activation"}
    unknown = {key: value for key, value in args.items() if key not in known}
    return MegatronConfig(model=model, parallelism=parallelism, training=training, raw_args=args, unknown_args=unknown)
