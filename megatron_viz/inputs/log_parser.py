"""Parse Megatron-LM arguments from training logs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from megatron_viz.inputs.shell_parser import parse_megatron_args

_LOG_KV_RE = re.compile(
    r"^\s*(?:\[[^\]]+\]\s*)?(?P<key>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s*"
    r"(?:=|:|\.\.\.+)\s*(?P<value>[^,;\n]+?)\s*$"
)
_USING_WORLD_RE = re.compile(
    r"using world size:\s*(?P<world>\d+).*?"
    r"data[-_ ]parallel[-_ ]size:\s*(?P<dp>\d+).*?"
    r"tensor[-_ ]model[-_ ]parallel size:\s*(?P<tp>\d+).*?"
    r"pipeline[-_ ]model[-_ ]parallel size:\s*(?P<pp>\d+)",
    re.IGNORECASE,
)
_KNOWN_ALIASES = {
    "world size": "world_size",
    "world_size": "world_size",
    "tensor model parallel size": "tensor_model_parallel_size",
    "tensor_model_parallel_size": "tensor_model_parallel_size",
    "pipeline model parallel size": "pipeline_model_parallel_size",
    "pipeline_model_parallel_size": "pipeline_model_parallel_size",
    "context parallel size": "context_parallel_size",
    "context_parallel_size": "context_parallel_size",
    "expert model parallel size": "expert_model_parallel_size",
    "expert_model_parallel_size": "expert_model_parallel_size",
    "expert tensor parallel size": "expert_tensor_parallel_size",
    "expert_tensor_parallel_size": "expert_tensor_parallel_size",
    "data parallel size": "data_parallel_size",
    "data_parallel_size": "data_parallel_size",
    "num layers": "num_layers",
    "num_layers": "num_layers",
    "hidden size": "hidden_size",
    "hidden_size": "hidden_size",
    "ffn hidden size": "ffn_hidden_size",
    "ffn_hidden_size": "ffn_hidden_size",
    "num attention heads": "num_attention_heads",
    "num_attention_heads": "num_attention_heads",
    "num query groups": "num_query_groups",
    "num_query_groups": "num_query_groups",
    "seq length": "seq_length",
    "seq_length": "seq_length",
    "max position embeddings": "max_position_embeddings",
    "max_position_embeddings": "max_position_embeddings",
    "vocab size": "vocab_size",
    "vocab_size": "vocab_size",
    "normalization": "normalization",
    "micro batch size": "micro_batch_size",
    "micro_batch_size": "micro_batch_size",
    "global batch size": "global_batch_size",
    "global_batch_size": "global_batch_size",
    "num experts": "num_experts",
    "num_experts": "num_experts",
    "moe router topk": "moe_router_topk",
    "moe_router_topk": "moe_router_topk",
    "moe ffn hidden size": "moe_ffn_hidden_size",
    "moe_ffn_hidden_size": "moe_ffn_hidden_size",
    "transformer impl": "transformer_impl",
    "transformer_impl": "transformer_impl",
    "bf16": "bf16",
    "fp16": "fp16",
    "sequence parallel": "sequence_parallel",
    "sequence_parallel": "sequence_parallel",
}


def _coerce_log_value(value: str) -> Any:
    cleaned = value.strip().strip("'\"")
    lowered = cleaned.lower()
    if lowered in {"true", "yes", "enabled", "on"}:
        return True
    if lowered in {"false", "no", "disabled", "off"}:
        return False
    try:
        return int(cleaned)
    except ValueError:
        pass
    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def _canonical_key(key: str) -> str | None:
    normalized = re.sub(r"\s+", " ", key.strip().lower().replace("-", "_")).strip()
    return _KNOWN_ALIASES.get(normalized)


def parse_megatron_log(text: str) -> dict[str, Any]:
    """Parse launch args and common resolved-argument lines from a Megatron log."""

    args = parse_megatron_args(text)
    for line in text.splitlines():
        world_match = _USING_WORLD_RE.search(line)
        if world_match:
            args["world_size"] = int(world_match.group("world"))
            args["data_parallel_size"] = int(world_match.group("dp"))
            args["tensor_model_parallel_size"] = int(world_match.group("tp"))
            args["pipeline_model_parallel_size"] = int(world_match.group("pp"))
            continue
        match = _LOG_KV_RE.match(line)
        if not match:
            continue
        key = _canonical_key(match.group("key"))
        if key:
            args[key] = _coerce_log_value(match.group("value"))
    return args


def parse_log_file(path: str | Path) -> dict[str, Any]:
    """Read and parse a Megatron-LM log file."""

    return parse_megatron_log(Path(path).read_text(encoding="utf-8"))
