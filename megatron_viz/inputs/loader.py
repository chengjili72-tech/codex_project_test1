"""Load normalized configs from scripts, commands, or logs."""

from __future__ import annotations

from pathlib import Path

from megatron_viz.inputs.log_parser import parse_megatron_log
from megatron_viz.inputs.shell_parser import parse_megatron_args
from megatron_viz.ir.config import MegatronConfig
from megatron_viz.ir.normalizer import normalize_args

_LOG_SUFFIXES = {".log", ".out", ".txt"}


def parse_text_by_kind(text: str, *, filename: str = "", kind: str = "auto") -> dict[str, object]:
    """Parse text as a shell script/command or log file."""

    if kind == "log" or (kind == "auto" and Path(filename).suffix.lower() in _LOG_SUFFIXES):
        return parse_megatron_log(text)
    return parse_megatron_args(text)


def config_from_text(text: str, *, filename: str = "", kind: str = "auto") -> MegatronConfig:
    """Parse and normalize text into a MegatronConfig."""

    return normalize_args(parse_text_by_kind(text, filename=filename, kind=kind))


def config_from_file(path: str | Path, *, kind: str = "auto") -> MegatronConfig:
    """Read a file and normalize it into a MegatronConfig."""

    file_path = Path(path)
    return config_from_text(file_path.read_text(encoding="utf-8"), filename=file_path.name, kind=kind)
