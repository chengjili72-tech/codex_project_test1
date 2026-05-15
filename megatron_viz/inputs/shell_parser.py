"""Parse Megatron-LM arguments from copied commands or simple shell scripts."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

_BOOL_NEGATION_PREFIX = "no-"


def _strip_comments_preserving_quotes(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        in_single = False
        in_double = False
        escaped = False
        chars: list[str] = []
        for char in line:
            if escaped:
                chars.append(char)
                escaped = False
                continue
            if char == "\\" and not in_single:
                chars.append(char)
                escaped = True
                continue
            if char == "'" and not in_double:
                in_single = not in_single
            elif char == '"' and not in_single:
                in_double = not in_double
            if char == "#" and not in_single and not in_double:
                break
            chars.append(char)
        lines.append("".join(chars))
    return "\n".join(lines)


def _join_line_continuations(text: str) -> str:
    return re.sub(r"\\\s*\n", " ", text)


def tokenize_command(text: str) -> list[str]:
    """Tokenize a copied launch command or simple shell script without executing it."""

    cleaned = _join_line_continuations(_strip_comments_preserving_quotes(text))
    return shlex.split(cleaned, posix=True)


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_megatron_args(text: str) -> dict[str, Any]:
    """Extract --style arguments from a Megatron launch command or simple script.

    The parser intentionally does not execute shell code. It supports the common
    Megatron forms ``--flag``, ``--key value``, and ``--key=value``. Repeated
    keys keep the last value, matching argparse's usual behavior for scalars.
    """

    tokens = tokenize_command(text)
    args: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--") or token == "--":
            i += 1
            continue

        key_value = token[2:]
        if "=" in key_value:
            key, value = key_value.split("=", 1)
            args[key.replace("-", "_")] = _coerce_value(value)
            i += 1
            continue

        key = key_value.replace("-", "_")
        if key_value.startswith(_BOOL_NEGATION_PREFIX):
            args[key[len(_BOOL_NEGATION_PREFIX) :].replace("-", "_")] = False
            i += 1
            continue

        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            args[key] = _coerce_value(tokens[i + 1])
            i += 2
        else:
            args[key] = True
            i += 1
    return args


def parse_script_file(path: str | Path) -> dict[str, Any]:
    """Read and parse a launch script file."""

    return parse_megatron_args(Path(path).read_text(encoding="utf-8"))
