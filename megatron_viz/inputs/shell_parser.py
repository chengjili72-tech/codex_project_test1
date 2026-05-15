"""Parse Megatron-LM arguments from copied commands or shell launch scripts."""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path
from typing import Any

_BOOL_NEGATION_PREFIX = "no-"
_ASSIGNMENT_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_VAR_REF_RE = re.compile(
    r"\$\(\((?P<arith>[^)]*)\)\)|"
    r"\$\{#(?P<array_len>[A-Za-z_][A-Za-z0-9_]*)\[@\]\}|"
    r"\$\{(?P<braced_array>[A-Za-z_][A-Za-z0-9_]*)\[(?P<array_index>\d+)\]\}|"
    r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|"
    r"\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)"
)


class _SafeArithmeticEvaluator(ast.NodeVisitor):
    """Evaluate the tiny integer arithmetic subset used by launch scripts."""

    _BINARY = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Div: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
    }
    _UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}

    def visit_Expression(self, node: ast.Expression) -> int:  # noqa: N802
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> int:  # noqa: N802
        if isinstance(node.value, int):
            return node.value
        raise ValueError("only integer constants are supported")

    def visit_BinOp(self, node: ast.BinOp) -> int:  # noqa: N802
        op_type = type(node.op)
        if op_type not in self._BINARY:
            raise ValueError(f"unsupported arithmetic operator: {op_type.__name__}")
        return self._BINARY[op_type](self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> int:  # noqa: N802
        op_type = type(node.op)
        if op_type not in self._UNARY:
            raise ValueError(f"unsupported unary operator: {op_type.__name__}")
        return self._UNARY[op_type](self.visit(node.operand))

    def generic_visit(self, node: ast.AST) -> int:
        raise ValueError(f"unsupported arithmetic expression: {type(node).__name__}")


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
            if char == "#" and not in_single and not in_double and "".join(chars[-2:]) != "${":
                break
            chars.append(char)
        lines.append("".join(chars))
    return "\n".join(lines)


def _join_line_continuations(text: str) -> str:
    return re.sub(r"\\\s*\n", " ", text)


def _quote_delta(text: str) -> int:
    in_single = False
    in_double = False
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\" and not in_single:
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
    return int(in_single) + int(in_double)


def _collect_assignment(lines: list[str], start: int) -> tuple[str, str, int] | None:
    match = _ASSIGNMENT_RE.match(lines[start])
    if not match:
        return None
    name, rhs = match.group(1), match.group(2)
    consumed = 1
    while _quote_delta(rhs) and start + consumed < len(lines):
        rhs += "\n" + lines[start + consumed]
        consumed += 1
    return name, rhs, consumed


def _strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _safe_eval_arithmetic(expr: str, variables: dict[str, str]) -> str:
    substituted = re.sub(r"\$?[A-Za-z_][A-Za-z0-9_]*", lambda m: variables.get(m.group(0).lstrip("$"), "0"), expr)
    if not re.fullmatch(r"[0-9+\-*/%()\s]+", substituted):
        return f"$(({expr}))"
    try:
        parsed = ast.parse(substituted, mode="eval")
        return str(_SafeArithmeticEvaluator().visit(parsed))
    except (SyntaxError, ValueError, ZeroDivisionError):
        return f"$(({expr}))"


def _expand_value(value: str, variables: dict[str, str], arrays: dict[str, list[str]]) -> str:
    def replace(match: re.Match[str]) -> str:
        if match.group("arith") is not None:
            return _safe_eval_arithmetic(match.group("arith"), variables)
        if match.group("array_len") is not None:
            return str(len(arrays.get(match.group("array_len"), [])))
        if match.group("braced_array") is not None:
            values = arrays.get(match.group("braced_array"), [])
            index = int(match.group("array_index"))
            return values[index] if index < len(values) else ""
        name = match.group("braced") or match.group("plain")
        return variables.get(name, "")

    previous = None
    expanded = value
    # Run a few passes so WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) works after NNODES was expanded.
    for _ in range(4):
        if expanded == previous:
            break
        previous = expanded
        expanded = _VAR_REF_RE.sub(replace, expanded)
    return expanded


def _evaluate_assignment(rhs: str, variables: dict[str, str], arrays: dict[str, list[str]]) -> tuple[str, list[str] | None]:
    rhs = rhs.strip()
    if rhs.startswith("(") and rhs.endswith(")"):
        array_items = shlex.split(rhs[1:-1], posix=True)
        return " ".join(array_items), array_items
    stripped = _strip_wrapping_quotes(rhs)
    return _expand_value(stripped, variables, arrays), None


def extract_shell_variables(text: str) -> dict[str, str]:
    """Extract statically resolvable shell variables without executing the script."""

    lines = _strip_comments_preserving_quotes(text).splitlines()
    variables: dict[str, str] = {}
    arrays: dict[str, list[str]] = {}
    index = 0
    while index < len(lines):
        collected = _collect_assignment(lines, index)
        if collected is None:
            index += 1
            continue
        name, rhs, consumed = collected
        value, array_value = _evaluate_assignment(rhs, variables, arrays)
        variables[name] = value
        if array_value is not None:
            arrays[name] = array_value
        index += consumed
    return variables


def expand_shell_variables(text: str) -> str:
    """Expand statically resolvable shell variables inside a copied script."""

    lines = _strip_comments_preserving_quotes(text).splitlines()
    variables: dict[str, str] = {}
    arrays: dict[str, list[str]] = {}
    expanded_lines: list[str] = []
    index = 0
    while index < len(lines):
        collected = _collect_assignment(lines, index)
        if collected is not None:
            name, rhs, consumed = collected
            value, array_value = _evaluate_assignment(rhs, variables, arrays)
            variables[name] = value
            if array_value is not None:
                arrays[name] = array_value
            index += consumed
            continue
        expanded_lines.append(_expand_value(lines[index], variables, arrays))
        index += 1
    return "\n".join(expanded_lines + list(variables.values()))


def tokenize_command(text: str) -> list[str]:
    """Tokenize a copied launch command or shell script without executing it."""

    expanded = expand_shell_variables(text)
    cleaned = _join_line_continuations(expanded)
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
    """Extract --style arguments from a Megatron launch command or shell script.

    The parser intentionally does not execute shell code. It supports the common
    Megatron forms ``--flag``, ``--key value``, and ``--key=value``. Repeated
    keys keep the last value, matching argparse's usual behavior for scalars.
    It also resolves simple shell assignments and quoted argument blocks such as
    ``TP=2`` and ``GPT_ARGS="--tensor-model-parallel-size ${TP}"``.
    """

    variables = extract_shell_variables(text)
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
    variable_aliases = {
        "WORLD_SIZE": "world_size",
        "TP": "tensor_model_parallel_size",
        "PP": "pipeline_model_parallel_size",
        "EP": "expert_model_parallel_size",
        "CP": "context_parallel_size",
        "NUM_LAYERS": "num_layers",
        "SEQ_LEN": "seq_length",
        "MBS": "micro_batch_size",
        "GBS": "global_batch_size",
    }
    for variable_name, arg_name in variable_aliases.items():
        if arg_name not in args and variable_name in variables:
            args[arg_name] = _coerce_value(variables[variable_name])
    return args


def parse_script_file(path: str | Path) -> dict[str, Any]:
    """Read and parse a launch script file."""

    return parse_megatron_args(Path(path).read_text(encoding="utf-8"))
