"""Command line interface for the Megatron-LM visualizer MVP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from megatron_viz.inputs.shell_parser import parse_megatron_args, parse_script_file
from megatron_viz.ir.normalizer import normalize_args
from megatron_viz.renderers.markdown import render_markdown, render_mermaid


def _write_output(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")


def _emit(args_dict: dict[str, object], output_format: str, out: str | None) -> None:
    config = normalize_args(args_dict)
    if output_format == "json":
        text = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    elif output_format == "mermaid":
        text = render_mermaid(config) + "\n"
    else:
        text = render_markdown(config)
    _write_output(text, out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="megatron-viz", description="Visualize Megatron-LM launch configurations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--format", choices=["markdown", "mermaid", "json"], default="markdown")
        subparser.add_argument("--out", help="Output path. Defaults to stdout.")

    from_command = subparsers.add_parser("from-command", help="Parse a copied Megatron-LM launch command.")
    from_command.add_argument("launch_command", help="The full launch command, quoted as a single shell argument.")
    add_common(from_command)

    from_script = subparsers.add_parser("from-script", help="Parse a simple Megatron-LM shell launch script.")
    from_script.add_argument("path", help="Path to a launch script.")
    add_common(from_script)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    if parsed.command == "from-command":
        args_dict = parse_megatron_args(parsed.launch_command)
    elif parsed.command == "from-script":
        args_dict = parse_script_file(parsed.path)
    else:
        parser.error(f"unsupported command: {parsed.command}")
    _emit(args_dict, parsed.format, parsed.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
