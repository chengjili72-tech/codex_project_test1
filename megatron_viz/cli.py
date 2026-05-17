"""Command line interface for the Megatron-LM visualizer MVP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from megatron_viz.inputs.log_parser import parse_log_file
from megatron_viz.inputs.shell_parser import parse_megatron_args, parse_script_file
from megatron_viz.ir.normalizer import normalize_args
from megatron_viz.renderers.html import render_html
from megatron_viz.renderers.markdown import render_markdown, render_mermaid
from megatron_viz.server import run_server


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
    elif output_format == "html":
        text = render_html(config)
    else:
        text = render_markdown(config)
    _write_output(text, out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="megatron-viz", description="Visualize Megatron-LM launch configurations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--format", choices=["markdown", "mermaid", "json", "html"], default="markdown")
        subparser.add_argument("--out", help="Output path. Defaults to stdout.")

    from_command = subparsers.add_parser("from-command", help="Parse a copied Megatron-LM launch command.")
    from_command.add_argument("launch_command", help="The full launch command, quoted as a single shell argument.")
    add_common(from_command)

    from_script = subparsers.add_parser("from-script", help="Parse a simple Megatron-LM shell launch script.")
    from_script.add_argument("path", help="Path to a launch script.")
    add_common(from_script)

    from_log = subparsers.add_parser("from-log", help="Parse a Megatron-LM training log.")
    from_log.add_argument("path", help="Path to a training log.")
    add_common(from_log)

    serve = subparsers.add_parser("serve", help="Start a local upload service for visualization and config comparison.")
    serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind. Defaults to 127.0.0.1.")
    serve.add_argument("--port", type=int, default=8765, help="Port to bind. Defaults to 8765.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    if parsed.command == "from-command":
        args_dict = parse_megatron_args(parsed.launch_command)
    elif parsed.command == "from-script":
        args_dict = parse_script_file(parsed.path)
    elif parsed.command == "from-log":
        args_dict = parse_log_file(parsed.path)
    elif parsed.command == "serve":
        run_server(host=parsed.host, port=parsed.port)
        return 0
    else:
        parser.error(f"unsupported command: {parsed.command}")
    _emit(args_dict, parsed.format, parsed.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
