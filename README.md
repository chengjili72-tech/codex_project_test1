# Megatron Visualizer MVP

`megatron-viz` is an early Megatron-LM visualization tool. The current MVP follows **方案 A**: it parses a copied launch command or a simple Megatron-LM shell launch script, normalizes the key model and parallelism arguments, and emits Markdown, Mermaid, JSON, or standalone browser HTML.

The project is intentionally structured around a normalized intermediate representation so later work can add a richer frontend, Megatron log parsing, resolved-args integration, and per-layer tensor examples without rewriting the parser or renderers.

## Current capabilities

- Parse common Megatron-LM CLI argument forms:
  - `--flag`
  - `--key value`
  - `--key=value`
  - `--no-some-flag`
- Normalize important model, parallelism, MoE, and training fields.
- Infer data parallel size when `world_size` is available.
- Generate:
  - Markdown report
  - Mermaid pipeline diagram
  - normalized JSON
  - standalone browser HTML visualization
- Run basic diagnostics for TP, PP, hidden/head divisibility, world size, and MoE expert splits.
- Emit lightweight tensor-shape examples as a foundation for the future per-layer tensor walkthrough.

## Install for local development

```bash
python -m pip install -e '.[dev]'
```

The package has no runtime dependencies beyond the Python standard library.

## Usage

### From a copied command

```bash
megatron-viz from-command 'torchrun pretrain_gpt.py \
  --world-size 64 \
  --num-layers 24 \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --seq-length 2048 \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 2 \
  --micro-batch-size 2 \
  --global-batch-size 256 \
  --bf16 \
  --sequence-parallel \
  --swiglu' --out report.md
```

### From a simple script

```bash
megatron-viz from-script tests/fixtures/train_gpt.sh --out report.md
```

### Mermaid only

```bash
megatron-viz from-script tests/fixtures/train_gpt.sh --format mermaid
```

### Normalized JSON

```bash
megatron-viz from-script tests/fixtures/train_gpt.sh --format json --out config.json
```

### Browser visualization

```bash
megatron-viz from-script tests/fixtures/deepseek_moe_npu.sh --format html --out report.html
python -m webbrowser report.html
```

The HTML file is standalone and can be opened directly in a browser. It contains metric cards, logical model structure, a pipeline SVG visualization, diagnostics, tensor-shape examples, and embedded normalized JSON for future frontend work.

## Report sections

The Markdown report contains:

1. model summary;
2. TP/PP/CP/DP/EP parallelism summary;
3. training hints;
4. pipeline layer layout;
5. Mermaid diagram;
6. initial tensor-shape examples;
7. diagnostics;
8. normalized JSON for later tooling.

## Architecture

```text
Megatron command or script
  -> megatron_viz.inputs.shell_parser
  -> megatron_viz.ir.normalizer
  -> MegatronConfig IR
  -> renderers / diagnostics / future frontend
```

Important modules:

- `megatron_viz.inputs.shell_parser`: safe static parsing of copied commands and simple scripts. It does not execute shell code.
- `megatron_viz.ir.config`: dataclass-based model, parallelism, training, and top-level config objects.
- `megatron_viz.ir.normalizer`: converts parsed raw args into the stable IR and infers derived values.
- `megatron_viz.reports.diagnostics`: validation, pipeline layer ranges, and initial tensor-shape examples.
- `megatron_viz.renderers.markdown`: Markdown and Mermaid renderers.
- `megatron_viz.renderers.html`: standalone browser renderer for interactive HTML/SVG visualization.
- `megatron_viz.cli`: `megatron-viz` command line entrypoint.

## Roadmap

### Next: richer frontend

The current Mermaid output is intentionally simple. A later frontend can consume `--format json` and render:

- expandable model tree;
- PP stage swimlanes;
- TP/CP/EP overlays;
- parameter and tensor shape panels;
- warnings and recommendations;
- exportable SVG/PNG/HTML demo pages.

### Later: per-layer tensor examples

The current report already includes basic tensor-shape examples such as token, embedding, per-CP-rank, and per-TP-head shapes. A future tensor walkthrough can extend the IR with layer-level nodes like:

```json
{
  "name": "decoder.layers.0.self_attention.qkv",
  "input_shape": ["micro_batch", "seq", "hidden"],
  "output_shape": ["micro_batch", "seq", "3 * hidden / TP"],
  "parallelism": {"tp_split": "column"}
}
```

This should be generated from the same `MegatronConfig` IR so the future frontend and tensor example engine stay consistent.

## Limitations

- This MVP statically parses scripts and does not execute Bash.
- Common scalar variables, arrays, arithmetic such as `WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))`, and quoted argument blocks are resolved statically. Complex sourced files, conditionals, loops, and command substitutions are not executed.
- Megatron-LM logs are not parsed yet.
- Megatron-LM's own argparse defaults are approximated only where explicitly implemented.

For highest accuracy in a later version, add a resolved-args JSON path that reuses Megatron-LM's own argument parser before visualization.
