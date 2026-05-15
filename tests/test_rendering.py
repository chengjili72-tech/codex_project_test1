from megatron_viz.inputs.shell_parser import parse_script_file
from megatron_viz.ir.normalizer import normalize_args
from megatron_viz.renderers.markdown import render_markdown, render_mermaid


def test_normalize_infers_parallelism_and_defaults():
    config = normalize_args(parse_script_file("tests/fixtures/train_gpt.sh"))

    assert config.parallelism.data_parallel_size == 8
    assert config.model.ffn_hidden_size == 16384
    assert config.model.activation == "SwiGLU"


def test_markdown_contains_pipeline_diagnostics_and_tensor_examples():
    config = normalize_args(parse_script_file("tests/fixtures/train_gpt.sh"))
    markdown = render_markdown(config)

    assert "PP stage 0: Embedding + Layers 0-11" in markdown
    assert "**OK**: world_size=64 is divisible" in markdown
    assert "Tensor Shape Examples" in markdown
    assert "embedding output" in markdown


def test_mermaid_renders_pipeline_nodes():
    config = normalize_args(parse_script_file("tests/fixtures/train_gpt.sh"))
    mermaid = render_mermaid(config)

    assert "flowchart LR" in mermaid
    assert "PP0[PP0: Layers 0-11]" in mermaid
    assert "PP1[PP1: Layers 12-23]" in mermaid
