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
from megatron_viz.renderers.html import render_html


def test_html_renderer_contains_browser_visualization_for_complex_script():
    config = normalize_args(parse_script_file("tests/fixtures/deepseek_moe_npu.sh"))
    html = render_html(config)

    assert "<!doctype html>" in html
    assert "Megatron-LM 模型结构可视化" in html
    assert "PP Stage 0" in html
    assert "MLA / Multi-Latent Attention" in html
    assert "MoE Experts" in html
    assert "window.MEGATRON_CONFIG" in html
