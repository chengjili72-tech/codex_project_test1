from megatron_viz.inputs.loader import config_from_text
from megatron_viz.inputs.log_parser import parse_log_file, parse_megatron_log
from megatron_viz.renderers.compare_html import render_compare_html
from megatron_viz.reports.comparison import changed_diffs, compare_configs
from megatron_viz.server import parse_multipart_form


def test_parse_log_file_extracts_resolved_args():
    parsed = parse_log_file("tests/fixtures/megatron_train.log")

    assert parsed["world_size"] == 64
    assert parsed["data_parallel_size"] == 8
    assert parsed["num_layers"] == 16
    assert parsed["hidden_size"] == 7168
    assert parsed["tensor_model_parallel_size"] == 2
    assert parsed["expert_model_parallel_size"] == 16
    assert parsed["bf16"] is True


def test_loader_uses_log_parser_for_log_suffix():
    text = open("tests/fixtures/megatron_train.log", encoding="utf-8").read()
    config = config_from_text(text, filename="run.log")

    assert config.model.num_layers == 16
    assert config.training.transformer_impl == "transformer_engine"


def test_compare_configs_and_html_highlight_key_differences():
    left = config_from_text(open("tests/fixtures/train_gpt.sh", encoding="utf-8").read(), filename="train_gpt.sh")
    right = config_from_text(open("tests/fixtures/deepseek_moe_npu.sh", encoding="utf-8").read(), filename="deepseek_moe_npu.sh")

    diffs = compare_configs(left, right)
    changed_paths = {diff.path for diff in changed_diffs(left, right)}
    html = render_compare_html(left, right, left_name="base.sh", right_name="candidate.sh")

    assert any(diff.changed for diff in diffs)
    assert "model.hidden_size" in changed_paths
    assert "parallelism.expert_model_parallel_size" in changed_paths
    assert "Megatron-LM 配置对比" in html
    assert "tr class='changed'" in html
    assert "base.sh" in html
    assert "candidate.sh" in html


def test_parse_multipart_form_extracts_fields_and_files():
    boundary = "----test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="kind"\r\n\r\n'
        "script\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="train.sh"\r\n'
        "Content-Type: text/x-shellscript\r\n\r\n"
        "--num-layers 2 --hidden-size 128\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    fields, files = parse_multipart_form(body, f"multipart/form-data; boundary={boundary}")

    assert fields["kind"] == "script"
    assert files["file"].filename == "train.sh"
    assert b"--num-layers 2" in files["file"].content


def test_parse_log_with_embedded_launch_command():
    parsed = parse_megatron_log("python pretrain_gpt.py --num-layers 8 --hidden-size 1024\n")

    assert parsed["num_layers"] == 8
    assert parsed["hidden_size"] == 1024
