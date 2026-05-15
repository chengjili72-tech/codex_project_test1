from megatron_viz.inputs.shell_parser import parse_megatron_args, parse_script_file


def test_parse_command_supports_boolean_key_value_and_equals_forms():
    parsed = parse_megatron_args(
        "torchrun pretrain_gpt.py --num-layers 24 --hidden-size=4096 --bf16 --no-sequence-parallel"
    )

    assert parsed["num_layers"] == 24
    assert parsed["hidden_size"] == 4096
    assert parsed["bf16"] is True
    assert parsed["sequence_parallel"] is False


def test_parse_script_file_with_line_continuations():
    parsed = parse_script_file("tests/fixtures/train_gpt.sh")

    assert parsed["world_size"] == 64
    assert parsed["tensor_model_parallel_size"] == 4
    assert parsed["pipeline_model_parallel_size"] == 2
    assert parsed["swiglu"] is True


def test_parse_complex_script_resolves_argument_blocks_variables_arrays_and_arithmetic():
    parsed = parse_script_file("tests/fixtures/deepseek_moe_npu.sh")

    assert parsed["world_size"] == 64
    assert parsed["nproc_per_node"] == 16
    assert parsed["nnodes"] == 4
    assert parsed["master_addr"] == "29.210.144.80"
    assert parsed["tensor_model_parallel_size"] == 2
    assert parsed["pipeline_model_parallel_size"] == 4
    assert parsed["expert_model_parallel_size"] == 16
    assert parsed["num_layers"] == 16
    assert parsed["seq_length"] == 4096
    assert parsed["micro_batch_size"] == 1
    assert parsed["global_batch_size"] == 64
    assert parsed["multi_latent_attention"] is True
    assert parsed["num_experts"] == 128
    assert parsed["moe_router_topk"] == 8
