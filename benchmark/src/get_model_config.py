import torch
import itertools
from tests.utils import get_model_config


# TODO: get "OpenGVLab/InternVL3_5-8B", "deepseek-ai/DeepSeek-OCR" config failed, need to investigate
model_lists = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "openbmb/MiniCPM-V-4", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen2.5-VL-32B-Instruct", "deepseek-ai/DeepSeek-V2-Lite"]


def gen_cutlass_fused_moe_correctness_configs():
    mnk =[
        (1, 5120, 8192),
        (4, 5120, 8192),
        (16, 5120, 8192),
        (1024, 5120, 8192),
        (8192, 5120, 8192),
    ]
    experts = [16]
    topk = [1]
    dtype = [torch.float16, torch.bfloat16]
    w_dtype = [torch.float8_e5m2, torch.float8_e4m3fn, None]
    has_bias = [True, False]

    configs = list(
        itertools.product(mnk, experts, topk, dtype, w_dtype, has_bias))
    return configs


def gen_cutlass_fused_moe_perf_configs():
    configs = []
    topk = [1]
    dtype = [torch.float16, torch.bfloat16]
    w_dtype = [torch.float8_e5m2, torch.float8_e4m3fn, None]
    has_bias = [True, False]
    input_lens = [1, 4, 16, 1024, 8192]

    for model in model_lists:
        model_config = get_model_config(model, tp_size=1)
        if model_config["is_moe"] == False:
            continue

        hidden_size = model_config["hidden_size"]
        intermediate_size = model_config["intermediate_size"]
        mnk = list(zip(input_lens, [hidden_size] * len(input_lens), [intermediate_size] * len(input_lens)))

        configs += list(
            itertools.product(mnk, [model_config["moe_config"]["moe_top_k"]], topk, dtype, w_dtype, has_bias))
    configs = set(configs)  # remove duplicates
    def sort_key(x):
        (m, n, k), moe_topk, topk_, dtype_, w_dtype_, bias_ = x

        return (
            m, n, k,
            moe_topk,
            topk_,
            str(dtype_),
            str(w_dtype_),
            bias_
        )

    configs = sorted(configs, key=sort_key)
    return configs


def gen_cutlass_flash_attn_varlen_correctness_configs():
    # seq_lens = [[(1, 1328), (5, 18), (129, 463)]]
    num_seqs = [3]
    query_lens = ["1,5,129"]
    kv_lens = ["1328,18,463"]

    num_heads = [(4, 4), (8, 2), (10, 2), (16, 1)]
    head_size = [64, 128, 192, 256]
    block_size = [64, 128]
    window_size = [(-1, 127), (127, -1), (64, 64), (-1, -1)]
    dtype = [torch.float16, torch.bfloat16]
    soft_cap = [None]
    num_blocks = [32768, 16324, 2048]
    fa_versions = [2]
    q_dtype = [None]
    is_sink = [False, True]
    is_causal = [False, True]
    is_paged = [False, True]
    fp8_dtype = [torch.float8_e5m2, torch.float8_e4m3fn, None]

    configs = list(
        itertools.product(num_seqs, query_lens, kv_lens, num_heads, head_size, block_size, window_size, dtype, soft_cap, num_blocks, fa_versions, q_dtype, is_sink, is_causal, is_paged, fp8_dtype)
    )
    return configs


def gen_cutlass_flash_attn_varlen_perf_configs():
    num_seqs = [3]
    query_lens = ["1024,2048,2048"]
    kv_lens = ["1024,1024,2048"]

    block_size = [64, 128]
    window_size = [(-1, -1)]
    dtype = [torch.float16, torch.bfloat16]
    soft_cap = [None]
    num_blocks = [16324]
    fa_versions = [2]
    q_dtype = [None]
    is_sink = [False, True]
    is_causal = [False, True]
    is_paged = [False, True]
    fp8_dtype = [torch.float8_e5m2, torch.float8_e4m3fn, None]

    configs = []
    num_heads = []
    head_size = []
    for model in model_lists:
        model_config = get_model_config(model, tp_size=1)
        head_size = [model_config["head_dim"]]
        num_heads = [(model_config["num_attention_heads"], model_config["num_key_value_heads"])]

        configs += list(
            itertools.product(num_seqs, query_lens, kv_lens, num_heads, head_size, block_size, window_size, dtype, soft_cap, num_blocks, fa_versions, q_dtype, is_sink, is_causal, is_paged, fp8_dtype)
        )
    configs = set(configs)  # remove duplicates
    def sort_key(x):
        (num_seq, query_len, kv_len, num_head, head_size, block_size, window_size, dtype_, soft_cap, num_blocks, fa_version, q_dtype, is_sink, is_causal, is_paged, fp8_dtype) = x

        return (
            num_seq,
            query_len,
            kv_len,
            num_head,
            head_size,
            block_size,
            window_size,
            str(dtype_),
            soft_cap if soft_cap is not None else -1,
            num_blocks,
            fa_version,
            str(q_dtype),
            is_sink,
            is_causal,
            is_paged,
            str(fp8_dtype)
        )
    configs = sorted(configs, key=sort_key)
    return configs