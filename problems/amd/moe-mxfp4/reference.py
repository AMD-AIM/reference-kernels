from utils import make_match_reference
from task import input_t, output_t
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from aiter.utility import fp4_utils
from aiter.ops.shuffle import shuffle_weight


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
MXFP4_BLOCK_SIZE = 32
PAD_ALIGN = 256


def _pad_to(x: int, align: int) -> int:
    return (x + align - 1) // align * align


# ──────────────────────────────────────────────────────────────────────
# ref_kernel: calls AITER fused_moe with MXFP4 quantized weights
# ──────────────────────────────────────────────────────────────────────
def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation using AITER's fused_moe kernel with MXFP4 quantized weights.

    Matches the Mxfp4MoEMethod.apply() codepath in atom/model_ops/moe.py.

    Input data tuple:
        hidden_states:                [M, d_hidden]                           bf16
        gate_up_weight:               [E, 2*d_expert_pad, d_hidden_pad//2]    fp4x2  (raw, before shuffle)
        down_weight:                  [E, d_hidden_pad, d_expert_pad//2]      fp4x2  (raw, before shuffle)
        gate_up_weight_scale:         [E, 2*d_expert_pad, scale_K]            e8m0   (raw, before shuffle)
        down_weight_scale:            [E, d_hidden_pad, scale_K]              e8m0   (raw, before shuffle)
        gate_up_weight_shuffled:      [E, 2*d_expert_pad, d_hidden_pad//2]    fp4x2  (shuffled for CK kernel)
        down_weight_shuffled:         [E, d_hidden_pad, d_expert_pad//2]      fp4x2  (shuffled for CK kernel)
        gate_up_weight_scale_shuffled:[padded, flat]                          e8m0   (shuffled for CK kernel)
        down_weight_scale_shuffled:   [padded, flat]                          e8m0   (shuffled for CK kernel)
        topk_weights:                 [M, top_k]                              float32
        topk_ids:                     [M, top_k]                              int32
        config:                       dict

    Returns:
        output: [M, d_hidden] bf16
    """
    (
        hidden_states,
        gate_up_weight,
        down_weight,
        gate_up_weight_scale,
        down_weight_scale,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
        config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    output = fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,  # MXFP4 uses per_1x32 block scaling
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )

    return output


# ──────────────────────────────────────────────────────────────────────
# generate_input: produce all tensors needed by ref_kernel
#
# Models DeepSeek-R1 MoE layer shapes:
#   - d_hidden = 7168
#   - d_expert = moe_intermediate_size (full=2048, or TP-split)
#   - E = local number of experts on this GPU
#   - top_k = 8
#
# ──────────────────────────────────────────────────────────────────────
def generate_input(
    dhidden: int,
    dexpert: int,
    nroutedexperts: int,
    nexpertspertoken: int,
    bs: int,
    seed: int,
) -> input_t:
    d_hidden = dhidden
    d_expert = dexpert
    n_routed_experts = nroutedexperts
    top_k = nexpertspertoken
    M = bs  # number of tokens

    # Padded dimensions (AITER MXFP4 requires 256-alignment)
    d_hidden_pad = _pad_to(d_hidden, PAD_ALIGN)
    d_expert_pad = _pad_to(d_expert, PAD_ALIGN)

    config = {
        "d_hidden": d_hidden,
        "d_expert": d_expert,
        "d_hidden_pad": d_hidden_pad,
        "d_expert_pad": d_expert_pad,
        "n_routed_experts": n_routed_experts,
        "n_experts_per_token": top_k,
        "bs": M,
    }

    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    # ── hidden_states [M, d_hidden] ──
    hidden_states = torch.randn(
        (M, d_hidden), device='cuda', dtype=torch.bfloat16, generator=gen,
    )

    # ── Router: softmax top-k ──
    router_weight = torch.randn(
        (n_routed_experts, d_hidden), device='cuda', dtype=torch.bfloat16, generator=gen,
    ) / math.sqrt(d_hidden)
    router_logits = F.linear(hidden_states, router_weight)  # [M, E]
    scores = router_logits.softmax(dim=-1)
    topk_weights, topk_ids = torch.topk(
        scores, k=top_k, dim=-1, sorted=False
    )
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    # ── Expert weights: bf16 -> quantize to MXFP4 ──
    # gate_up = fused [gate_proj; up_proj] per expert: [2*d_expert_pad, d_hidden_pad]
    # down    = down_proj                  per expert: [d_hidden_pad, d_expert_pad]
    gate_up_q_list, gate_up_s_list = [], []
    down_q_list, down_s_list = [], []

    for _ in range(n_routed_experts):
        # gate_proj + up_proj -> fused [2*d_expert_pad, d_hidden_pad]
        gate_bf16 = torch.randn(
            (d_expert_pad, d_hidden_pad), device='cuda', dtype=torch.bfloat16, generator=gen
        ) / math.sqrt(d_hidden)
        up_bf16 = torch.randn(
            (d_expert_pad, d_hidden_pad), device='cuda', dtype=torch.bfloat16, generator=gen
        ) / math.sqrt(d_hidden)
        gate_up_bf16 = torch.cat([gate_bf16, up_bf16], dim=0)

        # down_proj -> [d_hidden_pad, d_expert_pad]
        down_bf16 = torch.randn(
            (d_hidden_pad, d_expert_pad), device='cuda', dtype=torch.bfloat16, generator=gen
        ) / math.sqrt(d_expert)

        # Quantize to MXFP4
        gu_q, gu_s = fp4_utils.dynamic_mxfp4_quant(gate_up_bf16)
        dn_q, dn_s = fp4_utils.dynamic_mxfp4_quant(down_bf16)

        gate_up_q_list.append(gu_q)
        gate_up_s_list.append(gu_s)
        down_q_list.append(dn_q)
        down_s_list.append(dn_s)

    # Stack into [E, ...] tensors — raw (before shuffle)
    E = n_routed_experts
    gate_up_weight = torch.stack(gate_up_q_list)          # [E, 2*d_expert_pad, d_hidden_pad//2]  fp4x2
    gate_up_weight_scale = torch.stack(gate_up_s_list)    # [E, 2*d_expert_pad, scale_K]          e8m0
    down_weight = torch.stack(down_q_list)                # [E, d_hidden_pad, d_expert_pad//2]    fp4x2
    down_weight_scale = torch.stack(down_s_list)          # [E, d_hidden_pad, scale_K]            e8m0

    # Shuffle for AITER CK kernel
    gate_up_weight_shuffled = shuffle_weight(gate_up_weight.clone())
    down_weight_shuffled = shuffle_weight(down_weight.clone())
    gate_up_weight_scale_shuffled = fp4_utils.e8m0_shuffle(gate_up_weight_scale.reshape(E, -1))
    down_weight_scale_shuffled = fp4_utils.e8m0_shuffle(down_weight_scale.reshape(E, -1))

    return (
        hidden_states,                  # [M, d_hidden]                          bf16
        gate_up_weight,                 # [E, 2*d_expert_pad, d_hidden_pad//2]   fp4x2  (raw)
        down_weight,                    # [E, d_hidden_pad, d_expert_pad//2]     fp4x2  (raw)
        gate_up_weight_scale,           # [E, 2*d_expert_pad, scale_K]           e8m0   (raw)
        down_weight_scale,              # [E, d_hidden_pad, scale_K]             e8m0   (raw)
        gate_up_weight_shuffled,        # [E, 2*d_expert_pad, d_hidden_pad//2]   fp4x2  (shuffled)
        down_weight_shuffled,           # [E, d_hidden_pad, d_expert_pad//2]     fp4x2  (shuffled)
        gate_up_weight_scale_shuffled,  # [padded, flat]                         e8m0   (shuffled)
        down_weight_scale_shuffled,     # [padded, flat]                         e8m0   (shuffled)
        topk_weights,                   # [M, top_k]                             float32
        topk_ids,                       # [M, top_k]                             int32
        config,
    )


check_implementation = make_match_reference(ref_kernel, rtol=1e-2, atol=1e-2)
