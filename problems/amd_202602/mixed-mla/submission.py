"""
MLA (Multi-head Latent Attention) decode kernel — submission template.

Implement custom_kernel() to beat the aiter reference.

DeepSeek R1 forward_absorb MLA config:
  num_heads        = 16     (query heads, after TP split)
  num_kv_heads     = 1      (shared latent KV head)
  kv_lora_rank     = 512    (latent dim)
  qk_rope_head_dim = 64     (RoPE dim)
  qk_head_dim      = 576    (kv_lora_rank + qk_rope_head_dim, absorbed q/k dim)
  v_head_dim       = 512    (= kv_lora_rank, output dim)
  sm_scale         = 1/sqrt(576)

KV buffer format (forward_absorb):
  - Full 576 dims used as keys (for Q@K^T score computation)
  - First 512 dims (kv_lora_rank) used as values (for output computation)

KV cache dtype: bf16, fp8, or mxfp4 (specified in config["kv_dtype"])
  - bf16:  kv_buffer is bfloat16 (total_kv, 1, 576), kv_scale=None
  - fp8:   kv_buffer is fp8 (total_kv, 1, 576), kv_scale is a scalar float32 tensor
           → dequantize: kv_buffer.to(bf16) * kv_scale
  - mxfp4: kv_buffer is fp4x2 (total_kv, 1, 288), kv_scale is fp8_e8m0 (total_kv, N_blocks)
           Block size = 32. Uses aiter's native MXFP4 format.
           → dequantize to bf16 first, then run bf16 MLA
           → dequantize: aiter.utility.fp4_utils.{mxfp4_to_f32, e8m0_to_f32}
           → optimization: fuse dequant + attention to skip bf16 materialization

Input tuple:
  q:          (total_q, 16, 576)        bfloat16
  kv_buffer:  see above per dtype
  qo_indptr:  (batch_size + 1,)         int32
  kv_indptr:  (batch_size + 1,)         int32
  config:     dict with MLA parameters

Output:
  attention output: (total_q, 16, 512)  bfloat16
"""

import torch
import torch.nn.functional as F
from task import input_t, output_t

from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32


# ---------------------------------------------------------------------------
# KV buffer dequantization
# ---------------------------------------------------------------------------

def dequantize_kv_buffer(kv_buffer, config):
    """Dequantize kv_buffer to bfloat16 based on config['kv_dtype']."""
    kv_dtype = config.get("kv_dtype", "bf16")
    kv_scale = config.get("kv_scale")

    if kv_dtype == "bf16":
        return kv_buffer  # already bf16
    elif kv_dtype == "fp8":
        # Per-tensor FP8: dequantize = cast_to_bf16 * scale
        return kv_buffer.to(torch.bfloat16) * kv_scale
    elif kv_dtype == "mxfp4":
        # MXFP4: unpack fp4x2 → f32 values, then apply E8M0 block scales
        # Note: scale_e8m0 may be padded by dynamic_mxfp4_quant; trim to actual dims
        B, M, N_half = kv_buffer.shape
        N = N_half * 2
        num_rows = B * M
        block_size = 32
        num_blocks = N // block_size  # actual blocks (e.g. 576/32 = 18)

        # Unpack FP4 to float32 (handles both fp4x2 and uint8 dtypes)
        kv_2d = kv_buffer.reshape(num_rows, N_half)
        float_vals = mxfp4_to_f32(kv_2d)  # (num_rows, N) float32

        # Convert E8M0 scales to float32 and trim padded dimensions
        scale_f32 = e8m0_to_f32(kv_scale)  # (padded_rows, padded_blocks)
        scale_f32 = scale_f32[:num_rows, :num_blocks]  # (num_rows, num_blocks)

        float_blocked = float_vals.view(num_rows, num_blocks, block_size)
        scaled = float_blocked * scale_f32.unsqueeze(-1)

        return scaled.view(B, M, N).to(torch.bfloat16)
    else:
        raise ValueError(f"Unknown kv_dtype: {kv_dtype}")


# ---------------------------------------------------------------------------
# Naive torch attention (baseline — participants should optimize this)
# ---------------------------------------------------------------------------

def custom_kernel(data: input_t) -> output_t:
    q, kv_buffer, qo_indptr, kv_indptr, config = data

    num_heads = config["num_heads"]
    kv_lora_rank = config["kv_lora_rank"]
    qk_rope_head_dim = config["qk_rope_head_dim"]
    sm_scale = config["sm_scale"]

    # Dequantize kv_buffer to bf16 if needed
    # TODO: This is a naive implementation. We should use a more efficient implementation.
    # TODO: Try to fuse the dequantization and the attention computation.
    # TODO: Try to implement the attention computation with lower precision. (e.g. fp8, mxfp4)
    kv_bf16 = dequantize_kv_buffer(kv_buffer, config)

    batch_size = qo_indptr.shape[0] - 1
    out_list = []

    for i in range(batch_size):
        q_s, q_e = int(qo_indptr[i].item()), int(qo_indptr[i + 1].item())
        kv_s, kv_e = int(kv_indptr[i].item()), int(kv_indptr[i + 1].item())

        qi = q[q_s:q_e]               # (seq_q, nhead, 576)
        kvc = kv_bf16[kv_s:kv_e, 0]   # (seq_kv, 576)  squeeze kv_heads dim

        # Key: full 576 dims; Value: first 512 dims (kv_lora_rank)
        ki = kvc                       # (seq_kv, 576) — broadcast over heads
        vi = kvc[:, :kv_lora_rank]     # (seq_kv, 512)

        # Attention: (nhead, seq_q, 576) @ (576, seq_kv) → (nhead, seq_q, seq_kv)
        qi_t = qi.float().permute(1, 0, 2)  # (nhead, seq_q, 576)
        scores = torch.matmul(qi_t * sm_scale, ki.float().T)  # (nhead, seq_q, seq_kv)

        scores = F.softmax(scores, dim=-1)

        # Output: (nhead, seq_q, seq_kv) @ (seq_kv, 512) → (nhead, seq_q, 512)
        oi = torch.matmul(scores, vi.float())  # (nhead, seq_q, 512)
        oi = oi.permute(1, 0, 2)               # (seq_q, nhead, 512)
        out_list.append(oi.to(torch.bfloat16))

    return torch.cat(out_list, dim=0)
