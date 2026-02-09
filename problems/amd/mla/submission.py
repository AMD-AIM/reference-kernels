"""
MLA (Multi-head Latent Attention) decode kernel — submission template.

Implement custom_kernel() to beat the aiter a8w8 reference (fp8 Q + fp8 KV).

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

Input tuple:
  q:          (total_q, 16, 576)       bfloat16 — absorbed query
  kv_data:    dict with three KV cache formats:
    kv_data["bf16"]  — Tensor (total_kv, 1, 576) bfloat16
    kv_data["fp8"]   — (Tensor, Tensor): kv_buffer fp8 (total_kv,1,576) + scalar scale
    kv_data["mxfp4"] — (Tensor, Tensor): kv_buffer fp4x2 (total_kv,1,288) + fp8_e8m0 scale
  qo_indptr:  (batch_size + 1,)        int32    — query segment pointers
  kv_indptr:  (batch_size + 1,)        int32    — KV segment pointers
  config:     dict with MLA parameters

Output:
  attention output: (total_q, 16, 512) bfloat16

The reference uses aiter's a8w8 persistent MLA kernel (fp8 Q + fp8 KV),
which is ~2-3x faster than bf16. To beat it, consider:
  1. Use mxfp4 KV cache for even lower memory bandwidth
     - Fuse dequantization with attention to avoid bf16 materialization
  2. Custom kernel with tighter memory access patterns
  3. MQA: 1 KV head shared across 16 query heads — minimize redundant memory loads
  4. Variable-length batching: indptr-based segmented attention
  5. Split K/V from buffer: full 576 dims for keys, first 512 dims for values
"""

import torch
import torch.nn.functional as F
from task import input_t, output_t


# ---------------------------------------------------------------------------
# Naive torch attention (baseline — participants should optimize this)
# ---------------------------------------------------------------------------

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    num_heads = config["num_heads"]
    kv_lora_rank = config["kv_lora_rank"]
    sm_scale = config["sm_scale"]

    # This naive baseline uses bf16 KV directly.
    # For better performance, use kv_data["fp8"] or kv_data["mxfp4"]
    # which are (kv_buffer, kv_scale) tuples. See docstring for optimization opportunities.
    kv_buffer_bf16 = kv_data["bf16"]

    batch_size = qo_indptr.shape[0] - 1
    out_list = []

    for i in range(batch_size):
        q_s, q_e = int(qo_indptr[i].item()), int(qo_indptr[i + 1].item())
        kv_s, kv_e = int(kv_indptr[i].item()), int(kv_indptr[i + 1].item())

        qi = q[q_s:q_e]                       # (seq_q, nhead, 576)
        kvc = kv_buffer_bf16[kv_s:kv_e, 0]    # (seq_kv, 576)  squeeze kv_heads dim

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
