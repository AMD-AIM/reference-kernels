import torch
from typing import TypeVar, TypedDict

# DeepSeek R1 MLA forward_absorb format:
#
# Input: (q, kv_buffer, qo_indptr, kv_indptr, config)
#   q:          (total_q, num_heads, qk_head_dim)       bfloat16
#   kv_buffer:  bf16:  (total_kv, 1, 576)               bfloat16
#               fp8:   (total_kv, 1, 576)               float8 (aiter_dtypes.fp8)
#               mxfp4: (total_kv, 1, 288)               fp4x2 (aiter_dtypes.fp4x2)
#   qo_indptr:  (batch_size + 1,)                        int32
#   kv_indptr:  (batch_size + 1,)                        int32
#   config:     dict with MLA parameters
#
# where qk_head_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
#
# Output: attention output tensor (total_q, num_heads, v_head_dim) bfloat16
#   where v_head_dim = kv_lora_rank = 512
#
# The kv_buffer stores the compressed KV representation:
#   - Full 576 dims used as keys (for Q@K^T score computation)
#   - First 512 dims (kv_lora_rank) used as values (for output computation)
#
# KV cache quantization:
#   - bf16:  no quantization, kv_scale=None
#   - fp8:   dynamic per-tensor quantization (sglang style), kv_scale is scalar float32
#   - mxfp4: aiter dynamic_mxfp4_quant (block-32, E2M1 values, E8M0 scales)
#            kv_scale is fp8_e8m0 tensor (total_kv, num_blocks)

input_t = TypeVar(
    "input_t",
    bound=tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict],
)
output_t = TypeVar("output_t", bound=torch.Tensor)


class TestSpec(TypedDict):
    batchsize: int
    qseqlen: int
    kvseqlen: int
    kvdtype: str   # "bf16", "fp8", or "mxfp4"
    seed: int
