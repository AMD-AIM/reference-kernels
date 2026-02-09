# MLA (Multi-head Latent Attention) Decode Kernel

## Description

Implement a custom MLA attention decode kernel optimized for MI355X.

This is the **inner attention kernel** from DeepSeek R1's `forward_absorb` MLA path.
The absorbed query and compressed KV cache are provided directly — you implement the
attention computation with variable-length batching.

The reference uses **aiter MLA a8w8 decode kernel** (`mla_decode_fwd`, fp8 Q + fp8 KV,
persistent mode), which is ~2-3x faster than bf16 on MI355X with negligible accuracy loss.

Source: [sglang/scripts/radix_attention_standalone.py](https://github.com/sgl-project/sglang)

## DeepSeek R1 Forward-Absorb MLA Config

| Parameter | Value | Notes |
|---|---|---|
| num_heads | 16 | Query heads (after TP split) |
| num_kv_heads | 1 | Single shared latent KV head |
| kv_lora_rank | 512 | Latent dimension |
| qk_rope_head_dim | 64 | RoPE embedding dimension |
| qk_head_dim | 576 | kv_lora_rank + qk_rope_head_dim (absorbed q/k dim) |
| v_head_dim | 512 | = kv_lora_rank (output dim) |
| sm_scale | 1/sqrt(576) | |
| q dtype | bfloat16 | Always bf16 |
| kv dtype | bf16 / fp8 / mxfp4 | All three provided simultaneously |
| mode | decode | q_seq_len=1, kv_seq_len up to 8k |

## KV Buffer Format (forward_absorb)

The compressed KV buffer has `qk_head_dim=576` dimensions:
- **Full 576 dims** are used as **keys** (for Q@K^T score computation)
- **First 512 dims** (kv_lora_rank) are used as **values** (for output computation)

## KV Cache Quantization

| dtype | kv_buffer | kv_scale | Quantization | Bandwidth |
|---|---|---|---|---|
| bf16 | bfloat16 `(total_kv, 1, 576)` | None | No quantization | 1x |
| fp8 | fp8 `(total_kv, 1, 576)` | scalar float32 | Dynamic per-tensor (sglang `scaled_fp8_quant`) | 2x savings |
| mxfp4 | fp4x2 `(total_kv, 1, 288)` | fp8_e8m0 `(total_kv, N_blocks)` | Block-32 MXFP4 (aiter `dynamic_mxfp4_quant`) | 4x savings |

### FP8 quantization (sglang `scaled_fp8_quant`)

- **Granularity**: per-tensor
- **Scale**: `kv_scale = max(abs(kv_bf16)) / fp8_max`
- **Quantize**: `kv_fp8 = (kv_bf16 / kv_scale).clamp(...).to(fp8)`
- **Dequantize**: `kv_bf16 ≈ kv_fp8.to(bf16) * kv_scale`
- **kv_scale**: scalar float32 tensor

### MXFP4 quantization (aiter `dynamic_mxfp4_quant`)

- **Granularity**: per-block of 32 elements
- **FP4 format**: E2M1 — values `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`, max = 6.0
- **Scale format**: E8M0 — exponent-only scale stored in `aiter.dtypes.fp8_e8m0`
- **Packing**: 2 FP4 values packed per byte (low nibble = even index, high nibble = odd index)
- **kv_buffer**: `(total_kv, 1, 288)` in `aiter.dtypes.fp4x2` — packed FP4 data
- **kv_scale**: `(total_kv, N_blocks)` in `aiter.dtypes.fp8_e8m0` — per-block E8M0 scale factors
- **Dequantize**: `aiter.utility.fp4_utils.mxfp4_to_f32` + `e8m0_to_f32` for block-wise scaling
- **Reference approach**: dequantize MXFP4 → bf16, then run bf16 MLA kernel
- **Optimization opportunity**: fuse dequantization with attention to avoid materializing the full bf16 buffer

### aiter dtype reference

| Logical type | aiter dtype | PyTorch native (if available) | Fallback |
|---|---|---|---|
| fp4x2 | `aiter.dtypes.fp4x2` | `torch.float4_e2m1fn_x2` | `torch.uint8` |
| fp8_e8m0 | `aiter.dtypes.fp8_e8m0` | `torch.float8_e8m0fnu` | `torch.uint8` |
| fp8 | `aiter.dtypes.fp8` | `torch.float8_e4m3fnuz` (gfx942) / `torch.float8_e4m3fn` (gfx950+) | `torch.uint8` |

## Input

A tuple `(q, kv_data, qo_indptr, kv_indptr, config)`:

```
q:          (total_q, 16, 576)     bfloat16  — absorbed queries
kv_data:    dict with three KV cache formats (see below)
qo_indptr:  (batch_size + 1,)      int32     — query segment pointers
kv_indptr:  (batch_size + 1,)      int32     — KV segment pointers
config:     dict                              — MLA parameters
```

### kv_data dict

All three KV cache formats are provided simultaneously. Each entry is either a
`Tensor` (bf16) or a `(Tensor, Tensor)` tuple (quantized buffer + scale):

```python
kv_data = {
    "bf16":  kv_buffer_bf16,                        # Tensor (total_kv, 1, 576) bfloat16
    "fp8":   (kv_buffer_fp8, kv_scale_fp8),         # (fp8 Tensor, scalar float32)
    "mxfp4": (kv_buffer_mxfp4, kv_scale_mxfp4),    # (fp4x2 Tensor, fp8_e8m0 Tensor)
}
```

The `config` dict includes: `batch_size`, `num_heads`, `num_kv_heads`, `qk_head_dim`,
`kv_lora_rank`, `qk_rope_head_dim`, `v_head_dim`, `q_seq_len`, `kv_seq_len`, `sm_scale`.

## Output

```
attention_output: (total_q, 16, 512) bfloat16
```

## Optimization Opportunities

1. **MQA pattern**: 1 KV head shared across 16 query heads — minimize redundant KV loads
2. **Decode workload**: q_seq_len=1 vs large kv_seq_len (up to 8k) — memory-bound
3. **Quantized KV cache**: We have provided 3 dtypes of kv cache returned by `generate_input`. fp8/mxfp4 reduces memory bandwidth (the key bottleneck for decode), optional strategy you might use:

   **Strategy A — Fuse dequantization with attention (keep Q in bf16):**
   The reference dequantizes fp8/mxfp4 KV cache to bf16 in a separate pass, then runs
   bf16 MLA attention. This materializes a full bf16 KV buffer in HBM, wasting bandwidth.
   Instead, fuse the dequant into the attention inner loop: load quantized KV tiles from
   HBM, dequantize in registers/LDS to fp16/bf16, and immediately compute QK^T and
   softmax·V — never writing the decompressed KV back to HBM. This eliminates the extra
   read/write of the bf16 intermediate buffer, roughly halving the memory traffic for fp8
   and quartering it for mxfp4 compared to the naive dequant-then-attend approach.

   **Strategy B — Quantize Q to match KV precision (full low-precision compute):**
   Since Q is bf16 and KV is fp8/mxfp4, the QK^T matmul still requires mixed-precision.
   To fully exploit low-precision matrix units (e.g. fp8 MFMA on MI355X), dynamically
   quantize Q from bf16 → fp8 (per-tensor or per-head-row scaling), then compute QK^T
   entirely in fp8×fp8 using MFMA instructions. For mxfp4 KV, quantize Q to mxfp4 as
   well to use fp4×fp4 MFMA. The softmax is still done in fp32 for numerical stability,
   and V accumulation can use fp8×fp8 → fp32 or bf16 for the output. This trades a small
   amount of accuracy (from quantizing Q) for significantly higher throughput on the
   matrix units, which matters when batch sizes are large enough to become compute-bound.
4. **Variable-length batching**: indptr-based segmented attention across batch
5. **Split K/V from buffer**: full 576 dims for keys, first 512 for values

## Benchmark Cases

All three KV formats (bf16, fp8, mxfp4) are provided in every test case.

| batch_size | q_seq_len | kv_seq_len |
|---|---|---|
| 4 | 1 | 1024 |
| 4 | 1 | 8192 |

| 64 | 1 | 1024 |
| 64 | 1 | 8192 |
| 256 | 1 | 1024 |
| 256 | 1 | 8192 |
