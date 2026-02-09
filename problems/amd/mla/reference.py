"""
Reference implementation for MLA (Multi-head Latent Attention) decode kernel.

Uses aiter MLA kernels (mla_decode_fwd / mla_prefill_fwd) as the reference.
DeepSeek R1 forward_absorb MLA: absorbed q (576), compressed kv_buffer (576),
output v_head_dim = kv_lora_rank = 512.

KV cache dtype options:
  - bf16:  bfloat16, no scale — directly used by bf16 MLA kernel
  - fp8:   dynamic per-tensor quantization (sglang scaled_fp8_quant style),
           scalar kv_scale — passed to bf16-Q + fp8-KV persistent MLA kernel
  - mxfp4: block-32 MXFP4 quantization (aiter dynamic_mxfp4_quant),
           kv_buffer in fp4x2 dtype + kv_scale in fp8_e8m0 dtype
           → dequantized to bf16 first, then bf16 MLA kernel is used

Decode uses persistent mode; prefill uses non-persistent mode.
"""

import torch
import torch.nn.functional as F
from task import input_t, output_t
from utils import match_reference

from aiter.mla import mla_decode_fwd, mla_prefill_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.utility.fp4_utils import (
    dynamic_mxfp4_quant,
    mxfp4_to_f32,
    e8m0_to_f32,
)

# ---------------------------------------------------------------------------
# DeepSeek R1 latent MQA constants (forward_absorb path)
# https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/main/config.json
# ---------------------------------------------------------------------------
NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM   # 576
V_HEAD_DIM = KV_LORA_RANK                        # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

PAGE_SIZE = 1
NUM_KV_SPLITS = 32

# FP8 dtype (platform-specific via aiter)
FP8_DTYPE = aiter_dtypes.fp8

# Tolerance per kv dtype
TOLERANCES = {
    "bf16":  {"rtol": 2e-02, "atol": 8e-03},
    "fp8":   {"rtol": 5e-02, "atol": 1e-02},
    "mxfp4": {"rtol": 1e-01, "atol": 5e-02},
}


# ---------------------------------------------------------------------------
# FP8 quantization (sglang style: dynamic per-tensor)
# ---------------------------------------------------------------------------

def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamic per-tensor FP8 quantization (following sglang scaled_fp8_quant).

    Args:
        tensor: bf16 tensor to quantize

    Returns:
        (fp8_tensor, scale) where scale is a scalar float32 tensor.
        Dequantize: fp8_tensor.to(bf16) * scale
    """
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


# ---------------------------------------------------------------------------
# MXFP4 quantization (aiter native: block-32, fp4x2 + fp8_e8m0 dtypes)
# Uses aiter.utility.fp4_utils.dynamic_mxfp4_quant
# ---------------------------------------------------------------------------

def quantize_mxfp4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MXFP4 block-wise quantization using aiter's dynamic_mxfp4_quant.

    Block size = 32. Each block gets an E8M0 scale factor.
    Two FP4 E2M1 values are packed per byte.

    Args:
        tensor: bf16 tensor of shape [B, M, N] (N must be divisible by 32)

    Returns:
        (fp4_data, scale_e8m0)
        - fp4_data:   shape [B, M, N//2] in aiter_dtypes.fp4x2
        - scale_e8m0: shape [B*M, ceil(N/32)] padded, in aiter_dtypes.fp8_e8m0
    """
    orig_shape = tensor.shape  # (B, M, N)
    B, M, N = orig_shape

    # dynamic_mxfp4_quant expects 2D: (B*M, N)
    tensor_2d = tensor.reshape(B * M, N)
    fp4_data_2d, scale_e8m0 = dynamic_mxfp4_quant(tensor_2d)

    # Reshape fp4_data back to 3D: (B, M, N//2)
    fp4_data = fp4_data_2d.view(B, M, N // 2)

    return fp4_data, scale_e8m0


def dequantize_mxfp4(
    fp4_data: torch.Tensor,
    scale_e8m0: torch.Tensor,
    orig_shape: tuple,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize MXFP4 tensor using aiter utilities.

    Note: dynamic_mxfp4_quant may pad both row and block dimensions in scale_e8m0.
    We trim scales to match the actual data dimensions.

    Args:
        fp4_data:   packed FP4 data, shape [B, M, N//2] in fp4x2 or uint8
        scale_e8m0: E8M0 block scale factors (possibly padded) in fp8_e8m0
        orig_shape: original (B, M, N) for reshaping
        dtype:      output dtype

    Returns:
        Dequantized tensor of shape orig_shape.
    """
    B, M, N = orig_shape
    num_rows = B * M
    block_size = 32
    num_blocks = N // block_size  # actual blocks needed (e.g. 576/32 = 18)

    # Unpack FP4 to float32: mxfp4_to_f32 expects (..., N//2) -> (..., N)
    fp4_data_2d = fp4_data.reshape(num_rows, N // 2)
    float_vals = mxfp4_to_f32(fp4_data_2d)  # (num_rows, N)

    # Convert E8M0 scales to float32 and trim padded dimensions
    scale_f32 = e8m0_to_f32(scale_e8m0)  # (padded_rows, padded_blocks)
    scale_f32 = scale_f32[:num_rows, :num_blocks]  # (num_rows, num_blocks)

    # Apply block scales
    float_vals_blocked = float_vals.view(num_rows, num_blocks, block_size)
    scaled = float_vals_blocked * scale_f32.unsqueeze(-1)

    return scaled.view(B, M, N).to(dtype)


# ---------------------------------------------------------------------------
# Persistent mode metadata helpers
# ---------------------------------------------------------------------------

def _make_mla_decode_metadata(
    batch_size: int,
    max_q_len: int,
    nhead: int,
    nhead_kv: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_kv_splits: int = NUM_KV_SPLITS,
):
    """Allocate and populate work buffers for persistent mla_decode_fwd."""
    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, nhead, q_dtype, kv_dtype,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = work

    # Populate the metadata buffers
    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        nhead // nhead_kv,   # num_heads_per_head_k
        nhead_kv,            # num_heads_k
        True,                # is_causal
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=False,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
    )

    return {
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }


# ---------------------------------------------------------------------------
# Aiter reference kernel
# ---------------------------------------------------------------------------

def _aiter_mla_attention(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """
    MLA attention using aiter kernels (forward_absorb style).

    Decode: persistent mode with get_mla_metadata_v1.
      - bf16:  bf16 MLA kernel directly
      - fp8:   bf16-Q + fp8-KV kernel (a16w8), kv_scale passed through
      - mxfp4: dequantize to bf16 first, then bf16 MLA kernel

    Prefill: non-persistent mode (bf16 only).

    q:          (total_q, num_heads, 576)  bf16
    kv_buffer:  bf16:  (total_kv, 1, 576)  bfloat16
                fp8:   (total_kv, 1, 576)  fp8 dtype
                mxfp4: (total_kv, 1, 288)  fp4x2 dtype (packed 2 fp4 per byte)
    """
    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    kv_dtype = config.get("kv_dtype", "bf16")
    kv_scale = config.get("kv_scale")

    # For mxfp4: dequantize to bf16 first, then use bf16 MLA kernel
    if kv_dtype == "mxfp4":
        kv_buffer = dequantize_mxfp4(
            kv_buffer, kv_scale,
            orig_shape=(kv_buffer.shape[0], kv_buffer.shape[1], dq),
        )
        kv_scale = None
        actual_kv_torch_dtype = torch.bfloat16
    elif kv_dtype == "fp8":
        actual_kv_torch_dtype = FP8_DTYPE
    else:
        actual_kv_torch_dtype = torch.bfloat16

    total_kv_len = int(kv_indptr[-1].item())
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")

    # Reshape kv_buffer to 4D for aiter: (total_kv, page_size=1, nhead_kv=1, dim)
    kv_buffer_4d = kv_buffer.view(kv_buffer.shape[0], PAGE_SIZE, nkv, kv_buffer.shape[-1])

    is_decode = q_seq_len <= 4

    if is_decode:
        max_q_len = q_seq_len
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

        # Build persistent-mode metadata
        meta = _make_mla_decode_metadata(
            batch_size, max_q_len, nq, nkv,
            q.dtype, actual_kv_torch_dtype,
            qo_indptr, kv_indptr, kv_last_page_len,
            num_kv_splits=NUM_KV_SPLITS,
        )

        o = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
        mla_decode_fwd(
            q.view(-1, nq, dq),
            kv_buffer_4d,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            max_q_len,
            page_size=PAGE_SIZE,
            nhead_kv=nkv,
            sm_scale=SM_SCALE,
            logit_cap=0.0,
            num_kv_splits=NUM_KV_SPLITS,
            kv_scale=kv_scale,
            intra_batch_mode=True,
            **meta,
        )
        return o
    else:
        # Prefill: non-persistent bf16 mode (mxfp4/fp8 already dequantized above)
        max_q_len = int((qo_indptr[1:] - qo_indptr[:-1]).max().item())
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

        o = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
        mla_prefill_fwd(
            q.view(-1, nq, dq),
            kv_buffer_4d,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            max_q_len,
            sm_scale=SM_SCALE,
            logit_cap=0.0,
        )
        return o


# ---------------------------------------------------------------------------
# generate_input / ref_kernel / check_implementation
# ---------------------------------------------------------------------------

def generate_input(batchsize: int, qseqlen: int, kvseqlen: int, kvdtype: str, seed: int) -> input_t:
    """
    Generate absorbed q and compressed kv_buffer for MLA decode.

    KV cache quantization:
      - bf16:  kv_buffer stays bfloat16, kv_scale=None
      - fp8:   dynamic per-tensor quantization → fp8 kv_buffer + scalar kv_scale
      - mxfp4: aiter dynamic_mxfp4_quant (block-32)
               → fp4x2 kv_buffer + fp8_e8m0 kv_scale
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    total_q = batchsize * qseqlen
    total_kv = batchsize * kvseqlen

    # Absorbed query: (total_q, num_heads, 576) bf16
    q = torch.randn(
        (total_q, NUM_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16, device="cuda", generator=gen,
    ) * 0.02

    # Compressed KV buffer: (total_kv, 1, 576) bf16
    kv_buffer_bf16 = torch.randn(
        (total_kv, NUM_KV_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16, device="cuda", generator=gen,
    ) * 0.02

    # Quantize KV buffer to target dtype
    kv_scale = None
    if kvdtype == "bf16":
        kv_buffer = kv_buffer_bf16
    elif kvdtype == "fp8":
        # Dynamic per-tensor FP8 quantization (sglang style)
        kv_buffer, kv_scale = quantize_fp8(kv_buffer_bf16)
    elif kvdtype == "mxfp4":
        # Block-32 MXFP4 quantization using aiter (native fp4x2 + fp8_e8m0)
        # kv_buffer: (total_kv, 1, 288) fp4x2, kv_scale: (total_kv, num_blocks) fp8_e8m0
        kv_buffer, kv_scale = quantize_mxfp4(kv_buffer_bf16)
    else:
        raise ValueError(f"Unknown kvdtype: {kvdtype}. Use bf16, fp8, or mxfp4.")

    qo_indptr = torch.arange(0, batchsize + 1, dtype=torch.int32, device="cuda") * qseqlen
    kv_indptr = torch.arange(0, batchsize + 1, dtype=torch.int32, device="cuda") * kvseqlen

    config = {
        "batch_size": batchsize,
        "num_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "qk_head_dim": QK_HEAD_DIM,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        "q_seq_len": qseqlen,
        "kv_seq_len": kvseqlen,
        "sm_scale": SM_SCALE,
        "kv_dtype": kvdtype,
        "kv_scale": kv_scale,
    }

    return (q, kv_buffer, qo_indptr, kv_indptr, config)


def ref_kernel(data: input_t) -> output_t:
    """Reference MLA attention using aiter kernels."""
    q, kv_buffer, qo_indptr, kv_indptr, config = data
    return _aiter_mla_attention(q, kv_buffer, qo_indptr, kv_indptr, config)


def check_implementation(data, output):
    """Check submission output against aiter reference with per-dtype tolerance."""
    config = data[-1]
    kv_dtype = config.get("kv_dtype", "bf16")
    tol = TOLERANCES.get(kv_dtype, TOLERANCES["bf16"])
    return match_reference(data, output, ref_kernel, **tol)
