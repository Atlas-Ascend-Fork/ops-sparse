# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Backward helpers shared by all sparse matrix-product providers."""

import torch
from torch._C import _jit_get_operation as get_aten_operation


_COO_INDICES, _ = get_aten_operation("aten::_indices")


def _sparse_indices(tensor):
    if tensor.layout == torch.sparse_csr:
        return tensor.crow_indices(), tensor.col_indices()
    if tensor.layout == torch.sparse_csc:
        return tensor.ccol_indices(), tensor.row_indices()
    if tensor.layout == torch.sparse_coo:
        indices = tensor.indices() if tensor.is_coalesced() else _COO_INDICES(tensor)
        return (indices,)
    return ()


def _shares_pattern(left, right, out):
    if left.layout != right.layout or out.layout != left.layout:
        return False
    if left.shape != right.shape or out.shape != left.shape:
        return False
    # The fast path reads `.values()` positionally, which is only valid for
    # coalesced COO (and is always valid for the compressed CSR/CSC layouts).
    if left.layout == torch.sparse_coo and not (
            left.is_coalesced() and right.is_coalesced() and out.is_coalesced()):
        return False
    left_indices = _sparse_indices(left)
    right_indices = _sparse_indices(right)
    out_indices = _sparse_indices(out)
    return bool(left_indices) and all(
        a.shape == b.shape == c.shape and a.data_ptr() == b.data_ptr() == c.data_ptr()
        and a.dtype == b.dtype == c.dtype and a.device == b.device == c.device
        and a.stride() == b.stride() == c.stride()
        and a.is_conj() == b.is_conj() == c.is_conj()
        and a.is_neg() == b.is_neg() == c.is_neg()
        for a, b, c in zip(left_indices, right_indices, out_indices)
    )


def _sparse_add_cpu(left, right, alpha):
    """Compute the CPU-side sparse sum for the non-fast-path branch."""
    if left.layout == right.layout and left.layout in (
            torch.sparse_coo, torch.sparse_csr, torch.sparse_csc) and left.shape == right.shape:
        # COO addition handles complex and union patterns without a CPU MKL dependency.
        result_dtype = torch.result_type(left, right)
        compute_dtype = torch.float32 if result_dtype in (torch.float16, torch.bfloat16) else result_dtype
        result = torch.add(left.to(compute_dtype).to_sparse_coo(),
                           right.to(compute_dtype).to_sparse_coo(), alpha=alpha)
        if result.layout == torch.sparse_coo and not result.is_coalesced():
            result = result.coalesce()
        if left.layout == torch.sparse_csr:
            result = result.to_sparse_csr()
        elif left.layout == torch.sparse_csc:
            result = result.to_sparse_csc()
        return result.to(result_dtype)
    return torch.add(left, right, alpha=alpha)


def _resize_sparse_out(out, result):
    """Resize `out` to match the computed result layout/shape."""
    if out.layout == torch.strided:
        out.resize_(result.shape)
    elif out.layout != torch.sparse_csc:
        out.resize_as_sparse_(result)
    elif out.values().numel() != result.values().numel():
        raise RuntimeError("sparse CSC add requires matching gradient patterns")


@torch.no_grad()
def sparse_add_out(self, other, *, alpha=1, out):
    """Accumulate sparse gradients, keeping the common same-pattern path on device."""
    if self.device != other.device or out.device != self.device:
        raise RuntimeError("sparse add requires operands and out on the same device")
    operand_dtype = torch.result_type(self, other)
    # Use native scalar validation against the operands before touching out.
    # Empty CPU tensors check integral/complex/bool alpha without device work.
    probe = torch.empty(0, dtype=operand_dtype, device='cpu')
    torch.add(probe, probe, alpha=alpha)
    if not torch.can_cast(operand_dtype, out.dtype):
        raise RuntimeError("sparse add out has an incompatible dtype")
    if _shares_pattern(self, other, out):
        # 低精度值相乘、相加前升 FP32 避免中间溢出，最终一次转换回 out dtype。
        compute_dtype = torch.float32 if operand_dtype in (torch.float16, torch.bfloat16) else operand_dtype
        out.values().copy_(
            (self.values().to(compute_dtype) + other.values().to(compute_dtype) * alpha).to(out.dtype))
        return out
    result = _sparse_add_cpu(self.detach().cpu(), other.detach().cpu(), alpha)
    if result.layout != out.layout or not torch.can_cast(result.dtype, out.dtype):
        raise RuntimeError("sparse add out has an incompatible layout or dtype")
    result = result.to(device=out.device, dtype=out.dtype)
    _resize_sparse_out(out, result)
    out.copy_(result)
    return out


def sparse_add_inplace(self, other, *, alpha=1):
    return sparse_add_out(self, other, alpha=alpha, out=self)
