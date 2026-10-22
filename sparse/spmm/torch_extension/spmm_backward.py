# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details in the LICENSE file in the root of the software repository.
# ----------------------------------------------------------------------------------------------------------

"""First-order VJP for the CSR-dense SpMM route."""

from __future__ import annotations

from typing import Tuple

import torch
from torch._C import _jit_get_operation as get_aten_operation


_COO_INDICES, _ = get_aten_operation("aten::_indices")


def _scalar_conjugate(value, dtype):
    value = value.conjugate() if hasattr(value, "conjugate") else value
    return value if dtype.is_complex or not isinstance(value, complex) else value.real


def _scales(request):
    """Return the conjugated scalar multipliers for the matrix-product VJP.

    For ``Y = beta*self + alpha*(A@B)`` the complex Wirtinger gradient uses
    conj(beta) and conj(alpha); the real-valued dtypes are unaffected.
    """
    conjugated_alpha = _scalar_conjugate(request.alpha, request.mat1.dtype)
    return (_scalar_conjugate(request.beta, request.mat1.dtype),
            conjugated_alpha, conjugated_alpha)


def _as_csr(tensor):
    if tensor.layout == torch.sparse_csr:
        return tensor
    return tensor.to_sparse_csr()


def _coordinates(tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return one row/column coordinate for every stored sparse value."""
    if tensor.layout == torch.sparse_csr:
        offsets = tensor.crow_indices()
        counts = (offsets[1:] - offsets[:-1]).to(torch.int64)
        rows = torch.arange(tensor.size(0), device=tensor.device, dtype=torch.int64)
        rows = torch.repeat_interleave(rows, counts)
        return rows, tensor.col_indices().to(torch.int64)
    if tensor.layout == torch.sparse_csc:
        offsets = tensor.ccol_indices()
        counts = (offsets[1:] - offsets[:-1]).to(torch.int64)
        cols = torch.arange(tensor.size(1), device=tensor.device, dtype=torch.int64)
        cols = torch.repeat_interleave(cols, counts)
        return tensor.row_indices().to(torch.int64), cols
    if tensor.layout == torch.sparse_coo:
        indices = tensor.indices() if tensor.is_coalesced() else _COO_INDICES(tensor)
        return indices[0].to(torch.int64), indices[1].to(torch.int64)
    raise RuntimeError("SpMM backward requires a sparse matrix input")


def _complex_planes(tensor):
    """Materialize complex logical views once as float32 real/imag planes."""
    if not tensor.is_complex():
        return tensor
    logical = tensor.resolve_conj().resolve_neg()
    return torch.view_as_real(logical)


def _sample_rows(tensor, rows, is_complex):
    if is_complex:
        # Indexing the float plane avoids the unsupported complex64 2-D Gather
        # path while preserving logical conjugate/negative view semantics.
        return torch.view_as_complex(tensor[rows].contiguous())
    return tensor[rows]


def _sampled_values(left, right, rows, cols, dtype):
    """Evaluate selected entries of ``left @ right`` without dense matmul."""
    count = rows.numel()
    values = torch.empty(count, dtype=dtype, device=left.device)
    if count == 0:
        return values

    # Keep temporary indexed rows bounded for graph-shaped inputs with many
    # nonzeros. Accumulation is promoted for the low-precision value types.
    inner = max(1, left.size(1))
    chunk = max(1, (1 << 20) // inner)
    left_complex = left.is_complex()
    right_complex = right.is_complex()
    left = _complex_planes(left)
    right_transposed = _complex_planes(right.transpose(0, 1))
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        lhs = _sample_rows(left, rows[start:end], left_complex)
        rhs = _sample_rows(right_transposed, cols[start:end], right_complex)
        if lhs.dtype in (torch.float16, torch.bfloat16):
            lhs = lhs.float()
            rhs = rhs.float()
        product = lhs * rhs
        values[start:end] = product.sum(dim=1).to(dtype)
    return values


def _sampled_product(left, right, pattern):
    rows, cols = _coordinates(pattern)
    return _sampled_values(left, right, rows, cols, pattern.dtype)


def _dense_product(left, right):
    """Evaluate a dense product with index-select and vector reduction.

    The A2/A3 torch_npu runtime used by this adapter does not provide dense
    complex64 ``mm``. This path keeps the calculation on NPU while using the
    same vector operations as the sparse-value projection.
    """
    rows_count, cols_count = left.size(0), right.size(1)
    output = torch.empty((rows_count, cols_count), dtype=left.dtype, device=left.device)
    if rows_count == 0 or cols_count == 0:
        return output
    rows_per_chunk = max(1, (1 << 20) // max(1, cols_count))
    columns = torch.arange(cols_count, device=left.device, dtype=torch.int64)
    for start in range(0, rows_count, rows_per_chunk):
        end = min(start + rows_per_chunk, rows_count)
        rows = torch.arange(start, end, device=left.device, dtype=torch.int64)
        rows = rows.repeat_interleave(cols_count)
        cols = columns.repeat(end - start)
        values = _sampled_values(left, right, rows, cols, left.dtype)
        output[start:end] = values.reshape(end - start, cols_count)
    return output


def _rebuild_sparse(pattern, values):
    if pattern.layout == torch.sparse_csr:
        return torch.sparse_csr_tensor(
            pattern.crow_indices(), pattern.col_indices(), values,
            size=pattern.shape, device=pattern.device,
        )
    if pattern.layout == torch.sparse_csc:
        return torch.sparse_csc_tensor(
            pattern.ccol_indices(), pattern.row_indices(), values,
            size=pattern.shape, device=pattern.device,
        )
    indices = pattern.indices() if pattern.is_coalesced() else _COO_INDICES(pattern)
    return torch.sparse_coo_tensor(
        indices, values, pattern.shape, device=pattern.device,
        is_coalesced=pattern.is_coalesced(),
    )


def _sparse_left_gradient(request, gradient, scale):
    sparse = request.mat1.detach()
    # 前向把未 coalesce 的 COO 先 coalesce 再转 CSR，Autograd 却保留原始 COO；反向若不
    # 先 coalesce，会为每个重复坐标计算同一梯度并重建带重复项的稀疏梯度，后续累加会
    # 重复求和。这里先 coalesce，让梯度在唯一坐标上计算并返回正确的稀疏结构。
    if sparse.layout == torch.sparse_coo and not sparse.is_coalesced():
        sparse = sparse.coalesce()
    dense = request.mat2.detach()
    if request.schema == "aten::_sparse_addmm":
        values = _sampled_product(gradient, dense.mH, sparse) * scale
        return _rebuild_sparse(sparse, values)
    return _dense_product(gradient, dense.mH) * scale


def _sparse_right_gradient(request, gradient, scale):
    dense = request.mat1.detach()
    sparse = request.mat2.detach()
    if sparse.layout == torch.sparse_coo and not sparse.is_coalesced():
        sparse = sparse.coalesce()
    if request.schema == "aten::_sparse_addmm":
        values = _sampled_product(dense.mH, gradient, sparse) * scale
        return _rebuild_sparse(sparse, values)
    return _dense_product(dense.mH, gradient) * scale


@torch.no_grad()
def spmm_backward(request, grad_output):
    """Return ``(grad_self, grad_mat1, grad_mat2)`` for SpMM."""
    gradient = grad_output if grad_output.layout == torch.strided else grad_output.to_dense()
    beta, mat1_scale, mat2_scale = _scales(request)
    grad_self = None
    grad_mat1 = None
    grad_mat2 = None

    if request.self is not None and request.self.requires_grad:
        grad_self = (gradient * beta).sum_to_size(request.self.shape)

    mat1_sparse = request.mat1.layout != torch.strided
    mat2_sparse = request.mat2.layout != torch.strided
    if mat1_sparse == mat2_sparse:
        raise RuntimeError("SpMM backward requires exactly one sparse matrix operand")

    if request.mat1.requires_grad:
        if mat1_sparse:
            grad_mat1 = _sparse_left_gradient(request, gradient, mat1_scale)
        else:
            sparse = _as_csr(request.mat2.detach())
            grad_mat1 = torch.mm(sparse, gradient.mH).mH * mat1_scale

    if request.mat2.requires_grad:
        if mat2_sparse:
            grad_mat2 = _sparse_right_gradient(request, gradient, mat2_scale)
        else:
            sparse = _as_csr(request.mat1.detach())
            grad_mat2 = torch.mm(sparse.mH, gradient) * mat2_scale

    return grad_self, grad_mat1, grad_mat2
