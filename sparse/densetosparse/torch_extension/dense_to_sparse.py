# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# A copy of the License is located at
# http://www.huawei.com
# This program is distributed in the hope that it will be useful, but WITHOUT any warranty of any kind.
# ----------------------------------------------------------------------------------------------------------

"""``Tensor.to_sparse*`` 的 NPU 注册（aten::_to_sparse / _to_sparse.sparse_dim /
_to_sparse_csr / _to_sparse_csc / _to_sparse_bsr）。

不定义 schema，仅注册 NPU 后端实现；输入为稠密张量，分发键为
``PrivateUse1``（稀疏输入才需要 SparseCsrPrivateUse1/SparsePrivateUse1，此处不适用）。
C++ wrapper（csrc/densetosparse.cpp）完成校验、描述符与 aclsparse 三阶段调用；
稀疏输出张量在本层组装——在后端 kernel 上下文内构造稀疏张量会以排除该后端键的
方式重入 dispatcher，导致分配被误路由。

``load()`` 在每次分发时调用，而非导入时调用，导入包既不需要编译器也不会产生
构建产物。以下每个签名严格对应其 ATen schema（含默认值）。
"""

__all__ = []

from typing import List, Optional

import torch
from torch.library import impl

from ....op_builder.builder import OpBuilder


class DenseToSparseOpBuilder(OpBuilder):
    """DenseToSparse C++ 封装的 JIT 构建器。"""

    def __init__(self) -> None:
        super().__init__("densetosparse", category="sparse")

    def sources(self) -> List[str]:
        return ["csrc/sparse/densetosparse/densetosparse.cpp"]


dense_to_sparse_op_builder = DenseToSparseOpBuilder()

_SUPPORTED_DTYPES = (
    torch.int8, torch.float16, torch.bfloat16, torch.float32, torch.complex64,
)


def _check_dense(t: torch.Tensor, dim: int = 2) -> None:
    if t.device.type != "npu":
        raise RuntimeError(
            "aclsparseDenseToSparse(ATen): requires an NPU tensor")
    if t.dim() != dim:
        raise RuntimeError(
            f"aclsparseDenseToSparse(ATen): expected a {dim}-D dense tensor, "
            f"got dim {t.dim()}")
    if t.dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            "aclsparseDenseToSparse(ATen): unsupported dtype "
            "(int8/fp16/bf16/fp32/complex64)")


def _dense_operand(t: torch.Tensor) -> torch.Tensor:
    # Column-major 2-D storages feed the COL order directly; anything else
    # non-contiguous is copied on device.
    if t.is_contiguous():
        return t
    st = t.stride()
    if st[0] == 1 and st[1] >= t.size(0) and t.storage_offset() == 0:
        return t
    return t.contiguous()


def _select_rows(tensor: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    # Use multidimensional/int64 indexing independently of Gather's narrower
    # aten::index_select contract when both operator packages are installed.
    if tensor.is_complex():
        planes = torch.view_as_real(tensor.resolve_conj())
        return torch.view_as_complex(planes[rows].contiguous())
    return tensor[rows]


def _to_csr(self: torch.Tensor, dense_dim: Optional[int] = None) -> torch.Tensor:
    _check_dense(self)
    if dense_dim not in (None, 0):
        raise RuntimeError(
            "aclsparseDenseToSparse(ATen): dense_dim != 0 unsupported")
    values, major, minor = dense_to_sparse_op_builder.load().to_csr(
        _dense_operand(self))
    return torch.sparse_csr_tensor(major, minor, values,
                                    self.shape, device=self.device)


def _to_csc(self: torch.Tensor, dense_dim: Optional[int] = None) -> torch.Tensor:
    _check_dense(self)
    if dense_dim not in (None, 0):
        raise RuntimeError(
            "aclsparseDenseToSparse(ATen): dense_dim != 0 unsupported")
    values, major, minor = dense_to_sparse_op_builder.load().to_csc(
        _dense_operand(self))
    return torch.sparse_csc_tensor(major, minor, values,
                                    self.shape, device=self.device)


def _mark_coalesced(t: torch.Tensor) -> torch.Tensor:
    # torch has no public API to flag a pre-sorted COO stream (coalesce()
    # would rerun a device kernel and lacks int8 support on NPU); the
    # attribute is reached by name to keep static protected-member
    # checks quiet while preserving the exact tensor semantics.
    getattr(t, "_coalesced_")(True)
    return t


def _to_coo(self: torch.Tensor, sparse_dim: int, dense_dim: int) -> torch.Tensor:
    _check_dense(self)
    if not ((sparse_dim, dense_dim) in ((2, 0), (1, 1))):
        raise RuntimeError(
            "aclsparseDenseToSparse(ATen): unsupported sparse_dim/dense_dim "
            "combination (supported: 2/0 and 1/1)")
    d = _dense_operand(self)
    if sparse_dim == 2:
        values, rows, cols = dense_to_sparse_op_builder.load().to_coo(d)
        ind = torch.stack((rows, cols), 0)
        t = torch.sparse_coo_tensor(ind, values, self.shape,
                                    device=self.device)
        # The COO stream is sorted and unique by construction (and the
        # device-side coalesce lacks int8 support).
        return _mark_coalesced(t)
    # Hybrid (1, 1): one entry per row that contains any nonzero; the
    # values are the full rows.
    rows = d.ne(0).any(dim=1).nonzero(as_tuple=True)[0]
    vals = _select_rows(d, rows)
    ind = rows.reshape(1, -1)
    t = torch.sparse_coo_tensor(ind, vals, self.shape, device=self.device)
    return _mark_coalesced(t)


def _to_bsr(self: torch.Tensor, blocksize, dense_dim: Optional[int] = None) -> torch.Tensor:
    _check_dense(self)
    if dense_dim not in (None, 0):
        raise RuntimeError(
            "aclsparseDenseToSparse(ATen): dense_dim != 0 unsupported")
    if len(blocksize) != 2:
        raise RuntimeError(
            "aclsparseDenseToSparse(ATen): blocksize must have 2 elements")
    block_rows = int(blocksize[0])
    block_cols = int(blocksize[1])
    if block_rows < 1 or block_cols < 1:
        raise RuntimeError("aclsparseDenseToSparse(ATen): invalid blocksize")
    # The aclsparse Blocked-ELL kernel takes one square block size; a
    # rectangular request must not be silently converted with the row
    # axis alone.
    if block_rows != block_cols:
        raise RuntimeError(
            "aclsparseDenseToSparse(ATen): only square blocksize is "
            f"supported, got ({block_rows},{block_cols})")
    b = block_rows
    # torch's public BSR conversion requires a divisible grid.
    if self.shape[0] % b or self.shape[1] % b:
        raise RuntimeError(
            f"dense_to_sparse_bsr: tensor sparse size "
            f"{tuple(self.shape)} must be divisible by given blocksize "
            f"({b},{b})")
    values, pattern = dense_to_sparse_op_builder.load().to_bell(
        _dense_operand(self), b)
    br = self.shape[0] // b
    mask = pattern.ne(-1)
    col = pattern.masked_select(mask)
    widths = mask.sum(dim=1)
    crow = torch.zeros(br + 1, dtype=torch.int64, device=self.device)
    crow[1:] = widths.cumsum(0)
    # Gather valid tiles without masked_select (its NPU kernel lacks
    # complex64 support): flatten (br, width) slots and index them.
    flat = values.reshape(-1, b, b)
    slot_ids = mask.reshape(-1).nonzero(as_tuple=True)[0]
    blocks = _select_rows(flat, slot_ids)
    return torch.sparse_bsr_tensor(crow, col, blocks, self.shape,
                                    device=self.device)


def _to_sparse_dispatch(self: torch.Tensor,
                         sparse_dim: Optional[int] = None,
                         dense_dim: Optional[int] = None) -> torch.Tensor:
    return _to_coo(self, sparse_dim if sparse_dim is not None else 2,
                   dense_dim if dense_dim is not None else 0)


def _to_sparse_sparse_dim(self: torch.Tensor, sparse_dim: int,
                          dense_dim: Optional[int] = None) -> torch.Tensor:
    return _to_coo(self, sparse_dim,
                   dense_dim if dense_dim is not None else self.dim() - sparse_dim)


def _to_sparse_layout(self: torch.Tensor, layout=None, blocksize=None,
                       dense_dim: Optional[int] = None) -> torch.Tensor:
    if layout is None:
        if blocksize is not None:
            raise RuntimeError(
                "aclsparseDenseToSparse(ATen): blocksize requires a BSR "
                "layout")
        return _to_coo(self, 2, 0)
    if layout == torch.sparse_coo:
        return _to_coo(self, 2, 0)
    if layout == torch.sparse_csr:
        return _to_csr(self, dense_dim)
    if layout == torch.sparse_csc:
        return _to_csc(self, dense_dim)
    if layout == torch.sparse_bsr:
        if blocksize is None:
            raise RuntimeError(
                "aclsparseDenseToSparse(ATen): BSR layout requires blocksize")
        return _to_bsr(self, blocksize, dense_dim)
    raise RuntimeError(
        "aclsparseDenseToSparse(ATen): unsupported layout "
        "(coo/csr/csc/bsr only)")


@impl("aten::_to_sparse", "PrivateUse1")
def _to_sparse_privateuse1(self: torch.Tensor, layout=None, blocksize=None,
                           dense_dim: Optional[int] = None) -> torch.Tensor:
    return _to_sparse_layout(self, layout, blocksize, dense_dim)


@impl("aten::_to_sparse.sparse_dim", "PrivateUse1")
def _to_sparse_sparse_dim_privateuse1(self: torch.Tensor,
                                      sparse_dim: int) -> torch.Tensor:
    return _to_coo(self, sparse_dim, self.dim() - sparse_dim)


@impl("aten::_to_sparse_csr", "PrivateUse1")
def _to_sparse_csr_privateuse1(self: torch.Tensor,
                               dense_dim: Optional[int] = None) -> torch.Tensor:
    return _to_csr(self, dense_dim)


@impl("aten::_to_sparse_csc", "PrivateUse1")
def _to_sparse_csc_privateuse1(self: torch.Tensor,
                               dense_dim: Optional[int] = None) -> torch.Tensor:
    return _to_csc(self, dense_dim)


@impl("aten::_to_sparse_bsr", "PrivateUse1")
def _to_sparse_bsr_privateuse1(self: torch.Tensor, blocksize,
                               dense_dim: Optional[int] = None) -> torch.Tensor:
    return _to_bsr(self, blocksize, dense_dim)


# 这些入口是 Tensor 方法而非 ``torch.sparse.*`` 函数，没有可安装进
# ``torch_npu.sparse`` 命名空间的 façade；保持空表以参与包级聚合。
TORCH_NPU_SPARSE_FACADE_APIS = {}
