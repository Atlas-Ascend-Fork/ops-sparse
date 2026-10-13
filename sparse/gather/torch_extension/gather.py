# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Register the constrained Gather implementation on ``aten::index_select``."""

from __future__ import annotations

__all__ = ["gather", "index_select_npu"]

import torch
from torch.library import define, impl, register_fake
from torch import _check as check_shape

from ....op_builder.builder import OpBuilder


class GatherOpBuilder(OpBuilder):
    """JIT builder for the Gather ATen wrapper."""

    def __init__(self) -> None:
        super().__init__("gather", category="sparse")

    def sources(self) -> list[str]:
        return ["csrc/sparse/gather/gather.cpp"]


_gather_op_builder = GatherOpBuilder()


@impl("aten::index_select", "PrivateUse1")
def _index_select_privateuse1(self: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """Gather from a one-dimensional NPU tensor through ``aclsparseGather``."""
    if self.dim() != 1:
        raise RuntimeError(
            f"aclsparseGather index_select requires a 1-D input, got {self.dim()} dimensions"
        )
    if dim not in (0, -1):
        raise IndexError(f"Dimension out of range for 1-D Gather input: {dim}")
    return _gather_op_builder.load().index_select(self, index)


define("ops_sparse::gather", "(Tensor values, Tensor indices, int base=0) -> Tensor")


@impl("ops_sparse::gather", "PrivateUse1")
def _gather_privateuse1(values: torch.Tensor, indices: torch.Tensor, base: int = 0) -> torch.Tensor:
    return _gather_op_builder.load().gather(values, indices, base)


@register_fake("ops_sparse::gather")
def _gather_meta(values: torch.Tensor, indices: torch.Tensor, base: int = 0) -> torch.Tensor:
    if base not in (0, 1):
        raise RuntimeError("aclsparseGather base must be 0 or 1")
    if values.dim() != 1 or indices.dim() != 1:
        raise RuntimeError("aclsparseGather requires one-dimensional values and indices")
    if values.layout != torch.strided or indices.layout != torch.strided:
        raise RuntimeError("aclsparseGather requires strided values and indices")
    if values.device != indices.device:
        raise RuntimeError("aclsparseGather values and indices must be on the same device")
    if not values.is_contiguous() or not indices.is_contiguous():
        raise RuntimeError("aclsparseGather values and indices must be contiguous")
    if indices.dtype != torch.int32:
        raise RuntimeError("aclsparseGather indices must use int32")
    if values.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.complex64):
        raise RuntimeError("aclsparseGather values have unsupported dtype")
    if values.requires_grad:
        raise RuntimeError("aclsparseGather is forward-only and does not support autograd")
    check_shape((values.numel() > 0) | (indices.numel() == 0))
    return values.new_empty(indices.shape)


def gather(values: torch.Tensor, indices: torch.Tensor, base: int = 0) -> torch.Tensor:
    """Gather a one-dimensional NPU source using zero-based or one-based indices."""
    return torch.ops.ops_sparse.gather(values, indices, base)


def index_select_npu(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """Call the registered one-dimensional NPU index_select implementation."""
    if tensor.device.type != "npu":
        raise NotImplementedError("aclsparseGather index_select_npu requires an NPU input")
    return torch.index_select(tensor, dim, index)


# Gather exposes standard index_select and its base-aware custom schema; no sparse façade.
TORCH_NPU_SPARSE_FACADE_APIS = {}
