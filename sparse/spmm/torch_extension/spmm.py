# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Lazy SpMM capability for the shared matrix-product router."""

from __future__ import annotations

__all__ = ["MATMUL_CAPABILITIES"]

from typing import List

import torch

from ....op_builder.builder import OpBuilder
from ...matmul.dispatch import MatmulCapability
from .spmm_backward import spmm_backward


class SpMMOpBuilder(OpBuilder):
    """SpMM C++ 封装的 JIT 构建器。"""

    def __init__(self) -> None:
        super().__init__("spmm", category="sparse")

    def sources(self) -> List[str]:
        return ["csrc/sparse/spmm/spmm.cpp"]


_spmm_op_builder = SpMMOpBuilder()


def _copy_result(out: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    if tuple(out.shape) != tuple(result.shape):
        out.resize_(result.shape)
    if out.is_complex():
        physical_out = out.conj() if out.is_conj() else out
        physical_result = result.conj().resolve_conj() if out.is_conj() else result.resolve_conj()
        torch.view_as_real(physical_out).copy_(torch.view_as_real(physical_result))
    else:
        out.copy_(result)
    return out


def _execute_spmm(request):
    """Execute a normalized CSR-dense request selected by the shared router."""
    extension = _spmm_op_builder.load()
    shape = (request.mat1.size(0), request.mat2.size(1))
    out = request.out
    direct_out = out is not None and out.is_contiguous() and not out.is_conj()
    target = out if direct_out else torch.empty(
        shape, dtype=request.mat1.dtype, device=request.mat1.device)
    if request.operation == "mm":
        self_tensor = torch.empty((), dtype=request.mat1.dtype, device=request.mat1.device)
        result = extension.addmm_out(self_tensor, request.mat1, request.mat2, 0, 1, target)
    else:
        result = extension.addmm_out(
            request.self, request.mat1, request.mat2, request.beta, request.alpha, target)
    return _copy_result(out, result) if out is not None and target is not out else result


MATMUL_CAPABILITIES = (
    MatmulCapability(
        owner="spmm",
        route="spmm",
        forward=_execute_spmm,
        backward=spmm_backward,
        dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.complex64),
        device_types=("npu", "privateuseone"),
    ),
)
