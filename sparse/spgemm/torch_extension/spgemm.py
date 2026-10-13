# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Lazy SpGEMM implementation capabilities for the shared matrix router.

The framework owns ATen registrations and the public sparse facade. This
operator package only supplies its CSR/COO computation wrappers.
"""

__all__ = []

from typing import List

import torch

from ....op_builder.builder import OpBuilder
from ...matmul.dispatch import MatmulCapability
from .spgemm_backward import as_coo, sparse_add_inplace, sparse_add_out, sparse_mask, spgemm_backward


class SpGemmOpBuilder(OpBuilder):
    """SpGEMM C++ 封装的 JIT 构建器。"""

    def __init__(self) -> None:
        super().__init__("spgemm", category="sparse")

    def sources(self) -> List[str]:
        return ["csrc/sparse/spgemm/spgemm.cpp"]


_spgemm_op_builder = SpGemmOpBuilder()


def _execute_spgemm(request):
    # Load only after shared routing has selected a supported sparse product.
    if request.out is not None:
        raise RuntimeError("SpGEMM does not yet provide an out= implementation")
    extension = _spgemm_op_builder.load()
    if request.operation == "addmm":
        return extension.sparse_addmm(request.self, request.mat1, request.mat2,
                                      request.beta, request.alpha)
    if request.mat1.layout == torch.sparse_coo:
        return extension.sparse_sparse_matmul_coo(as_coo(request.mat1), as_coo(request.mat2))
    return extension.sparse_sparse_matmul_csr(request.mat1, request.mat2)


def _backward_spgemm(request, grad_output):
    return spgemm_backward(request, grad_output, _execute_spgemm)


MATMUL_CAPABILITIES = (
    MatmulCapability(owner="spgemm", route="spgemm", forward=_execute_spgemm,
                     backward=_backward_spgemm,
                     backward_helpers=(("sparse_mask", sparse_mask), ("add.out", sparse_add_out),
                                       ("add_.Tensor", sparse_add_inplace)),
                     dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.complex64),
                     device_types=("npu", "privateuseone")),
)
