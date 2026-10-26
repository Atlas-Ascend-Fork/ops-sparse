# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""注册 SDDMM 对应的标准 ATen 稀疏算子 NPU 实现。

``aten::sparse_sampled_addmm`` 在 NPU CSR 张量上分派到本模块，内部转调
ACLSparse 的三阶段接口。整条路径没有 CPU fallback。
"""

from dataclasses import dataclass

import torch
from torch.library import impl

from cann_ops_sparse.op_builder import OpBuilder


class SddmmOpBuilder(OpBuilder):
    """按需编译并加载 SDDMM 的 C++ wrapper。"""

    def __init__(self):
        super().__init__("sddmm", category="sparse")

    def sources(self):
        return ["csrc/sparse/sddmm/sddmm.cpp"]


_SDDMM_OP_BUILDER = SddmmOpBuilder()

# ``cann_ops_sparse.ops`` 收集该映射，并在所有算子包导入后创建
# ``torch_npu.sparse`` 中的对应 façade。
TORCH_NPU_SPARSE_FACADE_APIS = {"sampled_addmm": torch.sparse.sampled_addmm}


def ensure_sddmm_loaded():
    """编译并加载 SDDMM 的 C++ wrapper，返回该扩展模块。

    ``torch.ops.ops_sparse_test.sddmm_npu`` 由这个扩展在库加载时用
    ``TORCH_LIBRARY`` 注册，所以任何需要该算子的调用方都得先确保库已经加载。
    加载本身带缓存，重复调用不重复编译。
    """
    return _SDDMM_OP_BUILDER.load()


@dataclass(frozen=True)
class CsrOperands:
    """Tensor bindings retained by a prepared CSR operation."""

    row_ptr: torch.Tensor
    col_ind: torch.Tensor
    values: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor


def prepare_sddmm_csr(operands, alpha, beta, base):
    """Create caller-owned three-stage state for repeated CSR execution."""
    return _SDDMM_OP_BUILDER.load().prepare_sddmm_csr(
        operands.row_ptr, operands.col_ind, operands.values, operands.x, operands.y,
        complex(alpha), complex(beta), int(base)
    )


@impl("aten::sparse_sampled_addmm", "SparseCsrPrivateUse1")
def sparse_sampled_addmm_csr(self, mat1, mat2, beta=1, alpha=1):
    """NPU CSR 上的 ``C = (alpha * mat1 @ mat2 + beta * C) ∘ spy(C)``。"""
    return _SDDMM_OP_BUILDER.load().sparse_sampled_addmm_csr(
        self, mat1, mat2, beta, alpha
    )
