# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SpMV（稀疏矩阵×稠密向量）torch_npu 后端的 ATen 注册。

``torch.mv(csr, vec)``（CSR 二维稀疏矩阵 × 一维稠密向量）路由到
``aten::mv`` 的 ``SparseCsrPrivateUse1`` 分发键，语义固定为
``alpha=1, beta=0``，即 ``y = A @ vec``。

此处不定义 schema，仅注册 NPU 后端实现；``aten::mv`` 已有 PyTorch 自带的
Meta 实现，无需补充。
"""

__all__ = []

from typing import List

import torch
from torch.library import impl

from ....op_builder.builder import OpBuilder


class SpMVOpBuilder(OpBuilder):
    """SpMV C++ 封装的 JIT 构建器。"""

    def __init__(self) -> None:
        super().__init__("spmv", category="sparse")

    def sources(self) -> List[str]:
        return ["csrc/sparse/spmv/spmv.cpp"]


_spmv_op_builder = SpMVOpBuilder()


# ``load()`` 只在实际分发时调用，导入包不触发 JIT 编译。
# 签名与 aten::mv 的 ATen schema 完全一致：self 为 CSR 稀疏矩阵，vec 为一维稠密向量。
@impl("aten::mv", "SparseCsrPrivateUse1")
def _mv_sparse_csr_privateuse1(self: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return _spmv_op_builder.load().mv(self, vec)
