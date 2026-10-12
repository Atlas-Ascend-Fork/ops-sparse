# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""注册 index_copy_ 对应的标准 ATen 算子 NPU 实现。"""

import torch
from torch.library import impl

from cann_ops_sparse.op_builder import OpBuilder


class ScatterOpBuilder(OpBuilder):
    """按需编译并加载 aclsparseScatter 的 C++ wrapper。"""

    def __init__(self):
        super().__init__("scatter", category="sparse")

    def sources(self):
        return ["csrc/sparse/scatter/scatter.cpp"]


_SCATTER_OP_BUILDER = ScatterOpBuilder()


@impl("aten::index_copy_", "PrivateUse1")
def _index_copy_privateuse1(self, dim, index, source):
    """执行 ``Tensor.index_copy_(0, index, source)`` 的 NPU 原地散布。"""
    return _SCATTER_OP_BUILDER.load().index_copy_(self, dim, index, source)
