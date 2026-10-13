# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""The single owner of matrix-product ATen backend registrations."""

import torch

from .dispatch import BACKWARD_HELPER_KEYS, MatmulRequest
from .autograd import register_autograd


TORCH_NPU_SPARSE_FACADE_APIS = {"mm": torch.sparse.mm, "addmm": torch.sparse.addmm}


def register_aten(registry):
    """Return the Library guard; its lifetime must span all registered kernels."""
    library = torch.library.Library("aten", "IMPL")

    def mm(mat1, mat2):
        return registry.dispatch(MatmulRequest("mm", mat1, mat2, schema="aten::mm"))

    def sparse_sparse_mm(mat1, mat2):
        if mat1.layout == torch.strided or mat2.layout == torch.strided:
            raise RuntimeError("_sparse_sparse_matmul requires two sparse operands")
        return registry.dispatch(MatmulRequest("mm", mat1, mat2,
                                                schema="aten::_sparse_sparse_matmul"))

    def addmm(self, mat1, mat2, *, beta=1, alpha=1):
        return registry.dispatch(MatmulRequest("addmm", mat1, mat2, self, beta, alpha,
                                                schema="aten::addmm"))

    def addmm_out(self, mat1, mat2, *, beta=1, alpha=1, out):
        return registry.dispatch(MatmulRequest("addmm", mat1, mat2, self, beta, alpha, out,
                                                schema="aten::addmm.out"))

    def mm_out(mat1, mat2, *, out):
        return registry.dispatch(MatmulRequest("mm", mat1, mat2, out=out, schema="aten::mm.out"))

    def sparse_addmm(self, mat1, mat2, *, beta=1, alpha=1):
        return registry.dispatch(MatmulRequest("addmm", mat1, mat2, self, beta, alpha,
                                                schema="aten::_sparse_addmm"))

    implementations = {
        "mm": mm,
        "mm.out": mm_out,
        "addmm": addmm,
        "addmm.out": addmm_out,
        "_sparse_addmm": sparse_addmm,
        "_sparse_sparse_matmul": sparse_sparse_mm,
    }
    for dispatch_key in ("SparsePrivateUse1", "SparseCsrPrivateUse1"):
        for schema, implementation in implementations.items():
            library.impl(schema, implementation, dispatch_key)
        for schema, implementation in registry.backward_helpers.items():
            if dispatch_key in BACKWARD_HELPER_KEYS[schema]:
                library.impl(schema, implementation, dispatch_key)
    if any(capability.backward is not None for capability in registry.capabilities.values()):
        register_autograd(library, registry)
    return library
