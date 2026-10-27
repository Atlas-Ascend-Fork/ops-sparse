# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SDDMM registrations with separate native wrappers for arch22 and arch35."""

from dataclasses import dataclass
from typing import List, Union

import torch
from torch.library import impl

from ....op_builder.builder import OpBuilder

Scalar = Union[bool, int, float, complex]


def _uses_arch35_wrapper() -> bool:
    # Defer the device query until a native wrapper is requested.
    import torch_npu

    return torch_npu.npu.get_device_name().startswith("Ascend950")


class SddmmOpBuilder(OpBuilder):
    """Build the wrapper for the current NPU architecture on its first call."""

    def __init__(self) -> None:
        super().__init__("sddmm", category="sparse")

    def sources(self) -> List[str]:
        # The arch22 wrapper calls support kernels unavailable in arch35 builds.
        source = "sddmm.cpp" if _uses_arch35_wrapper() else "sddmm_arch22.cpp"
        return ["csrc/sparse/sddmm/" + source]


_sddmm_op_builder = SddmmOpBuilder()


TORCH_NPU_SPARSE_FACADE_APIS = {
    "sampled_addmm": torch.sparse.sampled_addmm,
}


def ensure_sddmm_loaded():
    """Load the current architecture's native SDDMM wrapper.

    The arch22 wrapper also registers ``torch.ops.ops_sparse_test.sddmm_npu``
    and exposes the prepared CSR interface.
    """
    return _sddmm_op_builder.load()


@dataclass(frozen=True)
class CsrOperands:
    """Tensor bindings retained by a prepared CSR operation on arch22."""

    row_ptr: torch.Tensor
    col_ind: torch.Tensor
    values: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor


def prepare_sddmm_csr(operands, alpha, beta, base):
    """Create caller-owned three-stage state for repeated CSR execution on arch22."""
    module = ensure_sddmm_loaded()
    prepare = getattr(module, "prepare_sddmm_csr", None)
    if prepare is None:
        raise NotImplementedError("prepare_sddmm_csr is available only on arch22")
    return prepare(
        operands.row_ptr, operands.col_ind, operands.values, operands.x, operands.y,
        complex(alpha), complex(beta), int(base)
    )


@impl("aten::sparse_sampled_addmm", "SparseCsrPrivateUse1")
def _sampled_addmm_sparse_csr_privateuse1(
    self: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor,
    *, beta: Scalar = 1, alpha: Scalar = 1,
) -> torch.Tensor:
    module = ensure_sddmm_loaded()
    if hasattr(module, "sampled_addmm"):
        return module.sampled_addmm(self, mat1, mat2, beta, alpha)
    return module.sparse_sampled_addmm_csr(self, mat1, mat2, beta, alpha)


@impl("aten::sparse_sampled_addmm.out", "SparseCsrPrivateUse1")
def _sampled_addmm_out_sparse_csr_privateuse1(
    self: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor,
    *, beta: Scalar = 1, alpha: Scalar = 1, out: torch.Tensor,
) -> torch.Tensor:
    module = ensure_sddmm_loaded()
    if not hasattr(module, "sampled_addmm_out"):
        raise NotImplementedError("sampled_addmm with out is available only on Ascend 950")
    module.sampled_addmm_out(self, mat1, mat2, beta, alpha, out)
    return out
