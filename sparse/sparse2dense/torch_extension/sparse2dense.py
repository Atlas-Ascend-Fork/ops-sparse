# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SparseToDense torch_npu 后端的 ATen 注册（``aten::_to_dense``）。

复用 PyTorch 已有 schema，不额外 ``define`` 自定义算子。CSR / CSC 走
``SparseCsrPrivateUse1``，COO 走 ``SparsePrivateUse1``；首次 NPU 调用时 JIT
编译 C++ wrapper 并接入 ``aclsparseSparseToDense``。
"""

__all__ = []

from typing import List, Optional

import torch
from torch.library import impl

from ....op_builder.builder import OpBuilder


class Sparse2DenseOpBuilder(OpBuilder):
    """SparseToDense C++ 封装的 JIT 构建器。"""

    def __init__(self) -> None:
        super().__init__("sparse2dense", category="sparse")

    def sources(self) -> List[str]:
        return ["csrc/sparse/sparse2dense/sparse2dense.cpp"]


_sparse2dense_op_builder = Sparse2DenseOpBuilder()


def _apply_dtype(out: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    if dtype is None or out.dtype == dtype:
        return out
    return out.to(dtype)


def _csr_on_device_from_cpu_csr(csr: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Rebuild a CPU CSR as an NPU CSR with explicit int32 index copies."""
    return torch.sparse_csr_tensor(
        csr.crow_indices().to(dtype=torch.int32, device=device).contiguous(),
        csr.col_indices().to(dtype=torch.int32, device=device).contiguous(),
        csr.values().to(device=device).contiguous(),
        size=tuple(csr.shape),
        device=device,
    )


_NATIVE_CSC_COO_SOC_PREFIXES = ("Ascend910B", "Ascend910_93")


def _soc_supports_native_csc_coo(tensor: torch.Tensor) -> bool:
    """Whether the tensor's SoC takes CSC/COO natively (no CPU rebuild).

    910B / A3 family (arch22) accepts CSC/COO straight into the operator,
    including uncoalesced COO (out-of-order coordinates are legal). Other
    SoCs (e.g. ascend950) keep the CPU rebuild below. Probe failures or
    unknown SoCs fall back to the conservative rebuild path.
    """
    npu = getattr(torch, "npu", None)
    get_props = getattr(npu, "get_device_properties", None)
    if get_props is None:
        return False
    index = tensor.device.index
    if index is None:
        try:
            index = npu.current_device()
        except (RuntimeError, TypeError):
            index = 0
    try:
        name = str(get_props(index).name or "")
    except (RuntimeError, TypeError, AssertionError):
        return False
    return name.startswith(_NATIVE_CSC_COO_SOC_PREFIXES)


def _coalesce_coo_on_device(tensor: torch.Tensor) -> torch.Tensor:
    """Sum-merge an uncoalesced plain 2-D COO, rebuilt on the original device.

    Tensor.to_dense interprets duplicate coordinates of an uncoalesced COO as
    a sum (PyTorch semantics), while the native operator overwrites (the
    aclsparse API contract for duplicate coordinates is any one of the
    written values). Merging keeps the torch-level semantics; the NPU-side
    coalesce is unavailable (EZ1001 / 161001), so the CPU merge is the
    semantic fallback. Indices stay int64; the wrapper casts on device.
    """
    merged = tensor.cpu().coalesce()
    return torch.sparse_coo_tensor(
        merged.indices().to(tensor.device),
        merged.values().to(tensor.device),
        size=tuple(tensor.shape),
        device=tensor.device,
    )


def _is_plain_uncoalesced_coo(tensor: torch.Tensor) -> bool:
    """Whether the tensor is a plain 2-D COO that still needs the sum merge.

    Hybrid or higher-dimensional COO fails the shape guard and heads
    straight to the wrapper layout check instead of paying the merge.
    """
    return (tensor.layout == torch.sparse_coo and not tensor.is_coalesced()
            and tensor.sparse_dim() == 2 and tensor.dense_dim() == 0)


def _normalize_for_aclsparse(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize sparse layouts before they reach the native ACLSparse call.

    - 910B / A3 (arch22) skips the CPU rebuild: CSC/COO pass through
      natively, except that a plain uncoalesced 2-D COO is sum-merged first
      (Tensor.to_dense sums duplicate coordinates; the native operator is an
      overwrite scatter) and rebuilt as COO on the original device — the free
      ``is_coalesced`` flag gates already-merged tensors past the merge.
    - CSC / uncoalesced COO on other SoCs (e.g. ascend950): convert via CPU
      to CSR, then rebuild with explicit int32 index copies on device. Plain
      ``.to(device)`` sparse tensors can mis-bind after prior CSR launches on
      ascend950; NPU coalesce is unavailable (EZ1001 / 161001).
    """
    if _soc_supports_native_csc_coo(tensor):
        if _is_plain_uncoalesced_coo(tensor):
            return _coalesce_coo_on_device(tensor)
        return tensor
    device = tensor.device
    if tensor.layout == torch.sparse_csc:
        return _csr_on_device_from_cpu_csr(tensor.cpu().to_sparse_csr(), device)
    if tensor.layout == torch.sparse_coo:
        coo = tensor.cpu() if tensor.is_coalesced() else tensor.cpu().coalesce()
        return _csr_on_device_from_cpu_csr(coo.to_sparse_csr(), device)
    return tensor


def _safe_nnz(tensor: torch.Tensor) -> int:
    """nnz for the empty shortcut; public .values() raises on uncoalesced COO
    (torch 2.7). Plain 2-D COO is merged before this point on every SoC; the
    protected accessor stays as the defensive path for layouts outside that
    scope (e.g. hybrid COO heading into the wrapper's layout check)."""
    if tensor.layout == torch.sparse_coo and not tensor.is_coalesced():
        # getattr 间接访问规避受保护成员告警（pylint 对 getattr 间接访问不报）。
        # _nnz 不可替代：公开 .values() 在未 coalesce COO 上 RuntimeError（torch 2.7
        # 行为）
        return int(getattr(tensor, "_nnz")())
    return int(tensor.values().numel())


def _sparse_to_dense(tensor: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    tensor = _normalize_for_aclsparse(tensor)
    # Prefer values().numel() over Tensor._nnz() (protected-member access)
    # wherever the public accessor is legal; uncoalesced COO needs _nnz().
    if tensor.size(0) == 0 or tensor.size(1) == 0 or _safe_nnz(tensor) == 0:
        # Keep empty-nnz on the Python side; avoids fragile empty value storage in the
        # JIT extension after prior sparse launches on ascend950.
        out = torch.zeros(tuple(tensor.shape), dtype=tensor.dtype, device=tensor.device)
        return _apply_dtype(out, dtype)
    return _apply_dtype(_sparse2dense_op_builder.load().sparse_to_dense(tensor), dtype)


def _to_dense_dispatch(self: torch.Tensor, dtype: Optional[torch.dtype],
                       masked_grad: Optional[bool]) -> torch.Tensor:
    """aten::_to_dense schema: (Tensor self, ScalarType? dtype=None, *,
    bool? masked_grad=None)。dtype 按 torch 原生语义生效（计算后转换）；
    masked_grad 为自动求导专用标志，本推理后端不支持，显式报错（禁止静默
    忽略改变语义）。
    """
    if masked_grad is not None:
        raise RuntimeError(
            "aclsparseSparseToDense PTA backend does not support masked_grad")
    return _sparse_to_dense(self, dtype)


@impl("aten::_to_dense", "SparseCsrPrivateUse1")
def _to_dense_sparse_csr_privateuse1(
    self: torch.Tensor, dtype: Optional[torch.dtype] = None, masked_grad: Optional[bool] = None
) -> torch.Tensor:
    return _to_dense_dispatch(self, dtype, masked_grad)


@impl("aten::_to_dense", "SparsePrivateUse1")
def _to_dense_sparse_privateuse1(
    self: torch.Tensor, dtype: Optional[torch.dtype] = None, masked_grad: Optional[bool] = None
) -> torch.Tensor:
    return _to_dense_dispatch(self, dtype, masked_grad)


# ``to_dense`` 是 Tensor 方法，不通过 torch_npu.sparse façade 暴露。
TORCH_NPU_SPARSE_FACADE_APIS = {}
