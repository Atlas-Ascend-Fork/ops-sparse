# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SparseToDense Torch Extension 端到端测试（aten::_to_dense）。

前置环境与构建/运行步骤见同目录 ``README.md``。
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
cann_ops_sparse = pytest.importorskip("cann_ops_sparse")

DEVICE = "npu:0"

# 值类型支持随架构不同：arch35（ascend950）含 int32；arch22（910B/A3）为
# int8/float16/bfloat16/float32/complex64（无 int32）。int32 参数在 910B/A3 上跳过。
_ARCH22_SOC_PREFIXES = ("Ascend910B", "Ascend910_93")


def _on_arch22() -> bool:
    try:
        name = str(torch.npu.get_device_properties(0).name or "")
    except (RuntimeError, TypeError, AssertionError):
        return False
    return name.startswith(_ARCH22_SOC_PREFIXES)


_SKIP_INT32_ON_ARCH22 = pytest.mark.skipif(
    _on_arch22(), reason="arch22 (910B/A3) sparse2dense has no int32 value dtype")

VALUE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.complex64,
    torch.int8,
    pytest.param(torch.int32, marks=_SKIP_INT32_ON_ARCH22),
)


def _has_npu() -> bool:
    return bool(getattr(torch, "npu", None) and torch.npu.is_available())


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU unavailable")


def _values(nnz: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    if dtype == torch.complex64:
        real = torch.randn(nnz, dtype=torch.float32, device=device)
        imag = torch.randn(nnz, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    if dtype == torch.int8:
        return torch.randint(-8, 9, (nnz,), dtype=torch.int8, device=device)
    if dtype == torch.int32:
        return torch.randint(-128, 129, (nnz,), dtype=torch.int32, device=device)
    return torch.randn(nnz, dtype=dtype, device=device)


def test_import_does_not_trigger_jit(tmp_path):
    cache = tmp_path / "extensions"
    cache.mkdir()
    environment = dict(os.environ, TORCH_EXTENSIONS_DIR=str(cache))
    script = (
        "import torch, torch_npu, cann_ops_sparse\n"
        "from cann_ops_sparse.ops.sparse.sparse2dense.sparse2dense "
        "import _sparse2dense_op_builder as b\n"
        "print('LOADED:', list(type(b)._loaded_ops.keys()))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    assert "LOADED: []" in completed.stdout
    assert not any(cache.iterdir()), "importing the package produced build artifacts"


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_csr_to_dense_matches_cpu(dtype):
    crow = torch.tensor([0, 2, 3, 5], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([0, 2, 1, 0, 3], dtype=torch.int32, device=DEVICE)
    values = _values(5, dtype, DEVICE)
    sparse = torch.sparse_csr_tensor(crow, col, values, size=(3, 4), device=DEVICE)
    out = sparse.to_dense()
    ref = torch.sparse_csr_tensor(
        crow.cpu(), col.cpu(), values.cpu(), size=(3, 4)
    ).to_dense()
    torch.npu.synchronize()
    assert out.is_npu
    assert torch.equal(out.cpu(), ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_csc_to_dense_matches_cpu(dtype):
    ccol = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=DEVICE)
    row = torch.tensor([0, 1, 0], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=DEVICE)
    sparse = torch.sparse_csc_tensor(ccol, row, values, size=(2, 3), device=DEVICE)
    out = sparse.to_dense()
    ref = torch.sparse_csc_tensor(
        ccol.cpu(), row.cpu(), values.cpu(), size=(2, 3)
    ).to_dense()
    torch.npu.synchronize()
    assert out.is_npu
    assert torch.equal(out.cpu(), ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_coo_to_dense_matches_cpu(dtype):
    indices = torch.tensor([[0, 1, 0], [0, 1, 2]], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=DEVICE)
    sparse = torch.sparse_coo_tensor(indices, values, size=(2, 3), device=DEVICE)
    out = sparse.to_dense()
    ref = sparse.cpu().coalesce().to_dense()
    torch.npu.synchronize()
    assert out.is_npu
    assert torch.equal(out.cpu(), ref)


def test_empty_nnz_csr():
    crow = torch.tensor([0, 0, 0], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([], dtype=torch.int32, device=DEVICE)
    values = torch.tensor([], dtype=torch.float32, device=DEVICE)
    sparse = torch.sparse_csr_tensor(crow, col, values, size=(2, 3), device=DEVICE)
    out = sparse.to_dense()
    torch.npu.synchronize()
    assert out.shape == (2, 3)
    assert torch.count_nonzero(out.cpu()) == 0
