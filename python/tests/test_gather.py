# -------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------
"""
End-to-end UT for the Python / torch adapter of aclsparseGather (arch22).

Covers:
* ``torch.ops.ops_sparse.gather`` (base 0 / 1) vs CPU golden -- exact (bit-wise).
* ``ops_sparse.index_select_npu`` (dim == 0) vs ``torch.index_select`` CPU golden.
* Unsupported combinations raise (no CPU fallback): dim != 0, non-1-D input,
  non-NPU tensors, non-int32 indices.

Run with pytest (requires torch + torch_npu + a usable NPU):
    pytest python/tests/test_gather.py
"""

import logging

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
ops_sparse = pytest.importorskip("ops_sparse")

gather = ops_sparse.gather
index_select_npu = ops_sparse.index_select_npu

DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.complex64]


def _golden(y_cpu, idx_cpu, base):
    if base == 1:
        idx_cpu = idx_cpu - 1
    return y_cpu[idx_cpu]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("base", [0, 1])
def test_gather_exact(dtype, base):
    torch.manual_seed(0)
    n = 1000
    nnz = 256
    y_cpu = torch.randn(n, dtype=dtype)
    hi = n if base == 0 else n + 1
    lo = base
    idx_cpu = torch.randint(lo, hi, (nnz,), dtype=torch.int32)

    y_npu = y_cpu.to("npu")
    idx_npu = idx_cpu.to("npu")

    out = gather(y_npu, idx_npu, base=base)
    assert out.dtype == dtype
    assert out.device.type == "npu"
    assert out.shape == (nnz,)

    out_cpu = out.cpu()
    golden = _golden(y_cpu, idx_cpu, base)
    # pure data movement -> bit-wise exact
    assert torch.equal(out_cpu, golden), f"mismatch dtype={dtype} base={base}"


@pytest.mark.parametrize("base", [0, 1])
def test_gather_repeat_valid_index(base):
    # y_len 可小于 nnz；重复有效索引合法（与 torch.index_select 一致）。
    y_cpu = torch.tensor([7.0], dtype=torch.float32)
    if base == 0:
        idx_cpu = torch.tensor([0, 0], dtype=torch.int32)
    else:
        idx_cpu = torch.tensor([1, 1], dtype=torch.int32)
    golden = _golden(y_cpu, idx_cpu, base)
    out = gather(y_cpu.to("npu"), idx_cpu.to("npu"), base=base)
    assert out.shape == (2,)
    assert out.dtype == torch.float32
    assert torch.equal(out.cpu(), golden), f"repeat valid index mismatch base={base}"


@pytest.mark.parametrize("dtype", DTYPES)
def test_index_select_npu_dim0(dtype):
    torch.manual_seed(1)
    n = 500
    nnz = 100
    y_cpu = torch.randn(n, dtype=dtype)
    idx_cpu = torch.randint(0, n, (nnz,), dtype=torch.int32)

    out = index_select_npu(y_cpu.to("npu"), 0, idx_cpu.to("npu"))
    golden = torch.index_select(y_cpu, 0, idx_cpu)
    assert torch.equal(out.cpu(), golden), f"index_select mismatch dtype={dtype}"


def test_nn0_returns_empty():
    y = torch.randn(16, dtype=torch.float32, device="npu")
    idx = torch.empty(0, dtype=torch.int32, device="npu")
    out = gather(y, idx, base=0)
    assert out.numel() == 0


def test_dim_not_zero_raises():
    y = torch.randn(16, dtype=torch.float32, device="npu")
    idx = torch.randint(0, 16, (4,), dtype=torch.int32, device="npu")
    with pytest.raises(NotImplementedError):
        index_select_npu(y, 1, idx)


def test_non_1d_input_raises():
    y = torch.randn(4, 4, dtype=torch.float32, device="npu")
    idx = torch.randint(0, 4, (2,), dtype=torch.int32, device="npu")
    with pytest.raises(NotImplementedError):
        gather(y, idx, base=0)


def test_cpu_tensor_raises():
    y = torch.randn(16, dtype=torch.float32)  # CPU
    idx = torch.randint(0, 16, (4,), dtype=torch.int32)
    with pytest.raises(NotImplementedError):
        gather(y, idx, base=0)


def test_non_int32_indices_raises():
    y = torch.randn(16, dtype=torch.float32, device="npu")
    idx = torch.randint(0, 16, (4,), dtype=torch.int64, device="npu")
    with pytest.raises(NotImplementedError):
        gather(y, idx, base=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)
    for dtype in DTYPES:
        for base in (0, 1):
            test_gather_exact(dtype, base)
    for base in (0, 1):
        test_gather_repeat_valid_index(base)
    for dtype in DTYPES:
        test_index_select_npu_dim0(dtype)
    test_nn0_returns_empty()
    test_dim_not_zero_raises()
    test_non_1d_input_raises()
    test_cpu_tensor_raises()
    test_non_int32_indices_raises()
    logging.info("ALL PYTHON E2E TESTS PASSED")
