# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SpMM Torch Extension 注册的端到端测试。

在真实 NPU 上对已安装的 ``cann_ops_sparse`` wheel 运行::

    bash build.sh --torch_extension --ops=spmm
    python3 -m pip install build_out/cann_ops_sparse-*.whl --force-reinstall --no-deps
    OPS_SPARSE_LIB_DIR=$PWD/build_out/lib64 python3 -m pytest test/spmm/python/ -v

``OPS_SPARSE_LIB_DIR`` 为 JIT 链接步骤指定构建树，无需预先安装运行包。
"""

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
cann_ops_sparse = pytest.importorskip("cann_ops_sparse")

DEVICE = "npu:0"
VALUE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.complex64)

# 每项依次为 atol、rtol、绝对误差硬上限
TOLERANCE = {
    torch.float16: (2.0**-9, 2.0**-9, 1.0e-1),
    torch.bfloat16: (2.0**-6, 2.0**-6, 1.0),
    torch.float32: (2.0**-16, 2.0**-10, 1.0e-2),
    torch.complex64: (2.0**-16, 2.0**-10, 1.0e-2),
}
FRACTION_BITS = {
    torch.float16: 10,
    torch.bfloat16: 7,
    torch.float32: 23,
    torch.complex64: 23,
}
MIN_SUBNORMAL_EXP = {
    torch.float16: -24,
    torch.bfloat16: -133,
    torch.float32: -149,
    torch.complex64: -149,
}
GOLDEN_DTYPE = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float64,
    torch.complex64: torch.complex128,
}

# ``_sparse_addmm`` 是 PyTorch 自带的 ATen 算子（下划线是 torch 的命名，不是受保护
# 成员）。用 getattr 取用，避免点式访问被静态检查误判为 protected-access。
SPARSE_ADDMM = getattr(torch.ops.aten, "_sparse_addmm")


def make_complex_backward_operands():
    crow = torch.tensor([0, 2, 3], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([0, 2, 1], dtype=torch.int32, device=DEVICE)
    values = torch.tensor([1.5 + 0.4j, 2.0 - 0.3j, -0.5 + 0.2j],
                          dtype=torch.complex64, device=DEVICE)
    mat1 = torch.sparse_csr_tensor(crow, col, values, size=(2, 3), device=DEVICE)
    mat1.requires_grad_()
    mat2 = torch.tensor([[2.0 - 0.5j, 3.0 + 0.2j],
                         [4.0 + 0.7j, 0.25 - 0.1j],
                         [1.0 + 0.3j, -2.0 + 0.4j]],
                        dtype=torch.complex64, device=DEVICE, requires_grad=True)
    gradient = torch.tensor([[0.2 + 0.4j, 1.3 - 0.2j],
                             [-0.7 + 0.5j, 2.0 + 0.3j]],
                            dtype=torch.complex64, device=DEVICE)
    return mat1, mat2, gradient


# ---------------------------------------------------------------------------
# 精度断言：与 test/spmm/arch22 的 C++ 用例同口径——99% 的点满足 atol/rtol，
# 且每个点的绝对误差不超过硬上限与 32 ulp 两者中的较大值
# ---------------------------------------------------------------------------

def assert_plane(actual, golden, dtype):
    atol, rtol, hard_cap = TOLERANCE[dtype]
    finite = torch.isfinite(golden) & torch.isfinite(actual)
    assert torch.equal(torch.isnan(actual), torch.isnan(golden))
    assert torch.equal(torch.isinf(actual), torch.isinf(golden))
    if torch.isinf(golden).any():
        infinite = torch.isinf(golden)
        assert torch.equal(torch.signbit(actual[infinite]), torch.signbit(golden[infinite]))
    if not finite.any():
        return
    actual_values = actual[finite].double()
    golden_values = golden[finite].double()
    error = (actual_values - golden_values).abs()
    within = (error <= atol + rtol * golden_values.abs()).double().mean().item()
    assert within >= 0.99, "only {:.2%} of points met atol/rtol".format(within)

    magnitude = golden_values.abs()
    subnormal_ulp = 2.0 ** MIN_SUBNORMAL_EXP[dtype]
    normal_ulp = torch.pow(2.0, torch.floor(torch.log2(magnitude)) - FRACTION_BITS[dtype])
    ulp = torch.maximum(normal_ulp, torch.full_like(normal_ulp, subnormal_ulp))
    ulp = torch.where(magnitude == 0, torch.full_like(ulp, subnormal_ulp), ulp)
    assert bool((error <= torch.maximum(torch.full_like(error, hard_cap), 32.0 * ulp)).all())


def assert_close(actual, golden, dtype):
    wide = torch.complex128 if dtype.is_complex else torch.float64
    actual = actual.to(wide)
    golden = golden.to(wide)
    assert_plane(actual.real, golden.real, dtype)
    if dtype.is_complex:
        assert_plane(actual.imag, golden.imag, dtype)


def make_operands(dtype, device=DEVICE, input_shape=(2, 2), noncontiguous=False):
    """A = [[1, 0, 2], [0, -1, 0]] (2x3 CSR), B = 3x2 dense, bias 可广播。"""
    crow = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    col = torch.tensor([0, 2, 1], dtype=torch.int32, device=device)
    values = torch.tensor([1.0, 2.0, -1.0], dtype=dtype, device=device)
    mat1 = torch.sparse_csr_tensor(crow, col, values, size=(2, 3), device=device)
    mat2 = torch.tensor([[1.0, -2.0], [0.5, 3.0], [-1.0, 4.0]], dtype=dtype, device=device)
    bias = torch.full(input_shape, 0.125, dtype=dtype, device=device)
    if noncontiguous:
        mat2_store = torch.empty((3, 4), dtype=dtype, device=device)
        mat2_view = mat2_store[:, ::2]
        mat2_view.copy_(mat2)
        bias_store = torch.empty((2, 4), dtype=dtype, device=device)
        bias_view = bias_store[:, ::2]
        bias_view.copy_(bias.expand(2, 2))
        mat2, bias = mat2_view, bias_view
    return bias, mat1, mat2


def cpu_golden(dtype, input_shape, alpha, beta):
    wide = GOLDEN_DTYPE[dtype]
    a = torch.tensor([[1.0, 0.0, 2.0], [0.0, -1.0, 0.0]], dtype=dtype).to(wide)
    b = torch.tensor([[1.0, -2.0], [0.5, 3.0], [-1.0, 4.0]], dtype=dtype).to(wide)
    bias = torch.full(input_shape, 0.125, dtype=dtype).to(wide).expand(2, 2)
    golden = alpha * (a @ b)
    if beta != 0:
        golden = golden + beta * bias
    return golden


# ---------------------------------------------------------------------------
# 1. 注册验证
# ---------------------------------------------------------------------------

def test_dispatch_key_is_registered():
    """CSR mat1 + 稠密 mat2 能路由到 NPU 实现。"""
    bias, mat1, mat2 = make_operands(torch.float32)
    out = torch.addmm(bias, mat1, mat2)
    torch.npu.synchronize()
    assert out.device == bias.device
    assert out.layout == torch.strided
    assert tuple(out.shape) == (2, 2)


def test_shared_router_registers_spmm_capabilities():
    registry = cann_ops_sparse.ops.MATMUL_REGISTRY
    assert registry.capabilities[("spmm", "mm")].owner == "spmm"
    assert registry.capabilities[("spmm", "addmm")].owner == "spmm"


def test_sparse_public_apis_route_to_spmm():
    bias, mat1, mat2 = make_operands(torch.float32)
    addmm = torch.sparse.addmm(bias, mat1, mat2)
    mm = torch.sparse.mm(mat1, mat2)
    assert_close(addmm.cpu(), cpu_golden(torch.float32, (2, 2), 1, 1), torch.float32)
    assert_close(mm.cpu(), cpu_golden(torch.float32, (2, 2), 1, 0), torch.float32)


@pytest.mark.parametrize("schema", ("sparse", "dense"))
def test_complex_spmm_backward_uses_schema_scalar_rules(schema):
    alpha = complex(1.2, -0.7)
    beta = complex(0.5, 0.2)
    mat1, mat2, gradient = make_complex_backward_operands()
    bias = torch.tensor([2.0 + 0.3j, 1.0 - 0.2j], dtype=torch.complex64,
                        device=DEVICE, requires_grad=True)

    if schema == "sparse":
        result = torch.sparse.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    else:
        result = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    result.backward(gradient)
    torch.npu.synchronize()

    a_dense_cpu = torch.tensor([[1.5 + 0.4j, 0, 2.0 - 0.3j],
                                [0, -0.5 + 0.2j, 0]], dtype=torch.complex128)
    mat2_cpu = mat2.detach().cpu().to(torch.complex128)
    gradient_cpu = gradient.cpu().to(torch.complex128)
    expected_b = (alpha.conjugate() * (a_dense_cpu.mH @ gradient_cpu)).to(torch.complex64)
    expected_bias = (beta.conjugate() * gradient_cpu.sum(0)).to(torch.complex64)
    torch.testing.assert_close(mat2.grad.cpu(), expected_b.cpu(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(bias.grad.cpu(), expected_bias.cpu(), rtol=1e-4, atol=1e-4)
    if schema == "sparse":
        expected_a = (alpha.conjugate() * (gradient_cpu @ mat2_cpu.mH)).to(torch.complex64)
        torch.testing.assert_close(mat1.grad.values().cpu(),
                                   expected_a.cpu()[[0, 0, 1], [0, 2, 1]],
                                   rtol=1e-4, atol=1e-4)
    else:
        expected_a = (alpha.conjugate() * (gradient_cpu @ mat2_cpu.mH)).to(torch.complex64)
        assert mat1.grad.layout == torch.strided
        torch.testing.assert_close(mat1.grad.cpu(), expected_a.cpu(), rtol=1e-4, atol=1e-4)


def test_sparse_csr_gradient_accumulates_across_branches():
    """SpMM-only wheels must provide the shared CSR add helper for VJP accumulation."""
    mat1, mat2, gradient = make_complex_backward_operands()
    mat2_other = mat2.detach().clone().requires_grad_()

    result = torch.sparse.mm(mat1, mat2) + torch.sparse.mm(mat1, mat2_other)
    result.backward(gradient)
    torch.npu.synchronize()

    dense_a = torch.tensor([[1.5 + 0.4j, 0, 2.0 - 0.3j],
                            [0, -0.5 + 0.2j, 0]], dtype=torch.complex128)
    gradient_cpu = gradient.cpu().to(torch.complex128)
    expected_a = (2 * (gradient_cpu @ mat2.detach().cpu().to(torch.complex128).mH))
    expected_b = (dense_a.mH @ gradient_cpu).to(torch.complex64)
    torch.testing.assert_close(mat1.grad.values().cpu(),
                               expected_a.to(torch.complex64)[[0, 0, 1], [0, 2, 1]],
                               rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(mat2.grad.cpu(), expected_b, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(mat2_other.grad.cpu(), mat2.grad.cpu(), rtol=1e-4, atol=1e-4)


def test_sparse_csr_gradient_accumulates_across_repeated_backward():
    """Repeated backward must add a CSR VJP into the existing sparse grad."""
    mat1, mat2, gradient = make_complex_backward_operands()
    result = torch.sparse.mm(mat1, mat2)
    result.backward(gradient, retain_graph=True)
    first = mat1.grad.values().detach().clone()
    result.backward(gradient)
    torch.npu.synchronize()

    torch.testing.assert_close(mat1.grad.values().cpu(), 2 * first.cpu(),
                               rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("layout", (torch.sparse_coo, torch.sparse_csr))
def test_sparse_gradient_accumulation_preserves_input_layout(layout):
    base = torch.tensor([[1.5, 0, 2.0], [0, -0.5, 0]], dtype=torch.float32, device=DEVICE)
    sparse = base.to_sparse(layout=layout).detach().requires_grad_()
    dense = torch.tensor([[2.0, 3.0], [4.0, 0.25], [1.0, -2.0]],
                         dtype=torch.float32, device=DEVICE, requires_grad=True)
    result = torch.sparse.mm(sparse, dense)
    result.sum().backward(retain_graph=True)
    result.sum().backward()
    torch.npu.synchronize()

    # CSC is deliberately not covered here: PyTorch's autograd engine both
    # enforces a CSC gradient layout for a CSC input and calls is_contiguous()
    # on it during AccumulateGrad, which CSC does not support. Sparse CSC
    # inputs therefore fall back to the CPU add path rather than on-device
    # gradient accumulation.
    assert sparse.grad.layout == layout
    expected = (2 * torch.ones((2, 2)) @ dense.detach().cpu().t()) * (base.cpu() != 0)
    torch.testing.assert_close(sparse.grad.cpu().to_dense(), expected)


def test_complex_spmm_backward_dense_left_sparse_right():
    alpha = complex(1.2, -0.7)
    beta = complex(0.5, 0.2)
    left = torch.tensor([[1.5 + 0.4j, 0, 2.0 - 0.3j],
                         [0, -0.5 + 0.2j, 0]],
                        dtype=torch.complex64, device=DEVICE, requires_grad=True)
    crow = torch.tensor([0, 2, 3, 4], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([0, 1, 1, 0], dtype=torch.int32, device=DEVICE)
    values = torch.tensor([2.0 - 0.5j, 3.0 + 0.2j, 4.0 + 0.7j, 1.0 + 0.3j],
                          dtype=torch.complex64, device=DEVICE)
    right = torch.sparse_csr_tensor(crow, col, values, size=(3, 2), device=DEVICE)
    right.requires_grad_()
    bias = torch.tensor([2.0 + 0.3j, 1.0 - 0.2j], dtype=torch.complex64,
                        device=DEVICE, requires_grad=True)
    gradient = torch.tensor([[0.2 + 0.4j, 1.3 - 0.2j],
                             [-0.7 + 0.5j, 2.0 + 0.3j]],
                            dtype=torch.complex64, device=DEVICE)

    torch.addmm(bias, left, right, alpha=alpha, beta=beta).backward(gradient)
    torch.npu.synchronize()

    left_cpu = left.detach().cpu().to(torch.complex128)
    right_cpu = torch.tensor([[2.0 - 0.5j, 3.0 + 0.2j],
                              [0, 4.0 + 0.7j],
                              [1.0 + 0.3j, 0]], dtype=torch.complex128)
    gradient_cpu = gradient.cpu().to(torch.complex128)
    expected_left = (gradient_cpu @ right_cpu.mH * alpha.conjugate()).to(torch.complex64)
    expected_right = (left_cpu.mH @ gradient_cpu * alpha.conjugate()).to(torch.complex64)
    expected_bias = (gradient_cpu.sum(0) * beta.conjugate()).to(torch.complex64)
    torch.testing.assert_close(left.grad.cpu(), expected_left, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(right.grad.cpu().to_dense(), expected_right,
                               rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(bias.grad.cpu(), expected_bias, rtol=1e-4, atol=1e-4)
    assert right.grad.layout == torch.strided


def test_complex_sparse_mm_backward_without_bias():
    alpha = complex(1.2, -0.7)
    mat1, mat2, gradient = make_complex_backward_operands()

    torch.sparse.mm(mat1, mat2).backward(gradient)
    torch.npu.synchronize()

    a_cpu = torch.tensor([[1.5 + 0.4j, 0, 2.0 - 0.3j],
                          [0, -0.5 + 0.2j, 0]], dtype=torch.complex128)
    b_cpu = mat2.detach().cpu().to(torch.complex128)
    g_cpu = gradient.cpu().to(torch.complex128)
    expected_b = (a_cpu.mH @ g_cpu).to(torch.complex64)
    expected_a = (g_cpu @ b_cpu.mH).to(torch.complex64)
    torch.testing.assert_close(mat2.grad.cpu(), expected_b.cpu(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(mat1.grad.values().cpu(),
                               expected_a.cpu()[[0, 0, 1], [0, 2, 1]],
                               rtol=1e-4, atol=1e-4)


def _make_complex_backward_sources(sparse_left):
    rows, inner, cols = (3, 2048, 513) if sparse_left else (513, 2048, 2)
    left = torch.zeros((rows, inner), dtype=torch.complex64)
    right = torch.zeros((inner, cols), dtype=torch.complex64)
    for index in range(2048):
        if sparse_left:
            left_row, left_col = index % rows, index // rows
            right_row, right_col = index // cols, index % cols
        else:
            left_row, left_col = index % rows, index // rows
            right_row, right_col = index // 2, index % 2
        left[left_row, left_col] = complex(1.0 + index * 0.001, -0.5 + index * 0.0005)
        right[right_row, right_col] = complex(-0.25 + index * 0.0007, 0.75 + index * 0.0003)
    sampled_width = cols if sparse_left else rows
    return rows, cols, left, right, sampled_width


def _apply_complex_backward_view(base_left, base_right, view_kind):
    if view_kind == "conj":
        return base_left.conj(), base_right.conj()
    if view_kind == "neg":
        return -base_left, -base_right
    if view_kind == "strided":
        rows, inner = base_left.shape
        cols = base_right.shape[1]
        left_store = torch.empty((rows, inner * 2), dtype=torch.complex64, device=DEVICE)
        right_store = torch.empty((inner * 2, cols), dtype=torch.complex64, device=DEVICE)
        left_store[:, ::2].copy_(base_left)
        right_store[::2].copy_(base_right)
        return left_store[:, ::2], right_store[::2]
    return base_left, base_right


def _sparse_grad_values(sparse_grad, dense_expected):
    crow = sparse_grad.crow_indices().cpu()
    row_indices = torch.repeat_interleave(torch.arange(dense_expected.size(0)), crow[1:] - crow[:-1])
    col_indices = sparse_grad.col_indices().cpu()
    return dense_expected.cpu()[row_indices, col_indices]


@pytest.mark.parametrize("sparse_left", [True, False])
@pytest.mark.parametrize("view_kind", ["plain", "conj", "neg", "strided"])
def test_complex_sparse_backward_views_and_cross_chunk(sparse_left, view_kind):
    """Exercise complex row sampling through logical views and multiple chunks."""
    rows, cols, left_cpu_source, right_cpu_source, sampled_width = (
        _make_complex_backward_sources(sparse_left))
    chunk_limit = (1 << 20) // sampled_width
    assert int((left_cpu_source != 0).sum()) == 2048
    assert int((right_cpu_source != 0).sum()) == 2048
    assert 2048 > chunk_limit
    base_left = left_cpu_source.to(DEVICE)
    base_right = right_cpu_source.to(DEVICE)
    base_left, base_right = _apply_complex_backward_view(base_left, base_right, view_kind)
    left_cpu = base_left.detach().resolve_conj().resolve_neg().cpu().to(torch.complex128)
    right_cpu = base_right.detach().resolve_conj().resolve_neg().cpu().to(torch.complex128)
    left = base_left.detach().requires_grad_(not sparse_left)
    right = base_right.detach().requires_grad_(sparse_left)
    if sparse_left:
        left = left.to_sparse_csr().requires_grad_()
    else:
        right = right.to_sparse_csr().requires_grad_()
    gradient_row = [1.0 + 0.0j] * cols
    gradient = torch.tensor([gradient_row] * rows, dtype=torch.complex64, device=DEVICE)
    result = torch.sparse.mm(left, right) if sparse_left else torch.mm(left, right)
    result.backward(gradient)
    torch.npu.synchronize()
    gradient_cpu = gradient.cpu().to(torch.complex128)
    expected_left = (gradient_cpu @ right_cpu.mH).to(torch.complex64)
    expected_right = (left_cpu.mH @ gradient_cpu).to(torch.complex64)
    actual_left = left.grad.values().cpu() if sparse_left else left.grad.cpu()
    actual_right = right.grad.cpu()
    expected_left_values = (_sparse_grad_values(left.grad, expected_left)
                            if sparse_left else expected_left.cpu())
    torch.testing.assert_close(actual_left, expected_left_values, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual_right, expected_right.cpu(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_sparse_addmm_backward_real_dtypes(dtype):
    alpha, beta = 1.5, -0.5
    crow = torch.tensor([0, 2, 3], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([0, 2, 1], dtype=torch.int32, device=DEVICE)
    values = torch.tensor([1.5, 2.0, -0.5], dtype=dtype, device=DEVICE)
    mat1 = torch.sparse_csr_tensor(crow, col, values, size=(2, 3), device=DEVICE)
    mat1.requires_grad_()
    mat2 = torch.tensor([[2.0, 3.0], [4.0, 0.25], [1.0, -2.0]],
                        dtype=dtype, device=DEVICE, requires_grad=True)
    bias = torch.tensor([2.0, 1.0], dtype=dtype, device=DEVICE, requires_grad=True)
    gradient = torch.tensor([[0.2, 1.3], [-0.7, 2.0]], dtype=dtype, device=DEVICE)

    torch.sparse.addmm(bias, mat1, mat2, alpha=alpha, beta=beta).backward(gradient)
    torch.npu.synchronize()

    a_cpu = torch.tensor([[1.5, 0, 2.0], [0, -0.5, 0]], dtype=torch.float32)
    b_cpu = mat2.detach().float().cpu()
    g_cpu = gradient.float().cpu()
    expected_a = (alpha * (g_cpu @ b_cpu.t())).to(dtype)
    expected_b = (alpha * (a_cpu.t() @ g_cpu)).to(dtype)
    expected_bias = (beta * g_cpu.sum(0)).to(dtype)
    atol, rtol, _ = TOLERANCE[dtype]
    torch.testing.assert_close(mat1.grad.values().cpu(), expected_a[[0, 0, 1], [0, 2, 1]],
                               rtol=rtol, atol=atol)
    torch.testing.assert_close(mat2.grad.cpu(), expected_b, rtol=rtol, atol=atol)
    torch.testing.assert_close(bias.grad.cpu(), expected_bias, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", (torch.float32, torch.complex64))
def test_dense_left_sparse_right_and_noncontiguous_out(dtype):
    left = torch.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]], dtype=dtype, device=DEVICE)
    crow = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([0, 1, 0], dtype=torch.int32, device=DEVICE)
    values = torch.tensor([2.0, -1.0, 0.5], dtype=dtype, device=DEVICE)
    right = torch.sparse_csr_tensor(crow, col, values, size=(3, 2), device=DEVICE)
    bias = torch.full((2, 2), 0.25, dtype=dtype, device=DEVICE)
    storage = torch.empty((2, 4), dtype=dtype, device=DEVICE)
    out = storage[:, ::2]

    result = torch.addmm(bias, left, right, beta=2, alpha=3, out=out)

    wide = GOLDEN_DTYPE[dtype]
    right_dense = torch.tensor([[2.0, 0.0], [0.0, -1.0], [0.5, 0.0]], dtype=wide)
    expected = 2 * bias.cpu().to(wide) + 3 * (left.cpu().to(wide) @ right_dense)
    assert result is out
    assert not out.is_contiguous()
    assert_close(out.cpu(), expected, dtype)


def test_dense_left_sparse_right_functional_output_is_contiguous():
    left = torch.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]], device=DEVICE)
    right = torch.tensor([[2.0, 0.0], [0.0, -1.0], [0.5, 0.0]], device=DEVICE).to_sparse_csr()
    result = torch.mm(left, right)
    assert result.is_contiguous()
    torch.testing.assert_close(result.cpu(), left.cpu() @ right.cpu().to_dense())


def test_import_does_not_trigger_jit(tmp_path):
    """导入包不应触发 JIT 编译（规范 3.1）。"""
    cache = tmp_path / "extensions"
    cache.mkdir()
    environment = dict(os.environ, TORCH_EXTENSIONS_DIR=str(cache))
    script = (
        "import torch, torch_npu, cann_ops_sparse\n"
        "from cann_ops_sparse.ops.sparse.spmm.spmm import _spmm_op_builder as b\n"
        "print('LOADED:', list(type(b)._loaded_ops.keys()))\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], env=environment,
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    assert "LOADED: []" in completed.stdout
    assert not any(cache.iterdir()), "importing the package produced build artifacts"


# ---------------------------------------------------------------------------
# 2. 功能与精度
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize("input_shape", [(2, 2), (2,), (1, 2), (2, 1), ()])
def test_dtypes_and_broadcast_shapes(dtype, input_shape):
    alpha = complex(1.5, 0.25) if dtype.is_complex else 1.5
    beta = complex(-0.5, -0.125) if dtype.is_complex else -0.5
    bias, mat1, mat2 = make_operands(dtype, input_shape=input_shape)
    actual = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta).cpu()
    assert_close(actual, cpu_golden(dtype, input_shape, alpha, beta), dtype)


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_noncontiguous_dense_operands(dtype):
    bias, mat1, mat2 = make_operands(dtype, noncontiguous=True)
    actual = torch.addmm(bias, mat1, mat2).cpu()
    assert_close(actual, cpu_golden(dtype, (2, 2), 1.0, 1.0), dtype)


def test_addmm_out_resolves_contiguous_mat2_lazy_conjugate():
    """addmm.out must read the logical values of a contiguous B.conj() view."""
    crow = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    # torch_npu 的 aclnnInplaceOne 不支持 complex64，显式构造相同的单位值。
    values = torch.tensor([complex(1.0, 0.0), complex(1.0, 0.0)],
                          dtype=torch.complex64, device=DEVICE)
    mat1 = torch.sparse_csr_tensor(crow, col, values, size=(2, 2), device=DEVICE)
    mat2_base = torch.tensor(
        [[complex(1.0, 2.0), complex(3.0, 4.0)],
         [complex(-5.0, 6.0), complex(7.0, -8.0)]],
        dtype=torch.complex64,
        device=DEVICE,
    )
    mat2 = mat2_base.conj()
    assert mat2.is_contiguous() and mat2.is_conj()

    bias = torch.zeros((2, 2), dtype=torch.complex64, device=DEVICE)
    out = torch.empty_like(bias)
    SPARSE_ADDMM.out(bias, mat1, mat2, beta=0, alpha=1, out=out)

    expected = torch.tensor(
        [[complex(1.0, -2.0), complex(3.0, -4.0)],
         [complex(-5.0, -6.0), complex(7.0, 8.0)]],
        dtype=torch.complex64,
    )
    assert_close(out.cpu(), expected, torch.complex64)


@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
def test_both_index_dtypes(index_dtype):
    crow = torch.tensor([0, 2, 3], dtype=index_dtype, device=DEVICE)
    col = torch.tensor([0, 2, 1], dtype=index_dtype, device=DEVICE)
    values = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32, device=DEVICE)
    mat1 = torch.sparse_csr_tensor(crow, col, values, size=(2, 3), device=DEVICE)
    _, _, mat2 = make_operands(torch.float32)
    bias = torch.full((2, 2), 0.125, dtype=torch.float32, device=DEVICE)
    actual = torch.addmm(bias, mat1, mat2).cpu()
    assert_close(actual, cpu_golden(torch.float32, (2, 2), 1.0, 1.0), torch.float32)


@pytest.mark.parametrize("m,k,n", [(0, 3, 4), (2, 0, 4), (2, 3, 0)])
def test_zero_length_dimensions(m, k, n):
    """零维输入按 C = beta * C 返回，与 README/arch22 host 行为一致。"""
    crow = torch.zeros(m + 1, dtype=torch.int32, device=DEVICE)
    col = torch.empty(0, dtype=torch.int32, device=DEVICE)
    values = torch.empty(0, dtype=torch.float32, device=DEVICE)
    mat1 = torch.sparse_csr_tensor(crow, col, values, size=(m, k), device=DEVICE)
    mat2 = torch.empty((k, n), dtype=torch.float32, device=DEVICE)
    bias = torch.ones((m, n), dtype=torch.float32, device=DEVICE)
    actual = torch.addmm(bias, mat1, mat2, beta=-0.5)
    assert tuple(actual.shape) == (m, n)
    assert torch.equal(actual.cpu(), torch.full((m, n), -0.5))


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_beta_zero_does_not_propagate_input_specials(dtype):
    """beta=0 时 self 里的 NaN/Inf 不得渗进结果。"""
    bias, mat1, mat2 = make_operands(dtype)
    bias[0, 0] = torch.nan
    bias[-1, -1] = torch.inf
    alpha = complex(1.0, 0.25) if dtype.is_complex else 1.0
    actual = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=0).cpu()
    checked = torch.view_as_real(actual) if dtype.is_complex else actual
    assert bool(torch.isfinite(checked).all())
    assert_close(actual, cpu_golden(dtype, (2, 2), alpha, 0), dtype)


# ---------------------------------------------------------------------------
# 3. out= 别名行为
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_out_may_alias_mat2(dtype):
    crow = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    values = torch.tensor([2.0, 3.0], dtype=dtype, device=DEVICE)
    mat1 = torch.sparse_csr_tensor(crow, col, values, size=(2, 2), device=DEVICE)
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, device=DEVICE)
    bias = torch.full((2, 2), 0.125, dtype=dtype, device=DEVICE)
    SPARSE_ADDMM.out(bias, mat1, mat2, beta=1, alpha=1, out=mat2)
    expected = torch.tensor([[2.125, 4.125], [9.125, 12.125]], dtype=GOLDEN_DTYPE[dtype])
    assert_close(mat2.cpu(), expected, dtype)


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize("input_shape", [(2, 2), (2,), ()])
def test_out_may_alias_self(dtype, input_shape):
    bias, mat1, mat2 = make_operands(dtype, input_shape=input_shape)
    SPARSE_ADDMM.out(bias, mat1, mat2, beta=1, alpha=1, out=bias)
    assert_close(bias.cpu(), cpu_golden(dtype, input_shape, 1.0, 1.0), dtype)


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_out_may_alias_mat1_values(dtype):
    # out = mat1.values().view(2,2) 与 CSR 值共享存储：PrepareOutput 会先写 out，
    # 若不先快照 CSR 值，稀疏值会被清空后再读，结果变全零而非原 A。bias 用一维广播
    # 以令 separateOutput=false，确保 PrepareOutput 真的会写 out。
    crow = torch.tensor([0, 2, 4], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=DEVICE)
    mat1 = torch.sparse_csr_tensor(crow, col, values, size=(2, 2), device=DEVICE)
    mat2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=dtype, device=DEVICE)
    bias = torch.zeros(2, dtype=dtype, device=DEVICE)
    out = mat1.values().view(2, 2)
    SPARSE_ADDMM.out(bias, mat1, mat2, beta=1, alpha=1, out=out)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=GOLDEN_DTYPE[dtype])
    assert_close(out.cpu(), expected, dtype)


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_out_may_alias_mat1_values_nonidentity_b(dtype):
    # 非单位阵 B + alpha=2 验证快照保留的是真实 CSR 值（而非仅"非零"）。
    crow = torch.tensor([0, 2, 4], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=DEVICE)
    mat1 = torch.sparse_csr_tensor(crow, col, values, size=(2, 2), device=DEVICE)
    mat2 = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=dtype, device=DEVICE)
    bias = torch.zeros(2, dtype=dtype, device=DEVICE)
    out = mat1.values().view(2, 2)
    SPARSE_ADDMM.out(bias, mat1, mat2, beta=1, alpha=2, out=out)
    # A=[[1,2],[3,4]]，A@B=[[3,5],[8,9]]，结果 = 2*(A@B)。
    expected = torch.tensor([[6.0, 10.0], [16.0, 18.0]], dtype=GOLDEN_DTYPE[dtype])
    assert_close(out.cpu(), expected, dtype)


def test_out_may_alias_mat1_col_indices():
    # out 通过 view(float32) 位重解释 col_indices 的 int32 存储（同为 4 字节），与列索引
    # 共享存储：PrepareOutput 写 out 会清掉尚未读取的 CSR 列索引，若不同步快照 crow/col，
    # 矩阵会被误解为所有非零元都在第 0 列，结果从 [[1,2],[3,4]] 变 [[3,0],[7,0]]。
    crow = torch.tensor([0, 2, 4], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=DEVICE)
    mat1 = torch.sparse_csr_tensor(crow, col, values, size=(2, 2), device=DEVICE)
    mat2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=DEVICE)
    bias = torch.zeros(2, dtype=torch.float32, device=DEVICE)
    out = mat1.col_indices().view(torch.float32).view(2, 2)
    SPARSE_ADDMM.out(bias, mat1, mat2, beta=1, alpha=1, out=out)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    assert_close(out.cpu(), expected, torch.float32)


def test_out_may_alias_mat1_crow_indices():
    # 同列索引，但针对 crow_indices：1×2 CSR 的 row offset 有 2 个 int32（8 字节），
    # 位重解释成 float32 后可 view 成 1×2，与行偏移共享存储。
    crow = torch.tensor([0, 2], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    values = torch.tensor([1.0, 2.0], dtype=torch.float32, device=DEVICE)
    mat1 = torch.sparse_csr_tensor(crow, col, values, size=(1, 2), device=DEVICE)
    mat2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=DEVICE)
    bias = torch.zeros(2, dtype=torch.float32, device=DEVICE)
    out = mat1.crow_indices().view(torch.float32).view(1, 2)
    SPARSE_ADDMM.out(bias, mat1, mat2, beta=1, alpha=1, out=out)
    expected = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    assert_close(out.cpu(), expected, torch.float32)


@pytest.mark.parametrize("operation", ("mm", "addmm"))
def test_out_resizes_wrong_shaped_noncontiguous_view(operation):
    bias, mat1, mat2 = make_operands(torch.float32)
    storage = torch.empty((1, 4), dtype=torch.float32, device=DEVICE)
    out = storage[:, ::2]
    data_ptr = out.data_ptr()

    if operation == "mm":
        result = torch.mm(mat1, mat2, out=out)
        expected = cpu_golden(torch.float32, (2, 2), 1, 0)
    else:
        result = torch.addmm(bias, mat1, mat2, beta=2, alpha=3, out=out)
        expected = cpu_golden(torch.float32, (2, 2), 3, 2)

    assert result is out
    assert tuple(out.shape) == (2, 2)
    assert out.data_ptr() == data_ptr == storage.data_ptr()
    assert_close(out.cpu(), expected, torch.float32)
    assert_close(storage.cpu().reshape(2, 2), expected, torch.float32)


@pytest.mark.parametrize("operation", ("mm", "addmm"))
@pytest.mark.parametrize("transpose", (False, True))
def test_complex_out_preserves_lazy_conjugate_view(operation, transpose):
    crow = torch.tensor([0, 2, 3], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([0, 2, 1], dtype=torch.int32, device=DEVICE)
    values_cpu = torch.tensor([1 + 0.5j, 2 - 0.25j, -1 + 0.75j], dtype=torch.complex64)
    mat2_cpu = torch.tensor(
        [[1 + 2j, -2 + 0.5j], [0.5 - 1j, 3 + 2j], [-1 + 0.25j, 4 - 3j]],
        dtype=torch.complex64,
    )
    bias_cpu = torch.tensor(
        [[0.125 + 0.5j, -0.25 + 0.75j], [1 - 0.5j, -0.75 - 0.25j]],
        dtype=torch.complex64,
    )
    mat1 = torch.sparse_csr_tensor(crow, col, values_cpu.to(DEVICE), size=(2, 3), device=DEVICE)
    mat2 = mat2_cpu.to(DEVICE)
    bias = bias_cpu.to(DEVICE)
    storage = torch.empty((2, 2), dtype=torch.complex64, device=DEVICE)
    out = storage.t().conj() if transpose else storage.conj()

    mat1_cpu = torch.tensor(
        [[values_cpu[0], 0, values_cpu[1]], [0, values_cpu[2], 0]],
        dtype=torch.complex64,
    )

    if operation == "mm":
        result = torch.mm(mat1, mat2, out=out)
        expected = mat1_cpu @ mat2_cpu
    else:
        result = torch.addmm(bias, mat1, mat2, beta=2, alpha=3, out=out)
        expected = 3 * (mat1_cpu @ mat2_cpu) + 2 * bias_cpu

    assert result is out
    assert out.is_conj()
    assert out.is_contiguous() is not transpose
    assert out.data_ptr() == storage.data_ptr()
    assert_close(out.cpu(), expected, torch.complex64)
    stored_expected = expected.conj().t() if transpose else expected.conj()
    assert_close(storage.cpu(), stored_expected, torch.complex64)


# ---------------------------------------------------------------------------
# 4. 非法输入
# ---------------------------------------------------------------------------

def test_mismatched_out_dtype_is_rejected():
    bias, mat1, mat2 = make_operands(torch.float32)
    out = torch.empty((2, 2), dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError, match="out dtype must match|Expected out tensor to have dtype"):
        SPARSE_ADDMM.out(bias, mat1, mat2, beta=0, alpha=1, out=out)


def test_mismatched_input_dtypes_are_rejected():
    bias, mat1, _ = make_operands(torch.float32)
    mat2 = torch.ones((3, 2), dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError, match="share (one )?dtype|expected .* to have the same dtype"):
        torch.addmm(bias, mat1, mat2)


def test_mismatched_shapes_are_rejected():
    bias, mat1, _ = make_operands(torch.float32)
    mat2 = torch.ones((4, 2), dtype=torch.float32, device=DEVICE)
    with pytest.raises(RuntimeError, match="must equal mat2.size\\(0\\)|cannot be multiplied"):
        torch.addmm(bias, mat1, mat2)


def test_int64_indices_must_fit_int32():
    crow = torch.tensor([0, 1, 1], dtype=torch.int64, device=DEVICE)
    col = torch.tensor([2**32], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    mat1 = torch.sparse_csr_tensor(crow, col, values, size=(2, 3), device=DEVICE,
                                   check_invariants=False)
    bias = torch.zeros((2, 2), dtype=torch.float32, device=DEVICE)
    mat2 = torch.ones((3, 2), dtype=torch.float32, device=DEVICE)
    with pytest.raises(RuntimeError, match="column-index values must fit signed int32"):
        torch.addmm(bias, mat1, mat2)


def test_complex_scalar_on_real_dtype_is_rejected():
    bias, mat1, mat2 = make_operands(torch.float32)
    with pytest.raises(RuntimeError, match="non-zero imaginary part"):
        torch.addmm(bias, mat1, mat2, alpha=complex(1.0, 1.0))


# ---------------------------------------------------------------------------
# 5. 设备与异步
# ---------------------------------------------------------------------------

@pytest.mark.skipif(torch.npu.device_count() < 2, reason="two NPUs are required")
def test_cross_device_inputs_are_rejected():
    bias, mat1, _ = make_operands(torch.float32, device="npu:0")
    _, _, mat2 = make_operands(torch.float32, device="npu:1")
    with pytest.raises(RuntimeError, match=r"must share dtype and device"):
        torch.addmm(bias, mat1, mat2)


@pytest.mark.skipif(torch.npu.device_count() < 2, reason="two NPUs are required")
def test_runs_on_non_default_device():
    bias, mat1, mat2 = make_operands(torch.float32, device="npu:1")
    actual = torch.addmm(bias, mat1, mat2)
    torch.npu.synchronize()
    assert actual.device == bias.device
    assert_close(actual.cpu(), cpu_golden(torch.float32, (2, 2), 1.0, 1.0), torch.float32)


def test_runs_on_non_default_stream():
    bias, mat1, mat2 = make_operands(torch.float32)
    stream = torch_npu.npu.Stream(device=DEVICE)
    with torch_npu.npu.stream(stream):
        actual = torch.addmm(bias, mat1, mat2)
    stream.synchronize()
    assert_close(actual.cpu(), cpu_golden(torch.float32, (2, 2), 1.0, 1.0), torch.float32)


def test_repeated_calls_are_consistent():
    bias, mat1, mat2 = make_operands(torch.float32)
    golden = cpu_golden(torch.float32, (2, 2), 1.0, 1.0)
    for _ in range(8):
        assert_close(torch.addmm(bias, mat1, mat2).cpu(), golden, torch.float32)
