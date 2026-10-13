# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CPU contract tests for routing, independent of NPU/JIT availability."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


_PATH = Path(__file__).resolve().parents[3] / "torch_extension/cann_ops_sparse/ops/matmul/dispatch.py"
_SPEC = importlib.util.spec_from_file_location("_matrix_router_under_test", _PATH)
router = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = router
_SPEC.loader.exec_module(router)


def _spmm(request):
    # A CPU stand-in exercises the provider protocol, not the NPU kernel.
    if request.operation == "mm":
        result = torch.mm(request.mat1, request.mat2)
    else:
        result = torch.addmm(request.self, request.mat1, request.mat2,
                             alpha=request.alpha, beta=request.beta)
    if request.out is not None:
        request.out.copy_(result)
        return request.out
    return result


def _registry():
    registry = router.MatmulRegistry()
    registry.register(router.MatmulCapability("cpu-spmm", "spmm", _spmm))
    return registry


@pytest.mark.parametrize("left_sparse,right_sparse", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_dense_output_routes(left_sparse, right_sparse, dtype):
    a = torch.tensor([[1, 2, 0], [0, 3, 4]], dtype=dtype)
    b = torch.tensor([[2, 0], [1, 3], [0, 4]], dtype=dtype)
    if dtype.is_complex:
        a = a + 1j * a
        b = b - 2j * b
    left = a.to_sparse_csr() if left_sparse else a
    right = b.to_sparse_csr() if right_sparse else b
    registry = _registry()
    torch.testing.assert_close(registry.dispatch(router.MatmulRequest("mm", left, right)), a @ b)
    self = torch.tensor([1, 2], dtype=dtype)
    request = router.MatmulRequest("addmm", left, right, self, beta=2, alpha=3)
    torch.testing.assert_close(registry.dispatch(request), 2 * self + 3 * (a @ b))


def test_sparse_sparse_selects_spgemm_without_spmm():
    calls = []
    registry = router.MatmulRegistry()

    def gemm(request):
        calls.append(request)
        return (request.mat1.to_dense() @ request.mat2.to_dense()).to_sparse_csr()

    registry.register(router.MatmulCapability("cpu-spgemm", "spgemm", gemm))
    a = torch.eye(2).to_sparse_csr()
    result = registry.dispatch(router.MatmulRequest("mm", a, a))
    assert result.layout == torch.sparse_csr
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="spmm capability is not registered"):
        registry.dispatch(router.MatmulRequest("mm", a, torch.eye(2)))


def test_duplicate_capability_registration_is_atomic():
    registry = router.MatmulRegistry()
    registry.register(router.MatmulCapability("first", "spmm", _spmm, operations=("addmm",)))
    request = router.MatmulRequest("addmm", torch.eye(2), torch.eye(2).to_sparse_csr(), torch.ones(2))
    assert registry.capability_for(request) is registry.capabilities[("spmm", "addmm")]
    with pytest.raises(RuntimeError, match="duplicate"):
        registry.register(router.MatmulCapability("second", "spmm", _spmm))
    a = torch.eye(2).to_sparse_csr()
    with pytest.raises(RuntimeError, match="not registered for mm"):
        registry.dispatch(router.MatmulRequest("mm", a, torch.eye(2)))


def test_sparse_layout_rejections_precede_execution():
    registry = router.MatmulRegistry()
    registry.register(router.MatmulCapability("never-called", "spgemm",
                                             lambda r: pytest.fail("invalid request executed")))
    a = torch.eye(2).to_sparse_csr()
    with pytest.raises(RuntimeError, match="matching COO or CSR"):
        registry.dispatch(router.MatmulRequest("mm", a, a.to_sparse_coo()))
    with pytest.raises(RuntimeError, match="self must use the sparse CSR"):
        registry.dispatch(router.MatmulRequest("addmm", a, a, torch.zeros(2, 2)))
    bsr = torch.eye(2).to_sparse_bsr((1, 1))
    with pytest.raises(RuntimeError, match="unsupported matrix layout"):
        registry.dispatch(router.MatmulRequest("mm", bsr, a))


def test_coo_duplicates_are_coalesced_for_spmm():
    a = torch.sparse_coo_tensor([[0, 0, 1], [0, 0, 1]], [2., 3., 4.], (2, 2))
    result = _registry().dispatch(router.MatmulRequest("mm", a, torch.eye(2)))
    torch.testing.assert_close(result, torch.diag(torch.tensor([5., 4.])))


def test_transposed_out_returns_original_tensor():
    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.eye(2).to_sparse_csr()
    out = torch.empty(2, 2)
    request = router.MatmulRequest("addmm", a, b, torch.ones(2), out=out)
    assert _registry().dispatch(request) is out
    torch.testing.assert_close(out, a + 1)


def test_transposed_out_resizes_empty_rectangular_output():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(12, dtype=torch.float32).reshape(3, 4).to_sparse_csr()
    out = torch.empty(0)
    request = router.MatmulRequest("mm", a, b, out=out)
    assert _registry().dispatch(request) is out
    torch.testing.assert_close(out, a @ b.to_dense())


def test_out_gradient_restriction():
    a = torch.eye(2, requires_grad=True)
    b = torch.eye(2).to_sparse_csr()
    with pytest.raises(RuntimeError, match="automatic differentiation"):
        _registry().dispatch(router.MatmulRequest("mm", a, b, out=torch.empty(2, 2)))


def test_normalization_preserves_first_order_gradients():
    values = torch.tensor([2., 3.], requires_grad=True)
    b = torch.sparse_coo_tensor([[0, 1], [0, 1]], values, (2, 2))
    a = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    self = torch.tensor([1., 2.], requires_grad=True)
    result = _registry().dispatch(router.MatmulRequest("addmm", a, b, self, beta=2, alpha=3))
    result.sum().backward()
    torch.testing.assert_close(a.grad, torch.tensor([[6., 9.], [6., 9.]]))
    torch.testing.assert_close(values.grad, torch.tensor([12., 18.]))
    torch.testing.assert_close(self.grad, torch.tensor([4., 4.]))


def test_backward_is_explicit_and_receives_original_operands():
    registry = _registry()
    a = torch.eye(2).to_sparse_csr()
    request = router.MatmulRequest("mm", torch.eye(2), a)
    with pytest.raises(RuntimeError, match="no custom backward"):
        registry.dispatch_backward(request, torch.ones(2, 2))
    registry = router.MatmulRegistry()

    def backward(original, grad):
        assert original is request
        return None, grad, grad

    registry.register(router.MatmulCapability("with-vjp", "spmm", _spmm, backward=backward))
    gradients = registry.dispatch_backward(request, torch.ones(2, 2))
    assert len(gradients) == 3
    with pytest.raises(RuntimeError, match="grad_output"):
        registry.dispatch_backward(request, torch.ones(3, 3))
