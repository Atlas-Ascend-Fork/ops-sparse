# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Exercise registered autograd callbacks without a device-specific backend."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

from test_matmul_dispatch import router
from test_spgemm_backward import backward


_PACKAGE = types.ModuleType("_matmul_autograd_unit")
_PACKAGE.__path__ = []
sys.modules[_PACKAGE.__name__] = _PACKAGE
sys.modules[_PACKAGE.__name__ + ".dispatch"] = router
_PATH = Path(__file__).resolve().parents[3] / "torch_extension/cann_ops_sparse/ops/matmul/autograd.py"
_SPEC = importlib.util.spec_from_file_location(_PACKAGE.__name__ + ".autograd", _PATH)
autograd = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = autograd
_SPEC.loader.exec_module(autograd)


class RecordingLibrary:
    """Capture the same implementation callbacks installed into ATen."""

    def __init__(self):
        self.implementations = {}

    def impl(self, schema, implementation, dispatch_key, *, with_keyset=False):
        assert dispatch_key == "AutogradPrivateUse1"
        assert with_keyset
        self.implementations[schema] = implementation


def _forward(request):
    if request.operation == "mm":
        return request.mat1.to_dense().mm(request.mat2)
    return torch.addmm(request.self, request.mat1.to_dense(), request.mat2,
                       beta=request.beta, alpha=request.alpha)


def _backward(request, gradient):
    assert request.mat1.layout == torch.sparse_coo
    scale = complex(request.alpha).conjugate() if request.mat1.is_complex() else request.alpha
    grad_mat1 = (scale * gradient.mm(request.mat2.mH)).sparse_mask(request.mat1)
    grad_mat2 = scale * request.mat1.to_dense().mH.mm(gradient)
    grad_self = None
    if request.self is not None:
        beta = complex(request.beta).conjugate() if request.self.is_complex() else request.beta
        grad_self = beta * gradient.sum(0)
    return grad_self, grad_mat1, grad_mat2


@pytest.mark.parametrize("operation", ["mm", "addmm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_registered_callback_preserves_tensor_gradients(operation, dtype):
    registry = router.MatmulRegistry()
    registry.register(router.MatmulCapability("cpu-vjp", "spmm", _forward, backward=_backward))
    library = RecordingLibrary()
    autograd.register_autograd(library, registry)
    dense_a = torch.tensor([[2., 0.], [0., 3.]], dtype=dtype)
    dense_b = torch.tensor([[1., 4.], [2., 5.]], dtype=dtype)
    if dtype.is_complex:
        dense_a = dense_a + 0.5j * dense_a
        dense_b = dense_b - 0.2j * dense_b
    a = dense_a.to_sparse_coo().detach().requires_grad_()
    b = dense_b.detach().requires_grad_()
    accumulated = torch.tensor([1., 2.], dtype=dtype, requires_grad=True)
    reference_a = dense_a.detach().requires_grad_()
    reference_b = dense_b.detach().requires_grad_()
    reference_self = accumulated.detach().clone().requires_grad_()
    if operation == "mm":
        result = library.implementations[operation](None, a, b)
        expected = reference_a.mm(reference_b)
    else:
        alpha, beta = (1.2 - 0.7j, 0.5 + 0.2j) if dtype.is_complex else (3., 2.)
        result = library.implementations[operation](None, accumulated, a, b, alpha=alpha, beta=beta)
        expected = torch.addmm(reference_self, reference_a, reference_b, alpha=alpha, beta=beta)
    gradient = torch.tensor([[0.2, 1.3], [-0.7, 2.0]], dtype=dtype)
    result.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(a.grad.coalesce().values(), reference_a.grad.diag())
    torch.testing.assert_close(b.grad, reference_b.grad)
    if operation == "addmm":
        torch.testing.assert_close(accumulated.grad, reference_self.grad)


def test_registered_dense_callback_retains_native_autograd():
    library = RecordingLibrary()
    autograd.register_autograd(library, router.MatmulRegistry())
    a = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    b = torch.tensor([[2., 0.], [1., 3.]], requires_grad=True)
    keys = torch.DispatchKeySet(torch.DispatchKey.CPU)
    result = library.implementations["mm"](keys, a, b)
    result.sum().backward()
    torch.testing.assert_close(a.grad, torch.tensor([[2., 4.], [2., 4.]]))
    torch.testing.assert_close(b.grad, torch.tensor([[4., 4.], [6., 6.]]))


def _spgemm_forward(request):
    # The local CPU build lacks MKL for CSR x CSR. Use native COO forward
    # while retaining original CSR inputs in the registered backward wrapper.
    result = torch.mm(request.mat1.to_sparse_coo(), request.mat2.to_sparse_coo())
    if request.operation == "addmm":
        result = (request.alpha * result.to_dense() + request.beta * request.self.to_dense()).to_sparse_coo()
    return result.to_sparse_csr()


def _spgemm_backward(request, gradient):
    return backward.spgemm_backward(request, gradient, lambda item: torch.mm(item.mat1, item.mat2))


def _spgemm_library():
    registry = router.MatmulRegistry()
    registry.register(router.MatmulCapability(
        "cpu-spgemm-vjp", "spgemm", _spgemm_forward, backward=_spgemm_backward))
    library = RecordingLibrary()
    autograd.register_autograd(library, registry)
    return library


@pytest.mark.parametrize("layout", [torch.strided, torch.sparse_coo, torch.sparse_csr])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_registered_csr_mm_gradient_retains_outside_pattern(dtype, layout):
    a_dense = torch.tensor([[2., 0.], [0., 3.]], dtype=dtype)
    b_dense = torch.tensor([[1., 4.], [2., 5.]], dtype=dtype)
    if dtype.is_complex:
        b_dense = b_dense + 0.5j * b_dense
    a, b = a_dense.to_sparse_csr().requires_grad_(), b_dense.to_sparse_csr().requires_grad_()
    reference_a = a_dense.to_sparse_coo().requires_grad_()
    reference_b = b_dense.to_sparse_coo().requires_grad_()
    gradient = torch.ones((2, 2), dtype=dtype)
    if layout != torch.strided:
        gradient = gradient.to_sparse_coo()
    native_gradient = gradient
    if layout == torch.sparse_csr:
        gradient = gradient.to_sparse_csr()
    expected = torch.autograd.grad(torch.mm(reference_a, reference_b), (reference_a, reference_b), native_gradient)
    _spgemm_library().implementations["mm"](None, a, b).backward(gradient)
    for actual, reference in zip((a.grad, b.grad), expected):
        if layout == torch.sparse_csr:
            reference = reference.to_sparse_csr()
            torch.testing.assert_close(actual.crow_indices(), reference.crow_indices())
            torch.testing.assert_close(actual.col_indices(), reference.col_indices())
            torch.testing.assert_close(actual.values(), reference.values())
        assert actual.layout == reference.layout == layout
        torch.testing.assert_close(actual.to_dense(), reference.to_dense())
    gradient_dense = a.grad.to_dense()
    assert gradient_dense[0, 1] != 0 and gradient_dense[1, 0] != 0


@pytest.mark.parametrize("schema", ["addmm", "_sparse_addmm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("layout", [torch.strided, torch.sparse_coo, torch.sparse_csr])
def test_registered_csr_addmm_self_gradient_preserves_upstream(schema, dtype, layout):
    accumulated = torch.eye(2, dtype=dtype).to_sparse_csr().requires_grad_()
    a = torch.tensor([[1., 2.], [3., 4.]], dtype=dtype).to_sparse_csr()
    b = torch.eye(2, dtype=dtype).to_sparse_csr()
    beta = 2 + 3j if dtype.is_complex else 2
    gradient = torch.tensor([[1., 7.], [11., 1.]], dtype=dtype)
    if layout == torch.sparse_coo:
        gradient = gradient.to_sparse_coo()
    elif layout == torch.sparse_csr:
        gradient = gradient.to_sparse_csr()
    # Use the native schema with supported COO x dense forward as reference:
    # only self requires gradients, so the matrix derivatives are irrelevant.
    reference_self = torch.eye(2, dtype=dtype, requires_grad=True)
    native = torch.addmm if schema == "addmm" else torch.sparse.addmm
    reference = native(reference_self, a.to_sparse_coo(), b.to_dense(), beta=beta)
    # A dense output rejects CSR upstream in native autograd. Compare its
    # numeric derivative with dense G, then preserve the extension's CSR G.
    native_gradient = gradient.to_dense() if layout == torch.sparse_csr else gradient
    expected = torch.autograd.grad(reference, reference_self, native_gradient)[0]
    if layout == torch.sparse_csr:
        expected = expected.to_sparse_csr()
    actual = _spgemm_library().implementations[schema](None, accumulated, a, b, beta=beta)
    actual.backward(gradient)
    assert a.grad is None and b.grad is None
    assert accumulated.grad.layout == expected.layout == gradient.layout
    torch.testing.assert_close(accumulated.grad.to_dense(), expected.to_dense())
    if layout == torch.sparse_csr:
        torch.testing.assert_close(accumulated.grad.col_indices(), gradient.col_indices())
        torch.testing.assert_close(accumulated.grad.crow_indices(), gradient.crow_indices())
    elif layout == torch.sparse_coo:
        torch.testing.assert_close(accumulated.grad.coalesce().indices(), gradient.indices())


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("layout", [torch.strided, torch.sparse_coo, torch.sparse_csr])
def test_registered_csr_addmm_matrix_gradients_are_complete(dtype, layout):
    a_dense = torch.diag(torch.tensor([2., 3.], dtype=dtype))
    b_dense = torch.tensor([[1., 4.], [2., 5.]], dtype=dtype)
    alpha = 3 + 2j if dtype.is_complex else 3
    a = a_dense.to_sparse_csr().requires_grad_()
    b = b_dense.to_sparse_csr().requires_grad_()
    accumulated = torch.eye(2, dtype=dtype).to_sparse_csr()
    gradient = torch.ones((2, 2), dtype=dtype)
    if layout != torch.strided:
        gradient = gradient.to_sparse_coo()
    native_gradient = gradient
    if layout == torch.sparse_csr:
        gradient = gradient.to_sparse_csr()
    # Native addmm uses the same unprojected matrix derivatives as mm.
    # COO mm avoids this CPU build's unavailable CSR x CSR forward kernel.
    reference_a = a_dense.to_sparse_coo().requires_grad_()
    reference_b = b_dense.to_sparse_coo().requires_grad_()
    expected = torch.autograd.grad(alpha * torch.mm(reference_a, reference_b),
                                   (reference_a, reference_b), native_gradient)
    result = _spgemm_library().implementations["addmm"](
        None, accumulated, a, b, alpha=alpha)
    result.backward(gradient)
    for actual, reference in zip((a.grad, b.grad), expected):
        if layout == torch.sparse_csr:
            reference = reference.to_sparse_csr()
            torch.testing.assert_close(actual.crow_indices(), reference.crow_indices())
            torch.testing.assert_close(actual.col_indices(), reference.col_indices())
            torch.testing.assert_close(actual.values(), reference.values())
        assert actual.layout == reference.layout == layout
        torch.testing.assert_close(actual.to_dense(), reference.to_dense())
    if not dtype.is_complex:
        torch.testing.assert_close(a.grad.to_dense(), torch.tensor([[15., 21.], [15., 21.]]))
    assert a.grad.to_dense()[0, 1] != 0 and a.grad.to_dense()[1, 0] != 0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_registered_csr_addmm_zero_upstream_preserves_nonfinite(dtype, value):
    a_dense = torch.diag(torch.tensor([2., 3.], dtype=dtype))
    b_dense = torch.diag(torch.tensor([value, 5.], dtype=dtype))
    a = a_dense.to_sparse_csr().requires_grad_()
    b = b_dense.to_sparse_csr().requires_grad_()
    accumulated = torch.eye(2, dtype=dtype).to_sparse_csr()
    reference_a = a_dense.clone().requires_grad_()
    reference_b = b_dense.clone().requires_grad_()
    gradient = torch.zeros((2, 2), dtype=dtype)
    expected = torch.autograd.grad(torch.addmm(torch.eye(2, dtype=dtype), reference_a, reference_b, alpha=3),
                                   (reference_a, reference_b), gradient)
    result = _spgemm_library().implementations["addmm"](None, accumulated, a, b, alpha=3)
    result.backward(gradient)
    for actual, reference in zip((a.grad, b.grad), expected):
        assert actual.layout == reference.layout == torch.strided
        torch.testing.assert_close(actual, reference, equal_nan=True)
