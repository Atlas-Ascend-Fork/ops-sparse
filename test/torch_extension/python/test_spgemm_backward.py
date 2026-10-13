# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Sparse VJP and projection contracts without an NPU dependency."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
from torch._C import _jit_get_operation as get_aten_operation

from test_matmul_dispatch import router


_ROOT = Path(__file__).resolve().parents[3]
for name in (
        "_spgemm_unit",
        "_spgemm_unit.ops",
        "_spgemm_unit.ops.matmul",
        "_spgemm_unit.ops.sparse",
        "_spgemm_unit.ops.sparse.spgemm",
):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
sys.modules["_spgemm_unit.ops.matmul.dispatch"] = router
_SPEC = importlib.util.spec_from_file_location(
    "_spgemm_unit.ops.sparse.spgemm.backward",
    _ROOT / "sparse/spgemm/torch_extension/spgemm_backward.py",
)
backward = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = backward
_SPEC.loader.exec_module(backward)
COO_INDICES, _ = get_aten_operation("aten::_indices")
COO_VALUES, _ = get_aten_operation("aten::_values")


def _components(tensor):
    """Return COO storage without coalescing duplicate coordinates."""
    return COO_INDICES(tensor), COO_VALUES(tensor)


def test_complex_conjugate_view_projection():
    dense = torch.tensor([[1 + 2j, 0], [0, 3 + 4j]]).conj()
    pattern = torch.eye(2, dtype=torch.complex64).to_sparse_coo()
    result = backward.sparse_mask(dense, pattern)
    torch.testing.assert_close(result.values(), torch.tensor([1 - 2j, 3 - 4j]))


def test_large_int32_sparse_pattern_keys():
    """Exercise the public VJP with wide coordinates and ordered projection."""
    mat1 = torch.sparse_coo_tensor(
        torch.tensor([[45000, 0], [45000, 1]], dtype=torch.int32),
        torch.tensor([2., 3.]),
        (50000, 50000),
    ).to_sparse_csr().requires_grad_()
    mat2 = mat1.detach().clone().requires_grad_()
    grad = torch.sparse_coo_tensor([[0], [0]], [1.], (50000, 50000)).coalesce()
    request = router.MatmulRequest("mm", mat1, mat2, schema="aten::_sparse_addmm")
    received = []
    product = torch.sparse_coo_tensor(
        [[0, 45000], [1, 45000]], [3., 2.], (50000, 50000)
    ).coalesce()

    def execute(item):
        received.append(item)
        return product

    actual_mat1, _ = backward.spgemm_backward(request, grad, execute)[1:]
    transposed_indices, _ = _components(received[0].mat2)
    torch.testing.assert_close(transposed_indices, torch.tensor([[1, 45000], [0, 45000]]))
    assert actual_mat1.layout == torch.sparse_csr
    result_indices, result_values = _components(actual_mat1.to_sparse_coo())
    torch.testing.assert_close(result_indices, torch.tensor([[0, 45000], [1, 45000]]))
    torch.testing.assert_close(result_values, torch.tensor([3., 2.]))


def test_sparse_mask_preserves_duplicate_pattern_order():
    source = torch.diag(torch.tensor([2., 3.])).to_sparse_coo()
    mask = torch.sparse_coo_tensor([[1, 0, 1], [1, 0, 1]], [0., 0., 0.], (2, 2))
    result = backward.sparse_mask(source, mask)
    assert not result.is_coalesced()
    result_indices, result_values = _components(result)
    mask_indices, _ = _components(mask)
    torch.testing.assert_close(result_indices, mask_indices)
    torch.testing.assert_close(result_values, torch.tensor([3., 2., 3.]))


def test_complex_duplicate_coo_values_are_summed():
    tensor = torch.sparse_coo_tensor(
        [[0, 0, 1], [0, 0, 1]], [1 + 2j, 3 - 1j, 2 + 3j], (2, 2)
    )
    result = backward.as_coo(tensor)
    torch.testing.assert_close(result.values(), torch.tensor([4 + 1j, 2 + 3j]))


def test_zero_scalars_preserve_native_nan_multiplication():
    a = torch.eye(2).to_sparse_csr().detach().requires_grad_()
    request = router.MatmulRequest("addmm", a, a, a, beta=0, alpha=0,
                                   schema="aten::_sparse_addmm")
    grad = torch.full((2, 2), float("nan"))
    calls = []

    def execute(request):
        calls.append(request)
        return torch.sparse_coo_tensor(
            [[0, 1], [0, 1]], [float("nan"), float("nan")], (2, 2)
        )

    gradients = backward.spgemm_backward(request, grad, execute)
    assert len(calls) == 2
    assert gradients[0].layout == torch.strided
    assert torch.isnan(gradients[0]).all()
    for tensor in gradients[1:]:
        assert torch.isnan(tensor.values()).all()


def _native_sparse_matmul_gradients(a, b, grad):
    # COO sparse.mm selects the native sparse-sparse schema under test.
    result = torch.sparse.mm(a, b)
    return torch.autograd.grad(result, (a, b), grad)


@pytest.mark.parametrize("input_has_more_entries", [False, True])
def test_sparse_sparse_backward_matches_native_projection_structure(input_has_more_entries):
    # The two branches of sparse_mask_like_grad select either sparse_mask or
    # _sparse_mask_projection, which intentionally produce different output
    # structures when the product and input patterns differ.
    if input_has_more_entries:
        a_indices, b_indices = [[0, 0, 1], [0, 1, 1]], [[0], [0]]
    else:
        a_indices, b_indices = [[0], [0]], [[0, 1, 2], [0, 1, 2]]
    a_values = torch.tensor([2., 3., 5.][:len(a_indices[0])])
    b_values = torch.tensor([7., 11., 13.][:len(b_indices[0])])
    a_native = torch.sparse_coo_tensor(a_indices, a_values, (3, 3)).coalesce().requires_grad_()
    b_native = torch.sparse_coo_tensor(b_indices, b_values, (3, 3)).coalesce().requires_grad_()
    grad = torch.sparse_coo_tensor([[0, 1], [0, 1]], [17., 19.], (3, 3)).coalesce()
    expected_a, expected_b = _native_sparse_matmul_gradients(a_native, b_native, grad)

    a = a_native.detach().requires_grad_()
    b = b_native.detach().requires_grad_()
    request = router.MatmulRequest("mm", a, b, schema="aten::_sparse_sparse_matmul")
    actual_a, actual_b = backward.spgemm_backward(
        request, grad, lambda item: torch.sparse.mm(item.mat1, item.mat2))[1:]
    for actual, expected in ((actual_a, expected_a), (actual_b, expected_b)):
        assert actual.is_coalesced() == expected.is_coalesced()
        actual_indices, actual_values = _components(actual)
        expected_indices, expected_values = _components(expected)
        torch.testing.assert_close(actual_indices, expected_indices)
        torch.testing.assert_close(actual_values, expected_values)


@pytest.mark.parametrize("dense_grad", [False, True])
def test_public_mm_backward_matches_native_unprojected_structure(dense_grad):
    a_native = torch.sparse_coo_tensor([[0, 0, 1], [0, 1, 1]], [2., 3., 5.], (3, 3)).coalesce().requires_grad_()
    b_native = torch.sparse_coo_tensor([[0], [0]], [7.], (3, 3)).coalesce().requires_grad_()
    grad = torch.sparse_coo_tensor([[0, 1], [0, 1]], [17., 19.], (3, 3)).coalesce()
    if dense_grad:
        grad = grad.to_dense()
    expected_gradients = torch.autograd.grad(torch.mm(a_native, b_native), (a_native, b_native), grad)

    a, b = a_native.detach().requires_grad_(), b_native.detach().requires_grad_()
    request = router.MatmulRequest("mm", a, b, schema="aten::mm")
    actual_a, actual_b = backward.spgemm_backward(
        request, grad, lambda item: torch.sparse.mm(item.mat1, item.mat2))[1:]
    for actual, expected in zip((actual_a, actual_b), expected_gradients):
        assert actual.layout == expected.layout
        if dense_grad:
            torch.testing.assert_close(actual, expected)
        else:
            assert actual.is_coalesced() == expected.is_coalesced()
            actual_indices, actual_values = _components(actual)
            expected_indices, expected_values = _components(expected)
            torch.testing.assert_close(actual_indices, expected_indices)
            torch.testing.assert_close(actual_values, expected_values)


def _cpu_sparse_execute(item):
    return torch.sparse.mm(item.mat1, item.mat2)


@pytest.mark.parametrize(
    ("schema", "native_addmm"),
    [("aten::_sparse_addmm", torch.sparse.addmm), ("aten::addmm", torch.addmm)],
)
def test_addmm_scalar_rules_match_native_complex_autograd(schema, native_addmm):
    alpha, beta = 2 + 3j, 5 + 7j
    self = torch.ones((2, 2), dtype=torch.complex64, requires_grad=True)
    mat1 = torch.sparse_coo_tensor(
        [[0, 0, 1, 1], [0, 1, 0, 1]],
        [1 + 2j, 2 - 1j, 3 + 4j, 4 - 2j],
        (2, 2),
    ).coalesce().requires_grad_()
    mat2 = torch.tensor(
        [[2 + 1j, 3 - 1j], [4 + 2j, 5 - 2j]], dtype=torch.complex64, requires_grad=True
    )
    grad = torch.tensor([[11 + 1j, 13 + 2j], [17 + 3j, 19 + 4j]])
    native_addmm(self, mat1, mat2, beta=beta, alpha=alpha).backward(grad)
    expected = (self.grad, mat1.grad, mat2.grad)

    request = router.MatmulRequest(
        "addmm", mat1.detach().requires_grad_(), mat2.detach().requires_grad_(),
        self.detach().requires_grad_(), beta=beta, alpha=alpha, schema=schema,
    )
    actual = backward.spgemm_backward(request, grad, _cpu_sparse_execute)
    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient.to_dense(), expected_gradient.to_dense())


def test_empty_product_projects_zero_values():
    empty = torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.int64), torch.empty(0), (2, 2))
    result = backward.sparse_mask(empty, torch.eye(2).to_sparse_csr())
    torch.testing.assert_close(result.values(), torch.zeros(2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("pattern_columns", [[0, 1], [1, 0]])
def test_csr_product_projection_sorts_keys_without_reordering_pattern(dtype, pattern_columns):
    mat1 = torch.sparse_csr_tensor(
        [0, 2], pattern_columns, torch.ones(2, dtype=dtype), size=(1, 2)
    ).requires_grad_()
    mat2 = torch.eye(2, dtype=dtype).to_sparse_csr()
    gradient_values = [7 + 0.5j, 11 - 0.3j] if dtype.is_complex else [7., 11.]
    product = torch.sparse_csr_tensor([0, 2], [1, 0], gradient_values, size=(1, 2), dtype=dtype)
    request = router.MatmulRequest("mm", mat1, mat2, schema="aten::_sparse_addmm")
    with torch.no_grad():
        grad_self, grad_mat1, grad_mat2 = backward.spgemm_backward(
            request, torch.ones((1, 2), dtype=dtype), lambda _request: product)
    expected = [gradient_values[1 - column] for column in pattern_columns]
    assert grad_self is None and grad_mat2 is None
    torch.testing.assert_close(grad_mat1.crow_indices(), mat1.crow_indices())
    torch.testing.assert_close(grad_mat1.col_indices(), mat1.col_indices())
    torch.testing.assert_close(grad_mat1.values(), torch.tensor(expected, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_csr_product_projection_sums_duplicate_coordinates(dtype):
    mat1 = torch.eye(2, dtype=dtype).to_sparse_csr().requires_grad_()
    mat2 = mat1.detach()
    product = torch.sparse_csr_tensor([0, 2, 2], [0, 0], [7., 11.], size=(2, 2), dtype=dtype)
    request = router.MatmulRequest("mm", mat1, mat2, schema="aten::_sparse_addmm")
    with torch.no_grad():
        grad_mat1 = backward.spgemm_backward(request, torch.ones((2, 2), dtype=dtype),
                                              lambda _request: product)[1]
    torch.testing.assert_close(grad_mat1.values(), torch.tensor([18., 0.], dtype=dtype))


def test_missing_native_projection_reports_pytorch_compatibility_error(monkeypatch):
    monkeypatch.setattr(backward, "_SPARSE_MASK_PROJECTION", None)
    operand = torch.eye(2).to_sparse_coo().requires_grad_()
    request = router.MatmulRequest("mm", operand, operand, schema="aten::_sparse_sparse_matmul")
    gradient = torch.eye(2).to_sparse_coo()
    with pytest.raises(RuntimeError, match="requires aten::_sparse_mask_projection"):
        backward.spgemm_backward(request, gradient, _cpu_sparse_execute)


def test_backward_helpers_register_atomically():
    registry = router.MatmulRegistry()
    capability = router.MatmulCapability("gemm", "spgemm", lambda r: r.mat1,
                                         backward_helpers=(("sparse_mask", backward.sparse_mask),))
    registry.register(capability)
    with pytest.raises(RuntimeError, match="duplicate matrix backward helper"):
        registry.register(router.MatmulCapability("mm", "spmm", lambda r: r.mat1,
                                                 backward_helpers=capability.backward_helpers))
    assert ("spmm", "mm") not in registry.capabilities


def test_sparse_mask_ignores_mask_values_dtype():
    source = torch.eye(2, dtype=torch.float32)
    mask = torch.eye(2, dtype=torch.float64).to_sparse_coo()
    result = backward.sparse_mask(source, mask)
    assert result.dtype == source.dtype
    torch.testing.assert_close(result.to_dense(), source.sparse_mask(mask).to_dense())
    empty = torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.int64), torch.empty(0), (2, 2))
    projected = backward.sparse_mask(empty, mask)
    assert projected.dtype == source.dtype
    torch.testing.assert_close(projected.values(), torch.zeros(2))


@pytest.mark.parametrize('sparse_dims', [1, 2, 3])
def test_sparse_mask_nd_and_hybrid_contract(sparse_dims):
    source = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    mask = torch.ones_like(source).to_sparse(sparse_dims)
    result = backward.sparse_mask(source, mask)
    torch.testing.assert_close(result.to_dense(), source.sparse_mask(mask).to_dense())
    sparse_source = source.to_sparse(sparse_dims)
    result = backward.sparse_mask(sparse_source, mask)
    torch.testing.assert_close(result.to_dense(), sparse_source.sparse_mask(mask).to_dense())


@pytest.mark.parametrize('layout', ['to_sparse_csc', 'to_sparse_bsr', 'to_sparse_bsc'])
def test_sparse_mask_compressed_layout_contract(layout):
    source = torch.tensor([[1., 2.], [3., 4.]])
    constructor = getattr(torch.eye(2), layout)
    mask = constructor((1, 1)) if layout.endswith(('bsr', 'bsc')) else constructor()
    result = backward.sparse_mask(source, mask)
    torch.testing.assert_close(result.to_dense(), source.sparse_mask(mask).to_dense())


def test_compressed_sparse_mask_retains_only_intersection():
    source = torch.tensor([[0., 2.], [0., 0.]]).to_sparse_csr()
    mask = torch.eye(2).to_sparse_csr()
    result = backward.sparse_mask(source, mask)
    assert result.values().numel() == source.sparse_mask(mask).values().numel() == 0
    torch.testing.assert_close(result.crow_indices(), torch.tensor([0, 0, 0]))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.complex64])
def test_csr_add_helpers_preserve_union_and_out_alias(dtype):
    left = torch.tensor([[1., 0.], [0., 2.]], dtype=dtype).to_sparse_csr()
    right = torch.tensor([[0., 3.], [4., 0.]], dtype=dtype).to_sparse_csr()
    out = torch.sparse_csr_tensor([0, 0, 0], [], torch.empty(0, dtype=dtype), size=(2, 2))
    assert backward.sparse_add_out(left, right, alpha=2, out=out) is out
    expected = torch.tensor([[1., 6.], [8., 2.]], dtype=dtype)
    torch.testing.assert_close(out.to_dense(), expected)
    original = left
    assert backward.sparse_add_inplace(left, right, alpha=2) is original
    torch.testing.assert_close(left.to_dense(), expected)


@pytest.mark.parametrize('schema', ["aten::mm", "aten::addmm"])
@pytest.mark.parametrize('layout', [torch.sparse_coo, torch.sparse_csr])
@pytest.mark.parametrize('value', [float('nan'), float('inf')])
@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_dense_matrix_zero_upstream_keeps_nonfinite_operand_contributions(value, dtype, layout, schema):
    a = torch.sparse_coo_tensor([[0], [0]], torch.tensor([value], dtype=dtype), (2, 2)).coalesce().requires_grad_()
    b = a.detach().clone().requires_grad_()
    grad = torch.zeros((2, 2), dtype=dtype)
    expected = torch.autograd.grad(torch.mm(a, b), (a, b), grad)
    if layout == torch.sparse_csr:
        a, b = a.detach().to_sparse_csr().requires_grad_(), b.detach().to_sparse_csr().requires_grad_()
    request = router.MatmulRequest('mm' if schema == "aten::mm" else "addmm", a, b, schema=schema)

    def no_sparse_execute(_):
        pytest.fail('native dense-upstream mm must preserve explicit zeros')

    actual = backward.spgemm_backward(request, grad, no_sparse_execute)[1:]
    for result, reference in zip(actual, expected):
        assert result.layout == torch.strided
        torch.testing.assert_close(result, reference, equal_nan=True)
