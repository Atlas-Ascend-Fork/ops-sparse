# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Exercise the production Python SpMM VJP with native CPU Torch tensors."""

import importlib.util
from pathlib import Path

import pytest
import torch

from test_matmul_dispatch import router

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_spmm_backward_precision", ROOT / "sparse/spmm/torch_extension/spmm_backward.py")
backward = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backward)


@pytest.mark.parametrize("layout", (torch.sparse_coo, torch.sparse_csr))
@pytest.mark.parametrize("sparse_left,schema", (
    (True, "aten::_sparse_addmm"), (True, "aten::addmm"), (False, "aten::addmm")))
@pytest.mark.parametrize("dense_value,grad_value,alpha", (
    (400.0, 400.0, 0.125), (400.0, 400.0, -0.125),
    (2.0 ** -12, 2.0 ** -13, 1024.0), (400.0, 400.0, 0.0), (4.0, 5.0, 0.125)))
def test_scale_before_fp16_rounding(layout, sparse_left, schema, dense_value, grad_value, alpha):
    sparse = torch.ones((1, 1), dtype=torch.float16).to_sparse(layout=layout).requires_grad_()
    dense = torch.tensor([[dense_value]], dtype=torch.float16)
    gradient = torch.tensor([[grad_value]], dtype=torch.float16)
    left, right = (sparse, dense) if sparse_left else (dense, sparse)
    request = router.MatmulRequest("addmm", left, right, torch.zeros_like(dense),
                                   alpha=alpha, beta=0, schema=schema)
    actual = backward.spmm_backward(request, gradient)[1 if sparse_left else 2].to_dense()
    # Independent FP64 native autograd; input numbers are exactly the original FP16 values.
    a = left.detach().to_dense().double().requires_grad_(sparse_left)
    b = right.detach().to_dense().double().requires_grad_(not sparse_left)
    torch.addmm(torch.zeros_like(a), a, b, alpha=alpha, beta=0).backward(gradient.double())
    expected = (a.grad if sparse_left else b.grad).half()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("dtype", (torch.float32, torch.complex64))
def test_existing_scaling_and_conjugation(dtype):
    a = torch.tensor([[1, 0], [0, 2]], dtype=dtype).to_sparse_coo().requires_grad_()
    b = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
    g = torch.tensor([[2, 3], [4, 5]], dtype=dtype)
    alpha = 0.5 if dtype == torch.float32 else 1 + 2j
    request = router.MatmulRequest("addmm", a, b, torch.zeros_like(b), alpha=alpha,
                                   beta=0, schema="aten::_sparse_addmm")
    actual = backward.spmm_backward(request, g)[1].to_dense()
    oracle_a = a.detach().to_dense().to(torch.complex128 if dtype.is_complex else torch.float64)
    oracle_a.requires_grad_()
    torch.addmm(torch.zeros_like(oracle_a), oracle_a, b.to(oracle_a.dtype), alpha=alpha,
                beta=0).backward(g.to(oracle_a.dtype))
    expected = torch.diag(torch.diag(oracle_a.grad)).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_empty_sparse_gradient_preserves_shape_and_dtype():
    a = torch.zeros((2, 2), dtype=torch.float16).to_sparse_coo().requires_grad_()
    b = torch.ones((2, 2), dtype=torch.float16)
    request = router.MatmulRequest("addmm", a, b, torch.zeros_like(b), alpha=1024,
                                   beta=0, schema="aten::_sparse_addmm")
    result = backward.spmm_backward(request, b)[1]
    assert result.shape == a.shape
    assert result.dtype == a.dtype
    assert result.coalesce().values().numel() == 0
