# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Shared golden comparison and native BSR invocation for SDDMM tests."""

import math
from dataclasses import dataclass

import torch
from cann_ops_sparse.ops.sparse.sddmm import ensure_sddmm_loaded


def assert_accuracy(self, actual, expected):
    self.assertEqual(actual.shape, expected.shape)
    if expected.numel() == 0:
        return
    dtype = expected.dtype
    rtol, atol, cap = {torch.float16: (2**-9, 2**-9, .1),
                       torch.bfloat16: (2**-6, 2**-6, 1.),
                       torch.float32: (2**-10, 2**-16, .01),
                       torch.complex64: (2**-10, 2**-16, .01)}[dtype]
    pairs = (
        ((actual.real, expected.real), (actual.imag, expected.imag))
        if expected.is_complex()
        else ((actual, expected),)
    )
    for got, ref in pairs:
        error = (got.double() - ref.double()).abs()
        self.assertGreaterEqual((error <= atol + rtol * ref.double().abs()).double().mean().item(), .99)
        mag = ref.abs()
        ulp = (torch.nextafter(mag, torch.full_like(mag, float("inf"))) - mag).double()
        self.assertTrue(bool((error <= torch.maximum(torch.full_like(error, cap), 32 * ulp)).all()))


def check_csr(case, inputs, output):
    rp, ci, values, x, y = [t.detach().cpu() for t in inputs]
    rp = rp.long() - case["base"]
    ci = ci.long() - case["base"]
    row = torch.repeat_interleave(torch.arange(case["m"]), rp[1:] - rp[:-1])
    work = (torch.complex128 if values.is_complex() else
            torch.float32 if values.dtype in (torch.float16, torch.bfloat16) else torch.float64)
    x, y = x.to(work), y.to(work)
    golden = torch.empty((x.shape[0], ci.numel()), dtype=values.dtype)
    for first in range(0, ci.numel(), 4096):
        last = min(first + 4096, ci.numel())
        dot = (x[:, row[first:last], :] * y.transpose(1, 2)[:, ci[first:last], :]).sum(-1)
        golden[:, first:last] = case["alpha"] * dot + case["beta"] * values[first:last]
    output = output.detach().cpu()
    if output.shape != golden.shape:
        return {"pass": False, "reason": "output shape mismatch"}
    if output.numel() == 0:
        return {"pass": True, "match_rate": 1.0, "cap_ratio": 0.0}
    dtype = case["dtype"]
    rtol, atol, cap = {"float16": (2**-9, 2**-9, 0.1),
                       "bfloat16": (2**-6, 2**-6, 1.0),
                       "float32": (2**-10, 2**-16, 0.01),
                       "complex64": (2**-10, 2**-16, 0.01)}[dtype]
    parts = [(output.real, golden.real), (output.imag, golden.imag)] if output.is_complex() else [(output, golden)]
    rate, worst = 1.0, 0.0
    for actual, expected in parts:
        mag = expected.abs()
        ulp = (torch.nextafter(mag, torch.full_like(mag, float("inf"))) - mag).double()
        error = (actual.double() - expected.double()).abs()
        bound = atol + rtol * expected.double().abs()
        rate = min(rate, (error <= bound).double().mean().item())
        limit = torch.maximum(torch.full_like(error, cap), 32 * ulp)
        ratio = (error / limit).max().item()
        worst = max(worst, ratio) if math.isfinite(ratio) else float("inf")
    return {"pass": rate >= 0.99 and worst <= 1.0, "match_rate": rate, "cap_ratio": worst}


@dataclass(frozen=True)
class BsrOperands:
    row_ptr: torch.Tensor
    col_ind: torch.Tensor
    values: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True)
class BsrOptions:
    alpha: complex = 1.0
    beta: complex = 0.0
    base: int = 0
    block_dim: int = 2
    direction: str = "row"
    batch: int = 1


def bsr_values(operands, options):
    return ensure_sddmm_loaded().sddmm_bsr_pattern(
        operands.row_ptr, operands.col_ind, operands.values, operands.x, operands.y,
        complex(options.alpha), complex(options.beta), options.base, options.block_dim,
        options.direction == "col", options.batch
    )
