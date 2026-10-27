# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Small-block BSR regression: DMA tails, grouped rows and storage boundaries."""

import unittest
from itertools import product

import torch
from _sddmm_utils import BsrOperands, BsrOptions, assert_accuracy, bsr_values, require_arch22

import torch_npu  # noqa: F401


PAIRS = ((torch.float32, torch.float32), (torch.float16, torch.float16),
         (torch.bfloat16, torch.bfloat16), (torch.float16, torch.float32),
         (torch.bfloat16, torch.float32))


class SmallBsrTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_arch22()

    assert_accuracy = assert_accuracy

    def check_case(self, shape, dtypes, direction, grouped=False, beta=.5):
        bd, k = shape
        dtype, ctype = dtypes
        torch.manual_seed(107)
        base = int(direction == "col")
        counts = torch.tensor([0, 643, 0, 19, 0, 5] if grouped else [0, 19, 0, 3, 2, 0])
        rp = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))).int() + base
        columns = [torch.arange(count) for count in counts]
        # Also exercise duplicate, descending and nonconsecutive block columns.
        columns[3] = columns[3].flip(0)
        columns[4] = torch.zeros_like(columns[4])
        ci = torch.cat(columns).int() + base
        m, n = len(counts) * bd, (int(counts.max()) + 3) * bd
        x = torch.randn(1, m, k, dtype=dtype)
        y = torch.randn(1, k, n, dtype=dtype)
        logical = torch.randn(len(ci), bd, bd, dtype=ctype)
        stored = logical if direction == "row" else logical.transpose(-2, -1).contiguous()
        rows = torch.repeat_interleave(torch.arange(len(counts)), counts)[:, None] * bd + torch.arange(bd)
        cols = (ci.long() - base)[:, None] * bd + torch.arange(bd)
        dense = x.double() @ y.double()
        expected = dense[:, rows[:, :, None], cols[:, None, :]] * .75
        if beta:
            expected += beta * logical.double()
        else:
            stored.fill_(float("nan"))
        expected = expected.to(ctype)
        if direction == "col":
            expected = expected.transpose(-2, -1).contiguous()
        operands = tuple(t.npu() for t in (rp, ci, stored, x, y))
        result = bsr_values(
            BsrOperands(*operands),
            BsrOptions(alpha=.75, beta=beta, base=base, block_dim=bd, direction=direction, batch=1)
        )
        self.assert_accuracy(result.cpu(), expected)

    def test_k_and_dtype_boundaries(self):
        ks = (1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 129, 255, 256, 257)
        for bd, k, (dtype, ctype), direction in product((2, 4, 8), ks, PAIRS, ("row", "col")):
            with self.subTest(bd=bd, k=k, dtype=dtype, ctype=ctype, direction=direction):
                self.check_case((bd, k), (dtype, ctype), direction)

    def test_grouped_rows_and_nan_c(self):
        for bd, (dtype, ctype), direction, beta in product((2, 4, 8), PAIRS, ("row", "col"), (0., -.25)):
            with self.subTest(bd=bd, dtype=dtype, ctype=ctype, direction=direction, beta=beta):
                self.check_case((bd, 100), (dtype, ctype), direction, grouped=True, beta=beta)


if __name__ == "__main__":
    unittest.main()
