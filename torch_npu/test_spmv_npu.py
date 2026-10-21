# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""End-to-end tests for the optional A5 SpMV PyTorch bridge.

These tests intentionally require torch_npu and an NPU. They never replace a
missing registration with a CPU implementation, because doing so would hide a
dispatcher regression.
"""

from __future__ import annotations

import os
import unittest
from itertools import product

import torch
import torch_npu  # noqa: F401  # registers the PrivateUse1 backend


def _load_extension() -> None:
    path = os.environ.get("OPS_SPARSE_TORCH_LIBRARY", "libops_sparse_torch.so")
    torch.ops.load_library(path)


class SpmvNpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _load_extension()
        cls.device = torch.device("npu:0")

    def test_torch_mv_dispatches_to_npu(self) -> None:
        matrix, _, _, _ = self._csr(torch.float32)
        vector = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        result = torch.mv(matrix, vector)
        self.assertTrue(result.device.type == "npu")
        self.assertTrue(torch.allclose(result.cpu(), torch.tensor([11.0, 8.0])))

    def test_mv_rejects_short_and_long_vectors(self) -> None:
        matrix, _, _, _ = self._csr(torch.float32)
        for length, entry in product((2, 4), (torch.mv, torch.ops.ops_sparse.spmv)):
            vector = torch.arange(1, length + 1, dtype=torch.float32, device=self.device)
            with self.subTest(length=length, entry=str(entry)):
                with self.assertRaisesRegex(
                    RuntimeError, "SpMV matrix columns must equal vector length"
                ):
                    entry(matrix, vector)

    def test_registered_hook_updates_y_in_place(self) -> None:
        _, row_ptr, col_ind, values = self._csr(torch.float32)
        x = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        y = torch.tensor([5.0, 6.0], device=self.device)
        returned = torch.ops.ops_sparse_test.spmv_npu(
            row_ptr, col_ind, values, x, y, 2.0, 3.0, "N", 0
        )
        self.assertEqual(returned.data_ptr(), y.data_ptr())
        self.assertTrue(torch.allclose(y.cpu(), torch.tensor([37.0, 34.0])))

    def test_supported_mixed_dtype(self) -> None:
        matrix, _, _, _ = self._csr(torch.float16)
        vector = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16, device=self.device)
        result = torch.mv(matrix, vector)
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(torch.allclose(result.cpu().float(), torch.tensor([11.0, 8.0])))

    def test_complex_conjugate_transpose_hook(self) -> None:
        row_ptr = torch.tensor([0, 2, 3], dtype=torch.int32, device=self.device)
        col_ind = torch.tensor([0, 1, 1], dtype=torch.int32, device=self.device)
        values = torch.tensor([1 + 2j, 3 - 1j, 2 + 4j],
                              dtype=torch.complex64, device=self.device)
        x = torch.tensor([1 - 1j, 2 + 3j], dtype=torch.complex64,
                         device=self.device)
        y = torch.empty(2, dtype=torch.complex64, device=self.device)
        result = torch.ops.ops_sparse_test.spmv_npu(
            row_ptr, col_ind, values, x, y, 1 + 0j, 0 + 0j, "H", 0
        )
        expected = torch.tensor([-1 - 3j, 20 - 4j], dtype=torch.complex64)
        self.assertEqual(result.data_ptr(), y.data_ptr())
        self.assertTrue(torch.allclose(result.cpu(), expected))

    def test_transpose_accepts_unsorted_and_duplicate_columns(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0, -1.0, 4.0], device=self.device)
        x = torch.tensor([2.0, -3.0], device=self.device)
        for base, ptrs, indices, expected in (
            (0, [0, 3, 5], [2, 0, 2, 1, 0], [-10.25, 4.25, 9.0]),
            (1, [1, 4, 6], [1, 3, 3, 1, 2], [6.0, -14.5, 11.5]),
        ):
            with self.subTest(base=base):
                row_ptr = torch.tensor(ptrs, dtype=torch.int32, device=self.device)
                col_ind = torch.tensor(indices, dtype=torch.int32, device=self.device)
                y = torch.tensor([0.5, -1.0, 2.0], device=self.device)
                returned = torch.ops.ops_sparse_test.spmv_npu(
                    row_ptr, col_ind, values, x, y, 1.25, -0.5, "T", base
                )
                self.assertEqual(returned.data_ptr(), y.data_ptr())
                self.assertTrue(torch.allclose(y.cpu(), torch.tensor(expected),
                                               rtol=0.0, atol=1e-6))

    def test_rejects_cpu_tensor_instead_of_fallback(self) -> None:
        row_ptr = torch.tensor([0, 0], dtype=torch.int32)
        col_ind = torch.empty(0, dtype=torch.int32)
        values = torch.empty(0, dtype=torch.float32)
        x = torch.empty(0, dtype=torch.float32)
        y = torch.empty(0, dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            torch.ops.ops_sparse_test.spmv_npu(
                row_ptr, col_ind, values, x, y, 1.0, 0.0, "N", 0
            )

    def _csr(self, dtype: torch.dtype):
        row_ptr = torch.tensor([0, 2, 3], dtype=torch.int32, device=self.device)
        col_ind = torch.tensor([0, 2, 1], dtype=torch.int32, device=self.device)
        values = torch.tensor([2, 3, 4], dtype=dtype, device=self.device)
        matrix = torch.sparse_csr_tensor(
            row_ptr, col_ind, values, size=(2, 3), device=self.device
        )
        return matrix, row_ptr, col_ind, values


if __name__ == "__main__":
    unittest.main()
