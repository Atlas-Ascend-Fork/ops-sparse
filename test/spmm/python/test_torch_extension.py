# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import unittest

import cann_ops_sparse  # noqa: F401 - importing the wheel registers its ATen handlers
import torch
import torch_npu


class SparseAddmmNpuTest(unittest.TestCase):
    @staticmethod
    def _complex_degree2_candidate():
        m, k, n = 1024, 257, 4096
        crow = torch.arange(0, 2 * (m + 1), 2, dtype=torch.int32)
        starts = torch.arange(m, dtype=torch.int64).remainder(k)
        col = torch.stack((starts, (starts + 1).remainder(k)), dim=1)
        col, _ = torch.sort(col, dim=1)
        col = col.reshape(-1).to(torch.int32)
        real = torch.linspace(-0.75, 0.75, 2 * m, dtype=torch.float32)
        values = torch.complex(real, real.flip(0) * 0.25)
        b_real = torch.linspace(-0.5, 0.5, k * n, dtype=torch.float32).reshape(k, n)
        b = torch.complex(b_real, b_real.flip(0) * 0.125)
        input_real = torch.linspace(-0.25, 0.25, m * n, dtype=torch.float32).reshape(m, n)
        input_tensor = torch.complex(input_real, input_real.flip(0) * -0.125)
        mat1 = torch.sparse_csr_tensor(
            crow.to("npu"), col.to("npu"), values.to("npu"),
            size=(m, k), device="npu"
        )
        return input_tensor.to("npu"), mat1, b.to("npu")

    @staticmethod
    def _complex_degree2_golden(input_tensor, mat1, mat2):
        col = mat1.col_indices().cpu().to(torch.int64).reshape(-1, 2)
        values = mat1.values().cpu().to(torch.complex128).reshape(-1, 2)
        b = mat2.cpu().to(torch.complex128)
        product = values[:, 0, None] * b[col[:, 0]]
        product.add_(values[:, 1, None] * b[col[:, 1]])
        return input_tensor.cpu().to(torch.complex128) + product

    @staticmethod
    def _csr(dtype):
        crow = torch.tensor([0, 2, 3, 5], dtype=torch.int32, device="npu")
        col = torch.tensor([0, 2, 1, 0, 3], dtype=torch.int32, device="npu")
        values = torch.tensor([1.25, -0.5, 0.75, -1.0, 0.375], dtype=dtype, device="npu")
        return torch.sparse_csr_tensor(crow, col, values, size=(3, 4), device="npu")

    @staticmethod
    def _golden(mat1, mat2, input_tensor, alpha, beta):
        if mat1.dtype == torch.complex64:
            golden_dtype = torch.complex128
        elif mat1.dtype == torch.float32:
            golden_dtype = torch.float64
        else:
            golden_dtype = torch.float32
        cpu_a = torch.sparse_csr_tensor(
            mat1.crow_indices().cpu(),
            mat1.col_indices().cpu(),
            mat1.values().cpu().to(golden_dtype),
            size=mat1.shape,
        )
        dense_a = cpu_a.to_dense()
        dense_b = mat2.cpu().to(golden_dtype)
        dense_input = input_tensor.cpu().to(golden_dtype).expand(3, 5)
        product = alpha * (dense_a @ dense_b)
        return product if beta == 0 else beta * dense_input + product

    @staticmethod
    def _empty_csr(dtype, m, k, index_dtype=torch.int32):
        rows = torch.zeros(m + 1, dtype=index_dtype, device="npu")
        columns = torch.empty(0, dtype=index_dtype, device="npu")
        values = torch.empty(0, dtype=dtype, device="npu")
        return torch.sparse_csr_tensor(rows, columns, values, size=(m, k), device="npu")

    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("Ascend NPU is unavailable")
        torch.npu.set_device(0)

    def test_zero_dimensions_and_empty_products(self):
        shapes = ((0, 4, 3), (2, 0, 3), (2, 4, 0), (2, 4, 3))
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.complex64):
            beta = complex(0.5, 0.25) if dtype == torch.complex64 else 0.5
            for m, k, n in shapes:
                with self.subTest(dtype=dtype, shape=(m, k, n)):
                    mat1 = self._empty_csr(dtype, m, k)
                    mat2 = torch.full((k, n), float("nan"), dtype=dtype, device="npu")
                    input_tensor = torch.arange(n, dtype=torch.float32, device="npu").to(dtype)
                    output = torch.sparse.addmm(input_tensor, mat1, mat2, beta=beta, alpha=0.75)
                    torch.npu.synchronize()
                    expected = beta * input_tensor.expand(m, n)
                    self.assertEqual(output.shape, (m, n))
                    self.assertEqual(output.dtype, dtype)
                    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)

    def test_empty_products_beta_zero_ignore_nan_and_inf(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.complex64):
            for k in (0, 4):
                with self.subTest(dtype=dtype, k=k):
                    mat1 = self._empty_csr(dtype, 2, k)
                    mat2 = torch.full((k, 3), float("nan"), dtype=dtype, device="npu")
                    input_tensor = torch.tensor(
                        [float("nan"), float("inf"), float("-inf")], dtype=dtype, device="npu"
                    )
                    output = torch.sparse.addmm(input_tensor, mat1, mat2, beta=0, alpha=1)
                    torch.npu.synchronize()
                    torch.testing.assert_close(output.cpu(), torch.zeros((2, 3), dtype=dtype),
                                               rtol=0, atol=0)

    def test_empty_products_validate_broadcast_and_indices(self):
        mat1 = self._empty_csr(torch.float32, 2, 0)
        mat2 = torch.empty((0, 3), dtype=torch.float32, device="npu")
        with self.assertRaises(RuntimeError):
            torch.sparse.addmm(torch.empty(4, device="npu"), mat1, mat2, beta=0)
        mat1 = self._empty_csr(torch.float32, 2, 0, torch.int64)
        output = torch.sparse.addmm(torch.zeros(3, device="npu"), mat1, mat2, beta=0)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), torch.zeros((2, 3)), rtol=0, atol=0)
        for rows, columns, values in (([0, 0, 1], [], []), ([0, 1, 0], [], []),
                                      ([0, 1, 1], [0], [1.0])):
            with self.subTest(rows=rows):
                mat1 = torch.sparse_csr_tensor(
                    torch.tensor(rows, dtype=torch.int32, device="npu"),
                    torch.tensor(columns, dtype=torch.int32, device="npu"),
                    torch.tensor(values, dtype=torch.float32, device="npu"),
                    size=(2, 0), device="npu", check_invariants=False
                )
                with self.assertRaises(RuntimeError):
                    torch.sparse.addmm(torch.zeros(3, device="npu"), mat1, mat2, beta=0)

    def test_dispatch_is_registered_for_sparse_csr_npu(self):
        mat1 = self._csr(torch.float32)
        input_tensor = torch.zeros((3, 5), dtype=torch.float32, device="npu")
        mat2 = torch.ones((4, 5), dtype=torch.float32, device="npu")
        output = torch.sparse.addmm(input_tensor, mat1, mat2)
        torch.npu.synchronize()
        self.assertEqual(output.device.type, "npu")
        self.assertEqual(output.layout, torch.strided)
        self.assertEqual(output.shape, (3, 5))

    def test_required_dtypes_and_scalars(self):
        configs = (
            (torch.float16, 0.75, -0.25, 2**-9, 2**-9),
            (torch.bfloat16, 0.75, -0.25, 2**-6, 2**-6),
            (torch.float32, 0.75, -0.25, 2**-10, 2**-16),
            (
                torch.complex64,
                complex(0.75, 0.25),
                complex(-0.25, 0.125),
                2**-10,
                2**-16,
            ),
        )
        for dtype, alpha, beta, rtol, atol in configs:
            with self.subTest(dtype=dtype):
                mat1 = self._csr(dtype)
                real = torch.linspace(-0.5, 0.5, 20, dtype=torch.float32).reshape(4, 5)
                if dtype == torch.complex64:
                    mat2_cpu = torch.complex(real, real.flip(0) * 0.125)
                    input_cpu = torch.complex(
                        torch.linspace(-1, 1, 5), torch.linspace(0.25, -0.25, 5)
                    )
                else:
                    mat2_cpu = real.to(dtype)
                    input_cpu = torch.linspace(-1, 1, 5).to(dtype)
                mat2 = mat2_cpu.to("npu")
                input_tensor = input_cpu.to("npu")

                output = torch.sparse.addmm(
                    input_tensor, mat1, mat2, alpha=alpha, beta=beta
                )
                torch.npu.synchronize()
                golden = self._golden(mat1, mat2, input_tensor, alpha, beta)

                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.device.type, "npu")
                actual = output.cpu().to(golden.dtype)
                if dtype == torch.complex64:
                    torch.testing.assert_close(
                        actual.real, golden.real, rtol=rtol, atol=atol
                    )
                    torch.testing.assert_close(
                        actual.imag, golden.imag, rtol=rtol, atol=atol
                    )
                else:
                    torch.testing.assert_close(actual, golden, rtol=rtol, atol=atol)

    def test_fp16_uses_fp32_golden_and_preserves_inputs(self):
        dtype = torch.float16
        mat1 = self._csr(dtype)
        base = torch.arange(20, dtype=dtype, device="npu").reshape(5, 4)
        mat2 = base.t()
        self.assertFalse(mat2.is_contiguous())
        input_tensor = torch.linspace(-1, 1, 5, dtype=dtype, device="npu")
        input_before = input_tensor.cpu().clone()
        mat2_before = mat2.cpu().clone()
        values_before = mat1.values().cpu().clone()

        output = torch.sparse.addmm(input_tensor, mat1, mat2, alpha=0.75, beta=-0.25)
        torch.npu.synchronize()
        golden = self._golden(mat1, mat2, input_tensor, 0.75, -0.25)

        self.assertEqual(output.dtype, dtype)
        self.assertEqual(output.device.type, "npu")
        torch.testing.assert_close(output.cpu().float(), golden, rtol=2**-9, atol=2**-9)
        torch.testing.assert_close(input_tensor.cpu(), input_before, rtol=0, atol=0)
        torch.testing.assert_close(mat2.cpu(), mat2_before, rtol=0, atol=0)
        torch.testing.assert_close(mat1.values().cpu(), values_before, rtol=0, atol=0)

    def test_fp16_beta_zero_does_not_propagate_input_nan_or_inf(self):
        dtype = torch.float16
        mat1 = self._csr(dtype)
        mat2 = torch.linspace(-0.5, 0.5, 20, dtype=dtype, device="npu").reshape(4, 5)
        input_tensor = torch.tensor(
            [[float("nan")], [float("inf")], [float("-inf")]], dtype=dtype, device="npu"
        )
        input_before = input_tensor.cpu().clone()

        output = torch.sparse.addmm(input_tensor, mat1, mat2, alpha=1, beta=0)
        torch.npu.synchronize()
        golden = self._golden(mat1, mat2, input_tensor, 1, 0)

        self.assertTrue(torch.isfinite(output).all().item())
        torch.testing.assert_close(output.cpu().float(), golden, rtol=2**-9, atol=2**-9)
        torch.testing.assert_close(input_tensor.cpu(), input_before, rtol=0, atol=0, equal_nan=True)

    def test_complex_contiguous_input_preserves_source(self):
        dtype = torch.complex64
        mat1 = self._csr(dtype)
        real = torch.linspace(-0.5, 0.5, 20, dtype=torch.float32).reshape(4, 5)
        mat2 = torch.complex(real, real.flip(0) * 0.125).to("npu")
        input_real = torch.linspace(-0.75, 0.75, 15, dtype=torch.float32).reshape(3, 5)
        input_tensor = torch.complex(
            input_real, input_real.flip(1) * -0.25
        ).to("npu")
        input_before = input_tensor.cpu().clone()

        output = torch.sparse.addmm(
            input_tensor, mat1, mat2,
            alpha=complex(0.75, 0.25), beta=complex(-0.25, 0.125)
        )
        torch.npu.synchronize()
        golden = self._golden(
            mat1, mat2, input_tensor,
            complex(0.75, 0.25), complex(-0.25, 0.125)
        )

        self.assertNotEqual(output.data_ptr(), input_tensor.data_ptr())
        torch.testing.assert_close(
            output.cpu().to(golden.dtype), golden, rtol=2**-10, atol=2**-16
        )
        torch.testing.assert_close(
            input_tensor.cpu(), input_before, rtol=0, atol=0
        )

    def test_fp32_contiguous_input_preserves_source(self):
        dtype = torch.float32
        mat1 = self._csr(dtype)
        mat2 = torch.linspace(
            -0.5, 0.5, 20, dtype=dtype, device="npu"
        ).reshape(4, 5)
        input_tensor = torch.linspace(
            -0.75, 0.75, 15, dtype=dtype, device="npu"
        ).reshape(3, 5)
        input_before = input_tensor.cpu().clone()

        output = torch.sparse.addmm(
            input_tensor, mat1, mat2, alpha=0.75, beta=-0.25
        )
        torch.npu.synchronize()
        golden = self._golden(mat1, mat2, input_tensor, 0.75, -0.25)

        self.assertNotEqual(output.data_ptr(), input_tensor.data_ptr())
        torch.testing.assert_close(
            output.cpu().to(golden.dtype), golden, rtol=2**-10, atol=2**-16
        )
        torch.testing.assert_close(
            input_tensor.cpu(), input_before, rtol=0, atol=0
        )

    def test_fp32_degree2_out_of_place_preserves_source(self):
        m, k, n = 1000, 257, 513
        crow = torch.arange(0, 2 * (m + 1), 2, dtype=torch.int32)
        starts = torch.arange(m, dtype=torch.int64).remainder(k)
        col = torch.stack((starts, (starts + 1).remainder(k)), dim=1)
        col, _ = torch.sort(col, dim=1)
        col = col.reshape(-1).to(torch.int32)
        values = torch.linspace(-0.75, 0.75, 2 * m, dtype=torch.float32)
        mat1 = torch.sparse_csr_tensor(
            crow.to("npu"), col.to("npu"), values.to("npu"),
            size=(m, k), device="npu"
        )
        mat2_cpu = torch.linspace(
            -0.5, 0.5, k * n, dtype=torch.float32
        ).reshape(k, n)
        input_cpu = torch.linspace(
            -0.25, 0.25, m * n, dtype=torch.float32
        ).reshape(m, n)
        input_tensor = input_cpu.to("npu")
        input_before = input_tensor.cpu().clone()

        output = torch.sparse.addmm(
            input_tensor, mat1, mat2_cpu.to("npu"), alpha=1, beta=1
        )
        torch.npu.synchronize()
        cpu_a = torch.sparse_csr_tensor(
            crow, col, values.to(torch.float64), size=(m, k)
        )
        golden = input_cpu.to(torch.float64) + (
            cpu_a.to_dense() @ mat2_cpu.to(torch.float64)
        )

        self.assertNotEqual(output.data_ptr(), input_tensor.data_ptr())
        torch.testing.assert_close(
            output.cpu().to(torch.float64), golden,
            rtol=2**-10, atol=2**-16
        )
        torch.testing.assert_close(
            input_tensor.cpu(), input_before, rtol=0, atol=0
        )

    def test_complex_nonuniform_reorder_preserved(self):
        dtype = torch.complex64
        m, k, n = 4, 5, 7
        crow = torch.tensor([0, 0, 1, 4, 5], dtype=torch.int32)
        col = torch.tensor([2, 0, 3, 4, 1], dtype=torch.int32)
        real = torch.linspace(-0.75, 0.75, col.numel(), dtype=torch.float32)
        values = torch.complex(real, real.flip(0) * 0.25)
        mat1 = torch.sparse_csr_tensor(
            crow.to("npu"), col.to("npu"), values.to("npu"),
            size=(m, k), device="npu"
        )
        b_real = torch.linspace(-0.5, 0.5, k * n, dtype=torch.float32).reshape(k, n)
        mat2_cpu = torch.complex(b_real, b_real.flip(0) * 0.125)
        input_real = torch.linspace(-0.25, 0.25, m * n, dtype=torch.float32).reshape(m, n)
        input_cpu = torch.complex(input_real, input_real.flip(1) * -0.125)

        output = torch.sparse.addmm(
            input_cpu.to("npu"), mat1, mat2_cpu.to("npu"), alpha=1, beta=1
        )
        torch.npu.synchronize()
        cpu_a = torch.sparse_csr_tensor(
            crow, col, values.to(torch.complex128), size=(m, k)
        )
        golden = input_cpu.to(torch.complex128) + (
            cpu_a.to_dense() @ mat2_cpu.to(torch.complex128)
        )

        torch.testing.assert_close(
            output.cpu().to(golden.dtype), golden, rtol=2**-10, atol=2**-16
        )

    def test_complex_degree2_cache_lifecycle_and_active_inputs(self):
        input_tensor, mat1, mat2 = self._complex_degree2_candidate()
        input_before = input_tensor.cpu().clone()

        first = torch.sparse.addmm(input_tensor, mat1, mat2)
        repeated = torch.sparse.addmm(input_tensor, mat1, mat2)
        torch.npu.synchronize()
        self._assert_complex_result(first, input_tensor, input_before, mat1, mat2)
        self._assert_complex_result(repeated, input_tensor, input_before, mat1, mat2)

        mat1.values().mul_(complex(-0.5, 0.25))
        mat2.mul_(complex(0.75, -0.125))
        active_update = torch.sparse.addmm(input_tensor, mat1, mat2)
        torch.npu.synchronize()
        self._assert_complex_result(
            active_update, input_tensor, input_before, mat1, mat2
        )

        mat1.col_indices()[:2].copy_(
            torch.tensor([1, 2], dtype=torch.int32, device="npu")
        )
        index_update = torch.sparse.addmm(input_tensor, mat1, mat2)
        torch.npu.synchronize()
        self._assert_complex_result(
            index_update, input_tensor, input_before, mat1, mat2
        )

        alternate_stream = torch.npu.Stream()
        with torch.npu.stream(alternate_stream):
            stream_update = torch.sparse.addmm(input_tensor, mat1, mat2)
        alternate_stream.synchronize()
        self._assert_complex_result(
            stream_update, input_tensor, input_before, mat1, mat2
        )

        noncandidate_b = mat2[:, :-1].contiguous()
        noncandidate_input = input_tensor[:, :-1].contiguous()
        noncandidate = torch.sparse.addmm(
            noncandidate_input, mat1, noncandidate_b
        )
        torch.npu.synchronize()
        noncandidate_golden = self._complex_degree2_golden(
            noncandidate_input, mat1, noncandidate_b
        )
        torch.testing.assert_close(
            noncandidate.cpu().to(noncandidate_golden.dtype), noncandidate_golden,
            rtol=2**-10, atol=2**-16
        )

    def _assert_complex_result(self, output, input_tensor, input_before, mat1, mat2):
        self.assertNotEqual(output.data_ptr(), input_tensor.data_ptr())
        torch.testing.assert_close(input_tensor.cpu(), input_before)
        golden = self._complex_degree2_golden(input_tensor, mat1, mat2)
        torch.testing.assert_close(
            output.cpu().to(golden.dtype), golden, rtol=2**-10, atol=2**-16
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
