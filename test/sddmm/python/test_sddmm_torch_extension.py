# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SDDMM Torch Extension tests through the installed cann_ops_sparse wheel.

Build and run instructions are in test/sddmm/README.md.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
cann_ops_sparse = pytest.importorskip("cann_ops_sparse")

_HAS_NPU = bool(getattr(torch, "npu", None) and torch.npu.is_available())


class CsrValues(NamedTuple):
    crow: torch.Tensor
    col: torch.Tensor
    values: torch.Tensor


class SampledCase(NamedTuple):
    pattern: torch.Tensor
    mat1: torch.Tensor
    mat2: torch.Tensor
    crow: torch.Tensor
    col: torch.Tensor
    values: torch.Tensor


def _case(dtype: torch.dtype, index_dtype: torch.dtype = torch.int64):
    crow = torch.tensor([0, 2, 3, 5], dtype=index_dtype)
    col = torch.tensor([0, 2, 1, 0, 3], dtype=index_dtype)
    if dtype.is_complex:
        values = torch.tensor(
            [1 + 0.5j, -2 + 1j, 0.25 - 0.5j, 3 - 1j, -0.75 + 2j],
            dtype=dtype,
        )
        mat1 = torch.tensor(
            [[1 + 1j, 2 - 1j], [0.5 + 2j, -1 + 0.25j], [3 - 0.5j, 1 + 1j]],
            dtype=dtype,
        )
        mat2 = torch.tensor(
            [[1 - 0.5j, 2 + 1j, -1 + 0.5j, 0.25 - 1j],
             [0.5 + 1j, -2 + 0.25j, 3 - 1j, 1 + 0.5j]],
            dtype=dtype,
        )
    else:
        values = torch.tensor([1.0, -2.0, 0.25, 3.0, -0.75], dtype=dtype)
        mat1 = torch.tensor([[1.0, 2.0], [0.5, -1.0], [3.0, 1.0]], dtype=dtype)
        mat2 = torch.tensor(
            [[1.0, 2.0, -1.0, 0.25], [0.5, -2.0, 3.0, 1.0]], dtype=dtype
        )
    pattern = torch.sparse_csr_tensor(
        crow.to("npu"), col.to("npu"), values.to("npu"), size=(3, 4)
    )
    return SampledCase(pattern, mat1.to("npu"), mat2.to("npu"), crow, col, values)


def _golden_values(pattern: CsrValues, mat1, mat2, alpha, beta):
    crow, col, values = pattern
    rows = torch.repeat_interleave(
        torch.arange(crow.numel() - 1, dtype=torch.int64), crow.diff().to(torch.int64)
    )
    if values.dtype == torch.float16:
        reference_dtype = torch.float32
    elif values.dtype == torch.float32:
        reference_dtype = torch.float64
    else:
        reference_dtype = torch.complex128
    product = mat1.cpu().to(reference_dtype) @ mat2.cpu().to(reference_dtype)
    old_values = values.cpu().to(reference_dtype)
    columns = col.to(torch.int64)
    if product.dim() == 2:
        sampled = product[rows, columns]
    else:
        flat_product = product.reshape(-1, product.size(-2), product.size(-1))
        flat_columns = columns.reshape(-1, columns.size(-1)) if columns.dim() > 1 else columns
        if flat_columns.dim() == 1:
            flat_columns = flat_columns.expand(flat_product.size(0), -1)
        flat_rows = rows.expand(flat_product.size(0), -1)
        flat_batches = torch.arange(flat_product.size(0)).unsqueeze(1)
        sampled = flat_product[flat_batches, flat_rows, flat_columns]
        sampled = sampled.reshape(*product.shape[:-2], rows.numel())
    expected = alpha * sampled + beta * old_values
    return expected.to(values.dtype)


def _complex_negative_view(tensor):
    # The imaginary view of a conjugate carries the lazy negative bit.
    # Re-expose the complete real/imaginary storage before viewing it as complex.
    # These public metadata views preserve storage and leave negation unresolved.
    real_shape = (*tensor.shape, 2)
    real_stride = (*[stride * 2 for stride in tensor.stride()], 1)
    negative_real = tensor.conj().imag.as_strided(
        real_shape, real_stride, tensor.storage_offset() * 2
    )
    return torch.view_as_complex(negative_real)


@unittest.skipUnless(_HAS_NPU, "torch_npu and an available NPU are required")
class TestSddmmTorchExtension(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        if not torch_npu.npu.get_device_name().startswith("Ascend950"):
            raise unittest.SkipTest("Ascend 950 public-wrapper tests require arch35")

    def setUp(self):
        torch.npu.set_device(0)

    def test_dispatcher_has_functional_and_out_kernels(self):
        self.assertTrue(
            torch.ops.aten.sparse_sampled_addmm.default.has_kernel_for_dispatch_key(
                "SparseCsrPrivateUse1"
            )
        )
        self.assertTrue(
            torch.ops.aten.sparse_sampled_addmm.out.has_kernel_for_dispatch_key(
                "SparseCsrPrivateUse1"
            )
        )

    def test_public_api_dtypes_and_functional_alias(self):
        for dtype in (torch.float16, torch.float32, torch.complex64):
            with self.subTest(dtype=dtype):
                pattern, mat1, mat2, crow, col, values = _case(dtype)
                alpha = 0.75 + (0.25j if dtype.is_complex else 0.0)
                beta = -0.5 + (0.125j if dtype.is_complex else 0.0)
                input_snapshot = pattern.values().cpu().clone()
                input_ptrs = (
                    pattern.crow_indices().untyped_storage().data_ptr(),
                    pattern.col_indices().untyped_storage().data_ptr(),
                    pattern.values().untyped_storage().data_ptr(),
                )

                result = torch.sparse.sampled_addmm(
                    pattern, mat1, mat2, alpha=alpha, beta=beta
                )
                expected = _golden_values(CsrValues(crow, col, values), mat1, mat2, alpha, beta)

                self.assertEqual(result.device.type, "npu")
                self.assertEqual(result.layout, torch.sparse_csr)
                torch.testing.assert_close(
                    result.values().cpu(), expected,
                    rtol=4e-3 if dtype == torch.float16 else 2e-4,
                    atol=4e-3 if dtype == torch.float16 else 2e-5,
                )
                torch.testing.assert_close(pattern.values().cpu(), input_snapshot, rtol=0, atol=0)
                output_ptrs = (
                    result.crow_indices().untyped_storage().data_ptr(),
                    result.col_indices().untyped_storage().data_ptr(),
                    result.values().untyped_storage().data_ptr(),
                )
                self.assertTrue(all(lhs != rhs for lhs, rhs in zip(input_ptrs, output_ptrs)))

    def test_aten_entry_int32_indices_and_noncontiguous_dense(self):
        pattern, mat1, mat2, crow, col, values = _case(torch.float32, torch.int32)
        mat1_noncontiguous = mat1.transpose(0, 1).contiguous().transpose(0, 1)
        mat2_noncontiguous = mat2.transpose(0, 1).contiguous().transpose(0, 1)
        self.assertFalse(mat1_noncontiguous.is_contiguous())
        self.assertFalse(mat2_noncontiguous.is_contiguous())

        result = torch.ops.aten.sparse_sampled_addmm.default(
            pattern, mat1_noncontiguous, mat2_noncontiguous, beta=0.25, alpha=1.5
        )
        expected = _golden_values(
            CsrValues(crow, col, values), mat1_noncontiguous, mat2_noncontiguous, 1.5, 0.25
        )
        torch.testing.assert_close(result.values().cpu(), expected, rtol=2e-4, atol=2e-5)

    def test_complex_lazy_conjugate_and_negative_views(self):
        pattern, mat1, mat2, crow, col, values = _case(torch.complex64)
        mat1_view = mat1.conj()
        mat2_view = _complex_negative_view(mat2)
        self.assertTrue(mat1_view.is_conj())
        self.assertTrue(mat2_view.is_neg())
        self.assertEqual(
            mat2_view.untyped_storage().data_ptr(),
            mat2.untyped_storage().data_ptr(),
        )

        result = torch.sparse.sampled_addmm(
            pattern, mat1_view, mat2_view, alpha=0.75 + 0.25j, beta=-0.5j
        )
        expected = _golden_values(
            CsrValues(crow, col, values), mat1_view, mat2_view, 0.75 + 0.25j, -0.5j
        )
        torch.testing.assert_close(result.values().cpu(), expected, rtol=2e-4, atol=2e-5)

    def test_results_are_isolated_between_npu_streams(self):
        rows, columns, reduction = 4, 64, 32
        crow = torch.arange(0, (rows + 1) * columns, columns, dtype=torch.int64)
        col = torch.arange(columns, dtype=torch.int64).repeat(rows)
        values = torch.linspace(-1.0, 1.0, rows * columns, dtype=torch.float32)
        mat1_cpu = torch.linspace(
            -0.5, 0.75, rows * reduction, dtype=torch.float32
        ).reshape(rows, reduction)
        mat2_cpu = torch.linspace(
            -1.0, 0.5, reduction * columns, dtype=torch.float32
        ).reshape(reduction, columns)
        pattern = torch.sparse_csr_tensor(
            crow.to("npu"), col.to("npu"), values.to("npu"),
            size=(rows, columns),
        )
        mat1 = mat1_cpu.to("npu")
        mat2 = mat2_cpu.to("npu")
        mat1_second = (mat1_cpu * -0.25).to("npu")
        mat2_second = (mat2_cpu * 1.5).to("npu")
        first_stream = torch.npu.Stream()
        second_stream = torch.npu.Stream()
        current_stream = torch.npu.current_stream()
        first_stream.wait_stream(current_stream)
        second_stream.wait_stream(current_stream)

        with torch.npu.stream(first_stream):
            first = torch.sparse.sampled_addmm(
                pattern, mat1, mat2, alpha=0.75, beta=-0.25
            )
        with torch.npu.stream(second_stream):
            second = torch.sparse.sampled_addmm(
                pattern, mat1_second, mat2_second, alpha=-0.5, beta=0.5
            )
        first_stream.synchronize()
        second_stream.synchronize()

        first_expected = _golden_values(
            CsrValues(crow, col, values), mat1_cpu, mat2_cpu, 0.75, -0.25
        )
        second_expected = _golden_values(
            CsrValues(crow, col, values), mat1_cpu * -0.25, mat2_cpu * 1.5, -0.5, 0.5
        )
        torch.testing.assert_close(
            first.values().cpu(), first_expected, rtol=2e-4, atol=2e-5
        )
        torch.testing.assert_close(
            second.values().cpu(), second_expected, rtol=2e-4, atol=2e-5
        )

    def test_two_dimensional_pattern_broadcasts_over_dense_batch(self):
        pattern, mat1, mat2, crow, col, values = _case(torch.float32)
        mat1_batch = torch.stack((mat1, mat1 * 0.5), dim=0)
        mat2_batch = torch.stack((mat2, mat2 * -2.0), dim=0)
        result = torch.sparse.sampled_addmm(
            pattern, mat1_batch, mat2_batch, alpha=0.75, beta=-0.25
        )
        expected = _golden_values(
            CsrValues(crow, col, values), mat1_batch, mat2_batch, 0.75, -0.25
        )
        self.assertEqual(tuple(result.shape), (2, 3, 4))
        self.assertEqual(tuple(result.values().shape), (2, values.numel()))
        torch.testing.assert_close(result.values().cpu(), expected, rtol=2e-4, atol=2e-5)

        inplace, _, _, _, _, _ = _case(torch.float32)
        original_shape = tuple(inplace.shape)
        with self.assertRaisesRegex(RuntimeError, "out=self cannot broadcast"):
            torch.sparse.sampled_addmm(
                inplace, mat1_batch, mat2_batch,
                alpha=0.75, beta=-0.25, out=inplace,
            )
        self.assertEqual(tuple(inplace.shape), original_shape)

    def test_out_and_out_is_input_alias_contract(self):
        pattern, mat1, mat2, crow, col, values = _case(torch.float32)
        out, _, _, _, _, _ = _case(torch.float32)
        returned = torch.sparse.sampled_addmm(
            pattern, mat1, mat2, alpha=1.25, beta=0.5, out=out
        )
        self.assertIs(returned, out)
        expected = _golden_values(CsrValues(crow, col, values), mat1, mat2, 1.25, 0.5)
        torch.testing.assert_close(out.values().cpu(), expected, rtol=2e-4, atol=2e-5)
        self.assertNotEqual(
            out.crow_indices().untyped_storage().data_ptr(),
            pattern.crow_indices().untyped_storage().data_ptr(),
        )

        inplace, mat1, mat2, crow, col, values = _case(torch.float32)
        crow_ptr = inplace.crow_indices().untyped_storage().data_ptr()
        col_ptr = inplace.col_indices().untyped_storage().data_ptr()
        values_ptr = inplace.values().untyped_storage().data_ptr()
        returned = torch.sparse.sampled_addmm(
            inplace, mat1, mat2, alpha=0.5, beta=-1.0, out=inplace
        )
        self.assertIs(returned, inplace)
        expected = _golden_values(CsrValues(crow, col, values), mat1, mat2, 0.5, -1.0)
        torch.testing.assert_close(inplace.values().cpu(), expected, rtol=2e-4, atol=2e-5)
        self.assertEqual(crow_ptr, inplace.crow_indices().untyped_storage().data_ptr())
        self.assertEqual(col_ptr, inplace.col_indices().untyped_storage().data_ptr())
        self.assertEqual(values_ptr, inplace.values().untyped_storage().data_ptr())

    def test_out_is_input_with_dense_values_overlap(self):
        count = 513  # Exercise the shared scalar across SIMT blocks.
        cases = (("mat2", 0, 0.0), ("mat2", 1, 0.5),
                 ("mat1", 0, 0.0), ("mat1", 1, 0.5),
                 ("both", 0, 0.0), ("both", 1, 0.5))
        for operand, offset, beta in cases:
            with self.subTest(operand=operand, offset=offset, beta=beta):
                rows, columns = (1, count) if operand == "mat1" else (count, 1)
                crow = torch.tensor([0, count]) if rows == 1 else torch.arange(count + 1)
                col = torch.arange(count) if rows == 1 else torch.zeros(count, dtype=torch.int64)
                pattern = torch.sparse_csr_tensor(
                    crow.to("npu"), col.to("npu"), torch.ones(count, device="npu"),
                    size=(rows, columns),
                )
                values = pattern.values()
                shared = values[offset:offset + 1].view(1, 1)
                mat1 = shared if operand == "mat1" else torch.ones((rows, 1), device="npu")
                mat2 = torch.ones((1, columns), device="npu") if operand == "mat1" else shared
                if operand == "both":
                    mat1 = values.view(rows, 1)
                self.assertTrue(mat1.is_contiguous() and mat2.is_contiguous())
                values_ptr = values.data_ptr()

                returned = torch.sparse.sampled_addmm(
                    pattern, mat1, mat2, alpha=2.0, beta=beta, out=pattern
                )

                self.assertIs(returned, pattern)
                self.assertEqual(pattern.values().data_ptr(), values_ptr)
                torch.testing.assert_close(
                    pattern.values().cpu(), torch.full((count,), 2.0 + beta), rtol=0, atol=0
                )
                torch.testing.assert_close(pattern.crow_indices().cpu(), crow)
                torch.testing.assert_close(pattern.col_indices().cpu(), col)

    def test_out_is_input_with_dense_overlap_across_batches(self):
        count = 513
        crow = torch.arange(count + 1).repeat(2, 1)
        col = torch.zeros((2, count), dtype=torch.int64)
        pattern = torch.sparse_csr_tensor(
            crow.to("npu"), col.to("npu"), torch.ones((2, count), device="npu"),
            size=(2, count, 1),
        )
        values_ptr = pattern.values().data_ptr()
        mat1 = torch.ones((2, count, 1), device="npu")
        # Both batches read scalars held in the first batch's output storage.
        mat2 = pattern.values()[0, :2].view(2, 1, 1)
        self.assertTrue(mat2.is_contiguous())

        returned = torch.sparse.sampled_addmm(
            pattern, mat1, mat2, alpha=2.0, beta=0.5, out=pattern
        )

        self.assertIs(returned, pattern)
        self.assertEqual(pattern.values().data_ptr(), values_ptr)
        torch.testing.assert_close(
            pattern.values().cpu(), torch.full((2, count), 2.5), rtol=0, atol=0
        )
        torch.testing.assert_close(pattern.crow_indices().cpu(), crow)
        torch.testing.assert_close(pattern.col_indices().cpu(), col)

    def test_batched_csr_uses_each_batch_pattern(self):
        _, mat1, mat2, crow, col, values = _case(torch.float32)
        crow_batch = torch.stack((crow, crow), dim=0)
        col_batch = torch.stack((col, torch.tensor([1, 3, 0, 2, 3])), dim=0)
        values_batch = torch.stack((values, values * -0.5), dim=0)
        pattern = torch.sparse_csr_tensor(
            crow_batch.to("npu"), col_batch.to("npu"), values_batch.to("npu"),
            size=(2, 3, 4),
        )
        mat1_batch = torch.stack((mat1, mat1 * 1.5), dim=0)
        mat2_batch = torch.stack((mat2, mat2 * -0.25), dim=0)
        result = torch.sparse.sampled_addmm(
            pattern, mat1_batch, mat2_batch, alpha=0.5, beta=0.75
        )
        expected = _golden_values(
            CsrValues(crow_batch[0], col_batch, values_batch), mat1_batch, mat2_batch, 0.5, 0.75
        )
        torch.testing.assert_close(result.values().cpu(), expected, rtol=2e-4, atol=2e-5)

    def test_beta_zero_ignores_old_nan_and_inf(self):
        pattern, mat1, mat2, crow, col, values = _case(torch.float32)
        pattern.values().copy_(
            torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0, -2.0], device="npu")
        )
        result = torch.sparse.sampled_addmm(pattern, mat1, mat2, alpha=1.0, beta=0.0)
        expected = _golden_values(
            CsrValues(crow, col, torch.zeros_like(values)), mat1, mat2, 1.0, 0.0
        )
        self.assertTrue(torch.isfinite(result.values()).all().item())
        torch.testing.assert_close(result.values().cpu(), expected, rtol=2e-4, atol=2e-5)

    def test_validation_errors_stay_on_npu_dispatch_path(self):
        pattern, mat1, mat2, _, _, _ = _case(torch.float32)
        with self.assertRaisesRegex(RuntimeError, "unsupported NPU dtype"):
            bf16_pattern = torch.sparse_csr_tensor(
                pattern.crow_indices(), pattern.col_indices(),
                pattern.values().to(torch.bfloat16), size=pattern.shape,
            )
            torch.sparse.sampled_addmm(
                bf16_pattern, mat1.to(torch.bfloat16), mat2.to(torch.bfloat16)
            )
        with self.assertRaisesRegex(RuntimeError, "same dtype"):
            torch.sparse.sampled_addmm(pattern, mat1, mat2.to(torch.float16))
        with self.assertRaisesRegex(RuntimeError, "cannot be multiplied"):
            torch.sparse.sampled_addmm(pattern, mat1[:, :1], mat2)
        with self.assertRaisesRegex(RuntimeError, "same NPU device"):
            torch.sparse.sampled_addmm(pattern, mat1, mat2.cpu())
        with self.assertRaisesRegex(RuntimeError, "strided layout"):
            torch.ops.aten.sparse_sampled_addmm.default(
                pattern, mat1.to_sparse(), mat2, beta=1, alpha=1
            )

    def test_empty_patterns(self):
        for rows, columns in ((0, 4), (3, 0), (3, 4)):
            with self.subTest(rows=rows, columns=columns):
                crow = torch.zeros(rows + 1, dtype=torch.int32, device="npu")
                col = torch.empty(0, dtype=torch.int32, device="npu")
                values = torch.empty(0, dtype=torch.float32, device="npu")
                pattern = torch.sparse_csr_tensor(crow, col, values, size=(rows, columns))
                mat1 = torch.ones((rows, 2), device="npu")
                mat2 = torch.ones((2, columns), device="npu")
                result = torch.sparse.sampled_addmm(pattern, mat1, mat2)
                torch.npu.synchronize()
                self.assertEqual(tuple(result.shape), (rows, columns))
                self.assertEqual(result.values().numel(), 0)
                torch.testing.assert_close(result.crow_indices().cpu(), crow.cpu())
                torch.testing.assert_close(result.col_indices().cpu(), col.cpu())

    def test_zero_reduction_scales_values(self):
        pattern, _, _, crow, col, values = _case(torch.float32)
        mat1 = torch.empty((3, 0), device="npu")
        mat2 = torch.empty((0, 4), device="npu")
        result = torch.sparse.sampled_addmm(pattern, mat1, mat2, beta=-0.5)
        torch.testing.assert_close(result.values().cpu(), values * -0.5, rtol=0, atol=0)
        pattern.values().fill_(float("nan"))
        result = torch.sparse.sampled_addmm(pattern, mat1, mat2, beta=0.0)
        torch.testing.assert_close(result.values().cpu(), torch.zeros_like(values), rtol=0, atol=0)
        torch.testing.assert_close(result.crow_indices().cpu(), crow)
        torch.testing.assert_close(result.col_indices().cpu(), col)

    def test_queued_calls_preserve_each_result(self):
        pattern, mat1, mat2, crow, col, values = _case(torch.float32)
        expected = [
            _golden_values(CsrValues(crow, col, values), mat1, mat2, alpha, 0.5)
            for alpha in range(1, 9)
        ]
        results = [
            torch.sparse.sampled_addmm(pattern, mat1, mat2, alpha=alpha, beta=0.5)
            for alpha in range(1, 9)
        ]
        torch.npu.synchronize()
        for result, reference in zip(results, expected):
            torch.testing.assert_close(result.values().cpu(), reference, rtol=2e-4, atol=2e-5)

    def test_facade_exposes_public_api(self):
        self.assertIs(torch_npu.sparse.sampled_addmm, torch.sparse.sampled_addmm)

    def test_import_does_not_trigger_jit(self):
        with tempfile.TemporaryDirectory(prefix="sddmm_import_") as directory:
            environment = dict(os.environ, TORCH_EXTENSIONS_DIR=directory)
            script = (
                "import torch, torch_npu, cann_ops_sparse\n"
                "from cann_ops_sparse.ops.sparse.sddmm.sddmm import _sddmm_op_builder as b\n"
                "print('LOADED:', list(type(b)._loaded_ops.keys()))\n"
            )
            completed = subprocess.run(
                [sys.executable, "-c", script], env=environment,
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("LOADED: []", completed.stdout)
            self.assertFalse(any(Path(directory).iterdir()), "import produced JIT build artifacts")


if __name__ == "__main__":
    unittest.main(verbosity=2)
