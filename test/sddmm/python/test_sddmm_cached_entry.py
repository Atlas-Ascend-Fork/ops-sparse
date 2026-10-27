# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression tests for the native cached CSR dispatcher entry point.

Run with the built cann_ops_sparse package and OPS_SPARSE_LIB_DIR set. These
tests validate reuse/invalidation, not a replacement performance measurement.
"""
import unittest
from itertools import product

import torch
from _sddmm_utils import require_arch22
from cann_ops_sparse.ops.sparse.sddmm import CsrOperands, ensure_sddmm_loaded, prepare_sddmm_csr

import torch_npu  # noqa: F401


class CachedCsrEntryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        require_arch22()
        ensure_sddmm_loaded()

    def operands(self, dtype=torch.float32, base=0, strided=False):
        torch.manual_seed(163)
        rp = (torch.tensor([0, 2, 3, 5], dtype=torch.int32) + base).npu()
        ci = (torch.tensor([0, 2, 1, 0, 3], dtype=torch.int32) + base).npu()
        values = torch.randn(5, dtype=dtype).npu()
        # K >= 64 exercises preprocessed Y for the supported numeric paths.
        x = torch.randn(1, 3, 128 if strided else 64, dtype=dtype).npu()
        y = torch.randn(1, 64, 128 if strided else 64, dtype=dtype).npu()
        if strided:
            x, y = x[..., ::2], y[..., ::2]
        return [rp, ci, values, x, y]

    def check_call(self, inputs, base=0, alpha=0.75, beta=0.5):
        rp, ci, values, x, y = inputs
        rp_cpu, ci_cpu = rp.cpu().long() - base, ci.cpu().long() - base
        work = torch.complex128 if values.is_complex() else torch.float64
        rows = torch.repeat_interleave(torch.arange(3), rp_cpu[1:] - rp_cpu[:-1])
        dots = x.cpu().to(work)[0] @ y.cpu().to(work)[0]
        golden = (alpha * dots[rows, ci_cpu] + beta * values.cpu().to(work)).to(values.dtype)
        output = torch.ops.ops_sparse_test.sddmm_npu(
            "csr", rp, ci, values, x, y, alpha, beta, base, 1, "row", 1
        )
        self.assertEqual(output.shape, (1, 5))
        self.assertEqual(output.data_ptr(), values.data_ptr())
        rtol, atol = (2**-9, 2**-9) if values.dtype == torch.float16 else (2**-10, 2**-16)
        torch.testing.assert_close(output[0].cpu(), golden, rtol=rtol, atol=atol)

    def test_repeated_execute_reads_current_values(self):
        for dtype, base in product((torch.float16, torch.float32, torch.complex64), (0, 1)):
            with self.subTest(dtype=dtype, base=base):
                inputs = self.operands(dtype, base)
                for _ in range(5):
                    inputs[2].add_(0.25)
                    self.check_call(inputs, base)

    def test_dense_mutations_and_aliases_invalidate_preprocessing(self):
        for dtype, strided in product((torch.float16, torch.float32, torch.complex64), (False, True)):
            with self.subTest(dtype=dtype, strided=strided):
                inputs = self.operands(dtype, strided=strided)
                self.check_call(inputs)
                inputs[3][0].add_(0.125)
                self.check_call(inputs)
                inputs[4][0].mul_(-0.5)
                self.check_call(inputs)

    def test_pattern_mutations_invalidate_preprocessing(self):
        inputs = self.operands()
        self.check_call(inputs)
        inputs[1].copy_(torch.tensor([1, 3, 0, 1, 2], dtype=torch.int32).npu())
        self.check_call(inputs)
        inputs[0].copy_(torch.tensor([0, 1, 3, 5], dtype=torch.int32).npu())
        self.check_call(inputs)

    def test_storage_rebinding_and_new_tensor_identity(self):
        inputs = self.operands()
        self.check_call(inputs)
        inputs[4].set_(torch.randn_like(inputs[4]))
        self.check_call(inputs)
        inputs[3] = inputs[3].clone()
        self.check_call(inputs)
        inputs[2].set_(torch.randn_like(inputs[2]))
        self.check_call(inputs)
        self.check_call(inputs, alpha=-0.25, beta=0.75)

    def test_inference_tensors_do_not_reuse_unversioned_state(self):
        with torch.inference_mode():
            inputs = self.operands()
            self.check_call(inputs)
            inputs[4].add_(0.5)
            self.check_call(inputs)

    def test_stream_change_rebuilds_state(self):
        inputs = self.operands()
        self.check_call(inputs)
        torch.npu.synchronize()
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            self.check_call(inputs)
            inputs[4].mul_(0.5)
            self.check_call(inputs)
        stream.synchronize()
        self.check_call(inputs)

    def test_queued_calls_preserve_copy_execute_output_order(self):
        inputs = self.operands(torch.float16)
        rp, ci, values, x, y = inputs
        self.check_call(inputs)
        rows = torch.tensor([0, 0, 1, 2, 2])
        dots = (x.cpu().double()[0] @ y.cpu().double()[0])[rows, ci.cpu().long()]
        torch.npu.synchronize()
        stream = torch.npu.Stream()
        outputs = []
        with torch.npu.stream(stream):
            for i in range(30):
                values.fill_(i / 8)
                output = torch.ops.ops_sparse_test.sddmm_npu(
                    "csr", rp, ci, values, x, y, 0.75, 0.5, 0, 1, "row", 1
                )
                outputs.append(output.clone())
            actual = torch.cat(outputs).cpu()
        golden = torch.stack([(0.75 * dots + 0.5 * i / 8).half() for i in range(30)])
        torch.testing.assert_close(actual, golden, rtol=2**-9, atol=2**-9)

    def test_invalid_mutated_pattern_is_not_silently_reused(self):
        inputs = self.operands()
        self.check_call(inputs)
        inputs[1].fill_(64)  # Column == N is out of bounds.
        with self.assertRaisesRegex(RuntimeError, "aclsparseSDDMMPreprocess"):
            torch.ops.ops_sparse_test.sddmm_npu(
                "csr", *inputs, 0.75, 0.5, 0, 1, "row", 1
            )

    def test_mixed_fp16_operands_fp32_values(self):
        inputs = self.operands(torch.float16)
        inputs[2] = inputs[2].float()
        self.check_call(inputs)
        inputs[4].add_(0.25)
        self.check_call(inputs)

    def check_partition_values(self, operands, dots, base):
        rp, ci, values, x, y = (operands.row_ptr, operands.col_ind, operands.values, operands.x, operands.y)
        dtype = values.dtype
        for _ in range(2):
            golden = (0.75 * dots + 0.25 * values.cpu().double()).to(dtype)
            result = torch.ops.ops_sparse_test.sddmm_npu(
                "csr", rp, ci, values, x, y, 0.75, 0.25, base, 1, "row", 1
            )
            torch.testing.assert_close(
                result[0].cpu(), golden, rtol=2**-9, atol=2**-16
            )

    def test_precomputed_partition_bounds_with_empty_and_skewed_rows(self):
        torch.manual_seed(917)
        for (m, k), shape in product(((256, 32), (257, 128), (511, 512)), ('regular', 'irregular', 'one_long_row')):
            n = 160
            if shape == "regular":
                counts = torch.full((m,), 16, dtype=torch.int64)
            elif shape == "irregular":
                counts = torch.randint(0, 67, (m,))
                counts[::5] = 0
            else:
                counts = torch.zeros(m, dtype=torch.int64)
                counts[m // 2] = m * 8 + 3
            rp0 = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
            row = torch.repeat_interleave(torch.arange(m), counts)
            ci0 = ((torch.arange(row.numel()) - rp0[row]) + row * 3) % n
            x = (torch.randn(1, m, k) * 0.125).half().npu()
            y = (torch.randn(1, k, n) * 0.125).half().npu()
            dots = (x.cpu().double()[0] @ y.cpu().double()[0])[row, ci0]
            for dtype, base in product((torch.float16, torch.float32), (0, 1)):
                with self.subTest(m=m, k=k, shape=shape, dtype=dtype, base=base):
                    rp, ci = (rp0 + base).int().npu(), (ci0 + base).int().npu()
                    values = torch.randn(row.numel(), dtype=dtype).npu()
                    self.check_partition_values(CsrOperands(rp, ci, values, x, y), dots, base)

    def test_guarded_numeric_launch_never_writes_after_invalid_pattern(self):
        m, n, k = 257, 128, 32
        for (base, dtype), which in product(product((0, 1), (torch.float16, torch.float32)), ('rows', 'columns')):
            with self.subTest(base=base, dtype=dtype, which=which):
                rp = (torch.arange(m + 1, dtype=torch.int32) * 16 + base).npu()
                ci = (torch.arange(m * 16, dtype=torch.int32) % 16 + base).npu()
                values = torch.full((m * 16,), 3.0, dtype=dtype).npu()
                x = torch.ones(m, k, dtype=torch.float16).npu()
                y = torch.ones(n, k, dtype=torch.float16).npu().t()
                state = prepare_sddmm_csr(CsrOperands(rp, ci, values, x, y), 0.75, 0.5, base)
                state.execute()
                expected = torch.full((m * 16,), 25.5, dtype=dtype)
                torch.testing.assert_close(values.cpu(), expected, rtol=0, atol=0)
                target = rp if which == "rows" else ci
                version = ensure_sddmm_loaded().tensor_version(target)
                target.data[target.numel() // 2].add_(1)
                self.assertEqual(ensure_sddmm_loaded().tensor_version(target), version)
                with self.assertRaisesRegex(RuntimeError, "aclsparseSDDMM"):
                    state.execute()
                torch.testing.assert_close(values.cpu(), expected, rtol=0, atol=0)

    def check_snapshot_mutation(self, operands, original, selection):
        rp, ci, values, x, y = (operands.row_ptr, operands.col_ind, operands.values, operands.x, operands.y)
        rows, columns = original
        which, pos = selection
        rp.copy_(rows)
        ci.copy_(columns)
        state = prepare_sddmm_csr(CsrOperands(rp, ci, values, x, y), 1.0, 0.0, 0)
        state.execute()
        torch.testing.assert_close(values.cpu(), torch.ones(columns.numel()))
        target = rp if which == "rows" else ci
        old_version = ensure_sddmm_loaded().tensor_version(target)
        target.data[pos].add_(1)  # Deliberately bypass the source's version counter.
        self.assertEqual(ensure_sddmm_loaded().tensor_version(target), old_version)
        with self.assertRaisesRegex(RuntimeError, "aclsparseSDDMM"):
            state.execute()
        torch.testing.assert_close(values.cpu(), torch.ones(columns.numel()))

    def test_native_snapshot_detects_unversioned_changes_at_tile_boundaries(self):
        # Covers 1024/2048/4096 boundaries, tails and multiple comparison tiles
        # on the same core. Changes deliberately bypass Tensor version counters.
        m = 8193
        rows = torch.arange(m + 1, dtype=torch.int32) * 16
        rows[-1] += 1
        columns = torch.arange(int(rows[-1]), dtype=torch.int32) % 8
        rp, ci = rows.npu(), columns.npu()
        values = torch.zeros(columns.numel(), device="npu:0")
        x, y = torch.ones(m, 1, device="npu:0"), torch.ones(1, 8, device="npu:0")
        for which, positions in (
            ("rows", (0, 1023, 1024, 2047, 2048, 4095, 4096, 8192, m)),
            ("columns", (0, 1023, 1024, 2047, 2048, 4095, 4096, 8191, 8192,
                         81920, columns.numel() - 1)),
        ):
            for pos in positions:
                with self.subTest(which=which, pos=pos):
                    self.check_snapshot_mutation(CsrOperands(rp, ci, values, x, y), (rows, columns), (which, pos))
        for replacement in (-(2**31), 2**31 - 1):
            rp.copy_(rows)
            ci.copy_(columns)
            state = prepare_sddmm_csr(CsrOperands(rp, ci, values, x, y), 1.0, 0.0, 0)
            ci.data[2048] = replacement
            with self.assertRaisesRegex(RuntimeError, "aclsparseSDDMM"):
                state.execute()
        rp.copy_(rows)
        ci.copy_(columns)
        state = prepare_sddmm_csr(CsrOperands(rp, ci, values, x, y), 1.0, 0.0, 0)
        # Opposite differences in the same lane on successive core-owned tiles
        # must not cancel, as they would in a checksum-based approximation.
        ci.data[0].add_(1)
        ci.data[81920].sub_(1)
        with self.assertRaisesRegex(RuntimeError, "aclsparseSDDMM"):
            state.execute()


if __name__ == "__main__":
    unittest.main(verbosity=2)
