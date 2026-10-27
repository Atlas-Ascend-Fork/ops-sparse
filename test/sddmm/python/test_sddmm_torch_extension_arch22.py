# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Focused cache, precision, batching and output-ownership regression checks."""

import unittest
from itertools import product
from dataclasses import dataclass

import torch
from _sddmm_utils import BsrOperands, BsrOptions, assert_accuracy, bsr_values, check_csr, require_arch22
from cann_ops_sparse.ops.sparse.sddmm import CsrOperands, ensure_sddmm_loaded, prepare_sddmm_csr

import torch_npu  # noqa: F401


@dataclass(frozen=True)
class DirectOptions:
    alpha: complex = 1.
    beta: complex = 0.
    initial: torch.Tensor | None = None


class DevicePathsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        require_arch22()
        ensure_sddmm_loaded()

    def setUp(self):
        torch.manual_seed(71)

    def direct_values(self, kind, x, y, options=DirectOptions()):
        module = ensure_sddmm_loaded()
        m = x.size(0)
        if kind == "csr":
            rp = torch.arange(m + 1, dtype=torch.int32).npu()
            ci = torch.arange(m, dtype=torch.int32).npu()
            values = torch.zeros(m, dtype=x.dtype).npu() if options.initial is None else options.initial
            return module.sddmm_csr(rp, ci, values, x, y, options.alpha, options.beta, 0)
        bd = m
        rp = torch.tensor([0, 1], dtype=torch.int32).npu()
        ci = torch.tensor([0], dtype=torch.int32).npu()
        values = torch.zeros(1, bd, bd, dtype=x.dtype).npu() if options.initial is None else options.initial
        return module.sddmm_bsr_pattern(
            rp, ci, values, x.unsqueeze(0), y.unsqueeze(0),
            complex(options.alpha), complex(options.beta), 0, bd, False, 1
        )

    def test_complex_high_dynamic_range(self):
        # K=1 takes the separate CSR products; N=K=32 takes fused products.
        # BSR block sizes exercise vector and Cube calculations.
        for kind, m, k in (("csr", 1, 1), ("csr", 32, 32), ("bsr", 4, 1), ("bsr", 64, 32)):
            x = torch.zeros(m, k, dtype=torch.complex64)
            y = torch.zeros(k, m, dtype=torch.complex64)
            x[:, 0], y[0, :] = 1 + 1j, complex(2e38, 1e38)
            golden = x.to(torch.complex128) @ y.to(torch.complex128)
            expected = golden.diag() if kind == "csr" else golden.reshape(1, m, m)
            actual = self.direct_values(kind, x.npu(), y.npu()).cpu().reshape(expected.shape)
            with self.subTest(kind=kind, m=m, k=k):
                for got, ref in ((actual.real, expected.real), (actual.imag, expected.imag)):
                    self.assertTrue(bool(torch.isfinite(got).all()))
                    torch.testing.assert_close(got.double(), ref, rtol=2**-10, atol=0)

    def test_complex_zero_scalar_components(self):
        for kind, m, k in (("csr", 1, 1), ("csr", 32, 32), ("bsr", 4, 1), ("bsr", 64, 32)):
            x = torch.zeros(m, k, dtype=torch.complex64).npu()
            y = torch.zeros(k, m, dtype=torch.complex64).npu()
            shape = (m,) if kind == "csr" else (1, m, m)
            for beta, old, real, imag in (
                (1., complex(float("inf"), 2), float("inf"), 2.),
                (1j, complex(float("inf"), 2), -2., float("inf")),
            ):
                values = torch.full(shape, old, dtype=torch.complex64).npu()
                actual = self.direct_values(kind, x, y, DirectOptions(beta=beta, initial=values)).cpu().reshape(shape)
                with self.subTest(kind=kind, m=m, beta=beta):
                    torch.testing.assert_close(actual.real, torch.full(shape, real), rtol=0, atol=0)
                    torch.testing.assert_close(actual.imag, torch.full(shape, imag), rtol=0, atol=0)
        # K=0 uses the scale-only entry rather than the dot-product kernels.
        for beta, real, imag in ((1., float("inf"), 2.), (1j, -2., float("inf"))):
            x = torch.empty(1, 0, dtype=torch.complex64).npu()
            y = torch.empty(0, 1, dtype=torch.complex64).npu()
            values = torch.tensor([complex(float("inf"), 2)], dtype=torch.complex64).npu()
            actual = self.direct_values("csr", x, y, DirectOptions(beta=beta, initial=values)).cpu()
            torch.testing.assert_close(actual.real, torch.tensor([real]), rtol=0, atol=0)
            torch.testing.assert_close(actual.imag, torch.tensor([imag]), rtol=0, atol=0)

    def dense_view_inputs(self, m, k, layout):
        width = 2 * k if layout == "strided" else k
        x = torch.zeros(m, width, dtype=torch.complex64).npu()
        x = x[:, ::2] if layout == "strided" else x
        if layout == "column":
            y = torch.zeros(m, k, dtype=torch.complex64).npu().t()
        else:
            width = 2 * m if layout == "strided" else m
            y = torch.zeros(k, width, dtype=torch.complex64).npu()
            y = y[:, ::2] if layout == "strided" else y
        x[0, 0], y[0, 0] = 1 + 2j, 3 + 4j
        return x, y

    def negative_complex_view(self, x):
        # Conjugate imaginary views carry the lazy negative bit. Reinterpret the
        # original real/imaginary storage without copying or changing that bit.
        real = torch.view_as_real(x)
        negative = torch.as_strided(x.conj().imag, real.shape, real.stride(), 2 * x.storage_offset())
        return torch.view_as_complex(negative)

    def check_view_updates(self, kind, inputs, views, case):
        x, y = inputs
        xv, yv = views
        for step in range(2):
            before_x, before_y = x.cpu().clone(), y.cpu().clone()
            golden = xv.cpu().to(torch.complex128) @ yv.cpu().to(torch.complex128)
            expected = golden.diag() if kind == "csr" else golden.unsqueeze(0)
            actual = self.direct_values(kind, xv, yv).cpu().reshape(expected.shape)
            with self.subTest(**case, step=step):
                torch.testing.assert_close(actual, expected.to(torch.complex64), rtol=2**-10, atol=2**-16)
                torch.testing.assert_close(x.cpu(), before_x, rtol=0, atol=0)
                torch.testing.assert_close(y.cpu(), before_y, rtol=0, atol=0)
            x[0, 0].add_(.25 + .5j)
            y[0, 0].add_(.5 - .25j)

    def test_direct_conjugate_and_negative_views(self):
        for kind, k, layout, bits in product(
            ("csr", "bsr"), (1, 32), ("row", "column", "strided"),
            ((True, False), (False, True), (True, True)),
        ):
            m = 4 if kind == "bsr" else 32
            x, y = self.dense_view_inputs(m, k, layout)
            for negative in (False, True):
                xv = self.negative_complex_view(x) if negative else x
                xv = xv.conj() if bits[0] else xv
                yv = y.conj() if bits[1] else y
                self.assertEqual(xv.is_neg(), negative)
                case = dict(kind=kind, k=k, layout=layout, bits=bits, negative=negative)
                self.check_view_updates(kind, (x, y), (xv, yv), case)

    def check_scalar_overflow(self, kind, inputs, options):
        with self.assertRaisesRegex(RuntimeError, "overflow|convert"):
            self.direct_values(kind, *inputs, options)

    def test_checked_scalar_overflow(self):
        for kind, dtype in product(("csr", "bsr"), (torch.float32, torch.complex64)):
            m = 4 if kind == "bsr" else 1
            x = torch.ones(m, 1, dtype=dtype).npu()
            y = torch.ones(1, m, dtype=dtype).npu()
            invalid = (1e40, -1e40)
            if dtype == torch.complex64:
                invalid += (complex(0, 1e40), complex(0, -1e40))
            for value, name in product(invalid, ("alpha", "beta")):
                with self.subTest(kind=kind, dtype=dtype, name=name, value=value):
                    self.check_scalar_overflow(kind, (x, y), DirectOptions(**{name: value}))

    def test_checked_scalar_nan_and_inf(self):
        for kind, dtype in product(("csr", "bsr"), (torch.float32, torch.complex64)):
            m = 4 if kind == "bsr" else 1
            x = torch.ones(m, 1, dtype=dtype).npu()
            y = torch.ones(1, m, dtype=dtype).npu()
            accepted = (float("inf"), -float("inf"), float("nan"))
            if dtype == torch.complex64:
                accepted += (complex(0, float("inf")), complex(0, float("nan")))
            for value, name in product(accepted, ("alpha", "beta")):
                with self.subTest(kind=kind, dtype=dtype, name=name, value=value):
                    actual = self.direct_values(kind, x, y, DirectOptions(**{name: value})).cpu()
                    self.assertTrue(bool((~torch.isfinite(actual)).any()))

    def test_csr_four_broadcast_modes(self):
        for base, xb, yb in product((0, 1), (1, 3), (1, 3)):
            rp = (torch.tensor([0, 2, 3], dtype=torch.int64) + base).npu()
            ci = (torch.tensor([0, 2, 1], dtype=torch.int64) + base).npu()
            values = torch.randn(3).npu()
            x = torch.randn(xb, 2, 7).npu()
            y = torch.randn(yb, 7, 3).npu()
            before = values.cpu().clone()
            dense = x.cpu().double().expand(3, -1, -1) @ y.cpu().double().expand(3, -1, -1)
            expected = (.75 * dense[:, [0, 0, 1], [0, 2, 1]] + .25 * before).float()
            actual = torch.ops.ops_sparse_test.sddmm_npu(
                "csr", rp, ci, values, x, y, .75, .25, base, 1, "row", 3
            )
            with self.subTest(base=base, xb=xb, yb=yb):
                self.assert_accuracy(actual.cpu(), expected)
                torch.testing.assert_close(values.cpu(), before, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "batch counts"):
            torch.ops.ops_sparse_test.sddmm_npu("csr", rp, ci, values, x, y, 1., 0., base, 1, "row", 2)
        for batch in (0, 65536):
            with self.assertRaisesRegex(RuntimeError, "batch count"):
                torch.ops.ops_sparse_test.sddmm_npu("csr", rp, ci, values, x, y, 1., 0., base, 1, "row", batch)

    def test_prepared_strided_numeric_updates(self):
        for dtype in (torch.float32, torch.complex64):
            rp = torch.tensor([0, 2, 3], dtype=torch.int32).npu()
            ci = torch.tensor([0, 2, 1], dtype=torch.int32).npu()
            values = torch.ones(3, dtype=dtype).npu()
            x = torch.randn(2, 14, dtype=dtype).npu()[:, ::2]
            y = torch.randn(7, 6, dtype=dtype).npu()[:, ::2]
            state = prepare_sddmm_csr(CsrOperands(rp, ci, values, x, y), .75, 0., 0)
            for change in (0., .25, -.5):
                x.add_(change)
                y.add_(2 * change)
                expected = .75 * (x.cpu().to(torch.complex128 if dtype.is_complex else torch.float64) @
                                  y.cpu().to(torch.complex128 if dtype.is_complex else torch.float64))
                expected = expected[torch.tensor([0, 0, 1]), ci.cpu().long()].to(dtype)
                self.assert_accuracy(state.execute().cpu(), expected)
            x.transpose_(0, 1)
            with self.assertRaisesRegex(RuntimeError, "binding changed"):
                state.execute()

    def check_prepared_conjugate_updates(self, x, y, conjugate):
        conj_x, conj_y = conjugate
        x_view, y_view = x.conj() if conj_x else x, y.conj() if conj_y else y
        self.assertEqual(x_view.is_conj(), conj_x)
        self.assertEqual(y_view.is_conj(), conj_y)
        rp = torch.arange(17, dtype=torch.int32).npu()
        ci = torch.arange(16, dtype=torch.int32).npu()
        values = torch.zeros(16, dtype=torch.complex64).npu()
        # Call pybind directly, without the dispatcher or Python adapter.
        state = ensure_sddmm_loaded().prepare_sddmm_csr(
            rp, ci, values, x_view, y_view, complex(1), complex(0), 0
        )
        for step in range(3):
            before_x, before_y = x.cpu().clone(), y.cpu().clone()
            logical_x = before_x.conj() if conj_x else before_x
            logical_y = before_y.conj() if conj_y else before_y
            expected = (logical_x.to(torch.complex128) @ logical_y.to(torch.complex128)).diag()
            if step == 0 and conj_x and not conj_y:
                self.assertEqual(expected[0].item(), 11 - 2j)
            actual = state.execute()
            self.assertEqual(actual.data_ptr(), values.data_ptr())
            torch.testing.assert_close(actual.cpu(), expected.to(torch.complex64), rtol=2**-10, atol=2**-16)
            torch.testing.assert_close(x.cpu(), before_x, rtol=0, atol=0)
            torch.testing.assert_close(y.cpu(), before_y, rtol=0, atol=0)
            x[0, 0].add_(.25 + .5j)
            y[0, 0].add_(.5 - .25j)

    def test_direct_prepared_conjugate_views_refresh_numeric_inputs(self):
        for k, layout, conjugate in product(
            (16, 64), ("row", "column", "strided"), ((True, False), (False, True), (True, True))
        ):
            x = torch.zeros(16, k, dtype=torch.complex64).npu()
            if layout == "column":
                y = torch.zeros(16, k, dtype=torch.complex64).npu().t()
            elif layout == "strided":
                x = torch.zeros(16, 2 * k, dtype=torch.complex64).npu()[:, ::2]
                y = torch.zeros(k, 32, dtype=torch.complex64).npu()[:, ::2]
            else:
                y = torch.zeros(k, 16, dtype=torch.complex64).npu()
            x[0, 0], y[0, 0] = 1 + 2j, 3 + 4j
            with self.subTest(k=k, layout=layout, conjugate=conjugate):
                self.check_prepared_conjugate_updates(x, y, conjugate)

    def test_index_conversion_and_prepared_pattern_changes(self):
        x = torch.ones(2, 7).npu()
        y = torch.ones(7, 3).npu()
        values = torch.ones(3).npu()
        for dtype, strided in product((torch.int32, torch.int64), (False, True)):
            rp = torch.tensor([0, 2, 3], dtype=dtype).npu()
            ci = torch.tensor([0, 2, 1], dtype=dtype).npu()
            if strided:
                rp = rp.repeat_interleave(2)[::2]
                ci = ci.repeat_interleave(2)[::2]
            state = prepare_sddmm_csr(CsrOperands(rp, ci, values, x, y), 1., 0., 0)
            torch.testing.assert_close(state.execute(), torch.full_like(values, 7.))
            ci[0] = 1
            with self.subTest(dtype=dtype, strided=strided), self.assertRaises(RuntimeError):
                state.execute()
        rp = torch.tensor([0, 2, 3], dtype=torch.int64).npu()
        for bad in (2**31, 2**32, 2**32 + 1, -(2**32), -(2**63), 2**63 - 1):
            ci = torch.tensor([bad, 2, 1], dtype=torch.int64).npu()
            with self.assertRaisesRegex(RuntimeError, "aclsparseSDDMMPreprocess"):
                prepare_sddmm_csr(CsrOperands(rp, ci, values, x, y), 1., 0., 0)
            with self.assertRaisesRegex(RuntimeError, "sddmm_call_local"):
                ensure_sddmm_loaded().sddmm_csr(rp, ci, values, x, y, 1., 0., 0)
        with self.assertRaisesRegex(RuntimeError, "int32 or int64"):
            prepare_sddmm_csr(CsrOperands(rp.float(), rp.float(), values, x, y), 1., 0., 0)

    def test_prepared_index_refresh_submission_order(self):
        x, y, values = torch.ones(2, 7).npu(), torch.ones(7, 3).npu(), torch.ones(3).npu()
        for dtype, strided, repeat in product((torch.int32, torch.int64), (False, True), range(32)):
            rp = torch.tensor([0, 2, 3], dtype=dtype).npu()
            ci = torch.tensor([0, 2, 1], dtype=dtype).npu()
            if strided:
                rp, ci = rp.repeat_interleave(2)[::2], ci.repeat_interleave(2)[::2]
            state = prepare_sddmm_csr(CsrOperands(rp, ci, values, x, y), 1., 0., 0)
            torch.testing.assert_close(state.execute(), torch.full_like(values, 7.))
            ci[0] = 1
            with self.subTest(dtype=dtype, strided=strided, repeat=repeat):
                with self.assertRaises(RuntimeError):
                    state.execute()

    def test_prepared_input_validation(self):
        rp = torch.tensor([0, 2, 3], dtype=torch.int32).npu()
        ci = torch.tensor([0, 2, 1], dtype=torch.int32).npu()
        values = torch.ones(3).npu()
        x, y = torch.ones(2, 7).npu(), torch.ones(7, 3).npu()
        for bad_x in (x.flatten(), x.cpu()):
            with self.assertRaises(RuntimeError):
                prepare_sddmm_csr(CsrOperands(rp, ci, values, bad_x, y), 1., 0., 0)
        with self.assertRaises(RuntimeError):
            prepare_sddmm_csr(CsrOperands(rp, ci, values.reshape(1, 3), x, y), 1., 0., 0)

    def test_index_conversion_tile_boundaries(self):
        x, y = torch.ones(1, 7).npu(), torch.ones(7, 37).npu()
        for count, strided in product((1023, 1024, 1025, 32769), (False, True)):
            rp = torch.tensor([0, count], dtype=torch.int64).npu()
            ci = (torch.arange(count, dtype=torch.int64) % 37).npu()
            if strided:
                ci = ci.repeat_interleave(2)[::2]
            values = torch.ones(count).npu()
            state = prepare_sddmm_csr(CsrOperands(rp, ci, values, x, y), 1., 0., 0)
            torch.testing.assert_close(state.execute().cpu(), torch.full((count,), 7.))
            ci[-1] += 2**32
            with self.assertRaisesRegex(RuntimeError, "aclsparseSDDMM"):
                state.execute()
            torch.testing.assert_close(values.cpu(), torch.full((count,), 7.))
        rp = torch.tensor([0, 2**32 + 1], dtype=torch.int64).npu()
        with self.assertRaisesRegex(RuntimeError, "aclsparseSDDMMPreprocess"):
            prepare_sddmm_csr(
                CsrOperands(rp, torch.zeros(1, dtype=torch.int64).npu(), torch.ones(1).npu(), x, y), 1.0, 0.0, 0
            )

    def test_bsr_four_broadcast_modes(self):
        for bd, xb, yb in product((2, 16), (1, 3), (1, 3)):
            rp = torch.tensor([1, 2, 3], dtype=torch.int32).npu()
            ci = torch.tensor([2, 1], dtype=torch.int32).npu()
            values = torch.randn(2, bd, bd).npu()
            x = torch.randn(xb, 2 * bd, 7).npu()
            y = torch.randn(yb, 7, 2 * bd).npu()
            initial = values.cpu().clone()
            dense = x.cpu().double().expand(3, -1, -1) @ y.cpu().double().expand(3, -1, -1)
            blocks = torch.stack((dense[:, :bd, bd:], dense[:, bd:, :bd]), dim=1)
            expected = (.75 * blocks + .25 * initial).float()
            actual = bsr_values(
                BsrOperands(rp, ci, values, x, y),
                BsrOptions(alpha=.75, beta=.25, base=1, block_dim=bd, direction="row", batch=3)
            )
            self.assertEqual(tuple(actual.shape), (3, 2, bd, bd))
            self.assert_accuracy(actual.cpu(), expected)
            torch.testing.assert_close(values.cpu(), initial, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "batch counts"):
            bsr_values(
                BsrOperands(rp, ci, values, x, y),
                BsrOptions(alpha=1., beta=0., base=1, block_dim=bd, direction="row", batch=2)
            )

    def test_public_dispatch_and_zero_reduction(self):
        self.assertTrue(torch.ops.aten.sparse_sampled_addmm.default.has_kernel_for_dispatch_key(
            "SparseCsrPrivateUse1"))
        for dtype in (torch.float16, torch.float32, torch.complex64):
            rp = torch.tensor([0, 2, 3], dtype=torch.int32).npu()
            ci = torch.tensor([0, 2, 1], dtype=torch.int32).npu()
            values = torch.ones(3, dtype=dtype).npu()
            x = torch.empty(2, 0, dtype=dtype).npu()
            y = torch.empty(0, 3, dtype=dtype).npu()
            sparse = torch.sparse_csr_tensor(rp, ci, values, size=(2, 3), device=values.device)
            for beta in (0., .5):
                result = torch.ops.aten.sparse_sampled_addmm(sparse, x, y, beta=beta, alpha=1.)
                torch.testing.assert_close(result.values().cpu(), torch.full((3,), beta, dtype=dtype), rtol=0, atol=0)
                torch.testing.assert_close(values.cpu(), torch.ones(3, dtype=dtype), rtol=0, atol=0)

    assert_accuracy = assert_accuracy

    def test_native_random_group_and_k_tail(self):
        for k, dtype in product((32, 100, 128, 192), (torch.float16, torch.float32, torch.bfloat16)):
            m, n, per_row, base = 256, 512, 64, 1
            rp = (torch.arange(m + 1, dtype=torch.int32) * per_row + base).npu()
            ci = (
                ((torch.arange(m)[:, None] * 17 + torch.arange(per_row)[None, :]) % n)
                .sort(1)
                .values.flatten()
                .int()
                .add(base)
                .npu()
            )
            values = torch.randn(m * per_row, dtype=dtype).npu()
            x = torch.randn(m, k, dtype=dtype).npu()
            y = torch.randn(k, n, dtype=dtype).npu().t().contiguous().t()
            initial = tuple(t.cpu().clone() for t in (rp, ci, values, x.unsqueeze(0), y.unsqueeze(0)))
            state = prepare_sddmm_csr(CsrOperands(rp, ci, values, x, y), .75, -.25, base)
            output = state.execute().unsqueeze(0)
            case = {"base": base, "m": m, "alpha": .75, "beta": -.25,
                    "dtype": str(dtype).split(".")[-1]}
            with self.subTest(k=k, dtype=dtype):
                self.assertTrue(check_csr(case, initial, output)["pass"])

    def test_csr_nonconsecutive_interior_and_reduction_tails(self):
        for dtype, base, count in product((torch.float16, torch.float32), (0, 1), (56, 64, 72)):
            m, n, k = 256, 512, 128
            rp = torch.arange(m + 1, dtype=torch.int32) * count + base
            ci = (torch.arange(m)[:, None] % 8 * 17 + torch.arange(count)[None, :]).int()
            ci[::3, 1:3] = ci[::3, 1:3].flip(1)
            ci[1::3, 5] = ci[1::3, 4]
            ci[2::3, 7] = n - 1
            ci = ci.flatten() + base
            values = torch.randn(m * count, dtype=dtype)
            x = torch.randn(m, k, dtype=dtype)
            y = torch.randn(k, n, dtype=dtype).t().contiguous().t()
            before = (rp, ci, values.clone(), x.unsqueeze(0), y.unsqueeze(0))
            operands = tuple(t.npu() for t in (rp, ci, values, x, y))
            state = prepare_sddmm_csr(CsrOperands(*operands), .75, -.25, base)
            output = state.execute().unsqueeze(0)
            case = {"base": base, "m": m, "alpha": .75, "beta": -.25,
                    "dtype": str(dtype).split(".")[-1]}
            with self.subTest(dtype=dtype, base=base, count=count):
                self.assertTrue(check_csr(case, before, output)["pass"])
                for index in (0, 1, 3, 4):
                    self.assertTrue(torch.equal(operands[index].cpu(), (rp, ci, values, x, y)[index]))

    def test_preprocessed_row_hints_reject_false_contiguity(self):
        m, n, k, per = 256, 128, 128, 64
        rp = (torch.arange(m + 1, dtype=torch.int32) * per).npu()
        columns = torch.arange(per, dtype=torch.int32).repeat(m, 1)
        columns[3::7, 17] = 16  # Same endpoints, but a duplicate and a missing interior index.
        ci = columns.flatten().npu()
        values = torch.randn(m * per, dtype=torch.float16).npu()
        x = torch.randn(m, k, dtype=torch.float16).npu()
        y = torch.randn(k, n, dtype=torch.float16).npu().t().contiguous().t()
        case = {'m': m, 'base': 0, 'alpha': .75, 'beta': -.25, 'dtype': 'float16'}
        for changed in (False, True):
            initial = tuple(t.cpu().clone() for t in (rp, ci, values, x.unsqueeze(0), y.unsqueeze(0)))
            state = prepare_sddmm_csr(CsrOperands(rp, ci, values, x, y), .75, -.25, 0)
            with self.subTest(changed=changed):
                self.assertTrue(check_csr(case, initial, state.execute().unsqueeze(0))['pass'])
            before = values.cpu().clone()
            ci[0] = 2 if not changed else 3
            with self.assertRaises(RuntimeError):
                state.execute()
            torch.testing.assert_close(values.cpu(), before, rtol=0, atol=0)

    def test_bsr_batch_layout_and_mixed_precision(self):
        pairs = ((torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16),
                 (torch.float32, torch.float32), (torch.complex64, torch.complex64),
                 (torch.float16, torch.float32), (torch.bfloat16, torch.float32))
        for ((dtype, ctype), bd), (direction, base, k) in product(
            product(pairs, (2, 4, 8, 16, 32, 64, 128)), (("row", 0, 128), ("col", 1, 100))
        ):
            counts = torch.tensor([0, 35, 1, 0, 3, 2, 0] if bd <= 16 else [0, 3, 1, 0, 2])
            m, n, batch = counts.numel() * bd, (40 if bd <= 16 else 8) * bd, 2
            br = torch.arange(m // bd)
            rp = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))).int() + base
            ci = torch.cat([torch.arange(count) for count in counts]).int() + base
            logical = torch.randn(ci.numel(), bd, bd, dtype=ctype)
            values = logical if direction == "row" else logical.transpose(-2, -1).contiguous()
            x = torch.randn(batch, m, k, dtype=dtype)
            y = torch.randn(batch, k, n, dtype=dtype)
            work = torch.complex128 if ctype == torch.complex64 else torch.float64
            row_ids = torch.repeat_interleave(br, counts)[:, None] * bd + torch.arange(bd)[None, :]
            col_ids = (ci - base)[:, None] * bd + torch.arange(bd)[None, :]
            dense = x.to(work) @ y.to(work)
            dot = dense[:, row_ids[:, :, None], col_ids[:, None, :]]
            expected = (.75 * dot + .5 * logical.to(work)).to(ctype)
            if direction == "col":
                expected = expected.transpose(-2, -1).contiguous()
            operands = tuple(t.npu() for t in (rp, ci, values, x, y))
            actual = bsr_values(
                BsrOperands(*operands),
                BsrOptions(alpha=.75, beta=.5, base=base, block_dim=bd, direction=direction, batch=batch)
            )
            with self.subTest(dtype=dtype, ctype=ctype, bd=bd, direction=direction):
                self.assert_accuracy(actual.cpu(), expected)
                self.assertTrue(torch.equal(operands[2].cpu(), values))

    def test_bsr_zero_beta_and_empty(self):
        for bd, dtype, direction, count in product((2, 128), (torch.float16, torch.complex64),
                                                  ("row", "col"), (0, 1)):
            rp = torch.tensor([1, 1 + count, 1 + count], dtype=torch.int32).npu()
            ci = torch.ones(count, dtype=torch.int32).npu()
            values = torch.full((count, bd, bd), float("nan"), dtype=dtype).npu()
            x = torch.ones(1, 2 * bd, 32, dtype=dtype).npu()
            y = torch.ones(1, 32, 2 * bd, dtype=dtype).npu()
            actual = bsr_values(
                BsrOperands(rp, ci, values, x, y),
                BsrOptions(alpha=1., beta=0., base=1, block_dim=bd, direction=direction, batch=1)
            )
            with self.subTest(bd=bd, dtype=dtype, direction=direction, count=count):
                self.assertEqual(tuple(actual.shape), (1, count, bd, bd))
                self.assertTrue(torch.equal(actual.cpu(), torch.full((1, count, bd, bd), 32., dtype=dtype)))
                self.assertEqual(actual.data_ptr(), values.data_ptr())

    def assert_operands_unchanged(self, operands, before):
        for index in (0, 1, 3, 4):
            self.assertTrue(torch.equal(operands[index].cpu(), before[index]))

    def test_bsr_cube_wave_boundary_and_complex_scalars(self):
        # More than 16 MiB of products: exercise a complete wave plus its tail.
        for dtype, blocks in ((torch.complex64, 257), (torch.float32, 1025)):
            bd, k, base = 64, 33, 1
            m, n = bd, blocks * bd
            rp = torch.tensor([base, blocks + base], dtype=torch.int32)
            ci = torch.arange(blocks, dtype=torch.int32) + base
            x = torch.randn(1, m, k, dtype=dtype)
            y = torch.randn(1, k, n, dtype=dtype)
            initial = torch.randn(blocks, bd, bd, dtype=dtype)
            alpha = .75 + .25j if dtype == torch.complex64 else .75
            beta = -.125 + .5j if dtype == torch.complex64 else -.125
            work = torch.complex128 if dtype == torch.complex64 else torch.float64
            dot = (x.to(work) @ y.to(work)).reshape(bd, blocks, bd).permute(1, 0, 2)
            for direction in ("row", "col"):
                values = initial if direction == "row" else initial.transpose(-2, -1).contiguous()
                expected = (alpha * dot + beta * initial.to(work)).to(dtype)
                if direction == "col":
                    expected = expected.transpose(-2, -1).contiguous()
                operands = tuple(t.npu() for t in (rp, ci, values, x, y))
                before = tuple(t.cpu() for t in operands)
                actual = bsr_values(
                    BsrOperands(*operands),
                    BsrOptions(alpha=alpha, beta=beta, base=base, block_dim=bd, direction=direction, batch=1)
                )
                with self.subTest(dtype=dtype, direction=direction):
                    self.assert_accuracy(actual[0].cpu(), expected)
                    self.assert_operands_unchanged(operands, before)

    def test_bsr_cube_direct_output(self):
        torch.manual_seed(219)
        for (bd, blocks), dtype in product(
            ((16, 7), (32, 7), (64, 1025), (128, 257)), (torch.float16, torch.bfloat16, torch.float32)
        ):
            k, base = 33, 1
            rp = torch.tensor([base, base, base + blocks - 1, base + blocks], dtype=torch.int32)
            ci = torch.arange(blocks, dtype=torch.int32).flip(0) + base
            rows = torch.cat((torch.ones(blocks - 1, dtype=torch.long), torch.tensor([2])))
            lane = torch.arange(bd)
            x = torch.randn(1, 3 * bd, k, dtype=dtype)
            y = torch.randn(1, k, blocks * bd, dtype=dtype)
            dot = x.double()[:, rows[:, None] * bd + lane, :] @ y.double().transpose(1, 2)[
                :, (ci.long() - base)[:, None] * bd + lane, :].transpose(-2, -1)
            operands = tuple(t.npu() for t in (rp, ci, x, y))
            # Revisit the low-precision output after a float-output cache entry.
            for ctype, (alpha, beta) in product(
                (dtype, torch.float32, dtype), ((1.0, 0.0), (0.75, 0.25), (1.0, 0.0))
            ):
                initial = torch.full((blocks, bd, bd), float("nan") if beta == 0 else .5,
                                     dtype=ctype).npu()
                actual = bsr_values(
                    BsrOperands(operands[0], operands[1], initial, operands[2], operands[3]),
                    BsrOptions(alpha=alpha, beta=beta, base=base, block_dim=bd, direction="row", batch=1)
                )
                expected = (alpha * dot + beta * .5).to(ctype)
                with self.subTest(bd=bd, dtype=dtype, ctype=ctype, alpha=alpha):
                    self.assert_accuracy(actual.cpu(), expected)
                    self.assertEqual(actual.data_ptr(), initial.data_ptr())


    def test_bsr_complex_vector_tails_and_scalars(self):
        torch.manual_seed(331)
        for bd, k, direction in product((4, 8, 16, 32), (1, 31, 64, 65, 128), ("row", "col")):
            rp = torch.tensor([1, 1, 3, 3, 4], dtype=torch.int32)
            ci = torch.tensor([3, 1, 2], dtype=torch.int32)
            x = torch.randn(1, 4 * bd, k, dtype=torch.complex64)
            y = torch.randn(1, k, 4 * bd, dtype=torch.complex64)
            row = torch.tensor([1, 1, 3])[:, None] * bd + torch.arange(bd)
            col = (ci.long() - 1)[:, None] * bd + torch.arange(bd)
            dense = x.to(torch.complex128) @ y.to(torch.complex128)
            dots = dense[:, row[:, :, None], col[:, None, :]]
            for alpha, beta in ((1., 0.), (.75 + .25j, -.25 + .5j)):
                initial = torch.randn(3, bd, bd, dtype=torch.complex64)
                expected = alpha * dots + beta * initial.to(torch.complex128)
                if beta == 0:
                    initial.fill_(complex(float("nan"), float("nan")))
                if direction == "col":
                    expected = expected.transpose(-2, -1).contiguous()
                    initial = initial.transpose(-2, -1).contiguous()
                operands = tuple(t.npu() for t in (rp, ci, initial, x, y))
                result = bsr_values(
                    BsrOperands(*operands),
                    BsrOptions(alpha=alpha, beta=beta, base=1, block_dim=bd, direction=direction, batch=1)
                )
                with self.subTest(bd=bd, k=k, direction=direction, alpha=alpha):
                    self.assert_accuracy(result.cpu(), expected.to(torch.complex64))

    def test_bsr_complex_direct_layout_and_fallback(self):
        torch.manual_seed(317)
        counts = torch.tensor([0, 16, 0, 16] + [8] * 12)
        for bd, k in ((64, 32), (64, 128), (64, 256), (128, 128),
                      (64, 31), (64, 33), (64, 257)):
            m = n = 16 * bd
            rp0 = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))).int()
            ci0 = torch.cat([torch.arange(count).flip(0) for count in counts]).int()
            ci0[20:24] = 0
            row = torch.repeat_interleave(torch.arange(16), counts)[:, None] * bd + torch.arange(bd)
            col = ci0[:, None] * bd + torch.arange(bd)
            x = torch.randn(1, m, k, dtype=torch.complex64)
            y = torch.randn(1, k, n, dtype=torch.complex64)
            dense = x.to(torch.complex128) @ y.to(torch.complex128)
            dot = dense[:, row[:, :, None], col[:, None, :]]
            for base, direction, alpha, beta in ((0, "col", 1., 0.), (1, "col", 1., 0.),
                    (1, "row", 1., 0.), (1, "col", .75 + .25j, -.25 + .5j)):
                initial = torch.randn(len(ci0), bd, bd, dtype=torch.complex64)
                expected = alpha * dot
                if beta:
                    expected += beta * initial.to(torch.complex128)
                else:
                    initial.fill_(complex(float("nan"), float("nan")))
                if direction == "col":
                    initial = initial.transpose(-2, -1).contiguous()
                    expected = expected.transpose(-2, -1).contiguous()
                operands = tuple(t.npu() for t in (rp0 + base, ci0 + base, initial, x, y))
                result = bsr_values(
                    BsrOperands(*operands),
                    BsrOptions(alpha=alpha, beta=beta, base=base, block_dim=bd, direction=direction, batch=1)
                )
                with self.subTest(bd=bd, k=k, base=base, direction=direction, alpha=alpha):
                    self.assert_accuracy(result.cpu(), expected.to(torch.complex64))

    def test_bsr_complex_direct_broadcast_and_refresh(self):
        bd, k, m, n, blocks = 64, 128, 1024, 1024, 128
        rp = torch.arange(0, blocks + 1, 8, dtype=torch.int32).npu()
        columns = torch.arange(8, dtype=torch.int32).repeat(16)
        ci = columns.npu()
        row = torch.arange(16).repeat_interleave(8)[:, None] * bd + torch.arange(bd)
        for xb, yb in ((1, 1), (1, 2), (2, 1), (2, 2)):
            x = torch.randn(xb, m, k, dtype=torch.complex64).npu()
            y = torch.randn(yb, k, n, dtype=torch.complex64).npu()
            values = torch.full((blocks, bd, bd), complex(float("nan"), 0), dtype=torch.complex64).npu()
            for shift in (0, 4):
                ci.copy_(columns + shift)
                x.add_(.25 + .125j)
                y.mul_(.5 - .25j)
                col = (columns + shift)[:, None] * bd + torch.arange(bd)
                dense = x.cpu().to(torch.complex128) @ y.cpu().to(torch.complex128)
                expected = dense[:, row[:, :, None], col[:, None, :]].transpose(-2, -1).contiguous()
                expected = expected.expand(2, -1, -1, -1).to(torch.complex64)
                actual = bsr_values(
                    BsrOperands(rp, ci, values, x, y),
                    BsrOptions(alpha=1., beta=0., base=0, block_dim=bd, direction="col", batch=2)
                )
                with self.subTest(xb=xb, yb=yb, shift=shift):
                    self.assert_accuracy(actual.cpu(), expected)

    def test_bsr_grouped_cube_irregular_rows(self):
        for dtype, bd, direction in product((torch.float16, torch.bfloat16, torch.float32,
                                             torch.complex64), (16, 32), ("row", "col")):
            # Groups cross empty rows, partial rows and nonconsecutive columns.
            counts = torch.tensor([0, 7, 5, 11] * 8 + [1])
            rp = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))).int() + 1
            ci = torch.cat([torch.arange(count) for count in counts]).int() + 1
            m, n, k = len(counts) * bd, 16 * bd, 33
            x = torch.randn(1, m, k, dtype=dtype)
            y = torch.randn(1, k, n, dtype=dtype)
            values = torch.randn(len(ci), bd, bd, dtype=dtype)
            row_ids = torch.repeat_interleave(torch.arange(len(counts)), counts)[:, None] * bd + torch.arange(bd)
            work = torch.complex128 if dtype == torch.complex64 else torch.float64
            alpha = .75 + .25j if dtype == torch.complex64 else .75
            beta = -.25 + .5j if dtype == torch.complex64 else -.25
            operands = tuple(t.npu() for t in (rp, ci, values, x, y))
            for gaps in (False, True, False):
                columns = ci + (ci >= 5).int() * (3 if gaps else 0)
                operands[1].copy_(columns)
                operands[2].copy_(values)
                col_ids = (columns.long() - 1)[:, None] * bd + torch.arange(bd)
                dot = x.to(work)[:, row_ids, :] @ y.to(work).transpose(1, 2)[:, col_ids, :].transpose(-2, -1)
                if direction == "col":
                    dot = dot.transpose(-2, -1)
                expected = (alpha * dot + beta * values).to(dtype)
                actual = bsr_values(
                    BsrOperands(*operands),
                    BsrOptions(alpha=alpha, beta=beta, base=1, block_dim=bd, direction=direction, batch=1)
                )
                with self.subTest(dtype=dtype, bd=bd, direction=direction, gaps=gaps):
                    self.assert_accuracy(actual.cpu(), expected)


    def test_public_csr_output_ownership_and_beta(self):
        for k, beta in product((16, 32, 100, 128, 512, 768, 2048), (0., .5, .25 + .125j)):
            m, n, per_row, base = 16, 64, 13, 1
            rp = (torch.arange(m + 1, dtype=torch.int32) * per_row + base).npu()
            ci = torch.stack([torch.randperm(n)[:per_row] for _ in range(m)]).flatten().int().add(base).npu()
            values = torch.randn(m * per_row, dtype=torch.complex64).npu()
            x = torch.randn(m, k, dtype=torch.complex64).npu()
            y = torch.randn(k, n, dtype=torch.complex64).npu()
            if beta == 0.:
                values.fill_(complex(float("nan"), float("nan")))
            original = values.cpu().clone()
            initial = (rp.cpu(), ci.cpu(), torch.zeros_like(original) if beta == 0. else original,
                       x.cpu().unsqueeze(0), y.cpu().unsqueeze(0))
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                sparse = torch.sparse_csr_tensor(rp, ci, values, size=(m, n), device=values.device)
                output = torch.sparse.sampled_addmm(sparse, x, y, alpha=.75 + .25j, beta=beta)
                actual = output.values().cpu().unsqueeze(0)
            case = {"base": base, "m": m, "alpha": .75 + .25j, "beta": beta, "dtype": "complex64"}
            with self.subTest(k=k, beta=beta):
                self.assertTrue(check_csr(case, initial, actual)["pass"])
                self.assertNotEqual(output.values().data_ptr(), values.data_ptr())
                if beta == 0.:
                    self.assertTrue(bool(values.cpu().isnan().all()))
                else:
                    self.assertTrue(torch.equal(values.cpu(), original))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
