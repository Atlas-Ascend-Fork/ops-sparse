# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""End-to-end coverage for the public DenseToSparse ATen registrations.

Run this file against an installed ``cann_ops_sparse`` wheel on an NPU.  The
first public conversion deliberately reaches the lazy builder, then every
comparison reads sparse components on the CPU.  This also avoids NPU
``to_dense`` for complex sparse tensors, whose availability is unrelated to
DenseToSparse conversion.
"""

from __future__ import annotations

from typing import Callable, Tuple

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
cann_ops_sparse = pytest.importorskip("cann_ops_sparse")

DEVICE = "npu:0"
DTYPES = (torch.int8, torch.float16, torch.bfloat16, torch.float32, torch.complex64)
FORMATS = ("csr", "csc", "coo")


def _has_npu() -> bool:
    return bool(getattr(torch, "npu", None) and torch.npu.is_available())


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU unavailable")


def _conversion(format_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if format_name == "csr":
        return torch.Tensor.to_sparse_csr
    if format_name == "csc":
        return torch.Tensor.to_sparse_csc
    return torch.Tensor.to_sparse


def _layout(format_name: str) -> torch.layout:
    return {
        "csr": torch.sparse_csr,
        "csc": torch.sparse_csc,
        "coo": torch.sparse_coo,
    }[format_name]


def _dense(dtype: torch.dtype, case: str) -> torch.Tensor:
    if case == "empty_rows":
        return torch.empty((0, 5), dtype=dtype)
    if case == "empty_columns":
        return torch.empty((3, 0), dtype=dtype)
    if case == "empty_matrix":
        return torch.empty((0, 0), dtype=dtype)
    if case == "zeros":
        return torch.zeros((3, 4), dtype=dtype)

    real = torch.tensor(
        [[0.0, 2.0, 0.0, -3.0], [4.0, 0.0, 5.0, 0.0], [0.0, -6.0, 0.0, 7.0]],
        dtype=torch.float32,
    )
    if dtype == torch.complex64:
        imaginary = torch.tensor(
            [[0.0, 1.0, 0.0, -2.0], [3.0, 0.0, -4.0, 0.0], [0.0, 5.0, 0.0, -6.0]],
            dtype=torch.float32,
        )
        return torch.complex(real, imaginary)
    return real.to(dtype)


def _rebuild_with_dtype(reference: torch.Tensor, format_name: str,
                        dtype: torch.dtype) -> torch.Tensor:
    values = reference.values().to(dtype)
    if format_name == "csr":
        return torch.sparse_csr_tensor(
            reference.crow_indices(), reference.col_indices(), values,
            size=reference.shape,
        )
    if format_name == "csc":
        return torch.sparse_csc_tensor(
            reference.ccol_indices(), reference.row_indices(), values,
            size=reference.shape,
        )
    return torch.sparse_coo_tensor(
        reference.indices(), values, size=reference.shape,
    ).coalesce()


def _cpu_reference(source: torch.Tensor, format_name: str) -> torch.Tensor:
    convert = _conversion(format_name)
    try:
        reference = convert(source)
    except RuntimeError:
        if source.is_complex():
            raise
        # Some Torch CPU versions do not implement compressed conversion for
        # every supported real value dtype. These values retain their
        # zero/nonzero pattern in fp32; restore the requested value dtype.
        reference = _rebuild_with_dtype(convert(source.to(torch.float32)), format_name,
                                        source.dtype)
    return reference.coalesce() if format_name == "coo" else reference


def _components(tensor: torch.Tensor, format_name: str) -> Tuple[torch.Tensor, ...]:
    if format_name == "csr":
        return (tensor.crow_indices().cpu(), tensor.col_indices().cpu(), tensor.values().cpu())
    if format_name == "csc":
        return (tensor.ccol_indices().cpu(), tensor.row_indices().cpu(), tensor.values().cpu())
    return (tensor.indices().cpu(), tensor.values().cpu())


def _assert_matches_cpu(actual: torch.Tensor, reference: torch.Tensor,
                        format_name: str) -> None:
    torch.npu.synchronize()
    assert actual.device.type == "npu"
    assert actual.layout == _layout(format_name)
    assert actual.shape == reference.shape
    actual_parts = _components(actual, format_name)
    reference_parts = _components(reference, format_name)
    assert len(actual_parts) == len(reference_parts)
    for actual_part, reference_part in zip(actual_parts, reference_parts):
        assert torch.equal(actual_part, reference_part)


def test_public_to_sparse_coo_triggers_lazy_jit_and_returns_canonical_indices() -> None:
    """The public ``to_sparse`` entry is the lazy builder's real API path."""
    source = _dense(torch.complex64, "nonzero")
    actual = source.to(DEVICE).to_sparse()
    reference = _cpu_reference(source, "coo")
    _assert_matches_cpu(actual, reference, "coo")
    indices, values = _components(actual, "coo")
    assert indices.shape == (2, values.numel())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("format_name", FORMATS)
@pytest.mark.parametrize("case", ("nonzero", "zeros", "empty_rows", "empty_columns", "empty_matrix"))
def test_public_dense_to_sparse_matches_cpu_components(
        dtype: torch.dtype, format_name: str, case: str) -> None:
    """CSR, CSC and COO preserve indices and values for each supported dtype."""
    source = _dense(dtype, case)
    actual = _conversion(format_name)(source.to(DEVICE))
    _assert_matches_cpu(actual, _cpu_reference(source, format_name), format_name)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("format_name", FORMATS)
def test_public_dense_to_sparse_accepts_column_major_input(
        dtype: torch.dtype, format_name: str) -> None:
    """A transposed view exercises the column-major descriptor path."""
    source = _dense(dtype, "nonzero").transpose(0, 1)
    assert source.stride(0) == 1
    actual = _conversion(format_name)(source.to(DEVICE))
    _assert_matches_cpu(actual, _cpu_reference(source, format_name), format_name)


@pytest.mark.parametrize("format_name", FORMATS)
def test_public_dense_to_sparse_respects_non_default_stream(format_name: str) -> None:
    """Conversion waits for a preceding producer on the caller's stream."""
    source = _dense(torch.float32, "nonzero")
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        produced = source.to(DEVICE) * 2.0
        actual = _conversion(format_name)(produced)
    stream.synchronize()
    _assert_matches_cpu(actual, _cpu_reference(source * 2.0, format_name), format_name)
