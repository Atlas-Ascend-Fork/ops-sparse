# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SpGEMM Torch Extension 注册的端到端测试。

在真实 NPU 上对已安装的 ``cann_ops_sparse`` wheel 运行::

    bash build.sh --torch_extension --ops=spgemm
    python3 -m pip install build_out/cann_ops_sparse-*.whl --force-reinstall --no-deps
    OPS_SPARSE_LIB_DIR=$PWD/build_out/lib64 python3 -m pytest test/spgemm/python/ -v

``OPS_SPARSE_LIB_DIR`` 为 JIT 链接步骤指定构建树，无需预先安装运行包。
"""

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import List, Tuple

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
cann_ops_sparse = pytest.importorskip("cann_ops_sparse")

DEVICE = "npu:0"
VALUE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.complex64)
# NPU complex 张量的 to_dense() 触发 aclnnIndexAdd dtype 缺陷，
# 与 SpGEMM 无关，因此 complex 用例改为比较 CSR 分量。
DENSE_COMPARABLE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass
class CsrData:
    crow: List[int]
    col: List[int]
    values: List[float]
    shape: Tuple[int, int]


def csr(data: CsrData, dtype: torch.dtype, device: str = DEVICE,
        index_dtype: torch.dtype = torch.int32) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.tensor(data.crow, dtype=index_dtype, device=device),
        torch.tensor(data.col, dtype=index_dtype, device=device),
        torch.tensor(data.values, dtype=dtype, device=device),
        size=data.shape,
        device=device,
    )


def cpu_parts(tensor: torch.Tensor) -> Tuple[List[int], List[int], torch.Tensor]:
    torch.npu.synchronize()
    return (
        tensor.crow_indices().cpu().tolist(),
        tensor.col_indices().cpu().tolist(),
        tensor.values().cpu(),
    )


def random_dense(rows: int, cols: int, dtype: torch.dtype, density: float = 0.3,
                 seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    mask = (torch.rand(rows, cols, generator=generator) < density).to(torch.float32)
    dense = mask * torch.randn(rows, cols, generator=generator)
    if dtype == torch.complex64:
        imaginary = mask * torch.randn(rows, cols, generator=generator)
        return (dense + 1j * imaginary).to(dtype)
    return dense.to(dtype)


def tolerance(dtype: torch.dtype) -> float:
    return {torch.float16: 5e-3, torch.bfloat16: 4e-2, torch.float32: 1e-5,
            torch.complex64: 1e-5}[dtype]


# ---------------------------------------------------------------------------
# 1. 注册验证
# ---------------------------------------------------------------------------


def test_dispatch_keys_are_registered():
    """CSR 和 COO 操作数能正确路由到 NPU 实现。"""
    csr_a = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    csr_out = torch.sparse.mm(csr_a, csr_a)
    torch.npu.synchronize()
    assert csr_out.device == csr_a.device
    assert csr_out.layout == torch.sparse_csr

    coo_a = csr_a.to_sparse(layout=torch.sparse_coo)
    coo_out = torch.sparse.mm(coo_a, coo_a)
    torch.npu.synchronize()
    assert coo_out.device == csr_a.device
    assert coo_out.layout == torch.sparse_coo


def test_shared_matrix_schemas_are_registered():
    from torch._C import _dispatch_has_kernel_for_dispatch_key as has_dispatch_kernel

    for key in ("SparseCsrPrivateUse1", "SparsePrivateUse1"):
        for schema in ("mm", "mm.out", "addmm", "addmm.out",
                       "_sparse_addmm", "_sparse_sparse_matmul"):
            # Dispatcher diagnostics have no public equivalent.
            assert has_dispatch_kernel("aten::" + schema, key)


@pytest.mark.parametrize("api", [torch.mm, torch.sparse.mm])
def test_public_mm_entries_share_spgemm(api):
    a = csr(CsrData([0, 1, 2], [0, 1], [2.0, 3.0], (2, 2)), torch.float32)
    result = api(a, a)
    _, _, values = cpu_parts(result)
    torch.testing.assert_close(values, torch.tensor([4.0, 9.0]))


@pytest.mark.parametrize("api", [torch.addmm, torch.sparse.addmm])
def test_public_addmm_entries_share_spgemm(api):
    a = csr(CsrData([0, 1, 2], [0, 1], [2.0, 3.0], (2, 2)), torch.float32)
    result = api(a, a, a, beta=2, alpha=3)
    _, _, values = cpu_parts(result)
    torch.testing.assert_close(values, torch.tensor([16.0, 33.0]))


@pytest.mark.parametrize("operation", ["mm", "addmm"])
def test_spgemm_out_is_explicitly_unsupported(operation):
    a = csr(CsrData([0, 1], [0], [2.0], (1, 1)), torch.float32)
    out = csr(CsrData([0, 0], [], [], (1, 1)), torch.float32)
    with pytest.raises(RuntimeError, match="SpGEMM does not yet provide an out="):
        if operation == "mm":
            torch.mm(a, a, out=out)
        else:
            torch.addmm(a, a, a, out=out)


def test_sparse_dense_reports_missing_spmm_without_loading_spgemm(monkeypatch):
    from cann_ops_sparse.ops.sparse.spgemm.spgemm import _spgemm_op_builder

    monkeypatch.setattr(
        _spgemm_op_builder,
        "load",
        lambda: pytest.fail("missing SpMM must not trigger the SpGEMM JIT builder"),
    )
    a = csr(CsrData([0, 1], [0], [2.0], (1, 1)), torch.float32)
    dense = torch.ones(1, 1, device=DEVICE)
    with pytest.raises(RuntimeError, match="spmm capability is not registered"):
        torch.sparse.addmm(dense, a, dense)


def test_import_does_not_trigger_jit(tmp_path):
    """导入包不应触发 JIT 编译（规范 3.1）。"""
    cache = tmp_path / "extensions"
    cache.mkdir()
    environment = dict(os.environ, TORCH_EXTENSIONS_DIR=str(cache))
    script = (
        "import torch, torch_npu, cann_ops_sparse\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], env=environment,
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    assert not any(cache.iterdir()), "importing the package produced build artifacts"


def test_facade_exposes_public_apis():
    """torch_npu.sparse 转发公共 API，而非私有入口。"""
    assert torch_npu.sparse.mm is torch.sparse.mm
    assert torch_npu.sparse.addmm is torch.sparse.addmm


# ---------------------------------------------------------------------------
# 2. 功能与精度
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_required_dtypes(dtype):
    """迁移自旧适配器测试：精确验证结构和数值。"""
    a = csr(CsrData([0, 2, 3], [0, 1, 1], [1, 2, 3], (2, 2)), dtype)
    output = torch.sparse.mm(a, a)
    crow, col, values = cpu_parts(output)
    assert output.layout == torch.sparse_csr
    assert crow == [0, 2, 3]
    assert col == [0, 1, 1]
    torch.testing.assert_close(values, torch.tensor([1, 8, 9], dtype=dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_matches_dense_reference(dtype):
    a = random_dense(16, 12, dtype, seed=1)
    b = random_dense(12, 10, dtype, seed=2)
    output = torch.sparse.mm(a.to_sparse_csr().to(DEVICE), b.to_sparse_csr().to(DEVICE))
    reference = (a @ b).to_sparse_csr()
    out_crow, out_col, out_values = cpu_parts(output)
    assert out_crow == reference.crow_indices().tolist()
    assert out_col == reference.col_indices().tolist()
    torch.testing.assert_close(out_values, reference.values(), rtol=tolerance(dtype),
                               atol=tolerance(dtype))


@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
def test_both_index_dtypes(index_dtype):
    """PyTorch 允许 int64 索引；封装层在 device 上窄化为 int32。"""
    a = csr(CsrData([0, 2, 3], [0, 1, 1], [1.0, 2.0, 3.0], (2, 2)), torch.float32,
            index_dtype=index_dtype)
    crow, col, values = cpu_parts(torch.sparse.mm(a, a))
    assert crow == [0, 2, 3] and col == [0, 1, 1]
    assert values.tolist() == [1.0, 8.0, 9.0]


@pytest.mark.parametrize("beta,alpha", [(0.0, 1.0), (1.0, 1.0), (2.0, 3.0), (-1.5, 0.5)])
def test_addmm_with_beta(beta, alpha):
    """beta != 0 时累加稀疏加数；旧适配器不支持此功能。"""
    a = random_dense(8, 8, torch.float32, seed=3)
    b = random_dense(8, 8, torch.float32, seed=4)
    c = random_dense(8, 8, torch.float32, seed=5)
    output = torch.sparse.addmm(c.to_sparse_csr().to(DEVICE), a.to_sparse_csr().to(DEVICE),
                                b.to_sparse_csr().to(DEVICE), beta=beta, alpha=alpha)
    reference = beta * c + alpha * (a @ b)
    torch.npu.synchronize()
    torch.testing.assert_close(output.to_dense().cpu(), reference, rtol=1e-4, atol=1e-4)


def test_addmm_does_not_alias_the_addend():
    """Copy 会重写 matC 的 row offsets，因此 self 不能成为输出。"""
    a = random_dense(8, 8, torch.float32, seed=6)
    b = random_dense(8, 8, torch.float32, seed=7)
    c = random_dense(8, 8, torch.float32, seed=8)
    addend = c.to_sparse_csr().to(DEVICE)
    before = addend.crow_indices().cpu().clone()
    output = torch.sparse.addmm(addend, a.to_sparse_csr().to(DEVICE),
                                b.to_sparse_csr().to(DEVICE), beta=1.0, alpha=1.0)
    torch.npu.synchronize()
    assert torch.equal(addend.crow_indices().cpu(), before)
    assert output.values().data_ptr() != addend.values().data_ptr()


@pytest.mark.parametrize("alpha", (1.0, 2.5, -1.0))
def test_alpha_scales_the_product(alpha):
    """空 alpha 会被库读为 0.0，必须确保正确传递。"""
    a = csr(CsrData([0, 2, 3], [0, 1, 1], [1.0, 2.0, 3.0], (2, 2)), torch.float32)
    empty = csr(CsrData([0, 0, 0], [], [], (2, 2)), torch.float32)
    output = torch.sparse.addmm(empty, a, a, beta=0.0, alpha=alpha)
    _, _, values = cpu_parts(output)
    torch.testing.assert_close(values, torch.tensor([1.0, 8.0, 9.0]) * alpha,
                               rtol=1e-6, atol=1e-6)


def test_coo_public_path():
    """迁移自旧测试：COO 上的 torch.sparse.mm 路由到 _sparse_sparse_matmul。"""
    a = csr(CsrData([0, 2, 3], [0, 1, 1], [1.0, 2.0, 3.0], (2, 2)), torch.float32)
    coo = a.to_sparse(layout=torch.sparse_coo)
    output = torch.sparse.mm(coo, coo)
    torch.npu.synchronize()
    assert output.layout == torch.sparse_coo
    assert output.is_coalesced()
    assert output.indices().cpu().tolist() == [[0, 0, 1], [0, 1, 1]]
    assert output.values().cpu().tolist() == [1.0, 8.0, 9.0]


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_coo_matches_csr_path(dtype):
    """两种布局对同一矩阵必须产生相同乘积。"""
    a = random_dense(10, 10, dtype, seed=13).to(DEVICE)
    # 通过 CSR 构建而非 dense.to_sparse()：aclnnNonzero 不支持
    # complex64，这是稠密化缺陷，与 SpGEMM 无关。
    as_csr = a.to_sparse_csr()
    as_coo = as_csr.to_sparse(layout=torch.sparse_coo)
    from_csr = torch.sparse.mm(as_csr, as_csr)
    from_coo = torch.sparse.mm(as_coo, as_coo)
    torch.npu.synchronize()
    reference = from_csr.to_sparse(layout=torch.sparse_coo).coalesce()
    assert from_coo.indices().cpu().tolist() == reference.indices().cpu().tolist()
    torch.testing.assert_close(from_coo.values().cpu(), reference.values().cpu(), rtol=0, atol=0)


def test_complex_scalar_on_complex_dtype():
    a = csr(CsrData([0, 2, 3], [0, 1, 1], [1, 2, 3], (2, 2)), torch.complex64)
    empty = csr(CsrData([0, 0, 0], [], [], (2, 2)), torch.complex64)
    output = torch.sparse.addmm(empty, a, a, beta=0.0, alpha=2j)
    _, _, values = cpu_parts(output)
    torch.testing.assert_close(values,
                               torch.tensor([1, 8, 9], dtype=torch.complex64) * 2j,
                               rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. 边界条件
# ---------------------------------------------------------------------------


def test_explicit_zero_is_retained():
    """迁移自旧测试：抵消为零的乘积仍占据一个 nnz 槽位。"""
    a = csr(CsrData([0, 2], [0, 1], [1.0, 1.0], (1, 2)), torch.float32,
            index_dtype=torch.int64)
    b = csr(CsrData([0, 1, 2], [0, 0], [1.0, -1.0], (2, 1)), torch.float32,
            index_dtype=torch.int64)
    crow, col, values = cpu_parts(torch.sparse.mm(a, b))
    assert crow == [0, 1] and col == [0]
    assert values.tolist() == [0.0]


def test_no_matching_products():
    """迁移自旧测试：不相交模式产生空但结构有效的结果。"""
    a = csr(CsrData([0, 1], [0], [1.0], (1, 2)), torch.float32)
    b = csr(CsrData([0, 0, 1], [0], [1.0], (2, 1)), torch.float32)
    crow, col, values = cpu_parts(torch.sparse.mm(a, b))
    assert crow == [0, 0] and col == [] and values.numel() == 0


def test_empty_rows_are_preserved():
    """迁移自旧测试：即使 nnz(C)=0，row offsets 长度仍为 M+1。"""
    a = csr(CsrData([0, 0, 1, 1], [1], [2.0], (3, 3)), torch.float32)
    b = csr(CsrData([0, 1, 1, 2], [2, 0], [3.0, 4.0], (3, 3)), torch.float32)
    crow, col, values = cpu_parts(torch.sparse.mm(a, b))
    assert crow == [0, 0, 0, 0] and col == [] and values.numel() == 0


def test_zero_nnz_operands():
    a = csr(CsrData([0, 0, 0], [], [], (2, 3)), torch.float32)
    b = csr(CsrData([0, 0, 0, 0], [], [], (3, 2)), torch.float32)
    output = torch.sparse.mm(a, b)
    crow, col, values = cpu_parts(output)
    assert output.shape == (2, 2)
    assert crow == [0, 0, 0] and col == [] and values.numel() == 0


def test_minimal_shape():
    a = csr(CsrData([0, 1], [0], [3.0], (1, 1)), torch.float32)
    crow, col, values = cpu_parts(torch.sparse.mm(a, a))
    assert crow == [0, 1] and col == [0] and values.tolist() == [9.0]


def test_zero_row_dimension():
    a = csr(CsrData([0], [], [], (0, 3)), torch.float32)
    b = csr(CsrData([0, 0, 0, 0], [], [], (3, 2)), torch.float32)
    output = torch.sparse.mm(a, b)
    torch.npu.synchronize()
    assert output.shape == (0, 2)
    assert output.values().numel() == 0


# ---------------------------------------------------------------------------
# 4. 非法输入
# ---------------------------------------------------------------------------


def test_mismatched_dtypes_are_rejected():
    a = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    b = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float16)
    with pytest.raises(RuntimeError, match="dtype"):
        torch.sparse.mm(a, b)


def test_mismatched_shapes_are_rejected():
    a = csr(CsrData([0, 1], [0], [1.0], (1, 2)), torch.float32)
    b = csr(CsrData([0, 1], [0], [1.0], (1, 2)), torch.float32)
    with pytest.raises(RuntimeError, match="cannot be multiplied"):
        torch.sparse.mm(a, b)


def test_strided_operand_is_rejected():
    """Python 侧注册不会穿透，因此需在此处检查布局。"""
    a = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    dense = torch.ones(1, 1, device=DEVICE)
    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.sparse.addmm(a, a, dense)


def test_unsupported_value_dtype_is_rejected():
    a = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float64)
    with pytest.raises(RuntimeError, match="dtype"):
        torch.sparse.mm(a, a)


def test_complex_scalar_on_real_dtype_is_rejected():
    a = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    with pytest.raises(RuntimeError, match="complex scalar"):
        torch.sparse.addmm(a, a, a, beta=0.0, alpha=1j)


def test_addend_shape_is_validated():
    a = csr(CsrData([0, 1, 1], [0], [1.0], (2, 2)), torch.float32)
    wrong = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    with pytest.raises(RuntimeError, match="self must have shape"):
        torch.sparse.addmm(wrong, a, a, beta=1.0, alpha=1.0)


@pytest.mark.parametrize("beta", (0.0, 1.0))
def test_addend_is_validated_for_every_beta(beta):
    """aten::addmm 对所有 beta（含零）都校验 self，封装层不能在 beta=0 路径上跳过检查。"""
    a = csr(CsrData([0, 2, 3], [0, 1, 1], [1.0, 2.0, 3.0], (2, 2)), torch.float32)
    wrong_dtype = csr(CsrData([0, 0, 0], [], [], (2, 2)), torch.float16)
    with pytest.raises(RuntimeError, match="self must share the dtype"):
        torch.sparse.addmm(wrong_dtype, a, a, beta=beta, alpha=1.0)

    wrong_shape = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    with pytest.raises(RuntimeError, match="self must have shape"):
        torch.sparse.addmm(wrong_shape, a, a, beta=beta, alpha=1.0)

    dense_addend = torch.zeros(2, 2, device=DEVICE)
    with pytest.raises(RuntimeError, match="self must use the sparse CSR layout"):
        torch.sparse.addmm(dense_addend, a, a, beta=beta, alpha=1.0)


def test_cpu_operands_do_not_reach_the_npu_kernel():
    """注册绑定在 PrivateUse1 上，不应捕获 CPU 工作。"""
    coo = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to_sparse()
    result = torch.sparse.mm(coo, coo)
    assert result.device.type == "cpu"

    # 无 MKL 时 PyTorch 不支持 CPU CSR × CSR；错误应来自 PyTorch 而非本封装层。
    csr_cpu = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to_sparse_csr()
    with pytest.raises(RuntimeError) as failure:
        torch.sparse.mm(csr_cpu, csr_cpu)
    assert "SpGEMM" not in str(failure.value)


# ---------------------------------------------------------------------------
# 5. 设备与异步
# ---------------------------------------------------------------------------


def test_repeated_calls_are_stable():
    a = random_dense(24, 24, torch.float32, seed=9)
    operand = a.to_sparse_csr().to(DEVICE)
    reference = None
    for _ in range(5):
        _, _, values = cpu_parts(torch.sparse.mm(operand, operand))
        if reference is None:
            reference = values
        else:
            torch.testing.assert_close(values, reference, rtol=0, atol=0)


@pytest.mark.skipif(torch.npu.device_count() < 2, reason="needs a second NPU")
def test_non_default_device():
    """DeviceGuard 必须在分配输出前生效。"""
    a = random_dense(12, 12, torch.float32, seed=10)
    operand = a.to_sparse_csr().to("npu:1")
    output = torch.sparse.mm(operand, operand)
    torch.npu.synchronize("npu:1")
    assert output.device == operand.device
    torch.testing.assert_close(output.to_dense().cpu(), a @ a, rtol=1e-5, atol=1e-5)


def test_non_default_stream():
    a = random_dense(12, 12, torch.float32, seed=11)
    operand = a.to_sparse_csr().to(DEVICE)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        output = torch.sparse.mm(operand, operand)
    stream.synchronize()
    torch.testing.assert_close(output.to_dense().cpu(), a @ a, rtol=1e-5, atol=1e-5)


def test_ordering_with_preceding_npu_work():
    """封装层读取调用方在同一 stream 上刚产生的输入。"""
    a = random_dense(16, 16, torch.float32, seed=12)
    scaled = (a.to(DEVICE) * 2.0).to_sparse_csr()
    output = torch.sparse.mm(scaled, scaled)
    torch.npu.synchronize()
    torch.testing.assert_close(output.to_dense().cpu(), (a * 2.0) @ (a * 2.0),
                               rtol=1e-4, atol=1e-4)


def test_concurrent_calls_from_threads():
    """per-handle 锁必须序列化描述符/workspace 状态，并发调用者不能互相破坏。"""
    a = random_dense(20, 20, torch.float32, seed=14)
    operand = a.to_sparse_csr().to(DEVICE)
    expected = None
    for _ in range(1):
        _, _, expected = cpu_parts(torch.sparse.mm(operand, operand))

    results: List[torch.Tensor] = []
    failures: List[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            for _ in range(6):
                _, _, values = cpu_parts(torch.sparse.mm(operand, operand))
                with lock:
                    results.append(values)
        # Report background failures to the assertion in the calling thread.
        except BaseException as error:
            with lock:
                failures.append(error)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures, f"concurrent calls raised: {failures[0]!r}"
    assert len(results) == 24
    for values in results:
        torch.testing.assert_close(values, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 6. 打包验证
# ---------------------------------------------------------------------------


def test_wheel_contents_are_complete():
    root = os.path.dirname(cann_ops_sparse.__file__)
    for relative in ("ops/sparse/spgemm/__init__.py", "ops/sparse/spgemm/spgemm.py",
                     "csrc/sparse/spgemm/spgemm.cpp", "common/aclsparse_common.h",
                     "common/cann_ops_sparse.h"):
        assert os.path.isfile(os.path.join(root, relative)), f"missing {relative}"


def test_builder_sources_resolve_inside_the_package():
    from cann_ops_sparse.ops.sparse.spgemm.spgemm import SpGemmOpBuilder

    builder = SpGemmOpBuilder()
    root = os.path.dirname(cann_ops_sparse.__file__)
    for source in builder.sources():
        assert os.path.isfile(os.path.join(root, source)), f"unresolved source {source}"


def test_extension_is_compiled_once_per_process():
    from cann_ops_sparse.ops.sparse.spgemm.spgemm import _spgemm_op_builder

    a = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    torch.sparse.mm(a, a)
    first = _spgemm_op_builder.load()
    torch.sparse.mm(a, a)
    assert _spgemm_op_builder.load() is first


# ---------------------------------------------------------------------------
# 7. 命名空间
# ---------------------------------------------------------------------------


def test_namespace_install_is_idempotent():
    """二次导入必须返回已有命名空间，不应抛异常。"""
    import cann_ops_sparse.torch_npu_sparse as facade

    before = torch_npu.sparse
    again = facade.install_namespace(torch_npu, cann_ops_sparse.ops.TORCH_NPU_SPARSE_FACADE_APIS)
    assert again is before
    assert torch_npu.sparse.mm is torch.sparse.mm


def test_framework_declares_the_matrix_apis():
    from cann_ops_sparse.ops.matmul.registration import TORCH_NPU_SPARSE_FACADE_APIS

    assert TORCH_NPU_SPARSE_FACADE_APIS == {"mm": torch.sparse.mm, "addmm": torch.sparse.addmm}
    for name in TORCH_NPU_SPARSE_FACADE_APIS:
        assert cann_ops_sparse.ops.TORCH_NPU_SPARSE_FACADE_APIS[name] is (
            TORCH_NPU_SPARSE_FACADE_APIS[name]
        )


def test_third_party_namespace_is_not_overwritten():
    import cann_ops_sparse.torch_npu_sparse as facade

    class Foreign:
        sparse = "owned by someone else"

    with pytest.raises(RuntimeError):
        facade.install_namespace(Foreign, {"mm": torch.sparse.mm})
    assert Foreign.sparse == "owned by someone else"


# ---------------------------------------------------------------------------
# 8. NPU first-order autograd
# ---------------------------------------------------------------------------


def _gradient_operand(data: CsrData, dtype: torch.dtype, layout: str):
    tensor = csr(data, dtype)
    if layout == "coo":
        tensor = tensor.to_sparse_coo()
    return tensor.detach().requires_grad_()


def _components(tensor):
    """Return COO storage without coalescing duplicate coordinates."""
    from torch._C import _jit_get_operation as get_aten_operation

    indices, _ = get_aten_operation("aten::_indices")
    values, _ = get_aten_operation("aten::_values")
    return indices(tensor), values(tensor)


def _gradient_values(tensor):
    if tensor.layout == torch.sparse_csr:
        return tensor.values().cpu()
    # AccumulateGrad may store an uncoalesced COO gradient. Canonicalize the
    # copied components on CPU so validation does not depend on CANN's separate
    # CoalesceSparse complex-dtype support.
    indices, values = _components(tensor)
    return torch.sparse_coo_tensor(
        indices.cpu(), values.cpu(), tensor.shape
    ).coalesce().values()


def _gradient_dense_cpu(tensor):
    if tensor.layout == torch.strided:
        return tensor.cpu()
    if tensor.layout == torch.sparse_csr:
        return torch.sparse_csr_tensor(tensor.crow_indices().cpu(), tensor.col_indices().cpu(),
                                       tensor.values().cpu(), size=tensor.shape).to_dense()
    indices, values = _components(tensor)
    return torch.sparse_coo_tensor(indices.cpu(), values.cpu(), tensor.shape).to_dense()


@dataclass
class GradientCase:
    """Sparse inputs and their dense native-reference calculation."""

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    reference_a: torch.Tensor
    reference_b: torch.Tensor
    reference_c: torch.Tensor
    result: torch.Tensor
    expected: torch.Tensor
    dtype: torch.dtype
    reference_dtype: torch.dtype


def _make_gradient_case(dtype, layout, api_name):
    complex_values = dtype == torch.complex64
    av = [1.5 + 0.4j, 2.0 - 0.3j, -0.5 + 0.2j] if complex_values else [1.5, 2.0, -0.5]
    bv = [2.0 - 0.5j, 3.0 + 0.2j, 4.0 + 0.7j] if complex_values else [2.0, 3.0, 4.0]
    cv = [2.0 + 0.3j, 1.0 - 0.2j] if complex_values else [2.0, 1.0]
    a = _gradient_operand(CsrData([0, 2, 3], [0, 2, 1], av, (2, 3)), dtype, layout)
    b = _gradient_operand(CsrData([0, 1, 2, 3], [1, 0, 1], bv, (3, 2)), dtype, layout)
    c = _gradient_operand(CsrData([0, 1, 2], [0, 1], cv, (2, 2)), dtype, "csr")
    reference_dtype = torch.complex128 if complex_values else torch.float64
    reference_a = torch.tensor(
        [[av[0], 0, av[1]], [0, av[2], 0]], dtype=reference_dtype, requires_grad=True
    )
    reference_b = torch.tensor(
        [[0, bv[0]], [bv[1], 0], [0, bv[2]]], dtype=reference_dtype, requires_grad=True
    )
    reference_c = torch.diag(torch.tensor(cv, dtype=reference_dtype)).requires_grad_()
    alpha, beta = (1.2 - 0.7j, 0.5 + 0.2j) if complex_values else (1.5, -0.5)
    api = getattr(torch.sparse if api_name.startswith("sparse.") else torch, api_name.split(".")[-1])
    accumulated = "addmm" in api_name
    result = api(c, a, b, alpha=alpha, beta=beta) if accumulated else api(a, b)
    if api_name == "sparse.addmm":
        reference_a = reference_a.detach().to_sparse_coo().requires_grad_()
        # The public sparse API selects the native _sparse_addmm scalar rules.
        expected = torch.sparse.addmm(
            reference_c, reference_a, reference_b, alpha=alpha, beta=beta
        )
    elif accumulated:
        expected = beta * reference_c + alpha * (reference_a @ reference_b)
    else:
        expected = reference_a @ reference_b
    return GradientCase(
        a, b, c, reference_a, reference_b, reference_c, result, expected, dtype, reference_dtype
    )


def _upstream_gradient(case, layout, dense_grad):
    values = (
        [[0.2 + 0.4j, 1.3 - 0.2j], [-0.7 + 0.5j, 2.0 + 0.3j]]
        if case.dtype == torch.complex64 else [[0.2, 1.3], [-0.7, 2.0]]
    )
    grad_cpu = torch.tensor(values, dtype=case.dtype)
    if dense_grad:
        return grad_cpu.to(DEVICE), grad_cpu
    grad = csr(
        CsrData([0, 2, 4], [0, 1, 0, 1], grad_cpu.reshape(-1).tolist(), (2, 2)), case.dtype
    )
    return (grad.to_sparse_coo() if layout == "coo" else grad), grad_cpu


def _assert_first_order_gradients(case, api_name, dense_grad):
    a_gradient = case.reference_a.grad
    a_dense = a_gradient.to_dense() if a_gradient.layout != torch.strided else a_gradient
    a_expected = a_dense[[0, 0, 1], [0, 2, 1]].to(case.dtype)
    b_expected = case.reference_b.grad[[0, 1, 2], [1, 0, 1]].to(case.dtype)
    # aten::mm and aten::addmm return the complete matrix derivatives for a
    # dense upstream gradient.  Only _sparse_addmm (the torch.sparse.addmm
    # entry point) keeps matrix gradients on the input sparse pattern.
    full_matrix_gradient = api_name in ("mm", "addmm")
    if full_matrix_gradient:
        torch.testing.assert_close(_gradient_dense_cpu(case.a.grad), a_dense.to(case.dtype),
                                   rtol=tolerance(case.dtype), atol=tolerance(case.dtype))
        torch.testing.assert_close(_gradient_dense_cpu(case.b.grad), case.reference_b.grad.to(case.dtype),
                                   rtol=tolerance(case.dtype), atol=tolerance(case.dtype))
        # A complete matrix derivative follows the upstream layout.  In
        # particular, CSR upstream gradients must not be converted to COO
        # while materializing coordinates outside the input CSR pattern.
        expected_layout = (
            torch.strided if dense_grad else
            (torch.sparse_csr if case.a.layout == torch.sparse_csr else torch.sparse_coo)
        )
        assert case.a.grad.layout == case.b.grad.layout == expected_layout
    else:
        torch.testing.assert_close(_gradient_values(case.a.grad), a_expected,
                                   rtol=tolerance(case.dtype), atol=tolerance(case.dtype))
        torch.testing.assert_close(_gradient_values(case.b.grad), b_expected,
                                   rtol=tolerance(case.dtype), atol=tolerance(case.dtype))
        assert case.a.grad.layout == case.a.layout and case.b.grad.layout == case.b.layout
    if "addmm" in api_name:
        torch.testing.assert_close(_gradient_dense_cpu(case.c.grad),
                                   case.reference_c.grad.to(case.dtype),
                                   rtol=tolerance(case.dtype), atol=tolerance(case.dtype))
        expected_layout = torch.strided if dense_grad else case.a.layout
        assert case.c.grad.layout == expected_layout


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize("layout,api_name", [
    ("csr", "mm"), ("csr", "sparse.mm"), ("csr", "addmm"), ("csr", "sparse.addmm"),
    ("coo", "mm"), ("coo", "sparse.mm"),
])
@pytest.mark.parametrize("dense_grad", [False, True])
def test_first_order_gradients_match_dense_reference(dtype, layout, api_name, dense_grad):
    case = _make_gradient_case(dtype, layout, api_name)
    grad, grad_cpu = _upstream_gradient(case, layout, dense_grad)
    case.result.backward(grad)
    case.expected.backward(grad_cpu.to(case.reference_dtype))
    torch.npu.synchronize()
    _assert_first_order_gradients(case, api_name, dense_grad)


def _coo_cpu_reference(tensor):
    """CSR x CSR CPU kernels require MKL; COO keeps the same aten::mm schema."""
    return tensor.detach().cpu().to_sparse_coo().requires_grad_()


def test_csr_torch_mm_dense_gradient_is_not_projected_to_csr_pattern():
    a = _gradient_operand(CsrData([0, 1, 2], [0, 1], [2., 3.], (2, 2)), torch.float32, "csr")
    b = _gradient_operand(
        CsrData([0, 2, 4], [0, 1, 0, 1], [1., 4., 2., 5.], (2, 2)), torch.float32, "csr"
    )
    grad_cpu = torch.ones((2, 2), dtype=torch.float32)
    reference_a, reference_b = _coo_cpu_reference(a), _coo_cpu_reference(b)
    expected = torch.autograd.grad(torch.mm(reference_a, reference_b),
                                   (reference_a, reference_b), grad_cpu)

    actual = torch.autograd.grad(torch.mm(a, b), (a, b), grad_cpu.to(DEVICE))
    for gradient, reference in zip(actual, expected):
        assert gradient.layout == torch.strided
        torch.testing.assert_close(gradient.cpu(), reference)
    torch.testing.assert_close(actual[0].cpu(), torch.tensor([[5., 7.], [5., 7.]]))


def test_csr_torch_addmm_dense_gradient_is_not_projected_to_csr_pattern():
    self_tensor = torch.zeros((2, 2), device=DEVICE).to_sparse_csr()
    a = _gradient_operand(CsrData([0, 1, 2], [0, 1], [2., 3.], (2, 2)), torch.float32, "csr")
    b = _gradient_operand(
        CsrData([0, 2, 4], [0, 1, 0, 1], [1., 4., 2., 5.], (2, 2)), torch.float32, "csr"
    )
    grad_cpu = torch.ones((2, 2), dtype=torch.float32)
    # This CPU build lacks CSR x CSR forward. The addmm
    # derivative is nevertheless the native dense matrix derivative, so use
    # dense leaves to retain the full coordinates and explicit upstream zeroes.
    reference_a = a.detach().cpu().to_dense().requires_grad_()
    reference_b = b.detach().cpu().to_dense().requires_grad_()
    expected = torch.autograd.grad(torch.addmm(torch.zeros((2, 2)), reference_a, reference_b, alpha=3),
                                   (reference_a, reference_b), grad_cpu)

    actual = torch.autograd.grad(torch.addmm(self_tensor, a, b, alpha=3), (a, b), grad_cpu.to(DEVICE))
    for gradient, reference in zip(actual, expected):
        assert gradient.layout == torch.strided
        torch.testing.assert_close(gradient.cpu(), reference)
    torch.testing.assert_close(actual[0].cpu(), torch.tensor([[15., 21.], [15., 21.]]))


def _assert_full_gradient_layout(gradient, reference, upstream_layout):
    expected_layout = {
        "csr": torch.sparse_csr,
        "coo": torch.sparse_coo,
        "strided": torch.strided,
    }[upstream_layout]
    dtype = gradient.dtype
    assert gradient.layout == expected_layout
    torch.testing.assert_close(_gradient_dense_cpu(gradient), reference.to(dtype),
                               rtol=tolerance(dtype), atol=tolerance(dtype))
    if upstream_layout == "csr":
        expected_csr = reference.to(dtype).to_sparse_csr()
        crow, col, values = cpu_parts(gradient)
        assert crow == expected_csr.crow_indices().tolist()
        assert col == expected_csr.col_indices().tolist()
        torch.testing.assert_close(values, expected_csr.values(),
                                   rtol=tolerance(dtype), atol=tolerance(dtype))


@pytest.mark.parametrize("api_name", ["mm", "addmm"])
@pytest.mark.parametrize("upstream_layout", ["csr", "coo", "strided"])
@pytest.mark.parametrize("dtype,alpha", [
    (torch.float32, 3.0),
    (torch.complex64, 1.2 - 0.7j),
])
def test_csr_matrix_gradients_follow_upstream_layout(api_name, upstream_layout, dtype, alpha):
    """Full CSR-input derivatives retain a CSR upstream layout and coordinates.

    The local CPU build has no CSR x CSR kernel.  Native COO mm and dense
    addmm provide the numerical reference; its full derivative is compressed
    to the layout required by the upstream gradient.
    """
    complex_values = dtype == torch.complex64
    a_values = [2.0 + 0.5j, 3.0 - 0.25j] if complex_values else [2.0, 3.0]
    b_values = ([1.0 - 0.5j, 4.0 + 0.25j, 2.0 + 0.75j, 5.0 - 0.5j]
                if complex_values else [1., 4., 2., 5.])
    grad_values = ([[1.0 + 0.25j, 2.0 - 0.5j], [3.0 + 0.75j, 4.0 - 0.25j]]
                   if complex_values else [[1., 2.], [3., 4.]])
    a = _gradient_operand(CsrData([0, 1, 2], [0, 1], a_values, (2, 2)), dtype, "csr")
    b = _gradient_operand(CsrData([0, 2, 4], [0, 1, 0, 1], b_values, (2, 2)), dtype, "csr")
    grad_cpu = torch.tensor(grad_values, dtype=dtype)
    grad = grad_cpu.to(DEVICE)
    if upstream_layout == "csr":
        grad = grad.to_sparse_csr()
    elif upstream_layout == "coo":
        grad = grad.to_sparse_coo()

    if api_name == "mm":
        # COO is the native sparse CPU reference for this formula; the local
        # build does not provide a CSR x CSR CPU kernel.
        reference_a, reference_b = _coo_cpu_reference(a), _coo_cpu_reference(b)
        reference_result = torch.mm(reference_a, reference_b)
        actual_result = torch.mm(a, b)
        reference_grad = grad_cpu
    else:
        reference_dtype = torch.complex128 if complex_values else torch.float64
        reference_a = a.detach().cpu().to_dense().to(reference_dtype).requires_grad_()
        reference_b = b.detach().cpu().to_dense().to(reference_dtype).requires_grad_()
        reference_result = torch.addmm(torch.zeros((2, 2), dtype=reference_dtype),
                                       reference_a, reference_b, alpha=alpha)
        actual_result = torch.addmm(torch.zeros((2, 2), device=DEVICE, dtype=dtype).to_sparse_csr(),
                                    a, b, alpha=alpha)
        reference_grad = grad_cpu.to(reference_dtype)
    expected = torch.autograd.grad(reference_result, (reference_a, reference_b),
                                   reference_grad)
    actual = torch.autograd.grad(actual_result, (a, b), grad)

    for gradient, reference in zip(actual, expected):
        _assert_full_gradient_layout(gradient, reference, upstream_layout)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_csr_torch_mm_dense_zero_gradient_preserves_nonfinite_values(value):
    a = _gradient_operand(CsrData([0, 1, 2], [0, 1], [2., 3.], (2, 2)), torch.float32, "csr")
    b = _gradient_operand(CsrData([0, 1, 2], [0, 1], [value, 5.], (2, 2)), torch.float32, "csr")
    grad_cpu = torch.zeros((2, 2), dtype=torch.float32)
    reference_a, reference_b = _coo_cpu_reference(a), _coo_cpu_reference(b)
    expected = torch.autograd.grad(torch.mm(reference_a, reference_b),
                                   (reference_a, reference_b), grad_cpu)

    actual = torch.autograd.grad(torch.mm(a, b), (a, b), grad_cpu.to(DEVICE))
    for gradient, reference in zip(actual, expected):
        assert gradient.layout == torch.strided
        torch.testing.assert_close(gradient.cpu(), reference, equal_nan=True)
    assert torch.isnan(actual[0].cpu()[:, 0]).all()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_csr_torch_addmm_dense_zero_gradient_preserves_nonfinite_values(value):
    a = _gradient_operand(CsrData([0, 1, 2], [0, 1], [2., 3.], (2, 2)), torch.float32, "csr")
    b = _gradient_operand(CsrData([0, 1, 2], [0, 1], [value, 5.], (2, 2)), torch.float32, "csr")
    grad_cpu = torch.zeros((2, 2), dtype=torch.float32)
    reference_a = a.detach().cpu().to_dense().requires_grad_()
    reference_b = b.detach().cpu().to_dense().requires_grad_()
    expected = torch.autograd.grad(torch.addmm(torch.zeros((2, 2)), reference_a, reference_b),
                                   (reference_a, reference_b), grad_cpu)

    actual = torch.autograd.grad(torch.addmm(torch.zeros((2, 2), device=DEVICE).to_sparse_csr(), a, b),
                                (a, b), grad_cpu.to(DEVICE))
    for gradient, reference in zip(actual, expected):
        assert gradient.layout == torch.strided
        torch.testing.assert_close(gradient.cpu(), reference, equal_nan=True)
    assert torch.isnan(actual[0].cpu()[:, 0]).all()


@pytest.mark.parametrize("upstream_layout", ["strided", "csr"])
def test_sparse_addmm_self_gradient_keeps_upstream_coordinates(upstream_layout):
    self_tensor = _gradient_operand(
        CsrData([0, 1, 2], [0, 1], [1., 1.], (2, 2)), torch.float32, "csr"
    )
    a = csr(CsrData([0, 1, 2], [0, 1], [1., 1.], (2, 2)), torch.float32)
    b = csr(CsrData([0, 1, 2], [0, 1], [1., 1.], (2, 2)), torch.float32)
    grad_cpu = torch.tensor([[1., 7.], [11., 1.]])
    grad = grad_cpu.to(DEVICE)
    if upstream_layout == "csr":
        grad = csr(CsrData([0, 2, 4], [0, 1, 0, 1], grad_cpu.reshape(-1).tolist(), (2, 2)),
                   torch.float32)

    gradient, = torch.autograd.grad(torch.sparse.addmm(self_tensor, a, b, beta=2), self_tensor, grad)
    assert gradient.layout == (torch.strided if upstream_layout == "strided" else torch.sparse_csr)
    torch.testing.assert_close(_gradient_dense_cpu(gradient), 2 * grad_cpu)


@pytest.mark.parametrize("layout", ["csr", "coo"])
@pytest.mark.parametrize("dtype", DENSE_COMPARABLE_DTYPES)
def test_gradient_reaches_real_sparse_constructor_values(layout, dtype):
    values = torch.tensor([2., 3.], device=DEVICE, dtype=dtype, requires_grad=True)
    if layout == "csr":
        a = torch.sparse_csr_tensor(torch.tensor([0, 1, 2], device=DEVICE),
                                    torch.tensor([0, 1], device=DEVICE), values, size=(2, 2))
    else:
        a = torch.sparse_coo_tensor(torch.tensor([[0, 1], [0, 1]], device=DEVICE), values,
                                    size=(2, 2), is_coalesced=True)
    b = csr(CsrData([0, 1, 2], [0, 1], [4., 5.], (2, 2)), dtype)
    if layout == "coo":
        b = b.to_sparse_coo()
    result = torch.sparse.mm(a, b)
    result.values().sum().backward()
    torch.testing.assert_close(values.grad.cpu(), torch.tensor([4., 5.], dtype=dtype))


def test_existing_dense_npu_autograd_is_preserved():
    a = torch.tensor([[1., 2.], [3., 4.]], device=DEVICE, requires_grad=True)
    b = torch.tensor([[2., 0.], [1., 3.]], device=DEVICE, requires_grad=True)
    c = torch.ones(2, device=DEVICE, requires_grad=True)
    result = torch.addmm(c, a, b, alpha=3, beta=2) + torch.mm(a, b)
    result.sum().backward()
    torch.testing.assert_close(a.grad.cpu(), torch.tensor([[8., 16.], [8., 16.]]))
    torch.testing.assert_close(b.grad.cpu(), torch.tensor([[16., 16.], [24., 24.]]))
    torch.testing.assert_close(c.grad.cpu(), torch.tensor([4., 4.]))


def test_complex_conjugate_grad_output():
    a = _gradient_operand(
        CsrData([0, 1, 2], [0, 1], [2 + 1j, 3 - 2j], (2, 2)), torch.complex64, 'csr'
    )
    b = _gradient_operand(
        CsrData([0, 1, 2], [0, 1], [4 - 2j, 5 + 1j], (2, 2)), torch.complex64, 'csr'
    )
    grad = torch.tensor([[1 + 2j, 0], [0, 3 - 1j]], device=DEVICE).conj()
    torch.sparse.mm(a, b).backward(grad)
    expected_a = torch.tensor([1 - 2j, 3 + 1j]) * torch.tensor([4 + 2j, 5 - 1j])
    expected_b = torch.tensor([2 - 1j, 3 + 2j]) * torch.tensor([1 - 2j, 3 + 1j])
    torch.testing.assert_close(_gradient_values(a.grad), expected_a)
    torch.testing.assert_close(_gradient_values(b.grad), expected_b)


def test_band_single_segment_uses_a_values():
    a = csr(CsrData([0, 1, 2, 3], [0, 1, 0], [1.5, -0.5, 2.], (3, 2)), torch.float32)
    b = csr(CsrData([0, 2, 4], [0, 1, 0, 1], [0.2, 1.3, -0.7, 2.], (2, 2)), torch.float32)
    _, _, values = cpu_parts(torch.sparse.mm(a, b))
    torch.testing.assert_close(values, torch.tensor([0.3, 1.95, 0.35, -1., 0.4, 2.6]))


@pytest.mark.parametrize('dtype', VALUE_DTYPES)
def test_duplicate_coo_first_order_values(dtype):
    complex_values = dtype == torch.complex64
    values = torch.tensor([1 + 0.5j, 2 - 0.5j, 3 + 1j] if complex_values else [1., 2., 3.],
                           device=DEVICE, dtype=dtype, requires_grad=not complex_values)
    a = torch.sparse_coo_tensor(torch.tensor([[0, 0, 1], [0, 0, 1]], device=DEVICE), values,
                                size=(2, 2))
    if complex_values:
        a = a.detach().requires_grad_()
    b = csr(CsrData([0, 1, 2], [0, 1], [4., 5.], (2, 2)), dtype).to_sparse_coo()
    result = torch.sparse.mm(a, b)
    grad = csr(CsrData([0, 1, 2], [0, 1], [1., 1.], (2, 2)), dtype).to_sparse_coo()
    result.backward(grad)
    if complex_values:
        torch.testing.assert_close(_gradient_values(a.grad), torch.tensor([4., 5.], dtype=dtype))
    else:
        torch.testing.assert_close(values.grad.cpu(), torch.tensor([4., 4., 5.], dtype=dtype))


def test_complex_band_da8_non_arithmetic_heads():
    columns = [0, 1, 2, 3, 5, 6, 7, 8]
    values = [complex(i + 1, 0.1 * (i + 1)) for i in range(8)]
    a = csr(CsrData([0, 8], columns, values, (1, 9)), torch.complex64)
    b_columns = [col for row in range(9) for col in (row, row + 10)]
    b_values = [value for _ in range(9) for value in (1 + 0.2j, 2 - 0.1j)]
    b = csr(CsrData(list(range(0, 19, 2)), b_columns, b_values, (9, 19)), torch.complex64)
    _, col, result = cpu_parts(torch.sparse.mm(a, b))
    assert col == columns + [column + 10 for column in columns]
    expected = torch.tensor([v * (1 + 0.2j) for v in values] + [v * (2 - 0.1j) for v in values])
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('sparse_dims', [1, 2, 3])
def test_npu_sparse_mask_nd_mixed_dtype(sparse_dims):
    source = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    mask = torch.ones_like(source, dtype=torch.float16).to_sparse(sparse_dims)
    expected = source.sparse_mask(mask)
    npu_source, npu_mask = source.to(DEVICE), mask.to(DEVICE)
    result = npu_source.sparse_mask(npu_mask)
    assert result.dtype == torch.float32
    result_indices, result_values = _components(result)
    expected_indices, expected_values = _components(expected)
    torch.testing.assert_close(result_indices.cpu(), expected_indices)
    torch.testing.assert_close(result_values.cpu(), expected_values)
    sparse_source = source.to_sparse(sparse_dims).to(DEVICE)
    result = sparse_source.sparse_mask(npu_mask)
    _, result_values = _components(result)
    _, expected_values = _components(expected)
    torch.testing.assert_close(result_values.cpu(), expected_values)


@pytest.mark.parametrize('dtype', VALUE_DTYPES)
@pytest.mark.parametrize('layout', ['csr', 'coo'])
def test_same_sparse_operand_accumulates_both_gradients(dtype, layout):
    a = _gradient_operand(CsrData([0, 1, 2], [0, 1], [2., 3.], (2, 2)), dtype, layout)
    grad = csr(CsrData([0, 1, 2], [0, 1], [1., 1.], (2, 2)), dtype)
    if layout == 'coo':
        grad = grad.to_sparse_coo()
    torch.sparse.mm(a, a).backward(grad)
    torch.testing.assert_close(_gradient_values(a.grad), torch.tensor([4., 6.], dtype=dtype))


@pytest.mark.parametrize('dtype', VALUE_DTYPES)
def test_csr_gradients_accumulate_across_backward_calls(dtype):
    a = _gradient_operand(CsrData([0, 1, 2], [0, 1], [2., 3.], (2, 2)), dtype, 'csr')
    b = csr(CsrData([0, 1, 2], [0, 1], [4., 5.], (2, 2)), dtype)
    grad = csr(CsrData([0, 1, 2], [0, 1], [1., 1.], (2, 2)), dtype)
    for _ in range(2):
        torch.sparse.mm(a, b).backward(grad)
    torch.testing.assert_close(_gradient_values(a.grad), torch.tensor([8., 10.], dtype=dtype))


@pytest.mark.parametrize("row,col", [([1, 1, 2], [0, 1]),
                                     ([0, 3, 2], [0, 1]),
                                     ([0, 1, 2], [-1, 1]),
                                     ([0, 2, 2], [1, 0])])
def test_stream_ordered_validation_rejects_invalid_indices(row, col):
    bad = csr(CsrData(row, col, [1.0, 2.0], (2, 2)), torch.float32,
              index_dtype=torch.int64)
    good = csr(CsrData([0, 1, 2], [0, 1], [1.0, 1.0], (2, 2)), torch.float32)
    with pytest.raises(RuntimeError, match="aclsparseSpGEMMWorkEstimation"):
        torch.sparse.mm(bad, good)


@pytest.mark.parametrize("beta", [0, 1])
def test_stream_ordered_validation_checks_addend(beta):
    good = csr(CsrData([0, 1, 2], [0, 1], [1.0, 1.0], (2, 2)), torch.float32)
    bad = csr(CsrData([0, 1, 2], [0, 2], [1.0, 1.0], (2, 2)), torch.float32,
              index_dtype=torch.int64)
    if beta == 0:
        result = torch.sparse.addmm(bad, good, good, beta=0)
        torch.testing.assert_close(result.values().cpu(), torch.tensor([1.0, 1.0]))
    else:
        with pytest.raises(RuntimeError, match="aclsparseSpGEMMWorkEstimation"):
            torch.sparse.addmm(bad, good, good, beta=beta)


def test_index_conversion_is_ordered_on_non_default_stream():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(30):
            a = csr(CsrData([0, 1, 2], [0, 1], [2.0, 3.0], (2, 2)), torch.float32,
                    index_dtype=torch.int64)
            # Produce values asynchronously as well as narrowing indices.
            a.values().mul_(2)
            result = torch.sparse.mm(a, a)
            torch.testing.assert_close(result.values().cpu(), torch.tensor([16.0, 36.0]))
    stream.synchronize()


@pytest.mark.parametrize("api_name", ["mm", "sparse.mm"])
def test_coo_backward_structure_matches_cpu_torch(api_name):
    cpu_a = torch.sparse_coo_tensor([[0, 0, 1], [0, 1, 1]], [2., 0., 3.], (2, 2)).coalesce().requires_grad_()
    cpu_b = torch.eye(2).to_sparse_coo().requires_grad_()
    cpu_grad = torch.eye(2).to_sparse_coo()
    api = torch.mm if api_name == "mm" else torch.sparse.mm
    expected = torch.autograd.grad(api(cpu_a, cpu_b), (cpu_a, cpu_b), cpu_grad)
    a = cpu_a.detach().to(DEVICE).requires_grad_()
    b = cpu_b.detach().to(DEVICE).requires_grad_()
    actual = torch.autograd.grad(api(a, b), (a, b), cpu_grad.to(DEVICE))
    for result, reference in zip(actual, expected):
        assert result.layout == reference.layout
        result_indices, result_values = _components(result)
        reference_indices, reference_values = _components(reference)
        torch.testing.assert_close(result_indices.cpu(), reference_indices)
        torch.testing.assert_close(result_values.cpu(), reference_values)


def test_zero_sparse_addmm_scales_preserve_nan_gradients():
    a = _gradient_operand(
        CsrData([0, 1, 2], [0, 1], [1., 1.], (2, 2)), torch.float32, "csr"
    )
    b = a.detach().clone().requires_grad_()
    c = a.detach().clone().requires_grad_()
    result = torch.sparse.addmm(c, a, b, alpha=0, beta=0)
    gradients = torch.autograd.grad(result, (c, a, b), torch.full((2, 2), float("nan"), device=DEVICE))
    assert torch.isnan(_gradient_dense_cpu(gradients[0])).all()
    for gradient in gradients[1:]:
        assert torch.isnan(gradient.values()).all()
    assert gradients[0].layout == torch.strided


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_dense_mm_zero_grad_preserves_nonfinite_values(value, dtype):
    cpu = torch.sparse_coo_tensor([[0], [0]], torch.tensor([value], dtype=dtype), (2, 2)).coalesce()
    a_cpu, b_cpu = cpu.detach().requires_grad_(), cpu.detach().requires_grad_()
    grad_cpu = torch.zeros((2, 2), dtype=dtype)
    expected = torch.autograd.grad(torch.mm(a_cpu, b_cpu), (a_cpu, b_cpu), grad_cpu)
    a, b = cpu.to(DEVICE).detach().requires_grad_(), cpu.to(DEVICE).detach().requires_grad_()
    actual = torch.autograd.grad(torch.mm(a, b), (a, b), grad_cpu.to(DEVICE))
    for result, reference in zip(actual, expected):
        assert result.layout == torch.strided
        torch.testing.assert_close(result.cpu(), reference, equal_nan=True)


@pytest.mark.parametrize('dtype', VALUE_DTYPES)
def test_five_way_fused_merge_overlaps_and_skips_empty_b_row(dtype):
    values = [0.5, -1.0, 1.5, -2.0, 2.5] * 2 + [99.0, 7.0]
    if dtype == torch.complex64:
        values = [complex(x, x / 8) for x in values]
    a_data = CsrData([0, 5, 11, 12], list(range(5)) + list(range(6)) + [5], values, (3, 6))
    b_values = [x for i in range(5) for x in (1.0 + i / 8, 2.0 - i / 8)]
    if dtype == torch.complex64:
        b_values = [complex(x, x / 16) for x in b_values]
    b_data = CsrData([0, 2, 4, 6, 8, 10, 10], [0, 1] * 5, b_values, (6, 3))
    a, b = csr(a_data, dtype), csr(b_data, dtype)
    result = torch.sparse.mm(a, b)
    reference_dtype = torch.complex128 if dtype == torch.complex64 else torch.float64
    ac = csr(a_data, dtype, device='cpu').to_dense().to(reference_dtype)
    bc = csr(b_data, dtype, device='cpu').to_dense().to(reference_dtype)
    torch.testing.assert_close(_gradient_dense_cpu(result), (ac @ bc).to(dtype),
                               rtol=tolerance(dtype), atol=tolerance(dtype))


def test_complex_non_arithmetic_heads_with_contiguous_segments():
    heads = [0, 1, 2, 3, 5, 6, 7, 8]
    values = [complex(i + 1, (i + 1) / 10) for i in range(8)]
    a = csr(CsrData([0, 8], list(range(8)), values, (1, 8)), torch.complex64)
    b_columns = [column for head in heads for column in (head, head + 10)]
    b_values = [value for _ in heads for value in (1 + 0.2j, 2 - 0.1j)]
    b = csr(CsrData(list(range(0, 17, 2)), b_columns, b_values, (8, 19)), torch.complex64)
    _, columns, result = cpu_parts(torch.sparse.mm(a, b))
    assert columns == heads + [head + 10 for head in heads]
    expected = torch.tensor([v * (1 + 0.2j) for v in values] + [v * (2 - 0.1j) for v in values])
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('dtype', VALUE_DTYPES)
def test_band_or_bound_false_positive_keeps_valid_eight_way_row(dtype):
    # B's band prefix is 9; OR(1..8) == 15 exceeds it although every A index is valid.
    heads = list(range(9)) + [10]
    values = [i / 8 for i in range(1, 9)]
    if dtype == torch.complex64:
        values = [complex(x, x / 8) for x in values]
    a_data = CsrData([0, 8], list(range(1, 9)), values, (1, 10))
    b_data = CsrData(list(range(0, 21, 2)),
                     [column for head in heads for column in (head, head + 12)],
                     [1.0, 2.0] * 10, (10, 23))
    result = torch.sparse.mm(csr(a_data, dtype), csr(b_data, dtype))
    reference_dtype = torch.complex128 if dtype == torch.complex64 else torch.float64
    ac = csr(a_data, dtype, device='cpu').to_dense().to(reference_dtype)
    bc = csr(b_data, dtype, device='cpu').to_dense().to(reference_dtype)
    torch.testing.assert_close(_gradient_dense_cpu(result), (ac @ bc).to(dtype),
                               rtol=tolerance(dtype), atol=tolerance(dtype))


@pytest.mark.parametrize('dtype', VALUE_DTYPES)
def test_five_way_band_or_bound_false_positive(dtype):
    # Five entries require General value loading even with a valid B band.
    # All indices are below prefix=9 despite their OR being 15.
    heads = list(range(9)) + [10]
    values = [0.5, -1.0, 1.5, -2.0, 2.5]
    if dtype == torch.complex64:
        values = [complex(x, x / 8) for x in values]
    a_data = CsrData([0, 5], [1, 2, 3, 4, 8], values, (1, 10))
    b_data = CsrData(list(range(0, 21, 2)),
                     [column for head in heads for column in (head, head + 12)],
                     [1.0, 2.0] * 10, (10, 23))
    result = torch.sparse.mm(csr(a_data, dtype), csr(b_data, dtype))
    reference_dtype = torch.complex128 if dtype == torch.complex64 else torch.float64
    ac = csr(a_data, dtype, device='cpu').to_dense().to(reference_dtype)
    bc = csr(b_data, dtype, device='cpu').to_dense().to(reference_dtype)
    torch.testing.assert_close(_gradient_dense_cpu(result), (ac @ bc).to(dtype),
                               rtol=tolerance(dtype), atol=tolerance(dtype))
