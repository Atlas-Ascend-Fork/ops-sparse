# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SpMV Torch Extension 注册的端到端测试。

在真实 NPU 上对已安装的 ``cann_ops_sparse`` wheel 运行::

    bash build.sh --torch_extension --ops=spmv
    python3 -m pip install build_out/cann_ops_sparse-*.whl --force-reinstall --no-deps
    OPS_SPARSE_LIB_DIR=$PWD/build_out/lib64 python3 -m pytest test/spmv/python/ -v

``OPS_SPARSE_LIB_DIR`` 为 JIT 链接步骤指定构建树，无需预先安装运行包。

公开入口用 ``torch.mv`` 覆盖 CSR×dense 向量的 NPU 分发（aten::mv，语义固定
alpha=1/beta=0）；完整 alpha/beta/trans/indexBase 参数面与 workspace 契约通过
pybind 模块的测试专用导出函数 ``spmv_npu`` / ``spmv_npu_workspace`` 直调覆盖，
二者不注册任何 torch schema。golden 在 CPU 上以 float64/complex128 稠密计算。
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Tuple, Union

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
cann_ops_sparse = pytest.importorskip("cann_ops_sparse")

DEVICE = "npu:0"

SPMV_DTYPES = (
    torch.int8,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int32,
    torch.complex64,
)

# 验收口径：out_dtype -> (rtol, atol, max_abs_error_limit, required_matched_ratio)
TOLERANCES = {
    torch.float16: (2**-9, 2**-9, 1e-1, 0.99),
    torch.bfloat16: (2**-6, 2**-6, 1e0, 0.99),
    torch.float32: (2**-10, 2**-16, 1e-2, 0.99),
    torch.complex64: (2**-10, 2**-16, 1e-2, 0.99),  # 实/虚部分别按 FP32
    torch.int32: (0.0, 0.0, 0.0, 1.0),  # exact match
}

# alpha/beta 标量可接受 Python int/float/bool/complex
Scalar = Union[bool, int, float, complex]


@dataclass
class DenseSpec:
    """随机稠密矩阵生成参数；long_tail_row=True 时首行构造长尾（行长 > 4096 tile）。"""

    rows: int
    cols: int
    density: float
    val_dtype: torch.dtype
    seed: int
    long_tail_row: bool = False


@dataclass
class SpmvCase:
    """一次完整参数面 SpMV 调用的全部输入（CPU 侧 dense/x/y 与调用参数）。

    alpha/beta 语义与 aclsparseSpMV 一致：``y = alpha * op(A) * x + beta * y``；
    trans 接受 bool（True 等价 "T"）或 "N"/"T"/"H"；base 为 CSR indexBase（0/1）。
    """

    dense: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    alpha: Scalar = 1
    beta: Scalar = 0
    trans: Union[bool, str] = "N"
    base: int = 0
    device: str = DEVICE


def spmv_ops():
    """返回 SpMV pybind 模块；首次调用触发 JIT 编译。"""
    from cann_ops_sparse.ops.sparse.spmv.spmv import _spmv_op_builder

    return _spmv_op_builder.load()


def out_dtype(val_dtype: torch.dtype) -> torch.dtype:
    """INT8 输入输出 INT32，其余与输入一致。"""
    return torch.int32 if val_dtype == torch.int8 else val_dtype


def gen_dense(spec: DenseSpec) -> torch.Tensor:
    """CPU 上生成随机稀疏矩阵的稠密形式（含空行），可选首行长尾。"""
    g = torch.Generator().manual_seed(spec.seed)
    mask = torch.rand(spec.rows, spec.cols, generator=g) < spec.density
    if spec.long_tail_row and spec.rows > 1 and spec.cols > 4200:
        mask[0] = torch.rand(spec.cols, generator=g) < 0.9  # 行长 > 4096 tile
        mask[1:] = torch.rand(spec.rows - 1, spec.cols, generator=g) < 0.01
    if spec.val_dtype in (torch.int8, torch.int32):
        dense = torch.randint(-5, 6, (spec.rows, spec.cols), generator=g,
                              dtype=torch.int64).to(spec.val_dtype)
    elif spec.val_dtype == torch.complex64:
        dense = (torch.randn(spec.rows, spec.cols, generator=g) +
                 1j * torch.randn(spec.rows, spec.cols, generator=g)).to(spec.val_dtype)
    else:
        dense = (torch.rand(spec.rows, spec.cols, generator=g) * 10 - 5).to(spec.val_dtype)
    return torch.where(mask, dense, torch.zeros((), dtype=dense.dtype))


def gen_vec(size: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if dtype in (torch.int8, torch.int32):
        return torch.randint(-5, 6, (size,), generator=g, dtype=torch.int64).to(dtype)
    if dtype == torch.complex64:
        return (torch.randn(size, generator=g) +
                1j * torch.randn(size, generator=g)).to(dtype)
    return (torch.rand(size, generator=g) * 10 - 5).to(dtype)


def golden_mv(case: SpmvCase) -> torch.Tensor:
    """float64 / complex128 稠密 golden。trans="H" 时按 A^H 计算。"""
    complex_mode = (case.dense.dtype.is_complex or case.x.dtype.is_complex
                    or case.y.dtype.is_complex
                    or isinstance(case.alpha, complex) or isinstance(case.beta, complex))
    wide = torch.complex128 if complex_mode else torch.float64
    mat = case.dense.to(wide)
    if case.trans == "H":
        mat = mat.conj().T
    elif case.trans in (True, "T"):
        mat = mat.T
    return case.alpha * (mat @ case.x.to(wide)) + case.beta * case.y.to(wide)


def check(actual: torch.Tensor, golden_wide: torch.Tensor, label: str) -> None:
    """验收口径：|actual-golden| <= atol + rtol*|golden|，匹配率与最大绝对误差双达标。"""
    torch.npu.synchronize()
    actual = actual.cpu()
    rtol, atol, max_abs, req_ratio = TOLERANCES[actual.dtype]
    if actual.dtype == torch.int32:
        golden_i32 = golden_wide.round().to(torch.int32)
        assert torch.equal(actual, golden_i32), (
            f"{label}: int32 exact match failed, "
            f"diff at {torch.nonzero(actual != golden_i32).flatten()[:5].tolist()}")
        return
    wide = torch.complex128 if actual.dtype.is_complex else torch.float64
    act = actual.to(wide)
    ref = golden_wide.to(wide)
    if actual.dtype.is_complex:
        # 实/虚部分别判定
        act = torch.view_as_real(act)
        ref = torch.view_as_real(ref)
    err = (act - ref).abs()
    tol = atol + rtol * ref.abs()
    matched = (err <= tol).double().mean().item()
    max_err = err.max().item() if err.numel() else 0.0
    assert matched >= req_ratio, f"{label}: matched_ratio={matched} < {req_ratio}"
    assert max_err <= max_abs, f"{label}: max_abs_error={max_err} > {max_abs}"


def csr_parts(dense_cpu: torch.Tensor, base: int,
              device: str = DEVICE) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 CPU 稠密矩阵转为 NPU 上的 int32 CSR 三分量（可选 indexBase 1）。"""
    csr = dense_cpu.to_sparse_csr()
    crow = csr.crow_indices().to(torch.int32) + base
    col = csr.col_indices().to(torch.int32) + base
    return crow.to(device), col.to(device), csr.values().to(device)


def run_spmv_npu(case: SpmvCase) -> torch.Tensor:
    """经 pybind 测试入口 spmv_npu 运行完整参数面的 SpMV。"""
    crow, col, values = csr_parts(case.dense, case.base, case.device)
    # trans 接受 bool 或 "N"/"T"/"H"
    trans_s = {False: "N", True: "T"}.get(case.trans, case.trans)
    return spmv_ops().spmv_npu(crow, col, values, case.x.to(case.device),
                               case.y.to(case.device), case.alpha, case.beta,
                               trans_s, case.base)


# ---------------------------------------------------------------------------
# 1. 注册验证
# ---------------------------------------------------------------------------

def test_dispatch_key_is_registered():
    """torch.mv(sparse_csr, dense) 路由到 aten::mv 的 NPU 实现。"""
    dense = gen_dense(DenseSpec(16, 24, 0.3, torch.float32, seed=1))
    x = gen_vec(24, torch.float32, seed=2)
    out = torch.mv(dense.to_sparse_csr().to(DEVICE), x.to(DEVICE))
    torch.npu.synchronize()
    assert out.device.type == torch.device(DEVICE).type
    assert out.layout == torch.strided
    check(out, golden_mv(SpmvCase(dense=dense, x=x, y=torch.zeros(16))), "dispatch")


def test_import_does_not_trigger_jit(tmp_path):
    """导入包不应触发 JIT 编译（规范 3.1）。"""
    cache = tmp_path / "extensions"
    cache.mkdir()
    environment = dict(os.environ, TORCH_EXTENSIONS_DIR=str(cache))
    script = (
        "import torch, torch_npu, cann_ops_sparse\n"
        "from cann_ops_sparse.ops.sparse.spmv.spmv import _spmv_op_builder as b\n"
        "print('LOADED:', list(type(b)._loaded_ops.keys()))\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], env=environment,
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    assert "LOADED: []" in completed.stdout
    assert not any(cache.iterdir()), "importing the package produced build artifacts"


# ---------------------------------------------------------------------------
# 2. 功能与精度（torch.mv 公开入口，固定 alpha=1/beta=0）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", SPMV_DTYPES)
def test_mv_dispatch_all_dtypes(dtype):
    dense = gen_dense(DenseSpec(64, 96, 0.2, dtype, seed=7))
    x = gen_vec(96, dtype, seed=8)
    y0 = torch.zeros(64, dtype=out_dtype(dtype))
    out = torch.mv(dense.to_sparse_csr().to(DEVICE), x.to(DEVICE))
    assert out.dtype == out_dtype(dtype), f"{dtype}: unexpected out dtype {out.dtype}"
    check(out, golden_mv(SpmvCase(dense=dense, x=x, y=y0)), f"mv-{dtype}")


@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
def test_both_index_dtypes(index_dtype):
    """PyTorch 允许 int64 索引；适配层在 device 上窄化为 int32。"""
    dense = gen_dense(DenseSpec(32, 48, 0.3, torch.float32, seed=36))
    csr = dense.to_sparse_csr()
    csr_indexed = torch.sparse_csr_tensor(
        csr.crow_indices().to(index_dtype), csr.col_indices().to(index_dtype),
        csr.values(), size=csr.shape)
    x = gen_vec(48, torch.float32, seed=37)
    out = torch.mv(csr_indexed.to(DEVICE), x.to(DEVICE))
    check(out, dense.double() @ x.double(), f"index-{index_dtype}")


# ---------------------------------------------------------------------------
# 2b. 零 nnz / 零列早退（PR #236 评审回归：empty 分配的 y 早退前必须清零）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", SPMV_DTYPES)
def test_mv_zero_nnz_returns_exact_zeros(dtype):
    """零非零元 CSR 经 torch.mv 必须逐元素精确返回 0（不接受 allclose 近似）。

    y 由 at::empty 分配，修复前 nnz==0 早退直接返回未初始化内存；重复调用
    以覆盖分配器复用未清零内存的偶发情形。
    """
    rows, cols = 3, 7
    mat = torch.sparse_csr_tensor(torch.zeros(rows + 1, dtype=torch.int64),
                                  torch.zeros(0, dtype=torch.int64),
                                  torch.zeros(0, dtype=dtype), size=(rows, cols))
    x = gen_vec(cols, dtype, seed=91)
    expect = torch.zeros(rows, dtype=out_dtype(dtype))
    for _ in range(8):
        out = torch.mv(mat.to(DEVICE), x.to(DEVICE))
        torch.npu.synchronize()
        assert torch.equal(out.cpu(), expect), (
            f"zero-nnz mv({dtype}) returned non-zero: {out.cpu().tolist()}")


@pytest.mark.parametrize("dtype", (torch.float32, torch.complex64))
def test_mv_zero_cols_returns_exact_zeros(dtype):
    """零列矩阵（n=0）与零 nnz 同路径，输出同样必须逐元素精确为 0。"""
    rows = 3
    mat = torch.sparse_csr_tensor(torch.zeros(rows + 1, dtype=torch.int64),
                                  torch.zeros(0, dtype=torch.int64),
                                  torch.zeros(0, dtype=dtype), size=(rows, 0))
    out = torch.mv(mat.to(DEVICE), torch.zeros(0, dtype=dtype).to(DEVICE))
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), torch.zeros(rows, dtype=dtype)), (
        f"zero-col mv({dtype}) returned non-zero: {out.cpu().tolist()}")


@pytest.mark.parametrize("trans", (False, True))
@pytest.mark.parametrize("beta", (0.0, 0.5))
def test_spmv_npu_zero_nnz_alpha_beta(trans, beta):
    """pybind 入口零 nnz：op(A)*x 贡献恒为 0，输出应精确等于 beta*y（含 alpha!=1）。"""
    rows, cols = 3, 5
    dense = torch.zeros(rows, cols, dtype=torch.float32)
    x = gen_vec(rows if trans else cols, torch.float32, seed=92)
    y = gen_vec(cols if trans else rows, torch.float32, seed=93)
    case = SpmvCase(dense=dense, x=x, y=y, alpha=1.5, beta=beta, trans=trans)
    out = run_spmv_npu(case)
    torch.npu.synchronize()
    golden = beta * y
    assert torch.equal(out.cpu(), golden), (
        f"zero-nnz spmv_npu(trans={trans}, beta={beta}) mismatch: "
        f"{out.cpu().tolist()} != {golden.tolist()}")


# ---------------------------------------------------------------------------
# 3. 完整参数面（pybind 测试入口 spmv_npu / spmv_npu_workspace）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", SPMV_DTYPES)
@pytest.mark.parametrize("trans", (False, True))
@pytest.mark.parametrize("base", (0, 1))
def test_dtype_combos(dtype, trans, base):
    """全部 dtype 组合 x N/T x base 0/1。"""
    odt = out_dtype(dtype)
    dense = gen_dense(DenseSpec(48, 72, 0.3, dtype, seed=11))
    x = gen_vec(48 if trans else 72, dtype, seed=12)
    y = gen_vec(72 if trans else 48, odt, seed=13)
    case = SpmvCase(dense=dense, x=x, y=y, trans=trans, base=base)
    out = run_spmv_npu(case)
    check(out, golden_mv(case),
          f"combo-{dtype}-{'T' if trans else 'N'}-base{base}")


@pytest.mark.parametrize("trans", (False, True))
def test_real_input_complex_output(trans):
    """FP32 -> complex64：实数输入复数输出。"""
    dense = gen_dense(DenseSpec(48, 72, 0.3, torch.float32, seed=14))
    x = gen_vec(48 if trans else 72, torch.float32, seed=15)
    y = gen_vec(72 if trans else 48, torch.complex64, seed=16)
    case = SpmvCase(dense=dense, x=x, y=y, trans=trans)
    out = run_spmv_npu(case)
    check(out, golden_mv(case), f"combo-f32-c64-{'T' if trans else 'N'}")


@pytest.mark.parametrize("dtype", (torch.complex64, torch.float32))
def test_hermitian_trans(dtype):
    """H（共轭转置）：complex64 走共轭路径，实数等价于 T。"""
    odt = out_dtype(dtype)
    dense = gen_dense(DenseSpec(48, 72, 0.3, dtype, seed=17))
    x = gen_vec(48, dtype, seed=18)
    y = gen_vec(72, odt, seed=19)
    case = SpmvCase(dense=dense, x=x, y=y, trans="H")
    out = run_spmv_npu(case)
    check(out, golden_mv(case), f"combo-{dtype}-H")


@pytest.mark.parametrize("dtype,alpha,beta", [
    (torch.float32, 2.0, 0.0), (torch.float32, 1.0, 1.0),
    (torch.float32, -1.5, 0.5), (torch.float32, 0.0, 3.0),
    (torch.int8, 2, 0), (torch.int8, -1, 1), (torch.int8, 0, 2),
    (torch.complex64, complex(1.5, -0.5), complex(0.2, 0.3)),
    (torch.complex64, complex(0.0, 1.0), 0),
])
@pytest.mark.parametrize("trans", (False, True))
def test_alpha_beta(dtype, alpha, beta, trans):
    dense = gen_dense(DenseSpec(40, 56, 0.3, dtype, seed=21))
    x = gen_vec(40 if trans else 56, dtype, seed=22)
    y = gen_vec(56 if trans else 40, out_dtype(dtype), seed=23)
    case = SpmvCase(dense=dense, x=x, y=y, alpha=alpha, beta=beta, trans=trans)
    out = run_spmv_npu(case)
    check(out, golden_mv(case),
          f"ab-{dtype}-{alpha}-{beta}-{'T' if trans else 'N'}")


def test_complex_scalars_on_real_input_complex_output():
    """FP32 -> complex64 复数 alpha/beta。"""
    dense = gen_dense(DenseSpec(40, 56, 0.3, torch.float32, seed=24))
    x = gen_vec(56, torch.float32, seed=25)
    y = gen_vec(40, torch.complex64, seed=26)
    case = SpmvCase(dense=dense, x=x, y=y,
                    alpha=complex(1.5, -0.5), beta=complex(0.2, 0.3))
    out = run_spmv_npu(case)
    check(out, golden_mv(case), "f32-c64-complex-ab")


@pytest.mark.parametrize("trans,expect_workspace", [
    ("N", False),  # 非转置路径不需要 workspace
    ("T", True),   # 转置 + FP32 计算需要 double-single 累加 workspace
])
def test_workspace_contract(trans, expect_workspace):
    dense = gen_dense(DenseSpec(32, 40, 0.3, torch.float32, seed=41))
    x = gen_vec(32 if trans == "T" else 40, torch.float32, seed=42)
    y = gen_vec(40 if trans == "T" else 32, torch.float32, seed=43)
    crow, col, values = csr_parts(dense, 0)
    workspace = spmv_ops().spmv_npu_workspace(
        crow, col, values, x.to(DEVICE), y.to(DEVICE), 1, 0, trans, 0)
    assert isinstance(workspace, int)
    assert (workspace > 0) == expect_workspace


# ---------------------------------------------------------------------------
# 4. 边界条件
# ---------------------------------------------------------------------------

def test_zero_nnz():
    """nnz=0 全空矩阵。"""
    dense = torch.zeros(16, 32, dtype=torch.float32)
    x = gen_vec(32, torch.float32, seed=31)
    out = torch.mv(dense.to_sparse_csr().to(DEVICE), x.to(DEVICE))
    check(out, torch.zeros(16, dtype=torch.float64), "nnz0")


def test_single_nnz():
    dense = torch.zeros(8, 8, dtype=torch.float32)
    dense[3, 5] = 2.5
    x = gen_vec(8, torch.float32, seed=32)
    out = torch.mv(dense.to_sparse_csr().to(DEVICE), x.to(DEVICE))
    check(out, dense.double() @ x.double(), "nnz1")


def test_extreme_value_twoprod_split_overflow():
    """TwoProd 拆分乘法溢出回归（PR 评审意见）。

    Dekker 拆分需计算 4097*a，|a| > FLT_MAX/4097 ≈ 8.3e34 时溢出为 Inf，
    误差项变 NaN（修复前这些用例输出 NaN）；修复后该项回退为普通乘积精度。
    覆盖：nnz=1 精确还原、1e35×1e-35、短行/长行混合、转置路径。
    """
    big, small = 2.0 ** 116, 2.0 ** -116

    # nnz=1 短行（Dot2ShortRows 补偿路径）：数学结果与普通 FP32 乘积均为精确的 1
    dense = torch.zeros(4, 8, dtype=torch.float32)
    dense[1, 3] = big
    x = torch.zeros(8, dtype=torch.float32)
    x[3] = small
    out = torch.mv(dense.to_sparse_csr().to(DEVICE), x.to(DEVICE))
    assert torch.isfinite(out).all(), "extreme nnz=1: 结果出现 Inf/NaN"
    assert out[1].item() == 1.0, f"extreme nnz=1: 期望精确 1.0，实际 {out[1].item()}"

    # 1e35 × 1e-35：同样越过拆分溢出边界（8.3e34）
    dense2 = torch.zeros(4, 8, dtype=torch.float32)
    dense2[2, 5] = 1e35
    x2 = torch.zeros(8, dtype=torch.float32)
    x2[5] = 1e-35
    out2 = torch.mv(dense2.to_sparse_csr().to(DEVICE), x2.to(DEVICE))
    assert torch.isfinite(out2).all(), "extreme 1e35: 结果出现 Inf/NaN"
    check(out2, dense2.double() @ x2.double(), "extreme-1e35")

    # 短行混合：极端项与普通项共存（补偿项与非补偿项混合累加）
    dense3 = torch.zeros(2, 8, dtype=torch.float32)
    dense3[0, 0] = big
    dense3[0, 1] = 3.0
    x3 = torch.zeros(8, dtype=torch.float32)
    x3[0] = small
    x3[1] = 2.0
    case3 = SpmvCase(dense=dense3, x=x3, y=torch.zeros(2), trans="N")
    out3 = run_spmv_npu(case3)
    assert torch.isfinite(out3).all(), "extreme mixed short row: 结果出现 Inf/NaN"
    check(out3, golden_mv(case3), "extreme-mixed-short")

    # 长行（行长 ≥ 128 走向量路径）中嵌入极端项
    dense4 = torch.zeros(2, 300, dtype=torch.float32)
    dense4[0, 0] = big
    dense4[0, 1:] = 1.0
    x4 = torch.zeros(300, dtype=torch.float32)
    x4[0] = small
    x4[1:] = 1.0
    case4 = SpmvCase(dense=dense4, x=x4, y=torch.zeros(2), trans="N")
    out4 = run_spmv_npu(case4)
    assert torch.isfinite(out4).all(), "extreme mixed long row: 结果出现 Inf/NaN"
    check(out4, golden_mv(case4), "extreme-mixed-long")

    # 转置路径（T kernel 的列累加同样使用 TwoProd）
    x5 = torch.zeros(4, dtype=torch.float32)
    x5[1] = small
    case5 = SpmvCase(dense=dense, x=x5, y=torch.zeros(8), trans="T")
    out5 = run_spmv_npu(case5)
    assert torch.isfinite(out5).all(), "extreme transpose: 结果出现 Inf/NaN"
    check(out5, golden_mv(case5), "extreme-trans")


def test_long_tail_row():
    """长尾行（行长 > 4096 tile），N 与 T 路径。"""
    dense = gen_dense(DenseSpec(8, 6000, 0.01, torch.float32, seed=33, long_tail_row=True))
    assert dense.to_sparse_csr().crow_indices()[1] > 4096
    x = gen_vec(6000, torch.float32, seed=34)
    out = torch.mv(dense.to_sparse_csr().to(DEVICE), x.to(DEVICE))
    check(out, dense.double() @ x.double(), "long-tail-row")
    x_t = gen_vec(8, torch.float32, seed=35)
    case = SpmvCase(dense=dense, x=x_t, y=torch.zeros(6000), trans=True)
    out_t = run_spmv_npu(case)
    check(out_t, dense.double().T @ x_t.double(), "long-tail-row-T")


# ---------------------------------------------------------------------------
# 5. 非法输入
# ---------------------------------------------------------------------------

def test_mismatched_shapes_are_rejected():
    dense = gen_dense(DenseSpec(8, 12, 0.3, torch.float32, seed=51))
    x = gen_vec(8, torch.float32, seed=52)
    with pytest.raises(RuntimeError, match="cannot be multiplied"):
        torch.mv(dense.to_sparse_csr().to(DEVICE), x.to(DEVICE))


def test_unsupported_value_dtype_is_rejected():
    dense = gen_dense(DenseSpec(8, 12, 0.3, torch.float32, seed=53)).double()
    x = torch.ones(12, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="int8, float16, bfloat16, float32, int32"):
        torch.mv(dense.to_sparse_csr().to(DEVICE), x.to(DEVICE))


def test_mismatched_dtypes_are_rejected():
    dense = gen_dense(DenseSpec(8, 12, 0.3, torch.float32, seed=54))
    x = gen_vec(12, torch.float16, seed=55)
    with pytest.raises(RuntimeError, match="same dtype"):
        torch.mv(dense.to_sparse_csr().to(DEVICE), x.to(DEVICE))


def test_cpu_operands_do_not_reach_the_npu_kernel():
    """pybind 测试入口显式拒绝 CPU 张量，不做 CPU fallback。"""
    dense = gen_dense(DenseSpec(8, 12, 0.3, torch.float32, seed=56))
    crow, col, values = csr_parts(dense, 0)
    x = gen_vec(12, torch.float32, seed=57)
    y = torch.zeros(8)
    with pytest.raises(RuntimeError, match="Ascend NPU"):
        spmv_ops().spmv_npu(crow.cpu(), col.cpu(), values.cpu(), x, y, 1, 0, "N", 0)


@pytest.mark.parametrize("bad_trans", ("n", "X", ""))
def test_invalid_trans_is_rejected(bad_trans):
    dense = gen_dense(DenseSpec(8, 12, 0.3, torch.float32, seed=58))
    crow, col, values = csr_parts(dense, 0)
    x = gen_vec(12, torch.float32, seed=59).to(DEVICE)
    y = torch.zeros(8, device=DEVICE)
    with pytest.raises(RuntimeError, match="trans must be N, T or H"):
        spmv_ops().spmv_npu(crow, col, values, x, y, 1, 0, bad_trans, 0)


def test_invalid_base_is_rejected():
    dense = gen_dense(DenseSpec(8, 12, 0.3, torch.float32, seed=60))
    crow, col, values = csr_parts(dense, 0)
    x = gen_vec(12, torch.float32, seed=61).to(DEVICE)
    y = torch.zeros(8, device=DEVICE)
    with pytest.raises(RuntimeError, match="base must be 0 or 1"):
        spmv_ops().spmv_npu(crow, col, values, x, y, 1, 0, "N", 2)


def test_invalid_dtype_combo_is_rejected():
    """complex64 输入配实数输出不在支持表内，必须稳定报错。"""
    dense = gen_dense(DenseSpec(8, 12, 0.3, torch.complex64, seed=62))
    crow, col, values = csr_parts(dense, 0)
    x = gen_vec(12, torch.complex64, seed=63).to(DEVICE)
    y = torch.zeros(8, device=DEVICE)
    with pytest.raises(RuntimeError, match="unsupported SpMV dtype combination"):
        spmv_ops().spmv_npu(crow, col, values, x, y, 1, 0, "N", 0)


# ---------------------------------------------------------------------------
# 6. 设备与异步
# ---------------------------------------------------------------------------

def test_repeated_calls_are_stable():
    dense = gen_dense(DenseSpec(24, 24, 0.3, torch.float32, seed=9))
    mat = dense.to_sparse_csr().to(DEVICE)
    x = gen_vec(24, torch.float32, seed=10).to(DEVICE)
    reference = None
    for _ in range(5):
        out = torch.mv(mat, x)
        torch.npu.synchronize()
        values = out.cpu()
        if reference is None:
            reference = values
        else:
            torch.testing.assert_close(values, reference, rtol=0, atol=0)


@pytest.mark.skipif(torch.npu.device_count() < 2, reason="needs a second NPU")
def test_non_default_device():
    """DeviceGuard 必须在分配输出前生效。"""
    dense = gen_dense(DenseSpec(12, 12, 0.3, torch.float32, seed=64))
    mat = dense.to_sparse_csr().to("npu:1")
    x = gen_vec(12, torch.float32, seed=65).to("npu:1")
    out = torch.mv(mat, x)
    torch.npu.synchronize("npu:1")
    assert out.device == mat.values().device
    torch.testing.assert_close(out.cpu(), dense @ x.cpu(), rtol=1e-5, atol=1e-5)


def test_non_default_stream():
    dense = gen_dense(DenseSpec(12, 12, 0.3, torch.float32, seed=66))
    mat = dense.to_sparse_csr().to(DEVICE)
    x = gen_vec(12, torch.float32, seed=67).to(DEVICE)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        out = torch.mv(mat, x)
    stream.synchronize()
    torch.testing.assert_close(out.cpu(), dense @ x.cpu(), rtol=1e-5, atol=1e-5)


def test_ordering_with_preceding_npu_work():
    """适配层读取调用方在同一 stream 上刚产生的输入。"""
    dense = gen_dense(DenseSpec(16, 16, 0.3, torch.float32, seed=68))
    scaled = (dense.to(DEVICE) * 2.0).to_sparse_csr()
    x = (gen_vec(16, torch.float32, seed=69).to(DEVICE) * 0.5)
    out = torch.mv(scaled, x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), (dense * 2.0) @ (gen_vec(16, torch.float32, seed=69) * 0.5),
                               rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. 打包验证
# ---------------------------------------------------------------------------

def test_wheel_contents_are_complete():
    root = os.path.dirname(cann_ops_sparse.__file__)
    for relative in ("ops/sparse/spmv/__init__.py", "ops/sparse/spmv/spmv.py",
                     "csrc/sparse/spmv/spmv.cpp", "common/aclsparse_common.h",
                     "common/cann_ops_sparse.h"):
        assert os.path.isfile(os.path.join(root, relative)), f"missing {relative}"


def test_builder_sources_resolve_inside_the_package():
    from cann_ops_sparse.ops.sparse.spmv.spmv import SpMVOpBuilder

    builder = SpMVOpBuilder()
    root = os.path.dirname(cann_ops_sparse.__file__)
    for source in builder.sources():
        assert os.path.isfile(os.path.join(root, source)), f"unresolved source {source}"


def test_extension_is_compiled_once_per_process():
    from cann_ops_sparse.ops.sparse.spmv.spmv import _spmv_op_builder

    dense = gen_dense(DenseSpec(8, 8, 0.3, torch.float32, seed=70))
    x = gen_vec(8, torch.float32, seed=71)
    torch.mv(dense.to_sparse_csr().to(DEVICE), x.to(DEVICE))
    first = _spmv_op_builder.load()
    torch.mv(dense.to_sparse_csr().to(DEVICE), x.to(DEVICE))
    assert _spmv_op_builder.load() is first
