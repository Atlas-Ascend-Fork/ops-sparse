#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""aclsparseSparseToDense torch_extension（aten::_to_dense NPU 后端）端到端 UT。

在真实 NPU 上对已安装的 ``cann_ops_sparse`` wheel 运行::

    bash build.sh --ops=sparse2dense                       # 主线算子库（libops_sparse.so）
    bash build.sh --torch_extension --ops=sparse2dense     # torch extension wheel
    python3 -m pip install build_out/cann_ops_sparse-*.whl --force-reinstall --no-deps
    OPS_SPARSE_LIB_DIR=$PWD/build_out/lib64 ASCEND_RT_VISIBLE_DEVICES=4 \\
        python3 test/sparse2dense/python/test_sparse2dense_torch.py

覆盖：
  1. Dispatcher 注册断言：aten::_to_dense 在 NPU 稀疏后端路径由本适配层接管
     （正控制：float64 values 必须抛 RuntimeError——torch_npu 原生路径可正常转换）。
  2. CSR/CSC/COO × 5 dtype（int8/float16/bfloat16/float32/complex64）：NPU t.to_dense()
     vs CPU golden（赋值语义，-0.0 位元保留）bit-wise 一致（位元视图比较，含
     ±0.0/±inf/nan；未覆盖元素必须为正零位元）；索引 int32 直通 / int64 Device 侧
     显式 I32 cast。
  3. 乱序唯一坐标 COO + 同输入重复执行 bit-wise 确定性 + COO 重复坐标求和语义
     （Tensor.to_dense 对未 coalesce COO 的重复坐标按求和解释，适配层预归并后与
     同参数 CPU 构建的参考 bit-wise 一致；CSR/CSC 段内重复遵循 aclsparse 原生
     契约——任一写入值，非 torch 层归一范围）。
  4. 异常路径：float64 values / 非 None dtype / 非 None masked_grad / 3-D / hybrid COO
     → 均显式 RuntimeError（无 CPU fallback）。
  5. CPU sparse tensor 不被本注册拦截（原生 to_dense 照常，含 float64）。
  6. 同进程多卡切换 npu:0 → npu:1 → npu:0（每卡 to_dense ×2）：数据校验链记忆化 host 层
     per-device 隔离回归（可见卡 < 2 时显式 SKIP）。
  7. 原地改写索引数组回归（数据校验链记忆化内容指纹守卫）：同一 sparse tensor 先 to_dense
     成功，再原地改写索引为越界坐标（指针/规格均不变）→ 必须显式 RuntimeError（不得
     复用缓存的 SUCCESS 使越界坐标进入 scatter）；改回合法坐标后恢复成功且输出
     bit-wise 正确；原地改写 values 输出即时更新（仅校验结论被记忆化）。
"""
from __future__ import annotations

import dataclasses
import logging
import sys

import torch
import torch_npu  # noqa: F401  # NPU 后端（cann_ops_sparse 的注册覆盖须在 torch_npu 之后）
import cann_ops_sparse  # noqa: F401  # 触发 aten::_to_dense NPU 稀疏后端注册（JIT 延迟编译）

# torch 内省/Dispatcher 直入口（模块级绑定一次；getattr 字符串引用，无公开替代——
# 未 coalesce COO 的公开 .indices()/.values() 直接 RuntimeError）
_dispatch_has_kernel = getattr(getattr(torch, "_C"), "_dispatch_has_kernel_for_dispatch_key")
aten_to_dense = getattr(torch.ops.aten, "_to_dense")

ROWS, COLS, NNZ = 13, 9, 17
FORMATS = ("csr", "csc", "coo")
DTYPE_MAP = {
    "int8": torch.int8,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "complex64": torch.complex64,
}

PASSED = 0
FAILED = 0
FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        FAILURES.append((name, detail))
        logging.info("  [FAIL] %s: %s", name, detail)


def bits(t: torch.Tensor) -> torch.Tensor:
    """位元视图（bit-wise 判定；torch.equal 对 NaN 恒 False，必须走位元比较）。"""
    t = t.cpu().contiguous()
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype in (torch.float16, torch.bfloat16):
        return t.view(torch.int16)
    if t.dtype == torch.complex64:
        return t.view(torch.int32)  # 实/虚交错展开
    return t  # int8 位元即值


@dataclasses.dataclass
class GoldenParts:
    """compute_golden 的具名参数组（值数组 + 双索引 + 形状）。"""

    values: torch.Tensor
    primary: torch.Tensor
    secondary: torch.Tensor
    rows: int
    cols: int


def compute_golden(fmt: str, parts: GoldenParts) -> torch.Tensor:
    """官方赋值语义 CPU golden（B[row, col] = values[p] 位元搬运，-0.0 位元保留）。

    注：原生 t.to_dense() 为累加语义（0.0 + (-0.0) = +0.0），丢失 -0.0 位元；
    本算子与官方 _reference 均为位元搬运赋值语义 → bit-wise 口径以本 golden 为准。
    """
    values, primary, secondary = parts.values, parts.primary, parts.secondary
    rows, cols = parts.rows, parts.cols
    out = torch.zeros((rows, cols), dtype=values.dtype)
    base = 0
    if fmt == "csr":
        for row in range(rows):
            start, end = int(primary[row].item()) - base, int(primary[row + 1].item()) - base
            out[row, (secondary[start:end] - base).long()] = values[start:end]
    elif fmt == "csc":
        for col in range(cols):
            start, end = int(primary[col].item()) - base, int(primary[col + 1].item()) - base
            out[(secondary[start:end] - base).long(), col] = values[start:end]
    else:
        out[(primary - base).long(), (secondary - base).long()] = values
    return out


def special_values(dtype: torch.dtype, n: int) -> torch.Tensor:
    """确定性 values：嵌入正负零 / ±inf / nan 位元 + 普通值（纯位搬运，零算术）。"""
    if dtype == torch.int8:
        return torch.tensor([i - 8 for i in range(n)], dtype=torch.int8)
    if dtype == torch.complex64:
        vals = [
            complex(0.0, 0.0), complex(-0.0, 0.0), complex(0.0, -0.0),
            complex(float("inf"), 1.0), complex(1.0, float("-inf")),
            complex(float("nan"), 0.0), complex(0.0, float("nan")),
            complex(2.5, -1.75),
        ] + [complex(float(k) / 2.0, -float(k)) for k in range(1, 17)]
    else:
        vals = [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1.5, -2.25, 0.125]
        vals += [float(k) / 4.0 * (-1 if k % 2 else 1) for k in range(1, 17)]
    return torch.tensor(vals[:n], dtype=dtype)


def plain_values(dtype: torch.dtype, n: int) -> torch.Tensor:
    """普通有限非零值（无 ±0/inf/nan）——用于与原生 to_dense 的兼容性对照。"""
    if dtype == torch.int8:
        return torch.tensor([(i % 15) + 1 for i in range(n)], dtype=torch.int8)
    if dtype == torch.complex64:
        return torch.tensor([complex(k / 4.0 + 1.0, -k / 2.0 - 1.0) for k in range(n)],
                            dtype=torch.complex64)
    return torch.tensor([k / 4.0 + 1.0 for k in range(n)], dtype=dtype)


def make_coords(rows: int, cols: int, nnz: int, seed: int) -> list:
    """sorted unique 坐标（线性探测，确定性）。"""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    flat = set()
    while len(flat) < nnz:
        flat.add(int(torch.randint(0, rows * cols, (1,), generator=g)))
    return sorted((f // cols, f % cols) for f in flat)


@dataclasses.dataclass
class SparseSpec:
    """make_sparse 的具名参数组（坐标 + 值 + 形状 + 设备/索引 dtype）。"""

    coords: list
    values: torch.Tensor
    rows: int
    cols: int
    dev: str
    idx_dtype: torch.dtype


def make_sparse(fmt: str, spec: SparseSpec) -> torch.Tensor:
    """按格式构造 sparse tensor（CSR 行主序 / CSC 列主序排序展开；COO 保持给定乱序）。"""
    coords, values = spec.coords, spec.values
    rows, cols, dev, idx_dtype = spec.rows, spec.cols, spec.dev, spec.idx_dtype
    r = torch.tensor([c[0] for c in coords], dtype=idx_dtype)
    c = torch.tensor([c[1] for c in coords], dtype=idx_dtype)
    v = values.to(dev)
    if fmt == "csr":
        counts = torch.bincount(r.to(torch.int64), minlength=rows).to(idx_dtype)
        crow = torch.cat((torch.zeros(1, dtype=idx_dtype), counts.cumsum(0)))
        return torch.sparse_csr_tensor(crow.to(dev), c.to(dev), v, (rows, cols), device=dev)
    if fmt == "csc":
        order = torch.argsort(c.to(torch.int64) * rows + r.to(torch.int64))
        c, r = c[order], r[order]
        counts = torch.bincount(c.to(torch.int64), minlength=cols).to(idx_dtype)
        ccol = torch.cat((torch.zeros(1, dtype=idx_dtype), counts.cumsum(0)))
        return torch.sparse_csc_tensor(ccol.to(dev), r.to(dev), v, (rows, cols), device=dev)
    return torch.sparse_coo_tensor(torch.stack((r, c)).to(dev), v, (rows, cols), device=dev)


def reference_golden(t_cpu: torch.Tensor, fmt: str) -> torch.Tensor:
    """赋值语义 CPU golden（与 compute_golden 同构，从 sparse tensor 元数据展开）。"""
    rows, cols = t_cpu.shape
    if fmt == "csr":
        return compute_golden("csr", GoldenParts(t_cpu.values(), t_cpu.crow_indices(),
                                                t_cpu.col_indices(), rows, cols))
    if fmt == "csc":
        return compute_golden("csc", GoldenParts(t_cpu.values(), t_cpu.ccol_indices(),
                                                t_cpu.row_indices(), rows, cols))
    # 未 coalesce 的 COO：公开 .indices()/.values() 直接 RuntimeError（coalesce 会重排
    # 坐标），只能走底层访问——getattr 字符串引用
    idx = getattr(t_cpu, "_indices")()
    vals = getattr(t_cpu, "_values")()
    return compute_golden("coo", GoldenParts(vals, idx[0], idx[1], rows, cols))


def expect_runtime_error(name: str, fn) -> None:
    try:
        fn()
    except RuntimeError:
        check(name, True)
        return
    except Exception as e:  # noqa: BLE001
        check(name, False, f"expected RuntimeError, got {type(e).__name__}: {e}")
        return
    check(name, False, "expected RuntimeError, but call succeeded (CPU fallback or native path?)")


def run_dispatcher_checks(dev: str) -> None:
    """[1] Dispatcher 注册断言（含 float64 正控制：接管者必须是本适配层）。"""
    logging.info("=== [1] Dispatcher 注册断言 ===")
    check("has_kernel SparsePrivateUse1",
          _dispatch_has_kernel("aten::_to_dense", "SparsePrivateUse1"))
    check("has_kernel SparseCsrPrivateUse1",
          _dispatch_has_kernel("aten::_to_dense", "SparseCsrPrivateUse1"))
    # 正控制：float64（不支持 values dtype）必须抛错且错误信息出自本适配层
    # （"unsupported values dtype"）——证明接管者是我们的内核而非 torch_npu 原生路径
    coords = make_coords(ROWS, COLS, 3, 42)
    v64 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    for fmt in FORMATS:
        t64 = make_sparse(fmt, SparseSpec(coords, v64, ROWS, COLS, dev, torch.int64))
        try:
            t64.to_dense()
            check(f"float64 {fmt} t.to_dense() raises (ours, no fallback)", False,
                  "call succeeded (torch_npu native path took over?)")
        except RuntimeError as e:
            check(f"float64 {fmt} t.to_dense() raises (ours, no fallback)",
                  "unsupported values dtype" in str(e) or "SparseToDense supports" in str(e),
                  f"unexpected message: {e}")
        except Exception as e:  # noqa: BLE001
            check(f"float64 {fmt} t.to_dense() raises (ours, no fallback)", False,
                  f"{type(e).__name__}: {e}")
    t64_csr = make_sparse("csr", SparseSpec(coords, v64, ROWS, COLS, dev, torch.int64))
    expect_runtime_error("float64 csr torch.ops.aten._to_dense raises",
                         lambda: aten_to_dense(t64_csr))


def check_uncovered_zero_bits(fmt: str, dtype_name: str, out: torch.Tensor, coords: list) -> None:
    """未覆盖元素必须为正零位元（清零 kernel 语义；complex64 位元视图最后一维 ×2）。"""
    uncovered = torch.ones((ROWS, COLS), dtype=torch.bool)
    for r_, c_ in coords:
        uncovered[r_, c_] = False
    if dtype_name == "complex64":
        uncovered = uncovered.repeat_interleave(2, dim=1)
    check(f"{fmt}/{dtype_name} uncovered elements are +0 bits",
          bool((bits(out)[uncovered] == 0).all()))


def run_bitwise_special_values(dev: str) -> None:
    """[2] CSR/CSC/COO × 5 dtype bit-wise（含 ±0/±inf/nan 位元；未覆盖元素为正零）。"""
    logging.info("=== [2] CSR/CSC/COO × 5 dtype bit-wise（含 ±0/±inf/nan 位元；未覆盖元素为正零） ===")
    for di, (dtype_name, dtype) in enumerate(DTYPE_MAP.items()):
        for fi, fmt in enumerate(FORMATS):
            coords = make_coords(ROWS, COLS, NNZ, 1000 + di * 10 + fi)
            vals = special_values(dtype, NNZ)
            t = make_sparse(fmt, SparseSpec(coords, vals, ROWS, COLS, dev, torch.int64))
            out = t.to_dense()
            check(f"{fmt}/{dtype_name} output shape", tuple(out.shape) == (ROWS, COLS),
                  f"got {tuple(out.shape)}")
            check(f"{fmt}/{dtype_name} output dtype", out.dtype == dtype, f"got {out.dtype}")
            check(f"{fmt}/{dtype_name} output contiguous", out.is_contiguous())
            check(f"{fmt}/{dtype_name} output on NPU", out.device.type == "npu")
            golden = reference_golden(t.cpu(), fmt)
            check(f"{fmt}/{dtype_name} bit-wise vs CPU golden",
                  torch.equal(bits(out), bits(golden)))
            if dtype_name != "int8":  # 未覆盖元素必须为正零位元（清零 kernel 语义）
                check_uncovered_zero_bits(fmt, dtype_name, out, coords)


def run_native_compat(dev: str) -> None:
    """[2b] 普通有限值与原生 to_dense 兼容性对照（COO × 5 dtype）。"""
    logging.info("=== [2b] 普通有限值与原生 to_dense 兼容性对照（COO × 5 dtype） ===")
    for di, (dtype_name, dtype) in enumerate(DTYPE_MAP.items()):
        coords = make_coords(ROWS, COLS, NNZ, 1700 + di)
        vals = plain_values(dtype, NNZ)
        t = make_sparse("coo", SparseSpec(coords, vals, ROWS, COLS, dev, torch.int64))
        out = t.to_dense()
        check(f"coo/{dtype_name} bit-wise vs native to_dense (plain values)",
              torch.equal(bits(out), bits(t.cpu().to_dense())))
        if dtype_name == "float32":
            out_direct = aten_to_dense(t)  # Dispatcher 直入口（绕过方法绑定）
            check("coo/float32 torch.ops.aten._to_dense direct path bit-wise",
                  torch.equal(bits(out), bits(out_direct)))


def run_index_dtype(dev: str) -> None:
    """[3] 索引 dtype：int32 直通 / int64 Device 侧 I32 cast（float32 抽样）。"""
    logging.info("=== [3] 索引 dtype：int32 直通 / int64 Device 侧 I32 cast（float32 抽样） ===")
    for fi, fmt in enumerate(FORMATS):
        for idx_dtype in (torch.int32, torch.int64):
            coords = make_coords(ROWS, COLS, NNZ, 2000 + fi)
            vals = special_values(torch.float32, NNZ)
            t = make_sparse(fmt, SparseSpec(coords, vals, ROWS, COLS, dev, idx_dtype))
            out = t.to_dense()
            check(f"{fmt}/float32 idx={str(idx_dtype).split('.')[-1]} bit-wise",
                  torch.equal(bits(out), bits(reference_golden(t.cpu(), fmt))))


def run_determinism(dev: str) -> None:
    """[4] 乱序唯一坐标 COO + 重复执行 bit-wise 确定性。"""
    logging.info("=== [4] 乱序唯一坐标 COO + 重复执行 bit-wise 确定性 ===")
    g = torch.Generator(device="cpu")
    g.manual_seed(7)
    perm = torch.randperm(NNZ, generator=g).tolist()
    for dtype_name in ("float32", "complex64"):
        coords = make_coords(ROWS, COLS, NNZ, 3000)
        shuffled = [coords[i] for i in perm]  # 乱序唯一坐标
        vals = special_values(DTYPE_MAP[dtype_name], NNZ)
        t = make_sparse("coo", SparseSpec(shuffled, vals, ROWS, COLS, dev, torch.int64))
        out1 = t.to_dense()
        out2 = t.to_dense()  # 重复执行
        golden = reference_golden(t.cpu(), "coo")
        check(f"coo/{dtype_name} unordered-unique coords bit-wise",
              torch.equal(bits(out1), bits(golden)))
        check(f"coo/{dtype_name} repeat-run determinism", torch.equal(bits(out1), bits(out2)))
    for fmt in ("csr", "csc"):
        coords = make_coords(ROWS, COLS, NNZ, 3100)
        vals = special_values(torch.float16, NNZ)
        t = make_sparse(fmt, SparseSpec(coords, vals, ROWS, COLS, dev, torch.int64))
        out1, out2 = t.to_dense(), t.to_dense()
        check(f"{fmt}/float16 repeat-run determinism", torch.equal(bits(out1), bits(out2)))


def run_error_paths(dev: str) -> None:
    """[5] 异常路径（显式 RuntimeError，无 CPU fallback）。"""
    logging.info("=== [5] 异常路径（显式 RuntimeError，无 CPU fallback） ===")
    coords = make_coords(ROWS, COLS, 5, 5000)
    vf = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float16)
    t16 = make_sparse("csr", SparseSpec(coords, vf, ROWS, COLS, dev, torch.int64))
    # dtype 按 torch 原生语义生效（计算后转换），不再报错；
    # masked_grad 为自动求导标志，推理后端显式 RuntimeError。
    out_same = t16.to_dense(torch.float16)
    check("non-None dtype (same as values) applies", out_same.dtype == torch.float16)
    out_diff = t16.to_dense(torch.float32)
    check("non-None dtype (different) converts",
          out_diff.dtype == torch.float32 and float(out_diff[coords[0][0], coords[0][1]].cpu()) == 1.0)
    expect_runtime_error("non-None masked_grad raises",
                         lambda: t16.to_dense(None, masked_grad=True))

    idx3 = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int64, device=dev)
    t3d = torch.sparse_coo_tensor(idx3, torch.tensor([1.0, 2.0], device=dev), (2, 2, 2), device=dev)
    expect_runtime_error("3-D COO raises", lambda: t3d.to_dense())

    idxh = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64, device=dev)
    th = torch.sparse_coo_tensor(idxh, torch.ones(2, 3, device=dev), (2, 2, 3), device=dev)
    expect_runtime_error("hybrid COO (dense_dim=1) raises", lambda: th.to_dense())



def run_duplicate_coords(dev: str) -> None:
    """[4b] COO 重复坐标求和语义（vs PyTorch CPU 参考）。

    Tensor.to_dense 对未 coalesce COO 的重复坐标按求和解释（PyTorch 语义），
    适配层进入原生算子前预归并；归并后每位置单次写入，各子例的和在对应 dtype
    下精确可表示 → device 结果与同参数 CPU 构建的 to_dense 参考 bit-wise 一致。
    CSR/CSC 段内重复遵循 aclsparse 原生契约（任一写入值），不在本断言范围。
    """
    logging.info("=== [4b] COO 重复坐标求和语义（vs PyTorch CPU 参考） ===")

    def cpu_ref(coords: list, vals: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        idx = torch.tensor([[r for r, _ in coords], [c for _, c in coords]], dtype=torch.int64)
        return torch.sparse_coo_tensor(idx, vals, (rows, cols)).to_dense()

    # 1. 基本 fp32：同 (0,0) 值 1.0/3.0 → 求和 4.0（非任取其一）
    dup = [(0, 0), (1, 1), (0, 0)]
    vals1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    out1 = make_sparse("coo", SparseSpec(dup, vals1, ROWS, COLS, dev, torch.int64)).to_dense()
    check("coo duplicates sum: (0,0)==4.0", float(out1[0, 0].cpu()) == 4.0)
    check("coo duplicates sum: (1,1)==2.0", float(out1[1, 1].cpu()) == 2.0)
    check("coo duplicates sum: count_nonzero==2", int(torch.count_nonzero(out1).item()) == 2)
    check("coo duplicates sum: bit-wise vs CPU reference",
          torch.equal(bits(out1), bits(cpu_ref(dup, vals1, ROWS, COLS))))

    # 2. 乱序多重复组 fp32：3 组重复坐标乱序排列，含同坐标 3 次重复
    coords2 = [(2, 3), (5, 1), (2, 3), (0, 7), (5, 1), (2, 3), (0, 7), (9, 4), (5, 1)]
    vals2 = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
                         dtype=torch.float32)
    out2 = make_sparse("coo", SparseSpec(coords2, vals2, ROWS, COLS, dev, torch.int64)).to_dense()
    check("coo shuffled multi-group duplicates bit-wise vs CPU reference",
          torch.equal(bits(out2), bits(cpu_ref(coords2, vals2, ROWS, COLS))))

    # 3. 负值抵消 fp32：同坐标 1.0 + (-1.0) → 精确 0.0（任一写入值语义不可能得 0）
    cancel = [(4, 2), (4, 2), (6, 8)]
    vals3 = torch.tensor([1.0, -1.0, 5.0], dtype=torch.float32)
    out3 = make_sparse("coo", SparseSpec(cancel, vals3, ROWS, COLS, dev, torch.int64)).to_dense()
    check("coo negative cancellation: (4,2)==0.0 exactly", float(out3[4, 2].cpu()) == 0.0)
    check("coo negative cancellation: count_nonzero==1", int(torch.count_nonzero(out3).item()) == 1)
    check("coo negative cancellation: bit-wise vs CPU reference",
          torch.equal(bits(out3), bits(cpu_ref(cancel, vals3, ROWS, COLS))))

    # 4. complex64：同坐标 (1+2j) + (3-1j) → 4+1j
    ccoords = [(3, 5), (3, 5)]
    vals4 = torch.tensor([1.0 + 2.0j, 3.0 - 1.0j], dtype=torch.complex64)
    out4 = make_sparse("coo", SparseSpec(ccoords, vals4, ROWS, COLS, dev, torch.int64)).to_dense()
    check("coo complex64 duplicates sum to 4+1j", complex(out4[3, 5].cpu()) == 4.0 + 1.0j)
    check("coo complex64 duplicates bit-wise vs CPU reference",
          torch.equal(bits(out4), bits(cpu_ref(ccoords, vals4, ROWS, COLS))))

    # 5. fp16：同坐标 1.0 + 3.0 → 4.0（fp16 精确可表示）
    hcoords = [(7, 6), (7, 6)]
    vals5 = torch.tensor([1.0, 3.0], dtype=torch.float16)
    out5 = make_sparse("coo", SparseSpec(hcoords, vals5, ROWS, COLS, dev, torch.int64)).to_dense()
    check("coo fp16 duplicates sum to 4.0", float(out5[7, 6].cpu()) == 4.0)
    check("coo fp16 duplicates bit-wise vs CPU reference",
          torch.equal(bits(out5), bits(cpu_ref(hcoords, vals5, ROWS, COLS))))


def run_cpu_passthrough() -> None:
    """[6] CPU sparse tensor 不被本注册拦截（原生路径不变）。"""
    logging.info("=== [6] CPU sparse tensor 不被本注册拦截（原生路径不变） ===")
    for fmt in FORMATS:
        coords = make_coords(ROWS, COLS, 5, 6000)
        v64 = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5], dtype=torch.float64)
        t_cpu = make_sparse(fmt, SparseSpec(coords, v64, ROWS, COLS, "cpu", torch.int64))
        try:
            d = t_cpu.to_dense()
            check(f"cpu {fmt} float64 native to_dense works", tuple(d.shape) == (ROWS, COLS)
                  and d.dtype == torch.float64)
        except Exception as e:  # noqa: BLE001
            check(f"cpu {fmt} float64 native to_dense works", False, f"{type(e).__name__}: {e}")


def run_multi_device_switch_revisit() -> None:
    """[7] 同进程多卡切换回归（数据校验链记忆化 host 层按 devId 隔离）。

    npu:0 → npu:1 → npu:0，每次到访在目标卡新建张量（各卡独立 Device 分配）并
    to_dense ×2（首调指纹 miss 全量校验登记、再调指纹 hit 稳态）；覆盖适配层
    NPUGuard/AclSparseHandleCache（handle 按 device+stream 键控）与 host 记忆化层
    per-device 状态的整链。可见卡 < 2 显式 SKIP（单卡链路由 [1]-[6] 常规覆盖）。
    """
    count = torch.npu.device_count()
    if count < 2:
        logging.info("=== [7] 多卡切换回归 SKIP（device_count=%d < 2） ===", count)
        return
    logging.info("=== [7] 同进程多卡切换回归 npu:0 → npu:1 → npu:0（每卡 to_dense ×2） ===")
    for fmt in FORMATS:
        coords = make_coords(ROWS, COLS, NNZ, 7000 + FORMATS.index(fmt))
        vals = special_values(torch.float32, NNZ)
        for visit, dev in (("v1", "npu:0"), ("v2", "npu:1"), ("v3", "npu:0")):
            t = make_sparse(fmt, SparseSpec(coords, vals, ROWS, COLS, dev, torch.int64))
            out1 = t.to_dense()
            out2 = t.to_dense()
            golden = reference_golden(t.cpu(), fmt)
            check(f"{fmt} {visit} {dev} bit-wise vs CPU golden",
                  torch.equal(bits(out1), bits(golden)))
            check(f"{fmt} {visit} {dev} repeat determinism",
                  torch.equal(bits(out1), bits(out2)))


def run_inplace_index_mutation(dev: str) -> None:
    """[8] 原地改写索引数组回归：记忆化命中必须以内容指纹为前提。

    同一 sparse tensor 先 to_dense 成功（缓存登记 SUCCESS），随后原地改写索引数组为
    越界坐标——指针与规格完全不变，若缓存仅按指针命中会复用陈旧 SUCCESS，使新的
    越界坐标直接进入 scatter 越界写输出。指纹守卫下：内容必变 → 指纹必变 → miss →
    全量坐标重校 → 显式 RuntimeError。改回合法坐标恢复成功（输出 bit-wise 不变）；
    原地改写 values 时输出即时更新（被记忆化的只有校验结论，计算每调用重跑）。
    """
    logging.info("=== [8] 原地改写索引数组回归（指纹守卫：越界改写必须报错） ===")
    for fmt in FORMATS:
        coords = make_coords(ROWS, COLS, NNZ, 8000 + FORMATS.index(fmt))
        t = make_sparse(fmt, SparseSpec(coords, plain_values(torch.float32, NNZ), ROWS, COLS, dev, torch.int64))
        out0 = t.to_dense()
        check(f"{fmt} initial to_dense succeeds", tuple(out0.shape) == (ROWS, COLS))

        # 索引视图（CSR=col_indices / CSC=row_indices / COO=_indices 行分量——未
        # coalesce COO 公开访问 RuntimeError，走底层 getattr；视图写入直通底层
        # 存储且共享版本计数，适配层 int64→I32 cast 记忆化随之失效重转）
        if fmt == "csr":
            idx_view = t.col_indices()
            oob_value = COLS + 7
        elif fmt == "csc":
            idx_view = t.row_indices()
            oob_value = ROWS + 7
        else:
            idx_view = getattr(t, "_indices")()[0]
            oob_value = ROWS + 7
        saved = idx_view[0].clone()
        idx_view[0] = oob_value  # 原地写：指针/规格不变，仅内容变化
        expect_runtime_error(f"{fmt} in-place OOB index mutation raises", lambda: t.to_dense())

        # 改回合法坐标 → 恢复成功且输出 bit-wise 不变（同一坐标集合）
        idx_view[0] = saved
        out1 = t.to_dense()
        check(f"{fmt} restored valid index succeeds with unchanged output",
              torch.equal(bits(out0), bits(out1)))

        # 原地改写 values → 输出即时更新（校验结论记忆化不缓存计算结果）
        vals_view = getattr(t, "_values")() if fmt == "coo" else t.values()
        vals_view[0] = vals_view[0] + 1.0
        out2 = t.to_dense()
        golden = reference_golden(t.cpu(), fmt)
        check(f"{fmt} in-place values mutation reflected in output",
              torch.equal(bits(out2), bits(golden)))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    torch.npu.set_device(0)
    dev = "npu:0"

    logging.info("torch %s / torch_npu %s / device %s", torch.__version__, torch_npu.__version__, dev)
    run_dispatcher_checks(dev)
    run_bitwise_special_values(dev)
    run_native_compat(dev)
    run_index_dtype(dev)
    run_determinism(dev)
    run_duplicate_coords(dev)
    run_error_paths(dev)
    run_cpu_passthrough()
    run_multi_device_switch_revisit()
    run_inplace_index_mutation(dev)

    logging.info("\n=== RESULT: %d passed, %d failed ===", PASSED, FAILED)
    if FAILURES:
        for name, detail in FAILURES:
            logging.info("  FAIL %s: %s", name, detail)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
