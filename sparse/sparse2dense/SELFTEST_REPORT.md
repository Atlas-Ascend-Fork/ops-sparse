# aclsparseSparseToDense 自测报告

> 对应任务书交付件「自测报告」。950 实测以严格交付 stamp **`20260902_153138`**（P+内存）+ ATK 闭环 stamp **`20260902_154106`** 为准。

## 0. 任务书符合性一览

| 任务书条款 | 要求摘要 | 状态 | 证据 |
|------------|----------|------|------|
| §2.0 / §3.5 | Python/ATen：`to_dense`→`aten::_to_dense`，NPU 无 CPU fallback | **已达** | `sparse/sparse2dense/torch_extension/`（统一 torch_extension）；pytest `test/sparse2dense/python/` |
| §3.2 | 精度 exact / bit-wise；200 条任务包 | **已达** | ATK **200/200**；ACC_FB 200；CSV ST |
| §3.3 | P-01/02/03 ≥0.3×，warmup≥10 / samples≥30 | **已达** | `P_RATIO 90/90`（stamp `153138`） |
| §3.3+ | 泛化 extra 200 ≥0.3×（同 warmup/samples） | **已达** | stamp `20260907_090339`：`EXTRA_RATIO 200/200`（`0672c50`） |
| §3.4 | 内存：IO>500MB 额外≤50% **或** workspace≤L2 | **已达** | 290 条全过：12 条走 peak_50%；其余 workspace=0≤L2 |
| 交付「自测用例」 | 覆盖自验证核心场景（CSR/CSC/COO×5 dtype×base×layout+边界） | **已达** | **自测主集 500/500**（另有任务包 200） |
| §4 设计文档 | competitions 设计 PR | **已有** | cann-ops-competitions 设计 MR |
| §3.5 A2/A3 交叉 | A2/A3 回归 | **未跑** | 需另申请算力（非 A5 950 门禁） |

说明：任务书正文未写死「恰好 500 条」；「500」在 §3.4 指 **500MB 内存阈值**。自测主集 500 条是按 §3.5/交付件「覆盖核心场景」补齐的交付用例集。

## 1. 环境

| 项 | 值 |
|----|-----|
| 硬件 | Ascend 950PR（A5） |
| CANN | 9.1.0 |
| 代码仓 | https://gitcode.com/longcat_chen/ops-sparse |
| 分支 | `feature/aclsparse-sparsetodense-950` |
| 验证 HEAD | ATK：`9f57680`；报告文档：`7997810`+ |
| 编程模型 | arch35 SIMT |

## 2. 自测用例（按任务书覆盖添加）

| 集合 | 数量 | 文件 | 覆盖 |
|------|------|------|------|
| **自测主集** | **500** | `selftest_500_cases.json` | 480=CSR/CSC/COO×5dtype×base0/1×ROW/COL×8 shape + 边界 20 |
| 任务包精度 | 200 | `accuracy_cases.json` | ATK / ACC_FB（任务书 §5） |
| CSV ST | gtest | `sparse2dense_test` | C++ BitwiseGolden + L2 负例 |
| P / 内存 | 90+290 | `performance_cases.json` | §3.3 / §3.4 |
| PTA/ATen UT | pytest | `tests/test_sparsetodense_npu.py` | hook + `to_dense` CSR/CSC/COO + 异常 |

生成 / 执行 500：

```bash
python3 scripts/ci/generate_selftest_500.py
python3 scripts/ci/run_sparsetodense_selftest_500.py
```

## 3. 门禁结果

### 3.1 任务书严格 P + 内存（`20260902_153138`，warmup=10 / samples=30）

| 项 | 结果 |
|----|------|
| BUILD / PKG / INSTALL / ACC / PYTEST | **0** |
| SELFTEST_500 | **500/500** |
| **P_RATIO** | **90/90 ≥0.3** |
| **EXTRA_RATIO** | **200/200 ≥0.3**（stamp `20260907_090339`） |
| **P_MEM_COMPARE** | **ok=290 fail=0**（>500MB：12 条 peak 额外比=0；workspace 恒 0） |

### 3.2 ATK（`20260902_154106`）

| 项 | 结果 |
|----|------|
| ATK | **Total Task: 200, success 200** |
| ACC_FB / SELFTEST_500 | **0** |

## 4. PyTorch / ATen（§2.0）

- 绑定：统一 `torch_extension`（与 SpGEMM 同构）
- Dispatcher：`aten::_to_dense` SparseCsrPrivateUse1 / SparsePrivateUse1 → aclsparse
- UT：`test/sparse2dense/python/test_sparse2dense_torch_extension.py`（950：12 passed）

## 5. 内存（§3.4）

- `aclsparseSparseToDense_bufferSize` **恒返回 0** → 满足「workspace ≤ L2」
- IO>500MB 的 12 个 P 场景：NPU peak ≤ GPU peak（`extra_ratio=0`）→ 同时满足「额外 ≤50%」
- 采集/对比：`collect_sparse_ops_*_memory.py` + `scripts/ci/compare_sparsetodense_memory_taskbook.py`

## 6. 实现要点

- Kernel：同 stream SIMT zero → scatter（CSR 行 / CSC 列 / COO nnz）
- Host：nnz>0 热路径无 `aclrtMemsetAsync`
- torch_extension：`OpBuilder` JIT + `aten::_to_dense`
