# ltmatmul 白盒测试分支覆盖说明

## 1. 概述

本文档记录 `sparseLt/matmul/arch35/matmul_kernel.cpp` 中阶段一新增的 bias+ReLU+GeLU epilogue 链的原始设计，
并保留当时的分支与白盒用例映射（case_id 400-415）。

> **当前基线说明**：本文档中的映射、覆盖状态、验证结论及统计均为原始设计历史记录，不代表当前 CSV 加载的用例或当前分支覆盖率。在原始集合 400—415 中，当前 `ltmatmul_test.csv` 仅保留 case 400、403、406、412、413；其余 11 个 case 已不在当前参数化测试数据中。CSV 是当前用例集合的事实来源。

**目标芯片**: Ascend 950PR (arch35 / DAV_3510), CANN 9.1.0
**测试框架**: GTest + CSV 参数化驱动, Eigen FP64 golden

## 2. 分支清单与覆盖映射

### 2.1 路径选择分支 (host: launch_matmul_kernels)

| # | 分支 | 条件 | 覆盖用例 | 状态 |
|---|------|------|----------|------|
| B01 | 融合路径 (splitK==1) | `td.splitK == 1` → `splt_fused_matmul_kernel_launch` | 200-209, 300-399, 400-414 (除404,405,415) | 覆盖 |
| B02 | 非融合路径 (splitK>1) | `td.splitK > 1` → memset + matmul + epilogue | 210, 371-373, 394, 404, 405, 415 | 覆盖 |

### 2.2 FastPath vs GeneralPath (fused: SpltFusedEpilogueTileLoop L1634-1651)

| # | 分支 | 条件 | 覆盖用例 | 状态 |
|---|------|------|----------|------|
| B03 | FastPath (无 bias/act) | `alphaVec==0 && alpha==1 && beta==0 && biasDevPtr==0 && act==0` | 1-6, 14-17, 94, 98 (非 epilogue 用例) | 覆盖 |
| B04 | GeneralPath (有 bias 或 act) | 上述条件任一不满足 | 200-209, 300-399, 400-415 | 覆盖 |

### 2.3 INT32 快速路径旁路 (non-fused: SpltEpilogueProcessChunk L800-807)

| # | 分支 | 条件 | 覆盖用例 | 状态 |
|---|------|------|----------|------|
| B05 | INT32 快速路径 (int32 直出) | `TempType!=float && alphaVec==0 && alpha==1 && beta==0 && biasDevPtr==0 && act==0` | 117, 118, 119 | 覆盖 |
| B06 | INT32 快速路径旁路 (bias/act 触发) | 上述条件不满足 (bias 或 act 存在) | 207, 208, 373, 415 | 覆盖 |

> 注：B06 用例设计正确，覆盖了 bias/act 触发 INT32 快速路径旁路的分支（原 bias 缺陷已于 commit 96bf47b 修复，见 §4）。

### 2.4 bias 加载路径 (fused: SpltFusedEpilogueImpl L1761-1820)

| # | 分支 | 条件 | 覆盖用例 | 状态 |
|---|------|------|----------|------|
| B07 | 全量加载 biasChunkMode=0, FP32 直接加载 | `biasDevPtr!=0 && biasChunkMode==0 && (T==float \|\| T==int8_t)` → `DataCopyPad(biasVecUB, biasVecGM[0], ..., sizeof(float))` | 200, 203, 340-347, 407-409, 413 | 覆盖 |
| B08 | 全量加载 biasChunkMode=0, FP32 直接加载（FP16 输入） | `biasDevPtr!=0 && biasChunkMode==0 && T==__fp16` → FP32 直接加载（BUGFIX 后不再 `Cast`，统一 `sizeof(float)`） | 205, 312, 332, 337, 403, 410 | 覆盖 |
| B09 | 全量加载 biasChunkMode=0, FP32 直接加载（BF16 输入） | `biasDevPtr!=0 && biasChunkMode==0 && T==__bf16` → FP32 直接加载（BUGFIX 后不再 `SpltCastBf16ToFp32Vec`，统一 `sizeof(float)`） | 206, 314, 333, 346, 401 | 覆盖 |
| B10 | per-chunk 降级 biasChunkMode=1, FP32 直接加载 | `biasDevPtr!=0 && biasChunkMode==1 && (CType==float \|\| CType==int8_t)` | 302, 309, 335, 406 | 覆盖 |
| B11 | per-chunk 降级 biasChunkMode=1, FP32 直接加载（FP16 输入） | `biasDevPtr!=0 && biasChunkMode==1 && CType==__fp16` → FP32 直接加载（BUGFIX 后不再 `Cast`，统一 `sizeof(float)`） | **400** | 覆盖 |
| B12 | per-chunk 降级 biasChunkMode=1, FP32 直接加载（BF16 输入） | `biasDevPtr!=0 && biasChunkMode==1 && CType==__bf16` → FP32 直接加载（BUGFIX 后不再 `SpltCastBf16ToFp32Vec`，统一 `sizeof(float)`） | **401** | 覆盖 |

> bias 在所有 dtype 路径下统一按 FP32 直接加载（commit 96bf47b BUGFIX）。此前 FP16/BF16 路径按 `sizeof(T)` 读取导致半数数据丢失，修复后不再需要类型转换分支。详见 §4。

### 2.5 bias 加载路径 (non-fused: SpltEpilogueImpl L975-1009)

| # | 分支 | 条件 | 覆盖用例 | 状态 |
|---|------|------|----------|------|
| B13 | 非融合 FP32/INT8 bias 直接加载 | `biasDevPtr!=0 && (T==float \|\| T==int8_t)` → `DataCopyPad(biasVecUB, biasVecGM)` | 210, 371, 404 | 覆盖 |
| B14 | 非融合 FP16 bias FP32 直接加载 | `biasDevPtr!=0 && T==__fp16` → FP32 直接加载（BUGFIX 后不再 `Cast`，统一 `sizeof(float)`） | 372, 410(非融合) | 覆盖 |
| B15 | 非融合 BF16 bias FP32 直接加载 | `biasDevPtr!=0 && T==__bf16` → FP32 直接加载（BUGFIX 后不再 `SpltCastBf16ToFp32Vec`，统一 `sizeof(float)`） | **405** | 覆盖 |

### 2.6 activation 分支 (SpltEpilogueProcessChunk L836-841 / SpltFusedEpilogueGeneralPath L1552-1562)

| # | 分支 | 条件 | 覆盖用例 | 状态 |
|---|------|------|----------|------|
| B16 | 无 activation (act==0) | `activationType == 0` | 200, 408, 414 | 覆盖 |
| B17 | ReLU (act==1) | `activationType == 1` → `SpltApplyReLU(dFp32UB, count, reluThreshold, reluUpperBound)` | 201, 203, 205, 340-343, 407, 408 | 覆盖 |
| B18 | GeLU (act==2) | `activationType == 2` → `SpltApplyGeLU(dFp32UB, geluTemp, count, geluScaling)` | 202, 204, 206, 344-346, 402, 409, 410, 413 | 覆盖 |

> B18 中 case 402（无 bias 的 GeLU）原用于对比验证无 bias 路径正确性；bias 缺陷修复后所有含 bias 的 GeLU 用例同样通过（见 §4）。

### 2.7 GeLU Exp clamp z≤20 (SpltApplyGeLU L725)

| # | 分支 | 条件 | 覆盖用例 | 状态 |
|---|------|------|----------|------|
| B19 | GeLU Exp clamp 触发 | `Mins(geluTemp, geluTemp, 20.0f, count)` — 输入值 |x|>6 时 z>20 触发 clamp | **402**(range[-10,10]), **403**(range[-10,10]), **404**(range[-10,10]), **410**(range[-5,5]), **413**(range[-10,10]) | 覆盖 |

> case 402 (无 bias) 验证 GeLU clamp 在无 bias 时正确工作。

### 2.8 ReLU 参数变化 (SpltApplyReLU L690-698)

| # | 分支 | 条件 | 覆盖用例 | 状态 |
|---|------|------|----------|------|
| B20 | ReLU 默认参数 (thr=0, ub=FLT_MAX) | `reluThreshold==0.0f && reluUpperBound==FLT_MAX` | 201, 340, 380 | 覆盖 |
| B21 | ReLU 自定义 thr>0 | `reluThreshold > 0.0f` | 343 (thr=1), **407** (thr=5) | 覆盖 |
| B22 | ReLU 自定义 thr<0 | `reluThreshold < 0.0f` | 342 (thr=-1), **408** (thr=-5) | 覆盖 |
| B23 | ReLU 自定义 ub (有限值) | `reluUpperBound < FLT_MAX` | 212 (ub=6), 341 (ub=6), 343 (ub=100), **407** (ub=50), **408** (ub=5) | 覆盖 |
| B24 | ReLU ub=0 (全零输出) | `reluUpperBound == 0.0f` | 347, 385 | 覆盖 |

### 2.9 GeLU scaling 变化 (SpltApplyGeLU L731)

| # | 分支 | 条件 | 覆盖用例 | 状态 |
|---|------|------|----------|------|
| B25 | GeLU 默认 scaling=1.0 | `geluScaling == 1.0f` | 202, 344, 402 | 覆盖 |
| B26 | GeLU scaling<1.0 | `geluScaling < 1.0f` | 213 (scale=0.5), 345 (scale=0.5), **410** (scale=0.25) | 覆盖 |
| B27 | GeLU scaling>1.0 | `geluScaling > 1.0f` | 346 (scale=2.0), **409** (scale=3.0) | 覆盖 |
| B28 | GeLU scaling=0 (全零输出) | `geluScaling == 0.0f` | 388 | 覆盖 |

### 2.10 INT8 路径 bias + activation

| # | 分支 | 条件 | 覆盖用例 | 状态 |
|---|------|------|----------|------|
| B29 | INT8 INT32 + bias + ReLU (int32→float Cast in GeneralPath) | `AccType==int32_t && biasDevPtr!=0 && act==1` → `Cast(dFp32UB, accUB, CAST_NONE)` + ReLU | 207, **406** | 覆盖 |
| B30 | INT8 INT32 + bias + GeLU (int32→float Cast + GeLU) | `AccType==int32_t && biasDevPtr!=0 && act==2` | 208, 318, 319 | 覆盖 |
| B31 | INT8 INT8 + bias + activation (saturation Cast) | `OutType==int8_t && biasDevPtr!=0` → Mins/Maxs clamp to [-128,127] before Cast | 208, 316, 317, **412** | 覆盖 |
| B32 | INT8 INT32 splitK>1 + bias + activation (非融合 INT32 域 epilogue) | `splitK>1 && T==int8_t && biasDevPtr!=0` → `ApplyEpilogueInt32` path | 373, **415** | 覆盖 |

### 2.11 epilogue 链组合

| # | 分支 | 条件 | 覆盖用例 | 状态 |
|---|------|------|----------|------|
| B33 | alpha=0 + beta=1 + bias + GeLU (D=C+bias→GeLU) | `alpha==0 && beta==1 && biasDevPtr!=0 && act==2` | **411**, 384(act=2,alpha=0,beta=0) | 覆盖 |
| B34 | alpha=1 + beta=0 + bias only (无 act) | `alpha==1 && beta==0 && biasDevPtr!=0 && act==0` | 200, 350 | 覆盖 |
| B35 | 3vec + bias + ReLU (alphaVec+betaVec+bias 三向量) | `alphaVec==1 && betaVec==1 && biasDevPtr!=0 && act==1` | 209, 300, 304, 350, 353, **400** | 覆盖 |
| B36 | 3vec + bias + GeLU (三向量 + GeLU) | `alphaVec==1 && betaVec==1 && biasDevPtr!=0 && act==2` | 301, 305, 353, **401** | 覆盖 |

### 2.12 尾核/非对齐 shape

| # | 分支 | 条件 | 覆盖用例 | 状态 |
|---|------|------|----------|------|
| B37 | 小 m (m=16, 最小 2:4 稀疏行数) + bias + GeLU | `m==16 && biasDevPtr!=0 && act==2` → 尾核 curM < baseM | **413** | 覆盖 |
| B38 | 小 n (n=16, L0C C0 对齐) + bias + ReLU | `n==16 && biasDevPtr!=0 && act==1` → per-row bias 广播 n=16 | **414** | 覆盖 |
| B39 | 大 m (m=15737) + 3vec + bias → biasChunkMode 降级 | `m==15737 && alphaVec==1 && betaVec==1 && biasDevPtr!=0` → UB 超限降级 | 302, 309, 335, **400**, **401**, **406** | 覆盖 |
| B40 | 非对齐 m/k/n (37/151/1793) + bias + activation | `m=37, k=37, n=151` → curM/curN/curKL1 尾块 | 300-309, 330-339 | 覆盖 |

### 2.13 bias 特殊值 (golden SP1~SP10 对应)

| # | 分支 | 条件 | 覆盖用例 | 状态 |
|---|------|------|----------|------|
| B41 | bias 全零 (biasVec=[0,...]) | `bias_enabled==1 && biasSeed%10==0` | 380 (SP1), 200(seed=2230→pattern=0) | 覆盖 |
| B42 | bias 全正 (biasVec=[2.0,...]) | `bias_enabled==1 && biasSeed%10==1` | 381 (SP2) | 覆盖 |
| B43 | bias 全负 (biasVec=[-2.0,...]) | `bias_enabled==1 && biasSeed%10==2` | 382 (SP3) | 覆盖 |
| B44 | INT8 bias=127 (饱和上界) | `bias_enabled==1 && biasSeed%10==3` | 388 (SP9) | 覆盖 |
| B45 | INT8 bias=-128 (饱和下界) | `bias_enabled==1 && biasSeed%10==4` | 389 (SP10) | 覆盖 |

## 3. 原始设计基线的分支覆盖统计（历史记录）

| 分支组 | 总分支数 | 已覆盖 | 新增覆盖 | 未覆盖 |
|--------|---------|--------|---------|--------|
| 路径选择 (B01-B02) | 2 | 2 | 0 | 0 |
| FastPath/GeneralPath (B03-B04) | 2 | 2 | 0 | 0 |
| INT32 快速路径旁路 (B05-B06) | 2 | 2 | 1 (B06) | 0 |
| bias 加载-融合 (B07-B12) | 6 | 6 | 2 (B11,B12) | 0 |
| bias 加载-非融合 (B13-B15) | 3 | 3 | 1 (B15) | 0 |
| activation 分支 (B16-B18) | 3 | 3 | 0 | 0 |
| GeLU Exp clamp (B19) | 1 | 1 | 1 | 0 |
| ReLU 参数 (B20-B24) | 5 | 5 | 2 (B21,B22) | 0 |
| GeLU scaling (B25-B28) | 4 | 4 | 2 (B26,B27) | 0 |
| INT8 bias+act (B29-B32) | 4 | 4 | 2 (B31,B32) | 0 |
| epilogue 链组合 (B33-B36) | 4 | 4 | 1 (B33) | 0 |
| 尾核/非对齐 (B37-B40) | 4 | 4 | 3 (B37,B38,B39) | 0 |
| bias 特殊值 (B41-B45) | 5 | 5 | 0 | 0 |
| **合计** | **45** | **45** | **15** | **0** |

**原始设计记录中的分支覆盖率声明: 45/45 = 100%**（历史统计，不代表当前覆盖率）

> 新增白盒用例 (case_id 400-415) 补充覆盖了 15 个此前未覆盖的分支：
> - B11: per-chunk FP16 bias FP32 直接加载 (case 400)
> - B12: per-chunk BF16 bias FP32 直接加载 (case 401)
> - B15: 非融合 BF16 bias FP32 直接加载 (case 405)
> - B19: GeLU Exp clamp z≤20 (case 402, 403, 404, 410, 413)
> - B21-B22: ReLU 自定义 thr>0/thr<0 (case 407, 408)
> - B26-B27: GeLU scaling<1.0/>1.0 (case 410, 409)
> - B31-B32: INT8 INT8 bias+act / INT8 splitK2 bias+act (case 412, 415)
> - B33: alpha=0+beta=1+bias+GeLU 链 (case 411)
> - B37-B39: 小 m/小 n/大 m biasChunkMode 降级 (case 413, 414, 400/401/406)

## 4. 已知算子缺陷与修复记录（历史）

> **历史修复记录（非本轮验证）**：本节原记录的 bias+activation epilogue 系统性缺陷已通过 commit 96bf47b 的多处 `[BUGFIX]` 修复。以下保留缺陷描述与修复记录，供追溯。

**原缺陷描述**：bias+activation epilogue 路径曾存在系统性缺陷，所有 `bias_enabled=1` 的用例 NPU 结果与 golden 不匹配；不含 bias 的用例（如 case 2, 22, 41, 402）均通过。

**原根因**：当 `biasDevPtr != 0` 时 FastPath 条件为 false，GeneralPath 被触发。GeneralPath 中 `Muls(dFp32UB, accUB, alpha, ...)` 后的 bias 加载/应用与 tile loop 的 CrossCore 同步或 V pipeline 产生冲突，导致 matmul 结果（acc）被破坏。

**修复项（commit 96bf47b）**：

1. **accBuf 分配顺序修复**（`matmul_kernel.cpp:1779-1784`）：调整 accBuf 的 UB 分配优先级，避免与 bias UB 冲突。
2. **bias FP32 直接加载修复**（`matmul_kernel.cpp:1796-1801, 1804-1820`）：bias UB 分配顺序修复 + bias 始终按 FP32 直接加载（`sizeof(float)`）。此前 FP16/BF16 路径按 `sizeof(T)` 读取 bias 导致半数数据丢失；修复后所有 dtype 路径统一 FP32 加载，不再需要 `Cast(biasVecUB, biasLoadUB, ...)` 类型转换分支。
3. **CrossCore localTileIdx 每 batch 重置修复**（`matmul_kernel.cpp:1213-1241`）：`localTileIdx` 每 batch 重置为 0，与 AIV 侧 `SpltFusedEpilogueTileLoop` 的 per-batch 重置对齐。此前 localTileIdx 跨 batch 累加导致 AIC 奇偶翻转与 AIV 不匹配引发 CrossCore 死锁；修复后每 batch 末尾 drain 残留 flag，无 tile 的 block 跳过 CrossCoreSetFlag 避免残留 flag。

**原始验证记录**：本节当时记载修复后所有 bias+activation 用例（含当时的用例 200-399 和 400-415）NPU 结果与 golden 一致，并据此更新 §2 的历史分支覆盖状态。该记录不表示这些用例仍全部存在于当前 CSV，也不表示这些结果来自本轮 CI。

## 5. 新增白盒用例清单（原始设计历史记录；非当前 CSV 加载清单）

| case_id | 用例名 | 目标分支 | dtype | shape (m×k×n) | bias | act | splitK | 关键参数 |
|---------|--------|---------|-------|---------------|------|-----|--------|---------|
| 400 | WB1-FP16-perchunk-bias-Cast-ReLU | B11 | FP16 | 15737×37×151 | 1 | ReLU | 1 | 3vec, transAB, B-sparse, biasChunkMode=1 |
| 401 | WB2-BF16-perchunk-bias-Cast-GeLU | B12 | BF16 | 15737×37×151 | 1 | GeLU | 1 | 3vec, transAB, B-sparse, biasChunkMode=1 |
| 402 | WB3-GeLU-ExpClamp-z20-range10 | B19 | FP32 | 128×128×128 | 0 | GeLU | 1 | range=[-10,10], Exp clamp z≤20 |
| 403 | WB4-FP16-bias-GeLU-ExpClamp-range10 | B08,B19 | FP16 | 128×128×128 | 1 | GeLU | 1 | range=[-10,10], FP16 bias FP32 直接加载 + Exp clamp |
| 404 | WB5-splitK2-bias-GeLU-ExpClamp-range10 | B13,B19 | FP32 | 128×128×128 | 1 | GeLU | 2 | 非融合 bias + Exp clamp |
| 405 | WB6-BF16-splitK2-bias-ReLU-Cast | B15 | BF16 | 128×128×128 | 1 | ReLU | 2 | 非融合 BF16 bias FP32 直接加载 |
| 406 | WB7-INT8-perchunk-bias-ReLU-int32cast | B10,B29 | INT8/INT32 | 15737×37×151 | 1 | ReLU | 1 | 3vec, transAB, B-sparse, INT8 biasChunkMode + int32→float |
| 407 | WB8-ReLU-thr5-ub50-range1 | B21,B23 | FP32 | 128×128×128 | 1 | ReLU | 1 | thr=5.0, ub=50.0 |
| 408 | WB9-ReLU-symm-clamp-neg5-pos5-range1 | B22,B23 | FP32 | 128×128×128 | 1 | ReLU | 1 | thr=-5.0, ub=5.0 (对称 clamp) |
| 409 | WB10-GeLU-scale3-range1 | B27 | FP32 | 128×128×128 | 1 | GeLU | 1 | geluScaling=3.0 |
| 410 | WB11-FP16-GeLU-scale025-ExpClamp-range5 | B08,B19,B26 | FP16 | 128×128×128 | 1 | GeLU | 1 | geluScaling=0.25, range=[-5,5], Exp clamp |
| 411 | WB12-beta1-bias-GeLU-chain-range1 | B33 | FP32 | 128×128×128 | 1 | GeLU | 1 | alpha=0, beta=1 (D=C+bias→GeLU) |
| 412 | WB13-INT8-INT8-bias-GeLU-saturation-range5 | B31 | INT8/INT8 | 128×128×128 | 1 | GeLU | 1 | INT8 输出 saturation Cast + GeLU |
| 413 | WB14-m16-bias-GeLU-ExpClamp-minRow | B19,B37 | FP32 | 16×128×128 | 1 | GeLU | 1 | m=16 (最小 2:4 稀疏行), Exp clamp |
| 414 | WB15-n16-bias-ReLU-smallN-broadcast | B38 | FP32 | 128×128×16 | 1 | ReLU | 1 | n=16 (L0C C0 对齐), per-row bias 广播 |
| 415 | WB16-INT8-INT32-splitK2-bias-ReLU-transB | B06,B32 | INT8/INT32 | 128×128×128 | 1 | ReLU | 2 | 非融合 INT32 域 epilogue, transB=true |
