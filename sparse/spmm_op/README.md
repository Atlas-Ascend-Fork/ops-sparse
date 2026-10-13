# aclsparseSpMMOp 算子文档

## 算子概述

**稀疏矩阵-稠密矩阵乘法（in-place 输出 + 可定制 epilogue）**。

计算如下表达式：

$$
\mathbf{C}_{ij} = \text{epilogue}\!\left(\alpha \sum_k \text{op}(A)_{ik} \cdot \text{op}(B)_{kj} + \beta \cdot C_{ij}\right)
$$

其中：
- $A$ 是 $m \times k$ 的稀疏矩阵（CSR 格式）
- $B$ 是稠密矩阵，$\text{op}(B) = B$（$k \times n$，当 `opB=NON_TRANSPOSE`）或 $B^T$（$n \times k$，当 `opB=TRANSPOSE`）
- $C$ 是 $m \times n$ 的稠密矩阵（in-place：读入 $\beta \cdot C$ 并写出结果）
- $\alpha$, $\beta$ 是标量，可为 HOST 或 DEVICE 指针
- $\text{epilogue}$ 为自定义 elementwise 函数（NPU 侧固定为 identity，即不附加操作）

本算子对标 cuSPARSE `cusparseSpMM()` Generic API 语义，主要特性：
- **确定性（bit-wise 可重复）**：ALG1/ALG2 均保证
- **行切分 + nTile=2048**：每个 AIV 核心处理一组行，每 tile 至多处理 2048 列；Vector 指令按 128 个元素分批执行（DataCopyPad + Muls + Add）；ALG1 静态均匀行切分，ALG2 按 nnz 负载均衡切分
- **FP32 累加**：无论输入/输出 dtype，累加均在 FP32 完成
- **FP16 饱和截断**：FP16 输出时截断到 ±65504，避免 Inf/NaN
- **execute 异步执行**：execute 阶段 kernel 异步入队到 handle 关联的 stream，函数立即返回；ALG2 的 createDescr 预处理为 host-side 同步执行
- **支持 opB 转置**：`opB ∈ {NON_TRANSPOSE, TRANSPOSE}`
- **B/C 独立主序**：支持 4 种 order_pair 组合（RR/RC/CR/CC）
- **支持未排序 CSR 索引**：colInd 无需行内递增，kernel 按行偏移区间遍历
- **支持值更新（ALG1）**：可在不重建 descr/plan 的情况下更新 CSR 值数组

## 接口列表

| 接口名 | 说明 |
|--------|------|
| `aclsparseSpMMOp_bufferSize` | 获取 workspace 大小 |
| `aclsparseSpMMOp_createDescr` | 创建描述符并执行预处理 |
| `aclsparseSpMMOp_destroyDescr` | 销毁描述符 |
| `aclsparseSpMMOp_createPlan` | 创建执行计划 |
| `aclsparseSpMMOp_destroyPlan` | 销毁执行计划 |
| `aclsparseSpMMOp_setGlobalUserData` | 设置 epilogue 用户数据（NPU 侧为 no-op） |
| `aclsparseSpMMOp` | 执行计算 |

## 产品支持情况

| 芯片系列 | 支持情况 |
|---------|---------|
| Ascend950 (arch35) | ✅ |

## 接口详情

### aclsparseSpMMOp_bufferSize

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMMOp_bufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMOpAlg_t alg,
    size_t *bufferSize);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | IN | `aclsparseHandle_t` | aclsparse 句柄 | Host |
| opA | IN | `aclsparseOperation_t` | op(A)，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE` | Host |
| opB | IN | `aclsparseOperation_t` | op(B)，支持 `NON_TRANSPOSE` / `TRANSPOSE` | Host |
| matA | IN | `aclsparseConstSpMatDescr_t` | CSR 稀疏矩阵描述符 | Host |
| matB | IN | `aclsparseConstDnMatDescr_t` | 稠密矩阵 B（可为 NULL） | Host |
| matC | IN | `aclsparseDnMatDescr_t` | 稠密矩阵 C（可为 NULL） | Host |
| computeType | IN | `aclDataType` | 计算类型，仅 `ACL_FLOAT` | Host |
| alg | IN | `aclsparseSpMMOpAlg_t` | 算法选择 | Host |
| bufferSize | OUT | `size_t *` | 返回所需 workspace 字节数 | Host |

#### 约束说明

| 约束 | 说明 |
|------|------|
| ALG1/DEFAULT | `bufferSize` 返回 0（无需 workspace） |
| ALG2 | `bufferSize` 返回 header + reorder + bin_edge 三块总大小（按 64 字节对齐）。ALG2 preprocess 在 host 侧同步完成（std::stable_sort + aclrtMemcpy），排序临时缓冲位于 host 侧 std::vector，不占用 device workspace |
| matA 格式 | 必须为 CSR |
| matA 值类型 | FP32 或 FP16（A/B/C 必须一致） |
| matA 索引类型 | ptrType 支持 I32/I64；IdxType 必须为 I32 |
| matA 索引基 | 支持 ZERO / ONE |
| CSR 内容契约 | colInd 值域须 ∈ [indexBase, indexBase+k)，rowOffsets 须单调不减，rowOffsets[0] 与 rowOffsets[m] 须与 nnz 一致。以上由调用方保证，库不做运行时校验 |
| computeType | 必须为 ACL_FLOAT |

---

### aclsparseSpMMOp_createDescr

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMMOp_createDescr(
    aclsparseHandle_t handle,
    aclsparseSpMMOpDescr_t *descr,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMOpAlg_t alg,
    void *buffer);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | IN | `aclsparseHandle_t` | aclsparse 句柄 | Host |
| descr | OUT | `aclsparseSpMMOpDescr_t *` | 输出描述符 | Host |
| opA | IN | `aclsparseOperation_t` | op(A) | Host |
| opB | IN | `aclsparseOperation_t` | op(B) | Host |
| matA | IN | `aclsparseConstSpMatDescr_t` | CSR 稀疏矩阵描述符 | Host |
| matB | IN | `aclsparseConstDnMatDescr_t` | 稠密矩阵 B（可为 NULL） | Host |
| matC | IN | `aclsparseDnMatDescr_t` | 稠密矩阵 C（可为 NULL） | Host |
| computeType | IN | `aclDataType` | 计算类型 | Host |
| alg | IN | `aclsparseSpMMOpAlg_t` | 算法选择 | Host |
| buffer | IN | `void *` | workspace buffer（ALG2 必需） | Device |

#### 约束说明

| 约束 | 说明 |
|------|------|
| ALG2 必需 buffer | ALG2 算法时 buffer 不可为 NULL，且 ≥ bufferSize 返回值 |
| buffer 生命周期 | buffer 必须在 descr 销毁前保持有效 |
| ALG2 preprocess 执行方式 | host-side 同步执行（D2H 读取 rowOffsets → `std::stable_sort` 排序 → 计算 bin_edge → H2D 写入 reorder/bin_edge），不涉及 kernel launch；preprocess 仅在 createDescr 阶段执行一次，plan 生命周期内复用 |
| matA 绑定语义 | matA 的 CSR 指针（csrRowOffsets/csrColInd/csrValues）以弱引用存入 descr，plan 生命周期内不可更换 matA |
| opB 绑定语义 | opB 在 createDescr 阶段绑定，plan 生命周期固定 |
| ALG1 值更新 | ALG1 允许原地更新 csrValues 指针指向的数据，不重建 descr/plan |
| ALG2 值更新警告 | ALG2 下原地修改 csrValues 会使 reorder/bin_edge 失效（用户责任，无运行时检测） |
| CSR 内容契约 | colInd 值域须 ∈ [indexBase, indexBase+k)，rowOffsets 须单调不减，rowOffsets[0] 与 rowOffsets[m] 须与 nnz 一致。以上由调用方保证，库不做运行时校验 |

---

### aclsparseSpMMOp_destroyDescr

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMMOp_destroyDescr(aclsparseSpMMOpDescr_t descr);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| descr | IN | `aclsparseSpMMOpDescr_t` | 要销毁的描述符 | Host |

#### 约束说明

| 约束 | 说明 |
|------|------|
| 幂等语义 | descr 为 nullptr 时直接返回 SUCCESS |
| 生命周期约束 | 必须先于 destroyDescr 调用 destroyPlan |

---

### aclsparseSpMMOp_createPlan

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMMOp_createPlan(
    aclsparseHandle_t handle,
    aclsparseSpMMOpDescr_t descr,
    aclsparseSpMMOpPlan_t *plan,
    const void *epilogueLTOBuffer,
    size_t epilogueLTOBufferSize);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | IN | `aclsparseHandle_t` | aclsparse 句柄 | Host |
| descr | IN | `aclsparseSpMMOpDescr_t` | 来自 createDescr 的描述符（非 const，对标 cuSPARSE） | Host |
| plan | OUT | `aclsparseSpMMOpPlan_t *` | 输出执行计划 | Host |
| epilogueLTOBuffer | IN | `const void *` | epilogue LTO-IR（NPU 侧必须为 NULL） | Host |
| epilogueLTOBufferSize | IN | `size_t` | LTO-IR 大小（NPU 侧必须为 0） | Host |

#### 约束说明

| 约束 | 说明 |
|------|------|
| epilogue 不支持 | NPU 侧使用 identity epilogue；传入非 NULL LTO-IR 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` |

---

### aclsparseSpMMOp_destroyPlan

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMMOp_destroyPlan(aclsparseSpMMOpPlan_t plan);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| plan | IN | `aclsparseSpMMOpPlan_t` | 要销毁的执行计划 | Host |

#### 约束说明

| 约束 | 说明 |
|------|------|
| 幂等语义 | plan 为 nullptr 时直接返回 SUCCESS |
| 生命周期约束 | destroyPlan 必须先于 destroyDescr（否则 plan->descr 悬垂） |

---

### aclsparseSpMMOp_setGlobalUserData

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMMOp_setGlobalUserData(
    aclsparseHandle_t handle,
    aclsparseSpMMOpPlan_t plan,
    const char *epilogueDataName,
    void *epilogueData,
    size_t epilogueDataSize);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | IN | `aclsparseHandle_t` | aclsparse 句柄 | Host |
| plan | IN | `aclsparseSpMMOpPlan_t` | 执行计划 | Host |
| epilogueDataName | IN | `const char *` | epilogue 数据变量名 | Host |
| epilogueData | IN | `void *` | epilogue 数据（非 const，对标 cuSPARSE） | Host |
| epilogueDataSize | IN | `size_t` | 数据字节数 | Host |

#### 约束说明

| 约束 | 说明 |
|------|------|
| NPU 侧为 no-op | identity epilogue 无需辅助数据，直接返回 SUCCESS |
| 参数一致性 | epilogueData 和 epilogueDataSize 必须同时有效或同时无效 |

---

### aclsparseSpMMOp（主执行入口）

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMMOp(
    aclsparseHandle_t handle,
    aclsparseSpMMOpPlan_t plan,
    const void *alpha,
    const void *beta,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | IN | `aclsparseHandle_t` | aclsparse 句柄 | Host |
| plan | IN | `aclsparseSpMMOpPlan_t` | 执行计划 | Host |
| alpha | IN | `const void *` | 标量 α（FP32 类型，HOST 或 DEVICE 指针，由 handle pointerMode 决定） | Host/Device |
| beta | IN | `const void *` | 标量 β（FP32 类型，HOST 或 DEVICE 指针） | Host/Device |
| matB | IN | `aclsparseConstDnMatDescr_t` | 稠密矩阵 B | Host |
| matC | IN/OUT | `aclsparseDnMatDescr_t` | 稠密矩阵 C（in-place） | Host |

#### 约束说明

| 约束 | 说明 |
|------|------|
| alpha/beta 指针类型 | 由 `aclsparseSetPointerMode` 设置，HOST 时直接解引用，DEVICE 时从 Device 内存读 |
| matB/matC 可变 | matB/matC 可跨多次 execute 更换，n/ldb/ldc/order 可变；但 m/k/opB 必须与 descr 一致 |
| 异步返回 | 调用后 kernel 入队到 handle 关联的 stream，函数立即返回 |
| 确定性 | ALG1/ALG2 均提供 bit-wise 可重复结果 |
| 维度校验 | matC.rows == m，matC.cols == n；opB=N 时 matB 为 (k×n)，opB=T 时 matB 为 (n×k)；matA.rows / matC.cols 不超过 INT32_MAX。零维支持：m=0 或 n=0 时输出为空矩阵，跳过 kernel 直接返回 SUCCESS；k=0（含 nnz=0）时积项为空，走 BetaC 路径计算 C = beta·C 后返回 SUCCESS |
| dtype 一致性 | matA/matB/matC 的 valueType 必须一致（FP32 或 FP16） |
| beta==0 短路 | HOST mode + beta==0 时跳过 matC.values 非空校验；DEVICE mode 跳过 matC.values 非空校验 |
| nnz==0 快捷路径 | nnz==0 时跳过 CSR 遍历，直接做 C = beta * C |

## 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 唯一支持的格式；ptrType 支持 I32/I64，IdxType 必须 I32；indexBase 支持 ZERO/ONE |
| COO | ❌ | |
| CSC | ❌ | |
| BSR | ❌ | |
| Blocked-ELL | ❌ | |
| Sliced-ELL | ❌ | |

## 支持的算法

| 算法 | 说明 |
|------|------|
| `ACL_SPARSE_SPMMOP_ALG_DEFAULT` | 默认算法，当前等同于 ALG1 |
| `ACL_SPARSE_SPMMOP_ALG1` | 确定性算法；host-side 静态行切分，无 workspace，支持不重建 descr/plan 的情况下更新 CSR 值数组 |
| `ACL_SPARSE_SPMMOP_ALG1_HIGH_PRECISION` | FP32 Kahan 补偿求和算法；workspace 语义同 ALG1（无 workspace）；仅 FP32 dtype 生效（FP16 静默忽略，退化为 ALG1 行为）；累加使用 Kahan 补偿求和以压制长行舍入与抵消误差 |
| `ACL_SPARSE_SPMMOP_ALG2` | 确定性算法；host-side merge sort（`std::stable_sort`）+ bin_edge 负载均衡，按每行 nnz 均衡各 block 工作量，性能可能更高，但不支持值更新且需要 workspace |

### 算法说明

#### ALG1（host-side 静态行切分）

ALG1 采用 host 侧均匀行切分策略，无预处理、无 workspace：

1. host 侧计算 `useBlocks = min(aivCoreNum, ceil(m, 128))`，`rowsPerBlock = ceil(m, useBlocks)`。
2. 每个 block 处理连续行区间 `[blockId × rowsPerBlock, min((blockId+1) × rowsPerBlock, m))`。
3. kernel 采用 SIMD 模式（class-based），每个 AIV 核心顺序遍历所分行，对 n 维按 nTile=2048 分 tile，每次用 `DataCopyPad` 将 B 行段从 GM 加载到 UB。
4. 对每行的 CSR 非零元素遍历，用 `Muls`（标量×向量）和 `Add`（向量累加）做 FP32 累加：`accBuf[0:nTile] += values[p] × bBuf[0:nTile]`。
5. 累加完成后应用 `alpha`/`beta`：`Muls(outBuf, accBuf, alpha)` + `Add(outBuf, outBuf, beta×cOldBuf)`，写回 C。

**负载特征**：当 CSR 行间 nnz 分布不均时，各 block 工作量差异大（ALG1 不做负载均衡）。
**适用场景**：nnz 分布较均匀、m 较小、或需原地更新 csrValues（ALG2 的 reorder 会失效）。

#### ALG2（host-side merge sort + bin_edge 负载均衡）

ALG2 在 createDescr 阶段执行 host-side 预处理，按每行 nnz 做负载均衡：

1. **D2H 读取**：`aclrtMemcpy`（同步）将 CSR `rowOffsets[m+1]` 从 device 拷贝到 host。按 `rowOffsetType`（I32/I64）决定拷贝字节数；I32 路径先拷到临时 `int32_t` buffer 再提升为 `int64_t`，I64 路径直接拷入 `int64_t` buffer。
2. **计算每行 nnz**：`rowNnz[i] = rowOffsets[i+1] - rowOffsets[i]`，初始化 `reorder[i] = i`。
3. **排序**：`std::stable_sort` 按 nnz 降序排序 `reorder`（O(m log m)），nnz 相同时保持行号顺序。
4. **计算 bin_edge**：按排序后 nnz 累计，每达到 `totalNnz / numBlocks` 即切一个 bin，得到 `bin_edge[numBlocks+1]`，使各 bin 的 nnz 总量大致相等。
5. **H2D 写回**：`aclrtMemcpy`（同步）将 `reorder[m]` 和 `bin_edge[numBlocks+1]` 写入 device workspace。

execute 阶段，kernel 内 block `b` 处理行范围 `[binEdge[b], binEdge[b+1])`，这些行在 reorder 后是 nnz 降序排列的连续区间，各 block 的 nnz 总量大致相等。

**负载特征**：各 block 的 nnz 总量大致相等，吸收行间 nnz 不均。
**适用场景**：nnz 分布极不均匀（幂律分布）、m 较大。
**限制**：createDescr 后原地修改 csrValues 会使 reorder/bin_edge 失效（用户责任，无运行时检测）。

> **平台说明**：ALG2 preprocess 采用 host-side 实现（而非 device-side kernel），原因是在 dav-3510 平台上 AIV-only `__aicore__` kernel 直接访问 `__gm__` 在非极小矩阵上不稳定（AIC error 507035）。仓内 `sparse/spmm/arch35/spmm_csr_mat.cpp` 亦使用 host-side 预处理作为先例。

### 精度说明

| 项目 | 说明 |
|------|------|
| 累加精度 | FP32 累加（无论输入/输出 dtype）。每个 AIV 核心持有 2048 个 `float` 累加器（nTile=2048，存储在 UB LocalTensor 中）。默认朴素累加使用 `Muls` + `Add` 向量指令；`ALG1_HIGH_PRECISION` 使用向量化 Kahan 补偿求和（5 条向量指令：`Sub→Add→Sub→Sub→Copy`）以压制长行舍入与抵消误差。Kahan 路径仅 FP32 dtype 生效，FP16 静默忽略 |
| FP32 路径 | 输入 FP32 → FP32 累加 → 输出 FP32 |
| FP16 路径 | 输入 FP16 → `Cast` 转 FP32 累加 → 输出时 `Mins`/`Maxs` 向量饱和截断到 ±65504 后 `Cast` 回 FP16，避免 Inf/NaN |
| computeType | 固定 `ACL_FLOAT`（FP32），不可更改 |
| 确定性 | ALG1/ALG2 均保证 bit-wise 可重复结果 |

## 支持的数据类型

| 数据类型组合 | A | B | C | computeType | 累加精度 |
|-------------|---|---|---|-------------|---------|
| FP32 | FP32 | FP32 | FP32 | FP32 | FP32 |
| FP16 | FP16 | FP16 | FP16 | FP32 | FP32（输入转 FP32，输出饱和截断到 FP16） |

> 不支持 INT8 / BF16 / FP64。不支持混合 dtype（A/B/C 必须一致）。computeType 固定 ACL_FLOAT。

## 支持的操作类型

| 操作 | opA | opB |
|------|-----|-----|
| NON_TRANSPOSE | ✅（唯一支持） | ✅ |
| TRANSPOSE | ❌（CSR 不可转置） | ✅ |
| CONJUGATE_TRANSPOSE | ❌ | ❌ |

## B/C 主序支持

| order_pair | matB.order | matC.order | 编码值 |
|-----------|-----------|-----------|-------|
| RR | ROW | ROW | 0 |
| RC | ROW | COL | 1 |
| CR | COL | ROW | 2 |
| CC | COL | COL | 3 |

## 返回值 / 错误码

所有 7 个 API 均返回 `aclsparseStatus_t` 枚举值：

| 返回值 | 含义 | 触发场景 |
|--------|------|----------|
| `ACL_SPARSE_STATUS_SUCCESS` (0) | 执行成功 | 正常完成 |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为 nullptr | 所有 API 的 handle 参数为 NULL |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | 参数非法 | descr/plan/bufferSize/alpha/beta/matB/matC 为 nullptr；CSR device 指针为 nullptr；维度不匹配；ld 不合法；matC.cols < 0；epilogueData 与 epilogueDataSize 不一致 |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 不支持的配置 | matA 格式非 CSR；valueType 非 FP32/FP16；ptrType 非 I32/I64；IdxType 非 I32；indexBase 非 ZERO/ONE；opA 非 NON_TRANSPOSE；opB 非 NON_TRANSPOSE/TRANSPOSE；computeType 非 ACL_FLOAT；alg 非法；epilogueLTOBuffer 非 NULL；m/k/ld/nnz 超过 INT32_MAX；dtype 组合不支持 |
| `ACL_SPARSE_STATUS_ALLOC_FAILED` | 内存分配失败 | `new aclsparseSpMMOpDescr` 或 `new SpmmOpPlanData` 失败；createDescr 内 `std::vector` 分配抛 `std::bad_alloc` |
| `ACL_SPARSE_STATUS_EXECUTION_FAILED` | 执行失败 | ALG2 preprocess 的 `aclrtMemcpy`（D2H 或 H2D）失败 |
| `ACL_SPARSE_STATUS_INTERNAL_ERROR` | 内部错误 | createDescr 内 `std::vector` 操作抛非 `bad_alloc` 异常 |

> `destroyDescr` / `destroyPlan` 对 nullptr 输入直接返回 `ACL_SPARSE_STATUS_SUCCESS`（幂等语义）。

## 生命周期与调用顺序

7 函数遵循 descriptor/plan 生命周期模式，调用顺序必须为：

```
bufferSize → createDescr → createPlan → [setGlobalUserData] → execute(可重复) → destroyPlan → destroyDescr
```

| 阶段 | 函数 | 说明 |
|------|------|------|
| 1. 查询 workspace | `aclsparseSpMMOp_bufferSize` | 返回 workspace 所需字节数（ALG1/DEFAULT 返回 0，ALG2 返回 > 0） |
| 2. 创建描述符 | `aclsparseSpMMOp_createDescr` | 绑定 matA（CSR pattern 固定）+ opB；ALG2 在此阶段执行 host-side preprocess |
| 3. 创建执行计划 | `aclsparseSpMMOp_createPlan` | 基于 descr 创建 plan；NPU 侧 epilogueLTOBuffer 必须为 NULL |
| 4.（可选）设置 epilogue | `aclsparseSpMMOp_setGlobalUserData` | NPU 侧 identity epilogue，no-op，直接返回 SUCCESS |
| 5. 执行计算 | `aclsparseSpMMOp` | 可重复调用；matB/matC 可跨多次 execute 更换（n/ldb/ldc/order 可变），但 m/k/opB 必须与 descr 一致 |
| 6. 销毁执行计划 | `aclsparseSpMMOp_destroyPlan` | 幂等（nullptr → SUCCESS） |
| 7. 销毁描述符 | `aclsparseSpMMOp_destroyDescr` | 幂等（nullptr → SUCCESS） |

**关键约束**：
- `destroyPlan` 必须先于 `destroyDescr` 调用，否则 plan 内部弱引用的 descr 成为悬垂指针，行为未定义。
- matA 在 createDescr 阶段绑定，plan 生命周期内不可更换 matA（ALG1 允许原地更新 csrValues 指针指向的数据，不重建 descr/plan）。
- opB 在 createDescr 阶段绑定，plan 生命周期固定。
- matB/matC 在 execute 阶段传入，可跨多次 execute 更换。

## 调用示例

```cpp
#include <stdio.h>
#include <stdlib.h>
#include "cann_ops_sparse.h"
#include "acl/acl.h"

// 错误宏：简化返回值检查
#define CHECK_STATUS(call)                                                      \
    do {                                                                        \
        aclsparseStatus_t _st = (call);                                         \
        if (_st != ACL_SPARSE_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "ERROR: %s failed at line %d, status=%d\n",         \
                    #call, __LINE__, static_cast<int>(_st));                     \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CHECK_ALLOC(ptr)                                                        \
    do {                                                                        \
        if ((ptr) == nullptr) {                                                 \
            fprintf(stderr, "ERROR: allocation failed at line %d\n", __LINE__); \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CHECK_ACL_RT(call)                                                      \
    do {                                                                        \
        aclError _st = (call);                                                  \
        if (_st != ACL_SUCCESS) {                                               \
            fprintf(stderr, "ERROR: %s failed at line %d, status=%d\n",         \
                    #call, __LINE__, static_cast<int>(_st));                     \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main() {
    aclrtStream stream = nullptr;  // 此处省略 stream 创建
    int64_t rows = 4, cols = 4, nnz = 5;  // A: m×k = 4×4
    int64_t n = 6;                       // C: m×n = 4×6

    // ...此处省略设备内存分配...
    int32_t *csrRowOffsets = nullptr;  // Device pointer
    int32_t *csrColInd = nullptr;       // Device pointer
    float *csrValues = nullptr;         // Device pointer
    float *matB = nullptr, *matC = nullptr;  // Device pointers

    // 1. 创建 handle
    aclsparseHandle_t handle = nullptr;
    CHECK_STATUS(aclsparseCreate(&handle));
    CHECK_STATUS(aclsparseSetStream(handle, stream));

    // 2. 创建 CSR 稀疏矩阵描述符 matA
    aclsparseConstSpMatDescr_t matA;
    CHECK_STATUS(aclsparseCreateConstCsr(&matA, rows, cols, nnz,
        csrRowOffsets, csrColInd, csrValues,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT));

    // 3. 创建稠密矩阵描述符（matB: k×n, matC: m×n）
    aclsparseConstDnMatDescr_t matB;
    aclsparseDnMatDescr_t matC;
    CHECK_STATUS(aclsparseCreateConstDnMat(&matB, cols, n, n, matB, ACL_FLOAT, ACL_SPARSE_ORDER_ROW));
    CHECK_STATUS(aclsparseCreateDnMat(&matC, rows, n, n, matC, ACL_FLOAT, ACL_SPARSE_ORDER_ROW));

    // 4. 获取 workspace 大小
    size_t bufferSize = 0;
    CHECK_STATUS(aclsparseSpMMOp_bufferSize(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, matA, matB, matC, ACL_FLOAT,
        ACL_SPARSE_SPMMOP_ALG_DEFAULT, &bufferSize));

    // 5. 分配 workspace（ALG1 时 bufferSize=0，可传 nullptr）
    void *buffer = nullptr;
    if (bufferSize > 0) {
        CHECK_ACL_RT(aclrtMalloc(&buffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // 6. 创建描述符（ALG2 时执行 host-side 预处理：同步排序 + bin_edge 计算）
    aclsparseSpMMOpDescr_t descr = nullptr;
    CHECK_STATUS(aclsparseSpMMOp_createDescr(handle, &descr,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA, matB, matC, ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG_DEFAULT, buffer));

    // 7. 创建执行计划（identity epilogue: NULL, 0）
    aclsparseSpMMOpPlan_t plan = nullptr;
    CHECK_STATUS(aclsparseSpMMOp_createPlan(handle, descr, &plan, nullptr, 0));

    // 8. 执行（C = alpha * A * B + beta * C）
    float alpha = 1.0f, beta = 0.0f;
    CHECK_STATUS(aclsparseSpMMOp(handle, plan, &alpha, &beta, matB, matC));

    // 9. 同步
    CHECK_ACL_RT(aclrtSynchronizeStream(stream));

    // 10. 清理（按创建顺序逆序销毁）
    aclsparseSpMMOp_destroyPlan(plan);
    aclsparseSpMMOp_destroyDescr(descr);
    if (buffer) {
        aclrtFree(buffer);
    }
    aclsparseDestroyDnMat(matC);
    aclsparseDestroyDnMat(matB);
    aclsparseDestroySpMat(matA);
    aclsparseDestroy(handle);

    return 0;
}
```
