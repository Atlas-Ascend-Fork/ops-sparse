# aclsparseSpSM

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- **算子功能**：aclsparseSpSM（Sparse Triangular Solve with Multiple Right-Hand Sides）用于求解稀疏三角线性方程组 `op(A)·C = α·op(B)`。其中 A 为 CSR、CSC 或 COO 格式的稀疏三角方阵（m×m），B 为稠密右端项矩阵，C 为稠密解矩阵且可与 B 共用 Device 地址，α 为标量系数。该算子对标 cuSPARSE Generic API 中的 `cusparseSpSM`，采用 BufferSize、Analysis、Solve 三阶段模型，并支持 Analysis 后更新矩阵数值。
- **目标平台**：<term>Ascend 950PR/Ascend 950DT</term>（arch35 / DAV_3510）、Atlas A2/A3 训练与推理系列（arch22 / DAV_2201）。
- **编程模型**：Ascend C Vector API。格式规范化、数值更新与三角求解在 NPU 上执行；Analysis/Update 的同步及 Host 回读范围见对应接口说明。

### 数学原理

#### 问题定义

求解稀疏三角线性方程组：

```
op(A) · C = α · op(B)
```

| 符号 | 含义 | 维度/类型 |
|------|------|-----------|
| A | CSR、CSC 或 COO 稀疏三角方阵 | m×m，FP32 或 complex64 |
| B | 稠密右端项矩阵 | `op(B)` 为 m×nrhs，FP32 或 complex64 |
| C | 稠密解矩阵（可与 B 共用 Device 地址） | m×nrhs，FP32 或 complex64 |
| α | 标量系数 | 与 computeType 一致，Host 或 Device pointer mode |
| op(A) / op(B) | 原矩阵、转置或共轭转置 | N / T / H |
| fillMode | A 参与计算的三角部分 | LOWER / UPPER |
| diagType | 对角线类型 | UNIT（使用隐式 1，忽略存储值）/ NON_UNIT（实际值） |

A 与 B 的转置独立处理。对实数输入，H 与 T 等价；对 complex64 输入，H 同时执行转置和共轭。B/C 可分别使用 ROW 或 COL 布局。

#### 行更新通式

对每一行 i，沿求解方向遍历，已解行构成依赖集 `deps(i)`：

```
             α·op(B)[i, :] − Σ_{j∈deps(i)} op(A)[i,j] · C[j, :]
C[i, :] = ─────────────────────────────────────────────────────  （NON_UNIT）
                                  op(A)[i,i]

C[i, :] = α·op(B)[i, :] − Σ_{j∈deps(i)} op(A)[i,j] · C[j, :]    （UNIT）
```

关键点：

- **除法作用域**：`/op(A)[i,i]` 作用于整个分子（α·op(B)[i,:] − Σ ...），非仅求和项。
- **UNIT 是特例**：A[i,i]≡1.0 时除法省略，设计以 NON_UNIT 通式为基础、UNIT 作 fast path。
- **重复坐标**：按 `(row, col, originalSlot)` 稳定排序并按固定顺序归并，相同设备和输入产生确定结果。

#### 四种 (opA, fillMode) 依赖模式

deps(i) 由 (opA, fillMode) 决定：

| opA | fillMode | 代入方式 | 遍历方向 | deps(i) |
|-----|----------|---------|---------|---------|
| N | LOWER | 前向 | i=0→m−1 | {j<i \| A[i,j]≠0} |
| N | UPPER | 后向 | i=m−1→0 | {j>i \| A[i,j]≠0} |
| T/H | LOWER | 后向 | i=m−1→0 | {j>i \| op(A)[i,j]≠0} |
| T/H | UPPER | 前向 | i=0→m−1 | {j<i \| op(A)[i,j]≠0} |

Analysis 在 NPU 上把 CSR、CSC 或 COO 规范化为 base-0 的 `op(A)` 有序表示，保存数值更新和后续求解所需的计划。不同平台的回读范围见 `aclsparseSpSMAnalysis`。

#### 除法与边界处理

- **NON_UNIT 除法**：缺失或合计为零的对角线在 Analysis 或 Update 时返回错误；Atlas A2/A3 返回 `ACL_SPARSE_STATUS_INVALID_VALUE`，Ascend 950PR/DT 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **UNIT diag**：省略除法（A[i,i]≡1.0）。
- **奇异矩阵检测**：Analysis 阶段检测对角线零元（显式零或缺失对角线项），若检测到零对角元则返回错误码（对齐 cuSPARSE zeroPivot 机制）。
- **数值稳定性**：FP32 和 complex64 均保存中间值的高、低位，使用 FMA 残差、补偿求和和除法修正降低误差传播。输出类型仍为 FP32/complex64；真解超出可表示范围时可能输出无穷。
- **零维输入**：Atlas A2/A3 接受满足其余接口约束的 m=0 或 nrhs=0 输入，Solve 不执行数值计算；Ascend 950PR/DT 返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- **异步执行**：Solve 异步提交，调用方读取输出或释放输入/workspace 前必须同步 handle stream。
- **in-place 正确性**：B/C 共用同一 Device 地址时先保存 B，再执行求解和布局转换。

## 接口说明

aclsparseSpSM 使用 SpSM 描述符在 BufferSize、Analysis、Solve 和 UpdateMatrix 阶段间保存计划状态。

### aclsparseSpSMCreateDescr

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSMCreateDescr(aclsparseSpSMDescr_t *spsmDescr);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| spsmDescr | 输出 | aclsparseSpSMDescr_t* | 接收新建 SpSM 描述符的 Host 指针；不得用它覆盖尚未销毁的描述符 |

#### 约束说明

- spsmDescr 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- 该描述符在 BufferSize、Analysis、Solve 和 UpdateMatrix 阶段间共享，须在调用这些接口前创建。

### aclsparseSpSMDestroyDescr

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSMDestroyDescr(aclsparseSpSMDescr_t spsmDescr);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| spsmDescr | 输入 | aclsparseSpSMDescr_t | 待销毁的 SpSM 描述符句柄，Host 内存 |

#### 约束说明

- spsmDescr 为 nullptr 时，Atlas A2/A3 返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；Ascend 950PR/DT 返回 `ACL_SPARSE_STATUS_SUCCESS`。
- 销毁不隐式等待异步 Solve；调用方须先保证相关 stream 工作完成。
- 销毁后不应再使用该句柄，重复销毁返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。

### aclsparseSpSMBufferSize

查询 SpSM 所需 workspace 大小（字节）。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSMBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr, size_t *bufferSize);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 上下文，携带 device、stream 和 pointer mode |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，支持 N/T/H |
| opB | 输入 | aclsparseOperation_t | 稠密矩阵 B 的操作类型，支持 N/T/H |
| alpha | 输入 | const void* | 与 computeType 一致的标量，支持 Host/Device pointer mode |
| matA | 输入 | aclsparseConstSpMatDescr_t | CSR/CSC/COO 稀疏三角方阵描述符 |
| matB | 输入 | aclsparseConstDnMatDescr_t | 稠密右端项描述符，op 后形状与 A 匹配 |
| matC | 输出 | aclsparseDnMatDescr_t | 稠密解矩阵描述符，允许与 B 共用 Device 地址 |
| computeType | 输入 | aclDataType | `ACL_FLOAT` 或 `ACL_COMPLEX64` |
| alg | 输入 | aclsparseSpSMAlg_t | 算法类型，仅支持 `ACL_SPARSE_SPSM_ALG_DEFAULT`，Host 内存 |
| spsmDescr | 输入/输出 | aclsparseSpSMDescr_t | 有效 SpSM 描述符；Ascend 950PR/DT 在此保存参数签名和 workspace 计划，Atlas A2/A3 在 Analysis 时绑定计划 |
| bufferSize | 输出 | size_t* | 输出所需 workspace 大小（字节），Host 内存 |

Atlas A2/A3 的 BufferSize 依赖当前设备核数，须在后续 Analysis 使用的设备上查询；失败时不修改 `*bufferSize`。B/C 的 values 可以延迟绑定。

### aclsparseSpSMAnalysis

SpSM 分析阶段：在 NPU 上完成格式规范化、索引校验、对角检查、依赖分析和 Update 槽位映射，绑定后续 Solve 使用的 workspace。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSMAnalysis(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr, void *buffer);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | 携带当前 device、stream 和 pointer mode；Ascend 950PR/DT 须与 BufferSize 阶段相同 |
| opA / opB | 输入 | aclsparseOperation_t | 支持 N/T/H；Ascend 950PR/DT 须与 BufferSize 阶段一致，Atlas A2/A3 在 Analysis 时绑定 |
| alpha | 输入 | const void* | 地址位置须符合 pointer mode；Ascend 950PR/DT 的地址和 pointer mode 须与 BufferSize 阶段一致；Atlas A2/A3 不读取其数值 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 有效稀疏矩阵；Ascend 950PR/DT 须与 BufferSize 阶段一致；Atlas A2/A3 在本阶段绑定格式、结构、数组地址和属性 |
| matB / matC | 输入/输出 | 稠密矩阵描述符 | 形状、布局和 ld 须满足公式；Ascend 950PR/DT 的描述符须与 BufferSize 阶段一致；Analysis 不读取 B/C 数值 |
| computeType | 输入 | aclDataType | `ACL_FLOAT` 或 `ACL_COMPLEX64`，须与矩阵一致 |
| alg | 输入 | aclsparseSpSMAlg_t | 算法类型，仅支持 `ACL_SPARSE_SPSM_ALG_DEFAULT`，Host 内存 |
| spsmDescr | 输入/输出 | aclsparseSpSMDescr_t | 保存 analyzed 状态和 NPU 计划 |
| buffer | 输入 | void* | 64 字节对齐且不少于 BufferSize 返回字节数的 Device workspace；Atlas A2/A3 要求显式传入，Ascend 950PR/DT 在为空时使用 handle 当前 workspace |

Analysis 会同步 handle stream。Ascend 950PR/DT 回读固定 64 字节校验摘要；Atlas A2/A3 回读每核校验状态、规范化行偏移及少量调度元数据，在检查纯对角结构时还会回读对应列索引，数值求解仍在 NPU 上执行。

B/C 的 values 可在创建描述符后通过公共 setter 延迟绑定，但首次非零维 Solve 前必须恢复有效 Device 地址。Atlas A2/A3 在 Analysis 时绑定非空 B/C 地址；当时为空的地址由首次成功 Solve 绑定。Atlas A2/A3 的 Host 参数校验失败不提交工作并保留旧计划；Device 数据校验或回读失败会使计划失效，须修正数据后重新 Analysis。

### aclsparseSpSM

SpSM 求解阶段：异步执行 `op(A)·C = α·op(B)`，复用 Analysis 生成的执行计划。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSM(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | 须与已分析计划绑定的 handle、device 和 stream 一致 |
| opA / opB | 输入 | aclsparseOperation_t | 须与已分析计划一致，支持 N/T/H |
| alpha | 输入 | const void* | pointer mode 须与已分析计划一致；HOST 值在本次调用时复制，DEVICE 值在 kernel 执行时读取。Ascend 950PR/DT 还要求地址与计划一致；Atlas A2/A3 可替换符合位置与生命周期约束的 alpha 地址 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏结构及描述符身份须与 Analysis 阶段一致 |
| matB | 输入 | aclsparseConstDnMatDescr_t | 稠密 RHS，Device values 在异步工作完成前有效 |
| matC | 输出 | aclsparseDnMatDescr_t | 稠密结果，允许与 B 共用 Device 地址 |
| computeType | 输入 | aclDataType | `ACL_FLOAT` 或 `ACL_COMPLEX64` |
| alg | 输入 | aclsparseSpSMAlg_t | 算法类型，仅支持 `ACL_SPARSE_SPSM_ALG_DEFAULT`，Host 内存 |
| spsmDescr | 输入 | aclsparseSpSMDescr_t | SpSM 描述符，须已完成 Analysis 阶段，Host 内存 |

### aclsparseSpSMUpdateMatrix

Analysis 后更新 A 的数值而不改变稀疏 pattern，成功后可继续调用 Solve。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSMUpdateMatrix(
    aclsparseHandle_t handle, aclsparseSpSMDescr_t spsmDescr,
    const void *newValues, aclsparseSpSMUpdate_t updatePart);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | 须与已分析计划绑定的 handle、device 和 stream 一致 |
| spsmDescr | 输入/输出 | aclsparseSpSMDescr_t | 须已成功完成 Analysis |
| newValues | 输入 | const void* | 与 Analysis 的 computeType 一致的 Device 数组；GENERAL 为 nnz 个原始存储顺序值。DIAGONAL：Atlas A2/A3 同样要求完整 nnz 数组；Ascend 950PR/DT 要求 m 个按原 A 行号排列的值 |
| updatePart | 输入 | aclsparseSpSMUpdate_t | `ACL_SPARSE_SPSM_UPDATE_GENERAL` 或 `ACL_SPARSE_SPSM_UPDATE_DIAGONAL` |

#### 约束说明

- Update 不改变稀疏结构或 matA 的值数组；A 的索引须保持与 Analysis 一致。GENERAL 按原始存储顺序刷新所有值，后续 GENERAL 会覆盖此前的 DIAGONAL 更新。opA=H 时仍传 A 本身的值。
- Atlas A2/A3 的 DIAGONAL 仅采用完整 newValues 数组中 `row == col` 的条目，其余条目保留计划当前数值。输入须为当前设备上 4 字节对齐的内存，且不得与 workspace 重叠。
- Atlas A2/A3 每次 Update 都同步 stream 并回读 Device 校验状态。数值校验失败返回 `ACL_SPARSE_STATUS_INVALID_VALUE` 并使计划失效；后续 Solve 返回 `ACL_SPARSE_STATUS_NOT_INITIALIZED`，须重新 Analysis。Host 参数校验失败保留旧计划且不提交工作。
- Ascend 950PR/DT 的 NON_UNIT Update 先校验候选对角值并同步回读摘要，校验失败保留旧计划。UNIT 忽略对角值，DIAGONAL 为成功无操作。
- newValues 及绑定的 A 索引和 workspace 须保持有效直到相关 stream 工作完成。

### 矩阵属性接口

SpSM 通过稀疏矩阵描述符的属性接口 `aclsparseSpMatSetAttribute` 设置三角矩阵的 fillMode 与 diagType，须在调用 BufferSize/Analysis 前设置。

```cpp
aclsparseStatus_t aclsparseSpMatSetAttribute(aclsparseSpMatDescr_t spMatDescr,
                                             aclsparseSpMatAttribute_t attribute,
                                             const void *data, size_t dataSize);
aclsparseStatus_t aclsparseSpMatGetAttribute(aclsparseConstSpMatDescr_t spMatDescr,
                                             aclsparseSpMatAttribute_t attribute,
                                             void *data, size_t dataSize);
```

- **attribute=`ACL_SPARSE_SPMAT_FILL_MODE`**：data 指向 `aclsparseFillMode_t`，dataSize 须为 `sizeof(aclsparseFillMode_t)`，值为 `ACL_SPARSE_FILL_MODE_LOWER`（下三角）或 `ACL_SPARSE_FILL_MODE_UPPER`（上三角）。
- **attribute=`ACL_SPARSE_SPMAT_DIAG_TYPE`**：data 指向 `aclsparseDiagType_t`，dataSize 须为 `sizeof(aclsparseDiagType_t)`，值为 `ACL_SPARSE_DIAG_TYPE_UNIT`（使用隐式对角线 1，忽略存储值）或 `ACL_SPARSE_DIAG_TYPE_NON_UNIT`（实际对角线值须由稀疏矩阵数据提供）。

### 返回码

| 返回码 | 说明 |
|--------|------|
| `ACL_SPARSE_STATUS_SUCCESS` | 操作成功；Solve 仅表示工作已提交 |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | 空描述符/alpha、非法枚举或索引、维度/ld/阶段不匹配、错误设备、地址无效或 workspace 重叠 |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 不支持的计算类型或索引宽度；Ascend 950PR/DT 还用于不支持的算法、规模及 NON_UNIT 缺失/合计为零的对角线 |
| `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED` | matA 不是 CSR、CSC 或 COO |
| `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` | Ascend 950PR/DT 的 workspace 计算溢出或 handle workspace 容量不足；Atlas A2/A3 的 workspace 布局或参数无效返回 `INVALID_VALUE` |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为 nullptr |
| `ACL_SPARSE_STATUS_EXECUTION_FAILED` | Ascend 950PR/DT 的 stream 同步或摘要回读失败 |
| `ACL_SPARSE_STATUS_INTERNAL_ERROR` | Atlas A2/A3 的 Analysis/Update 状态回读、结构/调度处理或 HOST alpha 复制失败 |
| `ACL_SPARSE_STATUS_NOT_INITIALIZED` | Atlas A2/A3 在未完成 Analysis 或计划失效后调用 Solve/Update；或当前设备不可用 |
| `ACL_SPARSE_STATUS_ALLOC_FAILED` | 描述符内存分配失败 |

## 支持规格

| 规格项 | 支持值 | 说明 |
|--------|--------|------|
| 数据类型（computeType） | `ACL_FLOAT`、`ACL_COMPLEX64` | matA/matB/matC 值类型与 computeType 一致 |
| 稀疏格式 | CSR、CSC、COO | 支持未排序索引和重复坐标 |
| opA / opB | N、T、H | 实数 H 等价于 T，复数 H 执行共轭转置 |
| fillMode | `ACL_SPARSE_FILL_MODE_LOWER`、`ACL_SPARSE_FILL_MODE_UPPER` | 均支持 |
| diagType | `ACL_SPARSE_DIAG_TYPE_UNIT`、`ACL_SPARSE_DIAG_TYPE_NON_UNIT` | UNIT 使用隐式 1 并忽略存储的对角条目；NON_UNIT 对重复对角条目求和，合计须非零 |
| order（B/C 布局） | `ACL_SPARSE_ORDER_ROW`、`ACL_SPARSE_ORDER_COL` | B/C 可独立组合 |
| pointer mode | HOST、DEVICE | alpha 地址类型须与 handle pointer mode 匹配 |
| indexBase | `ACL_SPARSE_INDEX_BASE_ZERO`、`ACL_SPARSE_INDEX_BASE_ONE` | 均支持；ONE 内部归一化为 ZERO |
| 索引类型 | `ACL_SPARSE_INDEX_32I` | 行偏移与列索引类型须均为 32I；64I 不支持 |
| in-place | 支持 | matB.values 与 matC.values 可指向同一 Device 地址，包括不同布局 |
| Update | GENERAL、DIAGONAL | 保持 Analysis 后的 pattern 不变 |
| alg | `ACL_SPARSE_SPSM_ALG_DEFAULT` | 仅默认算法 |
| 矩阵规模 | Atlas A2/A3：`0 ≤ m, nrhs ≤ INT32_MAX`；Ascend 950PR/DT：`0 < m < INT32_MAX`、`0 < nrhs ≤ INT32_MAX` | `nnz + base ≤ INT32_MAX`；m=0 时 nnz 须为 0；实际规模还受地址步长、字节数溢出检查和可用 workspace 容量约束 |

## 约束说明

- **描述符与类型**：A 必须为方阵，A/B/C/computeType 使用一致的 FP32 或 complex64；ptr/idx 为 I32，base 为 ZERO 或 ONE。
- **操作与布局**：opA/opB 支持 N/T/H，B/C 可分别采用 ROW/COL；`op(A)`、`op(B)` 和 C 的形状必须满足公式。
- **零维输入**：Atlas A2/A3 接受 m（matA.rows）或 nrhs（matC.cols）为 0，仍须完成有效描述符、alpha、workspace 与阶段校验；m=0 时 nnz 须为 0，Solve 不提交数值计算。Ascend 950PR/DT 拒绝零维输入。
- **leading dimension 约束**：matB/matC 的 leading dimension（ldb/ldc）须满足布局约束——行主序（`ACL_SPARSE_ORDER_ROW`）`ld >= cols`，列主序（`ACL_SPARSE_ORDER_COL`）`ld >= rows`，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- **奇异矩阵检测**：diagType=NON_UNIT 时检查缺失或合计为零的对角线，Atlas A2/A3 返回 `ACL_SPARSE_STATUS_INVALID_VALUE`，Ascend 950PR/DT 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。UNIT 使用隐式对角线 1。
- **矩阵规模上限**：见支持规格。Atlas A2/A3 的尺寸或布局超限返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；Ascend 950PR/DT 的尺寸超限返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **阶段一致性**：同一计划绑定 handle、device、stream、pointer mode、三个矩阵描述符身份、A 的数组地址和结构属性；Ascend 950PR/DT 还绑定 alpha 地址。修改绑定项后须重新查询 workspace 并 Analysis；Atlas A2/A3 的 BufferSize 仅查询大小，Analysis 绑定计划。
- **空 values**：公共 CreateDnMat 要求非空 values。BufferSize/Analysis 不读取 B/C 数值；可在描述符创建后通过公共 setter 暂时置空，并在首次非零维 Solve 前恢复有效地址。Atlas A2/A3 已绑定的非空 B/C 地址不可替换，须重新 Analysis。
- **buffer 生命周期**：Atlas A2/A3 要求显式传入非空 external buffer；Ascend 950PR/DT 在参数为空时使用 handle 当前 workspace。workspace 须为当前设备上 64 字节对齐的内存，容量不少于 BufferSize 返回值，并与 A/B/C、Device alpha、Update 输入互不重叠，直到 stream 完成为止。
- **异步执行**：Solve 不同步、不回读。调用方须在读取 C、释放 A/B/C/workspace 或销毁描述符前同步 handle stream。
- **unsorted indices**：支持 CSR/CSC/COO 的未排序及重复坐标；所有存储索引仍须合法。
- **in-place 布局**：任意 orderB/orderC 组合均支持 B/C 共用同一 Device 地址，求解前会保存 B。Atlas A2/A3 拒绝 B/C 地址不同但部分重叠，以及 C 与 A 数组重叠。
- **地址检查边界**：原始 `void*` 不能证明逻辑 dtype、子数组长度或分配代次，调用方仍须保证真实类型、容量和生命周期。Atlas A2/A3 要求 A 索引/值、非零维 B/C 值及 Device alpha 为当前设备上 4 字节对齐的内存；不检测已销毁的 handle 或矩阵描述符。
- **并发**：多个独立计划可在不同 stream 上执行；同一计划不允许并发调用。
- **属性设置时机**：fillMode 与 diagType 须在调用 BufferSize/Analysis 前通过 `aclsparseSpMatSetAttribute` 设置（attribute 分别取 `ACL_SPARSE_SPMAT_FILL_MODE` 与 `ACL_SPARSE_SPMAT_DIAG_TYPE`）。

## 调用说明

调用流程为三阶段法（Generic API Descriptor 模式）：

1. **BufferSize**：查询所需 workspace 大小，分配 Device 内存。
2. **Analysis**：在 NPU 上规范化格式、校验索引和对角线、准备求解及 Update 计划，并绑定 active buffer。
3. **Solve / Update**：异步执行三角求解；如只更新数值，可调用 UpdateMatrix 后复用计划继续 Solve。

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

// 辅助：分配 Device 内存并拷贝 Host 数据
static void* AllocAndCopyDevice(const void *hostPtr, size_t sizeBytes)
{
    void *dPtr = nullptr;
    if (aclrtMalloc(&dPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return nullptr;
    }
    if (hostPtr != nullptr && sizeBytes > 0) {
        if (aclrtMemcpy(dPtr, sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            aclrtFree(dPtr);
            return nullptr;
        }
    }
    return dPtr;
}

// 所有返回路径都先等待异步任务，再销毁描述符并释放设备内存。
struct ExampleResources {
    explicit ExampleResources(aclrtStream value) : stream(value) {}
    ExampleResources(const ExampleResources&) = delete;
    ExampleResources& operator=(const ExampleResources&) = delete;
    ~ExampleResources()
    {
        aclrtSynchronizeStream(stream);
        aclsparseSpSMDestroyDescr(spsmDescr);
        if (matA != nullptr) { aclsparseDestroySpMat(matA); }
        if (matB != nullptr) { aclsparseDestroyDnMat(matB); }
        if (matC != nullptr) { aclsparseDestroyDnMat(matC); }
        for (void *ptr : {dRowPtrA, dColIndA, dValA, dB, dC, dBuffer}) {
            if (ptr != nullptr) { aclrtFree(ptr); }
        }
    }
    aclrtStream stream = nullptr;
    aclsparseSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    aclsparseSpSMDescr_t spsmDescr = nullptr;
    void *dRowPtrA = nullptr, *dColIndA = nullptr, *dValA = nullptr;
    void *dB = nullptr, *dC = nullptr, *dBuffer = nullptr;
};

int aclsparseSpSMTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(rawHandle, aclsparseDestroy);
    ExampleResources resources(stream);

    sparseRet = aclsparseSetStream(static_cast<aclsparseHandle_t>(handlePtr.get()), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 2. 设置 PointerMode
    sparseRet = aclsparseSetPointerMode(static_cast<aclsparseHandle_t>(handlePtr.get()), ACL_SPARSE_POINTER_MODE_HOST);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetPointerMode failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 3. 准备 Host 端 CSR 数据
    //    A (3x3, 下三角, NON_UNIT, nnz=6):     B (3x2):
    //    [2.0  0.0  0.0]      [2.0  4.0]
    //    [1.0  3.0  0.0]      [4.0  8.0]
    //    [4.0  5.0  6.0]      [6.0 12.0]
    //
    //    CSR (含对角线, NON_UNIT):
    //      rowOff = [0, 1, 3, 6]
    //      colInd = [0, 0, 1, 0, 1, 2]
    //      values = [2.0, 1.0, 3.0, 4.0, 5.0, 6.0]
    //
    //    求解 A * X = 1.0 * B (下三角, 前向替换, NON_UNIT):
    //      X[0] = B[0] / A[0,0]                         = [1.0,  2.0]
    //      X[1] = (B[1] - A[1,0]*X[0]) / A[1,1]         = [1.0,  2.0]
    //      X[2] = (B[2] - A[2,0]*X[0] - A[2,1]*X[1]) / A[2,2] = [-0.5, -1.0]
    int64_t m = 3, n = 2;
    int64_t nnzA = 6;
    float hAlpha = 1.0f;

    std::vector<int> hRowPtrA = {0, 1, 3, 6};
    std::vector<int> hColIndA = {0, 0, 1, 0, 1, 2};
    std::vector<float> hValA  = {2.0f, 1.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // B: 行主序 3x2
    int64_t ldb = n, ldc = n;
    aclsparseOrder_t orderB = ACL_SPARSE_ORDER_ROW;
    aclsparseOrder_t orderC = ACL_SPARSE_ORDER_ROW;
    std::vector<float> hB(static_cast<size_t>(m) * n, 0.0f);
    hB[0 * n + 0] = 2.0f; hB[0 * n + 1] = 4.0f;
    hB[1 * n + 0] = 4.0f; hB[1 * n + 1] = 8.0f;
    hB[2 * n + 0] = 6.0f; hB[2 * n + 1] = 12.0f;

    std::vector<float> hC(static_cast<size_t>(m) * n, 0.0f);

    // 4. 拷贝数据到 Device
    auto &dRowPtrA = resources.dRowPtrA;
    dRowPtrA = AllocAndCopyDevice(hRowPtrA.data(), (m + 1) * sizeof(int));
    auto &dColIndA = resources.dColIndA;
    dColIndA = AllocAndCopyDevice(hColIndA.data(), nnzA * sizeof(int));
    auto &dValA = resources.dValA;
    dValA = AllocAndCopyDevice(hValA.data(),    nnzA * sizeof(float));
    auto &dB = resources.dB;
    dB = AllocAndCopyDevice(hB.data(),       static_cast<size_t>(m) * n * sizeof(float));
    auto &dC = resources.dC;
    dC = AllocAndCopyDevice(hC.data(),       static_cast<size_t>(m) * n * sizeof(float));

    if (dRowPtrA == nullptr || dColIndA == nullptr || dValA == nullptr || dB == nullptr || dC == nullptr) {
        LOG_PRINT("分配或拷贝设备数据失败\n");
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    // 5. 创建稀疏矩阵描述符并设置三角属性
    auto &matA = resources.matA;
    sparseRet = aclsparseCreateCsr(&matA, m, m, nnzA, dRowPtrA, dColIndA, dValA,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 5.1 设置三角矩阵属性：fillMode=LOWER, diagType=NON_UNIT
    aclsparseFillMode_t fillMode = ACL_SPARSE_FILL_MODE_LOWER;
    aclsparseDiagType_t diagType = ACL_SPARSE_DIAG_TYPE_NON_UNIT;
    sparseRet = aclsparseSpMatSetAttribute(matA, ACL_SPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode));
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSpMatSetAttribute(FILL_MODE) failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    sparseRet = aclsparseSpMatSetAttribute(matA, ACL_SPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType));
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSpMatSetAttribute(DIAG_TYPE) failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    auto &matB = resources.matB;
    auto &matC = resources.matC;
    sparseRet = aclsparseCreateDnMat(&matB, m, n, ldb, dB, ACL_FLOAT, orderB);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateDnMat B failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    sparseRet = aclsparseCreateDnMat(&matC, m, n, ldc, dC, ACL_FLOAT, orderC);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateDnMat C failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 6. 创建 SpSM 描述符
    auto &spsmDescr = resources.spsmDescr;
    sparseRet = aclsparseSpSMCreateDescr(&spsmDescr);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSpSMCreateDescr failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 7. Step 1 — BufferSize
    size_t bufferSize = 0;
    sparseRet = aclsparseSpSMBufferSize(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, matC, ACL_FLOAT,
        ACL_SPARSE_SPSM_ALG_DEFAULT, spsmDescr, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpSMBufferSize failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes\n", bufferSize);

    auto &dBuffer = resources.dBuffer;
    auto aclRet = aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for buffer failed. ERROR: %d\n", aclRet); return aclRet);

    // 8. Step 2 — Analysis (NPU 规范化、拓扑分层与奇异检测)
    sparseRet = aclsparseSpSMAnalysis(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, matC, ACL_FLOAT,
        ACL_SPARSE_SPSM_ALG_DEFAULT, spsmDescr, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpSMAnalysis failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 9. Step 3 — Solve (异步启动, 算子内部不做 stream 同步)
    sparseRet = aclsparseSpSM(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, matC, ACL_FLOAT,
        ACL_SPARSE_SPSM_ALG_DEFAULT, spsmDescr);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpSM failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 10. 等待异步 Solve 完成（由调用方同步；Analysis / NON_UNIT Update 会同步校验）
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 11. 将结果拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hC.data(), static_cast<size_t>(m) * n * sizeof(float),
                         dC, static_cast<size_t>(m) * n * sizeof(float),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);

    for (int64_t i = 0; i < m; i++) {
        LOG_PRINT("C[%lld] = %.1f, %.1f\n", static_cast<long long>(i), hC[i * n + 0], hC[i * n + 1]);
    }

    // resources 在返回时清理资源，先于 handlePtr 析构。
    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseSpSMTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSpSMTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
bufferSize = <以实际查询结果为准> bytes
C[0] = 1.0, 2.0
C[1] = 1.0, 2.0
C[2] = -0.5, -1.0
```

> in-place 用法：将上述示例中 `dC` 改为复用 `dB`（即 `aclsparseCreateDnMat(&matC, m, n, ldc, dB, ...)`）。实现先保存 B 的值再计算，支持 B/C 独立布局；调用者须确保共用内存的容量覆盖两者，示例的 resources.dC 应保持独立分配或置为 nullptr，确保析构时同一地址只释放一次。

## 参考资源

- 对标接口：cuSPARSE [cusparseSpSM](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespsm)（Generic API，三阶段执行）。
