# SpSV算子

## 产品支持情况

| 产品 | 是否支持 | 实现说明 |
| --- | --- | --- |
| Ascend 950PR / Ascend 950DT | √ | 本页的Ascend 950接口说明 |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ | [arch22](arch22/README.md) |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ | [arch22](arch22/README.md) |
| Atlas 200I/500 A2 推理产品 | × | — |
| Atlas 推理系列产品 | × | — |
| Atlas 训练系列产品 | × | — |

Atlas A2的`ascend910b*`与Atlas A3的`ascend910_93*`选择arch22，支持FP32和complex64。
A2/A3的参数、同步行为、workspace和调用示例见[arch22接口说明](arch22/README.md)，
编译与回归入口见[测试说明](../../test/spsv/README.md)。

arch22默认提交已加载torch_npu的Host队列，纯ACL调用方可通过
`OPS_SPARSE_SPSV_TORCH_QUEUE=0`关闭该互操作。触发条件、GIL行为、告警降级、
提交失败和解释器生命周期约束见[框架队列互操作](arch22/README.md#框架队列互操作)。

下述详细接口与示例针对Ascend 950的arch35实现；不同架构的约束分别维护。

## 算子概述

SpSV（Sparse triangular Solve - Vector）算子实现稀疏三角矩阵的线性方程组求解。核心运算为 op(A)·Y = alpha·X，其中 A 为 m×m 稀疏三角矩阵（CSR/CSC/COO/SLICED_ELL 格式），X 为输入稠密向量，Y 为输出稠密向量。

数学表达式：

```
op(A) · Y = alpha · X
```

其中：
- op(A) 为稀疏方阵 A 的操作：`ACL_SPARSE_OP_NON_TRANSPOSE`（非转置）、`ACL_SPARSE_OP_TRANSPOSE`（转置）、`ACL_SPARSE_OP_CONJUGATE_TRANSPOSE`（共轭转置，float32 下等价于转置）
- alpha 为 float32 标量，指针位置由 `aclsparseSetPointerMode` 控制（Host 或 Device 内存）
- 矩阵属性通过 `aclsparseSpMatSetAttribute` 设置：fillMode（LOWER/UPPER）和 diagType（NON_UNIT/UNIT）

SpSV 采用三阶段执行模型（bufferSize → analysis → solve），对标 cuSPARSE cusparseSpSV Generic API。analysis 阶段在 device 侧完成 Level Scheduling 预处理，solve 阶段按 level 顺序执行前代/回代求解（支持单核/多核自适应执行）。所有接口支持异步执行，host 侧无 D2H 同步操作。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSpSV_createDescr | 创建 SpSV 描述符 |
| aclsparseSpSV_destroyDescr | 销毁 SpSV 描述符 |
| aclsparseSpSV_bufferSize | 查询所需 workspace 大小 |
| aclsparseSpSV_analysis | 预处理（Level Scheduling），构建求解调度信息 |
| aclsparseSpSV_solve | 执行三角求解（前代/回代） |
| aclsparseSpSV_updateMatrix | 更新矩阵值（无需重新 analysis） |
| aclsparseSpMatSetAttribute | 设置稀疏矩阵属性（fillMode / diagType） |
| aclsparseSpMatGetAttribute | 获取稀疏矩阵属性 |

## 算子执行接口

### aclsparseSpSV_createDescr

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSV_createDescr(aclsparseSpSVDescr_t *spsvDescr)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| spsvDescr | 输出 | aclsparseSpSVDescr_t* | SpSV 描述符指针，Host 内存 |

#### 约束说明

- spsvDescr 不可为 nullptr
- 纯 host 内存分配，无 device 操作

---

### aclsparseSpSV_destroyDescr

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSV_destroyDescr(aclsparseSpSVDescr_t spsvDescr)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| spsvDescr | 输入 | aclsparseSpSVDescr_t | 要销毁的 SpSV 描述符，Host 内存 |

#### 约束说明

- spsvDescr 为 nullptr 时直接返回 SUCCESS（幂等语义）

---

### aclsparseSpSV_bufferSize

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSV_bufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr, size_t *bufferSize)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针（float32），Host 或 Device 内存（由 `aclsparseSetPointerMode` 控制）。(bufferSize 仅校验 alpha 非空，不读取其值；为保持与 analysis/solve 签名一致性而保留此参数) |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏三角矩阵 A 的描述符，Host 内存 |
| vecX | 输入 | aclsparseConstDnVecDescr_t | 输入稠密向量 x 的描述符，可为 NULL，Host 内存 |
| vecY | 输入 | aclsparseDnVecDescr_t | 输出稠密向量 y 的描述符，可为 NULL，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，仅支持 ACL_FLOAT，Host 内存 |
| alg | 输入 | aclsparseSpSVAlg_t | 算法类型，仅支持 ACL_SPARSE_SPSV_ALG_DEFAULT，Host 内存 |
| spsvDescr | 输入 | aclsparseSpSVDescr_t | SpSV 描述符，Host 内存 |
| bufferSize | 输出 | size_t* | 所需 workspace 字节数，Host 内存 |

#### 约束说明

- handle 不可为 nullptr
- alpha 不可为 nullptr
- bufferSize 不可为 nullptr
- vecX 和 vecY 可为 NULL（bufferSize 阶段不访问向量数据）
- 纯 host 算术计算，无 device 操作，异步安全
- `matA` 不可为 nullptr
- `spsvDescr` 不可为 nullptr
- `opA` 须为 ACL_SPARSE_OP_NON_TRANSPOSE / ACL_SPARSE_OP_TRANSPOSE / ACL_SPARSE_OP_CONJUGATE_TRANSPOSE 之一

---

### aclsparseSpSV_analysis

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSV_analysis(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr, void *externalBuffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针（float32），Host 或 Device 内存（由 `aclsparseSetPointerMode` 控制） |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏三角矩阵 A 的描述符，Host 内存 |
| vecX | 输入 | aclsparseConstDnVecDescr_t | 输入稠密向量 x 的描述符，可为 NULL，Host 内存 |
| vecY | 输入 | aclsparseDnVecDescr_t | 输出稠密向量 y 的描述符，可为 NULL，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，仅支持 ACL_FLOAT，Host 内存 |
| alg | 输入 | aclsparseSpSVAlg_t | 算法类型，仅支持 ACL_SPARSE_SPSV_ALG_DEFAULT，Host 内存 |
| spsvDescr | 输入/输出 | aclsparseSpSVDescr_t | SpSV 描述符，Host 内存 |
| externalBuffer | 输入 | void* | workspace 缓冲区（由 bufferSize 查询大小后分配），Device 内存 |

#### 约束说明

- handle 不可为 nullptr，且须先调用 `aclsparseSetStream` 设置 stream
- alpha 不可为 nullptr
- vecX 和 vecY 可为 NULL（analysis 阶段不访问向量数据）
- 当 nnz > 0 时，externalBuffer 不可为 nullptr
- 当 nnz == 0 时，externalBuffer 可为 nullptr（无需 workspace）
- analysis 启动 device kernel 异步执行 Level Scheduling，立即返回
- 当 nnz == 0 时，analysis 不启动 device kernel，直接在 host 侧缓存矩阵属性并返回 SUCCESS
- 对同一矩阵结构（sparsity pattern）只需 analysis 一次；矩阵值变化后通过 updateMatrix 更新，无需重新 analysis
- `matA` 不可为 nullptr
- `spsvDescr` 不可为 nullptr
- `opA` 须为 ACL_SPARSE_OP_NON_TRANSPOSE / ACL_SPARSE_OP_TRANSPOSE / ACL_SPARSE_OP_CONJUGATE_TRANSPOSE 之一

---

### aclsparseSpSV_solve

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSV_solve(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针（float32），Host 或 Device 内存（由 `aclsparseSetPointerMode` 控制） |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏三角矩阵 A 的描述符，Host 内存 |
| vecX | 输入 | aclsparseConstDnVecDescr_t | 输入稠密向量 x 的描述符，Host 内存 |
| vecY | 输入/输出 | aclsparseDnVecDescr_t | 输出稠密向量 y 的描述符，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，仅支持 ACL_FLOAT，Host 内存 |
| alg | 输入 | aclsparseSpSVAlg_t | 算法类型，仅支持 ACL_SPARSE_SPSV_ALG_DEFAULT，Host 内存 |
| spsvDescr | 输入 | aclsparseSpSVDescr_t | SpSV 描述符（须已完成 analysis），Host 内存 |

#### 约束说明

- handle 不可为 nullptr，且须先调用 `aclsparseSetStream` 设置 stream
- alpha、vecX、vecY 不可为 nullptr
- 必须先调用 `aclsparseSpSV_analysis` 完成预处理，否则返回 ACL_SPARSE_STATUS_INVALID_VALUE
- 支持 in-place 操作：vecX 和 vecY 可指向同一设备内存
- solve 提供确定性（bit-wise）结果：相同输入产生完全相同的输出
- solve 启动 device kernel 异步执行，立即返回
- 当 nnz == 0 时，solve 不执行常规求解 kernel，而是直接写入 vecY：
  - `ACL_SPARSE_DIAG_TYPE_UNIT`：Y = alpha * X（scale copy）
  - `ACL_SPARSE_DIAG_TYPE_NON_UNIT`：Y = 全零（三角矩阵无非零元时解为零向量）
- `matA` 不可为 nullptr
- `spsvDescr` 不可为 nullptr
- `opA` 须为 ACL_SPARSE_OP_NON_TRANSPOSE / ACL_SPARSE_OP_TRANSPOSE / ACL_SPARSE_OP_CONJUGATE_TRANSPOSE 之一

---

### aclsparseSpSV_updateMatrix

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSV_updateMatrix(
    aclsparseHandle_t handle, aclsparseSpSVDescr_t spsvDescr,
    void *newValues, aclsparseSpSVUpdate_t updatePart)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，Host 内存 |
| spsvDescr | 输入 | aclsparseSpSVDescr_t | SpSV 描述符（须已完成 analysis），Host 内存 |
| newValues | 输入 | void* | 新矩阵值指针，Device 内存。GENERAL 模式：nnz 个元素；DIAGONAL 模式：m 个元素 |
| updatePart | 输入 | aclsparseSpSVUpdate_t | 更新策略，Host 内存 |

#### 约束说明

- handle 不可为 nullptr
- spsvDescr 不可为 nullptr，且须已完成 analysis
- newValues 不可为 nullptr
- updatePart 支持：
  - `ACL_SPARSE_SPSV_UPDATE_GENERAL`：更新全部非零值（nnz 个元素），矩阵结构不变
  - `ACL_SPARSE_SPSV_UPDATE_DIAGONAL`：仅更新对角线值（m 个元素）
- 更新后无需重新 analysis（Level Schedule 仅依赖矩阵结构，与值无关）
- CSR + NON_TRANSPOSE 路径：仅更新 descr 内部的 values 指针引用（无 device kernel 启动），solve 直接从用户 values 数组读取新值；其余路径（COO/SLICED_ELL、转置操作）通过 kernel 将新值按 perm 映射拷贝至 workspace 内的 CSR/转置副本
- 异步执行，立即返回
- 当 analysis 缓存的 cachedNnz 为 0 时，updateMatrix 不启动 kernel，直接返回 ACL_SPARSE_STATUS_SUCCESS

---

### aclsparseSpMatSetAttribute

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpMatSetAttribute(
    aclsparseSpMatDescr_t spMatDescr, aclsparseSpMatAttribute_t attribute,
    const void *data, size_t dataSize)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| spMatDescr | 输入 | aclsparseSpMatDescr_t | 稀疏矩阵描述符，Host 内存 |
| attribute | 输入 | aclsparseSpMatAttribute_t | 属性类型：`ACL_SPARSE_SPMAT_FILL_MODE` 或 `ACL_SPARSE_SPMAT_DIAG_TYPE`，Host 内存 |
| data | 输入 | const void* | 属性值指针，Host 内存 |
| dataSize | 输入 | size_t | 属性值大小（字节），须与属性类型匹配 |

#### 约束说明

- spMatDescr 不可为 nullptr
- data 不可为 nullptr
- attribute 为 `ACL_SPARSE_SPMAT_FILL_MODE` 时：data 指向 `aclsparseFillMode_t`，dataSize 须为 `sizeof(aclsparseFillMode_t)`，值为 `ACL_SPARSE_FILL_MODE_LOWER` 或 `ACL_SPARSE_FILL_MODE_UPPER`
- attribute 为 `ACL_SPARSE_SPMAT_DIAG_TYPE` 时：data 指向 `aclsparseDiagType_t`，dataSize 须为 `sizeof(aclsparseDiagType_t)`，值为 `ACL_SPARSE_DIAG_TYPE_NON_UNIT` 或 `ACL_SPARSE_DIAG_TYPE_UNIT`
- 纯 host 字段写入，无 device 操作
- `data` 指针须满足对应枚举类型的自然对齐要求（`alignof(aclsparseFillMode_t)` 或 `alignof(aclsparseDiagType_t)`），否则返回 INVALID_VALUE
- SpSV 使用前建议显式调用 aclsparseSpMatSetAttribute 设置 fillMode 和 diagType（默认为 LOWER / NON_UNIT，如不需要更改可直接使用默认值）

---

### aclsparseSpMatGetAttribute

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpMatGetAttribute(
    aclsparseConstSpMatDescr_t spMatDescr, aclsparseSpMatAttribute_t attribute,
    void *data, size_t dataSize)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| spMatDescr | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵描述符，Host 内存 |
| attribute | 输入 | aclsparseSpMatAttribute_t | 属性类型：`ACL_SPARSE_SPMAT_FILL_MODE` 或 `ACL_SPARSE_SPMAT_DIAG_TYPE`，Host 内存 |
| data | 输出 | void* | 属性值输出指针，Host 内存 |
| dataSize | 输入 | size_t | 属性值大小（字节），须与属性类型匹配 |

#### 约束说明

- spMatDescr 不可为 nullptr
- data 不可为 nullptr
- dataSize 须与属性类型匹配（同 SetAttribute）
- 纯 host 字段读取，无 device 操作
- `data` 指针须满足对应枚举类型的自然对齐要求（`alignof(aclsparseFillMode_t)` 或 `alignof(aclsparseDiagType_t)`），否则返回 INVALID_VALUE

## 支持数据类型

| matA 值类型 | vecX 值类型 | vecY 值类型 | computeType | 说明 |
|------------|------------|------------|-------------|------|
| ACL_FLOAT | ACL_FLOAT | ACL_FLOAT | ACL_FLOAT | 当前唯一支持的类型组合 |

## 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 主路径，直接按行访问 |
| CSC | ✅ | 零拷贝实现：指针重映射（colPtr→rowPtr, rowInd→colInd），同时隐式翻转 fillMode（LOWER↔UPPER）和 opA（NON_TRANSPOSE↔TRANSPOSE/CONJ），转为等效 CSR 路径处理 |
| COO | ✅ | analysis 阶段在 device 侧构建 CSR 副本，需要额外 workspace |
| SLICED_ELL | ✅ | analysis 阶段在 device 侧构建 CSR 副本，需要额外 workspace |

## 支持的索引类型

| 索引类型 | 支持 | 说明 |
|----------|------|------|
| ACL_SPARSE_INDEX_32I | ✅ | 32-bit 有符号整数索引 |
| ACL_SPARSE_INDEX_64I | ✅ | 64-bit 有符号整数索引 |

SpSV 允许 rowOffsets（ptrType）与 colInd（idxType）独立指定索引宽度，但需满足以下支持矩阵：

| rowOffsets 类型 | colInd 类型 | 支持 | 典型场景 |
|:---:|:---:|:---:|------|
| I32 | I32 | ✅ | 标准路径；nnz ≤ INT32_MAX |
| I64 | I32 | ✅ | 混合宽度路径；nnz 可超过 INT32_MAX，colInd 受限于 m ≤ INT32_MAX，I32 即可 |
| I64 | I64 | ✅ | 全 I64 路径；nnz 可超过 INT32_MAX |
| I32 | I64 | ❌ | 不受支持：I32 rowPtr 无法寻址 nnz > INT32_MAX，此组合无实用场景 |

无论使用哪种索引类型组合，矩阵维度 m（rows/cols）始终须 ≤ INT32_MAX（kernel 内部 levelRow/diagPtr 等数组采用 int32_t）。

## 支持的矩阵操作

| 操作 | 枚举值 | 说明 |
|------|--------|------|
| 非转置 | ACL_SPARSE_OP_NON_TRANSPOSE | op(A) = A |
| 转置 | ACL_SPARSE_OP_TRANSPOSE | op(A) = A^T |
| 共轭转置 | ACL_SPARSE_OP_CONJUGATE_TRANSPOSE | op(A) = A^H（float32 下等价于 A^T） |

## 支持的矩阵属性

| fillMode | 枚举值 | 说明 |
|----------|--------|------|
| 下三角 | ACL_SPARSE_FILL_MODE_LOWER | 矩阵非零元素在主对角线及以下 |
| 上三角 | ACL_SPARSE_FILL_MODE_UPPER | 矩阵非零元素在主对角线及以上 |

| diagType | 枚举值 | 说明 |
|----------|--------|------|
| 非单位对角 | ACL_SPARSE_DIAG_TYPE_NON_UNIT | 对角线元素参与计算（需除法） |
| 单位对角 | ACL_SPARSE_DIAG_TYPE_UNIT | 对角线元素视为 1（跳过除法） |

## 约束说明

### 通用约束

- handle 不可为 nullptr，且须先调用 `aclsparseSetStream` 设置 stream
- matA 必须为方阵（rows == cols）
- matA 的索引基址支持 `ACL_SPARSE_INDEX_BASE_ZERO`（C 兼容性）和 `ACL_SPARSE_INDEX_BASE_ONE`（Fortran 兼容性）；1-based 输入时 analysis 阶段在 device 侧构建 0-based workspace CSR 副本，需要额外 workspace
- computeType 仅支持 `ACL_FLOAT`
- alg 仅支持 `ACL_SPARSE_SPSV_ALG_DEFAULT`
- matA 的值类型仅支持 `ACL_FLOAT`
- matA 的 rowOffsets 与 colInd 索引宽度可独立指定，支持组合详见上方"支持的索引类型"章节；仅 (I32, I64) 组合不受支持
- 矩阵维度 m（rows/cols）不得超过 INT32_MAX（2,147,483,647）；nnz 可超过 INT32_MAX，但此时必须使用 I64 rowOffsets（ptrType=ACL_SPARSE_INDEX_64I），colInd 可为 I32 或 I64
- 允许 matA 索引未排序（colInd 无需按行内升序排列）
- 当 m == 0 时，所有调用均合法：bufferSize 返回 0，analysis 和 solve 不启动 device kernel 直接返回 SUCCESS
- 所有接口支持异步执行，host 侧无 D2H 同步操作

### 调用顺序约束

1. 创建 SpSV 描述符：`aclsparseSpSV_createDescr`
2. 设置矩阵属性：`aclsparseSpMatSetAttribute`（fillMode + diagType）
3. 查询 workspace 大小：`aclsparseSpSV_bufferSize`
4. 分配 workspace（Device 内存）
5. 执行预处理：`aclsparseSpSV_analysis`
6. 执行求解：`aclsparseSpSV_solve`（可多次调用）
7. 可选：更新矩阵值：`aclsparseSpSV_updateMatrix`（无需重新 analysis）
8. 销毁描述符：`aclsparseSpSV_destroyDescr`

### SpSVDescr 生命周期

- analysis 完成后，descr 内部缓存以下运行时常量：
  - 矩阵属性缓存（analysis 时从 matA 快照）：cachedM、cachedNnz、cachedFormat、cachedFillMode、cachedDiagType、cachedOpA、cachedIndexType（rowPtr 索引宽度）、cachedColIndType（colInd 索引宽度）、cachedIdxBase（索引基址）
  - 运行时上下文（solve/updateMatrix 时使用）：workspaceBuffer（workspace 指针）、currentValues（values 弱引用，由 updateMatrix 更新）、以及 workspace 内部偏移字段（levelPtrOffset、levelRowOffset、diagPtrOffset、csrRowPtrOffset、csrColIndOffset、csrValuesOffset、permOffset、transValuesOffset、transPermOffset）。这些内部字段用户不应直接访问。

### 维度匹配

- vecX 大小 >= m（矩阵阶数）
- vecY 大小 >= m（矩阵阶数）

### 奇异矩阵

- NON_UNIT 模式下，若对角线元素为 0（或矩阵缺少显式对角线元素），IEEE-754 除零将产生 Inf/NaN
- 与 cuSPARSE 行为一致，不做特殊防御

### Workspace 说明

- **首部 TilingData 区域**：workspace 首 512 字节预留为 `SpsvTilingData` 运行时数据区。analysis kernel 将 `numLevels`（层数）和 `maxLevelWidth`（最大层宽）写入该区域，solve kernel 从中读取（而非从 host 传入的 kernel 参数中读取）
- CSR/CSC 格式（0-based）：workspace 存储 Level Schedule 数据（diagPtr + levelPtr + levelRow + validCount）
  - `validCount`（m × int32_t）：每行的有效依赖元素计数，用于 unsorted CSR 场景下跳过已处理的依赖项，提升求解性能
- COO/SLICED_ELL 格式：workspace 额外存储 CSR 副本（rowPtr + colInd + values + perm），用于格式转换
- 1-based CSR/CSC 格式（INDEX_BASE_ONE）：workspace 额外存储 0-based CSR 副本（rowPtr + colInd + values + perm），用于将 1-based 输入归一化为 0-based 进行下游处理
- 转置操作（TRANSPOSE/CONJUGATE_TRANSPOSE）：workspace 额外存储转置后的 CSR 副本
- workspace 大小由 `aclsparseSpSV_bufferSize` 计算，用户自行分配
- workspace 在 analysis 和 solve 之间须保持有效，不可释放
- **对齐要求**：workspace 首地址须 512B 对齐（满足 GM 基址对齐），workspace 内部各子分配区域按 64B 对齐
- 当 nnz == 0 时，bufferSize 返回 0，workspace 指针可为 NULL（analysis 和 solve 均不使用 workspace）

## 返回值 / 错误码

| 返回值 | 含义 |
|--------|------|
| ACL_SPARSE_STATUS_SUCCESS | 操作成功完成 |
| ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR | handle 为空指针 |
| ACL_SPARSE_STATUS_INVALID_VALUE | 参数无效（nullptr、维度不匹配、analysis 未完成等） |
| ACL_SPARSE_STATUS_NOT_SUPPORTED | 不支持的参数组合（computeType、alg、format、indexType 等） |
| ACL_SPARSE_STATUS_ALLOC_FAILED | 内存分配失败（createDescr） |

## 调用示例

```cpp
#include "acl/acl.h"
#include "cann_ops_sparse.h"

// 初始化 ACL 环境
aclInit(nullptr);
aclrtSetDevice(0);
aclrtStream stream = nullptr;
aclrtCreateStream(&stream);

// 创建 ops-sparse handle
aclsparseHandle_t handle = nullptr;
aclsparseCreate(&handle);
aclsparseSetStream(handle, stream);

// 假设已有 m×m 下三角 CSR 矩阵数据（Device 内存）
// dRowOffsets: int32_t[m+1], dColInd: int32_t[nnz], dValues: float[nnz]
// dX: float[m] 输入向量, dY: float[m] 输出向量
int64_t m = 1024;
int64_t nnz = /* 非零元数 */;

// 1. 创建稀疏矩阵描述符
aclsparseSpMatDescr_t matA = nullptr;
aclsparseCreateCsr(&matA, m, m, nnz,
    dRowOffsets, dColInd, dValues,
    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
    ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);

// 2. 设置矩阵属性（SpSV 必须）
aclsparseFillMode_t fillMode = ACL_SPARSE_FILL_MODE_LOWER;
aclsparseDiagType_t diagType = ACL_SPARSE_DIAG_TYPE_NON_UNIT;
aclsparseSpMatSetAttribute(matA, ACL_SPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode));
aclsparseSpMatSetAttribute(matA, ACL_SPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType));

// 3. 创建稠密向量描述符
aclsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
aclsparseCreateDnVec(&vecX, m, dX, ACL_FLOAT);
aclsparseCreateDnVec(&vecY, m, dY, ACL_FLOAT);

// 4. 创建 SpSV 描述符
aclsparseSpSVDescr_t spsvDescr = nullptr;
aclsparseSpSV_createDescr(&spsvDescr);

// 5. 查询 workspace 大小
float alpha = 1.0f;
aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE;
size_t bufferSize = 0;
aclsparseSpSV_bufferSize(handle, opA, &alpha, matA, vecX, vecY,
    ACL_FLOAT, ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr, &bufferSize);

// 6. 分配 workspace
void *dWorkspace = nullptr;
if (bufferSize > 0) {
    aclrtMalloc(&dWorkspace, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

// 7. 执行 analysis（预处理）
aclsparseSpSV_analysis(handle, opA, &alpha, matA, vecX, vecY,
    ACL_FLOAT, ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr, dWorkspace);

// 8. 执行 solve（可多次调用）
aclsparseSpSV_solve(handle, opA, &alpha, matA, vecX, vecY,
    ACL_FLOAT, ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr);

// 9. 可选：更新矩阵值后再次求解（无需重新 analysis）
// aclsparseSpSV_updateMatrix(handle, spsvDescr, dNewValues, ACL_SPARSE_SPSV_UPDATE_GENERAL);
// aclsparseSpSV_solve(handle, opA, &alpha, matA, vecX, vecY,
//     ACL_FLOAT, ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr);

// 10. 同步并清理
aclrtSynchronizeStream(stream);

aclsparseSpSV_destroyDescr(spsvDescr);
aclsparseDestroyDnVec(vecX);
aclsparseDestroyDnVec(vecY);
aclsparseDestroySpMat(matA);
if (dWorkspace) aclrtFree(dWorkspace);
aclsparseDestroy(handle);
aclrtDestroyStream(stream);
aclrtResetDevice(0);
aclFinalize();
```

## 性能数据参考

以下数据在 Ascend 950PR 上采集，配置为 CSR + LOWER + NON_UNIT + NON_TRANSPOSE + alpha=1.0 + sparsity=0.5，100 次迭代取平均：

| 矩阵规模 (m) | 非零元数 (nnz) | solve 平均耗时 (us) | 有效带宽 (GB/s) |
|:---:|:---:|:---:|:---:|
| 64 | ~1K | 待补充 | - |
| 256 | ~16K | 待补充 | - |
| 1024 | ~262K | 待补充 | - |
| 4096 | ~4.2M | 待补充 | - |
| 8192 | ~16.8M | 待补充 | - |

> 注：当前版本无硬性性能指标，以功能正确性为主。求解阶段支持多核并行执行：当矩阵维度 m 大于 SIMT 线程数（根据 m 选取 64/128/256/2048）且估算平均层宽（≈ m²/nnz）≥ 256 时启动多核，否则回退至单核执行以避免多核间 SyncAll 同步开销。多核 analysis 采用三阶段拆分（串行/并行/收尾），通过同一 stream 排序保证跨 kernel 的 GM 可见性。
>
> 由于 ComputeNthreads 在 m ≤ 256 时返回 m（选取 64/128/256 之一），仅在 m > 256 时 nthreads 升至 2048，因此多核执行的实际有效触发条件为 m > 2048 且估算平均层宽（≈ m²/nnz）≥ 256。

## 支持芯片

| 芯片 | 架构 | 支持情况 |
|------|------|---------|
| Ascend 950PR | arch35 (DAV_3510) | ✅ 支持 |
| Atlas A3 系列 | arch22 (DAV_2201) | ✅ 支持 |
| Atlas A2 系列 | arch22 (DAV_2201) | ✅ 支持 |

## 算法说明

### Level Scheduling（层级调度）

SpSV 的核心并行化方法为 Level Scheduling：

1. **构建依赖 DAG**：从稀疏矩阵结构提取行间依赖关系
2. **拓扑分层**：将行按依赖关系分为若干 level，同一 level 内的行互不依赖
3. **按 level 顺序求解**：level 间串行（保证依赖），level 内并行（线程级）

### 三阶段执行模型

| 阶段 | 接口 | 执行位置 | 说明 |
|------|------|---------|------|
| 查询 | bufferSize | Host | 纯算术计算 workspace 大小 |
| 预处理 | analysis | Device kernel | Level Scheduling + 对角线定位 + 格式转换（COO/SLICED_ELL→CSR） |
| 求解 | solve | Device kernel | 按 level 顺序执行前代/回代 |

### 求解方向决策

求解方向（forward / backward）由 fillMode、opA 和矩阵格式共同决定：

**有效 fillMode 与 opA**（经 CSC 隐式翻转和转置归一化后）：

| 有效 fillMode | 有效 opA | 求解方向 |
|:---:|:---:|:---:|
| LOWER | NON_TRANSPOSE | forward（前代） |
| LOWER | TRANSPOSE/CONJ | backward（回代） |
| UPPER | NON_TRANSPOSE | backward（回代） |
| UPPER | TRANSPOSE/CONJ | forward（前代） |

> CSC 格式采用零拷贝实现：将 colPtr 重映射为 rowPtr、rowInd 重映射为 colInd，同时隐式翻转 fillMode（LOWER↔UPPER）和 opA（NON_TRANSPOSE↔TRANSPOSE/CONJ），使 CSC 矩阵自动转为等效 CSR 路径处理。因此用户无需对 CSC 矩阵做额外操作，直接传入即可。

### 求解优化

- **validCount 优化**：analysis 阶段为每行预计算有效依赖元素计数（validCount），solve 阶段利用该计数跳过已处理的依赖项（按列索引排序的前缀），避免冗余遍历，对 unsorted CSR 场景性能提升显著
- **多核并行**：当 m > nthreads 且估算平均层宽（≈ m²/nnz）≥ 256 时启动多核求解；小矩阵或深层结构（平均层宽 < 256）回退至单核执行，避免多核 SyncAll 同步开销
- **多核 analysis**：多核路径拆分为串行（格式转换 + 顺序层计算）、并行（levelRow scatter + diagPtr/validCount）、收尾（levelPtr 修正）三个阶段，在同一 stream 上依次调度，由 stream 排序保证跨 kernel 的全局内存可见性

### 确定性保证

solve 阶段提供 bit-wise 确定性结果：
- Level Scheduling 是确定性的：给定相同矩阵结构，level 计算结果唯一
- Level 内行的求解顺序不影响结果：同一 level 内的行互不依赖
- 不引入随机化或依赖运行时调度的非确定性因素
