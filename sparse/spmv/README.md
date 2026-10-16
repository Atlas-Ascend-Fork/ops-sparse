# SpMV算子

## 算子概述

SpMV（Sparse Matrix - Dense Vector Multiplication）算子实现稀疏矩阵与稠密向量的乘法运算。核心运算为 y = alpha * op(A) * x + beta * y，其中 A 为 CSR 格式稀疏矩阵，x 和 y 为稠密向量。

数学表达式：

```
y = alpha * op(A) * x + beta * y
```

其中 op(A) 为稀疏矩阵 A 的操作（转置/非转置），x 为输入稠密向量，y 为输入/输出稠密向量。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSpMV | 执行稀疏矩阵-稠密向量乘法 y = alpha * op(A) * x + beta * y |
| aclsparseSpMVGetBufferSize | 查询所需 workspace 大小 |
| aclsparseSpMVPreprocess | 对转置路径构建可复用的确定性 CSR-to-CSC 索引（Ascend 950PR/950DT；Atlas A2/A3 暂未支持） |
| aclsparseCsrSetStrides / GetStrides | 设置/获取 CSR rowOffsets、colInd、values 的元素步长 |
| aclsparseDnVecSetStride / GetStride | 设置/获取稠密向量的元素步长 |

## 算子执行接口

### aclsparseSpMV

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpMV(aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX, const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType, aclsparseSpMVAlg_t alg, void *externalBuffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，类型须与 computeType 匹配。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 A 的描述符，仅支持 CSR 格式，Host 内存 |
| vecX | 输入 | aclsparseConstDnVecDescr_t | 输入稠密向量 x 的描述符，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针，类型须与 computeType 匹配。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| vecY | 输入/输出 | aclsparseDnVecDescr_t | 输入/输出稠密向量 y 的描述符，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，支持 `ACL_FLOAT` 或 `ACL_INT32`，Host 内存 |
| alg | 输入 | aclsparseSpMVAlg_t | 算法类型，随产品不同：Ascend 950PR/950DT 支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`、`ACL_SPARSE_SPMV_CSR_ALG1`、`ACL_SPARSE_SPMV_CSR_ALG2`；Atlas A2/A3 支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`（向量单元实现）与 `ACL_SPARSE_SPMV_CSR_ALG_CUBE`（cube 段求和实现），Host 内存 |
| externalBuffer | 输入 | void* | 工作缓冲区，大小由 `aclsparseSpMVGetBufferSize` 查询；该接口返回 0 时可传 nullptr，返回非 0 时（950 转置且 NNZ>0、A2/A3 cube 算法）必须按该大小分配，Device 内存 |

#### 约束说明

- handle 不可为 nullptr，且须先调用 `aclsparseSetStream` 设置 stream
- matA、vecX、vecY 不可为 nullptr
- matA 仅支持 CSR 格式（`ACL_SPARSE_FORMAT_CSR`）
- matA 的行偏移和列索引类型必须均为 `ACL_SPARSE_INDEX_32I`，且两者类型相同
- matA 的索引基址仅支持 `ACL_SPARSE_INDEX_BASE_ZERO`
- opA 支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置
- Ascend 950PR/950DT 上 alg 支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`、`ACL_SPARSE_SPMV_CSR_ALG1`、`ACL_SPARSE_SPMV_CSR_ALG2`；Atlas A2/A3 上 alg 支持 `ACL_SPARSE_SPMV_ALG_DEFAULT` 与 `ACL_SPARSE_SPMV_CSR_ALG_CUBE`
- `ACL_SPARSE_SPMV_CSR_ALG_CUBE` 额外约束：opA 须为 `ACL_SPARSE_OP_NON_TRANSPOSE`，computeType 与 vecY 值类型均为 `ACL_FLOAT`，matA / vecX 值类型为 `ACL_FLOAT` 或 `ACL_FLOAT16`（fp16 输入时 cube 以 fp32 累加、输出 fp32）；由于归约顺序不同，其结果与默认算法可能存在浮点误差差异，fp16 输入时乘积亦在 fp16 精度下计算
- computeType 仅支持 `ACL_FLOAT` 或 `ACL_INT32`
- vecX 的值类型须与 matA 的值类型一致
- 维度匹配：
  - 非转置模式：vecX 大小 >= A.cols，vecY 大小 >= A.rows
  - 转置模式：vecX 大小 >= A.rows，vecY 大小 >= A.cols
- 支持 M=0、N=0 和 NNZ=0；输出长度为 0 时成功返回，NNZ=0 时结果为 `beta * y`
- `aclsparseCreateCsr` 和 `aclsparseCreateDnVec` 创建的描述符默认元素步长为 1；通过
  `aclsparseCsrSetStrides` 和 `aclsparseDnVecSetStride` 可原生描述正步长非连续 Tensor
- CSR rowOffsets、colInd、values 以及 vecX、vecY 的 stride 均按元素计数且必须大于 0；vecY 按其 stride 原位读写，间隙元素不会被覆盖
- 非连续 stride 当前仅在 Ascend 950PR/950DT 路径生效；Atlas A2/A3 路径仅支持默认步长 1
- 调用方负责保证 CSR 内容合法：rowOffsets 必须从 0 开始、以 NNZ 结束且单调非降，colInd 必须全部位于
  `[0, cols)`。非法 CSR 内容属于未定义行为；SpMV 和 Preprocess 不同步回读 Device 数据，也不承诺返回内容校验错误
- 相同输入、硬件和软件环境下，非转置及稳定预处理后的转置路径使用固定归约顺序，可获得逐比特确定结果
- INT8→INT32 路径对完整的 `alpha * dot(A, x) + beta * y` 做最终 INT32 饱和，保留正负项抵消；
  使用乘法前范围检查和平移后的饱和边界，避免中间 INT64 乘加溢出。连续、stride 和转置路径语义一致。
- 支持的值类型组合：
  - matA=ACL_FLOAT, vecX=ACL_FLOAT, vecY=ACL_FLOAT, computeType=ACL_FLOAT
  - matA=ACL_FLOAT16, vecX=ACL_FLOAT16, vecY=ACL_FLOAT 或 ACL_FLOAT16, computeType=ACL_FLOAT
  - matA=ACL_BF16, vecX=ACL_BF16, vecY=ACL_FLOAT 或 ACL_BF16, computeType=ACL_FLOAT
  - matA=ACL_INT8, vecX=ACL_INT8, vecY=ACL_INT32, computeType=ACL_INT32
  - matA=ACL_INT8, vecX=ACL_INT8, vecY=ACL_FLOAT, computeType=ACL_FLOAT

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 稀疏矩阵 A 支持 CSR 格式 |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

#### 调用示例

具体编译、三阶段调用和自测命令请参考 [SPMV 测试说明](../../test/spmv/README.md)；通用工程构建方式见
[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

非连续 Tensor 在创建普通描述符后设置元素步长，SpMV 三阶段接口保持不变：

```cpp
aclsparseCsrSetStrides(matA, rowOffsetsStride, colIndStride, valuesStride);
aclsparseDnVecSetStride(vecX, xStride);
aclsparseDnVecSetStride(vecY, yStride);
```

未调用 stride 设置接口时保持兼容行为，所有步长均为 1。更新 CSR 指针或 stride 后，转置路径必须重新调用
`aclsparseSpMVPreprocess`。

### aclsparseSpMVGetBufferSize

查询 `aclsparseSpMV` 所需的 workspace 字节数。在 arch22 默认算法的转置模式下，FP16/BF16 输出使用每列 4 字节的 FP32 累加工作区，最后一次转换到输出类型；未传入 `externalBuffer` 时使用句柄工作区。调用方负责按返回字节数分配 Device 内存。
此阶段只检查描述符元数据及参数，不读取 CSR 或 x/y 数据，允许这些数据指针尚未绑定。

- Ascend 950PR/950DT 非转置路径：返回 0。
- Ascend 950PR/950DT 转置路径：返回稳定 CSR-to-CSC 元数据所需空间，包含列偏移、
  行索引和原 CSR value 下标。
- Atlas A2/A3 `ACL_SPARSE_SPMV_ALG_DEFAULT`（向量版）：所有中间量都在 UB 内，返回 0，
  `externalBuffer` 可传 `nullptr`。
- Atlas A2/A3 `ACL_SPARSE_SPMV_CSR_ALG_CUBE`：返回分段求和所需的 workspace 大小（gather 乘积区 +
  块内推测扫描结果区）。调用方须按该大小申请显存，并作为 `aclsparseSpMV` 的
  `externalBuffer` 传入，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
  空矩阵（`nnz == 0`）会回退到向量版实现，返回 0。

workspace 由调用方持有，`aclsparseSpMV` 只在其上排队异步任务、不做 stream 同步；
因此复用同一块 buffer 前需自行保证前一次 SpMV 已完成。

### aclsparseSpMVPreprocess

Atlas A2/A3 暂未支持该接口。Ascend 950PR/950DT 上，非转置路径为成功的空操作；转置路径在调用 stream 上异步构建稳定的按列索引：
每列内严格保留原 CSR 的行序和元素序，后续每个输出元素由单一 SIMT 线程组按固定顺序
累加，避免浮点 atomic add 引入非确定性。同一 sparsity pattern 可复用同一 workspace；
更新 CSR rowOffsets/colInd 或其 stride 后必须重新调用 Preprocess。非转置及 NNZ=0 时无需读矩阵数据；
转置且 NNZ>0 时只要求 rowOffsets/colInd 和 workspace 已绑定，不要求 values 或 x/y 已绑定。
预处理与执行均将 Device 任务入队到 handle 的 stream，不同步回读 CSR；同一 stream 上的异步 H2D 可以直接排在调用前，
调用方负责在读取结果或复用/释放相关内存前同步，并保证所有异步操作期间内存有效。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|--------|--------|----------|----------|----------|
| amelia_2026 | 社区开发者 | SpMV | 2026-08-26 | Ascend 950PR/950DT 三阶段接口、确定性转置预处理、SIMT kernel、自测代码与文档 |
