# SDDMM算子

## 算子概述

SDDMM（Sampled Dense-Dense Matrix Multiplication，采样稠密-稠密矩阵乘法）只在稀疏矩阵 C 的 pattern 位置计算两个稠密矩阵的乘积，并将结果原地写回 C 的 values 数组。接口语义参考 cuSPARSE Generic API 的 `cusparseSDDMM`。

本目录同时包含 Atlas A2/A3（`arch22/`）和 Ascend 950（`arch35/`）实现。
下文详细说明 Ascend 950；Atlas A2/A3 的支持矩阵、预处理和 stream 约定见
[arch22 说明](arch22/README.md)。两种架构均支持 CSR/BSR、base 0/1、
Host/Device 标量、复数、空矩阵和 BSR strided batch，具体限制按各架构说明执行。

数学表达式：

```
C_out = (alpha * op(X) * op(Y) + beta * C) ∘ spy(C)
```

其中：

- `op(X)` 为 m×k，`op(Y)` 为 k×n；X/Y 均支持 ROW/COL 布局和 NON_TRANSPOSE/TRANSPOSE。
  原始描述符形状由操作标志决定：`opX=NON_TRANSPOSE` 时 X 为 m×k，`opX=TRANSPOSE` 时 X
  为 k×m；`opY=NON_TRANSPOSE` 时 Y 为 k×n，`opY=TRANSPOSE` 时 Y 为 n×k
- C 为 m×n CSR 或方块 BSR 稀疏矩阵，spy(C) 为固定稀疏 pattern
- ∘ 表示 Hadamard 积（逐元素乘），即只在 C 的非零位置保存计算结果
- alpha、beta 为 computeType 标量，内存位置由 handle 的 pointer mode 决定

对 CSR 每个非零元素以及 BSR 每个非零块内元素（逻辑行 i、列 j），算子计算 `op(X)[i,:]` 与 `op(Y)[:,j]` 的 K 维点积，结果为 `alpha * dot + beta * old_value`，并原地写回对应 values。`beta=0` 时不读取旧 values。

调用流程为三步法：

1. **BufferSize**：查询所需 workspace 大小
2. **Preprocess**（可选）：在 Device 上校验 row offsets/column indices，并把完整 pattern 状态绑定到 matC 描述符
3. **Execute**：执行采样矩阵乘法，结果原地更新 C 的 values 数组

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSDDMMBufferSize | 查询 SDDMM 所需 workspace 大小（字节） |
| aclsparseSDDMMPreprocess | 校验 pattern 并建立 descriptor-local prepared state |
| aclsparseSDDMM | 执行采样矩阵乘法 C_out = (alpha * op(X) * op(Y) + beta * C) ∘ spy(C)；未命中 prepared state 时自动预处理 |
| aclsparseCreateBsr / aclsparseCreateConstBsr | 创建 mutable/const BSR 描述符 |
| aclsparseBsrSetStridedBatch | 设置 BSR 的结构和值 batch stride |
| aclsparseDnMatGetStridedBatch / aclsparseDnMatSetStridedBatch | 查询或设置稠密矩阵 batch 属性 |

## 实现目录

Ascend 950 实现按 Host、Device 及稀疏格式组织：

```text
arch35/
├── host/
│   ├── sddmm_host.cpp                 # 三阶段接口、参数校验与执行规划
│   └── sddmm_pattern_host.h           # 校验分核规划与 Host 结果归并
├── kernels/
│   ├── csr/
│   │   ├── sddmm_csr_cube_kernel.cpp
│   │   ├── sddmm_csr_cube_kernel.h
│   │   ├── sddmm_csr_cube_gather_kernel.h
│   │   ├── sddmm_csr_cube_traits.h
│   │   ├── sddmm_column_vector_kernel.h
│   │   └── sddmm_packed_vector_kernel.h
│   ├── bsr/
│   │   ├── sddmm_bsr_cube_kernel.cpp
│   │   ├── sddmm_complex_bsr_cube_kernel.cpp
│   │   └── sddmm_bsr8_vector_kernel.h
│   └── shared/
│       ├── sddmm_kernel.cpp           # CSR/BSR 通用调度、校验、打包与 SIMT 计算
│       ├── sddmm_complex_row_kernel.cpp
│       ├── sddmm_device_math.h
│       ├── sddmm_pattern_validator.h   # 按行、列及 batch 分片的 pattern 校验
│       └── sddmm_simt_convert.h
├── sddmm.h                           # 兼容入口
├── sddmm_kernel.h                    # Host/Device 共用 launch 声明
├── sddmm_tiling_data.h               # 共用数据布局与常量
├── sddmm_bsr8_common.h
├── sddmm_bsr_cube_common.h
├── sddmm_complex_bsr_cube_common.h
├── sddmm_complex_row_common.h
└── sddmm_csr_cube_common.h            # 各路径共用的规划、约束与 launch 声明
```

根目录的 `*_common.h` 由 Host 和 Device 共用，不包含独立计算 Kernel 实现。
构建系统递归收集当前架构的 `.cpp`；实现文件通过相对路径引用跨目录头文件。

## PyTorch 接口

Ascend 950 上通过统一 `cann_ops_sparse` wheel 使用 `torch.sparse.sampled_addmm`：

```python
import torch
import torch_npu
import cann_ops_sparse

result = torch.sparse.sampled_addmm(pattern, mat1, mat2, alpha=1.0, beta=0.0)
# 将结果写入指定 CSR 张量：
torch.sparse.sampled_addmm(pattern, mat1, mat2, out=result)
```

`pattern` 为 NPU CSR 张量，`mat1`、`mat2` 为同设备、同类型的稠密张量。
支持 FP16、FP32、complex64 和 int32/int64 索引，输出保留 CSR pattern；
默认入口返回独立结果，`out=pattern` 可原地更新 values。
`import cann_ops_sparse` 完成公开接口注册，第一次 NPU 调用时 JIT 编译 C++ wrapper。
构建和接口规格见 [Torch Extension 接口说明](../../torch_extension/cann_ops_sparse/docs/zh/sddmm.md)，
功能测试步骤见 [SDDMM 测试说明](../../test/sddmm/README.md#pytorch-端到端测试)。

## 算子执行接口

### aclsparseSDDMMBufferSize

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSDDMMBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, size_t *size)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opX | 输入 | aclsparseOperation_t | 稠密矩阵 X 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置，Host 内存 |
| opY | 输入 | aclsparseOperation_t | 稠密矩阵 Y 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，类型须与 computeType 匹配。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matX | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 X 的描述符：`opX=NON_TRANSPOSE` 时为 m×k，`opX=TRANSPOSE` 时为 k×m，Host 内存 |
| matY | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 Y 的描述符：`opY=NON_TRANSPOSE` 时为 k×n，`opY=TRANSPOSE` 时为 n×k，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针，类型须与 computeType 匹配。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matC | 输入/输出 | aclsparseSpMatDescr_t | 稀疏矩阵 C 的 mutable CSR/BSR 描述符，pattern 只读、values 原地更新，Host 内存 |
| computeType | 输入 | aclDataType | 累加及 alpha/beta 类型；arch35 支持 `ACL_FLOAT` 或 `ACL_COMPLEX64`，Host 内存 |
| alg | 输入 | aclsparseSDDMMAlg_t | 算法类型，仅支持 `ACL_SPARSE_SDDMM_ALG_DEFAULT`，Host 内存 |
| size | 输出 | size_t* | 输出所需 workspace 大小（字节），Host 内存 |

#### 约束说明

- handle 不可为 nullptr
- alpha、beta、size 不可为 nullptr
- matX、matY、matC 不可为 nullptr
- matC 支持 CSR 或方块 BSR；row offsets 和 column indices 均须为 `ACL_SPARSE_INDEX_32I`
- arch35 CSR 的 row offsets、column indices、values 元素 stride 均须为 1；非连续 CSR 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`
- index base 支持 `ACL_SPARSE_INDEX_BASE_ZERO` 和 `ACL_SPARSE_INDEX_BASE_ONE`
- BSR 块高与块宽必须相等，支持 2/4/8/16/32/64/128；块内布局支持 ROW/COL
- opX 支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置
- opY 支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置
- arch35 数据类型组合见下表；X 与 Y 的类型必须相同

| matC 格式 | X/Y | C values | computeType |
|-----------|-----|----------|-------------|
| CSR/BSR | ACL_FLOAT | ACL_FLOAT | ACL_FLOAT |
| CSR/BSR | ACL_COMPLEX64 | ACL_COMPLEX64 | ACL_COMPLEX64 |
| CSR/BSR | ACL_FLOAT16 | ACL_FLOAT16 | ACL_FLOAT |
| CSR/BSR | ACL_FLOAT16 | ACL_FLOAT | ACL_FLOAT |
| CSR | ACL_BF16 | ACL_FLOAT | ACL_FLOAT |
| BSR | ACL_BF16 | ACL_BF16 | ACL_FLOAT |
| BSR | ACL_BF16 | ACL_FLOAT | ACL_FLOAT |

- 维度须满足 `op(X).shape=[m,k]`、`op(Y).shape=[k,n]`、`C.shape=[m,n]`。
  因此，`opX=NON_TRANSPOSE` 时 X 为 m×k，`opX=TRANSPOSE` 时 X 为 k×m；
  `opY=NON_TRANSPOSE` 时 Y 为 k×n，`opY=TRANSPOSE` 时 Y 为 n×k。
- m、n、k 以及 CSR nnz/BSR blockNnz 的 I32 编码结果不得超过 `INT32_MAX`
- arch35 支持动态 shape、空行和零 nnz；m 或 n 为 0 时 pattern 必须为空，三阶段均成功返回；k 为 0 时输出为 beta * C（beta 为 0 时不读取旧 values）。arch22 对 X/Y/C 任一维度为 0 的输入返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- workspace 要求 64 字节对齐。m 或 n 为 0 时为 0 字节；m、n 均大于 0（包括 k 为 0）时，常规路径为 64 字节 pattern 校验区。满足连续 ROW/ROW、opX/opY=N、CSR FP32/FP16 或单 batch BSR b8/K128 等复用条件时，BufferSize 还会返回转置打包 Y 所需的动态 scratch。packed-Y 预留大小包含前缀与对齐后不超过 64 MiB；CSR packed-Y 与 CSR Cube scratch 的执行分支互斥，共享同一个私有 offset。仅当 CSR Cube 需要比 generic 路径更多空间时，BufferSize 才在 handle stream 上检查实际 Device pattern 和首列对齐：命中时预留二者最大值，回退时只返回 generic 容量。该检查使用运行时获取的 AIV 核数并行判定，每个活跃核独占 64 字节临时结果槽，总大小为 `64 × min(AIV 核数, m)` 字节；一次同步读回后合并结果，不占用其他算子的 workspace，也不替代后续 Preprocess 的完整 pattern 校验。实数 BSR Cube 与 complex64 split-real Cube 使用各自的有界私有布局。所有区域按 64 字节向上对齐；实数 BSR Cube 与 CSR Cube 私有区上限仍为 64 MiB，complex64 BSR 路径的私有区上限为 128 MiB 减 64 字节，并额外约束包含校验前缀与对齐的公共 workspace 总量不超过 128 MiB
- complex64 BSR Cube 在 Host 查询平台 L0A/L0B 字节容量，并核对当前 tile（包括 B16/K128 分组路径）的空间需求。容量不足或无法取得有效容量时使用通用路径；该检查在 BufferSize、Preprocess、Execute 的共享规划中执行，首次查询回退时不预留 Cube 私有区。
- arch35 每次成功 BufferSize 查询会替换 matC 的容量计划并使旧 prepared state 失效；之后传入的 buffer 必须满足此次查询大小。shape、layout、dtype、base 或 batch 等元数据变化后须重新查询。仅原地修改 pattern 或替换同规格索引指针时可沿用已分配容量，但须重新预处理；即使新 pattern 适合 Cube，也只在原容量足够时启用。alpha/beta 或 pointer mode 的变化不会扩大容量。BufferSize 的 pattern 检查不替代 Preprocess 对当前索引内容的重新校验。
- 需要并行检查 pattern 时，若没有可用 AIV 核则返回 `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`；临时区分配失败返回 `ACL_SPARSE_STATUS_ALLOC_FAILED`，校验同步、读回或释放失败返回 `ACL_SPARSE_STATUS_EXECUTION_FAILED`，失败查询不替换已有容量计划。同步失败时描述符保留该临时区，后续 SDDMM 调用先尝试同步并释放；描述符销毁时若仍无法完成同步，保留分配以避免释放仍被 Device 写入的空间。
- arch35 的 Preprocess（包括冷 Execute 自动预处理）按运行时 AIV 核数和索引数量裁剪校验核数，小任务使用单核。offsets 按行区间检查相邻边界，columns 按独立索引区间检查，非共享的 batch pattern 也参与分片；offsets/columns 均广播时只检查一份结构。各核写入独立的 64 字节临时结果槽，Host 在同步读回后归并状态与 pattern 属性，再更新原 workspace 前缀；不增加 BufferSize 已查询的用户 workspace 容量。错误仍按最早出错 batch 及该 batch 内的端点、offsets、columns 顺序判定，失败时不会建立 prepared state。
- DnMat/BSR batchCount 范围为 1..65535。X/Y 可为 batch 1 广播或与 C batch 相同；BSR offsets/columns stride 为 0 表示共享 pattern，values 不允许跨 batch 广播

> **与 cuSPARSE opB 的语义对照（迁移注意）**
>
> 本算子公式为 `C = (α·op(X)·op(Y) + βC) ∘ spy(C)`。`opX`、`opY` 分别决定 X、Y
> 描述符的原始形状和参与乘积的有效形状：`opX=NON_TRANSPOSE` 时 X 为 m×k，
> `opX=TRANSPOSE` 时 X 为 k×m；`opY=NON_TRANSPOSE` 时 Y 为 k×n，
> `opY=TRANSPOSE` 时 Y 为 n×k。cuSPARSE `cusparseSDDMM` 中 op(A)/op(B) 的语义
> 分别与本接口 opX/opY 一致，opY 的对应关系如下：
>
> | 本接口 opY | 实际计算的乘积 | 等价的 cuSPARSE opB |
> |-----------|---------------|---------------------|
> | `ACL_SPARSE_OP_NON_TRANSPOSE` | op(X)·Y（Y 描述符 k×n） | `CUSPARSE_OPERATION_NON_TRANSPOSE` |
> | `ACL_SPARSE_OP_TRANSPOSE` | op(X)·Y^T（Y 描述符 n×k） | `CUSPARSE_OPERATION_TRANSPOSE` |
>
> 从 cuSPARSE 迁移时，第二个稠密矩阵的操作标志语义一致，可直接对应传入：cuSPARSE 使用 `opB=NON_TRANSPOSE` 的场景，本接口传 `opY=NON_TRANSPOSE`；cuSPARSE 使用 `opB=TRANSPOSE` 的场景，本接口传 `opY=TRANSPOSE`。

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | I32 index，base 0/1 |
| BSR | ✅ | 方块 BSR，ROW/COL 块内布局及 strided batch；Atlas A2/A3 约束见 arch22 说明 |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

---

### aclsparseSDDMMPreprocess

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSDDMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opX | 输入 | aclsparseOperation_t | 稠密矩阵 X 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置，Host 内存 |
| opY | 输入 | aclsparseOperation_t | 稠密矩阵 Y 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matX | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 X 的描述符：`opX=NON_TRANSPOSE` 时为 m×k，`opX=TRANSPOSE` 时为 k×m，Host 内存 |
| matY | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 Y 的描述符：`opY=NON_TRANSPOSE` 时为 k×n，`opY=TRANSPOSE` 时为 n×k，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matC | 输入/输出 | aclsparseSpMatDescr_t | 稀疏矩阵 C 的 mutable CSR/BSR 描述符，Host 内存 |
| computeType | 输入 | aclDataType | 累加及 alpha/beta 类型；arch35 支持 `ACL_FLOAT` 或 `ACL_COMPLEX64`，Host 内存 |
| alg | 输入 | aclsparseSDDMMAlg_t | 算法类型，仅支持 `ACL_SPARSE_SDDMM_ALG_DEFAULT`，Host 内存 |
| buffer | 输入 | void* | workspace 缓冲区（由 BufferSize 返回的大小分配），Device 内存 |

#### 约束说明

- 同 aclsparseSDDMMBufferSize 的约束
- BufferSize 非零时，buffer 不可为 nullptr，且需按查询结果分配并满足 64 字节对齐
- 预处理在调用 stream 上校验 CSR/BSR offsets 的端点与单调性以及 column indices 范围，并缓存用于路径选择的 pattern flags 和首列信息
- prepared state 属于 matC 描述符，完整 key 包含 pattern 指针、结构代际、shape、base、块/布局、batch、X/Y 元数据、算法和 workspace
- values-only 更新不会使 pattern state 失效；结构指针或结构属性实际变化会失效。若调用方原地修改 offsets/columns 内容，必须重新调用 Preprocess
- matC 和 workspace 均由调用方管理生命周期；实现不分配或释放调用方 workspace

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | I32 index，base 0/1 |
| BSR | ✅（Ascend 950） | 方块 BSR，ROW/COL 块内布局及 strided batch |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

---

### aclsparseSDDMM

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSDDMM(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opX | 输入 | aclsparseOperation_t | 稠密矩阵 X 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置，Host 内存 |
| opY | 输入 | aclsparseOperation_t | 稠密矩阵 Y 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matX | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 X 的描述符：`opX=NON_TRANSPOSE` 时为 m×k，`opX=TRANSPOSE` 时为 k×m，Host 内存 |
| matY | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 Y 的描述符：`opY=NON_TRANSPOSE` 时为 k×n，`opY=TRANSPOSE` 时为 n×k，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matC | 输入/输出 | aclsparseSpMatDescr_t | 稀疏矩阵 C 的 mutable CSR/BSR 描述符，values 原地更新，Host 内存 |
| computeType | 输入 | aclDataType | 累加及 alpha/beta 类型；arch35 支持 `ACL_FLOAT` 或 `ACL_COMPLEX64`，Host 内存 |
| alg | 输入 | aclsparseSDDMMAlg_t | 算法类型，仅支持 `ACL_SPARSE_SDDMM_ALG_DEFAULT`，Host 内存 |
| buffer | 输入 | void* | workspace 缓冲区（由 BufferSize 返回的大小分配），Device 内存 |

#### 约束说明

- 同 aclsparseSDDMMBufferSize 的约束
- BufferSize 非零时 buffer 不可为 nullptr，并须与 prepared key 中的 workspace 一致
- Preprocess 为可选调用。Execute 未命中 prepared state 时会自动完成一次预处理；该冷路径为返回结构校验状态会同步当前 stream
- prepared state 命中时 Execute 不进行 Host/Device 状态搬运或 Host 同步；常规路径异步启动计算 Kernel，连续 Y 快路径依次异步启动打包与计算 Kernel，Cube 快路径依次启动准备、Cube 计算和 AIV 尾处理 Kernel。调用方需在读取 C values 前同步同一 stream，性能统计须包含同一次 Execute 发出的全部 Kernel
- X/Y 与 C pattern 只读，C values 原地更新；同一 matC 的 Preprocess/Execute 并发由调用方串行。不同 matC 仅在使用互不重叠的 workspace 时可独立并发；不同 stream 还必须使用独立 handle，并满足各自 stream 时序

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | I32 index，base 0/1 |
| BSR | ✅（Ascend 950） | 方块 BSR，ROW/COL 块内布局及 strided batch |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

### 调用示例

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
    aclError ret = aclrtMalloc(&dPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return nullptr;
    }
    if (hostPtr != nullptr && sizeBytes > 0) {
        aclrtMemcpy(dPtr, sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return dPtr;
}

int aclsparseSDDMMTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(rawHandle, aclsparseDestroy);

    sparseRet = aclsparseSetStream(static_cast<aclsparseHandle_t>(handlePtr.get()), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 2. 设置 PointerMode
    sparseRet = aclsparseSetPointerMode(static_cast<aclsparseHandle_t>(handlePtr.get()), ACL_SPARSE_POINTER_MODE_HOST);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetPointerMode failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 3. 准备 Host 端数据
    //    X (4x2):           Y (3x2):
    //    [1.0  2.0]         [1.0  0.0]
    //    [3.0  4.0]         [0.0  1.0]
    //    [5.0  6.0]         [1.0  1.0]
    //    [7.0  8.0]
    //
    //    C (4x3, nnz=5), 非零位置: (0,0)(0,2)(1,1)(2,0)(3,2), 初始值全 0
    //    C = 1.0 * X * Y^T ∘ spy(C) (beta=0):
    //    C[0,0]=X[0]·Y[0]=1.0  C[0,2]=X[0]·Y[2]=3.0
    //    C[1,1]=X[1]·Y[1]=4.0  C[2,0]=X[2]·Y[0]=5.0
    //    C[3,2]=X[3]·Y[2]=15.0
    int64_t m = 4, n = 3, k = 2;
    int64_t nnzC = 5;
    float hAlpha = 1.0f;
    float hBeta = 0.0f;

    std::vector<int> hRowPtrC = {0, 2, 3, 4, 5};
    std::vector<int> hColIndC = {0, 2, 1, 0, 2};
    std::vector<float> hValC  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // 本示例使用 opX=NON_TRANSPOSE、opY=TRANSPOSE，因此 X 为 m×k (4x2)，Y 为 n×k (3x2)。
    // 若切换操作类型，描述符原始形状也要随之切换：opX=TRANSPOSE 时 X 为 k×m，
    // opY=NON_TRANSPOSE 时 Y 为 k×n。
    int64_t ldx = k, ldy = k;
    aclsparseOrder_t orderX = ACL_SPARSE_ORDER_ROW;
    aclsparseOrder_t orderY = ACL_SPARSE_ORDER_ROW;
    std::vector<float> hX(static_cast<size_t>(m) * k, 0.0f);
    hX[0 * k + 0] = 1.0f; hX[0 * k + 1] = 2.0f;
    hX[1 * k + 0] = 3.0f; hX[1 * k + 1] = 4.0f;
    hX[2 * k + 0] = 5.0f; hX[2 * k + 1] = 6.0f;
    hX[3 * k + 0] = 7.0f; hX[3 * k + 1] = 8.0f;

    std::vector<float> hY(static_cast<size_t>(n) * k, 0.0f);
    hY[0 * k + 0] = 1.0f; hY[0 * k + 1] = 0.0f;
    hY[1 * k + 0] = 0.0f; hY[1 * k + 1] = 1.0f;
    hY[2 * k + 0] = 1.0f; hY[2 * k + 1] = 1.0f;

    // 4. 拷贝数据到 Device
    void *dRowPtrC = AllocAndCopyDevice(hRowPtrC.data(), (m + 1) * sizeof(int));
    void *dColIndC = AllocAndCopyDevice(hColIndC.data(), nnzC * sizeof(int));
    void *dValC    = AllocAndCopyDevice(hValC.data(),    nnzC * sizeof(float));
    void *dX       = AllocAndCopyDevice(hX.data(),       static_cast<size_t>(m) * k * sizeof(float));
    void *dY       = AllocAndCopyDevice(hY.data(),       static_cast<size_t>(n) * k * sizeof(float));

    // 5. 创建描述符
    aclsparseSpMatDescr_t matC = nullptr;
    sparseRet = aclsparseCreateCsr(&matC, m, n, nnzC, dRowPtrC, dColIndC, dValC,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    aclsparseDnMatDescr_t matX = nullptr, matY = nullptr;
    sparseRet = aclsparseCreateDnMat(&matX, m, k, ldx, dX, ACL_FLOAT, orderX);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateDnMat X failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    sparseRet = aclsparseCreateDnMat(&matY, n, k, ldy, dY, ACL_FLOAT, orderY);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateDnMat Y failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 6. Step 1 — BufferSize
    size_t bufferSize = 0;
    sparseRet = aclsparseSDDMMBufferSize(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
        &hAlpha, matX, matY, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SDDMMBufferSize failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes\n", bufferSize);

    void *dBuffer = nullptr;
    auto aclRet = aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for buffer failed. ERROR: %d\n", aclRet); return aclRet);

    // 7. Step 2 — Preprocess
    sparseRet = aclsparseSDDMMPreprocess(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
        &hAlpha, matX, matY, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SDDMM_ALG_DEFAULT, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SDDMMPreprocess failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 8. Step 3 — Execute（结果原地写回 C 的 values 数组）
    sparseRet = aclsparseSDDMM(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
        &hAlpha, matX, matY, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SDDMM_ALG_DEFAULT, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SDDMM failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 9. 同步等待计算完成
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 10. 将结果拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hValC.data(), static_cast<size_t>(nnzC) * sizeof(float),
                         dValC, static_cast<size_t>(nnzC) * sizeof(float),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);

    for (int64_t p = 0; p < nnzC; p++) {
        LOG_PRINT("C_values[%lld] = %.1f\n", static_cast<long long>(p), hValC[p]);
    }

    // 11. 清理资源
    aclsparseDestroySpMat(matC);
    aclsparseDestroyDnMat(matX);
    aclsparseDestroyDnMat(matY);
    if (dRowPtrC) aclrtFree(dRowPtrC);
    if (dColIndC) aclrtFree(dColIndC);
    if (dValC)    aclrtFree(dValC);
    if (dX)       aclrtFree(dX);
    if (dY)       aclrtFree(dY);
    if (dBuffer)  aclrtFree(dBuffer);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseSDDMMTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSDDMMTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
bufferSize = 64 bytes
C_values[0] = 1.0
C_values[1] = 3.0
C_values[2] = 4.0
C_values[3] = 5.0
C_values[4] = 15.0
```

## 精度标准

SDDMM 算子精度验证采用框架 MIXED_TOLERANCE 模式（`test/frame/verify.h::applyMixedTolerance` + `Verifier::verifyVector`），按数据类型自动选择 atol/rtol 与 maxAbsErrLimit：

- 逐元素通过条件：`|actual - golden| ≤ max(atol + rtol * |golden|, maxAbsErrLimit)`，其中 `maxAbsErrLimit = max(fixedValue, 32 * ULP(golden))`，`ULP(x) = 2^(floor(log2|x|) - mantissaBits)`。
- 整体通过条件：`matchedRatio ≥ 0.99`。

| 数据类型 | rtol（相对容差） | atol（绝对容差） | maxAbsErrLimit fixedValue |
|----------|------------------|------------------|---------------------------|
| FLOAT16  | 1.953125e-3 (2^-9)  | 1.953125e-3 (2^-9)  | 1e-1 |
| BFLOAT16 | 1.5625e-2 (2^-6) | 1.5625e-2 (2^-6) | 1.0 |
| FLOAT32  | 9.765625e-4 (2^-10) | 1.52587890625e-5 (2^-23) | 1e-2 |

complex64 的实部和虚部分别按 FLOAT32 规则比较；CPU Golden 使用 complex128。FP16/BF16 输入采用 FP32 计算与累加，FP32 输入使用 FP64 Golden。

> 说明：FP16 写回遵循 IEEE 转换语义，有限 FP32 结果超出 FP16 范围时得到有符号 Inf，不进行 65504 饱和截断；Inf/NAN 按生态算子精度标准验收。

## 支持芯片

- Ascend 950PR / Ascend 950DT（架构 arch35 / DAV_3510）
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Ascend 910B（架构 arch22 / DAV_2201）：对应 Atlas A2 训练系列产品 / Atlas A2 推理系列产品
