# Cube SpMM 算子

## 算子概述

Cube SpMM 算子完成基于 BCSR（Block Compressed Sparse Row）格式的稀疏矩阵与稠密矩阵的乘法：

```
C = alpha * A * B + beta * C
```

其中：

- $A$ 为稀疏矩阵，输入形态为 COO，经 `aclsparseCubeSpmmPreprocess` 预处理后转换为 BCSR 格式（`rwPtr` 为 `int64_t`，`colRef` 与 `coreInfo` 为 `int32_t`，`vals` 为 float16）。
- $B$ 为稠密矩阵，数据类型为 `ACL_FLOAT16`，形状为 $[K, N]$，当前仅支持 $N$ 为 16 的倍数。
- $C$ 为输出矩阵，数据类型为 `ACL_FLOAT`，形状为 $[M, N]$。
- `computeType` 固定为 `ACL_FLOAT`。
- Kernel 内部在 $N$ 方向按 `kNChunkSize = 512` 进行分块，local buffer（B1/B2/CO1 以及清零用 buffer）均按 `min(N, 512)` 分配，以控制单核内存用量。该分块对用户透明，不影响接口层面的 $N$ 约束。

## 维度取值限制

当前实现为了保证 Host 到 Kernel 的 TilingData 以及 BCSR 内部索引不发生溢出，对 $M$、$K$、$N$ 有如下限制：

- $M > 0$，$K > 0$，$N > 0$，且三者均须不大于 `INT32_MAX`（$2{,}147{,}483{,}647$）。该上限来源于 `aclsparseCubeSpmmPreprocess` 当前接受的 COO 行/列索引为 `int32_t`.
- $M \times K \le$ `INT64_MAX`（约 $9.2 \times 10^{18}$），保证 BCSR 的块索引产品（`numBlocks * 16`、`numBlocks * 256`）不超出 int64_t 范围。
- $N$ 须为 16 的倍数，且 $N \le 32{,}767$（按 16 对齐后的有效上限为 $32{,}752$），这是由 Kernel 内部 `DataCopyParams.blockLen` 为 uint16_t 决定的。
- 稠密 $B$ 的字节数 $K \times N \times 2$ 以及输出 $C$ 的字节数 $M \times N \times 4$ 必须能放入 `size_t`。

超出上述范围时，接口会返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` 或 `ACL_SPARSE_STATUS_INVALID_VALUE`。

调用流程为以下步骤：

1. **CreateCubeSpmmMat**：创建 Cube SpMM 稀疏矩阵描述符。
2. **Preprocess**：调用 `aclsparseCubeSpmmPreprocess` 将 COO 转换为 BCSR。
3. **PadDenseMatrixB**：调用 `aclsparseCubeSpmmPadDenseMatrixB` 对稠密 $B$ 做 padding 对齐（$A$ 不变而 $B$ 变化时可单独调用，复用 BCSR 数据）。
4. **GetBufferSize**：查询 SpMM 所需 workspace 大小（当前实现固定返回 0：tiling 作为 kernel 启动参数随 `<<<>>>` 下发，无需 device workspace）。
5. **CubeSpmm**：在 AICore 上执行矩阵乘法。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| `aclsparseCreateCubeSpmmMat` | 创建 Cube SpMM 稀疏矩阵描述符 |
| `aclsparseDestroyCubeSpmmMat` | 销毁 Cube SpMM 稀疏矩阵描述符 |
| `aclsparseCubeSpmmPreprocess` | 稀疏矩阵 $A$ 的 COO -> BCSR 转换 |
| `aclsparseCubeSpmmPadDenseMatrixB` | 稠密矩阵 $B$ 的 padding 对齐 |
| `aclsparseCubeSpmmGetBufferSize` | 查询 Cube SpMM 所需 workspace 大小（字节） |
| `aclsparseCubeSpmm` | 执行 BCSR 稀疏矩阵与稠密矩阵乘法 |

## 算子执行接口

### aclsparseCreateCubeSpmmMat

#### 产品支持情况

- Atlas A2系列产品：支持
- 对应芯片为 Ascend 910B。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCreateCubeSpmmMat(
    aclsparseCubeSpmmMatDescr_t *descr,
    int64_t rows, int64_t cols, int64_t nnz,
    int64_t blockM, int64_t blockK, int32_t numCores);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| descr | 输出 | `aclsparseCubeSpmmMatDescr_t*` | 输出的稀疏矩阵描述符，Host 内存 |
| rows | 输入 | `int64_t` | 稀疏矩阵行数 $M$，Host 内存 |
| cols | 输入 | `int64_t` | 稀疏矩阵列数 $K$，Host 内存 |
| nnz | 输入 | `int64_t` | COO 非零元个数，Host 内存 |
| blockM | 输入 | `int64_t` | BCSR 行方向块大小，当前固定为 16，Host 内存 |
| blockK | 输入 | `int64_t` | BCSR 列方向块大小，当前固定为 16，Host 内存 |
| numCores | 输入 | `int32_t` | 后续 SpMM 使用的 AICore 数量，Host 内存 |

#### 约束说明

- `descr` 不可为 `nullptr`。
- `rows > 0`、`cols > 0`、`nnz >= 0`。
- `blockM` 与 `blockK` 必须固定为 $16$（当前 kernel 硬性要求）。
- `numCores > 0`。
- 当前固定使用 0-based int32 索引和 float16 值（COO 输入阶段）；BCSR 内部仅 `rwPtr` 升级为 `int64_t`，`colRef` 与 `coreInfo` 仍为 `int32_t`。
- `rows` 和 `cols` 必须不大于 `INT32_MAX`（受 COO `int32_t` 索引限制），且 `rows * cols` 必须不大于 `INT64_MAX`（保证 BCSR 块索引产品安全）。
- `nnz` 不能超过 `rows * cols`。

---

### aclsparseDestroyCubeSpmmMat

#### 产品支持情况

- Atlas A2系列产品：支持
- 对应芯片为 Ascend 910B。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseDestroyCubeSpmmMat(
    aclsparseConstCubeSpmmMatDescr_t descr);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| descr | 输入 | `aclsparseConstCubeSpmmMatDescr_t` | 要销毁的 Cube SpMM 稀疏矩阵描述符，Host 内存 |

#### 约束说明

- 传入 `nullptr` 时直接返回 `ACL_SPARSE_STATUS_SUCCESS`。
- 会释放由 `aclsparseCubeSpmmPreprocess` 分配在描述符内的设备内存（`rwPtr`、`colRef`、`vals`、`coreInfo`）。

---

### aclsparseCubeSpmmPadDenseMatrixB

#### 产品支持情况

- Atlas A2系列产品：支持
- 对应芯片为 Ascend 910B。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCubeSpmmPadDenseMatrixB(
    aclsparseHandle_t handle,
    int64_t bRows, int64_t bCols, int64_t bLd, const void *bValues,
    aclDataType bType, aclsparseOrder_t bOrder,
    int64_t *nPadOut, void **bPadOut);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | `aclsparseHandle_t` | ops-sparse 库上下文句柄，Host 内存 |
| bRows | 输入 | `int64_t` | 稠密 $B$ 的行数，应等于 `matA->cols`，Host 内存 |
| bCols | 输入 | `int64_t` | 稠密 $B$ 的原始列数 $N$，Host 内存 |
| bLd | 输入 | `int64_t` | 稠密 $B$ 的 leading dimension，Host 内存 |
| bValues | 输入 | `const void*` | 稠密 $B$ 的主机内存指针（float16），Host 内存；接口内部完成 padding 后一次性拷贝到 Device |
| bType | 输入 | `aclDataType` | 稠密 $B$ 的数据类型，必须为 `ACL_FLOAT16`，Host 内存 |
| bOrder | 输入 | `aclsparseOrder_t` | 稠密 $B$ 的存储顺序，必须为 `ACL_SPARSE_ORDER_ROW`，Host 内存 |
| nPadOut | 输出 | `int64_t*` | 输出 padding 后的 $N$ 大小，Host 内存 |
| bPadOut | 输出 | `void**` | 输出 padding 后的 $B$ 设备内存指针，需由调用者使用 `aclrtFree` 释放，Device 内存 |

#### 约束说明

- `handle`、`bValues`、`nPadOut`、`bPadOut` 均不可为 `nullptr`。
- `bRows > 0`、`bCols > 0`、`bLd >= bCols`。
- `bType` 必须为 `ACL_FLOAT16`，`bOrder` 必须为 `ACL_SPARSE_ORDER_ROW`。
- `bCols` 建议为 16 的倍数；非倍数时内部会 padding 到满足 L0B / CopyInB 对齐要求的最小值。
- 输出的 `bPadOut` 内存由本接口通过 `aclrtMalloc` 分配，调用者须负责释放。
- 当稀疏矩阵 $A$ 不变而稠密 $B$ 变化时，可单独重复调用本接口，复用同一份 BCSR 数据。

---

### aclsparseCubeSpmmGetBufferSize

#### 产品支持情况

- Atlas A2系列产品（Ascend 910B）：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCubeSpmmGetBufferSize(
    aclsparseHandle_t handle,
    const void *alpha, aclsparseConstCubeSpmmMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    size_t *size);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | `aclsparseHandle_t` | ops-sparse 库上下文句柄，Host 内存 |
| alpha | 输入 | `const void*` | 标量 alpha 指针，当前未使用，可传 `nullptr`，Host 内存 |
| matA | 输入 | `aclsparseConstCubeSpmmMatDescr_t` | 经 `aclsparseCubeSpmmPreprocess` 预处理后的稀疏矩阵描述符，Host 内存 |
| matB | 输入 | `aclsparseConstDnMatDescr_t` | 稠密矩阵 $B$ 的描述符（padding 后的 `bPad`），Host 内存 |
| beta | 输入 | `const void*` | 标量 beta 指针，当前未使用，可传 `nullptr`，Host 内存 |
| matC | 输入/输出 | `aclsparseDnMatDescr_t` | 稠密矩阵 $C$ 的描述符（padding 后的 `cPad`），Host 内存 |
| computeType | 输入 | `aclDataType` | 计算精度类型，必须为 `ACL_FLOAT`，Host 内存 |
| size | 输出 | `size_t*` | 输出所需 workspace 大小（字节），当前实现固定为 0，Host 内存 |

#### 约束说明

- `size` 不可为 `nullptr`。
- `matA`、`matB`、`matC` 不可为 `nullptr`。
- `computeType` 必须为 `ACL_FLOAT`。
- `matB` 的数据类型必须为 `ACL_FLOAT16`，`matC` 的数据类型必须为 `ACL_FLOAT`。
- `matB` 与 `matC` 的存储顺序必须为 `ACL_SPARSE_ORDER_ROW`。
- 维度匹配：`matA->cols == matB->rows`、`matA->rows == matC->rows`、`matB->cols == matC->cols`。
- `matB->cols` 必须在 padding 后和 `Block_K` 相乘为 512B 的倍数。
- 需满足 [维度取值限制](#维度取值限制) 中的 $M$、$K$、$N$ 范围要求。
- `matB->ld >= matB->cols`、`matC->ld >= matC->cols`；`ld > cols` 的情况现在会被正确处理，Kernel 内部通过 TilingData 中的 `bLd` / `cLd` 计算 GM 偏移。
- `matB->ld` 与 `matC->ld` 均须不大于 `INT32_MAX`。
- `matB->values` 与 `matC->values` 不可为 `nullptr`。

---

### aclsparseCubeSpmm

#### 产品支持情况

- Atlas A2系列产品（Ascend 910B）：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCubeSpmm(
    aclsparseHandle_t handle,
    const void *alpha, aclsparseConstCubeSpmmMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    void *buffer);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | `aclsparseHandle_t` | ops-sparse 库上下文句柄，Host 内存 |
| alpha | 输入 | `const void*` | 标量 alpha 指针，当前未使用，可传 `nullptr`，Host 内存 |
| matA | 输入 | `aclsparseConstCubeSpmmMatDescr_t` | 经预处理后的稀疏矩阵描述符，Host 内存 |
| matB | 输入 | `aclsparseConstDnMatDescr_t` | 稠密矩阵 $B$ 的描述符（padding 后），Host 内存 |
| beta | 输入 | `const void*` | 标量 beta 指针，当前未使用，可传 `nullptr`，Host 内存 |
| matC | 输入/输出 | `aclsparseDnMatDescr_t` | 稠密矩阵 $C$ 的描述符（padding 后），Host 内存 |
| computeType | 输入 | `aclDataType` | 计算精度类型，必须为 `ACL_FLOAT`，Host 内存 |
| buffer | 输入 | `void*` | workspace 缓冲区，当前实现未使用（GetBufferSize 返回 0），可传 `nullptr` |

#### 约束说明

- 同 `aclsparseCubeSpmmGetBufferSize` 的约束。
- `buffer` 当前未使用，可传 `nullptr`。
- 调用前须先调用 `aclsparseCubeSpmmPreprocess` 进行预处理，填充 `matA` 内部的 BCSR 数据。
- 仅支持 `opA = NON_TRANSPOSE`、`opB = NON_TRANSPOSE`。

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| Cube-BCSR | ✅ | 稀疏矩阵 $A$ 支持 Cube-BCSR 格式，由 `aclsparseCubeSpmmPreprocess` 生成 |
| COO | ❌ | 原始 COO 输入需先经过预处理 |
| CSR | ❌ | 不支持 |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"

int main()
{
    // 1. 初始化 ACL 并创建 stream
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // 2. 创建 ops-sparse 句柄
    aclsparseHandle_t handle = nullptr;
    aclsparseCreate(&handle);
    aclsparseSetStream(handle, stream);

    // 3. 准备 COO 数据（示例：M=32, K=32, nnz=10）
    int64_t M = 32;
    int64_t K = 32;
    int64_t N = 128;
    int64_t nnz = 10;
    int32_t numCores = 8;

    std::vector<int32_t> hCooRows(nnz, 0);
    std::vector<int32_t> hCooCols(nnz, 0);
    std::vector<uint16_t> hCooVals(nnz, 0);
    // ... 填充 COO 数据 ...

    // 4. 上传 COO 到设备
    void *dCooRows = nullptr;
    void *dCooCols = nullptr;
    void *dCooVals = nullptr;
    aclrtMalloc(&dCooRows, nnz * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dCooCols, nnz * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dCooVals, nnz * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dCooRows, nnz * sizeof(int32_t), hCooRows.data(),
                nnz * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dCooCols, nnz * sizeof(int32_t), hCooCols.data(),
                nnz * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dCooVals, nnz * sizeof(uint16_t), hCooVals.data(),
                nnz * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // 5. 创建 Cube SpMM 稀疏矩阵描述符
    aclsparseCubeSpmmMatDescr_t matA = nullptr;
    aclsparseCreateCubeSpmmMat(&matA, M, K, nnz, 16, 16, numCores);

    // 6. 生成稠密 B（host 端）
    std::vector<uint16_t> hB(static_cast<size_t>(K) * N, 0);
    // ... 填充 B ...

    // 7. 预处理：COO -> BCSR
    aclsparseCubeSpmmPreprocess(
        handle, nullptr, matA,
        static_cast<const int32_t *>(dCooRows),
        static_cast<const int32_t *>(dCooCols),
        static_cast<const uint16_t *>(dCooVals),
        ACL_FLOAT);

    // 8. 对 B 做 padding 对齐：host -> device 一次性 H2D 拷贝
    int64_t nPad = 0;
    void *dBPad = nullptr;
    aclsparseCubeSpmmPadDenseMatrixB(
        handle, K, N, N, hB.data(),
        ACL_FLOAT16, ACL_SPARSE_ORDER_ROW,
        &nPad, &dBPad);

    // 9. 创建 padding 后的 B/C 稠密描述符
    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;
    void *dC = nullptr;
    aclrtMalloc(&dC, M * nPad * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dC, M * nPad * sizeof(float), 0, M * nPad * sizeof(float));

    aclsparseCreateDnMat(&matB, K, nPad, nPad, dBPad,
                         ACL_FLOAT16, ACL_SPARSE_ORDER_ROW);
    aclsparseCreateDnMat(&matC, M, nPad, nPad, dC,
                         ACL_FLOAT, ACL_SPARSE_ORDER_ROW);

    // 10. 查询 workspace（当前实现固定为 0）并执行 SpMM
    size_t workspaceSize = 0;
    aclsparseCubeSpmmGetBufferSize(
        handle, nullptr, matA, matB, nullptr, matC,
        ACL_FLOAT, &workspaceSize);

    aclsparseCubeSpmm(
        handle, nullptr, matA, matB, nullptr, matC,
        ACL_FLOAT, nullptr);

    aclrtSynchronizeStream(stream);

    // 11. 清理资源
    aclsparseDestroyDnMat(matC);
    aclsparseDestroyDnMat(matB);
    aclsparseDestroyCubeSpmmMat(matA);
    aclrtFree(dC);
    aclrtFree(dBPad);
    aclrtFree(dCooRows);
    aclrtFree(dCooCols);
    aclrtFree(dCooVals);
    aclsparseDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
