# SpMM算子（arch22 / DAV-2201）

## 算子概述

SpMM（Sparse Matrix - Dense Matrix Multiplication）算子实现稀疏矩阵与稠密矩阵的乘法运算。核心运算为 C = alpha * op(A) * op(B) + beta * C，其中 A 为 CSR 格式稀疏矩阵，B 和 C 为稠密矩阵。

本目录为 **arch22（DAV-2201）架构实现**，覆盖 Ascend910_93 系列与 Ascend910B 系列产品（二者均映射到 arch22，见根 CMakeLists 的 SOC_VERSION -> NPU_ARCH 映射）。

数学表达式：

```
C = alpha * op(A) * op(B) + beta * C
```

其中 op(A) 为稀疏矩阵 A 的操作，op(B) 为稠密矩阵 B 的操作。arch22 实现中 op(A)、op(B) 均仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`（非转置）。

调用流程为三步法：
1. **GetBufferSize**：查询所需 workspace 大小
2. **Preprocess**：对稀疏矩阵进行预处理（贪心行装箱 + 构建 Tiling），加速后续计算
3. **SpMM**：执行 C = alpha * op(A) * op(B) + beta * C（异步，调用方自行同步）

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSpMMGetBufferSize | 查询 SpMM 所需 workspace 大小（字节） |
| aclsparseSpMMPreprocess | 对稀疏矩阵进行预处理（行重排 + 分箱 + Tiling） |
| aclsparseSpMM | 执行稀疏矩阵-稠密矩阵乘法 C = alpha * op(A) * op(B) + beta * C |

## 算子执行接口

### aclsparseSpMMGetBufferSize

#### 产品支持情况

- Ascend910_93 系列：支持
- Ascend910B 系列：支持
- Ascend 950PR / Ascend 950DT：不支持（请使用 `sparse/spmm/arch35/` 实现）

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpMMGetBufferSize(aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB, const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB, const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType, aclsparseSpMMAlg_t alg, size_t *size)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| opB | 输入 | aclsparseOperation_t | 稠密矩阵 B 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，类型须为 float（ACL_FLOAT）。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 A 的描述符，仅支持 CSR 格式，Host 内存 |
| matB | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 B 的描述符，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针，类型须为 float（ACL_FLOAT）。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matC | 输入/输出 | aclsparseDnMatDescr_t | 稠密矩阵 C 的描述符，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，仅支持 `ACL_FLOAT`，Host 内存 |
| alg | 输入 | aclsparseSpMMAlg_t | 算法类型，支持 DEFAULT、CSR_ALG1、FP32_HIGH_PRECISION、CSR_ALG2 和 CSR_ALG3，Host 内存 |
| size | 输出 | size_t* | 输出所需 workspace 大小（字节），Host 内存 |

#### 约束说明

- handle 不可为 nullptr
- matA、matB、matC 不可为 nullptr
- matA 仅支持 CSR 格式（`ACL_SPARSE_FORMAT_CSR`）
- matA 的行偏移和列索引类型必须均为 `ACL_SPARSE_INDEX_32I`，且两者类型相同
- matA 的索引基址仅支持 `ACL_SPARSE_INDEX_BASE_ZERO`
- opA 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`
- opB 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`（不支持转置）
- 数据类型组合仅支持一种：
  - matA=ACL_FLOAT, matB=ACL_FLOAT, matC=ACL_FLOAT, computeType=ACL_FLOAT
- 维度匹配：A.cols == B.rows，A.rows == C.rows，B.cols == C.cols
- matB、matC 仅支持 `ACL_SPARSE_ORDER_ROW`（紧凑行主序）且 `ld == cols`；不支持 COL-major 或带 padding 的 ld（kernel 硬编码紧凑行主序寻址，非紧凑布局会被校验拒绝，详见 `IsSupportedDnMatLayout`，多布局需求请使用 arch35 实现）
- size 不可为 nullptr

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 稀疏矩阵 A 支持 CSR 格式 |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

---

### aclsparseSpMMPreprocess

#### 产品支持情况

- Ascend910_93 系列：支持
- Ascend910B 系列：支持
- Ascend 950PR / Ascend 950DT：不支持（请使用 `sparse/spmm/arch35/` 实现）

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpMMPreprocess(aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB, const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB, const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType, aclsparseSpMMAlg_t alg, void *buffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| opB | 输入 | aclsparseOperation_t | 稠密矩阵 B 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，类型须为 float。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 A 的描述符，仅支持 CSR 格式，Host 内存 |
| matB | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 B 的描述符，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针，类型须为 float。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matC | 输入/输出 | aclsparseDnMatDescr_t | 稠密矩阵 C 的描述符，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，仅支持 `ACL_FLOAT`，Host 内存 |
| alg | 输入 | aclsparseSpMMAlg_t | 算法类型，Host 内存 |
| buffer | 输入 | void* | workspace 缓冲区（由 GetBufferSize 返回的大小分配），Device 内存 |

#### 约束说明

- 同 aclsparseSpMMGetBufferSize 的约束
- buffer 不可为 nullptr，需按 GetBufferSize 返回的大小分配

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 稀疏矩阵 A 支持 CSR 格式 |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

---

### aclsparseSpMM

#### 产品支持情况

- Ascend910_93 系列：支持
- Ascend910B 系列：支持
- Ascend 950PR / Ascend 950DT：不支持（请使用 `sparse/spmm/arch35/` 实现）

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpMM(aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB, const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB, const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType, aclsparseSpMMAlg_t alg, void *buffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| opB | 输入 | aclsparseOperation_t | 稠密矩阵 B 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，类型须为 float。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 A 的描述符，仅支持 CSR 格式，Host 内存 |
| matB | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 B 的描述符，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针，类型须为 float。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matC | 输入/输出 | aclsparseDnMatDescr_t | 稠密矩阵 C 的描述符，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，仅支持 `ACL_FLOAT`，Host 内存 |
| alg | 输入 | aclsparseSpMMAlg_t | 算法类型，Host 内存 |
| buffer | 输入 | void* | workspace 缓冲区（由 GetBufferSize 返回的大小分配），Device 内存 |

#### 约束说明

- 同 aclsparseSpMMGetBufferSize 的约束
- buffer 不可为 nullptr，需按 GetBufferSize 返回的大小分配
- 调用前须以相同的 CSR pattern、稠密矩阵形状/布局、opA/opB 和 alg 调用 `aclsparseSpMMPreprocess`。若该 buffer 被其他矩阵或参数组合重新预处理覆盖，须再次调用 Preprocess；执行接口不会同步回读 CSR 数据并自动重建预处理结果
- **异步执行**：本接口下发 kernel 后立即返回，不等待计算完成。调用方须自行调用 `aclrtSynchronizeStream` 等待结果，并在同步完成前保证 buffer 与矩阵数据有效

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 稀疏矩阵 A 支持 CSR 格式 |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/develop/compile_and_run_example.md)。

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

// 辅助：分配 Device 内存并拷贝 Host 数据
static void* AllocAndCopyDevice(const void *hostPtr, size_t sizeBytes)
{
    void *dPtr = nullptr;
    aclrtMalloc(&dPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (hostPtr != nullptr && sizeBytes > 0) {
        aclrtMemcpy(dPtr, sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return dPtr;
}

int aclsparseSpMMArch22Test(int32_t deviceId)
{
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t handle = nullptr;
    auto sparseRet = aclsparseCreate(&handle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    sparseRet = aclsparseSetStream(handle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    sparseRet = aclsparseSetPointerMode(handle, ACL_SPARSE_POINTER_MODE_HOST);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    // 2. 准备 Host 端 CSR 数据
    //    A (4x3, nnz=5):      B (3x2):
    //    [1.0  0.0  2.0]      [1.0  0.0]
    //    [0.0  3.0  0.0]      [0.0  1.0]
    //    [4.0  0.0  0.0]      [1.0  1.0]
    //    [0.0  0.0  5.0]
    //
    //    C = 1.0 * A * B (4x2):
    //    C[0] = [3.0, 2.0]
    //    C[1] = [0.0, 3.0]
    //    C[2] = [4.0, 0.0]
    //    C[3] = [5.0, 5.0]
    int64_t m = 4, k = 3, n = 2;
    int64_t nnzA = 5;
    float hAlpha = 1.0f;
    float hBeta = 0.0f;

    std::vector<int> hRowPtrA = {0, 2, 3, 4, 5};
    std::vector<int> hColIndA = {0, 2, 1, 0, 2};
    std::vector<float> hValA  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // B: 行主序 3x2
    int64_t ldb = n, ldc = n;
    std::vector<float> hB(static_cast<size_t>(k) * n, 0.0f);
    hB[0 * n + 0] = 1.0f; hB[0 * n + 1] = 0.0f;
    hB[1 * n + 0] = 0.0f; hB[1 * n + 1] = 1.0f;
    hB[2 * n + 0] = 1.0f; hB[2 * n + 1] = 1.0f;

    std::vector<float> hC(static_cast<size_t>(m) * n, 0.0f);

    // 3. 拷贝数据到 Device
    void *dRowPtrA = AllocAndCopyDevice(hRowPtrA.data(), (m + 1) * sizeof(int));
    void *dColIndA = AllocAndCopyDevice(hColIndA.data(), nnzA * sizeof(int));
    void *dValA    = AllocAndCopyDevice(hValA.data(),    nnzA * sizeof(float));
    void *dB       = AllocAndCopyDevice(hB.data(),       static_cast<size_t>(k) * n * sizeof(float));
    void *dC       = AllocAndCopyDevice(hC.data(),       static_cast<size_t>(m) * n * sizeof(float));

    // 4. 创建描述符
    aclsparseSpMatDescr_t matA = nullptr;
    sparseRet = aclsparseCreateCsr(&matA, m, k, nnzA, dRowPtrA, dColIndA, dValA,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    aclsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    sparseRet = aclsparseCreateDnMat(&matB, k, n, ldb, dB, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);
    sparseRet = aclsparseCreateDnMat(&matC, m, n, ldc, dC, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    // 5. Step 1 — GetBufferSize
    size_t bufferSize = 0;
    sparseRet = aclsparseSpMMGetBufferSize(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes\n", bufferSize);

    void *dBuffer = nullptr;
    auto aclRet = aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 6. Step 2 — Preprocess
    sparseRet = aclsparseSpMMPreprocess(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SPMM_ALG_DEFAULT, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    // 7. Step 3 — SpMM（异步，返回后需自行同步）
    sparseRet = aclsparseSpMM(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SPMM_ALG_DEFAULT, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    // 8. 同步等待计算完成（arch22 的 SpMM 为异步接口）
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 9. 将结果拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hC.data(), static_cast<size_t>(m) * n * sizeof(float),
                          dC, static_cast<size_t>(m) * n * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    for (int64_t i = 0; i < m; i++) {
        LOG_PRINT("C[%lld] = %.1f, %.1f\n", static_cast<long long>(i), hC[i * n + 0], hC[i * n + 1]);
    }

    // 10. 清理资源
    aclsparseDestroySpMat(matA);
    aclsparseDestroyDnMat(matB);
    aclsparseDestroyDnMat(matC);
    aclsparseDestroy(handle);
    if (dRowPtrA) aclrtFree(dRowPtrA);
    if (dColIndA) aclrtFree(dColIndA);
    if (dValA)    aclrtFree(dValA);
    if (dB)       aclrtFree(dB);
    if (dC)       aclrtFree(dC);
    if (dBuffer)  aclrtFree(dBuffer);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    return aclsparseSpMMArch22Test(0);
}
```

预期输出如下：

```
bufferSize = 512 bytes
C[0] = 3.0, 2.0
C[1] = 0.0, 3.0
C[2] = 4.0, 0.0
C[3] = 5.0, 5.0
```
