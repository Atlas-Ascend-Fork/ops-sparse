# SpMM算子

## 算子概述

SpMM（Sparse Matrix - Dense Matrix Multiplication）算子实现稀疏矩阵与稠密矩阵的乘法运算。核心运算为 C = alpha * op(A) * op(B) + beta * C，其中 A 为 CSR 格式稀疏矩阵，B 和 C 为稠密矩阵。

数学表达式：

```
C = alpha * op(A) * op(B) + beta * C
```

其中 op(A) 为稀疏矩阵 A 的操作（转置/非转置），op(B) 为稠密矩阵 B 的操作。

调用流程为三步法：
1. **GetBufferSize**：查询所需 workspace 大小
2. **Preprocess（可选）**：对稀疏矩阵进行预处理（行重排 + 分箱），加速后续计算；省略时首次 SpMM 自动构建
3. **SpMM**：执行 C = alpha * op(A) * op(B) + beta * C

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSpMMGetBufferSize | 查询 SpMM 所需 workspace 大小（字节） |
| aclsparseSpMMPreprocess | 对稀疏矩阵进行预处理（行重排 + 分箱） |
| aclsparseSpMM | 执行稀疏矩阵-稠密矩阵乘法 C = alpha * op(A) * op(B) + beta * C |

## 算子执行接口

### aclsparseSpMMGetBufferSize

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpMMGetBufferSize(aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB, const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB, const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType, aclsparseSpMMAlg_t alg, size_t *size)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型；支持非转置、转置和共轭转置，CSR_ALG3 仅支持非转置，Host 内存 |
| opB | 输入 | aclsparseOperation_t | 稠密矩阵 B 的操作类型；支持非转置、转置及合法算法组合下的共轭转置，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，类型须与 computeType 匹配。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 A 的描述符，仅支持 CSR 格式，Host 内存 |
| matB | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 B 的描述符，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针，类型须与 computeType 匹配。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matC | 输入/输出 | aclsparseDnMatDescr_t | 稠密矩阵 C 的描述符，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型；实数使用 `ACL_FLOAT`，复数使用 `ACL_COMPLEX64`；Ascend 950 的 INT8 组合使用 `ACL_INT32`，Host 内存 |
| alg | 输入 | aclsparseSpMMAlg_t | 算法类型；A2/A3 支持 DEFAULT、CSR_ALG1、CSR_ALG2、CSR_ALG3 和 FP32_HIGH_PRECISION，Host 内存；不同算法会选择对应的布局/确定性路径 |
| size | 输出 | size_t* | 输出所需 workspace 大小（字节），Host 内存 |

#### 约束说明

- handle 不可为 nullptr
- matA、matB、matC 不可为 nullptr
- matA 仅支持 CSR 格式（`ACL_SPARSE_FORMAT_CSR`）
- matA 的行偏移和列索引类型必须均为 `ACL_SPARSE_INDEX_32I`，且两者类型相同
##### Ascend 950

- matA 的索引基址支持 `ACL_SPARSE_INDEX_BASE_ZERO` 和 `ACL_SPARSE_INDEX_BASE_ONE`
- opA 支持非转置、转置和共轭转置；CSR_ALG3 仅支持非转置
- opB 支持 `ACL_SPARSE_OP_NON_TRANSPOSE`、`ACL_SPARSE_OP_TRANSPOSE` 和 `ACL_SPARSE_OP_CONJUGATE_TRANSPOSE`；CSR_ALG3 不支持共轭转置
- 支持以下数据类型组合：
  - matA=ACL_FLOAT, matB=ACL_FLOAT, matC=ACL_FLOAT, computeType=ACL_FLOAT
  - matA=ACL_FLOAT16, matB=ACL_FLOAT16, matC=ACL_FLOAT16, computeType=ACL_FLOAT
  - matA=ACL_FLOAT16, matB=ACL_FLOAT16, matC=ACL_FLOAT, computeType=ACL_FLOAT
  - matA=ACL_BF16, matB=ACL_BF16, matC=ACL_BF16, computeType=ACL_FLOAT
  - matA=ACL_BF16, matB=ACL_BF16, matC=ACL_FLOAT, computeType=ACL_FLOAT
  - matA=ACL_COMPLEX64, matB=ACL_COMPLEX64, matC=ACL_COMPLEX64, computeType=ACL_COMPLEX64
  - matA=ACL_INT8, matB=ACL_INT8, matC=ACL_INT32, computeType=ACL_INT32
- 设 op(A) 的逻辑形状为 `m x k`，C 的形状为 `m x n`：
  - `opB=ACL_SPARSE_OP_NON_TRANSPOSE` 时，B 描述符形状为 `k x n`；
  - `opB=ACL_SPARSE_OP_TRANSPOSE` 时，B 描述符形状为 `n x k`；
  - 行主序矩阵的 `ld` 不小于物理列数，列主序矩阵的 `ld` 不小于物理行数。
- 零维输入：m / k / n（matA.rows / matA.cols / matC.cols）任一为 0 时返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- opA 为转置或共轭转置时，逻辑形状为原始 A 的列数乘行数；维度校验和 workspace 大小均按逻辑形状计算。
- 转置预处理保存 CSR 结构和原始 values 的置换关系，执行时读取当前 values；Complex64 共轭转置同时取复数共轭。
- 非零维矩阵允许 `nnz=0` 和空行。

##### A2/A3

- A2/A3 的 matA 索引基址支持 `ACL_SPARSE_INDEX_BASE_ZERO` 和 `ACL_SPARSE_INDEX_BASE_ONE`
- A2/A3 的 opA/opB 支持 `ACL_SPARSE_OP_NON_TRANSPOSE`、`ACL_SPARSE_OP_TRANSPOSE` 和
  `ACL_SPARSE_OP_CONJUGATE_TRANSPOSE`
- A2/A3 的 B/C 支持行主序和列主序；行主序 `ld >= max(1, cols)`，列主序
  `ld >= max(1, rows)`
- A2/A3 数据类型组合支持以下四种：
  - matA=ACL_FLOAT, matB=ACL_FLOAT, matC=ACL_FLOAT, computeType=ACL_FLOAT
  - matA=ACL_FLOAT16, matB=ACL_FLOAT16, matC=ACL_FLOAT16, computeType=ACL_FLOAT
  - matA=ACL_BF16, matB=ACL_BF16, matC=ACL_BF16, computeType=ACL_FLOAT
  - matA=ACL_COMPLEX64, matB=ACL_COMPLEX64, matC=ACL_COMPLEX64, computeType=ACL_COMPLEX64
- 维度匹配按 opA/opB 后的逻辑形状校验
- M/K/N 均允许为 0；零长度输出直接成功返回，K=0 时结果遵循 `C = beta * C`
- A2/A3 的 `CSR_ALG1` 使用布局感知通用路径，优先用于列主序 B/C；`CSR_ALG2` 在紧凑行主序
  N/N 组合下使用行主序快路径，其余合法布局回退通用路径；`CSR_ALG3` 使用固定原始行序，
  要求 opA 为非转置且 opB 不是共轭转置
- A2/A3 的 `FP32_HIGH_PRECISION` 仅支持 fp32，使用 Kahan 补偿求和
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

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpMMPreprocess(aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB, const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB, const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType, aclsparseSpMMAlg_t alg, void *buffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型；支持非转置、转置和共轭转置，CSR_ALG3 仅支持非转置，Host 内存 |
| opB | 输入 | aclsparseOperation_t | 稠密矩阵 B 的操作类型；支持非转置、转置及合法算法组合下的共轭转置，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 A 的描述符，仅支持 CSR 格式，Host 内存 |
| matB | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 B 的描述符，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matC | 输入/输出 | aclsparseDnMatDescr_t | 稠密矩阵 C 的描述符，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型；实数使用 `ACL_FLOAT`，复数使用 `ACL_COMPLEX64`；Ascend 950 的 INT8 组合使用 `ACL_INT32`，Host 内存 |
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

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpMM(aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB, const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB, const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType, aclsparseSpMMAlg_t alg, void *buffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型；支持非转置、转置和共轭转置，CSR_ALG3 仅支持非转置，Host 内存 |
| opB | 输入 | aclsparseOperation_t | 稠密矩阵 B 的操作类型；支持非转置、转置及合法算法组合下的共轭转置，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 A 的描述符，仅支持 CSR 格式，Host 内存 |
| matB | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 B 的描述符，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matC | 输入/输出 | aclsparseDnMatDescr_t | 稠密矩阵 C 的描述符，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型；实数使用 `ACL_FLOAT`，复数使用 `ACL_COMPLEX64`；Ascend 950 的 INT8 组合使用 `ACL_INT32`，Host 内存 |
| alg | 输入 | aclsparseSpMMAlg_t | 算法类型，Host 内存 |
| buffer | 输入 | void* | workspace 缓冲区（由 GetBufferSize 返回的大小分配），Device 内存 |

#### 约束说明

- 同 aclsparseSpMMGetBufferSize 的约束
- buffer 不可为 nullptr，需按 GetBufferSize 返回的大小分配
- A2/A3 调用前须先调用 `aclsparseSpMMPreprocess` 进行预处理
- Ascend 950 的 `aclsparseSpMMPreprocess` 为可选步骤，首次直接执行会构建 workspace
- A2/A3 支持 DEFAULT、CSR_ALG1、CSR_ALG2、CSR_ALG3、FP32_HIGH_PRECISION；算法会参与 host
  侧路径选择，CSR_ALG3 的操作限制见 GetBufferSize 约束
- A2/A3 的 `ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG` 仅支持 fp32，使用 Kahan 补偿求和

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 稀疏矩阵 A 支持 CSR 格式 |
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
    aclrtMalloc(&dPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (hostPtr != nullptr && sizeBytes > 0) {
        aclrtMemcpy(dPtr, sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return dPtr;
}

int aclsparseSpMMTest(AclContext& ctx)
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

    // 3. 准备 Host 端 CSR 数据
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
    aclsparseOrder_t orderB = ACL_SPARSE_ORDER_ROW;
    aclsparseOrder_t orderC = ACL_SPARSE_ORDER_ROW;
    std::vector<float> hB(static_cast<size_t>(k) * n, 0.0f);
    hB[0 * n + 0] = 1.0f; hB[0 * n + 1] = 0.0f;
    hB[1 * n + 0] = 0.0f; hB[1 * n + 1] = 1.0f;
    hB[2 * n + 0] = 1.0f; hB[2 * n + 1] = 1.0f;

    std::vector<float> hC(static_cast<size_t>(m) * n, 0.0f);

    // 4. 拷贝数据到 Device
    void *dRowPtrA = AllocAndCopyDevice(hRowPtrA.data(), (m + 1) * sizeof(int));
    void *dColIndA = AllocAndCopyDevice(hColIndA.data(), nnzA * sizeof(int));
    void *dValA    = AllocAndCopyDevice(hValA.data(),    nnzA * sizeof(float));
    void *dB       = AllocAndCopyDevice(hB.data(),       static_cast<size_t>(k) * n * sizeof(float));
    void *dC       = AllocAndCopyDevice(hC.data(),       static_cast<size_t>(m) * n * sizeof(float));

    // 5. 创建描述符
    aclsparseSpMatDescr_t matA = nullptr;
    sparseRet = aclsparseCreateCsr(&matA, m, k, nnzA, dRowPtrA, dColIndA, dValA,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    aclsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    sparseRet = aclsparseCreateDnMat(&matB, k, n, ldb, dB, ACL_FLOAT, orderB);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateDnMat B failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    sparseRet = aclsparseCreateDnMat(&matC, m, n, ldc, dC, ACL_FLOAT, orderC);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateDnMat C failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 6. Step 1 — GetBufferSize
    size_t bufferSize = 0;
    sparseRet = aclsparseSpMMGetBufferSize(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SPMM_CSR_ALG1, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpMMGetBufferSize failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes\n", bufferSize);

    void *dBuffer = nullptr;
    auto aclRet = aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for buffer failed. ERROR: %d\n", aclRet); return aclRet);

    // 7. Step 2 — Preprocess
    sparseRet = aclsparseSpMMPreprocess(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SPMM_CSR_ALG1, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpMMPreprocess failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 8. Step 3 — SpMM
    sparseRet = aclsparseSpMM(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SPMM_CSR_ALG1, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpMM failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 9. 同步等待计算完成
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 10. 将结果拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hC.data(), static_cast<size_t>(m) * n * sizeof(float),
                          dC, static_cast<size_t>(m) * n * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);

    for (int64_t i = 0; i < m; i++) {
        LOG_PRINT("C[%lld] = %.1f, %.1f\n", static_cast<long long>(i), hC[i * n + 0], hC[i * n + 1]);
    }

    // 11. 清理资源
    aclsparseDestroySpMat(matA);
    aclsparseDestroyDnMat(matB);
    aclsparseDestroyDnMat(matC);
    if (dRowPtrA) aclrtFree(dRowPtrA);
    if (dColIndA) aclrtFree(dColIndA);
    if (dValA)    aclrtFree(dValA);
    if (dB)       aclrtFree(dB);
    if (dC)       aclrtFree(dC);
    if (dBuffer)  aclrtFree(dBuffer);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseSpMMTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSpMMTest failed. ERROR: %d\n", ret); return ret);
    return 0;
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

## PyTorch/ATen 入口

torch_extension 适配位于 `sparse/spmm/torch_extension/`，注册 `aten::addmm.out` 的
`SparseCsrPrivateUse1` 实现，使标准公开入口直接走 Dispatcher 到 `aclsparseSpMM*`，
不经过 CPU 回填：

```python
import torch
import torch_npu
import cann_ops_sparse  # 导入即完成注册

out = torch.addmm(bias, csr_mat1, dense_mat2, beta=-0.5, alpha=1.5)
```

调用链为 `torch.addmm` → `aten::addmm.out` → SparseCsrPrivateUse1 实现 →
`aclsparseSpMMGetBufferSize`/`Preprocess`/`aclsparseSpMM` → Ascend C Kernel。

该公开入口固定 opA/opB 为非转置、索引基址为 0（PyTorch 的 `sparse_csr_tensor` 和
`torch.addmm` 都没有暴露这两个控制项），支持 fp16/bf16/fp32/complex64 四种 dtype、
可广播的 `self`、以及零维 M/K/N。转置与 1 基索引仍可通过上面的裸 C++ 接口使用。

安装、参数与约束详见 [torch_extension/cann_ops_sparse/docs/zh/spmm.md](../../torch_extension/cann_ops_sparse/docs/zh/spmm.md)，
其中也说明了当前 `torch.sparse.addmm` 稀疏×稠密调用会被 SpGEMM 的注册遮蔽、需改用
`torch.addmm` 的已知问题。
