# Csrgeam2算子

## 算子概述

csrgeam2 算子用于计算两个 CSR（Compressed Sparse Row）格式稀疏矩阵的线性组合：

```
C = α * A + β * B
```

其中 A、B、C 均为 m × n 稀疏矩阵，以 CSR 格式存储；α 和 β 为 FP32 标量。

csrgeam2 采用**两步法**：
1. **Nnz 阶段**：计算输出矩阵 C 的行指针 `csrRowPtrC` 和非零元素总数 `nnzC`
2. **Compute 阶段**：根据 C 的行指针，计算列索引 `csrColIndC` 和非零元素值 `csrValC`

C 的非零模式为 A 和 B 的**并集**（A ∪ B），与 α、β 的具体数值无关。

**仅支持非转置（No-Transpose）模式**，不支持 NT/TN/TT 组合。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseScsrgeam2_bufferSizeExt | 查询 workspace 大小（字节） |
| aclsparseXcsrgeam2Nnz | 计算 C 的行指针 csrRowPtrC 和非零元素总数 nnzC |
| aclsparseScsrgeam2 | 执行 C = α * A + β * B，计算列索引和非零元素值 |

## 算子执行接口

### aclsparseScsrgeam2_bufferSizeExt

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseScsrgeam2_bufferSizeExt(
    aclsparseHandle_t         handle,
    int                       m,
    int                       n,
    const float              *alpha,
    const aclsparseMatDescr_t descrA,
    int                       nnzA,
    const float              *csrSortedValA,
    const int                *csrSortedRowPtrA,
    const int                *csrSortedColIndA,
    const float              *beta,
    const aclsparseMatDescr_t descrB,
    int                       nnzB,
    const float              *csrSortedValB,
    const int                *csrSortedRowPtrB,
    const int                *csrSortedColIndB,
    const aclsparseMatDescr_t descrC,
    const float              *csrSortedValC,
    const int                *csrSortedRowPtrC,
    const int                *csrSortedColIndC,
    size_t                   *pBufferSizeInBytes);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|---------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream、pointerMode 等信息，Host 内存 |
| m | 输入 | int | 矩阵 A、B、C 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 A、B、C 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float* | 标量 α 指针，用于 A 的乘法系数。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| descrA | 输入 | const aclsparseMatDescr_t | 矩阵 A 的描述符，仅支持 `ACL_SPARSE_MATRIX_TYPE_GENERAL` 类型，Host 内存 |
| nnzA | 输入 | int | 矩阵 A 的非零元素个数，nnzA >= 0，Host 内存 |
| csrSortedValA | 输入 | const float* | A 的非零元素值数组，长度为 nnzA（nnzA=0 时可为 nullptr，本函数不读取），Device 内存 |
| csrSortedRowPtrA | 输入 | const int* | A 的行指针数组，长度为 m+1（m=0 时可为 nullptr），Device 内存 |
| csrSortedColIndA | 输入 | const int* | A 的列索引数组，长度为 nnzA（nnzA=0 时可为 nullptr），Device 内存 |
| beta | 输入 | const float* | 标量 β 指针，用于 B 的乘法系数。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| descrB | 输入 | const aclsparseMatDescr_t | 矩阵 B 的描述符，仅支持 `ACL_SPARSE_MATRIX_TYPE_GENERAL` 类型，Host 内存 |
| nnzB | 输入 | int | 矩阵 B 的非零元素个数，nnzB >= 0，Host 内存 |
| csrSortedValB | 输入 | const float* | B 的非零元素值数组，长度为 nnzB（nnzB=0 时可为 nullptr，本函数不读取），Device 内存 |
| csrSortedRowPtrB | 输入 | const int* | B 的行指针数组，长度为 m+1（m=0 时可为 nullptr），Device 内存 |
| csrSortedColIndB | 输入 | const int* | B 的列索引数组，长度为 nnzB（nnzB=0 时可为 nullptr），Device 内存 |
| descrC | 输入 | const aclsparseMatDescr_t | 矩阵 C 的描述符，仅支持 `ACL_SPARSE_MATRIX_TYPE_GENERAL` 类型，Host 内存 |
| csrSortedValC | 输入 | const float* | C 的非零元素值数组（可为 nullptr，本函数不读取），Device 内存 |
| csrSortedRowPtrC | 输入 | const int* | C 的行指针数组（可为 nullptr，本函数不读取），Device 内存 |
| csrSortedColIndC | 输入 | const int* | C 的列索引数组（可为 nullptr，本函数不读取），Device 内存 |
| pBufferSizeInBytes | 输出 | size_t* | 输出所需 workspace 大小（字节），Host 内存 |

#### 约束说明

- m >= 0，n >= 0
- nnzA >= 0，nnzB >= 0
- handle 不可为 nullptr
- descrA / descrB / descrC 不可为 nullptr，且 type 仅支持 `ACL_SPARSE_MATRIX_TYPE_GENERAL`
- descrA / descrB / descrC 的 indexBase 必须为 `ACL_SPARSE_INDEX_BASE_ZERO` 或 `ACL_SPARSE_INDEX_BASE_ONE`
- alpha / beta 不可为 nullptr
- 当 m > 0 时，csrSortedRowPtrA / csrSortedRowPtrB 不可为 nullptr
- 当 nnzA > 0 时，csrSortedColIndA 不可为 nullptr
- 当 nnzB > 0 时，csrSortedColIndB 不可为 nullptr
- pBufferSizeInBytes 不可为 nullptr
- workspace 大小公式：`(m + 1) * sizeof(int32_t)`（前 m 个 int32_t 用于 nnzPerRow，第 m+1 个用于 nnzC）
- csrSortedValA / csrSortedValB / csrSortedValC / csrSortedRowPtrC / csrSortedColIndC 可为 nullptr（本函数不读取这些指针，仅查询 workspace 大小）

---

### aclsparseXcsrgeam2Nnz

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseXcsrgeam2Nnz(
    aclsparseHandle_t         handle,
    int                       m,
    int                       n,
    const aclsparseMatDescr_t descrA,
    int                       nnzA,
    const int                *csrSortedRowPtrA,
    const int                *csrSortedColIndA,
    const aclsparseMatDescr_t descrB,
    int                       nnzB,
    const int                *csrSortedRowPtrB,
    const int                *csrSortedColIndB,
    const aclsparseMatDescr_t descrC,
    int                      *csrSortedRowPtrC,
    int                      *nnzTotalDevHostPtr,
    void                     *workspace);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|---------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，Host 内存 |
| m | 输入 | int | 矩阵行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵列数，n >= 0，Host 内存 |
| descrA | 输入 | const aclsparseMatDescr_t | 矩阵 A 的描述符，Host 内存 |
| nnzA | 输入 | int | 矩阵 A 的非零元素个数，Host 内存 |
| csrSortedRowPtrA | 输入 | const int* | A 的行指针数组，长度为 m+1，Device 内存 |
| csrSortedColIndA | 输入 | const int* | A 的列索引数组，长度为 nnzA，Device 内存 |
| descrB | 输入 | const aclsparseMatDescr_t | 矩阵 B 的描述符，Host 内存 |
| nnzB | 输入 | int | 矩阵 B 的非零元素个数，Host 内存 |
| csrSortedRowPtrB | 输入 | const int* | B 的行指针数组，长度为 m+1，Device 内存 |
| csrSortedColIndB | 输入 | const int* | B 的列索引数组，长度为 nnzB，Device 内存 |
| descrC | 输入 | const aclsparseMatDescr_t | 矩阵 C 的描述符，Host 内存 |
| csrSortedRowPtrC | 输出 | int* | C 的行指针数组（输出），长度为 m+1。调用前需分配，Device 内存 |
| nnzTotalDevHostPtr | 输出 | int* | C 的非零元素总数 nnzC。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| workspace | 输入 | void* | workspace 缓冲区（由 bufferSizeExt 返回的大小分配），Device 内存 |

#### 约束说明

- m >= 0，n >= 0
- nnzA >= 0，nnzB >= 0
- handle 不可为 nullptr
- descrA / descrB / descrC 不可为 nullptr，且 type 仅支持 `ACL_SPARSE_MATRIX_TYPE_GENERAL`
- descrA / descrB / descrC 的 indexBase 必须为 `ACL_SPARSE_INDEX_BASE_ZERO` 或 `ACL_SPARSE_INDEX_BASE_ONE`
- 当 m > 0 时，csrSortedRowPtrA / csrSortedRowPtrB 不可为 nullptr
- 当 nnzA > 0 时，csrSortedColIndA 不可为 nullptr
- 当 nnzB > 0 时，csrSortedColIndB 不可为 nullptr
- csrSortedRowPtrC 不可为 nullptr（调用前需分配 m+1 大小的 int 数组）
- nnzTotalDevHostPtr 不可为 nullptr
- workspace 不可为 nullptr（需按 bufferSizeExt 返回的大小分配）
- 当 m = 0 或 n = 0 时，函数直接返回成功，nnzC = 0
- **注意**：此函数使用 X 前缀（大写），表示与数据类型无关（不涉及值数组，仅操作索引结构）

---

### aclsparseScsrgeam2

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseScsrgeam2(
    aclsparseHandle_t         handle,
    int                       m,
    int                       n,
    const float              *alpha,
    const aclsparseMatDescr_t descrA,
    int                       nnzA,
    const float              *csrSortedValA,
    const int                *csrSortedRowPtrA,
    const int                *csrSortedColIndA,
    const float              *beta,
    const aclsparseMatDescr_t descrB,
    int                       nnzB,
    const float              *csrSortedValB,
    const int                *csrSortedRowPtrB,
    const int                *csrSortedColIndB,
    const aclsparseMatDescr_t descrC,
    float                    *csrSortedValC,
    int                      *csrSortedRowPtrC,
    int                      *csrSortedColIndC,
    void                     *pBuffer);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|---------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，Host 内存 |
| m | 输入 | int | 矩阵行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵列数，n >= 0，Host 内存 |
| alpha | 输入 | const float* | 标量 α 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| descrA | 输入 | const aclsparseMatDescr_t | 矩阵 A 的描述符，Host 内存 |
| nnzA | 输入 | int | 矩阵 A 的非零元素个数，Host 内存 |
| csrSortedValA | 输入 | const float* | A 的非零元素值数组，长度为 nnzA，Device 内存 |
| csrSortedRowPtrA | 输入 | const int* | A 的行指针数组，长度为 m+1，Device 内存 |
| csrSortedColIndA | 输入 | const int* | A 的列索引数组，长度为 nnzA，Device 内存 |
| beta | 输入 | const float* | 标量 β 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| descrB | 输入 | const aclsparseMatDescr_t | 矩阵 B 的描述符，Host 内存 |
| nnzB | 输入 | int | 矩阵 B 的非零元素个数，Host 内存 |
| csrSortedValB | 输入 | const float* | B 的非零元素值数组，长度为 nnzB，Device 内存 |
| csrSortedRowPtrB | 输入 | const int* | B 的行指针数组，长度为 m+1，Device 内存 |
| csrSortedColIndB | 输入 | const int* | B 的列索引数组，长度为 nnzB，Device 内存 |
| descrC | 输入 | const aclsparseMatDescr_t | 矩阵 C 的描述符，Host 内存 |
| csrSortedValC | 输出 | float* | C 的非零元素值数组（输出），长度为 nnzC，Device 内存 |
| csrSortedRowPtrC | 输入 | int* | C 的行指针数组（由 Nnz 阶段填充），长度为 m+1，Device 内存 |
| csrSortedColIndC | 输出 | int* | C 的列索引数组（输出），长度为 nnzC，Device 内存 |
| pBuffer | 输入 | void* | workspace 缓冲区（由 bufferSizeExt 返回的大小分配），Device 内存 |

#### 约束说明

- m >= 0，n >= 0
- nnzA >= 0，nnzB >= 0
- handle 不可为 nullptr
- descrA / descrB / descrC 不可为 nullptr，且 type 仅支持 `ACL_SPARSE_MATRIX_TYPE_GENERAL`
- descrA / descrB / descrC 的 indexBase 必须为 `ACL_SPARSE_INDEX_BASE_ZERO` 或 `ACL_SPARSE_INDEX_BASE_ONE`
- 当 m > 0 时，csrSortedRowPtrA / csrSortedRowPtrB / csrSortedRowPtrC 不可为 nullptr
- 当 nnzA > 0 时，csrSortedValA / csrSortedColIndA 不可为 nullptr
- 当 nnzB > 0 时，csrSortedValB / csrSortedColIndB 不可为 nullptr
- csrSortedValC / csrSortedColIndC 不可为 nullptr（需按 nnzC 分配）
- pBuffer 不可为 nullptr
- 调用前必须先调用 `aclsparseXcsrgeam2Nnz` 填充 `csrRowPtrC` 和获取 `nnzC`
- 当 m = 0 或 n = 0 时，函数直接返回成功

### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | A、B、C 三个矩阵均为 CSR 格式 |
| COO | ❌ | csrgeam2 仅支持 CSR |
| CSC | ❌ | csrgeam2 仅支持 CSR |

### CSR 格式索引结构

| 数组 | 长度 | 说明 |
|------|------|------|
| csrRowPtr | m + 1 | 行偏移指针，csrRowPtr[i] 是第 i 行非零元素的起始索引 |
| csrColInd | nnz | 列索引数组 |
| csrVal | nnz | 非零元素值数组 |

### 索引基址

通过 MatDescr 的 `indexBase` 字段指定：

- `ACL_SPARSE_INDEX_BASE_ZERO`：0-based（C 风格）
- `ACL_SPARSE_INDEX_BASE_ONE`：1-based（Fortran 风格）

A、B、C 的索引基址可以**不同**，各自通过独立的 MatDescr 指定。

### 数据类型

| 精度前缀 | 计算类型 | 值类型 | 说明 |
|---------|---------|--------|------|
| S | FP32 | FP32 | 当前唯一支持的精度（S 前缀） |

**索引类型**：int32（32 位有符号整数）。

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
static void* AllocAndCopyDevice(void *hostPtr, size_t sizeBytes)
{
    void *dPtr = nullptr;
    aclrtMalloc(&dPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (hostPtr != nullptr && sizeBytes > 0) {
        aclrtMemcpy(dPtr, sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return dPtr;
}

int aclsparseCsrgeam2Test(AclContext &ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreate failed: %d\n", sparseRet);
              return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(rawHandle, aclsparseDestroy);
    aclsparseSetStream(handlePtr.get(), stream);
    aclsparseSetPointerMode(handlePtr.get(), ACL_SPARSE_POINTER_MODE_HOST);

    // 2. 准备 Host 端 CSR 数据
    //    示例：两个 3x3 的稀疏矩阵 A 和 B
    //
    //    A (3x3, nnzA=4):      B (3x3, nnzB=2):
    //    [1.0  0.0  2.0]      [0.0  0.0  5.0]
    //    [0.0  4.0  0.0]      [6.0  0.0  0.0]
    //    [3.0  0.0  0.0]      [0.0  0.0  0.0]
    //
    //    C = 1.0 * A + 1.0 * B:
    //    C[0] = [1.0,  0.0,  7.0 ]  (nnz: 2)
    //    C[1] = [6.0,  4.0,  0.0 ]  (nnz: 2)
    //    C[2] = [3.0,  0.0,  0.0 ]  (nnz: 1)
    //    => nnzC = 5

    int m = 3;
    int n = 3;
    int nnzA = 4;
    int nnzB = 2;
    float hAlpha = 1.0f;
    float hBeta = 1.0f;

    // A: CSR 数据 (3x3, nnzA=4)
    std::vector<int> hRowPtrA = {0, 2, 3, 4};
    std::vector<int> hColIndA = {0, 2, 1, 0};
    std::vector<float> hValA  = {1.0f, 2.0f, 4.0f, 3.0f};

    // B: CSR 数据 (3x3, nnzB=2)
    std::vector<int> hRowPtrB = {0, 1, 2, 2};
    std::vector<int> hColIndB = {2, 0};
    std::vector<float> hValB  = {5.0f, 6.0f};

    // 3. 创建 MatDescr（A/B/C 各自独立）
    aclsparseMatDescr_t descrA = nullptr, descrB = nullptr, descrC = nullptr;
    aclsparseCreateMatDescr(&descrA);
    aclsparseCreateMatDescr(&descrB);
    aclsparseCreateMatDescr(&descrC);
    aclsparseSetMatType(descrA, ACL_SPARSE_MATRIX_TYPE_GENERAL);
    aclsparseSetMatType(descrB, ACL_SPARSE_MATRIX_TYPE_GENERAL);
    aclsparseSetMatType(descrC, ACL_SPARSE_MATRIX_TYPE_GENERAL);
    aclsparseSetMatIndexBase(descrA, ACL_SPARSE_INDEX_BASE_ZERO);
    aclsparseSetMatIndexBase(descrB, ACL_SPARSE_INDEX_BASE_ZERO);
    aclsparseSetMatIndexBase(descrC, ACL_SPARSE_INDEX_BASE_ZERO);

    // 4. 拷贝 A/B 的 CSR 数据到 Device
    void *dRowPtrA = AllocAndCopyDevice(hRowPtrA.data(), (m + 1) * sizeof(int));
    void *dColIndA = AllocAndCopyDevice(hColIndA.data(), nnzA * sizeof(int));
    void *dValA    = AllocAndCopyDevice(hValA.data(),    nnzA * sizeof(float));

    void *dRowPtrB = AllocAndCopyDevice(hRowPtrB.data(), (m + 1) * sizeof(int));
    void *dColIndB = AllocAndCopyDevice(hColIndB.data(), nnzB * sizeof(int));
    void *dValB    = AllocAndCopyDevice(hValB.data(),    nnzB * sizeof(float));

    // 5. Step 1 — bufferSizeExt: 查询 workspace 大小
    size_t bufferSize = 0;
    sparseRet = aclsparseScsrgeam2_bufferSizeExt(
        handlePtr.get(), m, n,
        &hAlpha,
        descrA, nnzA,
        static_cast<const float*>(dValA),
        static_cast<const int*>(dRowPtrA),
        static_cast<const int*>(dColIndA),
        &hBeta,
        descrB, nnzB,
        static_cast<const float*>(dValB),
        static_cast<const int*>(dRowPtrB),
        static_cast<const int*>(dColIndB),
        descrC, nullptr, nullptr, nullptr,
        &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("bufferSizeExt failed: %d\n", sparseRet);
              return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes\n", bufferSize);

    // 分配 workspace
    void *dWorkspace = nullptr;
    aclrtMalloc(&dWorkspace, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 6. Step 2 — Xcsrgeam2Nnz: 计算 C 的 csrRowPtrC 和 nnzC
    // 预分配 csrRowPtrC (m+1)
    void *dRowPtrC = nullptr;
    aclrtMalloc(&dRowPtrC, (m + 1) * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);

    int nnzC = 0;  // HOST 模式下，nnzC 直接写入 host 变量
    sparseRet = aclsparseXcsrgeam2Nnz(
        handlePtr.get(), m, n,
        descrA, nnzA,
        static_cast<const int*>(dRowPtrA),
        static_cast<const int*>(dColIndA),
        descrB, nnzB,
        static_cast<const int*>(dRowPtrB),
        static_cast<const int*>(dColIndB),
        descrC,
        static_cast<int*>(dRowPtrC),
        &nnzC,       // HOST 指针
        dWorkspace);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("Xcsrgeam2Nnz failed: %d\n", sparseRet);
              return sparseRet);
    LOG_PRINT("nnzC = %d\n", nnzC);

    // 7. Step 3 — Scsrgeam2: 计算 C 的列索引和值
    void *dColIndC = nullptr;
    void *dValC    = nullptr;
    aclrtMalloc(&dColIndC, nnzC * sizeof(int),   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dValC,    nnzC * sizeof(float),  ACL_MEM_MALLOC_HUGE_FIRST);

    // 分配主计算 pBuffer（可与 Nnz 阶段的 workspace 共用或独立分配）
    void *dPBuffer = nullptr;
    aclrtMalloc(&dPBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);

    sparseRet = aclsparseScsrgeam2(
        handlePtr.get(), m, n,
        &hAlpha,
        descrA, nnzA,
        static_cast<const float*>(dValA),
        static_cast<const int*>(dRowPtrA),
        static_cast<const int*>(dColIndA),
        &hBeta,
        descrB, nnzB,
        static_cast<const float*>(dValB),
        static_cast<const int*>(dRowPtrB),
        static_cast<const int*>(dColIndB),
        descrC,
        static_cast<float*>(dValC),
        static_cast<int*>(dRowPtrC),
        static_cast<int*>(dColIndC),
        dPBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("Scsrgeam2 failed: %d\n", sparseRet);
              return sparseRet);

    // 8. 同步等待计算完成
    aclrtSynchronizeStream(stream);

    // 9. 将结果拷贝回 Host 并打印
    std::vector<int>   hRowPtrC(m + 1, 0);
    std::vector<int>   hColIndC(nnzC, 0);
    std::vector<float> hValC(nnzC, 0.0f);

    aclrtMemcpy(hRowPtrC.data(), (m + 1) * sizeof(int),  dRowPtrC, (m + 1) * sizeof(int),  ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hColIndC.data(), nnzC * sizeof(int),      dColIndC, nnzC * sizeof(int),    ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hValC.data(),    nnzC * sizeof(float),    dValC,    nnzC * sizeof(float),  ACL_MEMCPY_DEVICE_TO_HOST);

    LOG_PRINT("\nResult C (CSR):\n");
    LOG_PRINT("  rowPtr: ");
    for (int i = 0; i <= m; i++) {
        LOG_PRINT("%d ", hRowPtrC[i]);
    }
    LOG_PRINT("\n  colInd: ");
    for (int i = 0; i < nnzC; i++) {
        LOG_PRINT("%d ", hColIndC[i]);
    }
    LOG_PRINT("\n  val:    ");
    for (int i = 0; i < nnzC; i++) {
        LOG_PRINT("%.1f ", hValC[i]);
    }
    LOG_PRINT("\n");

    // 10. 清理资源
    aclsparseDestroyMatDescr(descrA);
    aclsparseDestroyMatDescr(descrB);
    aclsparseDestroyMatDescr(descrC);

    if (dRowPtrA)  aclrtFree(dRowPtrA);
    if (dColIndA)  aclrtFree(dColIndA);
    if (dValA)     aclrtFree(dValA);
    if (dRowPtrB)  aclrtFree(dRowPtrB);
    if (dColIndB)  aclrtFree(dColIndB);
    if (dValB)     aclrtFree(dValB);
    if (dRowPtrC)  aclrtFree(dRowPtrC);
    if (dColIndC)  aclrtFree(dColIndC);
    if (dValC)     aclrtFree(dValC);
    if (dWorkspace) aclrtFree(dWorkspace);
    if (dPBuffer)   aclrtFree(dPBuffer);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseCsrgeam2Test(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCsrgeam2Test failed: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
bufferSize = 16 bytes
nnzC = 5

Result C (CSR):
  rowPtr: 0 2 4 5 
  colInd: 0 2 0 1 0 
  val:    1.0 7.0 6.0 4.0 3.0 
```
