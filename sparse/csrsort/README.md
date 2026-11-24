# CSRSORT算子

## 算子概述

csrsort算子用于对CSR（Compressed Sparse Row，压缩稀疏行）格式稀疏矩阵的每一行执行稳定排序。算子原地重排列索引数组`csrColInd`，并使用相同的排列重排数组`P`；行偏移数组`csrRowPtr`保持不变。

对于第`row`行，排序区间和输出关系如下：

```text
begin = csrRowPtr[row] - indexBase
end   = csrRowPtr[row + 1] - indexBase
csrColInd_out[begin:end] = stable_sort(csrColInd_in[begin:end])
P_out[begin:end] = P_in[stable_permutation(begin:end)]
```

当调用方将`P`初始化为`0, 1, ..., nnz - 1`时，排序后的`P[i]`表示输出位置`i`对应的原始非零元素下标，可据此同步重排稀疏矩阵的值数组。

算子采用两步调用流程：先调用`aclsparseXcsrsort_bufferSizeExt`查询workspace大小并分配Device内存，再调用`aclsparseXcsrsort`执行排序。排序接口异步下发到handle绑定的stream，读取结果前需要同步该stream。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseXcsrsort_bufferSizeExt | 查询排序所需的Device workspace大小 |
| aclsparseXcsrsort | 对CSR矩阵每行的列索引执行原地稳定升序排序，并同步重排`P` |

## 算子执行接口

### aclsparseXcsrsort_bufferSizeExt

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseXcsrsort_bufferSizeExt(
    aclsparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const int *csrRowPtr,
    const int *csrColInd,
    size_t *pBufferSizeInBytes)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse库上下文句柄 | Host |
| m | 输入 | int | 稀疏矩阵行数 | Host |
| n | 输入 | int | 稀疏矩阵列数；仅用于Host参数校验，不参与排序计算 | Host |
| nnz | 输入 | int | 稀疏矩阵非零元素个数 | Host |
| csrRowPtr | 输入 | const int*（INT32） | CSR行偏移数组，长度为`m + 1` | Device |
| csrColInd | 输入 | const int*（INT32） | CSR列索引数组，长度为`nnz` | Device |
| pBufferSizeInBytes | 输出 | size_t* | 返回workspace大小，单位为字节 | Host |

#### 约束说明

- `handle`不可为nullptr，否则返回`ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`。
- `m >= 0`、`n >= 0`且`nnz >= 0`。
- 当`m == 0`或`n == 0`时，`nnz`必须为0。
- 当`m > 0`时，`csrRowPtr`不可为nullptr。
- 当`nnz > 0`时，`csrColInd`不可为nullptr。
- `pBufferSizeInBytes`不可为nullptr。
- 当`nnz == 0`时，接口返回成功并将`*pBufferSizeInBytes`置为0；否则返回`2 * nnz * sizeof(int32_t)`。
- 本接口不引入Device同步。

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 输入使用CSR行偏移和列索引数组 |
| COO | ❌ | 不支持COO格式 |
| CSC | ❌ | 不支持CSC格式 |

### aclsparseXcsrsort

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseXcsrsort(
    aclsparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const aclsparseMatDescr_t descrA,
    const int *csrRowPtr,
    int *csrColInd,
    int *P,
    void *pBuffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse库上下文句柄，提供执行stream | Host |
| m | 输入 | int | 稀疏矩阵行数 | Host |
| n | 输入 | int | 稀疏矩阵列数；仅用于Host参数校验，不参与排序计算 | Host |
| nnz | 输入 | int | 稀疏矩阵非零元素个数 | Host |
| descrA | 输入 | const aclsparseMatDescr_t | Legacy矩阵描述符；使用其索引基址，其他属性不参与排序 | Host |
| csrRowPtr | 输入 | const int*（INT32） | CSR行偏移数组，长度为`m + 1`，排序期间保持不变 | Device |
| csrColInd | 输入/输出 | int*（INT32） | CSR列索引数组，长度为`nnz`；输出时每行内部稳定升序 | Device |
| P | 输入/输出 | int*（INT32） | 排列数组，长度为`nnz`；使用与`csrColInd`相同的稳定排列重排 | Device |
| pBuffer | 输入 | void* | 由`aclsparseXcsrsort_bufferSizeExt`查询大小并由调用方分配的workspace | Device |

#### 约束说明

- `handle`不可为nullptr，否则返回`ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`。
- `m >= 0`、`n >= 0`且`nnz >= 0`。
- 当`m == 0`或`n == 0`时，`nnz`必须为0。
- `descrA`不可为nullptr，索引基址仅支持`ACL_SPARSE_INDEX_BASE_ZERO`和`ACL_SPARSE_INDEX_BASE_ONE`。
- 当`m > 0`时，`csrRowPtr`不可为nullptr。
- 当`nnz > 0`时，`csrColInd`、`P`和`pBuffer`均不可为nullptr。
- 调用方须保证CSR结构合法，包括`csrRowPtr`单调不减、`csrRowPtr[m] - indexBase == nnz`，以及列索引位于矩阵列范围内。
- `pBuffer`无需128字节对齐，但不得与`csrRowPtr`、`csrColInd`或`P`的存储区域重叠。
- 当`nnz == 0`时，接口直接返回成功，不下发kernel。
- 本接口异步执行；读取`csrColInd`和`P`前，调用方须同步handle绑定的stream。

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 对每个CSR行的列索引独立排序 |
| COO | ❌ | 不支持COO格式 |
| CSC | ❌ | 不支持CSC格式 |

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
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

using DevicePtr = std::unique_ptr<void, aclError (*)(void *)>;

static int AllocAndCopyDevice(DevicePtr &devicePtr, const void *hostPtr, size_t sizeBytes)
{
    void *rawPtr = nullptr;
    auto ret = aclrtMalloc(&rawPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    devicePtr.reset(rawPtr);

    if (hostPtr != nullptr && sizeBytes > 0) {
        ret = aclrtMemcpy(devicePtr.get(), sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy data to Device failed. ERROR: %d\n", ret); return ret);
    }
    return ACL_SUCCESS;
}

static void PrintVector(const char *name, const std::vector<int> &values)
{
    LOG_PRINT("%s:", name);
    for (int value : values) {
        LOG_PRINT(" %d", value);
    }
    LOG_PRINT("\n");
}

int aclsparseCsrsortTest(AclContext &ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建ops-sparse句柄并绑定stream
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet); return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(
        rawHandle, aclsparseDestroy);

    sparseRet = aclsparseSetStream(handlePtr.get(), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet); return sparseRet);

    // 2. 准备Host端CSR数据
    const int m = 3;
    const int n = 5;
    const int nnz = 7;
    std::vector<int> hCsrRowPtr = {0, 3, 5, 7};
    std::vector<int> hCsrColInd = {4, 1, 3, 2, 0, 4, 1};
    std::vector<int> hP(nnz);
    std::iota(hP.begin(), hP.end(), 0);

    // 3. 申请Device内存并拷贝CSR数据和P
    DevicePtr dCsrRowPtr(nullptr, aclrtFree);
    DevicePtr dCsrColInd(nullptr, aclrtFree);
    DevicePtr dP(nullptr, aclrtFree);
    auto aclRet = AllocAndCopyDevice(
        dCsrRowPtr, hCsrRowPtr.data(), hCsrRowPtr.size() * sizeof(int));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(
        dCsrColInd, hCsrColInd.data(), hCsrColInd.size() * sizeof(int));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dP, hP.data(), hP.size() * sizeof(int));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 4. 查询workspace大小
    size_t bufferSize = 0;
    sparseRet = aclsparseXcsrsort_bufferSizeExt(
        handlePtr.get(), m, n, nnz,
        static_cast<const int *>(dCsrRowPtr.get()),
        static_cast<const int *>(dCsrColInd.get()), &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("bufferSizeExt failed. ERROR: %d\n", sparseRet); return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes\n", bufferSize);

    DevicePtr dBuffer(nullptr, aclrtFree);
    aclRet = AllocAndCopyDevice(dBuffer, nullptr, bufferSize);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 5. 创建并配置Legacy MatDescr，由unique_ptr自动销毁
    aclsparseMatDescr_t rawDescr = nullptr;
    sparseRet = aclsparseCreateMatDescr(&rawDescr);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateMatDescr failed. ERROR: %d\n", sparseRet); return sparseRet);
    std::unique_ptr<aclsparseMatDescr, aclsparseStatus_t (*)(aclsparseMatDescr_t)> descrPtr(
        rawDescr, aclsparseDestroyMatDescr);
    aclsparseSetMatType(descrPtr.get(), ACL_SPARSE_MATRIX_TYPE_GENERAL);
    aclsparseSetMatIndexBase(descrPtr.get(), ACL_SPARSE_INDEX_BASE_ZERO);

    // 6. 执行CSR行内稳定排序
    sparseRet = aclsparseXcsrsort(
        handlePtr.get(), m, n, nnz, descrPtr.get(),
        static_cast<const int *>(dCsrRowPtr.get()),
        static_cast<int *>(dCsrColInd.get()),
        static_cast<int *>(dP.get()), dBuffer.get());
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseXcsrsort failed. ERROR: %d\n", sparseRet); return sparseRet);

    // 7. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 8. 将结果拷贝回Host并打印
    std::vector<int> hSortedColInd(nnz, 0);
    std::vector<int> hSortedP(nnz, 0);
    aclRet = aclrtMemcpy(hSortedColInd.data(), hSortedColInd.size() * sizeof(int),
                         dCsrColInd.get(), hSortedColInd.size() * sizeof(int),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("copy csrColInd to Host failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(hSortedP.data(), hSortedP.size() * sizeof(int),
                         dP.get(), hSortedP.size() * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("copy P to Host failed. ERROR: %d\n", aclRet); return aclRet);

    PrintVector("csrRowPtr", hCsrRowPtr);
    PrintVector("csrColInd", hSortedColInd);
    PrintVector("P", hSortedP);
    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseCsrsortTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCsrsortTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```text
bufferSize = 56 bytes
csrRowPtr: 0 3 5 7
csrColInd: 1 3 4 0 2 1 4
P: 1 2 0 4 3 6 5
```
