# Csr2coo算子

## 算子概述

csr2coo 算子用于将 CSR（Compressed Sparse Row）格式的压缩行偏移数组（`csrRowPtr`，长度 m+1）展开为 COO（Coordinate）格式的逐元素行索引数组（`cooRowInd`，长度 nnz）。属于 Legacy API 体系下的 Format Conversion 类算子，仅操作索引数组，不涉及浮点/定点数值计算。

同一函数也可用于 CSC→COO 转换：将 CSC 的列偏移数组（`colPtr`）作为 `csrRowPtr` 参数传入、列数作为 `m` 参数传入，即可完成 CSC 列指针到 COO 列索引的展开。

**计算公式**：

$$
cooRowInd[j] = i + base, \quad \forall j \in [csrRowPtr[i] - base,\ csrRowPtr[i+1] - base),\ \forall i \in [0, m)
$$

其中 `base` 为索引基址（0 或 1），由 `idxBase` 参数指定。

**计算示例**：

0-based（`idxBase = ACL_SPARSE_INDEX_BASE_ZERO`，m = 4，nnz = 6）：
```
csrRowPtr = [0, 2, 2, 5, 6]   →   cooRowInd = [0, 0, 2, 2, 2, 3]
```

1-based（`idxBase = ACL_SPARSE_INDEX_BASE_ONE`，m = 4，nnz = 6）：
```
csrRowPtr = [1, 3, 3, 6, 7]   →   cooRowInd = [1, 1, 3, 3, 3, 4]
```

**CSC 转换**：本算子为纯索引操作，不依赖 CSR 格式语义。将 CSC 的列偏移数组（`colPtr`）作为 `csrRowPtr` 参数传入、列数作为 `m` 参数传入，即可完成 CSC 列指针到 COO 列索引的展开。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseXcsr2coo | 将 CSR 行偏移数组展开为 COO 行索引数组（亦支持 CSC→COO） |

## 算子执行接口

### aclsparseXcsr2coo

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseXcsr2coo(
    aclsparseHandle_t   handle,
    const int32_t      *csrRowPtr,
    int64_t             nnz,
    int64_t             m,
    int32_t            *cooRowInd,
    aclsparseIndexBase_t idxBase)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，须先通过 `aclsparseCreate` 创建。可通过 `aclsparseSetStream` 设置 stream；未设置时使用默认 stream。不可为 nullptr，Host 内存 |
| csrRowPtr | 输入 | const int32_t* | CSR 行偏移数组（或 CSC 列偏移数组），长度 m+1，升序排列。当 m > 0 时不可为 nullptr，Device 内存 |
| nnz | 输入 | int64_t | 稀疏矩阵中非零元个数，必须 ≥ 0，Host 内存 |
| m | 输入 | int64_t | 矩阵行数（或 CSC 的列数），必须 ≥ 0。当 m == 0 时 nnz 必须为 0，Host 内存 |
| cooRowInd | 输出 | int32_t* | COO 行索引数组（或 CSC 的 COO 列索引数组），长度 nnz。当 nnz > 0 时不可为 nullptr，Device 内存 |
| idxBase | 输入 | aclsparseIndexBase_t | 索引基址：`ACL_SPARSE_INDEX_BASE_ZERO`（0-based）或 `ACL_SPARSE_INDEX_BASE_ONE`（1-based）。其他值无效，Host 内存 |

#### 约束说明

- handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_NOT_INITIALIZED`。
- csrRowPtr 和 cooRowInd 仅支持 `int32_t`，不支持 `int64_t`。
- `csrRowPtr[m] - idxBase` 应等于 nnz。若不一致，kernel 按各行实际 rowNnz 写入，未覆盖的输出元素保留原值，接口仍返回 `ACL_SPARSE_STATUS_SUCCESS`。
- csrRowPtr 必须单调递增（非递减）。Host 不校验单调性；当某行 `csrRowPtr[i+1] < csrRowPtr[i]`（rowNnz < 0）或 `csrRowPtr[i] < idxBase`（rowStart < 0）时，Kernel 检测到后跳过该异常行，继续处理后续行，不报错，接口仍返回 `ACL_SPARSE_STATUS_SUCCESS`，但异常行的输出不正确。
- nnz 和 m 均不能超过 INT32_MAX (2^31-1)，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- nnz < 0 或 m < 0 时返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- 当 m > 0 时，csrRowPtr 不可为 nullptr；当 nnz > 0 时，cooRowInd 不可为 nullptr；idxBase 必须为 `ACL_SPARSE_INDEX_BASE_ZERO` 或 `ACL_SPARSE_INDEX_BASE_ONE`，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- 当 m == 0 时，nnz 必须为 0，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- 当 nnz == 0 时，不启动 Kernel，直接返回 `ACL_SPARSE_STATUS_SUCCESS`。此时 cooRowInd 可为 nullptr。
- UB 资源不足时返回 `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`。
- 内部错误（如 `GetUbSize` 返回 0、`GetAivCoreCount` 失败）时返回 `ACL_SPARSE_STATUS_INTERNAL_ERROR`。
- Kernel launch 为异步操作，用户需在读取 cooRowInd 输出前调用 `aclrtSynchronizeStream` 同步。
- 本算子完全在芯片片内缓存（UB）完成计算，无需 GM workspace，无 GetBufferSize 接口。
- Legacy API 扁平参数，索引基址通过 `idxBase` 参数直接传入，不使用 `aclsparseMatDescr_t`。

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 输入为 CSR 行偏移数组 `csrRowPtr`（长度 m+1） |
| CSC | ✅ | 将 CSC 列偏移数组作为 `csrRowPtr` 传入、列数作为 `m` 传入 |
| COO | ❌ | csr2coo 仅用于 CSR/CSC 偏移数组的展开，不支持 COO 输入 |

**CSR 格式索引结构**：

| 数组 | 长度 | 说明 |
|------|------|------|
| csrRowPtr | m + 1 | 行偏移指针，`csrRowPtr[i]` 是第 i 行非零元素的起始索引 |
| cooRowInd | nnz | 输出的 COO 行索引数组，每个非零元素的行号 |

**索引基址**：通过 `idxBase` 参数直接指定：

- `ACL_SPARSE_INDEX_BASE_ZERO`：0-based（C 风格）
- `ACL_SPARSE_INDEX_BASE_ONE`：1-based（Fortran 风格）

**数据类型**：

| 精度前缀 | 计算类型 | 值类型 | 说明 |
|---------|---------|--------|------|
| （无前缀） | int32 | 不涉及值 | csr2coo 仅操作 int32 索引数组，不读取/写入值数组 |

索引类型：int32（32 位有符号整数）。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdint>
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

using DevicePtr = std::unique_ptr<void, aclError (*)(void *)>;

// 辅助：分配 Device 内存并拷贝 Host 数据，内存由 DevicePtr 自动释放
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

int TestCsr2coo(aclrtStream stream)
{
    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed: %d\n", sparseRet); return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)>
        handlePtr(rawHandle, aclsparseDestroy);

    sparseRet = aclsparseSetStream(static_cast<aclsparseHandle_t>(handlePtr.get()), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed: %d\n", sparseRet); return sparseRet);

    // 2. 准备 CSR 输入：m=3, nnz=5, 0-based
    //    csrRowPtr = [0, 2, 3, 5]   ← row0 有 2 个非零元，row1 有 1 个，row2 有 2 个
    int64_t m = 3, nnz = 5;
    std::vector<int32_t> hRowPtr = {0, 2, 3, 5};

    // 3. 申请 Device 内存并拷贝 csrRowPtr
    DevicePtr dRowPtr(nullptr, aclrtFree);
    auto aclRet = AllocAndCopyDevice(dRowPtr, hRowPtr.data(), hRowPtr.size() * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 4. 申请输出内存：cooRowInd (nnz * sizeof(int32_t))
    DevicePtr dCooRow(nullptr, aclrtFree);
    void *rawCooRow = nullptr;
    aclRet = aclrtMalloc(&rawCooRow, static_cast<size_t>(nnz) * sizeof(int32_t),
                          ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dCooRow failed: %d\n", aclRet); return aclRet);
    dCooRow.reset(rawCooRow);

    // 5. 调用 aclsparseXcsr2coo（Legacy API，扁平参数，0-based）
    sparseRet = aclsparseXcsr2coo(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        static_cast<const int32_t *>(dRowPtr.get()),
        nnz, m,
        static_cast<int32_t *>(dCooRow.get()),
        ACL_SPARSE_INDEX_BASE_ZERO);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseXcsr2coo failed: %d\n", sparseRet); return sparseRet);

    // 6. 同步等待并将结果拷贝回 Host
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", aclRet); return aclRet);

    std::vector<int32_t> hCooRow(static_cast<size_t>(nnz), 0);
    aclRet = aclrtMemcpy(hCooRow.data(), static_cast<size_t>(nnz) * sizeof(int32_t),
                          dCooRow.get(), static_cast<size_t>(nnz) * sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed: %d\n", aclRet); return aclRet);

    // 7. 打印结果：预期 [0, 0, 1, 2, 2]
    LOG_PRINT("CSR->COO (0-based, m=3, nnz=5):\n");
    for (int64_t i = 0; i < nnz; i++) {
        LOG_PRINT("  cooRowInd[%ld] = %d\n", static_cast<int64_t>(i), hCooRow[i]);
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

int TestCsc2coo(aclrtStream stream)
{
    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed: %d\n", sparseRet); return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)>
        handlePtr(rawHandle, aclsparseDestroy);

    sparseRet = aclsparseSetStream(static_cast<aclsparseHandle_t>(handlePtr.get()), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed: %d\n", sparseRet); return sparseRet);

    // 2. 准备 CSC 输入：cols=3, nnz=5, 0-based
    //    colPtr = [0, 2, 3, 5]   ← col0 有 2 个非零元，col1 有 1 个，col2 有 2 个
    int64_t cols = 3, nnz = 5;
    std::vector<int32_t> hColPtr = {0, 2, 3, 5};

    // 3. 申请 Device 内存并拷贝 colPtr
    DevicePtr dColPtr(nullptr, aclrtFree);
    auto aclRet = AllocAndCopyDevice(dColPtr, hColPtr.data(), hColPtr.size() * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 4. 申请输出内存：cooColInd (nnz * sizeof(int32_t))
    DevicePtr dCooCol(nullptr, aclrtFree);
    void *rawCooCol = nullptr;
    aclRet = aclrtMalloc(&rawCooCol, static_cast<size_t>(nnz) * sizeof(int32_t),
                          ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dCooCol failed: %d\n", aclRet); return aclRet);
    dCooCol.reset(rawCooCol);

    // 5. 将 colPtr 作为 csrRowPtr 传入，cols 作为 m 传入
    sparseRet = aclsparseXcsr2coo(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        static_cast<const int32_t *>(dColPtr.get()),
        nnz, cols,
        static_cast<int32_t *>(dCooCol.get()),
        ACL_SPARSE_INDEX_BASE_ZERO);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("CSC->COO failed: %d\n", sparseRet); return sparseRet);

    // 6. 同步等待并将结果拷贝回 Host
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", aclRet); return aclRet);

    std::vector<int32_t> hCooCol(static_cast<size_t>(nnz), 0);
    aclRet = aclrtMemcpy(hCooCol.data(), static_cast<size_t>(nnz) * sizeof(int32_t),
                          dCooCol.get(), static_cast<size_t>(nnz) * sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed: %d\n", aclRet); return aclRet);

    // 7. 打印结果：预期 [0, 0, 1, 2, 2]
    LOG_PRINT("CSC->COO (0-based, cols=3, nnz=5):\n");
    for (int64_t i = 0; i < nnz; i++) {
        LOG_PRINT("  cooColInd[%ld] = %d\n", static_cast<int64_t>(i), hCooCol[i]);
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("AclContext init failed: %d\n", ret); return ret);

    ret = TestCsr2coo(ctx.Stream());
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("TestCsr2coo failed: %d\n", ret); return ret);

    ret = TestCsc2coo(ctx.Stream());
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("TestCsc2coo failed: %d\n", ret); return ret);

    LOG_PRINT("All tests passed.\n");
    return 0;
}
```

预期输出如下：

```
CSR->COO (0-based, m=3, nnz=5):
  cooRowInd[0] = 0
  cooRowInd[1] = 0
  cooRowInd[2] = 1
  cooRowInd[3] = 2
  cooRowInd[4] = 2
CSC->COO (0-based, cols=3, nnz=5):
  cooColInd[0] = 0
  cooColInd[1] = 0
  cooColInd[2] = 1
  cooColInd[3] = 2
  cooColInd[4] = 2
All tests passed.
```
