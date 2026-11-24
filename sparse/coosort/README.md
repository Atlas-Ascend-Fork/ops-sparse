# COOSORT算子

## 算子概述

coosort 算子用于对 COO（Coordinate）格式稀疏矩阵的坐标数组进行**稳定排序**，属于 Legacy API 体系下的 Reorderings / Sort 类算子。

算子对 `(cooRowsA, cooColsA)` 坐标对执行**双键字典序稳定排序**，并将排列作用于 `cooRowsA`、`cooColsA`（原地重排）以及调用方预置的排列数组 `P`：

```
排序后： sortedRow[i] = cooRowsA_in[P[i]]
         sortedCol[i] = cooColsA_in[P[i]]
即：     sortedVal[i] = origVal[P[i]]
```

其中 `P` 在调用前由调用方预置为 `P = 0:1:(nnz-1)`，排序后变为"排序后位置 i → 原始下标"的映射。

**双键语义**：

| 接口 | 主键 | 次键 | 行为 |
|------|------|------|------|
| aclsparseXcoosortByRow | row 升序 | col 升序 | 同 row 内按 col 升序 |
| aclsparseXcoosortByColumn | col 升序 | row 升序 | 同 col 内按 row 升序 |

- **稳定**：完全相等的 `(主键, 次键)` 元素保持原输入相对顺序。
- **支持负数索引**：int32 有符号比较，原生支持负值。
- `m`、`n` 仅校验为正，**不参与排序**。

底层采用**两趟 LSB radix 稳定排序**（AscendC `Sort` API，`SortConfig{SortType::RADIX_SORT, isDescend=false}`）。小规模数据由单核直接排序；大规模数据按运行时 UB 容量切成多个有序 run，各核可循环处理多个 run，随后在 GM workspace 中执行多轮并行 merge-path 稳定归并。int32 signed 升序排序原生支持负数，无需符号位归一。

**`X` 前缀**表示类型无关：算子只对 int 索引排序，不涉及值数组的数据类型。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseXcoosort_bufferSizeExt | 查询 workspace 大小（字节），不带 `P` 参数 |
| aclsparseXcoosortByRow | 按 row 主键、col 次键双键稳定排序，原地重排 row/col 并输出排列 P |
| aclsparseXcoosortByColumn | 按 col 主键、row 次键双键稳定排序，原地重排 row/col 并输出排列 P |

## 算子执行接口

### aclsparseXcoosort_bufferSizeExt

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseXcoosort_bufferSizeExt(
    aclsparseHandle_t handle,
    int               m,
    int               n,
    int               nnz,
    const int        *cooRowsA,
    const int        *cooColsA,
    size_t           *pBufferSizeInBytes)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream 等信息 | Host |
| m | 输入 | int | 矩阵行数，m > 0（仅校验，不参与排序） | Host |
| n | 输入 | int | 矩阵列数，n > 0（仅校验，不参与排序） | Host |
| nnz | 输入 | int | 非零元素个数，nnz >= 0 | Host |
| cooRowsA | 输入 | const int* | COO 行索引数组，长度为 nnz（nnz=0 时可为 nullptr） | Device |
| cooColsA | 输入 | const int* | COO 列索引数组，长度为 nnz（nnz=0 时可为 nullptr） | Device |
| pBufferSizeInBytes | 输出 | size_t* | 输出所需 workspace 大小（字节） | Host |

#### 约束说明

- m > 0，n > 0
- nnz >= 0
- handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`
- pBufferSizeInBytes 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- 当 nnz > 0 时，cooRowsA / cooColsA 不可为 nullptr
- 当 nnz > `kCoosortSingleCoreMaxNnz`（2048）时，自动进入多核、多 run 路径，不是错误条件
- 当 nnz == 0 时，返回 `ACL_SPARSE_STATUS_SUCCESS`，`*pBufferSizeInBytes` 置为 0
- 单核直排路径预留 `nnz * 12` 字节；多核路径使用两份 `[row,col,P]` ping-pong 区，约为 `nnz * 24` 字节并附带内部对齐余量；最终均向上按 128 字节对齐
- **不带 `P` 参数**（查询接口不依赖排列 P）
- 不校验 row/col 值域（支持负数）
- 不校验 pBuffer 对齐（Ascend 950PR&950DT系列产品硬件无此要求）

---

### aclsparseXcoosortByRow

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseXcoosortByRow(
    aclsparseHandle_t handle,
    int               m,
    int               n,
    int               nnz,
    int              *cooRowsA,
    int              *cooColsA,
    int              *P,
    void             *pBuffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄 | Host |
| m | 输入 | int | 矩阵行数，m > 0（仅校验，不参与排序） | Host |
| n | 输入 | int | 矩阵列数，n > 0（仅校验，不参与排序） | Host |
| nnz | 输入 | int | 非零元素个数，nnz >= 0 | Host |
| cooRowsA | 输入/输出 | int* | COO 行索引数组，长度为 nnz；排序后按 row 升序（同 row 内按 col 升序）原地重排 | Device |
| cooColsA | 输入/输出 | int* | COO 列索引数组，长度为 nnz；排序后跟随 cooRowsA 重排，保持 (row, col) 配对 | Device |
| P | 输入/输出 | int* | 排列数组，长度为 nnz；调用前需预置 `P[i] = i`（即 `0:1:(nnz-1)`），排序后满足 `sortedVal[i] = origVal[P[i]]` | Device |
| pBuffer | 输入 | void* | workspace 缓冲区，由 bufferSizeExt 返回的大小分配（无对齐要求） | Device |

#### 约束说明

- m > 0，n > 0
- nnz >= 0
- handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`
- 当 nnz > 0 时，cooRowsA / cooColsA / P / pBuffer 不可为 nullptr
- 当 nnz > `kCoosortSingleCoreMaxNnz`（2048）时，自动进入多核、多 run 路径
- 当 nnz == 0 时，返回 `ACL_SPARSE_STATUS_SUCCESS`（空操作）
- 调用前必须先调用 `aclsparseXcoosort_bufferSizeExt` 获取 pBuffer 大小并分配 pBuffer
- 调用前必须将 `P` 预置为 `P[i] = i`（`0:1:(nnz-1)`）
- 排序为**稳定**排序：完全相等的 `(row, col)` 保持原输入相对顺序
- 支持 int32 有符号负数索引
- 不校验 row/col 值域，不校验 pBuffer 对齐
- 异步 launch，调用后需 `aclrtSynchronizeStream` 等待完成

---

### aclsparseXcoosortByColumn

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseXcoosortByColumn(
    aclsparseHandle_t handle,
    int               m,
    int               n,
    int               nnz,
    int              *cooRowsA,
    int              *cooColsA,
    int              *P,
    void             *pBuffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄 | Host |
| m | 输入 | int | 矩阵行数，m > 0（仅校验，不参与排序） | Host |
| n | 输入 | int | 矩阵列数，n > 0（仅校验，不参与排序） | Host |
| nnz | 输入 | int | 非零元素个数，nnz >= 0 | Host |
| cooRowsA | 输入/输出 | int* | COO 行索引数组，长度为 nnz；排序后跟随 cooColsA 重排，保持 (row, col) 配对 | Device |
| cooColsA | 输入/输出 | int* | COO 列索引数组，长度为 nnz；排序后按 col 升序（同 col 内按 row 升序）原地重排 | Device |
| P | 输入/输出 | int* | 排列数组，长度为 nnz；调用前需预置 `P[i] = i`（即 `0:1:(nnz-1)`），排序后满足 `sortedVal[i] = origVal[P[i]]` | Device |
| pBuffer | 输入 | void* | workspace 缓冲区，由 bufferSizeExt 返回的大小分配（无对齐要求） | Device |

#### 约束说明

- m > 0，n > 0
- nnz >= 0
- handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`
- 当 nnz > 0 时，cooRowsA / cooColsA / P / pBuffer 不可为 nullptr
- 当 nnz > `kCoosortSingleCoreMaxNnz`（2048）时，自动进入多核、多 run 路径
- 当 nnz == 0 时，返回 `ACL_SPARSE_STATUS_SUCCESS`（空操作）
- 调用前必须先调用 `aclsparseXcoosort_bufferSizeExt` 获取 pBuffer 大小并分配 pBuffer
- 调用前必须将 `P` 预置为 `P[i] = i`（`0:1:(nnz-1)`）
- 排序为**稳定**排序：完全相等的 `(col, row)` 保持原输入相对顺序
- 支持 int32 有符号负数索引
- 不校验 row/col 值域，不校验 pBuffer 对齐
- 异步 launch，调用后需 `aclrtSynchronizeStream` 等待完成

## 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| COO | ✅ | 输入为 COO 格式的 (cooRowsA, cooColsA) 坐标数组 |
| CSR | ❌ | coosort 仅支持 COO |
| CSC | ❌ | coosort 仅支持 COO |

### COO 格式索引结构

| 数组 | 长度 | 说明 |
|------|------|------|
| cooRowsA | nnz | 行索引数组，每个非零元素的行号 |
| cooColsA | nnz | 列索引数组，每个非零元素的列号 |
| P | nnz | 排列数组，调用方预置 `0:1:(nnz-1)`，排序后为排列映射 |

### 索引基址

coosort 不使用 MatDescr，**不涉及索引基址（0-based / 1-based）转换**。调用方传入的 `cooRowsA` / `cooColsA` 按原始值排序，输出的 `P` 反映原始下标。如需 1-based，由调用方在排序前后自行调整。

### 数据类型

| 精度前缀 | 计算类型 | 值类型 | 说明 |
|---------|---------|--------|------|
| X | int32 | 不涉及值 | `X` 前缀表示类型无关，只对 int 索引排序，不读取/写入值数组 |

**索引类型**：int32（32 位有符号整数），支持负数。

### nnz 与分块策略

`kCoosortSingleCoreMaxNnz`（2048）仅是单核直排与多核路径的分发阈值，不是算子 nnz 上限。多核路径由 Host 根据真实 UB 容量反推单个 `runSize`，计算 `runCount = ceil(nnz / runSize)`，启动 `min(AIV 核数, runCount)` 个核；当 `runCount` 大于启动核数时，每个核以跨步方式循环排序多个 run。初始 run 全部完成后，再对所有 run 做 GM ping-pong 两两归并；每轮输出按 merge-path 对角线切成 UB 安全片段，由各启动核跨步领取，实际参与核数取决于片段数量。即使只剩最后一对 run，只要输出被切成多个片段，也可由多个核并行处理，不会在实现上固定由 0 号核归并全部 nnz。因此“多核均分后单核仍装不下全部数据”不会再直接返回资源不足。

接口中的 `nnz` 类型为 `int`，其数值范围构成 API 层面的上限；实际可处理规模还受设备可分配 workspace 容量约束。

## 调用示例

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

int aclsparseCoosortTest(AclContext &ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreate failed: %d\n", sparseRet);
              return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(rawHandle, aclsparseDestroy);

    sparseRet = aclsparseSetStream(static_cast<aclsparseHandle_t>(handlePtr.get()), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetStream failed: %d\n", sparseRet);
              return sparseRet);

    // 2. 准备 Host 端 COO 数据
    //    示例：5 个非零元的 COO 坐标（对标 CUDA 实测双键语义锚点）
    //    rows = {1, 0, 0, 1, 0}
    //    cols = {9, 7, 3, 2, 5}
    //    ByRow 期望输出（row 主/col 次升序）：
    //      rows = {0, 0, 0, 1, 1}
    //      cols = {3, 5, 7, 2, 9}
    //      P    = {2, 4, 1, 3, 0}
    int m = 2;
    int n = 10;
    int nnz = 5;

    std::vector<int> hCooRowsA = {1, 0, 0, 1, 0};
    std::vector<int> hCooColsA = {9, 7, 3, 2, 5};

    // 3. 拷贝 COO row/col 数据到 Device
    DevicePtr dCooRowsA(nullptr, aclrtFree);
    DevicePtr dCooColsA(nullptr, aclrtFree);
    auto aclRet = AllocAndCopyDevice(dCooRowsA, hCooRowsA.data(), nnz * sizeof(int));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dCooColsA, hCooColsA.data(), nnz * sizeof(int));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 4. Step 1 — bufferSizeExt: 查询 workspace 大小（不带 P）
    size_t bufferSize = 0;
    sparseRet = aclsparseXcoosort_bufferSizeExt(
        handlePtr.get(), m, n, nnz,
        static_cast<const int*>(dCooRowsA.get()),
        static_cast<const int*>(dCooColsA.get()),
        &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("bufferSizeExt failed: %d\n", sparseRet);
              return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes\n", bufferSize);

    // 5. 分配 workspace
    DevicePtr pBuffer(nullptr, aclrtFree);
    if (bufferSize > 0) {
        void *rawBuffer = nullptr;
        aclRet = aclrtMalloc(&rawBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for workspace failed. ERROR: %d\n", aclRet);
                  return aclRet);
        pBuffer.reset(rawBuffer);
    }

    // 6. 预置 P = 0:1:(nnz-1)（调用方责任）
    std::vector<int> hP(nnz);
    for (int i = 0; i < nnz; i++) {
        hP[i] = i;
    }
    DevicePtr dP(nullptr, aclrtFree);
    aclRet = AllocAndCopyDevice(dP, hP.data(), nnz * sizeof(int));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 7. Step 2 — ByRow: 双键稳定排序（row 主/col 次），原地重排 row/col + 输出 P
    sparseRet = aclsparseXcoosortByRow(
        handlePtr.get(), m, n, nnz,
        static_cast<int*>(dCooRowsA.get()),
        static_cast<int*>(dCooColsA.get()),
        static_cast<int*>(dP.get()),
        pBuffer.get());
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("ByRow failed: %d\n", sparseRet);
              return sparseRet);

    // 8. 同步等待计算完成
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              return aclRet);

    // 9. 将结果拷贝回 Host 并打印
    std::vector<int> hSortedRows(nnz, 0);
    std::vector<int> hSortedCols(nnz, 0);
    std::vector<int> hSortedP(nnz, 0);

    aclRet = aclrtMemcpy(hSortedRows.data(), nnz * sizeof(int), dCooRowsA.get(), nnz * sizeof(int),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy rows to Host failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(hSortedCols.data(), nnz * sizeof(int), dCooColsA.get(), nnz * sizeof(int),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy cols to Host failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(hSortedP.data(), nnz * sizeof(int), dP.get(), nnz * sizeof(int),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy P to Host failed. ERROR: %d\n", aclRet); return aclRet);

    LOG_PRINT("\nResult (ByRow):\n");
    LOG_PRINT("  rows: ");
    for (int i = 0; i < nnz; i++) {
        LOG_PRINT("%d ", hSortedRows[i]);
    }
    LOG_PRINT("\n  cols: ");
    for (int i = 0; i < nnz; i++) {
        LOG_PRINT("%d ", hSortedCols[i]);
    }
    LOG_PRINT("\n  P:    ");
    for (int i = 0; i < nnz; i++) {
        LOG_PRINT("%d ", hSortedP[i]);
    }
    LOG_PRINT("\n");

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseCoosortTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCoosortTest failed: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
bufferSize = 128 bytes

Result (ByRow):
  rows: 0 0 0 1 1
  cols: 3 5 7 2 9
  P:    2 4 1 3 0
```
