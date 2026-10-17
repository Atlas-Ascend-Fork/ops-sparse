# COO2CSR算子

## 算子概述

coo2csr 算子用于将 COO（Coordinate）格式稀疏矩阵的行索引数组转换为 CSR（Compressed Sparse Row）格式的行指针数组，属于 Legacy API 体系下的 Format Conversion 类算子。核心运算为统计每行非零元素个数（rowCount）并对其做前缀和，生成 CSR 行指针数组 `csrRowPtr[m+1]`。

数学表达式：

```text
rowCount[i] = |{ j : cooRowInd[j] - idxBase == i, 0 <= j < nnz }|,  i ∈ [0, m-1]
csrRowPtr[0]     = idxBase
csrRowPtr[i + 1] = csrRowPtr[i] + rowCount[i],  i ∈ [0, m-1]
```

其中 idxBase 为索引基址（0-based 或 1-based），m 为矩阵行数，nnz 为非零元素个数。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseXcoo2csr | 将 COO 格式行索引数组转换为 CSR 格式行指针数组 |

## 算子执行接口

### aclsparseXcoo2csr

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseXcoo2csr(aclsparseHandle_t handle, const int *cooRowInd, int nnz, int m, int *csrRowPtr, aclsparseIndexBase_t idxBase)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| cooRowInd | 输入 | const int*（INT32） | COO 格式行索引数组，长度为 nnz，须按行非递减排序；当 nnz = 0 时可为 nullptr，Device 内存 |
| nnz | 输入 | int | 非零元素个数，即 cooRowInd 数组长度，须 >= 0，Host 内存 |
| m | 输入 | int | 矩阵行数，须 >= 0；m = 0 时 csrRowPtr 仅含一个元素（值为 idxBase），Host 内存 |
| csrRowPtr | 输出 | int*（INT32） | CSR 格式行指针数组，长度为 m + 1，调用前须由调用方分配好 Device 内存，Device 内存 |
| idxBase | 输入 | aclsparseIndexBase_t | 索引基址，取值为 ACL_SPARSE_INDEX_BASE_ZERO（0-based）或 ACL_SPARSE_INDEX_BASE_ONE（1-based），Host 内存 |

#### 约束说明

- handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`
- csrRowPtr 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- nnz 须 >= 0，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- 当 `nnz == INT32_MAX` 且 `idxBase == ACL_SPARSE_INDEX_BASE_ONE` 时，`csrRowPtr[m] = nnz + idxBase` 超出 int32 表示范围，返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。`idxBase == ACL_SPARSE_INDEX_BASE_ZERO` 时 nnz 可达 INT32_MAX
- m 须 >= 0，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- 当 nnz > 0 且 m == 0 时返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。m = 0 时 nnz 必须为 0
- 当 nnz > 0 时，cooRowInd 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。nnz = 0 时 cooRowInd 可为 nullptr
- idxBase 必须为 `ACL_SPARSE_INDEX_BASE_ZERO` 或 `ACL_SPARSE_INDEX_BASE_ONE`，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- 调用方须保证 cooRowInd 按行非递减排序。算子内部不做排序校验。未排序输入仍能得到正确结果，但 RLE 压缩失效，atomic 竞争增加，性能下降。建议排序以获得最佳性能
- 调用方须保证 cooRowInd 中每个元素取值在 `[idxBase, m + idxBase - 1]` 范围内。越界值被跳过（不计入 rowCount），导致对应行计数偏少，csrRowPtr 值偏小
- nnz > 0 时算子需从 handle workspace 分配临时空间：`m × 4` 字节（行计数数组）+ `aivCoreNum × 4` 字节（多核前缀和块间归约数组，aivCoreNum 为 AI Core 数）。nnz == 0 时不要求 workspace，走快速路径。handle 默认 workspace 为 4 MiB；仅当 `m + aivCoreNum ≤ 1048576` 时，默认 workspace 才足以容纳上述临时空间。当 workspace 不足以同时容纳行计数数组与块间归约数组时，返回 `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`。当 `m + aivCoreNum > 1048576` 时，调用方须先调用 `aclsparseSetWorkspace` 提供不少于 `(m + aivCoreNum) × 4` 字节的 Device 内存，否则返回 `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`
- nnz > 0 时，handle 无 workspace（workspace 为 nullptr）时返回 `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`。nnz == 0 时不检查 workspace
- 边界行为：
  - m = 0：`csrRowPtr[0] = idxBase`，返回 `ACL_SPARSE_STATUS_SUCCESS`
  - nnz = 0, idxBase = 0：`csrRowPtr[0..m]` 全部清零（值为0），返回 `ACL_SPARSE_STATUS_SUCCESS`
  - nnz = 0, idxBase = 1：`csrRowPtr[0..m]` 全部填充为 1，返回 `ACL_SPARSE_STATUS_SUCCESS`
- 算子根据矩阵规模自动选择执行路径：

  | 路径 | 触发条件 | 说明 |
  |------|---------|------|
  | memset 清零 | nnz == 0 且 idxBase == 0 | 直接 memset csrRowPtr，不启动 kernel，不要求 workspace |
  | Fused kernel 填充 | nnz == 0 且 idxBase == 1 | 单 block 填充 csrRowPtr 为 idxBase，不启动计数/前缀和，不要求 workspace |
  | 融合 kernel | nnz > 0 且 m ≤ 1024 且 nnz ≤ 1024 | 单 block 内完成 CountRows + PrefixSum |
  | 多核并行前缀和 | nnz > 0 且 m > 1024 | K1（CountRows）+ 3 阶段并行前缀和（Phase A/B/C） |
  | 单核前缀和 | nnz > 0 且 m ≤ 1024 且 nnz > 1024 | K1（CountRows）+ 单核串行前缀和 |

- 当 `nnz > 0` 且 `m + aivCoreNum > 1048576` 时，handle 默认 workspace（4 MiB）不足以容纳行计数数组与块间归约数组（总计 `(m + aivCoreNum) × 4` 字节）。此时须在调用 `aclsparseXcoo2csr` 之前，先通过 `aclsparseSetWorkspace` 向 handle 注入一块不小于 `(m + aivCoreNum) × 4` 字节的 Device 内存，例如：
  ```cpp
  uint32_t aivCoreNum = 48;  // 请根据实际设备 AI Core 数调整
  void *userWorkspace = nullptr;
  size_t wsSize = static_cast<size_t>(m + aivCoreNum) * sizeof(int);
  aclrtMalloc(&userWorkspace, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclsparseSetWorkspace(handle, userWorkspace, wsSize);
  ```

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 作为输出格式，生成 CSR 行指针数组 `csrRowPtr[m+1]` |
| COO | ✅ | 作为输入格式，读取 COO 行索引数组 `cooRowInd[nnz]` |
| CSC | ✅ | 间接支持，本接口也可用于将 COO 列索引数组转换为 CSC 列指针数组，接口签名不变，语义由调用方决定 |

#### 调用示例

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

int aclsparseXcoo2csrTest(AclContext &ctx)
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

    // 2. 准备 Host 端 COO 行索引数据
    //    示例：4×4 矩阵，5 个非零元素，行索引已按行非递减排序（0-based）
    //    cooRowInd = {0, 0, 1, 2, 3}
    //    预期输出：csrRowPtr = {0, 2, 3, 4, 5}
    int m = 4;
    int nnz = 5;
    std::vector<int> hCooRowInd = {0, 0, 1, 2, 3};

    // 3. 拷贝 COO 行索引数据到 Device
    DevicePtr dCooRowInd(nullptr, aclrtFree);
    auto aclRet = AllocAndCopyDevice(dCooRowInd, hCooRowInd.data(), nnz * sizeof(int));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 4. 分配 csrRowPtr 输出空间（Device 内存，长度 m+1）
    DevicePtr dCsrRowPtr(nullptr, aclrtFree);
    {
        void *rawPtr = nullptr;
        aclRet = aclrtMalloc(&rawPtr, (m + 1) * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for csrRowPtr failed. ERROR: %d\n", aclRet);
                  return aclRet);
        dCsrRowPtr.reset(rawPtr);
    }

    // 5. 调用 aclsparseXcoo2csr 完成 COO -> CSR 行指针转换
    sparseRet = aclsparseXcoo2csr(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        static_cast<const int *>(dCooRowInd.get()),
        nnz,
        m,
        static_cast<int *>(dCsrRowPtr.get()),
        ACL_SPARSE_INDEX_BASE_ZERO);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseXcoo2csr failed: %d\n", sparseRet);
              return sparseRet);

    // 6. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 7. 将结果从 Device 拷贝回 Host 并打印
    std::vector<int> hCsrRowPtr(m + 1, 0);
    aclRet = aclrtMemcpy(hCsrRowPtr.data(), (m + 1) * sizeof(int), dCsrRowPtr.get(), (m + 1) * sizeof(int),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy csrRowPtr to Host failed. ERROR: %d\n", aclRet); return aclRet);

    LOG_PRINT("csrRowPtr:");
    for (int i = 0; i <= m; i++) {
        LOG_PRINT(" %d", hCsrRowPtr[i]);
    }
    LOG_PRINT("\n");

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseXcoo2csrTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseXcoo2csrTest failed: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```text
csrRowPtr: 0 2 3 4 5
```
