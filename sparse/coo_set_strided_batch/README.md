# CooSetStridedBatch算子

## 算子概述

coo_set_strided_batch 算子用于设置 COO（Coordinate）格式稀疏矩阵描述符的 batch 元数据，对标 cuSPARSE 的 `cusparseCooSetStridedBatch`，属于 SparseAccessor（描述符访问器）类接口。

本算子为**纯 host 描述符操作**：`aclsparseCooSetStridedBatch` 将 batchCount 与 batchStride 写入描述符，使单个 COO 描述符可表达 batch 化稀疏矩阵。全程**不启动 NPU kernel、不涉及 stream 同步、无 NPU 计算**。语义与 cuSPARSE 完全对齐，便于 cuSPARSE 用户无缝迁移。

batchStride 为相邻两个 batch 的 `cooRowInd` / `cooColInd` / `cooValues` 起始地址间的**元素偏移**（非字节偏移）。特殊值 `0` 表示广播——所有 batch 共享同一组 row/col/values 数组（同一稀疏模式广播到全部 batch，对齐 cuSPARSE 官方样例与本仓 BSR 的 0=广播约定）；此时数组实际长度仍为 `nnz`。其余合法取值必须满足 `batchStride >= nnz`，即用户数组按 `[batchStride * batchCount]` 元素布局，第 `i` 个 batch 的数组起始偏移为 `batchStride * i`。

batchCount 的读回由配套算子 [aclsparseSpMatGetStridedBatch](../spmat_get_strided_batch/README.md)（sparse/spmat_get_strided_batch/）提供，两者构成 Set/Get 闭环。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseCooSetStridedBatch | 设置 COO 描述符的 batch 数与 batch 步长 |

## 算子执行接口

### aclsparseCooSetStridedBatch

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCooSetStridedBatch(aclsparseSpMatDescr_t spMatDescr,
    int batchCount, int64_t batchStride);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入/输出 | aclsparseSpMatDescr_t | COO 稀疏矩阵描述符（由 aclsparseCreateCoo 创建） | Host |
| batchCount | 输入 | int | batch 数，范围为 [1, 65535] | Host |
| batchStride | 输入 | int64_t | 相邻 batch 间元素偏移；0 表示广播（所有 batch 共享同一组 row/col/values 数组）；其余取值必须 >= nnz 且 (batchCount-1)*batchStride+nnz 不溢出 int64，负值非法 | Host |

#### 约束说明

- spMatDescr 不可为 nullptr 且签名须合法，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- 描述符格式必须为 COO，否则返回 `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`
- batchCount 范围为 [1, 65535]（与 Bsr/DnMat 的 SetStridedBatch 全仓一致），否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- batchStride 必须为 `0`（广播）或 `>= nnz`：负值与 `0 < batchStride < nnz`（batch 数据区间重叠）均返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- 地址范围 `(batchCount-1)*batchStride + nnz` 溢出 int64 时返回 `ACL_SPARSE_STATUS_INVALID_VALUE`（受检乘加，与 Bsr/DnMat 的溢出防护一致）
- 纯 host 操作，不启动 NPU kernel，不涉及 stream 同步；可对同一描述符多次调用，后值覆盖前值
- batch 元数据实际变化时失效描述符既有预处理缓存（sddmm 等），与 `aclsparseBsrSetStridedBatch` 语义一致

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

int main()
{
    auto aclRet = aclInit(nullptr);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtSetDevice(0);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 1. 准备 batch=3、每 batch nnz=4 的 COO 数据：batchStride=8 元素（>= nnz，留 4 元素间隔）
    const int64_t rows = 4;
    const int64_t cols = 4;
    const int64_t nnz = 4;
    const int batchCount = 3;
    const int64_t batchStride = 8;
    const int64_t totalNnz = batchStride * batchCount;

    std::vector<int32_t> hRowInd(totalNnz);
    std::vector<int32_t> hColInd(totalNnz);
    std::vector<float> hValues(totalNnz);
    for (int b = 0; b < batchCount; b++) {
        for (int64_t i = 0; i < nnz; i++) {
            hRowInd[b * batchStride + i] = static_cast<int32_t>(i % rows);
            hColInd[b * batchStride + i] = static_cast<int32_t>((i + b) % cols);
            hValues[b * batchStride + i] = 1.0f * (b + 1);
        }
    }

    // 2. 拷贝到 Device
    DevicePtr dRowInd(nullptr, aclrtFree);
    DevicePtr dColInd(nullptr, aclrtFree);
    DevicePtr dValues(nullptr, aclrtFree);
    aclRet = AllocAndCopyDevice(dRowInd, hRowInd.data(), totalNnz * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dColInd, hColInd.data(), totalNnz * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dValues, hValues.data(), totalNnz * sizeof(float));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 3. 创建 COO 描述符并设置 strided batch
    aclsparseSpMatDescr_t matDescr = nullptr;
    auto sparseRet = aclsparseCreateCoo(&matDescr, rows, cols, nnz,
        dRowInd.get(), dColInd.get(), dValues.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCreateCoo failed: %d\n", sparseRet); return sparseRet);

    sparseRet = aclsparseCooSetStridedBatch(matDescr, batchCount, batchStride);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCooSetStridedBatch failed: %d\n", sparseRet);
        aclsparseDestroySpMat(matDescr); return sparseRet);

    // 4. 读回 batch 数验证
    int outBatchCount = 0;
    sparseRet = aclsparseSpMatGetStridedBatch(matDescr, &outBatchCount);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseSpMatGetStridedBatch failed: %d\n", sparseRet);
        aclsparseDestroySpMat(matDescr); return sparseRet);
    CHECK_RET(outBatchCount == batchCount,
        LOG_PRINT("batchCount mismatch: %d != %d\n", outBatchCount, batchCount);
        aclsparseDestroySpMat(matDescr); return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    // 5. 广播模式：batchStride = 0 表示所有 batch 共享同一组数组（对齐 cuSPARSE 官方样例）
    sparseRet = aclsparseCooSetStridedBatch(matDescr, batchCount, 0);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("broadcast aclsparseCooSetStridedBatch failed: %d\n", sparseRet);
        aclsparseDestroySpMat(matDescr); return sparseRet);

    LOG_PRINT("aclsparseCooSetStridedBatch / aclsparseSpMatGetStridedBatch succeeded.\n");
    LOG_PRINT("batchCount read back: %d\n", outBatchCount);

    // 6. 销毁描述符
    aclsparseDestroySpMat(matDescr);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
    return 0;
}
```

预期输出如下：

```
aclsparseCooSetStridedBatch / aclsparseSpMatGetStridedBatch succeeded.
batchCount read back: 3
```

## 配套算子说明

本算子与配套算子（Set/Get 闭环）均构建于同一 `libops_sparse`。测试二进制已内嵌配套算子实现，支持单算子独立构建测试：

```bash
bash build.sh --ops=coo_set_strided_batch --run            # 单算子：Set
bash build.sh --ops=spmat_get_strided_batch --run          # 单算子：Get
bash build.sh --ops=coo_set_strided_batch,spmat_get_strided_batch --run  # 成对（默认推荐）
```
