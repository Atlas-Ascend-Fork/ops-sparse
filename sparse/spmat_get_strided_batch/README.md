# SpMatGetStridedBatch算子

## 算子概述

spmat_get_strided_batch 算子用于读回稀疏矩阵描述符的 batch 数，对标 cuSPARSE 的 `cusparseSpMatGetStridedBatch`，属于 SparseAccessor（描述符访问器）类接口。

本算子为**纯 host 访问器**：从**任意格式**（COO / CSR / CSC / SELL / BLOCKED_ELL）稀疏矩阵描述符读回 batchCount，全程**不启动 NPU kernel、不涉及 stream 同步、无 NPU 计算**。接受 const 描述符，非 const 描述符同样支持（隐式转换）。语义与 cuSPARSE 完全对齐，便于 cuSPARSE 用户无缝迁移。

仅返回 batchCount 而不返回 batchStride，与 cuSPARSE 设计一致：stride 签名是格式相关的（COO 为 1 个 `batchStride`，CSR 为 `offsetsBatchStride` + `columnsValuesBatchStride` 2 个，BSR 为 3 个），通用 getter 唯一能跨格式统一暴露的公共量是 batchCount。batch 元数据的写入由各格式的 SetStridedBatch 接口完成：配套算子 [aclsparseCooSetStridedBatch](../coo_set_strided_batch/README.md)（COO）与 `aclsparseBsrSetStridedBatch`（BSR）共用同一 batchCount 字段，本接口统一读回两路写入值。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSpMatGetStridedBatch | 读回稀疏矩阵描述符的 batch 数（任意格式） |

## 算子执行接口

### aclsparseSpMatGetStridedBatch

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpMatGetStridedBatch(aclsparseConstSpMatDescr_t spMatDescr,
    int *batchCount);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵描述符（任意格式；非 const 描述符同样支持） | Host |
| batchCount | 输出 | int* | batch 数 | Host |

#### 约束说明

- spMatDescr 不可为 nullptr 且签名须合法（错型/损坏句柄将被拒绝），batchCount 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- 适用于任意格式稀疏矩阵描述符；未调用过 SetStridedBatch 的新建描述符返回 `1`（未 batch 化）
- 纯 host 操作，不启动 NPU kernel，不涉及 stream 同步

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

    // 1. 准备 3 个非零元的 3x3 COO 矩阵
    const int64_t rows = 3;
    const int64_t cols = 3;
    const int64_t nnz = 3;
    std::vector<int32_t> hRowInd = {0, 1, 2};
    std::vector<int32_t> hColInd = {0, 2, 1};
    std::vector<float> hValues = {1.0f, 2.0f, 3.0f};

    DevicePtr dRowInd(nullptr, aclrtFree);
    DevicePtr dColInd(nullptr, aclrtFree);
    DevicePtr dValues(nullptr, aclrtFree);
    aclRet = AllocAndCopyDevice(dRowInd, hRowInd.data(), nnz * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dColInd, hColInd.data(), nnz * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dValues, hValues.data(), nnz * sizeof(float));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 2. 创建 COO 描述符
    aclsparseSpMatDescr_t matDescr = nullptr;
    auto sparseRet = aclsparseCreateCoo(&matDescr, rows, cols, nnz,
        dRowInd.get(), dColInd.get(), dValues.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCreateCoo failed: %d\n", sparseRet); return sparseRet);

    // 3. 新建描述符（未 batch 化）：Get 返回 1
    int batchCount = 0;
    sparseRet = aclsparseSpMatGetStridedBatch(matDescr, &batchCount);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseSpMatGetStridedBatch failed: %d\n", sparseRet);
        aclsparseDestroySpMat(matDescr); return sparseRet);
    LOG_PRINT("default batchCount = %d\n", batchCount);

    // 4. 设置 strided batch 后再读回（配套算子 aclsparseCooSetStridedBatch）
    const int setBatchCount = 3;
    const int64_t setBatchStride = 4;  // >= nnz
    sparseRet = aclsparseCooSetStridedBatch(matDescr, setBatchCount, setBatchStride);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCooSetStridedBatch failed: %d\n", sparseRet);
        aclsparseDestroySpMat(matDescr); return sparseRet);
    sparseRet = aclsparseSpMatGetStridedBatch(matDescr, &batchCount);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseSpMatGetStridedBatch failed: %d\n", sparseRet);
        aclsparseDestroySpMat(matDescr); return sparseRet);
    CHECK_RET(batchCount == setBatchCount,
        LOG_PRINT("batchCount mismatch: %d != %d\n", batchCount, setBatchCount);
        aclsparseDestroySpMat(matDescr); return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    LOG_PRINT("after Set, batchCount = %d\n", batchCount);

    // 5. 销毁描述符
    aclsparseDestroySpMat(matDescr);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
    return 0;
}
```

预期输出如下：

```
default batchCount = 1
after Set, batchCount = 3
```

## 配套算子说明

本算子与配套算子（Set/Get 闭环）均构建于同一 `libops_sparse`。测试二进制已内嵌配套算子实现，支持单算子独立构建测试：

```bash
bash build.sh --ops=coo_set_strided_batch --run            # 单算子：Set
bash build.sh --ops=spmat_get_strided_batch --run          # 单算子：Get
bash build.sh --ops=coo_set_strided_batch,spmat_get_strided_batch --run  # 成对（默认推荐）
```
