# CsrSetStridedBatch算子

## 算子概述

csr_set_strided_batch 算子用于设置 CSR（Compressed Sparse Row）格式稀疏矩阵描述符的批量（strided batch）属性，对标 cuSPARSE 的 `cusparseCsrSetStridedBatch`，属于 SparseAccessor（描述符访问器）类接口。batch 数的读回由通用算子 `aclsparseSpMatGetStridedBatch`（sparse/spmat_get_strided_batch/，适用于全部稀疏格式）提供。

CsrSetStridedBatch 为**纯 host 访问器**：批量属性的写入均在 Host 侧完成，**不启动 NPU kernel、不涉及 stream 同步、无 NPU 计算**。`aclsparseCsrSetStridedBatch` 将批量矩阵个数与相邻 batch 间 CSR 数组的地址间隔写入描述符，使其可表达 batch 化稀疏矩阵。接口签名与 cuSPARSE 逐参数一致，便于 cuSPARSE 用户迁移。

两个步长均按**元素个数**计，且取 0 均表示**广播**（所有 batch 共享对应数组，对齐 cuSPARSE 官方样例与本仓 `aclsparseBsrSetStridedBatch` / `aclsparseCooSetStridedBatch` 的 0=广播约定）：

- `offsetsBatchStride`：相邻 batch 之间行偏移数组（`csrRowOffsets`，长度 `rows + 1`）的地址间隔；
- `columnsValuesBatchStride`：相邻 batch 之间列索引数组（`csrColInd`）与值数组（`csrValues`，长度均为 `nnz`）的地址间隔，两数组共用（写入描述符共享字段 `columnsBatchStride` 与 `valuesBatchStride` 同值，后续批量消费算子可与 BSR 拆分字段同源读取）。

新建描述符未调用 SetStridedBatch 时，默认 `batchCount = 1`、两个步长为 0，即未启用批量语义的单矩阵。

> **重要：当前版本计算接口尚不消费批量属性**。`batchCount` 与两个步长写入描述符后，SpMM / SpMV 等计算接口仍按单矩阵布局计算、GetBufferSize 不按 batch 膨胀；批量计算（如 cuSPARSE 的 `C_i = A_i · B_i` 批量 SpMM / SDDMM 模式）待消费算子交付。按 cuSPARSE 官方样例模式（Set 后直接调批量计算）迁移的用户请注意此差异。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseCsrSetStridedBatch | 设置 CSR 描述符的批量矩阵个数与相邻 batch 间 CSR 数组的元素间隔 |

> **实现位置说明**：实现位于 `sparse/csr_set_strided_batch/arch35/csr_set_strided_batch_host.cpp`（与 coo_set_strided_batch 等描述符算子交付模式一致）；描述符共享 batch 字段定义位于 `sparse/common/aclsparse_descr_internal.h`。测试位于 `test/csr_set_strided_batch/`。

## 算子执行接口

### aclsparseCsrSetStridedBatch

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCsrSetStridedBatch(aclsparseSpMatDescr_t spMatDescr,
    int batchCount, int64_t offsetsBatchStride, int64_t columnsValuesBatchStride);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入/输出 | aclsparseSpMatDescr_t | CSR 稀疏矩阵描述符（须由 aclsparseCreateCsr 创建，其 batchCount / offsetsBatchStride / columnsBatchStride / valuesBatchStride 字段被写入） | Host |
| batchCount | 输入 | int | batch 数，范围为 1..65535 | Host |
| offsetsBatchStride | 输入 | int64_t | 相邻 batch 行偏移数组的元素偏移；合法取值为 0（广播）或 >= rows + 1 | Host |
| columnsValuesBatchStride | 输入 | int64_t | 相邻 batch 列索引/值数组的元素偏移；合法取值为 0（广播）或 >= nnz | Host |

#### 约束说明

- spMatDescr 不可为 nullptr 且签名须合法（错误类型句柄返回 `ACL_SPARSE_STATUS_INVALID_VALUE`）
- 描述符必须是 CSR 格式，否则返回 `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`
- batchCount 超出 [1, 65535] 返回 `ACL_SPARSE_STATUS_INVALID_VALUE`（batchCount 为全格式共享字段，上限为全仓不变量，与 BSR/COO/DnMat setter 一致）
- 任一步长为负、或非 0 且小于对应数组长度（offsets < rows+1、columnsValues < nnz，即相邻 batch 数据区间重叠）返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；步长为 0 表示广播（所有 batch 共享对应数组）
- 地址范围 (batchCount-1)*步长+数组长度 溢出 int64 返回 `ACL_SPARSE_STATUS_INVALID_VALUE`（受检乘加防护，与 BSR/COO/DnMat setter 一致）；rows = INT64_MAX 时 rows + 1 跨度构造溢出同样返回 `ACL_SPARSE_STATUS_INVALID_VALUE`（受检加法防护，延迟绑定下 rows 可达 INT64_MAX）
- 校验失败时不改写描述符的既有批量属性
- batch 元数据实际变化时失效描述符既有预处理缓存（sddmm 等），需重新执行对应 Preprocess
- aclsparseCsrSetStridedBatch 为纯 host 操作，不启动 NPU kernel，不涉及 stream 同步
- 当前版本计算接口尚不消费批量属性（见上方算子概述的重要声明）

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
        }                             \
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

int aclsparseCsrSetStridedBatchTest(AclContext &ctx)
{
    (void)ctx.Stream();  // setter 不使用 handle / stream，此处仅占位

    // 1. 准备 Host 端 CSR 数据：单 batch 为 3x3 矩阵、3 个非零元
    //    [1.0  0.0  0.0]
    //    [0.0  0.0  2.0]
    //    [0.0  3.0  0.0]
    int64_t rows = 3;
    int64_t cols = 3;
    int64_t nnz = 3;

    std::vector<int32_t> hCsrRowOffsets = {0, 1, 2, 3};
    std::vector<int32_t> hCsrColInd = {0, 2, 1};
    std::vector<float> hCsrValues = {1.0f, 2.0f, 3.0f};

    // 2. 拷贝 rowOffsets/colInd/values 到 Device
    DevicePtr dCsrRowOffsets(nullptr, aclrtFree);
    DevicePtr dCsrColInd(nullptr, aclrtFree);
    DevicePtr dCsrValues(nullptr, aclrtFree);
    auto aclRet = AllocAndCopyDevice(dCsrRowOffsets, hCsrRowOffsets.data(), (rows + 1) * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dCsrColInd, hCsrColInd.data(), nnz * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dCsrValues, hCsrValues.data(), nnz * sizeof(float));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 3. aclsparseCreateCsr：创建单矩阵 CSR 描述符
    aclsparseSpMatDescr_t matDescr = nullptr;
    auto sparseRet = aclsparseCreateCsr(&matDescr, rows, cols, nnz,
        dCsrRowOffsets.get(), dCsrColInd.get(), dCsrValues.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet); return sparseRet);

    // 4. 新建描述符未设置批量语义时，默认 batchCount = 1
    int defaultBatchCount = 0;
    sparseRet = aclsparseSpMatGetStridedBatch(matDescr, &defaultBatchCount);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseSpMatGetStridedBatch failed. ERROR: %d\n", sparseRet);
        aclsparseDestroySpMat(matDescr); return sparseRet);
    LOG_PRINT("default batchCount = %d\n", defaultBatchCount);

    // 5. aclsparseCsrSetStridedBatch：设置为 4 个 batch 的批量矩阵
    //    行偏移数组间隔 = rows + 1 = 4（元素个数），列索引/值数组间隔 = nnz = 3
    int batchCount = 4;
    int64_t offsetsBatchStride = rows + 1;        // = 4
    int64_t columnsValuesBatchStride = nnz;       // = 3
    sparseRet = aclsparseCsrSetStridedBatch(matDescr, batchCount,
        offsetsBatchStride, columnsValuesBatchStride);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCsrSetStridedBatch failed. ERROR: %d\n", sparseRet);
        aclsparseDestroySpMat(matDescr); return sparseRet);

    // 6. aclsparseSpMatGetStridedBatch：读回批量个数并断言
    int outBatchCount = 0;
    sparseRet = aclsparseSpMatGetStridedBatch(matDescr, &outBatchCount);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseSpMatGetStridedBatch failed. ERROR: %d\n", sparseRet);
        aclsparseDestroySpMat(matDescr); return sparseRet);
    CHECK_RET(outBatchCount == batchCount,
        LOG_PRINT("batchCount mismatch: %d / %d\n", outBatchCount, batchCount);
        aclsparseDestroySpMat(matDescr); return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    LOG_PRINT("aclsparseCsrSetStridedBatch succeeded.\n");
    // batchCount 为 getter 回读值；两个步长为 Set 入参回显（公共 getter 只回读 batchCount）
    LOG_PRINT("batchCount read back = %d (input strides: offsetsBatchStride=%lld, columnsValuesBatchStride=%lld)\n",
        outBatchCount, static_cast<long long>(offsetsBatchStride),
        static_cast<long long>(columnsValuesBatchStride));

    // 7. 销毁描述符
    aclsparseDestroySpMat(matDescr);
    LOG_PRINT("aclsparseDestroySpMat done.\n");

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseCsrSetStridedBatchTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCsrSetStridedBatchTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
default batchCount = 1
aclsparseCsrSetStridedBatch succeeded.
batchCount read back = 4 (input strides: offsetsBatchStride=4, columnsValuesBatchStride=3)
aclsparseDestroySpMat done.
```

## 配套算子说明

本算子与配套算子（Set/Get 闭环）均构建于同一 `libops_sparse`。测试二进制已内嵌配套算子实现，支持单算子独立构建测试：

```bash
bash build.sh --ops=csr_set_strided_batch --run            # 单算子：Set
bash build.sh --ops=spmat_get_strided_batch --run          # 单算子：Get
bash build.sh --ops=csr_set_strided_batch,spmat_get_strided_batch --run  # 成对（默认推荐）
```
