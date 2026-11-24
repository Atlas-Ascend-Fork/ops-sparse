# 非零元素统计算子

## 算子概述

nnz 算子用于统计稠密矩阵中每行或每列的非零元素个数，以及总非零元素个数。该函数是 dense-to-sparse 格式转换流程的第一步，典型用法为：先调用 nnz 获取每行/列非零元素个数和总非零元素个数，再根据统计结果构建 CSR 或 CSC 格式的偏移数组（通过前缀和），最后执行 dense-to-sparse 转换。

核心运算为：

```
dirA = ROW:    nnzPerRow[i] = sum_{j=0}^{n-1} [A[i][j] != 0],  i in [0, m)
dirA = COLUMN: nnzPerCol[j] = sum_{i=0}^{m-1} [A[i][j] != 0],  j in [0, n)
nnzTotal = sum(nnzPerRow) = sum(nnzPerCol)
```

其中 `[A[i][j] != 0]` 为指示函数，当 `A[i][j] != 0` 时取 1，否则取 0。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSnnz | 统计 FP32 稠密矩阵每行或每列的非零元素个数及总非零元素个数 |

## 算子执行接口

### aclsparseSnnz

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSnnz(aclsparseHandle_t handle, aclsparseDirection_t dirA, int m, int n, const aclsparseMatDescr_t descrA, const float *A, int lda, int *nnzPerRowColumn, int *nnzTotalDevHostPtr)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| dirA | 输入 | aclsparseDirection_t | 统计方向：`ACL_SPARSE_DIRECTION_ROW`（按行）或 `ACL_SPARSE_DIRECTION_COLUMN`（按列），Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| descrA | 输入 | const aclsparseMatDescr_t | 矩阵描述符，仅支持 `ACL_SPARSE_MATRIX_TYPE_GENERAL` 类型，Host 内存 |
| A | 输入 | const float*（FP32） | 稠密矩阵数据，列主序存储，维度为 lda × n，元素 `A[i][j] = A[j * lda + i]`，Device 内存 |
| lda | 输入 | int | 矩阵 A 的 leading dimension，Host 内存 |
| nnzPerRowColumn | 输出 | int* | 每行（dirA=ROW 时大小为 m）或每列（dirA=COLUMN 时大小为 n）的非零元素个数数组，Device 内存 |
| nnzTotalDevHostPtr | 输出 | int* | 总非零元素个数，内存位置由 `aclsparseSetPointerMode` 控制：HOST 模式为 Host 内存，DEVICE 模式为 Device 内存 |

#### 约束说明

- m >= 0
- n >= 0
- lda >= max(1, m)
- descrA 不可为 nullptr，且 descrA.type 仅支持 `ACL_SPARSE_MATRIX_TYPE_GENERAL`
- 当 m > 0 且 n > 0 时，A 不可为 nullptr
- 当 totalUnits > 0 时（totalUnits = m 或 n，取决于 dirA），nnzPerRowColumn 不可为 nullptr
- nnzTotalDevHostPtr 不可为 nullptr
- 当 m = 0 或 n = 0 时，函数直接返回成功，nnzTotal 置 0，nnzPerRowColumn 不写入

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

int aclsparseSnnzTest(AclContext& ctx)
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

    // 使用 DEVICE 模式：nnzTotalDevHostPtr 指向 Device 内存
    sparseRet = aclsparseSetPointerMode(static_cast<aclsparseHandle_t>(handlePtr.get()), ACL_SPARSE_POINTER_MODE_DEVICE);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetPointerMode failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 2. 准备 Host 数据（列主序 3x3 矩阵）
    //    矩阵内容：
    //    [1.0  0.0  3.0]
    //    [0.0  0.0  4.0]
    //    [2.0  0.0  0.0]
    int m = 3, n = 3, lda = 3;
    std::vector<float> hA(static_cast<size_t>(lda) * n, 0.0f);
    hA[0 * lda + 0] = 1.0f;  // A[0][0]
    hA[0 * lda + 2] = 2.0f;  // A[2][0]
    hA[2 * lda + 0] = 3.0f;  // A[0][2]
    hA[2 * lda + 1] = 4.0f;  // A[1][2]

    // 3. 申请 Device 内存并拷贝数据
    void* rawMemA = nullptr;
    auto aclRet = aclrtMalloc(&rawMemA, static_cast<size_t>(lda) * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dA(rawMemA, aclrtFree);

    aclRet = aclrtMemcpy(dA.get(), static_cast<size_t>(lda) * n * sizeof(float),
                          hA.data(), static_cast<size_t>(lda) * n * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    void* rawMemNnzPerRow = nullptr;
    aclRet = aclrtMalloc(&rawMemNnzPerRow, m * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for nnzPerRow failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dNnzPerRow(rawMemNnzPerRow, aclrtFree);

    void* rawMemNnzTotal = nullptr;
    aclRet = aclrtMalloc(&rawMemNnzTotal, sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for nnzTotal failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dNnzTotal(rawMemNnzTotal, aclrtFree);

    // 4. 创建 MatDescr（Legacy API）
    aclsparseMatDescr_t matDescr = nullptr;
    sparseRet = aclsparseCreateMatDescr(&matDescr);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateMatDescr failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    aclsparseSetMatType(matDescr, ACL_SPARSE_MATRIX_TYPE_GENERAL);
    aclsparseSetMatIndexBase(matDescr, ACL_SPARSE_INDEX_BASE_ZERO);

    // 5. 调用 aclsparseSnnz（按行统计）
    aclsparseDirection_t dirA = ACL_SPARSE_DIRECTION_ROW;
    sparseRet = aclsparseSnnz(
        static_cast<aclsparseHandle_t>(handlePtr.get()), dirA, m, n, matDescr,
        static_cast<const float*>(dA.get()), lda,
        static_cast<int*>(dNnzPerRow.get()), static_cast<int*>(dNnzTotal.get()));
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSnnz failed. ERROR: %d\n", sparseRet);
              aclsparseDestroyMatDescr(matDescr); return sparseRet);

    // 6. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 7. 将结果从 Device 拷贝回 Host 并打印
    std::vector<int> hNnzPerRow(m, 0);
    aclRet = aclrtMemcpy(hNnzPerRow.data(), m * sizeof(int),
                          dNnzPerRow.get(), m * sizeof(int),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy nnzPerRow from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);

    int hNnzTotal = 0;
    aclRet = aclrtMemcpy(&hNnzTotal, sizeof(int),
                          dNnzTotal.get(), sizeof(int),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy nnzTotal from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);

    for (int i = 0; i < m; i++) {
        LOG_PRINT("nnzPerRow[%d] is: %d\n", i, hNnzPerRow[i]);
    }
    LOG_PRINT("nnzTotal is: %d\n", hNnzTotal);

    // 8. 清理 MatDescr
    aclsparseDestroyMatDescr(matDescr);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseSnnzTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSnnzTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
nnzPerRow[0] is: 2
nnzPerRow[1] is: 1
nnzPerRow[2] is: 1
nnzTotal is: 4
```
