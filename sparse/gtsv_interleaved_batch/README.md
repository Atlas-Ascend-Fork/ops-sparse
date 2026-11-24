# 三对角线性方程组批量求解算子（Interleaved Batch）

## 算子概述

gtsv_interleaved_batch 算子用于批量求解多个独立的三对角线性方程组。给定 `batchCount` 个三对角矩阵 **A^(k)** 和对应的右端向量 **b^(k)**，求解：

**A^(k) · x^(k) = b^(k)**, k = 0, 1, ..., batchCount - 1

其中每个三对角矩阵 **A^(k)** 为 m × m 方阵，由三条对角线表示：

- `dl`：下对角线（sub-diagonal）
- `d`：主对角线（main diagonal）
- `du`：上对角线（super-diagonal）

采用 **row-major interleaved** 数据布局，即将每条对角线视为 `[m][batchCount]` 的二维数组，同一行索引的所有 batch 数据在内存中连续存放：

```
data[row][batch] = data[row * batchCount + batch]
```

当前版本使用 Thomas 算法（algo=0）实现，算法流程分为两步：

1. **前向消元（Forward Sweep）**：逐行修改主对角线和右端项，得到等价的上三角系统
2. **后向回代（Backward Substitution）**：从最后一行向上逐步求解 x

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSgtsvInterleavedBatch | 批量求解三对角线性方程组（FP32），Thomas 算法 |
| aclsparseSgtsvInterleavedBatch_bufferSizeExt | 查询 aclsparseSgtsvInterleavedBatch 所需工作区大小 |

## 算子执行接口

### aclsparseSgtsvInterleavedBatch

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSgtsvInterleavedBatch(
    aclsparseHandle_t handle,
    int algo, int m,
    float *dl, float *d, float *du, float *x,
    int batchCount, void *pBuffer);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream | Host 内存 |
| algo | 输入 | int | 算法选择：0 = Thomas 算法（当前仅支持此选项） | Host 内存 |
| m | 输入 | int | 线性系统大小（行数 = 列数），需 ≥ 1 | Host 内存 |
| dl | 输入 | float* | 下对角线数组，大小为 m × batchCount（row-major interleaved）。`dl[0][k]`（即 `dl[k]`，k = 0..batchCount-1）为越界元素，调用前必须置 0 | Device 内存 |
| d | 输入 | float* | 主对角线数组，大小为 m × batchCount（row-major interleaved），不可为 nullptr | Device 内存 |
| du | 输入 | float* | 上对角线数组，大小为 m × batchCount（row-major interleaved）。`du[m-1][k]`（即 `du[(m-1)*batchCount + k]`）为越界元素，调用前必须置 0 | Device 内存 |
| x | 输入/输出 | float* | 右端项 b（输入）/ 解 x（输出），大小为 m × batchCount（row-major interleaved）。输入时为右端向量 b，计算完成后原地覆盖为解向量 x | Device 内存 |
| batchCount | 输入 | int | 批次数（独立方程组个数），需 ≥ 1 | Host 内存 |
| pBuffer | 输入 | void* | 工作区 buffer，地址必须 128 字节对齐。当 m = 1 时可为 nullptr；当 m > 1 时必须为有效指针，大小不少于 bufferSizeExt 返回的字节数 | Device 内存 |

#### 约束说明

- m ≥ 1
- batchCount ≥ 1
- algo 当前仅支持 0（Thomas 算法），algo = 1 或 2 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`，algo < 0 或 algo > 2 返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- dl、d、du、x 均不可为 nullptr
- 当 m > 1 时，pBuffer 不可为 nullptr，且地址必须是 128 字节的整数倍
- pBuffer 大小必须 ≥ `aclsparseSgtsvInterleavedBatch_bufferSizeExt` 返回值
- `dl[0][k]`（k = 0..batchCount-1）必须在调用前置 0
- `du[m-1][k]`（k = 0..batchCount-1）必须在调用前置 0
- Thomas 算法（algo=0）为**无 pivoting 的消元法**，仅对**对角占优**或**良态（well-conditioned）**的三对角矩阵数值稳定。算子不对奇异矩阵做任何保护：若前向消元或后向回代过程中任一主元 d'[i] 为零或接近零，IEEE-754 除零将在输出中产生 Inf/NaN。**调用方必须保证输入的三对角矩阵良态，否则该 batch 的精度不可预期**。

### aclsparseSgtsvInterleavedBatch_bufferSizeExt

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSgtsvInterleavedBatch_bufferSizeExt(
    aclsparseHandle_t handle,
    int algo, int m,
    const float *dl, const float *d, const float *du, const float *x,
    int batchCount, size_t *pBufferSizeInBytes);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄 | Host 内存 |
| algo | 输入 | int | 算法选择：0 = Thomas 算法（当前仅支持此选项） | Host 内存 |
| m | 输入 | int | 线性系统大小（行数 = 列数），需 ≥ 1 | Host 内存 |
| dl | 输入 | const float* | 下对角线数组指针（查询时可传 nullptr，bufferSizeExt 不读取） | Device 内存 |
| d | 输入 | const float* | 主对角线数组指针（查询时可传 nullptr，bufferSizeExt 不读取） | Device 内存 |
| du | 输入 | const float* | 上对角线数组指针（查询时可传 nullptr，bufferSizeExt 不读取） | Device 内存 |
| x | 输入 | const float* | 右端项/解向量数组指针（查询时可传 nullptr，bufferSizeExt 不读取） | Device 内存 |
| batchCount | 输入 | int | 批次数（独立方程组个数），需 ≥ 1 | Host 内存 |
| pBufferSizeInBytes | 输出 | size_t* | 输出 aclsparseSgtsvInterleavedBatch 所需工作区大小（字节），不可为 nullptr | Host 内存 |

#### 约束说明

- m ≥ 1
- batchCount ≥ 1
- algo 当前仅支持 0（Thomas 算法），algo = 1 或 2 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`，algo < 0 或 algo > 2 返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- pBufferSizeInBytes 不可为 nullptr
- 当 m = 1 时，返回 0（无需工作区）
- 返回的工作区大小已包含 128 字节对齐

## 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | 否 | 不适用（该算子为三对角求解器，不使用 CSR 格式） |
| COO | 否 | 不适用（该算子为三对角求解器，不使用 COO 格式） |
| CSC | 否 | 不适用（该算子为三对角求解器，不使用 CSC 格式） |
| 三对角（Tridiagonal） | 是 | 使用三条对角线（dl, d, du）的密集数组表示，row-major interleaved 布局 |

## 多精度版本列表

| 精度前缀 | 数据类型 | 支持情况 |
|---------|---------|---------|
| S | float（FP32） | 支持 |
| D | double（FP64） | 不支持 |
| C | cuComplex（FP32 复数） | 不支持 |
| Z | cuDoubleComplex（FP64 复数） | 不支持 |

当前仅实现 S 前缀（FP32）版本。

## 调用示例

示例代码如下，仅供参考，具体编译和运行过程请参考[编译与运行样例](../../docs/zh/install/quick_install.md)。

```cpp
#include <cstdio>
#include <cstdlib>
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

int aclsparseSgtsvInterleavedBatchTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet); return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)>
        handlePtr(rawHandle, aclsparseDestroy);

    sparseRet = aclsparseSetStream(static_cast<aclsparseHandle_t>(handlePtr.get()), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet); return sparseRet);

    // 2. 准备 Host 数据
    //    示例：m=3, batchCount=2，求解 2 个 3x3 三对角方程组
    //
    //    方程组 0:              方程组 1:
    //    [ 2  1  0] [x0]   [1]   [ 3  1  0] [x0]   [4]
    //    [ 1  2  1] [x1] = [0]   [ 1  3  1] [x1] = [3]
    //    [ 0  1  2] [x2]   [1]   [ 0  1  3] [x2]   [4]
    //
    //    row-major interleaved: data[row * batchCount + batch]

    const int m = 3;
    const int batchCount = 2;
    const int totalElems = m * batchCount;  // 6

    // dl: 下对角线 [m][batchCount], dl[0][*] 必须为 0
    std::vector<float> hDl(totalElems, 0.0f);
    // dl[0][0]=0, dl[0][1]=0 (越界元素，已初始化为 0)
    hDl[1 * batchCount + 0] = 1.0f;  // dl[1][0]
    hDl[1 * batchCount + 1] = 1.0f;  // dl[1][1]
    hDl[2 * batchCount + 0] = 1.0f;  // dl[2][0]
    hDl[2 * batchCount + 1] = 1.0f;  // dl[2][1]

    // d: 主对角线 [m][batchCount]
    std::vector<float> hD(totalElems, 0.0f);
    hD[0 * batchCount + 0] = 2.0f;  // d[0][0]
    hD[0 * batchCount + 1] = 3.0f;  // d[0][1]
    hD[1 * batchCount + 0] = 2.0f;  // d[1][0]
    hD[1 * batchCount + 1] = 3.0f;  // d[1][1]
    hD[2 * batchCount + 0] = 2.0f;  // d[2][0]
    hD[2 * batchCount + 1] = 3.0f;  // d[2][1]

    // du: 上对角线 [m][batchCount], du[m-1][*] 必须为 0
    std::vector<float> hDu(totalElems, 0.0f);
    hDu[0 * batchCount + 0] = 1.0f;  // du[0][0]
    hDu[0 * batchCount + 1] = 1.0f;  // du[0][1]
    hDu[1 * batchCount + 0] = 1.0f;  // du[1][0]
    hDu[1 * batchCount + 1] = 1.0f;  // du[1][1]
    // du[2][0]=0, du[2][1]=0 (越界元素，已初始化为 0)

    // x: 右端项 b [m][batchCount] (输入), 将被覆盖为解
    std::vector<float> hX(totalElems, 0.0f);
    hX[0 * batchCount + 0] = 1.0f;  // b[0][0]
    hX[0 * batchCount + 1] = 4.0f;  // b[0][1]
    hX[1 * batchCount + 0] = 0.0f;  // b[1][0]
    hX[1 * batchCount + 1] = 3.0f;  // b[1][1]
    hX[2 * batchCount + 0] = 1.0f;  // b[2][0]
    hX[2 * batchCount + 1] = 4.0f;  // b[2][1]

    // 3. 申请 Device 内存并拷贝数据
    const size_t dataSize = static_cast<size_t>(totalElems) * sizeof(float);

    void *dDl = nullptr, *dD = nullptr, *dDu = nullptr, *dX = nullptr;
    auto aclRet = aclrtMalloc(&dDl, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dl failed\n"); return aclRet);
    aclRet = aclrtMalloc(&dD, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for d failed\n"); return aclRet);
    aclRet = aclrtMalloc(&dDu, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for du failed\n"); return aclRet);
    aclRet = aclrtMalloc(&dX, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed\n"); return aclRet);

    aclrtMemcpy(dDl, dataSize, hDl.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dD, dataSize, hD.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dDu, dataSize, hDu.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dX, dataSize, hX.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // 4. 查询所需 buffer 大小
    size_t bufferSize = 0;
    sparseRet = aclsparseSgtsvInterleavedBatch_bufferSizeExt(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        0,  // algo = Thomas
        m,
        static_cast<const float*>(dDl),
        static_cast<const float*>(dD),
        static_cast<const float*>(dDu),
        static_cast<const float*>(dX),
        batchCount,
        &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("bufferSizeExt failed. ERROR: %d\n", sparseRet); return sparseRet);

    // 5. 分配并初始化工作区 buffer（128 字节对齐由 aclrtMalloc 保证）
    void *pBuffer = nullptr;
    if (bufferSize > 0) {
        aclRet = aclrtMalloc(&pBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS,
                  LOG_PRINT("aclrtMalloc for pBuffer failed\n"); return aclRet);
    }

    // 6. 调用算子求解
    sparseRet = aclsparseSgtsvInterleavedBatch(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        0,  // algo = Thomas
        m,
        static_cast<float*>(dDl),
        static_cast<float*>(dD),
        static_cast<float*>(dDu),
        static_cast<float*>(dX),
        batchCount,
        pBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSgtsvInterleavedBatch failed. ERROR: %d\n", sparseRet);
              if (pBuffer) aclrtFree(pBuffer); return sparseRet);

    // 7. 同步并读取结果
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    std::vector<float> hResult(totalElems, 0.0f);
    aclRet = aclrtMemcpy(hResult.data(), dataSize, dX, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("copy result failed. ERROR: %d\n", aclRet); return aclRet);

    // 打印解向量
    for (int k = 0; k < batchCount; k++) {
        LOG_PRINT("Batch %d solution:\n", k);
        for (int i = 0; i < m; i++) {
            LOG_PRINT("  x[%d][%d] = %f\n", i, k, hResult[i * batchCount + k]);
        }
    }

    // 8. 清理
    if (pBuffer) aclrtFree(pBuffer);
    aclrtFree(dDl);
    aclrtFree(dD);
    aclrtFree(dDu);
    aclrtFree(dX);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseSgtsvInterleavedBatchTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSgtsvInterleavedBatchTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
Batch 0 solution:
  x[0][0] = 1.000000
  x[1][0] = -1.000000
  x[2][0] = 1.000000
Batch 1 solution:
  x[0][1] = 1.285714
  x[1][1] = 0.142857
  x[2][1] = 1.285714
```
