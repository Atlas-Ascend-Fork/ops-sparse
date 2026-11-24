# 五对角线性方程组批量求解算子（Interleaved Batch）

## 算子概述

gpsv_interleaved_batch 算子用于批量求解多个独立的五对角线性方程组。给定 `batchCount` 个五对角矩阵 **A^(k)** 和对应的右端向量 **b^(k)**，求解：

**A^(k) · x^(k) = b^(k)**, k = 0, 1, ..., batchCount - 1

其中每个五对角矩阵 **A^(k)** 为 m × m 方阵，由五条对角线表示：

- `ds`：下第二对角线（distance-2 sub-diagonal）
- `dl`：下对角线（sub-diagonal）
- `d`：主对角线（main diagonal）
- `du`：上对角线（super-diagonal）
- `dw`：上第二对角线（distance-2 super-diagonal）

采用 **row-major interleaved** 数据布局，即将每条对角线视为 `[m][batchCount]` 的二维数组，同一行索引的所有 batch 数据在内存中连续存放：

```
data[row][batch] = data[row * batchCount + batch]
```

当前版本使用 QR 分解算法（algo=0，基于 Givens 旋转）实现，算法流程分为两步：

1. **前向 Givens 旋转（Forward Givens Rotation）**：逐行通过正交旋转消除下第二对角线 `ds` 和下对角线 `dl`，得到上三角带状矩阵 R。对于第 i 行：
   - 先对第 i-2 行和第 i 行施加 Givens 旋转，消除 `ds[i]`
   - 再对第 i-1 行和第 i 行施加 Givens 旋转，消除 `dl[i]`
2. **后向回代（Backward Substitution）**：从最后一行向上逐步求解 x

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSgpsvInterleavedBatch | 批量求解五对角线性方程组（FP32），QR 分解算法（Givens 旋转） |
| aclsparseSgpsvInterleavedBatch_bufferSizeExt | 查询 aclsparseSgpsvInterleavedBatch 所需工作区大小 |

## 算子执行接口

### aclsparseSgpsvInterleavedBatch

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSgpsvInterleavedBatch(
    aclsparseHandle_t handle,
    int algo, int m,
    float *ds, float *dl, float *d, float *du, float *dw, float *x,
    int batchCount, void *pBuffer);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream | Host 内存 |
| algo | 输入 | int | 算法选择：0 = QR 分解（Givens 旋转，当前仅支持此选项） | Host 内存 |
| m | 输入 | int | 线性系统大小（行数 = 列数），需 ≥ 1 | Host 内存 |
| ds | 输入 | float* | 下第二对角线数组，大小为 m × batchCount（row-major interleaved）。`ds[0][*]` 和 `ds[1][*]` 为越界元素，调用前必须置 0 | Device 内存 |
| dl | 输入 | float* | 下对角线数组，大小为 m × batchCount（row-major interleaved）。`dl[0][*]` 为越界元素，调用前必须置 0 | Device 内存 |
| d | 输入 | float* | 主对角线数组，大小为 m × batchCount（row-major interleaved），不可为 nullptr | Device 内存 |
| du | 输入 | float* | 上对角线数组，大小为 m × batchCount（row-major interleaved）。`du[m-1][*]` 为越界元素，调用前必须置 0 | Device 内存 |
| dw | 输入 | float* | 上第二对角线数组，大小为 m × batchCount（row-major interleaved）。`dw[m-2][*]` 和 `dw[m-1][*]` 为越界元素，调用前必须置 0 | Device 内存 |
| x | 输入/输出 | float* | 右端项 b（输入）/ 解 x（输出），大小为 m × batchCount（row-major interleaved）。输入时为右端向量 b，计算完成后原地覆盖为解向量 x | Device 内存 |
| batchCount | 输入 | int | 批次数（独立方程组个数），需 ≥ 1 | Host 内存 |
| pBuffer | 输入 | void* | 工作区 buffer。m=1 时可传 nullptr；m>1 时需要工作区存储 R 因子对角线，大小由 bufferSizeExt 查询，地址必须 128 字节对齐 | Device 内存 |

#### 约束说明

- m ≥ 1
- batchCount ≥ 1
- algo 当前仅支持 0（QR 分解 Givens 旋转），algo = 1 或 2 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`，algo < 0 或 algo > 2 返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- ds、dl、d、du、dw、x 均不可为 nullptr
- m=1 时 pBuffer 可为 nullptr（无需工作区）；m>1 时 pBuffer 不可为 nullptr，地址必须是 128 字节的整数倍
- 越界元素约束（用户 Responsibility）：
  - `ds[0][k]` = `ds[1][k]` = 0（k = 0..batchCount-1），即 `ds[k]` 和 `ds[batchCount + k]` 必须为 0
  - `dl[0][k]` = 0（k = 0..batchCount-1），即 `dl[k]` 必须为 0
  - `du[m-1][k]` = 0（m-1≥0 时），即 `du[(m-1)*batchCount + k]` 必须为 0
  - `dw[m-2][k]` = `dw[m-1][k]` = 0（对应行索引≥0 时），即 `dw[(m-2)*batchCount + k]` 和 `dw[(m-1)*batchCount + k]` 必须为 0
- QR 分解算法（algo=0）基于 Givens 正交旋转，对一般五对角系统（包括非对角占优矩阵）具有较好的数值稳定性。但若矩阵奇异或接近奇异，旋转过程中可能出现主元为零或接近零的情况，导致解不稳定或产生 IEEE-754 标准的 Inf/NaN。**调用方必须保证输入的五对角矩阵良态，否则该 batch 的精度不可预期**。

### aclsparseSgpsvInterleavedBatch_bufferSizeExt

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSgpsvInterleavedBatch_bufferSizeExt(
    aclsparseHandle_t handle,
    int algo, int m,
    const float *ds, const float *dl, const float *d, const float *du,
    const float *dw, const float *x,
    int batchCount, size_t *pBufferSizeInBytes);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄 | Host 内存 |
| algo | 输入 | int | 算法选择：0 = QR 分解（Givens 旋转，当前仅支持此选项） | Host 内存 |
| m | 输入 | int | 线性系统大小（行数 = 列数），需 ≥ 1 | Host 内存 |
| ds | 输入 | const float* | 下第二对角线数组指针（查询时可传 nullptr，bufferSizeExt 不读取） | Device 内存 |
| dl | 输入 | const float* | 下对角线数组指针（查询时可传 nullptr，bufferSizeExt 不读取） | Device 内存 |
| d | 输入 | const float* | 主对角线数组指针（查询时可传 nullptr，bufferSizeExt 不读取） | Device 内存 |
| du | 输入 | const float* | 上对角线数组指针（查询时可传 nullptr，bufferSizeExt 不读取） | Device 内存 |
| dw | 输入 | const float* | 上第二对角线数组指针（查询时可传 nullptr，bufferSizeExt 不读取） | Device 内存 |
| x | 输入 | const float* | 右端项/解向量数组指针（查询时可传 nullptr，bufferSizeExt 不读取） | Device 内存 |
| batchCount | 输入 | int | 批次数（独立方程组个数），需 ≥ 1 | Host 内存 |
| pBufferSizeInBytes | 输出 | size_t* | 输出 aclsparseSgpsvInterleavedBatch 所需工作区大小（字节），不可为 nullptr | Host 内存 |

#### 约束说明

- m ≥ 1
- batchCount ≥ 1
- algo 当前仅支持 0（QR 分解 Givens 旋转），algo = 1 或 2 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`，algo < 0 或 algo > 2 返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- pBufferSizeInBytes 不可为 nullptr
- m=1 时返回 0（无需工作区）；m>1 时需要 `5 * ceil(m * batchCount / 32) * 32 * sizeof(float)` 字节的工作区（用于存储 R 因子的 5 条上行对角线）

## 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | 否 | 不适用（该算子为五对角求解器，不使用 CSR 格式） |
| COO | 否 | 不适用（该算子为五对角求解器，不使用 COO 格式） |
| CSC | 否 | 不适用（该算子为五对角求解器，不使用 CSC 格式） |
| 五对角（Pentadiagonal） | 是 | 使用五条对角线（ds, dl, d, du, dw）的密集数组表示，row-major interleaved 布局 |

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

int aclsparseSgpsvInterleavedBatchTest(AclContext& ctx)
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
    //    示例：m=4, batchCount=2，求解 2 个 4x4 五对角方程组
    //
    //    方程组 0:                         方程组 1:
    //    [ 4  1  1  0] [x0]   [ 6]         [ 5  1  1  0] [x0]   [ 7]
    //    [ 1  4  1  1] [x1] = [ 7]         [ 1  5  1  1] [x1] = [ 8]
    //    [ 1  1  4  1] [x2]   [ 7]         [ 1  1  5  1] [x2] = [ 8]
    //    [ 0  1  1  4] [x3]   [ 6]         [ 0  1  1  5] [x3]   [ 7]
    //
    //    row-major interleaved: data[row * batchCount + batch]

    const int m = 4;
    const int batchCount = 2;
    const int totalElems = m * batchCount;  // 8

    // ds: 下第二对角线 [m][batchCount], ds[0][*] 和 ds[1][*] 必须为 0
    std::vector<float> hDs(totalElems, 0.0f);
    // ds[0][0]=ds[0][1]=0, ds[1][0]=ds[1][1]=0 (越界元素，已初始化为 0)
    hDs[2 * batchCount + 0] = 1.0f;  // ds[2][0]
    hDs[2 * batchCount + 1] = 1.0f;  // ds[2][1]
    hDs[3 * batchCount + 0] = 1.0f;  // ds[3][0] = 1
    hDs[3 * batchCount + 1] = 1.0f;  // ds[3][1] = 1

    // dl: 下对角线 [m][batchCount], dl[0][*] 必须为 0
    std::vector<float> hDl(totalElems, 0.0f);
    // dl[0][0]=0, dl[0][1]=0 (越界元素，已初始化为 0)
    hDl[1 * batchCount + 0] = 1.0f;  // dl[1][0]
    hDl[1 * batchCount + 1] = 1.0f;  // dl[1][1]
    hDl[2 * batchCount + 0] = 1.0f;  // dl[2][0]
    hDl[2 * batchCount + 1] = 1.0f;  // dl[2][1]
    hDl[3 * batchCount + 0] = 1.0f;  // dl[3][0]
    hDl[3 * batchCount + 1] = 1.0f;  // dl[3][1]

    // d: 主对角线 [m][batchCount]
    std::vector<float> hD(totalElems, 0.0f);
    hD[0 * batchCount + 0] = 4.0f;   // d[0][0]
    hD[0 * batchCount + 1] = 5.0f;   // d[0][1]
    hD[1 * batchCount + 0] = 4.0f;   // d[1][0]
    hD[1 * batchCount + 1] = 5.0f;   // d[1][1]
    hD[2 * batchCount + 0] = 4.0f;   // d[2][0]
    hD[2 * batchCount + 1] = 5.0f;   // d[2][1]
    hD[3 * batchCount + 0] = 4.0f;   // d[3][0]
    hD[3 * batchCount + 1] = 5.0f;   // d[3][1]

    // du: 上对角线 [m][batchCount], du[m-1][*] 必须为 0
    std::vector<float> hDu(totalElems, 0.0f);
    hDu[0 * batchCount + 0] = 1.0f;  // du[0][0]
    hDu[0 * batchCount + 1] = 1.0f;  // du[0][1]
    hDu[1 * batchCount + 0] = 1.0f;  // du[1][0]
    hDu[1 * batchCount + 1] = 1.0f;  // du[1][1]
    hDu[2 * batchCount + 0] = 1.0f;  // du[2][0]
    hDu[2 * batchCount + 1] = 1.0f;  // du[2][1]
    // du[3][0]=0, du[3][1]=0 (越界元素，已初始化为 0)

    // dw: 上第二对角线 [m][batchCount], dw[m-2][*] 和 dw[m-1][*] 必须为 0
    std::vector<float> hDw(totalElems, 0.0f);
    hDw[0 * batchCount + 0] = 1.0f;  // dw[0][0]
    hDw[0 * batchCount + 1] = 1.0f;  // dw[0][1]
    hDw[1 * batchCount + 0] = 1.0f;  // dw[1][0]
    hDw[1 * batchCount + 1] = 1.0f;  // dw[1][1]
    // dw[2][0]=dw[2][1]=0, dw[3][0]=dw[3][1]=0 (越界元素，已初始化为 0)

    // x: 右端项 b [m][batchCount] (输入), 将被覆盖为解
    std::vector<float> hX(totalElems, 0.0f);
    hX[0 * batchCount + 0] = 6.0f;   // b[0][0]
    hX[0 * batchCount + 1] = 7.0f;   // b[0][1]
    hX[1 * batchCount + 0] = 7.0f;   // b[1][0]
    hX[1 * batchCount + 1] = 8.0f;   // b[1][1]
    hX[2 * batchCount + 0] = 7.0f;   // b[2][0]
    hX[2 * batchCount + 1] = 8.0f;   // b[2][1]
    hX[3 * batchCount + 0] = 6.0f;   // b[3][0]
    hX[3 * batchCount + 1] = 7.0f;   // b[3][1]

    // 3. 申请 Device 内存并拷贝数据
    const size_t dataSize = static_cast<size_t>(totalElems) * sizeof(float);

    void *dDs = nullptr, *dDl = nullptr, *dD = nullptr, *dDu = nullptr, *dDw = nullptr, *dX = nullptr;
    auto aclRet = aclrtMalloc(&dDs, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for ds failed\n"); return aclRet);
    aclRet = aclrtMalloc(&dDl, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dl failed\n"); return aclRet);
    aclRet = aclrtMalloc(&dD, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for d failed\n"); return aclRet);
    aclRet = aclrtMalloc(&dDu, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for du failed\n"); return aclRet);
    aclRet = aclrtMalloc(&dDw, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dw failed\n"); return aclRet);
    aclRet = aclrtMalloc(&dX, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed\n"); return aclRet);

    aclrtMemcpy(dDs, dataSize, hDs.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dDl, dataSize, hDl.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dD, dataSize, hD.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dDu, dataSize, hDu.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dDw, dataSize, hDw.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dX, dataSize, hX.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // 4. 查询所需 buffer 大小
    size_t bufferSize = 0;
    sparseRet = aclsparseSgpsvInterleavedBatch_bufferSizeExt(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        0,  // algo = QR/Givens
        m,
        static_cast<const float*>(dDs),
        static_cast<const float*>(dDl),
        static_cast<const float*>(dD),
        static_cast<const float*>(dDu),
        static_cast<const float*>(dDw),
        static_cast<const float*>(dX),
        batchCount,
        &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("bufferSizeExt failed. ERROR: %d\n", sparseRet); return sparseRet);

    // 5. 分配工作区 buffer（QR 算法 m>1 时需要工作区存储 R 因子对角线）
    void *pBuffer = nullptr;
    if (bufferSize > 0) {
        aclRet = aclrtMalloc(&pBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS,
                  LOG_PRINT("aclrtMalloc for pBuffer failed\n"); return aclRet);
    }

    // 6. 调用算子求解
    sparseRet = aclsparseSgpsvInterleavedBatch(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        0,  // algo = QR/Givens
        m,
        static_cast<float*>(dDs),
        static_cast<float*>(dDl),
        static_cast<float*>(dD),
        static_cast<float*>(dDu),
        static_cast<float*>(dDw),
        static_cast<float*>(dX),
        batchCount,
        pBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSgpsvInterleavedBatch failed. ERROR: %d\n", sparseRet);
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
    aclrtFree(dDs);
    aclrtFree(dDl);
    aclrtFree(dD);
    aclrtFree(dDu);
    aclrtFree(dDw);
    aclrtFree(dX);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseSgpsvInterleavedBatchTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSgpsvInterleavedBatchTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
Batch 0 solution:
  x[0][0] = 1.000000
  x[1][0] = 1.000000
  x[2][0] = 1.000000
  x[3][0] = 1.000000
Batch 1 solution:
  x[0][1] = 1.000000
  x[1][1] = 1.000000
  x[2][1] = 1.000000
  x[3][1] = 1.000000
```

## 附录：算法说明

### QR 分解（Givens 旋转）用于五对角带状矩阵

该算子采用 QR 分解（基于 Givens 正交旋转）求解五对角线性方程组。对于一个 m × m 五对角矩阵 **A**，其非零元素仅分布在五条对角线上：

```
A[i][i-2] = ds[i]   (i ≥ 2)
A[i][i-1] = dl[i]   (i ≥ 1)
A[i][i]   = d[i]    (0 ≤ i < m)
A[i][i+1] = du[i]   (0 ≤ i < m-1)
A[i][i+2] = dw[i]   (0 ≤ i < m-2)
```

**前向 Givens 旋转**：对于每行 i（从第 2 行开始），依次执行两步旋转：

1. **消除 ds[i]**：对第 i-2 行和第 i 行构造 Givens 旋转矩阵 G(i-2, i)，使得旋转后第 i 行的下第二对角线元素被消除。旋转参数为 `cs = d'[i-2] / hyp, sn = ds'[i] / hyp`，其中 `hyp = sqrt(d'[i-2]² + ds'[i]²)`
2. **消除 dl[i]**：对第 i-1 行和第 i 行构造 Givens 旋转矩阵 G(i-1, i)，使得旋转后第 i 行的下对角线元素被消除。旋转参数同理计算

旋转完成后，原矩阵被转化为上三角带状矩阵 **R** = Q^T A，同时右端项变换为 b' = Q^T b。由于 Givens 旋转的 fill-in 效应，R 包含 5 条上行对角线：

- `d'[i]`：R 的主对角线
- `du'[i]`：R 的上第一对角线 (R[i, i+1])
- `dw'[i]`：R 的上第二对角线 (R[i, i+2])
- `d4'[i]`：R 的上第三对角线 (R[i, i+3])，仅 i ≤ m-4 时有效
- `d5'[i]`：R 的上第四对角线 (R[i, i+4])，仅 i ≤ m-5 时有效
- `b'[i]`：变换后的右端项

**后向回代**：从最后一行向上求解：

```
x[m-1] = b'[m-1] / d'[m-1]
x[m-2] = (b'[m-2] - du'[m-2] * x[m-1]) / d'[m-2]
x[m-3] = (b'[m-3] - du'[m-3] * x[m-2] - dw'[m-3] * x[m-1]) / d'[m-3]
x[m-4] = (b'[m-4] - du'[m-4] * x[m-3] - dw'[m-4] * x[m-2] - d4'[m-4] * x[m-1]) / d'[m-4]
x[i]   = (b'[i] - du'[i]*x[i+1] - dw'[i]*x[i+2] - d4'[i]*x[i+3] - d5'[i]*x[i+4]) / d'[i]  (i = m-5, ..., 0)
```

该算法需要额外工作区（workspace）来存储 R 的 5 条上行对角线（d', du', dw', d4', d5'），工作区大小由 `aclsparseSgpsvInterleavedBatch_bufferSizeExt` 查询。输入对角线 ds/dl/d/du/dw 在计算过程中保持只读，不被修改。
