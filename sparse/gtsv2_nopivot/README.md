# 三对角线性方程组无选主元求解算子（Multiple RHS）

## 算子概述

gtsv2_nopivot 算子用于求解单个三对角线性方程组，支持多个右端项（Multiple Right-Hand Sides）。给定一个 m × m 的三对角矩阵 **A** 和 m × n 的稠密右端矩阵 **B**，求解：

**A · X = B**

其中三对角矩阵 **A** 由三条对角线表示：

- `dl`：下对角线（sub-diagonal），dl[0] 必须为 0
- `d`：主对角线（main diagonal）
- `du`：上对角线（super-diagonal），du[m-1] 必须为 0

B 为 **column-major** 布局的稠密矩阵（ldb × n），解 X 原地覆盖 B。

采用 **Thomas 算法**（无选主元的简化 LU 分解），对每列 RHS 独立求解，流程分为两步：

1. **前向消元（Forward Sweep）**：逐行修改主对角线和右端项
2. **后向回代（Backward Substitution）**：从最后一行向上逐步求解 x

多核并行沿 RHS 列数 n 维度分布，每个 AI Core 处理一组连续的 RHS 列。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSgtsv2Nopivot | 求解三对角线性方程组（FP32），Thomas 算法 |
| aclsparseSgtsv2Nopivot_bufferSizeExt | 查询 aclsparseSgtsv2Nopivot 所需工作区大小 |

## 算子执行接口

### aclsparseSgtsv2Nopivot

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSgtsv2Nopivot(
    aclsparseHandle_t handle,
    int m, int n,
    const float *dl, const float *d, const float *du,
    float *B, int ldb, void *pBuffer);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream | Host 内存 |
| m | 输入 | int | 线性系统大小（行数 = 列数），需 ≥ 3 | Host 内存 |
| n | 输入 | int | 右端项列数，需 ≥ 1 | Host 内存 |
| dl | 输入 | const float* | 下对角线数组 [m]，dl[0] 必须在调用前置 0 | Device 内存 |
| d | 输入 | const float* | 主对角线数组 [m] | Device 内存 |
| du | 输入 | const float* | 上对角线数组 [m]，du[m-1] 必须在调用前置 0 | Device 内存 |
| B | 输入/输出 | float* | 右端项 b（输入）/ 解 X（输出），大小为 ldb × n（column-major）。输入时为右端矩阵 B，计算完成后原地覆盖为解矩阵 X | Device 内存 |
| ldb | 输入 | int | B 的 leading dimension，需 ≥ max(1, m) | Host 内存 |
| pBuffer | 输入 | void* | 工作区 buffer，地址必须 128 字节对齐；合法输入（m ≥ 3）时必须为有效指针，大小不少于 bufferSizeExt 返回的字节数 | Device 内存 |

#### 约束说明

- m ≥ 3
- n ≥ 1
- ldb ≥ max(1, m)
- dl、d、du、B 均不可为 nullptr
- 当 m > 1 时，pBuffer 不可为 nullptr，且地址必须是 128 字节的整数倍
- pBuffer 大小必须 ≥ `aclsparseSgtsv2Nopivot_bufferSizeExt` 返回值
- dl[0] 必须在调用前置 0
- du[m-1] 必须在调用前置 0
- Thomas 算法为**无 pivoting 的消元法**，仅对**对角占优**或**良态（well-conditioned）**的三对角矩阵数值稳定。算子不对奇异矩阵做任何保护：若前向消元或后向回代过程中任一主元为零或接近零，IEEE-754 除零将在输出中产生 Inf/NaN。**调用方必须保证输入的三对角矩阵良态，否则该 RHS 列的精度不可预期**。

### aclsparseSgtsv2Nopivot_bufferSizeExt

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSgtsv2Nopivot_bufferSizeExt(
    aclsparseHandle_t handle,
    int m, int n,
    const float *dl, const float *d, const float *du,
    const float *B, int ldb,
    size_t *pBufferSizeInBytes);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄 | Host 内存 |
| m | 输入 | int | 线性系统大小（行数 = 列数），需 ≥ 3 | Host 内存 |
| n | 输入 | int | 右端项列数，需 ≥ 1 | Host 内存 |
| dl | 输入 | const float* | 下对角线数组 [m]，dl[0] 必须为 0 | Device 内存 |
| d | 输入 | const float* | 主对角线数组 [m] | Device 内存 |
| du | 输入 | const float* | 上对角线数组 [m]，du[m-1] 必须为 0 | Device 内存 |
| B | 输入 | const float* | 右端项矩阵 [ldb × n]（column-major） | Device 内存 |
| ldb | 输入 | int | B 的 leading dimension，需 ≥ max(1, m) | Host 内存 |
| pBufferSizeInBytes | 输出 | size_t* | 输出 aclsparseSgtsv2Nopivot 所需工作区大小（字节），不可为 nullptr | Host 内存 |

#### 约束说明

- m ≥ 3
- n ≥ 1
- ldb ≥ max(1, m)
- dl、d、du、B、pBufferSizeInBytes 均不可为 nullptr
- 当 m < 3 时，返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- 返回的工作区大小已包含 128 字节对齐

## 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | 否 | 不适用（该算子为三对角求解器，不使用 CSR 格式） |
| COO | 否 | 不适用（该算子为三对角求解器，不使用 COO 格式） |
| CSC | 否 | 不适用（该算子为三对角求解器，不使用 CSC 格式） |
| 三对角（Tridiagonal） | 是 | 使用三条对角线（dl, d, du）的密集数组表示 |

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

int aclsparseSgtsv2NopivotTest(AclContext& ctx)
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
    //    示例：m=4, n=2，求解 4x4 三对角方程组（2 列 RHS）
    //
    //    三对角矩阵 A:
    //    [ 2  1  0  0]
    //    [ 1  2  1  0]
    //    [ 0  1  2  1]
    //    [ 0  0  1  2]
    //
    //    RHS 矩阵 B (column-major, ldb=4):
    //    列 1: [1, 0, 0, 1]^T
    //    列 2: [2, 1, 0, 0]^T

    const int m = 4;
    const int n = 2;
    const int ldb = 4;

    // dl: 下对角线 [m], dl[0] 必须为 0
    std::vector<float> hDl = {0.0f, 1.0f, 1.0f, 1.0f};

    // d: 主对角线 [m]
    std::vector<float> hD = {2.0f, 2.0f, 2.0f, 2.0f};

    // du: 上对角线 [m], du[m-1] 必须为 0
    std::vector<float> hDu = {1.0f, 1.0f, 1.0f, 0.0f};

    // B: 右端项 [ldb * n] column-major
    //    列 0: [1, 0, 0, 1], 列 1: [2, 1, 0, 0]
    std::vector<float> hB(ldb * n, 0.0f);
    hB[0] = 1.0f;  hB[4] = 2.0f;
    hB[1] = 0.0f;  hB[5] = 1.0f;
    hB[2] = 0.0f;  hB[6] = 0.0f;
    hB[3] = 1.0f;  hB[7] = 0.0f;

    // 3. 申请 Device 内存并拷贝数据
    const size_t diagSize = static_cast<size_t>(m) * sizeof(float);
    const size_t bSize = static_cast<size_t>(ldb) * n * sizeof(float);

    void *dDl = nullptr, *dD = nullptr, *dDu = nullptr, *dB = nullptr;
    aclrtMalloc(&dDl, diagSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dD,  diagSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dDu, diagSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dB,  bSize,    ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dDl, diagSize, hDl.data(), diagSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dD,  diagSize, hD.data(),  diagSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dDu, diagSize, hDu.data(), diagSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dB,  bSize,    hB.data(),  bSize,    ACL_MEMCPY_HOST_TO_DEVICE);

    // 4. 查询所需 buffer 大小
    size_t bufferSize = 0;
    sparseRet = aclsparseSgtsv2Nopivot_bufferSizeExt(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        m, n,
        static_cast<const float*>(dDl),
        static_cast<const float*>(dD),
        static_cast<const float*>(dDu),
        static_cast<const float*>(dB),
        ldb, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("bufferSizeExt failed. ERROR: %d\n", sparseRet); return sparseRet);

    // 5. 分配并初始化工作区 buffer
    void *pBuffer = nullptr;
    if (bufferSize > 0) {
        aclrtMalloc(&pBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 6. 调用算子求解（B 原地覆盖为解 X）
    sparseRet = aclsparseSgtsv2Nopivot(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        m, n,
        static_cast<const float*>(dDl),
        static_cast<const float*>(dD),
        static_cast<const float*>(dDu),
        static_cast<float*>(dB),
        ldb, pBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSgtsv2Nopivot failed. ERROR: %d\n", sparseRet);
              if (pBuffer) aclrtFree(pBuffer); return sparseRet);

    // 7. 同步并读取结果
    aclrtSynchronizeStream(stream);

    std::vector<float> hResult(ldb * n, 0.0f);
    aclrtMemcpy(hResult.data(), bSize, dB, bSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // 打印解向量（column-major: 每列 m 个元素）
    for (int j = 0; j < n; j++) {
        LOG_PRINT("RHS column %d solution:\n", j);
        for (int i = 0; i < m; i++) {
            LOG_PRINT("  X[%d][%d] = %f\n", i, j, hResult[i + j * ldb]);
        }
    }

    // 8. 清理
    if (pBuffer) aclrtFree(pBuffer);
    aclrtFree(dDl);
    aclrtFree(dD);
    aclrtFree(dDu);
    aclrtFree(dB);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseSgtsv2NopivotTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSgtsv2NopivotTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
RHS column 0 solution:
  X[0][0] = 0.600000
  X[1][0] = -0.200000
  X[2][0] = -0.200000
  X[3][0] = 0.600000
RHS column 1 solution:
  X[0][1] = 1.000000
  X[1][1] = 0.000000
  X[2][1] = 0.000000
  X[3][1] = 0.000000
```
