# aclsparseSgtsv2StridedBatch 算子文档

## 算子概述

aclsparseSgtsv2StridedBatch 算子用于批量求解三对角线性方程组。对于 `batchCount` 个独立的 m×m 三对角方程组，求解：

```
A^(i) · y^(i) = x^(i),  i = 0, 1, ..., batchCount-1
```

其中 A^(i) 是由下对角线 `dl`、主对角线 `d`、上对角线 `du` 三个向量定义的 m×m 三对角矩阵，x^(i) 为右端项，解 y^(i) 原地覆盖 x^(i)。

三对角矩阵结构示意（m = 6）：

```
    [ d0   du0   0     0     0     0   ]
    [ dl1  d1    du1   0     0     0   ]
A = [ 0    dl2   d2    du2   0     0   ]
    [ 0    0     dl3   d3    du3   0   ]
    [ 0    0     0     dl4   d4    du4 ]
    [ 0    0     0     0     dl5   d5  ]
```

每个 batch 的三对角线数据在内存中以 `batchStride` 为间隔排列，第 i 个 batch 起始于 `ptr + batchStride × i`。

算法采用 Cyclic Reduction（CR，循环归约）紧凑化变体，通过 ⌈log₂(m)⌉ 层正向归约和反向回代求解。正向归约逐层消去偶数位置方程，每层将活跃元素紧凑化到前半部分连续区域；反向回代逐层展开，恢复完整解向量。每层归约操作使用 SIMD 向量指令（Div/Mul/Sub）批量执行，充分利用向量流水宽度。

### 支持范围

| 路径 | m 范围 | workspace | 说明 |
|------|--------|-----------|------|
| 纯 UB 路径 | 3 ≤ m ≤ 2048 | bufferSize=0 | 全程在 UB 内完成 CR 归约/回代，无需 GM workspace |
| 外层 GM 分块 CR 路径 | 2048 < m ≤ 2^30 | bufferSize>0 | 外层在 GM workspace 中按 tile 逐层正向归约（m_pad → … → 2048），归约到 2048 后整体搬入 UB 由内层 CR 完成求解，再逐层反向回代；workspace 通过 bufferSizeExt 查询 |

> m 上界 2^30 是 int32 batchStride ≥ m_pad 契约的天花板（非算法上限）：m > 2^30 时 m_pad = 2^31 超出 int32 batchStride 可表达范围，主接口返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。m ≤ 2^30 时的实际上限受 GM 内存约束（单 batch workspace ≈ 40×m_pad 字节）。

### 接口列表

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSgtsv2StridedBatch_bufferSizeExt | 查询 workspace 大小（字节） |
| aclsparseSgtsv2StridedBatch | 批量求解三对角线性方程组 A·y = x，解 y 原地覆盖 x |

## 产品支持情况

| 芯片系列 | 支持情况 |
|---------|---------|
| Ascend 950PR&950DT系列产品 | ✅ 支持（arch35, SIMD 编程模型） |
| Ascend910B | ❌ 不支持 |
| Ascend910_93 | ❌ 不支持 |

## 接口详情

### aclsparseSgtsv2StridedBatch_bufferSizeExt

#### 产品支持情况

| 芯片系列 | 支持情况 |
|---------|---------|
| Ascend 950PR&950DT系列产品 | ✅ 支持 |
| Ascend910B | ❌ 不支持 |
| Ascend910_93 | ❌ 不支持 |

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSgtsv2StridedBatch_bufferSizeExt(
    aclsparseHandle_t handle,
    int m,
    const float *dl,
    const float *d,
    const float *du,
    const float *x,
    int batchCount,
    int batchStride,
    size_t *bufferSizeInBytes);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，需通过 aclsparseCreate 初始化 | Host |
| m | 输入 | int | 每个三对角系统的维度，3 ≤ m ≤ 2^30 | Host |
| dl | 输入 | const float* | 下对角线数组，第 i 个 batch 起始于 `dl + batchStride × i` | Device |
| d | 输入 | const float* | 主对角线数组，第 i 个 batch 起始于 `d + batchStride × i` | Device |
| du | 输入 | const float* | 上对角线数组，第 i 个 batch 起始于 `du + batchStride × i` | Device |
| x | 输入 | const float* | 右端项数组，第 i 个 batch 起始于 `x + batchStride × i` | Device |
| batchCount | 输入 | int | 批量个数，batchCount ≥ 0 | Host |
| batchStride | 输入 | int | batch 间元素步长，≥ ceil(m_pad/8)×8 且为 8 的整数倍（m_pad = 2^⌈log₂(m)⌉） | Host |
| bufferSizeInBytes | 输出 | size_t* | 输出所需 workspace 大小（字节）。m ≤ 2048 时返回 0（纯 UB 路径）；2048 < m ≤ 2^30 时返回非 0（外层 GM 分块 CR 路径） | Host |

#### 约束说明

- handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`
- m ≥ 3 且 m ≤ 2^30，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`（m < 3）或 `ACL_SPARSE_STATUS_NOT_SUPPORTED`（m > 2^30）
- dl / d / du / x 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- batchCount ≥ 0，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- batchStride ≥ ceil(m_pad/8)×8（m_pad = 2^⌈log₂(m)⌉，DataCopy 32B 对齐约束）且 batchStride mod 8 = 0（kAlignFactor 对齐约束，非整数倍时第二个 batch 起始地址不满足 32B 对齐），否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- bufferSizeInBytes 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- m ≤ 2048 时返回 bufferSize = 0（纯 UB 路径，pBuffer 仅作 API 合约占位）；2048 < m ≤ 2^30 时返回预估 workspace 大小（外层 GM 分块 CR 路径，按 batch 隔离，每 batch 字节数 = 4×(S_A + S_B)×4B + 4×(m_pad − 2048)×4B，其中 S_A = align8(m_pad)+8、S_B = align8(m_pad/2)+8，单位为 float，约 40×m_pad 字节；总值按 128B 对齐）。数值示例：m=3072（m_pad=4096）时每 batch 131328 字节。注意：2048 < m ≤ 4096 区间的 workspace 较旧实现增大约 2×（旧实现仅将 save 数据外置 GM，新实现为完整外层分块 CR），属行为变化；遵循「先查询后分配」契约的调用方无感
- bufferSizeExt 不限制 m 上界校验时机：m > 2^30 时返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` 并置 bufferSize = 0

---

### aclsparseSgtsv2StridedBatch

#### 产品支持情况

| 芯片系列 | 支持情况 |
|---------|---------|
| Ascend 950PR&950DT系列产品 | ✅ 支持 |
| Ascend910B | ❌ 不支持 |
| Ascend910_93 | ❌ 不支持 |

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSgtsv2StridedBatch(
    aclsparseHandle_t handle,
    int m,
    const float *dl,
    const float *d,
    const float *du,
    float *x,
    int batchCount,
    int batchStride,
    void *pBuffer);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，需通过 aclsparseCreate 初始化 | Host |
| m | 输入 | int | 每个三对角系统的维度，3 ≤ m ≤ 2^30 | Host |
| dl | 输入 | const float* | 下对角线数组，第 i 个 batch 起始于 `dl + batchStride × i`。每个 batch 的 dl[0] = 0（由用户保证） | Device |
| d | 输入 | const float* | 主对角线数组，第 i 个 batch 起始于 `d + batchStride × i` | Device |
| du | 输入 | const float* | 上对角线数组，第 i 个 batch 起始于 `du + batchStride × i`。每个 batch 的 du[m-1] = 0（由用户保证） | Device |
| x | 输入/输出 | float* | 输入时存储右端项，输出时存储解向量（原地覆盖）。第 i 个 batch 起始于 `x + batchStride × i` | Device |
| batchCount | 输入 | int | 批量个数，batchCount ≥ 0 | Host |
| batchStride | 输入 | int | batch 间元素步长，≥ ceil(m_pad/8)×8 且为 8 的整数倍（m_pad = 2^⌈log₂(m)⌉） | Host |
| pBuffer | 输入 | void* | workspace 缓冲区，需 128B 对齐。m ≤ 2048（纯 UB 路径）时仅作 API 合约占位可为 nullptr；2048 < m ≤ 2^30（外层 GM 分块 CR 路径）时必须非 nullptr 且大小 ≥ bufferSizeExt 返回值 | Device |

#### 约束说明

- handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`
- 3 ≤ m ≤ 2^30，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`（m < 3）或 `ACL_SPARSE_STATUS_NOT_SUPPORTED`（m > 2^30）
- dl / d / du / x 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- batchCount ≥ 0，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- batchCount = 0 时直接返回 `ACL_SPARSE_STATUS_SUCCESS`，不启动 kernel
- batchStride ≥ ceil(m_pad/8)×8（m_pad = 2^⌈log₂(m)⌉，DataCopy 32B 对齐约束。原因：Kernel 使用 DataCopy 搬运 alignedM = ceil(m_pad/8)×8 个元素，若 batchStride < alignedM 会跨越 batch 边界读取相邻 batch 数据）且 batchStride mod 8 = 0（kAlignFactor 对齐约束，防止第二个 batch GM 地址不 32B 对齐），否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- pBuffer 约束：
  - m ≤ 2048（纯 UB 路径）：pBuffer 可为 nullptr（Kernel 不使用 workspace）
  - 2048 < m ≤ 2^30（外层 GM 分块 CR 路径）：pBuffer 不可为 nullptr 且需 128B 对齐，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；大小需 ≥ bufferSizeExt 返回值
- 每个 batch 的 dl[0] = 0 和 du[m-1] = 0 由用户保证（Device 侧内存无法在 Host 校验），违反将产生 Inf/NaN 结果
- 矩阵奇异（除数为 0）时产生 Inf/NaN 结果，算法不做选主元（no pivoting）

## 支持的稀疏格式

本算子为三对角批量求解，三对角矩阵由 dl/d/du 三条对角线向量隐式定义，不涉及 CSR/COO/CSC 等通用稀疏存储格式。

| 格式 | 支持 | 说明 |
|------|------|------|
| Tridiagonal (Implicit) | ✅ | 由 dl/d/du 三条向量隐式定义 |
| CSR | ❌ | 不适用（三对角为隐式格式） |
| COO | ❌ | 不适用 |
| CSC | ❌ | 不适用 |

### 多精度版本列表

本算子为 Legacy API，通过函数名精度前缀区分数据类型。当前仅支持 FP32（S 前缀）：

| 精度前缀 | 函数名 | 数据类型 | 是否支持 |
|---------|--------|---------|---------|
| S | aclsparseSgtsv2StridedBatch | FP32 | ✅ |
| D | aclsparseDgtsv2StridedBatch | FP64 | ❌ |
| C | aclsparseCgtsv2StridedBatch | Complex64 (FP32 实部 + FP32 虚部) | ❌ |
| Z | aclsparseZgtsv2StridedBatch | Complex128 (FP64 实部 + FP64 虚部) | ❌ |

### 扁平参数说明

本算子采用 Legacy API 风格，参数以扁平方式直接传入，不使用稀疏矩阵描述符（SpMatDescr）或稠密向量描述符（DnVecDescr）。参数按以下顺序排列：

1. **上下文参数**：handle（库句柄）
2. **维度参数**：m（每个三对角系统的维度）、batchCount（批量数）、batchStride（batch 间步长）
3. **数据指针**：dl（下对角线）、d（主对角线）、du（上对角线）、x（右端项/解，原地覆盖）
4. **workspace 指针**：pBuffer（workspace 缓冲区）

三对角矩阵的对角线数据通过 dl/d/du 三个裸指针直接传递，解向量通过 x 指针原地输入输出。

### MatDescr 使用方式

本算子不需要 MatDescr（矩阵描述符）。gtsv2 系列接口通过精度前缀和扁平参数直接定义矩阵属性，无矩阵格式属性需要设置。

### 数据布局

各数组在内存中以 `batchStride` 为间隔排列，每个 batch 占用前 m 个有效元素，其后为 padding 区域：

```
batchCount = 2, m = 4, batchStride = 8:

dl: [dl0[0], dl0[1], dl0[2], dl0[3], pad, pad, pad, pad, dl1[0], dl1[1], dl1[2], dl1[3], pad, pad, pad, pad]
     |<-------- batch 0 (m=4) ------>|                    |<-------- batch 1 (m=4) ------>|
     |<------------- batchStride = 8 -------------->|

d, du, x: 同上布局
```

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/install/quick_install.md)。

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

// 辅助：分配 Device 内存并拷贝 Host 数据
static void* AllocAndCopyDevice(void *hostPtr, size_t sizeBytes)
{
    void *dPtr = nullptr;
    aclrtMalloc(&dPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (hostPtr != nullptr && sizeBytes > 0) {
        aclrtMemcpy(dPtr, sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return dPtr;
}

int aclsparseGtsv2StridedBatchTest(AclContext &ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreate failed: %d\n", sparseRet);
              return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(rawHandle, aclsparseDestroy);
    aclsparseSetStream(handlePtr.get(), stream);

    // 2. 准备 Host 端三对角数据
    //    示例：2 个 batch，每个 batch 为 4x4 三对角方程组 A·y = x
    //
    //    Batch 0:
    //      A = [2 1 0 0]    x = [3]    解 y = [1]
    //          [1 2 1 0]        [4]          [1]
    //          [0 1 2 1]        [4]          [1]
    //          [0 0 1 2]        [3]          [1]
    //
    //    Batch 1:
    //      A = [2 1 0 0]    x = [4]    解 y = [1]
    //          [1 2 1 0]        [7]          [2]
    //          [0 1 2 1]        [7]          [2]
    //          [0 0 1 2]        [4]          [1]
    //
    //    边界条件：dl[0] = 0, du[m-1] = 0（由用户保证）
    //    m_pad = 4, alignedM = ceil(4/8)*8 = 8, batchStride >= 8

    int m = 4;
    int batchCount = 2;
    int batchStride = 8;  // >= ceil(m_pad/8)*8 = 8

    // Host 端数据（batchStride = 8，每个 batch 占 8 个元素，前 4 个有效，后 4 个为 padding）
    std::vector<float> hDl = {
        0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // batch 0: dl[0]=0
        0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f   // batch 1: dl[0]=0
    };
    std::vector<float> hD = {
        2.0f, 2.0f, 2.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // batch 0
        2.0f, 2.0f, 2.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f   // batch 1
    };
    std::vector<float> hDu = {
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // batch 0: du[m-1]=0
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f   // batch 1: du[m-1]=0
    };
    std::vector<float> hX = {
        3.0f, 4.0f, 4.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // batch 0: RHS
        4.0f, 7.0f, 7.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f   // batch 1: RHS
    };

    // 3. 拷贝数据到 Device
    size_t totalBytes = static_cast<size_t>(batchCount * batchStride) * sizeof(float);
    void *dDl = AllocAndCopyDevice(hDl.data(), totalBytes);
    void *dD  = AllocAndCopyDevice(hD.data(),  totalBytes);
    void *dDu = AllocAndCopyDevice(hDu.data(), totalBytes);
    void *dX  = AllocAndCopyDevice(hX.data(),  totalBytes);

    // 4. 查询 workspace 大小
    size_t bufferSize = 0;
    sparseRet = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handlePtr.get(), m,
        static_cast<const float*>(dDl),
        static_cast<const float*>(dD),
        static_cast<const float*>(dDu),
        static_cast<const float*>(dX),
        batchCount, batchStride,
        &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("bufferSizeExt failed: %d\n", sparseRet);
              return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes\n", bufferSize);

    // 5. 分配 workspace（bufferSize=0 时 pBuffer 可传 nullptr，此处分配 128 字节占位以保持一致性）
    void *pBuffer = nullptr;
    size_t allocSize = (bufferSize > 0) ? bufferSize : 128;
    aclrtMalloc(&pBuffer, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 6. 调用算子求解 A·y = x，解 y 原地覆盖 x
    sparseRet = aclsparseSgtsv2StridedBatch(
        handlePtr.get(), m,
        static_cast<const float*>(dDl),
        static_cast<const float*>(dD),
        static_cast<const float*>(dDu),
        static_cast<float*>(dX),
        batchCount, batchStride,
        pBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSgtsv2StridedBatch failed: %d\n", sparseRet);
              return sparseRet);

    // 7. 同步等待计算完成
    aclrtSynchronizeStream(stream);

    // 8. 将结果拷贝回 Host 并打印
    std::vector<float> hResult(batchCount * batchStride, 0.0f);
    aclrtMemcpy(hResult.data(), totalBytes, dX, totalBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    LOG_PRINT("\nSolution:\n");
    for (int b = 0; b < batchCount; b++) {
        LOG_PRINT("  Batch %d: ", b);
        for (int i = 0; i < m; i++) {
            LOG_PRINT("%.1f ", hResult[b * batchStride + i]);
        }
        LOG_PRINT("\n");
    }

    // 9. 清理资源
    if (pBuffer) aclrtFree(pBuffer);
    if (dDl) aclrtFree(dDl);
    if (dD)  aclrtFree(dD);
    if (dDu) aclrtFree(dDu);
    if (dX)  aclrtFree(dX);

    return 0;
}
```

> 上面 `AclContext`、`AllocAndCopyDevice`、`aclsparseGtsv2StridedBatchTest` 为纯 UB 路径的完整调用。`main` 函数（同时调用纯 UB 路径与外层 GM 分块 CR 路径）见下方外层 GM 分块 CR 路径示例末尾。

预期输出如下：

```
bufferSize = 0 bytes

Solution:
  Batch 0: 1.0 1.0 1.0 1.0
  Batch 1: 1.0 2.0 2.0 1.0
```

> 上例中 m=4 ≤ 2048，走**纯 UB 路径**，`bufferSizeExt` 返回 0，pBuffer 仅作占位。下面给出**外层 GM 分块 CR 路径**（m > 2048）的调用示例。

### 外层 GM 分块 CR 路径调用示例（m = 3072）

当 `m > 2048` 时，单核 UB 容量不足以容纳全量 CR 工作集，Kernel 采用外层 GM 分块 CR：在 GM workspace 中按 tile 逐层正向归约（m_pad → … → 2048），归约到 2048 后整体搬入 UB 完成内层 CR 求解，再逐层反向回代展开解。调用方必须：

1. 调用 `aclsparseSgtsv2StridedBatch_bufferSizeExt` 查询所需 workspace 大小（返回值 > 0）。
2. 按返回大小分配 pBuffer（`aclrtMalloc` 默认保证 128B 对齐）。
3. 将非 nullptr 的 pBuffer 传入主接口。

下面示例复用上文 `AclContext` 与 `AllocAndCopyDevice` 辅助函数，求解 2 个 batch、m=3072 的对角占优三对角方程组（d=2, dl=du=-1）。

```cpp
// 构造对角占优三对角矩阵数据（d=2, dl=du=-1，dl[0]=0, du[m-1]=0）
// 右端项 x[i] = 1 (0 < i < m-1), x[0] = x[m-1] = 0
// 理论解 y[i] = (-i^2 + (m-1)*i + (m-2)) / 2，便于人工核验首尾与中点
static void BuildDiagDomTridiag(int m, int batchCount, int batchStride,
                                std::vector<float>& hDl,
                                std::vector<float>& hD,
                                std::vector<float>& hDu,
                                std::vector<float>& hX)
{
    size_t total = static_cast<size_t>(batchCount) * batchStride;
    hDl.assign(total, 0.0f);
    hD.assign(total, 0.0f);
    hDu.assign(total, 0.0f);
    hX.assign(total, 0.0f);

    for (int b = 0; b < batchCount; b++) {
        int64_t base = static_cast<int64_t>(b) * batchStride;
        for (int i = 0; i < m; i++) {
            hD[base + i]  = 2.0f;
            hDl[base + i] = (i == 0)     ? 0.0f : -1.0f;  // dl[0] = 0
            hDu[base + i] = (i == m - 1) ? 0.0f : -1.0f;  // du[m-1] = 0
            // 右端项：内部为 1，首尾为 0
            hX[base + i] = (i == 0 || i == m - 1) ? 0.0f : 1.0f;
        }
    }
}

int aclsparseGtsv2StridedBatchGMPathTest(AclContext &ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreate failed: %d\n", sparseRet);
              return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(rawHandle, aclsparseDestroy);
    aclsparseSetStream(handlePtr.get(), stream);

    // 2. 准备 Host 端三对角数据
    //    m=3072 > 2048 -> 走外层 GM 分块 CR 路径
    //    m_pad = 4096, alignedM = ceil(4096/8)*8 = 4096, batchStride >= 4096
    int m = 3072;
    int batchCount = 2;
    int batchStride = 4096;  // >= alignedM = 4096

    std::vector<float> hDl, hD, hDu, hX;
    BuildDiagDomTridiag(m, batchCount, batchStride, hDl, hD, hDu, hX);

    // 3. 拷贝数据到 Device
    size_t totalBytes = static_cast<size_t>(batchCount * batchStride) * sizeof(float);
    void *dDl = AllocAndCopyDevice(hDl.data(), totalBytes);
    void *dD  = AllocAndCopyDevice(hD.data(),  totalBytes);
    void *dDu = AllocAndCopyDevice(hDu.data(), totalBytes);
    void *dX  = AllocAndCopyDevice(hX.data(),  totalBytes);

    // 4. 查询 workspace 大小（GM 路径返回 > 0）
    size_t bufferSize = 0;
    sparseRet = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handlePtr.get(), m,
        static_cast<const float*>(dDl),
        static_cast<const float*>(dD),
        static_cast<const float*>(dDu),
        static_cast<const float*>(dX),
        batchCount, batchStride,
        &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("bufferSizeExt failed: %d\n", sparseRet);
              return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes (outer GM CR path, m=%d > 2048)\n", bufferSize, m);
    CHECK_RET(bufferSize > 0, LOG_PRINT("ERROR: GM path should return bufferSize > 0\n"); return -1);

    // 5. 分配 workspace（GM 路径必须非 nullptr，aclrtMalloc 默认 128B 对齐）
    void *pBuffer = nullptr;
    aclrtMalloc(&pBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(pBuffer != nullptr, LOG_PRINT("aclrtMalloc pBuffer failed\n"); return -1);

    // 6. 调用算子求解 A·y = x，解 y 原地覆盖 x
    sparseRet = aclsparseSgtsv2StridedBatch(
        handlePtr.get(), m,
        static_cast<const float*>(dDl),
        static_cast<const float*>(dD),
        static_cast<const float*>(dDu),
        static_cast<float*>(dX),
        batchCount, batchStride,
        pBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSgtsv2StridedBatch failed: %d\n", sparseRet);
              return sparseRet);

    // 7. 同步等待计算完成
    aclrtSynchronizeStream(stream);

    // 8. 将结果拷贝回 Host 并抽检首尾元素
    std::vector<float> hResult(batchCount * batchStride, 0.0f);
    aclrtMemcpy(hResult.data(), totalBytes, dX, totalBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    // 对角占优矩阵 (d=2, dl=du=-1) 在 x[i]=1(内部) 下解为
    // y[i] = (-i^2 + (m-1)*i + (m-2)) / 2
    // 抽检 batch 0 的首/中/尾：y[0]=y[m-1]=(m-2)/2，y[mid] 取最大值
    int mid = (m - 1) / 2;
    float expectEnd  = static_cast<float>(m - 2) / 2.0f;
    float expectMid  = (-static_cast<float>(mid) * mid
                        + static_cast<float>(m - 1) * mid
                        + static_cast<float>(m - 2)) / 2.0f;
    LOG_PRINT("  Batch 0 y[0]=%.1f (expect %.1f), y[mid=%d]=%.1f (expect %.1f), y[m-1]=%.1f (expect %.1f)\n",
              hResult[0], expectEnd, mid, hResult[mid], expectMid,
              hResult[m - 1], expectEnd);

    // 9. 清理资源
    if (pBuffer) aclrtFree(pBuffer);
    if (dDl) aclrtFree(dDl);
    if (dD)  aclrtFree(dD);
    if (dDu) aclrtFree(dDu);
    if (dX)  aclrtFree(dX);

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 纯 UB 路径（m=4 <= 2048，bufferSize=0）
    ret = aclsparseGtsv2StridedBatchTest(ctx);
    CHECK_RET(ret == 0, LOG_PRINT("UB path test failed: %d\n", ret); return ret);

    // 外层 GM 分块 CR 路径（m=3072 > 2048，bufferSize>0）
    ret = aclsparseGtsv2StridedBatchGMPathTest(ctx);
    CHECK_RET(ret == 0, LOG_PRINT("GM path test failed: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下（bufferSize 实际值取决于 batchCount，y 值为理论解）：

```
bufferSize = 0 bytes

Solution:
  Batch 0: 1.0 1.0 1.0 1.0
  Batch 1: 1.0 2.0 2.0 1.0
bufferSize = 262656 bytes (outer GM CR path, m=3072 > 2048)
  Batch 0 y[0]=1535.0 (expect 1535.0), y[mid=1535]=1180415.0 (expect 1180415.0), y[m-1]=1535.0 (expect 1535.0)
```

### 两种路径对比

| 项 | 纯 UB 路径（m ≤ 2048） | 外层 GM 分块 CR 路径（2048 < m ≤ 2^30） |
|----|----------------------|------------------------------------------|
| bufferSizeExt 返回 | 0 | > 0（每 batch ≈ 40×m_pad 字节，× batchCount 后按 128B 对齐） |
| pBuffer 要求 | 可为 nullptr（仅占位） | 必须 非 nullptr 且 ≥ bufferSize 字节，128B 对齐 |
| Kernel 行为 | 全程在 UB 内完成 CR 归约/回代 | 外层在 GM workspace 中按 tile 逐层归约至 2048，内层整体搬入 UB 完成 CR 求解，再逐层反向回代 |
| CR 层数 | L = ⌈log₂(m_pad)⌉ ≤ 11 | 内层恒定 11 层 + 外层 K = L − 11 层（L = ⌈log₂(m_pad)⌉ ≤ 30，K ≤ 19） |
| pBuffer 对齐校验 | 不校验（UB 路径不要求） | 强制 128B 对齐 |

> 实现层数约束说明：`GTSV2_MAX_CR_LEVELS`（=16）仅约束**内层** UB CR 层数（内层恒定 11 层 ≤ 16），不再限制整体 m；外层 GM 分块 CR 层数上限由 `GTSV2_MAX_OUTER_CR_LEVELS`（=20）约束（m ≤ 2^30 时 K ≤ 19）。
>
> 行为变化说明：2048 < m ≤ 4096 区间在旧实现中走「save 数据外置 GM」路径，新实现统一为外层 GM 分块 CR；CR 逐元素数学公式与运算次序不变，求解结果与旧路径一致，但 workspace 需求增大约 2×（如 m=3072 每 batch 由约 64KB 增至 131328 字节），且 GM 流量同数量级略增。

### 精度标准

| dtype | rtol | atol | 说明 |
|-------|------|------|------|
| FP32 | ≤ 1e-5 | ≤ 1e-5 | 测试使用 MERE_MARE 模式，MERE threshold = 2^-13 ≈ 0.000122，MARE multiplier = 10.0 |
