# 三对角线性方程组带选主元求解算子（Multiple RHS）

## 算子概述

gtsv2 算子用于求解单个三对角线性方程组，支持多个右端项（Multiple Right-Hand Sides）。给定一个 m × m 的三对角矩阵 **A** 和 m × n 的稠密右端矩阵 **B**，求解：

**A · X = B**

其中三对角矩阵 **A** 由三条对角线表示：

- `dl`：下对角线（sub-diagonal），dl[0] 必须为 0
- `d`：主对角线（main diagonal）
- `du`：上对角线（super-diagonal），du[m-1] 必须为 0

B 为 **column-major** 布局的稠密矩阵（ldb × n），解 X 原地覆盖 B。

采用 **带部分选主元（partial pivoting）的 Thomas 算法**，流程分为两步：

1. **前向消元（Forward Elimination）**：逐行消去下对角线元素，每步比较 |d'[i-1]| 与 |dl[i]| 选主元；若 |dl[i]| > |d'[i-1]| 则交换行 i-1 与 i 的三对角系数及 B 对应行，消元后 du 和 du2 均产生 fill-in
2. **后向回代（Backward Substitution）**：从最后一行向上逐步求解 x，使用含 fill-in 的修改后 du' 和 du2'

相比 gtsv2_nopivot（无主元版本），带主元的 Thomas 算法数值更稳定、精度更高。不同架构采用不同实现：arch35 为每个 RHS 独立保存 3 组因子；arch22 将公共矩阵只分解一次，并以 4 组 FP32 因子和 1 组逐行 pivot 标记供全部 RHS 复用。

多核并行沿 RHS 列数 n 维度分布：arch35 的每个 SIMT thread 处理一个 RHS 列；arch22 的快速路径由每个 AIV core 以 grid-stride 方式处理一个或多个 RHS 列。

arch22 对具有充分行对角占优裕量的矩阵（每行 `abs(d) > 2*(abs(dl)+abs(du))`）提供 FP32 向量 Jacobi 快速路径。
每个 RHS 还须满足所有输入有限、绝对值求和不溢出及保守归一化界 `max(abs(B_column))/min(abs(d)) <= 4`。
收缩因子小于 1/2，40 次固定迭代消除截断误差；RHS 范围限制控制舍入误差，不放宽对外精度标准。
`128<=m<=6144` 使用带显式边界区的 UB 双缓冲；`6144<m<=8192` 的多 RHS 路径使用六组对齐 FP32 数组，最多占 192 KiB UB，
将归一化 RHS 临时写入原地 B，并在每次迭代重新载入，最终原地写回解，不增加 workspace。
单 RHS 的 UB 内向量快路与直接求解均位于同一个 AIV Kernel 内；不满足上述条件、较小系统或大于 UB 的系统保留带主元直接路径。
所有选择只依赖尺寸、矩阵性质与输入动态范围，不依赖测试种子或用例编号；Inf/NaN 和大动态范围 RHS 在改写 B 前回退。

arch22 的直接/向量快速路径还要求 `2^-24 <= min(abs(d))`、`max(abs(d)) <= 2^24`、`max(abs(B)) <= 2^24`，在充分行对角占优条件下给消元与回代保留 FP32 动态范围裕量。分解 Kernel 在改写 B 前检查所有 RHS。

其他有限系统由 core 0 使用扩展指数的带主元 LU：每个中间值分别保存两组 FP32 尾数和一个 int64 指数，避免 Schur 补、RHS 前向消元及回代中的中间溢出。每行以精确二次幂进行均衡，只调整独立指数；同一行或不同行中大小悬殊的系数均独立保留。初始输出的分量后向误差超过 1e-6 时，比较不均衡分解并保留误差较小者。五组长度 m 的中间数组和一组原始 FP32 RHS 供所有列复用。有限输出始终尝试两轮扩展指数残差修正：在原始方程计算 b−Ax，复用所选 LU 求解修正量，整列确认更新值有限后写回。病态系统即使残差很小也可能有较大的解误差，因此不能仅因残差小跳过修正；此方法不保证所有病态系统的前向精度。运算仍在 NPU 上执行，不使用硬件 FP64 指令或 CPU fallback。尾数的关键和/积使用舍入屏障维持补偿运算顺序，最终按 FP32 舍入并保留非规格化输出。非有限输入与精确零主元回退到原有 FP32 运算顺序以保留 Inf/NaN 传播语义。

workspace 查询在 `n>0` 时返回 `align_up(align_up(17*m,128)+85*m,128)`，`n=0` 返回 0。调用方必须重新查询当前库所需容量，不得硬编码历史容量；仍须提供足量、128 字节对齐的 workspace，公共 API 与 stream 异步语义不变。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSgtsv2 | 求解三对角线性方程组（FP32），带部分选主元的 Thomas 算法 |
| aclsparseSgtsv2_bufferSizeExt | 查询 aclsparseSgtsv2 所需工作区大小 |

## 算子执行接口

### aclsparseSgtsv2

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22 / DAV_2201）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22 / DAV_2201）

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSgtsv2(
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
| n | 输入 | int | 右端项列数，需 ≥ 0；n=0 时提前返回 SUCCESS 不计算 | Host 内存 |
| dl | 输入 | const float* | 下对角线数组 [m]，dl[0] 必须在调用前置 0 | Device 内存 |
| d | 输入 | const float* | 主对角线数组 [m] | Device 内存 |
| du | 输入 | const float* | 上对角线数组 [m]，du[m-1] 必须在调用前置 0 | Device 内存 |
| B | 输入/输出 | float* | 右端项 b（输入）/ 解 X（输出），大小为 ldb × n（column-major）。输入时为右端矩阵 B，计算完成后原地覆盖为解矩阵 X | Device 内存 |
| ldb | 输入 | int | B 的 leading dimension，需 ≥ max(1, m) | Host 内存 |
| pBuffer | 输入 | void* | 工作区 buffer，地址必须 128 字节对齐。当 n > 0 时必须为有效指针，大小不少于 bufferSizeExt 返回的字节数 | Device 内存 |

#### 约束说明

- m ≥ 3
- n ≥ 0（n = 0 时提前返回 SUCCESS 不计算）
- ldb ≥ max(1, m)
- 当 n > 0 时，dl、d、du、B 均不可为 nullptr
- 当 n > 0 时，pBuffer 不可为 nullptr，且地址必须是 128 字节的整数倍
- pBuffer 大小必须 ≥ `aclsparseSgtsv2_bufferSizeExt` 返回值
- dl[0] 必须在调用前置 0
- du[m-1] 必须在调用前置 0
- 零主元可能产生 Inf/NaN，沿用 Legacy API 的传播行为。有限病态输入仍须按 FP64 golden 与任务精度阈值验证，不能仅以“病态”标签豁免精度失败；扩展指数和双分量尾数也不是对任意条件数矩阵的无限精度保证。

arch22 实现说明：

- workspace 为 `align_up(align_up(17*m,128)+85*m,128)` 字节，与 RHS 数量 n 无关；包括原有四组 FP32 因子/pivot 标记，以及扩展指数因子、单列 RHS、原始 FP32 RHS 和独立 pivot scratch。`m=8192` 时为 835584 字节。
- `m <= 8192` 时，多核沿 RHS 列并行；受数值条件保护的向量 Jacobi 与带主元直接求解共存。`m > 8192` 使用单核 GM 标量回退路径，不做 CPU fallback。
- 当 `ldb × sizeof(float)` 不是 32 字节对齐时，为避免非对齐 DMA 的相邻列边界冲突，arch22 采用单核求解；B 的 padding 行不会被改写。

### aclsparseSgtsv2_bufferSizeExt

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22 / DAV_2201）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22 / DAV_2201）

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSgtsv2_bufferSizeExt(
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
| n | 输入 | int | 右端项列数，需 ≥ 0；n=0 时返回 0 | Host 内存 |
| dl | 输入 | const float* | 下对角线数组 [m]，dl[0] 必须为 0 | Device 内存 |
| d | 输入 | const float* | 主对角线数组 [m] | Device 内存 |
| du | 输入 | const float* | 上对角线数组 [m]，du[m-1] 必须为 0 | Device 内存 |
| B | 输入 | const float* | 右端项矩阵 [ldb × n]（column-major）；查询接口不使用其值，可为 nullptr | Device 内存 |
| ldb | 输入 | int | B 的 leading dimension，需 ≥ max(1, m) | Host 内存 |
| pBufferSizeInBytes | 输出 | size_t* | 输出 aclsparseSgtsv2 所需工作区大小（字节），不可为 nullptr | Host 内存 |

#### 约束说明

- m ≥ 3
- n ≥ 0（n = 0 时返回 0，无需工作区）
- ldb ≥ max(1, m)
- dl、d、du、pBufferSizeInBytes 均不可为 nullptr
- B 可为 nullptr（查询接口不使用 B 的值）
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
| C | Complex64（FP32 实部 + FP32 虚部） | 不支持 |
| Z | Complex128（FP64 实部 + FP64 虚部） | 不支持 |

当前仅实现 S 前缀（FP32）版本。

## 精度标准

| dtype | rtol | atol | 说明 |
|-------|------|------|------|
| FP32 | 2^-10 | 2^-16 | 逐元素混合容差；整体匹配率 ≥ 0.99，绝对误差上限为 max(1e-2, 32 × ULP(golden)) |

## 调用示例

示例代码如下，仅供参考，具体编译和运行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

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

int aclsparseSgtsv2Test(AclContext& ctx)
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
    //    使用非对角占优矩阵以演示 partial pivoting 的效果
    //
    //    三对角矩阵 A:
    //    [ 1  1  0  0]
    //    [ 2  2  1  0]
    //    [ 0  3  2  1]
    //    [ 0  0  4  2]
    //
    //    RHS 矩阵 B (column-major, ldb=4):
    //    列 1: [2, 5, 6, 6]^T
    //    列 2: [1, 0, 0, 0]^T

    const int m = 4;
    const int n = 2;
    const int ldb = 4;

    // dl: 下对角线 [m], dl[0] 必须为 0
    std::vector<float> hDl = {0.0f, 2.0f, 3.0f, 4.0f};

    // d: 主对角线 [m]
    std::vector<float> hD = {1.0f, 2.0f, 2.0f, 2.0f};

    // du: 上对角线 [m], du[m-1] 必须为 0
    std::vector<float> hDu = {1.0f, 1.0f, 1.0f, 0.0f};

    // B: 右端项 [ldb * n] column-major
    //    列 0: [2, 5, 6, 6], 列 1: [1, 0, 0, 0]
    std::vector<float> hB(ldb * n, 0.0f);
    hB[0] = 2.0f;  hB[4] = 1.0f;
    hB[1] = 5.0f;  hB[5] = 0.0f;
    hB[2] = 6.0f;  hB[6] = 0.0f;
    hB[3] = 6.0f;  hB[7] = 0.0f;

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
    sparseRet = aclsparseSgtsv2_bufferSizeExt(
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
    sparseRet = aclsparseSgtsv2(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        m, n,
        static_cast<const float*>(dDl),
        static_cast<const float*>(dD),
        static_cast<const float*>(dDu),
        static_cast<float*>(dB),
        ldb, pBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSgtsv2 failed. ERROR: %d\n", sparseRet);
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

    ret = aclsparseSgtsv2Test(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSgtsv2Test failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
RHS column 0 solution:
  X[0][0] = 1.000000
  X[1][0] = 1.000000
  X[2][0] = 1.000000
  X[3][0] = 1.000000
RHS column 1 solution:
  X[0][1] = 1.000000
  X[1][1] = 0.000000
  X[2][1] = -2.000000
  X[3][1] = 4.000000
```
