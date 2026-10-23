# aclsparseLtMatmulAlgSetAttribute

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> Ascend 950PR/Ascend 950DT 上的 aclsparseLtMatmulAlgSetAttribute 依赖 CANN asc-devkit >= 9.1.0（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

## 功能说明

- **算子功能**：aclsparseLtMatmulAlgSetAttribute 用于在 `aclsparseLtMatmulPlanInit` 之前设置 Matmul 算法选择描述符（`aclsparseLtMatmulAlgSelection_t`）的配置属性，包括算法配置 ID（`algConfigId`）、splitK 切分因子（`splitK`）、搜索迭代次数（`searchIterations`）等。这些属性影响后续 PlanInit 计算的 cube tiling 策略与 workspace 布局。配套接口 `aclsparseLtMatmulAlgGetAttribute` 用于查询属性当前值。两个接口均为纯 Host 端元数据操作，不启动任何 kernel。该算子对标 cuSPARSELt 中的 `cusparseLtMatmulAlgSetAttribute` / `cusparseLtMatmulAlgGetAttribute`。
- **目标平台**：<term>Ascend 950PR/Ascend 950DT</term>（arch35 / DAV_3510）。
- **编程模型**：Host 端属性读写，直接操作 `aclsparseLtMatmulAlgSelection` 结构体字段。不涉及 Device 内存访问，不消耗 NPU 计算资源。

### 作用原理

属性设置发生在 aclsparseLt 调用链的"算法选择"阶段，位于 Matmul 描述符初始化之后、计划初始化之前：

```
aclsparseLtMatmulAlgSelectionInit  (默认 algConfigId=0, splitK=1, searchIterations=5)
        ↓
aclsparseLtMatmulAlgSetAttribute   (覆盖默认值)
        ↓
aclsparseLtMatmulPlanInit          (读取 algConfigId/splitK 计算 tiling)
```

各属性对 Matmul 执行的影响如下：

| 属性 | 影响 |
|------|------|
| `algConfigId` | 决定 cube tiling 的 baseM/baseN/baseK。不同配置在不同矩阵形状下性能表现不同，PlanInit 据此选择 tiling 参数 |
| `splitK` | splitK=1 走融合路径（L0C→UB→GM，无 temp 缓冲区）；splitK>1 走两段式路径（Cube 各段写 temp → epilogue 归约写 D），workspace 需分配 splitK×m×n×4 字节的 temp 区 |
| `searchIterations` | 当前仅存储，不参与执行（预留用于未来自动搜索） |
| `splitKMode` | 参与 effectiveSplitK 计算：ONE_KERNEL 模式下 effectiveSplitK=1（融合路径，无 temp 缓冲区）；TWO_KERNELS 模式下 effectiveSplitK=splitK（两段式路径，使用 temp 缓冲区） |
| `splitKBuffers` | 当前仅存储，不参与执行（预留） |

> PlanInit 会冻结 tiling 与 workspace 元数据到计划中。PlanInit 之后修改属性不会生效，需重新初始化计划。

## 算子执行接口

### aclsparseLtMatmulAlgSetAttribute

设置算法选择属性。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtMatmulAlgSetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attr,
    const void* attrValue,
    size_t attrValueSize);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseLtConstHandle_t | aclsparseLt 库句柄的 const 指针，Host 内存 |
| algSelection | 输入 | aclsparseLtMatmulAlgSelection_t* | 算法选择描述符，由 `aclsparseLtMatmulAlgSelectionInit` 创建，Host 内存 |
| attr | 输入 | aclsparseLtMatmulAlgAttribute_t | 要设置的属性枚举，Host 内存 |
| attrValue | 输入 | const void* | 属性值指针，指向 int32_t 类型值，Host 内存 |
| attrValueSize | 输入 | size_t | attrValue 缓冲区大小（字节），须为 sizeof(int32_t)，Host 内存 |

#### 可设置的属性

| 属性枚举 | 值类型 | 取值范围 | 说明 |
|----------|--------|---------|------|
| `ACLSPARSELT_MATMUL_ALG_CONFIG_ID` | int32_t | 0 / 1 | 算法配置 ID，影响 cube tiling（baseM/baseN/baseK）。需在 PlanInit 前设置 |
| `ACLSPARSELT_MATMUL_SPLIT_K` | int32_t | [1, K] | splitK 切分因子。需在 PlanInit 前设置 |
| `ACLSPARSELT_MATMUL_SEARCH_ITERATIONS` | int32_t | > 0 | 搜索迭代次数（默认 5），由 `aclsparseLtMatmulSearch` 搜索入口活读消费 |
| `ACLSPARSELT_MATMUL_SPLIT_K_MODE` | int32_t | 0 / 1 | splitK 模式（ONE_KERNEL / TWO_KERNELS），参与 effectiveSplitK 计算：ONE_KERNEL 时 effectiveSplitK=1（融合路径），TWO_KERNELS 时 effectiveSplitK=splitK（两段式路径） |
| `ACLSPARSELT_MATMUL_SPLIT_K_BUFFERS` | int32_t | [0, splitK-1] | splitK 缓冲区数，当前仅存储不参与执行 |
| `ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID` | — | 只读 | 不可设置，返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` |

#### 约束说明

- handle 不可为 nullptr 且 *handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`。
- algSelection 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- attrValue 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- attrValueSize 须与属性值类型大小匹配（int32_t 为 4 字节），否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `ACLSPARSELT_MATMUL_ALG_CONFIG_ID` 取值仅 0 或 1，其他值返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `ACLSPARSELT_MATMUL_SPLIT_K` 须 ≥ 1；若 algSelection 已绑定 matmulDesc 且 K > 0，splitK 须 ≤ K，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `ACLSPARSELT_MATMUL_SEARCH_ITERATIONS` 须 > 0，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `ACLSPARSELT_MATMUL_SPLIT_K_BUFFERS` 须在 [0, splitK-1] 范围内，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID` 为只读属性，设置时返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **属性设置时机**：须在 `aclsparseLtMatmulPlanInit` 之前调用，PlanInit 读取 algConfigId 与 splitK 计算 tiling。PlanInit 之后修改属性不会生效（需重新初始化计划）。

### aclsparseLtMatmulAlgGetAttribute

查询算法选择属性当前值。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtMatmulAlgGetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attr,
    void* attrValue,
    size_t attrValueSize);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseLtConstHandle_t | aclsparseLt 库句柄的 const 指针，Host 内存 |
| algSelection | 输入 | aclsparseLtConstMatmulAlgSelection_t* | 算法选择描述符，Host 内存 |
| attr | 输入 | aclsparseLtMatmulAlgAttribute_t | 要查询的属性枚举，Host 内存 |
| attrValue | 输出 | void* | 属性值输出缓冲区，须为 int32_t 类型，Host 内存 |
| attrValueSize | 输入 | size_t | attrValue 缓冲区大小（字节），须为 sizeof(int32_t)，Host 内存 |

#### 可查询的属性

| 属性枚举 | 返回值 | 说明 |
|----------|--------|------|
| `ACLSPARSELT_MATMUL_ALG_CONFIG_ID` | int32_t (0 / 1) | 当前算法配置 ID |
| `ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID` | int32_t (2) | 算法配置总数（algConfigId 0 和 1 共 2 种） |
| `ACLSPARSELT_MATMUL_SPLIT_K` | int32_t | 当前 splitK 切分因子 |
| `ACLSPARSELT_MATMUL_SEARCH_ITERATIONS` | int32_t | 当前搜索迭代次数（默认 5） |
| `ACLSPARSELT_MATMUL_SPLIT_K_MODE` | int32_t | 当前 splitK 模式 |
| `ACLSPARSELT_MATMUL_SPLIT_K_BUFFERS` | int32_t | 当前 splitK 缓冲区数 |

#### 约束说明

- handle 不可为 nullptr 且 *handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`。
- algSelection 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- attrValue 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- attrValueSize 须为 sizeof(int32_t)，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- 不支持的 attr 枚举值返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。

### 返回码

| 返回码 | 说明 |
|--------|------|
| `ACL_SPARSE_STATUS_SUCCESS` | 操作成功 |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为 nullptr 或 *handle 为 nullptr |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | algSelection/attrValue 为 nullptr、*algSelection 为 nullptr（已销毁）、attrValueSize 不匹配、algConfigId/splitK/searchIterations/splitKBuffers 取值非法 |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 设置只读属性（ALG_CONFIG_MAX_ID）、attr 枚举值不被支持 |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

以下示例演示通过 `aclsparseLtMatmulAlgSetAttribute` 设置 `algConfigId=1` 与 `splitK=2`，再通过 `aclsparseLtMatmulAlgGetAttribute` 回读验证，最后执行完整的剪枝 + 结构化稀疏矩阵乘流程。计算 `D = 1.0 * A_pruned * B + 0.0 * C`，其中 A 为 m×k（FP32，行主序），B 为 k×n，C/D 为 m×n，splitK=2 将 K 维度切分为 2 段并行计算。

```cpp
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparseLt.h"

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

// 辅助：分配 Device 内存并拷贝 Host 数据
static void* AllocAndCopyDevice(const void *hostPtr, size_t sizeBytes)
{
    void *dPtr = nullptr;
    aclrtMalloc(&dPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (hostPtr != nullptr && sizeBytes > 0) {
        aclrtMemcpy(dPtr, sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return dPtr;
}

int aclsparseLtAlgSetAttributeTest()
{
    int32_t deviceId = 0;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, aclFinalize(); return -1);
    aclrtStream stream = nullptr;
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, aclrtResetDevice(deviceId); aclFinalize(); return -1);

    // 1. 初始化 aclsparseLt 库句柄
    aclsparseLtHandle_t handle = nullptr;
    auto ret = aclsparseLtInit(&handle);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseLtInit failed: %d\n", ret); return ret);

    // 2. 准备矩阵维度与 Host 数据 (FP32, 行主序)
    //    k 须为 2 的倍数（FP32 剪枝分组为 2）；splitK=2 须 ≤ k
    int64_t m = 64, k = 64, n = 128;
    float alpha = 1.0f, beta = 0.0f;
    std::vector<float> hA(static_cast<size_t>(m) * k);
    std::vector<float> hB(static_cast<size_t>(k) * n);
    std::vector<float> hC(static_cast<size_t>(m) * n, 0.0f);
    for (size_t i = 0; i < hA.size(); i++) { hA[i] = static_cast<float>(i % 11) - 5.0f; }
    for (size_t i = 0; i < hB.size(); i++) { hB[i] = static_cast<float>(i % 7) - 3.0f; }

    // 3. 拷贝数据到 Device
    void *dA = AllocAndCopyDevice(hA.data(), static_cast<size_t>(m) * k * sizeof(float));
    void *dB = AllocAndCopyDevice(hB.data(), static_cast<size_t>(k) * n * sizeof(float));
    void *dC = AllocAndCopyDevice(hC.data(), static_cast<size_t>(m) * n * sizeof(float));
    void *dD = nullptr;
    aclrtMalloc(&dD, static_cast<size_t>(m) * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    void *dAPruned = nullptr;
    aclrtMalloc(&dAPruned, static_cast<size_t>(m) * k * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    // 4. 创建矩阵描述符
    aclsparseLtMatDescriptor_t matA = nullptr;
    aclsparseLtMatDescriptor_t matB = nullptr, matC = nullptr, matD = nullptr;
    aclsparseLtStructuredDescriptorInit(
        &handle, &matA, m, k, k, 16, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    aclsparseLtDenseDescriptorInit(&handle, &matB, k, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matC, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matD, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);

    // 5. 创建 Matmul 描述符 (opA=N, opB=N, FP32 累加)
    aclsparseLtMatmulDescriptor_t matmulDesc = nullptr;
    ret = aclsparseLtMatmulDescriptorInit(
        &handle, &matmulDesc,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &matA, &matB, &matC, &matD, ACL_SPARSE_COMPUTE_32F);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("MatmulDescriptorInit failed: %d\n", ret); return ret);

    // 6. 创建算法选择描述符（默认 algConfigId=0, splitK=1, searchIterations=5）
    aclsparseLtMatmulAlgSelection_t algSel = nullptr;
    ret = aclsparseLtMatmulAlgSelectionInit(&handle, &algSel, &matmulDesc, ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("AlgSelectionInit failed: %d\n", ret); return ret);

    // 7. ★ 核心接口：设置 algConfigId=1 与 splitK=2
    int32_t algConfigId = 1;
    int32_t splitK = 2;
    ret = aclsparseLtMatmulAlgSetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_ALG_CONFIG_ID,
                                           &algConfigId, sizeof(int32_t));
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("AlgSetAttribute(ALG_CONFIG_ID) failed: %d\n", ret); return ret);
    ret = aclsparseLtMatmulAlgSetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_SPLIT_K,
                                           &splitK, sizeof(int32_t));
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("AlgSetAttribute(SPLIT_K) failed: %d\n", ret); return ret);

    // 8. ★ 通过 AlgGetAttribute 回读验证设置是否生效
    int32_t getAlgConfigId = -1;
    int32_t getSplitK = -1;
    ret = aclsparseLtMatmulAlgGetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_ALG_CONFIG_ID,
                                           &getAlgConfigId, sizeof(int32_t));
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("AlgGetAttribute(ALG_CONFIG_ID) failed: %d\n", ret); return ret);
    ret = aclsparseLtMatmulAlgGetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_SPLIT_K,
                                           &getSplitK, sizeof(int32_t));
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("AlgGetAttribute(SPLIT_K) failed: %d\n", ret); return ret);
    LOG_PRINT("algConfigId: set=%d, get=%d\n", algConfigId, getAlgConfigId);
    LOG_PRINT("splitK: set=%d, get=%d\n", splitK, getSplitK);

    //    查询 ALG_CONFIG_MAX_ID（只读，返回 2，表示 algConfigId 0 和 1 共 2 种配置）
    int32_t maxId = 0;
    ret = aclsparseLtMatmulAlgGetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
                                           &maxId, sizeof(int32_t));
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("AlgGetAttribute(MAX_ID) failed: %d\n", ret); return ret);
    LOG_PRINT("ALG_CONFIG_MAX_ID = %d\n", maxId);

    // 9. 创建执行计划（PlanInit 读取 algConfigId=1 与 splitK=2 计算 tiling）
    aclsparseLtMatmulPlan_t plan = nullptr;
    ret = aclsparseLtMatmulPlanInit(&handle, &plan, &matmulDesc, &algSel);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("PlanInit failed: %d\n", ret); return ret);

    // 10. 查询 workspace 大小并分配（splitK>1 时包含 temp 区）
    size_t workspaceSize = 0;
    ret = aclsparseLtMatmulGetWorkspace(&handle, &plan, &workspaceSize);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("GetWorkspaceSize failed: %d\n", ret); return ret);
    LOG_PRINT("workspaceSize = %zu bytes (splitK=2 includes temp buffer)\n", workspaceSize);
    void *dWorkspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&dWorkspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 11. 剪枝：dA -> dAPruned
    ret = aclsparseLtSpMMAPrune(
        &handle, &matmulDesc, dA, dAPruned, ACLSPARSELT_PRUNE_SPMMA_STRIP, stream);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpMMAPrune failed: %d\n", ret); return ret);

    // 12. 执行 Matmul：D = alpha * A_pruned * B + beta * C
    //     splitK=2 时 Cube 将 K 维度切分为 2 段并行，epilogue 归约求和写 D
    ret = aclsparseLtMatmul(
        &handle, &plan,
        &alpha,
        dAPruned, dB,
        &beta, dC, dD,
        dWorkspace, &stream, 1);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("Matmul failed: %d\n", ret); return ret);

    // 13. 同步等待完成（调用方负责同步，算子内部不同步）
    aclrtSynchronizeStream(stream);

    // 14. 将结果拷贝回 Host
    std::vector<float> hD(static_cast<size_t>(m) * n, 0.0f);
    aclrtMemcpy(hD.data(), static_cast<size_t>(m) * n * sizeof(float),
                dD, static_cast<size_t>(m) * n * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
    LOG_PRINT("matmul done: D[0]=%.4f, D[%zu]=%.4f\n", hD[0], hD.size() - 1, hD.back());

    // 15. 清理资源（按依赖逆序销毁：plan -> algSel -> matmulDesc -> mat -> handle）
    aclsparseLtMatmulPlanDestroy(&plan);
    aclsparseLtMatmulAlgSelectionDestroy(&algSel);
    aclsparseLtMatmulDescriptorDestroy(&matmulDesc);
    aclsparseLtMatDescriptorDestroy(&matA);
    aclsparseLtMatDescriptorDestroy(&matB);
    aclsparseLtMatDescriptorDestroy(&matC);
    aclsparseLtMatDescriptorDestroy(&matD);
    aclsparseLtDestroy(&handle);

    aclrtFree(dA);
    aclrtFree(dB);
    aclrtFree(dC);
    aclrtFree(dD);
    aclrtFree(dAPruned);
    if (dWorkspace) { aclrtFree(dWorkspace); }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

int main()
{
    return aclsparseLtAlgSetAttributeTest();
}
```

> splitK=1 用法：将上述示例中 splitK 改为 1，PlanInit 走融合路径（L0C→UB→GM），workspace 不含 temp 区，大小相应减小。

> 属性查询用法：AlgGetAttribute 可在任何时刻调用（PlanInit 前后均可），返回 algSelection 结构体中存储的当前值。`ALG_CONFIG_MAX_ID` 查询始终返回 2（algConfigId 0 和 1 共 2 种配置）。

## 约束与限制

- **属性设置时机**：`aclsparseLtMatmulAlgSetAttribute` 须在 `aclsparseLtMatmulPlanInit` 之前调用。PlanInit 读取 algConfigId 与 splitK 计算 tiling 并冻结到计划中，PlanInit 之后修改属性不会生效，需重新初始化计划。
- **algConfigId 取值**：仅支持 0 或 1。不同配置对应不同的 cube tiling（baseM/baseN/baseK），在不同矩阵形状下性能表现不同。`ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID` 查询返回 2（共 2 种配置）。
- **splitK 取值**：须在 [1, K] 范围内。splitK=1 走融合路径（无 temp 缓冲区）；splitK>1 走两段式路径（Cube 各段写 temp → epilogue 归约写 D），temp 区大小为 splitK×m×n×4 字节。
- **searchIterations**：默认值为 5（由 AlgSelectionInit 设置）。当前仅存储不参与执行，预留用于未来自动搜索功能。
- **splitKMode**：splitK 模式（0=ONE_KERNEL, 1=TWO_KERNELS）。参与 effectiveSplitK 计算：ONE_KERNEL 模式下 effectiveSplitK=1（融合路径，无 temp 缓冲区）；TWO_KERNELS 模式下 effectiveSplitK=splitK（两段式路径，使用 temp 缓冲区）。
- **splitKBuffers**：splitK 缓冲区数，取值范围为 [0, splitK-1]。当前仅存储不参与执行，预留用于对齐 cuSPARSELt 的 API 完整性。
- **ALG_CONFIG_MAX_ID 只读**：该属性不可通过 AlgSetAttribute 设置，尝试设置返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`；仅可通过 AlgGetAttribute 查询，返回 2。
- **attrValueSize 匹配**：所有 int32_t 类型属性的 attrValueSize 须为 sizeof(int32_t)（4 字节），不匹配返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- **依赖关系**：algSelection 须由 `aclsparseLtMatmulAlgSelectionInit` 创建并绑定有效的 matmulDesc。splitK 上界校验（≤ K）依赖 algSelection 绑定的 matmulDesc 中的 K 值。
- **版本依赖**：依赖 CANN asc-devkit >= 9.1.0（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低版本环境下编译与运行将跳过此算子。

## 参考资源

- 对标接口：cuSPARSELt [cusparseLtMatmulAlgSetAttribute](https://docs.nvidia.com/cuda/cusparselt/index.html#cusparseltmatmulalgsetattribute) / [cusparseLtMatmulAlgGetAttribute](https://docs.nvidia.com/cuda/cusparselt/index.html#cusparseltmatmulalggetattribute)。
- 配套算子：[aclsparseLtMatmul](../matmul/README.md)（消费 algSelection 属性执行结构化稀疏矩阵乘）、[aclsparseLtSpMMAPrune](../prune/README.md)（剪枝，产出 Matmul 的稀疏侧输入 A_pruned）。
