# aclsparseLtSpMMAPruneCheck算子

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- **算子功能**：aclsparseLtSpMMAPruneCheck 校验稠密矩阵是否满足 2:4 结构化稀疏条件：沿剪枝维度逐分组统计非零元素个数，全部分组的非零数不超过保留上限时 `d_valid=0`（满足条件），任一分组超限时 `d_valid=1`（不满足）。该算子对标 cuSPARSELt 中的 `cusparseLtSpMMAPruneCheck`。校验对象由 Matmul 描述符中 matA/matB 的结构化稀疏标记决定（须恰有一侧为 `ACL_SPARSE_LT_SPARSITY_50_PERCENT`），校验条件为组级（基本 2:4）条件，与剪枝算法（TILE/STRIP）无关；经 `aclsparseLtSpMMAPrune` 产出的数据保证通过本检查。支持多 batch：`d_valid` 为全部 batch 判定结果的按位或（任一 batch 存在违规分组 → 1，全部 batch 合法 → 0），单 batch 判定逻辑不变。
- **目标平台**：<term>Ascend 950PR/Ascend 950DT</term>（arch35 / DAV_3510）。
- **编程模型**：Ascend C Vector（RegBase / AIV），多核按行带 × 列块二维 tile 网格切分并行（grid-stride 领取），全程按等宽无符号整数位视图做位级判零（无数据类型转换）。算子异步启动，内部不执行 stream 同步，调用方如需读取 d_valid 须自行同步。

### 数学原理

#### 问题定义

```
d_valid = PruneCheck_2:4(d_in)                    # 单 batch（numBatches=1，默认）
d_valid = OR_b( PruneCheck_2:4(d_in + b×batchStride) )   # 多 batch（numBatches>1）
```

| 符号 | 含义 | 维度/类型 |
|------|------|-----------|
| d_in | 待校验矩阵（稠密存储，行主序） | m×k（A 稀疏）或 k×n（B 稀疏），FP32/FP16/BF16/INT8 |
| d_valid | 校验结果（Device 侧 int 标量） | 0 = 满足条件（correct），1 = 不满足（wrong） |
| numBatches | batch 数量（稀疏侧描述符，默认 1） | ≥ 1 |
| batchStride | batch 间步长（稀疏侧描述符，元素数） | numBatches>1 时须 > 0 且为 32B/元素字节数的整数倍 |

#### 校验规则

沿剪枝维度按固定大小分组，每组校验非零元素个数是否超过保留上限：

| 数据类型 | 分组大小 | 每组保留上限 keep | 稀疏模式 |
|----------|---------|------------------|----------|
| FP16（ACL_FLOAT16） | 4 | 2 | 2:4 |
| BF16（ACL_BF16） | 4 | 2 | 2:4 |
| INT8（ACL_INT8） | 4 | 2 | 2:4 |
| FP32（ACL_FLOAT） | 2 | 1 | 1:2（等价 2:4 密度） |

设沿剪枝维度将矩阵划分为分组集合 {g_1, ..., g_G}：

```
nnz(g_i) = #{ x ∈ g_i : x != 0 }
d_valid  = 0，若 ∀ i: nnz(g_i) ≤ keep
d_valid  = 1，若 ∃ i: nnz(g_i) > keep
```

**批量判定规则**：`numBatches > 1` 时 `d_valid = OR(全部 batch 的判定)`——任一 batch 存在违规分组 → 1，全部 batch 合法 → 0；各 batch 独立判定（batch b 基址 = `d_in + b×batchStride` 元素），单 batch 判定逻辑（分组、判零、方向派生）与 `numBatches=1` 完全一致。

**判零语义**：±0（如 `+0.0`/`-0.0`）均按零处理；NaN 与非规格数（subnormal）按非零处理。浮点数据按位视图（幅值掩码屏蔽符号位后判零）实现，与 `x != 0` 的 IEEE 语义逐位等价。

#### 校验方向

校验方向（分组方向）由稀疏侧矩阵描述符的 order 与 Matmul 描述符的 op 联合决定，与 `aclsparseLtSpMMAPrune` 的剪枝方向推导完全一致：

| op（稀疏侧） | order | 校验方向 | 含义 |
|-----|-------|---------|------|
| NON_TRANSPOSE | ROW | 沿行方向 | 每组在行内 K 维度上取 g 个连续元素 |
| NON_TRANSPOSE | COL | 沿列方向 | 每组在同列 M 维度上取 g 个相邻元素 |
| TRANSPOSE | ROW | 沿列方向 | 同上（op 与 order 共同决定） |
| TRANSPOSE | COL | 沿行方向 | 同上 |

> **order 语义说明**：与 prune 一致，order 仅影响校验方向（沿行或沿列），不改变内存访问模式。d_in 一律以行主序物理存储解释，行步长为描述符 ld（元素数）。

## 接口原型

aclsparseLtSpMMAPruneCheck 直接接收 Matmul 描述符，从中读取稀疏侧的维度、数据类型、order 与 op 派生校验参数。调用前须完成库句柄初始化、矩阵描述符与 Matmul 描述符初始化。

```cpp
aclsparseStatus_t aclsparseLtSpMMAPruneCheck(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    int* d_valid,
    aclrtStream stream);
```

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseLtConstHandle_t | aclsparseLt 库句柄的 const 指针，Host 内存 |
| matmulDescr | 输入 | aclsparseLtConstMatmulDescriptor_t* | Matmul 操作描述符，算子从中读取稀疏侧的维度、数据类型、order 与 op，Host 内存 |
| d_in | 输入 | const void* | 待校验矩阵的 Device 内存指针，行主序存储，须 32 字节对齐，不可为 nullptr，Device 内存 |
| d_valid | 输出 | int* | 校验结果的 Device 内存 int 标量指针（0 满足，1 不满足），不可为 nullptr，Device 内存 |
| stream | 输入 | aclrtStream | ACL 流，算子在此流上异步执行，可为 nullptr（表示使用默认流），Host 内存 |

## 支持数据类型

稀疏侧矩阵（d_in）支持以下数据类型，其余类型返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`：

| 数据类型 | 分组语义（沿剪枝维度） |
|----------|------------------------|
| `ACL_FLOAT`（FP32） | 1:2（组 2 保 1，等价 2:4 密度） |
| `ACL_FLOAT16`（FP16） | 2:4（组 4 保 2） |
| `ACL_BF16`（BF16） | 2:4（组 4 保 2） |
| `ACL_INT8`（INT8） | 2:4（组 4 保 2） |

判零语义与分组方向推导规则详见「功能说明 > 数学原理」。

## 约束说明

- handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`。
- matmulDescr 不可为 nullptr，且其绑定的 matA/matB 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- **唯一稀疏侧约束**：matA 与 matB 须恰有一个为结构化稀疏描述符（经 `aclsparseLtStructuredDescriptorInit` 创建）；两者均非稀疏或均为稀疏时返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **批量约束**：稀疏侧 numBatches 须 ≥ 1（默认 1；经 `aclsparseLtMatDescSetAttribute` 的 `ACLSPARSELT_MAT_NUM_BATCHES` 设置，A/B/C/D 四侧须一致，由 Matmul 描述符初始化校验）。numBatches > 1 时为批量校验（`d_valid` = 全部 batch 判定的 OR），batchStride（`ACLSPARSELT_MAT_BATCH_STRIDE`，元素数）须 > 0 且为 32 字节/元素字节数的整数倍（与 ld 同款对齐：FP32 为 8、FP16/BF16 为 16、INT8 为 32 的倍数），否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；全量跨度 `(numBatches−1)×batchStride + (rows−1)×ld + cols` 换算字节后溢出 int64/uintptr 时同样返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。numBatches = 1 时 batchStride 不参与判定。
- 稀疏侧数据类型支持 `ACL_FLOAT`（FP32）/ `ACL_FLOAT16`（FP16）/ `ACL_BF16`（BF16）/ `ACL_INT8`（INT8），其他类型返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- d_in 须 32 字节对齐（作为 DataCopy 搬运源地址的硬件对齐要求，严于 cuSPARSELt 的 16 字节惯例）且不可为 nullptr，d_valid 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- 扫描维度与 ld 须为 32 字节/元素字节数的整数倍（结构化描述符初始化保证：FP32 为 8、FP16/BF16 为 16、INT8 为 32）；**ld 须不小于物理行长**（沿行校验时为行长维度、沿列校验时为列数）且**须不超过 INT32_MAX（2147483647）**（防止超长 ld 下内部行偏移地址运算溢出，属防御性拦截），否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。ld 大于物理行长时 padding 区数据不参与判定。
- stream 可为 nullptr（表示使用默认流）。
- **异步执行**：算子内部先在同 stream 上将 d_valid 预清零，再启动校验 kernel；内部不执行 `aclrtSynchronizeStream`，调用方如需读取 d_valid 须自行同步 stream。
- **内存布局**：d_in 以行主序物理存储解释（详见"校验方向"中的 order 语义说明）。

## 返回值 / 错误码

| 返回值 | 含义 |
|--------|------|
| `ACL_SPARSE_STATUS_SUCCESS` | 操作成功 |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为 nullptr |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | matmulDescr/matA/matB/d_in/d_valid 为 nullptr、d_in 未 32 字节对齐、维度/ld 非法、批量参数非法（numBatches<1，或 numBatches>1 时 batchStride≤0/非 32B 对齐）、扫描跨度溢出 |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | matA/matB 无唯一稀疏侧、数据类型非 FP32/FP16/BF16/INT8 |
| `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` | 平台核数/UB 资源不足 |
| `ACL_SPARSE_STATUS_EXECUTION_FAILED` | d_valid 预清零或 kernel 启动失败 |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

以下示例演示校验 m×k 的 FP32 矩阵 A 是否满足 1:2 结构化稀疏条件。校验需要先创建 Matmul 描述符（绑定 A/B/C/D），即使仅执行校验而不执行 Matmul，B/C/D 描述符仍须创建（维度须与 A 一致以保证派生元数据合理）。

```cpp
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparseLt.h"

#define CHECK_RET(cond, action) \
    do {                        \
        if (!(cond)) {          \
            action;             \
            goto cleanup;       \
        }                       \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// 辅助：分配 Device 内存并拷贝 Host 数据（分配或拷贝失败返回 nullptr）
static void* AllocAndCopyDevice(const void *hostPtr, size_t sizeBytes)
{
    void *dPtr = nullptr;
    if (aclrtMalloc(&dPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return nullptr;
    }
    if (hostPtr != nullptr && sizeBytes > 0) {
        if (aclrtMemcpy(dPtr, sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            aclrtFree(dPtr);
            return nullptr;
        }
    }
    return dPtr;
}

int aclsparseLtPruneCheckTest()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclsparseLtHandle_t handle = nullptr;
    aclsparseLtMatDescriptor_t matA = nullptr, matB = nullptr, matC = nullptr, matD = nullptr;
    aclsparseLtMatmulDescriptor_t matmulDesc = nullptr;
    void *dA = nullptr;
    int *dValid = nullptr;
    int ret = 0;
    // 矩阵维度与 Host 侧数据（声明须位于首个 goto cleanup 之前：C++ 禁止 goto 跨越带初始化的变量声明）
    int64_t m = 16, k = 32;  // A：m×k FP32；k 须为 8 的倍数（FP32 结构化描述符对齐要求，校验分组为 2）
    int64_t n = 128;         // 占位描述符维度：B(k×n)、C/D(m×n)
    std::vector<float> hA(static_cast<size_t>(m) * k, 0.0f);  // Host 端矩阵 A（m×k，行主序）
    int hValid = -1;         // Host 侧校验结果

    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, LOG_PRINT("aclInit failed\n"); return -1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed\n"); aclFinalize(); return -1);
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed\n"); aclrtResetDevice(deviceId); aclFinalize(); return -1);

    // 1. 初始化 aclsparseLt 库句柄
    {
        auto s = aclsparseLtInit(&handle);
        CHECK_RET(s == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseLtInit failed: %d\n", s); ret = -1);
    }

    // 2. 准备 Host 端矩阵 A (m x k, FP32, 行主序，满足 1:2 条件：每 2 个元素至多 1 个非零)
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < k; j += 2) {
            hA[static_cast<size_t>(i) * k + j] = static_cast<float>((i + j) % 7) - 3.0f; // 偶数位非零，奇数位为零
        }
    }

    // 3. 拷贝数据到 Device，分配 d_valid
    dA = AllocAndCopyDevice(hA.data(), static_cast<size_t>(m) * k * sizeof(float));
    CHECK_RET(dA != nullptr, LOG_PRINT("alloc/copy dA failed\n"); ret = -1);
    CHECK_RET(aclrtMalloc(reinterpret_cast<void**>(&dValid), sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
              LOG_PRINT("malloc dValid failed\n"); ret = -1);

    // 4. 创建矩阵描述符
    //    矩阵 A：结构化稀疏描述符（FP32 下分组为 1:2），数据在 PruneCheck 时通过 d_in 显式传入
    {
        auto s = aclsparseLtStructuredDescriptorInit(
            &handle, &matA, m, k, k, 16, ACL_FLOAT,
            ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
        CHECK_RET(s == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("StructuredDescriptorInit A failed: %d\n", s); ret = -1);
    }

    //    B/C/D：占位稠密描述符（校验不消费其数据，但 Matmul 描述符初始化需要绑定）
    //    维度须与 A 一致：B(k x n)、C/D(m x n)
    aclsparseLtDenseDescriptorInit(&handle, &matB, k, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matC, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matD, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);

    // 5. 创建 Matmul 描述符（opA=N，决定沿行方向校验）
    {
        auto s = aclsparseLtMatmulDescriptorInit(
            &handle, &matmulDesc,
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &matA, &matB, &matC, &matD, ACL_SPARSE_COMPUTE_32F);
        CHECK_RET(s == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("MatmulDescriptorInit failed: %d\n", s); ret = -1);
    }

    // 6. 执行校验：dA -> d_valid
    //    第二参为 const 描述符指针的指针（aclsparseLtConstMatmulDescriptor_t*）：
    //    先声明 const 版本描述符再传地址即可自然匹配类型。
    {
        aclsparseLtConstMatmulDescriptor_t constMatmulDesc = matmulDesc;
        auto s = aclsparseLtSpMMAPruneCheck(&handle, &constMatmulDesc, dA, dValid, stream);
        CHECK_RET(s == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpMMAPruneCheck failed: %d\n", s); ret = -1);
    }

    // 7. 同步等待校验完成（调用方负责同步，算子内部不同步）
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed\n"); ret = -1);

    // 8. 将结果拷贝回 Host
    CHECK_RET(aclrtMemcpy(&hValid, sizeof(int), dValid, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
              LOG_PRINT("copy dValid to host failed\n"); ret = -1);
    LOG_PRINT("prune check done: d_valid=%d (0=correct, 1=wrong)\n", hValid);

cleanup:
    // 9. 清理资源（按依赖逆序销毁，异常路径与正常路径统一从此处释放）
    if (matmulDesc != nullptr) { aclsparseLtMatmulDescriptorDestroy(&matmulDesc); }
    if (matA != nullptr) { aclsparseLtMatDescriptorDestroy(&matA); }
    if (matB != nullptr) { aclsparseLtMatDescriptorDestroy(&matB); }
    if (matC != nullptr) { aclsparseLtMatDescriptorDestroy(&matC); }
    if (matD != nullptr) { aclsparseLtMatDescriptorDestroy(&matD); }
    if (handle != nullptr) { aclsparseLtDestroy(&handle); }
    if (dA != nullptr) { aclrtFree(dA); }
    if (dValid != nullptr) { aclrtFree(dValid); }
    if (stream != nullptr) { aclrtDestroyStream(stream); }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}

int main()
{
    return aclsparseLtPruneCheckTest();
}
```

预期输出如下（示例数据满足 1:2 条件）：

```
prune check done: d_valid=0 (0=correct, 1=wrong)
```
