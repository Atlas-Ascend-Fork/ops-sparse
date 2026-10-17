# aclsparseLtSpMMAPrune

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

- **算子功能**：aclsparseLtSpMMAPrune 对稠密矩阵执行 2:4 结构化稀疏剪枝，输出与输入同型的稠密存储矩阵（被置零元素以 0 表示）。该算子对标 cuSPARSELt 中的 `cusparseLtSpMMAPrune`，是结构化稀疏矩阵乘法的前置步骤。剪枝对象由 Matmul 描述符中 matA/matB 的 sparsity 字段决定：matA 为 `ACL_SPARSE_LT_SPARSITY_50_PERCENT` 时剪枝矩阵 A（A-sparse，剪枝结果作为 Matmul 的稀疏侧输入 matA）；matB 为 `ACL_SPARSE_LT_SPARSITY_50_PERCENT` 时剪枝矩阵 B（B-sparse，剪枝结果作为 Matmul 的稀疏侧输入 matB）。支持两种剪枝算法：STRIP（逐分组剪枝）与 TILE（逐 tile 剪枝），详见"剪枝算法（pruneAlg）"。
- **目标平台**：<term>Ascend 950PR/Ascend 950DT</term>（arch35 / DAV_3510）。
- **编程模型**：ascendc vector API，多核按行（沿行剪枝）或按行块（沿列剪枝）切分并行；不使用矩阵乘硬件（Cube）。剪枝是确定性操作（按绝对值比较），不涉及随机性。算子异步启动，内部不执行 stream 同步，调用方如需读取结果须自行同步。

### 数学原理

#### 问题定义

对 m×k 稠密矩阵 A，按结构化稀疏模式剪枝：

```
A_pruned = Prune_2:4(A)
```

| 符号 | 含义 | 维度/类型 |
|------|------|-----------|
| A | 稠密输入矩阵（待剪枝） | m×k，FP32/FP16/BF16/INT8 |
| A_pruned | 剪枝后的矩阵（稠密存储，置零元素以 0 表示） | m×k，与 A 同型 |
| Prune_2:4(·) | 2:4 结构化剪枝算子 | — |

#### 剪枝规则

剪枝按固定大小的分组进行，每组保留绝对值最大的若干元素，其余置零：

| 数据类型 | 分组大小 | 每组保留元素数 | 稀疏度 |
|----------|---------|---------------|--------|
| FP16（ACL_FLOAT16） | 4 | 2（保留绝对值最大的 2 个） | 50%（2:4） |
| BF16（ACL_BF16） | 4 | 2（保留绝对值最大的 2 个） | 50%（2:4） |
| INT8（ACL_INT8） | 4 | 2（保留绝对值最大的 2 个） | 50%（2:4） |
| FP32（ACL_FLOAT） | 2 | 1（保留绝对值最大的 1 个） | 50%（1:2，等价 2:4 密度） |

> 说明：FP32 统一采用 1:2 模式（分组为 2、保留 1）以简化实现，与 NVIDIA cuSPARSELt 规范的 1:2 一致；FP16/BF16/INT8 采用标准 2:4 模式（分组为 4、保留 2）。所有数据类型稀疏度均为 50%。

#### 剪枝方向

剪枝方向由矩阵 A 描述符的 order 与 Matmul 描述符的 opA 共同决定：

| opA | order | 剪枝方向 | 含义 |
|-----|-------|---------|------|
| NON_TRANSPOSE | ROW | 沿行方向 | 每组在行的 K 维度上取 top-N |
| NON_TRANSPOSE | COL | 沿列方向 | 每组在列的 M 维度上取 top-N |
| TRANSPOSE | ROW | 沿列方向 | 同上（opA 与 order 共同决定） |
| TRANSPOSE | COL | 沿行方向 | 同上 |

> **order 语义说明**：当前实现中 order 仅影响剪枝方向（沿行或沿列），不改变内存访问模式。kernel 始终以行主序访问内存。矩阵须以行主序存储。

#### 剪枝算法（pruneAlg）

pruneAlg 参数选择剪枝算法，两种算法均保证 50% 稀疏度，但剪枝粒度与最优性不同：

| 算法 | 说明 |
|------|------|
| `ACLSPARSELT_PRUNE_SPMMA_STRIP` | 逐分组剪枝。FP16/BF16/INT8 按 4 元素分组保留绝对值最大的 2 个，FP32 按 2 元素分组保留绝对值最大的 1 个。每组独立选择，不跨组优化。 |
| `ACLSPARSELT_PRUNE_SPMMA_TILE` | 逐 tile 剪枝，对标 cuSPARSELt `cusparseLtSpMMAPrune` 的 TILE 规格。以 tile 为单位枚举所有满足 2:4 约束的有效配置，选 L1-norm（保留元素绝对值之和）最大的配置，其余位置置零。相比 STRIP 可在 tile 范围内保留更大总绝对值。 |

**TILE 算法细节**：

- **FP16/BF16/INT8（4×4 tile）**：将矩阵划分为 4×4 的 tile，在每个 tile 中枚举全部 90 种有效配置（每行恰好 2 个非零、每列恰好 2 个非零），选 L1-norm 最大的配置，其余位置置零。平局取配置序号较小者（确定性）。90 种配置由枚举 C(4,2)^4 = 1296 种行选择并筛选列和为 2 的配置生成。
- **FP32（2×2 tile）**：将矩阵划分为 2×2 的 tile，枚举 2 种有效配置（对角线和反对角线，每行每列恰好 1 个非零），选 L1-norm 最大的配置。
- **Edge tile 回退**：当矩阵行列尾部不构成完整 tile 时，尾部元素回退到 STRIP 算法逐分组剪枝。

> TILE 与 STRIP 的剪枝方向逻辑一致：均由 opA 与 order 共同决定沿行或沿列剪枝。TILE 的三个子路径（沿行 / 沿列 / transA 行序）与 STRIP 一一对应。

## 接口说明

aclsparseLtSpMMAPrune 直接接收 Matmul 描述符，从中读取矩阵 A 的维度、数据类型、order 与 opA 派生剪枝参数。调用前须完成库句柄初始化、矩阵描述符与 Matmul 描述符初始化。

### aclsparseLtSpMMAPrune

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtSpMMAPrune(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    void* d_out,
    aclsparseLtPruneAlg_t pruneAlg,
    aclrtStream stream);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseLtConstHandle_t | aclsparseLt 库句柄的 const 指针，Host 内存 |
| matmulDescr | 输入 | aclsparseLtConstMatmulDescriptor_t* | Matmul 操作描述符，算子从中读取矩阵 A 的维度（m、k）、数据类型、order 与 opA，Host 内存 |
| d_in | 输入 | const void* | 待剪枝的稠密矩阵 A 的 Device 内存指针，须 16 字节对齐，不可为 nullptr。Device 内存 |
| d_out | 输出 | void* | 剪枝结果 A_pruned 的 Device 内存指针，须 16 字节对齐，不可为 nullptr。支持 in-place（d_in == d_out），Device 内存 |
| pruneAlg | 输入 | aclsparseLtPruneAlg_t | 剪枝算法，支持 `ACLSPARSELT_PRUNE_SPMMA_STRIP` 与 `ACLSPARSELT_PRUNE_SPMMA_TILE`，详见"剪枝算法（pruneAlg）"，Host 内存 |
| stream | 输入 | aclrtStream | ACL 流，算子在此流上异步执行，可为 nullptr（表示使用默认流），Host 内存 |

#### 约束说明

- handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`。
- matmulDescr 不可为 nullptr，且其绑定的矩阵 A 描述符（matA）不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- stream 可为 nullptr（表示使用默认流）。
- pruneAlg 支持 `ACLSPARSELT_PRUNE_SPMMA_STRIP` 与 `ACLSPARSELT_PRUNE_SPMMA_TILE`，其他值返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- 矩阵 A 的数据类型支持 `ACL_FLOAT`（FP32）/ `ACL_FLOAT16`（FP16）/ `ACL_BF16`（BF16）/ `ACL_INT8`（INT8），其他类型返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- d_in 与 d_out 须 16 字节对齐，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- d_in 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- d_out 不可为 nullptr（无 workspace 回退路径），否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- 支持 in-place 操作：d_in 与 d_out 可指向同一 Device 内存。**例外**：转置路径（opA=TRANSPOSE 且 order=ROW，即沿列剪枝的转置场景）不支持 in-place，因为输出布局 (m,k) 与输入布局 (k,m) 不同，多核并行写入会覆盖后续 block 的读取区域；此时若 d_in == d_out 将返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **K 维度对齐约束**：K 须满足结构化描述符初始化（`aclsparseLtStructuredDescriptorInit`）的对齐要求——FP32 为 8 的倍数、FP16/BF16 为 16 的倍数、INT8 为 32 的倍数。描述符初始化会强制校验并向上对齐 rows/cols，因此经 API 调用时 K 必然为对应分组大小的倍数。
- **UB 容量约束**：沿行剪枝时，STRIP 需单行数据（k × elemSize）不超过单核 UB 容量，TILE 需 TS 行数据（TS × k × elemSize，TS=4 for FP16 / 2 for FP32，每行按 32 字节对齐）不超过单核 UB 容量；沿列剪枝时行块数据（k × elemSize × 16，每行按 32 字节对齐）须不超过单核 UB 容量。超限时返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **异步执行**：算子异步启动，内部不执行 `aclrtSynchronizeStream`；调用方如需读取 d_out 结果，须自行同步 stream。
- **内存布局**：矩阵 A 须以行主序存储（详见"剪枝方向"中的 order 语义说明）。

### 返回码

| 返回码 | 说明 |
|--------|------|
| `ACL_SPARSE_STATUS_SUCCESS` | 操作成功 |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为 nullptr |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | matmulDescr/matA/d_in/d_out 为 nullptr、数据指针未 16 字节对齐 |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | pruneAlg 非 STRIP/TILE、数据类型非 FP32/FP16/BF16/INT8、K 维度超过 UB 容量、转置路径 in-place（d_in == d_out） |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

以下示例演示对 m×k 的 FP32 矩阵 A 执行 2:4 剪枝。剪枝需要先创建 Matmul 描述符（绑定 A/B/C/D），即使仅执行剪枝而不执行 Matmul，B/C/D 描述符仍须创建（维度须与 A 一致以保证 tiling 元数据合理）。

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

int aclsparseLtPruneTest()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclsparseLtHandle_t handle = nullptr;
    aclsparseLtMatDescriptor_t matA = nullptr, matB = nullptr, matC = nullptr, matD = nullptr;
    aclsparseLtMatmulDescriptor_t matmulDesc = nullptr;
    void *dA = nullptr;
    void *dAPruned = nullptr;
    int ret = 0;

    // 声明放在可能跳转到 cleanup 的检查之前，避免跨越 C++ 对象初始化。
    int64_t m = 16, k = 32;
    std::vector<float> hA(static_cast<size_t>(m) * k);
    int64_t n = 128;
    std::vector<float> hAPruned(static_cast<size_t>(m) * k, 0.0f);
    int zeroCnt = 0;

    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, LOG_PRINT("aclInit failed\n"); return -1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed\n"); aclFinalize(); return -1);
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed\n"); aclrtResetDevice(deviceId); aclFinalize(); return -1);

    // 1. 初始化 aclsparseLt 库句柄
    {
        auto s = aclsparseLtInit(&handle);
        CHECK_RET(s == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseLtInit failed: %d\n", s); ret = -1);
    }

    // 2. 准备 Host 端矩阵 A (m x k, FP32, 行主序)
    //    k 须为 8 的倍数（FP32 结构化描述符对齐要求，剪枝分组为 2）
    for (size_t i = 0; i < hA.size(); i++) { hA[i] = static_cast<float>(i % 7) - 3.0f; }

    // 3. 拷贝数据到 Device
    dA = AllocAndCopyDevice(hA.data(), static_cast<size_t>(m) * k * sizeof(float));
    aclrtMalloc(&dAPruned, static_cast<size_t>(m) * k * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    // 4. 创建矩阵描述符
    //    矩阵 A：结构化稀疏描述符（2:4），数据在 SpMMAPrune 时通过 d_in 显式传入
    {
        auto s = aclsparseLtStructuredDescriptorInit(
            &handle, &matA, m, k, k, 16, ACL_FLOAT,
            ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
        CHECK_RET(s == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("StructuredDescriptorInit A failed: %d\n", s); ret = -1);
    }

    //    B/C/D：占位稠密描述符（剪枝不消费其数据，但 Matmul 描述符初始化需要绑定）
    //    维度须与 A 一致：B(k x n)、C/D(m x n)
    aclsparseLtDenseDescriptorInit(&handle, &matB, k, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matC, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matD, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);

    // 5. 创建 Matmul 描述符（opA=N，决定沿行方向剪枝）
    {
        auto s = aclsparseLtMatmulDescriptorInit(
            &handle, &matmulDesc,
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &matA, &matB, &matC, &matD, ACL_SPARSE_COMPUTE_32F);
        CHECK_RET(s == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("MatmulDescriptorInit failed: %d\n", s); ret = -1);
    }

    // 6. 执行剪枝：dA -> dAPruned
    //    第二参需 aclsparseLtConstMatmulDescriptor_t*（const 指针的指针），
    //    &matmulDesc 是 aclsparseLtMatmulDescriptor_t*，C++ 不允许 T** -> const T** 隐式转换，须 const_cast。
    {
        auto s = aclsparseLtSpMMAPrune(
            &handle, const_cast<aclsparseLtConstMatmulDescriptor_t*>(&matmulDesc),
            dA, dAPruned, ACLSPARSELT_PRUNE_SPMMA_STRIP, stream);
        CHECK_RET(s == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpMMAPrune failed: %d\n", s); ret = -1);
    }

    // 7. 同步等待剪枝完成（调用方负责同步，算子内部不同步）
    aclrtSynchronizeStream(stream);

    // 8. 将结果拷贝回 Host
    aclrtMemcpy(hAPruned.data(), static_cast<size_t>(m) * k * sizeof(float),
                dAPruned, static_cast<size_t>(m) * k * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);

    // 验证：FP32 下每 2 个元素保留 1 个（绝对值较大者），另一个置零
    for (size_t i = 0; i < hAPruned.size(); i++) {
        if (hAPruned[i] == 0.0f) { zeroCnt++; }
    }
    LOG_PRINT("prune done: %d / %zu elements zeroed (~50%% expected)\n", zeroCnt, hAPruned.size());

cleanup:
    // 9. 清理资源（按依赖逆序销毁，异常路径与正常路径统一从此处释放）
    if (matmulDesc != nullptr) { aclsparseLtMatmulDescriptorDestroy(&matmulDesc); }
    if (matA != nullptr) { aclsparseLtMatDescriptorDestroy(&matA); }
    if (matB != nullptr) { aclsparseLtMatDescriptorDestroy(&matB); }
    if (matC != nullptr) { aclsparseLtMatDescriptorDestroy(&matC); }
    if (matD != nullptr) { aclsparseLtMatDescriptorDestroy(&matD); }
    if (handle != nullptr) { aclsparseLtDestroy(&handle); }
    if (dA != nullptr) { aclrtFree(dA); }
    if (dAPruned != nullptr) { aclrtFree(dAPruned); }
    if (stream != nullptr) { aclrtDestroyStream(stream); }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}

int main()
{
    return aclsparseLtPruneTest();
}
```

预期输出如下（具体零元素数量取决于输入数据，约为总元素数的 50%）：

```
prune done: 256 / 512 elements zeroed (~50% expected)
```

> in-place 用法：将上述示例中 `dAPruned` 改为复用 `dA`（即 `aclsparseLtSpMMAPrune(&handle, const_cast<aclsparseLtConstMatmulDescriptor_t*>(&matmulDesc), dA, dA, ...)`），剪枝结果直接覆写输入矩阵 A 的 Device 内存。

## 参考资源

- 对标接口：cuSPARSELt [cusparseLtSpMMAPrune](https://docs.nvidia.com/cuda/cusparselt/index.html#cusparseltspmmaprune)。
