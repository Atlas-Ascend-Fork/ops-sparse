# aclsparseLtMatmulSearch

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> Ascend 950PR/Ascend 950DT 上的 aclsparseLtMatmulSearch 依赖 CANN asc-devkit >= 9.1.0（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

## 功能说明

- **算子功能**：aclsparseLtMatmulSearch 是 Matmul 算法自动搜索（auto-tuning）接口。该接口评估 plan 描述的 matmul 的可用算法候选（algConfigId × splitKMode × splitK，候选空间由库内生成规则枚举），每个候选在 streams[0] 上按 `aclsparseLtMatmul` 相同语义计时执行 searchIterations 次（默认 5，经 `aclsparseLtMatmulAlgSetAttribute` + `ACLSPARSELT_MATMUL_SEARCH_ITERATIONS` 设置）并取最小耗时，自动选择最快的配置并回写：plan 绑定的 algSelection 属性（algConfigId / splitKMode / splitK，可经 `aclsparseLtMatmulAlgGetAttribute` 查询）、plan 内 tilingData 与 workspaceSize（按选中配置重算）。面向同一 matmul 重复执行多次的场景：以一次性搜索成本换取后续每次执行的稳定收益。
- **实现形态**：纯 Host 侧控制流（枚举候选 → 改写 algSelection → 重算 tiling/workspace → 调既有 `aclsparseLtMatmul` 计时 → 选优回写 → 以选中配置终态执行一次），无 Search 专属 device kernel（非紧凑 dense 输入的布局预处理复用 aclsparseLtMatmul 既有路径，见"行为声明"）。数值计算 100% 由既有 aclsparseLtMatmul kernel 承载，候选仅改变 tiling 切分字段，不改变数学语义 `D = Activation(alpha * op(A) * op(B) + beta * C + bias)`，matD 数值精度与 aclsparseLtMatmul 完全一致。
- **目标平台**：<term>Ascend 950PR/Ascend 950DT</term>（arch35 / DAV_3510）。
- **编程模型**：Host 侧候选枚举与 aclrt event 计时（Device 侧时间戳），Device 侧计算复用既有 aclsparseLtMatmul kernel。
- **支持数据类型**：与 aclsparseLtMatmul 相同——FP32 / FP16 / BF16（`computeType = ACL_SPARSE_COMPUTE_32F`，FP32 累加）；INT8（`computeType = ACL_SPARSE_COMPUTE_32I`，int32 累加，支持 INT8/INT32 双输出）。搜索本身不引入新的 dtype 限制；C/D dtype 组合随 aclsparseLtMatmul 支持域（INT8 A/B 下 C ∈ {INT8, INT32, FP16, BF16} 超集，D 与 C 同型，C=INT8 时 D 亦可为 INT32）。
- **batch 支持**：与 aclsparseLtMatmul 相同（A/B-sparse 两侧均支持）。A-sparse × batch 场景须先经 `aclsparseLtSpMMAPrune`（其 structured 侧描述符的 `NUM_BATCHES`/`BATCH_STRIDE` 生效，一次调用产出全 batch 的 A_pruned）产出 A_pruned 后显式经 matA 传入——matA=nullptr 的 workspace 回退仅支持 numBatches=1（workspace 的 aPruned 区为单 batch 容量）。

### 作用原理

aclsparseLtMatmulSearch 位于 aclsparseLt 调用链的"执行前调优"位置：在 PlanInit / GetWorkspace / 剪枝完成之后、正式执行 Matmul 之前调用一次，之后以选中配置反复执行 Matmul。推荐时序：

```
Init → 矩阵描述符（可选 batch 属性）→ MatmulDescriptorInit →（可选）描述符属性
    → AlgSelectionInit →（可选）AlgSetAttribute（须在 PlanInit 前）→ PlanInit
    → GetWorkspace → 分配 →（A-sparse 路径）SpMMAPrune
    → ★ MatmulSearch（阻塞，相对 streams[0]）→ 重新 GetWorkspace 校验
    → Matmul（多次复用选中配置，异步）→ 逆序销毁
```

搜索内部流程（单次调用）：

1. **校验**：参数 / 指针 / 对齐 / dtype 一次性全部完成（参数类失败全部拦截在搜索开始前）；**快照**：保存进入时的 algSelection 三元组、tilingData 与 workspaceSize，供失败恢复；
2. **候选枚举**：按"搜索空间与选优规则"生成候选表（≤ 13 项，栈上固定数组，无堆分配）；
3. **逐候选计时**：将候选三元组写入 plan 绑定的 algSelection → 按 PlanInit 同源路径重算 tilingData / workspaceSize → 在 streams[0] 上执行 searchIterations 次 `RecordEvent(start) → aclsparseLtMatmul（数据指针全透传）→ RecordEvent(stop) → SynchronizeEvent → EventElapsedTime`，取最小耗时。候选内任一同步调用或内部 Matmul 返回失败 → 该候选视为不可用，日志留痕（OP_LOGW）后跳过，继续评估其余候选；
4. **回写**：将最优候选无条件幂等回写（algSelection 三元组 + tilingData + workspaceSize 重算，终态等价于用选中配置重新执行一次 `aclsparseLtMatmulPlanInit`）；
5. **终态执行与返回**：以选中配置再执行一次 aclsparseLtMatmul（不计时），matD 终态 = 选中配置的完整输出（非 in-place 为完整覆盖写；in-place 多累积一次，见"行为声明"）；检查终态执行与流同步返回值（兜底计时循环内不可见的异步 kernel 错误）。全部候选失败、终态执行失败或同步检查失败 → 恢复快照，返回 `ACL_SPARSE_STATUS_EXECUTION_FAILED`；成功 → 销毁事件，阻塞返回。

### 搜索空间与选优规则

搜索维度为 algSelection 中**实际参与执行分发**的属性三元组 `(algConfigId, splitKMode, splitK)`；`searchIterations` 是搜索输入（每候选重复测量次数）而非搜索维度，`splitKBuffers` 仅存储不参与执行（不纳入搜索）。候选集构成如下（按执行顺序）：

| 候选来源 | 构成规则 | 说明 |
|----------|----------|------|
| fused 等价类代表 | `(g, ONE_KERNEL, 1)`，g ∈ {0, 1}，恒纳入 | ONE_KERNEL（任意 splitK）与 TWO_KERNELS + splitK=1 在 kernel 层完全等价（effectiveSplitK=1，单融合 kernel 路径），每个 algConfigId 只计时 1 次。workspace 需求为该 shape 的布局最小值，恒通过裁剪 |
| splitK 幂次候选（两段式） | `(g, TWO_KERNELS, s)`，s ∈ {2, 4, 8, ...}（2 的幂，s ≤ K），g ∈ {0, 1} | s=2 恒为两段式代表点；此后按幂次翻倍扩展，一旦该 algConfigId 的 MN tile 数 × (s/2) 已达 Cube 核数（950PR 为 32）即停止（splitK 并行度收益以填满核数为限）。K < 2 时无两段式候选（正常退化） |
| 当前配置保底 | 进入搜索时的配置等价类未被上述候选覆盖时追加（至多 1 项） | 保证候选集包含当前配置 → "搜索不劣化"结构性成立 |
| workspace 方案 A 裁剪 | 仅评估 workspace 需求 ≤ 进入搜索时 workspaceSize 的候选 | 两段式候选的 temp 区为 numBatches × s × m × n × 4 字节，随 s 增大增长，超限候选在枚举期即被排除（不执行） |

- 候选数上界 13（2 + 2×5 + 1），典型 4~12 项。默认预置（splitK=1）时两段式候选通常全被方案 A 裁剪，搜索空间退化为 2 个 fused 候选（algConfigId 二选一），仍保证不劣化。
- **计时统计量**：每候选 searchIterations 次取最小值（min，抗偶发调度噪声）；无预热（min 统计量对首跑冷启动天然鲁棒）。计时包络含 tiling H2D 拷贝（~230 B），与用户后续以选中配置反复调用 aclsparseLtMatmul 的真实单次成本同构。
- **选优**：可用候选（计时全程无失败）中最小耗时候选胜出，回写 plan / algSelection；个别候选执行失败视为不可用跳过（日志留痕，不影响搜索返回），全部候选失败才报错（见"行为声明"）。近并列候选（耗时差在计时噪声内）间选中结果可能抖动，均满足"搜索不劣化"保证（见"约束与限制"）。

## 算子执行接口

### aclsparseLtMatmulSearch

搜索 matmul 最优算法配置并回写 plan。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtMatmulSearch(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulPlan_t* plan,
    const void* alpha,
    const void* matA, const void* matB,
    const void* beta,
    const void* matC, void* matD,
    void* workspace,
    aclrtStream* streams, int32_t numStreams);
```

> plan 为非 const（区别于 aclsparseLtMatmul 的 `aclsparseLtConstMatmulPlan_t*` 只读 plan）：搜索会更新 plan 绑定的 algSelection 属性及 plan 内 tilingData / workspaceSize。

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseLtConstHandle_t | aclsparseLt 库句柄的 const 指针，不可为 nullptr，Host 内存 |
| plan | 输入/输出 | aclsparseLtMatmulPlan_t* | 执行计划（`aclsparseLtMatmulPlanInit` 产出）。搜索更新 plan 绑定的 algSelection 属性（algConfigId / splitKMode / splitK ← 最优值）及 plan 内 tilingData / workspaceSize（按最优配置重算）。Host 内存 |
| alpha | 输入 | const void* | alpha 缩放因子。标量模式为 Host 端 float 指针（nullptr 按 1.0 处理）；向量缩放模式（启用 `ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING`）为 Device 端 float[M] 指针，不可为 nullptr。计时执行按 aclsparseLtMatmul 相同语义透传，Host/Device 内存 |
| matA | 输入 | const void* | 矩阵 A 数据指针（A-sparse 路径为剪枝后的 A_pruned），须 16 字节对齐。可为 nullptr（A-sparse 回退到 workspace 的 aPruned 区，须已由 aclsparseLtSpMMAPrune 填充，仅限 numBatches=1；numBatches>1 时须先显式调用 aclsparseLtSpMMAPrune 产出 A_pruned 再传入；dense×dense 与 B-sparse 路径须显式传入），Device 内存 |
| matB | 输入 | const void* | 矩阵 B 数据指针，须 16 字节对齐，不可为 nullptr，Device 内存 |
| beta | 输入 | const void* | beta 缩放因子。标量模式为 Host 端 float 指针（nullptr 按 0.0 处理）；向量缩放模式（启用 `ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING`）为 Device 端 float[M] 指针，不可为 nullptr；启用 `ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING` 且未启用 `BETA_VECTOR_SCALING` 时可为 nullptr（按 0.0 处理）。Host/Device 内存 |
| matC | 输入 | const void* | 矩阵 C 数据指针（累加项），须 16 字节对齐，不可为 nullptr。按 matC 描述符 (order, ld) 布局访问（ROW=行主序 / COL=列主序，ld 支持 padding，pad 区域不读；语义同 aclsparseLtMatmul）。支持 in-place（matC == matD，此时 matD 值在搜索中累积，见"行为声明"），Device 内存 |
| matD | 输出 | void* | 矩阵 D 数据指针（输出），须 16 字节对齐，不可为 nullptr。按 matD 描述符 (order, ld) 布局写出（ld 支持 padding，pad 区域不写）。搜索过程被反复写入；搜索结束后以选中配置无条件再执行一次，返回时内容为选中配置的完整输出，Device 内存 |
| workspace | 输入 | void* | workspace 指针（进入搜索前按当前配置 `aclsparseLtMatmulGetWorkspace` 查询并分配），须 16 字节对齐，不可为 nullptr。计时执行复用该 workspace；纳入评估的候选 workspace 需求均不超过进入搜索时的 workspaceSize，无越界风险，Device 内存 |
| streams | 输入 | aclrtStream* | ACL 流指针数组，不可为 nullptr；streams[0] 可为 nullptr（表示默认流）。当前仅使用 streams[0]（多流不支持，沿用 aclsparseLtMatmul 既有差异），Host 内存 |
| numStreams | 输入 | int32_t | 流数量，须 > 0；当前仅支持 1（> 1 时不报错，仅使用 streams[0]），Host 内存 |

#### 行为声明

- **阻塞语义**：本接口相对 streams[0] 为**阻塞调用**（返回前完成流同步），区别于 aclsparseLtMatmul 的异步语义。搜索返回后可直接回读 matD，无需额外同步。
- **matD 内容**：搜索过程被反复写入；搜索结束后以选中配置**无条件再执行一次**（不计时），返回时内容为选中配置的**完整输出**（非 in-place 时为完整覆盖写，可直接与 golden 比对）；in-place 时多累积一次，见下条。
- **in-place（matC == matD）**：matD 值随执行次数累积（含选中配置终态执行的一次），搜索返回后内容不可预测。需要确定结果时，建议改用非 in-place 布局或重置 matD 后再执行。
- **非紧凑布局直通**：dense 侧非紧凑布局（ld 大于物理维度）时，候选直接以调用方原始指针与布局执行（cube GM 搬运按 ld 行距直通跳过行尾 padding，无入口拷贝或预处理步骤），候选计时与终态执行均按该语义进行（布局访问与描述符属性活读同 `aclsparseLtMatmul`）。
- **候选失败处理**：个别候选执行失败 → 视为不可用候选，**跳过并日志留痕**，不影响搜索返回，搜索在可用候选中选优；**全部候选执行失败**、选中配置终态执行失败或最终流同步检查失败 → 返回 `ACL_SPARSE_STATUS_EXECUTION_FAILED`，plan 与 algSelection 恢复进入搜索前的状态（失败调用无 plan/algSelection 副作用，可继续用原配置执行 Matmul）；matD 已被写入的内容不回滚。
- **搜索后 workspace**：plan->workspaceSize 可能变小（选中配置的 temp 区需求更小），已分配 workspace 必然够用；建议重新调用 `aclsparseLtMatmulGetWorkspace` 校验（见"约束与限制"）。
- **输入格式要求（稀疏侧）**：matA/matB 的稀疏侧须为 `aclsparseLtSpMMAPrune` 输出的稠密置零格式（与 aclsparseLtMatmul 相同）；`aclsparseLtSpMMACompress` 的压缩格式输出不可直接传入，否则将产生静默错误结果。

#### 返回码

| 返回码 | 说明 |
|--------|------|
| `ACL_SPARSE_STATUS_SUCCESS` | 搜索完成，最优配置已写入 plan/algSelection |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为 nullptr 或 *handle 为 nullptr |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | plan/streams 为 nullptr、numStreams <= 0、matB/matC/matD 为 nullptr、数据指针未 16 字节对齐、matA 为 nullptr 且不满足 A-sparse 回退条件、searchIterations <= 0、启用向量缩放时 alpha/beta 为 nullptr（以上均为搜索入口一次性校验；缩放向量的 32 字节对齐与 device 内存判定经候选执行链沿用 aclsparseLtMatmul 执行期校验） |
| `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` | workspace 为 nullptr |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | dtype 不支持（仅支持 FP32/FP16/BF16/INT8） |
| `ACL_SPARSE_STATUS_EXECUTION_FAILED` | 全部候选执行失败、选中配置的最终回写执行失败，或 aclrt 计时/同步调用失败（失败时 plan/algSelection 恢复进入搜索前状态） |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

以下示例演示完整的"剪枝 → 算法搜索 → 选中配置执行"流程。计算 `D = 1.0 * A_pruned * B + 0.0 * C`，其中 A 为 m×k（FP32，行主序，2:4 结构化稀疏），B 为 k×n，C/D 为 m×n。示例预置 splitK=8 + splitKMode=TWO_KERNELS 以打开两段式候选的 workspace 窗口（见文末用法说明）。

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

// 辅助：分配 Device 内存并拷贝 Host 数据（失败返回 nullptr）
static void* AllocAndCopyDevice(const void *hostPtr, size_t sizeBytes)
{
    void *dPtr = nullptr;
    if (aclrtMalloc(&dPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed (%zu bytes)\n", sizeBytes);
        return nullptr;
    }
    if (hostPtr != nullptr && sizeBytes > 0) {
        if (aclrtMemcpy(dPtr, sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            LOG_PRINT("aclrtMemcpy H2D failed (%zu bytes)\n", sizeBytes);
            aclrtFree(dPtr);
            return nullptr;
        }
    }
    return dPtr;
}

int aclsparseLtMatmulSearchTest()
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
    //    k 须为 8 的倍数（FP32 结构化描述符对齐要求）
    int64_t m = 128, k = 4096, n = 128;
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
    CHECK_RET(dA != nullptr && dB != nullptr && dC != nullptr,
              LOG_PRINT("alloc/copy device input buffer failed\n"); return -1);
    void *dD = nullptr;
    CHECK_RET(aclrtMalloc(&dD, static_cast<size_t>(m) * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc dD failed\n"); return -1);
    void *dAPruned = nullptr;
    CHECK_RET(aclrtMalloc(&dAPruned, static_cast<size_t>(m) * k * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc dAPruned failed\n"); return -1);

    // 4. 创建矩阵描述符（A 为 structured，B/C/D 为 dense，均 FP32）
    aclsparseLtMatDescriptor_t matA = nullptr;
    aclsparseLtMatDescriptor_t matB = nullptr, matC = nullptr, matD = nullptr;
    ret = aclsparseLtStructuredDescriptorInit(
        &handle, &matA, m, k, k, 16, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("StructuredDescriptorInit failed: %d\n", ret); return ret);
    ret = aclsparseLtDenseDescriptorInit(&handle, &matB, k, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("DenseDescriptorInit(matB) failed: %d\n", ret); return ret);
    ret = aclsparseLtDenseDescriptorInit(&handle, &matC, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("DenseDescriptorInit(matC) failed: %d\n", ret); return ret);
    ret = aclsparseLtDenseDescriptorInit(&handle, &matD, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("DenseDescriptorInit(matD) failed: %d\n", ret); return ret);

    // 5. 创建 Matmul 描述符 (opA=N, opB=N, FP32 累加)
    aclsparseLtMatmulDescriptor_t matmulDesc = nullptr;
    ret = aclsparseLtMatmulDescriptorInit(
        &handle, &matmulDesc,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &matA, &matB, &matC, &matD, ACL_SPARSE_COMPUTE_32F);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("MatmulDescriptorInit failed: %d\n", ret); return ret);

    // 6. 创建算法选择描述符并预置属性（须在 PlanInit 之前）。
    //    预置 splitK=8 + splitKMode=TWO_KERNELS 以打开两段式候选的 workspace
    //    评估窗口（机制见文末"默认预置 / 想搜索更大 splitK"用法说明；
    //    searchIterations 默认 5，如需调整经 ACLSPARSELT_MATMUL_SEARCH_ITERATIONS
    //    设置，须 > 0）
    aclsparseLtMatmulAlgSelection_t algSel = nullptr;
    ret = aclsparseLtMatmulAlgSelectionInit(&handle, &algSel, &matmulDesc, ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("AlgSelectionInit failed: %d\n", ret); return ret);
    int32_t splitK = 8;
    ret = aclsparseLtMatmulAlgSetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_SPLIT_K,
                                           &splitK, sizeof(int32_t));
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("AlgSetAttribute(SPLIT_K) failed: %d\n", ret); return ret);
    int32_t splitKMode = static_cast<int32_t>(ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS);
    ret = aclsparseLtMatmulAlgSetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_SPLIT_K_MODE,
                                           &splitKMode, sizeof(int32_t));
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("AlgSetAttribute(SPLIT_K_MODE) failed: %d\n", ret); return ret);

    // 7. 创建执行计划（PlanInit 冻结 tiling 与 workspace 元数据）
    aclsparseLtMatmulPlan_t plan = nullptr;
    ret = aclsparseLtMatmulPlanInit(&handle, &plan, &matmulDesc, &algSel);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("PlanInit failed: %d\n", ret); return ret);

    // 8. 查询 workspace 大小并分配
    //    注意：Const 版本描述符/计划参数（aclsparseLtConstXxx_t*，即 const struct Xxx**）
    //    与局部变量取地址（struct Xxx**）之间不可隐式转换，需 const_cast；
    //    aclsparseLtMatmulSearch 的 plan 参数为非 const，直接传 &plan
    size_t workspaceSize = 0;
    ret = aclsparseLtMatmulGetWorkspace(&handle, const_cast<aclsparseLtConstMatmulPlan_t*>(&plan), &workspaceSize);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("GetWorkspaceSize failed: %d\n", ret); return ret);
    LOG_PRINT("workspaceSize (before search) = %zu bytes\n", workspaceSize);
    void *dWorkspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&dWorkspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
                  LOG_PRINT("aclrtMalloc workspace failed\n"); return -1);
    }

    // 9. 剪枝：dA -> dAPruned
    ret = aclsparseLtSpMMAPrune(
        &handle, const_cast<aclsparseLtConstMatmulDescriptor_t*>(&matmulDesc),
        dA, dAPruned, ACLSPARSELT_PRUNE_SPMMA_STRIP, stream);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpMMAPrune failed: %d\n", ret); return ret);

    // 10. ★ 算法搜索：枚举候选并逐候选计时取 min，最优配置回写 plan/algSelection，
    //     并以选中配置终态执行一次（matD 终态 = 选中配置的完整输出）。
    //     阻塞调用：返回前完成 streams[0] 同步
    ret = aclsparseLtMatmulSearch(
        &handle, &plan,
        &alpha,
        dAPruned, dB,
        &beta, dC, dD,
        dWorkspace, &stream, 1);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("MatmulSearch failed: %d\n", ret); return ret);

    // 11. 查询选中配置（经 AlgGetAttribute；SPLIT_K / SPLIT_K_MODE 同理）
    int32_t selAlgConfigId = -1;
    ret = aclsparseLtMatmulAlgGetAttribute(&handle, const_cast<aclsparseLtConstMatmulAlgSelection_t*>(&algSel),
                                           ACLSPARSELT_MATMUL_ALG_CONFIG_ID,
                                           &selAlgConfigId, sizeof(int32_t));
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("AlgGetAttribute(ALG_CONFIG_ID) failed: %d\n", ret); return ret);
    LOG_PRINT("selected algConfigId=%d\n", selAlgConfigId);

    // 12. 搜索后重新校验 workspaceSize（选中配置需求可能更小，方案 A 下必 <= 已分配值，
    //     原 workspace 可直接复用，无需重分配）
    size_t workspaceSizePost = 0;
    ret = aclsparseLtMatmulGetWorkspace(&handle, const_cast<aclsparseLtConstMatmulPlan_t*>(&plan), &workspaceSizePost);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("GetWorkspaceSize(post) failed: %d\n", ret); return ret);
    LOG_PRINT("workspaceSize (after search) = %zu bytes\n", workspaceSizePost);

    // 13. 以选中配置执行 Matmul（plan 终态 == 用选中配置重新 PlanInit，可直接复用；
    //     异步执行，读取结果前须同步）
    ret = aclsparseLtMatmul(
        &handle, const_cast<aclsparseLtConstMatmulPlan_t*>(&plan),
        &alpha,
        dAPruned, dB,
        &beta, dC, dD,
        dWorkspace, &stream, 1);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("Matmul failed: %d\n", ret); return ret);

    // 14. 同步等待完成（Matmul 为异步接口，调用方负责同步）
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed\n"); return -1);

    // 15. 将结果拷贝回 Host
    std::vector<float> hD(static_cast<size_t>(m) * n, 0.0f);
    CHECK_RET(aclrtMemcpy(hD.data(), static_cast<size_t>(m) * n * sizeof(float),
                          dD, static_cast<size_t>(m) * n * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy D2H failed\n"); return -1);
    LOG_PRINT("matmul done with searched config: D[0]=%.4f, D[%zu]=%.4f\n",
              hD[0], hD.size() - 1, hD.back());

    // 16. 清理资源（按依赖逆序销毁：plan -> algSel -> matmulDesc -> mat -> handle）
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
    return aclsparseLtMatmulSearchTest();
}
```

> **默认预置（不设置 splitK）用法**：AlgSelectionInit 默认 splitK=1 / splitKMode=ONE_KERNEL（融合路径），此时 workspace 无 temp 区，两段式候选全被方案 A 裁剪，搜索空间退化为 2 个 fused 候选（algConfigId 0/1 二选一）。当前配置保底保证"搜索不劣化"仍然成立，适合小 shape 或不希望额外分配大 workspace 的场景。

> **想搜索更大 splitK 的用法**：先 `aclsparseLtMatmulAlgSetAttribute` 设大 splitK 并将 splitKMode 设为 `ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS` → `aclsparseLtMatmulPlanInit` → 按新 workspaceSize 分配 → 再调用本接口。仅设大 splitK 而保持默认 ONE_KERNEL 时 splitK 被吸收（effectiveSplitK=1，workspace 不含 temp 区），无法打开两段式候选窗口。方案 A 仅评估 workspace 需求不超过进入时 workspaceSize 的候选，更大的预置 splitK 意味着更大的候选评估窗口。

> **in-place 用法**：matC 与 matD 传同一 Device 指针时功能可用（返回 SUCCESS、回写合法最优配置），但搜索过程 matD 值随执行次数累积，返回后内容不可预测；需要确定输出时请使用非 in-place 布局。

> **searchIterations 调整用法**：经 `aclsparseLtMatmulAlgSetAttribute(ACLSPARSELT_MATMUL_SEARCH_ITERATIONS, &n, sizeof(int32_t))` 设置（默认 5，须 > 0）。本属性为搜索入口活读属性：PlanInit 之后设置对随后调用的 MatmulSearch 仍生效（区别于三元组的 PlanInit 快照语义）。

## 约束与限制

- **属性设置时机**：三元组属性（algConfigId / splitK / splitKMode）经 `aclsparseLtMatmulAlgSetAttribute` 须在 `aclsparseLtMatmulPlanInit` 之前设置（PlanInit 读取三元组计算 tiling，之后修改需重新初始化计划）；`ACLSPARSELT_MATMUL_SEARCH_ITERATIONS` 为搜索入口活读属性，无 PlanInit 冻结门禁，PlanInit 之后设置对随后的 Search 仍生效。推荐调用时序：AlgSelectionInit → AlgSetAttribute → PlanInit → GetWorkspace →（可选 SpMMAPrune）→ **MatmulSearch（阻塞）** → 重新 GetWorkspace 校验 → Matmul（多次复用选中配置）。
- **搜索后 workspace 校验**：搜索可能选中 workspace 需求更小的配置，plan->workspaceSize 随之变小，已分配 workspace 必然够用；建议搜索后重新调用 `aclsparseLtMatmulGetWorkspace` 校验实际值。方案 A 保证选中配置需求 ≤ 进入搜索时的 workspaceSize，不会出现要求更大分配的情况；被裁剪后搜索空间缩小属预期行为（极端情况下退化为 fused 二选一），非缺陷。
- **searchIterations 建议**：默认 5，须 > 0，无上限。搜索为阻塞调用，总时延 ≈ 候选数 × searchIterations × 单次 matmul 耗时 + 计时同步开销，建议保持默认值；设置极大值会显著拉长阻塞时间，仅建议在计时噪声异常大的环境下适度增大。
- **近并列候选选优抖动**：当多个候选的耗时差在计时噪声范围内（近并列）时，不同次搜索可能选中不同配置，属设计预期内（min 统计量 + 不劣化保证的设计覆盖）。实测（5 轮独立进程）轮间选中配置存在抖动，全部轮次选中配置耗时均满足 ≤ 1.10 × 默认配置耗时的不劣化判据。
- **数据精度**：搜索不改变数值计算路径（候选仅改变 tiling 切分字段，数学语义不变），matD 数值精度与 aclsparseLtMatmul 完全一致。INT8 路径精度口径与 aclsparseLtMatmul 相同，两级分派：α==1 且 β==0 且无 bias/activation 时 EXACT（二进制一致），其余走混合容差 atol=1 / rtol=0 / max_abs_error_limit=65。
- **指针约束**：workspace / matA（非空时）/ matB / matC / matD 须 16 字节对齐；matB / matC / matD / workspace / streams 不可为 nullptr；matA 为 nullptr 仅在 A-sparse 路径合法（回退 workspace aPruned 区，须已由 aclsparseLtSpMMAPrune 填充），B-sparse 路径须显式传入。
- **流约束**：仅使用 streams[0]；numStreams > 1 时不报错，仅使用 streams[0]（多流不支持，沿用 aclsparseLtMatmul 既有差异）；streams[0] 可为 nullptr（ACL runtime 默认流）。
- **阻塞性质与线程模型**：相对 streams[0] 为阻塞调用，搜索期间独占该流与调用线程；单线程 C API 库，本接口不引入线程安全保证。
- **失败语义**：分层处理——个别候选执行失败跳过并日志留痕（在可用候选中选优）；全部候选失败或终态检查失败时返回 `ACL_SPARSE_STATUS_EXECUTION_FAILED`，plan 与 algSelection 恢复进入搜索前状态，可继续用原配置执行 Matmul；matD 已写入内容不回滚。

## 精度与性能实测参考

> 数据为实测结果，环境 Ascend 950PR_9589（单卡）/ CANN 9.2.0。精度：ST 131 用例（88 CSV + 43 条异常 TEST_F）。性能：5 轮独立进程稳态中位数，判定口径 T(c\*) median ≤ 1.10 × T(default) median（ε=10%）。

**搜索收益**（T(default) 为默认配置 algConfigId=0 / splitK=1 / ONE_KERNEL 的耗时，T(c\*) 为搜索选中配置的耗时）：

| 用例 | shape (m×k×n) | dtype | T(default) med (ms) | T(c\*) med (ms) | T(c\*)/T(default) | speedup | 选中配置 |
|------|------|-------|------------|------------|------------|------|------|
| T1（K 重） | 128×4096×128 | FP16 | 0.0710 | 0.0533 | 0.7515 | 1.340× | (algConfigId=1, splitK=8, TWO_KERNELS) |
| T2（默认预置） | 128×128×128 | FP32 | 0.0414 | 0.0412 | 0.9936 | 1.006× | (algConfigId=1, splitK=1, ONE_KERNEL) |
| T3（K 重） | 128×2048×128 | INT8→INT32 | 0.0448 | 0.0442 | 0.9866 | 1.006× | (algConfigId=0, splitK=1, ONE_KERNEL) |
| T4（中型默认预置） | 256×256×256 | BF16 | 0.0577 | 0.0514 | 0.8912 | 1.132× | (algConfigId=1, splitK=1, ONE_KERNEL) |

- 4/4 用例达标（无退化）；K 重 shape（T1）经 splitK 两段式搜索获得 1.34× 实质收益，中型 shape（T4）经 algConfigId tiling 切换获得 1.13× 收益；T2/T4 为 fused-only 候选空间，speedup ≈ 1.0 语义为"等价类内不退化"。
- 搜索调用自身时延（一次性成本，信息性）：全候选窗口用例（8 候选 × 5 迭代 = 40 次评估）≈ 15 ms，fused-only（2 × 5 = 10 次）≈ 3 ms。

**精度**（matD 混合容差 `|actual−golden| ≤ atol + rtol×|golden|` 且 matched_ratio ≥ 0.99 且 max_abs_error ≤ max(fixed, 32×ULP)；INT8 路径两级分派：α==1 ∧ β==0 ∧ 无 bias/activation 时 EXACT，否则 atol=1 / rtol=0 / limit=65）：

| dtype / 场景 | 阈值 | 实测最差 maxAbsErr | matched_ratio | 结论 |
|-------|------|------|------|------|
| FP32 | rtol=9.77e-4，atol=1.53e-5，limit=1e-2 | 7.2e-5 | 1.0000 | 达标 |
| FP16 | rtol=atol=1.95e-3，limit=1e-1 | 1.5625e-2 | 1.0000 | 达标 |
| BF16 | rtol=atol=1.56e-2，limit=1.0 | 7.8125e-3 | 1.0000 | 达标 |
| INT8→INT32 / INT8→INT8（EXACT 档：α=1、β=0、无 bias/activation） | EXACT（二进制一致或绝对误差为 0） | mismatches=0 | — | 达标 |
| INT8→INT32（atol=1 档：非单位缩放 / bias / activation） | atol=1，rtol=0，limit=65 | 1.0 ≤ 65 | 1.0000 | 达标 |

- 全 dtype 搜索返回输出与搜索后再执行输出均通过精度比对且**逐位一致**（bitwiseDiff=0，含多场景耦合组合 C 组 10 用例与正交补测 E 组 6 用例）——"搜索结束后以选中配置无条件再执行一次"契约的实测证据。
- **配套修复**：随测试增强暴露并修复的存量 kernel 缺陷——`aclsparseLtMatmul` 两段式（splitK>1）epilogue 向量缩放路径在 n 不满足对应 dtype 的 32B 对齐粒度且启用向量缩放或 bias 时存在向量核异常（507035），已修复（epilogue 分段对齐处理），修复后任意合法 shape 可用，ltmatmul 全量 136/136 回归通过；搜索的候选失败分层语义同时保证此类个别候选异常不再导致搜索整体失败。ST 侧 T 组 4 条计时观测用例超阈（>1.10）仅输出 WARN 不 FAIL（event 计时含不可消除的环境噪声，选优有效性的深入验证由多轮 median 的性能工具承载）。

## 参考资源

- 配套算子：[aclsparseLtMatmul](../matmul/README.md)（计时执行载体与选中配置执行）、[aclsparseLtSpMMAPrune](../prune/README.md)（剪枝，产出 Matmul 的稀疏侧输入 A_pruned）、[aclsparseLtMatmulAlgSetAttribute](../alg_set_attribute/README.md) / aclsparseLtMatmulAlgGetAttribute（算法属性设置 / 查询）。
