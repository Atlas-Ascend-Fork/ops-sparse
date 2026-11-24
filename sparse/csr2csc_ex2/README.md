# Csr2cscEx2 算子

> 6.1 文档补全阶段产出。developer-doc 依据算子代码与设计文档编写，面向使用者。

## 功能描述

`aclsparseCsr2cscEx2` 算子用于将同一个逻辑 `m x n` 稀疏矩阵从 CSR（Compressed Sparse Row）格式转换为 CSC（Compressed Sparse Column）格式；这不是数学转置。每个非零元素的逻辑坐标 `(i, j, v)` 保持不变：它在输出 CSC 的第 `j` 列中仍以行索引 `i` 保存。输出的 `cscColPtr` 为长度 `n + 1` 的列偏移数组，`cscRowInd` 为行索引数组，`cscVal` 为非零值数组，输出列内行索引升序排列。

该接口为 Legacy API 中的 type-generic 接口，通过 `valType` 参数分发数据类型（不使用 MatDescr），支持 INT8 / FP16 / BF16 / FP32 四种数据类型；支持 0-based 与 1-based 两种索引基址；支持 SYMBOLIC（仅计算结构）与 NUMERIC（计算结构并拷贝值）两种操作模式。转换仅涉及索引重排与数据搬运，不涉及浮点运算，输出与参考实现逐 bit 一致（位精确）。

## 接口原型

```cpp
aclsparseStatus_t aclsparseCsr2cscEx2_bufferSize(
    aclsparseHandle_t        handle,
    int                      m,
    int                      n,
    int                      nnz,
    const void              *csrVal,
    const int               *csrRowPtr,
    const int               *csrColInd,
    void                    *cscVal,
    int                     *cscColPtr,
    int                     *cscRowInd,
    aclDataType              valType,
    aclsparseAction_t        copyValues,
    aclsparseIndexBase_t     idxBase,
    aclsparseCsr2CscAlg_t    alg,
    size_t                  *bufferSize);

aclsparseStatus_t aclsparseCsr2cscEx2(
    aclsparseHandle_t        handle,
    int                      m,
    int                      n,
    int                      nnz,
    const void              *csrVal,
    const int               *csrRowPtr,
    const int               *csrColInd,
    void                    *cscVal,
    int                     *cscColPtr,
    int                     *cscRowInd,
    aclDataType              valType,
    aclsparseAction_t        copyValues,
    aclsparseIndexBase_t     idxBase,
    aclsparseCsr2CscAlg_t    alg,
    void                    *buffer);
```

## 参数说明

### aclsparseCsr2cscEx2_bufferSize

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|----------|------|------|------|
| handle | Host | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream 等信息，不可为 nullptr |
| m | Host | 输入 | int | 逻辑矩阵行数，m >= 0 |
| n | Host | 输入 | int | 逻辑矩阵列数；`cscColPtr` 长度为 n+1，n >= 0 |
| nnz | Host | 输入 | int | 非零元素个数，nnz >= 0 |
| csrVal | Device | 输入 | const void* | CSR 非零元素值数组，长度为 nnz（本函数不读取，可为 nullptr） |
| csrRowPtr | Device | 输入 | const int* | CSR 行偏移数组，长度为 m+1（本函数不读取，可为 nullptr） |
| csrColInd | Device | 输入 | const int* | CSR 列索引数组，长度为 nnz（本函数不读取，可为 nullptr） |
| cscVal | Device | 输入 | void* | CSC 非零元素值数组，长度为 nnz（本函数不读取，可为 nullptr） |
| cscColPtr | Device | 输入 | int* | CSC 列偏移数组，长度为 n+1（本函数不读取，可为 nullptr） |
| cscRowInd | Device | 输入 | int* | CSC 行索引数组，长度为 nnz（本函数不读取，可为 nullptr） |
| valType | Host | 输入 | aclDataType | 非零元素的数据类型，支持 ACL_INT8 / ACL_FLOAT16 / ACL_BF16 / ACL_FLOAT |
| copyValues | Host | 输入 | aclsparseAction_t | 操作类型：ACL_SPARSE_ACTION_SYMBOLIC 或 ACL_SPARSE_ACTION_NUMERIC |
| idxBase | Host | 输入 | aclsparseIndexBase_t | 索引基址：ACL_SPARSE_INDEX_BASE_ZERO（0-based）或 ACL_SPARSE_INDEX_BASE_ONE（1-based） |
| alg | Host | 输入 | aclsparseCsr2CscAlg_t | 算法选择：ACL_SPARSE_CSR2CSC_ALG_DEFAULT 或 ACL_SPARSE_CSR2CSC_ALG1（当前行为一致） |
| bufferSize | Host | 输出 | size_t* | 输出所需 workspace 大小（字节），不可为 nullptr |

### aclsparseCsr2cscEx2

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|----------|------|------|------|
| handle | Host | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream 等信息，不可为 nullptr |
| m | Host | 输入 | int | 逻辑矩阵行数，m >= 0 |
| n | Host | 输入 | int | 逻辑矩阵列数；`cscColPtr` 长度为 n+1，n >= 0 |
| nnz | Host | 输入 | int | 非零元素个数，nnz >= 0 |
| csrVal | Device | 输入 | const void* | CSR 非零元素值数组，长度为 nnz；nnz > 0 且 copyValues == NUMERIC 时不可为 nullptr |
| csrRowPtr | Device | 输入 | const int* | CSR 行偏移数组，长度为 m+1；m > 0 时不可为 nullptr。输入数据合法性为调用方前置条件：必须非降且 csrRowPtr[m] - idxBase == nnz，算子不对输入数据内容做边界校验 |
| csrColInd | Device | 输入 | const int* | CSR 列索引数组，长度为 nnz；nnz > 0 时不可为 nullptr。输入数据合法性为调用方前置条件：元素值域 ∈ [idxBase, idxBase+n)，算子不对输入数据内容做边界校验 |
| cscVal | Device | 输出 | void* | CSC 非零元素值数组，长度为 nnz；仅 NUMERIC 模式写入，SYMBOLIC 模式不修改；nnz > 0 且 copyValues == NUMERIC 时不可为 nullptr |
| cscColPtr | Device | 输出 | int* | CSC 列偏移数组，长度为 n+1，不可为 nullptr（nnz == 0 快路径也会写入） |
| cscRowInd | Device | 输出 | int* | CSC 行索引数组，长度为 nnz；nnz > 0 时不可为 nullptr |
| valType | Host | 输入 | aclDataType | 非零元素的数据类型，支持 ACL_INT8 / ACL_FLOAT16 / ACL_BF16 / ACL_FLOAT |
| copyValues | Host | 输入 | aclsparseAction_t | 操作类型：ACL_SPARSE_ACTION_SYMBOLIC（仅计算 cscColPtr 和 cscRowInd）或 ACL_SPARSE_ACTION_NUMERIC（同时拷贝 cscVal） |
| idxBase | Host | 输入 | aclsparseIndexBase_t | 索引基址：ACL_SPARSE_INDEX_BASE_ZERO（0-based）或 ACL_SPARSE_INDEX_BASE_ONE（1-based） |
| alg | Host | 输入 | aclsparseCsr2CscAlg_t | 算法选择：ACL_SPARSE_CSR2CSC_ALG_DEFAULT 或 ACL_SPARSE_CSR2CSC_ALG1（当前行为一致，ALG1 仅为接口兼容性保留） |
| buffer | Device | 输入 | void* | workspace 缓冲区（大小由 bufferSize 接口查询）；若为 nullptr 则回退使用 handle 预置的 workspace |

## 支持数据类型

| 数据类型 | 说明 |
|----------|------|
| INT8 (ACL_INT8) | 有符号 8 位整数，valSize = 1 byte |
| FP16 (ACL_FLOAT16) | 半精度浮点，valSize = 2 bytes |
| BF16 (ACL_BF16) | bfloat16，valSize = 2 bytes |
| FP32 (ACL_FLOAT) | 单精度浮点，valSize = 4 bytes |

索引类型统一为 int32（32 位有符号整数），所有索引数组使用同一类型。

## 约束说明

**shape 约束**：`m >= 0`、`n >= 0`、`nnz >= 0`。`n + 1` 不得超过 int32 上限（n <= 2147483646）；`nnz` 不得超过 int32 上限（nnz <= 2147483647）；1-based 时 `nnz + idxBase` 不得溢出 int32（nnz <= 2147483646）。

**dtype 约束**：`valType` 必须为 ACL_INT8 / ACL_FLOAT16 / ACL_BF16 / ACL_FLOAT 之一，否则返回 ACL_SPARSE_STATUS_NOT_SUPPORTED。

**枚举值约束**：`copyValues` 必须为 ACL_SPARSE_ACTION_SYMBOLIC 或 ACL_SPARSE_ACTION_NUMERIC；`idxBase` 必须为 ACL_SPARSE_INDEX_BASE_ZERO 或 ACL_SPARSE_INDEX_BASE_ONE；`alg` 必须为 ACL_SPARSE_CSR2CSC_ALG_DEFAULT 或 ACL_SPARSE_CSR2CSC_ALG1（当前两者行为一致，ALG1 仅为接口兼容性保留）。违反上述任一约束返回 ACL_SPARSE_STATUS_INVALID_VALUE。

**指针约束**：`handle` 不可为 nullptr；`bufferSize`（bufferSize 接口）不可为 nullptr；`cscColPtr`（compute 接口）不可为 nullptr；`csrRowPtr` 在 m > 0 时不可为 nullptr；`csrColInd` / `cscRowInd` 在 nnz > 0 时不可为 nullptr；`csrVal` / `cscVal` 在 nnz > 0 且 copyValues == NUMERIC 时不可为 nullptr。违反上述任一约束返回 ACL_SPARSE_STATUS_INVALID_VALUE。

**输入数据契约（调用方前置条件）**：`csrRowPtr` 必须非降且 `csrRowPtr[m] - idxBase == nnz`；`csrColInd` 元素值域必须 ∈ [idxBase, idxBase+n)。算子不对输入数据内容做边界校验，违反契约的后果由调用方承担。

**边界行为**：
- `m == 0` 或 `n == 0` 或 `nnz == 0` 时，执行路径不使用 workspace，`bufferSize = 0`，`cscColPtr` 全部填充为 idxBase（长度恒为 n+1 >= 1，仍会被写入），`cscRowInd` / `cscVal` 不写入。
- `m == 0` 且 `nnz > 0`，或 `n == 0` 且 `nnz > 0`，违反 CSR 数据契约，返回 ACL_SPARSE_STATUS_INVALID_VALUE。
- `nnz == 0` 且 `idxBase == 1` 时，`cscColPtr` 填充使用同步 H2D 拷贝（避免栈上缓冲区 use-after-free），此路径会短暂阻塞 host 线程；其余路径异步执行。

**异步执行**：函数异步执行，返回时结果可能尚未就绪，调用后需通过 `aclrtSynchronizeStream` 同步。

**平台约束**：Kernel 使用 `asc_atomic_add` 原子操作，目标架构为 arch35（Ascend 950PR&950DT系列产品），其他芯片不支持。

**workspace 大小**：`nnz > 0` 时 `bufferSize = (1 + stripeCount) × (n + 1) × sizeof(int32_t)`，其中 `stripeCount = min(⌈nnz / 256⌉, AIV 核数, kCsr2CscMaxStripeWorkspaceBytes / ((n+1) × sizeof(int32_t)))`（至少为 1）。段 0 为 colCount（n+1 个 int32），段 1 起为 stripeHist / 游标区（stripeCount 段，每段 n+1 个 int32）。

## 返回值 / 错误码

| 返回值 | 含义 |
|--------|------|
| ACL_SPARSE_STATUS_SUCCESS | 执行成功 |
| ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR | handle 为 nullptr |
| ACL_SPARSE_STATUS_INVALID_VALUE | 参数非法：m/n/nnz < 0；m==0 或 n==0 但 nnz>0（违反数据契约）；copyValues / idxBase / alg 枚举值非法；bufferSize / cscColPtr 为 nullptr；依据 nnz 与 copyValues 模式必填的指针为 nullptr |
| ACL_SPARSE_STATUS_NOT_SUPPORTED | valType 不在支持范围内（非 ACL_INT8 / ACL_FLOAT16 / ACL_BF16 / ACL_FLOAT） |
| ACL_SPARSE_STATUS_EXECUTION_FAILED | Kernel 执行或 stream 同步失败（Runtime 错误统一映射） |
| ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES | buffer 为 nullptr 且 handle 未预置 workspace，无法分配 workspace |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdint>
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

// Device 内存智能指针：带 aclrtFree deleter，离开作用域自动释放
using DevicePtr = std::unique_ptr<void, aclError (*)(void *)>;

// 辅助：分配 Device 内存，失败返回空指针
// ACL_MEM_MALLOC_HUGE_FIRST 优先使用大页内存（HugePage），可提升 HBM 读写带宽
static DevicePtr MallocDevice(size_t sizeBytes)
{
    void *rawPtr = nullptr;
    auto ret = aclrtMalloc(&rawPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
              return DevicePtr(nullptr, aclrtFree));
    return DevicePtr(rawPtr, aclrtFree);
}

// 辅助：分配 Device 内存并拷贝 Host 数据，失败返回空指针
static DevicePtr AllocAndCopyDevice(const void *hostPtr, size_t sizeBytes)
{
    DevicePtr devPtr = MallocDevice(sizeBytes);
    CHECK_RET(devPtr != nullptr, return DevicePtr(nullptr, aclrtFree));
    if (hostPtr != nullptr && sizeBytes > 0) {
        auto ret = aclrtMemcpy(devPtr.get(), sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret);
                  return DevicePtr(nullptr, aclrtFree));
    }
    return devPtr;
}

int aclsparseCsr2cscEx2Test(AclContext &ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreate failed: %d\n", sparseRet);
              return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(rawHandle, aclsparseDestroy);
    sparseRet = aclsparseSetStream(handlePtr.get(), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetStream failed: %d\n", sparseRet);
              return sparseRet);

    // 2. 准备 Host 端 CSR 数据
    //    示例：3x3 稀疏矩阵，nnz=4，0-based 索引
    //
    //    A (3x3, nnz=4):
    //    [1.0  2.0  0.0]
    //    [0.0  0.0  3.0]
    //    [4.0  0.0  0.0]
    //
    //    CSR 表示:
    //    csrRowPtr = [0, 2, 3, 4]
    //    csrColInd = [0, 1, 2, 0]
    //    csrVal    = [1.0, 2.0, 3.0, 4.0]
    //
    //    CSC 预期输出:
    //    cscColPtr = [0, 2, 3, 4]
    //    cscRowInd = [0, 2, 0, 1]
    //    cscVal    = [1.0, 4.0, 2.0, 3.0]

    int m = 3;
    int n = 3;
    int nnz = 4;

    std::vector<int> hCsrRowPtr = {0, 2, 3, 4};
    std::vector<int> hCsrColInd = {0, 1, 2, 0};
    std::vector<float> hCsrVal  = {1.0f, 2.0f, 3.0f, 4.0f};

    // 3. 拷贝 CSR 数据到 Device（DevicePtr 自动管理释放，错误路径不泄露）
    DevicePtr dCsrRowPtr = AllocAndCopyDevice(hCsrRowPtr.data(), (m + 1) * sizeof(int));
    CHECK_RET(dCsrRowPtr != nullptr, LOG_PRINT("AllocAndCopyDevice dCsrRowPtr failed\n");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    DevicePtr dCsrColInd = AllocAndCopyDevice(hCsrColInd.data(), nnz * sizeof(int));
    CHECK_RET(dCsrColInd != nullptr, LOG_PRINT("AllocAndCopyDevice dCsrColInd failed\n");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    DevicePtr dCsrVal    = AllocAndCopyDevice(hCsrVal.data(),    nnz * sizeof(float));
    CHECK_RET(dCsrVal != nullptr, LOG_PRINT("AllocAndCopyDevice dCsrVal failed\n");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    // 4. 分配 CSC 输出数组
    DevicePtr dCscColPtr = MallocDevice((n + 1) * sizeof(int));
    CHECK_RET(dCscColPtr != nullptr, LOG_PRINT("MallocDevice dCscColPtr failed\n");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    DevicePtr dCscRowInd = MallocDevice(nnz * sizeof(int));
    CHECK_RET(dCscRowInd != nullptr, LOG_PRINT("MallocDevice dCscRowInd failed\n");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    DevicePtr dCscVal    = MallocDevice(nnz * sizeof(float));
    CHECK_RET(dCscVal != nullptr, LOG_PRINT("MallocDevice dCscVal failed\n");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    // 5. 查询 workspace 大小
    size_t bufferSize = 0;
    sparseRet = aclsparseCsr2cscEx2_bufferSize(
        handlePtr.get(), m, n, nnz,
        dCsrVal.get(), static_cast<const int*>(dCsrRowPtr.get()), static_cast<const int*>(dCsrColInd.get()),
        dCscVal.get(), static_cast<int*>(dCscColPtr.get()), static_cast<int*>(dCscRowInd.get()),
        ACL_FLOAT, ACL_SPARSE_ACTION_NUMERIC, ACL_SPARSE_INDEX_BASE_ZERO,
        ACL_SPARSE_CSR2CSC_ALG_DEFAULT, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("bufferSize failed: %d\n", sparseRet);
              return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes\n", bufferSize);

    // 6. 分配 workspace（bufferSize == 0 时无需 workspace，跳过分配避免 aclrtMalloc(0) 报错）
    DevicePtr dWorkspace(nullptr, aclrtFree);
    if (bufferSize > 0) {
        dWorkspace = MallocDevice(bufferSize);
        CHECK_RET(dWorkspace != nullptr,
                  LOG_PRINT("MallocDevice workspace failed\n");
                  return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    }

    // 7. 执行 CSR → CSC 格式转换
    sparseRet = aclsparseCsr2cscEx2(
        handlePtr.get(), m, n, nnz,
        dCsrVal.get(), static_cast<const int*>(dCsrRowPtr.get()), static_cast<const int*>(dCsrColInd.get()),
        dCscVal.get(), static_cast<int*>(dCscColPtr.get()), static_cast<int*>(dCscRowInd.get()),
        ACL_FLOAT, ACL_SPARSE_ACTION_NUMERIC, ACL_SPARSE_INDEX_BASE_ZERO,
        ACL_SPARSE_CSR2CSC_ALG_DEFAULT, dWorkspace.get());
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("csr2cscEx2 failed: %d\n", sparseRet);
              return sparseRet);

    // 8. 同步等待计算完成（Runtime 错误统一映射为 ACL_SPARSE_STATUS_EXECUTION_FAILED，不与 aclError 混用）
    auto aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    // 9. 将结果拷贝回 Host 并打印
    std::vector<int>   hCscColPtr(n + 1, 0);
    std::vector<int>   hCscRowInd(nnz, 0);
    std::vector<float> hCscVal(nnz, 0.0f);

    aclRet = aclrtMemcpy(hCscColPtr.data(), (n + 1) * sizeof(int), dCscColPtr.get(), (n + 1) * sizeof(int),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", aclRet);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    aclRet = aclrtMemcpy(hCscRowInd.data(), nnz * sizeof(int), dCscRowInd.get(), nnz * sizeof(int),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", aclRet);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    aclRet = aclrtMemcpy(hCscVal.data(), nnz * sizeof(float), dCscVal.get(), nnz * sizeof(float),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", aclRet);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    LOG_PRINT("\nResult CSC:\n");
    LOG_PRINT("  colPtr: ");
    for (int i = 0; i <= n; i++) {
        LOG_PRINT("%d ", hCscColPtr[i]);
    }
    LOG_PRINT("\n  rowInd: ");
    for (int i = 0; i < nnz; i++) {
        LOG_PRINT("%d ", hCscRowInd[i]);
    }
    LOG_PRINT("\n  val:    ");
    for (int i = 0; i < nnz; i++) {
        LOG_PRINT("%.1f ", hCscVal[i]);
    }
    LOG_PRINT("\n");

    // 10. Device 内存由 DevicePtr 在函数返回时自动释放，句柄由 handlePtr 自动销毁
    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseCsr2cscEx2Test(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCsr2cscEx2Test failed: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下（arch35 设备示例值）：

```
bufferSize = 32 bytes

Result CSC:
  colPtr: 0 2 3 4 
  rowInd: 0 2 0 1 
  val:    1.0 4.0 2.0 3.0 
```

## 支持芯片

Ascend 950PR&950DT系列产品（arch35）。其他芯片（Ascend910B / Ascend910_93 / Ascend310P 等）不支持。
