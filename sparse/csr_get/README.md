# CsrGet算子

## 算子概述

csr_get 算子用于从 CSR（Compressed Sparse Row）格式稀疏矩阵描述符中读回全部字段，对标 cuSPARSE 的 `cusparseCreateCsr` / `cusparseCsrGet` / `cusparseCreateConstCsr` / `cusparseConstCsrGet`，属于 SparseAccessor（描述符访问器）类接口。

CsrGet 为**纯 host 访问器**：描述符的构造与字段读取均在 Host 侧完成，**不启动 NPU kernel、不涉及 stream 同步、无 GPU 计算**。`aclsparseCreateCsr` 将调用方传入的行偏移、列索引与值的设备指针连同维度、两个索引类型、索引基、值类型一并存入描述符；`aclsparseCsrGet` 则从描述符中把这些字段原样读回，读回的设备指针与 Create 时传入的指针逐字一致（accessor 不拷贝、不移动设备数据）。语义与 cuSPARSE 完全对齐，便于 cuSPARSE 用户无缝迁移。

CSR 格式以三个数组描述稀疏矩阵：行偏移 `csrRowOffsets`（长度 `rows + 1`，按行压缩存储每行非零元素的起始位置）、列索引 `csrColInd`（长度 `nnz`）与非零值 `csrValues`（长度 `nnz`）。与 COO 的关键差异在于：**行偏移索引类型（`csrRowOffsetsType`）与列索引类型（`csrColIndType`）是两个相互独立的参数**，各自可取 `ACL_SPARSE_INDEX_32I` / `ACL_SPARSE_INDEX_64I`（四种组合均合法），因此 `aclsparseCsrGet` 有两个独立的索引类型输出参数。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseCreateCsr | 创建 CSR 格式稀疏矩阵描述符，存入维度与设备指针 |
| aclsparseCreateConstCsr | 创建只读(const)CSR 格式稀疏矩阵描述符 |
| aclsparseCsrGet | 从 CSR 描述符读回全部字段（维度 / 设备指针 / 两个索引类型 / 索引基 / 值类型） |
| aclsparseConstCsrGet | 从只读(const)CSR 描述符读回全部字段 |

## 算子执行接口

### aclsparseCreateCsr

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCreateCsr(aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *csrRowOffsets, void *csrColInd, void *csrValues, aclsparseIndexType_t csrRowOffsetsType,
    aclsparseIndexType_t csrColIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入/输出 | aclsparseSpMatDescr_t* | CSR 稀疏矩阵描述符（输出句柄），调用前无需初始化 | Host |
| rows | 输入 | int64_t | 矩阵行数，rows >= 0 | Host |
| cols | 输入 | int64_t | 矩阵列数，cols >= 0 | Host |
| nnz | 输入 | int64_t | 非零元素个数，nnz >= 0 | Host |
| csrRowOffsets | 输入 | void* | CSR 行偏移数组，长度为 rows + 1 | Device |
| csrColInd | 输入 | void* | CSR 列索引数组，长度为 nnz | Device |
| csrValues | 输入 | void* | CSR 非零值数组，长度为 nnz | Device |
| csrRowOffsetsType | 输入 | aclsparseIndexType_t | 行偏移数组的数据类型，支持 ACL_SPARSE_INDEX_32I / ACL_SPARSE_INDEX_64I | Host |
| csrColIndType | 输入 | aclsparseIndexType_t | 列索引数组的数据类型，支持 ACL_SPARSE_INDEX_32I / ACL_SPARSE_INDEX_64I，可独立于 csrRowOffsetsType 选择 | Host |
| idxBase | 输入 | aclsparseIndexBase_t | 索引基值（ACL_SPARSE_INDEX_BASE_ZERO / ACL_SPARSE_INDEX_BASE_ONE） | Host |
| valueType | 输入 | aclDataType | 非零值数据类型 | Host |

#### 约束说明

- rows >= 0，cols >= 0，nnz >= 0，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- spMatDescr（输出句柄）不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- csrRowOffsetsType 与 csrColIndType **相互独立**，各自仅支持 `ACL_SPARSE_INDEX_32I` 与 `ACL_SPARSE_INDEX_64I`（四种组合均接受），其他取值返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。注意：accessor 层与 cuSPARSE 一致接受全部四种组合；但本仓 CSR 计算类接口（SpMV / SpMM 的 execute 路径）当前仅支持 32I 索引，混用计算接口时以计算接口的约束为准
- idxBase 仅接受 `ACL_SPARSE_INDEX_BASE_ZERO` / `ACL_SPARSE_INDEX_BASE_ONE`，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- valueType 须为库支持的稀疏值类型（如 ACL_FLOAT / ACL_FLOAT16 / ACL_DOUBLE / ACL_BF16 / 各整型等），否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- csrRowOffsets / csrColInd / csrValues 允许为 nullptr（延迟绑定：Create 仅记录指针，有效性推迟到后续 SetPointers / 计算接口使用时校验）；此时 `aclsparseCsrGet` 读回的对应指针为 nullptr
- aclsparseCreateCsr 为纯 host 操作，不启动 NPU kernel，不涉及 stream 同步
- const 变体（aclsparseCreateConstCsr）构造出的描述符只能传给接收 const 形参的接口；销毁统一用 `aclsparseDestroySpMat`（接受 const 变体，const 与非 const 描述符均可销毁）

---

### aclsparseCreateConstCsr

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCreateConstCsr(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols,
    int64_t nnz, const void *csrRowOffsets, const void *csrColInd, const void *csrValues,
    aclsparseIndexType_t csrRowOffsetsType, aclsparseIndexType_t csrColIndType,
    aclsparseIndexBase_t idxBase, aclDataType valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入/输出 | aclsparseConstSpMatDescr_t* | 只读(const)CSR 稀疏矩阵描述符（输出句柄） | Host |
| rows | 输入 | int64_t | 矩阵行数，rows >= 0 | Host |
| cols | 输入 | int64_t | 矩阵列数，cols >= 0 | Host |
| nnz | 输入 | int64_t | 非零元素个数，nnz >= 0 | Host |
| csrRowOffsets | 输入 | const void* | CSR 行偏移数组，长度为 rows + 1 | Device |
| csrColInd | 输入 | const void* | CSR 列索引数组，长度为 nnz | Device |
| csrValues | 输入 | const void* | CSR 非零值数组，长度为 nnz | Device |
| csrRowOffsetsType | 输入 | aclsparseIndexType_t | 行偏移数组的数据类型 | Host |
| csrColIndType | 输入 | aclsparseIndexType_t | 列索引数组的数据类型（独立于 csrRowOffsetsType） | Host |
| idxBase | 输入 | aclsparseIndexBase_t | 索引基值 | Host |
| valueType | 输入 | aclDataType | 非零值数据类型 | Host |

#### 约束说明

- 约束同 aclsparseCreateCsr（rows / cols / nnz 非负，spMatDescr 非 nullptr，索引类型各自为 32I / 64I，idxBase 与 valueType 合法）
- 构造出的描述符为 const 变体，只能传给接收 const 形参的接口（如 `aclsparseConstCsrGet`）；销毁统一用 `aclsparseDestroySpMat`
- 其余约束与 aclsparseCreateCsr 一致

---

### aclsparseCsrGet

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCsrGet(aclsparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
    int64_t *nnz, void **csrRowOffsets, void **csrColInd, void **csrValues,
    aclsparseIndexType_t *csrRowOffsetsType, aclsparseIndexType_t *csrColIndType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入 | aclsparseSpMatDescr_t | CSR 稀疏矩阵描述符 | Host |
| rows | 输出 | int64_t* | 矩阵行数 | Host |
| cols | 输出 | int64_t* | 矩阵列数 | Host |
| nnz | 输出 | int64_t* | 非零元素个数 | Host |
| csrRowOffsets | 输出 | void** | 行偏移数组指针（与 Create 传入的设备指针一致，长度 rows + 1） | Device |
| csrColInd | 输出 | void** | 列索引数组指针（与 Create 传入的设备指针一致，长度 nnz） | Device |
| csrValues | 输出 | void** | 非零值数组指针（与 Create 传入的设备指针一致，长度 nnz） | Device |
| csrRowOffsetsType | 输出 | aclsparseIndexType_t* | 行偏移数组的数据类型 | Host |
| csrColIndType | 输出 | aclsparseIndexType_t* | 列索引数组的数据类型（独立于 csrRowOffsetsType 读回） | Host |
| idxBase | 输出 | aclsparseIndexBase_t* | 索引基值 | Host |
| valueType | 输出 | aclDataType* | 非零值数据类型 | Host |

#### 约束说明

- spMatDescr（输入描述符）不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- spMatDescr 必须为稀疏矩阵描述符（误传其他类型描述符句柄，如稠密向量/稠密矩阵描述符）时返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- 对非 CSR 格式的描述符（COO / CSC / BELL / SLICED_ELL 等）返回 `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`
- 任一 OUT 指针可为 nullptr，此时仅返回请求的字段（对应字段不回写），与 `aclsparseSpMatGetSize` / `aclsparseSpVecGet` / `aclsparseCooGet` 措辞一致
- csrRowOffsetsType 与 csrColIndType 为两个独立输出，分别读回 Create 时传入的 csrRowOffsetsType 与 csrColIndType（四种 32I/64I 组合均可正确读回）
- aclsparseCsrGet 为纯 host 操作，不启动 NPU kernel，不涉及 stream 同步

---

### aclsparseConstCsrGet

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseConstCsrGet(aclsparseConstSpMatDescr_t spMatDescr, int64_t *rows,
    int64_t *cols, int64_t *nnz, const void **csrRowOffsets, const void **csrColInd,
    const void **csrValues, aclsparseIndexType_t *csrRowOffsetsType,
    aclsparseIndexType_t *csrColIndType, aclsparseIndexBase_t *idxBase, aclDataType *valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入 | aclsparseConstSpMatDescr_t | 只读(const)CSR 稀疏矩阵描述符 | Host |
| rows | 输出 | int64_t* | 矩阵行数 | Host |
| cols | 输出 | int64_t* | 矩阵列数 | Host |
| nnz | 输出 | int64_t* | 非零元素个数 | Host |
| csrRowOffsets | 输出 | const void** | 行偏移数组指针（与 Create 传入的指针一致） | Device |
| csrColInd | 输出 | const void** | 列索引数组指针（与 Create 传入的指针一致） | Device |
| csrValues | 输出 | const void** | 非零值数组指针（与 Create 传入的指针一致） | Device |
| csrRowOffsetsType | 输出 | aclsparseIndexType_t* | 行偏移数组的数据类型 | Host |
| csrColIndType | 输出 | aclsparseIndexType_t* | 列索引数组的数据类型 | Host |
| idxBase | 输出 | aclsparseIndexBase_t* | 索引基值 | Host |
| valueType | 输出 | aclDataType* | 非零值数据类型 | Host |

#### 约束说明

- 约束同 aclsparseCsrGet（spMatDescr 非 nullptr 且须为稀疏矩阵描述符句柄，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；非 CSR 格式返回 `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`）
- 任一 OUT 指针可为 nullptr，此时仅返回请求的字段
- aclsparseConstCsrGet 为纯 host 操作，不启动 NPU kernel，不涉及 stream 同步

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

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

using DevicePtr = std::unique_ptr<void, aclError (*)(void *)>;

// 辅助：分配 Device 内存并拷贝 Host 数据，内存由 DevicePtr 自动释放
static int AllocAndCopyDevice(DevicePtr &devicePtr, const void *hostPtr, size_t sizeBytes)
{
    void *rawPtr = nullptr;
    auto ret = aclrtMalloc(&rawPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    devicePtr.reset(rawPtr);

    if (hostPtr != nullptr && sizeBytes > 0) {
        ret = aclrtMemcpy(devicePtr.get(), sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy data to Device failed. ERROR: %d\n", ret); return ret);
    }
    return ACL_SUCCESS;
}

int aclsparseCsrGetTest(AclContext &ctx)
{
    (void)ctx.Stream();  // accessor 不使用 handle / stream，此处仅占位

    // 1. 准备 Host 端 CSR 数据：3 个非零元的 3x3 矩阵
    //    [1.0  0.0  0.0]
    //    [0.0  0.0  2.0]
    //    [0.0  3.0  0.0]
    int64_t rows = 3;
    int64_t cols = 3;
    int64_t nnz = 3;

    std::vector<int32_t> hCsrRowOffsets = {0, 1, 2, 3};  // 长度 rows + 1
    std::vector<int32_t> hCsrColInd = {0, 2, 1};
    std::vector<float> hCsrValues = {1.0f, 2.0f, 3.0f};

    // 2. 拷贝 rowOffsets/colInd/values 到 Device
    DevicePtr dCsrRowOffsets(nullptr, aclrtFree);
    DevicePtr dCsrColInd(nullptr, aclrtFree);
    DevicePtr dCsrValues(nullptr, aclrtFree);
    auto aclRet = AllocAndCopyDevice(dCsrRowOffsets, hCsrRowOffsets.data(), (rows + 1) * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dCsrColInd, hCsrColInd.data(), nnz * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dCsrValues, hCsrValues.data(), nnz * sizeof(float));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 3. aclsparseCreateCsr：将维度与设备指针存入描述符
    //    （行偏移与列索引类型相互独立，此处均为 32I，也可分别选择 32I/64I）
    aclsparseSpMatDescr_t matDescr = nullptr;
    auto sparseRet = aclsparseCreateCsr(&matDescr, rows, cols, nnz,
        dCsrRowOffsets.get(), dCsrColInd.get(), dCsrValues.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCreateCsr failed: %d\n", sparseRet); return sparseRet);

    // 4. aclsparseCsrGet：读回描述符全部字段
    int64_t outRows = 0, outCols = 0, outNnz = 0;
    void *outRowOffsets = nullptr, *outColInd = nullptr, *outValues = nullptr;
    aclsparseIndexType_t outRowOffsetsType{};
    aclsparseIndexType_t outColIndType{};
    aclsparseIndexBase_t outIdxBase{};
    aclDataType outValueType{};
    sparseRet = aclsparseCsrGet(matDescr, &outRows, &outCols, &outNnz,
        &outRowOffsets, &outColInd, &outValues, &outRowOffsetsType, &outColIndType,
        &outIdxBase, &outValueType);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCsrGet failed: %d\n", sparseRet); aclsparseDestroySpMat(matDescr); return sparseRet);

    // 5. accessor 语义断言：读回字段与 Create 传入一致
    CHECK_RET(outRows == rows && outCols == cols && outNnz == nnz,
        LOG_PRINT("size mismatch: rows=%lld/%lld cols=%lld/%lld nnz=%lld/%lld\n",
            static_cast<long long>(outRows), static_cast<long long>(rows),
            static_cast<long long>(outCols), static_cast<long long>(cols),
            static_cast<long long>(outNnz), static_cast<long long>(nnz));
        aclsparseDestroySpMat(matDescr); return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    CHECK_RET(outRowOffsetsType == ACL_SPARSE_INDEX_32I && outColIndType == ACL_SPARSE_INDEX_32I &&
              outIdxBase == ACL_SPARSE_INDEX_BASE_ZERO && outValueType == ACL_FLOAT,
        LOG_PRINT("attr mismatch\n"); aclsparseDestroySpMat(matDescr); return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    // 读回的设备指针应与 Create 传入的逐字一致（accessor 不拷贝设备数据）
    CHECK_RET(outRowOffsets == dCsrRowOffsets.get() && outColInd == dCsrColInd.get() &&
              outValues == dCsrValues.get(),
        LOG_PRINT("device pointer mismatch\n"); aclsparseDestroySpMat(matDescr);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    LOG_PRINT("aclsparseCreateCsr / aclsparseCsrGet succeeded.\n");
    LOG_PRINT("read back: rows=%lld, cols=%lld, nnz=%lld\n",
        static_cast<long long>(outRows), static_cast<long long>(outCols), static_cast<long long>(outNnz));
    LOG_PRINT("  attrs match: rowOffsetsType=ACL_SPARSE_INDEX_32I, colIndType=ACL_SPARSE_INDEX_32I, idxBase=ACL_SPARSE_INDEX_BASE_ZERO, valueType=ACL_FLOAT\n");
    LOG_PRINT("  devPtr match: rowOff=%d, colInd=%d, val=%d\n",
        outRowOffsets == dCsrRowOffsets.get() ? 1 : 0,
        outColInd == dCsrColInd.get() ? 1 : 0,
        outValues == dCsrValues.get() ? 1 : 0);

    // 6. 销毁描述符（统一用 aclsparseDestroySpMat）
    aclsparseDestroySpMat(matDescr);
    LOG_PRINT("aclsparseDestroySpMat done.\n");

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseCsrGetTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCsrGetTest failed: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
aclsparseCreateCsr / aclsparseCsrGet succeeded.
read back: rows=3, cols=3, nnz=3
  attrs match: rowOffsetsType=ACL_SPARSE_INDEX_32I, colIndType=ACL_SPARSE_INDEX_32I, idxBase=ACL_SPARSE_INDEX_BASE_ZERO, valueType=ACL_FLOAT
  devPtr match: rowOff=1, colInd=1, val=1
aclsparseDestroySpMat done.
```
