# CscGet算子

## 算子概述

csc_get 算子用于从 CSC（Compressed Sparse Column）格式稀疏矩阵描述符中读回全部字段，对标 cuSPARSE 的 `cusparseCreateCsc` / `cusparseCscGet` / `cusparseCreateConstCsc` / `cusparseConstCscGet`，属于 SparseAccessor（描述符访问器）类接口。

CscGet 为**纯 host 访问器**：描述符的构造与字段读取均在 Host 侧完成，**不启动 NPU kernel、不涉及 stream 同步、无 GPU 计算**。`aclsparseCreateCsc` 将调用方传入的列偏移、行索引与值的设备指针连同维度、索引类型、索引基、值类型一并存入描述符；`aclsparseCscGet` 则从描述符中把这些字段原样读回，读回的设备指针与 Create 时传入的指针逐字一致（accessor 不拷贝、不移动设备数据）。语义与 cuSPARSE 完全对齐，便于 cuSPARSE 用户无缝迁移。

CSC 格式以三个数组描述稀疏矩阵：列偏移 `cscColOffsets`、行索引 `cscRowInd` 与非零值 `cscValues`。其中 `cscColOffsets` 长度为 `cols + 1`（区别于 CSR 的 `rows + 1`），`cscRowInd` 与 `cscValues` 长度均为 `nnz`。CSC 拥有**两个独立的索引类型**：`cscColOffsetsType` 约束列偏移数组，`cscRowIndType` 约束行索引数组，二者可分别选择 `ACL_SPARSE_INDEX_32I` 或 `ACL_SPARSE_INDEX_64I`。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseCreateCsc | 创建 CSC 格式稀疏矩阵描述符，存入维度与设备指针 |
| aclsparseCreateConstCsc | 创建只读(const)CSC 格式稀疏矩阵描述符 |
| aclsparseCscGet | 从 CSC 描述符读回全部字段（维度 / 设备指针 / 索引类型 / 索引基 / 值类型） |
| aclsparseConstCscGet | 从只读(const)CSC 描述符读回全部字段 |

## 算子执行接口

### aclsparseCreateCsc

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCreateCsc(aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *cscColOffsets, void *cscRowInd, void *cscValues, aclsparseIndexType_t cscColOffsetsType,
    aclsparseIndexType_t cscRowIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入/输出 | aclsparseSpMatDescr_t* | CSC 稀疏矩阵描述符（输出句柄），调用前无需初始化 | Host |
| rows | 输入 | int64_t | 矩阵行数，rows >= 0 | Host |
| cols | 输入 | int64_t | 矩阵列数，cols >= 0 | Host |
| nnz | 输入 | int64_t | 非零元素个数，nnz >= 0 | Host |
| cscColOffsets | 输入 | void* | CSC 列偏移数组，长度为 cols+1 | Device |
| cscRowInd | 输入 | void* | CSC 行索引数组，长度为 nnz | Device |
| cscValues | 输入 | void* | CSC 非零值数组，长度为 nnz | Device |
| cscColOffsetsType | 输入 | aclsparseIndexType_t | 列偏移数组的数据类型 | Host |
| cscRowIndType | 输入 | aclsparseIndexType_t | 行索引数组的数据类型 | Host |
| idxBase | 输入 | aclsparseIndexBase_t | 索引基值（ACL_SPARSE_INDEX_BASE_ZERO / ACL_SPARSE_INDEX_BASE_ONE） | Host |
| valueType | 输入 | aclDataType | 非零值数据类型 | Host |

#### 约束说明

- rows >= 0，cols >= 0，nnz >= 0，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- spMatDescr（输出句柄）不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- cscColOffsetsType 与 cscRowIndType 为**两个独立的索引类型**，各自支持 `ACL_SPARSE_INDEX_32I` 与 `ACL_SPARSE_INDEX_64I`（四种组合均接受），非法索引类型返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- cscColOffsets / cscRowInd / cscValues 允许为 nullptr（延迟绑定语义），可在使用前通过 `aclsparseCscSetPointers` 重新绑定
- cscColOffsets / cscRowInd / cscValues 传入非空时，须分别与 cscColOffsetsType / cscRowIndType / valueType 所指定数据类型的大小对齐（如 `ACL_SPARSE_INDEX_32I` / `ACL_FLOAT` 为 4 字节对齐、`ACL_SPARSE_INDEX_64I` 为 8 字节对齐、`ACL_FLOAT16` 为 2 字节对齐）。该要求面向后续消费该描述符的计算接口；本算子的访问器接口仅存取指针本身，不解引用设备数据
- aclsparseCreateCsc 为纯 host 操作，不启动 NPU kernel，不涉及 stream 同步
- const 变体（aclsparseCreateConstCsc）构造出的描述符只能传给接收 const 形参的接口；销毁统一用 `aclsparseDestroySpMat`（接受 const 变体，const 与非 const 描述符均可销毁）

---

### aclsparseCreateConstCsc

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCreateConstCsc(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols,
    int64_t nnz, const void *cscColOffsets, const void *cscRowInd, const void *cscValues,
    aclsparseIndexType_t cscColOffsetsType, aclsparseIndexType_t cscRowIndType,
    aclsparseIndexBase_t idxBase, aclDataType valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入/输出 | aclsparseConstSpMatDescr_t* | 只读(const)CSC 稀疏矩阵描述符（输出句柄） | Host |
| rows | 输入 | int64_t | 矩阵行数，rows >= 0 | Host |
| cols | 输入 | int64_t | 矩阵列数，cols >= 0 | Host |
| nnz | 输入 | int64_t | 非零元素个数，nnz >= 0 | Host |
| cscColOffsets | 输入 | const void* | CSC 列偏移数组，长度为 cols+1 | Device |
| cscRowInd | 输入 | const void* | CSC 行索引数组，长度为 nnz | Device |
| cscValues | 输入 | const void* | CSC 非零值数组，长度为 nnz | Device |
| cscColOffsetsType | 输入 | aclsparseIndexType_t | 列偏移数组的数据类型 | Host |
| cscRowIndType | 输入 | aclsparseIndexType_t | 行索引数组的数据类型 | Host |
| idxBase | 输入 | aclsparseIndexBase_t | 索引基值 | Host |
| valueType | 输入 | aclDataType | 非零值数据类型 | Host |

#### 约束说明

- 约束同 aclsparseCreateCsc（rows / cols / nnz 非负，spMatDescr 非 nullptr，索引类型合法）
- 构造出的描述符为 const 变体，只能传给接收 const 形参的接口（如 `aclsparseConstCscGet`）；销毁统一用 `aclsparseDestroySpMat`
- 索引类型支持与 aclsparseCreateCsc 一致（`ACL_SPARSE_INDEX_32I` 与 `ACL_SPARSE_INDEX_64I` 及其四种组合均接受）
- 其余约束与 aclsparseCreateCsc 一致

---

### aclsparseCscGet

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCscGet(aclsparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz,
    void **cscColOffsets, void **cscRowInd, void **cscValues, aclsparseIndexType_t *cscColOffsetsType,
    aclsparseIndexType_t *cscRowIndType, aclsparseIndexBase_t *idxBase, aclDataType *valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入 | aclsparseSpMatDescr_t | CSC 稀疏矩阵描述符 | Host |
| rows | 输出 | int64_t* | 矩阵行数 | Host |
| cols | 输出 | int64_t* | 矩阵列数 | Host |
| nnz | 输出 | int64_t* | 非零元素个数 | Host |
| cscColOffsets | 输出 | void** | 列偏移数组指针（与 Create 传入的设备指针一致） | Device |
| cscRowInd | 输出 | void** | 行索引数组指针（与 Create 传入的设备指针一致） | Device |
| cscValues | 输出 | void** | 非零值数组指针（与 Create 传入的设备指针一致） | Device |
| cscColOffsetsType | 输出 | aclsparseIndexType_t* | 列偏移数组的数据类型 | Host |
| cscRowIndType | 输出 | aclsparseIndexType_t* | 行索引数组的数据类型 | Host |
| idxBase | 输出 | aclsparseIndexBase_t* | 索引基值 | Host |
| valueType | 输出 | aclDataType* | 非零值数据类型 | Host |

#### 约束说明

- spMatDescr（输入描述符句柄）不可为 nullptr：句柄为空指针视为非法入参，返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- spMatDescr 必须为稀疏矩阵描述符（误传其他类型描述符句柄，如稠密向量/稠密矩阵描述符）时返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- spMatDescr 须为 `aclsparseCreateCsc` 创建的 CSC 格式描述符；对其他格式（CSR / COO / SLICED_ELL 等）的描述符返回 `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`
- 任一 OUT 指针可为 nullptr，此时仅返回请求的字段（对应字段不回写），与 `aclsparseSpMatGetSize` / `aclsparseCooGet` 措辞一致
- aclsparseCscGet 为纯 host 操作，不启动 NPU kernel，不涉及 stream 同步

---

### aclsparseConstCscGet

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseConstCscGet(aclsparseConstSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
    int64_t *nnz, const void **cscColOffsets, const void **cscRowInd, const void **cscValues,
    aclsparseIndexType_t *cscColOffsetsType, aclsparseIndexType_t *cscRowIndType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入 | aclsparseConstSpMatDescr_t | 只读(const)CSC 稀疏矩阵描述符 | Host |
| rows | 输出 | int64_t* | 矩阵行数 | Host |
| cols | 输出 | int64_t* | 矩阵列数 | Host |
| nnz | 输出 | int64_t* | 非零元素个数 | Host |
| cscColOffsets | 输出 | const void** | 列偏移数组指针 | Device |
| cscRowInd | 输出 | const void** | 行索引数组指针 | Device |
| cscValues | 输出 | const void** | 非零值数组指针 | Device |
| cscColOffsetsType | 输出 | aclsparseIndexType_t* | 列偏移数组的数据类型 | Host |
| cscRowIndType | 输出 | aclsparseIndexType_t* | 行索引数组的数据类型 | Host |
| idxBase | 输出 | aclsparseIndexBase_t* | 索引基值 | Host |
| valueType | 输出 | aclDataType* | 非零值数据类型 | Host |

#### 约束说明

- 约束同 aclsparseCscGet（spMatDescr 非 nullptr 且须为稀疏矩阵描述符句柄，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；非 CSC 格式返回 `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`）
- 任一 OUT 指针可为 nullptr，此时仅返回请求的字段
- aclsparseConstCscGet 为纯 host 操作，不启动 NPU kernel，不涉及 stream 同步

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

int aclsparseCscGetTest(AclContext &ctx)
{
    (void)ctx.Stream();  // accessor 不使用 handle / stream，此处仅占位

    // 1. 准备 Host 端 CSC 数据：3 个非零元的 3x3 矩阵（按列压缩）
    //    [1.0  0.0  0.0]
    //    [0.0  0.0  2.0]
    //    [0.0  3.0  0.0]
    //    列 0 含 (0,0)=1.0，列 1 含 (2,1)=3.0，列 2 含 (1,2)=2.0
    int64_t rows = 3;
    int64_t cols = 3;
    int64_t nnz = 3;

    std::vector<int32_t> hCscColOffsets = {0, 1, 2, 3};   // 长 cols+1 = 4
    std::vector<int32_t> hCscRowInd = {0, 2, 1};          // 长 nnz = 3
    std::vector<float> hCscValues = {1.0f, 3.0f, 2.0f};   // 长 nnz = 3

    // 2. 拷贝 colOffsets/rowInd/values 到 Device
    DevicePtr dCscColOffsets(nullptr, aclrtFree);
    DevicePtr dCscRowInd(nullptr, aclrtFree);
    DevicePtr dCscValues(nullptr, aclrtFree);
    auto aclRet = AllocAndCopyDevice(dCscColOffsets, hCscColOffsets.data(), (cols + 1) * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dCscRowInd, hCscRowInd.data(), nnz * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dCscValues, hCscValues.data(), nnz * sizeof(float));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 3. aclsparseCreateCsc：将维度与设备指针存入描述符
    aclsparseSpMatDescr_t matDescr = nullptr;
    auto sparseRet = aclsparseCreateCsc(&matDescr, rows, cols, nnz,
        dCscColOffsets.get(), dCscRowInd.get(), dCscValues.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCreateCsc failed: %d\n", sparseRet); return sparseRet);

    // 4. aclsparseCscGet：读回描述符全部字段
    int64_t outRows = 0, outCols = 0, outNnz = 0;
    void *outColOffsets = nullptr, *outRowInd = nullptr, *outValues = nullptr;
    aclsparseIndexType_t outColOffsetsType{};
    aclsparseIndexType_t outRowIndType{};
    aclsparseIndexBase_t outIdxBase{};
    aclDataType outValueType{};
    sparseRet = aclsparseCscGet(matDescr, &outRows, &outCols, &outNnz,
        &outColOffsets, &outRowInd, &outValues, &outColOffsetsType, &outRowIndType, &outIdxBase, &outValueType);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCscGet failed: %d\n", sparseRet); aclsparseDestroySpMat(matDescr); return sparseRet);

    // 5. accessor 语义断言：读回字段与 Create 传入一致
    CHECK_RET(outRows == rows && outCols == cols && outNnz == nnz,
        LOG_PRINT("size mismatch: rows=%lld/%lld cols=%lld/%lld nnz=%lld/%lld\n",
            static_cast<long long>(outRows), static_cast<long long>(rows),
            static_cast<long long>(outCols), static_cast<long long>(cols),
            static_cast<long long>(outNnz), static_cast<long long>(nnz));
        aclsparseDestroySpMat(matDescr); return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    CHECK_RET(outColOffsetsType == ACL_SPARSE_INDEX_32I && outRowIndType == ACL_SPARSE_INDEX_32I &&
              outIdxBase == ACL_SPARSE_INDEX_BASE_ZERO && outValueType == ACL_FLOAT,
        LOG_PRINT("attr mismatch\n"); aclsparseDestroySpMat(matDescr); return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    // 读回的设备指针应与 Create 传入的逐字一致（accessor 不拷贝设备数据）
    CHECK_RET(outColOffsets == dCscColOffsets.get() && outRowInd == dCscRowInd.get() &&
              outValues == dCscValues.get(),
        LOG_PRINT("device pointer mismatch\n"); aclsparseDestroySpMat(matDescr);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    LOG_PRINT("aclsparseCreateCsc / aclsparseCscGet succeeded.\n");
    LOG_PRINT("read back: rows=%lld, cols=%lld, nnz=%lld\n",
        static_cast<long long>(outRows), static_cast<long long>(outCols), static_cast<long long>(outNnz));
    LOG_PRINT("  attrs match: colOffsetsType=ACL_SPARSE_INDEX_32I, rowIndType=ACL_SPARSE_INDEX_32I, idxBase=ACL_SPARSE_INDEX_BASE_ZERO, valueType=ACL_FLOAT\n");
    LOG_PRINT("  devPtr match: colOffsets=%d, rowInd=%d, val=%d\n",
        outColOffsets == dCscColOffsets.get() ? 1 : 0,
        outRowInd == dCscRowInd.get() ? 1 : 0,
        outValues == dCscValues.get() ? 1 : 0);

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

    ret = aclsparseCscGetTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCscGetTest failed: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
aclsparseCreateCsc / aclsparseCscGet succeeded.
read back: rows=3, cols=3, nnz=3
  attrs match: colOffsetsType=ACL_SPARSE_INDEX_32I, rowIndType=ACL_SPARSE_INDEX_32I, idxBase=ACL_SPARSE_INDEX_BASE_ZERO, valueType=ACL_FLOAT
  devPtr match: colOffsets=1, rowInd=1, val=1
aclsparseDestroySpMat done.
```
