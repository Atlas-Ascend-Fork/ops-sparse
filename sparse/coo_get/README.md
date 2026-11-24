# CooGet算子

## 算子概述

coo_get 算子用于从 COO（Coordinate）格式稀疏矩阵描述符中读回全部字段，对标 cuSPARSE 的 `cusparseCreateCoo` / `cusparseCooGet` / `cusparseCreateConstCoo` / `cusparseConstCooGet`，属于 SparseAccessor（描述符访问器）类接口。

CooGet 为**纯 host 访问器**：描述符的构造与字段读取均在 Host 侧完成，**不启动 NPU kernel、不涉及 stream 同步、无 GPU 计算**。`aclsparseCreateCoo` 将调用方传入的行/列索引与值的设备指针连同维度、索引类型、索引基、值类型一并存入描述符；`aclsparseCooGet` 则从描述符中把这些字段原样读回，读回的设备指针与 Create 时传入的指针逐字一致（accessor 不拷贝、不移动设备数据）。语义与 cuSPARSE 完全对齐，便于 cuSPARSE 用户无缝迁移。

COO 格式以三个等长数组描述稀疏矩阵：行索引 `cooRowInd`、列索引 `cooColInd` 与非零值 `cooValues`，数组长度均为 `nnz`。`cooIdxType` 同时约束行/列索引的数据类型。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseCreateCoo | 创建 COO 格式稀疏矩阵描述符，存入维度与设备指针 |
| aclsparseCreateConstCoo | 创建只读(const)COO 格式稀疏矩阵描述符 |
| aclsparseCooGet | 从 COO 描述符读回全部字段（维度 / 设备指针 / 索引类型 / 索引基 / 值类型） |
| aclsparseConstCooGet | 从只读(const)COO 描述符读回全部字段 |

## 算子执行接口

### aclsparseCreateCoo

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCreateCoo(aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *cooRowInd, void *cooColInd, void *cooValues, aclsparseIndexType_t cooIdxType,
    aclsparseIndexBase_t idxBase, aclDataType valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入/输出 | aclsparseSpMatDescr_t* | COO 稀疏矩阵描述符（输出句柄），调用前无需初始化 | Host |
| rows | 输入 | int64_t | 矩阵行数，rows >= 0 | Host |
| cols | 输入 | int64_t | 矩阵列数，cols >= 0 | Host |
| nnz | 输入 | int64_t | 非零元素个数，nnz >= 0 | Host |
| cooRowInd | 输入 | void* | COO 行索引数组，长度为 nnz | Device |
| cooColInd | 输入 | void* | COO 列索引数组，长度为 nnz | Device |
| cooValues | 输入 | void* | COO 非零值数组，长度为 nnz | Device |
| cooIdxType | 输入 | aclsparseIndexType_t | 行/列索引数据类型，同时约束 cooRowInd 与 cooColInd | Host |
| idxBase | 输入 | aclsparseIndexBase_t | 索引基值（ACL_SPARSE_INDEX_BASE_ZERO / ACL_SPARSE_INDEX_BASE_ONE） | Host |
| valueType | 输入 | aclDataType | 非零值数据类型 | Host |

#### 约束说明

- rows >= 0，cols >= 0，nnz >= 0，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- spMatDescr（输出句柄）不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- cooIdxType 同时约束行/列索引类型；**支持 `ACL_SPARSE_INDEX_32I` 与 `ACL_SPARSE_INDEX_64I`**（accessor 层无 NPU 计算，与 cuSPARSE 一致）。注意：这与 CSR 不同——CSR 当前 SpMV / SpMM 计算路径仅支持 `ACL_SPARSE_INDEX_32I`；COO accessor 两种索引类型均接受
- aclsparseCreateCoo 为纯 host 操作，不启动 NPU kernel，不涉及 stream 同步
- const 变体（aclsparseCreateConstCoo）构造出的描述符只能传给接收 const 形参的接口；销毁统一用 `aclsparseDestroySpMat`（接受 const 变体，const 与非 const 描述符均可销毁）

---

### aclsparseCreateConstCoo

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCreateConstCoo(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols,
    int64_t nnz, const void *cooRowInd, const void *cooColInd, const void *cooValues,
    aclsparseIndexType_t cooIdxType, aclsparseIndexBase_t idxBase, aclDataType valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入/输出 | aclsparseConstSpMatDescr_t* | 只读(const)COO 稀疏矩阵描述符（输出句柄） | Host |
| rows | 输入 | int64_t | 矩阵行数，rows >= 0 | Host |
| cols | 输入 | int64_t | 矩阵列数，cols >= 0 | Host |
| nnz | 输入 | int64_t | 非零元素个数，nnz >= 0 | Host |
| cooRowInd | 输入 | const void* | COO 行索引数组，长度为 nnz | Device |
| cooColInd | 输入 | const void* | COO 列索引数组，长度为 nnz | Device |
| cooValues | 输入 | const void* | COO 非零值数组，长度为 nnz | Device |
| cooIdxType | 输入 | aclsparseIndexType_t | 行/列索引数据类型，同时约束 cooRowInd 与 cooColInd | Host |
| idxBase | 输入 | aclsparseIndexBase_t | 索引基值 | Host |
| valueType | 输入 | aclDataType | 非零值数据类型 | Host |

#### 约束说明

- 约束同 aclsparseCreateCoo（rows / cols / nnz 非负，spMatDescr 非 nullptr）
- 构造出的描述符为 const 变体，只能传给接收 const 形参的接口（如 `aclsparseConstCooGet`）；销毁统一用 `aclsparseDestroySpMat`
- 索引类型支持与 aclsparseCreateCoo 一致（`ACL_SPARSE_INDEX_32I` 与 `ACL_SPARSE_INDEX_64I` 均接受）
- 其余约束与 aclsparseCreateCoo 一致

---

### aclsparseCooGet

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCooGet(aclsparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz,
    void **cooRowInd, void **cooColInd, void **cooValues, aclsparseIndexType_t *cooIdxType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入 | aclsparseSpMatDescr_t | COO 稀疏矩阵描述符 | Host |
| rows | 输出 | int64_t* | 矩阵行数 | Host |
| cols | 输出 | int64_t* | 矩阵列数 | Host |
| nnz | 输出 | int64_t* | 非零元素个数 | Host |
| cooRowInd | 输出 | void** | 行索引数组指针（与 Create 传入的设备指针一致） | Device |
| cooColInd | 输出 | void** | 列索引数组指针（与 Create 传入的设备指针一致） | Device |
| cooValues | 输出 | void** | 非零值数组指针（与 Create 传入的设备指针一致） | Device |
| cooIdxType | 输出 | aclsparseIndexType_t* | 行/列索引数据类型 | Host |
| idxBase | 输出 | aclsparseIndexBase_t* | 索引基值 | Host |
| valueType | 输出 | aclDataType* | 非零值数据类型 | Host |

#### 约束说明

- spMatDescr（输入描述符）不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- 对非 COO 格式的描述符返回 `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`
- 任一 OUT 指针可为 nullptr，此时仅返回请求的字段（对应字段不回写），与 `aclsparseSpMatGetSize` / `aclsparseSpVecGet` 措辞一致
- aclsparseCooGet 为纯 host 操作，不启动 NPU kernel，不涉及 stream 同步

---

### aclsparseConstCooGet

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseConstCooGet(aclsparseConstSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
    int64_t *nnz, const void **cooRowInd, const void **cooColInd, const void **cooValues,
    aclsparseIndexType_t *cooIdxType, aclsparseIndexBase_t *idxBase, aclDataType *valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入 | aclsparseConstSpMatDescr_t | 只读(const)COO 稀疏矩阵描述符 | Host |
| rows | 输出 | int64_t* | 矩阵行数 | Host |
| cols | 输出 | int64_t* | 矩阵列数 | Host |
| nnz | 输出 | int64_t* | 非零元素个数 | Host |
| cooRowInd | 输出 | const void** | 行索引数组指针 | Device |
| cooColInd | 输出 | const void** | 列索引数组指针 | Device |
| cooValues | 输出 | const void** | 非零值数组指针 | Device |
| cooIdxType | 输出 | aclsparseIndexType_t* | 行/列索引数据类型 | Host |
| idxBase | 输出 | aclsparseIndexBase_t* | 索引基值 | Host |
| valueType | 输出 | aclDataType* | 非零值数据类型 | Host |

#### 约束说明

- 约束同 aclsparseCooGet（spMatDescr 非 nullptr，非 COO 格式返回 `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`）
- 任一 OUT 指针可为 nullptr，此时仅返回请求的字段
- aclsparseConstCooGet 为纯 host 操作，不启动 NPU kernel，不涉及 stream 同步

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

int aclsparseCooGetTest(AclContext &ctx)
{
    (void)ctx.Stream();  // accessor 不使用 handle / stream，此处仅占位

    // 1. 准备 Host 端 COO 数据：3 个非零元的 3x3 矩阵
    //    [1.0  0.0  0.0]
    //    [0.0  0.0  2.0]
    //    [0.0  3.0  0.0]
    int64_t rows = 3;
    int64_t cols = 3;
    int64_t nnz = 3;

    std::vector<int32_t> hCooRowInd = {0, 1, 2};
    std::vector<int32_t> hCooColInd = {0, 2, 1};
    std::vector<float> hCooValues = {1.0f, 2.0f, 3.0f};

    // 2. 拷贝 row/col/values 到 Device
    DevicePtr dCooRowInd(nullptr, aclrtFree);
    DevicePtr dCooColInd(nullptr, aclrtFree);
    DevicePtr dCooValues(nullptr, aclrtFree);
    auto aclRet = AllocAndCopyDevice(dCooRowInd, hCooRowInd.data(), nnz * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dCooColInd, hCooColInd.data(), nnz * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dCooValues, hCooValues.data(), nnz * sizeof(float));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 3. aclsparseCreateCoo：将维度与设备指针存入描述符
    aclsparseSpMatDescr_t matDescr = nullptr;
    auto sparseRet = aclsparseCreateCoo(&matDescr, rows, cols, nnz,
        dCooRowInd.get(), dCooColInd.get(), dCooValues.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCreateCoo failed: %d\n", sparseRet); return sparseRet);

    // 4. aclsparseCooGet：读回描述符全部字段
    int64_t outRows = 0, outCols = 0, outNnz = 0;
    void *outRowInd = nullptr, *outColInd = nullptr, *outValues = nullptr;
    aclsparseIndexType_t outIdxType{};
    aclsparseIndexBase_t outIdxBase{};
    aclDataType outValueType{};
    sparseRet = aclsparseCooGet(matDescr, &outRows, &outCols, &outNnz,
        &outRowInd, &outColInd, &outValues, &outIdxType, &outIdxBase, &outValueType);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCooGet failed: %d\n", sparseRet); aclsparseDestroySpMat(matDescr); return sparseRet);

    // 5. accessor 语义断言：读回字段与 Create 传入一致
    CHECK_RET(outRows == rows && outCols == cols && outNnz == nnz,
        LOG_PRINT("size mismatch: rows=%lld/%lld cols=%lld/%lld nnz=%lld/%lld\n",
            static_cast<long long>(outRows), static_cast<long long>(rows),
            static_cast<long long>(outCols), static_cast<long long>(cols),
            static_cast<long long>(outNnz), static_cast<long long>(nnz));
        aclsparseDestroySpMat(matDescr); return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    CHECK_RET(outIdxType == ACL_SPARSE_INDEX_32I && outIdxBase == ACL_SPARSE_INDEX_BASE_ZERO &&
              outValueType == ACL_FLOAT,
        LOG_PRINT("attr mismatch\n"); aclsparseDestroySpMat(matDescr); return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    // 读回的设备指针应与 Create 传入的逐字一致（accessor 不拷贝设备数据）
    CHECK_RET(outRowInd == dCooRowInd.get() && outColInd == dCooColInd.get() &&
              outValues == dCooValues.get(),
        LOG_PRINT("device pointer mismatch\n"); aclsparseDestroySpMat(matDescr);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    LOG_PRINT("aclsparseCreateCoo / aclsparseCooGet succeeded.\n");
    LOG_PRINT("read back: rows=%lld, cols=%lld, nnz=%lld\n",
        static_cast<long long>(outRows), static_cast<long long>(outCols), static_cast<long long>(outNnz));
    LOG_PRINT("  attrs match: idxType=ACL_SPARSE_INDEX_32I, idxBase=ACL_SPARSE_INDEX_BASE_ZERO, valueType=ACL_FLOAT\n");
    LOG_PRINT("  devPtr match: row=%d, col=%d, val=%d\n",
        outRowInd == dCooRowInd.get() ? 1 : 0,
        outColInd == dCooColInd.get() ? 1 : 0,
        outValues == dCooValues.get() ? 1 : 0);

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

    ret = aclsparseCooGetTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCooGetTest failed: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
aclsparseCreateCoo / aclsparseCooGet succeeded.
read back: rows=3, cols=3, nnz=3
  attrs match: idxType=ACL_SPARSE_INDEX_32I, idxBase=ACL_SPARSE_INDEX_BASE_ZERO, valueType=ACL_FLOAT
  devPtr match: row=1, col=1, val=1
aclsparseDestroySpMat done.
```
