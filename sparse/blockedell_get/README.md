# BlockedEllGet算子

## 算子概述

blockedell_get 算子用于从 Blocked-ELL（Blocked Ellpack）格式稀疏矩阵描述符中读回全部字段，对标 cuSPARSE 的 `cusparseCreateBlockedEll` / `cusparseBlockedEllGet` / `cusparseConstBlockedEllGet`，属于 SparseAccessor（描述符访问器）类接口。

BlockedEllGet 为**纯 host 访问器**：描述符的构造与字段读取均在 Host 侧完成，**不启动 NPU kernel、不涉及 stream 同步、无 GPU 计算**。`aclsparseCreateBlockedEll` 将调用方传入的列块索引与值的设备指针连同维度、块大小、索引类型、索引基、值类型一并存入描述符；`aclsparseBlockedEllGet` 则从描述符中把这些字段原样读回，读回的设备指针为描述符**当前绑定**的指针（初始即 Create 时传入的值；经 `aclsparseSpMatSetValues` 重绑定 values 后，Get 读回新指针，与 cuSPARSE `cusparseSpMatSetValues` + Get 的语义一致；accessor 不拷贝、不移动设备数据）。接口语义与 cuSPARSE 对齐，便于 cuSPARSE 用户迁移（`ellValue` 物理排布差异见下文格式说明）。

Blocked-ELL 格式以固定大小 `ellBlockSize × ellBlockSize` 的块为基本单位描述稀疏矩阵。记 `b = ellBlockSize`，块网格为 `ceil(rows/b)` 个块行 × `ceil(ellCols/b)` 个列块槽（slot）：

- `ellColInd` 为列块索引数组，按块网格行主序排列（`task = 块行号 * ceil(ellCols/b) + 槽号`，与 `ellValue` 的 task 一一对应），长度为 `ceil(rows/b) * ceil(ellCols/b)`，取值范围 `[idxBase, ceil(cols/b)-1+idxBase]`，空槽用 `-1` 表示；
- `ellValue` 为值数组，按 task 连续存放，每个 task 占据连续的 `b × b` 个标量（一个完整块），**块内列主序**：`valuePos = task * b^2 + innerCol * b + innerRow`（`innerRow` / `innerCol` 为块内行 / 列偏移）；数组总容量为 `ceil(rows/b) * ceil(ellCols/b) * b^2`，尾块不足 `b × b` 处补零（仅当 rows 与 ellCols 均被 b 整除时，总容量才等于 `rows * ellCols`）。

注意：aclsparse 的 `ellValue` 物理排布与 cuSPARSE 不同——cuSPARSE 的 `ellValue` 按 `rows × ellCols` 跨步（strided）行主序存放且块内行主序；aclsparse 存放**连续块**且**块内列主序**（与仓内 DenseToSparse 算子消费 / 产出的 BELL 排布一致，见 densetosparse 算子）。两者 `ellColInd` 的槽序一致；从 cuSPARSE 迁移 BELL 数据时需按上述公式重排 `ellValue`。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseCreateBlockedEll | 创建 Blocked-ELL 格式稀疏矩阵描述符，存入维度与设备指针 |
| aclsparseBlockedEllGet | 从 Blocked-ELL 描述符读回全部字段（维度 / 块大小 / 设备指针 / 索引类型 / 索引基 / 值类型） |
| aclsparseConstBlockedEllGet | 从只读(const)Blocked-ELL 描述符读回全部字段 |

## 算子执行接口

### aclsparseCreateBlockedEll

#### 产品支持情况

`aclsparseCreateBlockedEll` 为架构无关的纯 host 接口（实现位于公共层，随 libops_sparse 在全平台构建）：

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（BELL 描述符可作为 DenseToSparse 的 BELL 输入，见 densetosparse 算子）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（同上）

> 注意：`aclsparseBlockedEllGet` / `aclsparseConstBlockedEllGet` 两个访问器仅提供 arch35（Ascend 950PR / Ascend 950DT）实现，A2 / A3 构建不含其符号——A2 / A3 平台可创建并使用 BELL 描述符，但无法通过 Get 接口回读字段（见下两节支持矩阵）。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCreateBlockedEll(
    aclsparseSpMatDescr_t *descr, int64_t rows, int64_t cols,
    int64_t ellBlockSize, int64_t ellCols, void *ellColInd, void *ellValue,
    aclsparseIndexType_t indexType, aclsparseIndexBase_t indexBase,
    aclDataType valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| descr | 输入/输出 | aclsparseSpMatDescr_t* | Blocked-ELL 稀疏矩阵描述符（输出句柄），调用前无需初始化 | Host |
| rows | 输入 | int64_t | 矩阵行数，rows >= 0 | Host |
| cols | 输入 | int64_t | 矩阵列数，cols >= 0 | Host |
| ellBlockSize | 输入 | int64_t | ELL 块大小（正方形块边长），ellBlockSize > 0 | Host |
| ellCols | 输入 | int64_t | Blocked-ELL 格式的实际列数（ellValue 的列数），ellCols >= 0 | Host |
| ellColInd | 输入 | void* | 列块索引数组，长度为 ceil(rows/ellBlockSize) * ceil(ellCols/ellBlockSize) | Device |
| ellValue | 输入 | void* | 非零值数组，按块网格行主序连续存放 b×b 块（块内列主序），总容量为 ceil(rows/ellBlockSize) * ceil(ellCols/ellBlockSize) * ellBlockSize^2 | Device |
| indexType | 输入 | aclsparseIndexType_t | ellColInd 的数据类型 | Host |
| indexBase | 输入 | aclsparseIndexBase_t | 索引基值（ACL_SPARSE_INDEX_BASE_ZERO / ACL_SPARSE_INDEX_BASE_ONE） | Host |
| valueType | 输入 | aclDataType | 非零值数据类型 | Host |

#### 约束说明

- rows >= 0，cols >= 0，ellBlockSize > 0，ellCols >= 0，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- descr（输出句柄）不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- indexType 仅支持 `ACL_SPARSE_INDEX_32I` 与 `ACL_SPARSE_INDEX_64I`（accessor 层无 NPU 计算，与 cuSPARSE 一致）
- ellCols 的块槽数上界为 `ceil(cols/ellBlockSize)`，超出返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；有列尾块时 ellCols 可超过 cols（尾块 slot 存在并填零）
- 容量按补齐后的块网格 `ceil(rows/ellBlockSize) * ceil(ellCols/ellBlockSize) * ellBlockSize^2` 计，int64 溢出返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- 索引数组 / 值数组非空（容量 > 0）时对应指针不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- aclsparseCreateBlockedEll 为纯 host 操作，不启动 NPU kernel，不涉及 stream 同步

---

### aclsparseBlockedEllGet

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseBlockedEllGet(
    aclsparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
    int64_t *ellBlockSize, int64_t *ellCols, void **ellColInd, void **ellValue,
    aclsparseIndexType_t *ellIdxType, aclsparseIndexBase_t *idxBase,
    aclDataType *valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入 | aclsparseSpMatDescr_t | Blocked-ELL 稀疏矩阵描述符 | Host |
| rows | 输出 | int64_t* | 矩阵行数 | Host |
| cols | 输出 | int64_t* | 矩阵列数 | Host |
| ellBlockSize | 输出 | int64_t* | ELL 块大小 | Host |
| ellCols | 输出 | int64_t* | Blocked-ELL 格式的实际列数 | Host |
| ellColInd | 输出 | void** | 列块索引数组指针（描述符当前绑定的设备指针，初始即 Create 传入值） | Device |
| ellValue | 输出 | void** | 非零值数组指针（描述符当前绑定的设备指针，初始即 Create 传入值；经 aclsparseSpMatSetValues 重绑定后读回新指针，与 cuSPARSE 语义一致） | Device |
| ellIdxType | 输出 | aclsparseIndexType_t* | ellColInd 的数据类型 | Host |
| idxBase | 输出 | aclsparseIndexBase_t* | 索引基值 | Host |
| valueType | 输出 | aclDataType* | 非零值数据类型 | Host |

#### 约束说明

- spMatDescr 不可为 nullptr 且签名须合法（错误类型句柄返回 `ACL_SPARSE_STATUS_INVALID_VALUE`）
- 对非 Blocked-ELL 格式的描述符返回 `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`
- 任一 OUT 指针可为 nullptr，此时仅返回请求的字段（对应字段不回写），与 `aclsparseSpMatGetSize` / `aclsparseCooGet` 措辞一致
- aclsparseBlockedEllGet 为纯 host 操作，不启动 NPU kernel，不涉及 stream 同步

---

### aclsparseConstBlockedEllGet

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseConstBlockedEllGet(
    aclsparseConstSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
    int64_t *ellBlockSize, int64_t *ellCols, const void **ellColInd,
    const void **ellValue, aclsparseIndexType_t *ellIdxType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| spMatDescr | 输入 | aclsparseConstSpMatDescr_t | 只读(const)Blocked-ELL 稀疏矩阵描述符 | Host |
| rows | 输出 | int64_t* | 矩阵行数 | Host |
| cols | 输出 | int64_t* | 矩阵列数 | Host |
| ellBlockSize | 输出 | int64_t* | ELL 块大小 | Host |
| ellCols | 输出 | int64_t* | Blocked-ELL 格式的实际列数 | Host |
| ellColInd | 输出 | const void** | 列块索引数组指针 | Device |
| ellValue | 输出 | const void** | 非零值数组指针 | Device |
| ellIdxType | 输出 | aclsparseIndexType_t* | ellColInd 的数据类型 | Host |
| idxBase | 输出 | aclsparseIndexBase_t* | 索引基值 | Host |
| valueType | 输出 | aclDataType* | 非零值数据类型 | Host |

#### 约束说明

- 约束同 aclsparseBlockedEllGet（spMatDescr 不可为 nullptr 且签名须合法，非 Blocked-ELL 格式返回 `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`）
- 任一 OUT 指针可为 nullptr，此时仅返回请求的字段
- aclsparseConstBlockedEllGet 为纯 host 操作，不启动 NPU kernel，不涉及 stream 同步

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

int aclsparseBlockedEllGetTest(AclContext &ctx)
{
    (void)ctx.Stream();  // accessor 不使用 handle / stream，此处仅占位

    // 1. 准备 Host 端 Blocked-ELL 数据：4x4 矩阵按 bs=2 分块，两个非空列块
    //    [1 2 | 0 0]
    //    [3 4 | 0 0]
    //    [-----|----]
    //    [0 0 | 5 6]
    //    [0 0 | 7 8]
    //    块行 0 的非零块在列块 0（[1 2; 3 4]），块行 1 的非零块在列块 1（[5 6; 7 8]）
    int64_t rows = 4;
    int64_t cols = 4;
    int64_t ellBlockSize = 2;
    int64_t ellCols = 4;
    // 块行数 ceil(4/2)=2，每块行槽数 ceil(4/2)=2，索引数 = 2*2 = 4
    // ellColInd 按块网格行主序（task = 块行号*槽数 + 槽号）：
    //   task0=(块行0,槽0)->列块0；task1=(块行0,槽1)->空(-1)；
    //   task2=(块行1,槽0)->空(-1)；task3=(块行1,槽1)->列块1
    std::vector<int32_t> hEllColInd = {0, -1, -1, 1};
    // ellValue 按 task 连续存放 2x2 块、块内列主序（valuePos = task*4 + innerCol*2 + innerRow）：
    //   task0（[1 2; 3 4]）块内列主序 = {1,3, 2,4}；task1 / task2 空块补零；
    //   task3（[5 6; 7 8]）块内列主序 = {5,7, 6,8}
    std::vector<float> hEllValue = {1.0f, 3.0f, 2.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 7.0f, 6.0f, 8.0f};

    // 2. 拷贝 ellColInd / ellValue 到 Device
    DevicePtr dEllColInd(nullptr, aclrtFree);
    DevicePtr dEllValue(nullptr, aclrtFree);
    auto aclRet = AllocAndCopyDevice(dEllColInd, hEllColInd.data(), hEllColInd.size() * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dEllValue, hEllValue.data(), hEllValue.size() * sizeof(float));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 3. aclsparseCreateBlockedEll：将维度与设备指针存入描述符
    aclsparseSpMatDescr_t matDescr = nullptr;
    auto sparseRet = aclsparseCreateBlockedEll(&matDescr, rows, cols, ellBlockSize, ellCols,
        dEllColInd.get(), dEllValue.get(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseCreateBlockedEll failed: %d\n", sparseRet); return sparseRet);

    // 4. aclsparseBlockedEllGet：读回描述符全部字段
    int64_t outRows = 0, outCols = 0, outBlockSize = 0, outEllCols = 0;
    void *outColInd = nullptr;
    void *outValue = nullptr;
    aclsparseIndexType_t outIdxType{};
    aclsparseIndexBase_t outIdxBase{};
    aclDataType outValueType{};
    sparseRet = aclsparseBlockedEllGet(matDescr, &outRows, &outCols, &outBlockSize, &outEllCols,
        &outColInd, &outValue, &outIdxType, &outIdxBase, &outValueType);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
        LOG_PRINT("aclsparseBlockedEllGet failed: %d\n", sparseRet); aclsparseDestroySpMat(matDescr); return sparseRet);

    // 5. accessor 语义断言：读回字段与 Create 传入一致
    CHECK_RET(outRows == rows && outCols == cols && outBlockSize == ellBlockSize &&
              outEllCols == ellCols,
        LOG_PRINT("size mismatch: rows=%lld/%lld cols=%lld/%lld bs=%lld/%lld ellCols=%lld/%lld\n",
            static_cast<long long>(outRows), static_cast<long long>(rows),
            static_cast<long long>(outCols), static_cast<long long>(cols),
            static_cast<long long>(outBlockSize), static_cast<long long>(ellBlockSize),
            static_cast<long long>(outEllCols), static_cast<long long>(ellCols));
        aclsparseDestroySpMat(matDescr); return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    CHECK_RET(outIdxType == ACL_SPARSE_INDEX_32I && outIdxBase == ACL_SPARSE_INDEX_BASE_ZERO &&
              outValueType == ACL_FLOAT,
        LOG_PRINT("attr mismatch\n"); aclsparseDestroySpMat(matDescr); return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    // 读回的设备指针应与描述符当前绑定的指针一致（本例未做 SetValues 重绑定，即 Create 传入值；accessor 不拷贝设备数据）
    CHECK_RET(outColInd == dEllColInd.get() && outValue == dEllValue.get(),
        LOG_PRINT("device pointer mismatch\n"); aclsparseDestroySpMat(matDescr);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    LOG_PRINT("aclsparseCreateBlockedEll / aclsparseBlockedEllGet succeeded.\n");
    LOG_PRINT("read back: rows=%lld, cols=%lld, ellBlockSize=%lld, ellCols=%lld\n",
        static_cast<long long>(outRows), static_cast<long long>(outCols),
        static_cast<long long>(outBlockSize), static_cast<long long>(outEllCols));
    LOG_PRINT("  attrs match: idxType=ACL_SPARSE_INDEX_32I, idxBase=ACL_SPARSE_INDEX_BASE_ZERO, valueType=ACL_FLOAT\n");
    LOG_PRINT("  devPtr match: colInd=%d, val=%d\n",
        outColInd == dEllColInd.get() ? 1 : 0,
        outValue == dEllValue.get() ? 1 : 0);

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

    ret = aclsparseBlockedEllGetTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseBlockedEllGetTest failed: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
aclsparseCreateBlockedEll / aclsparseBlockedEllGet succeeded.
read back: rows=4, cols=4, ellBlockSize=2, ellCols=4
  attrs match: idxType=ACL_SPARSE_INDEX_32I, idxBase=ACL_SPARSE_INDEX_BASE_ZERO, valueType=ACL_FLOAT
  devPtr match: colInd=1, val=1
aclsparseDestroySpMat done.
```
