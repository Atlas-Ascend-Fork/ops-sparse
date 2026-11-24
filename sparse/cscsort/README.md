# CSCSORT算子

## 算子概述

cscsort算子用于对CSC（Compressed Sparse Column，压缩稀疏列）格式稀疏矩阵的每一列执行稳定排序。算子原地重排行索引数组`cscRowInd`，并使用相同的排列重排数组`P`；列偏移数组`cscColPtr`保持不变。

对于第`col`列，排序区间和输出关系如下：

```text
begin = cscColPtr[col] - indexBase
end   = cscColPtr[col + 1] - indexBase
cscRowInd_out[begin:end] = stable_sort(cscRowInd_in[begin:end])
P_out[begin:end] = P_in[stable_permutation(begin:end)]
```

当调用方将`P`初始化为`0, 1, ..., nnz - 1`时，排序后的`P[i]`表示输出位置`i`对应的原始非零元素下标，可据此同步重排稀疏矩阵的值数组。

cscsort是csrsort的列向对偶：csrsort按CSR行（`csrRowPtr`划分）排序`csrColInd`，cscsort按CSC列（`cscColPtr`划分）排序`cscRowInd`，两者算法结构完全对称。本算子对齐cuSPARSE Legacy API `cusparseXcscsort` / `cusparseXcscsort_bufferSizeExt`。

算子采用两步调用流程：先调用`aclsparseXcscsort_bufferSizeExt`查询workspace大小并分配Device内存，再调用`aclsparseXcscsort`执行排序。排序接口异步下发到handle绑定的stream，读取结果前需要同步该stream。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseXcscsort_bufferSizeExt | 查询排序所需的Device workspace大小 |
| aclsparseXcscsort | 对CSC矩阵每列的行索引执行原地稳定升序排序，并同步重排`P` |

## 算子执行接口

### aclsparseXcscsort_bufferSizeExt

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseXcscsort_bufferSizeExt(
    aclsparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const int *cscColPtr,
    const int *cscRowInd,
    size_t *pBufferSizeInBytes)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse库上下文句柄 | Host |
| m | 输入 | int | 稀疏矩阵行数 | Host |
| n | 输入 | int | 稀疏矩阵列数 | Host |
| nnz | 输入 | int | 稀疏矩阵非零元素个数 | Host |
| cscColPtr | 输入 | const int*（INT32） | CSC列偏移数组，长度为`n + 1` | Device |
| cscRowInd | 输入 | const int*（INT32） | CSC行索引数组，长度为`nnz` | Device |
| pBufferSizeInBytes | 输出 | size_t* | 返回workspace大小，单位为字节 | Host |

#### 约束说明

- `handle`不可为nullptr，否则返回`ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`。
- `m >= 0`、`n >= 0`且`nnz >= 0`。
- 当`m == 0`或`n == 0`时，`nnz`必须为0。
- 当`n > 0`时，`cscColPtr`不可为nullptr。
- 当`nnz > 0`时，`cscRowInd`不可为nullptr。
- `pBufferSizeInBytes`不可为nullptr。
- 当`nnz == 0`时，接口返回成功并将`*pBufferSizeInBytes`置为0；否则返回`2 * nnz * sizeof(int32_t)`。
- 本接口不引入Device同步。

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSC | ✅ | 输入使用CSC列偏移和行索引数组 |
| CSR | ❌ | 不支持CSR格式 |
| COO | ❌ | 不支持COO格式 |

### aclsparseXcscsort

#### 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseXcscsort(
    aclsparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const aclsparseMatDescr_t descrA,
    const int *cscColPtr,
    int *cscRowInd,
    int *P,
    void *pBuffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse库上下文句柄，提供执行stream | Host |
| m | 输入 | int | 稀疏矩阵行数 | Host |
| n | 输入 | int | 稀疏矩阵列数 | Host |
| nnz | 输入 | int | 稀疏矩阵非零元素个数 | Host |
| descrA | 输入 | const aclsparseMatDescr_t | Legacy矩阵描述符；使用其索引基址，其他属性不参与排序 | Host |
| cscColPtr | 输入 | const int*（INT32） | CSC列偏移数组，长度为`n + 1`，排序期间保持不变 | Device |
| cscRowInd | 输入/输出 | int*（INT32） | CSC行索引数组，长度为`nnz`；输出时每列内部稳定升序 | Device |
| P | 输入/输出 | int*（INT32） | 排列数组，长度为`nnz`；调用方须先初始化为`0, 1, ..., nnz - 1`，排序后使用与`cscRowInd`相同的稳定排列重排 | Device |
| pBuffer | 输入 | void* | 由`aclsparseXcscsort_bufferSizeExt`查询大小并由调用方分配的workspace，地址须128字节对齐 | Device |

#### 约束说明

- `handle`不可为nullptr，否则返回`ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`。
- `m >= 0`、`n >= 0`且`nnz >= 0`。
- 当`m == 0`或`n == 0`时，`nnz`必须为0。
- `descrA`不可为nullptr，索引基址仅支持`ACL_SPARSE_INDEX_BASE_ZERO`和`ACL_SPARSE_INDEX_BASE_ONE`。
- 当`n > 0`时，`cscColPtr`不可为nullptr。
- 当`nnz > 0`时，`cscRowInd`、`P`和`pBuffer`均不可为nullptr。
- `pBuffer`地址必须128字节对齐，否则返回`ACL_SPARSE_STATUS_INVALID_VALUE`；`pBuffer`不得与`cscColPtr`、`cscRowInd`或`P`的存储区域重叠。
- 调用方须保证CSC结构合法，包括`cscColPtr`单调不减、`cscColPtr[n] - indexBase == nnz`，以及行索引位于矩阵行范围内（`[indexBase, indexBase + m)`）。
- 调用方须先将`P`初始化为`0, 1, ..., nnz - 1`（identity permutation）；本算子不负责创建`P`，仅在排序过程中同步重排。
- 矩阵类型隐式视为`ACL_SPARSE_MATRIX_TYPE_GENERAL`，忽略`descrA`中的对称性等属性。
- `indexBase`仅影响`cscColPtr`偏移换算，不影响排序逻辑（排序基于实际行索引值）。
- 当`nnz == 0`时，接口直接返回成功，不下发kernel。
- 本接口异步执行；读取`cscRowInd`和`P`前，调用方须同步handle绑定的stream。

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSC | ✅ | 对每个CSC列的行索引独立排序 |
| CSR | ❌ | 不支持CSR格式 |
| COO | ❌ | 不支持COO格式 |

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
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

static void PrintVector(const char *name, const std::vector<int> &values)
{
    LOG_PRINT("%s:", name);
    for (int value : values) {
        LOG_PRINT(" %d", value);
    }
    LOG_PRINT("\n");
}

int aclsparseCscsortTest(AclContext &ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建ops-sparse句柄并绑定stream
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet); return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(
        rawHandle, aclsparseDestroy);

    sparseRet = aclsparseSetStream(handlePtr.get(), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet); return sparseRet);

    // 2. 准备Host端CSC数据
    const int m = 5;
    const int n = 4;
    const int nnz = 7;
    std::vector<int> hCscColPtr = {0, 3, 4, 6, 7};
    std::vector<int> hCscRowInd = {4, 1, 3, 2, 4, 1, 0};
    std::vector<int> hP(nnz);
    std::iota(hP.begin(), hP.end(), 0);

    // 3. 申请Device内存并拷贝CSC数据和P
    DevicePtr dCscColPtr(nullptr, aclrtFree);
    DevicePtr dCscRowInd(nullptr, aclrtFree);
    DevicePtr dP(nullptr, aclrtFree);
    auto aclRet = AllocAndCopyDevice(
        dCscColPtr, hCscColPtr.data(), hCscColPtr.size() * sizeof(int));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(
        dCscRowInd, hCscRowInd.data(), hCscRowInd.size() * sizeof(int));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dP, hP.data(), hP.size() * sizeof(int));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 4. 查询workspace大小
    size_t bufferSize = 0;
    sparseRet = aclsparseXcscsort_bufferSizeExt(
        handlePtr.get(), m, n, nnz,
        static_cast<const int *>(dCscColPtr.get()),
        static_cast<const int *>(dCscRowInd.get()), &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("bufferSizeExt failed. ERROR: %d\n", sparseRet); return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes\n", bufferSize);

    // aclrtMalloc返回的地址默认满足128字节对齐要求，可直接作为pBuffer使用。
    DevicePtr dBuffer(nullptr, aclrtFree);
    aclRet = AllocAndCopyDevice(dBuffer, nullptr, bufferSize);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 5. 创建并配置Legacy MatDescr，由unique_ptr自动销毁
    aclsparseMatDescr_t rawDescr = nullptr;
    sparseRet = aclsparseCreateMatDescr(&rawDescr);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateMatDescr failed. ERROR: %d\n", sparseRet); return sparseRet);
    std::unique_ptr<aclsparseMatDescr, aclsparseStatus_t (*)(aclsparseMatDescr_t)> descrPtr(
        rawDescr, aclsparseDestroyMatDescr);
    aclsparseSetMatType(descrPtr.get(), ACL_SPARSE_MATRIX_TYPE_GENERAL);
    aclsparseSetMatIndexBase(descrPtr.get(), ACL_SPARSE_INDEX_BASE_ZERO);

    // 6. 执行CSC列内稳定排序
    sparseRet = aclsparseXcscsort(
        handlePtr.get(), m, n, nnz, descrPtr.get(),
        static_cast<const int *>(dCscColPtr.get()),
        static_cast<int *>(dCscRowInd.get()),
        static_cast<int *>(dP.get()), dBuffer.get());
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseXcscsort failed. ERROR: %d\n", sparseRet); return sparseRet);

    // 7. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 8. 将结果拷贝回Host并打印
    std::vector<int> hSortedRowInd(nnz, 0);
    std::vector<int> hSortedP(nnz, 0);
    aclRet = aclrtMemcpy(hSortedRowInd.data(), hSortedRowInd.size() * sizeof(int),
                         dCscRowInd.get(), hSortedRowInd.size() * sizeof(int),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("copy cscRowInd to Host failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(hSortedP.data(), hSortedP.size() * sizeof(int),
                         dP.get(), hSortedP.size() * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("copy P to Host failed. ERROR: %d\n", aclRet); return aclRet);

    PrintVector("cscColPtr", hCscColPtr);
    PrintVector("cscRowInd", hSortedRowInd);
    PrintVector("P", hSortedP);
    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseCscsortTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCscsortTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```text
bufferSize = 56 bytes
cscColPtr: 0 3 4 6 7
cscRowInd: 1 3 4 2 1 4 0
P: 1 2 0 3 5 4 6
```

排序后可使用`P`同步重排稀疏矩阵的值数组，例如对CSC值数组`cscValues`执行`sortedValues[i] = origValues[P[i]]`，即可得到与排序后`cscRowInd`/`cscColPtr`一致的值数组。

## 算法说明

- Host侧根据运行时AIV核数、UB容量、矩阵列数和`nnz`生成tiling数据。
- 多核切分以累计`nnz`为权重把完整列区间分配给各核（`taskUpperBound = min(n, nnz)`），列边界通过`cscColPtr`二分定位，不拆分单列。各核只访问自己列区间对应的`cscRowInd`、`P`、workspace区域，互不重叠，无需核间同步。
- 短列在UB内使用`Sort<int32_t>`完成稳定排序（RADIX_SORT，稳定升序）；长列不预先生成UB run，而是以单元素有序段为起点，直接在GM原数组与workspace之间进行`width=1,2,4,...`的bottom-up SIMT merge-path稳定归并，必要时将最终结果拷回原数组。
- 稳定性保证：归并阶段`left.key <= right.key`时选左侧，同列内相同行索引的元素保持原始相对顺序。
- 列之间相互独立，不执行跨列排序；`cscColPtr`全程不修改。

Atlas A2系列产品和Atlas A3系列产品（arch22）实现差异：arch22 的 AscendC `Sort` 仅支持 half/float 键且无 SIMT 排序，kernel 采用唯一 float 合成键驱动硬件排序完成稳定升序——每段构造键 `key = (segMaxRow - row) * 2^posBits + (len - 1 - pos)`（段内实测最大行号 `segMaxRow` 由 kernel `ReduceMax` 自适应，不依赖 host 传入的 m），键为小于 2^24 的非负精确整数且段内唯一，排序结果完全确定、稳定性不依赖硬件 tie 行为；pad 槽位键 `-1.0f` 沉底。Sort 的 index 通道预填 iota，输出每秩源下标，Extract 解交织后按字节偏移 Gather 同步重排 `cscRowInd` 与 `P`。短列（`len <= runSize`）多列聚合成批：每列独占一个 32 字节对齐的 UB 槽位，逐列 DataCopyPad 搬入后整批一次硬件 `Sort<float, false>`（isFullSort=false 按 32 组独立排序，repeat=段数），单段固定开销按批摊薄（批内行域上界取整批一次 `ReduceMax` 实测，不依赖 host 传入的 m）；键宽超预算（`segRowBits + posBits > 24`）或批内混入大于 32 槽位的段时，回退逐段路径（短段标量归并、长段矢量全排）。长列（`len > runSize`）按 `runSize` 分块在 UB 排序写入 GM workspace 形成有序 run，再多趟两路稳定归并；归并的 GM 写一律经 UB 暂存 + DataCopyPad（MTE3 带字节使能），避免标量 GM 写与相邻核 MTE3 写在共享 32B 扇区上互踩。workspace 与 arch35 口径一致（`2 * nnz * sizeof(int32_t)`）。

## 对齐接口

本算子对齐cuSPARSE Legacy API：

| 接口 | cuSPARSE 对应接口 |
|------|------------------|
| `aclsparseXcscsort_bufferSizeExt` | `cusparseXcscsort_bufferSizeExt` |
| `aclsparseXcscsort` | `cusparseXcscsort` |

接口签名、参数顺序、`P`的identity permutation语义、`descrA`的`indexBase`使用方式均与cuSPARSE一致；差异在于`pBuffer`要求128字节对齐（cuSPARSE未要求），调用方使用`aclrtMalloc`分配的内存默认满足该约束。

## 参考资源

- [cuSPARSE cusparseXcscsort 官方文档](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsexcscsort)
- [编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)
