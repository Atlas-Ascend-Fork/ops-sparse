# DenseToSparse 算子

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR / Ascend 950DT</term> | √（arch35） |
| <term>Atlas A3 训练系列产品 / Atlas A3 推理系列产品</term> | √（arch22） |
| <term>Atlas A2 训练系列产品 / Atlas A2 推理系列产品</term> | √（arch22） |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明


`DenseToSparse` 将二维稠密矩阵转换为 CSR、CSC、COO 或 Blocked-ELL
（下文简称 BELL）稀疏矩阵。接口语义对标 cuSPARSE Generic API 的
`cusparseDenseToSparse_bufferSize`、`cusparseDenseToSparse_analysis` 和
`cusparseDenseToSparse_convert`。推荐按查询 workspace、分析结构、执行转换三个
阶段调用，以便从 Dense 自动生成目标标准描述符中的结构信息。

对于 CSR 和 CSC，Analysis 统计非零元素、写入 offsets 并更新 `matB` 的 `nnz`；
COO Analysis 通过 `matB.nnz` 发布实际非零元素数；Convert 信任 Analysis 发布的
结构（要求与 Analysis 使用同一块 workspace），只写入索引和值。BELL 的结构和
storage 容量由调用者预置，Convert 只提取固定 pattern 覆盖的稠密块，不发现或
扩充结构；BELL 使用向上取整的块网格，尾块的越界逻辑元素输出 dtype 正零，
values 载荷按填充网格（`ceil(rows/b)*b x ellCols`）分配，空槽的块列索引为
`-1`（小于 index base 即视为空）。

接口语义对标 cuSPARSE Generic API 的
`cusparseDenseToSparse_bufferSize`、`cusparseDenseToSparse_analysis` 和
`cusparseDenseToSparse_convert`，推荐按查询 workspace、分析结构、执行转换
三个阶段调用。

### 接口与职责

| 接口名 | 功能简述 |
| --- | --- |
| `aclsparseDenseToSparseGetBufferSize` | 查询 Analysis/Convert 所需 Device workspace 字节数（纯查询，无阶段副作用） |
| `aclsparseDenseToSparseAnalysis` | 分析结构：CSR/CSC 统计非零并写 offsets、更新 `nnz`；COO 发布 `nnz`；BELL 校验几何与 pattern |
| `aclsparseDenseToSparseConvert` | 执行转换：CSR/CSC/COO 写入 indices/values（信任 Analysis 前缀）；BELL 按 pattern 提取块值 |

## 稀疏格式语义


| 格式 | Analysis 前必须提供 | Analysis 后绑定 | 输出语义 |
|------|---------------------|----------------|----------|
| CSR | `rowOffsets[m + 1]` | `colIndices[nnz]`、`values[nnz]` | 按行压缩；每行内列索引递增 |
| CSC | `colOffsets[n + 1]` | `rowIndices[nnz]`、`values[nnz]` | 按列压缩；每列内行索引递增 |
| COO | 无结构数组要求 | `rowIndices[nnz]`、`colIndices[nnz]`、`values[nnz]` | 按逻辑 row-major 顺序输出坐标 |
| BELL | `ellColInd[(ellCols / blockSize) * (m / blockSize)]`、`ellValue[m * ellCols]` | 无 | 按预置 block-column pattern 提取块值 |

CSR/CSC 的 descriptor 创建能力与 DenseToSparse 的执行能力需要区分：
`aclsparseCreateCsr` 和 `aclsparseCreateCsc` 可接受 I32/I32、I32/I64、I64/I32、
I64/I64 四种 offset/index 组合；但 DenseToSparse 为与 cuSPARSE 的实际算子行为
保持一致，仅能执行 offset 类型与 element index 类型相同的组合。

| CSR/CSC offset 类型 | CSR/CSC element index 类型 | DenseToSparse 支持情况 |
|---------------------|-----------------------------|-------------------------|
| I32 | I32 | 支持 |
| I32 | I64 | 不支持 |
| I64 | I32 | 不支持 |
| I64 | I64 | 支持 |

对于 I32/I64 或 I64/I32 的已创建 CSR/CSC descriptor，
`aclsparseDenseToSparseGetBufferSize`、`aclsparseDenseToSparseAnalysis` 和
`aclsparseDenseToSparseConvert` 均返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
COO 的 row/column index 共用描述符指定的 index 类型。所有格式均支持
`ACL_SPARSE_INDEX_BASE_ZERO` 和 `ACL_SPARSE_INDEX_BASE_ONE`。

## Blocked-ELL（BELL）


BELL 转换使用调用者在调用前确定的固定 pattern，而不是从稠密矩阵发现结构。
`ellColInd` 中的每一项表示一个 **block column index**，并按描述符的
`indexBase` 使用 0-based 或 1-based 编码。Analysis 和 Convert 都不会修改
`ellColInd`；pattern 未覆盖的 Dense 非零元素会被忽略，不会扩充、重排 pattern
或报告容量不足。

### 几何与存储容量

令 `rows = m`、`cols = n`、`ellBlockSize = b`，则必须满足：

- `b > 0`；
- `rows % b == 0`、`cols % b == 0`、`ellCols % b == 0`；
- `0 <= ellCols <= cols`。

BELL 有 `blockRows = rows / b` 个 block row，每个 block row 有
`slots = ellCols / b` 个固定 slot。因此：

| 数组/字段 | 元素数或语义 |
|-----------|--------------|
| `ellColInd` | `blockRows * slots` 个 I32 或 I64 block-column indices |
| `ellValue` | `rows * ellCols` 个 value 标量 |
| `aclsparseSpMatGetSize(..., &nnz)` | `nnz = rows * ellCols`，即存储标量容量 |

这里的 `nnz` 不是 Dense 中的实际非零数，也不是有效 pattern 覆盖范围内的实际
非零数。创建 BELL 描述符前必须按上述元素数分配 `ellColInd` 和 `ellValue`；
三个 DenseToSparse 阶段都不会改变该容量。

### Pattern、padding 与越界项

对 base 为 `base` 的 pattern 项 `encoded`，仅当
`base <= encoded < base + cols / b` 时有效，对应的 0-based block column 为
`encoded - base`。`-1`、任何小于 `base` 的值以及超出 block-column 范围的值
均按 padding 处理；对应整个 `b * b` slot 在 `ellValue` 中写为该 value 类型的
**正零**。这也适用于 1-based pattern 中的 `0`。

`ellColInd` 位于 Device。接口不会把 pattern 同步到 Host 后逐项校验其内容，
因此 GetBufferSize 和 Analysis 不会因某个 pattern 项越界而报错；该项在 Convert
的结果语义就是上述 padding。调用者仍须保证 Device 指针、元素类型和分配容量
正确。

### 三阶段调用与生命周期

BELL 也可沿用推荐的
`GetBufferSize -> Analysis -> Convert` 顺序。其 GetBufferSize 返回
`bufferSize == 0`，Analysis 和 Convert 的 `buffer` 因而都可传 `nullptr`。
Analysis 不扫描 Dense、不读取 pattern，也不启动 Device kernel；Convert 直接
消费当前 BELL 描述符中的固定 pattern 和 storage，不依赖私有阶段状态或历史快照。

`ellColInd` 和 `ellValue` 必须在创建 BELL 描述符前准备并绑定，并保持地址和容量
有效，直至 Convert 所在 stream 执行完成。描述符只借用这些 Device 指针，不拥有
其内存；同步 stream 后才能销毁描述符或释放数组。

### Values 布局与 Dense 读取

任务编号按 block row 优先排列：

```text
task = blockRow * slots + slot
valuePos = task * b * b + innerCol * b + innerRow
```

因此 `ellValue` 先按 `task`（即 `ellColInd` 的 slot 顺序）排列，每个
`b * b` block 内部按 column-major 排列。Dense 的读取则遵循 `matA` 的 order：

```text
ROW: dense[row * ld + col]
COL: dense[col * ld + row]
```

两种 Dense order 得到相同的逻辑 BELL values，只是输入的物理寻址方式不同。

### BELL 专用调用片段

以下片段展示 1-based、I32 pattern 的关键调用顺序。示例中 `rows = cols = 4`、
`b = 2`、`ellCols = 4`，所以 pattern 有 4 项，values 有 16 个 FP32 标量。
其中 `-1` 是 padding。

```cpp
constexpr int64_t rows = 4;
constexpr int64_t cols = 4;
constexpr int64_t b = 2;
constexpr int64_t ellCols = 4;
const int32_t hostPattern[] = {1, -1, 2, -1};

void *devicePattern = nullptr;
void *deviceBellValues = nullptr;
aclrtMalloc(&devicePattern, sizeof(hostPattern), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(devicePattern, sizeof(hostPattern),
            hostPattern, sizeof(hostPattern), ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMalloc(&deviceBellValues, rows * ellCols * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);

// handle、matA 和 matA 引用的 FP32 Dense Device 数据已提前创建；
// stream 已通过 aclsparseSetStream(handle, stream) 绑定到 handle。
aclsparseSpMatDescr_t matB = nullptr;
aclsparseCreateBlockedEll(
    &matB, rows, cols, b, ellCols, devicePattern, deviceBellValues,
    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ONE, ACL_FLOAT);

size_t bufferSize = 0;
aclsparseDenseToSparseGetBufferSize(
    handle, matA, matB, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize);
// BELL 固定返回 0；这里仍按推荐的三阶段顺序调用。
if (bufferSize == 0) {
    aclsparseDenseToSparseAnalysis(
        handle, matA, matB, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, nullptr);
    aclsparseDenseToSparseConvert(
        handle, matA, matB, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, nullptr);
}
aclrtSynchronizeStream(stream);

aclsparseDestroySpMat(matB);
aclrtFree(deviceBellValues);
aclrtFree(devicePattern);
```

BELL 的 index 类型支持 `ACL_SPARSE_INDEX_32I` 和
`ACL_SPARSE_INDEX_64I`，value 类型仅支持 `ACL_INT8`、`ACL_FLOAT16`、
`ACL_BF16` 和 `ACL_FLOAT`，且必须与 Dense 的 value 类型一致。

## 支持数据类型


value 类型要求 `matA` 与 `matB` 相同；各架构支持范围如下。

| 数据类型 | 枚举 | arch22 (A2/A3) | arch35 |
|----------|------|----------------|---------|
| int8 | `ACL_INT8` | 支持 | 支持 |
| float16 | `ACL_FLOAT16` | 支持 | 支持 |
| bfloat16 | `ACL_BF16` | 支持 | 支持 |
| float32 | `ACL_FLOAT` | 支持 | 支持 |
| complex64 | `ACL_COMPLEX64` | 支持 | 不支持 |

不支持 float64；arch35 不支持任何复数类型。传入不支持类型返回
`ACL_SPARSE_STATUS_NOT_SUPPORTED`。

**Device 索引固定为 I32（`ACL_SPARSE_INDEX_32I`）**：I64 描述符在三阶段接口上
返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`（描述符创建本身可能成功，属公开创建
能力与算子执行能力之差）。矩阵行数、列数和 CSR/CSC/COO 的实际 `nnz` 范围为
`[0, INT32_MAX]`，所有输出 offset/index 必须能由 I32 表示。

## 数值、布局与边界语义


- 同时支持 `ACL_SPARSE_ORDER_ROW`（row-major）和
  `ACL_SPARSE_ORDER_COL`（column-major）。row-major 要求 `ld >= n`，
  column-major 要求 `ld >= m`。
- 对浮点类型，`+0` 和 `-0` 均视为零，不写入 CSR/CSC/COO。
- NaN（包括其 payload）、`+Inf`、`-Inf` 和普通有限非零值均视为非零，值按位搬运，
  不执行数值计算或类型转换。
- 支持零维矩阵、全零矩阵和全非零矩阵。零维 CSR/CSC 仍生成长度为主维加一的
  offsets，元素均为 index base；零维 COO 没有输出元素。
- 全零 CSR/CSC 的 offsets 全部等于 index base；全零 COO 的 `nnz` 为 0。
- CSR/CSC/COO 的实际 `nnz` 不超过 `m * n`。维度、`ld`、workspace 或地址计算
  溢出时返回参数错误。

## 接口说明

### 公共参数


三个接口的公共参数如下。

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|----------|------|------|------|
| `handle` | Host | 输入 | `aclsparseHandle_t` | `ops-sparse` 上下文句柄，携带执行 stream，不可为空 |
| `matA` | Host 描述符；数据在 Device | 输入 | `aclsparseConstDnMatDescr_t` | 稠密矩阵 A，包含行数、列数、`ld`、布局、数据类型和数据指针 |
| `matB` | Host 描述符；数组在 Device | 输入/输出 | `aclsparseSpMatDescr_t` | 可变稀疏矩阵 B；Analysis 可更新 CSR/CSC/COO 的 `nnz`，Convert 写入数组 |
| `alg` | Host | 输入 | `aclsparseDenseToSparseAlg_t` | 仅支持 `ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT` |

各阶段的专有参数如下。

| 接口 | 参数 | 内存位置 | 方向 | 说明 |
|------|------|----------|------|------|
| `GetBufferSize` | `bufferSize` | Host | 输出 | Analysis/Convert 所需 Device workspace 字节数，不可为空；该接口是纯查询，不记录阶段状态 |
| `Analysis` | `buffer` | Device | 输入/输出 | 不少于 `bufferSize` 字节的 workspace；`bufferSize == 0` 时可为空 |
| `Convert` | `buffer` | Device | 输入/输出 | 不少于 `bufferSize` 字节的 workspace；**必须与 Analysis 使用同一块 workspace**（Convert 信任 Analysis 写入该 workspace 的结构前缀，跨缓冲或未执行 Analysis 的调用返回参数错误）；`bufferSize == 0` 时可为空 |

### 调用流程


1. 创建 handle、稠密矩阵描述符和目标格式的可变稀疏矩阵描述符。
2. 调用 `aclsparseDenseToSparseGetBufferSize`，按返回大小分配 Device workspace。
3. 调用 `aclsparseDenseToSparseAnalysis`。
4. 对 CSR、CSC、COO，使用 `aclsparseSpMatGetSize` 查询实际 `nnz`，分配输出
   indices/values，并调用对应的 `aclsparseCsrSetPointers`、
   `aclsparseCscSetPointers` 或 `aclsparseCooSetPointers` 绑定指针。BELL 不执行
   此步骤，其固定数组在创建描述符前就应分配完成。
5. 调用 `aclsparseDenseToSparseConvert`，然后同步 handle 绑定的 stream，再读取结果。

这套顺序用于从 Dense 自动得到 `nnz` 和 CSR/CSC offsets，并据此分配 payload
数组。`GetBufferSize` 不修改描述符，也不是 Analysis 的状态前置条件。Convert
信任 Analysis 写入 workspace 的 status/level0 结构前缀（同一块 workspace 在
描述符上登记，跨缓冲或未执行 Analysis 的 Convert 调用返回参数错误），仅写
payload 不重扫结构；指针、容量和生命周期均由调用者保证。


- workspace 仅按 `GetBufferSize` 查询值分配，Analysis 与 Convert 复用同一块
  workspace（arch22 协议：Analysis 在描述符上登记所用 workspace，Convert 使用
  不同地址或未先执行 Analysis 时返回 `ACL_SPARSE_STATUS_INVALID_VALUE`，
  对齐 cuSPARSE DenseToSparse 的 analysis 前置契约）；接口没有 workspace 长度
  参数，无法代替调用者校验实际 allocation 大小。
- `bufferSize > 0` 时，传给当前 API 的 workspace 必须非空，并至少保持有效到该次
  stream 工作完成。BELL 以及不需要 scratch 的空输入返回 0，此时 `buffer` 可为空。
- CSR/CSC/COO 的 Analysis 会在返回前取得实际 `nnz` 并更新 Host 描述符；Convert
  按 handle 的 stream 异步执行，调用者必须在读取或释放输出、workspace、描述符及
  handle 前同步该 stream。
- arch22 的 Convert 信任 Analysis 写入同一 workspace 的 status/level0 结构
  前缀，仅消费当前标准描述符字段写 payload（不重扫结构）；调用者必须保证当前
  描述符数据与当前 Dense 一致，并为其声明的容量提供有效存储。
- CSR/CSC 的输出位置以当前 offsets 为权威来源；COO 以当前 `matB.nnz` 作为
  容量门槛（arch22 不做 Device 侧重计数校验，结构以 Analysis 前缀为准）。
- Convert 保持 stream 异步：上述 Device mismatch 不会通过 host 同步回读转换成
  本次 API launch 的同步错误返回。调用者仍须在读取结果或复用、释放资源前同步
  stream。
- BELL 的 `ellColInd`、`ellValue` 及其容量必须在 Convert 前绑定并保持有效；
  BELL Convert 按固定 pattern 读取调用时的稠密值。
- 描述符只借用 Device 指针，不拥有其内存。先同步 stream，再销毁描述符并释放其
  引用的 Device 内存；workspace 和数据指针均由调用者分配、释放。


| 返回值 | 含义 |
|--------|------|
| `ACL_SPARSE_STATUS_SUCCESS` | 成功 |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | 空参数、非法枚举/格式、shape/`ld`/index base/index 类型错误、溢出、缺少所需指针或 workspace 为空 |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | value 数据类型不支持；或者 CSR/CSC 的 offset 类型与 element index 类型不一致 |
| `ACL_SPARSE_STATUS_EXECUTION_FAILED` | Device/Runtime 执行、拷贝或 stream 同步失败 |
| `ACL_SPARSE_STATUS_INTERNAL_ERROR` | 内部资源或设备核信息异常 |

描述符创建及 Device 内存分配还可能分别返回
`ACL_SPARSE_STATUS_ALLOC_FAILED` 或 ACL Runtime 错误；这些错误应在进入三阶段接口前
处理。

## 调用示例


以下示例将 row-major FP32 稠密矩阵转换为 0-based、I32 CSR。示例特意使用
`aclsparseCreateDnMat` 创建非 const 描述符，并将其传给只读 `matA` 参数。

```cpp
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"

#define CHECK_ACL(expr)                                                       \
    do {                                                                      \
        const aclError acl_status = (expr);                                   \
        if (acl_status != ACL_SUCCESS) {                                      \
            std::fprintf(stderr, "%s failed: %d\n", #expr, acl_status);       \
            return 1;                                                         \
        }                                                                     \
    } while (0)

#define CHECK_SPARSE(expr)                                                    \
    do {                                                                      \
        const aclsparseStatus_t sparse_status = (expr);                       \
        if (sparse_status != ACL_SPARSE_STATUS_SUCCESS) {                     \
            std::fprintf(stderr, "%s failed: %d\n", #expr, sparse_status);    \
            return 1;                                                         \
        }                                                                     \
    } while (0)

int main()
{
    constexpr int32_t deviceId = 0;
    constexpr int64_t m = 2;
    constexpr int64_t n = 3;
    const std::vector<float> hostA = {
        1.0f, 0.0f, 2.0f,
        0.0f, 3.0f, -0.0f
    };

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclsparseHandle_t handle = nullptr;
    CHECK_SPARSE(aclsparseCreate(&handle));
    CHECK_SPARSE(aclsparseSetStream(handle, stream));

    void *deviceA = nullptr;
    void *rowOffsets = nullptr;
    CHECK_ACL(aclrtMalloc(&deviceA, hostA.size() * sizeof(float),
                          ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(deviceA, hostA.size() * sizeof(float),
                         hostA.data(), hostA.size() * sizeof(float),
                         ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMalloc(&rowOffsets, (m + 1) * sizeof(int32_t),
                          ACL_MEM_MALLOC_HUGE_FIRST));

    aclsparseDnMatDescr_t matA = nullptr;
    aclsparseSpMatDescr_t matB = nullptr;
    CHECK_SPARSE(aclsparseCreateDnMat(
        &matA, m, n, n, deviceA, ACL_FLOAT, ACL_SPARSE_ORDER_ROW));
    CHECK_SPARSE(aclsparseCreateCsr(
        &matB, m, n, 0, rowOffsets, nullptr, nullptr,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT));

    size_t bufferSize = 0;
    CHECK_SPARSE(aclsparseDenseToSparseGetBufferSize(
        handle, matA, matB, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
        &bufferSize));

    void *workspace = nullptr;
    if (bufferSize != 0) {
        CHECK_ACL(aclrtMalloc(
            &workspace, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_SPARSE(aclsparseDenseToSparseAnalysis(
        handle, matA, matB, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
        workspace));

    int64_t nnz = 0;
    CHECK_SPARSE(aclsparseSpMatGetSize(matB, nullptr, nullptr, &nnz));

    void *colIndices = nullptr;
    void *values = nullptr;
    if (nnz != 0) {
        CHECK_ACL(aclrtMalloc(
            &colIndices, nnz * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(
            &values, nnz * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_SPARSE(
        aclsparseCsrSetPointers(matB, rowOffsets, colIndices, values));
    CHECK_SPARSE(aclsparseDenseToSparseConvert(
        handle, matA, matB, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
        workspace));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<int32_t> hostOffsets(m + 1);
    std::vector<int32_t> hostColumns(static_cast<size_t>(nnz));
    std::vector<float> hostValues(static_cast<size_t>(nnz));
    CHECK_ACL(aclrtMemcpy(
        hostOffsets.data(), hostOffsets.size() * sizeof(int32_t),
        rowOffsets, hostOffsets.size() * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST));
    if (nnz != 0) {
        CHECK_ACL(aclrtMemcpy(
            hostColumns.data(), hostColumns.size() * sizeof(int32_t),
            colIndices, hostColumns.size() * sizeof(int32_t),
            ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(
            hostValues.data(), hostValues.size() * sizeof(float),
            values, hostValues.size() * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST));
    }

    std::printf("nnz = %lld\n", static_cast<long long>(nnz));
    for (int64_t i = 0; i < nnz; ++i) {
        std::printf("col[%lld] = %d, value = %.1f\n",
                    static_cast<long long>(i),
                    hostColumns[static_cast<size_t>(i)],
                    hostValues[static_cast<size_t>(i)]);
    }

    CHECK_SPARSE(aclsparseDestroySpMat(matB));
    CHECK_SPARSE(aclsparseDestroyDnMat(matA));
    if (values != nullptr) {
        CHECK_ACL(aclrtFree(values));
    }
    if (colIndices != nullptr) {
        CHECK_ACL(aclrtFree(colIndices));
    }
    if (workspace != nullptr) {
        CHECK_ACL(aclrtFree(workspace));
    }
    CHECK_ACL(aclrtFree(rowOffsets));
    CHECK_ACL(aclrtFree(deviceA));
    CHECK_SPARSE(aclsparseDestroy(handle));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}
```

预期结果为 `nnz = 3`，CSR offsets 为 `[0, 2, 3]`，column indices 为
`[0, 2, 1]`，values 为 `[1.0, 2.0, 3.0]`。

具体编译和链接方式请参考
[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

## 编译与测试

```bash
cmake -S . -B build \
  -DSOC_VERSION=ascend910b3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TEST=ON \
  -DOP_LIST=densetosparse
cmake --build build --target densetosparse_test densetosparse_perf --parallel
./build/test/densetosparse/densetosparse_test --gtest_color=no
```

PyTorch 接口（`Tensor.to_sparse*` 的 NPU 适配）通过 `torch_extension`
框架提供，构建和加载方式见
[`../../torch_extension/README.md`](../../torch_extension/README.md)。

## 目录结构

```text
sparse/densetosparse/
├── README.md
├── arch22/                        # Atlas A2/A3（AIV 向量编程模型）
│   ├── densetosparse_host.cpp     # 三阶段 Host 编排与校验
│   ├── densetosparse_count_kernel.cpp   # 计数/扫描链/offsets 收尾 + 启动编排
│   ├── densetosparse_convert_kernel.cpp # CSR/CSC/COO 载荷提取（面板快路径）
│   ├── densetosparse_bell_kernel.cpp    # Blocked-ELL 专换（任务环流水）
│   ├── densetosparse_judge.h      # 共享判定链/位图/几何选择
│   ├── densetosparse_kernel.h / densetosparse_kernel_shared.h
│   └── densetosparse_tiling_data.h
├── arch35/                        # Ascend 950PR
│   ├── densetosparse_host.cpp
│   └── densetosparse_kernel.cpp / densetosparse_kernel.h
└── torch_extension/               # PyTorch 接口注册（JIT 单独编译）
    ├── __init__.py / dense_to_sparse.py
    └── csrc/densetosparse.cpp
```

测试代码位于 `test/densetosparse/`（`arch22/` 为 C++ UT 与 perf 基准，
`python/` 为 ATen/端到端测试）。
