# Csr2gebsr算子

## 算子概述

csr2gebsr 算子用于将 CSR（Compressed Sparse Row）格式的稀疏矩阵转换为 GEBSR（General Block Sparse Row）格式，即分块 CSR 格式。CSR 以单个非零元为单位存储，GEBSR 以 `rowBlockDim × colBlockDim` 的块为单位存储，全零块被跳过以实现压缩。参考 CUDA cusparse 的 `cusparse<t>csr2gebsr` 接口。

采用 Legacy 三步法 API（对标 cuSPARSE）：
1. **bufferSize**：查询 workspace 大小（纯 Host 计算）
2. **Nnz**：计算 `bsrRowPtrC` 和非零块数 `nnzb`
3. **Convert**：填充 `bsrColIndC` + `bsrValC`

`dir` 参数指定块内内存布局：ROW = 行主序，COLUMN = 列主序。

### 支持数据类型

| 数据类型 | 精度版本前缀 | 元素大小 |
|----------|-------------|----------|
| float32 | S | 4B |
| float16 | H | 2B |
| bfloat16 | Bh | 2B |
| int32 | I | 4B |

### 支持产品系列

| 产品系列 | NPU 架构 | 架构目录 |
|----------|----------|----------|
| Ascend 950PR&950DT系列产品 | dav-3510 | arch35 |

## 算子执行接口

算子共提供 9 个公开接口（Legacy 三步法，4 精度版本 × bufferSize + Convert，1 个类型无关 Nnz）：

| 接口 | 说明 |
|------|------|
| `aclsparse{S\|H\|Bh\|I}csr2gebsr_bufferSize` | 查询 workspace 大小（4 精度版本） |
| `aclsparseXcsr2gebsrNnz` | 计算 bsrRowPtrC 和 nnzb（类型无关） |
| `aclsparse{S\|H\|Bh\|I}csr2gebsr` | 执行 CSR→GEBSR 转换（4 精度版本） |

### aclsparseScsr2gebsr_bufferSize

#### 产品支持情况

| 产品系列 | 支持状态 |
|----------|----------|
| Ascend 950PR&950DT系列产品 | 支持 |

#### 函数原型

```cpp
aclsparseStatus_t aclsparseScsr2gebsr_bufferSize(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n, const aclsparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    int rowBlockDim, int colBlockDim, size_t *pBufferSizeInBytes);
```

> `aclsparseHcsr2gebsr_bufferSize` / `aclsparseBhcsr2gebsr_bufferSize` / `aclsparseIcsr2gebsr_bufferSize` 参数相同，仅 `csrValA` 类型不同（`const void*` 或 `const int*`）。

#### 参数说明

| 参数 | 输入/输出 | 类型 | 说明 |
|------|-----------|------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，不可为 nullptr |
| dir | 输入 | aclsparseDirection_t | 块内内存布局（ROW 或 COLUMN） |
| m | 输入 | int | 矩阵行数，m >= 0 |
| n | 输入 | int | 矩阵列数，n >= 0 |
| descrA | 输入 | aclsparseMatDescr_t | 输入矩阵描述符，type 必须为 GENERAL |
| csrValA | 输入 | const float* | CSR 非零值数组（本函数不读取，可为 nullptr） |
| csrRowPtrA | 输入 | const int* | CSR 行偏移数组，长度 m+1，m > 0 时不可为 nullptr |
| csrColIndA | 输入 | const int* | CSR 列索引数组（本函数不读取，可为 nullptr） |
| rowBlockDim | 输入 | int | 块行维度，> 0 |
| colBlockDim | 输入 | int | 块列维度，> 0 |
| pBufferSizeInBytes | 输出 | size_t* | 所需 workspace 字节数，不可为 nullptr |

#### 约束说明

- `m >= 0`，`n >= 0`
- `rowBlockDim > 0`，`colBlockDim > 0`
- `m`、`n`、`rowBlockDim`、`colBlockDim` 各自 ≤ INT32_MAX/2
- `mb × nb ≤ INT32_MAX`（mb = ceil(m/rowBlockDim)，nb = ceil(n/colBlockDim)）
- `descrA` 的 matrixType 必须为 GENERAL，indexBase 必须为 ZERO 或 ONE
- `m > 0` 时 `csrRowPtrA` 不可为 nullptr

### aclsparseXcsr2gebsrNnz

#### 产品支持情况

| 产品系列 | 支持状态 |
|----------|----------|
| Ascend 950PR&950DT系列产品 | 支持 |

#### 函数原型

```cpp
aclsparseStatus_t aclsparseXcsr2gebsrNnz(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    int *bsrRowPtrC,
    int rowBlockDim, int colBlockDim,
    int *nnzTotalDevHostPtr,
    void *pBuffer);
```

#### 参数说明

| 参数 | 输入/输出 | 类型 | 说明 |
|------|-----------|------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，不可为 nullptr |
| dir | 输入 | aclsparseDirection_t | 块内内存布局（ROW 或 COLUMN） |
| m | 输入 | int | 矩阵行数，m >= 0 |
| n | 输入 | int | 矩阵列数，n >= 0 |
| descrA | 输入 | aclsparseMatDescr_t | 输入矩阵描述符 |
| csrRowPtrA | 输入 | const int* | CSR 行偏移数组，长度 m+1，m > 0 时不可为 nullptr |
| csrColIndA | 输入 | const int* | CSR 列索引数组，nnz > 0 时需提供合法指针 |
| descrC | 输入 | aclsparseMatDescr_t | 输出矩阵描述符，不可为 nullptr |
| bsrRowPtrC | 输出 | int* | 块行偏移数组，长度 mb+1，不可为 nullptr |
| rowBlockDim | 输入 | int | 块行维度，> 0 |
| colBlockDim | 输入 | int | 块列维度，> 0 |
| nnzTotalDevHostPtr | 输出 | int* | 非零块数输出指针，不可为 nullptr |
| pBuffer | 输入 | void* | workspace，不可为 nullptr |

#### 约束说明

- 同 bufferSize 约束
- `descrC` 不可为 nullptr，matrixType 必须为 GENERAL
- `bsrRowPtrC` 不可为 nullptr
- `nnzTotalDevHostPtr` 不可为 nullptr（HOST 模式为 host 指针，DEVICE 模式为 device 指针）
- `pBuffer` 不可为 nullptr
- `nnzTotalDevHostPtr` 的指针类型必须与 `aclsparseSetPointerMode` 设置一致：HOST 模式传 host 指针（算子 D2H 回传 nnzb），DEVICE 模式传 device 指针（算子 D2D 回传）
- 本 API 异步执行（on stream），调用方在调用 Convert 前须同步 stream 或读回 nnzb 确认 Nnz 完成

### aclsparseScsr2gebsr

#### 产品支持情况

| 产品系列 | 支持状态 |
|----------|----------|
| Ascend 950PR&950DT系列产品 | 支持 |

#### 函数原型

```cpp
aclsparseStatus_t aclsparseScsr2gebsr(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    float *bsrValC, int *bsrRowPtrC, int *bsrColIndC,
    int rowBlockDim, int colBlockDim,
    void *pBuffer);
```

> `aclsparseHcsr2gebsr` / `aclsparseBhcsr2gebsr` / `aclsparseIcsr2gebsr` 参数相同，仅 `csrValA`/`bsrValC` 类型不同。

#### 参数说明

| 参数 | 输入/输出 | 类型 | 说明 |
|------|-----------|------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，不可为 nullptr |
| dir | 输入 | aclsparseDirection_t | 块内内存布局（ROW 或 COLUMN） |
| m | 输入 | int | 矩阵行数，m >= 0 |
| n | 输入 | int | 矩阵列数，n >= 0 |
| descrA | 输入 | aclsparseMatDescr_t | 输入矩阵描述符 |
| csrValA | 输入 | const float* | CSR 非零值数组，m > 0 时不可为 nullptr |
| csrRowPtrA | 输入 | const int* | CSR 行偏移数组，长度 m+1，m > 0 时不可为 nullptr |
| csrColIndA | 输入 | const int* | CSR 列索引数组，m > 0 时不可为 nullptr |
| descrC | 输入 | aclsparseMatDescr_t | 输出矩阵描述符 |
| bsrValC | 输出 | float* | GEBSR 非零块值数组，长度 nnzb×rowBlockDim×colBlockDim，m > 0 且 n > 0 时不可为 nullptr |
| bsrRowPtrC | 输入/输出 | int* | 块行偏移数组，长度 mb+1，不可为 nullptr |
| bsrColIndC | 输出 | int* | 块列索引数组，长度 nnzb，m > 0 且 n > 0 时不可为 nullptr |
| rowBlockDim | 输入 | int | 块行维度，> 0 |
| colBlockDim | 输入 | int | 块列维度，> 0 |
| pBuffer | 输入 | void* | workspace，不可为 nullptr |

#### 约束说明

- 同 Nnz 约束
- `m > 0` 时 `csrValA`、`csrRowPtrA`、`csrColIndA` 不可为 nullptr
- `m > 0 && n > 0` 时 `bsrValC`、`bsrColIndC` 不可为 nullptr
- `m == 0 || n == 0` 时跳过 kernel，直接返回 SUCCESS
- 本 API 依赖前置 Nnz 已在 stream 上完成，调用方须在调用前同步 stream

### 返回值

| 返回值 | 说明 |
|--------|------|
| ACL_SPARSE_STATUS_SUCCESS | 操作成功 |
| ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR | handle 为 nullptr |
| ACL_SPARSE_STATUS_INVALID_VALUE | 参数非法（负值、非法枚举、nullptr 指针、维度溢出） |
| ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED | 矩阵类型非 GENERAL |
| ACL_SPARSE_STATUS_EXECUTION_FAILED | aclrt API 调用失败 |
| ACL_SPARSE_STATUS_INTERNAL_ERROR | 内部错误（如获取核数失败） |

### workspace 计算公式

- workspace 大小 = `(mb + 1 + mb×nb + aivCoreNum) × sizeof(int32_t)`
- workspace 布局：`[nnzBlocksPerRow(mb) | nnzbDev(1) | marker(mb×nb) | segSum(aivCoreNum)]`
- `mb = ceil(m / rowBlockDim)`，`nb = ceil(n / colBlockDim)`
- `aivCoreNum` 为 AI Core 数量

### 输入数据契约

调用方必须保证（Host 侧无法校验 device 数据内容）：
- CSR 行内列索引必须按升序排列（Convert 阶段依赖二分查找定位目标块）
- `csrColIndA` 每个元素（减去 baseA 后）必须落在 `[0, n)` 区间内

## 实现架构

采用 SIMT 三层结构（class-based Dispatcher + asc_vf_call 混合编程模式），共 3 个 kernel：

1. **Kernel 1 (Nnz CountBlocksPerRow)**：逐块行扫描 CSR 行，用 marker 数组标记非零块，统计每行非零块数 `nnzBlocksPerRow`。多核按块行 `mb` 切分并行。

2. **Kernel 2 (PrefixSum)**：将 `nnzBlocksPerRow` 做 exclusive prefix sum 生成 `bsrRowPtrC`，并回传 `nnzb`。
   - `mb ≤ 128`（useBlocks == 1）：退化串行路径，单 kernel O(mb)
   - `mb > 128`（useBlocks > 1）：融合并行路径，单 kernel 内 Phase1→SyncAll→Phase2→SyncAll→Phase3

3. **Kernel 3 (Convert)**：遍历 CSR 非零元，用 marker 做 BinarySearch 定位目标块位置，填充 `bsrColIndC` 和 `bsrValC`（按 dir 指定的 ROW/COLUMN 方向排列块内数据）。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include <iostream>
#include <memory>

#define CHECK_ACL(ret, msg) do { if ((ret) != ACL_SUCCESS) { \
    std::cerr << msg << ": " << ret << std::endl; return ret; } } while(0)
#define CHECK_SPARSE(ret, msg) do { if ((ret) != ACL_SPARSE_STATUS_SUCCESS) { \
    std::cerr << msg << ": " << ret << std::endl; return ret; } } while(0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}
    ~AclContext() {
        if (stream_) aclrtDestroyStream(stream_);
        if (deviceSet_) aclrtResetDevice(deviceId_);
        if (aclInited_) aclFinalize();
    }
    aclError Init() {
        auto ret = aclInit(nullptr);
        CHECK_ACL(ret, "aclInit failed");
        aclInited_ = true;
        ret = aclrtSetDevice(deviceId_);
        CHECK_ACL(ret, "aclrtSetDevice failed");
        deviceSet_ = true;
        ret = aclrtCreateStream(&stream_);
        CHECK_ACL(ret, "aclrtCreateStream failed");
        return ACL_SUCCESS;
    }
    aclrtStream Stream() const { return stream_; }
private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct DeviceBufferDeleter {
    void operator()(void* p) const { if (p) aclrtFree(p); }
};
using DeviceUPtr = std::unique_ptr<void, DeviceBufferDeleter>;

static DeviceUPtr AllocDevice(size_t bytes) {
    void* p = nullptr;
    aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    return DeviceUPtr(p);
}

static DeviceUPtr CopyToDevice(const void* host, size_t bytes) {
    auto d = AllocDevice(bytes);
    if (d) aclrtMemcpy(d.get(), bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    return d;
}

int Csr2GebsrTest(AclContext& ctx) {
    aclrtStream stream = ctx.Stream();

    aclsparseHandle_t rawHandle = nullptr;
    CHECK_SPARSE(aclsparseCreate(&rawHandle), "aclsparseCreate failed");
    std::unique_ptr<aclsparseContext, aclsparseStatus_t(*)(aclsparseHandle_t)> handle(
        rawHandle, aclsparseDestroy);
    aclsparseSetStream(handle.get(), stream);
    aclsparseSetPointerMode(handle.get(), ACL_SPARSE_POINTER_MODE_HOST);

    // 4x4 CSR matrix:
    //   1 0 0 0
    //   0 0 0 0
    //   0 0 0 2
    //   0 0 0 0
    int m = 4, n = 4, nnz = 2;
    int rowBlockDim = 2, colBlockDim = 2;
    int h_csrRowPtr[] = {0, 1, 1, 1, 2};
    int h_csrColInd[] = {0, 3};
    float h_csrVal[] = {1.0f, 2.0f};

    aclsparseMatDescr_t descrA = nullptr, descrC = nullptr;
    aclsparseCreateMatDescr(&descrA);
    aclsparseSetMatType(descrA, ACL_SPARSE_MATRIX_TYPE_GENERAL);
    aclsparseSetMatIndexBase(descrA, ACL_SPARSE_INDEX_BASE_ZERO);
    aclsparseCreateMatDescr(&descrC);
    aclsparseSetMatType(descrC, ACL_SPARSE_MATRIX_TYPE_GENERAL);
    aclsparseSetMatIndexBase(descrC, ACL_SPARSE_INDEX_BASE_ZERO);

    auto d_rowPtr = CopyToDevice(h_csrRowPtr, 5 * sizeof(int));
    auto d_colInd = CopyToDevice(h_csrColInd, 2 * sizeof(int));
    auto d_csrVal = CopyToDevice(h_csrVal, 2 * sizeof(float));

    // Step 1: bufferSize
    size_t bufSize = 0;
    CHECK_SPARSE(aclsparseScsr2gebsr_bufferSize(handle.get(), ACL_SPARSE_DIRECTION_ROW,
        m, n, descrA, reinterpret_cast<float*>(d_csrVal.get()),
        reinterpret_cast<int*>(d_rowPtr.get()), reinterpret_cast<int*>(d_colInd.get()),
        rowBlockDim, colBlockDim, &bufSize), "bufferSize failed");
    auto d_buffer = AllocDevice(bufSize);

    // Step 2: Nnz
    int mb = (m + rowBlockDim - 1) / rowBlockDim;
    auto d_bsrRowPtr = AllocDevice((mb + 1) * sizeof(int));
    int nnzb = 0;
    CHECK_SPARSE(aclsparseXcsr2gebsrNnz(handle.get(), ACL_SPARSE_DIRECTION_ROW,
        m, n, descrA, reinterpret_cast<int*>(d_rowPtr.get()),
        reinterpret_cast<int*>(d_colInd.get()), descrC,
        reinterpret_cast<int*>(d_bsrRowPtr.get()),
        rowBlockDim, colBlockDim, &nnzb, d_buffer.get()), "Nnz failed");
    aclrtSynchronizeStream(stream);

    // Step 3: Convert
    int blockSize = rowBlockDim * colBlockDim;
    auto d_bsrColInd = AllocDevice(nnzb * sizeof(int));
    auto d_bsrVal = AllocDevice(nnzb * blockSize * sizeof(float));
    CHECK_SPARSE(aclsparseScsr2gebsr(handle.get(), ACL_SPARSE_DIRECTION_ROW,
        m, n, descrA, reinterpret_cast<float*>(d_csrVal.get()),
        reinterpret_cast<int*>(d_rowPtr.get()), reinterpret_cast<int*>(d_colInd.get()),
        descrC, reinterpret_cast<float*>(d_bsrVal.get()),
        reinterpret_cast<int*>(d_bsrRowPtr.get()), reinterpret_cast<int*>(d_bsrColInd.get()),
        rowBlockDim, colBlockDim, d_buffer.get()), "Convert failed");
    aclrtSynchronizeStream(stream);

    // Read back
    int h_bsrRowPtr[3], h_bsrColInd[2];
    float h_bsrVal[8];
    aclrtMemcpy(h_bsrRowPtr, 3*sizeof(int), d_bsrRowPtr.get(), 3*sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(h_bsrColInd, 2*sizeof(int), d_bsrColInd.get(), 2*sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(h_bsrVal, 8*sizeof(float), d_bsrVal.get(), 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "nnzb=" << nnzb
              << " bsrRowPtr=[" << h_bsrRowPtr[0] << "," << h_bsrRowPtr[1] << "," << h_bsrRowPtr[2] << "]"
              << " bsrColInd=[" << h_bsrColInd[0] << "," << h_bsrColInd[1] << "]" << std::endl;

    aclsparseDestroyMatDescr(descrA);
    aclsparseDestroyMatDescr(descrC);
    return 0;
}

int main() {
    AclContext ctx(0);
    CHECK_ACL(ctx.Init(), "AclContext Init failed");
    return Csr2GebsrTest(ctx);
}
```

## 编译与运行

```bash
# 编译算子包
bash build.sh --pkg --soc=ascend950 --ops=csr2gebsr

# 运行测试
bash build.sh --soc=ascend950 --ops=csr2gebsr --run
```

## 测试覆盖

测试文件位于 `test/csr2gebsr/`，采用 GTest + CSV 参数化测试框架，在 arch35 实测设备上全部通过。

### 测试结构

| 文件 | 说明 |
|------|------|
| `csr2gebsr_param.h` | CSV 参数结构体 |
| `csr2gebsr_golden.h` | CPU golden 计算 |
| `dtype_utils.h` | dtype 转换工具（float ↔ FP16/BF16/INT32） |
| `arch35/csr2gebsr_test.csv` | L0/L1 正确性用例 |
| `arch35/csr2gebsr_npu_wrapper.h` | NPU 三步法执行封装 |
| `arch35/csr2gebsr_test.cpp` | GTest 主测试（含异常测试 + 性能采集） |

### L0/L1 正确性测试

| 级别 | 覆盖维度 |
|------|----------|
| L0 | 基本功能：FP32/FP16/BF16/INT32 四类型 × ROW/COLUMN × 0-based/1-based |
| L0 | 块维度：1×1 / 2×2 / 2×3 / 3×2 / 3×3 / 4×4 / 1×2 / 2×1 |
| L0 | 边界：空矩阵 / m=0 / n=0 / m=n=0 / 单元素 / 含空行 / 对角矩阵 / 全满矩阵 |
| L0 | 索引基组合：(0,0) / (1,1) / (0,1) / (1,0) |
| L1 | 中等规模：300×300 / 516×516 / 1000×10 |
| L1 | 大规模：302×302 / 1000×10 b1×1（PrefixSum 并行路径） |

### L2 异常测试

| 测试类别 | 覆盖内容 |
|----------|----------|
| 空指针 | handle / descrA / descrC / csrRowPtrA / bsrRowPtrC / nnzTotalDevHostPtr / pBuffer / pBufferSizeInBytes 为 nullptr |
| 非法维度 | m < 0 / n < 0 |
| 非法块维度 | rowBlockDim == 0 / colBlockDim == 0 |
| 非法枚举 | dir 为非法值 / indexBase 为非法值 |
| 矩阵类型 | type 非 GENERAL（SYMMETRIC） |
| 维度溢出 | m/n/blockDim > INT32_MAX/2 / mb×nb > INT32_MAX |
| Convert 空指针 | csrValA / csrColIndA / bsrValC / bsrColIndC 为 nullptr（nnz > 0 时） |

### 数据类型覆盖

| 数据类型 | 用例数 | 说明 |
|----------|--------|------|
| FP32 | 14 | 主类型，覆盖全维度组合 |
| FP16 | 4 | 含 COLUMN / 1-based |
| BF16 | 4 | 含 COLUMN / 1-based |
| INT32 | 4 | 含 COLUMN / 1-based |

### 特殊路径测试

| 测试 | 覆盖内容 |
|------|----------|
| PointerModeDeviceNnzb | DEVICE 指针模式 D2D 回传 nnzb（非空 + m=0 空矩阵） |
| NnzNullCsrColIndAEmpty | nnz=0 + null csrColIndA 合法路径 |
| PerfSweep | 性能采集框架（warmup + measure，输出 JSON） |

## 目录结构

```
sparse/csr2gebsr/
├── README.md
└── arch35/
    ├── csr2gebsr.h
    ├── csr2gebsr_host.cpp
    ├── csr2gebsr_kernel.cpp
    ├── csr2gebsr_kernel.h
    └── csr2gebsr_tiling_data.h

test/csr2gebsr/
├── CMakeLists.txt
├── csr2gebsr_golden.h
├── csr2gebsr_param.h
├── dtype_utils.h
└── arch35/
    ├── csr2gebsr_npu_wrapper.h
    ├── csr2gebsr_test.cpp
    └── csr2gebsr_test.csv
```
