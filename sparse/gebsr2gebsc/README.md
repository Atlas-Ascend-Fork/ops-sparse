# Gebsr2Gebsc算子

## 算子概述

gebsr2gebsc 算子用于将 GEBSR（General Block Sparse Row）格式的稀疏矩阵转换为 GEBSC（General Block Sparse Column）格式。当将每个 block 视为标量时，该操作等价于 csr2csc：块稀疏模式被转置（block (i,j) -> (j,i)），但块内内存布局不变。参考 CUDA cusparse 的 `cusparse<t>gebsr2gebsc` 接口。

块值拷贝模式由 `rowBlockDimC`/`colBlockDimC` 与 `rowBlockDimA`/`colBlockDimA` 的关系决定：

- **Direct copy**（`rC==rA && cC==cA`）：块元素原样拷贝
- **Block transpose**（`rC==cA && cC==rA`）：块元素转置

`dirA` 指定块内内存布局（ROW=行主序, COLUMN=列主序），输入输出布局不变。

### 支持数据类型

| 数据类型 | 枚举值 | 元素大小 |
|----------|--------|----------|
| float32 | ACL_FLOAT | 4B |
| float16 | ACL_FLOAT16 | 2B |
| bfloat16 | ACL_BF16 | 2B |
| int32 | ACL_INT32 | 4B |
| int8 | ACL_INT8 | 1B |

### 支持产品系列

| 产品系列 | NPU 架构 | 架构目录 |
|----------|----------|----------|
| Ascend 950PR&950DT系列产品 | dav-3510 | arch35 |

## 算子执行接口

算子共提供 2 个公开接口（type-generic，通过 valType 参数支持多种数据类型）：

| 接口 | 说明 |
|------|------|
| `aclsparseGebsr2gebsc_bufferSize` | 查询 workspace 大小 |
| `aclsparseGebsr2gebsc` | 执行 GEBSR→GEBSC 转换 |

### aclsparseGebsr2gebsc_bufferSize

#### 产品支持情况

| 产品系列 | 支持状态 |
|----------|----------|
| Ascend 950PR&950DT系列产品 | 支持 |

#### 函数原型

```cpp
aclsparseStatus_t aclsparseGebsr2gebsc_bufferSize(
    aclsparseHandle_t handle, int mb, int nb, int nnzb,
    const void *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA,
    int rowBlockDimA, int colBlockDimA,
    aclsparseDirection_t dirA,
    aclDataType valType,
    size_t *pBufferSizeInBytes);
```

#### 参数说明

| 参数 | 输入/输出 | 类型 | 说明 |
|------|-----------|------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，不可为 nullptr |
| mb | 输入 | int | 块行数，mb >= 0 |
| nb | 输入 | int | 块列数，nb >= 0 |
| nnzb | 输入 | int | 非零块数，nnzb >= 0 |
| bsrValA | 输入 | const void* | GEBSR 非零块值数组（本函数不读取，可为 nullptr） |
| bsrRowPtrA | 输入 | const int* | 块行偏移数组，长度 mb+1（本函数不读取，可为 nullptr） |
| bsrColIndA | 输入 | const int* | 块列索引数组，长度 nnzb（本函数不读取，可为 nullptr） |
| rowBlockDimA | 输入 | int | 输入块行维，> 0 |
| colBlockDimA | 输入 | int | 输入块列维，> 0 |
| dirA | 输入 | aclsparseDirection_t | 块内内存布局（ROW 或 COLUMN） |
| valType | 输入 | aclDataType | 数据类型（FP32/FP16/BF16/INT32/INT8） |
| pBufferSizeInBytes | 输出 | size_t* | 所需 workspace 字节数，不可为 nullptr |

#### 约束说明

- `mb >= 0`，`nb >= 0`，`nnzb >= 0`
- `rowBlockDimA > 0`，`colBlockDimA > 0`
- `nb <= INT32_MAX - 1`
- `mb == 0` 时 `nnzb` 必为 0，`nb == 0` 时 `nnzb` 必为 0
- `valType` 仅支持 ACL_FLOAT / ACL_FLOAT16 / ACL_BF16 / ACL_INT32 / ACL_INT8
- `mb == 0 || nb == 0 || nnzb == 0` 时 workspace 大小为 0

### aclsparseGebsr2gebsc

#### 产品支持情况

| 产品系列 | 支持状态 |
|----------|----------|
| Ascend 950PR&950DT系列产品 | 支持 |

#### 函数原型

```cpp
aclsparseStatus_t aclsparseGebsr2gebsc(
    aclsparseHandle_t handle, int mb, int nb, int nnzb,
    const void *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA,
    int rowBlockDimA, int colBlockDimA,
    void *bscVal, int *bscColPtr, int *bscRowInd,
    int rowBlockDimC, int colBlockDimC,
    aclsparseAction_t copyValues, aclsparseIndexBase_t idxBase,
    aclsparseDirection_t dirA,
    aclDataType valType,
    void *pBuffer);
```

#### 参数说明

| 参数 | 输入/输出 | 类型 | 说明 |
|------|-----------|------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，不可为 nullptr |
| mb | 输入 | int | 块行数，mb >= 0 |
| nb | 输入 | int | 块列数，nb >= 0 |
| nnzb | 输入 | int | 非零块数，nnzb >= 0 |
| bsrValA | 输入 | const void* | GEBSR 非零块值数组，长度 nnzb*rowBlockDimA*colBlockDimA |
| bsrRowPtrA | 输入 | const int* | 块行偏移数组，长度 mb+1 |
| bsrColIndA | 输入 | const int* | 块列索引数组，长度 nnzb |
| rowBlockDimA | 输入 | int | 输入块行维，> 0 |
| colBlockDimA | 输入 | int | 输入块列维，> 0 |
| bscVal | 输出 | void* | GEBSC 非零块值数组，长度 nnzb*rowBlockDimC*colBlockDimC |
| bscColPtr | 输出 | int* | 块列偏移数组，长度 nb+1 |
| bscRowInd | 输出 | int* | 块行索引数组，长度 nnzb |
| rowBlockDimC | 输入 | int | 输出块行维，> 0 |
| colBlockDimC | 输入 | int | 输出块列维，> 0 |
| copyValues | 输入 | aclsparseAction_t | SYMBOLIC（仅结构）或 NUMERIC（结构+值） |
| idxBase | 输入 | aclsparseIndexBase_t | 索引基址（0-based 或 1-based） |
| dirA | 输入 | aclsparseDirection_t | 块内内存布局（ROW 或 COLUMN） |
| valType | 输入 | aclDataType | 数据类型（FP32/FP16/BF16/INT32/INT8） |
| pBuffer | 输入 | void* | workspace，大小由 bufferSize 接口查询 |

#### 约束说明

- `mb >= 0`，`nb >= 0`，`nnzb >= 0`
- `rowBlockDimA > 0`，`colBlockDimA > 0`，`rowBlockDimC > 0`，`colBlockDimC > 0`
- `nb <= INT32_MAX - 1`
- `nnzb + idxBase <= INT32_MAX`
- `mb == 0` 时 `nnzb` 必为 0，`nb == 0` 时 `nnzb` 必为 0
- 输出块维度必须满足 direct copy（`rC==rA && cC==cA`）或 transpose（`rC==cA && cC==rA`），否则返回 NOT_SUPPORTED
- `mb > 0` 时 `bsrRowPtrA` 不可为 nullptr
- `nnzb > 0` 时 `bsrColIndA` 不可为 nullptr
- `nnzb > 0 && NUMERIC` 时 `bsrValA` 不可为 nullptr
- `bscColPtr` 不可为 nullptr
- `nnzb > 0` 时 `bscRowInd` 不可为 nullptr
- `nnzb > 0 && NUMERIC` 时 `bscVal` 不可为 nullptr

### 返回值

| 返回值 | 说明 |
|--------|------|
| ACL_SPARSE_STATUS_SUCCESS | 操作成功 |
| ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR | handle 为 nullptr |
| ACL_SPARSE_STATUS_INVALID_VALUE | 参数非法（负值、非法枚举、nullptr 指针、数据契约违反） |
| ACL_SPARSE_STATUS_NOT_SUPPORTED | 不支持的 valType 或块维度组合 |
| ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES | workspace 不足 |
| ACL_SPARSE_STATUS_EXECUTION_FAILED | aclrt API 调用失败 |
| ACL_SPARSE_STATUS_INTERNAL_ERROR | 内部错误（如获取核数失败） |

### workspace 计算公式

- workspace 大小 = `(1 + stripeCount) × (nb + 1) × sizeof(int32_t)`
- `stripeCount = min(ceil(nnzb/256), aivCoreNum, 16MB/((nb+1)×4))`
- `pBuffer` 可为 nullptr，此时使用 handle 内置 workspace（需足够大）

### 实现架构

采用与 csr2csc_ex2 相同的五阶段 kernel 流水线（块级结构转换等价）：

1. **CountCols**：遍历 bsrColIndA，原子加统计每 stripe 列直方图
2. **SumStripeHist**：对 stripeHist 按列求和重建 colCount
3. **PrefixSum**：单 warp 并行 exclusive prefix sum -> bscColPtr
4. **StripeBase**：stripe 直方图按列前缀和 -> 每 stripe 写游标基址
5. **Scatter**：每 block 单线程顺序 scatter 写 bscRowInd / bscVal（含块值 direct copy / transpose）

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

```cpp
#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include <iostream>

#define CHECK_ACL(x) do { aclError __e = (x); if (__e != ACL_SUCCESS) { \
    std::cerr << "ACL error " << __e << " at line " << __LINE__ << std::endl; return 1; } } while(0)

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclsparseHandle_t handle;
    if (aclsparseCreate(&handle) != ACL_SPARSE_STATUS_SUCCESS) return 1;
    aclsparseSetStream(handle, stream);

    int mb = 2, nb = 2, nnzb = 3;
    int rowBlockDimA = 2, colBlockDimA = 2;
    int bsrRowPtrA[] = {0, 2, 3};
    int bsrColIndA[] = {0, 1, 1};
    float bsrValA[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};

    int *d_rowPtr = nullptr, *d_colInd = nullptr;
    float *d_bsrVal = nullptr;
    CHECK_ACL(aclrtMalloc((void**)&d_rowPtr, 3*sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&d_colInd, 3*sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&d_bsrVal, 12*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(d_rowPtr, 3*sizeof(int), bsrRowPtrA, 3*sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_colInd, 3*sizeof(int), bsrColIndA, 3*sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_bsrVal, 12*sizeof(float), bsrValA, 12*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    int *d_bscColPtr = nullptr, *d_bscRowInd = nullptr;
    float *d_bscVal = nullptr;
    CHECK_ACL(aclrtMalloc((void**)&d_bscColPtr, 3*sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&d_bscRowInd, 3*sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&d_bscVal, 12*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));

    size_t bufSize = 0;
    if (aclsparseGebsr2gebsc_bufferSize(handle, mb, nb, nnzb,
        d_bsrVal, d_rowPtr, d_colInd, rowBlockDimA, colBlockDimA,
        ACL_SPARSE_DIRECTION_ROW, ACL_FLOAT, &bufSize) != ACL_SPARSE_STATUS_SUCCESS) return 1;

    void *d_buffer = nullptr;
    if (bufSize > 0) CHECK_ACL(aclrtMalloc(&d_buffer, bufSize, ACL_MEM_MALLOC_HUGE_FIRST));

    if (aclsparseGebsr2gebsc(handle, mb, nb, nnzb,
        d_bsrVal, d_rowPtr, d_colInd, rowBlockDimA, colBlockDimA,
        d_bscVal, d_bscColPtr, d_bscRowInd,
        rowBlockDimA, colBlockDimA,
        ACL_SPARSE_ACTION_NUMERIC, ACL_SPARSE_INDEX_BASE_ZERO,
        ACL_SPARSE_DIRECTION_ROW, ACL_FLOAT, d_buffer) != ACL_SPARSE_STATUS_SUCCESS) return 1;

    CHECK_ACL(aclrtSynchronizeStream(stream));

    if (d_buffer) aclrtFree(d_buffer);
    aclrtFree(d_rowPtr); aclrtFree(d_colInd); aclrtFree(d_bsrVal);
    aclrtFree(d_bscColPtr); aclrtFree(d_bscRowInd); aclrtFree(d_bscVal);
    aclsparseDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```

## 编译与运行

```bash
# 编译算子包
bash build.sh --pkg --soc=ascend950 --ops=gebsr2gebsc

# 运行测试
bash build.sh --soc=ascend950 --ops=gebsr2gebsc --run
```

## 测试覆盖

测试文件位于 `test/gebsr2gebsc/`，采用 GTest + CSV 参数化测试框架，在 arch35 实测设备上全部通过。

### 测试结构

| 文件 | 说明 |
|------|------|
| `gebsr2gebsc_param.h` | CSV 参数结构体 |
| `gebsr2gebsc_golden.h` | CPU golden 计算（含 Eigen 交叉校验） |
| `arch35/gebsr2gebsc_test.csv` | L0/L1 正确性用例 |
| `arch35/gebsr2gebsc_npu_wrapper.h` | NPU 输入生成 / 类型映射 / 执行封装 |
| `arch35/gebsr2gebsc_test.cpp` | GTest 主测试（含异常测试） |

### L0/L1 正确性测试

| 级别 | 覆盖维度 |
|------|----------|
| L0 | 基本功能：FP32/FP16/BF16/INT32/INT8 五类型 × direct copy/block transpose × ROW/COLUMN × SYMBOLIC/NUMERIC × base 0/1 |
| L0 | 块维度：1×1 / 2×2 / 2×3 / 3×2 / 2×4 / 4×2 / 3×3 / 4×4 / 3×5 / 5×3 / 4×8 / 8×4 |
| L0 | 边界：空矩阵 / 单块 / 全满 / 全空块 / 极宽/极高矩阵 |
| L1 | 中等规模：64×64 ~ 128×128 × direct/transpose × ROW/COLUMN |
| L1 | 高稀疏度：0.9 / 0.95 / 0.99 × 大矩阵 |
| L1 | 大规模：256×256 × direct/transpose |

### L2 异常测试

| 测试类别 | 覆盖内容 |
|----------|----------|
| 空指针 | handle / pBufferSizeInBytes / bscColPtr 为 nullptr |
| 非法维度 | mb < 0 / nnzb < 0 |
| 非法块维度 | rowBlockDimA == 0 |
| 非法枚举 | dirA 为非法值 |
| 不支持组合 | 块维度不满足 direct copy 或 transpose |

### 数据类型覆盖

| 数据类型 | 用例数 | 说明 |
|----------|--------|------|
| FP32 | 123 | 主类型，覆盖全维度组合 |
| FP16 | 12 | 含 transpose / 1-based / symbolic |
| BF16 | 12 | 含 transpose / 1-based / symbolic |
| INT32 | 12 | 含 transpose / 1-based / symbolic |
| INT8 | 12 | 含 transpose / 1-based / symbolic |

## 目录结构

```
sparse/gebsr2gebsc/
├── README.md
└── arch35/
    ├── gebsr2gebsc_host.cpp
    ├── gebsr2gebsc_kernel.cpp
    ├── gebsr2gebsc_kernel.h
    └── gebsr2gebsc_tiling_data.h

test/gebsr2gebsc/
├── CMakeLists.txt
├── gebsr2gebsc_golden.h
├── gebsr2gebsc_param.h
└── arch35/
    ├── gebsr2gebsc_npu_wrapper.h
    ├── gebsr2gebsc_test.cpp
    └── gebsr2gebsc_test.csv
```
