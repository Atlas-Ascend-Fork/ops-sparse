# aclsparseSpMVOp 算子文档

## 算子概述

**稀疏矩阵-向量乘法（带独立输出向量 + 可定制 epilogue）**。

计算如下表达式：

$$
\mathbf{Z}_i = \text{epilogue}\!\left(\alpha \sum_k \text{op}(A_{ik}) \cdot X_k + \beta \cdot Y_i\right)
$$

其中：
- $A$ 是 $m \times k$ 的稀疏矩阵（CSR 格式）
- $X$ 是长度 $k$ 的稠密向量（输入）
- $Y$ 是长度 $m$ 的稠密向量（额外的加法输入，与标准 SpMV 中"输出"角色不同）
- $Z$ 是长度 $m$ 的稠密向量（输出，可与 $Y$ 别名实现 in-place）
- $\alpha$, $\beta$ 是标量，可为 HOST 或 DEVICE 指针
- $\text{epilogue}$ 为自定义 elementwise 函数（NPU 侧固定为 identity，即不附加操作）

本算子对标 cuSPARSE `cusparseSpMVOp()` [EXPERIMENTAL]，主要特性：
- **确定性（bit-wise 可重复）**：ALG1/ALG2 均保证
- **高精度累加**：使用 Priest 双重补偿点积（FP32 累加精度等效 rtol ≤ 1e-6）
- **支持异步执行**：全部 7 个 API 函数均异步返回
- **支持未排序索引**
- **支持值更新（ALG1）**：可在不重建 descr/plan 的情况下更新 CSR 值数组

## 接口列表

| 接口名 | 说明 |
|--------|------|
| `aclsparseSpMVOp_bufferSize` | 获取 workspace 大小 |
| `aclsparseSpMVOp_createDescr` | 创建描述符并执行预处理 |
| `aclsparseSpMVOp_destroyDescr` | 销毁描述符 |
| `aclsparseSpMVOp_createPlan` | 创建执行计划 |
| `aclsparseSpMVOp_destroyPlan` | 销毁执行计划 |
| `aclsparseSpMVOp_setGlobalUserData` | 设置 epilogue 用户数据（NPU 侧为 no-op） |
| `aclsparseSpMVOp` | 执行计算 |

## 产品支持情况

| 芯片系列 | 支持情况 |
|---------|---------|
| Ascend950 (arch35) | ✅ |

## 接口详情

### aclsparseSpMVOp_bufferSize

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMVOp_bufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseDnVecDescr_t vecZ,
    aclDataType computeType,
    aclsparseSpMVOpAlg_t alg,
    size_t *bufferSize);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | IN | `aclsparseHandle_t` | aclsparse 句柄 | Host |
| opA | IN | `aclsparseOperation_t` | op(A)，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE` | Host |
| matA | IN | `aclsparseConstSpMatDescr_t` | CSR 稀疏矩阵描述符 | Host |
| vecX | IN | `aclsparseConstDnVecDescr_t` | 稠密向量 X（可为 NULL） | Host |
| vecY | IN | `aclsparseConstDnVecDescr_t` | 稠密向量 Y（可为 NULL） | Host |
| vecZ | IN | `aclsparseDnVecDescr_t` | 稠密向量 Z（可为 NULL） | Host |
| computeType | IN | `aclDataType` | 计算类型，仅 `ACL_FLOAT` | Host |
| alg | IN | `aclsparseSpMVOpAlg_t` | 算法选择 | Host |
| bufferSize | OUT | `size_t *` | 返回所需 workspace 字节数 | Host |

#### 约束说明

| 约束 | 说明 |
|------|------|
| ALG1/DEFAULT | `bufferSize` 返回 0（无需 workspace） |
| ALG2 | `bufferSize` 返回 header + reorder + tmpReorder + scratch + tmpScratch + bin_edge 六块总大小（按 64 字节对齐） |
| matA 格式 | 必须为 CSR |
| matA 值类型 | 必须为 FP32 |
| colInd 类型 | 必须为 int32 |
| rowOffsets 类型 | int32 或 int64 |

---

### aclsparseSpMVOp_createDescr

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMVOp_createDescr(
    aclsparseHandle_t handle,
    aclsparseSpMVOpDescr_t *descr,
    aclsparseOperation_t opA,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseDnVecDescr_t vecZ,
    aclDataType computeType,
    aclsparseSpMVOpAlg_t alg,
    void *buffer);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | IN | `aclsparseHandle_t` | aclsparse 句柄 | Host |
| descr | OUT | `aclsparseSpMVOpDescr_t *` | 输出描述符 | Host |
| opA | IN | `aclsparseOperation_t` | op(A) | Host |
| matA | IN | `aclsparseConstSpMatDescr_t` | CSR 稀疏矩阵描述符 | Host |
| vecX | IN | `aclsparseConstDnVecDescr_t` | 稠密向量 X | Host |
| vecY | IN | `aclsparseConstDnVecDescr_t` | 稠密向量 Y | Host |
| vecZ | IN | `aclsparseDnVecDescr_t` | 稠密向量 Z | Host |
| computeType | IN | `aclDataType` | 计算类型 | Host |
| alg | IN | `aclsparseSpMVOpAlg_t` | 算法选择 | Host |
| buffer | IN | `void *` | workspace buffer（ALG2 必需） | Device |

#### 约束说明

| 约束 | 说明 |
|------|------|
| ALG2 必需 buffer | ALG2 算法时 buffer 不可为 NULL，且 ≥ bufferSize 返回值 |
| buffer 生命周期 | buffer 必须在 descr 销毁前保持有效 |
| 异步 launch | 若 ALG2 预处理包含 kernel launch，将在 stream 上异步执行 |

---

### aclsparseSpMVOp_destroyDescr

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMVOp_destroyDescr(aclsparseSpMVOpDescr_t descr);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| descr | IN | `aclsparseSpMVOpDescr_t` | 要销毁的描述符 | Host |

#### 约束说明

| 约束 | 说明 |
|------|------|
| 幂等语义 | descr 为 nullptr 时直接返回 SUCCESS |

---

### aclsparseSpMVOp_createPlan

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMVOp_createPlan(
    aclsparseHandle_t handle,
    aclsparseSpMVOpDescr_t descr,
    aclsparseSpMVOpPlan_t *plan,
    const void *epilogueLTOBuffer,
    size_t epilogueLTOBufferSize);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | IN | `aclsparseHandle_t` | aclsparse 句柄 | Host |
| descr | IN | `aclsparseSpMVOpDescr_t` | 来自 createDescr 的描述符（非 const，对标 cuSPARSE） | Host |
| plan | OUT | `aclsparseSpMVOpPlan_t *` | 输出执行计划 | Host |
| epilogueLTOBuffer | IN | `const void *` | epilogue LTO-IR（NPU 侧必须为 NULL） | Host |
| epilogueLTOBufferSize | IN | `size_t` | LTO-IR 大小（NPU 侧必须为 0） | Host |

#### 约束说明

| 约束 | 说明 |
|------|------|
| epilogue 不支持 | NPU 侧使用 identity epilogue；传入非 NULL LTO-IR 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` |

---

### aclsparseSpMVOp_destroyPlan

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMVOp_destroyPlan(aclsparseSpMVOpPlan_t plan);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| plan | IN | `aclsparseSpMVOpPlan_t` | 要销毁的执行计划 | Host |

#### 约束说明

| 约束 | 说明 |
|------|------|
| 幂等语义 | plan 为 nullptr 时直接返回 SUCCESS |

---

### aclsparseSpMVOp_setGlobalUserData

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMVOp_setGlobalUserData(
    aclsparseHandle_t handle,
    aclsparseSpMVOpPlan_t plan,
    const char *epilogueDataName,
    void *epilogueData,
    size_t epilogueDataSize);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | IN | `aclsparseHandle_t` | aclsparse 句柄 | Host |
| plan | IN | `aclsparseSpMVOpPlan_t` | 执行计划 | Host |
| epilogueDataName | IN | `const char *` | epilogue 数据变量名 | Host |
| epilogueData | IN | `void *` | epilogue 数据（非 const，对标 cuSPARSE） | Host |
| epilogueDataSize | IN | `size_t` | 数据字节数 | Host |

#### 约束说明

| 约束 | 说明 |
|------|------|
| NPU 侧为 no-op | identity epilogue 无需辅助数据，直接返回 SUCCESS |

---

### aclsparseSpMVOp（主执行入口）

#### 函数原型

```c
aclsparseStatus_t aclsparseSpMVOp(
    aclsparseHandle_t handle,
    aclsparseSpMVOpPlan_t plan,
    const void *alpha,
    const void *beta,
    aclsparseConstDnVecDescr_t vecX,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseDnVecDescr_t vecZ);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| handle | IN | `aclsparseHandle_t` | aclsparse 句柄 | Host |
| plan | IN | `aclsparseSpMVOpPlan_t` | 执行计划 | Host |
| alpha | IN | `const void *` | 标量 α（FP32 类型，HOST 或 DEVICE 指针，由 handle pointerMode 决定） | Host/Device |
| beta | IN | `const void *` | 标量 β（FP32 类型，HOST 或 DEVICE 指针） | Host/Device |
| vecX | IN | `aclsparseConstDnVecDescr_t` | 稠密向量 X | Host |
| vecY | IN | `aclsparseConstDnVecDescr_t` | 稠密向量 Y | Host |
| vecZ | IN/OUT | `aclsparseDnVecDescr_t` | 稠密向量 Z（输出） | Host |

#### 约束说明

| 约束 | 说明 |
|------|------|
| alpha/beta 指针类型 | 由 `aclsparseSetPointerMode` 设置，HOST 时直接解引用，DEVICE 时从 Device 内存读 |
| vecZ 别名 | vecZ 可与 vecY 共享 Device 内存（in-place 语义） |
| 异步返回 | 调用后 kernel 入队到 handle 关联的 stream，函数立即返回 |
| 确定性 | ALG1/ALG2 均提供 bit-wise 可重复结果 |

## 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 唯一支持的格式；colIndType 必须为 int32，rowOffsetsType 可为 int32 或 int64 |
| COO | ❌ | |
| CSC | ❌ | |
| BSR | ❌ | |
| Blocked-ELL | ❌ | |
| Sliced-ELL | ❌ | |

## 支持的算法

| 算法 | 说明 |
|------|------|
| `ACL_SPARSE_SPMVOP_ALG_DEFAULT` | 默认算法，当前等同于 ALG1 |
| `ACL_SPARSE_SPMVOP_ALG1` | 确定性算法；支持不重建 descr/plan 的情况下更新 CSR 值数组 |
| `ACL_SPARSE_SPMVOP_ALG2` | 确定性算法；按 nnz 负载均衡切分，性能可能更高，但不支持值更新且需要更大 workspace |

## 调用示例

```cpp
#include <stdio.h>
#include <stdlib.h>
#include "cann_ops_sparse.h"
#include "acl/acl.h"

// 错误宏：简化返回值检查
#define CHECK_STATUS(call)                                                      \
    do {                                                                        \
        aclsparseStatus_t _st = (call);                                         \
        if (_st != ACL_SPARSE_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "ERROR: %s failed at line %d, status=%d\n",         \
                    #call, __LINE__, static_cast<int>(_st));                     \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CHECK_ALLOC(ptr)                                                        \
    do {                                                                        \
        if ((ptr) == nullptr) {                                                 \
            fprintf(stderr, "ERROR: allocation failed at line %d\n", __LINE__); \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CHECK_ACL_RT(call)                                                      \
    do {                                                                        \
        aclError _st = (call);                                                  \
        if (_st != ACL_SUCCESS) {                                               \
            fprintf(stderr, "ERROR: %s failed at line %d, status=%d\n",         \
                    #call, __LINE__, static_cast<int>(_st));                     \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main() {
    aclrtStream stream = nullptr;  // 此处省略 stream 创建
    int64_t rows = 4, cols = 4, nnz = 5;

    // ...此处省略设备内存分配...
    int32_t *csrRowOffsets = nullptr;  // Device pointer
    int32_t *csrColInd = nullptr;       // Device pointer
    float *csrValues = nullptr;         // Device pointer
    float *x = nullptr, *y = nullptr, *z = nullptr;  // Device pointers

    // 1. 创建 handle
    aclsparseHandle_t handle = nullptr;
    CHECK_STATUS(aclsparseCreate(&handle));
    CHECK_STATUS(aclsparseSetStream(handle, stream));

    // 2. 创建 CSR 稀疏矩阵描述符 matA
    aclsparseConstSpMatDescr_t matA;
    CHECK_STATUS(aclsparseCreateConstCsr(&matA, rows, cols, nnz,
        csrRowOffsets, csrColInd, csrValues,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT));

    // 3. 创建稠密向量描述符
    aclsparseConstDnVecDescr_t vecX, vecY;
    aclsparseDnVecDescr_t vecZ;
    CHECK_STATUS(aclsparseCreateConstDnVec(&vecX, cols, x, ACL_FLOAT));
    CHECK_STATUS(aclsparseCreateConstDnVec(&vecY, rows, y, ACL_FLOAT));
    CHECK_STATUS(aclsparseCreateDnVec(&vecZ, rows, z, ACL_FLOAT));

    // 4. 获取 workspace 大小
    size_t bufferSize = 0;
    CHECK_STATUS(aclsparseSpMVOp_bufferSize(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA, vecX, vecY, vecZ, ACL_FLOAT, ACL_SPARSE_SPMVOP_ALG_DEFAULT, &bufferSize));

    // 5. 分配 workspace（ALG1 时 bufferSize=0，可传 nullptr）
    void *buffer = nullptr;
    if (bufferSize > 0) {
        CHECK_ACL_RT(aclrtMalloc(&buffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // 6. 创建描述符（ALG2 时执行异步预处理 kernel）
    aclsparseSpMVOpDescr_t descr = nullptr;
    CHECK_STATUS(aclsparseSpMVOp_createDescr(handle, &descr, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA, vecX, vecY, vecZ, ACL_FLOAT, ACL_SPARSE_SPMVOP_ALG_DEFAULT, buffer));

    // 7. 创建执行计划（identity epilogue: NULL, 0）
    aclsparseSpMVOpPlan_t plan = nullptr;
    CHECK_STATUS(aclsparseSpMVOp_createPlan(handle, descr, &plan, nullptr, 0));

    // 8. 执行（Z = alpha * A * X + beta * Y）
    float alpha = 1.0f, beta = 0.0f;
    CHECK_STATUS(aclsparseSpMVOp(handle, plan, &alpha, &beta, vecX, vecY, vecZ));

    // 9. 同步
    CHECK_ACL_RT(aclrtSynchronizeStream(stream));

    // 10. 清理（按创建顺序逆序销毁）
    aclsparseSpMVOp_destroyPlan(plan);
    aclsparseSpMVOp_destroyDescr(descr);
    if (buffer) aclrtFree(buffer);
    aclsparseDestroyDnVec(vecZ);
    aclsparseDestroyDnVec(vecY);
    aclsparseDestroyDnVec(vecX);
    aclsparseDestroySpMat(matA);
    aclsparseDestroy(handle);

    return 0;
}

```