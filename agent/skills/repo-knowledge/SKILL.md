---
name: repo-knowledge
description: 仓库领域知识，提供本仓算子涉及的领域标准、概念与背景（cuSPARSE 接口规范、稀疏算子 API 体系）。
---

# ops-sparse 库规范

ops-sparse 对齐 cuSPARSE，支持两种 API 体系。开发新算子时，**按照目标 cuSPARSE 接口的归属选择对应规范**：

| API 体系 | 适用场景 | 特征 |
|----------|---------|------|
| **Generic API** | cuSPARSE Generic API 中的算子（SpMV/SpMM/SpGEMM/SDDMM/SpSM 等） | Descriptor 模式 + 三阶段执行 |
| **Legacy API** | cuSPARSE Legacy API 中的算子（gtsv/gtsv2 三对角、格式转换、排序等，尚未迁移到 Generic） | 直传扁平参数 + MatDescr 矩阵描述符 |

---

## 一、Generic API 规范

### 1.1 命名模式

```
aclsparse{Operation}
```

| 分类 | 命名模式 | 示例 |
|------|----------|------|
| 矩阵-向量运算 | aclsparseSp{MV/SV} | aclsparseSpMV |
| 矩阵-矩阵运算 | aclsparseSp{MM/SM/GEMM} | aclsparseSpMM |
| 描述符管理 | aclsparseCreate/Destroy{Type} | aclsparseCreateCsr |
| 资源管理 | aclsparseCreate/Destroy | aclsparseCreate |
| SDDMM | aclsparseSDDMM | aclsparseSDDMM |

### 1.2 前缀规范

- **aclsparse**：库前缀，小写，无空格
- **SpMV**：Sparse Matrix-Vector multiplication
- **SpMM**：Sparse Matrix-Matrix multiplication
- **SpSV**：Sparse triangular Solve (Vector)
- **SpSM**：Sparse triangular Solve (Matrix)
- **SpGEMM**：Sparse General Matrix-Matrix multiplication
- **SDDMM**：Sampled Dense-Dense Matrix Multiplication

### 1.3 Generic 描述符规范

| 描述符 | 用途 | 关键字段 |
|--------|------|----------|
| `aclsparseSpMatDescr_t` | 稀疏矩阵 | format, rows, cols, nnz, ptrs, idxs, values, baseType, ptrType, IdxType, valueType |
| `aclsparseDnVecDescr_t` | 稠密向量 | nums, values, valueType |
| `aclsparseDnMatDescr_t` | 稠密矩阵 | rows, cols, ld, order, values, valueType |

**生命周期**：

1. **创建**：`aclsparseCreateXxx(&descr, ...)` — host 侧分配描述符结构体，填充字段
2. **使用**：描述符作为 opaque pointer 传入计算函数
3. **销毁**：`aclsparseDestroyXxx(descr)` — 释放描述符结构体（注意 const 转换）

**规范**：
- Create 函数参数中的指针类型描述符必须校验 nullptr
- Destroy 函数必须安全处理 nullptr 输入（直接返回 SUCCESS）
- Const 描述符（`aclsparseConstXxxDescr_t`）表示只读访问
- `aclsparseConstSpMatDescr_t` 用于 SpMV/SpMM 等计算函数中的稀疏矩阵输入
- `aclsparseConstDnVecDescr_t` 用于计算函数中的输入稠密向量
- 非 const 描述符用于输出（如 SpMV 的输出向量 `vecY`）

### 1.4 Generic 参数顺序

```c
aclsparseStatus_t aclsparseXxx(
    aclsparseHandle_t handle,           // 1. Handle
    aclsparseOperation_t op,            // 2. 操作类型（转置等）
    const void *alpha,                  // 3. 标量 alpha（Host 指针）
    aclsparseConstSpMatDescr_t matA,    // 4. 稀疏矩阵描述符
    aclsparseConstDnVecDescr_t vecX,    // 5. 输入向量
    const void *beta,                   // 6. 标量 beta
    aclsparseDnVecDescr_t vecY,         // 7. 输出向量
    aclDataType computeType,            // 8. 计算类型
    XxxAlg_t alg,                       // 9. 算法选择
    void *buffer                        // 10. 外部 buffer（可选）
);
```

**标量传递**：`alpha`/`beta` 为 `const void *` 类型，实际数据类型由 `computeType` 决定。

### 1.5 Generic 三阶段执行

| 阶段 | 函数 | 作用 |
|------|------|------|
| 1. 询问 buffer | `aclsparseXxxGetBufferSize(...)` | 返回 workspace 所需大小 |
| 2. 预处理 | `aclsparseXxxPreprocess(...)` | 一次性预处理（格式转换/重排），写入 buffer |
| 3. 执行 | `aclsparseXxx(...)` | 实际计算，复用 buffer（可多次调用） |

> 预处理阶段是可选的。如果算子不需要预处理，阶段 1 返回 bufferSize=0，阶段 2 不存在。

---

## 二、Legacy API 规范

### 2.1 命名模式

```
aclsparse{Type}{Operation}
```

按 cuSPARSE Legacy 命名，**{Type}** 表示精度前缀：

| 精度前缀 | 数据类型 | 示例 |
|---------|---------|------|
| S | FP32 (float) | aclsparseSgtsv2 |
| D | FP64 (double) | aclsparseDgtsv2 |
| C | complex FP32 | aclsparseCgtsv2 |
| Z | complex FP64 | aclsparseZgtsv2 |
| H | FP16 | aclsparseSgtsv2_half（扩展） |

**特殊前缀**（操作特定而与类型无关）：

| 前缀 | 用途 | 示例 |
|------|------|------|
| X | 与类型无关的辅助函数 | aclsparseXcsrsort |
| cusparse | 部分管理函数沿用 cuSPARSE 名称 | aclsparseSetMatType |

### 2.2 Legacy 函数分类

| 分类 | 说明 | 示例 |
|------|------|------|
| Level 2 | 稀疏矩阵-向量运算 | aclsparseSbsrmv（已 DEPRECATED，由 SpMV 替代） |
| Level 3 | 稀疏矩阵-矩阵运算 | aclsparseSbsrmm（已 DEPRECATED，由 SpMM 替代） |
| Preconditioners | 预条件子（IC/ILU） | csric02/csrilu02（已 DEPRECATED） |
| Tridiagonal Solve | 三对角求解 | aclsparseSgtsv2 / aclsparseSgtsv2StridedBatch |
| Pentadiagonal Solve | 五对角求解 | aclsparseSgpsvInterleavedBatch |
| Format Conversion | 格式转换 | aclsparseSgebsr2gebsc / aclsparseSnnz / aclsparse{t}coo2csr / aclsparse{t}csr2coo |
| Reorderings / Sort | 排序 | aclsparseXcsrsort / aclsparseXcoosort |
| Extra | 扩展（如 csrgeam2） | aclsparseScsrgeam2 |

### 2.3 Legacy MatDescr 矩阵描述符

Legacy 不使用 SpMatDescr，而是使用 **MatDescr（矩阵描述符）** 标注格式属性：

```c
typedef struct aclsparseMatDescr *aclsparseMatDescr_t;

aclsparseStatus_t aclsparseCreateMatDescr(aclsparseMatDescr_t *descr);
aclsparseStatus_t aclsparseDestroyMatDescr(aclsparseMatDescr_t descr);
void aclsparseSetMatType(aclsparseMatDescr_t descr, aclsparseMatrixType_t type);
void aclsparseSetMatIndexBase(aclsparseMatDescr_t descr, aclsparseIndexBase_t base);
void aclsparseSetMatDiagType(aclsparseMatDescr_t descr, aclsparseDiagType_t diagType);
void aclsparseSetMatFillMode(aclsparseMatDescr_t descr, aclsparseFillMode_t fillMode);
aclsparseMatrixType_t aclsparseGetMatType(aclsparseMatDescr_t descr);
aclsparseIndexBase_t aclsparseGetMatIndexBase(aclsparseMatDescr_t descr);
```

**MatDescr 关键字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| type | aclsparseMatrixType_t | GENERAL / SYMMETRIC / HERMITIAN / TRIANGULAR |
| indexBase | aclsparseIndexBase_t | ZERO / ONE |
| diagType | aclsparseDiagType_t | NON_UNIT / UNIT |
| fillMode | aclsparseFillMode_t | LOWER / UPPER |

### 2.4 Legacy 参数顺序

遵循 Sparse BLAS 风格：

```c
aclsparseStatus_t aclsparseSgtsv2(
    aclsparseHandle_t handle,       // 1. Handle
    int m,                          // 2. 矩阵行数
    const float *dl,                // 3. 下对角线 (m-1 元素)
    const float *d,                 // 4. 主对角线 (m 元素)
    const float *du,                // 5. 上对角线 (m-1 元素)
    float *x,                       // 6. 解向量 / 右端项 (m × nrhs)
    int nrhs,                       // 7. 右端项个数（列数）
    int ldx,                        // 8. 解向量 leading dimension（≥ m）
    size_t *pBufferSizeInBytes      // 9. workspace 大小（输入输出）
);
```

**参数顺序通用规则**：

1. `handle` 永远第一
2. 维度参数（m, n, k, nnz, nrhs）紧随其后
3. 数据指针（带 const 修饰输入）
4. leading dimension / stride 参数
5. 工作区大小（可选，输出参数）

### 2.5 Legacy Buffer 模式（两种）

Legacy 算子的 buffer 管理不如 Generic 的三阶段清晰，常见两种模式：

**模式 A：内联 bufferSize 参数**
```c
aclsparseStatus_t aclsparseSgtsv2_bufferSize(handle, m, dl, d, du, x, nrhs, ldx, &bufferSize);
aclsparseStatus_t aclsparseSgtsv2(handle, m, dl, d, du, x, nrhs, ldx, &bufferSize);
```
bufferSize 作为独立函数返回，也可嵌入主函数参数。

**模式 B：无 buffer（一次性计算）**
```c
aclsparseStatus_t aclsparseS{t}coo2csr(
    handle, cooRowInd, nnz, m, csrRowPtr, idxBase);
```
格式转换类常见。

---

## 三、公共规范（两种体系共用）

### 3.1 枚举类型

#### aclsparseOperation_t
```c
typedef enum {
    ACL_SPARSE_OP_NON_TRANSPOSE = 0,
    ACL_SPARSE_OP_TRANSPOSE,
    ACL_SPARSE_OP_CONJUGATE_TRANSPOSE
} aclsparseOperation_t;
```

#### aclsparseStatus_t
```c
typedef enum {
    ACL_SPARSE_STATUS_SUCCESS = 0,
    ACL_SPARSE_STATUS_NOT_INITIALIZED,
    ACL_SPARSE_STATUS_ALLOC_FAILED,
    ACL_SPARSE_STATUS_INVALID_VALUE,
    ACL_SPARSE_STATUS_ARCH_MISMATCH,
    ACL_SPARSE_STATUS_EXECUTION_FAILED,
    ACL_SPARSE_STATUS_INTERNAL_ERROR,
    ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
    ACL_SPARSE_STATUS_NOT_SUPPORTED,
    ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES,
    ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
} aclsparseStatus_t;
```

#### aclsparseFormat_t
```c
typedef enum {
    ACL_SPARSE_FORMAT_COO = 0,
    ACL_SPARSE_FORMAT_CSR,
    ACL_SPARSE_FORMAT_CSC,
    ACL_SPARSE_FORMAT_BLOCKED_ELL,
    ACL_SPARSE_FORMAT_SLICED_ELL,
    ACL_SPARSE_FORMAT_BSR
} aclsparseFormat_t;
```

#### aclsparseIndexType_t / aclsparseIndexBase_t
```c
typedef enum { ACL_SPARSE_INDEX_32I = 0, ACL_SPARSE_INDEX_64I } aclsparseIndexType_t;
typedef enum { ACL_SPARSE_INDEX_BASE_ZERO = 0, ACL_SPARSE_INDEX_BASE_ONE } aclsparseIndexBase_t;
```

#### aclsparseMatrixType_t（仅 Legacy）
```c
typedef enum {
    ACL_SPARSE_MATRIX_TYPE_GENERAL = 0,
    ACL_SPARSE_MATRIX_TYPE_SYMMETRIC,
    ACL_SPARSE_MATRIX_TYPE_HERMITIAN,
    ACL_SPARSE_MATRIX_TYPE_TRIANGULAR
} aclsparseMatrixType_t;
```

#### aclsparseDiagType_t / aclsparseFillMode_t（仅 Legacy）
```c
typedef enum { ACL_SPARSE_DIAG_TYPE_NON_UNIT = 0, ACL_SPARSE_DIAG_TYPE_UNIT } aclsparseDiagType_t;
typedef enum { ACL_SPARSE_FILL_MODE_LOWER = 0, ACL_SPARSE_FILL_MODE_UPPER } aclsparseFillMode_t;
```

### 3.2 Handle 管理（共用）

```c
aclsparseStatus_t aclsparseCreate(aclsparseHandle_t *handle);
aclsparseStatus_t aclsparseDestroy(aclsparseHandle_t handle);
aclsparseStatus_t aclsparseSetStream(aclsparseHandle_t handle, aclrtStream stream);
```

### 3.3 稀疏格式索引结构（共用）

#### CSR
| 数组 | 长度 | 说明 |
|------|------|------|
| csrRowOffsets | rows + 1 | 行偏移 |
| csrColInd | nnz | 列索引 |
| csrValues | nnz | 非零元素值 |

#### COO
| 数组 | 长度 | 说明 |
|------|------|------|
| cooRowInd | nnz | 行索引 |
| cooColInd | nnz | 列索引 |
| cooValues | nnz | 非零元素值 |

#### CSC
| 数组 | 长度 | 说明 |
|------|------|------|
| cscColOffsets | cols + 1 | 列偏移 |
| cscRowInd | nnz | 行索引 |
| cscValues | nnz | 非零元素值 |

### 3.4 状态码使用规范

| 场景 | 状态码 |
|------|--------|
| 正常完成 | `ACL_SPARSE_STATUS_SUCCESS` |
| handle 为 nullptr | `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` |
| 描述符/指针为 nullptr | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 维度为负数或不合法 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 内存分配失败 | `ACL_SPARSE_STATUS_ALLOC_FAILED` |
| 不支持的稀疏格式 | `ACL_SPARSE_STATUS_NOT_SUPPORTED`（仅 Generic） |
| 不支持的数据类型 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| 不支持的算法 | `ACL_SPARSE_STATUS_NOT_SUPPORTED`（仅 Generic） |
| 芯片不支持 | `ACL_SPARSE_STATUS_ARCH_MISMATCH` |
| Kernel 执行失败 | `ACL_SPARSE_STATUS_EXECUTION_FAILED` |
| UB 容量不足 | `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` |
| 内部错误 | `ACL_SPARSE_STATUS_INTERNAL_ERROR` |

---

## 四、常见错误示例

### 命名错误

| ❌ 错误 | ✅ 正确 | 说明 |
|---------|---------|------|
| `aclsparse_spmv` | `aclsparseSpMV` | 禁止下划线分隔 |
| `aclsparseSPMV` | `aclsparseSpMV` | 禁止全大写 |
| `aclsparseSpMv` | `aclsparseSpMV` | MV 是缩写，应全大写 |

### 描述符使用错误（Generic API）

| ❌ 错误 | ✅ 正确 | 说明 |
|---------|---------|------|
| 传入 `aclsparseSpMatDescr_t` 给只读参数 | 传入 `aclsparseConstSpMatDescr_t` | 输入描述符必须用 const 版本 |
| 跳过 `GetBufferSize` 直接调用算子 | 仅当接口暴露 `GetBufferSize` 时：先查询 → 按返回值决定是否分配 buffer → 再调用 | 阶段按接口声明调用，并非所有接口都必选三阶段；Preprocess 为可选，无需预处理时 bufferSize=0 且阶段 2 不存在（见 §1.5），单步接口（如 Gather）按其自身约定调用 |
| 忘记 `DestroySpMat` | Create/Destroy 配对 | 描述符必须释放 |

### Legacy API 错误

| ❌ 错误 | ✅ 正确 | 说明 |
|---------|---------|------|
| `aclsparseGtsv2(...)` | `aclsparseSgtsv2(...)` | Legacy 必须带精度前缀 |
| 传入未初始化的 `pBufferSize` 指针 | 先声明 `size_t bufSize = 0;` 再传 `&bufSize` | 指针必须有效 |

### const 修饰错误

| ❌ 错误 | ✅ 正确 | 说明 |
|---------|---------|------|
| `void *alpha` | `const void *alpha` | 标量输入必须 const |
| `aclsparseDnVecDescr_t vecX`（输入） | `aclsparseConstDnVecDescr_t vecX` | 输入向量用 const 描述符 |

---

## 五、参考资源

| 资源 | 链接 | 说明 |
|------|------|------|
| cuSPARSE 官方文档 | https://docs.nvidia.com/cuda/cusparse/ | Generic API + Legacy API 完整参考 |
| cuSPARSE Generic API | https://docs.nvidia.com/cuda/cusparse/#cusparse-generic-apis | Descriptor 模式、三阶段执行 |
| cuSPARSE Legacy API | https://docs.nvidia.com/cuda/cusparse/#cusparse-legacy-apis | 已弃用接口，仅用于对标尚未迁移的算子 |
| Sparse BLAS 标准 | https://www.netlib.org/blas/ | Legacy API 命名和参数顺序参考 |

## 六、稀疏算子领域补充规则

> 以下规则为 ops-sparse 仓特有的领域约束，补充上述 API 规范。详细内容见 [references/sparse-domain-rules.md](references/sparse-domain-rules.md)。

| 主题 | 说明 |
|------|------|
| cuSPARSE 接口对齐原则 | 新增 API 必须严格对齐 cuSPARSE 对应接口的规范 |
| 描述符提取校验 | Host 侧从描述符提取信息时，必须校验格式、维度和数据类型 |
| 公共代码复用 | 优先复用 `sparse/common/` 下的公共模块，禁止重复定义 |
