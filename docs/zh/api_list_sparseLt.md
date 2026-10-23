# 稀疏结构化算子接口（aclsparseLt）

## 使用说明

为方便调用结构化稀疏相关算子，提供一套基于 C 的 API（以 aclsparseLt 为前缀），主要用于结构化稀疏矩阵乘法及其前置剪枝、压缩等场景。

调用 aclsparseLt 算子 API 时，需引用依赖的头文件和库文件。

- 依赖的头文件：`cann_ops_sparseLt.h`
- 依赖的库文件：`libops_sparseLt.so`

## 接口列表

| 接口名 | 说明 |
|--------|------|
| [aclsparseLtInit](#aclsparseltinit) | 初始化 aclsparseLt 库句柄 |
| [aclsparseLtDestroy](#aclsparseltdestroy) | 销毁 aclsparseLt 库句柄 |
| [aclsparseLtGetErrorName](#aclsparseltgeterrorname) | 获取状态码对应的枚举名字符串 |
| [aclsparseLtGetErrorString](#aclsparseltgeterrorstring) | 获取状态码对应的描述性字符串 |
| [aclsparseLtGetVersion](#aclsparseltgetversion) | 获取 aclsparseLt 库版本号 |
| [aclsparseLtGetProperty](#aclsparseltgetproperty) | 获取 aclsparseLt 库属性信息 |
| [aclsparseLtDenseDescriptorInit](#aclsparseltdensedescriptorinit) | 初始化稠密矩阵描述符 |
| [aclsparseLtStructuredDescriptorInit](#aclsparseltstructureddescriptorinit) | 初始化结构化稀疏矩阵描述符（2:4） |
| [aclsparseLtMatDescriptorDestroy](#aclsparseltmatdescriptordestroy) | 销毁矩阵描述符 |
| [aclsparseLtMatDescSetAttribute](#aclsparseltmatdescsetattribute) | 设置矩阵描述符属性（batch 配置） |
| [aclsparseLtMatDescGetAttribute](#aclsparseltmatdescgetattribute) | 获取矩阵描述符属性（batch 配置） |
| [aclsparseLtMatmulDescriptorInit](#aclsparseltmatmuldescriptorinit) | 初始化矩阵乘法描述符 |
| [aclsparseLtMatmulDescriptorDestroy](#aclsparseltmatmuldescriptordestroy) | 销毁矩阵乘法描述符 |
| [aclsparseLtMatmulDescSetAttribute](#aclsparseltmatmuldescsetattribute) | 设置 matmul 描述符属性 |
| [aclsparseLtMatmulDescGetAttribute](#aclsparseltmatmuldescgetattribute) | 获取 matmul 描述符属性 |
| [aclsparseLtMatmulAlgSelectionInit](#aclsparseltmatmulalgselectioninit) | 初始化 matmul 算法选择描述符 |
| [aclsparseLtMatmulAlgSelectionDestroy](#aclsparseltmatmulalgselectiondestroy) | 销毁 matmul 算法选择描述符 |
| [aclsparseLtMatmulAlgSetAttribute](#aclsparseltmatmulalgsetattribute) | 设置算法选择描述符属性 |
| [aclsparseLtMatmulAlgGetAttribute](#aclsparseltmatmulalggetattribute) | 获取算法选择描述符属性 |
| [aclsparseLtMatmulPlanInit](#aclsparseltmatmulplaninit) | 初始化 matmul 执行计划 |
| [aclsparseLtMatmulPlanDestroy](#aclsparseltmatmulplandestroy) | 销毁 matmul 执行计划 |
| [aclsparseLtMatmulGetWorkspace](#aclsparseltmatmulgetworkspace) | 获取 matmul 所需 workspace 大小 |
| [aclsparseLtMatmul](#aclsparseltmatmul) | 执行结构化稀疏矩阵乘法 |
| [aclsparseLtMatmulSearch](#aclsparseltmatmulsearch) | 搜索最优 matmul 算法并回写 plan（auto-tuning） |
| [aclsparseLtSpMMAPrune](#aclsparseltspmmaprune) | 对稠密矩阵执行 2:4 结构化稀疏剪枝 |
| [aclsparseLtSpMMAPruneCheck](#aclsparseltspmmaprunecheck) | 校验稠密矩阵是否已满足 2:4 结构化稀疏约束 |
| [aclsparseLtSpMMACompressedSize](#aclsparseltspmmacompressedsize) | 查询压缩后矩阵及临时空间所需存储大小（Ascend950） |
| [aclsparseLtSpMMACompress](#aclsparseltspmmacompress) | 将结构化稀疏矩阵压缩为紧凑存储（Ascend950） |

## 接口详情

### aclsparseLtInit

```c
aclsparseStatus_t aclsparseLtInit(aclsparseLtHandle_t* handle);
```

**功能**：初始化 aclsparseLt 库句柄。在主机端分配轻量级硬件资源，创建 aclsparseLt 库上下文。必须在调用任何其他 aclsparseLt 函数之前调用。库上下文绑定到当前 NPU 设备，多设备使用需为每个设备创建独立 handle。

**参数说明**：

- `handle`（OUT）：HOST，aclsparseLt 库句柄输出参数。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`*handle` 非空（防止覆盖已有句柄导致泄漏）
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseLtDestroy

```c
aclsparseStatus_t aclsparseLtDestroy(const aclsparseLtHandle_t* handle);
```

**功能**：释放 aclsparseLt 库句柄占用的全部资源。与特定 handle 关联的最后一次调用，调用后该 handle 不可再使用。

**参数说明**：

- `handle`（IN）：HOST，指向要销毁的 aclsparseLt 库句柄的指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空

---

### aclsparseLtGetErrorName

```c
const char* aclsparseLtGetErrorName(aclsparseStatus_t status);
```

**功能**：返回状态码对应的枚举名字符串（如 `"ACL_SPARSE_STATUS_SUCCESS"`）。未识别的状态码返回 `"unrecognized error code"`。

**参数说明**：

- `status`（IN）：HOST，要转换的状态码。

**返回值**：

- `const char*`：指向枚举名字符串的指针。

---

### aclsparseLtGetErrorString

```c
const char* aclsparseLtGetErrorString(aclsparseStatus_t status);
```

**功能**：返回状态码对应的描述性字符串。未识别的状态码返回 `"unrecognized error code"`。

**参数说明**：

- `status`（IN）：HOST，要转换的状态码。

**返回值**：

- `const char*`：指向描述性字符串的指针。

---

### aclsparseLtGetVersion

```c
aclsparseStatus_t aclsparseLtGetVersion(
    aclsparseLtConstHandle_t handle,
    int* version);
```

**功能**：获取 aclsparseLt 库的版本号。版本编码格式：MAJOR*10000 + MINOR*100 + PATCH（如 0.1.0 -> 100）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `version`（OUT）：HOST，返回库的版本号。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：version 为空指针

---

### aclsparseLtGetProperty

```c
aclsparseStatus_t aclsparseLtGetProperty(
    aclsparseLtLibraryPropertyType_t propertyType,
    int* value);
```

**功能**：获取 aclsparseLt 库的属性信息（如主版本号、次版本号、补丁版本号）。根据 propertyType 返回版本号的 MAJOR / MINOR / PATCH 分量到 int* value。

**参数说明**：

- `propertyType`（IN）：HOST，请求的属性类型（`ACLSPARSELT_MAJOR_VERSION` / `ACLSPARSELT_MINOR_VERSION` / `ACLSPARSELT_PATCH_LEVEL`）。
- `value`（OUT）：HOST，返回请求属性的值。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：value 为空指针、propertyType 非法

---

### aclsparseLtDenseDescriptorInit

```c
aclsparseStatus_t aclsparseLtDenseDescriptorInit(
    const aclsparseLtHandle_t*       handle,
    aclsparseLtMatDescriptor_t*      matDescr,
    int64_t                          rows,
    int64_t                          cols,
    int64_t                          ld,
    uint32_t                         alignment,
    aclDataType                      valueType,
    aclsparseOrder_t                 order);
```

**功能**：初始化稠密矩阵描述符。在主机端分配并填充矩阵描述符结构体，记录稠密矩阵的形状、布局、数据类型等信息。用于描述 matmul 中的稠密矩阵（或剪枝场景下的占位稠密矩阵）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matDescr`（OUT）：HOST，矩阵描述符输出。
- `rows`（IN）：HOST，行数。
- `cols`（IN）：HOST，列数。
- `ld`（IN）：HOST，leading dimension。
- `alignment`（IN）：HOST，内存对齐字节数。
- `valueType`（IN）：HOST，矩阵数据存储类型。
- `order`（IN）：HOST，内存布局（`ACL_SPARSE_ORDER_ROW` / `ACL_SPARSE_ORDER_COL`）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matDescr 为空、维度/对齐/布局非法
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：valueType 不在支持列表内
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseLtStructuredDescriptorInit

```c
aclsparseStatus_t aclsparseLtStructuredDescriptorInit(
    const aclsparseLtHandle_t*       handle,
    aclsparseLtMatDescriptor_t*      matDescr,
    int64_t                          rows,
    int64_t                          cols,
    int64_t                          ld,
    uint32_t                         alignment,
    aclDataType                      valueType,
    aclsparseOrder_t                 order,
    aclsparseLtSparsity_t            sparsity);
```

**功能**：初始化结构化稀疏矩阵描述符（2:4 结构化稀疏）。与 `aclsparseLtDenseDescriptorInit` 类似，但描述符标记为 structured，对齐倍数采用 structured 档位，并记录稀疏模式。用于描述 matmul 中的稀疏矩阵。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matDescr`（OUT）：HOST，矩阵描述符输出。
- `rows`（IN）：HOST，行数。
- `cols`（IN）：HOST，列数。
- `ld`（IN）：HOST，leading dimension。
- `alignment`（IN）：HOST，内存对齐字节数。
- `valueType`（IN）：HOST，矩阵数据存储类型。
- `order`（IN）：HOST，内存布局（`ACL_SPARSE_ORDER_ROW` / `ACL_SPARSE_ORDER_COL`）。
- `sparsity`（IN）：HOST，稀疏模式（`ACL_SPARSE_LT_SPARSITY_50_PERCENT`）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matDescr 为空、维度/对齐/布局/sparsity 非法
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：valueType 不在支持列表内
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseLtMatDescriptorDestroy

```c
aclsparseStatus_t aclsparseLtMatDescriptorDestroy(aclsparseLtMatDescriptor_t* matDescr);
```

**功能**：销毁矩阵描述符，释放其占用的主机内存。销毁成功后 `*matDescr` 被置为 nullptr。

**参数说明**：

- `matDescr`（IN/OUT）：HOST，指向要销毁的矩阵描述符的指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：matDescr 指针为空

---

### aclsparseLtMatDescSetAttribute

```c
aclsparseStatus_t aclsparseLtMatDescSetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatDescriptor_t* matDescr,
    aclsparseLtMatDescAttribute_t matAttribute,
    const void* data,
    size_t dataSize);
```

**功能**：设置矩阵描述符的指定属性（用于 batch 批量计算配置）。须在 `aclsparseLtMatmulDescriptorInit` 之前设置——Init 时校验 A/B/C/D 四个矩阵的 `numBatches` 一致性。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matDescr`（IN/OUT）：HOST，矩阵描述符。
- `matAttribute`（IN）：HOST，要设置的矩阵描述符属性。
- `data`（IN）：HOST，指向属性值的指针。
- `dataSize`（IN）：HOST，属性值字节数，须等于对应属性的 sizeof，用于校验。

可设置的属性如下：

| 属性枚举 | 值 | 值类型 | 默认值 | 说明 |
|----------|---|--------|--------|------|
| `ACLSPARSELT_MAT_NUM_BATCHES` | 0 | int32_t | 1 | batch 数量，须 >= 1。dataSize 须 = `sizeof(int32_t)` |
| `ACLSPARSELT_MAT_BATCH_STRIDE` | 1 | int64_t | 0 | batch 间步长（元素数），每个矩阵独立设置。dataSize 须 = `sizeof(int64_t)` |

> **numBatches 一致性校验**：`aclsparseLtMatmulDescriptorInit` 时校验 A/B/C/D 四个矩阵的 numBatches 须一致，不一致返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。本接口仅校验单个描述符的 numBatches >= 1，不跨矩阵校验。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matDescr 为空、matAttribute 非法、data 为空、dataSize 与属性不匹配、`NUM_BATCHES` 值 < 1
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：该属性不在支持范围内

---

### aclsparseLtMatDescGetAttribute

```c
aclsparseStatus_t aclsparseLtMatDescGetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatDescriptor_t* matDescr,
    aclsparseLtMatDescAttribute_t matAttribute,
    void* data,
    size_t dataSize);
```

**功能**：获取矩阵描述符的指定属性（如批数、批步长等）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matDescr`（IN）：HOST，矩阵描述符。
- `matAttribute`（IN）：HOST，要获取的矩阵描述符属性。
- `data`（OUT）：HOST，返回属性值的内存地址。
- `dataSize`（IN）：HOST，属性值字节数，须等于对应属性的 sizeof，用于校验。

支持查询的属性同 `aclsparseLtMatDescSetAttribute`，可查询 `ACLSPARSELT_MAT_NUM_BATCHES` 与 `ACLSPARSELT_MAT_BATCH_STRIDE` 的当前值。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matDescr 为空、matAttribute 非法、data 为空、dataSize 与属性不匹配
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：该属性不在支持范围内

---

### aclsparseLtMatmulDescriptorInit

```c
aclsparseStatus_t aclsparseLtMatmulDescriptorInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulDescriptor_t*         matmulDescr,
    aclsparseOperation_t                   opA,
    aclsparseOperation_t                   opB,
    const aclsparseLtMatDescriptor_t*      matA,
    const aclsparseLtMatDescriptor_t*      matB,
    const aclsparseLtMatDescriptor_t*      matC,
    const aclsparseLtMatDescriptor_t*      matD,
    aclsparseComputeType_t                 computeType);
```

**功能**：初始化矩阵乘法描述符。在主机端分配并填充 matmul 描述符结构体，记录 matmul 运算的完整信息：操作类型、四个矩阵描述符引用、计算精度。matA/matB/matC/matD 为非所有权引用。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matmulDescr`（OUT）：HOST，matmul 描述符输出。
- `opA`（IN）：HOST，作用于矩阵 A 的操作（`ACL_SPARSE_OP_NON_TRANSPOSE` / `ACL_SPARSE_OP_TRANSPOSE`）。
- `opB`（IN）：HOST，作用于矩阵 B 的操作。
- `matA`（IN）：HOST，矩阵 A 描述符（structured 或 dense）。
- `matB`（IN）：HOST，矩阵 B 描述符（structured 或 dense）。
- `matC`（IN）：HOST，矩阵 C 描述符（dense）。
- `matD`（IN）：HOST，矩阵 D 描述符（dense）。
- `computeType`（IN）：HOST，计算精度（`ACL_SPARSE_COMPUTE_16F` / `ACL_SPARSE_COMPUTE_32F` / `ACL_SPARSE_COMPUTE_32I`）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matmulDescr 为空、opA/opB 非法、结构化位置/C/D 一致性/操作布局组合/维度约束违反
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：computeType 不在支持列表内
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseLtMatmulDescriptorDestroy

```c
aclsparseStatus_t aclsparseLtMatmulDescriptorDestroy(aclsparseLtMatmulDescriptor_t* matmulDescr);
```

**功能**：销毁 matmul 描述符，释放其占用的主机内存。本接口不销毁 matA/matB/matC/matD 描述符，用户须自行调用 `aclsparseLtMatDescriptorDestroy` 释放它们。销毁成功后 `*matmulDescr` 被置为 nullptr。

**参数说明**：

- `matmulDescr`（IN/OUT）：HOST，指向要销毁的 matmul 描述符的指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：matmulDescr 指针为空

---

### aclsparseLtMatmulDescSetAttribute

```c
aclsparseStatus_t aclsparseLtMatmulDescSetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulDescriptor_t* matmulDescr,
    aclsparseLtMatmulDescAttribute_t matmulAttribute,
    const void* data,
    size_t dataSize);
```

**功能**：设置 matmul 描述符的指定属性，用于配置 epilogue 链的向量缩放（alpha/beta）、bias 与 activation。上述属性在执行期（`aclsparseLtMatmul` / `aclsparseLtMatmulSearch` 每次执行时）从描述符读取，`aclsparseLtMatmulPlanInit` 之后修改对后续执行生效（宽容时序，无需重新初始化计划）。设置 `BETA_VECTOR_SCALING` 为非 0 时隐含启用 `ALPHA_VECTOR_SCALING`（cuSPARSELt 语义）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matmulDescr`（IN/OUT）：HOST，matmul 描述符，须已初始化。
- `matmulAttribute`（IN）：HOST，要设置的 matmul 描述符属性。
- `data`（IN）：HOST，指向属性值的指针，值类型与 dataSize 随属性变化（见下表）。
- `dataSize`（IN）：HOST，属性值字节数，须等于对应属性的 sizeof，用于校验。

可设置的属性如下：

| 属性枚举 | 值 | 值类型 | 默认值 | 说明 |
|----------|---|--------|--------|------|
| `ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING` | 0 | int | 0 | alpha 逐行向量缩放开关，0=禁用（标量模式），非 0=启用（alpha 为 Device float[M]）。dataSize 须 = `sizeof(int)` |
| `ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING` | 1 | int | 0 | beta 逐行向量缩放开关，0=禁用，非 0=启用，启用时隐含启用 ALPHA_VECTOR_SCALING（cuSPARSELt 语义）。dataSize 须 = `sizeof(int)` |
| `ACLSPARSELT_MATMUL_BIAS_POINTER` | 2 | void* | NULL | bias 向量 Device 指针，长度 = m。dtype 与 C 矩阵相同（FP32 路径 bias 为 FP32，FP16 路径 bias 为 FP16，BF16 路径 bias 为 BF16），INT8 路径 bias 为 FP32（cuSPARSELt 规定）。NULL=不启用 bias。dataSize 须 = `sizeof(void*)` |
| `ACLSPARSELT_MATMUL_BIAS_STRIDE` | 3 | int64_t | 0 | batch 间 bias 步长（元素数），0=所有 batch 共用同一 bias。dataSize 须 = `sizeof(int64_t)` |
| `ACLSPARSELT_MATMUL_ACTIVATION_RELU` | 4 | int | 0 | ReLU 开关，0=禁用，非 0=启用。与 GeLU 互斥。dataSize 须 = `sizeof(int)` |
| `ACLSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND` | 5 | float | FLT_MAX | ReLU 上界（`min(upperBound, D)`）。dataSize 须 = `sizeof(float)` |
| `ACLSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD` | 6 | float | 0.0f | ReLU 阈值（`max(threshold, D)`）。dataSize 须 = `sizeof(float)` |
| `ACLSPARSELT_MATMUL_ACTIVATION_GELU` | 7 | int | 0 | GeLU 开关，0=禁用，非 0=启用。与 ReLU 互斥。dataSize 须 = `sizeof(int)` |
| `ACLSPARSELT_MATMUL_ACTIVATION_GELU_SCALING` | 8 | float | 1.0f | GeLU 缩放系数。设置此属性会隐含启用 GeLU（cuSPARSELt 语义：It implies ACTIVATION_GELU）。若 ReLU 已启用则不自动启用 GeLU（保持互斥语义）。dataSize 须 = `sizeof(float)` |

> **ReLU/GeLU 互斥校验**：SetAttribute 启用一种 activation 时，若另一种已启用，返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；设为 0（关闭）时不检查。例如：已启用 GeLU（`ACTIVATION_GELU=1`）后设置 `ACTIVATION_RELU=1` 将返回错误；先设置 `ACTIVATION_GELU=0` 关闭再启用 ReLU 则成功。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matmulDescr 为空、matmulAttribute 非法、data 为空、dataSize 与属性不匹配、ReLU/GeLU 互斥冲突
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：该属性不在支持范围内

---

### aclsparseLtMatmulDescGetAttribute

```c
aclsparseStatus_t aclsparseLtMatmulDescGetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    aclsparseLtMatmulDescAttribute_t matmulAttribute,
    void* data,
    size_t dataSize);
```

**功能**：获取 matmul 描述符的指定属性当前值。支持查询 `aclsparseLtMatmulDescSetAttribute` 列出的全部 9 个属性（向量缩放开关、bias 指针/步长、ReLU 开关/上界/阈值、GeLU 开关/缩放）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matmulDescr`（IN）：HOST，matmul 描述符。
- `matmulAttribute`（IN）：HOST，要获取的 matmul 描述符属性。
- `data`（OUT）：HOST，返回属性值的内存地址，值类型与 dataSize 随属性变化（同 SetAttribute 表）。
- `dataSize`（IN）：HOST，属性值字节数，须等于对应属性的 sizeof，用于校验。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matmulDescr 为空、matmulAttribute 非法、data 为空、dataSize 与属性不匹配
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：该属性不在支持范围内

---

### aclsparseLtMatmulAlgSelectionInit

```c
aclsparseStatus_t aclsparseLtMatmulAlgSelectionInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulAlgSelection_t*       algSelection,
    const aclsparseLtMatmulDescriptor_t*   matmulDescr,
    aclsparseLtMatmulAlg_t                 alg);
```

**功能**：初始化 matmul 算法选择描述符。在主机端分配并填充算法选择描述符结构体，记录算法模式及其关联的 matmul 描述符引用。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `algSelection`（OUT）：HOST，算法选择描述符输出。
- `matmulDescr`（IN）：HOST，matmul 描述符引用。
- `alg`（IN）：HOST，算法模式（`ACL_SPARSE_LT_MATMUL_ALG_DEFAULT`）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：algSelection 为空、matmulDescr 为空或未初始化、alg 非法
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseLtMatmulAlgSelectionDestroy

```c
aclsparseStatus_t aclsparseLtMatmulAlgSelectionDestroy(aclsparseLtMatmulAlgSelection_t* algSelection);
```

**功能**：销毁 matmul 算法选择描述符，释放其占用的主机内存。本接口不销毁 matmulDescr，用户须自行调用 `aclsparseLtMatmulDescriptorDestroy` 释放。销毁成功后 `*algSelection` 被置为 nullptr。

**参数说明**：

- `algSelection`（IN/OUT）：HOST，指向要销毁的算法选择描述符的指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：algSelection 指针为空

---

### aclsparseLtMatmulAlgSetAttribute

```c
aclsparseStatus_t aclsparseLtMatmulAlgSetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attr,
    const void* attrValue,
    size_t attrValueSize);
```

**功能**：设置算法选择描述符的指定属性（如算法配置 ID、splitK 切分因子、搜索迭代次数、Split-K 模式等）。三元组属性（algConfigId / splitK / splitKMode）须在 PlanInit 之前调用（PlanInit 读取它们计算 tiling）；`ACLSPARSELT_MATMUL_SEARCH_ITERATIONS` 为搜索入口活读属性，PlanInit 之后设置对随后的 Search 仍生效。ALG_CONFIG_MAX_ID 为只读属性，不可设置。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `algSelection`（IN/OUT）：HOST，算法选择描述符。
- `attr`（IN）：HOST，要设置的算法选择描述符属性。
- `attrValue`（IN）：HOST，指向属性值的指针（int32_t 类型）。
- `attrValueSize`（IN）：HOST，属性值字节数，须等于 `sizeof(int32_t)`，用于校验。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：algSelection 为空、attr 非法、attrValue 为空、attrValueSize 与属性不匹配或属性值非法
- `ACL_SPARSE_STATUS_NOT_INITIALIZED`：algSelection 未初始化（已销毁）
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：该属性暂不支持（如 ALG_CONFIG_MAX_ID 为只读）

---

### aclsparseLtMatmulAlgGetAttribute

```c
aclsparseStatus_t aclsparseLtMatmulAlgGetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attr,
    void* attrValue,
    size_t attrValueSize);
```

**功能**：获取算法选择描述符的指定属性（如算法配置 ID、搜索迭代次数、splitK 切分因子、Split-K 模式等）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `algSelection`（IN）：HOST，算法选择描述符。
- `attr`（IN）：HOST，要获取的算法选择描述符属性。
- `attrValue`（OUT）：HOST，返回属性值的内存地址（int32_t 类型）。
- `attrValueSize`（IN）：HOST，属性值字节数，须等于 `sizeof(int32_t)`，用于校验。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：algSelection 为空、attr 非法、attrValue 为空、attrValueSize 与属性不匹配
- `ACL_SPARSE_STATUS_NOT_INITIALIZED`：algSelection 未初始化（已销毁）
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：该属性暂不支持

---

### aclsparseLtMatmulPlanInit

```c
aclsparseStatus_t aclsparseLtMatmulPlanInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulPlan_t*               plan,
    const aclsparseLtMatmulDescriptor_t*   matmulDescr,
    const aclsparseLtMatmulAlgSelection_t* algSelection);
```

**功能**：初始化 matmul 执行计划。在主机端分配并填充 plan 结构体，绑定 matmul 描述符与算法选择描述符，作为后续 matmul 执行的规划对象。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `plan`（OUT）：HOST，执行计划输出。
- `matmulDescr`（IN）：HOST，matmul 描述符引用。
- `algSelection`（IN）：HOST，算法选择描述符引用。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：plan 为空、matmulDescr 或 algSelection 为空或未初始化
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseLtMatmulPlanDestroy

```c
aclsparseStatus_t aclsparseLtMatmulPlanDestroy(aclsparseLtMatmulPlan_t* plan);
```

**功能**：销毁 matmul 执行计划，释放其占用的主机内存。本接口不销毁 matmulDescr 和 algSelection，用户须自行调用对应 Destroy 释放。plan 须在 algSelection 与 matmul 描述符之前销毁。销毁成功后 `*plan` 被置为 nullptr。

**参数说明**：

- `plan`（IN/OUT）：HOST，指向要销毁的执行计划的指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：plan 指针为空

---

### aclsparseLtMatmulGetWorkspace

```c
aclsparseStatus_t aclsparseLtMatmulGetWorkspace(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* plan,
    size_t* workspaceSize);
```

**功能**：查询执行 matmul 所需的 workspace 大小（字节）。根据 PlanInit 阶段计算的 tiling 参数返回 workspace 字节数。调用方据此分配 Device 内存，再将指针传递给 aclsparseLtMatmul。workspaceSize 为 0 时无需分配。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `plan`（IN）：HOST，matmul 执行计划。
- `workspaceSize`（OUT）：HOST，返回所需 workspace 大小（字节）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：plan 为空、workspaceSize 为空

---

### aclsparseLtMatmul

```c
aclsparseStatus_t aclsparseLtMatmul(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* plan,
    const void* alpha,
    const void* matA,
    const void* matB,
    const void* beta,
    const void* matC,
    void* matD,
    void* workspace,
    aclrtStream* streams,
    int32_t numStreams);
```

**功能**：执行结构化稀疏矩阵乘法并融合 epilogue 后处理，计算 `D = Activation(alpha * op(A) * op(B) + beta * C + bias)`。epilogue 链依次执行：alpha 缩放 → beta·C 累加 → bias 逐行广播加（per-row broadcast，长度 = m 的逐行广播向量，dtype 与 C 相同，INT8 路径为 FP32）→ activation（ReLU/GeLU，可选，二者互斥）→ 类型转换写出。bias 与 activation 通过 `aclsparseLtMatmulDescSetAttribute` 预先配置到 matmul 描述符中，执行接口不额外传参。A 须为已剪枝的矩阵（通过 `aclsparseLtSpMMAPrune` 产出）。当 matA 为 null 时，回退到 workspace 中的 A_pruned 区域（须已由 SpMMAPrune 填充）。alpha/beta 可为标量（float 指针）或设备端向量（启用 ALPHA_VECTOR_SCALING 时为 float[M] 数组指针）。算子在指定 stream 上异步执行，调用方须通过 aclrtSynchronizeStream 同步后读取 D。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `plan`（IN）：HOST，matmul 执行计划。
- `alpha`（IN）：HOST/DEVICE，标量 α（`float` 类型指针）；标量时为 HOST 指针，向量缩放时为 DEVICE 指针。
- `matA`（IN）：DEVICE，矩阵 A 的指针（结构化稀疏或稠密）。可为 nullptr（回退到 workspace A_pruned 区域）。
- `matB`（IN）：DEVICE，矩阵 B 的指针（结构化稀疏或稠密）。
- `beta`（IN）：HOST/DEVICE，标量 β（`float` 类型指针）；标量时为 HOST 指针，向量缩放时为 DEVICE 指针。
- `matC`（IN）：DEVICE，稠密矩阵 C 的指针（累加项）。
- `matD`（OUT）：DEVICE，稠密矩阵 D 的输出指针（须预分配）。
- `workspace`（IN）：DEVICE，workspace 指针（大小由 GetWorkspaceSize 查询）。
- `streams`（IN）：HOST，指向 ACL stream 数组的指针。
- `numStreams`（IN）：HOST，`streams` 数组中的 stream 数量（须 > 0）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：plan 为空、矩阵指针为空
- `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`：workspace 为空但计划需要
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：不支持的数据类型组合或操作
- `ACL_SPARSE_STATUS_EXECUTION_FAILED`：kernel 执行或 aclrt 调用失败

---

### aclsparseLtMatmulSearch

```c
aclsparseStatus_t aclsparseLtMatmulSearch(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulPlan_t* plan,
    const void* alpha,
    const void* matA,
    const void* matB,
    const void* beta,
    const void* matC,
    void* matD,
    void* workspace,
    aclrtStream* streams,
    int32_t numStreams);
```

**功能**：搜索 matmul 最优算法配置并回写 plan（auto-tuning）。评估 plan 描述的 matmul 的可用算法候选（algConfigId × splitKMode × splitK），每个候选在 streams[0] 上按 aclsparseLtMatmul 相同语义计时执行 searchIterations 次（默认 5，经 `aclsparseLtMatmulAlgSetAttribute` + `ACLSPARSELT_MATMUL_SEARCH_ITERATIONS` 设置）取最小耗时，自动选择最快的配置并回写 plan 绑定的 algSelection 属性（可经 `aclsparseLtMatmulAlgGetAttribute` 查询）与 plan 内 tilingData / workspaceSize（按选中配置重算）。面向同一 matmul 重复执行多次的自动调优场景；数值计算由既有 aclsparseLtMatmul kernel 承载，数据类型支持与 aclsparseLtMatmul 相同（FP32/FP16/BF16/INT8）。

行为声明：

- **阻塞语义**：本接口相对 streams[0] 为阻塞调用（返回前完成流同步），区别于 aclsparseLtMatmul 的异步语义；返回后可直接回读 matD。
- **matD 终态**：搜索过程被反复写入；搜索结束后以选中配置无条件再执行一次（不计时），返回时内容为选中配置的完整输出；in-place（matC == matD）时 matD 值随执行次数累积，不可预测。
- **候选失败**：个别候选执行失败视为不可用候选跳过（日志留痕，不影响搜索返回），在可用候选中选优；全部候选执行失败时返回 EXECUTION_FAILED，此时 plan 与 algSelection 恢复进入搜索前的状态，可继续用原配置执行 Matmul。
- **搜索后 workspace**：方案 A（候选裁剪）保证选中配置的 workspace 需求不超过进入搜索时的 workspaceSize，已分配 workspace 必然够用，不会出现要求更大分配的情况；选中配置的需求可能更小（plan->workspaceSize 随之变小），建议重新调用 `aclsparseLtMatmulGetWorkspace` 校验实际值。想搜索更大 splitK 的候选，须先经 `aclsparseLtMatmulAlgSetAttribute` 设大 splitK 并将 splitKMode 设为 `ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS`，再重新 PlanInit 并按新 workspaceSize 分配（详见 [matmul_search README](../../sparseLt/matmul_search/README.md)）。三元组属性（algConfigId / splitK / splitKMode）须在 PlanInit 之前设置；`ACLSPARSELT_MATMUL_SEARCH_ITERATIONS` 为搜索入口活读属性，PlanInit 之后设置对随后的 Search 仍生效。
- **非紧凑布局直通**：dense 侧非紧凑布局（ld 大于物理维度）时，候选直接以调用方原始指针与布局执行（cube GM 搬运按 ld 行距直通跳过行尾 padding，无入口拷贝或预处理步骤）；候选计时与终态执行均按该语义进行。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `plan`（IN/OUT）：HOST，matmul 执行计划（非 const，区别于 aclsparseLtMatmul 的只读 plan）；搜索更新 plan 绑定的 algSelection 属性（algConfigId / splitKMode / splitK ← 最优值）及 plan 内 tilingData / workspaceSize（按最优配置重算）。
- `alpha`（IN）：HOST/DEVICE，标量 α（`float` 类型指针，nullptr 按 1.0 处理）；向量缩放（启用 `ALPHA_VECTOR_SCALING`）时为 DEVICE float[M] 指针，不可为 nullptr。计时执行按 aclsparseLtMatmul 相同语义透传。
- `matA`（IN）：DEVICE，矩阵 A 的指针（结构化稀疏侧为剪枝后矩阵），须 16 字节对齐。可为 nullptr（回退到 workspace 的 A_pruned 区域，须已由 SpMMAPrune 填充，仅限 numBatches=1；numBatches>1 时须先显式调用 SpMMAPrune 产出 A_pruned 再传入；dense×dense 与 B-sparse 路径须显式传入）。
- `matB`（IN）：DEVICE，矩阵 B 的指针，须 16 字节对齐，不可为 nullptr。
- `beta`（IN）：HOST/DEVICE，标量 β（`float` 类型指针，nullptr 按 0.0 处理）；向量缩放（启用 `BETA_VECTOR_SCALING`）时为 DEVICE float[M] 指针，不可为 nullptr；启用 `ALPHA_VECTOR_SCALING` 且未启用 `BETA_VECTOR_SCALING` 时可为 nullptr（按 0.0 处理）。
- `matC`（IN）：DEVICE，稠密矩阵 C 的指针（累加项），须 16 字节对齐，不可为 nullptr。按 matC 描述符 (order, ld) 布局访问（支持列主序与 ld padding，语义同 aclsparseLtMatmul）。支持 in-place（matC == matD，此时 matD 值在搜索中累积）。
- `matD`（OUT）：DEVICE，稠密矩阵 D 的输出指针，须 16 字节对齐，不可为 nullptr。按 matD 描述符 (order, ld) 布局写出。搜索过程被反复写入；返回时内容为选中配置的完整输出。
- `workspace`（IN）：DEVICE，workspace 指针（进入搜索前按当前配置 `aclsparseLtMatmulGetWorkspace` 查询并分配），须 16 字节对齐，不可为 nullptr。纳入评估的候选 workspace 需求均不超过进入时的 workspaceSize，无越界风险。
- `streams`（IN）：HOST，指向 ACL stream 数组的指针；streams[0] 可为 nullptr（默认流）。当前仅使用 streams[0]（多流不支持）。
- `numStreams`（IN）：HOST，`streams` 数组中的 stream 数量（须 > 0；> 1 时不报错，仅使用 streams[0]）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：搜索完成，最优配置已写入 plan/algSelection
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：plan/streams 为空、numStreams <= 0、matB/matC/matD 为空、数据指针未 16 字节对齐、matA 为空且不满足 A-sparse 回退条件、searchIterations <= 0、启用向量缩放时 alpha/beta 为 nullptr（以上均为搜索入口一次性校验；缩放向量的 32 字节对齐与 device 内存判定经候选执行链沿用 `aclsparseLtMatmul` 执行期校验）
- `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`：workspace 为空
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：不支持的数据类型（仅支持 FP32/FP16/BF16/INT8）
- `ACL_SPARSE_STATUS_EXECUTION_FAILED`：全部候选执行失败、选中配置的最终回写执行失败，或 aclrt 计时/同步调用失败（失败时 plan/algSelection 恢复进入搜索前状态）

完整调用链、约束说明与调用示例见 [matmul_search README](../../sparseLt/matmul_search/README.md)。

---

### aclsparseLtSpMMAPrune

```c
aclsparseStatus_t aclsparseLtSpMMAPrune(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    void* d_out,
    aclsparseLtPruneAlg_t pruneAlg,
    aclrtStream stream);
```

**功能**：对稠密矩阵 A 执行 2:4 结构化稀疏剪枝，将每 4 个元素（FP16/BF16/INT8）或每 2 个元素（FP32）中绝对值较小的元素置零，保留绝对值最大的元素，输出与 A 同型的稠密存储矩阵 A_pruned（被置零元素以 0 表示）。直接接收 Matmul 描述符，从中读取矩阵 A 的维度（m、k）、数据类型、order 与 opA 派生剪枝参数。算子异步启动，内部不执行 stream 同步，调用方如需读取结果须自行同步。

> 说明：本算子为软件实现的 2:4 结构化稀疏剪枝，不依赖 Ascend 950 硬件的 2:4 稀疏加速单元，仅通过 Vector API 完成剪枝计算。FP32 统一采用 1:2 模式（分组为 2、保留 1，即每 2 个元素中保留绝对值最大的 1 个），并非"4 选 2"；FP16/BF16/INT8 采用标准 2:4 模式（分组为 4、保留 2）。所有数据类型稀疏度均为 50%。

**剪枝规则**：

| 数据类型 | 分组大小 | 每组保留元素数 | 稀疏度 |
|----------|---------|---------------|--------|
| FP16（ACL_FLOAT16） | 4 | 2（保留绝对值最大的 2 个） | 50%（2:4） |
| BF16（ACL_BF16） | 4 | 2（保留绝对值最大的 2 个） | 50%（2:4） |
| INT8（ACL_INT8） | 4 | 2（保留绝对值最大的 2 个） | 50%（2:4） |
| FP32（ACL_FLOAT） | 2 | 1（保留绝对值最大的 1 个） | 50%（1:2，等价 2:4 密度） |

**产品支持情况**：

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |

> 依赖 CANN asc-devkit >= 9.1.0（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄的 const 指针。
- `matmulDescr`（IN）：HOST，Matmul 操作描述符，算子从中读取矩阵 A 的维度（m、k）、数据类型、order 与 opA。
- `d_in`（IN）：DEVICE，待剪枝的稠密矩阵 A 的指针。
- `d_out`（OUT）：DEVICE，剪枝结果 A_pruned 的指针。支持 in-place（d_in == d_out）。
- `pruneAlg`（IN）：HOST，剪枝算法，支持 `ACLSPARSELT_PRUNE_SPMMA_STRIP` 与 `ACLSPARSELT_PRUNE_SPMMA_TILE`。
- `stream`（IN）：HOST，ACL 流，算子在此流上异步执行，可为 nullptr（表示使用默认流）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为 nullptr
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matmulDescr/matA/d_in/d_out 为 nullptr、数据指针未 16 字节对齐
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：pruneAlg 非 STRIP/TILE、数据类型非 FP32/FP16/BF16/INT8、K 维度超过 UB 容量、转置路径 in-place（d_in == d_out）

---

### aclsparseLtSpMMAPruneCheck

```c
aclsparseStatus_t aclsparseLtSpMMAPruneCheck(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    int* d_valid,
    aclrtStream stream);
```

**功能**：校验稠密矩阵是否满足 2:4 结构化稀疏条件。沿校验方向将矩阵划分为固定大小分组，逐组统计非零元素个数并与保留上限比较：全部分组不超过上限时写入 `d_valid = 0`（满足，correct），任一分组超限时写入 `d_valid = 1`（不满足，wrong）。校验条件为组级基本条件，与剪枝算法（TILE/STRIP）无关；经 `aclsparseLtSpMMAPrune` 剪枝产出的数据保证通过本检查，可跳过本接口，仅用于校验自定义剪枝结果的合法性。本接口直接接收 Matmul 描述符，校验对象（A/B 中的稀疏侧，须恰有一侧为结构化稀疏描述符）、维度、数据类型、order 与 op 均从描述符派生，派生口径与 `aclsparseLtSpMMAPrune` 一致。支持多 batch（经 `aclsparseLtMatDescSetAttribute` 设置稀疏侧 `ACLSPARSELT_MAT_NUM_BATCHES`/`ACLSPARSELT_MAT_BATCH_STRIDE`）：`d_valid` 为全部 batch 判定结果的按位或——任一 batch 存在违规分组 → 1，全部 batch 合法 → 0；`numBatches=1`（默认）时为单 batch 判定，单 batch 判定逻辑不变。算子异步执行，无需额外存储（no extra storage），结果确定（deterministic）；内部不执行 stream 同步，调用方读取 `d_valid` 前须自行同步。

**校验规则**：

| 数据类型 | 分组大小 | 每组保留上限 | 稀疏模式 |
|----------|---------|--------------|----------|
| FP16（ACL_FLOAT16） | 4 | 2 | 2:4 |
| BF16（ACL_BF16） | 4 | 2 | 2:4 |
| INT8（ACL_INT8） | 4 | 2 | 2:4 |
| FP32（ACL_FLOAT） | 2 | 1 | 1:2（等价 2:4 密度） |

判零语义：`±0` 均按零处理；NaN、`±inf` 与非规格数（subnormal）按非零处理（位级判零，与 `x != 0` 的 IEEE 语义逐位等价）。

校验方向由稀疏侧矩阵描述符的 order 与 Matmul 描述符的 op 联合决定，与 `aclsparseLtSpMMAPrune` 的剪枝方向推导一致：

| op（稀疏侧） | order | 校验方向 |
|-----|-------|---------|
| NON_TRANSPOSE | ROW | 沿行（每组为行内 g 个连续元素） |
| NON_TRANSPOSE | COL | 沿列（每组为同列 g 个相邻元素） |
| TRANSPOSE | ROW | 沿列 |
| TRANSPOSE | COL | 沿行 |

`d_in` 一律按行主序物理存储解释，行步长为稀疏侧描述符的 ld（元素数）；order 仅决定校验方向，不改变内存访问模式。ld 大于物理行长时，padding 区数据不参与判定。

**产品支持情况**：

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |

> 依赖 CANN asc-devkit >= 9.1.0（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

**参数说明**（与 `cusparseLtSpMMAPruneCheck` 逐参对齐：`const cusparseLtHandle_t*` → `aclsparseLtConstHandle_t`，`const cusparseLtMatmulDescriptor_t*` → `aclsparseLtConstMatmulDescriptor_t*`，`cudaStream_t` → `aclrtStream`，返回 `cusparseStatus_t` → `aclsparseStatus_t`；`d_in`/`d_valid` 语义一致）：

- `handle`（IN）：HOST，aclsparseLt 库句柄的 const 指针。
- `matmulDescr`（IN）：HOST，Matmul 操作描述符，校验对象（稀疏侧）、维度、数据类型、order 与 op 从中派生；matA/matB 须恰有一个为结构化稀疏描述符。
- `d_in`（IN）：DEVICE，待校验矩阵的指针，行主序存储，须 32 字节对齐。
- `d_valid`（OUT）：DEVICE，校验结果 int 标量（`0` 表示满足约束，`1` 表示不满足）。
- `stream`（IN）：HOST，ACL 流，算子在此流上异步执行，可为 nullptr（表示使用默认流）。

**约束说明**：

- `handle` 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`。
- `matmulDescr` 不可为 nullptr，且其绑定的 matA/matB 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- **唯一稀疏侧**：matA 与 matB 须恰有一个为结构化稀疏描述符（经 `aclsparseLtStructuredDescriptorInit` 创建）；两者均非稀疏或均为稀疏时返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **批量约束**：稀疏侧 numBatches 须 ≥ 1（默认 1；经 `aclsparseLtMatDescSetAttribute` 的 `ACLSPARSELT_MAT_NUM_BATCHES` 设置，A/B/C/D 四侧须一致，由 Matmul 描述符初始化校验）。numBatches > 1 时为批量校验（`d_valid` = 全部 batch 判定的 OR，batch b 基址 = `d_in + b×batchStride` 元素），batchStride（`ACLSPARSELT_MAT_BATCH_STRIDE`，元素数）须 > 0 且为 32 字节/元素字节数的整数倍（FP32 为 8、FP16/BF16 为 16、INT8 为 32 的倍数，与 ld 同款对齐），否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；全量跨度 `(numBatches−1)×batchStride + (rows−1)×ld + cols` 换算字节后溢出时同样返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。numBatches = 1 时 batchStride 不参与判定。
- **数据类型**：稀疏侧支持 `ACL_FLOAT`（FP32）/ `ACL_FLOAT16`（FP16）/ `ACL_BF16`（BF16）/ `ACL_INT8`（INT8），其他类型返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- `d_in` 须 32 字节对齐（作为 DataCopy 搬运源地址的硬件对齐要求，严于 cuSPARSELt 的 16 字节惯例）且不可为 nullptr，`d_valid` 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- **维度约束**：扫描维度与 ld 须为 32 字节/元素字节数的整数倍（结构化描述符初始化保证：FP32 为 8、FP16/BF16 为 16、INT8 为 32）；rows/cols/ld 须非 0 且不超过 INT32_MAX（2147483647，超长 ld 属防御性拦截，防止内部行偏移地址运算溢出）；ld 须不小于物理行长（沿行校验时为行长维度、沿列校验时为列数）。违反时返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；ld 大于物理行长时 padding 区数据不参与判定。
- **内存布局**：`d_in` 按行主序物理存储解释（行步长为稀疏侧描述符 ld），order 仅决定校验方向。
- **异步执行**：算子内部先在同 stream 上将 `d_valid` 预清零，再启动校验 kernel；内部不执行 `aclrtSynchronizeStream`，调用方读取 `d_valid` 前须自行同步 stream。
- `stream` 可为 nullptr（表示使用默认流）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为 nullptr
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matmulDescr/matA/matB/d_in/d_valid 为 nullptr、d_in 未 32 字节对齐、维度/ld 非法、批量参数非法（numBatches<1，或 numBatches>1 时 batchStride≤0/非 32B 对齐）、扫描跨度溢出
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：matA/matB 无唯一稀疏侧、数据类型非 FP32/FP16/BF16/INT8
- `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`：平台核数/UB 资源不足
- `ACL_SPARSE_STATUS_EXECUTION_FAILED`：d_valid 预清零或 kernel 启动失败

**调用示例**：

```cpp
#include "acl/acl.h"
#include "cann_ops_sparseLt.h"

#define CHECK_RET(cond, action) \
    do {                        \
        if (!(cond)) {          \
            action;             \
            goto cleanup;       \
        }                       \
    } while (0)

// 校验 m×k 的 FP32 矩阵 A 是否满足 1:2 结构化稀疏条件（每 2 个元素至多 1 个非零）
int PruneCheckExample()
{
    // 变量声明须位于首个 goto cleanup 之前：C++ 禁止 goto 跨越带初始化的变量声明
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclsparseLtHandle_t handle = nullptr;
    aclsparseLtMatDescriptor_t matA = nullptr, matB = nullptr, matC = nullptr, matD = nullptr;
    aclsparseLtMatmulDescriptor_t matmulDesc = nullptr;
    void *dA = nullptr;     // Device 侧待校验矩阵（m×k FP32，行主序，32 字节对齐）
    int *dValid = nullptr;  // Device 侧校验结果 int 标量
    int hValid = -1;        // Host 侧校验结果
    int ret = 0;
    int64_t m = 16, k = 32, n = 128;  // k 须为 8 的倍数（FP32 结构化描述符对齐要求）

    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, aclFinalize(); return -1);
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, aclrtResetDevice(deviceId); aclFinalize(); return -1);
    CHECK_RET(aclsparseLtInit(&handle) == ACL_SPARSE_STATUS_SUCCESS, ret = -1);

    // 分配 Device 内存，并将待校验数据从 Host 拷入 dA（aclrtMemcpy H2D，此处略）
    CHECK_RET(aclrtMalloc(&dA, static_cast<size_t>(m) * k * sizeof(float),
                          ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, ret = -1);
    CHECK_RET(aclrtMalloc(reinterpret_cast<void**>(&dValid), sizeof(int),
                          ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, ret = -1);

    // 1. 构建描述符：A 为结构化稀疏（FP32 下分组为 1:2，即每 2 个元素至多 1 个非零）；
    //    B/C/D 为占位稠密描述符——仅执行校验时 B/C/D 数据不被消费，
    //    但 Matmul 描述符初始化须绑定四个矩阵，维度须与 A 一致（B: k×n，C/D: m×n）
    aclsparseLtStructuredDescriptorInit(&handle, &matA, m, k, k, 16,
        ACL_FLOAT, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    aclsparseLtDenseDescriptorInit(&handle, &matB, k, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matC, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matD, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);

    // 2. 构建 Matmul 描述符（opA=NON_TRANSPOSE + ROW → 沿行方向校验）
    aclsparseLtMatmulDescriptorInit(&handle, &matmulDesc,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &matA, &matB, &matC, &matD, ACL_SPARSE_COMPUTE_32F);

    // 3. 执行校验：dA -> dValid
    //    第二参为 const 描述符指针的指针（aclsparseLtConstMatmulDescriptor_t*）：
    //    先声明 const 版本描述符再传地址即可自然匹配类型
    {
        aclsparseLtConstMatmulDescriptor_t constMatmulDesc = matmulDesc;
        CHECK_RET(aclsparseLtSpMMAPruneCheck(&handle, &constMatmulDesc, dA, dValid, stream)
                      == ACL_SPARSE_STATUS_SUCCESS, ret = -1);
    }

    // 4. 读取 d_valid 前须同步 stream（算子内部不同步，调用方负责）
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, ret = -1);
    CHECK_RET(aclrtMemcpy(&hValid, sizeof(int), dValid, sizeof(int),
                          ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS, ret = -1);
    // hValid == 0：满足 1:2 结构化稀疏约束；hValid == 1：不满足

cleanup:
    // 5. 清理资源（按依赖逆序销毁，异常路径与正常路径统一从此处释放）
    if (matmulDesc != nullptr) { aclsparseLtMatmulDescriptorDestroy(&matmulDesc); }
    if (matA != nullptr) { aclsparseLtMatDescriptorDestroy(&matA); }
    if (matB != nullptr) { aclsparseLtMatDescriptorDestroy(&matB); }
    if (matC != nullptr) { aclsparseLtMatDescriptorDestroy(&matC); }
    if (matD != nullptr) { aclsparseLtMatDescriptorDestroy(&matD); }
    if (handle != nullptr) { aclsparseLtDestroy(&handle); }
    if (dA != nullptr) { aclrtFree(dA); }
    if (dValid != nullptr) { aclrtFree(dValid); }
    if (stream != nullptr) { aclrtDestroyStream(stream); }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
```

完整约束说明、数学原理与完整可编译调用示例见 [PruneCheck 算子说明](../../sparseLt/prune_check/README.md)。

---

### aclsparseLtSpMMACompressedSize

```c
aclsparseStatus_t aclsparseLtSpMMACompressedSize(
    const aclsparseLtHandle_t* handle,
    const aclsparseLtMatmulPlan_t* plan,
    size_t* compressedSize,
    size_t* compressBufferSize);
```

**功能**：查询结构化稀疏矩阵压缩后及临时空间所需的存储大小（字节），用于在调用 `aclsparseLtSpMMACompress` 前分配 Device 内存。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `plan`（IN）：HOST，已初始化的执行计划，从中读取稀疏侧矩阵的描述符和操作类型。
- `compressedSize`（OUT）：HOST，返回压缩后矩阵所需的存储大小（字节）。
- `compressBufferSize`（OUT）：HOST，返回压缩临时空间大小（字节），当前为 0。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 或其内部句柄为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：plan 无效、输出指针为空或矩阵参数不合法

完整返回码及约束见 [SpMMACompress 算子说明](../../sparseLt/spmma_compress/README.md)。

---

### aclsparseLtSpMMACompress

```c
aclsparseStatus_t aclsparseLtSpMMACompress(
    const aclsparseLtHandle_t* handle,
    const aclsparseLtMatmulPlan_t* plan,
    const void* d_dense,
    void* d_compressed,
    void* d_compressed_buffer,
    aclrtStream stream);
```

**功能**：将结构化稀疏矩阵压缩为紧凑存储。输入须满足 FP32 的 1:2 或 FP16/BF16/INT8 的 2:4 稀疏约束。算子在指定 stream 上异步执行。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `plan`（IN）：HOST，已初始化的执行计划，从中读取稀疏侧矩阵的描述符和操作类型。
- `d_dense`（IN）：DEVICE，待压缩的稀疏矩阵指针，存储布局须与描述符一致。
- `d_compressed`（OUT）：DEVICE，压缩结果输出指针，容量至少为查询所得的 compressedSize 字节。
- `d_compressed_buffer`（INOUT，可选）：DEVICE，压缩临时空间，当前忽略，可为 nullptr。
- `stream`（IN）：HOST，ACL 流，可为 nullptr（表示使用默认流）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：Host 校验和启动调用成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 或其内部句柄为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：plan 无效、输入输出指针为空或矩阵及内存参数不合法
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：不支持的数据类型

完整返回码、输入约束和输出布局见 [SpMMACompress 算子说明](../../sparseLt/spmma_compress/README.md)。

---

## 枚举说明

### aclsparseLtSparsity_t

结构化稀疏模式枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_LT_SPARSITY_50_PERCENT` | 2:4 结构化稀疏（50% 稀疏度） |
| `ACL_SPARSE_LT_SPARSITY_NONE` | 稠密矩阵（用于 dense×dense 路径，标识非结构化稀疏矩阵） |

### aclsparseComputeType_t

计算精度枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_COMPUTE_16F` | FP16 计算精度 |
| `ACL_SPARSE_COMPUTE_32F` | FP32 计算精度 |
| `ACL_SPARSE_COMPUTE_32I` | INT32 计算精度 |

### aclsparseLtMatmulAlg_t

matmul 算法模式枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_LT_MATMUL_ALG_DEFAULT` | 默认算法 |

### aclsparseLtPruneAlg_t

剪枝算法枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACLSPARSELT_PRUNE_SPMMA_TILE` | 逐 tile 剪枝，以 tile 为单位枚举所有满足 2:4 约束的有效配置，选 L1-norm 最大的配置 |
| `ACLSPARSELT_PRUNE_SPMMA_STRIP` | 逐分组剪枝，每组独立选择绝对值最大的元素，不跨组优化 |

### aclsparseLtLibraryPropertyType_t

库属性类型枚举（用于 `aclsparseLtGetProperty`）：

| 枚举值 | 说明 |
|--------|------|
| `ACLSPARSELT_MAJOR_VERSION` | 主版本号 |
| `ACLSPARSELT_MINOR_VERSION` | 次版本号 |
| `ACLSPARSELT_PATCH_LEVEL` | 补丁版本号 |

### aclsparseLtMatmulAlgAttribute_t

matmul 算法选择属性枚举（用于 `aclsparseLtMatmulAlgSetAttribute` / `aclsparseLtMatmulAlgGetAttribute`）：

| 枚举值 | 说明 |
|--------|------|
| `ACLSPARSELT_MATMUL_ALG_CONFIG_ID` | 算法配置 ID（0 或 1），影响 cube tiling |
| `ACLSPARSELT_MATMUL_SEARCH_ITERATIONS` | 搜索迭代次数（默认 5） |
| `ACLSPARSELT_MATMUL_SPLIT_K` | Split-K 切分因子，取值范围 [1, K] |
| `ACLSPARSELT_MATMUL_SPLIT_K_MODE` | Split-K 模式（`ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL` / `ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS`） |
| `ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID` | 只读：返回算法配置 ID 上限（返回 2） |
| `ACLSPARSELT_MATMUL_SPLIT_K_BUFFERS` | Split-K 缓冲区数，取值范围 [0, splitK-1] |

### aclsparseLtMatmulDescAttribute_t

matmul 描述符属性枚举（用于 `aclsparseLtMatmulDescSetAttribute` / `aclsparseLtMatmulDescGetAttribute`）：

| 枚举值 | 编号 | 值类型 | 默认值 | 说明 |
|--------|------|--------|--------|------|
| `ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING` | 0 | int | 0 | alpha 逐行向量缩放开关，0=禁用（标量模式），非 0=启用（alpha 为 Device float[M]） |
| `ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING` | 1 | int | 0 | beta 逐行向量缩放开关，启用时隐含启用 ALPHA_VECTOR_SCALING（cuSPARSELt 语义） |
| `ACLSPARSELT_MATMUL_BIAS_POINTER` | 2 | void* | NULL | bias 向量 Device 指针，长度 = m。dtype 与 C 矩阵相同（FP32 路径 bias 为 FP32，FP16 路径 bias 为 FP16，BF16 路径 bias 为 BF16），INT8 路径 bias 为 FP32（cuSPARSELt 规定）。NULL=不启用 bias |
| `ACLSPARSELT_MATMUL_BIAS_STRIDE` | 3 | int64_t | 0 | batch 间 bias 步长（元素数），0=所有 batch 共用同一 bias |
| `ACLSPARSELT_MATMUL_ACTIVATION_RELU` | 4 | int | 0 | ReLU 开关，0=禁用，非 0=启用。与 GeLU 互斥 |
| `ACLSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND` | 5 | float | FLT_MAX | ReLU 上界，计算 `min(upperBound, D)` |
| `ACLSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD` | 6 | float | 0.0f | ReLU 阈值，计算 `max(threshold, D)` |
| `ACLSPARSELT_MATMUL_ACTIVATION_GELU` | 7 | int | 0 | GeLU 开关，0=禁用，非 0=启用。与 ReLU 互斥 |
| `ACLSPARSELT_MATMUL_ACTIVATION_GELU_SCALING` | 8 | float | 1.0f | GeLU 缩放系数。设置此属性会隐含启用 GeLU（cuSPARSELt 语义：It implies ACTIVATION_GELU）。若 ReLU 已启用则不自动启用 GeLU（保持互斥语义） |

### aclsparseLtMatDescAttribute_t

矩阵描述符属性枚举（用于 `aclsparseLtMatDescSetAttribute` / `aclsparseLtMatDescGetAttribute`），用于 batch 批量计算配置：

| 枚举值 | 编号 | 值类型 | 默认值 | 说明 |
|--------|------|--------|--------|------|
| `ACLSPARSELT_MAT_NUM_BATCHES` | 0 | int32_t | 1 | batch 数量，须 >= 1。A/B/C/D 四个矩阵的 numBatches 须一致（`aclsparseLtMatmulDescriptorInit` 校验） |
| `ACLSPARSELT_MAT_BATCH_STRIDE` | 1 | int64_t | 0 | batch 间步长（元素数），每个矩阵独立设置 |

### aclsparseLtSplitKMode_t

Split-K 模式枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL` | 融合单 kernel 路径，effectiveSplitK=1，GEMM+reduction 在单次 launch 中完成 |
| `ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS` | 两 kernel 路径，matmul kernel + epilogue kernel，splitK 增加并行度 |

### 公共枚举

以下枚举为 aclsparse 与 aclsparseLt 共用，定义于 `cann_ops_sparse.h`，详细说明请参考 [api_list.md 枚举说明](./api_list.md#枚举说明)：

- `aclsparseStatus_t`：返回状态码
- `aclsparseOperation_t`：稀疏操作类型（`ACL_SPARSE_OP_NON_TRANSPOSE` / `ACL_SPARSE_OP_TRANSPOSE` / `ACL_SPARSE_OP_CONJUGATE_TRANSPOSE`）
- `aclsparseOrder_t`：稠密矩阵布局（`ACL_SPARSE_ORDER_ROW` / `ACL_SPARSE_ORDER_COL`）

---

## 推荐调用流程

aclsparseLt 矩阵乘法的工作流分为初始化、描述符构建、计划构建、执行、清理五个阶段。

1. 调用 `aclsparseLtInit` 创建库上下文。
2. 对稀疏侧矩阵（A 或 B）调用 `aclsparseLtStructuredDescriptorInit` 创建结构化描述符，对稠密侧矩阵调用 `aclsparseLtDenseDescriptorInit` 创建稠密描述符。如需 batch 批量计算，通过 `aclsparseLtMatDescSetAttribute` 设置 `NUM_BATCHES` / `BATCH_STRIDE`（须在步骤 3 之前）。
3. 调用 `aclsparseLtMatmulDescriptorInit` 构建 Matmul 描述符，传入 opA/opB、四个矩阵描述符（matA/matB/matC/matD）与 computeType（Init 时校验 A/B/C/D 四个矩阵 numBatches 一致性）。
4. （可选）通过 `aclsparseLtMatmulDescSetAttribute` 设置 bias（`BIAS_POINTER` / `BIAS_STRIDE`）、activation（`ACTIVATION_RELU` / `RELU_UPPERBOUND` / `RELU_THRESHOLD` / `ACTIVATION_GELU` / `GELU_SCALING`）、向量缩放（`ALPHA_VECTOR_SCALING` / `BETA_VECTOR_SCALING`）。上述描述符属性在执行期读取，PlanInit 之后修改对后续执行生效。
5. （可选）调用 `aclsparseLtSpMMAPrune` 对稠密矩阵执行 2:4 剪枝，生成稀疏侧输入。自行实现剪枝的输入（未经 `aclsparseLtSpMMAPrune` 产出）可调用 `aclsparseLtSpMMAPruneCheck` 校验是否满足 2:4 约束（`aclsparseLtSpMMAPrune` 的产出保证通过该检查，无需重复校验）。
6. 调用 `aclsparseLtMatmulAlgSelectionInit` 构建算法选择描述符。
7. 调用 `aclsparseLtMatmulPlanInit` 构建执行计划（读取上述全部属性计算 tiling）。
8. 调用 `aclsparseLtMatmulGetWorkspace` 查询所需 workspace 大小并分配设备内存。
9. （可选，auto-tuning）调用 `aclsparseLtMatmulSearch` 搜索最优算法配置（阻塞调用；搜索后建议重新调用 `aclsparseLtMatmulGetWorkspace` 校验），之后以选中配置多次调用 `aclsparseLtMatmul` 执行结构化稀疏矩阵乘法。bias 与 activation 已通过描述符属性配置，执行接口不额外传参。
10. 按依赖逆序销毁执行计划、算法选择描述符、Matmul 描述符、矩阵描述符，最后调用 `aclsparseLtDestroy` 释放库句柄。

压缩需先创建 Plan，具体步骤见[压缩调用说明](../../sparseLt/spmma_compress/README.md#调用说明)。压缩结果尚不支持作为现有 Matmul 或 MatmulSearch 的输入。

---

## 最小示例（伪代码）

```c
#include "acl/acl.h"
#include "cann_ops_sparseLt.h"

int aclsparseLtExample()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclsparseLtHandle_t handle = nullptr;
    aclsparseLtMatDescriptor_t matA = nullptr, matB = nullptr, matC = nullptr, matD = nullptr;
    aclsparseLtMatmulDescriptor_t matmulDesc = nullptr;
    aclsparseLtMatmulAlgSelection_t algSelection = nullptr;
    aclsparseLtMatmulPlan_t plan = nullptr;
    void *dA = nullptr, *dAPruned = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    // 1. 初始化库句柄
    aclsparseLtInit(&handle);

    // 2. 构建矩阵描述符：A 为结构化稀疏（2:4），m×k，FP32，行主序
    int64_t m = 16, k = 32, n = 128;
    aclsparseLtStructuredDescriptorInit(&handle, &matA, m, k, k, 16,
        ACL_FLOAT, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    aclsparseLtDenseDescriptorInit(&handle, &matB, k, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matC, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matD, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);

    // 3. 构建 Matmul 描述符
    aclsparseLtMatmulDescriptorInit(&handle, &matmulDesc,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &matA, &matB, &matC, &matD, ACL_SPARSE_COMPUTE_32F);

    // 3.1（可选）配置 bias + ReLU activation（执行期读取，PlanInit 前后设置均生效）
    //     bias：长度 = m 的逐行广播向量（dtype 与 C 相同，INT8 路径为 FP32），Device 内存
    //     activation：ReLU（threshold=0.0f、upperBound=FLT_MAX 退化为标准 ReLU）
    void *dBias = nullptr;
    {
        float hBias[m];  // 长度 = m，此处填 0.5f 仅作示意
        for (int64_t i = 0; i < m; i++) { hBias[i] = 0.5f; }
        aclrtMalloc(&dBias, static_cast<size_t>(m) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dBias, static_cast<size_t>(m) * sizeof(float),
                    hBias, static_cast<size_t>(m) * sizeof(float),
                    ACL_MEMCPY_HOST_TO_DEVICE);
    }
    int64_t biasStride = 0;  // 0=所有 batch 共用同一 bias（单 batch 时无影响）
    int reluEnable = 1;
    aclsparseLtMatmulDescSetAttribute(&handle, &matmulDesc,
        ACLSPARSELT_MATMUL_BIAS_POINTER, &dBias, sizeof(void*));
    aclsparseLtMatmulDescSetAttribute(&handle, &matmulDesc,
        ACLSPARSELT_MATMUL_BIAS_STRIDE, &biasStride, sizeof(int64_t));
    aclsparseLtMatmulDescSetAttribute(&handle, &matmulDesc,
        ACLSPARSELT_MATMUL_ACTIVATION_RELU, &reluEnable, sizeof(int));
    // 说明：GeLU 用法同理，设置 ACLSPARSELT_MATMUL_ACTIVATION_GELU=1，
    //       可选设置 ACLSPARSELT_MATMUL_ACTIVATION_GELU_SCALING（默认 1.0f）；
    //       不可同时启用 ReLU 与 GeLU（互斥，返回 INVALID_VALUE）。

    // 4. 执行 2:4 剪枝：dA -> dAPruned
    aclsparseLtSpMMAPrune(&handle, &matmulDesc,
        dA, dAPruned, ACLSPARSELT_PRUNE_SPMMA_STRIP, stream);
    aclrtSynchronizeStream(stream);  // 算子内部不同步，调用方负责同步

    // 5. 构建算法选择描述符
    aclsparseLtMatmulAlgSelectionInit(&handle, &algSelection, &matmulDesc,
        ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);

    // 6. 构建执行计划
    aclsparseLtMatmulPlanInit(&handle, &plan, &matmulDesc, &algSelection);

    // 7. 获取 workspace 大小并分配
    size_t workspaceSize = 0;
    aclsparseLtMatmulGetWorkspace(&handle, &plan, &workspaceSize);
    void *workspace = nullptr;
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 8. 执行结构化稀疏矩阵乘法
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream streams[] = {stream};
    aclsparseLtMatmul(&handle, &plan, &alpha, dAPruned, dB, &beta, dC, dD,
        workspace, streams, 1);
    aclrtSynchronizeStream(stream);  // 算子内部不同步，调用方负责同步

    // 9. 清理资源（按依赖逆序销毁）
    aclsparseLtMatmulPlanDestroy(&plan);
    aclsparseLtMatmulAlgSelectionDestroy(&algSelection);
    aclsparseLtMatmulDescriptorDestroy(&matmulDesc);
    aclsparseLtMatDescriptorDestroy(&matA);
    aclsparseLtMatDescriptorDestroy(&matB);
    aclsparseLtMatDescriptorDestroy(&matC);
    aclsparseLtMatDescriptorDestroy(&matD);
    aclsparseLtDestroy(&handle);

    aclrtFree(dBias);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

---

## 备注

- 接口能力与属性支持范围以当前实现版本为准，详细限制请参考各算子目录下的README文档（如`sparseLt/prune/README.md`）及头文件内注释。
- 若文档描述与头文件声明不一致，请**以头文件声明与实际实现行为为准**。

