# 稀疏算子接口（aclsparse）

## 使用说明

为方便调用稀疏算子，提供一套基于 C 的 API（以 aclsparse 为前缀），主要用于稀疏矩阵运算场景。

调用稀疏算子API时，需引用依赖的头文件和库文件。

- 依赖的头文件：`cann_ops_sparse.h`
- 依赖的库文件：稀疏算子库文件

## 接口列表

| 接口名 | 说明 |
|--------|------|
| [aclsparseCreate](#aclsparsecreate) | 创建稀疏矩阵处理器 |
| [aclsparseDestroy](#aclsparsedestroy) | 销毁稀疏矩阵处理器 |
| [aclsparseSetStream](#aclsparsesetstream) | 设置处理器使用的 stream |
| [aclsparseGetStream](#aclsparsegetstream) | 获取处理器当前的 stream |
| [aclsparseGetVersion](#aclsparsegetversion) | 获取版本号 |
| [aclsparseCreateDnVec](#aclsparsecreatednvec) | 创建稠密向量 |
| [aclsparseDestroyDnVec](#aclsparsedestroydnvec) | 销毁稠密向量描述符 |
| [aclsparseDnVecGet](#aclsparsednvecget) | 获取稠密向量描述符的全部字段 |
| [aclsparseDnVecGetValues](#aclsparsednvecgetvalues) | 获取稠密向量描述符的 values 指针 |
| [aclsparseDnVecSetValues](#aclsparsednvecsetvalues) | 设置稠密向量描述符的 values 指针 |
| [aclsparseCreateSpVec](#aclsparsecreatespvec) | 创建稀疏向量 |
| [aclsparseDestroySpVec](#aclsparsedestroyspvec) | 销毁稀疏向量描述符 |
| [aclsparseSpVecGet](#aclsparsespvecget) | 获取稀疏向量描述符的全部字段 |
| [aclsparseSpVecGetIndexBase](#aclsparsespvecgetindexbase) | 获取稀疏向量的索引基值 |
| [aclsparseSpVecGetValues](#aclsparsespvecgetvalues) | 获取稀疏向量描述符的 values 指针 |
| [aclsparseSpVecSetValues](#aclsparsespvecsetvalues) | 设置稀疏向量描述符的 values 指针 |
| [aclsparseGather](#aclsparsegather) | 从稠密向量中按稀疏索引收集元素 |
| [aclsparseCreateCsr](#aclsparsecreatecsr) | 创建CSR格式稀疏矩阵 |
| [aclsparseCreateCsc](#aclsparsecreatecsc) | 创建CSC格式稀疏矩阵 |
| [aclsparseCreateCoo](#aclsparsecreatecoo) | 创建COO格式稀疏矩阵 |
| [aclsparseCreateConstCoo](#aclsparsecreateconstcoo) | 创建只读(const)COO格式稀疏矩阵 |
| [aclsparseCooGet](#aclsparsecooget) | 获取COO稀疏矩阵描述符的全部字段 |
| [aclsparseConstCooGet](#aclsparseconstcooget) | 获取只读(const)COO稀疏矩阵描述符的全部字段 |
| [aclsparseDestroySpMat](#aclsparsedestroyspmat) | 销毁稀疏矩阵对象 |
| [aclsparseSpMatGetFormat](#aclsparsespmatgetformat) | 获取稀疏矩阵的存储格式 |
| [aclsparseSpMatGetValues](#aclsparsespmatgetvalues) | 获取稀疏矩阵描述符的 values 指针 |
| [aclsparseSpMatSetValues](#aclsparsespmatsetvalues) | 设置稀疏矩阵描述符的 values 指针 |
| [aclsparseSpMatGetSize](#aclsparsespmatgetsize) | 获取稀疏矩阵的行数、列数和非零元素个数 |
| [aclsparseSpMVGetBufferSize](#aclsparsespmvgetbuffersize) | 获取SpMV缓冲区大小 |
| [aclsparseSpMVPreprocess](#aclsparsespmvpreprocess) | SpMV预处理（**暂未支持**） |
| [aclsparseSpMV](#aclsparsespmv) | 稀疏矩阵向量乘法 |
| [aclsparseCreateDnMat](#aclsparsecreatednmat) | 创建稠密矩阵 |
| [aclsparseDestroyDnMat](#aclsparsedestroydnmat) | 销毁稠密矩阵描述符 |
| [aclsparseDnMatGet](#aclsparsednmatget) | 获取稠密矩阵描述符的全部字段 |
| [aclsparseDnMatGetValues](#aclsparsednmatgetvalues) | 获取稠密矩阵描述符的 values 指针 |
| [aclsparseDnMatSetValues](#aclsparsednmatsetvalues) | 设置稠密矩阵描述符的 values 指针 |
| [aclsparseSpMMGetBufferSize](#aclsparsespmmgetbuffersize) | 获取SpMM缓冲区大小 |
| [aclsparseSpMMPreprocess](#aclsparsespmmpreprocess) | SpMM预处理 |
| [aclsparseSpMM](#aclsparsespmm) | 稀疏矩阵-稠密矩阵乘法 |
| [aclsparseSpMVOp_bufferSize](#aclsparsespmvop_buffersize) | 获取 SpMVOp workspace 大小 |
| [aclsparseSpMVOp_createDescr](#aclsparsespmvop_createdescr) | 创建 SpMVOp 描述符并执行预处理 |
| [aclsparseSpMVOp_destroyDescr](#aclsparsespmvop_destroydescr) | 销毁 SpMVOp 描述符 |
| [aclsparseSpMVOp_createPlan](#aclsparsespmvop_createplan) | 创建 SpMVOp 执行计划 |
| [aclsparseSpMVOp_destroyPlan](#aclsparsespmvop_destroyplan) | 销毁 SpMVOp 执行计划 |
| [aclsparseSpMVOp_setGlobalUserData](#aclsparsespmvop_setglobaluserdata) | 设置 SpMVOp epilogue 用户数据（NPU 侧为 no-op） |
| [aclsparseSpMVOp](#aclsparsespmvop) | 执行稀疏矩阵-向量乘法 |
| [aclsparseSpMMOp_bufferSize](#aclsparsespmmop_buffersize) | 获取 workspace 大小 |
| [aclsparseSpMMOp_createDescr](#aclsparsespmmop_createdescr) | 创建描述符并执行预处理 |
| [aclsparseSpMMOp_destroyDescr](#aclsparsespmmop_destroydescr) | 销毁描述符 |
| [aclsparseSpMMOp_createPlan](#aclsparsespmmop_createplan) | 创建执行计划 |
| [aclsparseSpMMOp_destroyPlan](#aclsparsespmmop_destroyplan) | 销毁执行计划 |
| [aclsparseSpMMOp_setGlobalUserData](#aclsparsespmmop_setglobaluserdata) | 设置 epilogue 用户数据（NPU 侧为 no-op） |
| [aclsparseSpMMOp](#aclsparsespmmop) | 稀疏矩阵-稠密矩阵乘法（descriptor/plan 生命周期模式） |
| [aclsparseSgtsv2](#aclsparsesgtsv2) | 三对角线性方程组带选主元求解（FP32，Multiple RHS） |
| [aclsparseSgtsv2_bufferSizeExt](#aclsparsesgtsv2_buffersizeext) | 查询 aclsparseSgtsv2 所需工作区大小 |
| [aclsparseXcscsort_bufferSizeExt](#aclsparsexcscsort_buffersizeext) | 获取CSC排序缓冲区大小 |
| [aclsparseXcscsort](#aclsparsexcscsort) | CSC格式按列稳定排序 |
| [aclsparseLtMatmulAlgSetAttribute](#aclsparseltmatmulalgsetattribute) | 设置结构化稀疏矩阵乘法的算法属性 |
| [aclsparseLtSpMMAPrune](#aclsparseltspmmaprune) | 对稠密矩阵执行 2:4 结构化稀疏剪枝 |

## 接口详情

### aclsparseCreate

```c
aclsparseStatus_t aclsparseCreate(aclsparseHandle_t *handle);
```

**功能**：创建稀疏矩阵处理器。处理器的 stream 默认为 `nullptr`（默认 stream），如需指定请调用 `aclsparseSetStream`。

**参数说明**：

- `handle`（IN/OUT）：HOST，返回创建的稀疏矩阵处理器句柄。调用前 `*handle` 必须为 `nullptr`。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseDestroy

```c
aclsparseStatus_t aclsparseDestroy(aclsparseHandle_t handle);
```

**功能**：销毁稀疏矩阵处理器。stream 由用户托管，此接口不会销毁 stream。

**参数说明**：

- `handle`（IN）：稀疏矩阵处理器句柄。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSetStream

```c
aclsparseStatus_t aclsparseSetStream(aclsparseHandle_t handle, aclrtStream stream);
```

**功能**：设置处理器后续算子执行所使用的 stream。

**参数说明**：

- `handle`（IN）：稀疏矩阵处理器句柄。
- `stream`（IN）：要绑定的 stream，`nullptr` 表示使用默认 stream。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseGetStream

```c
aclsparseStatus_t aclsparseGetStream(aclsparseHandle_t handle, aclrtStream *stream);
```

**功能**：获取处理器当前绑定的 stream。

**参数说明**：

- `handle`（IN）：稀疏矩阵处理器句柄。
- `stream`（OUT）：返回当前 stream。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseGetVersion

```c
aclsparseStatus_t aclsparseGetVersion(aclsparseHandle_t handle, int *version);
```

**功能**：获取 ops-sparse 版本号。版本号编码为 `MAJOR * 10000 + MINOR * 100 + PATCH`，如 `10000` 表示 1.0.0。

**参数说明**：

- `handle`（IN）：HOST，稀疏矩阵处理器句柄，可为 `nullptr`。
- `version`（OUT）：HOST，输出参数，接收版本号。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`version` 为空
- 其他值：失败

---

### aclsparseCreateDnVec

```c
aclsparseStatus_t aclsparseCreateDnVec(
    aclsparseDnVecDescr_t *dnVecDescr,
    int64_t size,
    void *values,
    aclDataType valueType
);
```

**功能**：创建一个稠密向量。

**参数说明**：

- `dnVecDescr`（IN/OUT）：HOST，稀疏向量描述符。
- `size`（IN）：HOST，向量的大小。
- `values`（IN）：DEVICE，向量的值。
- `valueType`（IN）：HOST，值的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseDestroyDnVec

```c
aclsparseStatus_t aclsparseDestroyDnVec(aclsparseConstDnVecDescr_t dnVecDescr);
```

**功能**：销毁稠密向量描述符。

**参数说明**：

- `dnVecDescr`（IN）：HOST，要销毁的稠密向量描述符。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseDnVecGet

```c
aclsparseStatus_t aclsparseDnVecGet(
    aclsparseDnVecDescr_t dnVecDescr,
    int64_t *size,
    void **values,
    aclDataType *valueType
);
```

**功能**：获取稠密向量描述符的全部字段。

**参数说明**：

- `dnVecDescr`（IN）：HOST，稠密向量描述符。
- `size`（OUT）：HOST，向量的大小。
- `values`（OUT）：DEVICE，向量的值指针。
- `valueType`（OUT）：HOST，值的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`dnVecDescr` 为空指针

---

### aclsparseDnVecGetValues

```c
aclsparseStatus_t aclsparseDnVecGetValues(
    aclsparseDnVecDescr_t dnVecDescr,
    void **values
);
```

**功能**：获取稠密向量描述符的 values 指针。

**参数说明**：

- `dnVecDescr`（IN）：HOST，稠密向量描述符。
- `values`（OUT）：DEVICE，向量的值指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`dnVecDescr` 为空指针

---

### aclsparseDnVecSetValues

```c
aclsparseStatus_t aclsparseDnVecSetValues(
    aclsparseDnVecDescr_t dnVecDescr,
    void *values
);
```

**功能**：设置稠密向量描述符的 values 指针。

**参数说明**：

- `dnVecDescr`（IN）：HOST，稠密向量描述符。
- `values`（IN）：DEVICE，向量的值指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`dnVecDescr` 为空指针

---

### aclsparseCreateSpVec

```c
aclsparseStatus_t aclsparseCreateSpVec(
    aclsparseSpVecDescr_t *spVecDescr,
    int64_t size,
    int64_t nnz,
    void *indices,
    void *values,
    aclsparseIndexType_t idxType,
    aclsparseIndexBase_t idxBase,
    aclDataType valueType
);
```

**功能**：创建一个稀疏向量描述符。

**参数说明**：

- `spVecDescr`（IN/OUT）：HOST，稀疏向量描述符。
- `size`（IN）：HOST，稀疏向量的大小（元素总数）。
- `nnz`（IN）：HOST，稀疏向量的非零元素个数。
- `indices`（IN）：DEVICE，索引数组，长度为 `nnz`。
- `values`（IN）：DEVICE，值数组，长度为 `nnz`。
- `idxType`（IN）：HOST，`indices` 的数据类型。
- `idxBase`（IN）：HOST，`indices` 的索引基值。
- `valueType`（IN）：HOST，`values` 的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spVecDescr` 为空指针，或 `size < 0`，或 `nnz < 0`，或 `nnz > size`
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseDestroySpVec

```c
aclsparseStatus_t aclsparseDestroySpVec(aclsparseConstSpVecDescr_t spVecDescr);
```

**功能**：销毁稀疏向量描述符。接受 const 描述符，非 const 描述符同样支持。

**参数说明**：

- `spVecDescr`（IN）：HOST，要销毁的稀疏向量描述符。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功

---

### aclsparseSpVecGet

```c
aclsparseStatus_t aclsparseSpVecGet(
    aclsparseSpVecDescr_t spVecDescr,
    int64_t *size,
    int64_t *nnz,
    void **indices,
    void **values,
    aclsparseIndexType_t *idxType,
    aclsparseIndexBase_t *idxBase,
    aclDataType *valueType
);
```

**功能**：获取稀疏向量描述符的全部字段。输出参数允许为 `nullptr`，此时仅返回请求的字段。

**参数说明**：

- `spVecDescr`（IN）：HOST，稀疏向量描述符。
- `size`（OUT）：HOST，稀疏向量的大小。
- `nnz`（OUT）：HOST，非零元素个数。
- `indices`（OUT）：DEVICE，索引数组指针。
- `values`（OUT）：DEVICE，值数组指针。
- `idxType`（OUT）：HOST，索引数据类型。
- `idxBase`（OUT）：HOST，索引基值。
- `valueType`（OUT）：HOST，值数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spVecDescr` 为空指针

---

### aclsparseSpVecGetIndexBase

```c
aclsparseStatus_t aclsparseSpVecGetIndexBase(
    aclsparseConstSpVecDescr_t spVecDescr,
    aclsparseIndexBase_t *idxBase
);
```

**功能**：获取稀疏向量描述符的索引基值。接受 const 描述符，非 const 描述符同样支持。

**参数说明**：

- `spVecDescr`（IN）：HOST，稀疏向量描述符。
- `idxBase`（OUT）：HOST，索引基值。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spVecDescr` 为空指针

---

### aclsparseSpVecGetValues

```c
aclsparseStatus_t aclsparseSpVecGetValues(
    aclsparseSpVecDescr_t spVecDescr,
    void **values
);
```

**功能**：获取稀疏向量描述符的 values 指针。

**参数说明**：

- `spVecDescr`（IN）：HOST，稀疏向量描述符。
- `values`（OUT）：DEVICE，值数组指针，长度为 `nnz`。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spVecDescr` 为空指针

---

### aclsparseSpVecSetValues

```c
aclsparseStatus_t aclsparseSpVecSetValues(
    aclsparseSpVecDescr_t spVecDescr,
    void *values
);
```

**功能**：设置稀疏向量描述符的 values 指针。

**参数说明**：

- `spVecDescr`（IN）：HOST，稀疏向量描述符。
- `values`（IN）：DEVICE，值数组指针，长度为 `nnz`。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spVecDescr` 为空指针

---

### aclsparseGather

```c
aclsparseStatus_t aclsparseGather(
    aclsparseHandle_t handle,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseSpVecDescr_t vecX);
```

**功能**：从稠密向量 Y 中按稀疏向量 X 的索引数组收集元素，满足 `X.values[i] = Y[X.indices[i] - idxBase]`（i = 0 .. nnz-1）。调用后 `X.values` 被覆写为从 Y 中收集到的值。不需要 workspace，不需要 preprocess 阶段，单步调用即可完成计算。

**产品支持情况**：

- Ascend 950PR / Ascend 950DT（arch35）：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品（arch22）：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品（arch22）：支持

> 说明：arch22 与 arch35 支持的 dtype / 索引类型范围不同，详见 `sparse/gather/README.md`。

**参数说明**：

- `handle`（IN）：HOST，ops-sparse 库上下文句柄，携带 stream，须先调用 `aclsparseSetStream`。
- `vecY`（IN）：HOST，稠密向量 Y 的描述符（数据源），仅读取 values。
- `vecX`（IN/OUT）：HOST，稀疏向量 X 的描述符：读取 indices 和 idxBase，写入 values。

**约束说明**：

- handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`
- vecY、vecX 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- vecX.valueType 与 vecY.valueType 必须一致
- 基数约束：arch35（Ascend 950）要求 `vecX.size == vecY.size` 且 `0 <= nnz <= size`；arch22（Atlas A2/A3）不做基数约束，合法性取决于逐索引落点（`nnz` 可大于 `vecY.size`）
- vecX.indices 中每个索引值 `X.indices[i] - idxBase` 必须在 `[0, vecY.size)` 范围内：arch22 在 Host 侧逐索引校验，越界显式返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；arch35 遵循异步错误协议，合法索引为调用方前置条件
- 支持索引乱序（indices 不需要排序）
- 支持 vecX.indices 中存在重复元素

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：参数非法（空指针等）
- 其他值：失败

---

### 只读(const)稀疏向量描述符构造接口

只读(const)构造接口：数据指针为 `const`，构造出的描述符是 `const` 变体，只能传给接收 `const` 形参的接口。销毁接口（`aclsparseDestroySpVec`）统一接收 `const` 变体，因此 const 与非 const 描述符都可销毁。

```c
aclsparseStatus_t aclsparseCreateConstSpVec(aclsparseConstSpVecDescr_t *spVecDescr, int64_t size,
    int64_t nnz, const void *indices, const void *values, aclsparseIndexType_t idxType,
    aclsparseIndexBase_t idxBase, aclDataType valueType);

aclsparseStatus_t aclsparseConstSpVecGet(aclsparseConstSpVecDescr_t spVecDescr, int64_t *size,
    int64_t *nnz, const void **indices, const void **values, aclsparseIndexType_t *idxType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType);

aclsparseStatus_t aclsparseConstSpVecGetValues(aclsparseConstSpVecDescr_t spVecDescr,
    const void **values);
```

---

### aclsparseCreateCsr

```c
aclsparseStatus_t aclsparseCreateCsr(
    aclsparseSpMatDescr_t *spMatDescr,
    int64_t rows,
    int64_t cols,
    int64_t nnz,
    void *csrRowOffsets,
    void *csrColInd,
    void *csrValues,
    aclsparseIndexType_t csrRowOffsetsType,
    aclsparseIndexType_t csrColIndType,
    aclsparseIndexBase_t idxBase,
    aclDataType valueType
);
```

**功能**：创建一个CSR（Compressed Sparse Row）格式的稀疏矩阵。

> **索引类型支持**：当前版本 SpMV / SpMM 仅支持 `ACL_SPARSE_INDEX_32I`，且 `csrRowOffsetsType` 与 `csrColIndType` 必须相同。传入 `ACL_SPARSE_INDEX_64I` 或 ptr/col 类型不一致时，`aclsparseCreateCsr` 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。

**参数说明**：

- `spMatDescr`（IN/OUT）：HOST，稀疏矩阵描述符。
- `rows`（IN）：HOST，矩阵的行数。
- `cols`（IN）：HOST，矩阵的列数。
- `nnz`（IN）：HOST，矩阵的非零元素个数。
- `csrRowOffsets`（IN）：DEVICE，指向CSR格式行偏移量数组的指针。
- `csrColInd`（IN）：DEVICE，指向CSR格式列索引数组的指针。
- `csrValues`（IN）：DEVICE，指向CSR格式非零元素数组的指针。
- `csrRowOffsetsType`（IN）：HOST，行偏移量数组的数据类型（当前仅支持 `ACL_SPARSE_INDEX_32I`）。
- `csrColIndType`（IN）：HOST，列索引数组的数据类型（当前仅支持 `ACL_SPARSE_INDEX_32I`，且须与 `csrRowOffsetsType` 一致）。
- `idxBase`（IN）：HOST，索引的基值（0或1）。
- `valueType`（IN）：HOST，非零元素的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseCreateCsc

> **支持状态**：该接口可创建 CSC 稀疏矩阵描述符。当前版本 SpMV / SpMM 仅支持 CSR 稀疏矩阵，CSC 描述符不能用于这两个计算接口。

```c
aclsparseStatus_t aclsparseCreateCsc(
    aclsparseSpMatDescr_t *spMatDescr,
    int64_t rows,
    int64_t cols,
    int64_t nnz,
    void *cscColOffsets,
    void *cscRowInd,
    void *cscValues,
    aclsparseIndexType_t cscColOffsetsType,
    aclsparseIndexType_t cscRowIndType,
    aclsparseIndexBase_t idxBase,
    aclDataType valueType
);
```

**功能**：创建一个CSC（Compressed Sparse Column）格式的稀疏矩阵描述符。

**参数说明**：

- `spMatDescr`（IN/OUT）：HOST，稀疏矩阵描述符。
- `rows`（IN）：HOST，矩阵的行数。
- `cols`（IN）：HOST，矩阵的列数。
- `nnz`（IN）：HOST，矩阵的非零元素个数。
- `cscColOffsets`（IN）：DEVICE，指向列偏移数组的指针。
- `cscRowInd`（IN）：DEVICE，指向行索引数组的指针。
- `cscValues`（IN）：DEVICE，指向非零元素值数组的指针。
- `cscColOffsetsType`（IN）：HOST，列偏移数组的数据类型。
- `cscRowIndType`（IN）：HOST，行索引数组的数据类型。
- `idxBase`（IN）：HOST，索引的基础值。
- `valueType`（IN）：HOST，非零元素值的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseCreateCoo

```c
aclsparseStatus_t aclsparseCreateCoo(
    aclsparseSpMatDescr_t *spMatDescr,
    int64_t rows,
    int64_t cols,
    int64_t nnz,
    void *cooRowInd,
    void *cooColInd,
    void *cooValues,
    aclsparseIndexType_t cooIdxType,
    aclsparseIndexBase_t idxBase,
    aclDataType valueType
);
```

**功能**：创建一个 COO（Coordinate）格式的稀疏矩阵。

> **索引类型支持**：COO 描述符支持 `ACL_SPARSE_INDEX_32I` 与 `ACL_SPARSE_INDEX_64I`，cooIdxType 同时约束行/列索引。CooGet 为纯 host 访问器，不涉及 NPU 计算，故两种索引类型均接受（与 cuSPARSE cusparseCreateCoo 一致）。

**参数说明**：

- `spMatDescr`（IN/OUT）：HOST，稀疏矩阵描述符。
- `rows`（IN）：HOST，矩阵的行数。
- `cols`（IN）：HOST，矩阵的列数。
- `nnz`（IN）：HOST，矩阵的非零元素个数。
- `cooRowInd`（IN）：DEVICE，行索引数组，长度为 `nnz`。
- `cooColInd`（IN）：DEVICE，列索引数组，长度为 `nnz`。
- `cooValues`（IN）：DEVICE，非零元素值数组，长度为 `nnz`。
- `cooIdxType`（IN）：HOST，行/列索引的数据类型。
- `idxBase`（IN）：HOST，索引的基值。
- `valueType`（IN）：HOST，非零元素的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spMatDescr` 为空指针，或 `rows` / `cols` / `nnz` 为负
- 其他值：失败

---

### aclsparseCreateConstCoo

```c
aclsparseStatus_t aclsparseCreateConstCoo(
    aclsparseConstSpMatDescr_t *spMatDescr,
    int64_t rows,
    int64_t cols,
    int64_t nnz,
    const void *cooRowInd,
    const void *cooColInd,
    const void *cooValues,
    aclsparseIndexType_t cooIdxType,
    aclsparseIndexBase_t idxBase,
    aclDataType valueType
);
```

**功能**：创建一个只读(const)COO 格式的稀疏矩阵，是 `aclsparseCreateCoo` 的 const 变体：数据指针为 `const`，构造出的描述符为 `aclsparseConstSpMatDescr_t`，只能传给接收 const 形参的接口。索引类型支持同 `aclsparseCreateCoo`。

**参数说明**：

- `spMatDescr`（IN/OUT）：HOST，只读(const)稀疏矩阵描述符。
- `rows`（IN）：HOST，矩阵的行数。
- `cols`（IN）：HOST，矩阵的列数。
- `nnz`（IN）：HOST，矩阵的非零元素个数。
- `cooRowInd`（IN）：DEVICE，行索引数组，长度为 `nnz`。
- `cooColInd`（IN）：DEVICE，列索引数组，长度为 `nnz`。
- `cooValues`（IN）：DEVICE，非零元素值数组，长度为 `nnz`。
- `cooIdxType`（IN）：HOST，行/列索引的数据类型。
- `idxBase`（IN）：HOST，索引的基值。
- `valueType`（IN）：HOST，非零元素的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spMatDescr` 为空指针，或 `rows` / `cols` / `nnz` 为负
- 其他值：失败

---

### aclsparseCooGet

```c
aclsparseStatus_t aclsparseCooGet(
    aclsparseSpMatDescr_t spMatDescr,
    int64_t *rows,
    int64_t *cols,
    int64_t *nnz,
    void **cooRowInd,
    void **cooColInd,
    void **cooValues,
    aclsparseIndexType_t *cooIdxType,
    aclsparseIndexBase_t *idxBase,
    aclDataType *valueType
);
```

**功能**：获取 COO 稀疏矩阵描述符的全部字段。输出参数允许为 `nullptr`，此时仅返回请求的字段。

**参数说明**：

- `spMatDescr`（IN）：HOST，COO 稀疏矩阵描述符。
- `rows`（OUT）：HOST，矩阵的行数。
- `cols`（OUT）：HOST，矩阵的列数。
- `nnz`（OUT）：HOST，矩阵的非零元素个数。
- `cooRowInd`（OUT）：DEVICE，行索引数组指针。
- `cooColInd`（OUT）：DEVICE，列索引数组指针。
- `cooValues`（OUT）：DEVICE，非零元素值数组指针。
- `cooIdxType`（OUT）：HOST，行/列索引的数据类型。
- `idxBase`（OUT）：HOST，索引的基值。
- `valueType`（OUT）：HOST，非零元素的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spMatDescr` 为空指针
- `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`：描述符非 COO 格式
- 其他值：失败

---

### aclsparseConstCooGet

```c
aclsparseStatus_t aclsparseConstCooGet(
    aclsparseConstSpMatDescr_t spMatDescr,
    int64_t *rows,
    int64_t *cols,
    int64_t *nnz,
    const void **cooRowInd,
    const void **cooColInd,
    const void **cooValues,
    aclsparseIndexType_t *cooIdxType,
    aclsparseIndexBase_t *idxBase,
    aclDataType *valueType
);
```

**功能**：获取只读(const)COO 稀疏矩阵描述符的全部字段，是 `aclsparseCooGet` 的 const 变体：描述符为 `aclsparseConstSpMatDescr_t`，读回的设备指针为 `const void**`。输出参数允许为 `nullptr`，此时仅返回请求的字段。

**参数说明**：

- `spMatDescr`（IN）：HOST，只读(const)COO 稀疏矩阵描述符。
- `rows`（OUT）：HOST，矩阵的行数。
- `cols`（OUT）：HOST，矩阵的列数。
- `nnz`（OUT）：HOST，矩阵的非零元素个数。
- `cooRowInd`（OUT）：DEVICE，行索引数组指针。
- `cooColInd`（OUT）：DEVICE，列索引数组指针。
- `cooValues`（OUT）：DEVICE，非零元素值数组指针。
- `cooIdxType`（OUT）：HOST，行/列索引的数据类型。
- `idxBase`（OUT）：HOST，索引的基值。
- `valueType`（OUT）：HOST，非零元素的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spMatDescr` 为空指针
- `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`：描述符非 COO 格式
- 其他值：失败

---

### aclsparseDestroySpMat

```c
aclsparseStatus_t aclsparseDestroySpMat(aclsparseConstSpMatDescr_t spMatDescr);
```

**功能**：销毁稀疏矩阵对象。

**参数说明**：

- `spMatDescr`（IN）：HOST，稀疏矩阵描述符。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMatGetFormat

```c
aclsparseStatus_t aclsparseSpMatGetFormat(
    aclsparseConstSpMatDescr_t spMatDescr,
    aclsparseFormat_t *format
);
```

**功能**：获取稀疏矩阵描述符的存储格式。接受 const 描述符，非 const 描述符同样支持。

**参数说明**：

- `spMatDescr`（IN）：HOST，稀疏矩阵描述符。
- `format`（OUT）：HOST，存储格式。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spMatDescr` 为空指针

---

### aclsparseSpMatGetValues

```c
aclsparseStatus_t aclsparseSpMatGetValues(
    aclsparseSpMatDescr_t spMatDescr,
    void **values
);
```

**功能**：获取稀疏矩阵描述符的 values 指针。

**参数说明**：

- `spMatDescr`（IN）：HOST，稀疏矩阵描述符。
- `values`（OUT）：DEVICE，值数组指针，长度为 `nnz`。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spMatDescr` 为空指针

---

### aclsparseSpMatSetValues

```c
aclsparseStatus_t aclsparseSpMatSetValues(
    aclsparseSpMatDescr_t spMatDescr,
    void *values
);
```

**功能**：设置稀疏矩阵描述符的 values 指针。

**参数说明**：

- `spMatDescr`（IN）：HOST，稀疏矩阵描述符。
- `values`（IN）：DEVICE，值数组指针，长度为 `nnz`。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spMatDescr` 为空指针

---

### aclsparseSpMatGetSize

```c
aclsparseStatus_t aclsparseSpMatGetSize(
    aclsparseConstSpMatDescr_t spMatDescr,
    int64_t *rows,
    int64_t *cols,
    int64_t *nnz
);
```

**功能**：获取稀疏矩阵的行数、列数和非零元素个数。接受 const 描述符，非 const 描述符同样支持。输出参数允许为 `nullptr`，此时仅返回请求的字段。

**参数说明**：

- `spMatDescr`（IN）：HOST，稀疏矩阵描述符。
- `rows`（OUT）：HOST，稀疏矩阵的行数。
- `cols`（OUT）：HOST，稀疏矩阵的列数。
- `nnz`（OUT）：HOST，稀疏矩阵的非零元素个数。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spMatDescr` 为空指针

---

### 只读(const)描述符构造接口

只读(const)构造接口：数据指针为 `const`，构造出的描述符是 `const` 变体，只能传给接收 `const` 形参的接口。`aclsparseCreateConstCsc` 可创建只读 CSC 描述符，作为 SparseToDense 的 CSC 输入。销毁接口（`aclsparseDestroyDnVec` / `aclsparseDestroySpMat` / `aclsparseDestroyDnMat` / `aclsparseDestroySpVec`）统一接收 `const` 变体，因此 const 与非 const 描述符都可销毁。

```c
aclsparseStatus_t aclsparseCreateConstDnVec(aclsparseConstDnVecDescr_t *dnVecDescr, int64_t size,
    const void *values, aclDataType valueType);

aclsparseStatus_t aclsparseCreateConstCsr(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    const void *csrRowOffsets, const void *csrColInd, const void *csrValues, aclsparseIndexType_t csrRowOffsetsType,
    aclsparseIndexType_t csrColIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);

aclsparseStatus_t aclsparseCreateConstCsc(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    const void *cscColOffsets, const void *cscRowInd, const void *cscValues, aclsparseIndexType_t cscColOffsetsType,
    aclsparseIndexType_t cscRowIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);

aclsparseStatus_t aclsparseCreateConstDnMat(aclsparseConstDnMatDescr_t *dnMatDescr,
    int64_t rows, int64_t cols, int64_t ld, const void *values,
    aclDataType valueType, aclsparseOrder_t order);

aclsparseStatus_t aclsparseConstSpMatGetValues(aclsparseConstSpMatDescr_t spMatDescr,
    const void **values);

aclsparseStatus_t aclsparseCreateConstCoo(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    const void *cooRowInd, const void *cooColInd, const void *cooValues, aclsparseIndexType_t cooIdxType,
    aclsparseIndexBase_t idxBase, aclDataType valueType);

aclsparseStatus_t aclsparseConstCooGet(aclsparseConstSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
    int64_t *nnz, const void **cooRowInd, const void **cooColInd, const void **cooValues,
    aclsparseIndexType_t *cooIdxType, aclsparseIndexBase_t *idxBase, aclDataType *valueType);
```

> **支持状态**：`aclsparseCreateConstCsc` 可创建只读 CSC 描述符。当前版本 SpMV / SpMM 仅支持 CSR 稀疏矩阵，CSC 描述符不能用于这两个计算接口。

---

### aclsparseSpMVGetBufferSize

```c
aclsparseStatus_t aclsparseSpMVGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    const void *beta,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    size_t *bufferSize
);
```

**功能**：获取稀疏矩阵与向量相乘（SpMV）操作所需的缓冲区大小。

**参数说明**：

- `handle`（IN）：HOST，稀疏计算句柄。
- `opA`（IN）：HOST，稀疏操作类型。
- `alpha`（IN）：HOST/DEVICE，标量alpha参数。
- `matA`（IN）：HOST，稀疏矩阵描述符。
- `vecX`（IN）：HOST，输入向量x的描述符。
- `beta`（IN）：HOST/DEVICE，标量beta参数。
- `vecY`（IN）：HOST，输出向量y的描述符。
- `computeType`（IN）：HOST，计算的数据类型。
- `alg`（IN）：HOST，SpMV算法类型。
- `bufferSize`（OUT）：HOST，所需的缓冲区大小。

**说明**：`ACL_SPARSE_SPMV_ALG_DEFAULT`（向量版）的所有中间量都在 UB 内，返回 0。
`ACL_SPARSE_SPMV_CSR_ALG_CUBE` 返回分段求和所需的 workspace 大小，调用方须按该大小
申请显存并作为 `aclsparseSpMV` 的 `externalBuffer` 传入。空矩阵（`nnz == 0`）会回退到
向量版实现，同样返回 0。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMVPreprocess

> **支持状态**：暂未支持。当前版本仅提供头文件声明，库中尚无实现；调用会导致链接失败。

```c
aclsparseStatus_t aclsparseSpMVPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    const void *beta,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    void *externalBuffer
);
```

**功能**：对稀疏矩阵进行预处理，以便在后续的 SpMV 计算中复用工作缓冲区、加速计算。

**参数说明**：

- `handle`（IN）：HOST，稀疏计算句柄。
- `opA`（IN）：HOST，稀疏矩阵操作类型。
- `alpha`（IN）：HOST/DEVICE，标量 alpha 参数。
- `matA`（IN）：HOST，稀疏矩阵描述符。
- `vecX`（IN）：HOST，输入向量 x 的描述符。
- `beta`（IN）：HOST/DEVICE，标量 beta 参数。
- `vecY`（IN）：HOST，输出向量 y 的描述符。
- `computeType`（IN）：HOST，计算的数据类型。
- `alg`（IN）：HOST，SpMV 算法类型。
- `externalBuffer`（IN）：DEVICE，工作缓冲区（需先通过 `aclsparseSpMVGetBufferSize` 分配）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMV

```c
aclsparseStatus_t aclsparseSpMV(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    const void *beta,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    void *externalBuffer
);
```

**功能**：稀疏矩阵向量乘法（SpMV）。稀疏矩阵 `matA` 须为 **CSR** 格式（CSC / COO 等格式将返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`）。

**参数说明**：

- `handle`（IN）：HOST，稀疏矩阵处理器句柄。
- `opA`（IN）：HOST，稀疏矩阵操作符。
- `alpha`（IN）：HOST/DEVICE，标量alpha。
- `matA`（IN）：HOST，稀疏矩阵描述符。
- `vecX`（IN）：HOST，密集向量x描述符。
- `beta`（IN）：HOST/DEVICE，标量beta。
- `vecY`（IN/OUT）：HOST，密集向量y描述符。
- `computeType`（IN）：HOST，计算类型。
- `alg`（IN）：HOST，稀疏矩阵向量乘法算法。
- `externalBuffer`（IN）：DEVICE，工作缓冲区指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseCreateDnMat

```c
aclsparseStatus_t aclsparseCreateDnMat(
    aclsparseDnMatDescr_t *dnMatDescr,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    void *values,
    aclDataType valueType,
    aclsparseOrder_t order
);
```

**功能**：创建稠密矩阵描述符，用于描述 SpMM 中的 B / C 稠密矩阵。

**参数说明**：

- `dnMatDescr`（IN/OUT）：HOST，输出的稠密矩阵描述符。
- `rows`（IN）：HOST，矩阵的行数，允许为 0，不允许为负数。
- `cols`（IN）：HOST，矩阵的列数，允许为 0，不允许为负数。
- `ld`（IN）：HOST，leading dimension。行主序时需 `>= cols`；列主序时需 `>= rows`。必须大于 0，该约束同样适用于零维矩阵。
- `values`（IN）：DEVICE，矩阵数据指针，必须非空，包括零维矩阵。
- `valueType`（IN）：HOST，元素数据类型。作为 B 矩阵时支持 `ACL_FLOAT` / `ACL_FLOAT16` / `ACL_INT8`；作为 C 矩阵时支持 `ACL_FLOAT` / `ACL_FLOAT16` / `ACL_INT32`（`ACL_INT32` 仅在 int8 计算路径中作为 C 矩阵类型使用）。
- `order`（IN）：HOST，布局：`ACL_SPARSE_ORDER_ROW`（行主序）/ `ACL_SPARSE_ORDER_COL`（列主序）。

**说明**：rows/cols 为 0 的零维稠密矩阵描述符可正常创建；但计算类 API 是否接受零维输入由各算子决定——aclsparseSpMM / aclsparseSDDMM / aclsparseSpSM 对零维输入返回 `ACL_SPARSE_STATUS_INVALID_VALUE`，aclsparseSpMMOp 系列支持零维（详见对应算子文档）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseDestroyDnMat

```c
aclsparseStatus_t aclsparseDestroyDnMat(aclsparseConstDnMatDescr_t dnMatDescr);
```

**功能**：销毁稠密矩阵描述符。

**参数说明**：

- `dnMatDescr`（IN）：HOST，要销毁的稠密矩阵描述符。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseDnMatGet

```c
aclsparseStatus_t aclsparseDnMatGet(
    aclsparseDnMatDescr_t dnMatDescr,
    int64_t *rows,
    int64_t *cols,
    int64_t *ld,
    void **values,
    aclDataType *valueType,
    aclsparseOrder_t *order
);
```

**功能**：获取稠密矩阵描述符的全部字段。

**参数说明**：

- `dnMatDescr`（IN）：HOST，稠密矩阵描述符。
- `rows`（OUT）：HOST，矩阵的行数。
- `cols`（OUT）：HOST，矩阵的列数。
- `ld`（OUT）：HOST，leading dimension。
- `values`（OUT）：DEVICE，矩阵数据指针。
- `valueType`（OUT）：HOST，元素数据类型。
- `order`（OUT）：HOST，内存布局。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`dnMatDescr` 为空指针

---

### aclsparseDnMatGetValues

```c
aclsparseStatus_t aclsparseDnMatGetValues(
    aclsparseDnMatDescr_t dnMatDescr,
    void **values
);
```

**功能**：获取稠密矩阵描述符的 values 指针。

**参数说明**：

- `dnMatDescr`（IN）：HOST，稠密矩阵描述符。
- `values`（OUT）：DEVICE，矩阵数据指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`dnMatDescr` 为空指针

---

### aclsparseDnMatSetValues

```c
aclsparseStatus_t aclsparseDnMatSetValues(
    aclsparseDnMatDescr_t dnMatDescr,
    void *values
);
```

**功能**：设置稠密矩阵描述符的 values 指针。

**参数说明**：

- `dnMatDescr`（IN）：HOST，稠密矩阵描述符。
- `values`（IN）：DEVICE，矩阵数据指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`dnMatDescr` 为空指针

---

### aclsparseSpMMGetBufferSize

```c
aclsparseStatus_t aclsparseSpMMGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    size_t *size
);
```

**功能**：获取稀疏矩阵-稠密矩阵乘法（SpMM）操作所需的 workspace 缓冲区大小。

**参数说明**：

- `handle`（IN）：HOST，稀疏计算句柄。
- `opA`（IN）：HOST，稀疏矩阵 A 的操作类型，A2/A3 支持非转置、转置和共轭转置。
- `opB`（IN）：HOST，稠密矩阵 B 的操作类型，A2/A3 支持非转置、转置和共轭转置。
- `alpha`（IN）：HOST/DEVICE，标量 alpha 参数。
- `matA`（IN）：HOST，CSR 稀疏矩阵 A 的描述符。
- `matB`（IN）：HOST，输入稠密矩阵 B 的描述符。
- `beta`（IN）：HOST/DEVICE，标量 beta 参数。
- `matC`（IN）：HOST，输入/输出稠密矩阵 C 的描述符。
- `computeType`（IN）：HOST，计算的数据类型。
- `alg`（IN）：HOST，SpMM 算法类型。
- `size`（OUT）：HOST，所需的 workspace 缓冲区大小（字节）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMMPreprocess

```c
aclsparseStatus_t aclsparseSpMMPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    void *buffer
);
```

**功能**：对 CSR 稀疏矩阵进行预处理（行重排与分桶），以便在后续的 SpMM 计算中使用。

**参数说明**：

- `handle`（IN）：HOST，稀疏计算句柄。
- `opA`（IN）：HOST，稀疏矩阵 A 的操作类型。
- `opB`（IN）：HOST，稠密矩阵 B 的操作类型。
- `alpha`（IN）：HOST/DEVICE，标量 alpha 参数。
- `matA`（IN）：HOST，CSR 稀疏矩阵 A 的描述符。
- `matB`（IN）：HOST，输入稠密矩阵 B 的描述符。
- `beta`（IN）：HOST/DEVICE，标量 beta 参数。
- `matC`（IN）：HOST，输入/输出稠密矩阵 C 的描述符。
- `computeType`（IN）：HOST，计算的数据类型。
- `alg`（IN）：HOST，SpMM 算法类型。
- `buffer`（IN）：DEVICE，用于存储预处理结果的 workspace 缓冲区。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMM

```c
aclsparseStatus_t aclsparseSpMM(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    void *buffer
);
```

**功能**：稀疏矩阵-稠密矩阵乘法（SpMM），计算 `C = alpha * op(A) * op(B) + beta * C`。稀疏矩阵 A 须为 **CSR** 格式（CSC / COO 等格式传入 `matA` 将返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`）。零维输入：arch35 上 matA.rows / matA.cols / matC.cols（即 m/k/n）任一为 0 时返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；**A2/A3（arch22）不同**——M/K/N 均允许为 0，零长度输出直接返回 `ACL_SPARSE_STATUS_SUCCESS`（K=0 时结果遵循 `C = beta * C`），三个入口（GetBufferSize / Preprocess / SpMM）行为一致，详见 `sparse/spmm/README.md`。

**参数说明**：

- `handle`（IN）：HOST，稀疏矩阵处理器句柄。
- `opA`（IN）：HOST，稀疏矩阵 A 的操作类型，A2/A3 支持非转置、转置和共轭转置。
- `opB`（IN）：HOST，稠密矩阵 B 的操作类型。
- `alpha`（IN）：HOST/DEVICE，标量 alpha。
- `matA`（IN）：HOST，CSR 稀疏矩阵 A 的描述符。
- `matB`（IN）：HOST，输入稠密矩阵 B 的描述符。
- `beta`（IN）：HOST/DEVICE，标量 beta。
- `matC`（IN/OUT）：HOST，输入/输出稠密矩阵 C 的描述符。
- `computeType`（IN）：HOST，计算类型。
- `alg`（IN）：HOST，SpMM 算法类型。
- `buffer`（IN）：DEVICE，工作缓冲区指针（需先调用 `aclsparseSpMMGetBufferSize` 分配，建议先调用 `aclsparseSpMMPreprocess` 预处理）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMVOp_bufferSize

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

**功能**：获取 SpMVOp 所需 workspace 大小（字节数）。ALG1 / DEFAULT 返回 0；ALG2 返回重排表、临时重排表、scratch、临时 scratch 和 bin_edge 表的总大小（包含 64 字节对齐开销）。

SpMVOp 系列接口遵循 descriptor/plan 生命周期模式，调用顺序为：`aclsparseSpMVOp_bufferSize` → `aclsparseSpMVOp_createDescr` → `aclsparseSpMVOp_createPlan` → [可选 `aclsparseSpMVOp_setGlobalUserData`] → `aclsparseSpMVOp` → `aclsparseSpMVOp_destroyPlan` → `aclsparseSpMVOp_destroyDescr`。

**参数说明**：

- `handle`（IN）：HOST，aclsparse 句柄。
- `opA`（IN）：HOST，矩阵 A 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`。
- `matA`（IN）：HOST，CSR 稀疏矩阵描述符。
- `vecX`（IN）：HOST，输入稠密向量描述符。
- `vecY`（IN）：HOST，额外加法输入稠密向量描述符。
- `vecZ`（IN/OUT）：HOST，输出稠密向量描述符。
- `computeType`（IN）：HOST，计算类型，仅支持 `ACL_FLOAT`。
- `alg`（IN）：HOST，SpMVOp 算法类型。
- `bufferSize`（OUT）：HOST，返回所需 workspace 字节数。

**约束说明**：`matA` 必须为 CSR 格式，值类型必须为 FP32，列索引类型必须为 int32，行偏移类型可为 int32 或 int64。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMVOp_createDescr

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

**功能**：创建 SpMVOp 内部描述符并执行预处理。ALG2 的预处理在 handle 关联的 stream 上异步执行排序和 bin_edge 负载均衡切分。

**参数说明**：

- `handle`（IN）：HOST，aclsparse 句柄。
- `descr`（OUT）：HOST，返回 SpMVOp 描述符。
- `opA`（IN）：HOST，矩阵 A 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`。
- `matA`（IN）：HOST，CSR 稀疏矩阵描述符。
- `vecX`（IN）：HOST，输入稠密向量描述符。
- `vecY`（IN）：HOST，额外加法输入稠密向量描述符。
- `vecZ`（IN/OUT）：HOST，输出稠密向量描述符。
- `computeType`（IN）：HOST，计算类型，仅支持 `ACL_FLOAT`。
- `alg`（IN）：HOST，SpMVOp 算法类型。
- `buffer`（IN）：DEVICE，workspace 缓冲区，由调用者先经 `aclsparseSpMVOp_bufferSize` 查询大小并分配。

**约束说明**：ALG2 时 `buffer` 不可为 NULL，且其大小不得小于 `aclsparseSpMVOp_bufferSize` 返回值；`buffer` 必须在 descr 销毁前保持有效。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMVOp_destroyDescr

```c
aclsparseStatus_t aclsparseSpMVOp_destroyDescr(aclsparseSpMVOpDescr_t descr);
```

**功能**：销毁 SpMVOp 描述符并释放其管理的 host 资源。`descr` 为 nullptr 时直接返回 SUCCESS。必须先销毁执行计划，再销毁描述符。

**参数说明**：`descr`（IN）：HOST，要销毁的 SpMVOp 描述符。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMVOp_createPlan

```c
aclsparseStatus_t aclsparseSpMVOp_createPlan(
    aclsparseHandle_t handle,
    aclsparseSpMVOpDescr_t descr,
    aclsparseSpMVOpPlan_t *plan,
    const void *epilogueLTOBuffer,
    size_t epilogueLTOBufferSize);
```

**功能**：基于 descr 创建执行计划。NPU 侧使用 identity epilogue，`epilogueLTOBuffer` 必须为 NULL，`epilogueLTOBufferSize` 必须为 0，否则返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。

**参数说明**：

- `handle`（IN）：HOST，aclsparse 句柄。
- `descr`（IN）：HOST，来自 `aclsparseSpMVOp_createDescr` 的描述符。
- `plan`（OUT）：HOST，返回执行计划。
- `epilogueLTOBuffer`（IN）：HOST，epilogue LTO-IR 缓冲区，NPU 侧必须为 NULL。
- `epilogueLTOBufferSize`（IN）：HOST，epilogue LTO-IR 大小，NPU 侧必须为 0。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMVOp_destroyPlan

```c
aclsparseStatus_t aclsparseSpMVOp_destroyPlan(aclsparseSpMVOpPlan_t plan);
```

**功能**：销毁 SpMVOp 执行计划。`plan` 为 nullptr 时直接返回 SUCCESS；必须先于 `aclsparseSpMVOp_destroyDescr` 调用。

**参数说明**：`plan`（IN）：HOST，要销毁的 SpMVOp 执行计划。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMVOp_setGlobalUserData

```c
aclsparseStatus_t aclsparseSpMVOp_setGlobalUserData(
    aclsparseHandle_t handle,
    aclsparseSpMVOpPlan_t plan,
    const char *epilogueDataName,
    void *epilogueData,
    size_t epilogueDataSize);
```

**功能**：设置 epilogue 模块中的全局用户数据。NPU 侧使用 identity epilogue，本接口为 no-op，直接返回 SUCCESS。

**参数说明**：

- `handle`（IN）：HOST，aclsparse 句柄。
- `plan`（IN）：HOST，SpMVOp 执行计划。
- `epilogueDataName`（IN）：HOST，epilogue 数据变量名。
- `epilogueData`（IN）：HOST，epilogue 数据指针。
- `epilogueDataSize`（IN）：HOST，数据字节数。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMVOp

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

**功能**：执行稀疏矩阵向量乘法 `Z = alpha * A * X + beta * Y`。调用异步入队到 handle 关联的 stream；ALG1 和 ALG2 均提供 bit-wise 可重复结果。

**参数说明**：

- `handle`（IN）：HOST，aclsparse 句柄。
- `plan`（IN）：HOST，SpMVOp 执行计划。
- `alpha`（IN）：HOST/DEVICE，FP32 标量，指针模式由 `aclsparseSetPointerMode` 决定。
- `beta`（IN）：HOST/DEVICE，FP32 标量，指针模式由 `aclsparseSetPointerMode` 决定。
- `vecX`（IN）：HOST，输入稠密向量描述符。
- `vecY`（IN）：HOST，额外加法输入稠密向量描述符。
- `vecZ`（IN/OUT）：HOST，输出稠密向量描述符，可与 vecY 共享 device values 实现 in-place。

**约束说明**：仅支持 CSR 格式；`colIndType` 必须为 int32，`rowOffsetsType` 可为 int32 或 int64；FP32 是唯一支持的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMMOp_bufferSize

```c
aclsparseStatus_t aclsparseSpMMOp_bufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMOpAlg_t alg,
    size_t *bufferSize);
```

**功能**：获取 SpMMOp 所需 workspace 大小（字节数）。ALG1 / DEFAULT 返回 0（无需 workspace）；ALG2 返回 reorder 表 + bin_edge 表（含对齐开销）的总大小。

SpMMOp 系列 7 个接口遵循 descriptor/plan 生命周期模式，调用顺序必须为：`aclsparseSpMMOp_bufferSize` → `aclsparseSpMMOp_createDescr` → `aclsparseSpMMOp_createPlan` → [可选 `aclsparseSpMMOp_setGlobalUserData`] → `aclsparseSpMMOp`（可重复执行）→ `aclsparseSpMMOp_destroyPlan` → `aclsparseSpMMOp_destroyDescr`。

**参数说明**：

- `handle`（IN）：HOST，aclsparse 句柄。
- `opA`（IN）：HOST，稀疏矩阵 A 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`。
- `opB`（IN）：HOST，稠密矩阵 B 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` / `ACL_SPARSE_OP_TRANSPOSE`。
- `matA`（IN）：HOST，CSR 稀疏矩阵 A 的描述符。
- `matB`（IN）：HOST，输入稠密矩阵 B 的描述符（不可为 NULL）。
- `matC`（IN）：HOST，稠密矩阵 C 的描述符（不可为 NULL；签名为非 const 的 `aclsparseDnMatDescr_t`，与执行阶段的 in-place 语义一致）。
- `computeType`（IN）：HOST，计算类型，仅支持 `ACL_FLOAT`。
- `alg`（IN）：HOST，SpMMOp 算法类型。
- `bufferSize`（OUT）：HOST，返回所需 workspace 字节数。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMMOp_createDescr

```c
aclsparseStatus_t aclsparseSpMMOp_createDescr(
    aclsparseHandle_t handle,
    aclsparseSpMMOpDescr_t *descr,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMOpAlg_t alg,
    void *buffer);
```

**功能**：创建 SpMMOp 内部描述符并执行预处理，绑定 matA / opB / alg。matA 的 CSR 指针以弱引用存入 descr（CSR pattern 固定用于 plan 生命周期）；opB 在此阶段绑定，plan 生命周期内固定。ALG2 在此阶段**同步**执行 host 侧预处理（D2H 读取 rowOffsets → `std::stable_sort` 排序 → 计算 bin_edge 负载均衡切分 → H2D 写入 workspace）；DEFAULT 归一化为 ALG1。

**参数说明**：

- `handle`（IN）：HOST，aclsparse 句柄。
- `descr`（OUT）：HOST，输出的 SpMMOp 描述符。
- `opA`（IN）：HOST，稀疏矩阵 A 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`。
- `opB`（IN）：HOST，稠密矩阵 B 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` / `ACL_SPARSE_OP_TRANSPOSE`。
- `matA`（IN）：HOST，CSR 稀疏矩阵 A 的描述符。
- `matB`（IN）：HOST，输入稠密矩阵 B 的描述符（可为 NULL）。
- `matC`（IN）：HOST，稠密矩阵 C 的描述符（可为 NULL）。
- `computeType`（IN）：HOST，计算类型，仅支持 `ACL_FLOAT`。
- `alg`（IN）：HOST，SpMMOp 算法类型。
- `buffer`（IN）：DEVICE，workspace 缓冲区，由调用者先经 `aclsparseSpMMOp_bufferSize` 查询大小并分配（ALG2 必需）。

**约束说明**：

- ALG2 时 `buffer` 不可为 NULL 且大小不小于 `aclsparseSpMMOp_bufferSize` 返回值；`buffer` 须在 descr 销毁前保持有效。
- ALG1 允许在不重建 descr/plan 的情况下原地更新 csrValues 指向的数据；ALG2 下原地修改 csrValues 为不受支持的未定义行为（调用方责任，无运行时检测）；修改 CSR 结构（rowOffsets / colInd）会使 reorder / bin_edge 失效。
- CSR 内容契约（colInd 值域 ∈ [indexBase, indexBase+k)、rowOffsets 单调不减、rowOffsets[0] 与 rowOffsets[m] 与 nnz 一致）由调用方保证，库不做运行时校验。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMMOp_destroyDescr

```c
aclsparseStatus_t aclsparseSpMMOp_destroyDescr(aclsparseSpMMOpDescr_t descr);
```

**功能**：销毁 SpMMOp 描述符并释放其管理的 host 资源。幂等语义：`descr` 为 nullptr 时直接返回 SUCCESS。必须先调用 `aclsparseSpMMOp_destroyPlan` 销毁执行计划，再调用本接口销毁描述符。

**参数说明**：

- `descr`（IN）：HOST，要销毁的 SpMMOp 描述符。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMMOp_createPlan

```c
aclsparseStatus_t aclsparseSpMMOp_createPlan(
    aclsparseHandle_t handle,
    aclsparseSpMMOpDescr_t descr,
    aclsparseSpMMOpPlan_t *plan,
    const void *epilogueLTOBuffer,
    size_t epilogueLTOBufferSize);
```

**功能**：基于 descr 创建执行计划。NPU 侧 epilogue 固定为 identity（不附加任何 elementwise 操作），`epilogueLTOBuffer` 必须为 NULL、`epilogueLTOBufferSize` 必须为 0，否则返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。

**参数说明**：

- `handle`（IN）：HOST，aclsparse 句柄。
- `descr`（IN）：HOST，来自 `aclsparseSpMMOp_createDescr` 的描述符（非 const，对标 cuSPARSE 语义，createPlan 可修改 descr）。
- `plan`（OUT）：HOST，输出的执行计划。
- `epilogueLTOBuffer`（IN）：HOST，epilogue LTO-IR 缓冲区，NPU 侧必须为 NULL。
- `epilogueLTOBufferSize`（IN）：HOST，epilogue LTO-IR 大小（字节），NPU 侧必须为 0。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMMOp_destroyPlan

```c
aclsparseStatus_t aclsparseSpMMOp_destroyPlan(aclsparseSpMMOpPlan_t plan);
```

**功能**：销毁 SpMMOp 执行计划。幂等语义：`plan` 为 nullptr 时直接返回 SUCCESS。必须先于 `aclsparseSpMMOp_destroyDescr` 调用，否则 plan 内部弱引用的 descr 成为悬垂指针。

**参数说明**：

- `plan`（IN）：HOST，要销毁的执行计划。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMMOp_setGlobalUserData

```c
aclsparseStatus_t aclsparseSpMMOp_setGlobalUserData(
    aclsparseHandle_t handle,
    aclsparseSpMMOpPlan_t plan,
    const char *epilogueDataName,
    void *epilogueData,
    size_t epilogueDataSize);
```

**功能**：将 epilogue 模块中 `__constant__` 变量设置为指定数据（对标 cuSPARSE）。NPU 侧使用 identity epilogue，无需辅助数据，本接口为 no-op：仅校验参数一致性（`epilogueData` 与 `epilogueDataSize` 须同时有效或同时无效）后直接返回 SUCCESS。

**参数说明**：

- `handle`（IN）：HOST，aclsparse 句柄。
- `plan`（IN）：HOST，执行计划。
- `epilogueDataName`（IN）：HOST，epilogue 数据变量名。
- `epilogueData`（IN）：HOST，epilogue 数据指针。
- `epilogueDataSize`（IN）：HOST，数据字节数。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMMOp

```c
aclsparseStatus_t aclsparseSpMMOp(
    aclsparseHandle_t handle,
    aclsparseSpMMOpPlan_t plan,
    const void *alpha,
    const void *beta,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC);
```

**功能**：执行稀疏矩阵-稠密矩阵乘法：`C = alpha * op(A) * op(B) + beta * C`，其中 A 为 CSR(m×k) 稀疏矩阵，B 为稠密矩阵，C 为稠密矩阵（in-place：读入 `beta * C` 并写出结果）。**异步执行**：kernel 入队到 handle 关联的 stream 后函数立即返回，同步由调用方通过 stream 负责。`matB` / `matC` 在 execute 阶段传入，可跨多次 execute 更换（n / ldb / ldc / order 可变），但 m / k / opB 必须与 descr 绑定值一致。

**参数说明**：

- `handle`（IN）：HOST，aclsparse 句柄。
- `plan`（IN）：HOST，执行计划。
- `alpha`（IN）：HOST/DEVICE，标量 alpha（FP32；指针模式由 `aclsparseSetPointerMode` 决定）。
- `beta`（IN）：HOST/DEVICE，标量 beta（FP32；指针模式由 `aclsparseSetPointerMode` 决定）。
- `matB`（IN）：HOST，输入稠密矩阵 B 的描述符。
- `matC`（IN/OUT）：HOST，输入/输出稠密矩阵 C 的描述符（in-place）。

**约束说明**：

- `opA` 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`；`opB` 支持 `ACL_SPARSE_OP_NON_TRANSPOSE` / `ACL_SPARSE_OP_TRANSPOSE`。
- dtype 支持 FP32 / FP16，matA / matB / matC 的 valueType 必须一致；累加固定 FP32（FP16 输出时饱和截断到 ±65504，避免 Inf/NaN）；computeType 固定 `ACL_FLOAT`。
- matB / matC 独立支持 `ACL_SPARSE_ORDER_ROW` / `ACL_SPARSE_ORDER_COL` 主序。
- ALG1 / ALG2 均保证 bit-wise 可重复结果。
- 维度与 stride 上限：m / k / n 及 matB.ld / matC.ld 超过 INT32_MAX 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`；stride 上限按 dtype 约束——FP32 为 UINT32_MAX / 4、FP16 为 UINT32_MAX / 2（ld 以元素计），超限返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSgtsv2

```cpp
aclsparseStatus_t aclsparseSgtsv2(
    aclsparseHandle_t handle,
    int m, int n,
    const float *dl, const float *d, const float *du,
    float *B, int ldb, void *pBuffer);
```

**功能**：求解三对角线性方程组 **A · X = B**（FP32，支持多个右端项 Multiple RHS），其中 **A** 为 m × m 三对角矩阵，由下对角线 `dl`、主对角线 `d`、上对角线 `du` 表示；**B** 为 m × n 的稠密右端矩阵（column-major），解 **X** 原地覆盖 **B**。

采用**带部分选主元（partial pivoting）的 Thomas 算法**。前向消元逐行比较 `|d'[i-1]|` 与 `|dl[i]|` 选主元，必要时交换行并产生 du'/du2' 两类 fill-in；后向回代使用含 fill-in 的修改后系数求解。arch22 各 RHS 复用公共矩阵因子；扩展指数路径按初始解的后向误差检查分解方式，必要时切换行均衡或原始尺度并重新分解。对强对角占优且归一化 RHS 范围受限的有限系统，使用 40 次固定向量 Jacobi 迭代快路，其他输入保留直接求解，非强对角占优的有限系统增加两次补偿残差修正。精度标准及公开 FP32 ABI 不变，详细数值门禁与 UB 布局见算子 README。其 workspace 为 `align_up(align_up(17*m,128)+85*m,128)` 字节且与 n 无关，n=0 时为 0。85 字节/行的 scratch 包含五组扩展数值数组（80 字节/行）、原始 RHS（4 字节/行）和独立 pivot（1 字节/行）。例如 `m=3,n=1` 时返回 384 B；因子区的 128 字节对齐边界两侧，`m=7` 和 `m=8` 分别返回 768 B 和 1024 B。**调用方必须以当前 `bufferSizeExt` 查询值分配 workspace，不得按历史公式硬编码容量。**arch35 保持每 RHS 3 组 FP32 workspace 的既有实现。

**产品支持情况**：

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22 / DAV_2201）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22 / DAV_2201）

**参数说明**：

- `handle`（IN）：HOST，ops-sparse 库上下文句柄，携带 stream。
- `m`（IN）：HOST，线性系统大小（行数 = 列数），需 ≥ 3（n > 0 时校验；n=0 时跳过校验）。
- `n`（IN）：HOST，右端项列数，需 ≥ 0；n=0 时提前返回 SUCCESS 不计算。
- `dl`（IN）：DEVICE，下对角线数组 [m]，`dl[0]` 必须在调用前置 0。
- `d`（IN）：DEVICE，主对角线数组 [m]。
- `du`（IN）：DEVICE，上对角线数组 [m]，`du[m-1]` 必须在调用前置 0。
- `B`（IN/OUT）：DEVICE，右端项 b（输入）/ 解 X（输出），大小为 ldb × n（column-major）。计算完成后原地覆盖为解矩阵 X。
- `ldb`（IN）：HOST，B 的 leading dimension，需 ≥ max(1, m)。
- `pBuffer`（IN）：DEVICE，工作区 buffer，地址必须 128 字节对齐。当 n > 0 时必须为有效指针，大小不少于 `aclsparseSgtsv2_bufferSizeExt` 返回的字节数。

**约束说明**：

- m ≥ 3（n > 0 时校验；n=0 时跳过校验）
- n ≥ 0（n = 0 时提前返回 SUCCESS 不计算）
- ldb ≥ max(1, m)
- 当 n > 0 时，dl、d、du、B 均不可为 nullptr
- 当 n > 0 时，pBuffer 不可为 nullptr，且地址必须是 128 字节的整数倍
- pBuffer 大小必须 ≥ `aclsparseSgtsv2_bufferSizeExt` 返回值
- `dl[0]` 必须在调用前置 0
- `du[m-1]` 必须在调用前置 0
- 带部分选主元的 Thomas 算法对一般非奇异地对角系统数值稳定。算子不对奇异矩阵做任何保护：若前向消元或后向回代过程中任一主元为零或接近零，IEEE-754 除零将在输出中产生 Inf/NaN。**调用方必须保证输入的三对角矩阵良态，否则该 RHS 列的精度不可预期**

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：参数非法（m/n/ldb 越界、空指针、pBuffer 未对齐等）
- 其他值：失败

---

### aclsparseSgtsv2_bufferSizeExt

```cpp
aclsparseStatus_t aclsparseSgtsv2_bufferSizeExt(
    aclsparseHandle_t handle,
    int m, int n,
    const float *dl, const float *d, const float *du,
    const float *B, int ldb,
    size_t *pBufferSizeInBytes);
```

**功能**：查询 `aclsparseSgtsv2` 所需工作区大小（字节）。返回值已包含 128 字节对齐。

**产品支持情况**：

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22 / DAV_2201）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22 / DAV_2201）

**参数说明**：

- `handle`（IN）：HOST，ops-sparse 库上下文句柄。
- `m`（IN）：HOST，线性系统大小（行数 = 列数），需 ≥ 3（n > 0 时校验；n=0 时跳过校验）。
- `n`（IN）：HOST，右端项列数，需 ≥ 0；n=0 时返回 0。
- `dl`（IN）：DEVICE，下对角线数组 [m]，`dl[0]` 必须为 0。
- `d`（IN）：DEVICE，主对角线数组 [m]。
- `du`（IN）：DEVICE，上对角线数组 [m]，`du[m-1]` 必须为 0。
- `B`（IN）：DEVICE，右端项矩阵 [ldb × n]（column-major）；查询接口不使用其值，可为 nullptr。
- `ldb`（IN）：HOST，B 的 leading dimension，需 ≥ max(1, m)。
- `pBufferSizeInBytes`（OUT）：HOST，输出 `aclsparseSgtsv2` 所需工作区大小（字节），不可为 nullptr。

**约束说明**：

- m ≥ 3（n > 0 时校验；n=0 时跳过校验）
- n ≥ 0（n = 0 时返回 0，无需工作区）
- ldb ≥ max(1, m)
- dl、d、du、pBufferSizeInBytes 均不可为 nullptr
- B 可为 nullptr（查询接口不使用 B 的值）
- 返回的工作区大小已包含 128 字节对齐

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：参数非法（m/n/ldb 越界、空指针等）
- 其他值：失败

---

### aclsparseXcscsort_bufferSizeExt

```c
aclsparseStatus_t aclsparseXcscsort_bufferSizeExt(
    aclsparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const int *cscColPtr,
    const int *cscRowInd,
    size_t *pBufferSizeInBytes
);
```

**功能**：查询CSC格式按列稳定排序（`aclsparseXcscsort`）所需的workspace缓冲区大小。仅做参数校验与字节数计算，不启动kernel、不引入Device同步。

**参数说明**：

- `handle`（IN）：HOST，aclsparse句柄。
- `m`（IN）：HOST，矩阵行数。
- `n`（IN）：HOST，矩阵列数。
- `nnz`（IN）：HOST，非零元素个数。
- `cscColPtr`（IN）：DEVICE，CSC列偏移数组，长度为`n + 1`。
- `cscRowInd`（IN）：DEVICE，CSC行索引数组，长度为`nnz`；`nnz == 0`时可为`nullptr`。
- `pBufferSizeInBytes`（OUT）：HOST，返回所需的workspace字节数。`nnz == 0`时置为0；否则为`2 * nnz * sizeof(int32_t)`。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseXcscsort

```c
aclsparseStatus_t aclsparseXcscsort(
    aclsparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const aclsparseMatDescr_t descrA,
    const int *cscColPtr,
    int *cscRowInd,
    int *P,
    void *pBuffer
);
```

**功能**：对CSC格式稀疏矩阵每列的行索引执行原地稳定升序排序（单键排序），并使用相同排列重排`P`。`cscColPtr`不修改；`P`既是输入（调用方须预设为`0, 1, ..., nnz - 1`）又是输出（记录最终排列，满足`sortedVal[i] = origVal(P[i])`）。`pBuffer`须128字节对齐。接口异步执行，读取结果前须同步handle绑定的stream。

**参数说明**：

- `handle`（IN）：HOST，aclsparse句柄。
- `m`（IN）：HOST，矩阵行数。
- `n`（IN）：HOST，矩阵列数。
- `nnz`（IN）：HOST，非零元素个数。
- `descrA`（IN）：HOST，Legacy矩阵描述符，提供`indexBase`；其他属性不参与排序，矩阵类型隐式视为`GENERAL`。
- `cscColPtr`（IN）：DEVICE，CSC列偏移数组，长度为`n + 1`，排序期间不修改。
- `cscRowInd`（IN/OUT）：DEVICE，行索引数组，长度为`nnz`；输出时每列内部稳定升序。
- `P`（IN/OUT）：DEVICE，排列数组，长度为`nnz`；调用方须先初始化为`0, 1, ..., nnz - 1`，排序后`sortedVal[i] = origVal(P[i])`。
- `pBuffer`（IN）：DEVICE，由`aclsparseXcscsort_bufferSizeExt`查询大小并由调用方分配的workspace，地址须128字节对齐。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseLtMatmulAlgSetAttribute

```cpp
aclsparseStatus_t aclsparseLtMatmulAlgSetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attr,
    const void* attrValue,
    size_t attrValueSize);
```

**功能**：设置结构化稀疏矩阵乘法的算法选择属性。在 `aclsparseLtMatmulPlanInit` 之前调用，覆盖 `aclsparseLtMatmulAlgSelectionInit` 的默认值（algConfigId=0、splitK=1、searchIterations=5），影响后续 PlanInit 计算的 cube tiling 策略与 workspace 布局。纯 Host 端元数据操作，不启动任何 kernel、不消耗 NPU 计算资源。配套接口 `aclsparseLtMatmulAlgGetAttribute` 用于查询属性当前值。对标 cuSPARSELt 的 `cusparseLtMatmulAlgSetAttribute`。

**产品支持情况**：

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |

> 依赖 CANN asc-devkit >= 9.1.0（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄的 const 指针。
- `algSelection`（IN/OUT）：HOST，算法选择描述符，由 `aclsparseLtMatmulAlgSelectionInit` 创建。AlgSetAttribute 写入 algSelection 结构体的属性字段。
- `attr`（IN）：HOST，要设置的属性枚举（`aclsparseLtMatmulAlgAttribute_t`）。
- `attrValue`（IN）：HOST，属性值指针，指向 `int32_t` 类型值。
- `attrValueSize`（IN）：HOST，`attrValue` 缓冲区大小（字节），须为 `sizeof(int32_t)`。

**可设置的属性**：

| 属性枚举 | 值类型 | 取值范围 | 说明 |
|----------|--------|---------|------|
| `ACLSPARSELT_MATMUL_ALG_CONFIG_ID` | int32_t | 0 / 1 | 算法配置 ID，影响 cube tiling（baseM/baseN/baseK）。需在 PlanInit 前设置 |
| `ACLSPARSELT_MATMUL_SPLIT_K` | int32_t | [1, K] | splitK 切分因子。需在 PlanInit 前设置 |
| `ACLSPARSELT_MATMUL_SEARCH_ITERATIONS` | int32_t | > 0 | 搜索迭代次数（默认 5），当前仅存储不参与执行 |
| `ACLSPARSELT_MATMUL_SPLIT_K_MODE` | int32_t | 0 / 1 | splitK 模式（ONE_KERNEL / TWO_KERNELS），参与 effectiveSplitK 计算：ONE_KERNEL 时 effectiveSplitK=1（融合路径），TWO_KERNELS 时 effectiveSplitK=splitK（两段式路径） |
| `ACLSPARSELT_MATMUL_SPLIT_K_BUFFERS` | int32_t | [0, splitK-1] | splitK 缓冲区数，当前仅存储不参与执行 |
| `ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID` | — | 只读 | 不可设置，返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` |

**约束说明**：

- handle 不可为 nullptr 且 *handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`。
- algSelection 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- attrValue 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- attrValueSize 须与属性值类型大小匹配（int32_t 为 4 字节），否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `ACLSPARSELT_MATMUL_ALG_CONFIG_ID` 取值仅 0 或 1，其他值返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `ACLSPARSELT_MATMUL_SPLIT_K` 须 ≥ 1；若 algSelection 已绑定 matmulDesc 且 K > 0，splitK 须 ≤ K，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `ACLSPARSELT_MATMUL_SEARCH_ITERATIONS` 须 > 0，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `ACLSPARSELT_MATMUL_SPLIT_K_BUFFERS` 须在 [0, splitK-1] 范围内，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID` 为只读属性，设置时返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **属性设置时机**：须在 `aclsparseLtMatmulPlanInit` 之前调用，PlanInit 读取 algConfigId 与 splitK 计算 tiling 并冻结到计划中。PlanInit 之后修改属性不会生效，需重新初始化计划。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为 nullptr 或 *handle 为 nullptr
- `ACL_SPARSE_STATUS_INVALID_VALUE`：algSelection/attrValue 为 nullptr、attrValueSize 不匹配、algConfigId/splitK/searchIterations/splitKBuffers 取值非法
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：设置只读属性（ALG_CONFIG_MAX_ID）、attr 枚举值不被支持
- 其他值：失败

---

### aclsparseLtSpMMAPrune

```cpp
aclsparseStatus_t aclsparseLtSpMMAPrune(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    void* d_out,
    aclsparseLtPruneAlg_t pruneAlg,
    aclrtStream stream);
```

**功能**：对稠密矩阵 A 执行 2:4 结构化稀疏剪枝，将每 4 个元素（FP16）或每 2 个元素（FP32）中绝对值较小的元素置零，保留绝对值最大的元素，输出与 A 同型的稠密存储矩阵 A_pruned（被置零元素以 0 表示）。剪枝后的 A_pruned 作为 `aclsparseLtMatmul` 的稀疏侧输入。直接接收 Matmul 描述符，从中读取矩阵 A 的维度（m、k）、数据类型、order 与 opA 派生剪枝参数。算子异步启动，内部不执行 stream 同步，调用方如需读取结果须自行同步。对标 cuSPARSELt 的 `cusparseLtSpMMAPrune`。

**剪枝规则**：

| 数据类型 | 分组大小 | 每组保留元素数 | 稀疏度 |
|----------|---------|---------------|--------|
| FP16（ACL_FLOAT16） | 4 | 2（保留绝对值最大的 2 个） | 50%（2:4） |
| BF16（ACL_BF16） | 4 | 2（保留绝对值最大的 2 个） | 50%（2:4） |
| INT8（ACL_INT8） | 4 | 2（保留绝对值最大的 2 个） | 50%（2:4） |
| FP32（ACL_FLOAT） | 2 | 1（保留绝对值最大的 1 个） | 50%（1:2，等价 2:4 密度） |

> 说明：FP32 统一采用 1:2 模式（分组为 2、保留 1）以简化实现，与 NVIDIA cuSPARSELt 规范的 1:2 一致；FP16/BF16/INT8 采用标准 2:4 模式（分组为 4、保留 2）。所有数据类型稀疏度均为 50%。

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
- `d_in`（IN）：DEVICE，待剪枝的稠密矩阵 A 的指针，须 16 字节对齐，不可为 nullptr。
- `d_out`（OUT）：DEVICE，剪枝结果 A_pruned 的指针，须 16 字节对齐，不可为 nullptr。支持 in-place（d_in == d_out）。
- `pruneAlg`（IN）：HOST，剪枝算法，支持 `ACLSPARSELT_PRUNE_SPMMA_STRIP` 与 `ACLSPARSELT_PRUNE_SPMMA_TILE`。
- `stream`（IN）：HOST，ACL 流，算子在此流上异步执行，可为 nullptr（表示使用默认流）。

**约束说明**：

- handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`。
- matmulDescr 不可为 nullptr，且其绑定的矩阵 A 描述符（matA）不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- stream 可为 nullptr（表示使用默认流）。
- pruneAlg 支持 `ACLSPARSELT_PRUNE_SPMMA_STRIP` 与 `ACLSPARSELT_PRUNE_SPMMA_TILE`，其他值返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- 矩阵 A 的数据类型支持 `ACL_FLOAT`（FP32）/ `ACL_FLOAT16`（FP16）/ `ACL_BF16`（BF16）/ `ACL_INT8`（INT8），其他类型返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- d_in 与 d_out 须 16 字节对齐，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- d_in 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- d_out 不可为 nullptr（无 workspace 回退路径），否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- 支持 in-place 操作：d_in 与 d_out 可指向同一 Device 内存。
- **K 维度无对齐约束**：kernel 通过 DataCopyPad 和 tail-clearing 处理非对齐 K，无需 K 为分组大小的倍数。
- **UB 容量约束**：沿行剪枝时单行数据（k × elemSize）须不超过单核 UB 容量；沿列剪枝时行块数据（k × elemSize × 16）须不超过单核 UB 容量。超限时返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **异步执行**：算子异步启动，内部不执行 `aclrtSynchronizeStream`；调用方如需读取 d_out 结果，须自行同步 stream。
- **内存布局**：矩阵 A 须以行主序存储（当前实现中 order 仅影响剪枝方向，不改变内存访问模式；不支持真实列主序存储的矩阵）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为 nullptr
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matmulDescr/matA/d_in/d_out 为 nullptr、数据指针未 16 字节对齐
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：pruneAlg 非 STRIP/TILE、数据类型非 FP32/FP16/BF16/INT8、K 维度超过 UB 容量
- 其他值：失败

---

## 枚举说明

### aclsparseOperation_t

稀疏操作类型枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_OP_NON_TRANSPOSE` | 非转置操作 |
| `ACL_SPARSE_OP_TRANSPOSE` | 转置操作 |
| `ACL_SPARSE_OP_CONJUGATE_TRANSPOSE` | 共轭转置操作 |

### aclsparseStatus_t

返回状态枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_STATUS_SUCCESS` | 操作成功完成 |
| `ACL_SPARSE_STATUS_NOT_INITIALIZED` | aclSPARSE库未初始化 |
| `ACL_SPARSE_STATUS_ALLOC_FAILED` | 资源分配失败 |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | 无效的参数值 |
| `ACL_SPARSE_STATUS_ARCH_MISMATCH` | 设备架构不匹配 |
| `ACL_SPARSE_STATUS_EXECUTION_FAILED` | NPU程序执行失败 |
| `ACL_SPARSE_STATUS_INTERNAL_ERROR` | 内部aclSPARSE操作失败 |
| `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED` | 不支持的矩阵类型 |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 当前不支持该操作或数据类型组合 |
| `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` | 资源不足 |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为空 |

### aclsparseSpMVAlg_t

SpMV算法枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_SPMV_ALG_DEFAULT` | 默认算法 |
| `ACL_SPARSE_SPMV_COO_ALG1` | COO格式默认算法 |
| `ACL_SPARSE_SPMV_COO_ALG2` | COO格式确定性算法 |
| `ACL_SPARSE_SPMV_CSR_ALG1` | CSR/CSC格式默认算法 |
| `ACL_SPARSE_SPMV_CSR_ALG2` | CSR/CSC格式确定性算法 |
| `ACL_SPARSE_SPMV_SELL_ALG1` | SELL格式默认算法 |
| `ACL_SPARSE_SPMV_CSR_ALG_CUBE` | CSR格式 cube（矩阵单元）段求和算法，仅支持非转置计算，matA/vecX 为 fp32 或 fp16，vecY 为 fp32 |

### aclsparseSpMMAlg_t

SpMM算法枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_SPMM_ALG_DEFAULT` | 默认 CSR 算法；arch22 按布局自动选择快路径或通用路径 |
| `ACL_SPARSE_SPMM_CSR_ALG1` | CSR 布局感知通用算法，优先用于列主序稠密矩阵 |
| `ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG` | fp32 高精度算法；fp32 累加使用 Kahan 补偿求和；仅对 fp32 生效 |
| `ACL_SPARSE_SPMM_CSR_ALG2` | CSR 行主序算法；arch22 在紧凑行主序 N/N 时选择快路径，其他合法布局回退通用路径 |
| `ACL_SPARSE_SPMM_CSR_ALG3` | 固定原始 CSR 行序的确定性算法；`opA` 仅支持非转置，`opB` 不支持共轭转置 |

### aclsparseSpMMOpAlg_t

SpMMOp算法枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_SPMMOP_ALG_DEFAULT` | 默认算法，归一化为 ALG1，无 workspace |
| `ACL_SPARSE_SPMMOP_ALG1` | 确定性算法；host 侧静态均匀行切分，无 workspace，支持不重建 descr/plan 原地更新 CSR 值数组 |
| `ACL_SPARSE_SPMMOP_ALG2` | 确定性算法；host 侧 merge sort（`std::stable_sort`）+ bin_edge 按每行 nnz 负载均衡切分，需要 workspace，不支持原地更新 CSR 值 |
| `ACL_SPARSE_SPMMOP_ALG1_HIGH_PRECISION` | FP32 Kahan 补偿求和高精度算法；仅 FP32 dtype 生效（FP16 静默忽略，退化为 ALG1 行为）；workspace 语义同 ALG1（无 workspace） |

### aclsparseOrder_t

稠密矩阵布局枚举（SpMM 中 B / C 使用）：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_ORDER_ROW` | 行主序 |
| `ACL_SPARSE_ORDER_COL` | 列主序 |

### aclsparseIndexType_t

索引类型枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_INDEX_32I` | 32位有符号整数 [0, 2^31 - 1]；**当前 SpMV / SpMM 已实现** |
| `ACL_SPARSE_INDEX_64I` | 64位有符号整数 [0, 2^63 - 1]；**暂未支持**（`aclsparseCreateCsr` 传入时返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`） |

### aclsparseIndexBase_t

索引基值枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_INDEX_BASE_ZERO` | 索引从0开始（C语言兼容） |
| `ACL_SPARSE_INDEX_BASE_ONE` | 索引从1开始（Fortran兼容） |

### aclsparseFormat_t

稀疏矩阵格式枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_FORMAT_COO` | COO（Coordinate）格式 |
| `ACL_SPARSE_FORMAT_CSR` | CSR（Compressed Sparse Row）格式 |
| `ACL_SPARSE_FORMAT_CSC` | CSC（Compressed Sparse Column）格式 |
| `ACL_SPARSE_FORMAT_BLOCKED_ELL` | Blocked-ELL格式 |
| `ACL_SPARSE_FORMAT_SLICED_ELL` | Sliced-ELL格式 |
| `ACL_SPARSE_FORMAT_BSR` | BSR（Block Sparse Row）格式 |

---

## 推荐调用流程

1. 调用 `aclsparseCreate` 创建句柄，并通过 `aclsparseSetStream` 绑定 stream。
2. 使用 `aclsparseCreateCsr` 创建 CSR 稀疏矩阵 A，使用 `aclsparseCreateDnMat` 创建稠密矩阵 B、C。
3. 调用 `aclsparseSpMMGetBufferSize` 获取 workspace 大小并分配设备内存。
4. 调用 `aclsparseSpMMPreprocess` 对稀疏矩阵进行预处理（同一 sparsity pattern 可复用 workspace）。
5. 调用 `aclsparseSpMM` 执行稀疏矩阵-稠密矩阵乘法。
6. 依次销毁矩阵描述符与句柄，释放 workspace 与 ACL 资源。

---

## 最小示例（伪代码）

```c
aclsparseHandle_t handle = nullptr;
aclsparseCreate(&handle);
aclsparseSetStream(handle, stream);

// C = alpha * A * B + beta * C：A 为 rows × cols，B 为 cols × colsOut，C 为 rows × colsOut
aclsparseSpMatDescr_t matA;
aclsparseCreateCsr(&matA, rows, cols, nnz,
    dA_csrOffsets, dA_columns, dA_values,
    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
    ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);

aclsparseDnMatDescr_t matB, matC;
aclsparseCreateDnMat(&matB, cols, colsOut, colsOut, dB, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
aclsparseCreateDnMat(&matC, rows, colsOut, colsOut, dC, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);

size_t bufferSize = 0;
float alpha = 1.0f, beta = 0.0f;
aclsparseSpMMGetBufferSize(handle,
    ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
    &alpha, matA, matB, &beta, matC,
    ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, &bufferSize);

void *dBuffer = nullptr;
aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclsparseSpMMPreprocess(handle,
    ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
    &alpha, matA, matB, &beta, matC,
    ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, dBuffer);
aclsparseSpMM(handle,
    ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
    &alpha, matA, matB, &beta, matC,
    ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, dBuffer);

aclsparseDestroyDnMat(matB);
aclsparseDestroyDnMat(matC);
aclsparseDestroySpMat(matA);
aclsparseDestroy(handle);
aclrtFree(dBuffer);
```

---

## 备注

- 接口能力与属性支持范围以当前实现版本为准，详细限制请参考各算子目录下的README文档（如`test/spmv/README.md`、`test/spmm/README.md`）及头文件内注释。
- 若文档描述与头文件声明不一致，请**以头文件声明与实际实现行为为准**。
