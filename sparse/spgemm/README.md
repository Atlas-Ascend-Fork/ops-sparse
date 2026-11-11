# SpGEMM算子

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

SpGEMM（Sparse General Matrix-Matrix Multiplication）计算两个 CSR 稀疏矩阵的乘积：

```text
C' = alpha * op(A) * op(B) + beta * C
```

其中 A 的形状为 `M×K`，B 的形状为 `K×N`，输出 C 的形状为 `M×N`。当前版本仅支持
`op(A)=A`、`op(B)=B`。多个中间乘积落到同一坐标时会被合并；输出 CSR 每行列索引严格升序，
不含重复坐标。数值抵消得到的显式零仍保留在输出结构中。

算子采用多阶段 Generic API：先统计中间乘积数量并查询 workspace，再计算输出结构和数值，
最后把内部结果写入调用方分配的动态输出。Python 适配通过公开接口
`torch.sparse.mm(mat1, mat2)` 调用同一套 `aclsparseSpGEMM*` 实现。

### 接口与职责

| 接口名 | 功能简述 |
| --- | --- |
| `aclsparseSpGEMMCreateDescr` | 创建跨阶段复用的 SpGEMM 描述符 |
| `aclsparseSpGEMMWorkEstimation` | 查询/执行工作量统计，得到逐行及总中间乘积数量 |
| `aclsparseSpGEMMGetNumProducts` | 查询 WorkEstimation 得到的中间乘积总数 |
| `aclsparseSpGEMMEstimateMemory` | 查询 ALG2/ALG3 内存规划及 Compute workspace 大小 |
| `aclsparseSpGEMMCompute` | 查询/执行结构和数值计算，写出 rowOffsets 并更新 `nnz(C)` |
| `aclsparseSpGEMMCopy` | 将内部列索引和 values 写入调用方设置的 C 指针 |
| `aclsparseSpGEMMDestroyDescr` | 销毁 Host 描述符，不释放调用方 workspace |
| `aclsparseSpGEMMSetCInValid` | 兼容旧版 C 输入有效标记调用 |
| `aclsparseSpGEMMGetBufferSize` | 查询旧三阶段兼容接口的共用 workspace |
| `aclsparseSpGEMMPreprocess` | 执行旧三阶段兼容接口的结构预处理 |
| `aclsparseSpGEMM` | 执行旧三阶段兼容接口的数值计算 |

## 接口说明

### 描述符创建与销毁

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpGEMMCreateDescr(
    aclsparseSpGEMMDescr_t *descr);

aclsparseStatus_t aclsparseSpGEMMDestroyDescr(
    aclsparseSpGEMMDescr_t descr);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
| --- | --- | --- | --- |
| `descr` | 输出/输入 | `aclsparseSpGEMMDescr_t*` / `aclsparseSpGEMMDescr_t` | 不透明 Host 描述符；Create 前 `*descr` 必须为 `nullptr` |

#### 约束说明

- 描述符必须在其他 SpGEMM 阶段之前创建，并在所有异步工作完成后销毁。
- `aclsparseSpGEMMDestroyDescr(nullptr)` 返回成功。
- Destroy 只释放 Host 描述符，不同步 stream，也不释放 `externalBuffer1/2/3`。

### 公共参数

以下参数由 WorkEstimation、EstimateMemory、Compute 和 Copy 共用。

| 参数名 | 输入/输出 | 参数类型 | 内存位置 | 说明 |
| --- | --- | --- | --- | --- |
| `handle` | 输入 | `aclsparseHandle_t` | Host | ops-sparse 上下文句柄，须通过 `aclsparseSetStream` 绑定 stream |
| `opA` | 输入 | `aclsparseOperation_t` | Host | 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE` |
| `opB` | 输入 | `aclsparseOperation_t` | Host | 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE` |
| `alpha` | 输入 | `const void*` | Host/Device | 标量 alpha，地址空间由 handle 的 pointer mode 决定，dtype 等于 `computeType` |
| `matA` | 输入 | `aclsparseConstSpMatDescr_t` | Host 描述符，Device 数据 | `M×K` CSR 输入矩阵 A |
| `matB` | 输入 | `aclsparseConstSpMatDescr_t` | Host 描述符，Device 数据 | `K×N` CSR 输入矩阵 B |
| `beta` | 输入 | `const void*` | Host/Device | 标量 beta，地址空间由 handle 的 pointer mode 决定，dtype 等于 `computeType` |
| `matC` | 输入/输出 | `aclsparseSpMatDescr_t` | Host 描述符，Device 数据 | `M×N` CSR 输入/输出矩阵 C；Compute 更新 `nnz` |
| `computeType` | 输入 | `aclDataType` | Host | `ACL_FLOAT16`、`ACL_BF16`、`ACL_FLOAT` 或 `ACL_COMPLEX64` |
| `alg` | 输入 | `aclsparseSpGEMMAlg_t` | Host | DEFAULT、ALG1、ALG2 或 ALG3；同一描述符全流程不可改变 |
| `spgemmDescr` | 输入/输出 | `aclsparseSpGEMMDescr_t` | Host | 保存阶段状态、问题快照、workspace 规划、`numProds` 和 `nnz(C)` |

### aclsparseSpGEMMWorkEstimation

查询并执行中间乘积工作量统计。第一次调用传入 `externalBuffer1=nullptr` 查询大小；分配 Device
workspace 后第二次调用执行统计。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpGEMMWorkEstimation(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize1, void *externalBuffer1);
```

#### 阶段专用参数

| 参数名 | 输入/输出 | 参数类型 | 内存位置 | 说明 |
| --- | --- | --- | --- | --- |
| `bufferSize1` | 输入/输出 | `size_t*` | Host | 查询时写回所需字节数；执行时传入实际可用字节数 |
| `externalBuffer1` | 输入 | `void*` | Device | 第一次调用为 `nullptr`，第二次调用为不少于 `bufferSize1` 字节的 workspace |

执行成功后，`externalBuffer1` 必须保留到 Copy 完成。输入 CSR 非法、workspace 不足或中间乘积数
超过 `INT32_MAX` 时返回相应错误码。

### aclsparseSpGEMMGetNumProducts

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpGEMMGetNumProducts(
    aclsparseSpGEMMDescr_t spgemmDescr,
    int64_t *numProds);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
| --- | --- | --- | --- |
| `spgemmDescr` | 输入 | `aclsparseSpGEMMDescr_t` | 已成功执行 WorkEstimation 的描述符 |
| `numProds` | 输出 | `int64_t*` | 中间乘积总数，不等于去重后的最终 `nnz(C)` |

### aclsparseSpGEMMEstimateMemory

用于查询 Compute workspace。ALG2/ALG3 的 `chunkFraction` 必须位于 `(0, 1]`。当前 Ascend 950PR
确定性实现返回 `bufferSize3=0`，`externalBuffer3` 为兼容多阶段接口保留。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpGEMMEstimateMemory(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    float chunkFraction,
    size_t *bufferSize3, void *externalBuffer3,
    size_t *bufferSize2);
```

#### 阶段专用参数

| 参数名 | 输入/输出 | 参数类型 | 内存位置 | 说明 |
| --- | --- | --- | --- | --- |
| `chunkFraction` | 输入 | `float` | Host | ALG2/ALG3 必须在 `(0,1]`；DEFAULT/ALG1 保留该参数 |
| `bufferSize3` | 输出 | `size_t*` | Host | 当前实现写回 0 |
| `externalBuffer3` | 输入 | `void*` | Device | 当前实现保留，可传 `nullptr` |
| `bufferSize2` | 输出 | `size_t*` | Host | Compute 所需 workspace 字节数 |

### aclsparseSpGEMMCompute

第一次调用传入 `externalBuffer2=nullptr` 可直接查询大小；第二次调用执行结构及数值计算。Compute
写出 `matC.rowOffsets`，更新描述符中的 `nnz(C)`，但不会为输出列索引和 values 分配内存。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpGEMMCompute(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize2, void *externalBuffer2);
```

#### 阶段专用参数

| 参数名 | 输入/输出 | 参数类型 | 内存位置 | 说明 |
| --- | --- | --- | --- | --- |
| `bufferSize2` | 输入/输出 | `size_t*` | Host | 查询时写回所需字节数；执行时传入实际可用字节数 |
| `externalBuffer2` | 输入 | `void*` | Device | 第一次调用可为 `nullptr`；执行时为不少于 `bufferSize2` 字节的 workspace |

`matC` 在 Compute 前必须至少绑定长度为 `M+1` 的 Device rowOffsets。`beta=0` 时，可在 Compute
成功并通过 `aclsparseSpMatGetSize` 获得 `nnz(C)` 后再分配输出 colIndices 和 values；
`beta!=0` 时，输入 C 必须预先提供与结果相同的 CSR pattern 和有效 values。

为保持已有源码兼容，`aclsparseSpGEMMSetCInValid` 继续保留。新多阶段接口根据
`beta!=0` 以及 Compute 前已绑定的 `matC` CSR pattern/values 判定有效 `C_in`；旧调用方
无需删除 `SetCInValid` 调用，累加语义仍为 `C' = alpha * A * B + beta * C_in`。

旧三阶段接口 `aclsparseSpGEMMGetBufferSize`、`aclsparseSpGEMMPreprocess` 和
`aclsparseSpGEMM` 同样保留，并在内部复用上述多阶段实现。兼容 workspace 同时容纳工作量
统计和数值计算区域，其生命周期要求与旧接口一致。

### aclsparseSpGEMMCopy

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpGEMMCopy(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr);
```

Compute 后应调用 `aclsparseSpMatGetSize` 获取 `nnz(C)`，为 colIndices 和 values 分配 Device
内存，再通过 `aclsparseCsrSetPointers` 更新 `matC`。Copy 异步写出最终 CSR，读取结果或释放
workspace 前必须同步 handle 所绑定的 stream。

### 旧三阶段兼容接口

以下三个接口在内部复用上述多阶段实现，供旧调用方以"查询大小 → 分配 → 一次执行"的方式
使用；兼容 workspace 同时容纳工作量统计与数值计算区域。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpGEMMGetBufferSize(aclsparseHandle_t handle, aclsparseOperation_t opA,
                                               aclsparseOperation_t opB, const void* alpha,
                                               aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
                                               const void* beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
                                               aclsparseSpGEMMAlg_t alg, size_t* size);

aclsparseStatus_t aclsparseSpGEMMPreprocess(aclsparseHandle_t handle, aclsparseOperation_t opA,
                                            aclsparseOperation_t opB, const void* alpha,
                                            aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
                                            const void* beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
                                            aclsparseSpGEMMAlg_t alg, void* buffer);

aclsparseStatus_t aclsparseSpGEMM(aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
                                  const void* alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
                                  const void* beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
                                  aclsparseSpGEMMAlg_t alg, void* buffer);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
| --- | --- | --- | --- |
| `size` | 输出 | `size_t*` | GetBufferSize 写回兼容 workspace 所需字节数，不可为 `nullptr` |
| `buffer` | 输入 | `void*` | Device workspace；GetBufferSize 时不使用，Preprocess/SpGEMM 执行时必须非空且不少于查询结果 |
| 其余参数 | 输入 | — | 与 WorkEstimation/Compute/Copy 的公共参数一致 |

#### 约束说明

- `buffer` 为 `nullptr` 时 Preprocess/SpGEMM 返回 `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`；
  执行时内部布局与 GetBufferSize 查询值不一致返回 `ACL_SPARSE_STATUS_INTERNAL_ERROR`。
- 调用 Preprocess/SpGEMM 前，`matC` 必须已绑定长度为 `A 行数 + 1` 的 Device rowOffsets；由于
  旧接口内部会完成 Copy，colIndices 与 values 也必须在调用前绑定且容量不小于 `nnz(C)`。实现
  只校验输出指针非空，不做容量校验；容量不足会越界写设备内存。
- Preprocess 成功后，legacy 描述符与传入 `buffer` 挂接在 `matC` 上（含 `activeBuffer`）；
  后续以同一 `buffer`、同一问题（A/B/C/computeType/alg）调用 SpGEMM 直接命中缓存，不再预处理。
- 缓存命中判定（ProblemMatches）只比较 matA/matB/matC 描述符对象、shape、nnz、computeType 与
  alg，不比较 CSR 内容；原地改写输入 CSR 结构（即使 nnz 不变）后，必须重新调用
  `aclsparseSpGEMMPreprocess`（或换用不同 `buffer` 强制重预处理），否则会复用陈旧的行乘积
  统计导致结果错误。
- `matC` 未缓存 legacy 描述符、传入 `buffer` 与缓存不一致、或问题（A/B/C/computeType/alg）
  不匹配时，SpGEMM 会自动执行一次完整预处理（含 ProblemMatches 判定）；即 SpGEMM 可独立于
  显式 Preprocess 调用，但该冷路径会同步执行全部预处理阶段。
- 换用不同 `buffer` 或更换问题会触发上述隐式重预处理；调用方不应假设 SpGEMM 不做预处理。
- 兼容 workspace 的生命周期要求与旧接口一致：读取结果或释放前须同步 stream。

## 支持规格

| 规格项 | 支持值 | 说明 |
| --- | --- | --- |
| 输入/输出格式 | CSR | A、B、C 均为二维 CSR；不支持 COO、CSC、batch 或广播 |
| 数据类型 | FP16、BF16、FP32、Complex64 | A、B、C values 类型必须与 `computeType` 完全一致 |
| 索引类型 | `ACL_SPARSE_INDEX_32I` | rowOffsets 与 colIndices 均须为 int32 |
| 索引基址 | `ACL_SPARSE_INDEX_BASE_ZERO` | 不支持 one-based CSR |
| `opA` / `opB` | `ACL_SPARSE_OP_NON_TRANSPOSE` | 不支持转置或共轭转置 |
| 算法 | DEFAULT、ALG1、ALG2、ALG3 | DEFAULT 与 ALG1 使用确定性全量候选路径；四种算法输出语义一致 |
| scalar pointer mode | Host、Device | alpha 与 beta 必须同时符合 handle 当前 pointer mode |
| 输出结构 | 每行列索引严格升序、无重复坐标 | 数值为零的结构项仍保留 |
| 规模上限 | shape、nnz 和中间乘积数不超过 `INT32_MAX` | 同时受 `size_t` 和实际 Device 内存限制 |

## 约束说明

- A 的列数必须等于 B 的行数；C 的形状必须为 `A.rows × B.cols`。
- A、B、C 必须使用相同的 value dtype、int32 索引类型和零基索引。
- A、B 输入 CSR 的 rowOffsets 必须单调非降，列索引必须在合法范围内并按行非降。
- `alpha`、`beta`、各矩阵描述符、SpGEMM 描述符及阶段所需 size 指针不可为 `nullptr`。
- 同一描述符跨阶段不得更换矩阵描述符、shape、nnz、dtype 或算法；结构变化后必须重新执行
  WorkEstimation。
- WorkEstimation 和 Compute 都采用“空 buffer 查询大小，再传入 Device buffer 执行”的两次调用约定。
- `externalBuffer1` 和 `externalBuffer2` 的生命周期必须延续至 Copy 完成；执行时传入的字节数不得
  小于查询结果。
- Copy 异步提交；调用方读取输出、销毁输入数据或复用 workspace 前须同步 stream。
- `aclsparseSpGEMMDestroyDescr` 不会释放调用方创建的矩阵描述符或 Device 内存。

## 调用流程

推荐流程如下。省略了统一的返回码检查和 Device 分配辅助函数，完整可运行路径见
[`../../test/spgemm/arch35/spgemm_test.cpp`](../../test/spgemm/arch35/spgemm_test.cpp)。

```cpp
aclsparseSpGEMMDescr_t descr = nullptr;
aclsparseSpGEMMCreateDescr(&descr);

// A、B 是完整 CSR；C 已设置 shape、dtype 和长度 M+1 的 rowOffsets。
size_t size1 = 0;
aclsparseSpGEMMWorkEstimation(handle, opA, opB, alpha, matA, matB, beta,
    matC, computeType, alg, descr, &size1, nullptr);
void *buffer1 = DeviceMalloc(size1);
aclsparseSpGEMMWorkEstimation(handle, opA, opB, alpha, matA, matB, beta,
    matC, computeType, alg, descr, &size1, buffer1);

int64_t numProducts = 0;
aclsparseSpGEMMGetNumProducts(descr, &numProducts);

size_t size2 = 0;
if (alg == ACL_SPARSE_SPGEMM_ALG2 || alg == ACL_SPARSE_SPGEMM_ALG3) {
    size_t size3 = 0;
    aclsparseSpGEMMEstimateMemory(handle, opA, opB, alpha, matA, matB, beta,
        matC, computeType, alg, descr, 1.0F, &size3, nullptr, &size2);
} else {
    aclsparseSpGEMMCompute(handle, opA, opB, alpha, matA, matB, beta,
        matC, computeType, alg, descr, &size2, nullptr);
}

void *buffer2 = DeviceMalloc(size2);
aclsparseSpGEMMCompute(handle, opA, opB, alpha, matA, matB, beta,
    matC, computeType, alg, descr, &size2, buffer2);

int64_t rowsC = 0;
int64_t colsC = 0;
int64_t nnzC = 0;
aclsparseSpMatGetSize(matC, &rowsC, &colsC, &nnzC);
void *colIndicesC = DeviceMalloc(static_cast<size_t>(nnzC) * sizeof(int32_t));
void *valuesC = DeviceMalloc(static_cast<size_t>(nnzC) * ValueSize(computeType));
aclsparseCsrSetPointers(matC, rowOffsetsC, colIndicesC, valuesC);

aclsparseSpGEMMCopy(handle, opA, opB, alpha, matA, matB, beta,
    matC, computeType, alg, descr);
aclrtSynchronizeStream(stream);

aclsparseSpGEMMDestroyDescr(descr);
DeviceFree(buffer2);
DeviceFree(buffer1);
```

## 编译与测试

```bash
cmake -S . -B build \
  -DSOC_VERSION=ascend950 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TEST=ON \
  -DOP_LIST=spgemm
cmake --build build --target spgemm_test spgemm_perf --parallel
./build/test/spgemm/spgemm_test --gtest_color=no
```

PyTorch 接口注册通过 `torch_extension` 框架提供，构建和加载方式见
[`../../torch_extension/README.md`](../../torch_extension/README.md)，
SpGEMM 侧的接口说明见
[`../../torch_extension/cann_ops_sparse/docs/zh/spgemm.md`](../../torch_extension/cann_ops_sparse/docs/zh/spgemm.md)。

## 目录结构

```text
sparse/spgemm/
├── README.md
├── common/
│   ├── spgemm_common.h        # 跨 arch 共享的类型与规格定义
│   └── spgemm_validate.cpp    # 公共参数校验
├── arch22/                    # Atlas A2/A3
│   ├── README.md
│   ├── spgemm.h
│   ├── spgemm_host.cpp
│   ├── spgemm_count_kernel.cpp
│   ├── spgemm_symbolic_kernel.cpp
│   ├── spgemm_numeric_kernel.cpp
│   ├── spgemm_merge.h
│   ├── spgemm_merge_emit.h
│   ├── spgemm_merge_row.h
│   └── spgemm_value.h
├── arch35/                    # Ascend 950PR
│   ├── spgemm.h
│   ├── spgemm_host.cpp
│   ├── spgemm_kernel.cpp
│   ├── spgemm_kernel.h
│   ├── spgemm_csr_mat.cpp
│   ├── spgemm_csr_mat.h
│   └── spgemm_tiling_data.h
└── torch_extension/           # PyTorch 接口注册（由 PyTorch JIT 单独编译）
    ├── __init__.py
    ├── spgemm.py
    └── csrc/
        └── spgemm.cpp
```

测试代码位于 `test/spgemm/`。
