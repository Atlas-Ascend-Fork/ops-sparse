# SpMMACompress

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 将满足结构化稀疏条件、采用稠密存储的矩阵压缩为保留值（values）和位置索引（metadata），减少存储量。
- `aclsparseLtSpMMACompressedSize` 查询压缩结果和压缩临时空间的字节数；`aclsparseLtSpMMACompress` 在指定 stream 上异步执行压缩。
- 输入矩阵及分组方向由 Plan 中结构化稀疏一侧（A 或 B）的描述符和 op 确定。压缩不使用 Plan 的矩阵乘法 tiling 或 workspace，也不执行剪枝、解压或矩阵乘法。
- 沿行和沿列均采用 RegBase 向量实现，支持合法尺寸下不足一个完整向量的尾部。保留值按原始位模式复制，不进行浮点精度转换。

## 参数说明

接口声明见 [cann_ops_sparseLt.h](../../include/cann_ops_sparseLt.h)。以下数据类型列采用 C API 参数类型；矩阵元素类型见约束说明。

### aclsparseLtSpMMACompressedSize

```cpp
aclsparseStatus_t aclsparseLtSpMMACompressedSize(
    const aclsparseLtHandle_t* handle, const aclsparseLtMatmulPlan_t* plan,
    size_t* compressedSize, size_t* compressBufferSize);
```

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| handle | 输入 | 已初始化的库句柄，位于 Host。 | const aclsparseLtHandle_t* | - |
| plan | 输入 | 已初始化的 Plan，位于 Host；A、B 中恰有一个结构化稀疏矩阵。 | const aclsparseLtMatmulPlan_t* | - |
| compressedSize | 输出 | Host 指针，返回压缩结果字节数。不可为空，失败时保持原值。 | size_t* | - |
| compressBufferSize | 输出 | Host 指针，返回压缩临时空间字节数。不可为空，成功时为 0，失败时保持原值。 | size_t* | - |

### aclsparseLtSpMMACompress

```cpp
aclsparseStatus_t aclsparseLtSpMMACompress(
    const aclsparseLtHandle_t* handle, const aclsparseLtMatmulPlan_t* plan,
    const void* d_dense, void* d_compressed, void* d_compressed_buffer,
    aclrtStream stream);
```

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| handle | 输入 | 已初始化的库句柄，位于 Host。 | const aclsparseLtHandle_t* | - |
| plan | 输入 | 已初始化的 Plan，位于 Host；A、B 中恰有一个结构化稀疏矩阵。 | const aclsparseLtMatmulPlan_t* | - |
| d_dense | 输入 | 待压缩矩阵的 Device 指针，存储布局须与描述符一致。 | const void* | 行主序或列主序 |
| d_compressed | 输出 | 压缩结果的 Device 指针，容量至少为大小查询返回的 compressedSize 字节。 | void* | values + metadata，见输出格式 |
| d_compressed_buffer | 可选输入/输出 | 压缩临时空间的 Device 指针。当前实现无需临时空间，不解引用此参数，可传 nullptr。 | void* | - |
| stream | 输入 | 执行流句柄，位于 Host；nullptr 表示默认流。 | aclrtStream | - |

## 约束说明

### 数据类型与尺寸

| 矩阵元素类型 | 每组元素数 G | 每组保留数 H | rows、cols、ld 的倍数 |
| --- | ---: | ---: | ---: |
| FP32（ACL_FLOAT） | 2 | 1 | 8 |
| FP16（ACL_FLOAT16）、BF16（ACL_BF16） | 4 | 2 | 16 |
| INT8（ACL_INT8） | 4 | 2 | 32 |

- 稀疏矩阵描述符的 sparsity 须为 `ACL_SPARSE_LT_SPARSITY_50_PERCENT`。输入还须满足公共[矩阵描述符](../matmul_descriptor/README.md)和 [Plan](../matmul_plan/README.md) 的创建约束，包括 INT8 支持的 A/B、op、order 组合。
- rows、cols、ld 均为正，rows、cols 不超过 INT32_MAX。ld 和 batchStride 使用 64 位计算；跨度、输出大小和地址计算不能溢出。
- 两个接口都需要有效的 Ascend 950PR&950DT系列产品运行时环境。连续物理宽度对应的字节数无需为 256 的整数倍；宽矩阵按片上统一缓冲区（UB）容量分块。

### 输入布局与内存

- op 为 `ACL_SPARSE_OP_NON_TRANSPOSE`（N）时，沿原矩阵的行分组；为 `ACL_SPARSE_OP_TRANSPOSE`（T）时，沿列分组。A、B 两侧采用相同规则。
- order 决定物理存储：ROW 时 `R=rows,C=cols`，COL 时 `R=cols,C=rows`。物理坐标 `(b,r,c)` 的元素偏移为 `b*batchStride+r*ld+c`，偏移单位为元素，要求 `ld >= C`。
- 令 `alongRow=(op==N)==(order==ROW)`。为真时，每组包含同一物理行中连续的 G 个元素；为假时，每组包含同一物理列中连续 G 行的元素。
- 批次数为正，batchStride 非负。batchStride 为 0 时，各批次读取同一输入矩阵；非零时至少为 `(R-1)*ld+C`。
- 描述符的 alignment 至少为 16 且为 16 的倍数。输入与输出首址均须满足该对齐值，两个完整内存跨度不得重叠。
- 调用者须保证 Device 地址有效、分配容量充足，相关对象和内存在任务完成前保持有效；接口无法同步检查实际分配容量和对象生命周期。
- 使用 [Prune](../prune/README.md) 的输出时，须核对实际存储布局。Prune 对连续行主序和转置后输出有专门约定，部分布局的输出与原描述符不一致。

### 非零值与数值语义

- 每组最多含 H 个非零值，此条件由调用者保证。接口不读取 Device 数据进行同步合法性检查；组内非零数量超限时，不保证压缩数值结果。
- 浮点数屏蔽符号位后，剩余位全零才判为零：+0、-0 均按零处理，NaN、Inf 和非规格化数（subnormal）均按非零处理。INT8 按字节是否为零判断。
- 每组保留全部非零值；不足 H 个时，按原位置从小到大补选零值，最后按原位置升序输出。
- 选中值逐位复制，保留负零符号位、NaN 载荷位（payload）和 FP32 低位，不进行 TF32 转换。未选中零值的符号位不保证能够恢复。
- 输出使用下述本库格式，不保证与 cuSPARSELt 的完整压缩缓冲区二进制兼容。现有 Matmul、SpMM 及稀疏 MMA 接口尚不支持读取该格式。

## 输出格式

设批次数为 nb，矩阵元素类型为 T，总元素数 `E=nb*R*C`，总组数 `NG=E/G`。R、C、G、H 的含义见约束说明。输出依次存放所有批次的 values 和所有批次的 metadata，不含头部、格式版本字段或填充字节。

| 区域 | 字节偏移 | 字节数 |
| --- | --- | --- |
| values | 0 | `V=E/2*sizeof(T)` |
| metadata | V | `M=NG/2` |

`compressedSize=V+M`。各组先按批次排列：alongRow 为真时，批内组号为 `r*(C/G)+j`；否则为 `rg*C+c`。其中 j 为行内组号，rg 为每 G 行构成的行组编号。全局组 g 的第 s 个保留值位于 values 区域的元素 `g*H+s`。

每组使用一个四位位置码 q。metadata 字节 `g/2` 的低四位存偶数组，高四位存奇数组。

- FP16/BF16/INT8：`q=p0+4*p1`，p0<p1 为两个保留位置。位置对 (0,1)/(0,2)/(0,3)/(1,2)/(1,3)/(2,3) 分别对应 4/8/12/9/13/14。
- FP32：保留位置 0/1 分别编码为 4/14，位置编码不改变保留值的精度。
- 全零组选择位置 0 或 (0,1)，位置码均为 4。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclsparseLt C API | - | 通过 [cann_ops_sparseLt.h](../../include/cann_ops_sparseLt.h) 声明的大小查询与压缩接口调用。 |

### 调用流程

1. 初始化 ACL、设备、库句柄、矩阵描述符和合法 Plan。
2. 调用 `aclsparseLtSpMMACompressedSize`，查询结果和临时空间大小。当前 `compressBufferSize` 为 0，无需分配压缩临时空间；`compressedSize` 为实际输出字节数。
3. 分配 Device 输入、输出缓冲区，准备符合描述符和分组规则的输入矩阵。
4. 调用 `aclsparseLtSpMMACompress`，将 `d_compressed_buffer` 设为 nullptr，并检查返回值。
5. 同步对应 stream 并检查运行时错误，再读取结果或释放对象和内存。

接口内部不分配 Device 内存，也不进行 stream 同步。测试中的完整调用过程可参考[测试封装](../../test/spmma_compress/arch35/spmma_compress_npu_wrapper.h)的 `RunCompressCase`；构建与运行方式见[测试说明](../../test/spmma_compress/README.md)。

### 返回值

两个接口均返回 `aclsparseStatus_t`。Compress 返回成功表示 Host 校验和 kernel 启动调用已完成，设备任务可能仍在执行。

| 返回值 | 含义 |
| --- | --- |
| ACL_SPARSE_STATUS_SUCCESS | 大小查询成功，或压缩任务已发起。 |
| ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR | handle 外层或内部为空。 |
| ACL_SPARSE_STATUS_INVALID_VALUE | 非法 Plan、参数、尺寸、对齐、跨度或溢出。 |
| ACL_SPARSE_STATUS_EXECUTION_FAILED | Compress 启动前已有线程运行时错误，或启动后立即观察到错误。 |
| ACL_SPARSE_STATUS_NOT_SUPPORTED | 不支持的数据类型。 |
| ACL_SPARSE_STATUS_ARCH_MISMATCH | 平台不是 Ascend 950PR&950DT系列产品。 |
| ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES | 平台资源不可用，或 UB 无法容纳最小分块。 |

Compress 在 kernel 启动前后调用 `aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)`。线程已有错误时，不提交任务；启动后立即观察到错误时，也返回执行失败。接口不清除线程错误，由调用者处理。此检查仅覆盖 last-error 通道可立即观察到的错误，无法覆盖所有底层入队故障；设备执行期间的错误仍须通过后续 ACL 同步操作检查。
