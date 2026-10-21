# SpMV算子

## 算子概述

SpMV（Sparse Matrix - Dense Vector Multiplication）算子实现稀疏矩阵与稠密向量的乘法运算。核心运算为 y = alpha * op(A) * x + beta * y，其中 A 为 CSR 格式稀疏矩阵，x 和 y 为稠密向量。

数学表达式：

```
y = alpha * op(A) * x + beta * y
```

其中 op(A) 为稀疏矩阵 A 的操作（转置/非转置），x 为输入稠密向量，y 为输入/输出稠密向量。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSpMV | 执行稀疏矩阵-稠密向量乘法 y = alpha * op(A) * x + beta * y |
| aclsparseSpMVGetBufferSize | 查询所需 workspace 大小 |
| aclsparseSpMVPreprocess | 对稀疏矩阵进行预处理并绑定 active plan |
| aclsparseCsrSetStrides / GetStrides | 设置/获取 CSR rowOffsets、colInd、values 的元素步长 |
| aclsparseDnVecSetStride / GetStride | 设置/获取稠密向量的元素步长 |

## 算子执行接口

### aclsparseSpMV

#### 产品支持情况

- Ascend 950PR（A5，DAV_3510）：支持 CSR/I32 的本页 A5 能力矩阵
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpMV(aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX, const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType, aclsparseSpMVAlg_t alg, void *externalBuffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE`、`ACL_SPARSE_OP_TRANSPOSE` 和 `ACL_SPARSE_OP_CONJUGATE_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，类型须与 computeType 匹配。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 A 的描述符，仅支持 CSR 格式，Host 内存 |
| vecX | 输入 | aclsparseConstDnVecDescr_t | 输入稠密向量 x 的描述符，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针，类型须与 computeType 匹配。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| vecY | 输入/输出 | aclsparseDnVecDescr_t | 输入/输出稠密向量 y 的描述符，Host 内存 |
| computeType | 输入 | aclDataType | A5 支持 `ACL_INT32`、`ACL_FLOAT` 或 `ACL_COMPLEX64`；Atlas A2/A3 支持 `ACL_FLOAT` 或 `ACL_INT32`，Host 内存 |
| alg | 输入 | aclsparseSpMVAlg_t | 算法类型，随产品不同：Ascend 950PR/950DT 支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`、`ACL_SPARSE_SPMV_CSR_ALG1`、`ACL_SPARSE_SPMV_CSR_ALG2`；Atlas A2/A3 支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`（向量单元实现）与 `ACL_SPARSE_SPMV_CSR_ALG_CUBE`（cube 段求和实现），Host 内存 |
| externalBuffer | 输入 | void* | 大小由 `aclsparseSpMVGetBufferSize` 查询的 Device workspace；A5 始终需要非空、64 字节对齐的 buffer。Atlas A2/A3 查询返回 0 时可传 nullptr，否则按查询大小分配 |

#### 约束说明

- handle 不可为 nullptr，且须先调用 `aclsparseSetStream` 设置 stream
- matA、vecX、vecY 不可为 nullptr
- matA 仅支持 CSR 格式（`ACL_SPARSE_FORMAT_CSR`）
- A5 路径支持正的元素 stride：CSR 行偏移、列索引、值以及 x/y 分别使用描述符中的 stride，输出间隙不被修改
- matA 的行偏移和列索引类型必须均为 `ACL_SPARSE_INDEX_32I`，且两者类型相同

以下为 A5 约束；Atlas A2/A3 差异见后文。

- 由于 Kernel 按有符号 `int32_t` 读取 CSR 索引，`M`、`K` 不超过 `INT32_MAX`；`nnz` 不超过
  `INT32_MAX`（base 0）或 `INT32_MAX - 1`（base 1），避免 row offset/column 存储溢出
- matA 的索引基址支持 `ACL_SPARSE_INDEX_BASE_ZERO` 或 `ACL_SPARSE_INDEX_BASE_ONE`
- opA 支持 `ACL_SPARSE_OP_NON_TRANSPOSE`、`ACL_SPARSE_OP_TRANSPOSE` 和
  `ACL_SPARSE_OP_CONJUGATE_TRANSPOSE`；复数值在共轭转置时逐 entry 取共轭，实数值与普通转置相同
- alg 支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`、`ACL_SPARSE_SPMV_CSR_ALG1`、`ACL_SPARSE_SPMV_CSR_ALG2`
- computeType 支持 `ACL_INT32`、`ACL_FLOAT` 或 `ACL_COMPLEX64`
- vecX 的值类型须与 matA 的值类型一致
- 维度匹配：
  - 非转置模式：vecX 大小必须等于 A.cols，vecY 大小必须等于 A.rows
  - 转置模式：vecX 大小必须等于 A.rows，vecY 大小必须等于 A.cols
- 支持 M=0、N=0 和 NNZ=0；输出长度为 0 时成功返回，NNZ=0 时结果为 `beta * y`
- M=0 或 N=0 时必须同时满足 NNZ=0，否则查询、预处理和执行均返回 `INVALID_VALUE`。
- `aclsparseCreateCsr` 和 `aclsparseCreateDnVec` 创建的描述符默认元素步长为 1；通过
  `aclsparseCsrSetStrides` 和 `aclsparseDnVecSetStride` 可原生描述正步长非连续 Tensor
- CSR rowOffsets、colInd、values 以及 vecX、vecY 的 stride 均按元素计数且必须大于 0；vecY 按其 stride 原位读写，间隙元素不会被覆盖
- 非连续 stride 当前仅在 Ascend 950PR/950DT 路径生效；Atlas A2/A3 路径仅支持默认步长 1
- 调用方负责保证 CSR 内容合法：rowOffsets 必须从 base 开始、以 NNZ+base 结束且单调非降，colInd 必须全部位于
  `[base, cols+base)`。SpMV 和 Preprocess 不同步回读 Device 数据，也不承诺返回内容校验错误
- INT8→INT32 路径对完整的 `alpha * dot(A, x) + beta * y` 做最终 INT32 饱和，保留正负项抵消；
  使用乘法前范围检查和平移后的饱和边界，避免中间 INT64 乘加溢出。连续、stride 和转置路径语义一致。
- 支持的值类型组合：
  - matA=ACL_FLOAT, vecX=ACL_FLOAT, vecY=ACL_FLOAT, computeType=ACL_FLOAT
  - matA=ACL_FLOAT16, vecX=ACL_FLOAT16, vecY=ACL_FLOAT 或 ACL_FLOAT16, computeType=ACL_FLOAT
  - matA=ACL_BF16, vecX=ACL_BF16, vecY=ACL_FLOAT 或 ACL_BF16, computeType=ACL_FLOAT
  - matA=ACL_INT8, vecX=ACL_INT8, vecY=ACL_INT32, computeType=ACL_INT32
  - matA=ACL_INT8, vecX=ACL_INT8, vecY=ACL_FLOAT, computeType=ACL_FLOAT
  - matA=ACL_COMPLEX64, vecX=ACL_COMPLEX64, vecY=ACL_COMPLEX64, computeType=ACL_COMPLEX64
  - matA=ACL_FLOAT, vecX=ACL_FLOAT, vecY=ACL_COMPLEX64, computeType=ACL_COMPLEX64

Atlas A2/A3 差异：

- 仅支持 base 0、NON_TRANSPOSE/TRANSPOSE 和默认元素 stride 1，不支持共轭转置。
- computeType 支持 `ACL_FLOAT` 或 `ACL_INT32`；向量长度可大于所需维度。
- alg 支持 `ACL_SPARSE_SPMV_ALG_DEFAULT` 与 `ACL_SPARSE_SPMV_CSR_ALG_CUBE`。
- CUBE 仅支持非转置，computeType 与 y 为 FLOAT，A/x 为 FLOAT 或 FLOAT16；fp16 乘积以 fp16 计算、fp32 累加。归约顺序不同可能造成与默认算法的浮点误差差异。

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 稀疏矩阵 A 支持 CSR 格式 |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

#### 调用示例

暂无运行示例，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

### aclsparseSpMVGetBufferSize

查询 `aclsparseSpMV` 所需的 workspace 字节数。在 arch22 默认算法的转置模式下，FP16/BF16 输出使用每列 4 字节的 FP32 累加工作区，最后一次转换到输出类型；未传入 `externalBuffer` 时使用句柄工作区。调用方负责按返回字节数分配 Device 内存。
此阶段只检查描述符元数据及参数，不读取 CSR 或 x/y 数据，允许这些数据指针尚未绑定。

- Ascend 950PR/950DT：始终返回非零大小，包含 192 字节头部和归一化行偏移；转置时还包含累加区，各区域按 64 字节对齐。
- Atlas A2/A3 `ACL_SPARSE_SPMV_ALG_DEFAULT`（向量版）：转置且 FP16/BF16 输出时返回
  `cols * sizeof(float)`，用于 FP32 累加；其他路径返回 0。
  `externalBuffer` 可传 `nullptr`，需要工作区时由句柄提供。
- Atlas A2/A3 `ACL_SPARSE_SPMV_CSR_ALG_CUBE`：返回分段求和所需的 workspace 大小（gather 乘积区 +
  块内推测扫描结果区）。调用方须按该大小申请显存，并作为 `aclsparseSpMV` 的
  `externalBuffer` 传入，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
  空矩阵（`nnz == 0`）会回退到向量版实现，返回 0。

workspace 由调用方持有，`aclsparseSpMV` 只在其上排队异步任务、不做 stream 同步；
因此复用同一块 buffer 前需自行保证前一次 SpMV 已完成。

### aclsparseSpMVPreprocess

已支持 A5。对同一 CSR pattern 可选调用；它在 workspace 中生成 base 归一化 row pointer，
并将 matA 的 pattern、属性和 workspace 记录为 active plan，后续 `aclsparseSpMV` 可复用。
Atlas A2/A3 暂未支持该接口。A5 预处理只读取 rowOffsets（rows > 0 时必须绑定）及描述符元数据，
并要求 workspace 已分配；values、colInd 和 x/y 可在预处理完成后再绑定，执行阶段仍会完整检查数据指针。
后续更新指针可能使缓存 plan 失效，但可直接执行 SpMV，执行时回退读取原始 CSR，无需再次预处理。
这不适用于原地修改 CSR 结构：若 rowOffsets 或 colInd 的内容发生变化，即使指针地址不变，
也必须在更新完成后重新调用 Preprocess，再执行 SpMV。Host 不读取设备端索引内容，无法自动
发现这种变化；否则可能复用旧的归一化行偏移。仅原地更新 values/x/y 不要求重新预处理。
预处理与执行均将 Device 任务入队到 handle 的 stream，不同步回读 CSR；同一 stream 上的异步 H2D 可以直接排在调用前，
调用方负责在读取结果或复用/释放相关内存前同步，并保证所有异步操作期间内存有效。

## Ascend 950/A5 能力和调用约定

A5 实现位于 `sparse/spmv/arch35/`，目标架构为 `dav-3510`。所有声明组合均使用二维
CSR、I32 row pointer/column index、index base 0/1，并支持 `NON_TRANSPOSE`、`TRANSPOSE` 和
`CONJUGATE_TRANSPOSE`：

| A values / x | y | computeType |
|---|---|---|
| INT8 | INT32 | INT32 |
| INT8 | FP32 | FP32 |
| FP16 | FP32 | FP32 |
| BF16 | FP32 | FP32 |
| FP16 | FP16 | FP32（FP32 累加后转换） |
| BF16 | BF16 | FP32（FP32 累加后转换） |
| FP32 | FP32 | FP32 |
| complex64 | complex64 | complex64 |
| FP32 | complex64 | complex64（FP32 按实数嵌入） |

非转置时 `x.size == A.cols`、`y.size == A.rows`；转置/共轭转置时 `x.size == A.rows`、
`y.size == A.cols`。支持动态 shape、空行、`nnz=0/1`、不均匀长尾以及每行未排序/重复
column；重复坐标的贡献均计入，ALG2 按原始 entry 顺序累加。调用方应保证 `rowPtr` 单调且首尾值与 base/nnz 一致，
column 扣除 base 后位于 `[0, A.cols)`。Host 会校验可见的 descriptor 元数据；device CSR
结构错误的 status 回传仍需在 A5 环境完成并通过专项目测，在该决策门关闭前不得将其宣称为
已支持的异步参数错误。

算法支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`（映射 CSR ALG1）、`ACL_SPARSE_SPMV_CSR_ALG1`
和 `ACL_SPARSE_SPMV_CSR_ALG2`。ALG2 的转置路径使用一个 GM accumulator，按固定
row/entry 顺序单次遍历 `O(nnz + cols)` 完成确定性归约。ALG1/DEFAULT 对
`0 < rows <= kSpmvTransposeGatherRowsMax` 的矩阵先在设备侧检查每行 column 是否有序；有序时每个输出 column 由独占 warp
二分定位并归约，未排序时回退到原始 entry 顺序的无原子路径。较大矩阵采用逐行遍历和
workspace 原子累加。ALG1/DEFAULT 的浮点/复数结果不承诺 bit-wise 确定性。

路径阈值统一定义在 [spmv_tiling_data.h](arch35/spmv_tiling_data.h)，供 Host 启动逻辑与
Kernel 分派共用：`kSpmvTransposeGatherRowsMax` 当前为 4095（包含上界），零行或超过
该上界时 ALG1/DEFAULT 转置走原子路径；FP32 输入/输出/计算的非转置路径在
`rows <= kSpmvFloatSmallRowsMax` 时使用小矩阵实现，后者当前为 64（包含上界）。

`alpha`/`beta` 可按 `aclsparseSetPointerMode` 使用 Host 或 Device pointer；Device mode
下 Host 不解引用标量。`beta == 0` 时 kernel 不读取旧 `y`，`beta != 0` 时按
`alpha * op(A) * x + beta * y` 写回。workspace 必须是调用方分配的 Device 内存并满足
64 字节对齐；异步执行结束前不得释放或修改 workspace、descriptor、CSR、scalar、x 或 y。
接口不携带实际 workspace size，调用方必须使用 `GetBufferSize` 的结果，欠配空间不保证能
被接口可靠检测。

不支持 COO/CSC/SELL、I64、批量/广播、FP64/complex128 或未声明
dtype/computeType/algorithm；这些组合返回
`ACL_SPARSE_STATUS_NOT_SUPPORTED`（参数结构错误返回 `ACL_SPARSE_STATUS_INVALID_VALUE`）。

SpMV 的预处理记录使用独立的 `activeSpmvBuffer`。同一 descriptor/workspace 在
SpMV 与 SpMM/SDDMM 间复用时会使旧布局失效；CSR stride 或向量 stride 改变后不复用旧 plan。
预处理生成的行偏移始终连续，其他输入及输出仍按各自物理 stride 访问。workspace 头部为
192 字节，调用方应重新调用 `GetBufferSize` 查询，不能硬编码旧大小。

## Python/ATen 适配要求

Python 入口为 `torch.mv(mat, vec)`，内部映射 `aten::mv` 的 NPU Dispatcher，仅接受二维
CSR sparse matrix 和一维 dense vector。桥接层必须完成 device/layout/shape/stride/dtype/
index/alias 校验、SpMat/DnVec descriptor 和 NPU output Tensor 构造，固定以 `alpha=1`、
`beta=0` 调用底层标准 SpMV；不允许 CPU fallback。目标版本为 PyTorch 2.7+ 和
torch_npu 26.0+，异步 stream、allocator、异常和输出 dtype 语义需与目标版本一致。

桥接目标是可选构建的 `ops_sparse_torch`（见根目录 `BUILD_TORCH_ADAPTER`），并注册：

- `torch.ops.ops_sparse.spmv(mat, vec)`，以及 `aten::mv` 的
  `SparseCsrPrivateUse1` 路径，用于真实 CSR `torch.mv` 调用；不覆盖通用
  `PrivateUse1` 稠密 `mv` 注册，避免干扰 torch_npu 既有 dense 实现；
- `torch.ops.ops_sparse_test.spmv_npu(row_ptr, col_ind, values, x, y, alpha, beta, trans, base)`，
  供任务包 ATK、性能和内存脚本使用。该入口原地更新 `y` 并返回同一 Tensor。

构建示例（需先加载 CANN、PyTorch 和 torch_npu 环境）：

```bash
cmake -S . -B build -DSOC_VERSION=ascend950 -DBUILD_TORCH_ADAPTER=ON
cmake --build build --target ops_sparse ops_sparse_torch -j
OPS_SPARSE_TORCH_LIBRARY=build/torch_npu/libops_sparse_torch.so \
  python3 torch_npu/test_spmv_npu.py
```

桥接使用当前 NPU stream，并调用 `NPUCachingAllocator::recordStream` 保证临时 workspace
在异步 Kernel 完成前不会被缓存分配器复用；未安装 torch_npu、设备不是 NPU 或 dispatcher
未命中时直接报错，不提供 CPU reference fallback。

## A5 测试和验收

A5 C++ GTest/CSV 测试放在 `test/spmv/arch35/`，通过
`ops_sparse_add_gtest_tests(spmv ${OPS_SPARSE} EIGEN)` 接入。现有测试包含三阶段调用、workspace
复用、stride、base、转置、延迟绑定、INT8→INT32 正负溢出以及 FP16 舍入等场景。
现有仓库内 GTest 不能视为完整的九组 dtype 覆盖：BF16、输入只读/alias 断言和
ALG2 逐比特确定性尚缺少对应回归用例。算子能力矩阵不等同于测试覆盖矩阵。专项
`SpMV950/test_cases/` 还必须执行 200 条 ATK 精度用例、固定 P-01/P-02/P-03 GPU/NPU 性能
用例及内存对比；报告预热至少 10 次、采样至少 30 次，记录 median/P90、workspace、
峰值内存和 NPU Dispatch/Profiler 证据。无 CANN/NPU 环境时只可做静态检查，不能宣称验收通过。
