# SPMV 算子

## 概述

ops-sparse 仓库中的 SPMV (Sparse Matrix-Vector Multiplication) 算子实现了稀疏矩阵与稠密向量的乘法运算，支持非转置和转置两种模式：

$$
\begin{aligned}
\text{非转置:}\quad y &= \alpha \cdot A \cdot x + \beta \cdot y \\
\text{转置:}\quad y &= \alpha \cdot A^T \cdot x + \beta \cdot y
\end{aligned}
$$

其中 A 为 M×N 稀疏矩阵，x、y 为稠密向量，alpha、beta 为标量。算子基于 aclsparse 统一接口。

## 支持的 AI 处理器

| 产品                                        | 是否支持 |
| ------------------------------------------- | :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |
| Ascend 950PR / Ascend 950DT                |    √    |

## 转置 finalize 整数边界回归（无需 NPU）

在仓库根目录运行：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I test/spmv/host/stubs \
    test/spmv/host/spmv_trans_finalize_boundary_test.cpp \
    -o /tmp/spmv_trans_finalize_boundary_test
/tmp/spmv_trans_finalize_boundary_test
```

测试通过 host stub 执行实际 `SpmvTransFinalize` 模板中的循环，不分配显存。
覆盖两种 16 位输出类型、`beta=0/1`、空输出、256 元素分块边界、
`UINT32_MAX` 及相邻长度、多核分块和乘法提升边界，检查偏移、尾块长度和退出条件。
stub 不执行数值运算、DMA 或同步，不能替代设备精度测试。

## 目录结构介绍

### 公共测试文件

```
test/spmv/
├── CMakeLists.txt                    // 调用 ops_sparse_add_test(spmv ${OPS_SPARSE})
└── README.md                         // 说明文档
```

### arch22（Atlas A2/A3）

```
sparse/spmv/arch22/
├── kernels/                           // 各类型独立编译单元（每 TU 一个 kernel）
│   ├── spmv_kernel.h                 // 模板类 + DEFINE 宏
│   ├── spmv_kernel_f32.cpp           // <float, float, float>
│   ├── spmv_kernel_f32_half_half.cpp // <float, half, half>
│   ├── ...
├── spmv_host.cpp                     // Host 侧 API 实现
├── spmv_tiling_data.h                // Tiling 数据结构
└── spmv.h                            // 算子内部头文件

test/spmv/arch22/
└── spmv_test.cpp                     // 910B 算子调用样例
```

### arch35（Ascend 950PR/950DT）

```
sparse/spmv/arch35/
├── spmv.h                            // 950 workspace / tiling 数据结构
├── spmv_host.cpp                     // 950 三阶段 Host API 与参数校验
└── spmv_kernel.cpp                   // 950 CSR-to-CSC 预处理与 SIMT 计算内核

test/spmv/arch35/
├── spmv_test.cpp                     // Ascend 950PR/950DT 三阶段接口样例
└── spmv_test.csv                     // Ascend 950PR/950DT 的 200 条固定测试用例
```

## 算子描述

### 功能

SPMV 算子实现了 CSR 格式稀疏矩阵与稠密向量的乘法运算，对应的数学表达式为：

$$
y = \alpha \cdot op(A) \cdot x + \beta \cdot y
$$

其中 `op(A)` 可为 A（非转置）或 Aᵀ（转置）。

### 支持的类型组合

算子支持多种输入精度、计算精度与输出精度的组合：

| 输入 A、X 类型 | 计算类型 (computeType) | 输出 Y 类型 | 文件名中的模板参数                  |
| :------------: | :--------------------: | :---------: | :---------------------------------- |
|    float32    |        float32        |   float32   | `<float>`                         |
|      int8     |         int32         |    int32    | `<int32_t, int8_t, int32_t>`      |
|      int8     |        float32        |   float32   | `<float, int8_t, float>`          |
|    float16    |        float32        |   float32   | `<float, half, float>`            |
|    float16    |        float32        |   float16   | `<float, half, half>`             |
|    bfloat16    |        float32        |   float32   | `<float, bfloat16_t, float>`      |
|    bfloat16    |        float32        |  bfloat16  | `<float, bfloat16_t, bfloat16_t>` |

> **注意**：`alpha`/`beta` 的类型必须与 `computeType` 一致：`computeType=float32` 时传 `float` 指针，`computeType=int32` 时传 `int32_t` 指针，否则会导致类型双关错误。

### 存储格式

本实现采用 CSR (Compressed Sparse Row) 格式存储稀疏矩阵，该格式由三个数组组成：

- `csrRowPtr`：行偏移数组，存储每行第一个非零元素在 `csrVal` 和 `csrColInd` 数组中的位置
- `csrColInd`：列索引数组，存储非零元素的列索引
- `csrVal`：值数组，存储非零元素的值

### 实现原理

1. **架构隔离**：arch22 保留独立模板 kernel；arch35 使用 `spmv_host.cpp` 和 `spmv_kernel.cpp` 的统一三阶段接口，不改动已有 arch22 路径
2. **950 非转置路径**：每个输出行由固定的 4-lane SIMT 小组处理，行内使用四条独立 FMA 累加链并做固定顺序的两级归约
3. **950 转置路径**：预处理阶段稳定生成 CSR-to-CSC 索引，每个输出列仍由单一小组按固定顺序计算，不使用浮点原子加
4. **确定性**：相同输入在相同硬件和软件环境中连续执行 20 次，输出要求逐字节一致
5. **混合精度**：float16、bfloat16 和 int8 输入按接口约定提升至 float32 或 int32 计算，再转换到目标输出类型
6. **结果验证**：浮点 CPU golden 使用 double 累加后转换到目标输出类型，整数 golden 使用 Host 端 `__int128` 精确乘加后按 int32 饱和规则输出，避免参考结果在饱和前发生 int64 溢出；该类型不用于 Device 内核

### 算子规格参数说明

| 参数名    | 输入/输出/属性 | 描述                                           | 数据类型                          | 数据格式 |
| --------- | -------------- | ---------------------------------------------- | --------------------------------- | -------- |
| csrRowPtr | 输入           | 矩阵中每一行的第一个非零元素在 csrVal 中的位置 | int32_t                           | ND       |
| csrColInd | 输入           | csrVal 中每个非零元素的列索引                  | int32_t                           | ND       |
| csrVal    | 输入           | 稀疏矩阵中所有非零元素的值                     | float / int8_t / half / bfloat16  | ND       |
| xVec      | 输入           | 被乘稠密向量                                   | 与 csrVal 一致                    | ND       |
| yVec      | 输入/输出      | 结果向量（也作为 beta*y 的输入）               | float / int32_t / half / bfloat16 | ND       |
| alpha     | 输入           | 标量系数 alpha                                 | 与 computeType 一致               | scalar   |
| beta      | 输入           | 标量系数 beta                                  | 与 computeType 一致               | scalar   |
| opA       | 属性           | 矩阵操作类型（NON_TRANSPOSE / TRANSPOSE）      | enum                              | —       |

### 约束说明

- **矩阵格式**：当前仅支持 CSR 格式；CSC / COO 等格式在 SpMV / SpMM 路径上会返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`（`aclsparseCreateCsc` 亦暂未实现）
- **索引类型**：当前仅支持 `ACL_SPARSE_INDEX_32I`；`ACL_SPARSE_INDEX_64I` 暂未支持
- **算法类型**：arch35（Ascend 950）支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`、`ACL_SPARSE_SPMV_CSR_ALG1`、`ACL_SPARSE_SPMV_CSR_ALG2`，当前三个枚举共享同一正确性路径；arch22（Atlas A2/A3）支持 `ACL_SPARSE_SPMV_ALG_DEFAULT` 与 `ACL_SPARSE_SPMV_CSR_ALG_CUBE`（仅非转置，matA/vecX 为 fp32 或 fp16，输出 fp32）。传入所在架构不支持的算法会提示错误并返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`
- **转置支持**：支持非转置和转置，暂不支持共轭转置
- **非连续 Tensor**：CSR rowOffsets、colInd、values 通过 `aclsparseCsrSetStrides` 设置元素步长，x/y 通过
  `aclsparseDnVecSetStride` 设置元素步长；旧接口默认 stride=1，仅支持正步长，vecY 间隙元素保持不变
- **空输入**：支持 M=0、N=0 和 NNZ=0；输出长度为 0 时成功返回，NNZ=0 时仅计算 `beta * y`
- **950 workspace**：仅转置且 NNZ>0 时需要 workspace，大小必须由 `aclsparseSpMVGetBufferSize` 查询；不存在 arch22 UB 单行长度限制

### 测试实现

- **测试流程** (`spmv_test.cpp`)

1. **生成测试数据**：使用固定、可复现种子生成 CSR 矩阵（`GenerateCsr`）和稠密向量 x、y（`GenerateDenseVector`）；`arch35/spmv_test.csv` 中的 200 条固定用例按 [-5, 5] 范围执行，扩展泛化回归按 [-10, 10] 执行
2. **CPU 参考计算**：非转置使用 `SpmvCpu`，转置使用 `SpmvTransCpu`，采用高于算子计算类型的参考累加精度
3. **初始化 ACL 环境**：初始化 ACL 环境，设置设备和创建流
4. **创建设备内存**：使用 `CreateDeviceTensor` 拷贝矩阵和向量数据到 device，用 `unique_ptr` 托管生命周期
5. **创建描述符**：使用 `aclsparseCreateCsr` 和 `aclsparseCreateDnVec` 创建矩阵和向量描述符；非连续用例再通过 stride 设置接口写入五个数组的元素步长（arch35）
6. **执行 SpMV**：先调用 `aclsparseSpMVGetBufferSize` 查询 workspace 大小，按需分配 `externalBuffer`（返回 0 时传入 `nullptr`），再调用可选的 `aclsparseSpMVPreprocess`（仅 arch35 实现）和 `aclsparseSpMV`
7. **结果验证**：将 device 结果回读到 host，与 CPU 参考结果比较。浮点类型使用生态算子混合容差及逐元素最大绝对误差门禁，int32 类型要求逐元素完全匹配；NaN 和同符号 Infinity 按 IEEE 分类一致性验证
8. **清理资源**：销毁描述符，unique_ptr 自动释放设备/主机内存

- **测试覆盖**

| 测试组 | 覆盖内容 |
| ------ | -------- |
| 全量回归 | 非转置/转置、稠密/稀疏/随机形状、alpha/beta 正负值和 pass-through |
| 任务用例 | `arch35/spmv_test.csv` 中 200 个固定用例，覆盖 7 种数据类型组合 |
| 专项验收 | 7 种类型组合、3 种算法枚举、Host/Device 标量和每种类型 20 次确定性执行 |
| 边界验收 | M=0、N=0、NNZ=0、1x1、NaN、+Infinity、-Infinity |
| 非连续验收 | 5 个输入/输入输出 Tensor 使用不同非单位 stride，7 种 dtype 组合分别覆盖转置/非转置，共 14 个实机用例，并校验 vecY 间隙哨兵不被覆盖 |
| 延迟绑定与异步执行 | CSR 和 x/y 未绑定时可查询 workspace；execute 校验实际使用的指针；同 stream 异步 H2D 后直接执行，分别覆盖转置/非转置、显式/跳过 Preprocess |
| 整数溢出回归 | 16 种乘法溢出、加法溢出、极值、抵消及边界场景，分别覆盖转置/非转置、连续/stride、Host/Device 标量，共 128 组；逐元素精确匹配并检查输出间隙哨兵 |
| 性能门禁 | 任务书 4 个标准规模，按 95%/99%/97.5%/99.9% 稀疏率对比基准 |
| Cube 算法（arch22） | fp32 11 组固定用例 + 10 条随机参数，fp16 输入 10 组；覆盖稠密/稀疏/单行/宽矩阵、alpha/beta 组合以及 nnz=0 回退向量版 |

### 精度验证方法

- **float / half / bfloat16**：使用混合容差 `atol + rtol * |golden|` 统计匹配率，并同时检查逐元素绝对误差上限 `max(fixed_limit, 32*ULP)`：

  $$
  tolerance_i = atol + rtol \cdot |golden_i|
  $$

  - float32：rtol=2^-10、atol=2^-16、固定绝对误差上限 1e-2、匹配率至少 99%
  - float16：rtol=2^-9、atol=2^-9、固定绝对误差上限 1e-1、匹配率至少 99%
  - bfloat16：rtol=2^-6、atol=2^-6、固定绝对误差上限 1、匹配率至少 99%
  - NaN 与 NaN、同符号 Infinity 与 Infinity 视为分类一致；其他非有限值组合失败
- **int32_t**：采用绝对误差（Absolute Error）和逐元素精确匹配：

  $$
  aError_i = |npu_i - golden_i|
  $$

  - 所有元素必须精确相等才算通过
  - MARE：最大绝对误差
  - MERE：平均绝对误差
- **关键代码片段**

```cpp
// 生成随机 CSR 矩阵和稠密向量
GenerateCsr<T>(M, N, sparsity, csrRowPtr, csrColInd, csrVal, fixedSeed);
GenerateDenseVector<T>(xSize, xVec, rng);
GenerateDenseVector<T>(ySize, yVec, rng);

// 计算 CPU 参考结果
std::vector<OutT> output_cpu;
if (transpose)
    output_cpu = SpmvTransCpu<CompT, ValT, OutT>(csrRowPtr, csrColInd, csrVal, xVec, yVec, M, N, alpha, beta);
else
    output_cpu = SpmvCpu<CompT, ValT, OutT>(csrRowPtr, csrColInd, csrVal, xVec, yVec, alpha, beta);

// 执行 SpMV（alpha/beta 类型必须与 computeType 匹配）
aclsparseOperation_t op = transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;

// 查询并按需申请 workspace（向量版返回 0，externalBuffer 保持 nullptr）
size_t bufferSize = 0;
sparseRet = aclsparseSpMVGetBufferSize(spHandle, op, &alphaTyped,
                                       matDesc, vecXDesc, &betaTyped, vecYDesc,
                                       compDt, ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize);
void *externalBuffer = nullptr;
if (bufferSize > 0)
    aclrtMalloc(&externalBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);

sparseRet = aclsparseSpMV(spHandle, op, &alphaTyped,
                           matDesc, vecXDesc, &betaTyped, vecYDesc,
                           compDt, ACL_SPARSE_SPMV_ALG_DEFAULT,
                           externalBuffer);
```

## 编译运行

在 ops-sparse 仓库根目录下执行如下步骤，编译并执行 SPMV 算子测试。

### 配置环境变量

请根据当前环境上 CANN 开发套件包的安装方式，选择对应配置环境变量的命令。

- 默认路径，root 用户安装 CANN 软件包

  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```
- 默认路径，非 root 用户安装 CANN 软件包

  ```bash
  source $HOME/Ascend/cann/set_env.sh
  ```
- 指定路径 install_path，安装 CANN 软件包

  ```bash
  source ${install_path}/cann/set_env.sh
  ```

### 样例执行

```bash
bash build.sh --ops=spmv --run
```

Ascend 950PR 可在完成构建后分别复现专项验收、任务书 200 条用例、全量回归和性能门禁：

```bash
export LD_LIBRARY_PATH="$PWD/build:$LD_LIBRARY_PATH"
./build/test/spmv/spmv_test --acceptance
./build/test/spmv/spmv_test --strided
./build/test/spmv/spmv_test --integer-overflow
./build/test/spmv/spmv_test --task-cases test/spmv/arch35/spmv_test.csv
./build/test/spmv/spmv_test
./build/test/spmv/spmv_test --perf
```

执行结果如下，说明精度对比成功：

```txt
======== Float Basic Tests (alpha=1.0, beta=0.0) ========
====Test case: row num = 512 col num = 1024 sparsity (zero ratio) = 0.9 alpha = 1 beta = 0====
Verification...
...
====Test case pass!====

======== Int8->Int32 Basic Tests (alpha=1, beta=0) ========
====Test case: row num = 512 col num = 1024 sparsity (zero ratio) = 0.9 alpha = 1 beta = 0====
Verification...
Matched Ratio = 1; Max Absolute Error = 0
====Test case pass!====

======== Int8->Float32 Non-Transpose ========
...
====Test case pass!====
```

## 接口说明

### aclsparseSpMVGetBufferSize

arch35（Ascend 950）：非转置返回 0；转置按矩阵列数和 NNZ 返回确定性预处理 workspace 大小。
查询只验证描述符元数据，不要求 CSR 或 x/y 的数据指针已绑定。

arch22（Atlas A2/A3）：向量版（`ACL_SPARSE_SPMV_ALG_DEFAULT`）返回 0；cube 版
（`ACL_SPARSE_SPMV_CSR_ALG_CUBE`）返回分段求和所需的 workspace 大小，须作为
`aclsparseSpMV` 的 `externalBuffer` 传入。

**函数原型**：

```cpp
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
    size_t *bufferSize);
```

**参数说明**：

| 参数        | 方向 | 描述                                                                          |
| ----------- | :--: | ----------------------------------------------------------------------------- |
| handle      |  IN  | aclsparse 句柄                                                                |
| opA         |  IN  | 矩阵操作类型（`ACL_SPARSE_OP_NON_TRANSPOSE` / `ACL_SPARSE_OP_TRANSPOSE`） |
| alpha       |  IN  | 标量 alpha 指针，类型必须与 computeType 一致                                  |
| matA        |  IN  | 稀疏矩阵描述符                                                                |
| vecX        |  IN  | 输入稠密向量 x 描述符                                                         |
| beta        |  IN  | 标量 beta 指针，类型必须与 computeType 一致                                   |
| vecY        |  IN  | 输出稠密向量 y 描述符                                                         |
| computeType |  IN  | 计算数据类型（`ACL_FLOAT` / `ACL_INT32`）                                 |
| alg         |  IN  | 算法类型，arch35 支持 DEFAULT / CSR_ALG1 / CSR_ALG2，arch22 支持 DEFAULT / CSR_ALG_CUBE |
| bufferSize  | OUT | 所需工作缓冲区大小                                                            |

### aclsparseSpMVPreprocess

Ascend 950 已实现。非转置为成功的空操作；转置在调用 stream 上异步构建稳定的
CSR-to-CSC 索引。同一 sparsity pattern 可复用，rowOffsets/colInd 更新后必须重新预处理。
不在 Host 同步回读或校验 CSR 内容；调用方必须保证 rowOffsets 从 0 到 NNZ 单调非降，colInd 位于
`[0, cols)`，非法内容属于未定义行为。转置预处理只读取 CSR 索引，不要求 values 或 x/y 已绑定。

**函数原型**：

```cpp
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
    void *externalBuffer);
```

### aclsparseSpMV

**函数原型**：

```cpp
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
    void *externalBuffer);
```

**参数说明**：

| 参数           |  方向  | 描述                                                                          |
| -------------- | :----: | ----------------------------------------------------------------------------- |
| handle         |   IN   | aclsparse 句柄                                                                |
| opA            |   IN   | 矩阵操作类型（`ACL_SPARSE_OP_NON_TRANSPOSE` / `ACL_SPARSE_OP_TRANSPOSE`） |
| alpha          |   IN   | 标量 alpha 指针，类型必须与 computeType 一致                                  |
| matA           |   IN   | 稀疏矩阵描述符                                                                |
| vecX           |   IN   | 输入稠密向量 x 描述符                                                         |
| beta           |   IN   | 标量 beta 指针，类型必须与 computeType 一致                                   |
| vecY           | IN/OUT | 稠密向量 y 描述符（y 为被乘向量，结果覆盖写入 y）                             |
| computeType    |   IN   | 计算数据类型（`ACL_FLOAT` / `ACL_INT32`）                                 |
| alg            |   IN   | 算法类型，arch35 支持 DEFAULT / CSR_ALG1 / CSR_ALG2，arch22 支持 DEFAULT / CSR_ALG_CUBE |
| externalBuffer |   IN   | 工作缓冲区（大小由`GetBufferSize` 获取；arch35 转置且 NNZ>0、arch22 cube 算法时必需） |

**返回值**：

| 返回值                                       | 说明                                                                          |
| -------------------------------------------- | ----------------------------------------------------------------------------- |
| `ACL_SPARSE_STATUS_SUCCESS`                | 成功                                                                          |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`      | handle 为空                                                                   |
| `ACL_SPARSE_STATUS_INVALID_VALUE`          | 描述符或必要标量为空、元数据非法、向量过短，或当前阶段使用的数据指针未绑定 |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED`          | 矩阵格式非 CSR、computeType 不支持、算法不支持、或 valType/outType 组合不支持 |
| `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` | 转置且 NNZ>0 时 workspace 为空或不足                                           |
| `ACL_SPARSE_STATUS_INTERNAL_ERROR`         | 公共 GetAivCoreCount 返回 0；不使用固定核数兜底 |
