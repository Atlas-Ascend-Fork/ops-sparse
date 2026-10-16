# Cube SpMM 算子实现

## 概述

ops-sparse 仓库中的 **Cube SpMM** 算子实现了稀疏矩阵与稠密矩阵的乘法运算，专为 Ascend Cube 计算单元优化。与通用 `spmm` 算子不同，Cube SpMM 在预处理阶段将 COO 格式的稀疏矩阵转换为 Cube-BCSR 格式，并通过列凝聚（column condensation）与负载均衡提升 Cube 核上的计算效率。

该算子适用于稀疏矩阵规模较大、稠密矩阵宽度适中的场景，利用 fp16 稀疏矩阵与 fp16 稠密矩阵相乘、fp32 累加的方式获得较高吞吐。

## 产品支持情况

| 产品                                                         |  是否支持 |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 910B 系列</term>                                |    ✓    |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ✗    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ✗    |

> Cube SpMM 当前版本仅在 `arch22`（DAV-2201）平台交付。非 `ascend910b*` 平台编译时会跳过 `cube_spmm_test`。

## 目录结构介绍

测试与源码目录如下：

```txt
├── test/cube_spmm
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── README.md               // 说明文档
│   └── arch22
│       └── cube_spmm_test.cpp  // arch22 算子调用样例与精度测试
```

源码实现位于：

```txt
├── sparse/cube_spmm/arch22
│   ├── cube_spmm_host.cpp      // Host 侧 API 实现、Tiling 写入与 launch 调度
│   ├── cube_spmm_kernel.cpp    // Kernel 侧 Cube 计算实现
│   ├── cube_spmm.h             // 内部头文件与 Tiling 定义
│   ├── cube_spmm_preprocess.cpp // COO -> Cube-BCSR 预处理、列凝聚、负载均衡
│   └── cube_spmm_preprocess.h   // 内部 BCSR 构建辅助函数声明
```

## 算子描述

### 功能

Cube SpMM 算子实现了稀疏矩阵与稠密矩阵的乘法运算。对应的数学表达式为：

$$
C = \alpha \cdot A \cdot B + \beta \cdot C
$$

其中：

- $A$ 为稀疏矩阵，调用方以 COO 格式输入，算子内部转换为 Cube-BCSR 格式。
- $B$ 为稠密矩阵，数据类型为 `ACL_FLOAT16`（fp16），形状为 $K \times N$。
- $C$ 为稠密矩阵，数据类型为 `ACL_FLOAT`（fp32），形状为 $M \times N$。
- `computeType` 固定为 `ACL_FLOAT`。
- $\alpha$、$\beta$ 为标量（当前版本暂不支持非 1/0 缩放，调用时可传入 `nullptr`）。

### 存储格式

- **稀疏矩阵 A（输入）**：COO 格式，由三个设备数组组成：
  - `cooRows`：行索引，`int32`，0-based。
  - `cooCols`：列索引，`int32`，0-based。
  - `cooVals`：非零元素值，`fp16`（以 `uint16_t *` 传入）。
- **稠密矩阵 B / C**：仅支持行主序（`ACL_SPARSE_ORDER_ROW`），通过 `ld` 描述内存布局。
  - `B`：`fp16`，形状为 $K \times N$。
  - `C`：`fp32`，形状为 $M \times N$（实际分配需按 padding 后尺寸）。
  - `ld >= cols` 时必须通过 `ld` 正确计算行间偏移；当前 kernel 已通过 `bLd` / `cLd` 支持该场景。

### 实现原理

1. **参数校验与描述符创建**：在 `cube_spmm_host.cpp` 中校验矩阵维度、数据类型与布局，创建 Cube-BCSR 专用描述符。当前 `blockM` 与 `blockK` 必须固定为 $16$。
2. **稀疏矩阵预处理**（`cube_spmm_preprocess.cpp`）：
   - 将 Device 端 COO 拷贝到 Host。
   - 转换为 CSR，再按 $blockM \times blockK$（$16 \times 16$）分块得到 BCSR。
   - 对每个块行内的列索引做凝聚，减少重复列带来的冗余计算。
   - 按非零块数量在核间做负载均衡，生成 `coreInfo`。
   - 预处理结果（`rwPtr`、`colRef`、`vals`、`coreInfo`）保存在 `matA` 描述符内部。
   - 内部使用 RAII 临时 Device 资源；全部成功后再释放 `matA` 旧资源并提交新资源，避免部分失败或重复调用时泄漏。
3. **稠密矩阵 B padding**（`aclsparseCubeSpmmPadDenseMatrixB`）：
   - 与稀疏 A 的预处理解耦，A 不变而 B 变化时可单独调用。
   - 对 Host 端稠密 B 的 $N$ 维度做 padding，满足 L0B / CopyInB 对齐要求：
     - `blockK * sizeof(fp16)` 必须是 32B 的整数倍。
     - `blockK * N_pad * sizeof(fp16)` 必须是 512B 的整数倍。
     - 当 `blockK = 16`、`sizeof(fp16) = 2` 时，上述约束均满足，且 $N_{pad}$ 为 16 的倍数。
   - padding 完成后一次性 H2D 拷贝到 `*bPadOut`。
4. **Tiling 构建与 Kernel 启动**（`cube_spmm_host.cpp`）：
   - 每次 `aclsparseCubeSpmm` 都按当前 $M$、$N_{pad}$、$K$、`usedCoreNum`、`lastKLength`、`tileM`、`tileN`、`tailM`、`bLd`、`cLd` 在 Host 侧重建 `CubeSpmmTilingData`，并作为 kernel 启动参数按值随 `<<<>>>` 一并下发，无需写入 workspace（`GetBufferSize` 固定返回 0）。
   - Kernel 内部在 $N$ 方向按 `kNChunkSize = 512` 分块，local buffer（B1/B2/CO1/清零 buffer）按 `min(N, 512)` 分配。
   - 调用 `cube_spmm_kernel_launch` 启动 Cube 核函数。
5. **结果验证**：在 `cube_spmm_test.cpp` 中，通过 CPU 参考实现计算 golden 真值，只比较实际 $M \times N$ 区域，按 `precision.md` 的 FLOAT16 标准判定精度。

### 算子规格

- 参数说明：

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">CubeSpmm</td></tr>
  <tr>
  <tr><td rowspan="7" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A (COO)</td><td align="center">rows × cols</td><td align="center">fp16</td><td align="center">COO (int32 row/col)</td></tr>
  <tr><td align="center">B</td><td align="center">cols × N</td><td align="center">fp16</td><td align="center">行主序</td></tr>
  <tr><td align="center">C</td><td align="center">rows × N</td><td align="center">fp32</td><td align="center">行主序</td></tr>
  <tr><td align="center">alpha</td><td align="center">1</td><td align="center">-</td><td align="center">scalar（当前仅支持 nullptr / 1）</td></tr>
  <tr><td align="center">beta</td><td align="center">1</td><td align="center">-</td><td align="center">scalar（当前仅支持 nullptr / 0）</td></tr>
  <tr><td align="center">opA / opB</td><td align="center">-</td><td align="center">-</td><td align="center">仅支持 NON_TRANSPOSE</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">C</td><td align="center">rows × N</td><td align="center">fp32</td><td align="center">行主序</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="6" align="center">cube_spmm_kernel_launch</td></tr>
  </table>

- 支持的数据类型组合：

  | A | B | C | computeType |
  |---|---|---|-------------|
  | fp16 (COO) | fp16 | fp32 | fp32 |

- 约束限制（详见 [`sparse/cube_spmm/README.md`](../../sparse/cube_spmm/README.md)）：
  - 稀疏矩阵 A 的输入格式为 COO，索引类型仅支持 `int32`，且必须为 0-based。
  - `opA` 与 `opB` 当前仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`。
  - 稠密矩阵 B、C 仅支持行主序（`ACL_SPARSE_ORDER_ROW`）。
  - `aclsparseCreateCubeSpmmMat` 中 `blockM`、`blockK` 必须固定为 $16$。
  - $M$、$K$、$N$ 均大于 0 且不大于 `INT32_MAX`；$M \times K \le$ `INT64_MAX`。
  - 稠密 B/C 的 `values` 指针不可为 `nullptr`。
  - `numCores` 必须大于 0，且需与运行时的核数规划匹配。
  - 当前版本 `alpha` / `beta` 参数仅作占位，实际按 $\alpha=1$、$\beta=0$ 计算。

### 测试实现

测试流程 (`cube_spmm_test.cpp`)：

1. **初始化**：初始化 ACL 环境，设置设备并创建 stream。
2. **生成测试数据**：
   - 无参数时：在线生成随机 COO 稀疏矩阵与 fp16 稠密矩阵 B，默认 $N=1024$。
   - 有参数时：读取外部 Matrix Market (`.mtx`) 文件，可指定 `num_cores`、`device_id`、`N`。
3. **CPU 参考计算**：使用 CPU 实现计算 fp32 参考结果。
4. **设备内存管理**：分配并拷贝 COO 数据到设备内存。
5. **稀疏矩阵描述符**：调用 `aclsparseCreateCubeSpmmMat` 创建 Cube-BCSR 描述符，`blockM=blockK=16`。
6. **预处理 A**：调用 `aclsparseCubeSpmmPreprocess` 完成 COO -> BCSR 转换，结果写入 `matA`。
7. **padding 稠密 B**：调用 `aclsparseCubeSpmmPadDenseMatrixB`，传入 Host 端原始 B 与维度，获取 $N_{pad}$ 与 `bPadOut`。
8. **创建 padded B/C 描述符**：用 $N_{pad}$、`bPadOut` 与设备端 `dC` 创建 `matB` / `matC`。
9. **查询 workspace**：调用 `aclsparseCubeSpmmGetBufferSize`（当前固定返回 0，无需分配）。
10. **执行 Cube SpMM**：调用 `aclsparseCubeSpmm` 两次（`buffer` 传 `nullptr`），第二次前清零 C，验证重复调用结果一致。
11. **结果验证**：将结果拷贝回主机，提取前 $M \times N$ 区域，与 CPU golden 比较。
12. **清理资源**：释放设备内存并销毁描述符。

关键代码片段：

```cpp
// 1. 创建 Cube-BCSR 描述符
CHECK_ACL_SPARSE(aclsparseCreateCubeSpmmMat(
    &matA, M, K, nnz, 16, 16, numCores));

// 2. 预处理 A：COO -> BCSR
CHECK_ACL_SPARSE(aclsparseCubeSpmmPreprocess(
    handle, nullptr, matA,
    cooRowsDev, cooColsDev, cooValsDev,
    ACL_FLOAT));

// 3. 对 Host 端 B 做 padding 并 H2D
int64_t nPad = 0;
void *bPadOut = nullptr;
CHECK_ACL_SPARSE(aclsparseCubeSpmmPadDenseMatrixB(
    handle, K, N, N, hB.data(),
    ACL_FLOAT16, ACL_SPARSE_ORDER_ROW,
    &nPad, &bPadOut));

// 4. 创建 padding 后的 B/C 描述符
CHECK_ACL_SPARSE(aclsparseCreateDnMat(
    &matB, K, nPad, nPad, bPadOut,
    ACL_FLOAT16, ACL_SPARSE_ORDER_ROW));
CHECK_ACL_SPARSE(aclsparseCreateDnMat(
    &matC, M, nPad, nPad, dC,
    ACL_FLOAT, ACL_SPARSE_ORDER_ROW));

// 5. 查询 workspace（当前实现固定为 0）并执行 SpMM
size_t bufferSize = 0;
CHECK_ACL_SPARSE(aclsparseCubeSpmmGetBufferSize(
    handle, nullptr, matA, matB, nullptr, matC, ACL_FLOAT, &bufferSize));

CHECK_ACL_SPARSE(aclsparseCubeSpmm(
    handle, nullptr, matA, matB, nullptr, matC, ACL_FLOAT, nullptr));
CHECK_ACL(aclrtSynchronizeStream(stream));
```

## 编译运行

在 ops-sparse 仓库根目录下执行如下步骤，编译并执行 Cube SpMM 算子测试。

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

### 快速验证

无参数一键运行，使用内置随机稀疏矩阵：

```bash
bash build.sh --ops=cube_spmm --soc=<arch名称> --run
```

执行结果如下，说明精度对比成功：

```txt
Random mode: M=512 K=512 nnz=2621 N=1024 numCores=20
N_pad=1024
Mean Relative Error: 5.9609e-08 (threshold 0.000976562)
Max Relative Error:  0.00199062 (threshold 0.00976562)
Mismatch count: 0/524288
[Success] test case accuracy is verification passed.
[PASS] cube_spmm_test
```

### 使用外部数据集验证

也支持传入外部 `.mtx` 文件进行验证：

```bash
./build/test/cube_spmm/cube_spmm_test <mtx_file> [num_cores] [device_id] [N]
```

## 接口说明

Cube SpMM 采用调用流程：

```
CreateCubeSpmmMat
    -> CubeSpmmPreprocess(A)
    -> CubeSpmmPadDenseMatrixB(B)
    -> GetBufferSize
    -> CubeSpmm
```

完整 API 说明参见 [`include/cann_ops_sparse.h`](../../include/cann_ops_sparse.h)。

### aclsparseCreateCubeSpmmMat

创建 Cube-BCSR 专用稀疏矩阵描述符。

```cpp
aclsparseStatus_t aclsparseCreateCubeSpmmMat(
    aclsparseCubeSpmmMatDescr_t *descr,
    int64_t rows, int64_t cols, int64_t nnz,
    int64_t blockM, int64_t blockK, int32_t numCores);
```

### aclsparseCubeSpmmPreprocess

负责 COO → Cube-BCSR 转换、列凝聚、负载均衡。仅处理稀疏矩阵 A。

```cpp
aclsparseStatus_t aclsparseCubeSpmmPreprocess(
    aclsparseHandle_t handle,
    const void *alpha, aclsparseCubeSpmmMatDescr_t matA,
    const int32_t *cooRows, const int32_t *cooCols, const uint16_t *cooVals,
    aclDataType computeType);
```

### aclsparseCubeSpmmPadDenseMatrixB

负责稠密矩阵 B 的 padding 对齐与 H2D 拷贝。A 不变而 B 变化时可单独调用。

```cpp
aclsparseStatus_t aclsparseCubeSpmmPadDenseMatrixB(
    aclsparseHandle_t handle,
    int64_t bRows, int64_t bCols, int64_t bLd, const void *bValues,
    aclDataType bType, aclsparseOrder_t bOrder,
    int64_t *nPadOut, void **bPadOut);
```

### aclsparseCubeSpmmGetBufferSize / aclsparseCubeSpmm

标准的 workspace + compute 流程。当前实现 tiling 随 kernel 启动参数下发，不需要 device workspace：`GetBufferSize` 固定返回 0，`buffer` 传 `nullptr`。

```cpp
aclsparseStatus_t aclsparseCubeSpmmGetBufferSize(
    aclsparseHandle_t handle,
    const void *alpha, aclsparseConstCubeSpmmMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    size_t *size);

aclsparseStatus_t aclsparseCubeSpmm(
    aclsparseHandle_t handle,
    const void *alpha, aclsparseConstCubeSpmmMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    void *buffer);
```
