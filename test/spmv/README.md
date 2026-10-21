# SpMV 测试

## 平台与入口

SpMV 计算 `y = alpha * op(A) * x + beta * y`。平台能力以
[算子说明](../../sparse/spmv/README.md)为准；支持某种类型不代表仓库测试已完整覆盖。

| 平台 | 测试源码 | 构建入口 |
| --- | --- | --- |
| Atlas A2/A3（arch22） | `arch22/spmv_test.cpp` | `ops_sparse_add_test`，独立测试程序 |
| Ascend 950PR/950DT（arch35） | `arch35/spmv_test.cpp`、`arch35/spmv_test.csv` | `ops_sparse_add_gtest_tests(spmv ${OPS_SPARSE} EIGEN)` |

A2/A3 支持 DEFAULT 与 CSR_ALG_CUBE；CUBE 仅支持非转置、FP32/FP16 输入和 FP32 输出。
以下用例名、CSV 和 GTest 命令均针对 A5，不能直接用于 arch22 程序。

## A2/A3 转置 finalize 整数边界回归（无需 NPU）

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

## A5 实现与调用约定

实现位于 `sparse/spmv/arch35/`：

- `spmv.h`、`spmv_tiling_data.h` 定义 workspace 布局、tiling 和 active plan。
- `spmv_host.cpp` 实现 GetBufferSize、Preprocess、Execute 和参数校验。
- `spmv_kernel.cpp` 实现行偏移归一化以及非转置、转置计算。

A5 支持 CSR、I32 索引、base 0/1，支持非转置 N、转置 T 和共轭转置 H。
复数 H 对矩阵元素取共轭；实数 H 与 T 等价。非转置要求
`x.size == A.cols`、`y.size == A.rows`，转置/共轭转置要求
`x.size == A.rows`、`y.size == A.cols`。

A5 支持的类型组合如下，alpha/beta 类型必须与 computeType 一致：

| A / x | computeType | y |
| --- | --- | --- |
| INT8 | INT32 | INT32 |
| INT8 | FLOAT | FLOAT |
| FLOAT16 | FLOAT | FLOAT16 或 FLOAT |
| BF16 | FLOAT | BF16 或 FLOAT |
| FLOAT | FLOAT | FLOAT |
| COMPLEX64 | COMPLEX64 | COMPLEX64 |
| FLOAT | COMPLEX64 | COMPLEX64 |

CSR rowOffsets、colInd、values 及 x/y 均支持正的元素 stride，默认值为 1。
调用方负责保证 CSR 内容合法：rowOffsets 从 base 到 nnz+base 单调非降，
colInd 位于 `[base, cols+base)`。重复坐标可用，不要求 nnz <= rows*cols。
rows 或 cols 为 0 时必须 nnz=0；nnz=0 时执行 beta*y。

### Workspace 与预处理

1. GetBufferSize 校验元数据，不要求 CSR 和 x/y 数据已绑定。A5 的 N/T/H
   均返回非零 workspace 大小，调用方必须按查询值分配足量、64 字节对齐的设备内存。
   接口不接收实际分配大小，不能保证检测出欠配。
2. Preprocess 可选。它生成归一化的连续 rowOffsets，并记录 active plan，
   不构建按列重排的索引。rows>0 时要求 rowOffsets 已绑定；values、colInd 和 x/y
   可后续绑定。即使跳过预处理，Execute 仍需要查询所得的 workspace。
3. Execute 检查实际使用的数据指针；可复用有效 plan，否则读取原始 CSR。
   更新指针或属性可能使 plan 失效。原地修改 rowOffsets/colInd 内容后，必须重新
   Preprocess，因为 Host 不读取设备内容，无法仅凭不变的地址识别修改。
   仅原地修改 values/x/y 不要求重新预处理。
4. 三阶段调用使用 handle 的 stream。调用方须保证异步操作期间相关内存有效，
   在读取结果、复用或释放内存前同步。beta=0 时不读取旧 y。

workspace 包含 192 字节头部、归一化行偏移，转置另有累加区，各区域按 64 字节对齐。
A2/A3 DEFAULT 在转置且 FP16/BF16 输出时查询返回每列 4 字节的 FP32 累加空间，
其他 DEFAULT 路径返回 0；CUBE 按实现查询分段求和空间。该约定与 A5 不同。

### 算法与确定性

DEFAULT 映射 CSR_ALG1。非转置使用按行的计算路径。
ALG1/DEFAULT 转置对 `0 < rows <= kSpmvTransposeGatherRowsMax` 的矩阵检查索引是否有序，有序时按列 gather，
无序时回退无原子路径；较大矩阵使用 workspace 原子累加。
浮点/复数 ALG1/DEFAULT 不承诺逐比特确定性。

FP32 输入/输出/计算的非转置路径在 `rows <= kSpmvFloatSmallRowsMax` 时使用小矩阵实现。
两项包含上界的阈值均以 [共用头文件](../../sparse/spmv/arch35/spmv_tiling_data.h) 为准。

ALG2 转置按固定 row/entry 顺序归约。当前仓库测试没有对其执行逐字节重复比较，
不能将算法设计描述当作确定性回归已通过的证据。
INT8→INT32 结果最终饱和到 INT32 范围，缩放时保留抵消并检查中间溢出。

## A5 仓库内测试覆盖

`arch35/spmv_test.csv` 当前有 **6 条数据记录**（不含表头），全部用于 FP32 的
`SpmvA5Cases/SpmvArch35Test.ThreeStageAndExecute/*` 参数化测试。
字段为 case_name、op、index_base、alg、pointer_mode、alpha、beta。
数据由代码中的固定 2×3 CSR fixture 提供，CSV 不包含矩阵尺寸或 dtype 字段。
每条用例执行三阶段调用及一次 workspace 复用；第二次执行的 golden 使用第一次的输出作为旧 y，
这不是对相同输入做逐字节确定性比较。

| 当前用例 | 验证内容 |
| --- | --- |
| 6 条 CSV 参数化用例 | N/T、base 0/1、DEFAULT/ALG1/ALG2、Host/Device 标量，FP32 结果使用 1e-5 绝对容差 |
| RejectUnsupportedFormatAndAlgorithm | 不支持的格式与算法 |
| StridedStoragePreservesPadding | FP32 的不同步长、输出间隙哨兵及不同计算路径 |
| SeparateWorkspaceCacheAndStrideIdentity | workspace 归属与 stride 对 plan 命中的影响 |
| ReusesWorkspaceAcrossSpmvAndSpmm | SpMV/SpMM 复用缓存，仅同时构建 spmm 时启用 |
| ZeroNnzAndMisalignedWorkspace | 零 nnz 与错误 workspace 对齐 |
| AllowsRepeatedCsrCoordinates | 重复坐标 |
| RejectsNonzeroNnzWithZeroDimensions | N/T 的零行/零列且非零 nnz，三阶段均拒绝 |
| QueriesWorkspaceBeforeBindingStorage | 查询时未绑定数据、预处理后再绑定 values/x/y |
| SaturatesPositiveAndNegativeInt32Overflow | INT8→INT32 正负饱和，N/T/H、DEFAULT/ALG2、2/4096 行 |
| RoundsHalfwayResultsToEven | FP16 中点舍入 |
| ComplexConjugateTranspose | complex64 共轭转置 |

上述测试覆盖 FP32、INT8→INT32、FP16 和 complex64 场景，并非完整九组类型组合回归。
BF16、输入只读/alias 断言和 ALG2 逐比特确定性仍缺少仓库内对应回归用例。
各用例的精度要求由源码中的 EXPECT_NEAR、EXPECT_FLOAT_EQ、EXPECT_EQ 等断言决定；
这里不额外承诺统一的匹配率、ULP 或 NaN/Infinity 门禁。

任务书附带的官方精度、性能和内存对比属于独立验收流程，不是此 CSV 或 GTest 的选项。
应使用任务书对应版本的官方工具、case 文件及实际原始日志；本 README 不提供已验收通过的证明。

## A5 编译与运行

先加载实际安装路径的 CANN 环境，例如：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

在仓库根目录构建（build.sh 会重建 build 目录）：

```bash
bash build.sh --soc=ascend950 --ops=spmv
```

若需要 SpMV/SpMM 交叉复用用例，将上面的算子列表改为 `--ops=spmv,spmm`。
编译需要目标 CANN 工具链及 GTest 等依赖，执行需要对应 NPU。

CMake 会把 CSV 复制到测试二进制目录。测试按相对路径读取 `spmv_test.csv`，
因此手动执行时必须进入该目录：

```bash
export LD_LIBRARY_PATH="$PWD/build:$LD_LIBRARY_PATH"
cd build/test/spmv
./spmv_test --gtest_list_tests
./spmv_test
./spmv_test --gtest_filter='SpmvA5Cases/SpmvArch35Test.ThreeStageAndExecute/*'
./spmv_test --gtest_filter='SpmvArch35ExceptionTest.StridedStoragePreservesPadding'
./spmv_test --gtest_filter='SpmvArch35ExceptionTest.SaturatesPositiveAndNegativeInt32Overflow'
./spmv_test --gtest_output=xml:spmv_results.xml
```

本程序使用 GTest 标准参数，没有独立的专项验收、任务 JSON 或性能模式命令行接口。
先用 list_tests 确认实际注册的用例，再按需使用 filter。用例数量会因条件编译而变化。
GTest 结果只证明本次所执行的仓库用例，不代表官方全量精度或性能门禁已经通过。
