# cube_spmm_1d

完整的产品支持、接口原型及参数/返回值见[算子说明](../../sparse/cube_spmm_1d/README.md)，
公共接口索引见[接口列表](../../docs/zh/api_list.md#aclsparsecubespmm1dcreateplan)。
本页说明构建、测试和调用方法。

## 功能与支持范围

使用 Cube 计算单元执行稀疏矩阵与稠密矩阵乘法：

```text
C = A × B
A: M × K，FP16
B: K × N，FP16
C: M × N，FP32
```

支持 d=16 的垂直 vector-CSR，也提供标准 CSR 的内存准备接口。
实现位于 `sparse/cube_spmm_1d/arch22/`，已在 Ascend910B3、CANN 8.5 环境验证。
源码支持范围为 `arch22`；`arch35` 构建不编译或导出本接口族。
不支持转置、alpha/beta 缩放、批处理或任意 leading dimension；A/B 必须为有限数。

## 环境与构建

需要 CANN 开发套件（含 `ccec` 或 `bisheng`、`ld.lld` 和 ASC CMake 配置）、CMake >= 3.16、
C++17 编译器及 Python3。构建测试还需要 GTest 和 Eigen3 >= 3.3。
非系统目录安装的依赖可通过 `CMAKE_PREFIX_PATH` 指定安装前缀。
以下命令均在仓库根目录执行。

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
bash build.sh --ops=cube_spmm_1d --soc=ascend910b3
```

如果 CMake 未自动找到 `ASCConfig.cmake`，设置 `ASC_DIR` 为该文件所在目录后重试。
例如，部分 CANN 安装布局对应：

```bash
export ASC_DIR="$ASCEND_HOME_PATH/$(uname -m)-linux/lib64/cmake"
```

仅构建库、不构建测试时，无需 GTest/Eigen：

```bash
cmake -S . -B build -DOP_LIST=cube_spmm_1d \
    -DSOC_VERSION=ascend910b3 -DBUILD_TEST=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
cmake --install build
```

安装产物：

```text
build_out/
├── include/cann_ops_sparse.h
└── lib64/
    ├── libops_sparse.so
    └── cube_spmm_1d_kernels/*.o
```

运行时会根据动态库位置加载 `cube_spmm_1d_kernels/` 中的设备文件，
因此必须一起部署动态库和该目录，不能只复制 `.so`。

## 运行测试

先通过 `npu-smi info` 确认设备健康且可用，再选择设备编号：

```bash
bash build.sh --ops=cube_spmm_1d --soc=ascend910b3 --device=1 --run
```

测试使用仓库公共框架、GTest、CSV 用例和 Eigen FP64 参考，覆盖原生/CSR 输入、
配置选择、空行、补齐、重复执行、输入校验、双 stream 和缓冲区边界。
成功时输出 `[  PASSED  ]`，且仓库测试汇总中 `Failed: 0`。

额外的 Host 故障注入目标不初始化设备，以链接包装替代运行时调用；它直接编译生产
Host 源文件，验证二进制查找失败与重试、注册失败回滚、回滚失败重试、各分配点失败、函数注册成功后零分配、
以及计划部分释放失败后的重试和上下文检查：

```bash
cmake --build build --target cube_spmm_1d_fault_test -j4
./build/test/cube_spmm_1d/cube_spmm_1d_fault_test
```

CSV 中 `NonResidentNative` / `NonResidentCsr` 每个行块含3073个活跃列，超过64 KiB L0B
容纳完整 A 行块的上限；`AdaptiveNMajorLargeB` 使用192 MiB逻辑 B、N=8192，
在20核Ascend910B3上断言自适应选择key 57，而非通过显式配置强制该路径。
这些测试使用原生/CSR公共接口、非零 B、FP64参考以及C/workspace尾部guard。

## 接口与调用流程

接口声明位于 `include/cann_ops_sparse.h`，链接 `ops_sparse` 和 CANN 运行库。

| 接口 | 功能 |
| --- | --- |
| `aclsparseCubeSpmm1dCreatePlan` | 从 Host vector-CSR 数组准备并上传 A |
| `aclsparseCubeSpmm1dCreateCsrPlan` | 从 Host 标准 CSR 数组转换并上传 A |
| `aclsparseCubeSpmm1dGetPlanInfo` | 查询逻辑尺寸、补齐尺寸、配置及内存字节数 |
| `aclsparseCubeSpmm1dExecute` | 在 handle 的 stream 上清零 C 并异步提交计算 |
| `aclsparseCubeSpmm1dDestroyPlan` | 释放计划及其持有的 A；不等待 stream |

DestroyPlan 若返回设备释放失败，须保留原句柄重试销毁；已释放缓冲区会跳过。
开始销毁后不可再查询或执行该计划，完整约束见[计划销毁说明](../../sparse/cube_spmm_1d/README.md#aclsparsecubespmm1ddestroyplan)。

调用顺序：

1. 初始化 ACL，设置设备/上下文，创建 stream 和 `aclsparse` handle，并绑定 stream。
2. 将初值为 `NULL` 的 plan 交给其中一个准备接口。
3. 查询 PlanInfo，按 `bBytes`、`cBytes`、`workspaceBytes` 分配设备缓冲区。
4. 按 `paddedN` 填充 B，并将尾列补零；调用 Execute。
5. 在调用者 stream 上继续提交后续操作，或同步 stream 后读取结果。
6. 所有执行完成后，先销毁计划，再释放相关资源和 ACL 环境。

每个 API/ACL 调用都应检查返回值；即使后续提交失败，也应完成已排队工作后再释放资源。
准备阶段可以分配设备内存和执行 H2D，Execute 不分配设备内存或进行张量 H2D/D2H。

### A 的存储与准备

两种准备接口均接收 **Host 数组**，并在调用期间复制和上传 A。
准备返回后可以释放 Host 数组；修改原数组不会更新计划，要更新 A 必须重新准备。

| 格式 | 行指针 | 列索引 | 数值 |
| --- | --- | --- | --- |
| vector-CSR | `int32_t[M/16+1]` | `int32_t[P]` | `uint16_t[16*P]`，FP16 位模式 |
| 标准 CSR | `int64_t[M+1]` | `int64_t[nnz]` | `uint16_t[nnz]`，FP16 位模式 |

所有索引从 0 开始。行指针首项为 0，单调非递减，末项等于 P 或 nnz。
传入的数组长度必须与实际容量和行指针端点一致。空 A 允许列索引和数值指针为 `NULL`，
但行指针仍然必需。

**vector-CSR：** M 必须为 16 的倍数；每个行块的列索引严格递增且位于 `[0,K)`。
一个非零向量对应同一列中的连续 16 行，16 个值连续存放；P 是非零向量数。

**标准 CSR：** 支持任意正 M、无序列索引、重复项、显式零和空行。
转换先稳定按列排序，对重复项按输入顺序用 FP64 求和，再以 RNE 规则量化为 FP16，
去除零值；随后每 16 行取列并集、补零，并将 M 补齐到 16 的倍数。
重复项求和后重新量化可能与逐项 CSR 乘加产生不同舍入结果。
非有限值或合并后绝对值大于 65504 的数值返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。

M/N/K 必须为正，M 不超过 `INT32_MAX-15`，转换后 P 不超过 `INT32_MAX/16`。
补齐、字节数及 workspace 必须满足接口的整数范围和内存限制。
一般 CSR 的列并集可能增加存储量，可通过 PlanInfo 的 `persistentABytes` 查询准备后的 A 存储量。

### B、C 与 workspace

| 缓冲区 | 类型与布局 | 容量 |
| --- | --- | --- |
| B | FP16，行主序 `[K,paddedN]` | `info.bBytes` |
| C | FP32，列主序 `[paddedM,paddedN]` | `info.cBytes` |
| workspace | 设备工作区 | `info.workspaceBytes` |

- B 的逻辑元素位于 `B[i*paddedN+j]`；`j >= N` 的尾列必须为零。
  B 每次执行均可更换，调用者须保证所有输入值有限。
- C 的逻辑元素位于 `C[j*paddedM+i]`，只读取原始 `M×N` 范围。
  Execute 会先清零完整物理 C，不保留 C 原值；空行和补齐区输出为零。
- B/C/workspace 均需 32 字节对齐，属于计划创建时的设备/上下文，且互不重叠。
- Execute 的容量参数单位为字节，小于查询值返回 `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`。

Execute 为异步接口。计划、B、C 和 workspace 必须存活到相关 stream 的工作完成。
同一不可变计划可以复用；并行执行须使用不同的 C/workspace，并自行保证 stream 顺序。
DestroyPlan 不隐式等待，须在计划创建时的上下文中调用；销毁空计划返回成功。

### 配置选项

`options == NULL` 使用自适应配置。需要显式选项时，设置：

| 字段 | 取值 |
| --- | --- |
| `structSize` | `sizeof(aclsparseCubeSpmm1dOptions)` |
| `tilingKey` | `-1` 表示自适应；非负值显式选择已编译的配置 |
| `reserved` | 必须为 0 |
| `workspaceLimitBytes` | 0 表示无额外限制；非零值为 workspace 字节上限 |

显式 key 用于调试和复现，须来自算子已编译的配置集合。
总是使用 PlanInfo 返回的 `paddedN` 和 `workspaceBytes` 分配缓冲区，不在调用方推断替代值。

## C 调用示例

完整的初始化、CSR 准备、执行、校验和资源清理见
[`cube_spmm_1d_csr.c`](../../examples/cube_spmm_1d/cube_spmm_1d_csr.c)。
完成库安装后，可在仓库根目录编译运行：

```bash
gcc -std=c11 examples/cube_spmm_1d/cube_spmm_1d_csr.c \
    -Ibuild_out/include -I"$ASCEND_HOME_PATH/include" \
    -Lbuild_out/lib64 -L"$ASCEND_HOME_PATH/lib64" \
    -lops_sparse -lascendcl -o /tmp/cube_spmm_1d_csr

ASCEND_RT_VISIBLE_DEVICES=1 \
LD_LIBRARY_PATH="$PWD/build_out/lib64:$LD_LIBRARY_PATH" \
    /tmp/cube_spmm_1d_csr
```

示例成功时输出 `cube_spmm_1d C CSR example: PASS`。
如果运行环境尚未配置驱动依赖库，请按 CANN 安装说明补充运行库搜索路径。

## 性能测量

测量完整 Execute 耗时时，应在一致的 stream 同步边界计时，包含输出清零、提交和计算。
单独分析设备计算时，可使用 msprof 的 `MIX_AIC` 任务时间，不叠加 `MIX_AIV` 时间；
该任务不包含 Host 准备、A 转换/上传和任务外的 C 清零。
异步提交耗时不能作为已完成计算的耗时，纯设备任务耗时也不能代替完整接口耗时。
