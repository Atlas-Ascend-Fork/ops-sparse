# SpMM 测试与接口说明

本目录包含 `aclsparseSpMM` 的 C++ 功能、精度和异常测试，以及 PyTorch 适配测试。

ops-sparse 仓库中的 SPMM (Sparse Matrix-Dense Matrix Multiplication) 算子实现了稀疏矩阵与稠密矩阵的乘法运算，是高性能稀疏矩阵计算中的核心算子之一。

该算子根据 CSR 结构、数据类型和输出布局选择 SIMT/AIV、Cube 或 RegBase 计算路径，预处理阶段生成各路径所需的 CSR 元数据。

## 产品支持情况

| 产品                                                         |  是否支持 |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ✓    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ✓    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    ✓    |

> Ascend 950PR/950DT 的源码和测试分别位于 `sparse/spmm/arch35/` 与 `test/spmm/arch35/`，仅在 `ascend950*` 目标编译。

A2/A3 源码位于 `sparse/spmm/arch22/`，测试位于 `test/spmm/arch22/`。按目标 SOC 编译对应实现，接口约束见算子说明。

## 目录结构介绍

```txt
sparse/spmm/arch22/
├── kernels/              // A2/A3 AscendC kernel 与 dtype 分发
├── spmm_host.cpp         // Host 侧参数校验、workspace 与 Tiling
├── spmm_csr_mat.cpp      // CSR 预处理、行重排与分桶
├── spmm_csr_mat.h
└── spmm.h                // 公共接口与 Tiling 定义
```

Ascend 950PR/950DT 的实现单独位于 `arch35`，其内部结构如下：

```txt
sparse/spmm/arch35/
├── host/
│   ├── spmm_host.cpp          // Host 侧 API、workspace 管理与选路
│   ├── spmm_csr_mat.cpp       // CSR 矩阵预处理（行重排、分桶）
│   └── spmm_host_tiling.h     // Host 侧 Tiling 构造与序列化
├── kernel/
│   ├── spmm_kernel.cpp        // Kernel 启动与路径分发
│   ├── spmm_kernel_simt.cpp   // SIMT 计算
│   ├── spmm_kernel_cube.cpp   // 实数 Cube 计算
│   ├── spmm_kernel_complex_cube.cpp     // Complex64 Cube 计算
│   ├── spmm_kernel_fp32_regbase.cpp     // FP32 RegBase 计算
│   ├── spmm_kernel_complex_regbase.cpp  // Complex64 RegBase 计算
│   └── spmm_kernel_tiling.h   // Kernel 侧 Tiling 接收与解析
├── spmm_config.h             // 公共常量
└── spmm_tiling_wire.h        // Tiling 传输字段与布局校验
```

## 测试目录

```text
test/spmm/
├── CMakeLists.txt
├── README.md
├── arch22/             // A2/A3 功能、精度和缺陷回归
├── arch35/
│   ├── spmm_test.cpp
│   ├── spmm_task_dtype_test.cpp
│   ├── spmm_cube_path_test.cpp
│   ├── spmm_cube_dtype_test.cpp
│   ├── spmm_fp32_regbase_test.cpp
│   ├── spmm_fp32_degree2_test.cpp
│   ├── spmm_complex64_degree2_test.cpp
│   ├── spmm_workspace_owner_test.cpp
│   └── spmm_transpose_test.cpp
└── python/
    ├── test_torch_extension.py
    ├── test_sparse_addmm_registry.py
    └── test_op_builder_lifecycle.py
```

`arch35` 测试仅在 `ascend950*` 目标编译。以上 C++ 文件共同构建为 `spmm_test`，
复用仓内 `ops_sparse_add_gtest_tests`、公共 GTest main 与 device fixture。
转置回归覆盖 opA=T/H、矩形 CSR、0/1 基索引、workspace 交替复用和原位 values 更新。
基础配置及 packed 数值、精度回归通过 `TEST_F` / `TEST_P` 注册。

## 覆盖范围

数学语义为：

```text
C = alpha * op(A) * op(B) + beta * C
```

- A：二维 CSR，row offsets 和 column indices 均为 `ACL_SPARSE_INDEX_32I`，
  支持 index base 0/1；
- B/C：支持行主序、列主序和合法 leading dimension；
- dtype：FP16、BF16、FP32、Complex64；FP16/BF16 使用 FP32 累加，Complex64
  使用 Complex64 复数乘加；
- opA/opB、CSR_ALG1/2/3 和布局组合遵循
  [SpMM 接口说明](../../sparse/spmm/README.md)；不支持的组合返回错误；
- 覆盖 workspace 查询、可选 preprocess、SpMM 执行、Host/Device 标量、空
  CSR、空行、长尾、padding、非连续输入、输入只读性、beta=0、重复执行和资源
  释放；
- `test/spmm/python/test_torch_extension.py` 负责 Python/ATen 注册、四种
  dtype、广播、非连续 Tensor、输入保护和无 CPU fallback 的端到端回归。
- `test/spmm/python/test_spmm_algorithm.py` 在 CPU 上编译 C++ 适配层并模拟原生接口，
  检查三阶段调用、独立输出和回退路径使用同一 DEFAULT 算法。
- `spmm_workspace_owner_test.cpp` 覆盖同一 workspace 上的 `A1 -> A2 -> A1`
  交替调用、描述符更新、不同 workspace 的生命周期及登记淘汰后的重建，
  并覆盖 FP32/FP16/BF16 同指针改值、改为零和 FP32 补齐窗口边界。
- `spmm_cube_dtype_test.cpp` 覆盖 FP16/BF16 的末尾列窗口与小 K，检查实际分组的
  B 读取范围、数值结果和输入、输出、workspace guard。
- `spmm_complex64_degree2_test.cpp` 覆盖 Complex64 RegBase 数值刷新及 Cube 打包
  的两侧小值、实虚部、抵消、范围检查及部分列宽回退。
- `test_op_builder_lifecycle.py` 验证公共 builder 的模块缓存与退出清理顺序。

A2/A3 测试包含 FP16/BF16/FP32/Complex64、转置、共轭转置、索引基址 0/1、零维输入及确定性算法；对应接口约束见算子说明。

## C++ 调用流程

使用已有公共接口，顺序如下：

1. `aclsparseCreate`、`aclsparseSetStream` 和描述符创建；
2. `aclsparseSpMMGetBufferSize` 查询 workspace；
3. 分配 workspace，可选调用 `aclsparseSpMMPreprocess`；
4. 调用 `aclsparseSpMM` 执行；
5. 同步、校验输出，然后销毁描述符、workspace、stream 和 handle。

接口声明位于 `include/cann_ops_sparse.h`，完整参数约束、标量指针模式、布局、
算法限制和错误行为位于 `sparse/spmm/README.md`。

## 构建与运行

先加载与源码匹配的 CANN 环境，在仓库根目录执行：

```bash
bash build.sh --ops=spmm --soc=ascend950 --run
```

该命令通过 GTest 运行基础用例以及 dtype、Cube、RegBase、workspace 归属和
生命周期测试，任一用例失败时返回非零值。可对生成的 `spmm_test` 使用
`--gtest_list_tests` 查看用例，或使用 `--gtest_filter` 选择回归范围。

A2/A3 使用相同构建入口，并将 `--soc` 设为对应产品。

Python/ATen 适配测试：

```bash
bash build.sh --torch_extension --ops=spmm
python -m pip install --force-reinstall --no-deps build_out/cann_ops_sparse-*.whl
python -m pytest -q test/spmm/python/test_torch_extension.py
```

Python 测试提供功能、精度、注册及公共加载器生命周期覆盖。

```bash
python -m pytest -q test/spmm/python/
```

## 结果判定

C++ 测试使用 CPU Golden 校验输出，通过 GTest 报告各用例结果。Cube 边界用例同时读取
实际分组 metadata，检查补齐后的 B 读取窗口位于真实 K 范围内。输入、输出和
workspace guard 检查与数值检查共同执行。
