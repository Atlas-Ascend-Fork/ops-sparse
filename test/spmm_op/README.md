# SpMMOp 算子测试

## 概述

ops-sparse 仓库中的 SpMMOp 算子测试基于 GTest + CSV 驱动的参数化测试框架，对标 cuSPARSE `cusparseSpMM()` Generic API 语义，覆盖 7 函数 descriptor/plan 生命周期（bufferSize → createDescr → createPlan → [setGlobalUserData] → execute → destroyPlan → destroyDescr）。

对应的数学公式为：

$$
C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
$$

其中 A 为 CSR 格式稀疏矩阵（m×k），B 为稠密矩阵（k×n），C 为稠密矩阵（m×n），alpha/beta 为标量。

## 产品支持情况

| 产品 | 是否支持 |
|------|---------|
| Ascend 950PR/Ascend 950DT | ✓ |

> 非 `ascend950*` 平台编译时会跳过 `spmm_op_test`。

## 目录结构

```
test/spmm_op/
├── CMakeLists.txt                    // 调用 ops_sparse_add_gtest_tests(spmm_op ...)
├── README.md                         // 本文档
├── spmm_op_golden.h                  // FP64 CPU golden 参考实现 + CSR 生成器
├── spmm_op_param.h                   // CSV 参数解析结构体 SpmmOpTestParam
└── arch35/
    ├── spmm_op_test.cpp              // GTest 测试主体（TEST_P + TEST_F）
    ├── spmm_op_test.csv              // CSV 驱动用例数据（158 个用例）
    ├── spmm_op_npu_wrapper.h         // NPU 7 函数生命周期 RAII 封装
    └── spmm_op_perf.cpp              // 独立性能采集程序（60 用例）
```

## 测试框架

- **驱动方式**：GTest TEST_P + CSV 参数化（`test/frame/csv_loader.h`），通过 `GetCasesFromCsv<SpmmOpTestParam>` 加载 CSV 用例；异常/值更新/确定性用例使用 TEST_F 硬编码
- **精度验证**：MIXED_TOLERANCE 模式（`test/frame/verify.h`），dtype 驱动 atol/rtol + per-element max(abs_err) limit = max(fixedValue, 32\*ULP)
- **入口点**：`test/frame/test_main.cpp`（共享 main）
- **构建**：`bash build.sh --ops=spmm_op --soc=ascend950 --run`

### 精度阈值

| dtype | rtol | atol | fixedValue | mantissaBits | emin |
|-------|------|------|------------|--------------|------|
| ACL_FLOAT | 9.77e-4 | 1.53e-5 | 1e-2 | 23 | -126 |
| ACL_FLOAT16 | 1.95e-3 | 1.95e-3 | 1e-1 | 10 | -14 |

判定条件（逐元素）：`|npu - golden| <= (atol + rtol * |golden|)` 且 `|npu - golden| <= max(fixedValue, 32 * ULP)`，匹配率 >= 99%。

## 测试用例分类（182 个测试）

| 分类 | 数量 | 说明 |
|------|------|------|
| L0 功能基础 | 10 | FP32/FP16 基础尺寸，beta=0/1，HOST/DEVICE pointerMode |
| L1 功能覆盖 | 43 | opB=T、RC/CR/CC 主序、ALG2、非对齐尺寸、HOST pointerMode、alpha/beta 边界 |
| L2 大尺寸 | 8 | 2048×1024×2048 等，FP32/FP16 × ALG1/ALG2 |
| WB 边界 | 16 | 空矩阵(m/k/n=0)、单行/单列、全稠密、BetaC、死锁规避 |
| P1 精度 | 15 | 大值域、大K、小值域、混合量级（4 个 skip_precision） |
| P2 FP16 精度 | 14 | 饱和边界、下溢、混合符号抵消 |
| I64 索引 | 4 | int64_t rowOffsets × ALG1/ALG2/opB=T |
| indexBase=ONE | 3 | ZERO/ONE × I32/I64 组合 |
| Kahan 高精度 | 4 | ALG1_HIGH_PRECISION FP32（大K/大值域）+ FP16 静默忽略 |
| unsorted | 4 | colInd 行内打乱 × FP32/FP16/ALG1/ALG2 |
| G 泛化组合 | 20 | G01-G20，交叉覆盖 dtype、算法、转置、索引类型与稀疏度 |
| ZS 零维与 BetaC | 9 | BetaC 边界、m/n=0 与 opB=T 组合 |
| ST 多核与压力 | 8 | 多 block/tile、高密度、大 shape 与 I64 组合 |
| E01-E19 异常 | 19 | null handle/matA/matB/matC/alpha/beta/bufferSize/descr/plan，维度不匹配，不支持类型 |
| 值更新 | 2 | ALG1 csrValues 原地更新（FP32/FP16） |
| 确定性 | 3 | bit-identical 5 次重复、ALG1 vs ALG2 一致性、5 种子稳定性 |

> 其中 L0-L2、WB、P1、P2、I64、IB、KH、US、G、ZS、ST 共 158 个用例由 CSV 驱动（TEST_P）；E01-E19 异常、值更新、确定性共 24 个用例为 TEST_F 硬编码。

## CSV 列定义

| 列名 | 说明 |
|------|------|
| case_name | 用例标识 |
| description | 用例描述 |
| m, k, n | 矩阵维度（A: m×k, B: k×n, C: m×n） |
| sparsity_ratio | 稀疏度（0.0~1.0） |
| alpha, beta | 标量参数 |
| dtype | ACL_FLOAT / ACL_FLOAT16 |
| compute_type | ACL_FLOAT（固定） |
| op_b | ACL_SPARSE_OP_NON_TRANSPOSE / ACL_SPARSE_OP_TRANSPOSE |
| order_b, order_c | ACL_SPARSE_ORDER_ROW / ACL_SPARSE_ORDER_COL |
| alg | ACL_SPARSE_SPMMOP_ALG_DEFAULT / _ALG1 / _ALG2 / _ALG1_HIGH_PRECISION |
| pointer_mode | HOST / DEVICE |
| value_lo, value_hi | 值域范围 |
| expect_result | ACL_SPARSE_STATUS_SUCCESS |
| random_seed | 随机种子 |
| skip_precision | 是否跳过精度验证（0/1） |
| index_base | ACL_SPARSE_INDEX_BASE_ZERO / ACL_SPARSE_INDEX_BASE_ONE |
| row_offset_type | ACL_SPARSE_INDEX_32I / ACL_SPARSE_INDEX_64I |
| unsorted | 是否打乱 colInd 顺序（0/1） |

## golden 参考实现

`spmm_op_golden.h` 中的 `SpmmGolden` 函数使用 FP64 三重循环计算参考结果：

- 遍历 CSR 行偏移区间 [rowOffsets[i], rowOffsets[i+1])
- 对每行做 dot(A_row, B_col) 累加
- 支持 opB=TRANSPOSE、indexBase=ZERO/ONE
- FP16 路径先量化到 FP16 再用 FP64 计算，对齐 NPU 的 FP16 输入精度

CSR 生成器 `MakeSpmmSparsity` 使用确定性稀疏模式：位置 (i, j) 为非零当且仅当 `(i * 7 + j * 13) % 100 < int(ratio * 100)`，值由 `std::mt19937` + `uniform_real_distribution<double>` 按 `random_seed` 生成，保证 golden 与 NPU 两侧数据 bit-for-bit 一致。

## NPU 生命周期封装

`spmm_op_npu_wrapper.h` 中的 `SpmmOpNpu` 模板函数封装完整的 7 函数生命周期：

1. `aclsparseSpMMOp_bufferSize` → 获取 workspace 大小
2. `aclsparseSpMMOp_createDescr` → 创建描述符，绑定 matA（CSR）
3. `aclsparseSpMMOp_createPlan` → 创建执行计划
4. `aclsparseSpMMOp_setGlobalUserData` → 可选，no-op
5. `aclsparseSpMMOp` → 执行计算
6. `aclsparseSpMMOp_destroyPlan` → 销毁计划（RAII）
7. `aclsparseSpMMOp_destroyDescr` → 销毁描述符（RAII）

支持 I32/I64 rowOffsets 转换、indexBase ZERO/ONE、ROW/COL 主序 repack。`SpmmOpDescrGuard` / `SpmmOpPlanGuard` 提供 RAII 自动销毁；NPU 崩溃时通过 `release()` 跳过 destroy 避免二次 segfault。

## 编译运行

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
bash build.sh --ops=spmm_op --soc=ascend950 --run
```

预期输出：

```
[==========] 182 tests from 4 test suites ran.
[  PASSED  ] 182 tests.
[PASS] spmm_op_test
```

4 个 test suite 对应：

| test suite | 驱动方式 | 用例数 |
|------------|---------|--------|
| SpmmOpCases (SpmmOpTest) | TEST_P + CSV | 158 |
| SpmmOpExceptionTest | TEST_F | 19 |
| SpmmOpValueUpdateTest | TEST_F | 2 |
| SpmmOpDeterminismTest | TEST_F | 3 |

## 性能采集

`spmm_op_perf.cpp` 是独立的性能采集程序，复用 golden 稀疏生成器与 7 函数生命周期，但预分配所有 device buffer / descriptor / plan，仅对 Execute 阶段（`aclsparseSpMMOp`）用 `aclrtEvent` 计时，隔离 host 侧 setup 与 H2D copy 开销。

每个用例输出：kernel time（min / median / mean，us）、FLOPs 与 achieved GFLOPS、HBM bytes 与 achieved bandwidth（GB/s）、bandwidth utilization。

### 编译性能程序

```bash
source /usr/local/Ascend/cann/set_env.sh

g++ -std=c++17 -O2 \
    -I test/frame \
    -I test/spmm_op \
    -I test/spmm_op/arch35 \
    -I sparse/spmm_op/arch35 \
    -I include \
    -I ${ASCEND_HOME_PATH}/$(uname -m)-linux/include \
    test/spmm_op/arch35/spmm_op_perf.cpp \
    -o build/spmm_op_perf \
    -L build -lops_sparse \
    -L ${ASCEND_HOME_PATH}/lib64 -lascendcl
```

### 运行

```bash
export LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH
./build/spmm_op_perf --warmup=5 --iters=20 --peak=1421
```

性能用例矩阵共 60 个用例：

- 4 shapes × 2 dtypes × 2 ALGs（ALG1, ALG2）× 3 sparsities = 48
- 4 shapes × 1 dtype(FP32) × 1 ALG(HIGH_PRECISION) × 3 sparsities = 12
- （HIGH_PRECISION 仅对 FP32 有意义，FP16 静默忽略 Kahan 与普通 ALG1 等价，跳过）

可选参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--warmup` | 5 | 预热迭代次数 |
| `--iters` | 20 | 计时迭代次数 |
| `--peak` | 0 | 峰值 HBM 带宽（GB/s），用于计算 BW 利用率；0 表示不计算 |
| `--only` | 空 | 仅运行 label 包含该子串的用例 |
| `--bwbench` | false | 运行 HBM 带宽基准测试后退出 |
