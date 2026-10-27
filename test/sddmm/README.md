# SDDMM 测试

## 测试说明

测试覆盖 `aclsparseSDDMM` 三阶段接口，以及统一 `torch_extension` 注册的
`torch.sparse.sampled_addmm`。算子规格见 [SDDMM 算子说明](../../sparse/sddmm/README.md)。
以下命令从仓库根目录执行。Ascend 950（arch35）和 Atlas A2/A3（arch22）
分别使用各自匹配的 CANN、PyTorch 和 `torch_npu`；硬件能力以对应的
[arch35 说明](../../sparse/sddmm/README.md)和
[arch22 说明](../../sparse/sddmm/arch22/README.md)为准。

## 目录结构

```text
test/sddmm/
├── CMakeLists.txt
├── README.md
├── python/
│   ├── test_sddmm_torch_extension.py        # Ascend 950 公开 API 与 out/alias 回归
│   ├── test_sddmm_torch_extension_arch22.py # Atlas A2/A3 精度、batch 与输出所有权
│   ├── test_sddmm_cached_entry.py          # Atlas A2/A3 prepared/cache 回归
│   ├── test_sddmm_bsr.py                   # Atlas A2/A3 小块 BSR 回归
│   └── _sddmm_utils.py                    # Python Golden 与架构选择辅助
├── sddmm_golden.h                 # CPU 参考实现和数据生成
├── sddmm_param.h                  # CSV 参数解析
├── sddmm_test_layout.h            # 通用测试数据与布局辅助
├── sddmm_test_common.h            # Atlas A2/A3 参数化测试入口辅助
├── arch22/                       # Atlas A2/A3 CSV 和 SddmmContract 三阶段回归
└── arch35/
    ├── sddmm_test.cpp             # 全部 GTest、CSV、描述符与扩展回归测试
    ├── sddmm_test_support.h       # 辅助类型和函数声明
    ├── sddmm_test_bsr_support.cpp # BSR 数据、参考计算与执行辅助实现
    ├── sddmm_test_csr_support.cpp # CSR、状态与描述符辅助实现
    ├── sddmm_test_complex_support.cpp   # 复数特殊值回归辅助
    ├── sddmm_test_workspace_support.cpp # workspace 跨算子回归辅助
    ├── sddmm_test.csv             # 参数化精度用例
    ├── sddmm_npu_wrapper.h        # 三阶段 NPU 调用封装
    └── sddmm_perf.cpp             # 原生 Execute 性能程序
```

被测实现位于 `sparse/sddmm/arch35/host/` 与 `kernels/{csr,bsr,shared}/`，
目录职责见 [算子实现目录](../../sparse/sddmm/README.md#实现目录)。
Host/Device 共用规划头仍位于 `arch35/` 根目录；测试辅助头引用
`sddmm_complex_bsr_cube_common.h`、`sddmm_csr_cube_common.h` 和 `sddmm_tiling_data.h`。

## C++ 测试

先将 `ASCEND_HOME_PATH` 设置为实际 CANN 安装目录，再加载环境：

```bash
source "${ASCEND_HOME_PATH}/set_env.sh"
```

需要 C++17 编译器和 GTest 1.10 及以上版本。GTest 不在默认搜索路径时，
先将 `GTEST_PREFIX` 设置为其安装前缀，再通过环境变量提供给 CMake：

```bash
export CMAKE_PREFIX_PATH="${GTEST_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
```

使用仓库统一入口构建并运行：

```bash
bash build.sh --ops=sddmm --soc=ascend950 --run
```

此入口构建 `sddmm_test` 和 `sddmm_perf`；未设置 GTest 过滤条件时，
运行 Ascend 950 对应的全部 SDDMM C++ GTest 和 CSV 用例。测试从
`build/test/sddmm` 运行，以读取构建阶段复制的 CSV。所有扩展回归用例
与既有用例统一位于 `sddmm_test.cpp`；support 源文件仅提供辅助实现，
由同一个 `sddmm_test` 目标编译链接，无需额外入口。
验收时应检查退出码为 0、测试汇总无失败，并核对实际执行用例；
以本次运行结果作为验收依据。

Atlas A2/A3 使用实际目标 `--soc` 配置相同测试目标；CMake 根据
`SOC_ARCH_DIRS` 选择 `arch22/` 并额外注册 `sddmm_contract_test.cpp`。
`SddmmContract.*` 覆盖空维度、零归约、布局、标量指针模式、批次和错误输入。
共享描述符按最小可读跨度接受稠密 batch stride；arch35 SDDMM 仍要求完整
`major * ld` 跨度，并有专门用例验证短跨度在 BufferSize/Preprocess 阶段被拒绝。

需要同时检查关联的 SpMM 回归时，可执行：

```bash
bash build.sh --ops=sddmm,spmm --soc=ascend950 --run
```

PyTorch 测试需按下一节单独执行；原生性能采样需按“原生性能测试”一节执行。

arch35 覆盖 CSR/BSR、base 0/1、ROW/COL 布局、转置、batch、混合输出类型、
complex64、Host/Device pointer mode、prepared state、零维矩阵、异常参数及快速路径回退。
可以通过 `--gtest_filter` 选择以下测试组：

| 过滤器 | 覆盖内容 |
| --- | --- |
| `SddmmCases/SddmmTest.SddmmFunctional/*` | CSV 参数化 CSR 精度用例 |
| `SddmmExceptionTest.*` | 空指针、维度和不支持组合 |
| `SddmmDescriptorApiTest.*` | BSR 和 strided-batch 描述符接口 |
| `SddmmZeroDimensionTest.*` | M/N/K 为 0 的三阶段接口行为 |
| `SddmmExtendedNpuTest.*` | BSR、complex、batch、快速路径和回退 |
| `SddmmCubeResourceTest.*` | Host 侧 complex BSR L0 容量边界、分组路径需求及不支持的 tile；不执行 NPU Kernel |

CPU Golden 与输出按 `test/frame/verify.h` 的 `MIXED_TOLERANCE` 比较；
FP16/BF16 使用 FP32 累加，FP32 使用 FP64 Golden，complex64 的实部和虚部分别比较。

### 可选全仓回归

需要验证仓库全部原生测试时，可省略 `--ops`：

```bash
bash build.sh --soc=ascend950 --run
```

## PyTorch 端到端测试

先构建核心库，再构建并安装统一 `cann_ops_sparse` wheel：

```bash
CMAKE_BUILD_TYPE=Release bash build.sh --ops=sddmm --soc=ascend950
python3 -m pip install -r torch_extension/requirements.txt
bash build.sh --torch_extension --ops=sddmm
python3 -m pip install --force-reinstall --no-deps build_out/cann_ops_sparse-*.whl
OPS_SPARSE_LIB_DIR=$PWD/build_out/lib64 \
  python3 -m pytest test/sddmm/python/test_sddmm_torch_extension.py -v
```

`import cann_ops_sparse` 注册 `aten::sparse_sampled_addmm` 的默认和 `out` 入口。
C++ wrapper 在第一次 NPU 调用时由 PyTorch JIT 编译；`OPS_SPARSE_LIB_DIR` 指向
`libops_sparse.so` 目录，未设置时使用 `$ASCEND_HOME_PATH/lib64`。首次加载按当前
设备选择 arch22 或 arch35 wrapper；仅导入 Python 包不查询设备或触发 JIT。

测试覆盖 FP16/FP32/complex64、int32/int64 索引、非连续及延迟共轭/取负视图、
CSR batch 与广播、独立输出及 `out=input`、多 stream、连续队列调用、空 pattern、
K=0、beta=0 忽略旧 NaN/Inf，以及非法输入。还检查导入不触发 JIT 和
`torch_npu.sparse.sampled_addmm` 公开入口。缺少依赖或可用 NPU 时 pytest 会报告 skip。

两种架构均可发现完整 Python 测试目录：

```bash
OPS_SPARSE_LIB_DIR=$PWD/build_out/lib64 \
  python3 -m pytest test/sddmm/python -v
```

测试在类初始化时检查当前设备：Ascend 950 执行公开 API、`out` 和共享存储回归；
Atlas A2/A3 执行独立的公开接口、CSR prepared/cache 和 BSR 测试。
不适用的另一架构测试报告 skip，不计作本架构通过。A2/A3 的直接接口和
prepared 接口原地更新数值，标准 `sampled_addmm` 返回独立输出；高动态范围
complex64 测试分别比较实部和虚部。各平台结果须分别记录。

## 原生性能测试

`sddmm_perf` 在 BufferSize、Preprocess 和分配完成后，以 ACL Event 包围 Execute，
输出延迟统计和 JSON。可选择单个用例：

```bash
export LD_LIBRARY_PATH="$PWD/build_out/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
(cd build/test/sddmm && \
  ./sddmm_perf --only=sddmm-P-01-bsr-float16-base0 --warmup=10 --iters=30)
```

`--only` 接受用例名称子串，省略时运行全部原生锚点。
该结果只反映原生 Execute；记录性能结果时应同时记录硬件、CANN 版本、用例和采样次数。

## 与 SpMM 交替复用 workspace

同时构建 SDDMM 与 SpMM 时，`sddmm_test` 额外注册真实跨算子回归，覆盖
packed-Y 和 CSR Cube 两条写入路径，并分别验证同一 workspace 和独立 workspace：

```bash
bash build.sh --ops=sddmm,spmm --soc=ascend950 --run
```

仅构建 SDDMM 时，仍运行 warm Execute 的 workspace owner 失效回归；
真实 SpMM 调用用例仅在同时编入 SpMM 或全量构建时注册。
