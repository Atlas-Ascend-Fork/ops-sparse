# SpSM 测试

## 测试说明

SpSM 测试覆盖 `aclsparseSpSM*` 多阶段接口和 `aclsparseSpSMUpdateMatrix`，验证 Ascend 950 上的
CSR、CSC、COO 稀疏三角求解。覆盖 FP32、complex64，`opA`/`opB` 的 N/T/H 组合，LOWER/UPPER、
UNIT/NON_UNIT，ROW/COL 稠密布局，Host/Device pointer mode、原地求解、数值更新、异步 stream
语义及异常生命周期。算子规格、接口约束和调用流程见
[SpSM 算子说明](../../sparse/spsm/README.md)。

## 目录结构

```text
test/spsm/
├── CMakeLists.txt
├── README.md
└── spsm/
    ├── spsm_fp16_util.h                # FP16 数据转换辅助函数
    ├── spsm_golden.h                   # Eigen FP64/complex128 Golden
    ├── spsm_param.h                    # CSV 参数解析
    ├── spsm_verify.h                   # 任务书精度判据
    └── arch35/
        ├── spsm_npu_wrapper.h          # NPU 接口调用封装
        ├── spsm_test.cpp               # 参数化 C++ UT/ST
        ├── spsm_test.csv               # 基础与专项参数用例
        ├── spsm_extended_test.cpp       # Ascend 950 专项、边界及生命周期测试
        └── spsm_perf.cpp                # 独立性能测试程序
```

## 环境要求

- Ascend 950PR 设备
- 与仓库版本匹配的 CANN 开发环境
- Eigen3（构建测试时由工程依赖配置提供）

执行前加载 CANN 环境，例如：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

实际安装路径不同时，加载对应 CANN 安装目录下的 `set_env.sh`。

## 编译和运行

在仓库根目录执行：

```bash
bash build.sh --ops=spsm --soc=ascend950 --run
```

也可以在构建完成后单独运行：

```bash
(cd build/test/spsm && ./spsm_test --gtest_color=no)
```

当前 Ascend 950 完整测试目标包含 248 项测试。测试结果以 GTest 进程退出码和生成的 XML 为准；
需要保留报告时可执行：

```bash
(cd build/test/spsm && ./spsm_test \
  --gtest_color=no \
  --gtest_output=xml:spsm-test-report.xml)
```

测试分工如下：

| 测试类别 | 测试入口 | 主要覆盖内容 |
| --- | --- | --- |
| 参数化功能和精度 | `spsm_test.cpp`、`spsm_test.csv` | 数据类型、稀疏格式、转置、填充模式、对角类型和稠密布局 |
| 接口契约和异常 | `SpsmExceptionTest` | 空指针、尺寸、类型、算法、leading dimension、stage 顺序和无效 update |
| 生命周期和内存边界 | `SpsmExtendedTest` | workspace 大小与生命周期、销毁后的 plan、独立 stream、地址重叠和异步返回 |
| 性能 | `spsm_perf` | Analysis、Update、Solve 的设备侧耗时以及 workspace 字节数 |

## 性能测试

性能程序作为独立目标构建，不依赖 GTest。先完成 SpSM 工程配置，再执行：

```bash
cmake --build build --target spsm_perf -j
./build/test/spsm/spsm_perf \
  --m 4096 --rhs 32 --nnz-per-row 8 \
  --format csr --dtype fp32 --warmup 5 --samples 30
```

`--format` 支持 `csr`、`csc`、`coo`，`--dtype` 支持 `fp32`、`complex64`。如需覆盖全部组合：

```bash
for format in csr csc coo; do
  for dtype in fp32 complex64; do
    ./build/test/spsm/spsm_perf --format "${format}" --dtype "${dtype}"
  done
done
```

程序使用 ACL runtime event 计时，并输出单行 JSON。`analysis_*_us`、`update_*_us`、`solve_*_us`
分别给出各阶段的 min、median、mean、p90；`workspace_bytes` 是 BufferSize 返回的 workspace 大小，
`input_output_bytes` 是本用例稀疏矩阵与 B/C 数据占用的设备字节数。性能对比应固定设备、CANN 版本、
矩阵参数、预热次数和采样次数。

## 精度判定

参考结果使用实际 FP32 输入，并以 Eigen FP64/complex128 计算。实部和虚部分别检查：

- `rtol = 2^-13`
- `atol = 2^-13`
- 元素匹配率不低于 99%
- 每元素绝对误差不超过 `max(0.01, 32 ULP)`
- NaN 必须与 NaN 对应，Inf 必须同号对应

专项测试还覆盖未排序和重复坐标、不同 index base、空 values 的 BufferSize/Analysis、Update 后
重复求解、workspace 生命周期、独立 stream、B/C 地址重叠以及确定性结果。
