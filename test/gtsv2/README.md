# aclsparseSgtsv2 测试说明

## 概述

本目录验证 FP32 `aclsparseSgtsv2` Legacy API 的完整调用流程：查询
workspace、分配 128 字节对齐缓冲区、在句柄绑定的 stream 上异步求解，并将
解原地写回 column-major 的 `B`。

测试按 SOC 自动选择实现：

- `ascend910b`、`ascend910_93` 使用 `arch22/`（Atlas A2/A3）；
- `ascend950` 使用 `arch35/`（Ascend 950PR/950DT）。

## 目录结构

```text
test/gtsv2/
├── CMakeLists.txt
├── README.md
├── gtsv2_golden.h          # 全程 FP64 的带选主元 CPU Golden
├── gtsv2_param.h           # CSV 参数解析
├── arch22/
│   ├── gtsv2_test.cpp      # 公共回归 + A2/A3 专项测试
│   └── gtsv2_test.csv      # A2/A3 shape、pivot 与性能形状
└── arch35/
    ├── gtsv2_npu_wrapper.h
    ├── gtsv2_test.cpp
    └── gtsv2_test.csv
```

## A2/A3 覆盖范围

`arch22/gtsv2_test.cpp` 复用 arch35 的公共 API、数值和异常测试，并增加
A2/A3 实现专项检查。当前覆盖：

- 30 组 CSV 用例：`m=3`、`n=0/1/多 RHS`、`ldb=m/带 padding`、非对齐
  shape、确定性 pivot 分支、奇异矩阵、UB/GM 路径边界，以及 P-01/P-02/P-03
  正式性能 shape 的功能回归；
- 单 RHS 融合路径额外覆盖 `4096×1, ldb=4112`、`m=8192/8193` 的
  UB/GM 边界、连续/首次/末次 pivot，以及极端主元比例；
- 19 组公共异常用例：空句柄/指针、非法 `m/n/ldb`、空或未对齐 workspace
  及 `n=0` 优先级；
- arch22 专项：紧凑 workspace 公式和最大维度宽整数运算、输入对角线只读、
  `ldb` padding 保持、workspace guard，以及同一 stream 上异步
  `D2D -> aclsparseSgtsv2 -> D2D` 的顺序语义。

新增单 RHS 综合测试在零首主元条件下重复执行三次，验证 FP64 Golden 精度、
确定性、系数只读、padding、workspace guard 和同 stream 异步顺序。
另有原 seed-8 消去抵消矩阵在 17/129/8193 行、单/多 RHS 路径的消去抵消精度回归。
向量 Jacobi 新增 6 项专项：127/128/129、6143/6144/6145、8191/8192/8193 边界，接近 1/2 的收缩裕量、
正负对角系数、RHS 归一化范围阈值内外、17 列非对齐 leading dimension 下的事件复用、
Inf/NaN 回退后下一列的正常执行、65 列多核多轮执行；均复核重复结果、系数只读、padding 和 workspace 哨兵。
扩展指数路径新增 4 项专项：有限输入的 Schur 补/RHS 中间溢出（pivot/nonpivot、
单/多 RHS、127/128/129、6144/6145、8192/8193 边界）；同一矩阵内 FLT_MAX 与最小非规格化系数并存；
仅 RHS 超大时的回退及跨非有限列复用、正零、系数/padding/workspace 守卫；最终非规格化输出舍入。
混合行尺度新增 3 项专项：固化 m=12 独立回归输入的完整 FP32 十六进制系数与 RHS，
以 150 位十进制独立求解生成参考解，覆盖普通/混合行尺度、单/多 RHS 和尺寸边界，
同时检查解误差、逐行分量后向误差及 padding；另以精确值检查同一行内最大有限数与
最小非规格化系数并存时不会因归一化丢失小系数。该独立输入不冒充评审尚未提供的原始输入。
同时固化 3/4/12 行的独立反例，检查归一化改变主元时较小解分量仍被保留，以及跨 RHS 的因子复用；
这些反例参考解使用 500 位十进制求解。
逐行分量后向误差为 `abs(A*x-b)/(abs(b)+abs(A)*abs(x))`，新增回归阈值为 `1e-5`，
解误差保持 `1e-4*(1+abs(expected))`，不以放宽阈值消除失败。
新增固定 4×4 病态回归：旧版虽后向误差很小却存在大解误差，覆盖 4/129/8193 行、单/多 RHS 和 padding，并同时检查解误差与方程残差。

arch22 完整集合共 71 条：30 条数值用例、19 条异常用例、22 条专项测试。
当前 workspace 为 `align_up(align_up(17*m,128)+85*m,128)`，仍与 RHS 数量无关。

普通非奇异结果按项目 FP32 混合精度标准与 FP64 Golden 比较；奇异系统验证
Inf/NaN 传播位置。`dl`、`d`、`du` 均不得被修改。

## 编译与运行

先加载 CANN 9.1.0 或后续配套版本的环境，例如：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

在仓库根目录按实际硬件执行：

```bash
# Atlas A2（报告中分别记录 910B3、910B4 实际型号）
bash build.sh --ops=gtsv2 --soc=ascend910b --run

# Atlas A3
bash build.sh --ops=gtsv2 --soc=ascend910_93 --run

# A5 交叉回归
bash build.sh --ops=gtsv2 --soc=ascend950 --run
```

运行前应确认 `npu-smi info` 中的实际 SOC、驱动和固件版本；自测报告还需记录
CANN 版本、代码提交号、全部测试结果以及失败项。性能、峰值内存和 Profiler
采集应使用任务包提供的专项脚本，不能用本 GTest 的墙钟时间代替。
