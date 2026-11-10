# Cgtsv2 API 测试

## 测试概述

本目录通过公开 `aclsparseCgtsv2_bufferSizeExt`、`aclsparseCgtsv2` 在 NPU 上执行测试。
复用 `test/frame` 的 GTest/CSV、`AclEnvScope`、`HandleManager`、`DeviceBuffer` 等基础设施。
独立 complex128 带主元三对角参考不依赖 Eigen3，不复用设备运算。

## 构建与运行

在对应 CANN 环境下，从仓库根目录执行：

```bash
source /actual/CANN/set_env.sh
bash build.sh --soc=ascend910b3 --ops=cgtsv2 --run --device=0
```

SoC 和设备号按实际 A2/A3 环境选择。GTest 必需，Eigen3 不需要。
直接运行测试二进制时，应在其构建目录下执行，以便读取 `cgtsv2_test.csv`。

## 测试文件

| 文件 | 用途 |
| --- | --- |
| `CMakeLists.txt` | 注册共享 GTest 构建 |
| `arch22/cgtsv2_test.cpp` | 接口、数值、内存及比较器契约测试 |
| `arch22/cgtsv2_test.csv` | 确定性数值参数，共 26 条 |
| `arch22/cgtsv2_npu_wrapper.h` | Device 输入准备和公开查询/执行接口调用；沿用 frame RAII |
| `cgtsv2_param.h` | 共享 CSV 参数加载 |
| `cgtsv2_golden.h` | 无 Eigen3 的独立 complex128 参考 |
| `../frame/mixed_anchor.h` | 共享 FLOAT64 Golden、最大误差点 cap 比较策略，经 VerifyConfig/Verifier 调用 |

## 覆盖说明

当前共 40 项 GTest：26 条 CSV 参数化用例、9 项定向接口/数值回归、5 项 Host 比较器契约测试。
覆盖复数/零主元、纯虚对角线、多 RHS、padding、尾部、系数只读、workspace 哨兵、重复确定性、
同址改值、异步 stream 顺序、空 RHS、非法参数与奇异状态；小/大尺寸解析回归覆盖前向和回代溢出、
复数交叉项及被抵消的低位。小用例输出原始实际值与未收窄参考，便于独立复核。

大单 RHS dominant 用例 m=2049/4097/8193、ldb=2052/4100/8196，检查 B 的 64 字节对齐、
Jacobi 状态和原 B 快照，固定验证 parallelSingleRhs 路由。大 NaN RHS 用例检查 expanded 派发、
参考非有限值分类及 padding/系数只读。

Jacobi 边界包含占优比 1.26、1.35、1.36、1.5、2.0、2.01，每种均含单列 m=2049 和多列 m=4097。
输入包含大幅抵消与近零解；参考对实际量化后的输入求解。
1.26/1.35 断言 expanded；1.36/1.5/2.0/2.01 断言 refined Jacobi。
同时检查 Golden、额外严格边界误差、padding、workspace 哨兵及两次执行字节一致性。

## 精度比较器例外说明

[官方任务书](https://www.hiascend.com/activities/task-center/details/b8289b3a952d4e13a9e79c3a242a8338?menu=tasks)
第 3.2 节规定：保留 complex128 Golden，实/虚部分别使用 `rtol=atol=2^-13`，匹配率至少 0.99；
再在该分量最大有限绝对误差点检查 `max(0.01,32*ULP(g_low))`，`g_low` 为该点的 FLOAT32 Golden，
ULP 使用带符号值朝正无穷方向的相邻值。非有限值由独立特殊值测试处理。
仓内确定性 ST 将匹配率提高为 1.0，未改官方验收比较器。

按评审要求统一调用共享 `test/frame/VerifyConfig/Verifier`。框架新增显式
`MIXED_TOLERANCE_ANCHOR` 模式，接收 FLOAT32 输出和未舍入的 FLOAT64 Golden，
使用配置中的容差、匹配率及最大误差点 cap；删除算子目录内的自定义比较器。
既有 `MIXED_TOLERANCE` 模式仍保持逐元素 cap，其他算子的默认判定规则不变。
五项契约测试覆盖两种 cap 口径差异、未舍入参考精度、带符号 ULP、形状/配置错误、
空向量与非有限值，以及配置的容差和匹配率。官方特殊值测试仍使用原任务包比较器。

原任务包验收独立执行，输入、Golden、阈值及原比较器保持不变；仓外验收数量不冒充本目录用例数。
