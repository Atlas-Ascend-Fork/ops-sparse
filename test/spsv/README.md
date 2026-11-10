# SpSV测试（Atlas A2/A3）

测试沿用仓库的GTest构建与运行入口。生产接口和限制见
[arch22接口说明](../../sparse/spsv/arch22/README.md)。
`ascend910b*`（A2）和`ascend910_93*`（A3）均选择arch22；
Ascend 950的既有测试在arch35目录中独立构建。

arch22源码按`host/`、`kernel/`组织；本目录保留原生GTest入口。
对应内部头文件路径仅加入`spsv_test`的arch22构建配置。

## 构建与运行

从仓库根目录执行，使用目标设备配套的CANN环境：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
SOC=ascend910b3
# A3：SOC=ascend910_93；有具体后缀时使用环境对应的实际SOC名称
cmake -S . -B "build-$SOC" -DSOC_VERSION="$SOC" -DOP_LIST=spsv -DBUILD_TEST=ON
cmake --build "build-$SOC" -j2
SPSV_TEST_DEVICE_ID=0 "./build-$SOC/test/spsv/spsv_test"
```

可使用GTest标准选项`--gtest_filter`选择测试组，或使用`--gtest_output=xml:路径`
保存运行结果；无需额外配置报告目录。

## 测试范围

| 内容 | 入口或过滤器 |
| --- | --- |
| 四格式、N/T/H、dtype、fill/diag、base、pointer mode、原地与更新 | [spsv_test.cpp](arch22/spsv_test.cpp)，`CoreMatrix/*:SpSVArch22Matrix.*` |
| 规范化、索引与values置换、保护区、重复Analysis | `SpSVArch22Normalization.*` |
| 参数异常、描述符绑定和异步资源生命周期 | `SpSVArch22Test.*:SpSVArch22Parameters.*:SpSVArch22Lifecycle.*:SpSVArch22Lifetime.*` |
| 公开内存查询结果、能力降级和完整缓冲区边界 | `SpSVArch22Memory.*:SpSVArch22Test.PublicRangeChecksLiveAllocationAndOffsets:SpSVArch22Test.*Spans` |
| 数值范围、抵消与重复对角累加 | [spsv_range_tests.h](arch22/spsv_range_tests.h)、[spsv_duplicate_diagonal_tests.h](arch22/spsv_duplicate_diagonal_tests.h)，`SpSVArch22Range.*:DuplicateCancellation/*:SpSVArch22DuplicateLowPart.*` |

FP32/complex64的CPU参考解使用FP64/complex128计算。数值专项还检查有限结果、
ULP误差和重复执行一致性。生命周期负例由父测试在独立进程中执行子测试并核对退出码，
不继承已初始化ACL进程的运行时线程。

## 精度判定依据

本目录采用CANN社区任务“09-9：aclsparseSpSV算子开发(A2/A3)”的专项精度要求。
任务来源见[社区任务书索引](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/README.md#9月份发放的任务)。
专项参数同时记载于社区任务仓库的[SpSV A2/A3设计文档“精度标准/性能标准”](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/tasklist/09-9-aclsparseSpSV-A2A3/qq_69884268/docs/design.md)。
《aclsparseSpSV算子开发(A2/A3)任务书》第3.2节“精度要求”规定：

- FP32使用FP64 Golden；complex64使用complex128 Golden，并分别比较实部与虚部。
- 混合容差为`|actual - golden| <= 2^-16 + 2^-10 * |golden|`，匹配率至少为99%。
- 每个有限值比对点的绝对误差还须不超过`max(1e-2, 32 * ULP(golden))`；
  ULP按输出FP32分量收窄后的值计算。

`spsv_test.cpp`中的`Verify`对每个输出分量都执行上述容差与绝对误差检查，
要求100%分量通过，比专项要求的99%匹配率更严格；NaN/Inf按参考结果的类别和符号检查。
数值范围专项还验证有限Golden按输出类型收窄后的溢出语义。

上述`atol=2^-16、rtol=2^-10`来自SpSV任务专项要求，不能作为opbase通用FLOAT32阈值引用。
当前[opbase混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
第2.2节的通用FLOAT32阈值为`atol=rtol=2^-13`。二者的容差曲线不同，不能笼统视为
一套标准整体严于另一套；本目录的SpSV任务回归按上述专项参数判定。

## 内存边界测试

内存边界测试使用公开`aclrtMemGetAddressRange`；纯Host结果判定用例覆盖成功、
`ACL_ERROR_RT_FEATURE_NOT_SUPPORT`、其他查询错误、范围越界和整数回绕，不提交Kernel。
活跃分配用例按公开查询返回的实际范围校验，允许分配范围大于请求字节数。
完整缓冲区用例将输入或输出放到公开分配范围的尾部，覆盖四格式的offset/index/value数组、
FP32/complex64向量与设备端alpha、GENERAL存储values和DIAGONAL的m个values。
分别检查比所需容量少4字节时拒绝，以及容量恰好足够时可正常Analysis或更新、求解。
公开范围查询不支持时，完整缓冲区用例明确标为跳过；释放workspace负例由父测试标为跳过，
不启动子进程。
子测试在释放前后再次查询；公开检查无法证明释放地址会被拒绝时，报告失败并停止，
避免对已释放workspace调用Solve或UpdateMatrix。调用方仍须保证缓冲区容量及异步生命周期有效。
