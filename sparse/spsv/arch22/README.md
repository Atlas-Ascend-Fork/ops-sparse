# SpSV（arch22）

## 产品支持情况

| 产品 | 是否支持 | SOC与架构 |
| --- | --- | --- |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ | ascend910_93*，DAV_2201 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ | ascend910b*，DAV_2201 |

本路径使用CANN 9.1.0及配套ASC编译器，A2与A3均由构建系统选择arch22。
Ascend 950使用同仓独立的[arch35实现](../README.md)，不经过本Kernel路径。

## 功能说明

求解`op(A)Y = alpha * X`。支持FP32、complex64，CSR、CSC、COO及仓库描述符定义的
SLICED_ELL，N/T/H、LOWER/UPPER、UNIT/NON_UNIT、base0/1、Host/Device alpha和X/Y同址。
complex64的H操作对矩阵系数共轭；实数H等价于T。生产库不依赖PyTorch，无CPU计算回退。

Analysis在NPU上按原始存储顺序规范化为CSR，保留确定性重复坐标处理；非目标三角项被忽略。
合法CSR的N路径及经逐元素确认的纯对角T/H路径使用有界DMA保序复制；base1在UB中
用Adds规范化，位置映射由CreateVecIndex生成。每次Analysis完整校验rowPtr/colIdx，
过滤、一般转置及其他格式使用通用预处理。
纯对角结构采用128行对齐的独立多核分块，普通三角结构按依赖次序在单AIV核上求解。
紧凑矩阵的解高低分量完整驻留UB，以FP32补偿算术降低跨行舍入误差；较大矩阵使用块缓存
和补偿累加。当完整解的标量分量数不超过24576时，将Y固定驻留于已有UB缓存，
按列索引/values缓存的有效窗口分段遍历，移出逐非零元素的窗口检查；存储顺序、
Neumaier补偿与乘加顺序不变，显式UB缓冲总量仍为163872B。更大Y保留窗口回退路径。
普通范围的抵消先重算当前行；需要范围保护时，保留带指数的快速高低分量求解路径。
内部依赖的精度检查采用相对误差预算，不使用公开输出的绝对误差下限放行小残差。
当依赖精度不足时，从原始X重新求解依赖链。紧凑高低分量路径在严重抵消时也重新求解。
范围感知的前向代入在workspace中保留多个binary64分量，各分量具有独立二进制指数。
紧凑规模保留20个分量，更大规模保留4个分量；每行完成累加与除法后，后续行直接读取这一内部表示。
实数乘积使用整数窗口累加；复数乘积分量遵循binary64舍入规则，再累加各分量贡献。
整数累加跳过没有乘积贡献的指数窗口，并精确处理空区间的进位与符号扩展。
前向代入完成后统一舍入公开输出，尾项参与FP32中点舍入判定。
UNIT且标量分量数不超过24576时，在求解前估计绝对三角逆的上界，并据此选择整数位宽与二进制点。
该整数路径按三角方向直接求解，截断误差的放大界小于2^-24；其适用性检查在产生解向量之前完成。
该路径另占用30624B数学缓冲，全部显式UB缓冲总量为194496B，受共享预算常量和Host容量检查约束。
原地求解保留原始输入快照，直到全部依赖计算完成后写出结果。
求解分支依据矩阵规模与结构选择。

## 参数说明

公开原型沿用[公共头文件](../../../include/cann_ops_sparse.h)，不另增替代API。

| 参数 | 位置与规格 |
| --- | --- |
| handle | 有效句柄，已设置调用stream及pointer mode |
| opA / alg | N/T/H / DEFAULT |
| alpha | Host或当前逻辑Device上的FP32/complex64标量，与computeType一致 |
| matA | 方阵、四种支持格式、Device I32索引、连续Device values，base0/1 |
| vecX / vecY | 连续Device向量，长度m，dtype与A一致；BufferSize/Analysis允许NULL描述符 |
| computeType | ACL_FLOAT或ACL_COMPLEX64 |
| spsvDescr | createDescr创建，Analysis成功后才可Solve/UpdateMatrix |
| bufferSize / externalBuffer | Host字节数输出 / 调用方分配的512B对齐Device workspace |
| newValues / updatePart | 连续Device values / GENERAL或DIAGONAL |

GENERAL读取原始存储长度的values；DIAGONAL读取m个逻辑对角values。重复对角更新时仅
保留一份新对角值，其他重复项置零，避免重复计入。更新不改变稀疏结构，也不插入缺失的对角项。

## 约束说明

- m、nnz不超过I32可表示范围；所有matrix/vector stride为1。不支持其他dtype、I64 Device索引或其他算法枚举。
- SLICED_ELL沿用现有仓库语义：每个slice包含m个行组，每行`sliceNnz`个槽位，slice跨度为`m * sliceNnz`；小于base的列坐标为padding。
- 非空矩阵workspace为512B头部加64B对齐的CSR行指针、列索引、规范化values、原始位置映射、scratch、中间解和原地输入快照。UNIT且规模满足整数路径条件时另预留有界整数区；m或nnz为0时查询值为0。
- Analysis在handle stream下发NPU预处理后，同步并读取12B状态，用于返回索引/结构错误和确定求解分支。它不是完全异步接口；Solve与UpdateMatrix不主动同步。
- Analysis绑定handle、stream、pointer mode、矩阵描述符、结构和workspace。改变这些参数须重新Analysis。
- Analysis显式传入向量时绑定向量描述符及values；传NULL向量时允许后续Solve使用新的合规向量。
- X/Y可完全同址，但不允许部分重叠。A与X保持只读，原地调用时X仅作为Y输出被覆盖。
- UNIT忽略存储的对角数值。NON_UNIT的缺失/零对角在FP32路径按实数IEEE除法传播Inf/NaN，complex64零除数输出实部、虚部NaN；显式存储的零依赖不等同于未存储项。
- 调用方必须保证所有Device缓冲区、描述符和Device alpha在相关异步任务完成前有效；释放前同步stream。
- Host通过公开接口校验Device指针的位置、当前设备及对齐。`aclrtMemGetAddressRange`支持时，
  进一步校验分配范围，workspace须覆盖BufferSize字节数；其他查询错误按非法输入拒绝。
  该接口返回`ACL_ERROR_RT_FEATURE_NOT_SUPPORT`时，仅保留基础校验，不依赖驱动内部接口。
  此时不能保证检测已释放缓冲区或分配范围越界；缓存分配器内部对象释放和地址复用也不能仅由裸指针区分。
  调用方须保证缓冲区大小足够，并在异步任务完成前保持其有效。

## 框架队列互操作

本实现默认启用已加载torch_npu的队列提交，不需要额外加载适配模块。进程已初始化CPython且
调用线程具有Python线程状态时，Analysis、Solve和UpdateMatrix在下发前短暂持有GIL检查
已加载模块；仅当`torch_npu.npu`存在时，才查询当前逻辑设备的框架stream，并通过其
`npu_stream`属性提交Host队列。
该步骤可能阻塞Host，但不等待整个Device完成，也不替换handle stream或补建跨stream事件依赖。
纯原生进程及不具有Python线程状态的框架工作线程保留调用方管理的ACL顺序。

纯ACL调用方可在调用前设置以下环境变量，完全跳过解释器探测、GIL获取及框架模块加载：

```bash
export OPS_SPARSE_SPSV_TORCH_QUEUE=0
```

未设置该变量或将其恢复为`1`时使用默认互操作。关闭后，调用方须自行保证生产者已经提交任务，
并通过正确的stream顺序或事件依赖保证输入就绪；仅传入Device指针不能建立生产者依赖。
不要与正在执行的接口并发修改这些环境变量。

解释器或stream探测不可用、属性缺失和指针转换失败时，库记录告警并继续使用原ACL stream；
已经取得框架stream后，其队列提交属性抛出`RuntimeError`，或显式提交模块返回失败时，接口返回
`ACL_SPARSE_STATUS_INTERNAL_ERROR`。调用前已有的Python异常会保留。告警降级不能代替
输入依赖管理：若仍依赖框架队列，应先恢复可用的框架环境或由调用方明确完成生产者提交。

Python C API符号在首次启用互操作时缓存，不主动初始化解释器或导入框架；已初始化解释器中的
torch_npu可后续导入。嵌入式程序应在首次启用互操作前准备好解释器，并保证接口调用不与解释器
初始化或销毁并发。检测到解释器收尾时不再获取GIL或回调Python。信号处理及解释器生命周期
管理路径应关闭互操作，并遵守ACL自身的线程与调用约束。

`OPS_SPARSE_SPSV_TORCH_QUEUE_LIBRARY`可指定绝对路径的外部队列提交模块。
路径无效、模块不可用或ABI入口缺失时记录告警并回到默认探测；它不能替代上述关闭开关。
生产库不链接libtorch，框架互操作不改变NPU求解及Device缓冲区的异步生命周期要求。

## 调用说明

按`createDescr -> bufferSize -> analysis -> solve -> updateMatrix -> solve -> destroyDescr`
执行。更新可省略，Solve可重复；销毁与释放前调用`aclrtSynchronizeStream`。
端到端生命周期、错误行为、四格式、原地及异步资源释放示例在
[C++设备测试](../../../test/spsv/arch22/spsv_test.cpp)中。

从仓库根目录执行。`SOC`默认选择A2，构建A3时改为`ascend910_93`；
如环境提供具体的SOC后缀，则填写与目标设备配套的实际SOC名称。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
SOC=ascend910b3
# A3：SOC=ascend910_93
cmake -S . -B "build-$SOC" -DSOC_VERSION="$SOC" -DOP_LIST=spsv -DBUILD_TEST=ON
cmake --build "build-$SOC" -j4
SPSV_TEST_DEVICE_ID=0 "./build-$SOC/test/spsv/spsv_test"
```

数值测试使用同一份FP32/complex64输入，并升为FP64/complex128计算CPU参考解。
对于超出参考类型指数范围后返回可表示范围的依赖链，使用解析解核验，避免参考计算自身溢出或下溢。

可通过[测试说明](../../../test/spsv/README.md)选择功能、规范化、数值范围、参数和生命周期测试组。
重复坐标的逻辑对角使用补偿累加，GENERAL更新保留相同的坐标归并语义。

## 实现文件

arch22实现按Host与Kernel分目录维护，内部辅助文件按关联职责整合：

```text
arch22/
├── host/
│   ├── spsv_host.cpp
│   ├── spsv_host_utils.h
│   └── spsv_torch_compat.h
└── kernel/
    ├── spsv_kernel.cpp
    ├── spsv_kernel.h
    ├── spsv_tiling_data.h
    ├── spsv_analysis.h
    ├── spsv_solve.h
    └── 数值与内存辅助头文件
```

| 文件 | 职责 |
| --- | --- |
| `host/spsv_host.cpp`、`host/spsv_host_utils.h` | 公开接口、参数和内存范围校验、状态管理 |
| `host/spsv_torch_compat.h` | 已加载框架的Host队列互操作 |
| `kernel/spsv_kernel.cpp`、`kernel/spsv_kernel.h` | Device入口、启动实现及供Host调用的启动声明 |
| `kernel/spsv_tiling_data.h` | Host/Device共用的参数结构、workspace偏移及UB容量约束 |
| `kernel/spsv_analysis.h` | 四格式遍历、索引校验、CSR规范化、values置换及Analysis状态写回 |
| `kernel/spsv_solve.h`、`kernel/spsv_compact_solve.h`、`kernel/spsv_update.h` | 常规、对角和紧凑求解，以及GENERAL/DIAGONAL更新 |
| `kernel/spsv_kernel_memory.h` | UB缓存、DMA访问和原地输入保护 |
| `kernel/spsv_binary64.h`、`kernel/spsv_expansion64.h` | 软件binary64及独立指数表示、多分量表示与运算 |
| `kernel/spsv_dd_math.h`、`kernel/spsv_numeric.h`、`kernel/spsv_exact_sum.h` | 底层算术、求解数值辅助与精确累加 |
| `kernel/spsv_unit_precision.h`、`kernel/spsv_unit_integer.h`、`kernel/spsv_unit_vector_math.h` | 误差上界与精度判定、整数运算，以及向量进位与运算 |
| `kernel/spsv_scaled_solution.h`、`kernel/spsv_wide_solution.h` | 缩放依赖状态与多分量依赖状态，各自独立维护 |

启动声明只依赖tiling数据；Host通过`kernel/spsv_kernel.h`调用启动接口。
Kernel内部按底层算术、精确累加、误差判定和求解状态组织，数值辅助头文件不反向包含启动声明。
