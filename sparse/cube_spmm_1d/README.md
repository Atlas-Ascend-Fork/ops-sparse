# Cube SpMM 1D

## 算子概述

`cube_spmm_1d` 使用 Cube 计算单元实现稀疏矩阵与稠密矩阵乘法：

```text
C = A × B
A: M × K，有限 FP16
B: K × N，有限 FP16
C: M × N，FP32
```

原生稀疏格式为 d=16 的垂直 vector-CSR：每个非零向量对应同一列中的连续 16 行。
同时提供 Host 标准 CSR 数组的准备接口，转换后使用相同设备计算与自适应配置规则。
本算子与 `spmm`、`cube_spmm` 并存，不提供转置、alpha/beta 缩放、批处理或任意 leading dimension。

## 产品支持情况

| 产品 | 源码支持情况 |
| --- | --- |
| Atlas A2 训练/推理系列（Ascend 910B） | 支持，`arch22` / DAV-2201 |
| Atlas A3 训练/推理系列（Ascend 910_93） | 支持，`arch22` / DAV-2201 |
| Ascend 950PR / Ascend 950DT | 不支持，无 `arch35` 实现 |
| Ascend 310P | 不支持，无 `arch20` 实现 |

上述支持范围按源码与 SOC 构建选择说明。当前板上验证仅覆盖 Ascend910B3，
不代表其他型号已完成板测或具有相同性能。未选择 `arch22` 的构建不编译或导出本接口族，
不能在 950 上仅凭公共头中的声明调用它们。
软件环境与源码版本配套要求以[仓库 README](../../README.md)中的环境准备说明为准。

## 接口与调用顺序

公共声明位于 [`include/cann_ops_sparse.h`](../../include/cann_ops_sparse.h)，
链接 `libops_sparse.so` 和 CANN 运行库。各接口返回 `aclsparseStatus_t`。

| 接口 | 功能 |
| --- | --- |
| [aclsparseCubeSpmm1dCreatePlan](#aclsparsecubespmm1dcreateplan) | 从 Host vector-CSR 创建并上传不可变 A |
| [aclsparseCubeSpmm1dCreateCsrPlan](#aclsparsecubespmm1dcreatecsrplan) | 从 Host 标准 CSR 转换并准备同一计划 |
| [aclsparseCubeSpmm1dGetPlanInfo](#aclsparsecubespmm1dgetplaninfo) | 查询逻辑/物理尺寸、配置和内存需求 |
| [aclsparseCubeSpmm1dExecute](#aclsparsecubespmm1dexecute) | 在调用者 stream 上预清零 C 并异步计算 |
| [aclsparseCubeSpmm1dDestroyPlan](#aclsparsecubespmm1ddestroyplan) | 完成全部执行后释放计划及设备 A |

调用流程：

1. 初始化 ACL，设置设备/上下文，创建 stream 和公共 `aclsparse` handle，绑定 stream。
2. 选择一种准备接口，传入 Host 数组和初值为空的计划输出变量。
3. 调用 GetPlanInfo，按返回的物理字节数分配 B/C/workspace。
4. B 按 `paddedN` 行跨度填写，尾列补零；调用 Execute。
5. 在同一 stream 上排队后续操作，或由调用者同步 stream 后读取逻辑结果。
6. 所有相关执行完成后，在创建上下文中销毁计划，再清理剩余资源。

准备接口会复制输入、分配并上传设备 A，调用返回后 Host 数组可以释放。
A 在计划生命周期内固定；修改原数组不会更新计划，更新 A 须重新准备。
B 每次执行可以更换。计划、B、C 和 workspace 必须存活到相关 stream 上的工作完成。

## 计划选项与查询字段

`options == NULL` 使用结构自适应配置。显式选项各字段含义为：

| `aclsparseCubeSpmm1dOptions` 字段 | 要求 |
| --- | --- |
| `structSize` | `sizeof(aclsparseCubeSpmm1dOptions)` |
| `tilingKey` | `-1` 为自适应；非负值显式选择已编译的合法配置 |
| `reserved` | 必须为 0 |
| `workspaceLimitBytes` | 0 为无额外限制；非零值为设备 workspace 字节上限 |

自适应选择根据尺寸、稀疏行块结构和运行时核数确定配置与补齐。
特定融合选择保留设备与核数条件；不能将某型号的校准规则无条件套用于所有平台。

| `aclsparseCubeSpmm1dPlanInfo` 字段 | 含义 |
| --- | --- |
| `m` / `n` / `k` | 原始逻辑 M/N/K |
| `paddedM` / `paddedN` | 实际设备存储使用的 M/N |
| `tilingKey` / `blockDim` / `deviceId` | 计划配置、任务 block 数和准备时设备编号 |
| `vectors` | 设备 vector-CSR 非零向量数 P |
| `bBytes` | FP16 行主序 `[K,paddedN]` 的设备字节数 |
| `cBytes` | FP32 列主序 `[paddedM,paddedN]` 的设备字节数 |
| `workspaceBytes` | 调用者应提供的设备 workspace 字节数 |
| `persistentABytes` | 计划持有的设备 A 数组总字节数，不含 B/C/workspace |

所有字节数字段的单位为字节。始终使用查询结果，不以逻辑尺寸或自行推导的 key 替代。

## 算子接口详情

### aclsparseCubeSpmm1dCreatePlan

```c
aclsparseStatus_t aclsparseCubeSpmm1dCreatePlan(
    aclsparseHandle_t handle, int32_t m, int32_t n, int32_t k,
    const int32_t *hostOffsets, size_t offsetCount,
    const int32_t *hostColumns, const uint16_t *hostValues, size_t vectors,
    const aclsparseCubeSpmm1dOptions *options, aclsparseCubeSpmm1dPlan_t *plan);
```

**功能**：从 Host 原生 vector-CSR 创建计划并上传 A。

| 参数 | 方向/位置 | 说明 |
| --- | --- | --- |
| `handle` | IN，Host | 有效的公共库 handle，当前 ACL 设备/上下文已设置 |
| `m` / `n` / `k` | IN，Host | 逻辑尺寸，均为正；M 不超过 `INT32_MAX-15`，且为 16 的倍数 |
| `hostOffsets` | IN，Host | 非空 `int32_t[M/16+1]` 行块指针 |
| `offsetCount` | IN，Host | 必须为 `M/16+1` |
| `hostColumns` | IN，Host | `int32_t[P]` 列索引；P 为 0 时可以为空 |
| `hostValues` | IN，Host | `uint16_t[16*P]`，存储有限 IEEE binary16 位模式；P 为 0 时可以为空 |
| `vectors` | IN，Host | P，必须不超过 `INT32_MAX/16` |
| `options` | IN，Host | 可空的计划选项，见上表 |
| `plan` | OUT，Host | 非空输出地址，调用前 `*plan` 必须为空；成功后由调用者销毁 |

行指针首项为 0、单调非递减，末项等于 P。每个行块中的列索引严格递增且位于 `[0,K)`，
相邻行指针可以相等以表示空行块。每个向量的 16 个值连续存储。
各数组实际容量由调用者保证，长度和端点必须一致。

### aclsparseCubeSpmm1dCreateCsrPlan

```c
aclsparseStatus_t aclsparseCubeSpmm1dCreateCsrPlan(
    aclsparseHandle_t handle, int32_t m, int32_t n, int32_t k,
    const int64_t *hostOffsets, size_t offsetCount,
    const int64_t *hostColumns, const uint16_t *hostValues, size_t nnz,
    const aclsparseCubeSpmm1dOptions *options, aclsparseCubeSpmm1dPlan_t *plan);
```

**功能**：从 Host 标准 CSR 数组规范化并转换为 d=16 vector-CSR，创建同一设备计划。

| 参数 | 方向/位置 | 说明 |
| --- | --- | --- |
| `handle` | IN，Host | 与原生准备接口相同 |
| `m` / `n` / `k` | IN，Host | 正逻辑尺寸；M 不超过 `INT32_MAX-15`，无需为 16 的倍数 |
| `hostOffsets` | IN，Host | 非空 `int64_t[M+1]` CSR 行指针 |
| `offsetCount` | IN，Host | 必须为 `M+1` |
| `hostColumns` | IN，Host | `int64_t[nnz]` 列索引；nnz 为 0 时可以为空 |
| `hostValues` | IN，Host | `uint16_t[nnz]`，有限 FP16 位模式；nnz 为 0 时可以为空 |
| `nnz` | IN，Host | 实际存储的 CSR 元素数，包括重复项和显式零 |
| `options` | IN，Host | 与原生准备接口相同 |
| `plan` | OUT，Host | 非空输出地址，调用前 `*plan` 必须为空 |

所有索引从 0 开始；行指针首项为 0、单调非递减、末项为 nnz，列索引位于 `[0,K)`。
允许无序列、重复项、显式零和空行。转换步骤为：

1. 稳定按列排序，重复项按原输入顺序用 FP64 求和。
2. 结果以 RNE 重新量化为 FP16，去除精确零和量化零；非有限数或超出有限 FP16 范围的和被拒绝。
3. 每 16 行取列并集并补零，M 向上补齐到 16，得到实际设备 vector-CSR。

**重复项合并后重新量化可能与逐项 CSR 乘加有不同舍入结果**，并非对所有重复项无损。
一般 CSR 的列并集会增加存储量，转换后 P 同样不得超过 `INT32_MAX/16`；
预处理可能因范围、内存或 workspace 预算不足失败，不将普通稀疏模式转换成性能收益保证。

### aclsparseCubeSpmm1dGetPlanInfo

```c
aclsparseStatus_t aclsparseCubeSpmm1dGetPlanInfo(
    aclsparseCubeSpmm1dPlan_t plan, aclsparseCubeSpmm1dPlanInfo *info);
```

**功能**：将计划字段写入调用者的 Host 结构体，不启动设备任务。

| 参数 | 方向/位置 | 说明 |
| --- | --- | --- |
| `plan` | IN，Host | 成功准备且尚未销毁的非空计划 |
| `info` | OUT，Host | 非空输出结构体地址，各字段见“计划选项与查询字段” |

成功返回 `ACL_SPARSE_STATUS_SUCCESS`；计划或输出参数为空等情况返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。

### aclsparseCubeSpmm1dExecute

```c
aclsparseStatus_t aclsparseCubeSpmm1dExecute(
    aclsparseHandle_t handle, aclsparseCubeSpmm1dPlan_t plan,
    const void *b, size_t bBytes, void *c, size_t cBytes,
    void *workspace, size_t workspaceBytes);
```

**功能**：在 handle 当前绑定的 stream 上先清零完整物理 C，再异步提交 `C=A×B`。

| 参数 | 方向/位置 | 说明 |
| --- | --- | --- |
| `handle` | IN，Host | 有效公共 handle，绑定调用者要使用的 stream |
| `plan` | IN，Host | 当前设备/上下文中成功准备且仍有效的计划 |
| `b` | IN，Device | 有限 FP16，行主序 `[K,paddedN]`，32 字节对齐，非空 |
| `bBytes` | IN，Host | B 实际可用容量，至少为 `info.bBytes` |
| `c` | OUT，Device | FP32，列主序 `[paddedM,paddedN]`，32 字节对齐，非空 |
| `cBytes` | IN，Host | C 实际可用容量，至少为 `info.cBytes` |
| `workspace` | IN/OUT，Device | 32 字节对齐的非空工作区 |
| `workspaceBytes` | IN，Host | 工作区实际可用容量，至少为 `info.workspaceBytes` |

- B 的元素位置为 `B[i*paddedN+j]`，`j >= N` 的尾列必须为零；B 可以逐次更换，调用者每次保证全部值有限。
- C 的元素位置为 `C[j*paddedM+i]`；逻辑结果只取原 M×N，空行与补齐区为零，不保留 C 旧值。
- B/C/workspace 位于计划准备时的设备/上下文，彼此不得重叠，也不得与计划持有的 A 重叠。
- Execute 不分配设备内存，不进行张量 H2D/D2H。返回成功表示已提交，不表示设备已完成。
- 调用者同步 stream 检查异步错误；即使后续提交失败，也须确保已排队工作完成后再释放相关资源。
- 可复用不可变计划；并行调用需使用独立可写 C/workspace，并自行保证 stream 依赖顺序。

### aclsparseCubeSpmm1dDestroyPlan

```c
aclsparseStatus_t aclsparseCubeSpmm1dDestroyPlan(aclsparseCubeSpmm1dPlan_t plan);
```

**功能**：释放计划及其设备 A，不释放调用者的 B/C/workspace，也不隐式等待 stream。

`plan` 为 Host 输入句柄，空计划是成功的 no-op。非空计划必须在创建上下文中销毁，
且调用者须先完成所有使用它的设备工作。成功销毁后不得继续查询、执行或再次销毁同一非空句柄。
上下文不匹配等错误返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
设备 A 释放失败返回 `ACL_SPARSE_STATUS_EXECUTION_FAILED`，计划保留未释放缓冲区；
调用者须保留原句柄，在创建上下文中重试 DestroyPlan，已释放的缓冲区不会重复释放。
首次开始销毁后，不得再查询或执行该计划；只有销毁全部成功后句柄才失效。

## 返回值与调用责任

| 状态码 | 典型含义 |
| --- | --- |
| `ACL_SPARSE_STATUS_SUCCESS` | 准备/查询/销毁成功，或执行已成功提交 |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | 需要 handle 的接口传入空 handle |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | 非法尺寸、数组/选项、配置 key、对齐、别名或上下文等 |
| `ACL_SPARSE_STATUS_NOT_INITIALIZED` | 准备时未取得有效 ACL 上下文 |
| `ACL_SPARSE_STATUS_ALLOC_FAILED` | Host/设备 A 分配或上传失败 |
| `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` | 补齐/大小超限、配置或 workspace 预算不满足、执行容量不足等 |
| `ACL_SPARSE_STATUS_EXECUTION_FAILED` | 设备能力查询、输出清零、任务提交或设备 A 释放等运行时操作失败 |
| `ACL_SPARSE_STATUS_INTERNAL_ERROR` | 非预期 Host 异常或内部状态错误 |

以上为常见状态，调用者应检查每个 API 和 ACL 返回值。参数中的实际地址可用性、
数组容量、Device B 的有限性和补齐、跨 stream 生命周期仍由调用者保证，不能靠非空检查替代。

## 构建、测试与示例

环境配置、构建/安装、CSV/GTest 测试、配置校验和计时边界见
[测试与运行说明](../../test/cube_spmm_1d/README.md)。
独立 C 示例见 [`cube_spmm_1d_csr.c`](../../examples/cube_spmm_1d/cube_spmm_1d_csr.c)。
运行时必须一并部署 `libops_sparse.so` 及其相邻 `cube_spmm_1d_kernels/` 设备文件目录。
