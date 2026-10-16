# SpGEMM arch22 实现说明（Atlas A2 / A3，dav-2201）

本目录为 SpGEMM 在 Atlas A2 / A3（`NPU_ARCH=dav-2201`）上的实现，使用 **Ascend C 向量
编程模型**（`TPipe` / `TQue` / `DataCopyPad`），与仓内 `spmm/arch22` 一致。

硬件无关的描述符、状态机与入参校验位于 `../common/`，本目录只负责 arch22 的 tiling
规划、workspace 布局与 Kernel 下发。

## Kernel 划分

| Kernel | 阶段 | dtype 相关 | 职责 |
|--------|------|-----------|------|
| `spgemm_arch22_count` | WorkEstimation | **无关** | `rowProducts[r] = Σ_{k∈A.cols(r)} nnz(B.row(k))`，int64 计数 |
| `spgemm_arch22_symbolic` | Compute 第 1 趟 | **无关** | k 路归并去重求 `rowNnz[r]`；两级前缀和产出 `C.rowOffsets` |
| `spgemm_arch22_numeric_{fp32,fp16,bf16,c64}` | Compute 第 2 趟 | 模板特化 | 归并累加写出 `colIndices` / `values` |

三个 Kernel 均为 `KERNEL_TYPE_AIV_ONLY`（纯向量/标量，不用 Cube），blockDim 由 Host 侧
`GetAivCoreCount()` 运行时查询。

**符号阶段与 dtype 完全无关**——只读 A/B 的 `rowOffsets`/`colIndices`，不读 `values`。
这是 complex64 增量成本低的根本原因：新增类型只需扩展数值 Kernel 的模板实例。

## 分核策略

按每行中间乘积数 `P_i` 做**连续区间等工作量划分**，边界写入 `binEdge[blockDim+1]`。

之所以用连续区间而非 `spmm` 式的 reorder 贪心装箱：SpGEMM 的输出行必须按行号连续写入
`C.colIndices`/`C.values`（偏移由前缀和给出），连续区间让每核的输出也是连续的一段，
写回可批量 `DataCopyPad`，且 B 行访问在核内保持局部性。长尾行分布下，等工作量划分
同样能把「少数超大行」单独切成一个核的区间，达到与贪心装箱相同的均衡效果。

每行至少计 1 个单位成本，保证全零行也被均匀摊开，避免「某核拿到 M 个空行、另一核拿到
全部非空行」的退化划分。

## 两级前缀和

`C.rowOffsets` 的生成在符号 Kernel **内部**完成，不额外下发 kernel：

1. 各核串行扫描自己的行区间，得到局部总和写入 `coreSum[blockId+1]`；
2. `SyncAll()` 后，每核累加前序各核的局部总和作为基址偏移，写出全局 `C.rowOffsets`；
3. `C.rowOffsets[M]` 即 `nnz(C)`，由 Host 一次 D2H 读取。

`blockDim ≤ 64`，故第 2 步的标量累加开销可忽略。

## UB 预算（192KB，预留 24KB 给编译器临时与栈）

### T1 归并容量 `mergeCapacity`

单行归并所需 UB（元素数 cap）：

| 缓冲 | fp32 / fp16 / bf16 | complex64 |
|------|-------------------|-----------|
| B 段列索引暂存 | cap × 4 | cap × 4 |
| B 段值暂存 | cap × 4 | cap × 8 |
| 归并输出列索引 | cap × 4 | cap × 4 |
| 归并输出值（fp32 累加器） | cap × 4 | cap × 8 |

`mergeCapacity = floor(可用UB / perElem)`，32 元素对齐，上限 4096。complex64 时
`perElem` 从 16B 增至 24B，容量约为实数路径的 2/3——**complex64 使容量自动缩小**，
这是最容易踩的坑，故由 `SpgemmArch22MergeCapacity()` 按 dtype 字节宽动态计算，
绝不写死常量。

### T3 列分块宽度 `chunkWidth`

稠密累加器每列需 `accBytes` 值 + 4 字节命中标记：

```
chunkWidth = floor((可用UB / 2) / (accBytes + 4))
```

complex64 的 `accBytes` 为 8，故 `chunkWidth` 自动缩小。UB 占用固定，**与 N 无关**，
因此任意 N 与任意膨胀行都不会溢出 UB。

## workspace 布局（64B 对齐）

### buffer1（WorkEstimation 申请，Copy 结束前不得释放）

```
[header 64B][SpgemmArch22TilingData][binEdge int32×(blockDim+1)]
[rowProducts int64×M]
[rowNnz int32×M][coreSum int32×(blockDim+1)]
[scratchBase int64×(blockDim+1)][bApStride int32×max(K,1)]
[bHead int32×max(K,1)][bStat int32×(blockDim×16)][sync int32×(blockDim×8)]
```

各段分别按 64B 对齐。Host 对 A/B/C 仍执行完整 CSR 内容校验（偏移单调、列索引范围、
行内非降），但不再按 A 行长度或 complex64 的列间距收紧 B 的 Band 证书。

Band 建段只解析 B 的段范围和首列；成功的向量发射直接批量搬运 A values。若行回退到
标量或 General 归并，Kernel 才一次性将对应 A values 装入 `seg.val*`。因此 da=1/5、
complex64 da=8 的非等差段首，以及其它未命中向量路径的行都能使用正确值，同时正常
Band 快路径不会重复读取 A values。

WorkEstimation 末尾仅上传分核边界，不再次上传 tiling。
乘积数量和容量查询使用 host 描述符与 `rowProducts`；Compute 执行前重新构造并
完整上传 tiling。因此普通和风险输入均省去一次没有消费者的 H2D 上传。

### buffer2（Compute 申请，Copy 结束前不得释放）

```
[C.rowOffsets int32×(M+1)][C.colIndices int32×nnzUB][C.values sizeof(T)×nnzUB]
```

`nnzUB` 为 nnz(C) 上界，逐行取 `min(P_i, N)` 之和（再加 `beta != 0` 时的 `nnz(C_in)`）。

用 `min(P_i, N)` 而非直接用 `P_i`：高膨胀行的 `P_i` 可能远大于 N（同一列被多次命中），
去重后每行最多 N 个非零。这个上界比 `numProds` 紧得多，直接决定 buffer2 的大小——
对任务书最大用例（`nnz(C)=6.7e7`）是必要的，否则 workspace 会按 `numProds` 超额申请。

## 边界处理

| 场景 | 处理 |
|------|------|
| M=0 或 N=0 | `rowOffsets` 全零，`nnz(C)=0`，不启动符号/数值 Kernel |
| K=0 且 beta=0 | 同上 |
| K=0 且 beta!=0 | 保留 `C_in` 的 CSR 结构，`nnz(C)=nnz(C_in)`，数值为 `beta*C_in` |
| nnz(A)=0 或 nnz(B)=0（且 beta=0） | 同上 |
| A 某行为空 | `rowNnz[i]=0`，归并直接跳过 |
| B 中被引用行为空 | 该段长度 0，归并自动忽略 |
| 无交集乘积 | `rowNnz[i]=0` |
| 列索引越界 | Kernel 内防御性跳过，不参与计算（Host 已校验维度） |
| 数值抵消为 0 | **保留显式零**（结构由坐标决定，不做数值判断） |
| INF/NAN 输入 | 按 IEEE754 正常传播，结构不受影响 |

## 与 A5（arch35）共存

- `common/` 只放硬件无关逻辑，A5 的 SIMT 细节不进入公共层；
- `arch22/` 与 `arch35/` 目录互不重叠，CMake 依 `SOC_VERSION` 自动挑选
  （`ascend910b*` / `ascend910_93*` → `arch22`）；
- 唯一共享文件 `include/cann_ops_sparse.h` 沿用同一套 `aclsparseSpGEMM*` 声明，
  不重复新增同名或同功能接口；
- 核数不写死：一律走 `GetAivCoreCount()` 运行时查询，tiling 参数按查询结果推导，
  故 A2 上调好的实现在 A3（核数与带宽不同）无需改代码。
