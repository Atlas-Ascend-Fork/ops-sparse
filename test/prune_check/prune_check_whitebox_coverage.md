# aclsparseLtSpMMAPruneCheck 白盒测试分支覆盖说明

> 交付时间：2026-10-08（批量支持更新：2026-10-09；B 侧批量补格更新：2026-10-13；
> 第三批检视修复更新：2026-10-13——d_in 对齐 16B→32B（E7b 新增）、
> tiling 删 groupSize 字段（kernel 按 sizeof(U) 推导）、alongRow 改 bool、
> kernel 热循环除模重构为增量推进、host 校验/注释同步；
> padding ld 补格更新：2026-10-14——检视意见 193726682：ld > 物理行长的
> 正向路径零覆盖，新增 PruneCheckPaddingLdTest 4 条（RowScan/ColScan ×
> 合法/违规两态）+ wrapper ldOverride 扩展，K31 登记 srcStride padding
> 消费点）。
> 环境：Ascend950PR_9589
>（NpuArch=3510，**64 AI Vector**）、CANN 9.2.0（环境入口
> `source /usr/local/Ascend/cann/set_env.sh`，非默认安装路径用
> `source ${install_path}/cann/set_env.sh`）。
>
> 覆盖对象：
> - `sparseLt/prune_check/arch35/prune_check_kernel.cpp`（449 行：u32/u16/u8 kernel + launcher，含 batch tile 领取）
> - `sparseLt/prune_check/arch35/prune_check_host.cpp`（475 行：validate/tiling/launch 三段式，含批量参数校验）
>
> 批量支持变更（2026-10-09）：原 D-3「numBatches>1 → NOT_SUPPORTED」由批量
> 语义取代（d_valid = OR(全部 batch 判定)）；校验表编号同步调整——原 #5
>（numBatches 拒绝）删除，#5=dtype 白名单（原 #6）、#6=批量参数（新增）。
> B 侧批量补格（2026-10-13）：批量参数经稀疏侧描述符装配（DeriveBatchParams
> 读 sparse->numBatches/batchStride，B 侧走 md->matB 的 batchStride 装配链），
> 此前批量用例全部为 A 侧（B 侧非批量 18 条、A 侧批量 12 条均有，B 稀疏×batch
> 为正交覆盖缺口）；新增 wrapper `PruneCheckNpuBatchBSparse`（B 侧批量执行链）
> 与 4 条 B 侧批量用例（16 条批量用例 A/B 两侧分别承载）。
> 行号基准：本文行号为批量支持后版本；分支内容与映射关系随代码同步更新。
>
> 本说明与用例的映射关系为交付件的一部分；用例改动时须同步更新。

## 1. 交付物清单

| 交付物 | 位置 | 条数 | 内容 |
|---|---|---|---|
| CSV 白盒行（level=WB） | `arch35/prune_check_test.csv` case 101-112 | 12 | kernel 侧分支：tiling 列尾块 / grid-stride / 扫描几何×dtype / groupMask / 多写者 |
| CSV 最小对齐粒度边界行（level=L1） | `arch35/prune_check_test.csv` case 69-80 | 12 | 各 dtype 32B/elem 最小对齐倍数（且非更大粒度倍数：FP32{8,24}/BF16{16,48}/INT8{32,96}）shape × 耦合盲区：BF16×B 侧四组合全扫描（T+COL 两态必含）、INT8×B 侧 T+ROW+N+COL 两态、{BF16,INT8}×T+COL×A 侧、FP32×B 侧+T+COL，两态均衡（合法/违规各 6 条）；构造性实测 35/35 PASS（BF16 四组合均可构造；INT8 B 侧仅 T+ROW/N+COL 可构造，T+COL/N+ROW 被 `IsValidOpLayoutCombination` 拒） |
| TEST_F 白盒用例 | `arch35/prune_check_test.cpp` `PruneCheckWhiteboxTest` | 10 | host 侧校验表 #3(matB)/#6(numBatches<1 防御)/#9/#11/#12 防御分支 + 跨返回码有序性 + 错误注入（含 ld 上界断言） |
| TEST_F 批量语义用例 | `arch35/prune_check_test.cpp` `PruneCheckBatchTest` | 16 | A 侧 12 条：E13 重做（numBatches=2 全 batch 合法→SUCCESS/d_valid=0）+ 批量功能（batch0/末 batch/全部违规、T+ROW 转置物理、FP32、padding batchStride、numBatches=16 跨 batch grid-stride）+ 批量参数拦截（batchStride 0/负/非 32B 对齐、#11 全量跨度溢出、tiling totalTiles 溢出）+ numBatches 一致性（描述符层）；**B 侧批量补格 4 条（B 侧装配链：BATCH_STRIDE 落 matB，host 读 md->matB）**：numBatches=2 全 batch 合法（FP16 N+COL，ColScan×batch）→ 0、T+ROW 转置物理×3 batch 仅末违规 → 1、batchStride 非 32B 对齐（B 侧装配）→ INVALID_VALUE、FP32 N+COL 仅第二 batch 违规 → 1 |
| TEST_F padding ld 用例 | `arch35/prune_check_test.cpp` `PruneCheckPaddingLdTest` | 4 | **ld > 物理行长的正向路径（检视意见 193726682 补格）**：FP16 128×144、ld=160（每行 padding 16 元素非零垃圾），N+ROW（RowScan）/N+COL（ColScan）A 侧各 2 条 —— 数据区合法 + padding 全非零 → d_valid=0（padding 忽略正向佐证）、数据区违规 + padding 全非零 → d_valid=1（两态载体）；golden 双口径交叉（逻辑 golden == PruneCheckGoldenBatch 物理 ld golden）；wrapper 侧 `SparseSideSpecOf`/`PrepareSparseSideData`/`PruneCheckNpu(BSparse)` 扩展 ldOverride 透传 |
| 分支覆盖说明（本文） | `prune_check_whitebox_coverage.md` | — | 分支清单、映射、不可达项说明 |

合并去重原则：12 条 WB CSV 行的 (dtype, shape, op, order, side, pattern) 组合均与
既有 68 条 CSV 不同（逐一核对，见 §4 各行说明）；12 条最小对齐粒度边界 L1 行
（case 69-80）同样与既有 80 条 CSV 及彼此无重复（既有 BF16×B 侧 0 条、FP32×B 侧
0 条、FP32 T+COL 仅合法态、INT8 B 侧 T+ROW 仅合法态 / N+COL 仅违规态，均系盲区
增量）；TEST_F 白盒 10 条补充的均为既有 TEST_F 未覆盖的防御分支（#3 的 matB
操作数、#6 的 numBatches<1、#9 全部子条件（含 ld 上界）、#11、#12），批量
A 侧 12 条为批量支持新增语义/拦截路径（与既有单 batch 用例无重复），不重复
E1-E14 已覆盖项；**B 侧批量 4 条为 B 稀疏×batch 正交补格（A 侧批量 12 条均
A 稀疏侧，B 侧非批量 18 条均 numBatches=1，该组合既有 0 条）——批量参数装配
侧别（BATCH_STRIDE 落稀疏侧 matB）与描述符派生（m=md_k、k=md_n）为 B 侧链
独立节点，kernel 判定链单 batch 语义不变**；**padding ld 4 条为 ld>物理行长
维度补格（全部既有 136 条 ld 恒等于物理行长——wrapper 自动推导，该维度既有
0 条）：(dtype, shape, op, order) 组合与既有 CSV 重复（FP16 N+ROW/N+COL
128×144），但 ld=160 padding 布局为独立覆盖维度（K31），且 padding 区非零
垃圾 + 双态断言为该维度专属构造，不与既有用例语义重复**。

## 2. 设计基线（tiling 推导依据）

host tiling（`prune_check_host.cpp` L348-399）：`kTileRows=16`、`kMaxBlockDim=64`、
`kUbLimit=248KB`、`kUbReserve=4096`、reduceBuf=256B。本机 AIV 核数 64。
批量字段：`numBatches`（≥1，稀疏侧描述符）、`batchStride`（numBatches>1 时
须 >0 且为 32B/elem 元素倍数）。

| dtype | lanes(=256B/elem) | maxCols(budget=249600B) | 分组 g/keep（kernel 按 sizeof(U) 编译期推导） |
|---|---|---|---|
| FP32 (u32) | 64 | 3840 | 2 / 1（1:2） |
| FP16/BF16 (u16) | 128 | 7680 | 4 / 2（2:4） |
| INT8 (u8) | 256 | 15360 | 4 / 2（2:4） |

关键公式：`tileCols = max(lanes, min(ceil(cols/lanes)×lanes, maxCols))`；
`blockDim = clamp(min(64 核, totalTiles), 1, 64)`；
`tilesPerMatrix = ceil(rows/16) × ceil(cols/tileCols)`；`totalTiles = tilesPerMatrix × numBatches`。

> 对 UB 更小的设备同样成立：maxCols 只会更小，"cols > maxCols"（混合列块）与
> "cols 非 lanes 倍数"（列尾清零）的分支条件结论不变；描述符层强制
> rows/cols/ld 为 32B/elem 元素倍数（FP32:8 / FP16,BF16:16 / INT8:32），
> 本表所有 WB shape 均满足（可构造），且 `blockLen = validCols×elem` 恒为 32B 倍数。

## 3. 分支清单与用例映射

### 3.1 kernel 侧（prune_check_kernel.cpp）

| # | 分支 / 路径 | 源码位置 | 覆盖用例（含既有） |
|---|---|---|---|
| K1 | `GroupViolation2`（u32 判定链，keep=1） | L48-65 | 既有 FP32 全部（3,4,13,14,19,29,49,56,67）+ WB 101/102/103/108/111 + 批量 FP32（MultiBatchFp32SecondBatchViolation / **B 侧 MultiBatchBSparseFp32ColScanSecondBatchViolation**） |
| K2 | `GroupViolation4` `sizeof(U)==1`（INT8 省幅值 And） | L79-83 | 既有 7,8,16,50,58,62,68 + WB 110/112 |
| K3 | `GroupViolation4` else（u16 幅值 And 链） | L84-94 | 既有 FP16/BF16 全部 + WB 104-107/109 + 批量 FP16 全部（含 **B 侧 MultiBatchBSparse\* FP16 两态**） |
| K4 | `PruneCheckClearInput` 清零循环体 | L110-119 | 所有列尾块用例（见 K11）执行 |
| K5 | RowScan `groupSize==2`（单级 DeInterleave） | L162-165 | 同 K1 |
| K6 | RowScan `groupSize==4`（三级 DeInterleave） | L166-172 | 同 K2/K3 |
| K7 | **groupMask `VL32`**（groupLanes=32：u32/u16 RowScan 前缀子集掩码） | L141-142 | 既有 FP32/FP16/BF16 的 N+ROW、T+COL、T+ROW 用例 + WB 101-107 + 批量 N+ROW/T+ROW（含 B 侧 T+ROW） |
| K8 | **groupMask `VL64`**（groupLanes=64：仅 u8 RowScan） | L143-144 | 既有 7,8,16,50,58,68 + **WB 110**（每行 63 chunk，多次累或；256 tiles 多轮） |
| K9 | RowScan 行内多 chunk（offset 循环 ≥2 轮） | L156-176 | 既有 29,65 等 + WB 103/107/110 |
| K10 | ColScan 列向多 chunk（offset 循环 ≥2 轮） | L199-214 | 既有 13（FP32 T+COL，2 chunk）+ **WB 109（u16 2 chunk）/ 112（u8 2 chunk）** —— u16/u8 ColScan 多 chunk 为既有缺口 |
| K11 | **O2 列尾块条件预清零**（`validCols < tileCols` → ClearInput） | L365-369 | per 实例化：u32 既有 67 + WB 101/111；u16 既有 63 + WB 104 + **padding ld P1-P4（144 < tileCols256，u16×ld padding 组合）**；u8 既有 68 + WB 110；L1 边界 69-80（各 dtype 最小对齐粒度 cols<tileCols 全走清零，含 B 侧/T+COL 耦合） |
| K12 | **整块搬运跳过清零**（`validCols == tileCols`） | L365 条件为假 | per 实例化：u32 既有 3 + WB 102/103(tile0) + **批量 B 侧 FP32（64=tileCols）**；u16 既有 1 + WB 107(tile0) + **批量 B 侧 FP16（128=tileCols）**；**u8 既有无（全部 INT8 用例 cols≤64 < tileCols256 走清零）→ WB 110(tile0) 补齐** |
| K13 | grid-stride 单轮（blockDim ≥ tpm：block j < tpm 每 batch 领 1 tile，j ≥ tpm 空转；totalTiles < blockDim 时部分核未启动） | L322/L326 | 既有 63/67/68（blockDim=1/1/2）+ WB 101/104/111/112（blockDim=2/1/2/2）+ L1 边界 69-80 + 批量 numBatches=2/3（tpm=8 < blockDim=16/24：block j<8 跨 batch 每 batch 1 tile、j≥8 空转；含 **B 侧 tpm=16/24/8**） |
| K14 | **grid-stride 多轮领 tile**（tpm > blockDim：while 内每 block ≥2 tile 增量推进） | L326 | 既有 27/28/29（128/128/256 tiles）+ **WB 103/105/106/110** + **批量 MultiBatchGridStrideAcrossBatches（numBatches=16、tpm=8 < blockDim=64：block j<8 每 batch 1 tile 共 16 tile 跨 batch 交错 —— batch 维度领取的多轮载体）** |
| K15 | **V_S 事件同步**（`PipeBarrier<PIPE_V>` + `SetFlag/WaitFlag<V_S>`，acc→标量读可见性） | L333-334 | 全部 SUCCESS 用例必经（每 tile 一次）；WB 103/105/106/110 + 批量 16-batch 用例每 block 执行 ≥2 次（连续 tile 的重复同步） |
| K16 | V_MTE2 WAR 保护（下一 tile 覆写 inputBuf 前）+ 清零后 DMA 可见性 | L339 / L368 | 多轮用例（27/28/29、WB 103/105/106/110、批量 16-batch）+ 列尾块用例（K11 全体） |
| K17 | `blockViolation` 跨轮累积（后处理 tile 才违规，false→true） | L298, L335-337 | 既有 27（N+ROW，第 2 轮）+ **WB 105（block63 第 2 轮 tile127）** + **批量 MultiBatchGridStrideAcrossBatches（block j<8 第 16 个 tile 落末 batch，跨 batch 累积）** |
| K18 | **违规 block 条件写 d_valid 多写者**（`SetValue(0,1)`，写同值） | L353-355 | 既有 26（8 写者单轮）/2/8/64 + **WB 106** + **批量 MultiBatchAllBatchesViolating（跨 batch 多写者）** |
| K19 | 非违规 block 不写（memset 预清零保持 0） | L353 条件为假 | 既有全部 VALID 用例（1,3,12 等）+ WB 102/103/104/107/110 + 批量 MultiBatchTwoBatchesAllValid（全 batch 合法 → d_valid=0，跨 batch 预清零保持）+ **B 侧 MultiBatchBSparseTwoBatchesAllValid（B 侧链全 batch 合法 → 0）** |
| K20 | `Compute alongRow != 0`（RowScan 分派） | L397-398 | 既有全部 N+ROW/T+COL/T+ROW + WB 101-107/110 + 批量 N+ROW/T+ROW（含 **B 侧 T+ROW**） |
| K21 | `Compute alongRow == 0`（ColScan 分派） | L399-400 | 既有 9,13,15,17,18,20,22,23,52,54,60,62 + WB 108/109/111/112 + **批量 B 侧 N+COL（MultiBatchBSparseTwoBatchesAllValid / MultiBatchBSparseFp32ColScanSecondBatchViolation —— ColScan×batch 维度首证，A 侧批量均为 RowScan）** |
| K22 | `MakeTilePosition` 行带尾（validRows < tileRows=16） | L245-246 | 既有 67（FP32 8×8）+ **WB 101/102/111** + L1 边界 77-79 |
| K23 | `MakeTilePosition` 列块尾（validCols < tileCols） | L247-248 | 同 K11（含 **padding ld P1-P4**：144 < 256 列尾块 × ld padding 的 srcStride 组合） |
| K24 | launcher `dataBytes==4 → prune_check_u32` | L442-443 | FP32 全部用例（含批量 FP32 A/B 两侧） |
| K25 | launcher `dataBytes==2 → prune_check_u16` | L444-445 | FP16/BF16 全部用例（含批量 FP16 A/B 两侧 + **padding ld P1-P4**） |
| K26 | launcher else `→ prune_check_u8` | L446-447 | INT8 全部用例 |
| K27 | `MTE2_V` 搬入→计算可见性同步 | L388 | 全部 SUCCESS 用例必经 |
| **K28** | **batch tile 领取派生**（外层 batch 循环 + 内层 tile-in-matrix grid-stride，`tim = blockIdx + k·blockNum`；行带/列块二维坐标计数器增量推进——`colChunk += colStep` 条件减法进位 + `rowBand += rowStep`，除法仅初始化每 kernel O(1) 次，热循环零除模；numBatches==1 时 batchIdx 恒 0 退化为单 batch，覆盖集合与原「线性 tileId 整除还原」实现等价） | L312-321, L340-349 | 全部用例必经（numBatches=1 退化路径由既有 92 条 CSV 承载）；**batchIdx>0 寻址由批量用例承载：MultiBatchLastBatchViolationTransposed（末 batch）/ MultiBatchFp32SecondBatchViolation（第二 batch）/ MultiBatchGridStrideAcrossBatches（batchIdx 0-15）+ B 侧对偶 MultiBatchBSparseLastBatchViolationTransposed（末 batch）/ MultiBatchBSparseFp32ColScanSecondBatchViolation（第二 batch）** |
| **K29** | **batch 基址偏移**（CopyIn 源 `tile.batch×batchStride + rowOffset×ld + colOffset`；host #6 保证 batchStride 32B/dataBytes 对齐 → 各 batch 源地址对齐性质与 batch 0 一致） | L383-386 | 同 K28 批量用例（含 T+ROW 转置物理 × batch 偏移组合、padding batchStride=20480 非紧凑布局；**B 侧批量含 T+ROW 转置物理 × B 侧 batch 偏移、N+COL ColScan × batch 偏移**） |
| **K30** | **Init 全量跨度含 batch 项**（`(numBatches−1)×batchStride + (rows−1)×ld + cols` 作 SetGlobalBuffer 长度） | L284-285 | 全部用例必经（numBatches=1 时与扩展前一致）；批量跨度语义由批量功能用例承载（A/B 两侧） |
| **K31** | **CopyIn srcStride padding 跳过（ld > 物理行长，padding 能力唯一消费点）**：`DataCopyExtParams.srcStride = (ld − validCols)×sizeof(U)` 使行间源步长跨过 padding 区 —— padding 数据不进入 UB / 不参与判定（README「ld 大于物理行长时 padding 区数据不参与判定」的实现载体）。既有全部用例 ld == 物理行长：单列块 shape（cols ≤ tileCols）srcStride=0、多列块 shape 的块间推进不涉及行尾 padding —— **ld > 物理行长的正向路径此前零覆盖（检视意见 193726682）** | L375-379 | **padding ld P1（RowScan）/ P2（ColScan）**：FP16 128×144、ld=160，数据区合法 + padding 区 [144,160) 全非零垃圾（1.0 位模式）→ d_valid=0（若 padding 参与判定，每行 padding 16 元素构成 4 个全非零组 / padding 列垂直组全非零 → 必然 d_valid=1，断言冲突即暴露）；**P3/P4（加固）**：数据区违规 + padding 全非零 → d_valid=1（两态载体，判定只看数据区）。golden 双口径交叉（逻辑 golden == PruneCheckGoldenBatch 物理 ld golden，padding 不进入分组判定） |

### 3.2 扫描几何 × dtype 全组合矩阵（kernel 实例化 × `Compute` 分派）

沿行方向 `alongRow = (trans != isRowOrder)`：N+ROW / T+COL → RowScan(R=m,C=k)；
T+ROW → RowScan(R=k,C=m，物理转置)；N+COL → ColScan(R=m,C=k)。
"✓"= 既有 68 条 CSV 已覆盖，"WB"= 白盒批新增（101-112），"L1b"= 最小对齐粒度
边界批新增（69-80），"—"= 不存在该组合。

| 几何 | FP32(u32) | FP16(u16) | BF16(u16) | INT8(u8) |
|---|---|---|---|---|
| N+ROW（RowScan） | ✓ 3,4,29,49,56,67 + WB101,102,103 + L1b78（B 侧） | ✓ 1,2,24-28,44-48,51,55,63-65 + WB104,106,107 | ✓ 5,6,57 + L1b80（B 侧） | ✓ 7,8,58,68 |
| T+COL（RowScan） | ✓ 13 + L1b77（违规态）,79（B 侧违规态） | ✓ 17,22,54 | ✓ 15 + L1b69,70（B 侧两态）,75（违规态） | ✓ 50 + L1b76（合法态） |
| T+ROW（RowScan，物理转置） | ✓ 14,19 | ✓ 10,21,30,53,66 + **WB105**（>64 tiles 多轮） | ✓ 61 + L1b72（B 侧） | ✓ 16 + **WB110**（B 侧大 shape 多轮，u8 整块跳过唯一可承载路径）+ L1b73（B 侧违规态，最小边界） |
| N+COL（ColScan） | **既有缺口 → WB108,111** | ✓ 9,18,20,23,52 | ✓ 60 + **WB109**（违规态+多 chunk）+ L1b71（B 侧违规态） | ✓ 62 + **WB112**（多 chunk+整块跳过+违规态）+ L1b74（B 侧合法态，最小边界） |

B 侧派生（`SelectSparseSide` sparse=matB 路径）：既有 11,16,20-23,62 + WB 110/112
+ L1 边界 69-74,78-80（BF16 B 侧四组合全扫描 T+COL/N+COL/T+ROW/N+ROW、
INT8 T+ROW/N+COL、FP32 N+ROW/T+COL，最小对齐粒度边界 shape —— 构造性
实测：BF16 无低精度布局约束四组合均可构造且 kernel 判定三向一致；
INT8 B 侧仅 T+ROW/N+COL 可构造）+ **B 侧批量 4 条（TEST_F
MultiBatchBSparse\*：批量参数 B 侧装配链，FP16/FP32 无低精度布局约束，
N+COL / T+ROW 均可构造）**。
WB 110/112 用 B 侧的原因（INT8 低精度布局约束 `IsValidOpLayoutCombination`：
B 侧仅 T+ROW（ROW\|ROW→opA=N,opB=T）与 N+COL（ROW\|COL→opA=N,opB=N）可构造，
A 侧 N+ROW/T+COL 的占位 matB 为 dense COL(k,128,ld=128)，k>128 时违反
`IsLdValid`（COL 序 ld≥rows）不可构造）：u8 RowScan 的大列路径（C>maxCols 的
整块跳过 + 多 chunk + 多轮）仅 B 侧 T+ROW 可承载；kernel 分派与扫描几何不受
侧别影响（侧别只决定 m/k 派生源）。

### 3.3 host 侧（prune_check_host.cpp，校验表 #1-#12 + tiling + launch）

> 校验表编号（批量支持后）：#1 handle / #2 matmulDescr / #3 matA·matB null /
> #4 无唯一稀疏侧（D-1）/ **#5 dtype 白名单（原 #6）/ #6 批量参数（新增，取代
> 原 #5「numBatches>1 → NOT_SUPPORTED」D-3 拒绝）** / #7 指针 / #8 对齐 /
> #9 维度防御复检 / #10 ld 下限（D-2）/ #11 全量跨度溢出（含 batch 项）/
> #12 launch。

| # | 校验表/分支 | 源码位置 | 覆盖用例 |
|---|---|---|---|
| H1 | #1 handle null / *handle null → HANDLE_IS_NULLPTR | L313-316 | 既有 E1 / E2 |
| H2 | #2 matmulDescr null / *null → INVALID_VALUE | L318-321 | 既有 E3 / E4 |
| H3 | #3 `matA==nullptr` 操作数 | L84-87 | 既有 E4b（伪造） |
| H4 | #3 `matB==nullptr` 操作数 | 同上（\|\| 第二操作数） | **WB ForgedNullMatB**（伪造，公共 API 不可构造） |
| H5 | #4 无唯一稀疏侧（双稀疏/双稠密）→ NOT_SUPPORTED（D-1） | L90-94 | 既有 E8b / E9b（伪造；描述符层 E8/E9） |
| H6 | #5 dtype 白名单（FP32/FP16/BF16/INT8 + default→NOT_SUPPORTED） | L155-169 | 四支持 dtype 由 CSV 全覆盖；default 分支既有 E10（INT32） |
| H7 | **#6 批量参数：numBatches<1 → INVALID_VALUE**（SetAttribute 拒绝 <1，防伪造复检） | DeriveBatchParams L113-116 | **WB ForgedZeroBatchesRejected**（伪造 0/-1；公共 API 不可构造） |
| H8 | **#6 批量参数：numBatches>1 时 batchStride≤0 / 非 32B/elem 倍数 → INVALID_VALUE**（公共 API 可构造） | DeriveBatchParams L119-138（DeriveCheckShape L171-174 调用，dtype #5 先行） | **批量 MultiBatchStrideZeroRejected（0）/ MultiBatchStrideNegativeRejected（负）/ MultiBatchStrideUnalignedRejected（16384+8 非 16 倍数）+ B 侧 MultiBatchBSparseStrideUnalignedRejected（batchStride 经 B 侧 matB 装配，host 读 md->matB 后拦截）**。B 侧批量功能用例（合法 batchStride → SUCCESS/d_valid 三向一致）同时证明 B 侧装配链无「读字段错侧」缺陷（错侧读 matA 默认 batchStride=0 会被 #6 拒绝而非 SUCCESS） |
| H9 | #7 d_in / d_valid null | L254-257 | 既有 E5 / E6 |
| H10 | #8 d_in 32B 对齐（实现要求，严于官方 16B 惯例：d_in 为 kernel DataCopyPad 的 GM 源基地址，硬件要求 32B；16B 非 32B 进入未定义搬运路径） | L258-266 | 既有 E7（非 16B/32B 对齐）+ **E7b（16B 对齐但非 32B → INVALID_VALUE，32B 修复新增）** |
| H11 | #9 `rows==0`/`cols==0`/`ld==0` | L268-283 | **WB ForgedZeroScanDimsRejected**（3 子条件逐一伪造；E12 仅覆盖描述符层拦截，PruneCheck 防御复检此前不可达） |
| H12 | #9 `rows/cols/ld % divisor != 0` | L270-273 | **WB ForgedNonDivisibleScanDimsRejected**（FP16 divisor=16，130 逐一伪造） |
| H13 | #9 `rows/cols/ld > int32Max`（ld 上界拦截巨 ld+小 rows 场景：该场景 spanBytes 落在 int64 内、绕过 #11 致 kernel 越界读） | L271-272 | **WB ForgedInt32OverflowDimsRejected** + **WB ForgedLdUpperBoundRejected**（P1 上界拦截 / P2 负向 / P3 含边；「≤INT32_MAX 不被上界误拦」由 WB-H5 承载） |
| H14 | #10 ld < 物理行长 → INVALID_VALUE（D-2） | L284-293 | 既有 E14（黑盒可构造 N+COL） |
| H15 | **#11 全量跨度溢出（含 batch 项）**：`(numBatches−1)×batchStride + (rows−1)×ld + cols` 逐项 CheckedMulAdd 拒绝（numBatches=1 时批量项为 0，与扩展前一致） | ValidateScanSpan L211-246（#11 拆分为独立函数承载）| 单 batch 溢出 **WB ForgedSpanOverflowRejected**（FP32 rows=ld=2^31-8）+ **批量 MultiBatchSpanOverflowRejected**（numBatches=2 + batchStride=INT64_MAX 向下取 16 倍数：#6 全过、批量项溢出 int64 —— 批量项子分支载体） |
| H16 | #11 第三子条件 `spanBytes > uintptr_max − dIn` | L227 | 不可达（见 §5） |
| H17 | tiling `coreCount==0` / UB 不足 → INSUFFICIENT_RESOURCES | L351-357 | 不可构造（见 §5） |
| H18 | tiling `maxCols==0` → INSUFFICIENT_RESOURCES | L362-366 | 不可构造（同上） |
| H19 | **tiling `tilesPerMatrix`/`totalTiles` CheckedMulAdd 溢出 → INVALID_VALUE** | L386-393 | **批量 MultiBatchTotalTilesOverflowRejected**（原「数学不可达」；批量支持后公共 API 可构造：INT8 rows=cols=ld=2^31−32 + numBatches=INT32_MAX + batchStride=32 → #6/#9-#11 全过（totalElems≈4.61e18 < int64max）但 tilesPerMatrix(≈1.88e13)×numBatches(≈2.15e9)≈4.0e22 溢出 int64 → tiling 拒绝；设备访问前返回） |
| H20 | tiling blockDim clamp：=1 / ∈(1,64) / =64 | L395-397 | =1：63,67,WB104；∈(1,64)：68,WB101/111/112,批量 numBatches=2/3（A 侧 16/24；B 侧 16/24/8）；=64：27,28,29,WB103/105/106/110,批量 16-batch（128） |
| H21 | 维度映射三分支（alongRowPrune / sparseTrans / else） | L187-199 | CSV 四几何全组合（§3.2）+ B 侧派生 + 批量 T+ROW（MultiBatchLastBatchViolationTransposed）+ **B 侧批量 N+COL/T+ROW（MultiBatchBSparse\*：B 侧 m=md_k、k=md_n 派生 × 批量维度）** |
| H22 | #12 前置线程错误 → EXECUTION_FAILED（不下发任务） | L410-415 | **WB PreExistingThreadErrorNotEnqueued**（触发器：已销毁 stream 上的 memset；断言拒绝 → `aclrtGetLastError` 清理 → 恢复证明） |
| H23 | #12 `aclrtMemsetAsync` 失败 → EXECUTION_FAILED | L416-420 | **WB DestroyedStreamMemsetRejected**（已销毁 stream 句柄同步查表失败） |
| H24 | #12 launch 后错误探测 → EXECUTION_FAILED | L444-449 | 不可确定性构造（见 §5） |
| H25 | 校验顺序即优先级（跨返回码可观测组合） | L308-337 编排 | **WB ValidationOrderPriority**：#1>#3（HANDLE_IS_NULLPTR 先于 INVALID_VALUE）、#4>#7、**#5>#6（INT32 dtype 与 numBatches=2/batchStride=0 并存 → NOT_SUPPORTED，dtype 先于批量校验）**、#5>#8（NOT_SUPPORTED 先于 INVALID_VALUE） |

> 同返回码相邻项之间的顺序（如 #8 vs #9、#9 各子条件之间）对外不可观测，不作
> 用例断言；顺序性以跨返回码组合（H25）承载。

## 4. 白盒用例与分支要点对照（任务 7 项逐一回应 + 批量支持增量）

| 任务分支要点 | 承载用例 |
|---|---|
| 1. groupMask VL32/VL64 + V_S 事件同步 | VL32：K7（既有 FP32/FP16/BF16 RowScan + WB101-107 + 批量 FP16/FP32）；VL64：K8（既有 INT8 RowScan + WB110 多 chunk 多轮）；V_S：K15（全部用例必经；WB103/105/106/110 + 批量 16-batch 每 block ≥2 次重复同步） |
| 2. O2 列尾块条件预清零 / 整块跳过 | K11/K12 per dtype 实例化：u32（67+WB101/102/103/111+批量 FP32）、u16（63+WB104/107+批量 FP16）、u8（68+**WB110 补整块跳过缺口**） |
| 3. grid-stride 多轮 / totalTiles<64 空转 | 多轮 K14：既有 27/28/29 + WB103/105/106/110 + **批量 16-batch（跨 batch 交错多轮）**；空转 K13：63/67/68 + WB104（blockDim=1）/101/111/112 + 批量 numBatches=2/3 |
| 4. RowScan/ColScan 几何全组合 | §3.2 矩阵：四几何全 dtype 覆盖；WB 补 FP32-N+COL（108/111）、BF16-N+COL 违规态（109）、T+ROW 多轮（105/110）；批量补 T+ROW 转置物理 × batch 偏移（MultiBatchLastBatchViolationTransposed） |
| 5. u32/u16/u8 kernel 分派全路径 | K24-K26：三 dtype 全部用例；WB 使三实例化各自同时具备 RowScan+ColScan+清零/整块+违规/合法；批量补 u32/u16 的 batch 维度实例化 |
| 6. host 校验表 #1-#12 有序校验 | §3.3：既有 E1-E14 覆盖 #1-#2、#3(matA)、#4、#7-#8、#10；WB 补 #3(matB)、#5(default)、#6(numBatches<1 防御)、#9（4 组子条件，含 ld 上界）、#11（单 batch）、#12（两分支）、有序性（H25）；批量补 #6 合法性拦截（batchStride 0/负/非对齐）、#11 批量项、tiling 溢出（H19） |
| 7. 违规 block 条件写 d_valid 多写者 | K18：既有 26（8 写者单轮）+ WB106（64 写者 × 2 轮）+ **批量 MultiBatchAllBatchesViolating（跨 batch 多写者）**；K17 跨轮累积：27 + WB105 + 批量 16-batch（跨轮+跨 batch） |
| 8.（批量增量）batch tile 领取 / 基址偏移 / OR 归约 | K28/K29/K30 + H7/H8：E13 重做（全 batch 合法→0）+ batch0/末 batch/全部违规 + T+ROW 转置物理 + FP32 + padding batchStride + numBatches=16 多轮 + 拦截 4 条（#6×3 + #11 批量项）+ tiling 溢出 + numBatches 一致性（描述符层）+ **B 侧补格 4 条（装配链 BATCH_STRIDE 落 matB：全 batch 合法 / 末 batch 违规 T+ROW / batchStride 非对齐拦截 / FP32 N+COL 第二 batch 违规）** |

## 5. 不可达 / 不覆盖项说明（防御性分支，无公共构造路径）

| 项 | 不可达原因 |
|---|---|
| H16（#11 第三子条件） | 64 位平台 `uintptr_max(2^64-1) > int64max(2^63-1)`，且前段 CheckedMulAdd 已保证 `spanBytes ≤ int64max`，`spanBytes > uintptr_max − dIn` 需 d_in 高于 2^64−1−2^63 ≈ 9.2e18，设备地址空间不可达 |
| H17/H18（平台资源不足） | `GetAivCoreCount/GetUbSize` 由平台保证非零/充足（本机 64 核 / UB≥248KB），无 API 可注入 |
| H24（launch 后错误探测） | 需 memset 成功后 kernel 启动失败（运行时故障注入），公共 API 无确定性构造；WB DestroyedStreamMemsetRejected 在 memset 阶段即被拒，kernel 未启动 |
| kernel grid-stride 0 tile block（已启动但空转） | 增量推进实现下 tpm < blockDim 时 blockIdx ≥ tpm 的 block 空转（多 batch 小矩阵已知限制：覆盖完整性不受影响——每个 (batch, tileInMatrix) 恰由 tim%blockDim 的 block 领取，全网格无丢失无重复，见 K13）；numBatches==1 时 host blockDim ≤ totalTiles=tpm 保证无空转；另存在"未被启动的核"（blockDim < 64，K13 承载） |

> 批量支持前 H19（tiling totalTiles 溢出）为「数学不可达」（单 batch 下
> rowBands×tilesPerRowBand ≤ 2^52 < int64max）；批量支持后 totalTiles 含
> numBatches（≤ INT32_MAX）因子，极端合法参数下乘积可溢出 int64，已转为
> 公共 API 可构造并由 MultiBatchTotalTilesOverflowRejected 承载（§3.3 H19）。

## 6. 执行与验收

- 构建/运行：`source /usr/local/Ascend/cann/set_env.sh && bash build.sh --ops=prune_check --run --soc=ascend950`（CANN 环境入口按实际安装路径调整，`${install_path}/cann/set_env.sh`；默认 ascend910b3 会因 ENABLE_SPARSELT 拦截跳过，须显式 `--soc=ascend950`）
- 预期规模：80（CSV 既有 L0/L1 68 + 最小对齐粒度边界 12）+ 12（CSV WB）+ 18（TEST_F 既有 = E 系列 17（含 E7b：d_in 16B 对齐非 32B → INVALID_VALUE）+ 缺陷暴露 1，E13 由批量重做取代）+ 10（TEST_F WB，含 ForgedLdUpperBoundRejected + ForgedZeroBatchesRejected）+ 16（TEST_F 批量：E13 重做 + A 侧 B 系列 11 + **B 侧批量补格 4**）+ **4（TEST_F padding ld：PruneCheckPaddingLdTest，检视意见 193726682 补格——RowScan/ColScan × 合法/违规两态，ld=160 > 物理行长 144）** = **140 条**
- 回归约束：全量 140 条必须全部通过（含 padding ld 补格前 136 条的零回归：CSV 92 条不变 + TEST_F 既有 44 条语义不变），d_valid 三向一致（EXACT 零容差）与返回码枚举断言全数满足。
- 实测结论：见 §8

## 7. 备注与已知限制

1. WB-H4（int32 上限伪造）依赖 `md_m/md_k` 的 `static_cast<int32_t>` 补码回绕
   （x86-64 gcc 定义良好行为），注释已注明。
2. WB-H7/H8（#12 错误注入）在套件尾部执行：gtest 按注册顺序（文件内声明顺序）
   运行，CSV（INSTANTIATE 于前）→ E 系列 → 缺陷暴露 → WB 组，错误态注入的
   潜在污染面收敛到 WB 组内部；WB-H7 内部含"清理 + 恢复证明"双重防污染断言。
3. **线程错误置位机制的构造性实测结论**（CANN 9.2.0）：参数校验类失败
   （如 `aclrtMemsetAsync(dst=nullptr)` 返回 100000）
   **不置**线程级 last error（peek=0）；运行时句柄查表类失败（已销毁
   stream/event，返回 107017）**同步置位**（peek=107017）；`aclrtGetLastError`
   读取并清除（清除后 peek=0）。WB-H7 触发器据此选用已销毁 stream 上的 memset。
4. level=WB 为 CSV `level` 列取值之一（仅用于 caseId 后缀与日志区分，框架对
   level 无枚举校验）；L0/L1 语义不变。
5. 构造决策记录（对后续维护者有用）：① INT8 B 侧 N+ROW 不可构造
   （`IsValidOpLayoutCombination`：ROW|ROW 需 opB=T），case110 选用可构造的
   `16000×2048 T+ROW(B)`（`32×16000 N+ROW(B)` 在 MatmulDescriptorInit 即被拒
   INVALID_VALUE）；② WB-H7 触发器须置位线程错误，dst=null memset 属参数
   校验类失败、不置位（见备注 3），选用已销毁 stream 上的 memset。
6. 最小对齐粒度边界用例（case 69-80）构造性实测结论（35/35 PASS，
   未发现算子缺陷）：
   ① BF16 不属 `IsLowPrecisionType`，无 op/layout 组合约束，B 侧四组合
   （T+COL/N+COL/T+ROW/N+ROW）描述符链全部可构造，最小边界 kernel 判定
   双态（VALID/VIOLATION）均与 golden 三向一致 —— 重点观察项「B 侧 T+COL
   描述符派生」「BF16 B 侧 kernel 判定」无异常；② INT8 B 侧仅 T+ROW
   （ROW|ROW→opA=N,opB=T）/N+COL（ROW|COL→opA=N,opB=N）可构造，T+COL/
   N+ROW 在 MatmulDescriptorInit 即被拒（与备注 5 ① 同根因），故 12 条中
   INT8 B 侧仅承载该两组合并以 A 侧 T+COL 补足第 3 组；③ 最小对齐粒度
   边界 shape（FP32{8,24}/BF16{16,48}/INT8{32,96}，均非更大粒度倍数）
   经 StructuredDescriptorInit 全部可构造（32B/elem 整除恰好满足），无
   描述符层拦截误伤。
7. B 侧批量用例构造决策记录（对后续维护者有用）：FP16/FP32 不属
   `IsLowPrecisionType`，B 侧批量描述符链（matA dense ROW 方方阵 + matB
   structured）N+COL / T+ROW 均可构造（INT8 B 侧仅 T+ROW/N+COL 可构造，
   见备注 5 ①，本次未选 INT8 变体以避开该约束）；B 侧批量选取 N+COL
   （B1/B4）兼补 ColScan×batch 维度（A 侧批量 12 条均为 RowScan 几何），
   T+ROW（B2）对偶 A 侧末 batch 违规用例。
8. padding ld 用例构造决策记录（对后续维护者有用，检视意见 193726682）：
   ① shape 128×144 / ld=160 —— 144 与 160 均为 FP16 structured 32B/elem
   （16 元素）倍数、N+ROW（ROW 序 ld≥cols：160≥144）/N+COL（COL 序
   ld≥rows：160≥128）描述符均可构造；144 ∈ (lanes=128, tileCols=256)
   使单列块即为列尾块（K11/K23），srcStride=(160−144)×2B=32B 的 padding
   跳过路径直达（K31）；② wrapper 扩展为可选参数（`SparseSideSpecOf` 的
   `ldOverride`，缺省 0 = 自动推导），既有调用点（CSV 正常/DESC_INVALID
   路径、批量链、E 系列）零改动零回归；③ padding 垃圾取 1.0 位模式
   （`PruneCheckPaddingGarbage`，全 dtype 确定性非零），host 侧装配
   （`PrepareSparseSideData`）与设备缓冲行距（`physElems = 行数×ld`）同步
   扩展；④ golden 无需功能扩展 —— `PruneCheckGoldenBatch` 的物理提取
   仅取数据区 [0, 物理行长)（非转置 j∈[0,k)、转置 i∈[0,m)），padding 不
   进入分组判定，由 P1-P4 的「逻辑 golden == 物理 ld golden」双口径交叉
   断言固化；⑤ B 侧链（`PruneCheckNpuBSparse`）同步扩展 ldOverride 透传
   （对称能力，本批用例均为 A 侧几何，按用例几何选链）。

## 8. 全量运行结论（实测回填）

**当前基线（2026-10-14，padding ld 补格后）**：

```
$ source /usr/local/Ascend/cann/set_env.sh   # 非默认安装路径: ${install_path}/cann/set_env.sh
$ bash build.sh --ops=prune_check --run --soc=ascend950

[==========] 140 tests from 6 test suites ran. (76040 ms total)
[  PASSED  ] 140 tests.
```

- **140/140 PASSED**（四次全量复跑同口径 140/140：62903 / 81788 / 66289 /
  76040 ms，最终态完整日志
  `.cannbot/aclsparseLtSpMMAPruneCheck/tmp/prune_check_final_run.log`）：
  CSV 92 条（L0/L1 80 + WB 12，零回归）+ TEST_F 48 条（E 系列 17 + 缺陷暴露 1 +
  白盒 10 + 批量 16 + **padding ld 4**）全部通过，d_valid 三向一致（EXACT
  零容差）与返回码枚举断言全数满足。
- **本批新增实测**（padding ld 4/4 PASS，重点观察项均无异常）：
  ① P1 RowScan（FP16 128×144，ld=160，N+ROW A 侧）：数据区合法 + padding 区
  [144,160) 全非零垃圾（1.0 位模式）→ SUCCESS/d_valid=0 —— kernel CopyIn 的
  srcStride=(160−144)×2B=32B 跳过 padding（K31），padding 数据不参与判定
  （README 声明获用例佐证；若 padding 参与判定，每行 padding 16 元素构成 4 个
  全非零组 → 必然 d_valid=1，与实测 0 冲突即暴露）；② P2 ColScan（同几何
  N+COL）：padding 列垂直组全非零 → 同样 d_valid=0（ColScan 几何的 padding
  忽略）；③ P3/P4 加固：数据区违规 + padding 全非零 → d_valid=1（两态载体，
  判定只看数据区）；④ golden 双口径交叉（逻辑 golden == PruneCheckGoldenBatch
  物理 ld golden）全数一致 —— golden 的 ld 提取语义（padding 不进入分组判定）
  与 kernel 行为对齐。
- **零回归**：padding ld 补格前基线 136 条（2026-10-13）中，CSV 92 条 +
  TEST_F 既有 44 条全部原样通过（wrapper ldOverride 为缺省 0 的可选参数，
  既有调用点零改动）。
- 备注：复跑期间曾出现一次 `aclrtSetDevice failed: 507033`（设备占用类
  瞬时环境错误，发生于连续快速复跑的间隔；等待后同二进制复跑 140/140 通过），
  与本批代码变更无关。

**历史基线（2026-10-13，第三批检视修复后，136 条）**：

```
$ source /usr/local/Ascend/cann/set_env.sh   # 非默认安装路径: ${install_path}/cann/set_env.sh
$ bash build.sh --ops=prune_check --run --soc=ascend950

[==========] 136 tests from 5 test suites ran. (58448 ms total)
[  PASSED  ] 136 tests.
```

- **136/136 PASSED**（二次复跑同口径 136/136）：CSV 92 条（L0/L1 80 + WB 12，
  零回归）+ TEST_F 44 条（E 系列 17 —— 含 **E7b（d_in 16B 对齐但非 32B →
  INVALID_VALUE，32B 对齐修复新增）**、E13 由批量重做取代 + 缺陷暴露 1 +
  白盒 10 + 批量 16）全部通过，d_valid 三向一致（EXACT 零容差）与返回码
  枚举断言全数满足。
- **本批修复项实测**：① E7（偏移 4B，非 16B/32B 对齐）与 E7b（偏移 16B，
  16B 对齐但非 32B）均被校验表 #8 拦截（INVALID_VALUE）——32B 口径下两类
  非法对齐形态独立验证；② tiling 删 groupSize 字段 + alongRow 改 bool 后
  全部 dtype/方向用例零回归（kernel 按 sizeof(U) 推导分组，u32/u16/u8 三
  分派正常）；③ kernel 热循环增量推进（零除模）重构后 grid-stride/跨
  batch/多写者用例（WB 103/105/106/110、批量 16-batch、B 侧批量 4 条）
  全部通过——覆盖集合等价性（每个 (batch, tileInMatrix) 恰被一个 block
  领取）经全量用例佐证。
- **修复前基线（2026-10-13，B 侧批量补格后）**：135/135 PASSED（本批
  E7b 新增 1 条 → 136，既有 135 条零回归）。

**历史基线（2026-10-13，B 侧批量补格后，135 条）**：

```
[==========] 135 tests from 5 test suites ran. (55489 ms total)
[  PASSED  ] 135 tests.
Test Summary: Passed: 1 - prune_check / Failed: 0
```

- **135/135 PASSED**（二次复跑同口径 135/135，56404 ms）：CSV 92 条
  （L0/L1 80 + WB 12，零回归）+ TEST_F 43 条（E 系列 16 —— E13 由批量重做
  取代 + 缺陷暴露 1 + 白盒 10 + 批量 16）全部通过，d_valid 三向一致
  （EXACT 零容差：actual == golden == expect_valid）与返回码枚举断言全数满足。
- **B 侧批量补格实测**（4/4 PASS，重点观察项均无异常）：
  ① B 侧 numBatches=2 全 batch 合法（FP16 N+COL）→ SUCCESS/d_valid=0 ——
  host DeriveBatchParams 读 md->matB 的 batchStride 装配链功能正确（若读
  错侧 matA 默认 batchStride=0 会被 #6 拒绝而非 SUCCESS，「host 读字段
  错侧」疑点排除）；② B 侧 T+ROW 转置物理 × 3 batch 仅末 batch 违规 →
  d_valid=1（batchIdx>0 基址偏移 × 转置装配，B 侧链对偶）；③ B 侧
  batchStride=16384+8（非 32B/elem=16 元素倍数，经 matB 装配）→
  INVALID_VALUE（#6 对齐子分支）；④ B 侧 FP32（u32）N+COL 仅第二 batch
  违规 → d_valid=1（ColScan×batchIdx>0 寻址）—— 未发现算子缺陷（B 侧
  batchStride 装配链 / matmul 描述符 B 侧派生均正确）。
- **批量语义实测**（A 侧 12/12 + B 侧 4/4 PASS）：numBatches=2 全 batch
  合法 → SUCCESS/d_valid=0（E13 重做，公共 API 路径）；仅 batch0 / 仅末
  batch（T+ROW 转置物理）/ 全部 batch 违规 → 1；FP32（u32）批量违规 → 1；
  padding batchStride=20480 非紧凑布局 → 1；numBatches=16（totalTiles=128
  > 64，每 block 2 轮跨 batch 交错领 tile）仅末 batch 违规 → 1；batchStride
  0/负/非 32B 对齐 → INVALID_VALUE（#6，A/B 侧装配各承载）；batchStride=
  INT64_MAX 向下取 16 倍数 → INVALID_VALUE（#11 批量项溢出）；INT8 极限
  几何 + numBatches=INT32_MAX → INVALID_VALUE（tiling totalTiles 溢出）；
  numBatches 仅设 A 侧 → MatmulDescriptorInit INVALID_VALUE（一致性由描述
  符层承载）。
- **零回归**：B 侧批量补格前基线 131 条（2026-10-09）中，CSV 92 条全部
  原样通过；TEST_F 既有 39 条全部原样通过（批量 A 侧 12 条经公共尾段
  sparseIsA 显式化后语义不变）。
- 构造性实测（35/35 PASS）：BF16×B 侧四组合均可构造且 kernel 判定与
  golden 一致（重点观察项 B 侧 T+COL 描述符派生、BF16 B 侧 kernel 判定
  均无异常）；INT8×B 侧 T+COL/N+ROW 在 MatmulDescriptorInit 即被拒
  （INVALID_VALUE，`IsValidOpLayoutCombination` 低精度约束），仅 T+ROW/N+COL
  可构造 —— 未发现算子缺陷。
- 耗时最长 case110_WB（host 侧 8.2M 分组构造 + 32MB H2D），其余均 < 2s。

**历史基线（2026-10-09，批量支持后）**：

```
[==========] 131 tests from 5 test suites ran. (55728 ms total)
[  PASSED  ] 131 tests.
Test Summary: Passed: 1 - prune_check / Failed: 0
```

- 131/131 PASSED（二次复跑同口径 131/131，56208 ms）；单 batch 零回归
  （批量支持前基线 119 条（2026-10-08）原样通过，numBatches=1 数值路径
  与扩展前完全一致）。明细见本节上方「批量语义实测」（B 侧补格前 12 条
  口径）。
