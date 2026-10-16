# SDDMM算子 (arch22)

## 算子概述

SDDMM（Sampled Dense-Dense Matrix Multiplication）实现两个稠密矩阵相乘后按稀疏矩阵模式采样的运算，对标 cuSPARSE `cusparseSDDMM`。

```
C_out = (alpha * op(X) * op(Y) + beta * C) ∘ spy(C)
```

## 支持芯片

- Ascend 910B（架构 arch22 / DAV_2201）

## 接口与参数

公共接口定义、参数说明及调用示例见上层 [../README.md](../README.md)。本文件仅补充 arch22 特有的实现约束和说明。

零维输入：X / Y / C 任一维度（rows/cols）为 0 时返回 `ACL_SPARSE_STATUS_INVALID_VALUE`（host 侧 `SddmmValidateShapes` fail-fast 拒绝，arch35 与 arch22 行为一致）。

## 精度标准

采用 MIXED_TOLERANCE 模式：逐元素 `|actual - golden| ≤ max(atol + rtol * |golden|, maxAbsErrLimit)`，整体 `matchedRatio ≥ 0.99`。

| 数据类型 | rtol | atol | maxAbsErrLimit fixedValue |
|----------|------|------|---------------------------|
| FLOAT32  | 2⁻¹⁰ | 2⁻²³ | 1e-2 |
| FLOAT16  | 2⁻⁹  | 2⁻⁹  | 1e-1 |

FP16 路径在 FP32 中完成点积累加，写回前饱和截断到 [−65504, 65504]。

## Ascend 910B (arch22) 实现说明

FP32 主路径按非零元分块的向量化归约组织计算：

1. 每行将 X 对应行载入 UB 一次
2. 至多 `jTile` 个非零元的 Y 列连续暂存为 UB 中 `jTile × kPitch` 矩阵，连续列合并为一次多块 DMA
3. `src0RepStride=0` 广播 X 行做 `Mul`，`WholeReduceSum` 折叠为点积
4. `y = alpha * acc + beta * C` 全部在向量流水完成

连续和跨步的 Y 列分别实例化 FP32 内核。跨步列按 `kPitch` 批量收集，尾块也保持相同的对齐步长。`K > 4088` 时使用完整行缓存或分块点积；是否按非零元分区由实际归约维度 K 决定。逐非零元路径直接融合 CSR 更新，不依赖固定长度的点积结果数组，因此支持长稀疏行。`alpha == 0` 时省去点积，执行向量 `y = beta * C`。

FP16 路径在 FP32 精度下计算。K 不超过 4088 且每行至少有 8 个非零元时，每段最多 512 个点积结果在向量流水完成缩放、CSR 累加、饱和截断与 half 转换；该行的不足 8 个元素的尾段沿用同一路径。其余路径以标量合并点积结果，再精确写回有效 half 字节。所有 DMA、转换、归约及输出缓冲复用均按实际生产者/消费者设置流水同步。

当 K 等于 4088，且全部非零元能在可用 AIV 核上单轮完成时，FP16 可按非零元分区，避免小行数限制并行度。K 大于 4088 的 FP16 仍按行分区，通过分块标量累加完成点积。

测试覆盖连续/跨步布局、转置、长稀疏行、K 的分块边界、FP16 饱和以及向量段尾部；精度判定沿用上表标准。
