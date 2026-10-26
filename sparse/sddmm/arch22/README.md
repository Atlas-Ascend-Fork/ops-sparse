# SDDMM（Atlas A2/A3）

## 功能

SDDMM 按 CSR 或 BSR 的既有稀疏位置计算 `alpha * op(X) * op(Y) + beta * C`。
结果原地更新 C 的数值，稀疏索引和稠密输入保持只读。

## 支持范围

| 稀疏格式 | X/Y 类型 | C 数值类型 | 计算类型 |
| --- | --- | --- | --- |
| CSR/BSR | float32 | float32 | float32 |
| CSR/BSR | complex64 | complex64 | complex64 |
| CSR/BSR | float16 | float16 或 float32 | float32 |
| BSR | bfloat16 | bfloat16 或 float32 | float32 |

- 索引使用 int32，支持从 0 或 1 开始编号。
- BSR 支持方形块，边长为 2、4、8、16、32、64 或 128；块数值支持行主序和列主序。
- 稠密矩阵支持行主序、列主序、带填充的 leading dimension 和普通转置。
- 支持 `ACL_SPARSE_SDDMM_ALG_DEFAULT`，复数标量使用 `aclsparseComplex`。
- 支持 Host 和 Device 标量，以及空矩阵和 K=0。
- beta 为零时忽略 C 的原数值。
- BSR 批次数为 1 至 65535。X/Y 可各自广播，C 的每批可使用独立稀疏结构。

## 接口约定

依次调用 `aclsparseSDDMMBufferSize`、`aclsparseSDDMMPreprocess` 和 `aclsparseSDDMM`。
workspace 至少为查询大小，首地址按 32 字节对齐。
数值计算提交到调用方指定的 stream。释放输入、输出或 workspace 前应同步该 stream。

修改稀疏索引地址或内容后，需重新执行 Preprocess。
仅修改 X/Y 或 C 的数值可复用预处理状态。
修改形状、布局、操作、类型或批次属性后，需重新预处理。
同一 C 描述符的调用应串行；独立描述符和 workspace 支持并发调用。

逻辑维度和展开索引需处于 int32 范围，复数的双倍列范围也需满足此条件。
稠密行字节跨度需处于 uint32 范围。workspace 上限为 64 MiB。
超出支持范围时返回错误，调用方应保证实际分配大小符合查询结果。

Torch CSR 接口支持 int32 和 int64 索引，并拒绝超出 int32 范围的索引。
`torch.sparse.sampled_addmm` 返回独立结果；直接接口和 prepared 接口原地更新数值。
直接接口支持共轭和负号视图。prepared 接口重复执行时使用原始绑定的最新数值。
修改绑定张量的存储、形状或步长后应重新 prepare。

## 精度

float16 和 bfloat16 输入使用 float32 累加。complex64 的实部和虚部分别按 float32 验证。
正式测试使用 CPU 高精度参考结果，并按输出类型比较。

| 输出类型 | 相对容差 | 绝对容差 |
| --- | --- | --- |
| float16 | 2^-9 | 2^-9 |
| bfloat16 | 2^-6 | 2^-6 |
| float32/complex64 分量 | 2^-10 | 2^-16 |

回归覆盖不同索引基准、布局、转置、批次广播、复数标量、数值更新和高动态范围复数输入。
有限 alpha/beta 超出 float32 分量范围时，Torch 接口报告转换溢出；NaN/Inf 保留原生转换语义。

## 主线实现兼容

统一类型内核保留连续/跨步布局、长稀疏行、K 分块及 FP16 尾段处理。
分核依据实际归约维度 K；FP16 在单轮可覆盖全部非零元时支持按非零元分核。
主线新增的长行、分块边界、跨步尾块及饱和回归用例继续参与测试。
