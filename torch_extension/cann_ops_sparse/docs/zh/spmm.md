# SpMM

## 产品支持情况

| 产品 | 架构目录 | 是否支持 |
| :--- | :---: | :---: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | `arch22` | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | `arch22` | √ |
| <term>Ascend 950PR</term> | - | × |

## 功能说明

稀疏 CSR 矩阵与稠密矩阵相乘并累加到稠密矩阵上：

```text
out = beta * self + alpha * (mat1 @ mat2)
```

`mat1` / `mat2` 中恰有一个是稀疏矩阵，另一个是稠密矩阵；公共路由会把请求规范化
为 `m x k` 的稀疏 CSR 矩阵乘 `k x n` 的稠密矩阵。`self` 可广播到 `m x n`，
`out` 是 `m x n` 的稠密矩阵。

| 层次 | 名称 |
| :--- | :--- |
| public PyTorch API | `torch.mm` / `torch.addmm` / `torch.sparse.mm` / `torch.sparse.addmm` |
| ATen schema | 公共矩阵路由统一注册 `mm` / `addmm` / `_sparse_addmm` 及 out overload |
| 分发键 | `SparseCsrPrivateUse1` / `SparsePrivateUse1` |
| 底层 ACLSparse 接口 | `aclsparseSpMMGetBufferSize` / `aclsparseSpMMPreprocess` / `aclsparseSpMM` |

公共矩阵路由按输入布局选择 SpMM（稀疏×稠密）或 SpGEMM（稀疏×稀疏），并把
SpMM 请求规范化为 CSR×dense。适配层把 CSR 三元组转成 `aclsparseCreateCsr`
描述符、把稠密输入/输出转成稠密描述符，按
`GetBufferSize -> Preprocess -> SpMM` 顺序使用调用方当前 NPU stream 异步下发，
没有 CPU fallback、也不做设备同步。

### 关于注册的 ATen 算子

矩阵乘法族的 ATen schema 由 `ops/matmul/registration.py` 唯一注册。SpMM 算子包
仅通过 `MATMUL_CAPABILITIES` 声明执行能力，不直接占用任何 schema。公共路由根据
输入布局选择 SpMM 或 SpGEMM，因此两个算子可以装入同一个 wheel，且
`torch.sparse.addmm` 的稀疏×稠密调用不会再被 SpGEMM 路径遮蔽。

## 函数原型

```python
torch.addmm(input: Tensor, mat1: Tensor, mat2: Tensor, *,
            beta: Number = 1, alpha: Number = 1, out: Tensor | None = None) -> Tensor

torch.sparse.addmm(input: Tensor, mat1: Tensor, mat2: Tensor, *,
                   beta: Number = 1, alpha: Number = 1) -> Tensor

torch.sparse.mm(mat1: Tensor, mat2: Tensor) -> Tensor
```

## 参数说明

- `input`（IN，即 schema 中的 `self`）：稠密 strided NPU Tensor，dtype 与 `mat1` 相同，
  shape 可广播到 `m x n`（`()`、`(n,)`、`(1, n)`、`(m, 1)`、`(m, n)` 均可）。
  `beta = 0` 时其数值不参与计算，其中的 `NaN`/`Inf` 也不会渗入结果。
- `mat1` / `mat2`（IN）：二维 NPU Tensor，恰有一个使用稀疏 COO/CSR/CSC layout，
  另一个使用稠密 strided layout，二者 dtype 相同。公共路由会合并 COO 重复项并
  转成 CSR；int64 CSR 索引在校验不溢出后窄化为 int32。规范化后的 `m`、`k`、`nnz`
  均须可用有符号 int32 表示。
- 规范化后的稠密乘数若为紧凑行主序则零拷贝就地描述，其余合法 stride 会物化成
  连续副本。
- `beta`、`alpha`（IN）：Python `int`/`float`/`bool`/`complex` 标量。实数 dtype 下不接受
  虚部非零的复数；complex64 下按 `{re, im}` 两个 fp32 传给底层接口。
- `out`（OUT）：`m x n` 的稠密 NPU Tensor，dtype 与 `mat1` 相同。允许与 `mat2` 或
  `input` 别名；非连续输出由公共 provider 使用连续临时结果回写。

`mat1` / `mat2` / `input` / `out` 必须位于同一 NPU device。不做隐式 dtype 提升。

支持的 dtype 组合（与 `sparse/spmm/README.md` 的 arch22 约束一致）：

| `mat1` / `mat2` / `out` dtype | 底层 computeType |
| :--- | :--- |
| `torch.float32` | `ACL_FLOAT` |
| `torch.float16` | `ACL_FLOAT`（fp32 累加） |
| `torch.bfloat16` | `ACL_FLOAT`（fp32 累加） |
| `torch.complex64` | `ACL_COMPLEX64` |

## 输出

返回 `m x n` 的连续稠密 NPU Tensor，dtype 与 device 与 `mat1` 相同。使用 `out=` 时
返回的就是 `out` 本身（满足 `Tensor(a!)` 的别名契约）。`mat1`、`mat2` 保持不变，
除非调用方显式把它们作为 `out` 传入。

## 一阶反向支持

SpMM 的公共矩阵路由注册了一阶 VJP，支持 `float16`、`bfloat16`、`float32` 和
`complex64`，覆盖稀疏矩阵在左侧或右侧的 `mm`、`addmm`、`sparse.mm` 和
`sparse.addmm`。反向接收原始稀疏布局，并按调用的 ATen schema 保持 PyTorch 的标量
语义。设上游梯度为 `G`：

- `aten::_sparse_addmm`（即 `torch.sparse.addmm`）：`self` 和稀疏矩阵使用原始
  `beta`/`alpha`，稠密矩阵使用 `conj(alpha)`；`self` 的梯度归约回原始广播形状。
- `aten::addmm`（即普通 `torch.addmm`）：矩阵梯度使用 `conj(alpha)`，`self` 的
  梯度使用 `conj(beta)`，同样归约回原始广播形状。
- `aten::mm` / `torch.sparse.mm`：等价于 `alpha = beta = 1` 且没有 `self`。

对于 `torch.sparse.addmm` 和 `torch.sparse.mm`，稀疏矩阵的梯度保留原输入的稀疏布局和
存储模式；对于普通 `torch.addmm`，稀疏输入的梯度按 PyTorch 语义返回完整矩阵梯度。
索引张量不参与求导。公共自动求导包装器当前只支持一阶反向，不支持二阶反向和 `out=`
自动求导。

## 约束说明

- `opA`/`opB` 固定为非转置，索引基址固定为 0：`torch.sparse.addmm` 和 PyTorch 的
  `sparse_csr_tensor` 都没有暴露这两个控制项。转置与 1 基索引只能通过裸 C++
  `aclsparseSpMM` 接口使用。
- `mat1` 不是 CSR layout、维度不是 2、`mat1.size(1) != mat2.size(0)`、dtype 不一致、
  `out` dtype 与输入不符、device 不一致，或 `m`/`k`/`nnz` 超出 int32 时抛 `RuntimeError`。
- int64 CSR 索引窄化到 int32 会溢出时抛 `RuntimeError`，不会静默截断。
- `m`/`k`/`n` 任一为 0 时返回合法的零长度结果，取值遵循 `out = beta * self`，不启动
  主 Kernel（与 `sparse/spmm/README.md` 中 arch22 的零维行为一致）。
- functional 与 out overload 都由公共路由进入同一个 SpMM provider；out overload
  返回调用方传入的原对象。
- 正常路径全程异步，不做设备同步。handle 由公共层按 `device + stream` 缓存，
  workspace 与临时 Tensor 的 storage 记录到同一条 stream 上。
- `SPMM_ADAPTER_OUT_OF_PLACE=0` 可关闭 out-of-place 优化（该优化在 `self` 已是整块
  连续 `m x n` 时让底层直接读 `self`、省掉一次 `self -> out` 拷贝）；关闭后功能不变，
  只是多一次拷贝。
- C++ wrapper 在首次 NPU 调用时 JIT 编译；导入 `cann_ops_sparse` 不触发编译。

## 调用示例

```python
import torch
import torch_npu
import cann_ops_sparse  # 导入即装配公共矩阵路由与 SpMM 能力

device = "npu:0"
crow = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
col = torch.tensor([0, 2, 1], dtype=torch.int32, device=device)
values = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32, device=device)
# mat1 = [[1, 0, 2], [0, -1, 0]]
mat1 = torch.sparse_csr_tensor(crow, col, values, size=(2, 3), device=device)
mat2 = torch.tensor([[1.0, -2.0], [0.5, 3.0], [-1.0, 4.0]], dtype=torch.float32, device=device)
bias = torch.full((2, 2), 0.125, dtype=torch.float32, device=device)

out = torch.addmm(bias, mat1, mat2, beta=-0.5, alpha=1.5)
print(out)  # [[-1.5625, 8.9375], [-0.8125, -4.5625]]
```
