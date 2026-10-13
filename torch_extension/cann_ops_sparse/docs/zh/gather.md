# Gather

## 产品支持情况

| 产品 | 架构目录 | 是否支持 |
| :--- | :---: | :---: |
| <term>Ascend 950PR</term> | `arch35` | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | - | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | - | × |

## 功能说明

从一维稠密 NPU Tensor `input` 中按一维索引 `index` 收集元素：

```text
output[i] = input[index[i]]
```

| 层次 | 名称 |
| :--- | :--- |
| public PyTorch API | `torch.index_select` |
| ATen schema | `aten::index_select(Tensor self, int dim, Tensor index) -> Tensor` |
| 分发键 | `PrivateUse1` |
| 底层 ACLSparse 接口 | `aclsparseGather` |

适配层将 `input` 和 `index` 转为 DnVec/SpVec 描述符，使用调用方当前 NPU stream
异步下发 Ascend C Kernel。没有 CPU fallback、workspace 或额外的线性临时缓冲区。
索引长度可以大于输入长度；此时按输入长度对索引和输出的连续视图分段调用
`aclsparseGather`，保持公共 SpVec 描述符的 `nnz <= size` 契约。分段不复制 Tensor 数据，
不修改索引值，所有分段沿用同一 stream；索引长度不超过输入长度时仍只调用一次。

## 函数原型

```python
torch.index_select(input: Tensor, dim: int, index: Tensor, *, out: Tensor | None = None) -> Tensor
```

本适配实现默认返回新 Tensor，不覆盖 `out` overload。

## 参数说明

- `input`（IN）：一维、连续、strided NPU Tensor；支持 `torch.float16`、
  `torch.bfloat16`、`torch.float32`、`torch.complex64`，不支持 autograd。
- `dim`（IN）：一维输入仅支持 `0` 或等价的 `-1`。
- `index`（IN）：与 `input` 位于同一 NPU 的一维连续 `torch.int32` Tensor；支持乱序和重复，
  长度可以超过 `input.numel()`，每个元素必须位于 `[0, input.numel())`。

`torch.index_select` 使用零基索引。迁移后的自定义接口支持两种基数：

```python
torch.ops.ops_sparse.gather(values: Tensor, indices: Tensor, base: int = 0) -> Tensor

from cann_ops_sparse.ops.sparse.gather import gather, index_select_npu
```

`gather(values, indices, base)` 转发自定义接口；`index_select_npu(values, dim, indices)`
转发 `torch.index_select`，仅接受 NPU 输入。`base=0` 时索引范围为
`[0, values.numel())`，`base=1` 时为 `[1, values.numel()+1)`；其他 base 报错。
两种基数共用同一 JIT builder、描述符和当前 NPU stream，支持重复索引、分段输出及空输出。
自定义接口提供 Meta/FakeTensor 实现，推断输出的 shape、dtype 和 device 时不启动 JIT。
原 `python/ops_sparse` 的独立 ctypes 包已删除；导入入口统一为 `cann_ops_sparse`，
原 `torch.ops.ops_sparse.gather` schema 保留，不再需要配置 `PYTHONPATH=.../python`。

## 输出

返回长度为 `index.numel()` 的一维连续 NPU Tensor，dtype 与 device 均与 `input` 相同。
输出存储独立，不与 `input` 或 `index` 别名；输入与索引保持不变。相同输入重复调用的结果
逐位一致，包括 complex64、INF 和 NAN 的原始 payload。

## 约束说明

- `input` 或 `index` 不是 NPU Tensor、device 不一致、layout/维度/连续性不匹配时抛
  `RuntimeError`。
- `index` 不是 int32、`input` dtype 不受支持、空输入配非空索引或启用 autograd 时抛
  `RuntimeError`。
- 非法 `dim` 抛 `IndexError`。
- `index.numel() == 0` 时返回合法空输出且不启动 Kernel。
- 公开 PyTorch 入口在下发 Gather Kernel 前于 NPU 上归约索引范围；若存在负索引或索引值不小于
  `input.numel()`，同步抛出 `IndexError`，不启动 Gather Kernel。底层 C API 仍遵循异步接口约定，
  合法索引是其调用方前置条件。重复使用同一未修改索引 Tensor 和相同输入长度时，适配层根据
  Tensor 弱引用、版本计数、输入长度和 base 复用已验证结果；原地修改索引、改变输入长度
  或切换 base 后会重新校验。
- C++ wrapper 首次 NPU 调用时通过 JIT 编译；导入 `cann_ops_sparse` 不触发编译。

## 调用示例

```python
import torch
import torch_npu
import cann_ops_sparse  # 导入即完成 aten::index_select 的 NPU 注册

source = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32, device="npu:0")
index = torch.tensor([4, 0, 2, 2], dtype=torch.int32, device="npu:0")
output = torch.index_select(source, 0, index)
print(output)  # [50, 10, 30, 30]
```


```python
one_based = torch.tensor([5, 1, 3, 3], dtype=torch.int32, device="npu:0")
output = torch.ops.ops_sparse.gather(source, one_based, base=1)
# 与上面的 torch.index_select 示例输出相同
```
