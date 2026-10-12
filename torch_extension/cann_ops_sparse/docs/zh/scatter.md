<!-- 算子接口文档：index_copy_（aclsparseScatter） -->

# index_copy_

## 产品支持情况

| 产品 | 支持 |
| --- | --- |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 支持 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 支持 |

## 功能说明

- **PyTorch public API**：`Tensor.index_copy_(0, index, source)`
- **ATen schema**：`aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)`
- **底层 ACLSparse 接口**：`aclsparseScatter(handle, vecX, vecY)`
- **计算语义**：按稀疏向量索引将 source 值原地写入 self：

$$Y[index_i] = source_i,\quad i \in [0, nnz)$$

其中 `self` 为稠密输入输出向量（DnVec 语义），`index`/`source` 构成稀疏输入向量（SpVec 语义，分离存储）。
重复索引遵循 PyTorch `index_copy_` 语义，为非确定性 last-write-wins；无重复索引时结果与 CPU 参考逐位一致。

## 函数原型

```python
Tensor.index_copy_(dim, index, source) -> Tensor
```

## 参数说明

| 参数 | 说明 |
| --- | --- |
| self | 输入输出（原地写）：一维连续稠密 NPU 张量（`dim == 0`、一维、连续 layout）；dtype 须与 source 一致 |
| dim | 仅支持 `0` |
| index | 目标位置索引：一维连续 NPU `int32` 张量；取值须位于 `[0, self.numel())`；与 source 元素数一致 |
| source | 写入值：一维连续 NPU 张量；dtype ∈ {int8, float16, bfloat16, float32, complex64}，且与 self 一致 |

三个张量须位于同一 NPU device；index/source 与 self 不允许内存重叠。

## 输出

- 原地修改 `self` 并返回 `self`（别名返回，无新分配）。
- 无重复索引时输出与 CPU 参考逐位一致；`nnz=0` 时成功返回且不修改 `self`。

## 约束说明

- 仅支持 `dim == 0` 的一维稠密向量场景；不支持的组合显式报错，无 CPU fallback。
- index 仅支持 int32（aclsparse I32 契约）；int64 索引确定报错，不做隐式转换。
- 越界 index 返回确定错误（对齐 PyTorch `index_copy_` 语义，不依赖 Device 侧异步错误协议）。
- 重复索引结果非确定（last-write-wins），性能 case 与精度比对建议使用无重复索引。
- 规模上界：`nnz` 与 `self.numel()` 均不超过 `INT32_MAX`。
- 底层 aclsparse 句柄按 `device + stream` 缓存并串行化调用；首次调用触发 JIT 编译，不计入稳态性能。

## 调用示例

```python
import torch
import torch_npu
import cann_ops_sparse  # 导入即注册 aten::index_copy_ NPU 实现

device = "npu:0"
y = torch.randn(128256, dtype=torch.float32).to(device)
idx = torch.randperm(128256)[:8192].to(torch.int32).to(device)
src = torch.randn(8192, dtype=torch.float32).to(device)

y.index_copy_(0, idx, src)  # NPU 路径 → aclsparseScatter
torch.npu.synchronize()
```
