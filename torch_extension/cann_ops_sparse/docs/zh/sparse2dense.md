# SparseToDense Torch Extension 接口说明

## 概述

本扩展通过仓内统一的 `torch_extension` 框架，为 Ascend NPU 注册标准 ATen 算子
`aten::_to_dense` 的后端实现。用户直接调用 `Tensor.to_dense()`，无需自定义 schema。

构建与安装方式见 [`../../../README.md`](../../../README.md)。

## 支持规格

| 项 | 支持范围 |
| --- | --- |
| 稀疏布局 | CSR、CSC、COO（COO 调用前会 coalesce） |
| 稠密输出布局 | 行主序（与 `Tensor.to_dense()` 一致） |
| 值类型 | int8、int32、float16、bfloat16、float32、complex64 |
| 索引 | 运行时窄化为 int32；index base = 0 |
| workspace | `aclsparseSparseToDense_bufferSize` 恒为 0 |

## 调用示例

```python
import torch
import torch_npu
import cann_ops_sparse  # 注册 aten::_to_dense NPU 实现

crow = torch.tensor([0, 2, 3], dtype=torch.int32, device="npu:0")
col = torch.tensor([0, 1, 1], dtype=torch.int32, device="npu:0")
values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="npu:0")
sparse = torch.sparse_csr_tensor(crow, col, values, size=(2, 3), device="npu:0")
dense = sparse.to_dense()  # 首次调用触发 JIT
```

## 构建与测试

前置环境与完整命令见
[`test/sparse2dense/python/README.md`](../../../../test/sparse2dense/python/README.md)。

先加载 CANN 环境，再按上述测试文档执行构建、安装和 pytest 命令。

底层 C API 语义见 [`../../../../sparse/sparse2dense/README.md`](../../../../sparse/sparse2dense/README.md)。
