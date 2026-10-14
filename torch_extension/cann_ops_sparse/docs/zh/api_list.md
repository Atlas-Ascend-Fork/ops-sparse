# Torch Extension 接口列表

`cann_ops_sparse` 包通过 `torch.library` 将 ops-sparse 算子注册到 torch_npu 后端。
导入 `cann_ops_sparse` 即完成注册，无需显式调用私有接口；C++ wrapper 在首次 NPU
调用时由 PyTorch JIT 编译。

## ATen 适配

| public PyTorch API | ATen schema | 分发键 | 算子 | 文档 |
| :--- | :--- | :--- | :--- | :--- |
| `torch.index_select` | `aten::index_select` | `PrivateUse1` | gather | [gather.md](gather.md) |
| `Tensor.index_copy_` | `aten::index_copy_` | `PrivateUse1` | scatter | [scatter.md](scatter.md) |
| `torch.mm` / `torch.addmm` | `aten::mm` / `aten::addmm`（含 out overload） | `SparseCsrPrivateUse1` / `SparsePrivateUse1` | 公共路由 → spgemm / 后续 spmm 能力 | [spgemm.md](spgemm.md) |
| `torch.sparse.mm` / `torch.sparse.addmm` | `aten::_sparse_addmm` | `SparseCsrPrivateUse1` / `SparsePrivateUse1` | 公共路由 → spgemm / 后续 spmm 能力 | [spgemm.md](spgemm.md) |
| `torch.sparse.mm` | `aten::_sparse_sparse_matmul` | `SparseCsrPrivateUse1` / `SparsePrivateUse1` | 公共路由 → spgemm | [spgemm.md](spgemm.md) |

当前仓库只提供 SpGEMM 执行能力。SpMM 相关注册入口已预留，请求该能力时会明确报错。
SpGEMM 的 out overload 尚未实现，调用会明确报错。

## `torch_npu.sparse` façade

façade 仅转发同名 public PyTorch API，不绕过 ATen 分发器，语义与直接调用完全一致。

| façade API | 转发目标 | 声明算子 |
| :--- | :--- | :--- |
| `torch_npu.sparse.mm` | `torch.sparse.mm` | 公共矩阵路由框架 |
| `torch_npu.sparse.addmm` | `torch.sparse.addmm` | 公共矩阵路由框架 |

## custom API

| custom API | schema / 转发目标 | 算子 |
| :--- | :--- | :--- |
| `torch.ops.ops_sparse.gather(values, indices, base=0)` | `ops_sparse::gather`（PrivateUse1、Meta） | gather |
| `cann_ops_sparse.ops.sparse.gather.gather` | 上述 custom schema | gather |
| `cann_ops_sparse.ops.sparse.gather.index_select_npu` | `torch.index_select`，只接受 NPU 输入 | gather |

Gather 的 `base` 支持 0/1；旧独立 `ops_sparse` Python 包已迁入 wheel 并删除。
