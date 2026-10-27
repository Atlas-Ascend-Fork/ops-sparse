# SDDMM PyTorch 接口

## 产品支持情况

| 产品 | 架构 |
| --- | --- |
| Ascend 950 | arch35 |

使用 PyTorch 2.7 或更新版本及与之配套的 torch_npu。构建和安装遵循
[统一扩展说明](../../../README.md)，可通过
`bash build.sh --torch_extension --ops=sddmm` 单独打包本算子。

## 功能说明

本扩展为 `torch.sparse.sampled_addmm` 注册
`aten::sparse_sampled_addmm` 和 `aten::sparse_sampled_addmm.out` 的
`SparseCsrPrivateUse1` 实现。导入 `cann_ops_sparse` 完成注册，第一次 NPU 调用时
JIT 编译算子 wrapper；底层调用 `aclsparseSDDMMBufferSize`、
`aclsparseSDDMMPreprocess` 和 `aclsparseSDDMM`。

对于 CSR 输入的每个已存储位置 `(i, j)`，输出值为
`beta * input[i, j] + alpha * (mat1 @ mat2)[i, j]`，输出保留该稀疏结构。
`torch_npu.sparse.sampled_addmm` 是同一公共 PyTorch API 的入口。

## 函数原型

```python
torch.sparse.sampled_addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None)
torch_npu.sparse.sampled_addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None)
```

## 参数说明

| 参数 | 说明 |
| --- | --- |
| `input` | 非 hybrid 的 CSR Tensor，末两维为 `(M, N)`；所有成员 Tensor 位于同一 NPU |
| `mat1` | 同设备、同 dtype 的稠密 Tensor，末两维为 `(M, K)` |
| `mat2` | 同设备、同 dtype 的稠密 Tensor，末两维为 `(K, N)` |
| `beta`、`alpha` | Python 标量，默认均为 `1`；complex64 输入支持复数标量 |
| `out` | 可选同设备、同 dtype 的非 hybrid CSR Tensor，用于接收结果 |

值类型支持 `float16`、`float32`、`complex64`。CSR 行指针与列索引须具有相同的
`int32` 或 `int64` dtype，采用从零开始的索引。`int64` 索引经范围检查后无损转换
为 `int32`。`M`、`N`、`K`、`nnz` 不超过 `INT32_MAX`，展平后的批次数不超过 65535。

两侧稠密 Tensor 的 batch 前缀必须相同；二维 CSR 结构可向该 batch 前缀广播，
带 batch 的 CSR 输入则须具有相同前缀。稠密非连续输入在当前 NPU 上整理为连续
Tensor，conjugate/negative view 在传递设备指针前解析。

## 输出

返回与输入相同 dtype、设备和 CSR 布局的 Tensor，shape 为稠密 batch 前缀加
`(M, N)`。未指定 `out` 时，行指针、列索引和值均拥有独立 storage。
`out=input` 且 shape 一致时更新输入值；二维输入不能通过 `out=input` 原地扩展为
带 batch 的输出。指定 `out` 时返回该输出对象。

## 约束说明

- 公开 PyTorch 接口接收 CSR；BSR 使用底层 ACLSparse C API。
- 不支持 CSR `bfloat16`、混合 dtype、跨设备 Tensor 或 hybrid CSR。
- 空 batch、零维矩阵和零 `nnz` 按输入 shape 返回对应 CSR 结果；`K=0` 时仍应用
  `beta`。
- layout、shape、设备和索引类型不匹配，以及无法安全窄化的索引会报错。
- 调用在当前 NPU stream 执行。稠密整理和索引转换排队后，先提交 torch_npu 队列，
  再在同一 ACL stream 下发 SDDMM；Tensor storage 记录到该 stream，保护临时内存的
  异步使用。首次 `int64` 索引范围检查需要读取检查结果。
- handle 按公共层的设备与 stream 缓存，并串行使用；描述符按调用创建和释放，
  workspace 由公共层管理并记录到当前 stream。每次调用执行完整 SDDMM 三阶段序列。

## 调用示例

```python
import torch
import torch_npu
import cann_ops_sparse

device = "npu:0"
crow = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
col = torch.tensor([0, 1], dtype=torch.int32, device=device)
values = torch.tensor([1.0, 2.0], device=device)
pattern = torch.sparse_csr_tensor(crow, col, values, size=(2, 2), device=device)
x = torch.randn(2, 4, device=device)
y = torch.randn(4, 2, device=device)
result = torch.sparse.sampled_addmm(pattern, x, y, beta=0.5, alpha=2.0)
same_api_result = torch_npu.sparse.sampled_addmm(pattern, x, y)
torch.sparse.sampled_addmm(pattern, x, y, out=pattern)
```

测试入口为 `test/sddmm/python/test_sddmm_torch_extension.py`。完整输入与 C API
约束见 [SDDMM 算子说明](../../../../sparse/sddmm/README.md)。
