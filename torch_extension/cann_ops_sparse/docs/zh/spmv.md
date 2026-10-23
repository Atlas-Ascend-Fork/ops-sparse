# spmv

## 产品支持情况

| 产品 | 架构目录 | 是否支持 |
| :--- | :---: | :---: |
| <term>Ascend 950PR</term> | - | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | `arch22` | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | `arch22` | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | - | × |
| <term>Atlas 推理系列产品</term> | - | × |
| <term>Atlas 训练系列产品</term> | - | × |

Torch 接口本身与架构无关：注册逻辑只调用公共 `aclsparseSpMV*` 接口，具体由哪个
架构实现承接，取决于运行时加载的 `libops_sparse.so` 是为哪个 `SOC_VERSION` 构建的。

## 功能说明

计算 CSR 稀疏矩阵与稠密向量的乘积：

```text
y = A @ vec
```

其中 `A` 形状为 `M×N`（CSR 稀疏矩阵），`vec` 形状为 `(N,)`，输出形状为 `(M,)`。

| 层次 | 名称 |
| :--- | :--- |
| public PyTorch API | `torch.mv` |
| ATen schema | `aten::mv(Tensor self, Tensor vec) -> Tensor` |
| 分发键 | `SparseCsrPrivateUse1` |
| 底层 ACLSparse 接口 | `aclsparseCreateCsr` / `aclsparseCreateDnVec` / `aclsparseSpMVGetBufferSize` / `aclsparseSpMVPreprocess` / `aclsparseSpMV` / `aclsparseDestroySpMat` / `aclsparseDestroyDnVec` |
| `torch_npu.sparse` façade | 无（`torch.mv` 不属于 `torch_npu.sparse.*` 命名空间） |

`aten::mv` 的 ATen 语义不含 `alpha`/`beta`，适配层固定按 `alpha=1`、`beta=0`
调用底层接口。需要完整 `y = alpha * op(A) * x + beta * y`（含转置、indexBase 1、
输入输出 dtype 组合）的场景请直接使用 aclsparse C 三阶段接口
（`aclsparseSpMVGetBufferSize` / `aclsparseSpMVPreprocess` / `aclsparseSpMV`）。

## 函数原型

```python
torch.mv(input: Tensor, vec: Tensor) -> Tensor
```

## 参数说明

- `input`（IN）：CSR 稀疏矩阵。
  - shape：`(M, N)`，必须为 2 维；`M`、`N`、`nnz` 均须 `<= INT32_MAX`。
  - dtype：`torch.int8` / `torch.float16` / `torch.bfloat16` / `torch.float32` /
    `torch.int32` / `torch.complex64`。
  - layout：必须为 `torch.sparse_csr`；其他 layout 抛 `RuntimeError`。
  - device：NPU（`PrivateUse1`）。
  - 索引 dtype：`int32` 与 `int64` 均可；`int64` 由适配层在设备上窄化为 `int32`
    后传给 ACLSparse（ACLSparse 侧仅支持 `ACL_SPARSE_INDEX_32I`）。
  - 索引基值固定为 0（`ACL_SPARSE_INDEX_BASE_ZERO`）。
  - 连续性：`crow_indices` / `col_indices` / `values` 由适配层统一 `contiguous()`，
    调用方无需预先保证。
  - 同一行内允许重复列坐标，重复项按数学语义累加；列索引不要求有序。
- `vec`（IN）：稠密向量。
  - shape：`(N,)`，必须为 1 维，且 `vec.size(0) == input.size(1)`。
  - dtype：必须与 `input` 的 values dtype 完全一致，不做隐式提升。
  - layout/device：`torch.strided`、与 `input` 相同的 NPU device；
    非连续输入由适配层统一 `contiguous()`。

## 输出

单个稠密 Tensor：

- shape：`(M,)`。
- dtype：`input` 为 `torch.int8` 时输出 `torch.int32`（INT32 精确累加），
  其余与 `input` 的 values dtype 一致。
- layout：`torch.strided`；device 与 `input` 相同。
- 别名为新分配存储，不与任何输入共享内存，也不修改任何输入。

## 约束说明

**dtype**

- 仅支持上表列出的 6 种 value dtype；其他 dtype（如 `float64`、`int64`）抛
  `RuntimeError`。
- `input` 与 `vec` 的 dtype 必须一致。
- 本路径的输出 dtype 固定按上表映射（int8→int32、其余不变）；其余
  输入/输出 dtype 组合（如 int8→float32、float32→complex64）不经 `torch.mv`
  暴露，需走 aclsparse C 接口。

**layout 与 transpose**

- 仅支持 CSR × 一维稠密向量；CSC/COO 或稠密 `input` 不由本注册承接。
- 仅支持非转置（`op(A)=A`）。转置/共轭转置不经 `torch.mv` 暴露。

**index 类型与规模**

- `M`、`N`、`nnz` 任一超过 `INT32_MAX` 抛 `RuntimeError`。
- ACLSparse 侧仅支持 `ACL_SPARSE_INDEX_32I`。

**空 Tensor 与边界 shape**

- `nnz == 0` 或 `N == 0`：返回全零向量（`alpha=1, beta=0` 下 `A @ vec = 0`）。
- `M == 0`：返回 shape `(0,)` 的空向量。

**异常类型**

适配层的入参校验统一抛 `RuntimeError`（`TORCH_CHECK`），且在申请任何设备内存
之前完成；ACLSparse 返回非 `SUCCESS` 时同样抛 `RuntimeError`，消息中带
`ACLSparse status` 及状态码。

**stream 与同步**

- 适配层绑定 PyTorch 当前 NPU stream（`c10_npu::getCurrentNPUStream`），
  支持非默认 stream 与非默认 device；所有输出/workspace 张量都做了
  `recordStream`，可安全参与 caching allocator 的复用。
- 调用开始处会提交（submit）torch_npu 异步任务队列中已排队的工作
  （`NPUStream::stream(true)`）。这**不是** device 同步，只保证适配层内部为
  ACLSparse 做的索引窄化在库读取之前已下发；执行路径无主动同步。

**编译**

- C++ wrapper 由 PyTorch 在**首次 NPU 调用时**JIT 编译（ninja），导入
  `cann_ops_sparse` 本身不触发编译。链接需要 `libops_sparse.so`：默认从
  `$ASCEND_HOME_PATH/lib64` 查找，可用 `OPS_SPARSE_LIB_DIR` 指向构建产物目录。

## 确定性计算说明

- 非转置路径（即 `torch.mv` 所走路径）每行非零元按固定的分块归约顺序累加，
  相同输入重复调用结果**逐位一致**。
- INT32 计算路径（int8/int32 输出）为整数精确累加，与参考实现逐元素相等。
- 16bit 输出（float16/bfloat16）以 FP32 计算、末尾一次性按最近舍入
  （平局取偶，RNE）转换。
- 转置路径（仅 aclsparse C 接口暴露）采用 double-single 确定性累加或整数原子加，
  同样逐位可复现。

## 调用示例

```python
import torch
import torch_npu
import cann_ops_sparse  # 导入即完成 ATen 注册

device = "npu:0"

dense = torch.tensor([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], device=device)
a = dense.to_sparse_csr()
vec = torch.tensor([1.0, 1.0, 1.0], device=device)

y = torch.mv(a, vec)
print(y)  # tensor([3., 7.], device='npu:0')
```
