# spgemm

## 产品支持情况

| 产品 | 架构目录 | 是否支持 |
| :--- | :---: | :---: |
| <term>Ascend 950PR</term> | `arch35` | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | `arch22` | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | `arch22` | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | - | × |
| <term>Atlas 推理系列产品</term> | - | × |
| <term>Atlas 训练系列产品</term> | - | × |

Torch 接口本身与架构无关：注册逻辑只调用公共 `aclsparseSpGEMM*` 接口，具体由哪个
架构实现承接，取决于运行时加载的 `libops_sparse.so` 是为哪个 `SOC_VERSION` 构建的。

## 功能说明

计算两个 CSR 稀疏矩阵的乘积，并可选地累加一个稀疏矩阵：

```text
C = beta * self + alpha * (mat1 @ mat2)
```

其中 `mat1` 形状为 `M×K`、`mat2` 形状为 `K×N`、输出形状为 `M×N`。多个中间乘积落到
同一坐标时合并；输出 CSR 每行列索引严格升序、不含重复坐标；数值抵消得到的显式零
**保留**在输出结构中（`nnz` 计入该位置）。

| 层次 | 名称 |
| :--- | :--- |
| public PyTorch API | `torch.sparse.mm` / `torch.sparse.addmm` |
| ATen schema | `aten::_sparse_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor` |
| ATen schema | `aten::_sparse_sparse_matmul(Tensor self, Tensor other) -> Tensor` |
| 分发键 | 公共路由统一注册 `SparseCsrPrivateUse1` / `SparsePrivateUse1`，按布局和能力选择实现 |
| 底层 ACLSparse 接口 | `aclsparseSpGEMMCreateDescr` / `aclsparseSpGEMMWorkEstimation` / `aclsparseSpGEMMCompute` / `aclsparseSpMatGetSize` / `aclsparseCsrSetPointers` / `aclsparseSpGEMMCopy` / `aclsparseSpGEMMDestroyDescr` |
| `torch_npu.sparse` façade | `torch_npu.sparse.mm` / `torch_npu.sparse.addmm` |

ATen 注册与矩阵乘法 façade 由公共框架统一管理，SpGEMM 包仅声明执行能力。
façade 直接转发同名 public PyTorch API，不绕过 ATen 分发器，因此两种写法语义完全一致。

`torch.sparse.mm(csr, csr)` 由 PyTorch 以 `beta=0, alpha=1` 路由到 `aten::_sparse_addmm`；
`torch.sparse.mm(coo, coo)` 路由到 `aten::_sparse_sparse_matmul`。

## 函数原型

```python
torch.sparse.mm(mat1: Tensor, mat2: Tensor) -> Tensor

torch.sparse.addmm(mat: Tensor, mat1: Tensor, mat2: Tensor, *,
                   beta: Number = 1, alpha: Number = 1) -> Tensor

torch_npu.sparse.mm(mat1: Tensor, mat2: Tensor) -> Tensor
torch_npu.sparse.addmm(mat: Tensor, mat1: Tensor, mat2: Tensor, *,
                       beta: Number = 1, alpha: Number = 1) -> Tensor
```

## 参数说明

- `mat1`（IN）：左乘稀疏矩阵。
  - shape：`(M, K)`，必须为 2 维；`M`、`K`、`nnz` 均须 `<= INT32_MAX`。
  - dtype：`torch.float16` / `torch.bfloat16` / `torch.float32` / `torch.complex64`。
  - layout：`torch.sparse_csr`（`_sparse_addmm` 路径）或 `torch.sparse_coo`
    （`_sparse_sparse_matmul` 路径，内部转 CSR 计算）。
  - device：NPU（`PrivateUse1`）。
  - 索引 dtype：`int32` 与 `int64` 均可；`int64` 由适配层在设备上窄化为 `int32`
    后传给 ACLSparse（ACLSparse 侧仅支持 `ACL_SPARSE_INDEX_32I`）。
  - 连续性：`crow_indices` / `col_indices` / `values` 由适配层统一 `contiguous()`，
    调用方无需预先保证。
- `mat2`（IN）：右乘稀疏矩阵。shape `(K, N)`，其余约束同 `mat1`；
  必须满足 `mat1.size(1) == mat2.size(0)`，且与 `mat1` 的 dtype、device 一致。
- `mat` / `self`（IN）：被累加的稀疏矩阵，仅 `torch.sparse.addmm` 使用。
  - **校验与 `beta` 取值无关**：只要传入，就必须为 CSR、shape 恰为 `(M, N)`、
    dtype 与 device 与 `mat1` 一致，否则抛 `RuntimeError`。这与 `aten::addmm`
    的参考语义一致（CPU 上 `beta=0` 同样校验 `self`）。
  - `beta == 0` 时**不读取其数据**，仅参与校验（`torch.sparse.mm` 即此情形，
    PyTorch 会合成一个 shape `(M, N)`、dtype/device 与输入一致的空 CSR）。
  - `beta != 0` 时额外作为 C_in 绑定到 matC，从 WorkEstimation 阶段即生效。
- `beta`（IN）：`self` 的缩放系数，默认 `1`。可为 Python `int`/`float`/`bool`/`complex`。
- `alpha`（IN）：乘积的缩放系数，默认 `1`，取值范围同 `beta`。

`beta`、`alpha` 会按输出 dtype 转换后一次性暂存在 host，并以同一地址传给
WorkEstimation / Compute / Copy 各阶段——ACLSparse 会逐位比较各阶段读到的 `beta`，
不一致即返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。

## 输出

单个 Tensor：

- shape：`(M, N)`。
- dtype：与输入相同。
- layout：输入为 CSR 时返回 `torch.sparse_csr`；输入为 COO 时返回
  `torch.sparse_coo`，且已 coalesce（`is_coalesced()` 为 `True`）。
- device：与输入相同。
- 索引 dtype：`torch.int32`（无论输入索引是 `int32` 还是 `int64`）。
- 别名/原地行为：输出的 `crow_indices` / `col_indices` / `values` 均为新分配的存储，
  **不与 `self`、`mat1`、`mat2` 共享内存**，也不修改任何输入。即使
  `beta != 0`（此时 `self` 作为 C_in 参与计算），`self` 的数据也保持不变。

## 约束说明

**dtype**

- 仅支持 `float16` / `bfloat16` / `float32` / `complex64`。其他 dtype（如 `float64`、
  整型）抛 `RuntimeError`。
- 三个输入的 dtype 必须完全一致，不做隐式提升；不一致抛 `RuntimeError`。
- 实数 dtype 上传入虚部非零的 `beta`/`alpha` 抛 `RuntimeError`。

**layout**

- `mat1`、`mat2` 必须同为稀疏（CSR 或 COO）。稀疏 × 稠密不走本算子，
  由公共框架选择 SpMM 能力；当前未提供该能力时明确报错。
- 传入 strided（稠密）Tensor 会抛 `RuntimeError`。

**transpose**

- 仅支持 `op(A)=A`、`op(B)=B`。`ACL_SPARSE_OP_TRANSPOSE` / `CONJUGATE_TRANSPOSE`
  未实现，适配层固定传 `ACL_SPARSE_OP_NON_TRANSPOSE`，不暴露转置入口。

**index 类型与规模**

- `rows`、`cols`、`nnz` 任一超过 `INT32_MAX` 抛 `RuntimeError`（对应 ACLSparse 的
  `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`）。
- 索引基值固定为 0（`ACL_SPARSE_INDEX_BASE_ZERO`）。

**空 Tensor 与边界 shape**

- `nnz == 0`（含全空行、无匹配乘积）：正常返回，输出 `nnz == 0`，
  `crow_indices` 长度仍为 `M+1` 且全为 0。
- `M == 0` 或 `N == 0`：正常返回对应形状的空稀疏 Tensor。
- 最小 shape `(1, 1)`：正常计算。
- 数值抵消为 0 的位置**不被剪除**，作为显式零保留。

**异常类型**

适配层的入参校验统一抛 `RuntimeError`（`TORCH_CHECK`），且在申请任何设备内存
之前完成；ACLSparse 返回非 `SUCCESS` 时同样抛 `RuntimeError`，消息中带
`ACLSparse status` 及状态码。

**一阶反向支持**

| 接口 | 输入布局 | 已验证的 value dtype |
| :--- | :--- | :--- |
| `torch.mm` / `torch.sparse.mm` | CSR×CSR、COO×COO | float16、bfloat16、float32、complex64 |
| `torch.addmm` / `torch.sparse.addmm` | CSR self、CSR×CSR | float16、bfloat16、float32、complex64 |
| `torch_npu.sparse.mm/addmm` | 与对应 `torch.sparse` 接口相同 | 同上 |

反向计算两侧输入及累加项的梯度，indices 不求导；梯度布局和稀疏结构按对应 ATen
入口处理，不能统一强制为原输入 pattern。支持稠密或稀疏上游梯度及复数，浮点 CSR/COO
构造器的 values 求导链路也已验证；complex64 验证的是稀疏 Tensor 叶节点的梯度。
COO 梯度可为未 coalesce 的表示，结构与数值比较可将其分量复制到 CPU 后 coalesce。
已验证共享输入（如 `mm(A, A)`）及多次 backward 的梯度累加。CSR 梯度累加、
COO 梯度结构投影、稀疏源的 `sparse_mask`，以及 `torch.mm` 的稠密上游
反向目前通过 CPU 完成，包含设备间数据复制。后者保留稠密零值与 NaN/Inf 的原生传播；
CPU 不支持的 float16/bfloat16 稀疏计算使用 float32 累加后转回原 dtype。
当前支持一阶反向；二阶反向及 `out=` 的自动求导不支持。

COO 反向的结构投影依赖 PyTorch 原生 `aten::_sparse_mask_projection`。运行版本缺少
该算子时，相关反向会明确报告兼容性错误；`sparse_mask` 的结构语义不同，不能作为
等价替代。升级 PyTorch 后应重新运行原生梯度结构、布局和数值对照测试。

**PyTorch 2.10 反向规则与扩展支持**

公共路由保存原始 ATen schema，按入口分别执行反向，而非把所有接口视为同一个公式：

| 入口 | 反向规则 |
| :--- | :--- |
| `torch.mm(COO, COO)`、`torch.mm(CSR, CSR)` / `aten::mm` | 返回完整乘积梯度，不投影到输入 pattern；稠密上游返回稠密梯度；稀疏上游返回完整稀疏乘积；CSR 输入和 CSR 上游返回 CSR，COO 上游返回 COO |
| `torch.sparse.mm(COO, COO)` / `aten::_sparse_sparse_matmul` | 用原生 `sparse_mask_like_grad(..., false)` 的分支处理稀疏结构及重复坐标 |
| `torch.addmm` / `aten::addmm` | self 直接缩放上游、保留其布局和坐标，使用 `conj(beta)`；两侧返回完整乘积梯度，不投影到输入 pattern，使用 `conj(alpha)`；稠密上游保留显式零并返回稠密梯度；CSR 输入和 CSR 上游返回 CSR，COO 上游返回 COO |
| `torch.sparse.addmm` / `aten::_sparse_addmm` | 按该 schema 的原生规则：self 直接缩放上游、保留其布局和坐标，使用 beta；mat1 使用 alpha，mat2 使用 `conj(alpha)` |

alpha/beta 为零时仍按对应原生计算再缩放，保留 `NaN * 0`，不直接返回零梯度。
COO 的上述规则已与本地 PyTorch 2.10 CPU 原生接口对照结构、布局和数值；稠密上游
还覆盖零值与稀疏 operand 中 NaN/Inf 的传播。

额外支持包括 CSR×CSR 一阶反向、原生稀疏乘法不接受的稠密上游，以及部分 dtype /
上游布局组合。`aten::_sparse_addmm` 的 CSR 矩阵梯度保留输入
CSR pattern；`aten::mm` / `aten::addmm` 的矩阵梯度及所有入口的 self 梯度不作输入 pattern 裁剪。
这些扩展不意味着 PyTorch 2.10 原生支持相同输入域。SpGEMM addmm 的 self 仍要求同形 CSR，不扩展原生 SpMM 的广播域。

本地 CPU 构建缺少 MKL，无法运行原生 CSR×CSR forward；相关注册回调的回归
使用相同 `aten::mm` 的原生 COO 输入，以及原生 addmm 的 COO×dense 输入作对照。
实际 NPU 公共入口的 CSR 用例另列于端到端测试中。

对照依据为 PyTorch v2.10.0 的
[derivatives.yaml](https://github.com/pytorch/pytorch/blob/v2.10.0/tools/autograd/derivatives.yaml)
和 [FunctionsManual.cpp](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/autograd/FunctionsManual.cpp)。
CSR 上游的矩阵梯度布局依据
[SparseCsrTensorMath.cpp](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/sparse/SparseCsrTensorMath.cpp)
的 `_sparse_csr_mm`：CSR/CSC 两侧乘积返回 CSR；内部 COO 计算完成后恢复该布局。

本地验证环境为 CANN 9.1.0、torch 2.10.0、torch_npu 2.10.0，使用与仓库头文件匹配的
arch22 算子库。

**stream 与同步**

- 适配层绑定 PyTorch 当前 NPU stream（`c10_npu::getCurrentNPUStream`），
  支持非默认 stream 与非默认 device；所有输出/workspace 张量都做了
  `recordStream`，可安全参与 caching allocator 的复用。
- 适配层仅提交 torch_npu 队列，不主动等待设备。arch22 的 CSR 校验使用
  `aclrtMemcpyAsync` 将索引复制排到 handle stream，并在 CPU 校验前等待复制完成，
  确保索引转换与读取有序；A/B 共用一次等待，非零 beta 的累加项另行校验。
- **本算子的执行路径含固有的阻塞同步**：SpGEMM 是两趟算法，host 需要读回
  中间统计量才能继续——
  - `sparse/spgemm/arch22/spgemm_host.cpp:455`：读回 `rowProducts` 做行装箱；
  - `sparse/spgemm/arch22/spgemm_host.cpp:581`：读回 `nnz(C)` 以确定输出规模；
  - `aclsparseSpGEMMWorkEstimation` 执行阶段等待 CSR 索引的 stream 有序 D2H 复制完成后做结构校验。

  这是「输出 `nnz` 在计算前未知」这一算法本质带来的，无法在适配层消除，
  因此本算子**不满足**"执行路径无主动同步"这一通用要求。调用方应预期
  `torch.sparse.mm` / `torch.sparse.addmm` 在 NPU 稀疏乘法处存在同步点。

**编译**

- C++ wrapper 由 PyTorch 在**首次 NPU 调用时**JIT 编译（ninja），导入
  `cann_ops_sparse` 本身不触发编译。链接需要 `libops_sparse.so`：默认从
  `$ASCEND_HOME_PATH/lib64` 查找，可用 `OPS_SPARSE_LIB_DIR` 指向构建产物目录。

## 调用示例

```python
import torch
import torch_npu
import cann_ops_sparse  # 导入即完成 ATen 注册

device = "npu:0"

dense_a = torch.tensor([[1.0, 2.0], [0.0, 3.0]], device=device)
a = dense_a.to_sparse_csr()

# 1) 标准 PyTorch API：稀疏 × 稀疏
c = torch.sparse.mm(a, a)
print(c.layout, c.crow_indices(), c.col_indices(), c.values())

# 2) 带累加与缩放：C = 2 * b + 3 * (a @ a)
b = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device).to_sparse_csr()
c = torch.sparse.addmm(b, a, a, beta=2.0, alpha=3.0)
print(c.to_dense())

# 3) COO 布局
a_coo = a.to_sparse(layout=torch.sparse_coo)
c_coo = torch.sparse.mm(a_coo, a_coo)
print(c_coo.layout, c_coo.is_coalesced(), c_coo.indices(), c_coo.values())

# 4) torch_npu.sparse façade（与 1)、2) 等价）
c = torch_npu.sparse.mm(a, a)
c = torch_npu.sparse.addmm(b, a, a, beta=2.0, alpha=3.0)
```
