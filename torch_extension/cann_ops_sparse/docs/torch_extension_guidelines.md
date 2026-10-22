# torch_extension 开发规范

本文档约定 `cann_ops_sparse`（torch_extension）新增或修改 PyTorch 稀疏算子适配时的
目录组织、命名、各层实现、打包、测试和文档规范。开发前应先确认目标 PyTorch API、
ATen schema 和 dispatch key。本文以 `spmm` 算子为例说明实现形式；它只是示例，不是
框架的注册前提，新算子必须能够独立打包、安装和注册。

`cann_ops_sparse` 通过 `torch.utils.cpp_extension.load` 在算子首次执行时 JIT 编译
C++ kernel wrapper，将 `at::Tensor` 桥接到 ops-sparse 提供的 ACLSparse C API。完整适配通常
包括 Python 注册层、C++ kernel wrapper、wheel 收集规则、API 文档和测试。导入包时还可安装
`torch_npu.sparse` Python façade，但不得覆盖其他模块已占用的命名空间。

> 本仓优先复用 PyTorch 已有 ATen schema，底层调用 ACLSparse C API，并自行管理
> handle、descriptor、workspace 和 stream 生命周期。适配实现不得引入与本仓无关的
> 自定义 schema、ACLNN 包装或图模式转换依赖。

## 1. 目录组织

新增算子文件必须归属到对应算子目录，不得放在仓根 `torch_extension` 中：

```text
ops-sparse/
├── sparse/<op>/
│   ├── archXX/                         # 原有 host / kernel 实现
│   └── torch_extension/
│       ├── __init__.py                 # 导入模块，触发注册
│       ├── <op_api>.py                 # Python dispatch 注册
│       └── csrc/
│           └── <op_api>.cpp            # at::Tensor 到 ACLSparse 的 C++ kernel wrapper
└── torch_extension/
    ├── setup.py                         # 收集各算子的 extension 文件
    ├── README.md                        # 安装和使用说明
    └── cann_ops_sparse/
        ├── __init__.py                  # 包入口：加载注册并安装 façade
        ├── torch_npu_sparse.py          # torch_npu.sparse façade 集中管理
        ├── op_builder/                  # 通用 JIT builder
        ├── ops/                         # 自动发现 wheel 中的算子
        │   └── matmul/                  # 矩阵族路由、ATen 注册及 autograd
        └── docs/
            ├── torch_extension_guidelines.md
            └── zh/<op_api>.md           # 对外 API 文档
```

以 `spmm` 算子为例，其 extension 源码位于
`sparse/spmm/torch_extension/`。新增其他算子时应放在自己的
`<category>/<op>/torch_extension/` 目录；不得通过修改 `spmm` 的 Python 注册或 C++ wrapper
来实现注册。

当前 `setup.py` 的收集规则如下：

1. 将 `sparse/<op>/torch_extension/` 顶层的 `*.py` 复制到
   `cann_ops_sparse/ops/sparse/<op>/`；
2. 将 `sparse/<op>/torch_extension/csrc/` 顶层的 `*.cpp` 复制到
   `cann_ops_sparse/csrc/sparse/<op>/`；
3. 根据发现的算子生成 category 级 `__init__.py`，包导入时再由
   `cann_ops_sparse.ops` 自动发现并导入算子包。

因此，新增嵌套 Python 模块、头文件或其他编译源时，必须同步扩展 `setup.py` 的收集
规则并验证 wheel 内容。不同算子的 C++ 源码按算子目录隔离；同一算子内不得存在同名
源文件。生成的 `__init__.py` 必须是语法合法、可实际导入的 Python 文件，不能写入
字面量形式的转义换行符。

新增算子必须同时交付 `sources()` 声明的 C++ kernel wrapper，并完成 wheel/JIT 验证。每个算子
都应能通过 `bash build.sh --torch_extension --ops=<op>` 生成最小 wheel；该 wheel 不要求
包含或导入 `spmm` 的注册实现。

## 2. API 与命名规范

### 2.1 API 选择原则

1. **优先适配标准 PyTorch API**：先确认 PyTorch 是否已有 public Python API 和 ATen
   schema，例如 `torch.sparse.addmm` 对应 `aten::_sparse_addmm`。
2. **不得重复定义已有 schema**：已有 ATen 算子只注册 NPU 实现，不得再次调用
   `Library(..., "DEF")` 或 `define()` 定义同名 schema。
3. **签名严格一致**：dispatcher 的参数顺序、默认值、关键字参数和返回值必须与目标
   ATen schema 一致。不得为了适配底层接口私自改变 public API 语义。
4. **确认 dispatch key**：根据参与分发的 Tensor layout 注册所有必要的 key。CSR
   Tensor 可能优先进入 `SparseCsrPrivateUse1`，因此不能只注册 `PrivateUse1`。
5. **新增 custom op 是例外**：确无对应 ATen schema 时，须先完成 API 设计评审，明确
   custom namespace、schema、Meta/FakeTensor 实现、对外入口及兼容策略。自定义算子名
   不得伪装成 `aten` 算子。

### 2.2 各层命名

以下以 `spmm` 算子为例：

| 层级 | 命名约定 | 示例 |
| --- | --- | --- |
| public PyTorch API | 沿用 PyTorch 官方名称和语义 | `torch.sparse.addmm` |
| ATen schema | 使用 PyTorch 已有完整限定名 | `aten::_sparse_addmm` |
| 算子目录 | 与底层算子目录一致，小写蛇形 | `sparse/spmm/` |
| Python 文件 | 小写蛇形，表达适配的算子 | `spmm.py` |
| OpBuilder 子类 | 大驼峰 + `OpBuilder` | `SpMMOpBuilder` |
| builder 实例 | 单下划线前缀、小写蛇形 | `_spmm_op_builder` |
| dispatcher 函数 | 单下划线前缀，并体现 key 或 layout | `_sparse_addmm_sparse_csr_privateuse1` |
| C++ kernel wrapper 文件 | 小写蛇形，且 category 内唯一 | `spmm.cpp` |
| pybind 导出名 | 与 Python 调用 wrapper 的属性名一致 | `sparse_addmm` |
| API 文档 | 与 public API 或适配名对应 | `spmm.md` |

同一参数在 ATen schema、Python dispatcher、C++ kernel wrapper、底层 ACLSparse 调用和文档中
必须使用一致的语义。底层 ACLSparse 接口名保持 C API 原名，不为迎合 Python 名称而
重命名。

### 2.3 标识符与类型

- Python 函数、变量和参数使用小写蛇形；类使用大驼峰；模块常量使用全大写蛇形；
  内部符号使用单下划线前缀。
- Python 新增函数应提供类型注解；对外 façade 或 custom API 必须提供完整 docstring。
- C++ 函数和局部变量遵循仓内既有风格；常量使用具名 `const`/`constexpr`，禁止使用
  难以理解的裸魔数。
- 必选 Tensor 使用 `const at::Tensor &`；可选 Tensor 使用
  `const c10::optional<at::Tensor> &`；整型属性优先使用 `int64_t`，并在窄化转换前
  检查范围。
- dtype、layout、index type 等枚举必须显式转换并校验，不得依赖不同枚举值恰好相同。

## 3. Python 注册层

Python 文件负责声明 C++ 源码、注册 ATen dispatch，并在 NPU 调用发生时加载 C++ wrapper。

### 3.1 OpBuilder

每个算子应继承 `OpBuilder`，在构造函数中传入唯一的 JIT 模块名和 category，并实现
`sources()`。以下以 `spmm` 算子为例：

```python
class SpMMOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("spmm", category="sparse")

    def sources(self):
        return ["csrc/sparse/spmm/spmm.cpp"]
```

`sources()` 中的路径以安装后的 `cann_ops_sparse` 包根为基准，必须与 `setup.py` 的目标
路径一致。JIT 模块名在进程内必须唯一，不能与其他算子共用 `_loaded_ops` 缓存键。

模块导入时可以实例化 builder 并调用 `_ensure_initialized()` 完成后端初始化，但不得
调用 `load()`。`load()` 只能出现在实际 dispatcher 执行路径中，以保证“首次调用才
编译”，避免仅导入包就依赖编译器、CANN 环境或生成构建产物。

### 3.2 ATen dispatch 注册

#### 3.2.1 普通单算子注册

矩阵乘法族以外的算子继续在自己的 Python 模块中使用 `torch.library.impl`，
为已有 ATen schema 注册 NPU 实现。以下以仓内 Gather 的 `aten::index_select` 为例：

```python
import torch
from torch.library import impl


@impl("aten::index_select", "PrivateUse1")
def _index_select_privateuse1(self: torch.Tensor, dim: int,
                             index: torch.Tensor) -> torch.Tensor:
    if self.dim() != 1:
        raise RuntimeError("aclsparseGather index_select requires a 1-D input")
    if dim not in (0, -1):
        raise IndexError(f"Dimension out of range for 1-D Gather input: {dim}")
    return _gather_op_builder.load().index_select(self, index)
```

`_gather_op_builder` 按 3.1 节定义，模块顶层仅实例化 builder。
这里的 `PrivateUse1` 适用于该实现的稠密输入；其他算子若接受 COO/CSR 输入，
应按自身 schema 和输入布局注册 `SparsePrivateUse1` / `SparseCsrPrivateUse1`，
不能照搬 Gather 的 dispatch key。overload（如 `index_select.out`）需按其完整签名
单独注册，不支持的 overload 不得注册占位实现。

算子包的 `__init__.py` 显式导入注册模块，例如 `from . import gather`；框架自动发现
并导入这个包时装配注册，首次实际调用才执行 JIT。普通算子不声明
`MATMUL_CAPABILITIES`，也不通过矩阵统一路由调用。

已有 ATen 自动求导规则会继续使用，但其依赖的 NPU 算子必须单独验证；注册一个
前向 kernel 不等于支持反向。确需补充反向依赖时，按对应 ATen schema 注册，
不能把普通算子的 backward 放入矩阵能力表。

#### 3.2.2 矩阵乘法族统一注册

矩阵乘法族的 ATen 注册由 `ops/matmul/registration.py` 唯一管理。算子包只声明
`MATMUL_CAPABILITIES`，不得重复注册 `mm/addmm/_sparse_addmm/_sparse_sparse_matmul`
及其 out overload。

例如，后续 SpMM 包可声明：

```python
from ...matmul.dispatch import MatmulCapability


def execute_spmm(request):
    extension = _spmm_op_builder.load()
    if request.out is not None:
        raise RuntimeError("SpMM provider has no out= implementation")
    if request.operation == "mm":
        return extension.spmm(request.mat1, request.mat2)
    return extension.sparse_addmm(request.self, request.mat1, request.mat2,
                                  request.beta, request.alpha)


MATMUL_CAPABILITIES = (
    MatmulCapability(owner="spmm", route="spmm", forward=execute_spmm),
)
```

此处 `spmm` / `sparse_addmm` 是待接入 provider 的 pybind 签名示例，须随其真实
C++ 导出调整；本仓当前没有 SpMM provider。`forward` 接收 `MatmulRequest`：
operation 为 `mm` 或 `addmm`，包含 mat1/mat2、可选 self/out 和 alpha/beta，并保留原始 ATen `schema`。
SpMM 收到 CSR×dense 的规范化输入；dense×sparse 由框架用普通转置归约，
COO coalesce 后转 CSR，CSC 转 CSR。provider 必须保持设备、dtype 和数值语义，
out 实现负责写回并返回传入 out 对象，不支持的调用应明确报错。
SpGEMM 接收匹配的 COO×COO（mm）或 CSR×CSR；CSR addmm 的 self 必须为 CSR。

可通过 operations、dtypes、device_types 限定能力。重复 route/operation 注册会在
导入时失败；缺失能力在调用时明确报错。包导入仅装配注册，不调用 JIT load。

#### 3.2.3 矩阵算子反向接入参考

可选 `backward` 接收原始 request 和 `grad_output`，返回
`(grad_self, grad_mat1, grad_mat2)`；没有累加项或不需要梯度的输入返回 `None`。
公共层为已声明 backward 的稀疏请求安装 `AutogradPrivateUse1` 实现，保存输入并
调用该反向函数；其他请求继续分发到 ATen 的原有 autograd。

VJP 是 **vector–Jacobian product（向量–雅可比乘积）**，指将上游梯度
`grad_output` 通过算子的导数传播为各输入梯度。此处可以直接理解为“反向函数”，
不需要显式构造雅可比矩阵。

以未来 SpMM 的 `Y = beta * self + alpha * (A @ B)` 为例，设上游梯度为 `G`，
稠密形式的反向公式为：

```text
grad_A    = conj(alpha) * G @ Bᴴ
grad_B    = conj(alpha) * Aᴴ @ G
grad_self = conj(beta) * G，再归约到 self 的原始 shape
```

`ᴴ` 表示共轭转置。这是通常的稠密公式；正式实现必须以原始 `request.schema`
对应的 PyTorch 反向规则为准。PyTorch 2.10 的 `aten::_sparse_addmm` 对 self 和 mat1
分别使用未共轭的 beta/alpha，mat2 使用 `conj(alpha)`；普通 `aten::addmm` 三项
使用共轭标量。`aten::mm` 不做输入 pattern 投影，`aten::_sparse_sparse_matmul`
按原生 `sparse_mask_like_grad` 分支处理结构。indices 不求导。
下面给出有限数值、稠密上游的接入方式和数学参考实现，便于其他矩阵 provider 复用。
它通过稠密化计算展示语义；非有限数值及稀疏上游还须对照原生稀疏结构和传播规则。
正式 SpMM 实现应替换为自身的稀疏反向 kernel，以避免完整稠密矩阵的
内存与计算开销；所用 `to_dense`、复数运算、`sparse_mask` 等 NPU 依赖需实际验证。

```python
import torch
from ...matmul.dispatch import MatmulCapability


def _as_dense(tensor):
    return tensor if tensor.layout == torch.strided else tensor.to_dense()


def _conjugate_scalar(value):
    return value.conjugate() if isinstance(value, complex) else value


def _input_gradient(dense_gradient, original):
    if original.layout == torch.strided:
        return dense_gradient.sum_to_size(original.shape)
    return dense_gradient.sparse_mask(original)


def backward_spmm(request, grad_output):
    # backward 接收原始输入、原始 ATen schema 和上游梯度布局，
    # 并非 forward 已转换为 CSR×dense 的 request。
    a, b = request.mat1, request.mat2
    g = _as_dense(grad_output)
    alpha = _conjugate_scalar(request.alpha)
    beta = _conjugate_scalar(request.beta)
    alpha_a = request.alpha if request.schema == "aten::_sparse_addmm" else alpha
    beta_self = request.beta if request.schema == "aten::_sparse_addmm" else beta
    grad_a = grad_b = grad_self = None
    if a.requires_grad:
        dense_grad_a = alpha_a * torch.mm(g, _as_dense(b).mH)
        grad_a = (_input_gradient(dense_grad_a, a)
                  if request.schema == "aten::_sparse_addmm" else dense_grad_a)
    if b.requires_grad:
        grad_b = alpha * torch.mm(_as_dense(a).mH, g)
    if request.self is not None and request.self.requires_grad:
        grad_self = _input_gradient(beta_self * g, request.self)
    return grad_self, grad_a, grad_b

# execute_spmm 和 _spmm_op_builder 按上一节提供。
# 用这个声明替换前向示例的 MATMUL_CAPABILITIES，而非再追加同一路由。
MATMUL_CAPABILITIES = (
    MatmulCapability(owner="spmm", route="spmm", forward=execute_spmm,
                     backward=backward_spmm),
)
```

反向实现负责共轭、广播归约及原输入的稀疏梯度结构。若正式实现复用 forward 的
布局转换，必须自行将上游梯度和返回梯度变换回原输入布局；公共层不会自动执行
反向的转置或投影。不得对零 alpha/beta 直接返回零梯度，否则会屏蔽原生的 NaN 传播。

`backward_helpers` 可声明构造器反向或梯度累加所需的辅助实现，例如
`(("sparse_mask", sparse_mask_impl),)`。当前 provider 可声明的 schema 为
`sparse_mask`；CSR 梯度累加所需的 `add.out`、`add_.Tensor` 由公共 matmul 层统一实现和
注册，provider 不得重复声明。新增 helper 须同步扩展公共层并验证完整 ATen 语义。
声明 helper 不会替代主算子的 `backward`。

当前公共 autograd 包装器仅支持一阶反向，不支持二阶反向和 `out=` 自动求导。
新增 provider 应覆盖两侧输入、累加项、广播、复数、原始稀疏布局和梯度累加，
通过 NPU 验证后才能在支持列表中声明反向能力。

注册时必须满足：

1. 完整限定算子名、overload、参数顺序、默认值和返回结构与 ATen schema 一致；
2. 针对实际参与分发的稀疏 layout 补齐 dispatch key，并分别覆盖测试；
3. dispatcher 只做必要的参数转发，不复制 C++ wrapper 中的大段校验和资源管理逻辑；
4. 不在模块顶层执行 NPU 计算、同步设备或触发 JIT 编译；
5. 若现有 ATen 算子已提供 Meta 实现，不重复注册；若新增 custom op，则必须提供与真实
   输出 shape、dtype、layout 完全一致的 Meta/FakeTensor 实现。

### 3.3 算子包导入

`sparse/<op>/torch_extension/__init__.py` 应显式导入承担注册的模块，使 wheel 中的自动
发现能够收集能力或触发非矩阵算子的注册。矩阵算子包只导出自身能力，不互相依赖。内部注册模块不作为 public API 时，`__all__` 可保持为空；如确需导出 custom
API，应显式列入 `__all__`，禁止依赖通配符导入的偶然行为。

## 4. C++ 后端（C++ kernel wrapper）

`csrc/<op_api>.cpp` 负责把 PyTorch Tensor 安全地转换为 ACLSparse 描述符并下发计算。
新增源文件必须包含仓库许可证头，并遵循以下顺序。

### 4.1 参数校验

在访问数据指针或申请大块内存前使用 `TORCH_CHECK` 完成校验，至少覆盖：

- 所有 Tensor 均已定义，并位于支持的 NPU device；需要同设备的 Tensor 必须一致；
- sparse layout、稠密 layout、维度、shape、stride/连续性满足底层接口约束；
- value dtype、compute type、output dtype 的组合确有对应 kernel；
- CSR/COO 索引 dtype、index base、行列范围及 `nnz` 满足接口约束；
- 标量类型和取值范围正确，窄化到 `int32_t`/`uint32_t` 前不会溢出；
- 输出 shape 的计算经过溢出检查，不能由非法输入导致越界或超量分配。

错误信息应包含参数名、期望值和实际值。可预期的不支持组合应稳定返回可识别的错误，
不得静默选择其他 dtype/layout 的 kernel。

### 4.2 device、stream 与输出

1. 在申请输出和 workspace 前，根据主输入设置 `c10::OptionalDeviceGuard`，并使用
   `{}` 作用域将 DeviceGuard 与所有输出 Tensor 的申请包在一起。DeviceGuard 必须先于
   `at::empty` 等输出申请生效；否则在非默认 NPU 上调用时，输出可能被创建到当前默认
   设备，导致输入与输出 device 不一致。
2. 获取 PyTorch 当前 NPU stream，而不是默认 stream；将 ACLSparse handle 绑定到该
   stream，保证与调用方已有算子保持正确顺序。
3. 输出 Tensor 的 shape、dtype、layout 和 device 必须与 ATen 语义及 Meta 推导一致；
   禁止通过隐式 host round-trip 推导输出。
4. 正常执行路径必须保持异步，不得在每次调用末尾主动执行 device synchronize。

推荐写法如下：

```cpp
at::Tensor output;
{
    const c10::OptionalDeviceGuard device_guard(c10::Device(input.device()));
    output = at::empty(output_sizes, input.options());
    // 其余输出 Tensor 也必须在此作用域内申请。
}
```

### 4.3 ACLSparse 资源生命周期

- 每个 ACLSparse API 返回值必须通过公共层 `ACLSPARSE_CHECK(...)` 校验，并转换为包含
  接口名和状态码的 `TORCH_CHECK` 异常；不得忽略状态码。
- 稀疏/稠密 descriptor 和其他调用级原生资源必须在所有成功与异常路径中恰好释放
  一次。优先使用局部 RAII 封装，避免多处手工清理造成泄漏或重复释放。
- 公共层按 `device + stream` 缓存 ACLSparse handle，并为同一 handle 串行化调用；算子
  wrapper 不得自行缓存或跨 stream 复用 handle。缓存随扩展卸载或进程退出析构并释放。
- workspace 大小必须由对应 API 查询，并通过公共层 `AclSparseWorkspace` 申请；该类会
  记录 workspace 到 handle 初始化时绑定的 stream，不能在异步任务完成前复用或销毁其 storage。
- 临时 Tensor、输入和输出 storage 必须按 torch_npu 提供的机制记录到当前 stream。优先
  使用 `AclSparseContext::RecordTensors(...)`；记录的必须是 handle 初始化时绑定的 stream，
  不能依赖 Python 局部变量离开作用域后的偶然存活。
- 禁止缓存带有调用级状态的 descriptor、裸数据指针或 stream；除公共层的 handle 缓存外，
  如缓存只读全局对象，必须说明线程安全和多 device 行为。

公共层与算子层的边界如下：

| 公共层 `common/aclsparse_common.h` | 算子 C++ kernel wrapper |
| --- | --- |
| DeviceGuard、当前 stream、按 device + stream 缓存的 ACLSparse handle | 输入校验与 ATen 输出语义 |
| `ACLSPARSE_CHECK(...)` 状态检查 | 稀疏／稠密 descriptor 的构造和释放 |
| `AclSparseWorkspace` 申请与 stream record | `GetBufferSize → Preprocess → Compute` 等算子专属调用序列 |
| `RecordTensors(...)` 的 storage 生命周期管理 | 算法选择、tiling 与算子特有参数 |

以 `spmm` 算子为例，wrapper 应只声明 descriptor 和调用 SpMM 序列；不得在算子侧重复
创建 handle、设置 stream、手工申请 workspace 或重复实现状态检查。

### 4.4 wrapper 导出

使用 `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` 导出 Python dispatcher 实际调用的
函数，导出名必须与 `builder.load().<name>` 完全一致。C++ 函数签名、参数顺序和返回值
必须与 Python dispatcher 对齐；修改任意一层时须同步修改并补充测试。

## 5. `torch_npu.sparse` façade

标准 ATen 适配的主要入口仍是 PyTorch public API。`torch_npu.sparse` 仅提供等价的
Python façade，不应绕过 ATen dispatcher 直接调用 C++ wrapper。

所有 façade 安装和导出必须由 `torch_npu_sparse.py` 统一执行：

1. 同时检查 `torch_npu.sparse` 属性与 `sys.modules["torch_npu.sparse"]`；
2. 仅当两者都未被占用时创建模块；若被第三方占用，必须明确报错，严禁覆盖；
3. 仅允许本包通过 owner marker 识别并重复使用自己创建的同一模块；
4. 算子注册模块通过 `TORCH_NPU_SPARSE_FACADE_APIS` 声明 `{名称: public PyTorch API}`
   映射；矩阵乘法族由公共 `ops/matmul/registration.py` 统一声明该映射，算子包只声明能力。
   `cann_ops_sparse.ops` 自动收集，安装时自动生成 `namespace.__all__`；
5. 新增 façade 函数必须保证其签名和语义与对应 public PyTorch API 一致；
6. 算子自身的 `__init__.py` 和注册模块不得直接修改 `torch_npu` 属性或 `sys.modules`。

## 6. 打包与导入规范

- 构建逻辑必须保持确定性：目录和算子按稳定顺序收集，禁止依赖文件系统遍历顺序。
- 自动发现只导入包含 `__init__.py` 的算子目录；新增算子必须验证安装后的路径和导入
  行为，不能只在源码树中测试。
- wheel 必须包含 dispatcher 所声明的全部 C++ 源码和编译所需头文件。源码缺失应在
  首次调用时给出包含算子名和缺失路径的清晰错误。
- 通用 builder、自动发现和 façade 逻辑放在仓根 `torch_extension`；算子专有逻辑留在
  `sparse/<op>/torch_extension`，禁止将单算子判断持续堆叠到公共模块。
- 公共 builder 的错误信息不得硬编码某个具体 ATen 算子名，应使用当前 builder 的名称
  生成诊断信息。
- 新增算子的最小 wheel 不得依赖其他算子包的 import、副作用注册或 C++ wrapper 源码；以
  `spmm` 算子为例，其他算子的 `--ops=<op>` 构建不应收集它的文件。

## 7. 文档规范

每个对外 ATen 适配或 custom API 必须在
`torch_extension/cann_ops_sparse/docs/zh/<op_api>.md` 提供中文文档。以 `spmm` 算子为例，
章节顺序如下；其他算子按同一章节结构独立编写：

1. **产品支持情况**：以表格列出支持的 Ascend 产品和架构；
2. **功能说明**：写明 public PyTorch API、ATen schema、底层 ACLSparse 接口和
   `torch_npu.sparse` façade，并描述计算语义、公式及符号定义；
3. **函数原型**：给出完整 Python 签名，包含默认值和关键字参数；
4. **参数说明**：逐项说明语义、shape、dtype、layout、device、连续性、取值范围、默认
   值及参数间一致性约束；
5. **输出**：说明输出数量、shape、dtype、layout、device 及别名/原地行为；
6. **约束说明**：列出不支持的 dtype/layout/transpose/index 类型、空 Tensor 和边界
   shape 行为，以及对应异常类型；
7. **调用示例**：提供可运行的 NPU 示例，覆盖标准 PyTorch API；若有 façade，再给出
   等价调用方式；

如 API 有特殊 stream、同步或生命周期要求，须在“约束说明”中明确说明。

文档、Python docstring、ATen schema 和实现必须保持一致。API 支持列表发生变化时，应
同步更新 `docs/zh/api_list.md`。

## 8. 测试规范

测试放在对应算子的现有 `test/<op>/` 或仓库统一测试体系中，不在
`torch_extension/tests/` 新建脱离算子的测试树。以 `spmm` 算子为例，至少覆盖：

- **注册测试**：导入包后，各必需 dispatch key 已注册，且导入不会触发 JIT 编译；
- **功能与精度**：典型 shape、所有支持的 dtype/layout/index type、默认参数和非默认
  参数与 CPU/reference 结果一致；
- **边界场景**：空 Tensor、零 `nnz`、最小/最大合法 shape、非连续 Tensor 及转置场景
  按文档行为执行；
- **非法输入**：device/dtype/layout/shape/index type 不匹配和整数溢出能够在访问设备
  前稳定失败，错误信息可定位；
- **设备与异步**：非默认 NPU、非默认 stream、连续多次调用及必要的并发调用不存在
  错误设备、提前释放或隐式同步；
- **打包测试**：从构建出的 wheel 安装并导入，确认 Python/C++ 源码收集完整，首次 NPU
  调用能够 JIT 编译并加载；
- **命名空间测试**：`torch_npu.sparse` 首次安装、幂等导入和第三方占用冲突行为正确。

若当前环境无法执行 NPU 测试，提交说明中必须列出未执行项、原因和替代验证，不能将
“未执行”等同于“已通过”。

## 9. 通用编码约束

- 所有新增 `.py`、`.cpp`、`.h` 文件必须包含 Huawei 版权与 CANN Open Software
  License Agreement Version 2.0 许可证头，年份使用创建年份。
- 对外 API 必须提供 docstring；C++ 中对资源所有权、stream 生命周期和非显然的边界
  检查添加简洁注释，注释说明“为什么”，避免逐行复述代码。
- 优先复用公共 builder、状态检查和资源管理能力，不在每个算子中复制同类实现。
- 禁止吞掉异常、忽略返回码、使用裸 `new/delete` 管理调用级资源或通过主动同步掩盖
  生命周期问题。
- Python 代码遵循 PEP 8，C++ 代码遵循仓库现有格式；提交中不得夹带无关格式化。

## 10. 提交前自检清单

- [ ] 已确认 public PyTorch API、完整 ATen schema 和所有必要 dispatch key。
- [ ] 未重复定义已有 ATen schema，Python dispatcher 签名与 schema 完全一致。
- [ ] 文件位于 `sparse/<op>/torch_extension/`，命名符合规范且 C++ 文件名无冲突。
- [ ] `sources()` 路径与 wheel 内路径一致，导入包不会触发 JIT 编译。
- [ ] wheel 包含全部 Python、C++ 和头文件，并通过安装后导入/JIT 验证。
- [ ] C++ wrapper 完成 device、dtype、layout、shape、索引和窄化范围校验。
- [ ] 已用 `{}` 作用域包住 DeviceGuard 和全部输出申请，DeviceGuard 在申请前生效。
- [ ] handle 绑定当前 stream，执行路径无主动同步。
- [ ] handle、descriptor、workspace 和 Tensor storage 在所有路径上生命周期正确。
- [ ] pybind 导出名与 `builder.load()` 调用名一致。
- [ ] `torch_npu.sparse` façade 未覆盖第三方命名空间，`__all__` 已同步维护。
- [ ] 已覆盖功能、精度、非法输入、边界、dispatch、非默认 device/stream 和 wheel 测试。
- [ ] API 文档、docstring、支持列表与实现保持一致。
- [ ] 未执行的硬件或环境相关测试已在提交说明中明确记录。
