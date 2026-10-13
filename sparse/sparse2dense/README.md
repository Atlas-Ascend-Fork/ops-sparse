# aclsparseSparseToDense 算子

> 本文面向 `ops-sparse` 接口使用者。

## 功能描述

`SparseToDense` 将 CSR、CSC 或 COO 格式的稀疏矩阵转换为稠密矩阵。接口语义对标 cuSPARSE Generic API 的 `cusparseSparseToDense_bufferSize` 和 `cusparseSparseToDense`。推荐按查询 workspace、执行转换两个阶段调用。

计算如下表达式：

$$
B_{ij} = \begin{cases} A_{ij} & \text{若 } (i,j) \in \text{nnz}(A) \\ 0 & \text{否则} \end{cases}
$$

其中：
- $A$ 是 $m \times n$ 的稀疏矩阵（CSR / CSC / COO 格式）
- $B$ 是 $m \times n$ 的稠密矩阵（输出，行主序或列主序）

主要特性：
- **纯数据搬运**：kernel 仅执行 `B[idx] = A.val[p]` 的 scatter 写操作，无任何算术运算、无类型转换，输入输出同 dtype，故**零精度损失**（bit-exact）
- **三格式支持**：CSR、CSC、COO 三种稀疏格式，通过描述符 `format` 字段自动分流
- **Reg 优先（asc-devkit）**：arch35 使用 `AscendC::Reg::Scatter` / `Duplicate` 在 UB 条带内离散写回再 `DataCopyPad`；禁止与 SIMT 混编
- **多核并行**：按输出布局主轴（ROW→行 / COL→列）切分 AI Core；次轴按 `tileLen` 切 UB 条带
- **全 dtype 输出同型**：直接以原始 dtype 读写，不做 float 中转；含 COMPLEX64（按 uint64 bit-copy）
- **支持异步执行**：两个 API 均异步返回，同步由调用方通过 stream 负责
- **零 workspace**：`bufferSize` 恒返回 0
- **支持行/列主序输出**：通过 `aclsparseOrder_t` 指定，支持任意 leading dimension（ld）

## 架构支持

| 架构 | 硬件 | 实现目录 | 说明 |
|------|------|----------|------|
| arch35 | Ascend 950PR / 950DT | `arch35/` | Reg 优先 scatter；memset 清零；无数据级校验 |
| arch22 | Atlas A2 训练系列（910B3/910B4）、Atlas A3 训练系列（910_93），DAV_2201 | `arch22/` | 清零 + scatter 双 kernel；Host 校验链（CheckMeta → 坐标校验 → FillTiling），坐标越界/offsets 非法拒绝，重复坐标合法（结果取任一写入值） |

两架构 `bufferSize` 均恒返回 0（零 workspace 协议）。**aclsparse API 层**：重复 `(row, col)`（COO 直接重复与 CSR/CSC 段内展开重复）合法，结果取任一写入值（并行环境下写序不确定，本实现显式约定；cuSPARSE 对重复索引不做正确性承诺）；坐标越界仍返回 `ACL_SPARSE_STATUS_INVALID_VALUE`，乱序坐标可接受。**torch_extension 层**：遵循 PyTorch `Tensor.to_dense` 对未 coalesce COO 重复坐标的求和语义，所有 SoC 进入原生算子前预归并（见 `torch_extension/sparse2dense.py`）；CSR/CSC 段内重复在 torch 层同样取任一写入值。arch22 原生接受未 coalesce 的 COO 张量（C API 能力；torch_extension 层统一预归并后进入）。

## 接口原型

```cpp
aclsparseStatus_t aclsparseSparseToDense_bufferSize(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    size_t *bufferSize);

aclsparseStatus_t aclsparseSparseToDense(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    void *buffer);
```

`matA` 是只读稀疏矩阵描述符，`matB` 是可变稠密矩阵描述符。接口不会修改 `matA`
或其数据。

## 参数说明

两个接口的公共参数如下。

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|----------|------|------|------|
| `handle` | Host | 输入 | `aclsparseHandle_t` | `ops-sparse` 上下文句柄，携带执行 stream，不可为空 |
| `matA` | Host 描述符；数据在 Device | 输入 | `aclsparseConstSpMatDescr_t` | 稀疏矩阵 A，包含格式、行列数、索引类型、indexBase、值类型和数据指针 |
| `matB` | Host 描述符；数据在 Device | 输入/输出 | `aclsparseDnMatDescr_t` | 稠密矩阵 B，包含行数、列数、`ld`、布局、数据类型和数据指针；执行时写入 |
| `alg` | Host | 输入 | `aclsparseSparseToDenseAlg_t` | 仅支持 `ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT` |

各接口的专有参数如下。

| 接口 | 参数 | 内存位置 | 方向 | 说明 |
|------|------|----------|------|------|
| `bufferSize` | `bufferSize` | Host | 输出 | workspace 字节数，不可为空；当前恒返回 0 |
| `execute` | `buffer` | Device | 输入 | workspace；`bufferSize == 0` 时可为空 |

## 推荐调用流程

1. 创建 handle、稀疏矩阵描述符 `matA` 和稠密矩阵描述符 `matB`。
2. 调用 `aclsparseSparseToDense_bufferSize`，按返回大小分配 Device workspace（当前恒为 0，可跳过）。
3. 调用 `aclsparseSparseToDense` 执行转换。
4. 同步 handle 绑定的 stream，读取结果。

## 稀疏格式语义

| 格式 | `matA` 描述符字段 | 说明 |
|------|-------------------|------|
| CSR | `ptrs`=rowOffsets(m+1), `idxs`=colInd(nnz) | 按行压缩；kernel 按行切分多核 |
| CSC | `ptrs`=colOffsets(n+1), `idxs`=rowInd(nnz) | 按列压缩；kernel 按列切分多核 |
| COO | `ptrs`=cooRowInd(nnz), `idxs`=cooColInd(nnz) | 坐标格式；kernel 按 nnz grid-stride 切分 |

CSR/CSC 的 `ptrType` 与 `IdxType` 必须均为 `ACL_SPARSE_INDEX_32I`。COO 的 `cooIdxType`
必须为 `ACL_SPARSE_INDEX_32I`。所有格式均支持 `ACL_SPARSE_INDEX_BASE_ZERO` 和
`ACL_SPARSE_INDEX_BASE_ONE`。

重复的 `(row, col)`（COO 直接重复与 CSR/CSC 段内展开重复）合法，结果为其中任意一个值（并行环境下写顺序不确定，本实现显式约定；cuSPARSE 对重复索引不做正确性承诺）。坐标经 base 换算后仍须落在 `[0, rows) × [0, cols)` 内，越界在 Host 校验链返回参数错误，不发起 kernel。torch_extension 层对未 coalesce COO 重复坐标按 PyTorch `to_dense` 求和语义预归并（见 `torch_extension/sparse2dense.py`）；CSR/CSC 段内重复在 torch 层同样取任一写入值。

## 支持数据类型

arch35 实现支持以下 value 类型，且 `matA` 与 `matB` 的 value 类型必须相同。

| 数据类型 | 枚举 | 元素大小 |
|----------|------|---------|
| float32 | `ACL_FLOAT` | 4B |
| float16 | `ACL_FLOAT16` | 2B |
| bfloat16 | `ACL_BF16` | 2B |
| int32 | `ACL_INT32` | 4B |
| int8 | `ACL_INT8` | 1B |
| complex64 | `ACL_COMPLEX64` | 8B（Reg 路径按 `uint64_t` bit-copy） |

arch22 实现支持以下 value 类型（无 int32，含 complex64），`matA` 与 `matB` 的 value 类型必须相同。

| 数据类型 | 枚举 | 元素大小 |
|----------|------|---------|
| float32 | `ACL_FLOAT` | 4B |
| float16 | `ACL_FLOAT16` | 2B |
| bfloat16 | `ACL_BF16` | 2B |
| int8 | `ACL_INT8` | 1B |
| complex64 | `ACL_COMPLEX64` | 8B |

两架构均不支持 float64；传入不支持的类型返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。

## 数值、布局与边界语义

- 同时支持 `ACL_SPARSE_ORDER_ROW`（row-major）和 `ACL_SPARSE_ORDER_COL`（column-major）。
  row-major 要求 `ld >= n`，column-major 要求 `ld >= m`。
- leading dimension (ld) 可大于最小值（即允许 stride padding），用于满足对齐需求。
- matA 和 matB 的 value 类型必须一致，不支持跨类型转换。kernel 以原始 dtype 读写，
  保证 bit-exact 无精度损失。
- matB 的 Device 内存由算子内部清零（arch35：热路径为 SIMT uint64 grid-stride，`nnz=0`/空矩阵走 `aclrtMemsetAsync`；arch22：清零 kernel，覆盖全部存储缓冲含 ld padding），调用方无需预先清零。
- NaN（包括其 payload）、`+Inf`、`-Inf`、`+0`、`-0` 和普通有限非零值均按位搬运，
  不执行数值计算或类型转换。
- 支持零维矩阵、全零矩阵和全非零矩阵。`nnz=0` 时输出清零后直接返回（arch35 不 launch kernel；arch22 由清零 kernel 承接输出定义）。
- 矩阵行数、列数不可超过 `INT32_MAX`。

## Workspace 与生命周期

- workspace 是每次调用使用的临时 scratch，不保存跨调用状态。`bufferSize` 恒返回 0，
  `buffer` 可传 `nullptr`。
- 执行按 handle 的 stream 异步执行，调用者必须在读取或释放输出、描述符及 handle 前同步该 stream。
- 描述符只借用 Device 指针，不拥有其内存。先同步 stream，再销毁描述符并释放其引用的 Device 内存。

## 返回值 / 错误码

| 返回值 | 含义 |
|--------|------|
| `ACL_SPARSE_STATUS_SUCCESS` | 成功 |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为 NULL |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | matA/matB 为 NULL；bufferSize 为 NULL；stream 未设置；维度不匹配；**execute** 时 matB.values 为 NULL；nnz>0 但 idxs/values 为 NULL |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 格式非 CSR/CSC/COO；索引类型非 I32；值类型不在支持列表；matA/matB 值类型不一致；alg 非 DEFAULT；m/n 超过 INT32_MAX |

## 调用示例

```cpp
#include <cstdio>
#include <memory>
#include "cann_ops_sparse.h"
#include "acl/acl.h"

#define LOG_PRINT(fmt, ...) do { printf(fmt, ##__VA_ARGS__); } while (0)

#define CHECK_RET(cond, msg) do { \
    if (!(cond)) { LOG_PRINT("ERROR: %s at line %d\n", msg, __LINE__); return ret; } \
} while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}
    ~AclContext() {
        if (stream_) { aclrtDestroyStream(stream_); stream_ = nullptr; }
        if (deviceSet_) { aclrtResetDevice(deviceId_); deviceSet_ = false; }
        if (aclInited_) { aclFinalize(); aclInited_ = false; }
    }
    aclError Init() {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, "aclInit failed");
        aclInited_ = true;
        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice failed");
        deviceSet_ = true;
        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream failed");
        return ACL_SUCCESS;
    }
    aclrtStream Stream() const { return stream_; }
private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

static void *AllocAndCopyDevice(const void *hostPtr, size_t sizeBytes) {
    void *dPtr = nullptr;
    aclrtMalloc(&dPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (hostPtr != nullptr && sizeBytes > 0)
        aclrtMemcpy(dPtr, sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    return dPtr;
}

int main() {
    AclContext ctx(0);
    aclError ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, "AclContext init failed");

    // CSR 矩阵数据（4x4，nnz=5，0-based）
    int32_t hRowOff[] = {0, 2, 3, 5, 5};
    int32_t hColInd[] = {0, 2, 1, 0, 2};
    float   hValues[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int64_t rows = 4, cols = 4, nnz = 5;

    // 分配 Device 内存并拷贝 CSR 数据
    std::unique_ptr<int32_t, decltype(&aclrtFree)> dRowOff(
        static_cast<int32_t *>(AllocAndCopyDevice(hRowOff, sizeof(hRowOff))), &aclrtFree);
    std::unique_ptr<int32_t, decltype(&aclrtFree)> dColInd(
        static_cast<int32_t *>(AllocAndCopyDevice(hColInd, sizeof(hColInd))), &aclrtFree);
    std::unique_ptr<float, decltype(&aclrtFree)> dValues(
        static_cast<float *>(AllocAndCopyDevice(hValues, sizeof(hValues))), &aclrtFree);
    std::unique_ptr<float, decltype(&aclrtFree)> dDnMat(
        static_cast<float *>(AllocAndCopyDevice(nullptr, rows * cols * sizeof(float))), &aclrtFree);
    CHECK_RET(dRowOff && dColInd && dValues && dDnMat, "device alloc failed");

    // 创建 handle
    aclsparseHandle_t rawHandle = nullptr;
    auto st = aclsparseCreate(&rawHandle);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, "aclsparseCreate failed");
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handle(rawHandle, aclsparseDestroy);
    aclsparseSetStream(handle.get(), ctx.Stream());

    // 创建 CSR 稀疏矩阵描述符 matA
    aclsparseConstSpMatDescr_t rawMatA = nullptr;
    st = aclsparseCreateConstCsr(&rawMatA, rows, cols, nnz,
        dRowOff.get(), dColInd.get(), dValues.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, "createConstCsr failed");
    std::unique_ptr<aclsparseSpMatDescr, aclsparseStatus_t (*)(aclsparseConstSpMatDescr_t)> matA(
        const_cast<aclsparseSpMatDescr *>(reinterpret_cast<const aclsparseSpMatDescr *>(rawMatA)),
        aclsparseDestroySpMat);

    // 创建稠密矩阵描述符 matB（行主序，ld=cols）
    aclsparseDnMatDescr_t rawMatB = nullptr;
    st = aclsparseCreateDnMat(&rawMatB, rows, cols, cols, dDnMat.get(),
        ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, "createDnMat failed");
    std::unique_ptr<aclsparseDnMatDescr, aclsparseStatus_t (*)(aclsparseConstDnMatDescr_t)> matB(
        rawMatB, aclsparseDestroyDnMat);

    // 查询 workspace（恒为 0）
    size_t bufferSize = 0;
    st = aclsparseSparseToDense_bufferSize(handle.get(), matA.get(), matB.get(),
        ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, "bufferSize failed");

    // 执行 CSR → Dense 转换
    st = aclsparseSparseToDense(handle.get(), matA.get(), matB.get(),
        ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, nullptr);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, "execute failed");

    // 同步并取回结果
    ret = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(ret == ACL_SUCCESS, "sync failed");
    float hDnMat[16] = {0};
    ret = aclrtMemcpy(hDnMat, sizeof(hDnMat), dDnMat.get(), sizeof(hDnMat),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "D2H failed");

    // 打印结果
    printf("Dense matrix (%lld x %lld, row-major):\n", (long long)rows, (long long)cols);
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j)
            printf("%6.1f ", hDnMat[i * cols + j]);
        printf("\n");
    }
    return 0;
}
```

预期输出：

```
Dense matrix (4 x 4, row-major):
   1.0    0.0    2.0    0.0 
   0.0    3.0    0.0    0.0 
   4.0    0.0    5.0    0.0 
   0.0    0.0    0.0    0.0 
```

## 测试覆盖

测试文件位于 `test/sparse2dense/`，采用 GTest + CSV 参数化测试框架。

- **arch35（Ascend 950 真机）**：50 个测试用例（32 正确性 + 16 负向 + 2 跳过）全部通过。
- **arch22（Atlas A2/A3 真机）**：ST 全量 123 OK + 3 SKIP（L2_001/L2_016/MultiDevice 单卡口径跳过）、Host UT 73 PASS + 1 SKIP、Python e2e 133/133、官方 ATK 精度 200/200。

### arch35 测试结构

| 文件 | 说明 |
|------|------|
| `sparse2dense_param.h` | CSV 参数结构体 |
| `sparse2dense_golden.h` | CPU golden 计算（CSR/CSC/COO 三格式） |
| `arch35/sparse2dense_test.csv` | L0/L1 正确性用例（32 条） |
| `arch35/sparse2dense_l2_cases.csv` | L2 负向用例（16 条） |
| `arch35/sparse2dense_npu_wrapper.h` | NPU 输入生成 / 类型映射 / 执行封装 |
| `arch35/sparse2dense_test.cpp` | GTest 主测试 |

### arch35 L0/L1 正确性测试（32 条）

| 级别 | 覆盖维度 |
|------|----------|
| L0 | 基本功能：CSR/CSC/COO × FP32/FP16/BF16/INT8/INT32 × row/col × base 0/1 |
| L0 | 边界：空矩阵 / 单元素 / 全密 / ld padding / 宽矩阵 / 高矩阵 |
| L1 | 中等规模：128×128 ~ 512×512 × 三格式 × 五 dtype |
| L1 | indexBase=1 覆盖 |
| L1 | 三格式一致性（同矩阵 CSR/CSC/COO 输出必须一致） |

### arch35 L2 负向测试（16 条）

| 测试类别 | 覆盖内容 |
|----------|----------|
| 空指针 | handle / matA / matB / bufferSize 为 NULL |
| 维度不匹配 | matA 与 matB 行列数不一致 |
| 类型不匹配 | matA 与 matB 值类型不一致 |
| 非法 ld | row-major ld < cols |
| 非法格式 | format 枚举越界 |
| 非法算法 | alg 枚举越界 |
| 不支持 dtype | FP64 |
| nnz>0 空指针 | ptrs / idxs 为 NULL |
| INT32 越界 | m > INT32_MAX |
| 不支持索引类型 | I64 |

### arch22 测试结构

| 文件 | 说明 |
|------|------|
| `arch22/sparse2dense_test.cpp` | GTest 主测试（L0/L1/L2 + 多卡 + 坐标校验链负例） |
| `arch22/sparse2dense_test.csv` | 三级用例单文件（L0 12 + L1 85 + L2 37 = 134 条，测试运行时装载；L2_001/016 为 mock-only SKIP；L2_029~037 为第十轮评审整改公开 API 双入口负例） |
| `arch22/ut/` | Host 侧单元测试（含 aclrt mock） |
| `python/test_sparse2dense_torch.py` | torch_extension 端到端（aten::_to_dense 接管/格式×dtype bit-wise/错误路径/多卡） |

## 支持芯片

| 芯片 | 支持情况 |
|------|----------|
| Ascend 950PR / Ascend 950DT | 支持（arch35 / dav-3510） |
| Atlas A2 训练系列（Ascend 910B3 / 910B4） | 支持（arch22 / dav-2201） |
| Atlas A3 训练系列（Ascend 910_93） | 支持（arch22 / dav-2201） |

## PyTorch / Torch Extension

PyTorch 适配走仓内统一的 `torch_extension` 框架（与 SpGEMM 相同），注册标准
`aten::_to_dense`，用户调用 `Tensor.to_dense()`。源码目录：

```text
sparse/sparse2dense/torch_extension/
├── __init__.py
├── sparse2dense.py
└── csrc/sparse2dense.cpp
```

构建、安装与 pytest 命令见
[`../../test/sparse2dense/python/README.md`](../../test/sparse2dense/python/README.md)、
[`../../torch_extension/README.md`](../../torch_extension/README.md) 与
[`../../torch_extension/cann_ops_sparse/docs/zh/sparse2dense.md`](../../torch_extension/cann_ops_sparse/docs/zh/sparse2dense.md)。
