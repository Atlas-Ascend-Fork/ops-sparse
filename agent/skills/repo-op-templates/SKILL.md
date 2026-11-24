---
name: repo-op-templates
description: 算子代码模板库，提供代码模板与模板选择规则，作为算子代码开发的起点。
---

# Skill: sparse-op-templates

ops-sparse 算子代码模板库，为不同编程模型和目标架构提供标准化的代码骨架。Agent 在开发新算子时，应以对应类型的模板为起点，按算子需求填充业务逻辑。

---

## 算子分类体系

ops-sparse 仓中的算子按 **编程模型** 和 **目标架构** 两个维度分类：

### 编程模型

| 编程模型 | 关键特征 | 适用场景 | 模板目录 |
|---------|---------|---------|---------|
| **SIMD** | `TPipe` / `TQue` / `DataCopyPad` / `SetFlag/WaitFlag<HardEvent::...>` | 所有架构通用的向量编程模型（默认首选） | `references/simd/` |
| **RegBase** | `TBuf` / `LocalTensor` / `__VEC_SCOPE__` / `RegTensor` / `MicroAPI::DataCopy/Mul/Add/ReduceSum` | 寄存器级 SIMD 算子，在 UB 内使用寄存器张量计算（仅 arch35） | `references/regbase/` |
| **SIMT** | `__simt_vf__` / `asc_vf_call` / `threadIdx.x` / `blockDim.x` / `grid-stride loop` | 线程级并行算子（仅 arch35） | `references/simt/` |

> RegBase 模板与 SIMD 模板的关键区别：使用 `TBuf + LocalTensor`（非 `TQue`）、`__VEC_SCOPE__` 块内 `RegTensor` 寄存器级 API、host 侧预计算 UB buffer 大小。

> SIMT 模板与 SIMD/RegBase 模板的关键区别：使用 `__simt_vf__` 装饰器、`asc_vf_call` 调度、`grid-stride loop` 线程级并行、无 `TPipe/TQue/DataCopyPad`（直接 `__gm__` 指针访问）。

### 目标架构

| 架构 | SOC_VERSION | NPU_ARCH | 支持的编程模型 | 说明 |
|------|------------|----------|--------------|------|
| arch20 | ascend310p* | dav-2002 | SIMD | 推理芯片 |
| arch22 | ascend910b* / ascend910_93* | dav-2201 | SIMD | 训练/推理芯片 |
| **arch35** | **ascend950*** | **dav-3510** | SIMD / RegBase / SIMT | Ascend 950PR&950DT系列产品（当前重点） |

> **架构默认反选**：目标 arch 目录默认由 SOC 参数反选（ascend950 → arch35，ascend910b/910_93 → arch22，ascend310p → arch20），优先使用该默认值。

> SIMD 编程模型在所有架构上可用，是默认选择。Kernel 侧代码的架构差异由 AscendC 编译器屏蔽。RegBase 和 SIMT **仅 Ascend 950PR&950DT系列产品 (arch35) 可用**。

> **编程模型选择**：算子开发模板选择时，应优先评估 SIMD 方案（SIMD / RegBase）的可行性。仅在经评估后当前目标架构或算子数据依赖模式确实无法使用 SIMD 表达时，才允许退化为 SIMT 方案。

---

## 模板目录结构

每个编程模型的模板文件按仓库实际目录层级组织，从 `sparse/` 一级开始，仅包含算子实现代码（**2 层**，无 `{family}/` 中间层）：

```
references/<programming-model>/
  sparse/{op}/
    archxx/                       -- SIMD 通用（arch20/arch22/arch35 均适用）
      op_tiling_data.h            -- Tiling 数据结构（host/kernel 共享）
      op_kernel.h                 -- kernel_do 签名（host.cpp / kernel.cpp 共用）
      op_kernel.cpp               -- Device 侧 kernel 实现
      op_host.cpp                 -- Host 侧 API 入口 + Validate/Launch 拆分 + dlog 集成
    arch35/                       -- RegBase / SIMT（仅 arch35）
      op_tiling_data.h
      op_kernel.h
      op_kernel.cpp
      op_host.cpp
```

说明：
- `simd` 模板适用于 arch20/arch22/arch35，使用 `archxx/` 作为通用目录名
- `regbase` 模板仅适用于 arch35（DAV_3510 RegBase 模式），使用 `arch35/` 目录名
- `simt` 模板仅适用于 arch35（arch22 不支持 SIMT 编程模型），使用 `arch35/` 目录名
- `{op}` 为 snake_case 格式的算子名（如 `spmv`、`spmm`、`sp_geam`）

---

## 算子交付件目录结构

本 skill 覆盖算子的 **`sparse/` 代码交付件**（下方文件布局）。一个完整算子在仓库中的代码布局：

```
sparse/<operator_name>/
  archXX/
    <operator_name>_host.cpp
    <operator_name>_kernel.cpp
    <operator_name>_kernel.h
    <operator_name>_tiling_data.h
```

- 算子代码位于 `sparse/{op_name}/archXX/`，测试代码位于 `test/{op_name}/`
- `{op_name}` 使用 **snake_case** 格式（如 `spmv`、`spmm`、`sp_geam`）
- 本 skill 只覆盖 `sparse/` 代码交付件，不含算子测试代码（测试代码见 `repo-test-develop` skill）

---

## 命名规范

**重要**：算子目录和文件名使用 **snake_case**（下划线分隔），API 名和结构体名使用 **PascalCase**。

| 元素 | 规范 | 示例 |
|------|------|------|
| 算子目录 | `sparse/<operator_name>/`（snake_case） | `sparse/spmv/` |
| Kernel 文件 | `<operator_name>_kernel.cpp` | `spmv_kernel.cpp` |
| Kernel 头文件 | `<operator_name>_kernel.h` | `spmv_kernel.h` |
| Host 文件 | `<operator_name>_host.cpp` | `spmv_host.cpp` |
| Tiling 头文件 | `<operator_name>_tiling_data.h` | `spmv_tiling_data.h` |
| Tiling 结构体 | `<OpName>TilingData`（PascalCase） | `SpmvTilingData` |
| Kernel 类 | `<OpName>AIV`（SIMD/RegBase） | `SpmvAIV` |
| SIMT 计算函数 | `<OpName>SimtCompute` + `__simt_vf__` | `SpmvSimtCompute` |
| Kernel 入口 | `<operator_name>_kernel`（`__global__`） | `spmv_kernel` |
| Kernel 启动器 | `<operator_name>_kernel_do` | `spmv_kernel_do` |
| 公共 API | `aclsparse<OpName>`（PascalCase） | `aclsparseSpMV` |

---

## host.cpp 强制规范

- **函数拆分**：host.cpp 必须拆分为两个静态函数：
  - `static aclsparseStatus_t Validate{Op}Params(...)` — 参数校验（包含空指针、维度、格式、数据类型等）
  - `static aclsparseStatus_t Launch{Op}Kernel(...)` — 描述符解析 + tiling 计算 + kernel launch
  - API 入口 `aclsparse{OpName}(...)` 仅做 handle 校验 + 快速返回 + 调用上述两个函数
- **dlog 强制集成**：host.cpp 必须 `#include "log/log.h"`，按以下约定使用：
  - `OP_LOGE` — 参数校验失败、ACL Runtime 调用失败、不可恢复错误
  - `OP_LOGD` — tiling 数据详细字段
  - `OP_LOGI` — kernel launch 信息（block 数、core 数）
  - 禁止使用 `printf`、`std::cout`
- **kernel.h 共用头文件**：host.cpp 通过 `#include "{op}_kernel.h"` 引入 `kernel_do` 签名，**禁止**在 host.cpp 中写 extern 前向声明
- **异步 kernel 启动**：调用 `kernel_do(...)` 后立即返回，**禁止**在 host 侧调用 `aclrtSynchronizeStream`
- **Tiling 传递**：host 侧以 **`const` 引用** 传入 `kernel_do`；kernel 侧以 **by value** 接收；**禁止**使用 `aclrtMalloc` + `aclrtMemcpy(H2D)` 传递 tiling
- **host include 精简**：host.cpp **禁止**引入冗余 include（`acl/acl.h`、`cann_ops_sparse_common.h`、`tiling/platform/platform_ascendc.h` 均为冗余，由公共头文件间接引入）；仅保留必需头文件：`log/log.h`、`cann_ops_sparse.h`、`{op}_kernel.h`、`aclsparse_handle_internal.h`、`aclsparse_descr_internal.h`

---

## 描述符转换函数

新算子应 `#include "aclsparse_descr_internal.h"` 使用内部结构体，并在本地头文件 `{op_name}.h` 中定义转换函数：

```cpp
inline struct aclsparseContext *ToInternalHandle(aclsparseHandle_t handle) {
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

inline struct aclsparseSpMatDescr *ToMatInner(aclsparseConstSpMatDescr_t desc) {
    return const_cast<struct aclsparseSpMatDescr *>(
        reinterpret_cast<const struct aclsparseSpMatDescr *>(desc));
}

inline struct aclsparseDnVecDescr *ToVecInner(aclsparseConstDnVecDescr_t desc) {
    return const_cast<struct aclsparseDnVecDescr *>(
        reinterpret_cast<const struct aclsparseDnVecDescr *>(desc));
}

inline struct aclsparseDnMatDescr *ToDnMatInner(aclsparseConstDnMatDescr_t desc) {
    return const_cast<struct aclsparseDnMatDescr *>(
        reinterpret_cast<const struct aclsparseDnMatDescr *>(desc));
}
```

## kernel_do 签名模式

```cpp
// {op_name}_kernel.h
#ifndef {OP_NAME}_KERNEL_H_
#define {OP_NAME}_KERNEL_H_

#include "{op_name}_tiling_data.h"

#define GM_ADDR uint8_t *

// kernel_do 签名：
// - 数据指针按算子需求填写（GM_ADDR）
// - tiling 以 const 引用传入
// - numBlocks + stream 为通用参数
void {op_name}_kernel_do(
    GM_ADDR dataPtr1, GM_ADDR dataPtr2, ...,
    uint32_t numBlocks, const {Op}TilingData& tiling, void *stream);

#endif
```

> **Generic API 算子**：若需要 alpha/beta/computeType 参数，在 kernel_do 中按算子需求追加（如 `const void *alpha, const void *beta`），并在 host 侧从描述符或接口参数中提取。

## Host 骨架模式

```cpp
// {op_name}_host.cpp
// 禁止冗余 include：acl/acl.h、cann_ops_sparse_common.h、tiling/platform/platform_ascendc.h 均为冗余
#include <cstdint>
#include <new>
#include "log/log.h"
#include "cann_ops_sparse.h"
#include "{op_name}.h"
#include "{op_name}_kernel.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "host_utils.h"

namespace {
    static aclsparseStatus_t Validate{Op}Params(
        aclsparseHandle_t handle, ...) {
        // 参数校验 - 每个参数 nullptr/value 检查
    }

    static aclsparseStatus_t Launch{Op}Kernel(
        aclsparseHandle_t handle, ...) {
        // 描述符解析 -> TilingData 计算 -> kernel_do launch
    }
} // namespace

extern "C" {
    aclsparseStatus_t aclsparse{Op}(
        aclsparseHandle_t handle, ...) {
        aclsparseStatus_t status = Validate{Op}Params(handle, ...);
        if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
        return Launch{Op}Kernel(handle, ...);
    }
}
```

## 公共宏定义（来自现有代码）

以下是仓内已有的公共宏，新算子**必须复用**，**禁止重新定义**：

```cpp
#define GM_ADDR uint8_t *

#define CHECK_RET(cond, return_expr) \
    do { if (!(cond)) { return_expr; } } while (0)
```

---

## 使用方法

1. 根据算子的编程模型和目标架构，选择对应的模板目录
2. 将模板文件复制到算子目录，按命名规范重命名
3. 按模板中的 `// TEMPLATE:` 注释指引，替换占位逻辑为实际业务逻辑
4. 模板中的代码已包含标准框架（Init/Process、参数校验、Tiling 计算等），只需填充核心算法部分

---

## 参考资源

| 资源 | 路径 | 说明 |
|------|------|------|
| SIMD 模板 | [references/simd/](references/simd/) | 所有架构通用的向量编程模型算子全套模板（archxx 通用） |
| RegBase 模板 | [references/regbase/](references/regbase/) | 寄存器级 SIMD 算子全套模板（仅 arch35） |
| SIMT 模板 | [references/simt/](references/simt/) | 线程级并行算子全套模板（仅 arch35） |
