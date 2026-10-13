# 稀疏算子领域补充规则

> 本文件收录 ops-sparse 仓特有的领域约束，补充 `repo-knowledge` SKILL.md 中 API 规范未覆盖的部分。这些规则从工作流硬规则中提取，去除工作流步骤引用后保留领域本质。

---

## 1. cuSPARSE 接口对齐原则

新增 API 必须严格对齐 cuSPARSE 对应接口的规范，按算子所属 API 体系执行：

| API 体系 | 对齐要求 |
|----------|---------|
| **Generic 体系** | 接口签名参照 cuSPARSE Generic API、使用 `aclsparseSpMatDescr_t` / `aclsparseDnVecDescr_t` / `aclsparseDnMatDescr_t`、三阶段执行（若需要） |
| **Legacy 体系** | 接口签名参照 Sparse BLAS 风格、带精度前缀（S/D/C/Z）、使用 `aclsparseMatDescr_t`、扁平参数 |
| **两者共用** | 状态码使用 `aclsparseStatus_t` 枚举值、操作类型使用 `aclsparseOperation_t`、索引类型使用 `aclsparseIndexType_t` / `aclsparseIndexBase_t` |

> API 体系的选择在需求分析阶段确定，后续开发严格遵循所选体系的规范。

---

## 2. 描述符提取校验（Generic API）

所有稀疏算子的 Host 侧必须正确管理描述符（Descriptor）的创建和使用：

- **Create/Destroy API 必须配对调用**，禁止泄漏
- **Host 侧从描述符提取信息时，必须校验**：
  - 格式（`format`）：确认描述符的稀疏格式与算子支持的格式匹配（如 CSR/COO/CSC）
  - 维度（`rows`/`cols`/`nnz`）：确认维度合法且与算子预期一致
  - 数据类型（`valueType`）：确认数据类型被算子支持
- 空指针、非法维度等无效输入校验失败时返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；不支持的稀疏格式或数据类型返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`（具体见状态码表）。校验失败时通过 `OP_LOGE` 记录原因
- const 描述符的 `const_cast` 必须在 `ToMatInner`/`ToVecInner`/`ToDnMatInner` 等统一函数中完成，禁止在业务代码中直接 cast

**示例**：

```cpp
auto* matInner = ToMatInner(matA);
if (matInner == nullptr) {
    OP_LOGE("aclsparseSpMV", "matA is nullptr");
    return ACL_SPARSE_STATUS_INVALID_VALUE;
}
if (matInner->format != ACL_SPARSE_FORMAT_CSR) {
    OP_LOGE("aclsparseSpMV", "unsupported format: %d", matInner->format);
    return ACL_SPARSE_STATUS_NOT_SUPPORTED;
}
if (matInner->rows == 0 || matInner->cols == 0) {
    return ACL_SPARSE_STATUS_SUCCESS;  // 边界快速返回
}
```

---

## 3. 公共代码复用

优先复用 `sparse/common/` 下的公共模块，禁止在算子代码中重复定义公共宏或工具函数。

### 已有公共模块

| 模块 | 路径 | 职责 |
|------|------|------|
| `aclsparse_handle_internal.h` | `sparse/common/` | Handle 内部结构体 + `ToInternalHandle` 转换函数 |
| `aclsparse_descr_internal.h` | `sparse/common/` | 描述符内部结构体 + `ToMatInner`/`ToVecInner`/`ToDnMatInner` 转换函数 |
| `host_utils.h` | `sparse/common/` | `GetAivCoreCount` 等公共 Host 工具函数 |

### 强制规则

- **禁止**在算子 host 文件中重复定义 `GetAivCoreCount` — 由 `host_utils.h` 提供
- **禁止**在算子代码中重复定义 `GM_ADDR`、`CHECK_RET` 等公共宏 — 已由公共头文件提供
- **禁止**在算子代码中重复定义描述符转换函数 — 由 `aclsparse_descr_internal.h` 提供
- 若新增通用功能，必须先补充到 `sparse/common/` 对应头文件，再在算子代码中调用

---

## 4. Legacy API 描述符生命周期

Legacy API 无 SpMatDescr/DnVecDescr/DnMatDescr 描述符，但部分接口需要矩阵格式属性：

- 若接口需要矩阵格式属性，使用轻量 `aclsparseMatDescr_t`（`aclsparseCreateMatDescr` / `aclsparseDestroyMatDescr` 配对）
- 数据指针扁平传入 Host 侧（无描述符封装）
- `aclsparseMatDescr_t` 的 getter（`aclsparseGetMatType` / `aclsparseGetMatIndexBase` 等）在 Host 侧调用，禁止自行存储冗余字段
- 同一算子的多精度版本（S/D 函数）需按精度前缀分别命名，禁止共享实现（避免精度丢失）
- 扁平数据指针校验 nullptr（每个指针独立校验，不能批量）

---

## 5. 索引运算约定

稀疏算子涉及索引运算时的强制约定：

- 行偏移数组长度为 `rows + 1`，必须校验一致性（`csrRowOffsets[rows]` == nnz）
- nnz 必须与值数组/列索引数组的实际长度匹配
- 索引基址（0-based/1-based）处理逻辑必须显式标注，通过 `aclsparseIndexBase_t` 控制
- Device 指针使用 `GM_ADDR`（`uint8_t *`）传递
- Host 指针使用 `const void *`（alpha/beta 标量，仅 Generic 使用）
- 不允许在 Kernel 侧直接访问 Host 内存
