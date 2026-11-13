# SPVV 算子

## 概述

ops-sparse 仓库中的 SPVV (Sparse Vector-Vector Dot Product) 算子实现了稀疏向量与稠密向量的点积运算：

$$
\text{result} = \sum_{i=0}^{\text{nnz}-1} x\_values[i] \cdot y[x\_indices[i]]
$$

其中 x 为稀疏向量（含 nnz 个非零元素，索引升序唯一），y 为稠密向量，结果恒为 FP32 标量。算子基于 aclsparse 统一接口。

## 支持的 AI 处理器

| 产品                                        | 是否支持 |
| ------------------------------------------- | :------: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Ascend 950PR / Ascend 950DT                      |    ×    |

> 当前仅提供 arch22（DAV-2201）实现，覆盖 ascend910b（A2）与 ascend910_93（A3）。Ascend 950PR/950DT（arch35 / DAV-3510）尚未适配。

## 目录结构介绍

```
sparse/spvv/arch22/
├── spvv_tiling_data.h                // Tiling 数据结构与常量
├── spvv_kernel.h                     // kernel_do 启动器声明（Host/Kernel 共用）
├── spvv_host.cpp                     // Host 侧 API 实现（aclsparseSpvv）
└── spvv_kernel.cpp                   // Kernel 侧向量化实现

test/spvv/
├── CMakeLists.txt                    // 调用 ops_sparse_add_test(spvv ${OPS_SPARSE})
├── README.md                         // 说明文档
└── arch22/
    └── spvv_test.cpp                 // 910B 算子调用样例（正确性 + 性能 + 随机 fuzz）
```

> SpVec 描述符的创建/销毁（`aclsparseCreateSpVec` / `aclsparseDestroySpVec` 等）由 `sparse/common/aclsparse_descr.cpp` 统一实现，不在 spvv 目录内。

## 算子描述

### 功能

SPVV 算子计算稀疏向量 x 与稠密向量 y 的点积，结果为单个 FP32 标量。x 的非零元素由 `(index, value)` 对表示，索引必须严格升序且唯一，取值范围 `[0, yLen-1]`。

### 支持的类型组合

| valueType (x/y) | computeType (累加/输出) | 说明 |
| :-------------: | :---------------------: | :--- |
|    float32      |        float32          | FP32 输入，FP32 累加 |
|    float16      |        float32          | FP16 输入，Cast 到 FP32 后做乘法与累加（避免 half×half 精度损失） |

> **注意**：`computeType` 固定为 `ACL_FLOAT`（累加与输出恒为 FP32）；`valueType`（x/y 元素类型）可为 FP32 或 FP16，核函数按 `valueType` 分发。x 与 y 的 `valueType` 必须一致。

### 实现原理

1. **无 D2H 拷贝 + tiling 直传**：参考 cuSPARSE，kernel 端自行按 nnz 均匀分核，host 侧不再将 indices 拷贝到 host 做 tiling；`SpvvTilingData` 直接作为核函数参数传入，不走 device buffer。
2. **按 nnz 分核 + AtomicAdd 跨核规约**：`blockNum = ceil(nnz / nnzPerCore)`，每核处理连续一段 nnz；host 侧在 launch 前用 `aclrtMemsetAsync`（同 stream）将 output 置零，各核 partialSum 经 `SetAtomicAdd` 累加到 output GM，`PipeBarrier<PIPE_ALL>` 确保 DMA 完成后再退出。无需 kernel 内置零或核间同步。
3. **向量化 Gather + Mul + ReduceSum**：每 tile 把所需 y 片段加载到 UB，Gather 出对应 y 元素，与 x 值 Mul，ReduceSum 累加。
4. **fast path**：tile 的 y-span ≤ `YSLICE_MAX` 时单次加载 + 单次 Gather，跳过段循环。
5. **y 片段分段 + 32B 对齐拼接**：y 片段过长时按 `YSLICE_MAX` 分段，count 向下取整到 `ALIGN_ELEM`(32B) 保证 Gather dst 偏移对齐，拼接进同一 `yGathered`。
6. **二分查找 segment 边界**：indices 已排序，用 upper_bound 二分查找（O(log n)）替代线性扫描。
7. **Muls 预计算 + idxUb 复用**：整 tile 一次性 `idx*InputSz`，段内仅一次 `Adds`；直接在 `idxUb[nnzPos]` 子张量上 in-place 转偏移，避免每段 GM 重载。
8. **segYEnd 裁剪**：count 确定后将 segYEnd 裁剪到实际最后 index，减少 y 的 DMA 加载量。
9. **标量兜底**：极稀疏（段内 index 数 < `ALIGN_ELEM`）时用标量 `GetValue` 手动 gather，支持任意稀疏度。
10. **FP16 精度**：FP16 路径先 Cast 两输入到 FP32 再 Mul（FP32 乘积），消除 half×half 在大 nnz 下的累积误差/符号翻转。
11. **最小每核 nnz**：小 nnz 时每核至少 `SPVV_MIN_NNZ_PER_CORE`(512) 个元素，减少启动核数、摊薄 per-core 开销。
12. **核数获取**：通过 `PlatformAscendCManager::GetCoreNumAiv()` 获取 AIV 核数（kernel 直调惯用法）。

### 算子规格参数说明

| 参数名     | 输入/输出/属性 | 描述                                                   | 数据类型           | 数据格式 |
| ---------- | -------------- | ------------------------------------------------------ | ------------------ | -------- |
| x (SpVec)  | 输入           | 稀疏向量描述符（indices 升序唯一，范围 [0,yLen-1]）   | float / half       | ND       |
| y (DnVec)  | 输入           | 稠密向量描述符                                         | 与 x 一致          | ND       |
| result     | 输出           | 点积结果（device 指针，1 个 float）                    | float32            | scalar   |
| op         | 属性           | 操作类型（仅 `ACL_SPARSE_OP_NON_TRANSPOSE`）           | enum               | —        |
| computeType| 属性           | 累加/输出类型（仅 `ACL_FLOAT`）                        | enum               | —        |

### 约束说明

- **op**：仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`
- **idxType**：仅支持 `ACL_SPARSE_INDEX_32I`
- **idxBase**：仅支持 `ACL_SPARSE_INDEX_BASE_ZERO`
- **indices**：必须严格升序且唯一，取值 `[0, yLen-1]`（**不校验**，越界会导致 y 越界读、结果未定义/崩溃，由 caller 保证）
- **nnz ≤ yLen**：由 caller 保证（不校验）
- **computeType**：固定 `ACL_FLOAT`
- 无 device buffer 需求

### 测试实现

- **测试流程** (`spvv_test.cpp`)

1. **生成测试数据**：Fisher-Yates 部分洗牌生成唯一升序 indices，随机 x/y 值
2. **CPU 参考计算**：`compute_golden_fp32` / `compute_golden_fp16`（double 累加，高精度参考值）
3. **初始化 ACL 环境**：aclInit + aclrtSetDevice
4. **创建描述符**：`aclsparseCreateSpVec` + `aclsparseCreateDnVec`
5. **执行 SpVV**：`aclsparseSpvv`，同步后回读结果
6. **结果验证**：与 golden 比较，FP32 容差 1e-3 相对，FP16 容差 1e-2 相对

- **测试覆盖**

| 测试组            | 用例数 | 说明                                         |
| ----------------- | :----: | -------------------------------------------- |
| 正确性（表驱动）  |  10×2  | tiny/zero/sparse/small/medium/large × FP32+FP16 |
| benchmark         | 10×2  | scalar fallback + 小中大规模 × FP32+FP16     |
| 随机 fuzz         | 100×2 | 随机 nnz∈[0,yLen]、yLen∈[256,2M] × FP32+FP16 |

- **错误处理**：`CHECK_ACL` / `CHECK_ACL_SPARSE` 宏检测到错误时打印原因并 `return false`，不继续执行。
- **dtype 过滤**：测试二进制支持可选参数 `./spvv_test [fp32|fp16|all]`（默认 `all`），可单独验证某个 dtype 路径，便于隔离 FP32/FP16 特有问题。

### 精度验证方法

- **FP32**：相对误差 `|npu - golden| < 1e-3 * max(1, |golden|)`
- **FP16**：相对误差 `|npu - golden| < 1e-2 * max(1, |golden|)`（FP16 输入，FP32 乘积累加）
- golden 用 `double` 累加，消除参考值自身在大 nnz 下的累积误差

## 编译运行

在 ops-sparse 仓库根目录下执行：

### 配置环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 样例执行

```bash
bash build.sh --ops=spvv --run
```

执行结果如下，说明精度对比成功：

```txt
=== FP32 SpVV [large-1M]: nnz=100000, yLen=1000000 ===
  FP32 [large-1M] nnz=100000 yLen=1000000: golden=-47.697754 npu=-47.697735 diff=1.91e-05 → PASS
...
========================================
Total: 22 passed, 0 failed
========================================
```

### 性能参考（ascend910b3, 48 核, FP32, nnz=100K）

| yLen | 时延 (min) | 吞吐 (Mnnz/s) |
|------|-----------|---------------|
| 1M   | ~0.07 ms  | ~1400         |
| 10M  | ~0.11 ms  | ~900          |

> 时延为 benchmark 20 次迭代的最小值（含 host 侧 `aclrtMemsetAsync` 开销）；avg 约高出 20-40%。

## 接口说明

### aclsparseSpvv

**函数原型**：

```cpp
aclsparseStatus_t aclsparseSpvv(
    aclsparseHandle_t handle,
    aclsparseOperation_t op,
    aclsparseConstSpVecDescr_t x,
    aclsparseConstDnVecDescr_t y,
    void *result,
    aclDataType computeType);
```

**参数说明**：

| 参数        | 方向 | 描述                                                       |
| ----------- | :--: | ---------------------------------------------------------- |
| handle      |  IN  | aclsparse 句柄                                             |
| op          |  IN  | 操作类型（仅 `ACL_SPARSE_OP_NON_TRANSPOSE`）              |
| x           |  IN  | 稀疏向量描述符（indices 升序唯一，范围 [0,yLen-1]）      |
| y           |  IN  | 稠密向量描述符                                             |
| result      | OUT | device 指针，1 个 float（FP32，与 valueType 无关）        |
| computeType |  IN  | 累加/输出类型（仅 `ACL_FLOAT`）                           |

**返回值**：

| 返回值                                  | 说明                                        |
| --------------------------------------- | ------------------------------------------- |
| `ACL_SPARSE_STATUS_SUCCESS`           | 成功                                        |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为空                                 |
| `ACL_SPARSE_STATUS_INVALID_VALUE`     | x/y/result 为空，或 x/y valueType 不一致    |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED`     | op/computeType/idxType/idxBase/valueType 不支持 |

> **注意**：indices 的升序唯一性与值域 `[0, yLen-1]`、nnz ≤ yLen 均**不校验**，由 caller 保证；越界会导致 y 越界读、结果未定义或崩溃。

### aclsparseCreateSpVec / aclsparseCreateConstSpVec / aclsparseDestroySpVec

稀疏向量描述符的创建/销毁由 `sparse/common/aclsparse_descr.cpp` 统一实现，详见 `include/cann_ops_sparse.h`。
