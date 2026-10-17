# aclsparseLtMatmulAlgGetAttribute

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- **算子功能**：`aclsparseLtMatmulAlgGetAttribute` 查询算法选择描述符（`aclsparseLtMatmulAlgSelection_t`）的属性当前值。它是 `aclsparseLtMatmulAlgSetAttribute` 的查询伴生接口，对标 cuSPARSELt 中的 `cusparseLtMatmulAlgGetAttribute`。
- **纯 Host 操作**：该接口为纯 host 端元数据查询，不启动任何 device kernel，不涉及 Device 内存访问。
- **算子目录**：v2 版本将 `aclsparseLtMatmulAlgGetAttribute` 从 `alg_set_attribute` 目录拆分为独立算子目录 `alg_get_attribute/arch35/`，与 `alg_set_attribute` 目录解耦，便于独立维护与测试。SetAttribute 的 host 实现文件头注释中已标注 GetAttribute 的独立路径。

### 可查询属性

| 属性枚举 | 值类型 | 说明 |
|----------|--------|------|
| `ACLSPARSELT_MATMUL_ALG_CONFIG_ID` | int32_t | 当前算法配置 ID（0 或 1），影响 cube tiling（baseM/baseN/baseK） |
| `ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID` | int32_t | 算法配置 ID 上限（返回 2，即 algConfigId 0 和 1 共 2 种配置）。只读属性，不可通过 SetAttribute 设置 |
| `ACLSPARSELT_MATMUL_SEARCH_ITERATIONS` | int32_t | 搜索迭代次数（默认 5），当前仅存储不参与执行 |
| `ACLSPARSELT_MATMUL_SPLIT_K` | int32_t | splitK 切分因子，取值范围 [1, K] |
| `ACLSPARSELT_MATMUL_SPLIT_K_MODE` | int32_t | splitK 模式（0=ONE_KERNEL 融合单 kernel，1=TWO_KERNELS matmul+epilogue 两 kernel） |
| `ACLSPARSELT_MATMUL_SPLIT_K_BUFFERS` | int32_t | splitK 缓冲区数，取值范围 [0, splitK-1]，当前仅存储不参与执行 |

## 接口原型

```cpp
aclsparseStatus_t aclsparseLtMatmulAlgGetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attr,
    void* attrValue,
    size_t attrValueSize);
```

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseLtConstHandle_t | 库句柄的 const 指针，须非空且 *handle 非空，Host 内存 |
| algSelection | 输入 | aclsparseLtConstMatmulAlgSelection_t* | 算法选择描述符，须非空且已初始化（*algSelection 非 nullptr），Host 内存 |
| attr | 输入 | aclsparseLtMatmulAlgAttribute_t | 要查询的属性枚举，Host 内存 |
| attrValue | 输出 | void* | 属性值输出缓冲区，须非空，Host 内存。写入的数据类型与大小由 `attr` 决定（均为 int32_t，4 字节） |
| attrValueSize | 输入 | size_t | `attrValue` 缓冲区大小（字节），须等于 `sizeof(int32_t)`（4 字节），Host 内存 |

> 与 `aclsparseLtMatmulAlgSetAttribute` 参数语义对称，区别为 `attrValue` 方向为输出、`algSelection` 为 const 只读。

## 支持数据类型

本接口为纯 host 端元数据查询，不涉及矩阵数据类型。查询返回的属性值均为 `int32_t`。

## 约束说明

- **handle 有效性**：`handle` 须非空且 `*handle` 非空，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`。
- **algSelection 有效性**：`algSelection` 须非空且已通过 `aclsparseLtMatmulAlgSelectionInit` 初始化（`*algSelection` 非 nullptr）。已销毁的描述符（`*algSelection` 为 nullptr）返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- **attrValue 非空**：`attrValue` 须非空，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- **attrValueSize 匹配**：`attrValueSize` 须等于 `sizeof(int32_t)`（4 字节），否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- **attr 枚举范围**：`attr` 须为上表所列枚举值之一，否则返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **查询时机**：可在 `aclsparseLtMatmulAlgSelectionInit` 之后任意时刻查询（PlanInit 前后均可）。可设置属性始终读取 algSelection 中当前存储的值；PlanInit 后再次成功调用 SetAttribute，GetAttribute 也会返回更新后的属性值，而非已有 plan 的初始化快照。查询 SPLIT_K 返回存储的 splitK，不返回 PlanInit 计算的 effectiveSplitK。因此不能用本查询接口读取或验证已有 plan 的冻结 tiling/workspace 配置。
- **与 SetAttribute 的关系**：GetAttribute 读取的值为最近一次 SetAttribute 设置的值（或 AlgSelectionInit 的默认值：algConfigId=0、splitK=1、searchIterations=5、splitKMode=0、splitKBuffers=0）。

## 返回值 / 错误码

| 返回值 | 含义 |
|--------|------|
| `ACL_SPARSE_STATUS_SUCCESS` | 查询成功 |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为 nullptr 或 *handle 为 nullptr |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | algSelection 或 attrValue 为 nullptr、*algSelection 为 nullptr（已销毁），或 attrValueSize 不匹配 sizeof(int32_t) |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | attr 枚举值不在支持范围内 |

## 调用示例

以下示例演示完整的 AlgSelectionInit → AlgSetAttribute → AlgGetAttribute → 校验流程。计算 `D = A_pruned * B`（FP32，行主序），设置 algConfigId=0、splitK=2 后查询验证。

```cpp
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparseLt.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int aclsparseLtAlgGetAttributeTest()
{
    int32_t deviceId = 0;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, aclFinalize(); return -1);

    // 1. 初始化 aclsparseLt 库句柄
    aclsparseLtHandle_t handle = nullptr;
    auto ret = aclsparseLtInit(&handle);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseLtInit failed: %d\n", ret); return ret);

    // 2. 创建矩阵描述符 (FP32, 行主序)
    int64_t m = 64, k = 64, n = 128;
    aclsparseLtMatDescriptor_t matA = nullptr;
    aclsparseLtMatDescriptor_t matB = nullptr, matC = nullptr, matD = nullptr;
    aclsparseLtStructuredDescriptorInit(
        &handle, &matA, m, k, k, 16, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    aclsparseLtDenseDescriptorInit(&handle, &matB, k, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matC, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matD, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);

    // 3. 创建 Matmul 描述符与算法选择描述符
    aclsparseLtMatmulDescriptor_t matmulDesc = nullptr;
    aclsparseLtMatmulDescriptorInit(
        &handle, &matmulDesc,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &matA, &matB, &matC, &matD, ACL_SPARSE_COMPUTE_32F);

    aclsparseLtMatmulAlgSelection_t algSel = nullptr;
    aclsparseLtMatmulAlgSelectionInit(&handle, &algSel, &matmulDesc, ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);

    // 4. 设置 algConfigId=0、splitK=2
    int32_t algConfigId = 0;
    int32_t splitK = 2;
    aclsparseLtMatmulAlgSetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_ALG_CONFIG_ID,
                                      &algConfigId, sizeof(int32_t));
    aclsparseLtMatmulAlgSetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_SPLIT_K,
                                      &splitK, sizeof(int32_t));

    // 5. 查询属性并校验
    int32_t queryAlgConfigId = -1;
    int32_t querySplitK = -1;
    int32_t queryMaxId = -1;
    ret = aclsparseLtMatmulAlgGetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_ALG_CONFIG_ID,
                                            &queryAlgConfigId, sizeof(int32_t));
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("GetAttribute(ALG_CONFIG_ID) failed: %d\n", ret); return ret);
    ret = aclsparseLtMatmulAlgGetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_SPLIT_K,
                                            &querySplitK, sizeof(int32_t));
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("GetAttribute(SPLIT_K) failed: %d\n", ret); return ret);
    ret = aclsparseLtMatmulAlgGetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
                                            &queryMaxId, sizeof(int32_t));
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("GetAttribute(ALG_CONFIG_MAX_ID) failed: %d\n", ret); return ret);

    LOG_PRINT("algConfigId=%d (expected 0), splitK=%d (expected 2), maxId=%d (expected 2)\n",
              queryAlgConfigId, querySplitK, queryMaxId);

    // 6. 清理资源（按依赖逆序销毁）
    aclsparseLtMatmulAlgSelectionDestroy(&algSel);
    aclsparseLtMatmulDescriptorDestroy(&matmulDesc);
    aclsparseLtMatDescriptorDestroy(&matA);
    aclsparseLtMatDescriptorDestroy(&matB);
    aclsparseLtMatDescriptorDestroy(&matC);
    aclsparseLtMatDescriptorDestroy(&matD);
    aclsparseLtDestroy(&handle);

    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

int main()
{
    return aclsparseLtAlgGetAttributeTest();
}
```

## 支持芯片

<term>Ascend 950PR/Ascend 950DT</term>（arch35 / DAV_3510）

## 参考资源

- 对标接口：cuSPARSELt [cusparseLtMatmulAlgGetAttribute](https://docs.nvidia.com/cuda/cusparselt/index.html#cusparseltmatmulalggetattribute)
- 伴生接口：[aclsparseLtMatmulAlgSetAttribute](../alg_set_attribute/README.md)（设置算法选择属性）
- 完整调用链：[aclsparseLtMatmul](../matmul/README.md)（结构化稀疏矩阵乘，含描述符/算法选择/计划/执行完整流程）
