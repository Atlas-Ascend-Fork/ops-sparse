# sparseLtMatmulPlan 类接口

## 接口概述

sparseLtMatmulPlan 类接口用于创建和销毁 matmul 计算前的准备对象：algSelection（算法选择）与 plan（执行计划）。

algSelection 用于在计算执行前选中本次 matmul 运算采用的算法模式，并将其与参与运算的 matmul 描述符绑定；plan 用于记录完整的执行准备信息，将 matmul 描述符与 algSelection 绑定在一起，供后续 matmul 执行接口直接使用。本类接口只负责这两类对象的创建与销毁，不执行任何数值计算，实际计算由后续的 matmul 计算接口完成。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseLtMatmulAlgSelectionInit | 初始化 algSelection |
| aclsparseLtMatmulAlgSelectionDestroy | 销毁 algSelection |
| aclsparseLtMatmulPlanInit | 初始化 plan |
| aclsparseLtMatmulPlanDestroy | 销毁 plan |

**销毁顺序说明**：plan 与 algSelection 都保存了对 matmul 描述符的引用，销毁它们不会销毁所引用的 matmul 描述符。使用完毕后，应先销毁 plan，再销毁 algSelection，之后销毁 matmul 描述符。如果顺序颠倒，plan 中保存的 algSelection 引用将指向已释放的内存，产生非法访问风险。

## 接口说明

### aclsparseLtMatmulAlgSelectionInit

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 功能描述

初始化一个 algSelection 对象，把 matmul 描述符与算法模式绑定在一起，供后续创建 plan 时使用。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtMatmulAlgSelectionInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulAlgSelection_t*       algSelection,
    const aclsparseLtMatmulDescriptor_t*   matmulDescr,
    aclsparseLtMatmulAlg_t                 alg)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | const aclsparseLtHandle_t* | aclsparseLt 库句柄 | Host |
| algSelection | 输入/输出 | aclsparseLtMatmulAlgSelection_t* | 初始化的 algSelection 对象 | Host |
| matmulDescr | 输入 | const aclsparseLtMatmulDescriptor_t* | matmul 描述符 | Host |
| alg | 输入 | aclsparseLtMatmulAlg_t | 算法模式 | Host |

#### 约束说明

- handle 不可为 nullptr。
- algSelection 不可为 nullptr，且调用前 *algSelection 必须为 nullptr，防止覆盖已存在的 algSelection。
- matmulDescr 不可为 nullptr，且必须已初始化（由 aclsparseLtMatmulDescriptorInit 创建）。
- alg 支持的取值见[支持的算法模式](#支持的算法模式)章节，传入表中之外的取值返回 ACL_SPARSE_STATUS_INVALID_VALUE。

### aclsparseLtMatmulAlgSelectionDestroy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 功能描述

销毁 algSelection 对象，释放其内存并将 *algSelection 置为 nullptr。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtMatmulAlgSelectionDestroy(aclsparseLtMatmulAlgSelection_t* algSelection)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| algSelection | 输入/输出 | aclsparseLtMatmulAlgSelection_t* | 指向要销毁的 algSelection 对象的指针，销毁成功后 *algSelection 被置为 nullptr | Host |

#### 约束说明

- 若 algSelection 指针本身为 nullptr，返回 ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR。
- 若 *algSelection 为 nullptr，视为空操作，返回 ACL_SPARSE_STATUS_SUCCESS。
- 本接口不销毁其引用的 matmul 描述符。

### aclsparseLtMatmulPlanInit

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 功能描述

初始化一个 plan 对象，把 matmul 描述符与 algSelection 绑定在一起，形成一个可供后续 matmul 执行接口使用的完整 plan。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtMatmulPlanInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulPlan_t*               plan,
    const aclsparseLtMatmulDescriptor_t*   matmulDescr,
    const aclsparseLtMatmulAlgSelection_t* algSelection)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | const aclsparseLtHandle_t* | aclsparseLt 库句柄 | Host |
| plan | 输入/输出 | aclsparseLtMatmulPlan_t* | 初始化的 plan 对象 | Host |
| matmulDescr | 输入 | const aclsparseLtMatmulDescriptor_t* | matmul 描述符 | Host |
| algSelection | 输入 | const aclsparseLtMatmulAlgSelection_t* | algSelection 对象 | Host |

#### 约束说明

- handle 不可为 nullptr。
- plan 不可为 nullptr，且调用前 *plan 必须为 nullptr，防止覆盖已有的 plan。
- matmulDescr 不可为 nullptr，且必须已初始化（由 aclsparseLtMatmulDescriptorInit 创建）。
- algSelection 不可为 nullptr，且必须已初始化（由 aclsparseLtMatmulAlgSelectionInit 创建）。
- 传入的 matmulDescr 必须与 algSelection 内部绑定的 matmulDescr 一致（即创建 algSelection 时传入的 matmulDescr），否则返回 ACL_SPARSE_STATUS_INVALID_VALUE。

### aclsparseLtMatmulPlanDestroy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 功能描述

销毁 plan 对象，释放其内存并将 *plan 置为 nullptr。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtMatmulPlanDestroy(aclsparseLtMatmulPlan_t* plan)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| plan | 输入/输出 | aclsparseLtMatmulPlan_t* | 指向要销毁的 plan 对象的指针，销毁成功后 *plan 被置为 nullptr | Host |

#### 约束说明

- 若 plan 指针本身为 nullptr，返回 ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR。
- 若 *plan 为 nullptr，视为空操作，返回 ACL_SPARSE_STATUS_SUCCESS。
- 本接口不销毁其引用的 matmul 描述符与 algSelection。

## 支持的算法模式

| 算法 | 枚举名 | 说明 |
|------|------|------|
| DEFAULT | ACL_SPARSE_LT_MATMUL_ALG_DEFAULT | 默认算法 |

## 返回值

| 返回值 | 含义 |
|--------|------|
| ACL_SPARSE_STATUS_SUCCESS | 操作成功完成。 |
| ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR | 传入的库句柄为 nullptr，或销毁接口传入的对象指针本身为 nullptr。 |
| ACL_SPARSE_STATUS_INVALID_VALUE | 参数校验失败：输出指针为 nullptr 或调用前已非空（防止覆盖）、输入的对象引用为 nullptr 或未初始化、算法模式非法、algSelection 与 matmulDescr 不匹配。 |
| ACL_SPARSE_STATUS_ALLOC_FAILED | 内存分配失败。 |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

> 示例中 `ACL_SPARSE_COMPUTE_16F` 与 FP16 矩阵组合为合法用法（computeType↔valueType 绑定：16F↔仅 FP16；32F↔FP32/FP16/BF16；32I↔INT8，错配返回 NOT_SUPPORTED，绑定口径详见 [matmul_descriptor/README.md](../matmul_descriptor/README.md)）。本实现 16F 按 FP32 累加执行，精度与 32F 路径一致。

```cpp
#include "cann_ops_sparseLt.h"
#include "acl/acl.h"
#include <cstdio>

int main()
{
    // 0. 初始化 ACL runtime
    aclError aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", aclRet);
        return -1;
    }
    aclRet = aclrtSetDevice(0);
    if (aclRet != ACL_SUCCESS) {
        printf("aclrtSetDevice failed: %d\n", aclRet);
        aclFinalize();
        return -1;
    }

    // 1. 创建库句柄
    aclsparseLtHandle_t handle = nullptr;
    aclsparseStatus_t st = aclsparseLtInit(&handle);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        printf("aclsparseLtInit failed: %d\n", st);
        return -1;
    }

    // 2. 初始化矩阵描述符（承接 aclsparseLtMatmulDescriptorInit 的前置工作）
    //    A 为结构化稀疏（structured，FP16），B/C/D 为稠密（dense，FP16），列主序
    aclsparseLtMatDescriptor_t matA = nullptr, matB = nullptr, matC = nullptr, matD = nullptr;
    const int64_t rows = 32, cols = 32, ld = 32;
    const uint32_t alignment = 16;

    st = aclsparseLtStructuredDescriptorInit(
        &handle, &matA, rows, cols, ld, alignment,
        ACL_FLOAT16, ACL_SPARSE_ORDER_COL, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { printf("matA failed: %d\n", st); return -1; }

    st = aclsparseLtDenseDescriptorInit(&handle, &matB, cols, rows, ld, alignment, ACL_FLOAT16, ACL_SPARSE_ORDER_COL);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { printf("matB failed: %d\n", st); return -1; }

    st = aclsparseLtDenseDescriptorInit(&handle, &matC, rows, rows, ld, alignment, ACL_FLOAT16, ACL_SPARSE_ORDER_COL);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { printf("matC failed: %d\n", st); return -1; }

    st = aclsparseLtDenseDescriptorInit(&handle, &matD, rows, rows, ld, alignment, ACL_FLOAT16, ACL_SPARSE_ORDER_COL);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { printf("matD failed: %d\n", st); return -1; }

    // 3. 组装 matmul 描述符
    aclsparseLtMatmulDescriptor_t matmulDescr = nullptr;
    st = aclsparseLtMatmulDescriptorInit(
        &handle, &matmulDescr,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &matA, &matB, &matC, &matD, ACL_SPARSE_COMPUTE_16F);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { printf("matmulDescr failed: %d\n", st); return -1; }

    // 4. 初始化 algSelection：把 matmul 描述符与算法模式绑定在一起
    aclsparseLtMatmulAlgSelection_t algSelection = nullptr;
    st = aclsparseLtMatmulAlgSelectionInit(
        &handle, &algSelection, &matmulDescr, ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { printf("algSelection failed: %d\n", st); return -1; }

    // 5. 初始化 plan：把 matmul 描述符与 algSelection 绑定在一起
    aclsparseLtMatmulPlan_t plan = nullptr;
    st = aclsparseLtMatmulPlanInit(&handle, &plan, &matmulDescr, &algSelection);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { printf("plan failed: %d\n", st); return -1; }

    // 6. 销毁（顺序：先 plan，再 algSelection，再 matmul 描述符，再 mat 描述符，最后库句柄）
    aclsparseLtMatmulPlanDestroy(&plan);
    aclsparseLtMatmulAlgSelectionDestroy(&algSelection);
    aclsparseLtMatmulDescriptorDestroy(&matmulDescr);
    aclsparseLtMatDescriptorDestroy(&matD);
    aclsparseLtMatDescriptorDestroy(&matC);
    aclsparseLtMatDescriptorDestroy(&matB);
    aclsparseLtMatDescriptorDestroy(&matA);
    aclsparseLtDestroy(&handle);

    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```

接口调用成功后返回 0，无输出。