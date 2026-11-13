# SpVV（稀疏向量-稠密向量点积）算子

## 算子概述

`aclsparseSpvv` 计算稀疏向量与稠密向量的点积，结果为 Device 端单个 FP32 标量：

```text
result = sum(x_values[i] * y[x_indices[i]]),  i ∈ [0, nnz)
```

x 为稀疏向量（`aclsparseSpVecDescr_t`，索引须升序唯一、0 基、int32），y 为稠密向量（`aclsparseDnVecDescr_t`），result 为指向 Device 端 1 个 float 元素的指针。累加与输出恒为 FP32（computeType 仅支持 `ACL_FLOAT`）；x/y 元素类型可为 `ACL_FLOAT` 或 `ACL_FLOAT16`（两者必须一致，核函数按 valueType 分发）。

## 产品支持情况

| 产品 | 支持情况 |
| --- | --- |
| Atlas A2 训练/推理系列 | 支持，arch22 / DAV_2201 |
| Atlas A3 训练/推理系列 | 支持，arch22 / DAV_2201 |
| Ascend 950PR / Ascend 950DT | 不支持 |

## 接口说明

### aclsparseSpvv

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpvv(aclsparseHandle_t handle, aclsparseOperation_t op,
                                aclsparseConstSpVecDescr_t x, aclsparseConstDnVecDescr_t y,
                                void* result, aclDataType computeType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
| --- | --- | --- | --- |
| handle | 输入 | aclsparseHandle_t | 稀疏计算句柄，Host 内存 |
| op | 输入 | aclsparseOperation_t | 仅支持 ACL_SPARSE_OP_NON_TRANSPOSE |
| x | 输入 | aclsparseConstSpVecDescr_t | 稀疏向量描述符；索引须升序唯一，Device 内存 |
| y | 输入 | aclsparseConstDnVecDescr_t | 稠密向量描述符，Device 内存 |
| result | 输出 | void* | 指向 Device 端 1 个 float 元素的指针 |
| computeType | 输入 | aclDataType | 仅支持 ACL_FLOAT（累加/输出恒为 FP32） |

#### 约束说明

- op 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`（其余返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`）
- computeType 仅支持 `ACL_FLOAT`；x/y 元素类型可为 `ACL_FLOAT` 或 `ACL_FLOAT16`，两者必须一致（不一致返回 `ACL_SPARSE_STATUS_INVALID_VALUE`）
- x 索引类型仅支持 `ACL_SPARSE_INDEX_32I`，基址仅支持 `ACL_SPARSE_INDEX_BASE_ZERO`，索引须升序唯一（由调用方保证）
- 非空向量须绑定 values/indices 指针；`x`/`y`/`result` 不可为 nullptr
- result 所指元素在 stream 上异步写出，读取前须同步

#### 调用顺序

```cpp
aclsparseCreate(&handle);
aclsparseSetStream(handle, stream);
// CreateSpVec(x, size, nnz, indices, values, INDEX_32I, BASE_ZERO, FP32/FP16)
// CreateDnVec(y, size, values, 与 x 相同的 valueType)
aclsparseSpvv(handle, ACL_SPARSE_OP_NON_TRANSPOSE, x, y, dResult, ACL_FLOAT);
aclrtSynchronizeStream(stream);   // 读取 dResult 前
aclsparseDestroy(handle);
```

## 目录结构

```text
sparse/spvv/
└── arch22/               # Atlas A2/A3（DAV_2201）
    ├── spvv_host.cpp     # Host 侧 API（参数校验 + kernel 下发）
    ├── spvv_kernel.cpp   # Kernel 侧向量化实现
    ├── spvv_kernel.h     # kernel_do 启动器声明
    └── spvv_tiling_data.h
```

测试代码位于 `test/spvv/`（README 见该目录）。
