# SDDMM算子

## 算子概述

SDDMM（Sampled Dense-Dense Matrix Multiplication，采样稠密-稠密矩阵乘法）算子实现两个稠密矩阵相乘后按稀疏矩阵模式采样的运算。核心运算是只在稀疏矩阵 C 的非零位置上计算 X 与 Y 的点积，结果原地写回 C 的 values 数组。对标 cuSPARSE Generic API 中的 `cusparseSDDMM`。

数学表达式：

```
C_out = (alpha * op(X) * op(Y) + beta * C) ∘ spy(C)
```

其中：

- `op(X)` 为 m×k、`op(Y)` 为 k×n 的稠密矩阵，结果为 m×n；原始描述符形状由
  `opX`、`opY` 决定：`opX=NON_TRANSPOSE` 时 X 为 m×k，`opX=TRANSPOSE` 时 X
  为 k×m；`opY=NON_TRANSPOSE` 时 Y 为 k×n，`opY=TRANSPOSE` 时 Y 为 n×k
- C 为 m×n 稀疏矩阵（CSR 格式），spy(C) 为 C 的稀疏模式（非零位置为 1）
- ∘ 表示 Hadamard 积（逐元素乘），即只在 C 的非零位置保存计算结果
- alpha、beta 为标量缩放因子

对 CSR 每个非零元素 p（行 i，列 j=colInd[p]），算子计算 `op(X)[i,:]` 与
`op(Y)[:,j]` 的 K 维度点积，结果为 `alpha * dot + beta * C_values[p]`，并原地写回
`csrValues[p]`。

调用流程为三步法：

1. **BufferSize**：查询所需 workspace 大小
2. **Preprocess**（可选）：对稀疏矩阵进行预处理（行重排 + 贪心分箱负载均衡），加速后续计算
3. **Execute**：执行采样矩阵乘法，结果原地更新 C 的 values 数组

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSDDMMBufferSize | 查询 SDDMM 所需 workspace 大小（字节） |
| aclsparseSDDMMPreprocess | 对稀疏矩阵进行预处理（行重排 + 贪心分箱），加速后续计算 |
| aclsparseSDDMM | 执行采样矩阵乘法 C_out = (alpha * op(X) * op(Y) + beta * C) ∘ spy(C) |

## 算子执行接口

### aclsparseSDDMMBufferSize

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSDDMMBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, size_t *size)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opX | 输入 | aclsparseOperation_t | 稠密矩阵 X 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置，Host 内存 |
| opY | 输入 | aclsparseOperation_t | 稠密矩阵 Y 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，类型须与 computeType 匹配。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matX | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 X 的描述符：`opX=NON_TRANSPOSE` 时为 m×k，`opX=TRANSPOSE` 时为 k×m，Host 内存 |
| matY | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 Y 的描述符：`opY=NON_TRANSPOSE` 时为 k×n，`opY=TRANSPOSE` 时为 n×k，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针，类型须与 computeType 匹配。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matC | 输入/输出 | aclsparseSpMatDescr_t | 稀疏矩阵 C 的描述符（CSR 格式，m×n），values 数组原地更新，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，支持 `ACL_FLOAT` 或 `ACL_FLOAT16`，Host 内存 |
| alg | 输入 | aclsparseSDDMMAlg_t | 算法类型，仅支持 `ACL_SPARSE_SDDMM_ALG_DEFAULT`，Host 内存 |
| size | 输出 | size_t* | 输出所需 workspace 大小（字节），Host 内存 |

#### 约束说明

- handle 不可为 nullptr
- alpha、beta、size 不可为 nullptr
- matX、matY、matC 不可为 nullptr
- matC 仅支持 CSR 格式（`ACL_SPARSE_FORMAT_CSR`）
- matC 的行偏移和列索引类型必须均为 `ACL_SPARSE_INDEX_32I`，且两者类型相同
- matC 的索引基址仅支持 `ACL_SPARSE_INDEX_BASE_ZERO`
- opX 支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置
- opY 支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置
- 数据类型组合仅支持以下两种（X/Y/C 三者数据类型须一致且与 computeType 一致）：
  - matX=ACL_FLOAT, matY=ACL_FLOAT, matC=ACL_FLOAT, computeType=ACL_FLOAT
  - matX=ACL_FLOAT16, matY=ACL_FLOAT16, matC=ACL_FLOAT16, computeType=ACL_FLOAT16
- 维度匹配按操作后的有效形状检查：`op(X)` 为 m×k、`op(Y)` 为 k×n，且 C 为 m×n。
  因此，`opX=NON_TRANSPOSE` 时 X 为 m×k，`opX=TRANSPOSE` 时 X 为 k×m；
  `opY=NON_TRANSPOSE` 时 Y 为 k×n，`opY=TRANSPOSE` 时 Y 为 n×k。
- 零维输入：X / Y / C 任一维度（rows/cols）为 0 时返回 `ACL_SPARSE_STATUS_INVALID_VALUE`（arch35 与 arch22 行为一致）
- 各维度值不得超过 INT32_MAX

> **与 cuSPARSE opB 的语义对照（迁移注意）**
>
> 本算子公式为 `C = α(op(X)·op(Y)) ∘ spy(C) + βC`。`opX`、`opY` 分别决定 X、Y
> 描述符的原始形态和参与乘积的有效形态：`opX=NON_TRANSPOSE` 时 X 描述符为 m×k，
> `opX=TRANSPOSE` 时为 k×m；`opY=NON_TRANSPOSE` 时 Y 描述符为 k×n，
> `opY=TRANSPOSE` 时为 n×k。cuSPARSE `cusparseSDDMM` 公式为
> `C = α(op(A)·op(B)) ∘ spy(C) + βC`，op(B) 形态由 opB 决定，语义与本接口 opY
> 一致，对应关系如下：
>
> | 本接口 opY | 实际计算的乘积 | 等价的 cuSPARSE opB |
> |-----------|---------------|---------------------|
> | `ACL_SPARSE_OP_NON_TRANSPOSE` | X·Y（Y 描述符 k×n） | `CUSPARSE_OPERATION_NON_TRANSPOSE` |
> | `ACL_SPARSE_OP_TRANSPOSE` | X·Y^T（Y 描述符 n×k） | `CUSPARSE_OPERATION_TRANSPOSE` |
>
> 从 cuSPARSE 迁移时，第二个稠密矩阵的操作标志语义一致，可直接对应传入：cuSPARSE 使用 `opB=NON_TRANSPOSE` 的场景，本接口传 `opY=NON_TRANSPOSE`；cuSPARSE 使用 `opB=TRANSPOSE` 的场景，本接口传 `opY=TRANSPOSE`。

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 稀疏矩阵 C 支持 CSR 格式 |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

---

### aclsparseSDDMMPreprocess

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSDDMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opX | 输入 | aclsparseOperation_t | 稠密矩阵 X 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置，Host 内存 |
| opY | 输入 | aclsparseOperation_t | 稠密矩阵 Y 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matX | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 X 的描述符：`opX=NON_TRANSPOSE` 时为 m×k，`opX=TRANSPOSE` 时为 k×m，Host 内存 |
| matY | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 Y 的描述符：`opY=NON_TRANSPOSE` 时为 k×n，`opY=TRANSPOSE` 时为 n×k，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matC | 输入/输出 | aclsparseSpMatDescr_t | 稀疏矩阵 C 的描述符（CSR 格式，m×n），Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，支持 `ACL_FLOAT` 或 `ACL_FLOAT16`，Host 内存 |
| alg | 输入 | aclsparseSDDMMAlg_t | 算法类型，仅支持 `ACL_SPARSE_SDDMM_ALG_DEFAULT`，Host 内存 |
| buffer | 输入 | void* | workspace 缓冲区（由 BufferSize 返回的大小分配），Device 内存 |

#### 约束说明

- 同 aclsparseSDDMMBufferSize 的约束
- buffer 不可为 nullptr，需按 BufferSize 返回的大小分配
- 预处理会根据 CSR 行偏移计算各行非零数，按贪心分箱（greedy bin packing）策略进行行重排以均衡各核负载，并将重排表、分箱边界表与 TilingData 写入 device workspace

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 稀疏矩阵 C 支持 CSR 格式 |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

---

### aclsparseSDDMM

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSDDMM(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opX | 输入 | aclsparseOperation_t | 稠密矩阵 X 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置，Host 内存 |
| opY | 输入 | aclsparseOperation_t | 稠密矩阵 Y 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matX | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 X 的描述符：`opX=NON_TRANSPOSE` 时为 m×k，`opX=TRANSPOSE` 时为 k×m，Host 内存 |
| matY | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 Y 的描述符：`opY=NON_TRANSPOSE` 时为 k×n，`opY=TRANSPOSE` 时为 n×k，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matC | 输入/输出 | aclsparseSpMatDescr_t | 稀疏矩阵 C 的描述符（CSR 格式，m×n），values 数组原地更新为采样矩阵乘法结果，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，支持 `ACL_FLOAT` 或 `ACL_FLOAT16`，Host 内存 |
| alg | 输入 | aclsparseSDDMMAlg_t | 算法类型，仅支持 `ACL_SPARSE_SDDMM_ALG_DEFAULT`，Host 内存 |
| buffer | 输入 | void* | workspace 缓冲区（由 BufferSize 返回的大小分配），Device 内存 |

#### 约束说明

- 同 aclsparseSDDMMBufferSize 的约束
- handle、buffer 不可为 nullptr
- 调用前须先调用 `aclsparseSDDMMPreprocess` 进行预处理；若传入的 buffer 与 Preprocess 阶段标记的 activeBuffer 一致，则复用模式元数据并按值传入本次可变字段，否则重建模式元数据
- arch35 Execute 对已准备的 workspace 按值传入本次几何和 Host 标量，不读回或刷新 Device tiling；DEVICE 标量由 Kernel 在 handle stream 上读取。首次准备或失配重建仍包含同步元数据传输。
- arch22 为维持“模式变化即时返回 INVALID_VALUE 且 C 不变”的契约，等待独立校验事件；实数 CSR 数值 Kernel 可与该等待重叠，且检查全部批次的门控标志。其余路径仍等待校验前缀和 DEVICE 标量回传。
- 数值计算异步启动，调用者需自行同步后再读取 C。校验事件等待不等价于整个接口完全非阻塞。

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 稀疏矩阵 C 支持 CSR 格式 |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"

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

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

// 辅助：分配 Device 内存并拷贝 Host 数据
static void* AllocAndCopyDevice(const void *hostPtr, size_t sizeBytes)
{
    void *dPtr = nullptr;
    aclError ret = aclrtMalloc(&dPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return nullptr;
    }
    if (hostPtr != nullptr && sizeBytes > 0) {
        aclrtMemcpy(dPtr, sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return dPtr;
}

int aclsparseSDDMMTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(rawHandle, aclsparseDestroy);

    sparseRet = aclsparseSetStream(static_cast<aclsparseHandle_t>(handlePtr.get()), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 2. 设置 PointerMode
    sparseRet = aclsparseSetPointerMode(static_cast<aclsparseHandle_t>(handlePtr.get()), ACL_SPARSE_POINTER_MODE_HOST);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetPointerMode failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 3. 准备 Host 端数据
    //    X (4x2):           Y (3x2):
    //    [1.0  2.0]         [1.0  0.0]
    //    [3.0  4.0]         [0.0  1.0]
    //    [5.0  6.0]         [1.0  1.0]
    //    [7.0  8.0]
    //
    //    C (4x3, nnz=5), 非零位置: (0,0)(0,2)(1,1)(2,0)(3,2), 初始值全 0
    //    C = 1.0 * X * Y^T ∘ spy(C) (beta=0):
    //    C[0,0]=X[0]·Y[0]=1.0  C[0,2]=X[0]·Y[2]=3.0
    //    C[1,1]=X[1]·Y[1]=4.0  C[2,0]=X[2]·Y[0]=5.0
    //    C[3,2]=X[3]·Y[2]=15.0
    int64_t m = 4, n = 3, k = 2;
    int64_t nnzC = 5;
    float hAlpha = 1.0f;
    float hBeta = 0.0f;

    std::vector<int> hRowPtrC = {0, 2, 3, 4, 5};
    std::vector<int> hColIndC = {0, 2, 1, 0, 2};
    std::vector<float> hValC  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // 本示例使用 opX=NON_TRANSPOSE、opY=TRANSPOSE，因此 X 为 m×k (4x2)，Y 为 n×k (3x2)。
    // 若切换操作类型，描述符原始形状也要随之切换：opX=TRANSPOSE 时 X 为 k×m，
    // opY=NON_TRANSPOSE 时 Y 为 k×n。
    int64_t ldx = k, ldy = k;
    aclsparseOrder_t orderX = ACL_SPARSE_ORDER_ROW;
    aclsparseOrder_t orderY = ACL_SPARSE_ORDER_ROW;
    std::vector<float> hX(static_cast<size_t>(m) * k, 0.0f);
    hX[0 * k + 0] = 1.0f; hX[0 * k + 1] = 2.0f;
    hX[1 * k + 0] = 3.0f; hX[1 * k + 1] = 4.0f;
    hX[2 * k + 0] = 5.0f; hX[2 * k + 1] = 6.0f;
    hX[3 * k + 0] = 7.0f; hX[3 * k + 1] = 8.0f;

    std::vector<float> hY(static_cast<size_t>(n) * k, 0.0f);
    hY[0 * k + 0] = 1.0f; hY[0 * k + 1] = 0.0f;
    hY[1 * k + 0] = 0.0f; hY[1 * k + 1] = 1.0f;
    hY[2 * k + 0] = 1.0f; hY[2 * k + 1] = 1.0f;

    // 4. 拷贝数据到 Device
    void *dRowPtrC = AllocAndCopyDevice(hRowPtrC.data(), (m + 1) * sizeof(int));
    void *dColIndC = AllocAndCopyDevice(hColIndC.data(), nnzC * sizeof(int));
    void *dValC    = AllocAndCopyDevice(hValC.data(),    nnzC * sizeof(float));
    void *dX       = AllocAndCopyDevice(hX.data(),       static_cast<size_t>(m) * k * sizeof(float));
    void *dY       = AllocAndCopyDevice(hY.data(),       static_cast<size_t>(n) * k * sizeof(float));

    // 5. 创建描述符
    aclsparseSpMatDescr_t matC = nullptr;
    sparseRet = aclsparseCreateCsr(&matC, m, n, nnzC, dRowPtrC, dColIndC, dValC,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    aclsparseDnMatDescr_t matX = nullptr, matY = nullptr;
    sparseRet = aclsparseCreateDnMat(&matX, m, k, ldx, dX, ACL_FLOAT, orderX);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateDnMat X failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    sparseRet = aclsparseCreateDnMat(&matY, n, k, ldy, dY, ACL_FLOAT, orderY);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateDnMat Y failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 6. Step 1 — BufferSize
    size_t bufferSize = 0;
    sparseRet = aclsparseSDDMMBufferSize(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
        &hAlpha, matX, matY, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SDDMMBufferSize failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes\n", bufferSize);

    void *dBuffer = nullptr;
    auto aclRet = aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for buffer failed. ERROR: %d\n", aclRet); return aclRet);

    // 7. Step 2 — Preprocess
    sparseRet = aclsparseSDDMMPreprocess(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
        &hAlpha, matX, matY, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SDDMM_ALG_DEFAULT, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SDDMMPreprocess failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 8. Step 3 — Execute（结果原地写回 C 的 values 数组）
    sparseRet = aclsparseSDDMM(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
        &hAlpha, matX, matY, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SDDMM_ALG_DEFAULT, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SDDMM failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 9. 同步等待计算完成
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 10. 将结果拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hValC.data(), static_cast<size_t>(nnzC) * sizeof(float),
                         dValC, static_cast<size_t>(nnzC) * sizeof(float),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);

    for (int64_t p = 0; p < nnzC; p++) {
        LOG_PRINT("C_values[%lld] = %.1f\n", static_cast<long long>(p), hValC[p]);
    }

    // 11. 清理资源
    aclsparseDestroySpMat(matC);
    aclsparseDestroyDnMat(matX);
    aclsparseDestroyDnMat(matY);
    if (dRowPtrC) aclrtFree(dRowPtrC);
    if (dColIndC) aclrtFree(dColIndC);
    if (dValC)    aclrtFree(dValC);
    if (dX)       aclrtFree(dX);
    if (dY)       aclrtFree(dY);
    if (dBuffer)  aclrtFree(dBuffer);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseSDDMMTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSDDMMTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
bufferSize = 512 bytes
C_values[0] = 1.0
C_values[1] = 3.0
C_values[2] = 4.0
C_values[3] = 5.0
C_values[4] = 15.0
```

## 精度标准

SDDMM 算子精度验证采用框架 MIXED_TOLERANCE 模式（`test/frame/verify.h::applyMixedTolerance` + `Verifier::verifyVector`），按数据类型自动选择 atol/rtol 与 maxAbsErrLimit：

- 逐元素通过条件：`|actual - golden| ≤ max(atol + rtol * |golden|, maxAbsErrLimit)`，其中 `maxAbsErrLimit = max(fixedValue, 32 * ULP(golden))`，`ULP(x) = 2^(floor(log2|x|) - mantissaBits)`。
- 整体通过条件：`matchedRatio ≥ 0.99`。

| 数据类型 | rtol（相对容差） | atol（绝对容差） | maxAbsErrLimit fixedValue |
|----------|------------------|------------------|---------------------------|
| FLOAT16  | 1.953125e-3 (2^-9)  | 1.953125e-3 (2^-9)  | 1e-1 |
| FLOAT32  | 9.765625e-4 (2^-10) | 1.52587890625e-5 (2^-23) | 1e-2 |

基线测试数据（20 个功能用例）：FP32 与 FP16 全部 matchedRatio=1.0；FP32 maxAbsErr 4.77e-07 ~ 1.53e-05，FP16 maxAbsErr 2.24e-04 ~ 1.56e-02。

> 说明：FP16 路径在 FP32 中完成点积累加与 alpha/beta 缩放，写回前对结果做饱和截断到 FP16 范围 [−65504, 65504] 后再转换为 FP16，避免溢出。MIXED_TOLERANCE 模式下的 maxAbsErrLimit 随 |golden| 自适应放宽，给 K 维度累加噪声留出充足余量。

## 支持芯片

- Ascend 950PR / Ascend 950DT（架构 arch35 / DAV_3510）
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Ascend 910B（架构 arch22 / DAV_2201）：对应 Atlas A2 训练系列产品 / Atlas A2 推理系列产品