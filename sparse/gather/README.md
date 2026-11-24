# Gather 算子

## 算子概述

Gather 算子实现从稠密向量中按稀疏向量的索引数组收集元素，属于 Generic API 体系下的稀疏向量-稠密向量基本运算。

输出结果 X 满足：

```
X.values[i] = Y[X.indices[i] - idxBase]   for i = 0 .. nnz-1
```

其中 `X` 为稀疏向量（Sparse Vector），包含 `indices`（非零元素在稠密向量中的位置）和 `values`（非零元素的值）；`Y` 为稠密向量（Dense Vector）。调用后 `X.values` 被覆写为从 `Y` 中收集到的值。

当 `X.nnz > 0` 时，`Y` 的元素步长必须为 1。arch22 和 arch35 均不支持非单位步长，
通过 `aclsparseDnVecSetStride` 设置其他步长后调用 Gather 会返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`，
不会下发 Kernel。空输出的既有 no-op 行为不受此步长限制。

该算子不需要 workspace，也不需要 preprocess 阶段，单步调用即可完成计算。

## 算子执行接口

### aclsparseGather

#### 产品支持情况

- Ascend 950PR&950DT系列产品（arch35）：支持
- Atlas A3系列产品（arch22）：支持
- Atlas A2系列产品（arch22）：支持

> 说明：arch22（Atlas A2系列产品、Atlas A3系列产品）与 arch35（Ascend 950PR&950DT系列产品）支持的 dtype / 索引类型范围不同，见下方“支持的数据类型 / 索引类型”。arch22 实现位于 `sparse/gather/arch22/`，arch35 实现位于 `sparse/gather/arch35/`，二者按 SOC 在编译期分流，不会同时参与链接。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseGather(
    aclsparseHandle_t handle,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseSpVecDescr_t vecX)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，须先调用 `aclsparseSetStream` | Host |
| vecY | 输入 | aclsparseConstDnVecDescr_t | 稠密向量 Y 的描述符（数据源），仅读取 values | Host |
| vecX | 输出 | aclsparseSpVecDescr_t | 稀疏向量 X 的描述符：读取 indices 和 idxBase，写入 values | Host |

#### 约束说明

- handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`
- vecY、vecX 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- vecX.valueType 与 vecY.valueType 必须一致
- 合法索引是核心契约：`vecX.indices` 中每个索引值 `X.indices[i] - idxBase` 必须落在 `[0, vecY.size)` 范围内
  - arch22（Atlas A2系列产品、Atlas A3系列产品）：Host 侧做逐索引范围校验（D2H 拷贝索引后逐个判断），越界索引显式返回 `ACL_SPARSE_STATUS_INVALID_VALUE`，不会静默越界读取
  - arch35（Ascend 950PR&950DT系列产品）：Device 索引越界遵循异步错误协议，因此合法索引是调用方前置条件
- `vecX.size` 与 `vecY.size` 的基数约束：
  - arch35（Ascend 950PR&950DT系列产品）：`vecX.size` 必须等于 `vecY.size`，且 `0 <= nnz <= size`
  - arch22（Atlas A2系列产品、Atlas A3系列产品）：不做基数约束，合法性完全取决于逐索引落点（`nnz` 可大于 `vecY.size`）
- `size` 和 `nnz` 的接口上限为 `INT64_MAX`；实际还受 Device 可分配内存、元素字节数以及地址范围不溢出的约束
- `nnz=0` 成功返回且不启动 Kernel；`size=0` 仅允许 `nnz=0`。**校验顺序：handle → `nnz==0` 早退 → 描述符/数据指针**，因此空 handle 即便 `nnz=0` 也返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`，而零 `nnz` 时不要求 `indices`/`values` 有效
- 非空输入、索引和输出指针必须位于当前 NPU Device，且在异步执行完成前保持有效
- vecY.values、vecX.indices 只读；vecX.values 不得与任一输入缓冲区重叠
- 不需要 workspace，不需要预处理阶段
- 支持索引乱序（indices 不需要排序）
- 支持 `vecX.indices` 中存在重复元素（arch22 下 `nnz` 可大于 `vecY.size`，例如 `vecY.size=1, nnz=2, indices=[0,0]`（base0）会输出同一元素两次，与 `torch.index_select` 语义一致）

> arch22（Atlas A2系列产品、Atlas A3系列产品）Host 侧额外约束（不满足即显式返回错误，**不退回 CPU**）：
> - `vecX.valueType` 与 `vecY.valueType` 须一致，且须在 {`ACL_FLOAT`, `ACL_FLOAT16`, `ACL_BF16`, `ACL_COMPLEX64`} 内（FP64 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`）
> - `vecX.idxType` 须为 `ACL_SPARSE_INDEX_32I`（I64 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`）
> - `vecX.idxBase` 须为 0 或 1（其它值返回 `ACL_SPARSE_STATUS_INVALID_VALUE`）
> - `nnz == 0` 时直接返回 `ACL_SPARSE_STATUS_SUCCESS`，不启动 kernel（输出 `X.values` 保持不变）；该早退位于 handle 校验之后，零 `nnz` 时不要求 `indices`/`values` 指针有效

#### 支持的数据类型

| 数据类型 | 枚举值 | 支持 | 备注 |
|---------|--------|------|------|
| FP32 | `ACL_FLOAT` | ✅ | arch22 / arch35 均支持 |
| FP16 | `ACL_FLOAT16` | ✅ | arch22 / arch35 均支持 |
| BF16 | `ACL_BF16` | ✅ | arch22 / arch35 均支持 |
| FP64 | `ACL_DOUBLE` | ✅（仅 arch35） | arch22 不支持 |
| COMPLEX64 | `ACL_COMPLEX64` | ✅ | arch22 / arch35 均支持 |

Ascend 950PR&950DT系列产品任务声明的必选类型为 FP16、BF16、FP32 和 COMPLEX64。FP64 为既有低层 C++ 接口兼容能力；PyTorch/ATen 适配仅开放四种必选类型。
arch22（Atlas A2系列产品、Atlas A3系列产品）支持范围：**FP32 / FP16 / BF16 / COMPLEX64**，与任务书一致；不支持 FP64。

#### 支持的索引类型

| 索引类型 | 枚举值 | 支持 | 备注 |
|---------|--------|------|------|
| 32 位有符号整数 | `ACL_SPARSE_INDEX_32I` | ✅ | arch22 / arch35 均支持 |
| 64 位有符号整数 | `ACL_SPARSE_INDEX_64I` | ✅（仅 arch35） | arch22 仅支持 I32 |

任务公开的 PyTorch/ATen 契约仅接受 I32。I64 保留为既有低层 C++ 接口兼容能力。

#### 索引基址

| 基址 | 枚举值 | 说明 |
|------|--------|------|
| 0-based | `ACL_SPARSE_INDEX_BASE_ZERO` | C 兼容（默认） |
| 1-based | `ACL_SPARSE_INDEX_BASE_ONE` | Fortran 兼容 |

索引基址在创建稀疏向量描述符（`aclsparseCreateSpVec`）时指定，Gather 通过 `X.indices[i] - idxBase` 计算实际访问位置。

#### 特性说明

| 特性 | 支持 | 说明 |
|------|------|------|
| 额外 buffer | 不需要 | 无需 workspace 分配 |
| Preprocess | 不需要 | 无预处理阶段 |
| 确定性 | ✅ |每次调用结果 bit-wise 一致 |
| 索引乱序 | ✅ | indices 不要求排序 |
| 异步执行 | ✅ | 调用后需 `aclrtSynchronizeStream` 等待完成 |

## PyTorch/ATen 入口

适配支持 PyTorch 2.7+、torch_npu 26.0.0+。加载扩展后，公开入口保持标准 Dispatcher 路径：

```python
import torch
import torch_npu
import cann_ops_sparse

source = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32, device="npu")
index = torch.tensor([0, 2, 4], dtype=torch.int32, device="npu")
output = torch.index_select(source, 0, index)
```

调用链为 `torch.index_select` → `aten::index_select` → PrivateUse1 实现 → `aclsparseGather` → Ascend C Kernel，不通过 CPU 回填。该公开入口限一维连续 Strided NPU Tensor、`dim=0`（一维语义下也接受 `-1`）、I32 index 和四种任务必选 values dtype。PyTorch 公开入口采用 base 0；底层 C API 继续支持 base 0/1。适配为 forward-only，不支持 autograd，输出为独立连续 Tensor。

PyTorch 入口允许索引长度超过输入长度，索引可以乱序或重复，但每个索引仍必须位于输入范围。
长索引通过索引和输出的连续视图分段调用底层接口，全部沿用调用方 stream，不复制数据或增加
线性临时缓冲区；公共 SpVec 描述符和 C API 的 `nnz <= size` 契约保持不变。

构建和执行命令见 [`test/gather/README.md`](../../test/gather/README.md)。

## 调用示例

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

using DevicePtr = std::unique_ptr<void, aclError (*)(void *)>;

static int AllocAndCopyDevice(DevicePtr &devicePtr, const void *hostPtr, size_t sizeBytes)
{
    void *rawPtr = nullptr;
    auto ret = aclrtMalloc(&rawPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    devicePtr.reset(rawPtr);

    if (hostPtr != nullptr && sizeBytes > 0) {
        ret = aclrtMemcpy(devicePtr.get(), sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy data to Device failed. ERROR: %d\n", ret); return ret);
    }
    return ACL_SUCCESS;
}

static int AllocDevice(DevicePtr &devicePtr, size_t sizeBytes)
{
    void *rawPtr = nullptr;
    auto ret = aclrtMalloc(&rawPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    devicePtr.reset(rawPtr);
    return ACL_SUCCESS;
}

int aclsparseGatherTest(AclContext &ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreate failed: %d\n", sparseRet);
              return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(rawHandle, aclsparseDestroy);

    sparseRet = aclsparseSetStream(static_cast<aclsparseHandle_t>(handlePtr.get()), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetStream failed: %d\n", sparseRet);
              return sparseRet);

    // 2. 准备 Host 端数据
    //    稠密向量 Y = {10.0, 20.0, 30.0, 40.0, 50.0}
    //    稀疏向量 X.indices = {0, 2, 4}（0-based），nnz = 3
    //    期望输出 X.values = {10.0, 30.0, 50.0}
    int64_t vecSize = 5;
    int64_t nnz = 3;

    std::vector<float> hY = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    std::vector<int32_t> hIndices = {0, 2, 4};

    // 3. 拷贝数据到 Device
    DevicePtr dY(nullptr, aclrtFree);
    DevicePtr dIndices(nullptr, aclrtFree);
    DevicePtr dXValues(nullptr, aclrtFree);

    auto aclRet = AllocAndCopyDevice(dY, hY.data(), vecSize * sizeof(float));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dIndices, hIndices.data(), nnz * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocDevice(dXValues, nnz * sizeof(float));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 4. 创建稠密向量描述符（数据源 Y）
    aclsparseConstDnVecDescr_t dnVecY = nullptr;
    sparseRet = aclsparseCreateConstDnVec(&dnVecY, vecSize, dY.get(), ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateConstDnVec failed: %d\n", sparseRet);
              return sparseRet);

    // 5. 创建稀疏向量描述符（输出 X）
    aclsparseSpVecDescr_t spVecX = nullptr;
    sparseRet = aclsparseCreateSpVec(&spVecX, vecSize, nnz, dIndices.get(), dXValues.get(),
                                     ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateSpVec failed: %d\n", sparseRet);
              return sparseRet);

    // 6. 调用 Gather
    sparseRet = aclsparseGather(handlePtr.get(), dnVecY, spVecX);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseGather failed: %d\n", sparseRet);
              return sparseRet);

    // 7. 同步等待计算完成
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 8. 将结果拷贝回 Host 并打印
    std::vector<float> hXValues(nnz, 0.0f);
    aclRet = aclrtMemcpy(hXValues.data(), nnz * sizeof(float), dXValues.get(), nnz * sizeof(float),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy X.values to Host failed. ERROR: %d\n", aclRet); return aclRet);

    LOG_PRINT("\nResult:\n");
    LOG_PRINT("  X.values: ");
    for (int64_t i = 0; i < nnz; i++) {
        LOG_PRINT("%.1f ", hXValues[i]);
    }
    LOG_PRINT("\n");

    // 9. 清理描述符
    aclsparseDestroySpVec(spVecX);
    aclsparseDestroyDnVec(dnVecY);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseGatherTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseGatherTest failed: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
Result:
  X.values: 10.0 30.0 50.0
```
