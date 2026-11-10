# 复数三对角线性方程组求解算子

## 算子概述

`aclsparseCgtsv2` 求解单个复数三对角系统 `A·X=B`，支持多个右端项（RHS）。
矩阵由下对角线 `dl`、主对角线 `d` 和上对角线 `du` 表示；解原地覆盖列主序 `B`，系数只读，padding 不变。
通用路径采用相邻部分选主元，共享分解并保留换行产生的第二上对角线。
本算子不改变既有实数 `aclsparseSgtsv2` 接口。
调用顺序为：查询 workspace 容量 → 分配并对齐 workspace → 执行求解 → 同步 stream 后读取结果。

数学表达式：

```text
dl[i] * X[i-1,j] + d[i] * X[i,j] + du[i] * X[i+1,j] = B[i,j]
0 <= i < m, 0 <= j < n；越界项为 0，dl[0] = du[m-1] = 0。
```

包含以下接口：

| 接口名 | 功能简述 |
| --- | --- |
| aclsparseCgtsv2_bufferSizeExt | 查询调用者应提供的 Device workspace 字节数 |
| aclsparseCgtsv2 | 对复数三对角系统选主元求解，将解原位写回 B |


### 接口契约来源

本实现依据[官方 A2/A3 任务书](https://www.hiascend.com/activities/task-center/details/b8289b3a952d4e13a9e79c3a242a8338?menu=tasks)。
任务书第 2.2 节明确要求 Legacy C API 的 Kernel 直调模式，第 2.3 节指定
`aclsparseCgtsv2_bufferSizeExt`、`aclsparseCgtsv2`。此处按任务要求确认接口体系，未新增 Generic 描述符接口。
按评审建议复用仓库既有 `aclsparseComplex { float x; float y; }`，`aclsparseComplex_t` 仅为它的类型别名，
其中 `x` 表示实部、`y` 表示虚部。两种名称是同一个 C/C++ 类型，可直接赋值或传递指针，
不再维护第二份 complex64 struct。与任务书原 `real/imag` 表示相比，两个 FLOAT32 分量的顺序、
8 字节大小及公开 C 符号保持一致；源码成员访问统一为仓库已有的 `x/y`。
设计说明见[设计文档 PR](https://atomgit.com/cann/cann-ops-competitions/pull/2394)。

## 算子执行接口

### aclsparseCgtsv2

#### 产品支持情况

| 产品 | 支持情况 |
| --- | --- |
| Atlas A2 训练/推理系列 | 支持，arch22 / DAV_2201 |
| Atlas A3 训练/推理系列 | 支持，arch22 / DAV_2201 |
| Ascend 950PR / Ascend 950DT | 本 PR 未提供 arch35 数值后端，不宣称支持 |

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCgtsv2(aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, aclsparseComplex_t *B, int ldb, void *pBuffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
| --- | --- | --- | --- |
| handle | 输入 | aclsparseHandle_t | 有效库句柄，携带执行 stream，Host 内存 |
| m | 输入 | int | 系统行数，`m>=3`，Host 内存 |
| n | 输入 | int | RHS 列数，`n>=0`，Host 内存 |
| dl | 输入 | const aclsparseComplex_t* | 下对角线，m 个 complex64，`dl[0]=0`，Device 内存 |
| d | 输入 | const aclsparseComplex_t* | 主对角线，m 个 complex64，Device 内存 |
| du | 输入 | const aclsparseComplex_t* | 上对角线，m 个 complex64，`du[m-1]=0`，Device 内存 |
| B | 输入/输出 | aclsparseComplex_t* | `[ldb,n]` 列主序，前 m 行原地写回解，Device 内存 |
| ldb | 输入 | int | 物理列跨度，`ldb>=m`，Host 内存 |
| pBuffer | 输入/workspace | void* | 至少查询容量且基地址按 128 字节对齐，Device 内存 |

#### 约束说明

- 复数类型为连续 `float x/y`，大小 8 字节，对齐为 `alignof(float)`。
- `dl/d/du` 必须只读；B 的 `[m,ldb)` 行保留原始字节。无独立 X、batch、广播或任意 stride 参数。
- `n=0` 仍校验 handle、`m/ldb`，返回成功且不启动 kernel；允许空数据/workspace 指针。
- 非空调用须提供有效设备分配。裸指针 API 无法获知分配的真实容量、dtype 或 stride，由调用方负责。
- 并发调用须使用独立可写 workspace。输入、B 和 workspace 不得在 stream 完成前释放。
- 成功返回只表示校验和异步派发成功；设备错误还需在调用方同步 stream 时检查。
- 空 handle 返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`；非法尺寸、指针、对齐或地址跨度返回
  `ACL_SPARSE_STATUS_INVALID_VALUE`。平台发现失败返回已有内部错误码，并记录错误日志。
- 奇异和非有限输入采用有界执行及单独诊断语义，不作为普通有限值精度通过的证据。

#### 调用示例

示例为 `m=3,n=1,ldb=3`，矩阵主对角线为 4、两条副对角线为 1，右端项为 `[6,12,14]`，
解析解为 `[1,2,3]`，虚部均为 0。该接口不使用 MatDescr，也没有需要 PointerMode 的标量指针参数。
示例使用 `AclContext` 和 `std::unique_ptr` 管理 ACL、句柄、Device 输入及 workspace，包含完整查询、执行和结果检查。
编译与运行方式见[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)，目标 SoC 按实际 A2/A3 设备选择。

```cpp
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include "acl/acl.h"
#include "cann_ops_sparse.h"

void CheckAcl(aclError status, const char *operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
    }
}

void CheckSparse(aclsparseStatus_t status, const char *operation)
{
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
    }
}

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}
    AclContext(const AclContext &) = delete;
    AclContext &operator=(const AclContext &) = delete;
    ~AclContext()
    {
        if (stream_ != nullptr) { aclrtDestroyStream(stream_); }
        if (deviceSet_) { aclrtResetDevice(deviceId_); }
        if (initialized_) { aclFinalize(); }
    }
    void Init()
    {
        CheckAcl(aclInit(nullptr), "aclInit");
        initialized_ = true;
        CheckAcl(aclrtSetDevice(deviceId_), "aclrtSetDevice");
        deviceSet_ = true;
        CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream");
    }
    aclrtStream Stream() const { return stream_; }
private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool initialized_ = false;
    bool deviceSet_ = false;
};

struct DeviceDeleter {
    aclrtStream stream;
    void operator()(void *address) const
    {
        // Error paths also wait before releasing data used by an async launch.
        aclrtSynchronizeStream(stream);
        aclrtFree(address);
    }
};
using DevicePtr = std::unique_ptr<void, DeviceDeleter>;

DevicePtr Allocate(size_t bytes, aclrtStream stream)
{
    void *address = nullptr;
    CheckAcl(aclrtMalloc(&address, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    return DevicePtr(address, DeviceDeleter{stream});
}

void RunExample(AclContext &ctx)
{
    using Complex = aclsparseComplex;
    const std::array<Complex, 3> lower{{{0, 0}, {1, 0}, {1, 0}}};
    const std::array<Complex, 3> diagonal{{{4, 0}, {4, 0}, {4, 0}}};
    const std::array<Complex, 3> upper{{{1, 0}, {1, 0}, {0, 0}}};
    const std::array<Complex, 3> rhs{{{6, 0}, {12, 0}, {14, 0}}};
    constexpr size_t dataBytes = sizeof(Complex) * 3;
    aclsparseHandle_t rawHandle = nullptr;
    CheckSparse(aclsparseCreate(&rawHandle), "aclsparseCreate");
    std::unique_ptr<aclsparseContext, decltype(&aclsparseDestroy)> handle(rawHandle, aclsparseDestroy);
    CheckSparse(aclsparseSetStream(handle.get(), ctx.Stream()), "aclsparseSetStream");
    auto dl = Allocate(dataBytes, ctx.Stream());
    auto d = Allocate(dataBytes, ctx.Stream());
    auto du = Allocate(dataBytes, ctx.Stream());
    auto b = Allocate(dataBytes, ctx.Stream());
    CheckAcl(aclrtMemcpy(dl.get(), dataBytes, lower.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy dl");
    CheckAcl(aclrtMemcpy(d.get(), dataBytes, diagonal.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy d");
    CheckAcl(aclrtMemcpy(du.get(), dataBytes, upper.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy du");
    CheckAcl(aclrtMemcpy(b.get(), dataBytes, rhs.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy B");
    size_t bytes = 0;
    CheckSparse(aclsparseCgtsv2_bufferSizeExt(handle.get(), 3, 1,
        static_cast<const Complex *>(dl.get()), static_cast<const Complex *>(d.get()),
        static_cast<const Complex *>(du.get()), static_cast<const Complex *>(b.get()), 3, &bytes), "bufferSizeExt");
    if (bytes == 0 || bytes > std::numeric_limits<size_t>::max() - 127U) {
        throw std::runtime_error("invalid workspace size");
    }
    auto backing = Allocate(bytes + 127U, ctx.Stream());
    const auto base = reinterpret_cast<uintptr_t>(backing.get());
    void *workspace = reinterpret_cast<void *>((base + 127U) & ~uintptr_t(127U));
    CheckSparse(aclsparseCgtsv2(handle.get(), 3, 1,
        static_cast<const Complex *>(dl.get()), static_cast<const Complex *>(d.get()),
        static_cast<const Complex *>(du.get()), static_cast<Complex *>(b.get()), 3, workspace), "aclsparseCgtsv2");
    CheckAcl(aclrtSynchronizeStream(ctx.Stream()), "aclrtSynchronizeStream");
    std::array<Complex, 3> result{};
    CheckAcl(aclrtMemcpy(result.data(), dataBytes, b.get(), dataBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy result");
    for (size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result[i].x) || !std::isfinite(result[i].y) ||
            std::abs(result[i].x - static_cast<float>(i + 1)) > 1e-5f || std::abs(result[i].y) > 1e-5f) {
            throw std::runtime_error("unexpected solution");
        }
        std::printf("result[%zu] = (%.6f, %.6f)\n", i, result[i].x, result[i].y == 0 ? 0.0f : result[i].y);
    }
}

int main()
{
    try {
        AclContext ctx(0);
        ctx.Init();
        RunExample(ctx);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
```

预期输出如下：

```text
result[0] = (1.000000, 0.000000)
result[1] = (2.000000, 0.000000)
result[2] = (3.000000, 0.000000)
```

### aclsparseCgtsv2_bufferSizeExt

#### 产品支持情况

与执行接口相同，支持 Atlas A2/A3。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseCgtsv2_bufferSizeExt(aclsparseHandle_t handle,
    int m, int n, const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, const aclsparseComplex_t *B, int ldb,
    size_t *pBufferSizeInBytes)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
| --- | --- | --- | --- |
| handle | 输入 | aclsparseHandle_t | 有效库句柄，Host 内存 |
| m | 输入 | int | 系统行数，`m>=3`，Host 内存 |
| n | 输入 | int | RHS 列数，`n>=0`，Host 内存 |
| dl | 输入 | const aclsparseComplex_t* | m 个下对角线值，Device 内存 |
| d | 输入 | const aclsparseComplex_t* | m 个主对角线值，Device 内存 |
| du | 输入 | const aclsparseComplex_t* | m 个上对角线值，Device 内存 |
| B | 输入 | const aclsparseComplex_t* | 列主序 RHS，Device 内存 |
| ldb | 输入 | int | 物理列跨度，`ldb>=m`，Host 内存 |
| pBufferSizeInBytes | 输出 | size_t* | 非空字节数输出指针，Host 内存 |

#### 约束说明

输入约束与执行接口相同；`pBufferSizeInBytes` 不得为空。
合法 `n=0` 返回 0；非空调用按任务书第 2.4 节保留 `dl/d/du/B` 非空及可表示地址跨度校验。
查询不读设备输入、不分配设备内存、不启动 kernel。
`aclsparseGetStream` 仅验证 handle 签名，允许空 stream 值（默认 stream），不创建或同步 stream。
调用方必须按当前库重新查询容量，不能硬编码历史容量。

#### 调用示例

上一接口的完整示例先调用本接口取得 `bytes`，再分配并对齐 workspace，最后执行求解。
两个接口共用该示例及其预期输出，不需要单独初始化第二套 ACL 资源；返回的 workspace 容量由当前实现确定，不能硬编码。

## 实现说明

### 公共入口与架构后端

`cgtsv2_api.cpp` 统一两个公开 C 入口和尺寸、handle、指针、地址跨度、workspace 对齐校验。
`cgtsv2_backend.h` 声明内部 `Cgtsv2WorkspaceBytes` 和 `Cgtsv2Dispatch` 钩子。
`arch22/cgtsv2_host.cpp` 只负责本架构 workspace、UB 准入和 kernel 派发。
未来 arch35 后端可复用公共入口并独立计算自己的布局；本 PR 不提供该数值后端。

### 数值与内存路径

通用有限输入路径以高低 FP32 尾数及实虚部分别独立的 int64 指数表示中间值，减少中间溢出。
原始 B 保留到最终转换；列工作数组供求解和原方程残差修正复用，workspace 为 O(m)。
小于 1024 行的 UB 分解由四组复数因子加 uint32 pivot 构成，每行 36 字节；
求解增加一组 RHS 后为 44 字节，数组按 32 字节边界对齐。
Host 与 Device 共用 `arch22/cgtsv2_tiling_data.h` 常量；准入额外保留 16 KiB 给 TPipe/编译器使用。
该预留量是保守调度预算，不是硬件容量。已删除不可达的 tiled direct-solve 分支。
未对齐的物理写边界采用保守单写者调度；标量 GM 回退由一个核写入。
快路只依赖实际尺寸、矩阵性质及输入范围，不依赖用例编号或种子。

### Jacobi 认证及残差修正

大系统要求 `1<=maxComp(d)<=8`、RHS 每分量有限且绝对值不超过 16，且逐行
`maxComp(d)>1.351352*offBound`。offBound 以 `sqrt(1+t*t)` 的四段线性弦线、向上舍入常数
和 FP32 安全裕量保守估计复数模长之和，认证收缩率小于 0.74。
先进行 40 次零初值迭代，用原始 FP32 A/B 计算补偿残差，再用 40 次迭代求修正量并加回初始解。
精确算术截断界为 `0.74^80*16*sqrt(2)/(1-0.74)<3.01e-9`；FP32 舍入另由补偿残差和边界回归检查。
依赖半径最多 `39+1+39=79` 行，保留 80 行 original-B halo。
对齐的大单 RHS 由准备核发布原 B 快照，各核独占 512 行 center tile；其他合格调用按整 RHS 列调度。
认证失败（含非有限 RHS）回退 expanded；小尺寸 direct 保留独立准入条件。

## 测试说明

仓内测试位于 `test/cgtsv2/`，复用 GTest/CSV、ACL RAII 和内存检查设施；
构建、覆盖与任务特定比较器的差异说明见[测试说明](../../test/cgtsv2/README.md)。
官方验收使用原任务包及其原始比较器，独立于仓内回归；本文不将仓外验收数量列为仓内用例。
