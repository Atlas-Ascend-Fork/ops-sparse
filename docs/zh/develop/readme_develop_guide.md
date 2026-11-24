# {算子名称}算子

<!--
模板使用说明：
1. 将所有 {占位符} 替换为实际内容，占位符命名规则见下表。
2. 按实际包含的接口增删子章节（### 节）。如仅有单精度接口则只保留一个 ### 节；如有更多接口（如 aclsparseD{op}、aclsparseC{op}）则继续添加。
3. 参数说明列必须写清楚参数含义和内存位置（Host 内存/Device 内存）。标量指针参数（如 alpha、beta）的内存位置由 aclsparseSetPointerMode 控制，需注明"Host/Device"。
4. 约束说明如无约束则写"无"，不允许留空。
5. 函数原型必须写在同一行内，不允许换行，且末尾不加 `;`。若参数过多导致一行过长，可在代码块内换行缩进，但函数签名仍需保持完整。
6. 调用示例必须使用 RAII 模式（AclContext 类 + std::unique_ptr）管理 ACL 资源和 Device 内存，与 compile_and_run_example.md 一致；须由开发者在本地跑通后再上库。
7. 如算子涉及 aclsparseMatDescr_t，须在调用示例中展示 MatDescr 的创建与销毁。
8. 如算子涉及标量指针参数（如 alpha、beta），须在调用示例中展示 aclsparseSetPointerMode 的设置。
9. 如算子为多步调用（如 bufferSize → Nnz → Compute），须在调用示例中展示完整的多步调用流程，并在算子概述中说明调用顺序。
10. 如算子涉及稀疏矩阵格式（CSR/COO/CSC 等），须在约束说明后添加"支持的稀疏格式"子章节，列出各格式的支持情况；如不涉及则删除该子章节。
11. 调用示例代码块后须添加"预期输出如下："及预期输出代码块，供用户验证结果。
12. 完成文档后删除本使用说明注释块。

占位符约定：

| 占位符 | 含义 | 示例 |
|--------|------|------|
| {算子名称} | 算子中文名 | 非零元素统计 |
| {op} | 算子英文缩写（小写） | nnz |
| {功能描述} | 算子功能的一句话描述 | 统计稠密矩阵中每行或每列的非零元素个数 |
| {运算描述} | 核心数学运算描述 | nnzPerRow[i] = sum(A[i][j] != 0) |
| {参数含义} | 各参数的具体含义 | 矩阵行数 |
-->

## 算子概述

{一段话描述算子的功能定位和核心运算。例如："{op} 算子实现了{功能描述}，核心运算为{运算描述}。"}

数学表达式：

```
{数学公式，使用纯文本。如需要可用 LaTeX 格式（$$...$$）}
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseS{op} | {单精度功能描述} |
| aclsparseD{op} | {双精度功能描述} |

## 算子执行接口

### aclsparseS{op}

#### 产品支持情况

- Ascend 950PR&950DT系列产品：{支持/不支持}
- Atlas A3系列产品：{支持/不支持}
- Atlas A2系列产品：{支持/不支持}

#### 函数原型

```cpp
aclsparseStatus_t aclsparseS{op}(aclsparseHandle_t handle, int m, int n, const float *A, ...)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| m | 输入 | int | {参数含义}，Host 内存 |
| A | 输入 | const float*（FP32） | {参数含义}，Device 内存 |

#### 约束说明

- m >= 0
- n >= 0

{如无约束则写"无"，不允许留空。}

#### 支持的稀疏格式

{如算子涉及稀疏矩阵格式（CSR/COO/CSC 等），按下表列出支持情况；如不涉及稀疏格式则删除本节。}

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | {✅/❌} | {说明} |
| COO | {✅/❌} | {说明} |
| CSC | {✅/❌} | {说明} |

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](./compile_and_run_example.md)。

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

int aclsparseS{op}Test(AclContext& ctx)
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

    // 2. 设置 PointerMode（如算子含标量指针参数如 alpha、beta，须设置；否则可省略）
    // sparseRet = aclsparseSetPointerMode(static_cast<aclsparseHandle_t>(handlePtr.get()), ACL_SPARSE_POINTER_MODE_HOST);
    // CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetPointerMode failed. ERROR: %d\n", sparseRet);
    //           return sparseRet);

    // 3. 创建 MatDescr（如算子需要 aclsparseMatDescr_t，须创建并配置；否则可省略）
    // aclsparseMatDescr_t matDescr = nullptr;
    // sparseRet = aclsparseCreateMatDescr(&matDescr);
    // CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateMatDescr failed. ERROR: %d\n", sparseRet);
    //           return sparseRet);
    // aclsparseSetMatType(matDescr, ACL_SPARSE_MATRIX_TYPE_GENERAL);
    // aclsparseSetMatIndexBase(matDescr, ACL_SPARSE_INDEX_BASE_ZERO);

    // 4. 准备 Host 数据
    // {按算子需求定义参数并初始化 Host 数据，如 std::vector<float> hA(m * n, 0.0f); }

    // 5. 申请 Device 内存并拷贝数据
    // {按算子需求使用 aclrtMalloc + std::unique_ptr + aclrtFree 管理 Device 内存}
    // {按算子需求使用 aclrtMemcpy 拷贝数据到 Device}

    // 6. 调用 aclsparseS{op}
    // sparseRet = aclsparseS{op}(static_cast<aclsparseHandle_t>(handlePtr.get()), ...);
    // CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseS{op} failed. ERROR: %d\n", sparseRet);
    //           return sparseRet);

    // 7. 同步等待任务执行结束
    auto aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 8. 将结果从 Device 拷贝回 Host 并打印
    // {按算子需求使用 aclrtMemcpy 拷贝结果回 Host 并打印验证}

    // 9. 清理 MatDescr（如步骤 3 已创建）
    // aclsparseDestroyMatDescr(matDescr);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseS{op}Test(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseS{op}Test failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
{按算子实际输出填写，如：}
result[0] is: xxx
result[1] is: xxx
```

### aclsparseD{op}

#### 产品支持情况

- Ascend 950PR&950DT系列产品：{支持/不支持}
- Atlas A3系列产品：{支持/不支持}
- Atlas A2系列产品：{支持/不支持}

#### 函数原型

```cpp
aclsparseStatus_t aclsparseD{op}(aclsparseHandle_t handle, int m, int n, const double *A, ...)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| m | 输入 | int | {参数含义}，Host 内存 |
| A | 输入 | const double*（FP64） | {参数含义}，Device 内存 |

#### 约束说明

- m >= 0
- n >= 0

{如无约束则写"无"，不允许留空。}

#### 支持的稀疏格式

{如算子涉及稀疏矩阵格式（CSR/COO/CSC 等），按下表列出支持情况；如不涉及稀疏格式则删除本节。}

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | {✅/❌} | {说明} |
| COO | {✅/❌} | {说明} |
| CSC | {✅/❌} | {说明} |

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](./compile_and_run_example.md)。

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

int aclsparseD{op}Test(AclContext& ctx)
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

    // 2. 设置 PointerMode（如算子含标量指针参数如 alpha、beta，须设置；否则可省略）
    // sparseRet = aclsparseSetPointerMode(static_cast<aclsparseHandle_t>(handlePtr.get()), ACL_SPARSE_POINTER_MODE_HOST);
    // CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetPointerMode failed. ERROR: %d\n", sparseRet);
    //           return sparseRet);

    // 3. 创建 MatDescr（如算子需要 aclsparseMatDescr_t，须创建并配置；否则可省略）
    // aclsparseMatDescr_t matDescr = nullptr;
    // sparseRet = aclsparseCreateMatDescr(&matDescr);
    // CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateMatDescr failed. ERROR: %d\n", sparseRet);
    //           return sparseRet);
    // aclsparseSetMatType(matDescr, ACL_SPARSE_MATRIX_TYPE_GENERAL);
    // aclsparseSetMatIndexBase(matDescr, ACL_SPARSE_INDEX_BASE_ZERO);

    // 4. 准备 Host 数据
    // {按算子需求定义参数并初始化 Host 数据，如 std::vector<double> hA(m * n, 0.0); }

    // 5. 申请 Device 内存并拷贝数据
    // {按算子需求使用 aclrtMalloc + std::unique_ptr + aclrtFree 管理 Device 内存}
    // {按算子需求使用 aclrtMemcpy 拷贝数据到 Device}

    // 6. 调用 aclsparseD{op}
    // sparseRet = aclsparseD{op}(static_cast<aclsparseHandle_t>(handlePtr.get()), ...);
    // CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseD{op} failed. ERROR: %d\n", sparseRet);
    //           return sparseRet);

    // 7. 同步等待任务执行结束
    auto aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 8. 将结果从 Device 拷贝回 Host 并打印
    // {按算子需求使用 aclrtMemcpy 拷贝结果回 Host 并打印验证}

    // 9. 清理 MatDescr（如步骤 3 已创建）
    // aclsparseDestroyMatDescr(matDescr);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseD{op}Test(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseD{op}Test failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
{按算子实际输出填写，如：}
result[0] is: xxx
result[1] is: xxx
```
