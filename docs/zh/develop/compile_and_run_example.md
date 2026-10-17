# 编译与运行样例

## 前提说明

- 请确保基础环境已搭建完成，包括 NPU 驱动与固件、CANN 软件包（toolkit + ops）等。环境搭建请参考[环境部署](../install/quick_install.md)。

## 步骤一：编译并安装 ops-sparse

开始编译前，请先完成[环境部署](../install/quick_install.md)中的[下载源码](../install/quick_install.md)、[安装基础依赖](../install/quick_install.md#安装基础依赖)与[配置 CANN 环境变量](../install/quick_install.md#环境变量配置)。

1. 编译生成 run 安装包。

    以 Ascend 950 为例，在 ops-sparse 源码根目录下执行：

    ```bash
    bash build.sh --pkg --soc=ascend950
    ```
    - `--soc` 参数须与运行设备的芯片型号一致，支持的 SOC 值：`ascend910b`、`ascend910_93`、`ascend950`、`ascend310p`。
    - 未指定时默认 **Release**。调试 kernel 时可通过环境变量切换为 Debug：

      ```bash
      CMAKE_BUILD_TYPE=Debug bash build.sh --pkg --soc=ascend950
      ```

      直接调用 CMake 时传 `-DCMAKE_BUILD_TYPE=Debug` 同样生效。

    编译成功后，安装包生成在 `build_out/` 目录下，文件名类似 `cann-950-ops-sparse-<cann_version>_linux-<arch>.run`。

    - `${cann_version}`：表示 toolkit 安装包版本号，需满足上文的最低版本要求。
    - `${arch}`：表示 CPU 架构，如 `aarch64`、`x86_64`。
    

<a id="安装-run-包"></a>

2. 安装 run 包。

    ```bash
    # 赋予可执行权限
    chmod +x build_out/cann-950-ops-sparse-*.run
    # 安装到 ${install_path}/cann
    ./build_out/cann-950-ops-sparse-*.run --install --quiet --install-path="${install_path}"
    ```
    - 执行前请设置 `${install_path}` 为非空且不含空格的安装根目录，例如 `install_path=/opt/Ascend`。
    - 安装器在该根目录下追加 `cann/`：root 默认安装到 `/usr/local/Ascend/cann/`，普通用户默认安装到 `${HOME}/Ascend/cann/`。省略 `--install-path` 时，安装器也采用相同的默认根目录。

    安装后，ops-sparse 的头文件和库文件位于：

    ```
    ${install_path}/cann/
    ├── include/
    │   ├── cann_ops_sparse.h
    │   └── cann_ops_sparseLt.h
    └── lib64/
        ├── libops_sparse.so
        └── libops_sparseLt.so
    ```

## 步骤二：准备示例代码

本章以开发和运行环境合设场景为例，即代码开发和代码运行在同一台机器上。这里以 **aclsparseSnnz** 接口为例，调用其他计算接口请根据实际情况自行修改 API 调用脚本（\*.cpp）和编译脚本（CMakeLists.txt）。

1. 新建项目目录。

    ```bash
    mkdir -p ops-sparse-example
    cd ops-sparse-example
    ```

2. 在项目目录下创建代码文件 `test_snnz.cpp`。

    ```bash
    vi test_snnz.cpp
    ```

    将以下代码粘贴到文件中并保存：

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
    
    int aclsparseSnnzTest(AclContext& ctx)
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
    
        // 2. 设置 PointerMode（nnzTotalDevHostPtr 内存位置：HOST 模式为 Host 内存）
        sparseRet = aclsparseSetPointerMode(static_cast<aclsparseHandle_t>(handlePtr.get()), ACL_SPARSE_POINTER_MODE_HOST);
        CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetPointerMode failed. ERROR: %d\n", sparseRet);
                  return sparseRet);
    
        // 3. 准备 Host 数据（列主序 3x3 矩阵）
        //    矩阵内容：
        //    [1.0  0.0  3.0]
        //    [0.0  0.0  4.0]
        //    [2.0  0.0  0.0]
        int m = 3, n = 3, lda = 3;
        std::vector<float> hA(static_cast<size_t>(lda) * n, 0.0f);
        hA[0 * lda + 0] = 1.0f;  // A[0][0]
        hA[0 * lda + 2] = 2.0f;  // A[2][0]
        hA[2 * lda + 0] = 3.0f;  // A[0][2]
        hA[2 * lda + 1] = 4.0f;  // A[1][2]
    
        // 4. 申请 Device 内存并拷贝数据
        void* rawMemA = nullptr;
        auto aclRet = aclrtMalloc(&rawMemA, static_cast<size_t>(lda) * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
        std::unique_ptr<void, aclError (*)(void*)> dA(rawMemA, aclrtFree);
    
        aclRet = aclrtMemcpy(dA.get(), static_cast<size_t>(lda) * n * sizeof(float),
                              hA.data(), static_cast<size_t>(lda) * n * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);
    
        void* rawMemNnzPerRow = nullptr;
        aclRet = aclrtMalloc(&rawMemNnzPerRow, m * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for nnzPerRow failed. ERROR: %d\n", aclRet); return aclRet);
        std::unique_ptr<void, aclError (*)(void*)> dNnzPerRow(rawMemNnzPerRow, aclrtFree);
    
        int hNnzTotal = 0;
    
        // 5. 创建 MatDescr
        aclsparseMatDescr_t matDescr = nullptr;
        sparseRet = aclsparseCreateMatDescr(&matDescr);
        CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateMatDescr failed. ERROR: %d\n", sparseRet);
                  return sparseRet);
        aclsparseSetMatType(matDescr, ACL_SPARSE_MATRIX_TYPE_GENERAL);
        aclsparseSetMatIndexBase(matDescr, ACL_SPARSE_INDEX_BASE_ZERO);
    
        // 6. 调用 aclsparseSnnz（按行统计）
        aclsparseDirection_t dirA = ACL_SPARSE_DIRECTION_ROW;
        sparseRet = aclsparseSnnz(
            static_cast<aclsparseHandle_t>(handlePtr.get()), dirA, m, n, matDescr,
            static_cast<const float*>(dA.get()), lda,
            static_cast<int*>(dNnzPerRow.get()), &hNnzTotal);
        CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSnnz failed. ERROR: %d\n", sparseRet);
                  aclsparseDestroyMatDescr(matDescr); return sparseRet);
    
        // 7. 同步等待任务执行结束
        aclRet = aclrtSynchronizeStream(stream);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
                  aclsparseDestroyMatDescr(matDescr); return aclRet);
    
        // 8. 将结果从 Device 拷贝回 Host 并打印
        std::vector<int> hNnzPerRow(m, 0);
        aclRet = aclrtMemcpy(hNnzPerRow.data(), m * sizeof(int),
                              dNnzPerRow.get(), m * sizeof(int),
                              ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy nnzPerRow from device to host failed. ERROR: %d\n", aclRet);
                  aclsparseDestroyMatDescr(matDescr); return aclRet);
    
        for (int i = 0; i < m; i++) {
            LOG_PRINT("nnzPerRow[%d] is: %d\n", i, hNnzPerRow[i]);
        }
        LOG_PRINT("nnzTotal is: %d\n", hNnzTotal);
    
        // 9. 清理 MatDescr
        aclsparseDestroyMatDescr(matDescr);
    
        return ACL_SUCCESS;
    }
    
    int main()
    {
        AclContext ctx(0);
        auto ret = ctx.Init();
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    
        ret = aclsparseSnnzTest(ctx);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsparseSnnzTest failed. ERROR: %d\n", ret); return ret);
        return 0;
    }
    ```

3. 在项目目录下创建编译脚本 `CMakeLists.txt`。

    ```bash
    vi CMakeLists.txt
    ```

    将以下内容粘贴到文件中并保存（无需修改任何内容，路径通过环境变量自动获取）：

    ```cmake
    cmake_minimum_required(VERSION 3.16)

    project(ACLSPARSE_EXAMPLE)

    add_compile_options(-std=c++11)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    add_executable(opapi_test
                   test_snnz.cpp)

    # CANN toolkit 路径，优先读取环境变量，若未设置则使用默认路径
    if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()

    # ops-sparse 路径，若与 CANN 安装路径不同请通过环境变量 OPS_SPARSE_PATH 指定
    if(NOT "$ENV{OPS_SPARSE_PATH}" STREQUAL "")
        set(OPS_SPARSE_PATH $ENV{OPS_SPARSE_PATH})
    else()
        set(OPS_SPARSE_PATH "${ASCEND_PATH}")
    endif()

    include_directories(
        ${ASCEND_PATH}/include
        ${ASCEND_PATH}/include/aclnn
        ${OPS_SPARSE_PATH}/include
    )

    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libascendcl.so
                          ${OPS_SPARSE_PATH}/lib64/libops_sparse.so)

    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

4. 配置路径环境变量。

    CMakeLists.txt 通过环境变量自动获取路径，无需手动编辑。请在当前终端执行以下命令设置环境变量：

    ```bash
    # 配置 CANN 环境变量（source set_env.sh 后 ASCEND_HOME_PATH 会自动设置）
    source /usr/local/Ascend/cann/set_env.sh
    # 若为指定路径安装，请替换为：
    # source ${install_path}/cann/set_env.sh

    # 若 ops-sparse 安装路径与 CANN 不同，需额外指定（大多数情况无需设置）
    # export OPS_SPARSE_PATH=/your/ops-sparse/install/path
    ```

    > **说明**：
    >
    > - `source set_env.sh` 会自动设置 `ASCEND_HOME_PATH` 环境变量，CMakeLists.txt 通过该变量定位 CANN 头文件和库文件。
    > - 若 ops-sparse 的 run 包已安装到 CANN 目录下（参见 [安装 run 包](#安装-run-包)），则 `OPS_SPARSE_PATH` 默认与 `ASCEND_HOME_PATH` 相同，无需额外设置。

## 步骤三：编译与运行

以下步骤均在步骤二创建的项目目录下执行。

1. 新建 build 目录并执行编译。

    ```bash
    mkdir -p build
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
    make
    ```

    编译成功后，会在 `build/bin/` 目录下生成可执行文件 `opapi_test`。

2. 运行。

    ```bash
    cd bin
    ./opapi_test
    ```

    以 **aclsparseSnnz** 接口（3x3 矩阵，按行统计）为例，预期输出如下：

    ```bash
    nnzPerRow[0] is: 2
    nnzPerRow[1] is: 1
    nnzPerRow[2] is: 1
    nnzTotal is: 4
    ```
    
3. 若执行结果报错，未出现预期结果，可以使用 `aclGetRecentErrMsg` 接口获取报错具体信息。示例如下：

    ```cpp
    ret = aclsparseSnnz(handle, dirA, m, n, matDescr, A, lda, nnzPerRow, &nnzTotal);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        printf("aclsparseSnnz failed. ERROR: %d\n[ERROR msg]%s\n", ret, aclGetRecentErrMsg());
        return ret;
    }
    ```
