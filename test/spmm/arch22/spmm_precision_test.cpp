#include "spmm_defect_common.h"
#include <limits>
#include <cmath>

struct PrecisionCase {
    const char *name;
    float first;
    float second;
    float initial;
    float alpha;
    float beta;
    float expected;
};

struct PrecisionOperands {
    void *data = nullptr;
    void *workspace = nullptr;
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;

    ~PrecisionOperands()
    {
        if (handle != nullptr) { aclsparseDestroy(handle); }
        if (matA != nullptr) { aclsparseDestroySpMat(matA); }
        if (matB != nullptr) { aclsparseDestroyDnMat(matB); }
        if (matC != nullptr) { aclsparseDestroyDnMat(matC); }
        if (workspace != nullptr) { aclrtFree(workspace); }
        if (data != nullptr) { aclrtFree(data); }
    }
};

static int PreparePrecisionOperands(
    PrecisionOperands &ops, aclrtStream stream, bool transpose, const PrecisionCase &test)
{
    const int32_t offsetsRow[] = {0, 2, 2};
    const int32_t offsetsColumn[] = {0, 1, 2};
    const int32_t columnsRow[] = {0, 1};
    const int32_t columnsColumn[] = {0, 0};
    const float values[] = {test.first, test.second};
    const float dense[] = {1.0f, 1.0f};
    const float initial[] = {test.initial, 0.0f};
    CHECK_TEST(aclrtMalloc(&ops.data, 64, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_ERROR_NONE, "allocate operands");
    auto *bytes = static_cast<unsigned char *>(ops.data);
    CHECK_TEST(aclrtMemcpy(bytes, 12, transpose ? offsetsColumn : offsetsRow, 12,
        ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy offsets");
    CHECK_TEST(aclrtMemcpy(bytes + 12, 8, transpose ? columnsColumn : columnsRow, 8,
        ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy columns");
    CHECK_TEST(aclrtMemcpy(bytes + 20, 8, values, 8, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy values");
    CHECK_TEST(aclrtMemcpy(bytes + 28, 8, dense, 8, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy B");
    CHECK_TEST(aclrtMemcpy(bytes + 36, 8, initial, 8, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy C");
    CHECK_TEST(aclsparseCreate(&ops.handle) == ACL_SPARSE_STATUS_SUCCESS, "create handle");
    CHECK_TEST(aclsparseSetStream(ops.handle, stream) == ACL_SPARSE_STATUS_SUCCESS, "set stream");
    CHECK_TEST(aclsparseSetPointerMode(ops.handle, ACL_SPARSE_POINTER_MODE_HOST) ==
        ACL_SPARSE_STATUS_SUCCESS, "set pointer mode");
    CHECK_TEST(aclsparseCreateCsr(&ops.matA, 2, 2, 2, bytes, bytes + 12, bytes + 20,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT) ==
        ACL_SPARSE_STATUS_SUCCESS, "create A");
    CHECK_TEST(aclsparseCreateDnMat(&ops.matB, 2, 1, 1, bytes + 28, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS, "create B");
    CHECK_TEST(aclsparseCreateDnMat(&ops.matC, 2, 1, 1, bytes + 36, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS, "create C");
    return 0;
}

static int TestPrecisionCancellation(aclrtStream stream, bool transpose, const PrecisionCase &test)
{
    PrecisionOperands ops;
    CHECK_TEST(PreparePrecisionOperands(ops, stream, transpose, test) == 0, "prepare precision case");
    const auto operation = transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
    const auto algorithm = ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG;
    const float alpha = test.alpha;
    const float beta = test.beta;
    size_t size = 0;
    CHECK_TEST(aclsparseSpMMGetBufferSize(ops.handle, operation, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, ops.matA, ops.matB, &beta, ops.matC, ACL_FLOAT, algorithm, &size) ==
        ACL_SPARSE_STATUS_SUCCESS, "get precision workspace");
    CHECK_TEST(aclrtMalloc(&ops.workspace, size, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_ERROR_NONE, "allocate workspace");
    CHECK_TEST(aclsparseSpMMPreprocess(ops.handle, operation, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, ops.matA, ops.matB, &beta, ops.matC, ACL_FLOAT, algorithm, ops.workspace) ==
        ACL_SPARSE_STATUS_SUCCESS, "preprocess precision case");
    const auto status = aclsparseSpMM(ops.handle, operation, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, ops.matA, ops.matB, &beta, ops.matC, ACL_FLOAT, algorithm, ops.workspace);
    const auto syncStatus = aclrtSynchronizeStream(stream);
    CHECK_TEST(status == ACL_SPARSE_STATUS_SUCCESS && syncStatus == ACL_ERROR_NONE, "execute precision case");
    float result[2] = {};
    CHECK_TEST(aclrtMemcpy(result, sizeof(result), static_cast<unsigned char *>(ops.data) + 36,
        sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_ERROR_NONE, "read precision result");
    printf("  [precision] %s transpose=%d result=[%g,%g] expected=%g\n",
        test.name, transpose, result[0], result[1], test.expected);
    CHECK_TEST(result[0] == test.expected && result[1] == 0.0f, "incorrect compensated combination");
    return 0;
}

struct ComplexRangeCase {
    const char *name;
    float a[2];
    float b[2];
    float alpha[2];
    float expected[2];
};

// 高精度路径在应用 alpha 前先算 A*B，中间乘积可能溢出/下溢；这里用 1×1 CSR、
// 可配置 B 直接构造该场景（评审者原例 + 对称反例），验证缩放保护后结果正确。
struct OverflowCase {
    const char *name;
    float a;
    float b;
    float alpha;
    float expected;
};

// 1×2 CSR 的混合量级/累加和回归，A、B 独立取值（累加和溢出需要 A、B 不等）。
struct MixedOverflowCase {
    const char *name;
    float a0;
    float a1;
    float b0;
    float b1;
    float alpha;
    float expected;
};

static int TestPrecisionOverflow(aclrtStream stream, bool transpose, const OverflowCase &test)
{
    PrecisionOperands ops;
    const int32_t offsets[] = {0, 1};
    const int32_t column = 0;
    const float values[] = {test.a};
    const float dense[] = {test.b};
    const float initial[] = {0.0f};
    CHECK_TEST(aclrtMalloc(&ops.data, 64, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_ERROR_NONE, "allocate overflow operands");
    auto *bytes = static_cast<unsigned char *>(ops.data);
    CHECK_TEST(aclrtMemcpy(bytes, 8, offsets, 8, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy overflow offsets");
    CHECK_TEST(aclrtMemcpy(bytes + 8, 4, &column, 4, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy overflow column");
    CHECK_TEST(aclrtMemcpy(bytes + 12, 4, values, 4, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy overflow values");
    CHECK_TEST(aclrtMemcpy(bytes + 16, 4, dense, 4, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy overflow B");
    CHECK_TEST(aclrtMemcpy(bytes + 20, 4, initial, 4, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy overflow C");
    CHECK_TEST(aclsparseCreate(&ops.handle) == ACL_SPARSE_STATUS_SUCCESS, "create overflow handle");
    CHECK_TEST(aclsparseSetStream(ops.handle, stream) == ACL_SPARSE_STATUS_SUCCESS, "set overflow stream");
    CHECK_TEST(aclsparseSetPointerMode(ops.handle, ACL_SPARSE_POINTER_MODE_HOST) ==
        ACL_SPARSE_STATUS_SUCCESS, "set overflow pointer mode");
    CHECK_TEST(aclsparseCreateCsr(&ops.matA, 1, 1, 1, bytes, bytes + 8, bytes + 12,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT) ==
        ACL_SPARSE_STATUS_SUCCESS, "create overflow A");
    CHECK_TEST(aclsparseCreateDnMat(&ops.matB, 1, 1, 1, bytes + 16, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS, "create overflow B");
    CHECK_TEST(aclsparseCreateDnMat(&ops.matC, 1, 1, 1, bytes + 20, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS, "create overflow C");
    const auto operation = transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
    const auto algorithm = ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG;
    const float alpha = test.alpha;
    const float beta = 0.0f;
    size_t size = 0;
    CHECK_TEST(aclsparseSpMMGetBufferSize(ops.handle, operation, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, ops.matA, ops.matB, &beta, ops.matC, ACL_FLOAT, algorithm, &size) ==
        ACL_SPARSE_STATUS_SUCCESS, "get overflow workspace");
    CHECK_TEST(aclrtMalloc(&ops.workspace, size, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_ERROR_NONE, "allocate overflow workspace");
    CHECK_TEST(aclsparseSpMMPreprocess(ops.handle, operation, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, ops.matA, ops.matB, &beta, ops.matC, ACL_FLOAT, algorithm, ops.workspace) ==
        ACL_SPARSE_STATUS_SUCCESS, "preprocess overflow case");
    const auto status = aclsparseSpMM(ops.handle, operation, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, ops.matA, ops.matB, &beta, ops.matC, ACL_FLOAT, algorithm, ops.workspace);
    const auto syncStatus = aclrtSynchronizeStream(stream);
    CHECK_TEST(status == ACL_SPARSE_STATUS_SUCCESS && syncStatus == ACL_ERROR_NONE, "execute overflow case");
    float result = 0.0f;
    CHECK_TEST(aclrtMemcpy(&result, sizeof(result), bytes + 20, sizeof(result),
        ACL_MEMCPY_DEVICE_TO_HOST) == ACL_ERROR_NONE, "read overflow result");
    printf("  [precision-overflow] %s transpose=%d result=%g expected=%g\n", test.name, transpose, result, test.expected);
    const bool match = (result == test.expected) || (std::isnan(result) && std::isnan(test.expected));
    CHECK_TEST(match, "incorrect overflow-protected result");
    return 0;
}

// 1×2 CSR（一行两个非零元），验证同一输出元素内溢出量级与下溢量级乘积共存时，
// 尺度对齐不会上溢（下溢项被正确忽略）。
static int TestPrecisionOverflowMixed(aclrtStream stream, bool transpose, const MixedOverflowCase &test)
{
    PrecisionOperands ops;
    const int32_t offsets[] = {0, 2};
    const int32_t columns[] = {0, 1};
    const float values[] = {test.a0, test.a1};
    const float dense[] = {test.b0, test.b1};
    const float initial[] = {0.0f};
    CHECK_TEST(aclrtMalloc(&ops.data, 64, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_ERROR_NONE, "allocate mixed operands");
    auto *bytes = static_cast<unsigned char *>(ops.data);
    CHECK_TEST(aclrtMemcpy(bytes, 8, offsets, 8, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy mixed offsets");
    CHECK_TEST(aclrtMemcpy(bytes + 8, 8, columns, 8, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy mixed columns");
    CHECK_TEST(aclrtMemcpy(bytes + 16, 8, values, 8, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy mixed values");
    CHECK_TEST(aclrtMemcpy(bytes + 24, 8, dense, 8, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy mixed B");
    CHECK_TEST(aclrtMemcpy(bytes + 32, 4, initial, 4, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy mixed C");
    CHECK_TEST(aclsparseCreate(&ops.handle) == ACL_SPARSE_STATUS_SUCCESS, "create mixed handle");
    CHECK_TEST(aclsparseSetStream(ops.handle, stream) == ACL_SPARSE_STATUS_SUCCESS, "set mixed stream");
    CHECK_TEST(aclsparseSetPointerMode(ops.handle, ACL_SPARSE_POINTER_MODE_HOST) ==
        ACL_SPARSE_STATUS_SUCCESS, "set mixed pointer mode");
    CHECK_TEST(aclsparseCreateCsr(&ops.matA, 1, 2, 2, bytes, bytes + 8, bytes + 16,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT) ==
        ACL_SPARSE_STATUS_SUCCESS, "create mixed A");
    CHECK_TEST(aclsparseCreateDnMat(&ops.matB, 2, 1, 1, bytes + 24, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS, "create mixed B");
    CHECK_TEST(aclsparseCreateDnMat(&ops.matC, 1, 1, 1, bytes + 32, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS, "create mixed C");
    const auto operation = transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
    const auto algorithm = ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG;
    const float alpha = test.alpha;
    const float beta = 0.0f;
    size_t size = 0;
    CHECK_TEST(aclsparseSpMMGetBufferSize(ops.handle, operation, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, ops.matA, ops.matB, &beta, ops.matC, ACL_FLOAT, algorithm, &size) ==
        ACL_SPARSE_STATUS_SUCCESS, "get mixed workspace");
    CHECK_TEST(aclrtMalloc(&ops.workspace, size, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_ERROR_NONE, "allocate mixed workspace");
    CHECK_TEST(aclsparseSpMMPreprocess(ops.handle, operation, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, ops.matA, ops.matB, &beta, ops.matC, ACL_FLOAT, algorithm, ops.workspace) ==
        ACL_SPARSE_STATUS_SUCCESS, "preprocess mixed case");
    const auto status = aclsparseSpMM(ops.handle, operation, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, ops.matA, ops.matB, &beta, ops.matC, ACL_FLOAT, algorithm, ops.workspace);
    const auto syncStatus = aclrtSynchronizeStream(stream);
    CHECK_TEST(status == ACL_SPARSE_STATUS_SUCCESS && syncStatus == ACL_ERROR_NONE, "execute mixed case");
    float result = 0.0f;
    CHECK_TEST(aclrtMemcpy(&result, sizeof(result), bytes + 32, sizeof(result),
        ACL_MEMCPY_DEVICE_TO_HOST) == ACL_ERROR_NONE, "read mixed result");
    printf("  [precision-overflow-mixed] %s transpose=%d result=%g expected=%g\n",
        test.name, transpose, result, test.expected);
    CHECK_TEST(result == test.expected, "incorrect mixed-magnitude result");
    return 0;
}

// corrected = product - comp 本身溢出：product=-FLT_MAX、comp=+2^103 时 corrected 成为
// -Inf，缩放需在形成 corrected 前触发。1×3 CSR（评审者原例，A=[2^127+2^104,2^103,-F]、
// B=[1,1,1]，精确结果 -2^127+5*2^103）。
static int TestPrecisionCorrectedOverflow(aclrtStream stream)
{
    PrecisionOperands ops;
    const int32_t offsets[] = {0, 3};
    const int32_t columns[] = {0, 1, 2};
    const float values[] = {std::ldexp(1.0f, 127) + std::ldexp(1.0f, 104),
        std::ldexp(1.0f, 103), -std::numeric_limits<float>::max()};
    const float dense[] = {1.0f, 1.0f, 1.0f};
    const float initial[] = {0.0f};
    CHECK_TEST(aclrtMalloc(&ops.data, 64, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_ERROR_NONE, "allocate corrected operands");
    auto *bytes = static_cast<unsigned char *>(ops.data);
    CHECK_TEST(aclrtMemcpy(bytes, 8, offsets, 8, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy corrected offsets");
    CHECK_TEST(aclrtMemcpy(bytes + 8, 12, columns, 12, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy corrected columns");
    CHECK_TEST(aclrtMemcpy(bytes + 20, 12, values, 12, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy corrected values");
    CHECK_TEST(aclrtMemcpy(bytes + 32, 12, dense, 12, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy corrected B");
    CHECK_TEST(aclrtMemcpy(bytes + 44, 4, initial, 4, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy corrected C");
    CHECK_TEST(aclsparseCreate(&ops.handle) == ACL_SPARSE_STATUS_SUCCESS, "create corrected handle");
    CHECK_TEST(aclsparseSetStream(ops.handle, stream) == ACL_SPARSE_STATUS_SUCCESS, "set corrected stream");
    CHECK_TEST(aclsparseSetPointerMode(ops.handle, ACL_SPARSE_POINTER_MODE_HOST) == ACL_SPARSE_STATUS_SUCCESS, "set corrected pointer mode");
    CHECK_TEST(aclsparseCreateCsr(&ops.matA, 1, 3, 3, bytes, bytes + 8, bytes + 20,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT) ==
        ACL_SPARSE_STATUS_SUCCESS, "create corrected A");
    CHECK_TEST(aclsparseCreateDnMat(&ops.matB, 3, 1, 1, bytes + 32, ACL_FLOAT, ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS, "create corrected B");
    CHECK_TEST(aclsparseCreateDnMat(&ops.matC, 1, 1, 1, bytes + 44, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS, "create corrected C");
    const auto operation = ACL_SPARSE_OP_NON_TRANSPOSE;
    const auto algorithm = ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    size_t size = 0;
    CHECK_TEST(aclsparseSpMMGetBufferSize(ops.handle, operation, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, ops.matA, ops.matB, &beta, ops.matC, ACL_FLOAT, algorithm, &size) ==
        ACL_SPARSE_STATUS_SUCCESS, "get corrected workspace");
    CHECK_TEST(aclrtMalloc(&ops.workspace, size, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_ERROR_NONE, "allocate corrected workspace");
    CHECK_TEST(aclsparseSpMMPreprocess(ops.handle, operation, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, ops.matA, ops.matB, &beta, ops.matC, ACL_FLOAT, algorithm, ops.workspace) ==
        ACL_SPARSE_STATUS_SUCCESS, "preprocess corrected case");
    const auto status = aclsparseSpMM(ops.handle, operation, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, ops.matA, ops.matB, &beta, ops.matC, ACL_FLOAT, algorithm, ops.workspace);
    const auto syncStatus = aclrtSynchronizeStream(stream);
    CHECK_TEST(status == ACL_SPARSE_STATUS_SUCCESS && syncStatus == ACL_ERROR_NONE, "execute corrected case");
    float result = 0.0f;
    CHECK_TEST(aclrtMemcpy(&result, sizeof(result), bytes + 44, sizeof(result),
        ACL_MEMCPY_DEVICE_TO_HOST) == ACL_ERROR_NONE, "read corrected result");
    const float expected = -std::ldexp(1.0f, 127) + 5.0f * std::ldexp(1.0f, 103);
    printf("  [precision-corrected-overflow] result=%g expected=%g\n", result, expected);
    // 高精度 Kahan 补偿不承诺逐位正确舍入，本反例核心是避免 NaN；只要求有限且量级正确。
    const bool ok = std::isfinite(result) && std::fabs(result - expected) < std::ldexp(1.0f, 105);
    CHECK_TEST(ok, "incorrect corrected-overflow result");
    return 0;
}

static int TestComplexRange(aclrtStream stream, const ComplexRangeCase &test, bool generic)
{
    PrecisionOperands ops;
    const int32_t offsets[] = {0, 1};
    const int32_t column = 0;
    const float zero[] = {0.0f, 0.0f};
    CHECK_TEST(aclrtMalloc(&ops.data, 64, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_ERROR_NONE, "allocate complex operands");
    auto *bytes = static_cast<unsigned char *>(ops.data);
    CHECK_TEST(aclrtMemcpy(bytes, 8, offsets, 8, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy complex offsets");
    CHECK_TEST(aclrtMemcpy(bytes + 8, 4, &column, 4, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy complex column");
    CHECK_TEST(aclrtMemcpy(bytes + 12, 8, test.a, 8, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy complex A");
    CHECK_TEST(aclrtMemcpy(bytes + 20, 8, test.b, 8, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy complex B");
    CHECK_TEST(aclrtMemcpy(bytes + 28, 8, zero, 8, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_ERROR_NONE, "copy complex C");
    CHECK_TEST(aclsparseCreate(&ops.handle) == ACL_SPARSE_STATUS_SUCCESS, "create complex handle");
    CHECK_TEST(aclsparseSetStream(ops.handle, stream) == ACL_SPARSE_STATUS_SUCCESS, "set complex stream");
    CHECK_TEST(aclsparseSetPointerMode(ops.handle, ACL_SPARSE_POINTER_MODE_HOST) ==
        ACL_SPARSE_STATUS_SUCCESS, "set complex pointer mode");
    CHECK_TEST(aclsparseCreateCsr(&ops.matA, 1, 1, 1, bytes, bytes + 8, bytes + 12,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_COMPLEX64) ==
        ACL_SPARSE_STATUS_SUCCESS, "create complex A");
    CHECK_TEST(aclsparseCreateDnMat(&ops.matB, 1, 1, 1, bytes + 20, ACL_COMPLEX64,
        ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS, "create complex B");
    CHECK_TEST(aclsparseCreateDnMat(&ops.matC, 1, 1, 1, bytes + 28, ACL_COMPLEX64,
        ACL_SPARSE_ORDER_ROW) == ACL_SPARSE_STATUS_SUCCESS, "create complex C");
    const auto operation = ACL_SPARSE_OP_NON_TRANSPOSE;
    const auto algorithm = generic ? ACL_SPARSE_SPMM_CSR_ALG1 : ACL_SPARSE_SPMM_ALG_DEFAULT;
    size_t size = 0;
    CHECK_TEST(aclsparseSpMMGetBufferSize(ops.handle, operation, operation,
        test.alpha, ops.matA, ops.matB, zero, ops.matC, ACL_COMPLEX64, algorithm, &size) ==
        ACL_SPARSE_STATUS_SUCCESS, "get complex workspace");
    CHECK_TEST(aclrtMalloc(&ops.workspace, size, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_ERROR_NONE, "allocate complex workspace");
    CHECK_TEST(aclsparseSpMMPreprocess(ops.handle, operation, operation,
        test.alpha, ops.matA, ops.matB, zero, ops.matC, ACL_COMPLEX64, algorithm, ops.workspace) ==
        ACL_SPARSE_STATUS_SUCCESS, "preprocess complex case");
    const auto status = aclsparseSpMM(ops.handle, operation, operation,
        test.alpha, ops.matA, ops.matB, zero, ops.matC, ACL_COMPLEX64, algorithm, ops.workspace);
    const auto syncStatus = aclrtSynchronizeStream(stream);
    CHECK_TEST(status == ACL_SPARSE_STATUS_SUCCESS && syncStatus == ACL_ERROR_NONE, "execute complex case");
    float result[2] = {};
    CHECK_TEST(aclrtMemcpy(result, sizeof(result), bytes + 28, sizeof(result),
        ACL_MEMCPY_DEVICE_TO_HOST) == ACL_ERROR_NONE, "read complex result");
    printf("  [complex-range] %s generic=%d result=[%g,%g] expected=[%g,%g]\n",
        test.name, generic, result[0], result[1], test.expected[0], test.expected[1]);
    CHECK_TEST(result[0] == test.expected[0] && result[1] == test.expected[1], "complex range protection failed");
    return 0;
}

// 高精度路径溢出/下溢与混合量级回归：1×1/1×2 CSR 覆盖单元素与多元素尺度对齐。
static int run_overflow_regression_tests(aclrtStream stream)
{
    int failures = 0;
    const float low100 = std::ldexp(1.0f, -100);
    const float low126 = std::ldexp(1.0f, -126);
    const float high100 = std::ldexp(1.0f, 100);
    const float high126 = std::ldexp(1.0f, 126);
    const OverflowCase overflowCases[] = {
        {"overflow-alpha", 3.0e38f, 2.0f, 0.5f, 3.0e38f},
        {"underflow-alpha", low100, low100, high100, low100},
        {"overflow-alpha-symmetric", high100, high100, low100, high100},
        {"underflow-alpha-symmetric", low126, low126, high126, low126},
        {"infinite-A", std::numeric_limits<float>::infinity(), 2.0f, 1.0f,
            std::numeric_limits<float>::quiet_NaN()},
    };
    for (bool transpose : {false, true}) {
        for (const auto &test : overflowCases) {
            failures += TestPrecisionOverflow(stream, transpose, test);
        }
    }
    const float high70 = std::ldexp(1.0f, 70);
    const float low20 = std::ldexp(1.0f, -20);
    const float high120 = std::ldexp(1.0f, 120);
    const MixedOverflowCase mixedCases[] = {
        {"overflow-then-underflow", high70, low100, high70, low100, low20, high120},
        {"underflow-then-overflow", low100, high70, low100, high70, low20, high120},
        {"overflow-then-normal", std::ldexp(1.0f, 65), std::ldexp(1.0f, 63),
            std::ldexp(1.0f, 65), std::ldexp(1.0f, 63), low20, std::ldexp(17.0f, 106)},
        {"sum-overflow", std::ldexp(1.0f, 127), std::ldexp(1.0f, 127), 1.0f, 1.0f, 0.5f,
            std::ldexp(1.0f, 127)},
    };
    // 1×2 CSR 转置后维度非法（2×1 @ 2×1 内维不匹配），且行列路径共用同一个
    // AddGenericProductScaled，转置路径已由上面的单元素溢出用例覆盖，故这里仅非转置。
    for (const auto &test : mixedCases) {
        failures += TestPrecisionOverflowMixed(stream, false, test);
    }
    failures += TestPrecisionCorrectedOverflow(stream);
    return failures;
}

// 高精度相消与 beta/alpha 残差回归：2×2 CSR 覆盖既有精度反例。
static int run_cancellation_regression_tests(aclrtStream stream)
{
    int failures = 0;
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const PrecisionCase cases[] = {
        {"unit", 16777216.0f, 1.0f, 16777216.0f, 1.0f, -1.0f, 1.0f},
        {"negative", 16777216.0f, 1.0f, 16777216.0f, -1.0f, 1.0f, -1.0f},
        {"double", 16777216.0f, 1.0f, 16777216.0f, 2.0f, -2.0f, 2.0f},
        {"half", 16777216.0f, 1.0f, 16777216.0f, 0.5f, -0.5f, 0.5f},
        {"alpha-rounding", 16777218.0f, 1.0f, 20971524.0f, 1.25f, -1.0f, -0.25f},
        {"beta-rounding", -16785408.0f, 0.0f, 4097.0f, 1.0f, 4097.0f, 1.0f},
        {"beta-rounding-negative", 16785408.0f, 0.0f, 4097.0f, 1.0f, -4097.0f, -1.0f},
        {"alpha-beta-rounding", -13428327.0f, 0.0f, 4097.0f, 1.25f, 4097.0f, 0.25f},
        {"beta-zero-nan", 2.0f, 1.0f, nan, 1.0f, 0.0f, 3.0f},
        {"beta-zero-infinity", 2.0f, 1.0f, infinity, 1.0f, 0.0f, 3.0f},
        {"beta-negative-zero", 2.0f, 1.0f, nan, 1.0f, -0.0f, 3.0f},
        {"beta-product-overflow", 1.0f, 0.0f, 3.0e38f, 1.0f, 2.0f, infinity},
        {"infinite-C", 1.0f, 0.0f, infinity, 1.0f, 1.0f, infinity},
        {"overflow-sum", 3.0e38f, 0.0f, 3.0e38f, 1.0f, 1.0f, infinity},
    };
    for (bool transpose : {false, true}) {
        for (const auto &test : cases) {
            failures += TestPrecisionCancellation(stream, transpose, test);
        }
    }
    return failures;
}

int run_precision_regression_tests(aclrtStream stream)
{
    int failures = 0;
    failures += run_cancellation_regression_tests(stream);
    failures += run_overflow_regression_tests(stream);
    const float large = std::ldexp(1.0f, 126);
    const float high = std::ldexp(1.0f, 100);
    const float low = std::ldexp(1.0f, -100);
    const ComplexRangeCase complexCases[] = {
        {"large-alpha", {4, 4}, {0.25f, 0.25f}, {large, 0}, {0, 2 * large}},
        {"large-A", {large, large}, {0.25f, 0.25f}, {4, 0}, {0, 2 * large}},
        {"large-B", {4, 4}, {large, large}, {0.25f, 0}, {0, 2 * large}},
        {"four-terms", {4, 4}, {0.125f, 0.125f}, {large, large}, {-large, large}},
        {"small-A-B", {low, low}, {low, low}, {high, 0}, {0, 2 * low}},
        {"mixed-exponents", {high, high}, {low, low}, {low, 0}, {0, 2 * low}},
        {"near-max", {std::ldexp(11185126.0f, -23), std::ldexp(11185126.0f, -23)},
            {std::ldexp(14791752.0f, -23), std::ldexp(14791752.0f, -23)},
            {std::ldexp(14271485.0f, 102), 0}, {0, std::numeric_limits<float>::max()}},
        {"alpha-underflow", {std::ldexp(1.0f, -24), 0}, {std::ldexp(1.0f, 127), 0},
            {std::ldexp(1.0f, -126), 0}, {std::ldexp(1.0f, -23), 0}},
        {"alpha-underflow-single", {std::ldexp(1.0f, -24), 1.0f}, {std::ldexp(1.0f, 127), 0},
            {std::ldexp(1.0f, -126), 0}, {std::ldexp(1.0f, -23), 2.0f}},
    };
    for (bool generic : {false, true}) {
        for (const auto &test : complexCases) {
            failures += TestComplexRange(stream, test, generic);
        }
    }
    return failures;
}
