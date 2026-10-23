/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

// End-to-end test for aclsparseSpMM with FP32 / FP16 / INT8 data types.
//
//   C = alpha * A * op(B) + beta * C
//
// Coverage:
//   FP32: opB N/T x B/C row/column-major, padded leading dimensions, beta
//   FP16: A(fp16) * B(fp16) -> C(fp16),  computeType=fp32  (accumulate in fp32)
//   INT8: A(int8) * B(int8) -> C(int32), computeType=int32 (accumulate in int32)

#include <acl/acl.h>
#include <cann_ops_sparse.h>
#include "spmm_test_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <set>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

static double ElapsedMs(const TimePoint &start, const TimePoint &end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct VerifyResult {
    bool pass;
    double mere;
    double mare;
    double mae;
};

struct DenseStorageSpec {
    int32_t rows;
    int32_t cols;
    int32_t ld;
    size_t elementCount;
};

static DenseStorageSpec MakeDenseStorageSpec(
    int32_t rows, int32_t cols, aclsparseOrder_t order, int32_t ldPadding = 0)
{
    const bool rowMajor = (order == ACL_SPARSE_ORDER_ROW);
    const int32_t ld = (rowMajor ? cols : rows) + ldPadding;
    const size_t majorCount = static_cast<size_t>(rowMajor ? rows : cols);
    return DenseStorageSpec{rows, cols, ld, majorCount * static_cast<size_t>(ld)};
}

static bool IsLogicalDenseElement(
    size_t index, const DenseStorageSpec &spec, aclsparseOrder_t order)
{
    if (order == ACL_SPARSE_ORDER_ROW) {
        const int32_t row = static_cast<int32_t>(index / static_cast<size_t>(spec.ld));
        const int32_t col = static_cast<int32_t>(index % static_cast<size_t>(spec.ld));
        return row < spec.rows && col < spec.cols;
    }
    const int32_t col = static_cast<int32_t>(index / static_cast<size_t>(spec.ld));
    const int32_t row = static_cast<int32_t>(index % static_cast<size_t>(spec.ld));
    return row < spec.rows && col < spec.cols;
}

static bool VerifyFloatPaddingUnchanged(
    const std::vector<float> &got, const std::vector<float> &expect,
    const DenseStorageSpec &spec, aclsparseOrder_t order)
{
    for (size_t i = 0; i < got.size(); ++i) {
        if (!IsLogicalDenseElement(i, spec, order) && got[i] != expect[i]) {
            std::printf("[FP32] padding changed at physical index %zu: expected %.8f, got %.8f FAIL\n",
                        i, expect[i], got[i]);
            return false;
        }
    }
    std::printf("[FP32] dense padding unchanged PASS\n");
    return true;
}

static VerifyResult FailVerify()
{
    return VerifyResult{false, 0.0, 0.0, 0.0};
}

// RAII: free device buffers + destroy descriptors on any early CHECK_RET return
// (issues #158/#159/#160).
struct SpmmTestResources {
    void *dRowOff = nullptr;
    void *dColInd = nullptr;
    void *dVals = nullptr;
    void *dB = nullptr;
    void *dC = nullptr;
    void *dAlpha = nullptr;
    void *dBeta = nullptr;
    void *dBuffer = nullptr;
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;

    ~SpmmTestResources()
    {
        if (matB != nullptr) {
            aclsparseDestroyDnMat(matB);
        }
        if (matC != nullptr) {
            aclsparseDestroyDnMat(matC);
        }
        if (matA != nullptr) {
            aclsparseDestroySpMat(matA);
        }
        if (handle != nullptr) {
            aclsparseDestroy(handle);
        }
        if (dBuffer != nullptr) {
            aclrtFree(dBuffer);
        }
        if (dRowOff != nullptr) {
            aclrtFree(dRowOff);
        }
        if (dColInd != nullptr) {
            aclrtFree(dColInd);
        }
        if (dVals != nullptr) {
            aclrtFree(dVals);
        }
        if (dB != nullptr) {
            aclrtFree(dB);
        }
        if (dC != nullptr) {
            aclrtFree(dC);
        }
        if (dAlpha != nullptr) {
            aclrtFree(dAlpha);
        }
        if (dBeta != nullptr) {
            aclrtFree(dBeta);
        }
    }

    SpmmTestResources() = default;
    SpmmTestResources(const SpmmTestResources &) = delete;
    SpmmTestResources &operator=(const SpmmTestResources &) = delete;
};

#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do                              \
    {                               \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

void GenerateRandomCsr(int32_t m, int32_t k, int32_t nnz,
                      std::vector<int32_t> *rowOff,
                      std::vector<int32_t> *colInd,
                      std::vector<float>   *vals)
{
    rowOff->assign(static_cast<size_t>(m) + 1, 0);
    colInd->clear();
    vals->clear();
    colInd->reserve(nnz);
    vals->reserve(nnz);

    if (m <= 0 || k <= 0) {
        return;
    }

    int32_t produced = 0;
    for (int32_t i = 0; i < m; ++i) {
        int32_t rowNnz = nnz / m + (i < (nnz % m) ? 1 : 0);
        if (rowNnz > k) {
            rowNnz = k;
        }
        std::set<int32_t> picked;
        while (static_cast<int32_t>(picked.size()) < rowNnz) {
            picked.insert(std::rand() % k);
        }
        for (int32_t col : picked) {
            colInd->push_back(col);
            vals->push_back(-5.0f + 10.0f * static_cast<float>(produced % 10001) / 10000.0f);
            ++produced;
        }
        (*rowOff)[i + 1] = produced;
    }
}

void GenerateSkewedCsr(int32_t m, int32_t k, int32_t nnz,
                       std::vector<int32_t> *rowOff,
                       std::vector<int32_t> *colInd,
                       std::vector<float> *vals)
{
    rowOff->assign(static_cast<size_t>(m) + 1, 0);
    colInd->clear();
    vals->clear();
    if (m <= 0 || k <= 0 || nnz <= 0) {
        return;
    }

    int32_t remaining = std::min(nnz, m * k);
    int32_t produced = 0;
    for (int32_t row = 0; row < m; ++row) {
        const int32_t rowsAfter = m - row - 1;
        int32_t rowNnz = 0;
        if (row == 0) {
            rowNnz = std::min(k, std::max(0, remaining - rowsAfter));
        } else if (remaining > 0) {
            rowNnz = std::min(k, std::max(1, remaining / (rowsAfter + 1)));
        }
        rowNnz = std::min(rowNnz, remaining);
        for (int32_t col = 0; col < rowNnz; ++col) {
            colInd->push_back(col);
            vals->push_back(-3.0f + 0.25f * static_cast<float>(produced % 17));
            ++produced;
        }
        remaining -= rowNnz;
        (*rowOff)[row + 1] = produced;
    }
}

void SpmmCpuFp32(int32_t m, int32_t n,
                 const std::vector<int32_t> &rowOff,
                 const std::vector<int32_t> &colInd,
                 const std::vector<float>   &vals,
                 const std::vector<float>   &B,
                 int32_t ldb, bool bRowMajor, bool opBTranspose,
                 std::vector<float>         *C,
                 int32_t ldc, bool cRowMajor,
                 float alpha, float beta, int32_t indexBase = 0)
{
    for (int32_t r = 0; r < m; ++r) {
        for (int32_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            int32_t s = rowOff[r] - indexBase;
            int32_t e = rowOff[r + 1] - indexBase;
            for (int32_t p = s; p < e; ++p) {
                int32_t c = colInd[p] - indexBase;
                int64_t bIdx;
                if (!opBTranspose) {
                    bIdx = bRowMajor
                        ? static_cast<int64_t>(c) * ldb + j
                        : static_cast<int64_t>(j) * ldb + c;
                } else {
                    bIdx = bRowMajor
                        ? static_cast<int64_t>(j) * ldb + c
                        : static_cast<int64_t>(c) * ldb + j;
                }
                acc += vals[p] * B[static_cast<size_t>(bIdx)];
            }
            int64_t cIdx = cRowMajor
                ? static_cast<int64_t>(r) * ldc + j
                : static_cast<int64_t>(j) * ldc + r;
            float cOld = (*C)[static_cast<size_t>(cIdx)];
            (*C)[static_cast<size_t>(cIdx)] = alpha * acc + beta * cOld;
        }
    }
}

void SpmmCpuInt8(int32_t m, int32_t n,
                 const std::vector<int32_t> &rowOff,
                 const std::vector<int32_t> &colInd,
                 const std::vector<int8_t>  &vals,
                 const std::vector<int8_t>  &B,
                 int32_t ldb, bool bRowMajor, bool opBTranspose,
                 std::vector<int32_t>       *C,
                 int32_t ldc, bool cRowMajor,
                 int32_t alpha, int32_t beta, int32_t indexBase = 0)
{
    for (int32_t r = 0; r < m; ++r) {
        for (int32_t j = 0; j < n; ++j) {
            int64_t acc = 0;
            int32_t s = rowOff[r] - indexBase;
            int32_t e = rowOff[r + 1] - indexBase;
            for (int32_t p = s; p < e; ++p) {
                int32_t c = colInd[p] - indexBase;
                int64_t bIdx;
                if (!opBTranspose) {
                    bIdx = bRowMajor
                        ? static_cast<int64_t>(c) * ldb + j
                        : static_cast<int64_t>(j) * ldb + c;
                } else {
                    bIdx = bRowMajor
                        ? static_cast<int64_t>(j) * ldb + c
                        : static_cast<int64_t>(c) * ldb + j;
                }
                acc += static_cast<int64_t>(vals[p]) *
                       static_cast<int64_t>(B[static_cast<size_t>(bIdx)]);
            }
            int64_t cIdx = cRowMajor
                ? static_cast<int64_t>(r) * ldc + j
                : static_cast<int64_t>(j) * ldc + r;
            int64_t cWide = static_cast<int64_t>(alpha) * acc +
                            static_cast<int64_t>(beta) *
                            static_cast<int64_t>((*C)[static_cast<size_t>(cIdx)]);
            cWide = std::max<int64_t>(-2147483648LL,
                                      std::min<int64_t>(2147483647LL, cWide));
            (*C)[static_cast<size_t>(cIdx)] = static_cast<int32_t>(cWide);
        }
    }
}

VerifyResult VerifyFloat(const std::vector<float> &got, const std::vector<float> &expect,
                 int32_t m, int32_t n, int32_t ldc, bool cRowMajor, const char *dtypeLabel,
                 double mereThreshold, double mareThreshold)
{
    const int32_t total = m * n;
    double mereSum = 0.0;
    double mare = 0.0;
    double mae = 0.0;
    int32_t shownErrors = 0;

    for (int32_t r = 0; r < m; ++r) {
        for (int32_t j = 0; j < n; ++j) {
            int64_t idx = cRowMajor
                ? static_cast<int64_t>(r) * ldc + j
                : static_cast<int64_t>(j) * ldc + r;
            float g = got[static_cast<size_t>(idx)];
            float e = expect[static_cast<size_t>(idx)];
            double absDiff = std::fabs(static_cast<double>(g) - static_cast<double>(e));
            double absExpect = std::fabs(static_cast<double>(e));
            double relErr = absDiff / (absExpect + 1e-7);
            mereSum += relErr;
            if (relErr > mare) {
                mare = relErr;
            }
            if (absDiff > mae) {
                mae = absDiff;
            }
            if (relErr > mareThreshold && shownErrors < 10) {
                std::printf("  diff at (%d, %d): expected %.8f, got %.8f, relErr=%.6e\n",
                            r, j, e, g, relErr);
                ++shownErrors;
            }
        }
    }
    double mere = mereSum / static_cast<double>(total);

    bool pass = (mere < mereThreshold) && (mare < mareThreshold);
    std::printf("[%s] MERE=%.6e (threshold=%.6e) %s\n",
                dtypeLabel, mere, mereThreshold,
                mere < mereThreshold ? "PASS" : "FAIL");
    std::printf("[%s] MARE=%.6e (threshold=%.6e) %s\n",
                dtypeLabel, mare, mareThreshold,
                mare < mareThreshold ? "PASS" : "FAIL");
    std::printf("[%s] MAE =%.6e\n", dtypeLabel, mae);
    if (pass) {
        std::printf("[%s] Overall: PASS\n", dtypeLabel);
    } else {
        std::printf("[%s] Overall: FAIL\n", dtypeLabel);
    }
    return {pass, mere, mare, mae};
}

VerifyResult VerifyInt32(const std::vector<int32_t> &got, const std::vector<int32_t> &expect,
                 int32_t m, int32_t n, int32_t ldc, bool cRowMajor)
{
    int32_t errors = 0;
    int32_t shownErrors = 0;
    for (int32_t r = 0; r < m; ++r) {
        for (int32_t j = 0; j < n; ++j) {
            int64_t idx = cRowMajor
                ? static_cast<int64_t>(r) * ldc + j
                : static_cast<int64_t>(j) * ldc + r;
            int32_t g = got[static_cast<size_t>(idx)];
            int32_t e = expect[static_cast<size_t>(idx)];
            if (g != e) {
                if (shownErrors < 10) {
                    std::printf("  diff at (%d, %d): expected %d, got %d\n", r, j, e, g);
                    ++shownErrors;
                }
                ++errors;
            }
        }
    }
    if (errors > 0) {
        std::printf("[INT32] %d / %d elements mismatch (exact match) FAIL\n", errors, m * n);
        double errRate = static_cast<double>(errors) / static_cast<double>(m * n);
        return {false, errRate, errRate, errRate};
    }
    std::printf("[INT32] all %d elements pass (exact match) PASS\n", m * n);
    return {true, 0.0, 0.0, 0.0};
}

void PrintTiming(const char *label, double msDataGen, double msCpuRef,
                 double msDevAlloc, double msH2D, double msGetBuf,
                 double msPreprocess, double msSpmm, double msD2H, double msVerify)
{
    double msTotal = msDataGen + msCpuRef + msDevAlloc + msH2D +
                     msGetBuf + msPreprocess + msSpmm + msD2H + msVerify;
    std::printf("\n--- Timing: %s ---\n", label);
    std::printf("  Data Generation   : %10.3f ms\n", msDataGen);
    std::printf("  CPU Reference     : %10.3f ms\n", msCpuRef);
    std::printf("  Device Alloc      : %10.3f ms\n", msDevAlloc);
    std::printf("  H2D Copy          : %10.3f ms\n", msH2D);
    std::printf("  GetBufferSize     : %10.3f ms\n", msGetBuf);
    std::printf("  Preprocess        : %10.3f ms\n", msPreprocess);
    std::printf("  Spmm Execute      : %10.3f ms\n", msSpmm);
    std::printf("  D2H Copy          : %10.3f ms\n", msD2H);
    std::printf("  Verify            : %10.3f ms\n", msVerify);
    std::printf("  Total             : %10.3f ms\n", msTotal);
}

inline uint16_t Fp32ToFp16Bits(float v)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t rawExp = (bits >> 23) & 0xFFu;
    int32_t exp = static_cast<int32_t>(rawExp) - 127;
    uint32_t mant = bits & 0x007FFFFFu;

    // NaN / Inf: 保留 NaN（尾数非零）与 Inf 语义。
    if (rawExp == 0xFFu) {
        uint32_t nanBit = (mant != 0u) ? 0x0200u : 0u;
        return static_cast<uint16_t>(sign | 0x7C00u | nanBit);
    }

    // 溢出到 Inf。
    if (exp >= 16) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    // 规格化 fp16：exp ∈ [-14, 15]。使用 IEEE 754 就近偶数舍入。
    if (exp >= -14) {
        uint32_t mant10 = mant >> 13;          // 高 10 位尾数
        uint32_t remainder = mant & 0x1FFFu;   // 低 13 位舍入判据
        uint32_t result = (static_cast<uint32_t>(exp + 15) << 10) | mant10;
        // remainder > 半个 ULP，或恰为半个 ULP 且当前位为奇数时进位。
        if (remainder > 0x1000u || (remainder == 0x1000u && (mant10 & 1u))) {
            result += 1u;  // 进位可自然溢出到指数域（含规格化->Inf）
        }
        return static_cast<uint16_t>(sign | result);
    }

    // 次规格化 fp16：exp ∈ [-25, -15]，否则下溢为 0。
    if (exp >= -25) {
        uint32_t significand = 0x00800000u | mant;      // 含隐含位的 24 位有效数
        uint32_t shift = static_cast<uint32_t>(-exp - 1); // 14..24
        uint32_t frac = significand >> shift;
        uint32_t remainder = significand & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1u);
        if (remainder > half || (remainder == half && (frac & 1u))) {
            frac += 1u;  // 进位可自然升为最小规格化数
        }
        return static_cast<uint16_t>(sign | frac);
    }
    return static_cast<uint16_t>(sign);
}

inline float Fp16BitsToFp32(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            uint32_t shift = __builtin_clz(mant) - 21;
            mant <<= shift;
            exp = 1;
            f = sign | ((exp + 127 - 15 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float v;
    __builtin_memcpy(&v, &f, sizeof(float));
    return v;
}

} // namespace

// ============================================================================
// FP32 test: A(fp32) * B(fp32) -> C(fp32), computeType=ACL_FLOAT
// ============================================================================
static void BuildFp32DenseInputs(const DenseStorageSpec &bSpec, const DenseStorageSpec &cSpec,
                                 aclsparseOrder_t orderB, aclsparseOrder_t orderC, float beta,
                                 std::vector<float> *hB, std::vector<float> *hC)
{
    hB->assign(bSpec.elementCount, 97.0f);
    for (size_t i = 0; i < hB->size(); ++i) {
        if (IsLogicalDenseElement(i, bSpec, orderB)) {
            (*hB)[i] = -5.0f + 10.0f * static_cast<float>(i % 10001) / 10000.0f;
        }
    }
    hC->assign(cSpec.elementCount, -101.0f);
    for (size_t i = 0; i < hC->size(); ++i) {
        if (IsLogicalDenseElement(i, cSpec, orderC)) {
            (*hC)[i] = beta == 0.0f ? 0.0f :
                -5.0f + 10.0f * static_cast<float>(i % 10001) / 10000.0f;
        }
    }
}

static bool AllocateFp32Resources(int32_t m, int32_t nnz, const DenseStorageSpec &bSpec,
                                  const DenseStorageSpec &cSpec, bool deviceScalars,
                                  SpmmTestResources *res, int32_t **dRowOff, int32_t **dColInd,
                                  float **dVals, float **dB, float **dC)
{
    aclError aclRet = aclrtMalloc(&res->dRowOff, sizeof(int32_t) * (m + 1),
                                  ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return false);
    *dRowOff = static_cast<int32_t *>(res->dRowOff);
    aclRet = aclrtMalloc(&res->dColInd, sizeof(int32_t) * static_cast<size_t>(std::max(1, nnz)),
                         ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return false);
    *dColInd = static_cast<int32_t *>(res->dColInd);
    aclRet = aclrtMalloc(&res->dVals, sizeof(float) * static_cast<size_t>(std::max(1, nnz)),
                         ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return false);
    *dVals = static_cast<float *>(res->dVals);
    aclRet = aclrtMalloc(&res->dB, sizeof(float) * bSpec.elementCount, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return false);
    *dB = static_cast<float *>(res->dB);
    aclRet = aclrtMalloc(&res->dC, sizeof(float) * cSpec.elementCount, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return false);
    *dC = static_cast<float *>(res->dC);
    if (!deviceScalars) return true;
    aclRet = aclrtMalloc(&res->dAlpha, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc alpha failed. ERROR: %d\n", aclRet); return false);
    aclRet = aclrtMalloc(&res->dBeta, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc beta failed. ERROR: %d\n", aclRet); return false);
    return true;
}

static bool UploadFp32Resources(int32_t m, int32_t nnz, const DenseStorageSpec &bSpec,
                                const DenseStorageSpec &cSpec,
                                const std::vector<int32_t> &hRowOff,
                                const std::vector<int32_t> &hColInd,
                                const std::vector<float> &hValsFp32,
                                const std::vector<float> &hB, const std::vector<float> &hC,
                                float alpha, float beta, bool deviceScalars,
                                const SpmmTestResources &res, int32_t *dRowOff,
                                int32_t *dColInd, float *dVals, float *dB, float *dC)
{
    aclError aclRet = aclrtMemcpy(dRowOff, sizeof(int32_t) * (m + 1), hRowOff.data(),
                                  sizeof(int32_t) * (m + 1), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return false);
    if (nnz > 0) {
        aclRet = aclrtMemcpy(dColInd, sizeof(int32_t) * nnz, hColInd.data(),
                             sizeof(int32_t) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return false);
        aclRet = aclrtMemcpy(dVals, sizeof(float) * nnz, hValsFp32.data(),
                             sizeof(float) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return false);
    }
    aclRet = aclrtMemcpy(dB, sizeof(float) * bSpec.elementCount, hB.data(),
                         sizeof(float) * bSpec.elementCount, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return false);
    aclRet = aclrtMemcpy(dC, sizeof(float) * cSpec.elementCount, hC.data(),
                         sizeof(float) * cSpec.elementCount, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return false);
    if (!deviceScalars) return true;
    aclRet = aclrtMemcpy(res.dAlpha, sizeof(float), &alpha, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy alpha failed. ERROR: %d\n", aclRet); return false);
    aclRet = aclrtMemcpy(res.dBeta, sizeof(float), &beta, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy beta failed. ERROR: %d\n", aclRet); return false);
    return true;
}

static bool CreateFp32Descriptors(aclrtStream stream, int32_t m, int32_t k, int32_t nnz,
                                  const DenseStorageSpec &bSpec, const DenseStorageSpec &cSpec,
                                  aclsparseOrder_t orderB, aclsparseOrder_t orderC,
                                  int32_t indexBase, bool deviceScalars,
                                  SpmmTestResources *res, int32_t *dRowOff, int32_t *dColInd,
                                  float *dVals, float *dB, float *dC)
{
    aclsparseStatus_t sparseRet = aclsparseCreate(&res->handle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet); return false);
    sparseRet = aclsparseSetStream(res->handle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet); return false);
    if (deviceScalars) {
        sparseRet = aclsparseSetPointerMode(res->handle, ACL_SPARSE_POINTER_MODE_DEVICE);
        CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
                  LOG_PRINT("aclsparseSetPointerMode failed. ERROR: %d\n", sparseRet); return false);
    }
    sparseRet = aclsparseCreateCsr(&res->matA, m, k, nnz, dRowOff, dColInd, dVals,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   indexBase == 0 ? ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE,
                                   ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet); return false);
    sparseRet = aclsparseCreateDnMat(&res->matB, bSpec.rows, bSpec.cols, bSpec.ld, dB,
                                     ACL_FLOAT, orderB);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return false);
    sparseRet = aclsparseCreateDnMat(&res->matC, m, cSpec.cols, cSpec.ld, dC,
                                     ACL_FLOAT, orderC);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return false);
    return true;
}

static bool PrepareFp32Workspace(SpmmTestResources *res, aclsparseOperation_t opB,
                                 const void *alphaArg, const void *betaArg,
                                 aclDataType computeType, aclsparseSpMMAlg_t alg,
                                 bool usePreprocess, double *msGetBuf,
                                 double *msPreprocess)
{
    TimePoint t0 = Clock::now();
    size_t bufferSize = 0;
    aclsparseStatus_t sparseRet = aclsparseSpMMGetBufferSize(res->handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB, alphaArg, res->matA, res->matB, betaArg,
        res->matC, computeType, alg, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return false);
    *msGetBuf = ElapsedMs(t0, Clock::now());
    std::printf("Workspace bytes: %zu\n", bufferSize);
    const aclError aclRet = aclrtMalloc(&res->dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return false);
    if (!usePreprocess) return true;
    t0 = Clock::now();
    sparseRet = aclsparseSpMMPreprocess(res->handle, ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        alphaArg, res->matA, res->matB, betaArg, res->matC, computeType, alg, res->dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return false);
    *msPreprocess = ElapsedMs(t0, Clock::now());
    return true;
}

static bool ExecuteFp32(aclrtStream stream, SpmmTestResources &res,
                        aclsparseOperation_t opB, const void *alphaArg, const void *betaArg,
                        aclsparseSpMMAlg_t alg, bool repeated, double *msSpmm)
{
    const TimePoint t0 = Clock::now();
    aclsparseStatus_t sparseRet = aclsparseSpMM(res.handle, ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        alphaArg, res.matA, res.matB, betaArg, res.matC, ACL_FLOAT, alg, res.dBuffer);
    if (sparseRet != ACL_SPARSE_STATUS_SUCCESS) {
        if (repeated) LOG_PRINT("repeat aclsparseSpMM failed. ERROR: %d\n", sparseRet);
        return false;
    }
    const aclError aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        if (repeated) {
            LOG_PRINT("repeat synchronize failed. ERROR: %d\n", aclRet);
        } else {
            LOG_PRINT("[ERROR] aclsparseSpMM: aclrtSynchronizeStream failed, ret=%d\n", aclRet);
        }
        return false;
    }
    *msSpmm += ElapsedMs(t0, Clock::now());
    return true;
}

static bool RepeatFp32Execution(aclrtStream stream, SpmmTestResources &res,
                                aclsparseOperation_t opB, const void *alphaArg,
                                const void *betaArg, aclsparseSpMMAlg_t alg, float *dC,
                                const std::vector<float> &hC, int32_t repeatCount,
                                double *msSpmm)
{
    if (!ExecuteFp32(stream, res, opB, alphaArg, betaArg, alg, false, msSpmm)) return false;
    for (int32_t repeat = 1; repeat < std::max(1, repeatCount); ++repeat) {
        const aclError aclRet = aclrtMemcpy(dC, sizeof(float) * hC.size(), hC.data(),
                                             sizeof(float) * hC.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS,
                  LOG_PRINT("aclrtMemcpy repeat C failed. ERROR: %d\n", aclRet); return false);
        if (!ExecuteFp32(stream, res, opB, alphaArg, betaArg, alg, true, msSpmm)) return false;
    }
    if (repeatCount > 1) std::printf("[FP32] repeated execution count=%d PASS\n", repeatCount);
    return true;
}

static VerifyResult RunSpmmTestFp32(int32_t deviceId, aclrtStream stream,
                            int32_t m, int32_t k, int32_t n, int32_t nnz,
                            const std::vector<int32_t> &hRowOff,
                            const std::vector<int32_t> &hColInd,
                            const std::vector<float>   &hValsFp32,
                            float alpha, float beta,
                            aclsparseOperation_t opB,
                            aclsparseOrder_t orderB,
                            aclsparseOrder_t orderC,
                            int32_t bLdPadding,
                            int32_t cLdPadding,
                            bool usePreprocess = true,
                            bool deviceScalars = false,
                            int32_t indexBase = 0,
                            int32_t repeatCount = 1,
                            aclsparseSpMMAlg_t alg = ACL_SPARSE_SPMM_CSR_ALG1)
{
    TimePoint t0, t1;
    double msDataGen = 0, msCpuRef = 0, msDevAlloc = 0, msH2D = 0;
    double msGetBuf = 0, msPreprocess = 0, msSpmm = 0, msD2H = 0, msVerify = 0;

    const bool bRowMajor = (orderB == ACL_SPARSE_ORDER_ROW);
    const bool cRowMajor = (orderC == ACL_SPARSE_ORDER_ROW);
    const bool opBTranspose = (opB == ACL_SPARSE_OP_TRANSPOSE);
    const int32_t bRows = opBTranspose ? n : k;
    const int32_t bCols = opBTranspose ? k : n;
    const DenseStorageSpec bSpec = MakeDenseStorageSpec(bRows, bCols, orderB, bLdPadding);
    const DenseStorageSpec cSpec = MakeDenseStorageSpec(m, n, orderC, cLdPadding);
    const int32_t ldb = bSpec.ld;
    const int32_t ldc = cSpec.ld;

    const char *opBStr = opBTranspose ? "T" : "N";
    const char *obStr = bRowMajor ? "R" : "C";
    const char *ocStr = cRowMajor ? "R" : "C";
    std::printf("\n====== SPMM FP32 Test [opB=%s, B=%s, C=%s, ldb=%d, ldc=%d] ======\n",
                opBStr, obStr, ocStr, ldb, ldc);

    t0 = Clock::now();
    std::vector<float> hB;
    std::vector<float> hC;
    BuildFp32DenseInputs(bSpec, cSpec, orderB, orderC, beta, &hB, &hC);
    std::vector<float> hCRef(hC);
    t1 = Clock::now();
    msDataGen = ElapsedMs(t0, t1);

    t0 = Clock::now();
    SpmmCpuFp32(m, n, hRowOff, hColInd, hValsFp32, hB, ldb, bRowMajor, opBTranspose, &hCRef, ldc, cRowMajor, alpha, beta, indexBase);
    t1 = Clock::now();
    msCpuRef = ElapsedMs(t0, t1);

    t0 = Clock::now();
    SpmmTestResources res;
    int32_t *dRowOff = nullptr;
    int32_t *dColInd = nullptr;
    float   *dVals   = nullptr;
    float   *dB      = nullptr;
    float   *dC      = nullptr;
    if (!AllocateFp32Resources(m, nnz, bSpec, cSpec, deviceScalars, &res,
                               &dRowOff, &dColInd, &dVals, &dB, &dC)) return FailVerify();
    t1 = Clock::now();
    msDevAlloc = ElapsedMs(t0, t1);

    t0 = Clock::now();
    if (!UploadFp32Resources(m, nnz, bSpec, cSpec, hRowOff, hColInd, hValsFp32, hB, hC,
                             alpha, beta, deviceScalars, res, dRowOff, dColInd, dVals, dB, dC)) {
        return FailVerify();
    }
    t1 = Clock::now();
    msH2D = ElapsedMs(t0, t1);

    const void *alphaArg = deviceScalars ? static_cast<const void *>(res.dAlpha)
                                         : static_cast<const void *>(&alpha);
    const void *betaArg = deviceScalars ? static_cast<const void *>(res.dBeta)
                                        : static_cast<const void *>(&beta);
    if (!CreateFp32Descriptors(stream, m, k, nnz, bSpec, cSpec, orderB, orderC,
                               indexBase, deviceScalars, &res, dRowOff, dColInd, dVals, dB, dC)) {
        return FailVerify();
    }

    if (!PrepareFp32Workspace(&res, opB, alphaArg, betaArg, ACL_FLOAT, alg, usePreprocess,
                              &msGetBuf, &msPreprocess) ||
        !RepeatFp32Execution(stream, res, opB, alphaArg, betaArg, alg, dC, hC, repeatCount,
                             &msSpmm)) return FailVerify();

    t0 = Clock::now();
    const aclError aclRet = aclrtMemcpy(hC.data(), sizeof(float) * cSpec.elementCount, dC,
                                        sizeof(float) * cSpec.elementCount, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msD2H = ElapsedMs(t0, t1);

    t0 = Clock::now();
    VerifyResult vr = VerifyFloat(hC, hCRef, m, n, ldc, cRowMajor, "FP32",
                          1.0 / (1 << 13), 10.0 / (1 << 13));
    if (bLdPadding != 0 || cLdPadding != 0) {
        vr.pass = VerifyFloatPaddingUnchanged(hC, hCRef, cSpec, orderC) && vr.pass;
    }
    t1 = Clock::now();
    msVerify = ElapsedMs(t0, t1);

    PrintTiming("FP32", msDataGen, msCpuRef, msDevAlloc, msH2D, msGetBuf, msPreprocess, msSpmm, msD2H, msVerify);

    return vr;
}

// ============================================================================
// FP16 test: A(fp16) * B(fp16) -> C(fp16), computeType=ACL_FLOAT
//   Host uses uint16_t to store IEEE 754 half-precision bit patterns.
//   CPU reference computed in fp32 for accuracy; fp16 tolerance is looser.
// ============================================================================
static VerifyResult RunSpmmTestFp16(int32_t deviceId, aclrtStream stream,
                            int32_t m, int32_t k, int32_t n, int32_t nnz,
                            const std::vector<int32_t> &hRowOff,
                            const std::vector<int32_t> &hColInd,
                            const std::vector<float>   &hValsFp32,
                            float alpha, float beta,
                            aclsparseOperation_t opB,
                            aclsparseOrder_t orderB,
                            aclsparseOrder_t orderC)
{
    TimePoint t0, t1;
    double msDataGen = 0, msCpuRef = 0, msDevAlloc = 0, msH2D = 0;
    double msGetBuf = 0, msPreprocess = 0, msSpmm = 0, msD2H = 0, msVerify = 0;

    const bool bRowMajor = (orderB == ACL_SPARSE_ORDER_ROW);
    const bool cRowMajor = (orderC == ACL_SPARSE_ORDER_ROW);
    const bool opBTranspose = (opB == ACL_SPARSE_OP_TRANSPOSE);
    const int32_t bRows = opBTranspose ? n : k;
    const int32_t bCols = opBTranspose ? k : n;
    const DenseStorageSpec bSpec = MakeDenseStorageSpec(bRows, bCols, orderB);
    const DenseStorageSpec cSpec = MakeDenseStorageSpec(m, n, orderC);
    const int32_t ldb = bSpec.ld;
    const int32_t ldc = cSpec.ld;

    const char *opBStr = opBTranspose ? "T" : "N";
    const char *obStr = bRowMajor ? "R" : "C";
    const char *ocStr = cRowMajor ? "R" : "C";
    std::printf("\n====== SPMM FP16 Test [opB=%s, B=%s, C=%s] ======\n", opBStr, obStr, ocStr);

    t0 = Clock::now();
    std::vector<float> hBFp32(bSpec.elementCount);
    for (size_t i = 0; i < hBFp32.size(); ++i) {
        hBFp32[i] = -5.0f + 10.0f * static_cast<float>(i % 10001) / 10000.0f;
    }
    std::vector<float> hCRefFp32(cSpec.elementCount, 0.0f);
    if (beta != 0.0f) {
        for (size_t i = 0; i < hCRefFp32.size(); ++i) {
            hCRefFp32[i] = -5.0f + 10.0f * static_cast<float>(i % 10001) / 10000.0f;
        }
    }

    std::vector<uint16_t> hValsFp16(static_cast<size_t>(nnz));
    for (int32_t i = 0; i < nnz; ++i) {
        hValsFp16[i] = Fp32ToFp16Bits(hValsFp32[i]);
    }
    std::vector<uint16_t> hBFp16(bSpec.elementCount);
    for (size_t i = 0; i < hBFp16.size(); ++i) {
        hBFp16[i] = Fp32ToFp16Bits(hBFp32[i]);
    }
    std::vector<uint16_t> hCFp16(cSpec.elementCount);
    for (size_t i = 0; i < hCFp16.size(); ++i) {
        hCFp16[i] = Fp32ToFp16Bits(hCRefFp32[i]);
    }
    t1 = Clock::now();
    msDataGen = ElapsedMs(t0, t1);

    t0 = Clock::now();
    std::vector<float> hValsFp16RoundTrip(static_cast<size_t>(nnz));
    for (int32_t i = 0; i < nnz; ++i) {
        hValsFp16RoundTrip[i] = Fp16BitsToFp32(hValsFp16[i]);
    }
    std::vector<float> hBFp16RoundTrip(bSpec.elementCount);
    for (size_t i = 0; i < hBFp16RoundTrip.size(); ++i) {
        hBFp16RoundTrip[i] = Fp16BitsToFp32(hBFp16[i]);
    }
    std::vector<float> hCRefFp16RoundTrip(cSpec.elementCount);
    for (size_t i = 0; i < hCRefFp16RoundTrip.size(); ++i) {
        hCRefFp16RoundTrip[i] = Fp16BitsToFp32(hCFp16[i]);
    }
    SpmmCpuFp32(m, n, hRowOff, hColInd, hValsFp16RoundTrip, hBFp16RoundTrip, ldb, bRowMajor, opBTranspose, &hCRefFp16RoundTrip, ldc, cRowMajor, alpha, beta);
    constexpr float kFp16Max = 65504.0f;
    for (size_t i = 0; i < hCRefFp16RoundTrip.size(); ++i) {
        if (hCRefFp16RoundTrip[i] > kFp16Max) hCRefFp16RoundTrip[i] = kFp16Max;
        if (hCRefFp16RoundTrip[i] < -kFp16Max) hCRefFp16RoundTrip[i] = -kFp16Max;
        hCRefFp16RoundTrip[i] = Fp16BitsToFp32(Fp32ToFp16Bits(hCRefFp16RoundTrip[i]));
    }
    t1 = Clock::now();
    msCpuRef = ElapsedMs(t0, t1);

    t0 = Clock::now();
    SpmmTestResources res;
    int32_t  *dRowOff = nullptr;
    int32_t  *dColInd = nullptr;
    void     *dVals   = nullptr;
    void     *dB      = nullptr;
    void     *dC      = nullptr;
    aclError aclRet = aclrtMalloc((void **)&res.dRowOff, sizeof(int32_t) * (m + 1), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dRowOff = static_cast<int32_t *>(res.dRowOff);
    aclRet = aclrtMalloc((void **)&res.dColInd, sizeof(int32_t) * nnz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dColInd = static_cast<int32_t *>(res.dColInd);
    aclRet = aclrtMalloc(&res.dVals, sizeof(uint16_t) * nnz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dVals = res.dVals;
    aclRet = aclrtMalloc(&res.dB, sizeof(uint16_t) * bSpec.elementCount, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dB = res.dB;
    aclRet = aclrtMalloc(&res.dC, sizeof(uint16_t) * cSpec.elementCount, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dC = res.dC;
    t1 = Clock::now();
    msDevAlloc = ElapsedMs(t0, t1);

    t0 = Clock::now();
    aclRet = aclrtMemcpy(dRowOff, sizeof(int32_t) * (m + 1), hRowOff.data(), sizeof(int32_t) * (m + 1), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dColInd, sizeof(int32_t) * nnz, hColInd.data(), sizeof(int32_t) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dVals, sizeof(uint16_t) * nnz, hValsFp16.data(), sizeof(uint16_t) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dB, sizeof(uint16_t) * bSpec.elementCount, hBFp16.data(),
                         sizeof(uint16_t) * bSpec.elementCount, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dC, sizeof(uint16_t) * cSpec.elementCount, hCFp16.data(),
                         sizeof(uint16_t) * cSpec.elementCount, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msH2D = ElapsedMs(t0, t1);

    aclsparseStatus_t sparseRet = aclsparseCreate(&res.handle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet); return FailVerify());
    sparseRet = aclsparseSetStream(res.handle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet); return FailVerify());

    sparseRet = aclsparseCreateCsr(&res.matA, m, k, nnz, dRowOff, dColInd, dVals,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT16);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet);
              return FailVerify());

    sparseRet = aclsparseCreateDnMat(&res.matB, bSpec.rows, bSpec.cols, ldb, dB, ACL_FLOAT16, orderB);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    sparseRet = aclsparseCreateDnMat(&res.matC, m, n, ldc, dC, ACL_FLOAT16, orderC);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());

    t0 = Clock::now();
    size_t bufferSize = 0;
    sparseRet = aclsparseSpMMGetBufferSize(res.handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, res.matA, res.matB, &beta, res.matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    t1 = Clock::now();
    msGetBuf = ElapsedMs(t0, t1);
    std::printf("Workspace bytes: %zu\n", bufferSize);

    aclRet = aclrtMalloc(&res.dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());

    t0 = Clock::now();
    sparseRet = aclsparseSpMMPreprocess(res.handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, res.matA, res.matB, &beta, res.matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1, res.dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    t1 = Clock::now();
    msPreprocess = ElapsedMs(t0, t1);

    t0 = Clock::now();
    sparseRet = aclsparseSpMM(res.handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, res.matA, res.matB, &beta, res.matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1, res.dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclsparseSpMM: aclrtSynchronizeStream failed, ret=%d\n", aclRet);
              return FailVerify());
    t1 = Clock::now();
    msSpmm = ElapsedMs(t0, t1);

    t0 = Clock::now();
    aclRet = aclrtMemcpy(hCFp16.data(), sizeof(uint16_t) * cSpec.elementCount, dC,
                         sizeof(uint16_t) * cSpec.elementCount, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msD2H = ElapsedMs(t0, t1);

    t0 = Clock::now();
    std::vector<float> hCFp32FromFp16(cSpec.elementCount);
    for (size_t i = 0; i < hCFp32FromFp16.size(); ++i) {
        hCFp32FromFp16[i] = Fp16BitsToFp32(hCFp16[i]);
    }
    VerifyResult vr = VerifyFloat(hCFp32FromFp16, hCRefFp16RoundTrip, m, n, ldc, cRowMajor, "FP16",
                          1.0 / (1 << 10), 10.0 / (1 << 10));
    t1 = Clock::now();
    msVerify = ElapsedMs(t0, t1);

    PrintTiming("FP16", msDataGen, msCpuRef, msDevAlloc, msH2D, msGetBuf, msPreprocess, msSpmm, msD2H, msVerify);

    return vr;
}

// ============================================================================
// INT8 test: A(int8) * B(int8) -> C(int32), computeType=ACL_INT32
//   CPU reference computed in int32; exact match required.
// ============================================================================
static VerifyResult RunSpmmTestInt8(int32_t deviceId, aclrtStream stream,
                            int32_t m, int32_t k, int32_t n, int32_t nnz,
                            const std::vector<int32_t> &hRowOff,
                            const std::vector<int32_t> &hColInd,
                            int32_t alpha, int32_t beta,
                            aclsparseOperation_t opB,
                            aclsparseOrder_t orderB,
                            aclsparseOrder_t orderC,
                            int8_t fixedInput = 0)
{
    TimePoint t0, t1;
    double msDataGen = 0, msCpuRef = 0, msDevAlloc = 0, msH2D = 0;
    double msGetBuf = 0, msPreprocess = 0, msSpmm = 0, msD2H = 0, msVerify = 0;

    const bool bRowMajor = (orderB == ACL_SPARSE_ORDER_ROW);
    const bool cRowMajor = (orderC == ACL_SPARSE_ORDER_ROW);
    const bool opBTranspose = (opB == ACL_SPARSE_OP_TRANSPOSE);
    const int32_t bRows = opBTranspose ? n : k;
    const int32_t bCols = opBTranspose ? k : n;
    const DenseStorageSpec bSpec = MakeDenseStorageSpec(bRows, bCols, orderB);
    const DenseStorageSpec cSpec = MakeDenseStorageSpec(m, n, orderC);
    const int32_t ldb = bSpec.ld;
    const int32_t ldc = cSpec.ld;

    const char *opBStr = opBTranspose ? "T" : "N";
    const char *obStr = bRowMajor ? "R" : "C";
    const char *ocStr = cRowMajor ? "R" : "C";
    std::printf("\n====== SPMM INT8 Test [opB=%s, B=%s, C=%s] ======\n", opBStr, obStr, ocStr);

    t0 = Clock::now();
    std::vector<int8_t> hValsInt8(static_cast<size_t>(nnz));
    for (int32_t i = 0; i < nnz; ++i) {
        hValsInt8[i] = (fixedInput == 0) ? static_cast<int8_t>((i % 11) - 5) : fixedInput;
    }
    std::vector<int8_t> hBInt8(bSpec.elementCount);
    for (size_t i = 0; i < hBInt8.size(); ++i) {
        hBInt8[i] = (fixedInput == 0) ? static_cast<int8_t>((i % 11) - 5) : fixedInput;
    }
    std::vector<int32_t> hCInt32(cSpec.elementCount, 0);
    if (beta != 0) {
        for (size_t i = 0; i < hCInt32.size(); ++i) {
            hCInt32[i] = (fixedInput == 0) ? static_cast<int32_t>((i % 11) - 5)
                                           : static_cast<int32_t>(fixedInput);
        }
    }
    std::vector<int32_t> hCRefInt32(hCInt32);
    t1 = Clock::now();
    msDataGen = ElapsedMs(t0, t1);

    t0 = Clock::now();
    SpmmCpuInt8(m, n, hRowOff, hColInd, hValsInt8, hBInt8, ldb, bRowMajor, opBTranspose, &hCRefInt32, ldc, cRowMajor, alpha, beta);
    t1 = Clock::now();
    msCpuRef = ElapsedMs(t0, t1);

    t0 = Clock::now();
    SpmmTestResources res;
    int32_t  *dRowOff = nullptr;
    int32_t  *dColInd = nullptr;
    int8_t   *dVals   = nullptr;
    int8_t   *dB      = nullptr;
    int32_t  *dC      = nullptr;
    aclError aclRet = aclrtMalloc((void **)&res.dRowOff, sizeof(int32_t) * (m + 1), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dRowOff = static_cast<int32_t *>(res.dRowOff);
    aclRet = aclrtMalloc((void **)&res.dColInd,
                         sizeof(int32_t) * static_cast<size_t>(std::max(1, nnz)),
                         ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dColInd = static_cast<int32_t *>(res.dColInd);
    aclRet = aclrtMalloc((void **)&res.dVals,
                         sizeof(int8_t) * static_cast<size_t>(std::max(1, nnz)),
                         ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dVals = static_cast<int8_t *>(res.dVals);
    aclRet = aclrtMalloc((void **)&res.dB, sizeof(int8_t) * bSpec.elementCount, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dB = static_cast<int8_t *>(res.dB);
    aclRet = aclrtMalloc((void **)&res.dC, sizeof(int32_t) * cSpec.elementCount, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dC = static_cast<int32_t *>(res.dC);
    t1 = Clock::now();
    msDevAlloc = ElapsedMs(t0, t1);

    t0 = Clock::now();
    aclRet = aclrtMemcpy(dRowOff, sizeof(int32_t) * (m + 1), hRowOff.data(), sizeof(int32_t) * (m + 1), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    if (nnz > 0) {
        aclRet = aclrtMemcpy(dColInd, sizeof(int32_t) * nnz, hColInd.data(), sizeof(int32_t) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
        aclRet = aclrtMemcpy(dVals, sizeof(int8_t) * nnz, hValsInt8.data(), sizeof(int8_t) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    }
    aclRet = aclrtMemcpy(dB, sizeof(int8_t) * bSpec.elementCount, hBInt8.data(),
                         sizeof(int8_t) * bSpec.elementCount, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dC, sizeof(int32_t) * cSpec.elementCount, hCInt32.data(),
                         sizeof(int32_t) * cSpec.elementCount, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msH2D = ElapsedMs(t0, t1);

    aclsparseStatus_t sparseRet = aclsparseCreate(&res.handle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet); return FailVerify());
    sparseRet = aclsparseSetStream(res.handle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet); return FailVerify());

    sparseRet = aclsparseCreateCsr(&res.matA, m, k, nnz, dRowOff, dColInd, dVals,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_INT8);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet);
              return FailVerify());

    sparseRet = aclsparseCreateDnMat(&res.matB, bSpec.rows, bSpec.cols, ldb, dB, ACL_INT8, orderB);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    sparseRet = aclsparseCreateDnMat(&res.matC, m, n, ldc, dC, ACL_INT32, orderC);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());

    t0 = Clock::now();
    size_t bufferSize = 0;
    sparseRet = aclsparseSpMMGetBufferSize(res.handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, res.matA, res.matB, &beta, res.matC,
        ACL_INT32, ACL_SPARSE_SPMM_CSR_ALG1, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    t1 = Clock::now();
    msGetBuf = ElapsedMs(t0, t1);
    std::printf("Workspace bytes: %zu\n", bufferSize);

    aclRet = aclrtMalloc(&res.dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());

    t0 = Clock::now();
    sparseRet = aclsparseSpMMPreprocess(res.handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, res.matA, res.matB, &beta, res.matC,
        ACL_INT32, ACL_SPARSE_SPMM_CSR_ALG1, res.dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    t1 = Clock::now();
    msPreprocess = ElapsedMs(t0, t1);

    t0 = Clock::now();
    sparseRet = aclsparseSpMM(res.handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, res.matA, res.matB, &beta, res.matC,
        ACL_INT32, ACL_SPARSE_SPMM_CSR_ALG1, res.dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclsparseSpMM: aclrtSynchronizeStream failed, ret=%d\n", aclRet);
              return FailVerify());
    t1 = Clock::now();
    msSpmm = ElapsedMs(t0, t1);

    t0 = Clock::now();
    aclRet = aclrtMemcpy(hCInt32.data(), sizeof(int32_t) * cSpec.elementCount, dC,
                         sizeof(int32_t) * cSpec.elementCount, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msD2H = ElapsedMs(t0, t1);

    t0 = Clock::now();
    VerifyResult vr = VerifyInt32(hCInt32, hCRefInt32, m, n, ldc, cRowMajor);
    t1 = Clock::now();
    msVerify = ElapsedMs(t0, t1);

    PrintTiming("INT8", msDataGen, msCpuRef, msDevAlloc, msH2D, msGetBuf, msPreprocess, msSpmm, msD2H, msVerify);

    return vr;
}

// ============================================================================
// FP32 case runner + main
// ============================================================================
struct SpmmFp32CaseConfig {
    const char *name;
    int32_t m;
    int32_t k;
    int32_t n;
    int32_t nnzTarget; // 0 -> m * k / 10, negative -> empty CSR
    float alpha;
    float beta;
    aclsparseOperation_t opB;
    aclsparseOrder_t orderB;
    aclsparseOrder_t orderC;
    int32_t bLdPadding;
    int32_t cLdPadding;
    unsigned seed;
    bool usePreprocess = true;
    bool deviceScalars = false;
    int32_t indexBase = 0;
    int32_t repeatCount = 1;
    bool skewedRows = false;
    aclsparseSpMMAlg_t alg = ACL_SPARSE_SPMM_CSR_ALG1;
};

static bool RunOneFp32Case(int32_t deviceId, aclrtStream stream, const SpmmFp32CaseConfig &cfg)
{
    std::srand(cfg.seed);
    const int32_t nnzTarget = (cfg.nnzTarget < 0) ? 0 :
        ((cfg.nnzTarget > 0) ? cfg.nnzTarget : cfg.m * cfg.k / 10);
    std::vector<int32_t> hRowOff;
    std::vector<int32_t> hColInd;
    std::vector<float> hValsFp32;
    if (cfg.skewedRows) {
        GenerateSkewedCsr(cfg.m, cfg.k, nnzTarget, &hRowOff, &hColInd, &hValsFp32);
    } else {
        GenerateRandomCsr(cfg.m, cfg.k, nnzTarget, &hRowOff, &hColInd, &hValsFp32);
    }
    if (cfg.indexBase == 1) {
        for (auto &v : hRowOff) { ++v; }
        for (auto &v : hColInd) { ++v; }
    }
    const int32_t actualNnz = hRowOff.back() - cfg.indexBase;
    std::printf("\n--- FP32 case: %s (m=%d k=%d n=%d nnz=%d) ---\n",
                cfg.name, cfg.m, cfg.k, cfg.n, actualNnz);
    VerifyResult r = RunSpmmTestFp32(deviceId, stream, cfg.m, cfg.k, cfg.n, actualNnz,
        hRowOff, hColInd, hValsFp32, cfg.alpha, cfg.beta, cfg.opB, cfg.orderB, cfg.orderC,
        cfg.bLdPadding, cfg.cLdPadding, cfg.usePreprocess, cfg.deviceScalars,
        cfg.indexBase, cfg.repeatCount, cfg.alg);
    std::printf("  [%s] %s\n", cfg.name, r.pass ? "PASS" : "FAIL");
    return r.pass;
}

static void ExpectParameterStatus(const char *name, aclsparseStatus_t got,
                                  aclsparseStatus_t wanted, bool *pass)
{
    const bool ok = got == wanted;
    std::printf("[PARAM] %s: got=%d expected=%d %s\n",
                name, got, wanted, ok ? "PASS" : "FAIL");
    *pass = *pass && ok;
}

static void CheckBufferSizeParameters(aclsparseHandle_t handle,
                                      const float &scalar, size_t *size,
                                      bool *pass)
{
    ExpectParameterStatus("null_handle",
           aclsparseSpMMGetBufferSize(nullptr, ACL_SPARSE_OP_NON_TRANSPOSE,
                                       ACL_SPARSE_OP_NON_TRANSPOSE,
                                       &scalar, nullptr, nullptr, &scalar,
                                       nullptr, ACL_FLOAT,
                                       ACL_SPARSE_SPMM_CSR_ALG1, size),
           ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR, pass);
    ExpectParameterStatus("null_size",
           aclsparseSpMMGetBufferSize(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
                                      ACL_SPARSE_OP_NON_TRANSPOSE,
                                      &scalar, nullptr, nullptr, &scalar,
                                      nullptr, ACL_FLOAT,
                                      ACL_SPARSE_SPMM_CSR_ALG1, nullptr),
           ACL_SPARSE_STATUS_INVALID_VALUE, pass);
    ExpectParameterStatus("null_alpha",
           aclsparseSpMMGetBufferSize(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
                                      ACL_SPARSE_OP_NON_TRANSPOSE,
                                      nullptr, nullptr, nullptr, &scalar,
                                      nullptr, ACL_FLOAT,
                                      ACL_SPARSE_SPMM_CSR_ALG1, size),
           ACL_SPARSE_STATUS_INVALID_VALUE, pass);
    ExpectParameterStatus("null_beta",
           aclsparseSpMMGetBufferSize(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
                                      ACL_SPARSE_OP_NON_TRANSPOSE,
                                      &scalar, nullptr, nullptr, nullptr,
                                      nullptr, ACL_FLOAT,
                                      ACL_SPARSE_SPMM_CSR_ALG1, size),
           ACL_SPARSE_STATUS_INVALID_VALUE, pass);
}

static void CheckExecutionParameters(aclsparseHandle_t handle,
                                     const float &scalar, bool *pass)
{
    ExpectParameterStatus("preprocess_null_alpha",
           aclsparseSpMMPreprocess(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
                                   ACL_SPARSE_OP_NON_TRANSPOSE, nullptr,
                                   nullptr, nullptr, &scalar, nullptr,
                                   ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1,
                                   nullptr),
           ACL_SPARSE_STATUS_INVALID_VALUE, pass);
    ExpectParameterStatus("spmm_null_beta",
           aclsparseSpMM(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
                         ACL_SPARSE_OP_NON_TRANSPOSE, &scalar, nullptr,
                         nullptr, nullptr, nullptr, ACL_FLOAT,
                         ACL_SPARSE_SPMM_CSR_ALG1, nullptr),
           ACL_SPARSE_STATUS_INVALID_VALUE, pass);
}

static bool RunSpmmParameterValidationTests()
{
    aclsparseHandle_t handle = nullptr;
    if (aclsparseCreate(&handle) != ACL_SPARSE_STATUS_SUCCESS) {
        std::printf("[PARAM] aclsparseCreate failed FAIL\n");
        return false;
    }
    const float scalar = 1.0f;
    size_t size = 0;
    bool pass = true;
    CheckBufferSizeParameters(handle, scalar, &size, &pass);
    CheckExecutionParameters(handle, scalar, &pass);
    aclsparseDestroy(handle);
    std::printf("[PARAM] Overall: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// Zero dimensions are rejected by Host validation before launching a kernel.
static bool RunSpmmZeroDimCase(aclrtStream stream, const char *name,
                               int32_t m, int32_t k, int32_t n)
{
    std::printf("\n--- zero-dim case: %s (m=%d k=%d n=%d) ---\n", name, m, k, n);
    SpmmTestResources res;

    // CSR A: m×k, nnz=0；rowOffsets 非空（m=0 时 1 元素 [0]，否则 m+1 个 0）
    std::vector<int32_t> hRowOff(static_cast<size_t>(m) + 1, 0);
    aclError aclRet = aclrtMalloc(&res.dRowOff, sizeof(int32_t) * (m + 1), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return false);
    aclRet = aclrtMemcpy(res.dRowOff, sizeof(int32_t) * (m + 1), hRowOff.data(),
                         sizeof(int32_t) * (m + 1), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return false);

    // B: k×n ROW 序, C: m×n ROW 序；ld = (n>0 ? n : 1)
    // values 必须 ≥1 元素非空（0 维时描述符创建层仍要求非空指针）
    const int32_t ld = (n > 0) ? n : 1;
    const size_t bElems = static_cast<size_t>(k > 0 ? k : 1) * static_cast<size_t>(ld);
    const size_t cElems = static_cast<size_t>(m > 0 ? m : 1) * static_cast<size_t>(ld);
    aclRet = aclrtMalloc(&res.dB, sizeof(float) * bElems, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return false);
    aclRet = aclrtMalloc(&res.dC, sizeof(float) * cElems, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return false);

    aclsparseStatus_t sparseRet = aclsparseCreate(&res.handle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet); return false);
    sparseRet = aclsparseSetStream(res.handle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet); return false);

    sparseRet = aclsparseCreateCsr(&res.matA, m, k, 0, res.dRowOff, nullptr, nullptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet); return false);
    sparseRet = aclsparseCreateDnMat(&res.matB, k, n, ld, res.dB, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateDnMat(B) failed. ERROR: %d\n", sparseRet); return false);
    sparseRet = aclsparseCreateDnMat(&res.matC, m, n, ld, res.dC, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateDnMat(C) failed. ERROR: %d\n", sparseRet); return false);

    // dtype combo 全 FP32 合法、维度匹配（0==0 放行），INVALID_VALUE 只能来自零维守卫
    const float alpha = 1.0f;
    const float beta = 0.0f;
    size_t bufferSize = 0;
    sparseRet = aclsparseSpMMGetBufferSize(res.handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, res.matA, res.matB, &beta, res.matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1, &bufferSize);
    const bool pass = (sparseRet == ACL_SPARSE_STATUS_INVALID_VALUE);
    std::printf("  [%s] GetBufferSize ret=%d (expect %d INVALID_VALUE): %s\n",
                name, static_cast<int>(sparseRet),
                static_cast<int>(ACL_SPARSE_STATUS_INVALID_VALUE),
                pass ? "PASS" : "FAIL");
    return pass;
}

static const SpmmFp32CaseConfig kFp32Cases[] = {
    {"N_Brow_Crow", 37, 23, 11, 170, 1.0f, 0.0f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 0, 0, 42u},
    {"N_Brow_Ccol", 37, 23, 11, 170, 1.0f, 0.0f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL, 0, 0, 43u},
    {"N_Bcol_Crow", 37, 23, 11, 170, 1.0f, 0.0f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_COL, ACL_SPARSE_ORDER_ROW, 0, 0, 44u},
    {"N_Bcol_Ccol", 37, 23, 11, 170, 1.0f, 0.0f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_COL, ACL_SPARSE_ORDER_COL, 0, 0, 45u},
    {"T_Brow_Crow", 37, 23, 11, 170, 1.0f, 0.0f,
        ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 0, 0, 46u},
    {"T_Brow_Ccol", 37, 23, 11, 170, 1.0f, 0.0f,
        ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL, 0, 0, 47u},
    {"T_Bcol_Crow", 37, 23, 11, 170, 1.0f, 0.0f,
        ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_ORDER_COL, ACL_SPARSE_ORDER_ROW, 0, 0, 48u},
    {"T_Bcol_Ccol", 37, 23, 11, 170, 1.0f, 0.0f,
        ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_ORDER_COL, ACL_SPARSE_ORDER_COL, 0, 0, 49u},
    {"T_padded_ld_beta", 41, 29, 13, 205, -0.75f, 0.5f,
        ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_ORDER_COL, ACL_SPARSE_ORDER_ROW, 3, 5, 50u},
    {"small_non_square", 7, 5, 3, 14, 1.0f, 0.0f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 0, 0, 51u},
    {"legacy_baseline_row_N_beta0", 256, 256, 64, 0, 1.0f, 0.0f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 0, 0, 42u},
    {"legacy_opB_transpose", 256, 256, 64, 0, 1.0f, 0.0f,
        ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 0, 0, 43u},
    {"legacy_B_C_col_major", 256, 256, 64, 0, 1.0f, 0.0f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_COL, ACL_SPARSE_ORDER_COL, 0, 0, 44u},
    {"legacy_beta_nonzero", 256, 256, 64, 0, 1.0f, 0.5f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 0, 0, 45u},
    {"legacy_small_shape", 32, 32, 16, 80, 1.0f, 0.0f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 0, 0, 46u},
    {"device_scalars_without_preprocess", 19, 13, 7, 61, -1.25f, 0.75f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_COL, ACL_SPARSE_ORDER_ROW, 2, 3, 52u,
        false, true},
    {"one_based_csr", 17, 11, 5, 44, 0.5f, -0.25f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL, 0, 0, 53u,
        true, false, 1},
    {"alpha_zero_beta_zero_empty_csr", 3, 4, 2, -1, 0.0f, 0.0f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 0, 0, 54u},
    {"alpha_zero_beta_one", 9, 7, 3, 18, 0.0f, 1.0f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_COL, ACL_SPARSE_ORDER_COL, 1, 2, 55u},
    {"empty_csr_beta", 5, 6, 4, -1, 0.0f, 0.5f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_COL, ACL_SPARSE_ORDER_ROW, 0, 0, 56u,
        false, false, 0, 2},
    {"preprocess_workspace_reuse", 23, 17, 9, 82, 0.75f, -0.25f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL, 2, 3, 57u,
        true, false, 0, 3},
    {"unit_shape", 1, 1, 1, 1, 1.0f, 0.5f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 0, 0, 58u},
    {"skewed_row_lengths", 16, 64, 7, 79, -0.5f, 0.25f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL, 1, 2, 59u,
        true, false, 0, 1, true},
    {"csr_alg2_row_major", 31, 37, 16, 173, 1.0f, 1.0f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 0, 0, 60u,
        true, false, 0, 2, false, ACL_SPARSE_SPMM_CSR_ALG2},
    {"csr_alg3_deterministic", 31, 37, 16, 173, 1.0f, 0.0f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 0, 0, 61u,
        true, false, 0, 3, false, ACL_SPARSE_SPMM_CSR_ALG3},
};

static bool RunBaselineTypeCases(int32_t deviceId, aclrtStream stream)
{
    // One functional case per remaining dtype (FP16 / INT8), baseline layout
    constexpr int32_t kM = 256;
    constexpr int32_t kK = 256;
    constexpr int32_t kN = 64;
    constexpr int32_t kNnz = 256 * 256 / 10;
    const aclsparseOperation_t opB = ACL_SPARSE_OP_NON_TRANSPOSE;
    const aclsparseOrder_t orderB = ACL_SPARSE_ORDER_ROW;
    const aclsparseOrder_t orderC = ACL_SPARSE_ORDER_ROW;

    std::printf("\n========== SPMM FP16 / INT8 (baseline per dtype) ==========\n");
    std::srand(42);
    std::vector<int32_t> hRowOff;
    std::vector<int32_t> hColInd;
    std::vector<float> hValsFp32;
    GenerateRandomCsr(kM, kK, kNnz, &hRowOff, &hColInd, &hValsFp32);
    const int32_t actualNnz = hRowOff.back();
    std::printf("Generated CSR: m=%d, k=%d, nnz=%d, density=%.4f\n",
                kM, kK, actualNnz,
                static_cast<float>(actualNnz) / (static_cast<float>(kM) * kK));

    VerifyResult rFp16 = RunSpmmTestFp16(deviceId, stream, kM, kK, kN, actualNnz,
        hRowOff, hColInd, hValsFp32, 1.0f, 0.0f, opB, orderB, orderC);
    VerifyResult rInt8 = RunSpmmTestInt8(deviceId, stream, kM, kK, kN, actualNnz,
        hRowOff, hColInd, 1, 0, opB, orderB, orderC);

    // Exact scalar and saturation boundaries for the INT8 -> INT32 path.
    std::vector<int32_t> hSmallRowOff{0, 1};
    std::vector<int32_t> hSmallColInd{0};
    VerifyResult rInt8LargeScalar = RunSpmmTestInt8(
        deviceId, stream, 1, 1, 1, 1, hSmallRowOff, hSmallColInd,
        16777217, 0, opB, orderB, orderC, 1);

    std::srand(58);
    std::vector<int32_t> hSaturateRowOff;
    std::vector<int32_t> hSaturateColInd;
    std::vector<float> hUnusedVals;
    GenerateRandomCsr(1, 256, 256, &hSaturateRowOff, &hSaturateColInd, &hUnusedVals);
    VerifyResult rInt8Saturation = RunSpmmTestInt8(
        deviceId, stream, 1, 256, 1, 256, hSaturateRowOff, hSaturateColInd,
        2147483647, 0, opB, orderB, orderC, 127);

    return rFp16.pass && rInt8.pass && rInt8LargeScalar.pass && rInt8Saturation.pass;
}

namespace {

class SpmmEnvironment : public ::testing::Environment {
public:
    void SetUp() override
    {
        ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
        ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    }

    void TearDown() override
    {
        EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
        EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
    }
};

::testing::Environment *const kSpmmEnvironment =
    ::testing::AddGlobalTestEnvironment(new SpmmEnvironment);

} // namespace

using spmm_test::SpmmDeviceTest;

class Fp32SpmmTest : public SpmmDeviceTest,
                     public ::testing::WithParamInterface<SpmmFp32CaseConfig> {};

TEST_P(Fp32SpmmTest, MatchesGolden)
{
    ASSERT_TRUE(RunOneFp32Case(0, stream_, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(LayoutsAndScalars, Fp32SpmmTest, ::testing::ValuesIn(kFp32Cases));

TEST_F(SpmmDeviceTest, ParameterValidation)
{
    ASSERT_TRUE(RunSpmmParameterValidationTests());
}

TEST_F(SpmmDeviceTest, NativeZeroDimensions)
{
    EXPECT_TRUE(RunSpmmZeroDimCase(stream_, "zero_dim_m0", 0, 16, 8));
    EXPECT_TRUE(RunSpmmZeroDimCase(stream_, "zero_dim_k0", 8, 0, 8));
    EXPECT_TRUE(RunSpmmZeroDimCase(stream_, "zero_dim_n0", 8, 16, 0));
}

TEST_F(SpmmDeviceTest, BaselineFp16AndInt8)
{
    ASSERT_TRUE(RunBaselineTypeCases(0, stream_));
}
