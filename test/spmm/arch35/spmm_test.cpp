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
//   C = alpha * A * B + beta * C
//
// Three test cases:
//   FP32: A(fp32) * B(fp32) -> C(fp32),  computeType=fp32
//   FP16: A(fp16) * B(fp16) -> C(fp16),  computeType=fp32  (accumulate in fp32)
//   INT8: A(int8) * B(int8) -> C(int32), computeType=int32 (accumulate in int32)

#include <acl/acl.h>
#include "cann_ops_sparse.h"

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

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

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

void SpmmCpuFp32(int32_t m, int32_t n,
                 const std::vector<int32_t> &rowOff,
                 const std::vector<int32_t> &colInd,
                 const std::vector<float>   &vals,
                 const std::vector<float>   &B,
                 int32_t ldb, bool bRowMajor, bool opBTranspose,
                 std::vector<float>         *C,
                 int32_t ldc, bool cRowMajor,
                 float alpha, float beta)
{
    for (int32_t r = 0; r < m; ++r) {
        for (int32_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            int32_t s = rowOff[r];
            int32_t e = rowOff[r + 1];
            for (int32_t p = s; p < e; ++p) {
                int32_t c = colInd[p];
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
                 int32_t alpha, int32_t beta)
{
    for (int32_t r = 0; r < m; ++r) {
        for (int32_t j = 0; j < n; ++j) {
            int32_t acc = 0;
            int32_t s = rowOff[r];
            int32_t e = rowOff[r + 1];
            for (int32_t p = s; p < e; ++p) {
                int32_t c = colInd[p];
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
                acc += static_cast<int32_t>(vals[p]) *
                       static_cast<int32_t>(B[static_cast<size_t>(bIdx)]);
            }
            int64_t cIdx = cRowMajor
                ? static_cast<int64_t>(r) * ldc + j
                : static_cast<int64_t>(j) * ldc + r;
            int32_t cOld = (*C)[static_cast<size_t>(cIdx)];
            (*C)[static_cast<size_t>(cIdx)] = alpha * acc + beta * cOld;
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
static VerifyResult RunSpmmTestFp32(int32_t deviceId, aclrtStream stream,
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
    const int32_t ldb = bRowMajor ? n : k;
    const int32_t ldc = cRowMajor ? n : m;

    const char *opBStr = opBTranspose ? "T" : "N";
    const char *obStr = bRowMajor ? "R" : "C";
    const char *ocStr = cRowMajor ? "R" : "C";
    std::printf("\n====== SPMM FP32 Test [opB=%s, B=%s, C=%s] ======\n", opBStr, obStr, ocStr);

    t0 = Clock::now();
    std::vector<float> hB(static_cast<size_t>(k) * ldb);
    for (size_t i = 0; i < hB.size(); ++i) {
        hB[i] = -5.0f + 10.0f * static_cast<float>(i % 10001) / 10000.0f;
    }
    std::vector<float> hC(static_cast<size_t>(m) * ldc, 0.0f);
    if (beta != 0.0f) {
        for (size_t i = 0; i < hC.size(); ++i) {
            hC[i] = -5.0f + 10.0f * static_cast<float>(i % 10001) / 10000.0f;
        }
    }
    std::vector<float> hCRef(hC);
    t1 = Clock::now();
    msDataGen = ElapsedMs(t0, t1);

    t0 = Clock::now();
    SpmmCpuFp32(m, n, hRowOff, hColInd, hValsFp32, hB, ldb, bRowMajor, opBTranspose, &hCRef, ldc, cRowMajor, alpha, beta);
    t1 = Clock::now();
    msCpuRef = ElapsedMs(t0, t1);

    t0 = Clock::now();
    SpmmTestResources res;
    int32_t *dRowOff = nullptr;
    int32_t *dColInd = nullptr;
    float   *dVals   = nullptr;
    float   *dB      = nullptr;
    float   *dC      = nullptr;
    aclError aclRet = aclrtMalloc((void **)&res.dRowOff, sizeof(int32_t) * (m + 1), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dRowOff = static_cast<int32_t *>(res.dRowOff);
    aclRet = aclrtMalloc((void **)&res.dColInd, sizeof(int32_t) * nnz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dColInd = static_cast<int32_t *>(res.dColInd);
    aclRet = aclrtMalloc((void **)&res.dVals, sizeof(float) * nnz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dVals = static_cast<float *>(res.dVals);
    aclRet = aclrtMalloc((void **)&res.dB, sizeof(float) * k * ldb, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dB = static_cast<float *>(res.dB);
    aclRet = aclrtMalloc((void **)&res.dC, sizeof(float) * m * ldc, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dC = static_cast<float *>(res.dC);
    t1 = Clock::now();
    msDevAlloc = ElapsedMs(t0, t1);

    t0 = Clock::now();
    aclRet = aclrtMemcpy(dRowOff, sizeof(int32_t) * (m + 1), hRowOff.data(), sizeof(int32_t) * (m + 1), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dColInd, sizeof(int32_t) * nnz, hColInd.data(), sizeof(int32_t) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dVals, sizeof(float) * nnz, hValsFp32.data(), sizeof(float) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dB, sizeof(float) * k * ldb, hB.data(), sizeof(float) * k * ldb, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dC, sizeof(float) * m * ldc, hC.data(), sizeof(float) * m * ldc, ACL_MEMCPY_HOST_TO_DEVICE);
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
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet);
              return FailVerify());

    sparseRet = aclsparseCreateDnMat(&res.matB, k, n, ldb, dB, ACL_FLOAT, orderB);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    sparseRet = aclsparseCreateDnMat(&res.matC, m, n, ldc, dC, ACL_FLOAT, orderC);
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
    aclRet = aclrtMemcpy(hC.data(), sizeof(float) * m * ldc, dC, sizeof(float) * m * ldc, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msD2H = ElapsedMs(t0, t1);

    t0 = Clock::now();
    VerifyResult vr = VerifyFloat(hC, hCRef, m, n, ldc, cRowMajor, "FP32",
                          1.0 / (1 << 13), 10.0 / (1 << 13));
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
    const int32_t ldb = bRowMajor ? n : k;
    const int32_t ldc = cRowMajor ? n : m;

    const char *opBStr = opBTranspose ? "T" : "N";
    const char *obStr = bRowMajor ? "R" : "C";
    const char *ocStr = cRowMajor ? "R" : "C";
    std::printf("\n====== SPMM FP16 Test [opB=%s, B=%s, C=%s] ======\n", opBStr, obStr, ocStr);

    t0 = Clock::now();
    std::vector<float> hBFp32(static_cast<size_t>(k) * ldb);
    for (size_t i = 0; i < hBFp32.size(); ++i) {
        hBFp32[i] = -5.0f + 10.0f * static_cast<float>(i % 10001) / 10000.0f;
    }
    std::vector<float> hCRefFp32(static_cast<size_t>(m) * ldc, 0.0f);
    if (beta != 0.0f) {
        for (size_t i = 0; i < hCRefFp32.size(); ++i) {
            hCRefFp32[i] = -5.0f + 10.0f * static_cast<float>(i % 10001) / 10000.0f;
        }
    }

    std::vector<uint16_t> hValsFp16(static_cast<size_t>(nnz));
    for (int32_t i = 0; i < nnz; ++i) {
        hValsFp16[i] = Fp32ToFp16Bits(hValsFp32[i]);
    }
    std::vector<uint16_t> hBFp16(static_cast<size_t>(k) * ldb);
    for (size_t i = 0; i < hBFp16.size(); ++i) {
        hBFp16[i] = Fp32ToFp16Bits(hBFp32[i]);
    }
    std::vector<uint16_t> hCFp16(static_cast<size_t>(m) * ldc);
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
    std::vector<float> hBFp16RoundTrip(static_cast<size_t>(k) * ldb);
    for (size_t i = 0; i < hBFp16RoundTrip.size(); ++i) {
        hBFp16RoundTrip[i] = Fp16BitsToFp32(hBFp16[i]);
    }
    std::vector<float> hCRefFp16RoundTrip(static_cast<size_t>(m) * ldc);
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
    aclRet = aclrtMalloc(&res.dB, sizeof(uint16_t) * k * ldb, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dB = res.dB;
    aclRet = aclrtMalloc(&res.dC, sizeof(uint16_t) * m * ldc, ACL_MEM_MALLOC_HUGE_FIRST);
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
    aclRet = aclrtMemcpy(dB, sizeof(uint16_t) * k * ldb, hBFp16.data(), sizeof(uint16_t) * k * ldb, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dC, sizeof(uint16_t) * m * ldc, hCFp16.data(), sizeof(uint16_t) * m * ldc, ACL_MEMCPY_HOST_TO_DEVICE);
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

    sparseRet = aclsparseCreateDnMat(&res.matB, k, n, ldb, dB, ACL_FLOAT16, orderB);
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
    aclRet = aclrtMemcpy(hCFp16.data(), sizeof(uint16_t) * m * ldc, dC, sizeof(uint16_t) * m * ldc, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msD2H = ElapsedMs(t0, t1);

    t0 = Clock::now();
    std::vector<float> hCFp32FromFp16(static_cast<size_t>(m) * ldc);
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
                            aclsparseOrder_t orderC)
{
    TimePoint t0, t1;
    double msDataGen = 0, msCpuRef = 0, msDevAlloc = 0, msH2D = 0;
    double msGetBuf = 0, msPreprocess = 0, msSpmm = 0, msD2H = 0, msVerify = 0;

    const bool bRowMajor = (orderB == ACL_SPARSE_ORDER_ROW);
    const bool cRowMajor = (orderC == ACL_SPARSE_ORDER_ROW);
    const bool opBTranspose = (opB == ACL_SPARSE_OP_TRANSPOSE);
    const int32_t ldb = bRowMajor ? n : k;
    const int32_t ldc = cRowMajor ? n : m;

    const char *opBStr = opBTranspose ? "T" : "N";
    const char *obStr = bRowMajor ? "R" : "C";
    const char *ocStr = cRowMajor ? "R" : "C";
    std::printf("\n====== SPMM INT8 Test [opB=%s, B=%s, C=%s] ======\n", opBStr, obStr, ocStr);

    t0 = Clock::now();
    std::vector<int8_t> hValsInt8(static_cast<size_t>(nnz));
    for (int32_t i = 0; i < nnz; ++i) {
        hValsInt8[i] = static_cast<int8_t>((i % 11) - 5);
    }
    std::vector<int8_t> hBInt8(static_cast<size_t>(k) * ldb);
    for (size_t i = 0; i < hBInt8.size(); ++i) {
        hBInt8[i] = static_cast<int8_t>((i % 11) - 5);
    }
    std::vector<int32_t> hCInt32(static_cast<size_t>(m) * ldc, 0);
    if (beta != 0) {
        for (size_t i = 0; i < hCInt32.size(); ++i) {
            hCInt32[i] = static_cast<int32_t>((i % 11) - 5);
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
    aclRet = aclrtMalloc((void **)&res.dColInd, sizeof(int32_t) * nnz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dColInd = static_cast<int32_t *>(res.dColInd);
    aclRet = aclrtMalloc((void **)&res.dVals, sizeof(int8_t) * nnz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dVals = static_cast<int8_t *>(res.dVals);
    aclRet = aclrtMalloc((void **)&res.dB, sizeof(int8_t) * k * ldb, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dB = static_cast<int8_t *>(res.dB);
    aclRet = aclrtMalloc((void **)&res.dC, sizeof(int32_t) * m * ldc, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    dC = static_cast<int32_t *>(res.dC);
    t1 = Clock::now();
    msDevAlloc = ElapsedMs(t0, t1);

    t0 = Clock::now();
    aclRet = aclrtMemcpy(dRowOff, sizeof(int32_t) * (m + 1), hRowOff.data(), sizeof(int32_t) * (m + 1), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dColInd, sizeof(int32_t) * nnz, hColInd.data(), sizeof(int32_t) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dVals, sizeof(int8_t) * nnz, hValsInt8.data(), sizeof(int8_t) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dB, sizeof(int8_t) * k * ldb, hBInt8.data(), sizeof(int8_t) * k * ldb, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dC, sizeof(int32_t) * m * ldc, hCInt32.data(), sizeof(int32_t) * m * ldc, ACL_MEMCPY_HOST_TO_DEVICE);
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

    sparseRet = aclsparseCreateDnMat(&res.matB, k, n, ldb, dB, ACL_INT8, orderB);
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
    aclRet = aclrtMemcpy(hCInt32.data(), sizeof(int32_t) * m * ldc, dC, sizeof(int32_t) * m * ldc, ACL_MEMCPY_DEVICE_TO_HOST);
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
    int32_t nnzTarget; // 0 -> m * k / 10
    float alpha;
    float beta;
    aclsparseOperation_t opB;
    aclsparseOrder_t orderB;
    aclsparseOrder_t orderC;
    unsigned seed;
};

static bool RunOneFp32Case(int32_t deviceId, aclrtStream stream, const SpmmFp32CaseConfig &cfg)
{
    std::srand(cfg.seed);
    const int32_t nnzTarget = (cfg.nnzTarget > 0) ? cfg.nnzTarget : cfg.m * cfg.k / 10;
    std::vector<int32_t> hRowOff;
    std::vector<int32_t> hColInd;
    std::vector<float> hValsFp32;
    GenerateRandomCsr(cfg.m, cfg.k, nnzTarget, &hRowOff, &hColInd, &hValsFp32);
    const int32_t actualNnz = hRowOff.back();
    std::printf("\n--- FP32 case: %s (m=%d k=%d n=%d nnz=%d) ---\n",
                cfg.name, cfg.m, cfg.k, cfg.n, actualNnz);
    VerifyResult r = RunSpmmTestFp32(deviceId, stream, cfg.m, cfg.k, cfg.n, actualNnz,
        hRowOff, hColInd, hValsFp32, cfg.alpha, cfg.beta, cfg.opB, cfg.orderB, cfg.orderC);
    std::printf("  [%s] %s\n", cfg.name, r.pass ? "PASS" : "FAIL");
    return r.pass;
}

// ============================================================================
// Zero-dim guard cases: m/k/n 任一为 0 时 ValidateSpmmInputs 应返回
// INVALID_VALUE（维度匹配对 0==0 放行，由零维守卫 matA->rows/cols、
// matC->cols 拒绝）。仅调 GetBufferSize，不执行 kernel。
// 维度关系: A m×k CSR, B k×n DnMat (ROW), C m×n DnMat (ROW)
//   AclsparseValidateSpmmCommon 维度匹配: A.cols==B.rows(k), A.rows==C.rows(m), B.cols==C.cols(n)
// 描述符层约束: aclsparseCreateDnMat 放行 0 维（ld>0、ROW 序 ld>=cols、
// values 非空）；aclsparseCreateCsr 放行 nnz=0（colInd/values 可 null，
// rowOffsets 非空以满足校验要求）。
// ============================================================================
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

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return EXIT_FAILURE);

    static const SpmmFp32CaseConfig kFp32Cases[] = {
        {"baseline_row_N_beta0", 256, 256, 64, 0, 1.0f, 0.0f,
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 42u},
        {"opB_transpose", 256, 256, 64, 0, 1.0f, 0.0f,
            ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 43u},
        {"B_C_col_major", 256, 256, 64, 0, 1.0f, 0.0f,
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_COL, ACL_SPARSE_ORDER_COL, 44u},
        {"beta_nonzero", 256, 256, 64, 0, 1.0f, 0.5f,
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 45u},
        {"small_shape", 32, 32, 16, 80, 1.0f, 0.0f,
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, 46u},
    };

    std::printf("\n========== SPMM FP32 cases (%zu configs) ==========\n", sizeof(kFp32Cases) / sizeof(kFp32Cases[0]));
    bool allFp32Pass = true;
    for (const auto &cfg : kFp32Cases) {
        allFp32Pass = RunOneFp32Case(deviceId, stream, cfg) && allFp32Pass;
    }

    std::printf("\n========== SPMM zero-dim guard cases ==========\n");
    bool allZeroDimPass = true;
    allZeroDimPass = RunSpmmZeroDimCase(stream, "zero_dim_m0", 0, 16, 8) && allZeroDimPass;
    allZeroDimPass = RunSpmmZeroDimCase(stream, "zero_dim_k0", 8, 0, 8) && allZeroDimPass;
    allZeroDimPass = RunSpmmZeroDimCase(stream, "zero_dim_n0", 8, 16, 0) && allZeroDimPass;
    std::printf("  ZeroDim: %s\n", allZeroDimPass ? "PASS" : "FAIL");

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

    std::printf("\n========== Results ==========\n");
    std::printf("  FP32 (%zu cases): %s\n", sizeof(kFp32Cases) / sizeof(kFp32Cases[0]),
                allFp32Pass ? "PASS" : "FAIL");
    std::printf("  ZeroDim (3 cases): %s\n", allZeroDimPass ? "PASS" : "FAIL");
    std::printf("  FP16: %s  INT8: %s\n",
                rFp16.pass ? "PASS" : "FAIL",
                rInt8.pass ? "PASS" : "FAIL");

    bool allPass = allFp32Pass && allZeroDimPass && rFp16.pass && rInt8.pass;
    std::printf("  Overall: %s\n", allPass ? "PASS" : "FAIL");

    Finalize(deviceId, stream);
    return allPass ? EXIT_SUCCESS : EXIT_FAILURE;
}
