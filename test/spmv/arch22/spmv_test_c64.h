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

// complex64 测试辅助：由 spmv_test.cpp 在公共工具（容差参数、SpmvRunEnv 等）
// 定义之后包含，依赖其 include 与 Complex64 类型定义。
#ifndef SPMV_TEST_C64_H
#define SPMV_TEST_C64_H

// CPU golden 用 double 复数（complex128）计算，最后转回 float 比较
struct DComplex {
    double re;
    double im;
};

inline DComplex DAdd(DComplex a, DComplex b) { return {a.re + b.re, a.im + b.im}; }
inline DComplex DMul(DComplex a, DComplex b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
inline DComplex ToDC(Complex64 c) { return {static_cast<double>(c.re), static_cast<double>(c.im)}; }
inline DComplex ToDC(aclsparseComplex c) { return {static_cast<double>(c.x), static_cast<double>(c.y)}; }
inline DComplex ToDC(float f) { return {static_cast<double>(f), 0.0}; }

/* complex64 精度验证：实部/虚部分别按 FP32 混合容差 */
int32_t VerificationC64(const std::vector<DComplex> &cpuGolden,
                        const std::vector<Complex64> &npuRet,
                        float &MARE, float &MERE,
                        std::string *worstInfo = nullptr) {
    if (npuRet.size() != cpuGolden.size()) {
        std::cout << "[ERROR] The size of npuRet and cpuGolden is not equal!\n";
        return 1;
    }
    const MixedToleranceParams params = GetMixedToleranceParams<float>();

    int32_t status = 0;
    std::cout << "Verification (complex64, per-component FP32 tolerance)...\n";
    for (int i = 0; i < std::min(static_cast<int64_t>(npuRet.size()), static_cast<int64_t>(5)); ++i) {
        std::cout << "golden[" << i << "]=(" << cpuGolden[i].re << "," << cpuGolden[i].im << ")"
                  << " npu_result[" << i << "]=(" << npuRet[i].re << "," << npuRet[i].im << ")\n";
    }

    size_t passCount = 0;
    size_t worstIdx = 0;
    float maxAbsError = 0.0f;
    float worstMargin = 0.0f;

    for (size_t i = 0; i < npuRet.size(); ++i) {
        const float npuRe = npuRet[i].re, npuIm = npuRet[i].im;
        const float cpuRe = static_cast<float>(cpuGolden[i].re);
        const float cpuIm = static_cast<float>(cpuGolden[i].im);
        const float errRe = std::fabs(npuRe - cpuRe);
        const float errIm = std::fabs(npuIm - cpuIm);
        maxAbsError = std::max(maxAbsError, std::max(errRe, errIm));
        const float tolRe = params.atol + params.rtol * std::fabs(cpuRe);
        const float tolIm = params.atol + params.rtol * std::fabs(cpuIm);
        if (errRe <= tolRe && errIm <= tolIm) {
            passCount++;
        } else {
            const float margin = std::max(errRe - tolRe, errIm - tolIm);
            if (margin > worstMargin) {
                worstMargin = margin;
                worstIdx = i;
            }
        }
    }

    const float matchedRatio = npuRet.empty() ? 1.0f
                                              : static_cast<float>(passCount) / static_cast<float>(npuRet.size());
    MARE = maxAbsError;
    MERE = matchedRatio;

    status = CheckMatchedRatioAndAbsError(matchedRatio, maxAbsError, params);

    if (worstInfo) {
        std::ostringstream oss;
        oss << "worst[" << worstIdx << "] margin=" << worstMargin;
        *worstInfo = oss.str();
    }
    return status;
}

/* complex64 用例输入生成：csrVal/x 按 ValT（Complex64 或 float）、y 恒为 Complex64，
 * RNG 抽取次数与顺序与原内联实现一致 */
template <typename ValT>
void GenerateC64Inputs(std::uniform_real_distribution<float> &valDist, std::mt19937 &rng,
                       size_t nnz, uint32_t xSize, uint32_t ySize,
                       std::vector<ValT> &csrVal, std::vector<ValT> &xVec,
                       std::vector<Complex64> &yVec) {
    csrVal.assign(nnz, ValT{});
    for (size_t i = 0; i < nnz; ++i) {
        if constexpr (std::is_same_v<ValT, Complex64>) {
            csrVal[i] = Complex64{valDist(rng), valDist(rng)};
        } else {
            csrVal[i] = valDist(rng);
        }
    }
    xVec.assign(xSize, ValT{});
    for (auto &v : xVec) {
        if constexpr (std::is_same_v<ValT, Complex64>) {
            v = Complex64{valDist(rng), valDist(rng)};
        } else {
            v = valDist(rng);
        }
    }
    yVec.assign(ySize, Complex64{});
    for (auto &v : yVec)
        v = Complex64{valDist(rng), valDist(rng)};
}

/* CPU golden（complex128）：非转置按行累加，转置按列散布累加 */
template <typename ValT>
std::vector<DComplex> GoldenC64(const std::vector<int32_t> &csrRowPtr,
                                const std::vector<int32_t> &csrColInd,
                                const std::vector<ValT> &csrVal,
                                const std::vector<ValT> &xVec,
                                const std::vector<Complex64> &yVec,
                                size_t M, size_t N, bool transpose,
                                DComplex alpha, DComplex beta) {
    std::vector<DComplex> golden(transpose ? N : M, {0.0, 0.0});
    if (!transpose) {
        for (size_t i = 0; i < M; ++i) {
            DComplex sum{0.0, 0.0};
            for (int32_t j = csrRowPtr[i]; j < csrRowPtr[i + 1]; ++j)
                sum = DAdd(sum, DMul(ToDC(csrVal[j]), ToDC(xVec[csrColInd[j]])));
            golden[i] = DAdd(DMul(alpha, sum), DMul(beta, ToDC(yVec[i])));
        }
    } else {
        for (size_t j = 0; j < N; ++j)
            golden[j] = DMul(beta, ToDC(yVec[j]));
        for (size_t i = 0; i < M; ++i) {
            const DComplex contrib = DMul(alpha, ToDC(xVec[i]));
            for (int32_t k = csrRowPtr[i]; k < csrRowPtr[i + 1]; ++k)
                golden[csrColInd[k]] = DAdd(golden[csrColInd[k]], DMul(contrib, ToDC(csrVal[k])));
        }
    }
    return golden;
}

/* complex64 用例输入与 golden 准备：CSR 结构（preset 或随机生成）、输入向量、
 * complex128 golden 一次备好，RNG 抽取次数与顺序与原内联实现一致 */
template <typename ValT>
void PrepareC64Case(size_t M, size_t N, float sparsity, bool transpose,
                    const std::vector<int32_t> *presetRowPtr,
                    const std::vector<int32_t> *presetColInd,
                    float valRange,
                    aclsparseComplex alphaVal, aclsparseComplex betaVal,
                    std::vector<int32_t> &csrRowPtr, std::vector<int32_t> &csrColInd,
                    std::vector<ValT> &csrVal, std::vector<ValT> &xVec,
                    std::vector<Complex64> &yVec, std::vector<DComplex> &golden) {
    std::mt19937 rng(static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    // 长尾行等超长累加场景用较小值域，避免 fp32 累加的绝对误差超过 maxAbsError 上限
    std::uniform_real_distribution<float> valDist(-valRange, valRange);

    if (presetRowPtr != nullptr) {
        csrRowPtr = *presetRowPtr;
        csrColInd = *presetColInd;
    } else {
        std::vector<float> dummyVals;
        GenerateCsr<float>(M, N, sparsity, csrRowPtr, csrColInd, dummyVals);
    }
    const size_t nnz = csrColInd.size();

    uint32_t xSize = transpose ? static_cast<uint32_t>(M) : static_cast<uint32_t>(N);
    uint32_t ySize = transpose ? static_cast<uint32_t>(N) : static_cast<uint32_t>(M);

    GenerateC64Inputs<ValT>(valDist, rng, nnz, xSize, ySize, csrVal, xVec, yVec);

    // ===================== CPU golden（complex128） =====================
    golden = GoldenC64<ValT>(csrRowPtr, csrColInd, csrVal, xVec, yVec,
                             M, N, transpose, ToDC(alphaVal), ToDC(betaVal));
}

/* 读回 y 并做 complex64 精度校验；成功路径销毁描述符与 handle */
int ReadbackAndVerifyC64(SpmvRunEnv &env, aclrtStream stream,
                         const std::vector<DComplex> &golden,
                         float &MARE, float &MERE, std::string *worstInfo) {
    int aclRet = aclrtMemcpy(env.yHost, env.totalYByteSize, env.yDevice, env.totalYByteSize,
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::vector<Complex64> yResult;
    yResult.assign((Complex64 *)env.yHost, (Complex64 *)(env.yHost + env.totalYByteSize));

    int verifyRet = VerificationC64(golden, yResult, MARE, MERE, worstInfo);

    aclsparseDestroySpMat(env.matDesc);
    aclsparseDestroyDnVec(env.vecXDesc);
    aclsparseDestroyDnVec(env.vecYDesc);
    aclsparseDestroy(env.spHandle);

    return verifyRet;
}

#endif // SPMV_TEST_C64_H
