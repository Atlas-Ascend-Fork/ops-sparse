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

#ifndef SPMV_TEST_VERIFICATION_H
#define SPMV_TEST_VERIFICATION_H

template <typename ErrorT>
struct SpmvErrorLocation {
    size_t index = 0;
    ErrorT golden = 0;
    ErrorT actual = 0;
};

// Diagnostic outputs may be float; acceptance is always decided in ErrorT.
template <typename ErrorT>
struct SpmvVerificationState {
    size_t passCount = 0;
    ErrorT maxAbsError = 0;
    ErrorT worstMargin = 0;
    SpmvErrorLocation<ErrorT> maxAbsLocation;
    SpmvErrorLocation<ErrorT> mixedLocation;
};

template <typename T, typename GoldenT, typename ErrorT>
void AccumulateSpmvVerification(T actual, GoldenT golden, size_t index,
    const MixedToleranceParams &params, SpmvVerificationState<ErrorT> &state) {
    const ErrorT npuVal = static_cast<ErrorT>(actual);
    const ErrorT cpuVal = static_cast<ErrorT>(golden);
    const bool finite = std::isfinite(npuVal) && std::isfinite(cpuVal);
    const ErrorT error = finite ? std::fabs(npuVal - cpuVal) : std::numeric_limits<ErrorT>::infinity();
    const ErrorT tolerance = static_cast<ErrorT>(params.atol) +
        static_cast<ErrorT>(params.rtol) * std::fabs(cpuVal);
    bool matched = finite && error <= tolerance;
    if constexpr (std::is_same_v<T, int32_t>) {
        matched = actual == golden;
    }
    state.passCount += matched ? 1 : 0;
    const ErrorT margin = finite ? error - tolerance : error;
    if (error > state.maxAbsError) {
        state.maxAbsLocation = {index, cpuVal, npuVal};
        state.maxAbsError = error;
    }
    if (margin > state.worstMargin) {
        state.mixedLocation = {index, cpuVal, npuVal};
        state.worstMargin = margin;
    }
}

template <typename ErrorT>
std::string FormatSpmvVerification(const SpmvVerificationState<ErrorT> &state) {
    std::ostringstream oss;
    oss << "maxAbs[" << state.maxAbsLocation.index << "] golden=" << state.maxAbsLocation.golden
        << " npu=" << state.maxAbsLocation.actual << " error=" << state.maxAbsError
        << "; mixed[" << state.mixedLocation.index << "] golden=" << state.mixedLocation.golden
        << " npu=" << state.mixedLocation.actual << " margin=" << state.worstMargin;
    return oss.str();
}

template <typename T, typename GoldenT = T>
int32_t Verification(const std::vector<GoldenT> &cpuGolden, const std::vector<T> &npuRet,
    float &MARE, float &MERE, const MixedToleranceParams &params, std::string *worstInfo = nullptr) {
    if (npuRet.size() != cpuGolden.size()) {
        std::cout << "[ERROR] The size of npuRet and cpuGolden is not equal!\n";
        return 1;
    }
    using ErrorT = std::conditional_t<std::is_same_v<GoldenT, double> ||
        std::is_same_v<T, int32_t>, double, float>;
    SpmvVerificationState<ErrorT> state;
    std::cout << "Verification...\n";
    for (size_t i = 0; i < npuRet.size(); ++i) {
        if (i < 10) {
            std::cout << "golden[" << i << "]=" << static_cast<ErrorT>(cpuGolden[i])
                      << " npu_result[" << i << "]=" << static_cast<ErrorT>(npuRet[i]) << "\n";
        }
        AccumulateSpmvVerification(npuRet[i], cpuGolden[i], i, params, state);
    }
    // Preserve the existing float matched-ratio boundary (99/100 == 0.99f).
    const float ratio = npuRet.empty() ? 1.0f :
        static_cast<float>(state.passCount) / static_cast<float>(npuRet.size());
    int32_t status = 0;
    if constexpr (std::is_same_v<T, int32_t>) {
        status = ratio < params.requiredMatchedRatio ? 1 : 0;
    } else {
        status = CheckMatchedRatioAndAbsError(ratio, state.maxAbsError, params);
    }
    MARE = static_cast<float>(state.maxAbsError);
    MERE = static_cast<float>(ratio);
    if (worstInfo) {
        *worstInfo = FormatSpmvVerification(state);
    }
    return status;
}

template <typename T, typename GoldenT>
bool ExpectSpmvVerification(const std::vector<GoldenT> &golden, const std::vector<T> &actual,
    const MixedToleranceParams &params, bool expectedPass) {
    float maxError = 0;
    float ratio = 0;
    return (Verification<T>(golden, actual, maxError, ratio, params) == 0) == expectedPass;
}

bool TestSpmvMatchedRatio() {
    const auto params = GetMixedToleranceParams<float>();
    const std::vector<double> golden(100, 1.0);
    const std::vector<float> legacyGolden(100, 1.0f);
    std::vector<float> actual(100, 1.0f);
    actual[0] = 1.005f; // One mixed-tolerance miss, below the max-absolute limit.
    bool ok = ExpectSpmvVerification(golden, actual, params, true);
    ok &= ExpectSpmvVerification(legacyGolden, actual, params, true);
    // A single non-finite value must still fail with 99% finite matches.
    actual[0] = std::numeric_limits<float>::quiet_NaN();
    ok &= ExpectSpmvVerification(golden, actual, params, false);
    actual[0] = std::numeric_limits<float>::infinity();
    ok &= ExpectSpmvVerification(golden, actual, params, false);
    return ok;
}

bool TestSpmvVerificationDiagnostics() {
    std::vector<double> golden(100, 0.0);
    std::vector<float> actual(100, 0.0f);
    golden[0] = 1000.0;
    actual[0] = 1000.02f;
    actual[1] = 0.005f;
    float maxError = 0;
    float ratio = 0;
    std::string details;
    const int32_t status = Verification<float>(golden, actual, maxError, ratio,
        GetMixedToleranceParams<float>(), &details);
    std::cout << "SpMV diagnostic regression: " << details << "\n";
    return status == 1 && ratio == 0.99f && maxError > 0.01f &&
        details.find("maxAbs[0] golden=1000") != std::string::npos &&
        details.find("mixed[1] golden=0") != std::string::npos;
}

bool TestSpmvVerification() {
    const MixedToleranceParams exact{0, 0, 0, 1};
    const auto fp32 = GetMixedToleranceParams<float>();
    const double aboveLimit = static_cast<double>(fp32.maxAbsErrorLimit) + 1e-12;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    bool ok = TestSpmvGoldenPrecision() && TestSpmvMatchedRatio();
    ok &= TestSpmvVerificationDiagnostics();
    ok &= ExpectSpmvVerification<float, double>({1.0 + 2e-8}, {1.0f}, exact, false);
    // The per-element tolerance passes; only the unrounded maximum error fails.
    ok &= ExpectSpmvVerification<float, double>({aboveLimit}, {0.0f}, {0, 1, fp32.maxAbsErrorLimit, 1}, false);
    ok &= ExpectSpmvVerification<float, double>({1.0}, {1.0f}, fp32, true);
    ok &= ExpectSpmvVerification<float, double>({1.0}, {2.0f}, fp32, false);
    ok &= ExpectSpmvVerification<float, double>({1.0}, {nan}, fp32, false);
    ok &= ExpectSpmvVerification<float, double>({1.0}, {inf}, fp32, false);
    ok &= ExpectSpmvVerification<float, double>({std::numeric_limits<double>::infinity()}, {1.0f}, fp32, false);
    ok &= ExpectSpmvVerification<float, double>({1.0}, {}, fp32, false);
    ok &= ExpectSpmvVerification<int32_t, int32_t>({16777217}, {16777216}, exact, false);
    ok &= ExpectSpmvVerification<int32_t, int32_t>({16777217}, {16777217}, exact, true);
    std::cout << "SpMV host verification regression: " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

#endif // SPMV_TEST_VERIFICATION_H
