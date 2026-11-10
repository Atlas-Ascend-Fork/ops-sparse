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
#ifndef TEST_FRAME_MIXED_ANCHOR_H_
#define TEST_FRAME_MIXED_ANCHOR_H_

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#include "types.h"

namespace sparse_test {
struct MixedAnchorResult {
    bool passed = false;
    size_t total = 0;
    size_t matched = 0;
    size_t finite = 0;
    size_t maxErrorIndex = 0;
    double matchedRatio = 0;
    double maxAbsError = 0;
    double anchorGolden = 0;
    float anchorLow = 0;
    double anchorUlp = 0;
    double absoluteLimit = 0.01;
    std::string reason;
};

inline bool AccumulateComponent(const std::vector<float> &actual,
    const std::vector<double> &golden, MixedAnchorResult &result, const VerifyConfig &cfg)
{
    for (size_t i = 0; i < golden.size(); ++i) {
        const float a = actual[i];
        const double g = golden[i];
        if (std::isnan(a) || std::isnan(g)) {
            result.reason = "NaN belongs to the separate special-value suite";
            return false;
        }
        if (std::isfinite(a) && std::isfinite(g)) {
            const double error = std::abs(static_cast<double>(a) - g);
            if (error <= cfg.mixedAtol + cfg.mixedRtol * std::abs(g)) { ++result.matched; }
            // Strict > retains the first maximum, as the official argmax does.
            if (!result.finite || error > result.maxAbsError) {
                result.maxAbsError = error;
                result.maxErrorIndex = i;
                result.anchorGolden = g;
            }
            ++result.finite;
        } else {
            const float narrowed = static_cast<float>(g);
            if (std::isinf(a) && a == narrowed) {
                ++result.matched;
            } else {
                result.reason = "Inf position/sign mismatch after FLOAT32 RNE narrowing";
                return false;
            }
        }
    }
    return true;
}

inline bool CheckAnchorCap(MixedAnchorResult &result, const VerifyConfig &cfg)
{
    if (result.finite) {
        result.anchorLow = static_cast<float>(result.anchorGolden);
        if (!std::isfinite(result.anchorLow)) {
            result.reason = "finite DUT disagrees with overflowing FLOAT32 Golden at max error";
            return false;
        }
        // Do not take abs(anchorLow) before nextafter: a negative power of two
        // has a different interval towards +infinity from the positive value.
        result.anchorUlp = std::abs(static_cast<double>(std::nextafter(
            result.anchorLow, std::numeric_limits<float>::infinity())) - result.anchorLow);
        result.absoluteLimit = std::max(cfg.mixedMaxAbsErrorLimit, 32 * result.anchorUlp);
        return result.maxAbsError <= result.absoluteLimit;
    }
    return true;
}

// FLOAT32 output with an unrounded FLOAT64 reference. The cap is checked
// at the first maximum finite absolute error, using signed FLOAT32 nextafter.
inline MixedAnchorResult EvaluateMixedAnchor(const std::vector<float> &actual,
    const std::vector<double> &golden, const VerifyConfig &cfg)
{
    static_assert(std::numeric_limits<float>::is_iec559 &&
        std::numeric_limits<float>::digits == 24, "Cgtsv2 requires IEEE binary32");
    MixedAnchorResult result;
    result.total = golden.size();
    if (cfg.mode != PrecisionMode::MIXED_TOLERANCE_ANCHOR ||
        cfg.mixedRequiredMatchedRatio < 0 || cfg.mixedRequiredMatchedRatio > 1 ||
        !std::isfinite(cfg.mixedRequiredMatchedRatio) ||
        !std::isfinite(cfg.mixedAtol) || cfg.mixedAtol < 0 ||
        !std::isfinite(cfg.mixedRtol) || cfg.mixedRtol < 0 ||
        !std::isfinite(cfg.mixedMaxAbsErrorLimit) || cfg.mixedMaxAbsErrorLimit < 0) {
        result.reason = "invalid mixed-anchor configuration";
        return result;
    }
    if (actual.size() != golden.size()) {
        result.reason = "component shape mismatch";
        return result;
    }
    if (golden.empty()) {
        result.passed = true;
        result.matchedRatio = 1;
        result.reason = "legal empty RHS";
        return result;
    }
    // Conversions below use the actual host's round-to-nearest-even mode.
    // A changed floating environment is an invalid test environment, not a PASS.
    if (std::fegetround() != FE_TONEAREST) {
        result.reason = "FLOAT32 narrowing requires FE_TONEAREST (RNE)";
        return result;
    }
    if (!AccumulateComponent(actual, golden, result, cfg)) { return result; }
    result.matchedRatio = static_cast<double>(result.matched) / result.total;
    const bool capPass = CheckAnchorCap(result, cfg);
    if (!result.reason.empty()) { return result; }
    result.passed = result.matchedRatio >= cfg.mixedRequiredMatchedRatio && capPass;
    result.reason = result.matchedRatio < cfg.mixedRequiredMatchedRatio ? "mixed matched-ratio failure" :
        (!capPass ? "maximum-finite-error anchor cap failure" : "mixed tolerance and anchor cap satisfied");
    return result;
}
} // namespace sparse_test
#endif
