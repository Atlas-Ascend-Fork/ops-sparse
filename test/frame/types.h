/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FRAME_TYPES_H_
#define TEST_FRAME_TYPES_H_

#include <cstdint>

namespace sparse_test {

enum class PrecisionMode {
    ABS,
    REL,
    COMBINED,
    MERE_MARE,
    EXACT,
    INTEGER,
    MIXED_TOLERANCE,
    MIXED_TOLERANCE_ANCHOR, // FLOAT64 reference; cap at the maximum finite-error position.
};

struct VerifyConfig {
    PrecisionMode mode = PrecisionMode::MERE_MARE;
    double absTol = 1e-5;
    double relTol = 1e-5;
    double mereThreshold = 0.0001220703125;
    double mareMultiplier = 10.0;
    double epsilonForRel = 1e-7;

    VerifyConfig() = default;

    VerifyConfig& SetMode(PrecisionMode m) {
        mode = m;
        return *this;
    }
    VerifyConfig& SetAbsTol(double t) {
        absTol = t;
        return *this;
    }
    VerifyConfig& SetRelTol(double t) {
        relTol = t;
        return *this;
    }
    VerifyConfig& SetMERE(double t) {
        mereThreshold = t;
        return *this;
    }
    VerifyConfig& SetMARE(double m) {
        mareMultiplier = m;
        return *this;
    }
    VerifyConfig& SetEpsilon(double e) {
        epsilonForRel = e;
        return *this;
    }

    double mixedAtol = 1.52587890625e-5;
    double mixedRtol = 0.0009765625;
    double mixedRequiredMatchedRatio = 0.99;
    double mixedMaxAbsErrorLimit = 1e-2;
    int mixedMantissaBits = 23;
    int mixedEmin = -126;

    VerifyConfig& SetMixedTol(double a, double r) {
        mixedAtol = a;
        mixedRtol = r;
        return *this;
    }
    VerifyConfig& SetMixedRequiredRatio(double r) {
        mixedRequiredMatchedRatio = r;
        return *this;
    }
    VerifyConfig& SetMixedMaxAbsErrLim(double l) {
        mixedMaxAbsErrorLimit = l;
        return *this;
    }
};

struct CaseSummary {
    int totalCases = 0;
    int passCases = 0;
    int failCases = 0;
};

}

#endif
