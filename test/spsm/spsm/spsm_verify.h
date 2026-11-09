/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TEST_SPSM_VERIFY_H_
#define TEST_SPSM_VERIFY_H_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace sparse_test
{
// Keep the CPU Golden in FP64 all the way through the comparison. Call once
// for each component of complex64 so a passing real part cannot hide failures
// in the imaginary part. ULP is measured in the target FP32 format.
inline bool VerifySpsmComponent(
    const std::vector<float>& actual, const std::vector<double>& golden, const std::string& name)
{
    if (actual.size() != golden.size() || actual.empty())
    {
        return false;
    }
    size_t matched = 0, limitFailures = 0, specialFailures = 0;
    double maxError = 0;
    for (size_t i = 0; i < actual.size(); ++i)
    {
        double x = actual[i], y = golden[i];
        if (!std::isfinite(x) || !std::isfinite(y))
        {
            bool same = (std::isnan(x) && std::isnan(y))
                || (std::isinf(x) && std::isinf(y) && std::signbit(x) == std::signbit(y));
            matched += same;
            specialFailures += !same;
            continue;
        }
        double error = std::abs(x - y);
        maxError = std::max(maxError, error);
        // mixed_tolerance_standard.md 2026-09-17 修订版：FLOAT32 rtol = atol = 2^-13
        matched += error <= std::ldexp(1.0, -13) + std::ldexp(1.0, -13) * std::abs(y);
        int exponent = -125;
        if (y != 0)
        {
            std::frexp(std::abs(y), &exponent);
        }
        double ulp = std::ldexp(1.0, std::max(exponent - 1, -126) - 23);
        limitFailures += error > std::max(0.01, 32 * ulp);
    }
    bool pass = static_cast<double>(matched) / actual.size() >= .99 && limitFailures == 0 && specialFailures == 0;
    std::cout << '[' << name << "] matched=" << matched << '/' << actual.size() << " limit_failures=" << limitFailures
              << " special_failures=" << specialFailures << " max_abs=" << maxError << " passed=" << pass << '\n';
    return pass;
}
} // namespace sparse_test
#endif
