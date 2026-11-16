/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gpsv_interleaved_batch_golden.h"
#include "gpsv_zero_rotation_cases.h"
#include <iostream>

bool CheckRangeGolden()
{
    bool passed = true;
    for (const auto& test : sparse_test::MakeGpsvRangeCases()) {
        const auto& a = test.matrix;
        const auto actual = sparse_test::QrPentadiagSolveGolden(a.ds, a.dl, a.d, a.du, a.dw, test.rhs, a.m,
                                                                a.batchCount);
        if (actual.size() != test.expected.size()) {
            return false;
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            passed = passed && std::isfinite(actual[i]) && std::fabs(actual[i] - test.expected[i]) < 1e-5F;
        }
    }
    return passed;
}

int main()
{
    int failures = 0;
    int systems = 0;
    for (int m = 1; m <= 6; ++m) {
        for (float sign : {1.0F, -1.0F}) {
            const auto test = sparse_test::MakeGpsvPermutationCase(m, sign);
            const auto& a = test.matrix;
            const auto actual = sparse_test::QrPentadiagSolveGolden(a.ds, a.dl, a.d, a.du, a.dw, test.rhs, m,
                                                                    a.batchCount);
            bool passed = actual.size() == test.expected.size();
            for (size_t i = 0; i < actual.size(); ++i) {
                passed = passed && std::isfinite(actual[i]) && actual[i] == test.expected[i];
            }
            std::cout << "m=" << m << " sign=" << sign << " systems=" << a.batchCount
                      << " golden=" << (passed ? "PASS" : "FAIL") << '\n';
            failures += !passed;
            systems += a.batchCount;
        }
    }
    const bool rangePassed = CheckRangeGolden();
    std::cout << "scaled_range_golden=" << (rangePassed ? "PASS" : "FAIL") << '\n';
    failures += !rangePassed;
    std::cout << "systems=" << systems << " failed_groups=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
