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

#ifndef SPSV_RATIONAL_RECONSTRUCTION_H_
#define SPSV_RATIONAL_RECONSTRUCTION_H_

// The scalar target indices and complex component indices have different
// widths. Only their CRT residue merge and proof-based prime iteration agree.
template <typename TargetIndex>
__simt_callee__ inline void SpsvRationalMergeTargets(const TargetIndex* targets, int32_t targetCount,
                                                     __gm__ const uint32_t* residue, uint32_t determinant,
                                                     uint32_t prime, uint32_t inverseModulus, SpsvBigUInt* numerators,
                                                     const SpsvBigUInt& modulus)
{
    for (int32_t i = 0; i < targetCount; ++i) {
        uint32_t value = SpsvModMul(residue[targets[i]], determinant, prime);
        uint32_t old = SpsvBigMod(numerators[i], prime);
        uint32_t delta = value >= old ? value - old : prime - (old - value);
        SpsvBigCrtAdd(numerators[i], modulus, SpsvModMul(delta, inverseModulus, prime));
    }
}

template <typename TargetIndex, typename PrimeSystem>
__simt_callee__ inline void SpsvRationalReconstructPrimes(const TargetIndex* targets, int32_t targetCount,
                                                          __gm__ const uint32_t* residue, SpsvBigUInt* numerators,
                                                          SpsvBigUInt& modulus, int64_t determinantBoundBits,
                                                          const PrimeSystem& system)
{
    for (int32_t i = 0; i < targetCount; ++i) {
        SpsvBigSet(numerators[i], 0);
    }
    SpsvBigSet(modulus, 1);
    // Start with native uint32 modular products; extend with distinct larger
    // primes until the original determinant proof is satisfied, never a cap.
    uint32_t prime = 65521U;
    bool ascending = false;
    while (SpsvBigBits(modulus) <= determinantBoundBits + 1) {
        while (!SpsvRationalPrime(prime)) {
            SpsvRationalNextPrime(prime, ascending);
        }
        uint32_t determinant = 1;
        if (system.Solve(prime, determinant)) {
            uint32_t inverse = SpsvModPow(SpsvBigMod(modulus, prime), prime - 2, prime);
            SpsvRationalMergeTargets(targets, targetCount, residue, determinant, prime, inverse, numerators, modulus);
            SpsvBigMul(modulus, prime);
        }
        SpsvRationalNextPrime(prime, ascending);
    }
}

#endif // SPSV_RATIONAL_RECONSTRUCTION_H_
