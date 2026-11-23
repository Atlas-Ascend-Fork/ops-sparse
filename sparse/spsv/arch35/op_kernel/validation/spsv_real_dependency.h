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

#ifndef OP_KERNEL_VALIDATION_SPSV_REAL_DEPENDENCY_H_
#define OP_KERNEL_VALIDATION_SPSV_REAL_DEPENDENCY_H_

__simt_callee__ __attribute__((always_inline)) inline bool SpsvTwoBandScanRow(
    const SpsvRationalIndexArray& rowPtr, const SpsvRationalIndexArray& colInd, __gm__ const float* values,
    int64_t r, int64_t step, int64_t m, bool forward, float d, float& a, float& b)
{
    bool haveA = false, haveB = false;
    bool supported = SpsvBlockFiniteNormal(d) && d != 0.0f;
    for (int64_t at = rowPtr[r]; at < rowPtr[r + 1] && supported; ++at) {
        int64_t c = colInd[at], logical = forward ? c : m - 1 - c;
        if (logical >= step) {
            continue;
        }
        float v = values[at];
        supported &= SpsvBlockFiniteNormal(v);
        if (step - logical == 1 && !haveA) {
            a = v;
            haveA = true;
        } else if (step - logical == 2 && !haveB) {
            b = v;
            haveB = true;
        } else if (SpsvRationalNonzero(v)) {
            supported = false;
        }
    }
    return supported;
}

__simt_callee__ __attribute__((always_inline)) inline int64_t SpsvTwoBandBuildTransitions(
    const SpsvRationalIndexArray& rowPtr, const SpsvRationalIndexArray& colInd, __gm__ const float* values,
    const SpsvRationalIndexArray& diagPtr, int64_t m, bool forward, bool nonUnit, __gm__ float* scratch,
    __gm__ float* matrix, int32_t tid)
{
    int64_t first = m;
    for (int64_t step = tid; step < m; step += blockDim.x) {
        int64_t r = forward ? step : m - 1 - step;
        float d = nonUnit ? (diagPtr[r] >= 0 ? values[diagPtr[r]] : 0.0f) : 1.0f;
        float a = 0.0f, b = 0.0f;
        bool supported = SpsvTwoBandScanRow(rowPtr, colInd, values, r, step, m, forward, d, a, b);
        if (!supported) {
            first = step < first ? step : first;
        }
        __gm__ float* out = scratch + kSpsvTwoBandWords * step;
        out[5] = d;
        out[6] = a;
        out[7] = b;
        __gm__ float* transition = matrix + 4 * step;
        transition[0] = supported && step >= 2 ? -a / d : 1.0f;
        transition[1] = supported && step >= 2 ? -b / d : 0.0f;
        transition[2] = supported && step >= 2 ? 1.0f : 0.0f;
        transition[3] = supported && step >= 2 ? 0.0f : 1.0f;
    }
    return first;
}

__simt_callee__ __attribute__((always_inline)) inline __gm__ float* SpsvTwoBandComposeTransitions(
    __gm__ float* matrix, __gm__ float* next, int64_t first, int32_t tid)
{
    for (int64_t distance = 1; distance < first; distance *= 2) {
        for (int64_t step = tid; step < first; step += blockDim.x) {
            __gm__ const float* a = matrix + 4 * step;
            __gm__ float* out = next + 4 * step;
            if (step < distance) {
                for (int32_t j = 0; j < 4; ++j) {
                    out[j] = a[j];
                }
            } else {
                __gm__ const float* b = matrix + 4 * (step - distance);
                out[0] = fmaf(a[0], b[0], a[1] * b[2]);
                out[1] = fmaf(a[0], b[1], a[1] * b[3]);
                out[2] = fmaf(a[2], b[0], a[3] * b[2]);
                out[3] = fmaf(a[2], b[1], a[3] * b[3]);
            }
        }
        asc_threadfence();
        asc_syncthreads();
        __gm__ float* swap = matrix;
        matrix = next;
        next = swap;
    }
    return matrix;
}

__simt_callee__ __attribute__((always_inline)) inline uint32_t SpsvTwoBandBuildInverse(__gm__ float* scratch,
                                                                                       __gm__ float* matrix,
                                                                                       int64_t first,
                                                                                       int32_t tid)
{
    uint32_t invalid = 0;
    for (int64_t step = tid; step < first; step += blockDim.x) {
        __gm__ float* out = scratch + kSpsvTwoBandWords * step;
        float h0, h1;
        if (step == 0) {
            h0 = 1.0f / out[5];
            __gm__ const float* second = out + kSpsvTwoBandWords;
            h1 = first > 1 ? -(second[6] / second[5]) / out[5] : 0.0f;
        } else {
            __gm__ const float* previous = out - kSpsvTwoBandWords;
            float divisor = fmaf(out[0], previous[1], -out[1] * previous[0]) * out[5];
            h0 = previous[1] / divisor;
            h1 = -previous[0] / divisor;
        }
        out[2] = h0;
        out[3] = h1;
        invalid |= !SpsvBlockFiniteNormal(out[0]) || !SpsvBlockFiniteNormal(out[1]) ||
                   !SpsvBlockFiniteNormal(h0) || !SpsvBlockFiniteNormal(h1);
        matrix[step] = SpsvFloatBoundUp(SpsvAbsF(h0) > SpsvAbsF(h1) ? SpsvAbsF(h0) : SpsvAbsF(h1));
    }
    return invalid;
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvTwoBandPrefixNorm(__gm__ float* scratch,
                                                                                 __gm__ float* matrix,
                                                                                 __gm__ float* next,
                                                                                 int64_t first, int32_t tid)
{
    for (int64_t distance = 1; distance < first; distance *= 2) {
        for (int64_t step = tid; step < first; step += blockDim.x) {
            next[step] =
                step >= distance ? SpsvFloatBoundAdd(matrix[step], matrix[step - distance]) : matrix[step];
        }
        asc_threadfence();
        asc_syncthreads();
        __gm__ float* swap = matrix;
        matrix = next;
        next = swap;
    }
    for (int64_t step = tid; step < first; step += blockDim.x) {
        scratch[kSpsvTwoBandWords * step + 4] = matrix[step];
    }
    asc_threadfence();
    asc_syncthreads();
}

// An approximate inverse may be generated with rounded parallel products:
// the independent I-A*P certificate, rather than the construction, proves it.
__simt_callee__ inline void SpsvTwoBandStoreTransitions(__gm__ float* scratch, __gm__ const float* matrix,
                                                        int64_t first, int32_t tid)
{
    for (int64_t step = tid; step < first; step += blockDim.x) {
        __gm__ float* out = scratch + kSpsvTwoBandWords * step;
        out[0] = step == 0 ? 1.0f : matrix[4 * step + 1];
        out[1] = step == 0 ? 0.0f : matrix[4 * step];
    }
    asc_threadfence();
    asc_syncthreads();
}

__simt_callee__ inline int64_t SpsvTwoBandPrepareParallel(const SpsvRationalIndexArray& rowPtr,
                                                          const SpsvRationalIndexArray& colInd,
                                                          __gm__ const float* values,
                                                          const SpsvRationalIndexArray& diagPtr, float alpha,
                                                          int64_t m, bool forward, bool nonUnit,
                                                          __gm__ float* scratch)
{
    __ubuf__ int64_t firstUnsupported[8];
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    if (!SpsvBlockFiniteNormal(alpha)) {
        return -1;
    }
    int64_t first = m;
    __gm__ float* matrix = scratch + kSpsvTwoBandWords * m;
    __gm__ float* next = matrix + 4 * m;
    first = SpsvTwoBandBuildTransitions(rowPtr, colInd, values, diagPtr, m, forward, nonUnit, scratch, matrix,
                                        tid);
    // Row counts fit I32; use an I32 reduction to avoid a 64-bit warp API.
    int32_t stop = asc_reduce_min(static_cast<int32_t>(first));
    if ((tid & 31) == 0) {
        firstUnsupported[tid / 32] = stop;
    }
    asc_threadfence();
    asc_syncthreads();
    first = firstUnsupported[0];
    for (int32_t warp = 1; warp < 8; ++warp) {
        first = firstUnsupported[warp] < first ? firstUnsupported[warp] : first;
    }
    if (m - first > kSpsvBoundBlockRows) {
        return -1;
    }
    matrix = SpsvTwoBandComposeTransitions(matrix, next, first, tid);
    SpsvTwoBandStoreTransitions(scratch, matrix, first, tid);
    // Reuse the matrix buffers for a positive prefix norm of H.
    matrix = scratch + kSpsvTwoBandWords * m;
    next = matrix + 4 * m;
    uint32_t invalid = SpsvTwoBandBuildInverse(scratch, matrix, first, tid);
    invalid = asc_reduce_max(invalid);
    if ((tid & 31) == 0) {
        firstUnsupported[tid / 32] = invalid;
    }
    asc_threadfence();
    asc_syncthreads();
    for (int32_t warp = 0; warp < 8; ++warp) {
        if (firstUnsupported[warp] != 0) {
            return -1;
        }
    }
    SpsvTwoBandPrefixNorm(scratch, matrix, next, first, tid);
    return first;
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvPublishInverseCertificate(
    __gm__ float* results, int32_t block, int32_t tid, float residual, float defect)
{
    residual = asc_reduce_max(residual);
    defect = asc_reduce_max(defect);
    __ubuf__ float bounds[8];
    if ((tid & 31) == 0) {
        bounds[2 * (tid / 32)] = residual;
        bounds[2 * (tid / 32) + 1] = defect;
    }
    asc_syncthreads();
    if (tid == 0) {
        for (int32_t warp = 1; warp < 4; ++warp) {
            residual = bounds[2 * warp] > residual ? bounds[2 * warp] : residual;
            defect = bounds[2 * warp + 1] > defect ? bounds[2 * warp + 1] : defect;
        }
        results[2 * block] = residual;
        results[2 * block + 1] = defect;
    }
    asc_threadfence();
}

// Verify independent approximate-inverse rows across the solve's blocks.
// Only the small residual/defect summaries survive into the retry launch;
// its integer scratch is reused after those summaries have been consumed.
__simt_vf__ __aicore__ __launch_bounds__(128) inline void SpsvFloatInverseCertificate(
    __gm__ const uint8_t* rp, __gm__ const uint8_t* ci, __gm__ const float* values, __gm__ const float* rhs,
    __gm__ const float* vecY, __gm__ float* state, __gm__ const uint8_t* dp, float alpha, int64_t m,
    bool forward, bool nonUnit, bool rowWide, bool colWide, bool permWide)
{
    SpsvInvalidateSolveCache((__gm__ void*)state);
    int32_t blocks = static_cast<int32_t>(gridDim.x), block = static_cast<int32_t>(blockIdx.x);
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    __gm__ float* results = state + 4 * m;
    int64_t capacity = (kSpsvRationalBitsPerRow * m + 512 + 15) / 16;
    int64_t floats = (kSpsvRationalTargetBatch + 2) * capacity / 2;
    int32_t perBlock = static_cast<int32_t>((floats - 2 * blocks - kSpsvInverseRowWords * m) /
                                            (4 * kSpsvInverseScratch * blocks));
    perBlock = perBlock < 128 ? perBlock : 128;
    if (perBlock <= 0) {
        if (tid == 0) {
            results[2 * block] = results[2 * block + 1] = __builtin_inff();
        }
        for (int64_t row = static_cast<int64_t>(block) * blockDim.x + tid; row < m;
             row += static_cast<int64_t>(blocks) * blockDim.x) {
            results[2 * blocks + kSpsvInverseRowWords * row] = __builtin_inff();
        }
        return;
    }
    int32_t active = perBlock * blocks, lane = block * perBlock + tid;
    SpsvRationalIndexArray rowPtr{(__gm__ const int32_t*)rp, (__gm__ const int64_t*)rp, rowWide};
    SpsvRationalIndexArray colInd{(__gm__ const int32_t*)ci, (__gm__ const int64_t*)ci, colWide};
    SpsvRationalIndexArray diagPtr{(__gm__ const int32_t*)dp, (__gm__ const int64_t*)dp, permWide};
    float residual = 0.0f, defect = 0.0f;
    __gm__ float* rowBounds = results + 2 * blocks;
    __gm__ float* scratch = rowBounds + kSpsvInverseRowWords * m;
    for (int64_t row = lane; tid < perBlock && row < m; row += active) {
        float r, d;
        SpsvFloatInverseRowBound(rowPtr, colInd, values, rhs, diagPtr, alpha, m, forward, nonUnit, vecY,
                                 state, row, scratch, active, lane, r, d,
                                 rowBounds + kSpsvInverseRowWords * row);
        residual = r > residual ? r : residual;
        defect = d > defect ? d : defect;
    }
    SpsvPublishInverseCertificate(results, block, tid, residual, defect);
}

__simt_callee__ __attribute__((always_inline)) inline bool SpsvFloatBlockHasRisk(uint32_t risk,
                                                                                 __ubuf__ uint32_t* warpRisk,
                                                                                 int32_t tid)
{
    risk = asc_reduce_max(risk);
    if ((tid & 31) == 0) {
        warpRisk[tid / 32] = risk;
    }
    asc_syncthreads();
    if (tid == 0) {
        for (int32_t warp = 1; warp < 8; ++warp) {
            warpRisk[0] |= warpRisk[warp];
        }
    }
    asc_syncthreads();
    return warpRisk[0] != 0;
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvFloatCheckInverseSummary(
    __gm__ const uint16_t* digits, int32_t certificateBlocks, __ubuf__ uint32_t* warpRisk, int32_t tid)
{
    if (tid == 0) {
        __gm__ const float* results = (__gm__ const float*)digits;
        float residual = 0.0f, defect = 0.0f;
        for (int32_t block = 0; block < certificateBlocks; ++block) {
            residual = results[2 * block] > residual ? results[2 * block] : residual;
            defect = results[2 * block + 1] > defect ? results[2 * block + 1] : defect;
        }
        warpRisk[0] = SpsvFloatBoundAdd(residual, defect) < 1.0f ? 0U : 1U;
    }
    asc_syncthreads();
}

__simt_callee__ __attribute__((always_inline)) inline uint32_t SpsvFloatCheckInverseTail(
    __gm__ const float* vecY, __gm__ float* state, __gm__ int32_t* scale, int64_t m, bool forward,
    int32_t certificateBlocks, int32_t tid)
{
    // A row's uncertified tail precedes it by at least the inverse window.
    // A window-sized group is therefore independent and verifies in parallel;
    // only publication between groups is sequential.
    __gm__ float* normalized = (__gm__ float*)scale;
    uint32_t tailRisk = 0;
    for (int64_t base = 0; base < m; base += kSpsvInverseWindow) {
        int64_t step = base + tid;
        if (tid < kSpsvInverseWindow && step < m) {
            int64_t r = forward ? step : m - 1 - step;
            float radius;
            bool valid = SpsvFloatInverseTailRowBound(vecY, state, m, forward, certificateBlocks, normalized,
                                                      r, radius);
            normalized[r] = valid ? radius : __builtin_inff();
            tailRisk |= valid ? 0U : 1U;
        }
        asc_threadfence();
        asc_syncthreads();
    }
    return tailRisk;
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvFloatReduceTwoBandBounds(
    const SpsvRationalIndexArray& rowPtr, const SpsvRationalIndexArray& colInd, __gm__ const float* values,
    const SpsvRationalIndexArray& diagPtr, __gm__ const float* rhs, __gm__ float* vecY, __gm__ float* state,
    float alpha, int64_t m, bool forward, bool nonUnit, __ubuf__ int64_t* twoBandCompleted,
    __ubuf__ float* twoBandBounds, __ubuf__ uint32_t* warpRisk, int32_t tid, float inverseNorm,
    float residualNorm, float defectNorm, float allowedRadius)
{
    inverseNorm = asc_reduce_max(inverseNorm);
    residualNorm = asc_reduce_max(residualNorm);
    defectNorm = asc_reduce_max(defectNorm);
    allowedRadius = asc_reduce_min(allowedRadius);
    if ((tid & 31) == 0) {
        int32_t at = (tid / 32) * 4;
        twoBandBounds[at] = inverseNorm;
        twoBandBounds[at + 1] = residualNorm;
        twoBandBounds[at + 2] = defectNorm;
        twoBandBounds[at + 3] = allowedRadius;
    }
    asc_syncthreads();
    if (tid == 0) {
        for (int32_t warp = 1; warp < 8; ++warp) {
            int32_t at = warp * 4;
            inverseNorm = twoBandBounds[at] > inverseNorm ? twoBandBounds[at] : inverseNorm;
            residualNorm = twoBandBounds[at + 1] > residualNorm ? twoBandBounds[at + 1] : residualNorm;
            defectNorm = twoBandBounds[at + 2] > defectNorm ? twoBandBounds[at + 2] : defectNorm;
            allowedRadius = twoBandBounds[at + 3] < allowedRadius ? twoBandBounds[at + 3] : allowedRadius;
        }
        warpRisk[0] = SpsvTwoBandTailCertified(rowPtr, colInd, values, rhs, diagPtr, alpha, m, forward,
                                               nonUnit, vecY, state, twoBandCompleted[0], inverseNorm,
                                               residualNorm, defectNorm, allowedRadius)
                          ? 0U
                          : 1U;
    }
    asc_syncthreads();
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvFloatCheckTwoBand(
    const SpsvRationalIndexArray& rowPtr, const SpsvRationalIndexArray& colInd, __gm__ const float* values,
    const SpsvRationalIndexArray& diagPtr, __gm__ const float* rhs, __gm__ float* vecY, __gm__ float* state,
    float alpha, int64_t m, bool forward, bool nonUnit, __gm__ uint16_t* digits, __ubuf__ uint32_t* warpRisk,
    int32_t tid)
{
    // Generate the virtual inverse once, then verify its rows in parallel.
    // The existing exact-retry scratch is much larger than 8*m floats.
    __ubuf__ int64_t twoBandCompleted[1];
    __ubuf__ float twoBandBounds[32];
    __gm__ float* twoBandScratch = (__gm__ float*)digits;
    int64_t completed = SpsvTwoBandPrepareParallel(rowPtr, colInd, values, diagPtr, alpha, m, forward,
                                                   nonUnit, twoBandScratch);
    if (tid == 0) {
        twoBandCompleted[0] = completed;
    }
    asc_threadfence();
    asc_syncthreads();
    float inverseNorm = 0.0f, residualNorm = 0.0f, defectNorm = 0.0f;
    float allowedRadius = __builtin_inff();
    for (int64_t step = tid; step < twoBandCompleted[0]; step += blockDim.x) {
        float p, r, d, a;
        SpsvTwoBandRowBounds(twoBandScratch, rhs, alpha, step, m, forward, vecY, state, p, r, d, a);
        inverseNorm = p > inverseNorm ? p : inverseNorm;
        residualNorm = r > residualNorm ? r : residualNorm;
        defectNorm = d > defectNorm ? d : defectNorm;
        allowedRadius = a < allowedRadius ? a : allowedRadius;
    }
    SpsvFloatReduceTwoBandBounds(rowPtr, colInd, values, diagPtr, rhs, vecY, state, alpha, m, forward,
                                 nonUnit, twoBandCompleted, twoBandBounds, warpRisk, tid, inverseNorm,
                                 residualNorm, defectNorm, allowedRadius);
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvClassifyFloatRecoveryRows(
    const SpsvRationalIndexArray& rowPtr, const SpsvRationalIndexArray& colInd, __gm__ const float* values,
    const SpsvRationalIndexArray& diagPtr, __gm__ const float* rhs, __gm__ float* vecY, __gm__ float* state,
    float alpha, int64_t m, bool forward, bool nonUnit, __gm__ int32_t* scale)
{
    // Classify the original equations, never the rounded candidate Y.
    // Include stored zero edges: 0*NaN must retain the public IEEE behavior.
    for (int64_t step = 0; step < m; ++step) {
        int64_t r = forward ? step : m - 1 - step;
        float diagonal = nonUnit ? (diagPtr[r] >= 0 ? values[diagPtr[r]] : 0.0f) : 1.0f;
        bool valid = SpsvIsFiniteF(alpha) && SpsvIsFiniteF(rhs[r]) && SpsvIsFiniteF(diagonal) &&
                     SpsvRationalNonzero(diagonal);
        for (int64_t p = rowPtr[r]; p < rowPtr[r + 1]; ++p) {
            int32_t c = static_cast<int32_t>(colInd[p]);
            if (forward ? c < r : c > r) {
                valid &= SpsvIsFiniteF(values[p]) && scale[c] != 0;
            }
        }
        scale[r] = valid ? 1 : 0;
        if (!valid) {
            state[2 * r + 1] = kSpsvInvalidDependencyRadius;
        }
    }
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvFloatRationalRecovery(
    const SpsvRationalIndexArray& rowPtr, const SpsvRationalIndexArray& colInd, __gm__ const float* values,
    const SpsvRationalIndexArray& diagPtr, __gm__ const float* rhs, __gm__ float* vecY, __gm__ float* state,
    float alpha, int64_t m, bool forward, bool nonUnit, __gm__ int32_t* scale, __gm__ uint32_t* residue,
    SpsvBigUInt* numerators, SpsvBigUInt& modulus, SpsvBigUInt& denominator)
{
    for (int64_t first = 0; first < m; first += kSpsvRationalTargetBatch) {
        int32_t targets[kSpsvRationalTargetBatch];
        int32_t targetCount = 0;
        int64_t end = first + kSpsvRationalTargetBatch;
        end = end < m ? end : m;
        for (int64_t r = first; r < end; ++r) {
            if (SpsvFloatNeedsDependencyRetry(vecY[r], state[2 * r], state[2 * r + 1])) {
                targets[targetCount++] = static_cast<int32_t>(r);
            }
        }
        if (targetCount == 0) {
            continue;
        }
        float answers[kSpsvRationalTargetBatch];
        uint32_t answerValid[kSpsvRationalTargetBatch];
        if (SpsvRationalSolveBatch(rowPtr, colInd, values, rhs, diagPtr, alpha, m, targets, targetCount,
                                   forward, nonUnit, scale, residue, numerators, modulus, denominator,
                                   answers, answerValid)) {
            for (int32_t i = 0; i < targetCount; ++i) {
                if (answerValid[i] != 0) {
                    vecY[targets[i]] = answers[i];
                }
            }
        }
    }
}

#endif // OP_KERNEL_VALIDATION_SPSV_REAL_DEPENDENCY_H_
