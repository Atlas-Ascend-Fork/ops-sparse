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

#ifndef OP_KERNEL_SOLVE_SPSV_COMPLEX_FAST_ROW_H_
#define OP_KERNEL_SOLVE_SPSV_COMPLEX_FAST_ROW_H_

// The range check and absolute product bound share the accumulation's loads.
// Return false before publishing anything if range or conditioning requires
// the scaled/exact path. WARP rows make the decision collectively.
// Bound a product of normalized pairs without a per-row division.
__simt_callee__ inline SpsvDF SpsvPairProductEnclosed(SpsvDF a, SpsvDF b, float& error)
{
    float high = a.hi * b.hi;
    float p0 = fmaf(a.hi, b.hi, -high);
    float p1 = a.hi * b.lo, p2 = a.lo * b.hi;
    float low = (p0 + p1) + p2;
    // TwoSum-normalized inputs have low words <= u times the high
    // words. 256*u*u*abs(high) majorizes the omitted lo*lo and the rounded
    // low operations, including rounding of this positive majorant.
    error += SpsvAbsF(high) * 0x1p-40f + 0x1p-90f;
    return {high, low};
}
template <typename RowPtrT, typename ColIndT, bool FORWARD>
__simt_callee__ __attribute__((always_inline)) inline void SpsvAccumulateFastComplex(
    __gm__ const ColIndT* colInd, __gm__ const SpsvComplex64* values, __gm__ const float* yWide,
    __gm__ const float* rowPrep, RowPtrT start, RowPtrT end, int32_t first, int32_t step, int32_t row, float& rh,
    float& rl, float& ih, float& il, float& er, float& ei, float& dependencyError, bool& safe, uint32_t& flags)
{
    bool deferred = rowPrep[18 * static_cast<int64_t>(row) + 7] == kSpsvDeferredComplexCertificate;
    for (SpsvCsrCursor<RowPtrT> p = static_cast<SpsvCsrCursor<RowPtrT>>(start) + first; p < end; p += step) {
        int32_t col = static_cast<int32_t>(colInd[p]);
        if (FORWARD ? col < row : col > row) {
            int64_t at = 4 * static_cast<int64_t>(col);
            float yr = yWide[at], yi = yWide[at + 2];
            float yl = yWide[at + 1], yil = yWide[at + 3];
            if (yr != yr || yi != yi) {
                // The existing path publishes NaN whenever any solved
                // dependency is NaN. Nothing later in this lane can change
                // that verdict; reconverge at the collective flag reduction.
                flags |= 3U;
                break;
            }
            float ar = values[p].real, ai = values[p].imag;
            if (!deferred) {
                float pe = rowPrep[18 * static_cast<int64_t>(col) + 9];
                float pi = rowPrep[18 * static_cast<int64_t>(col) + 10];
                dependencyError += (SpsvAbsF(ar) + SpsvAbsF(ai)) * (pe > pi ? pe : pi);
            }
            safe = safe & SpsvFastDependency(ar) & SpsvFastDependency(ai) & SpsvFastDependency(yr) &
                   SpsvFastDependency(yi);
            SpsvAccEnclosed(rh, rl, er, -ar, yr, yl);
            SpsvAccEnclosed(rh, rl, er, ai, yi, yil);
            SpsvAccEnclosed(ih, il, ei, -ar, yi, yil);
            SpsvAccEnclosed(ih, il, ei, -ai, yr, yl);
        }
    }
}

template <bool BLOCK, int GROUP>
__simt_callee__ __attribute__((always_inline)) inline void SpsvMergeFastComplexGroup(int64_t rowLength, SpsvDF& re,
                                                                                     SpsvDF& im, float& er, float& ei,
                                                                                     float& dependencyError, bool& safe,
                                                                                     uint32_t& flags)
{
    flags |= safe ? 0U : 2U;
    if constexpr (GROUP < 32) {
        for (int32_t mask = 1; mask < GROUP; mask <<= 1) {
            flags |= asc_shfl_xor(flags, mask);
        }
    } else {
        flags = asc_reduce_max(flags);
    }
    if constexpr (GROUP < 32) {
        for (int32_t mask = 1; mask < GROUP; mask <<= 1) {
            dependencyError += asc_shfl_xor(dependencyError, mask);
        }
    } else {
        dependencyError = asc_reduce_add(dependencyError);
    }
    safe = flags == 0;
    if ((flags & 1U) == 0) {
        if constexpr (GROUP < 32) {
            re = SpsvGroupEnclosed<GROUP>(re, er);
            im = SpsvGroupEnclosed<GROUP>(im, ei);
        } else {
            int32_t width = 32;
            if constexpr (!BLOCK) {
                width = rowLength <= 4 ? 4 : rowLength <= 8 ? 8 : rowLength <= 16 ? 16 : 32;
            }
            re = SpsvWarpEnclosed(re, er, width);
            im = SpsvWarpEnclosed(im, ei, width);
        }
    }
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvMergeFastComplexBlock(int32_t lane, SpsvDF& re,
                                                                                     SpsvDF& im, float& er, float& ei,
                                                                                     float& dependencyError, bool& safe,
                                                                                     uint32_t& flags)
{
    __ubuf__ float partials[64 * 9];
    int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
    int32_t nwarps = static_cast<int32_t>(blockDim.x) / 32;
    if (lane == 0) {
        partials[warp * 9] = re.hi;
        partials[warp * 9 + 1] = re.lo;
        partials[warp * 9 + 2] = im.hi;
        partials[warp * 9 + 3] = im.lo;
        partials[warp * 9 + 4] = er;
        partials[warp * 9 + 5] = ei;
        partials[warp * 9 + 6] = safe ? 0.0f : 1.0f;
        partials[warp * 9 + 7] = static_cast<float>(flags);
        partials[warp * 9 + 8] = dependencyError;
    }
    asc_syncthreads();
    re = {0.0f, 0.0f};
    im = {0.0f, 0.0f};
    er = ei = 0.0f;
    dependencyError = 0.0f;
    uint32_t invalid = 0;
    flags = 0;
    for (int32_t w = lane; w < nwarps; w += 32) {
        re = SpsvMergeEnclosed(re, {partials[w * 9], partials[w * 9 + 1]}, er);
        im = SpsvMergeEnclosed(im, {partials[w * 9 + 2], partials[w * 9 + 3]}, ei);
        er += partials[w * 9 + 4];
        ei += partials[w * 9 + 5];
        invalid |= static_cast<uint32_t>(partials[w * 9 + 6]);
        flags |= static_cast<uint32_t>(partials[w * 9 + 7]);
        dependencyError += partials[w * 9 + 8];
    }
    re = SpsvWarpEnclosed(re, er);
    im = SpsvWarpEnclosed(im, ei);
    dependencyError = asc_reduce_add(dependencyError);
    safe = asc_reduce_max(invalid) == 0;
    flags = asc_reduce_max(flags);
}

__simt_callee__ __attribute__((always_inline)) inline void SpsvDivideFastComplex(__gm__ const float* rowPrep,
                                                                                 int64_t meta, float inverseError,
                                                                                 SpsvDF& re, SpsvDF& im, float& er,
                                                                                 float& ei)
{
    SpsvDF ir{rowPrep[meta + 11], rowPrep[meta + 12]};
    SpsvDF ii{rowPrep[meta + 13], rowPrep[meta + 14]};
    float nr = SpsvAbsF(re.hi) + SpsvAbsF(re.lo);
    float ni = SpsvAbsF(im.hi) + SpsvAbsF(im.lo);
    float pr = SpsvAbsF(ir.hi) + SpsvAbsF(ir.lo);
    float pi = SpsvAbsF(ii.hi) + SpsvAbsF(ii.lo);
    float qrError = 4.0f * (er * pr + ei * pi + (nr + ni) * inverseError);
    float qiError = 4.0f * (er * pi + ei * pr + (nr + ni) * inverseError);
    SpsvDF a = SpsvPairProductEnclosed(re, ir, qrError);
    SpsvDF b = SpsvPairProductEnclosed(im, ii, qrError);
    float mergeError = 0.0f;
    SpsvDF qr = SpsvMergeEnclosed(a, {-b.hi, -b.lo}, mergeError);
    qrError += mergeError * 0x1p-21f;
    a = SpsvPairProductEnclosed(re, ii, qiError);
    b = SpsvPairProductEnclosed(im, ir, qiError);
    mergeError = 0.0f;
    SpsvDF qi = SpsvMergeEnclosed(a, b, mergeError);
    qiError += mergeError * 0x1p-21f;
    re = qr;
    im = qi;
    er = qrError;
    ei = qiError;
}

template <bool NON_UNIT>
__simt_callee__ __attribute__((always_inline)) inline bool SpsvFastComplexCertified(SpsvDF re, SpsvDF im, float er,
                                                                                    float ei, bool safe)
{
    // A large finite answer may have a correctly certified rounding cell even
    // when its local enclosure exceeds the small-answer absolute budget.
    // Reuse the recovery check for that case; the full predecessor enclosure
    // is still published below and can independently require reconstruction.
    float br = SpsvAbsF(re.hi) * 0x1p-12f;
    float bi = SpsvAbsF(im.hi) * 0x1p-12f;
    br = br > 0x1p-18f ? br : 0x1p-18f;
    bi = bi > 0x1p-18f ? bi : 0x1p-18f;
    br = br < 0.0005f ? br : 0.0005f;
    bi = bi < 0.0005f ? bi : 0.0005f;
    bool realSafe = er <= br;
    bool imagSafe = ei <= bi;
    if constexpr (NON_UNIT) {
        if (!realSafe) {
            realSafe = !SpsvFloatNeedsDependencyRetry(re.hi, re.lo, er);
        }
        if (!imagSafe) {
            imagSafe = !SpsvFloatNeedsDependencyRetry(im.hi, im.lo, ei);
        }
    }
    if (!(safe && SpsvIsFiniteF(er) && SpsvIsFiniteF(ei) && realSafe && imagSafe && SpsvIsFiniteF(re.hi) &&
          SpsvIsFiniteF(re.lo) && SpsvIsFiniteF(im.hi) && SpsvIsFiniteF(im.lo))) {
        return false;
    }
    return true;
}

template <bool NON_UNIT>
__simt_callee__ __attribute__((always_inline)) inline void SpsvPublishFastComplex(
    __gm__ SpsvComplex64* vecY, __gm__ float* yWide, __gm__ const float* rowPrep, int32_t row, int64_t meta,
    int64_t rowLength, SpsvDF re, SpsvDF im, float er, float ei, float dependencyError)
{
    vecY[row].real = re.hi;
    vecY[row].imag = im.hi;
    yWide[4 * static_cast<int64_t>(row)] = re.hi;
    yWide[4 * static_cast<int64_t>(row) + 1] = re.lo;
    yWide[4 * static_cast<int64_t>(row) + 2] = im.hi;
    yWide[4 * static_cast<int64_t>(row) + 3] = im.lo;
    if (rowPrep[meta + 7] == kSpsvDeferredComplexCertificate) {
        // Never publish a local estimate as a global dependency certificate.
        ((__gm__ float*)rowPrep)[meta + 9] = __builtin_inff();
        ((__gm__ float*)rowPrep)[meta + 10] = __builtin_inff();
        return;
    }
    // Inflate the positive sum by its operation count, then use the
    // certified inverse norm. Keep summation slack separate from division
    // instead of paying a constant factor at every level of the graph.
    float propagated = SpsvFloatDependencyBound(dependencyError, rowLength);
    if constexpr (NON_UNIT) {
        propagated = SpsvFloatBoundMul(propagated, rowPrep[meta + 8]);
    }
    ((__gm__ float*)rowPrep)[meta + 9] = SpsvFloatBoundAdd(er, propagated);
    ((__gm__ float*)rowPrep)[meta + 10] = SpsvFloatBoundAdd(ei, propagated);
    ((__gm__ float*)rowPrep)[meta + 7] = -1.0f;
}

__simt_callee__ __attribute__((always_inline)) inline bool SpsvPublishFastComplexNaN(__gm__ SpsvComplex64* vecY,
                                                                                     __gm__ float* yWide,
                                                                                     __gm__ const float* rowPrep,
                                                                                     int32_t row, int64_t meta,
                                                                                     uint32_t flags)
{
    if (flags & 1U) {
        float nanValue = 0.0f / 0.0f;
        vecY[row].real = vecY[row].imag = nanValue;
        int64_t at = 4 * static_cast<int64_t>(row);
        yWide[at] = yWide[at + 2] = nanValue;
        yWide[at + 1] = yWide[at + 3] = 0.0f;
        ((__gm__ float*)rowPrep)[meta + 7] = -2.0f;
        ((__gm__ float*)rowPrep)[meta + 9] = __builtin_inff();
        ((__gm__ float*)rowPrep)[meta + 10] = __builtin_inff();
        return true;
    }
    return (flags & 1U) != 0;
}

template <bool NON_UNIT>
__simt_callee__ __attribute__((always_inline)) inline bool SpsvCertifyFastComplex(__gm__ const float* rowPrep,
                                                                                  int64_t meta, float inverseError,
                                                                                  SpsvDF& re, SpsvDF& im, float& er,
                                                                                  float& ei, bool safe)
{
    er = er * 0x1p-21f + 0x1p-90f;
    ei = ei * 0x1p-21f + 0x1p-90f;
    if constexpr (NON_UNIT) {
        SpsvDivideFastComplex(rowPrep, meta, inverseError, re, im, er, ei);
    }
    return SpsvFastComplexCertified<NON_UNIT>(re, im, er, ei, safe);
}

template <bool WARP, bool SHORT, bool BLOCK, int GROUP>
__simt_callee__ __attribute__((always_inline)) inline void SpsvFastComplexLanes(int32_t& lane, int32_t& first,
                                                                                int32_t& step)
{
    lane = WARP ? static_cast<int32_t>(threadIdx.x) & (GROUP - 1) : 0;
    first = (WARP && !SHORT) ? (BLOCK ? static_cast<int32_t>(threadIdx.x) : lane) : 0;
    step = (WARP && !SHORT) ? (BLOCK ? static_cast<int32_t>(blockDim.x) : GROUP) : 1;
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, bool WARP = false, bool SHORT = false,
          bool BLOCK = false, int GROUP = 32>
__simt_callee__ inline bool SpsvTryFastComplexRow(__gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd,
                                                  __gm__ const SpsvComplex64* values, __gm__ SpsvComplex64* vecY,
                                                  __gm__ float* yWide, __gm__ const float* rowPrep, int32_t row)
{
    const int64_t meta = 18 * static_cast<int64_t>(row);
    RowPtrT start = rowPtr[row], end = rowPtr[row + 1];
    float inverseError = rowPrep[meta + 15];
    bool safe = end - start <= (1 << 20) && rowPrep[meta + 8] > 0.0f && inverseError >= 0.0f;
    if (!safe) {
        return false;
    }
    int32_t lane, first, step;
    SpsvFastComplexLanes<WARP, SHORT, BLOCK, GROUP>(lane, first, step);
    float rh = 0.0f, rl = 0.0f, ih = 0.0f, il = 0.0f;
    float er = 0.0f, ei = 0.0f, dependencyError = 0.0f;
    uint32_t flags = 0;
    if constexpr (!WARP || SHORT) {
        rh = rowPrep[meta + 2];
        rl = rowPrep[meta + 3];
        ih = rowPrep[meta + 4];
        il = rowPrep[meta + 5];
        er = rowPrep[meta + 9] * 0x1p-23f;
        ei = rowPrep[meta + 10] * 0x1p-23f;
    }
    SpsvAccumulateFastComplex<RowPtrT, ColIndT, FORWARD>(colInd, values, yWide, rowPrep, start, end, first, step, row,
                                                         rh, rl, ih, il, er, ei, dependencyError, safe, flags);
    SpsvDF re = SpsvTwoSum(rh, rl), im = SpsvTwoSum(ih, il);
    if constexpr (WARP && !SHORT) {
        SpsvMergeFastComplexGroup<BLOCK, GROUP>(static_cast<int64_t>(end) - start, re, im, er, ei, dependencyError,
                                                safe, flags);
    }
    if constexpr (BLOCK) {
        SpsvMergeFastComplexBlock(lane, re, im, er, ei, dependencyError, safe, flags);
    }
    if constexpr (WARP && !SHORT) {
        re = SpsvMergeEnclosed(re, {rowPrep[meta + 2], rowPrep[meta + 3]}, er);
        im = SpsvMergeEnclosed(im, {rowPrep[meta + 4], rowPrep[meta + 5]}, ei);
        er += rowPrep[meta + 9] * 0x1p-23f;
        ei += rowPrep[meta + 10] * 0x1p-23f;
    }
    if (SpsvPublishFastComplexNaN(vecY, yWide, rowPrep, row, meta, flags)) {
        return true;
    }
    // Positive error accumulation is inflated by eight single-precision
    // roundoffs, covering the low-word sums and their warp/block merges.
    if (!SpsvCertifyFastComplex<NON_UNIT>(rowPrep, meta, inverseError, re, im, er, ei, safe)) {
        return false;
    }
    SpsvPublishFastComplex<NON_UNIT>(vecY, yWide, rowPrep, row, meta, static_cast<int64_t>(end) - start, re, im, er, ei,
                                     dependencyError);
    return true;
}

// A row of 256 or more nonzeros, taken by the whole block, and redone in a
// scaled domain if its answer left range.  The cooperative path wrote the row
// through the whole block, so the retry's reader needs a barrier; the retry
// is one warp's, so the caller owes another barrier before the block reads
// this row again.
template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT>
__simt_callee__ inline void SpsvSolveWideRowComplexChecked(SPSV_COMPLEX_LEVEL_PARAMS(RowPtrT, ColIndT),
                                                           SpsvComplex64 alpha, int32_t row)
{
    // Share the long dot product across the block. Warp leaders publish
    // partial sums in UB; a block barrier makes the second reduction safe.
    if (SpsvTryFastComplexRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, true, false, true>(rowPtr, colInd, values, vecY,
                                                                                      yWide, rowPrep, row)) {
        return;
    }
    SpsvFinishComplexColdRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, true, true>(rowPtr, colInd, values, vecX, vecY, yWide,
                                                                              alpha, row, rowPrep);
}

// A single-row level, in a scaled domain, and which form takes it.  A row of
// a few nonzeros is run whole on every lane: the warp form pays two exponent
// reductions and a float-float butterfly, five shuffle steps apiece, to share
// out a dot product of two terms.  kSpsvWarpRowMinNnz already names that
// trade for the fast path, with the measurement in its comment; the scaled
// path pays the two extra reductions, so the crossing does not move down.
//
// Every lane runs the whole row rather than one of them doing it, which is
// what the fast path does too: each lane then holds the answer and stores it,
// so no lane has to wait for a broadcast.
template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT, bool SHORT_ROWS>
__simt_callee__ inline void SpsvSolveComplexScaledNarrowRow(SPSV_COMPLEX_LEVEL_PARAMS(RowPtrT, ColIndT),
                                                            SpsvComplex64 alpha, int32_t row, RowPtrT rowLen)
{
    if (rowLen < static_cast<RowPtrT>(kSpsvWarpRowMinNnz)) {
        if (SpsvTryFastComplexRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, true, true>(rowPtr, colInd, values, vecY, yWide,
                                                                                   rowPrep, row)) {
            return;
        }
    } else if (SpsvTryFastComplexRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, true>(rowPtr, colInd, values, vecY, yWide,
                                                                                rowPrep, row)) {
        return;
    }
    SpsvFinishComplexColdRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, true>(rowPtr, colInd, values, vecX, vecY, yWide,
                                                                        alpha, row, rowPrep);
}

// A level wide enough to share out, one row per thread.  MULTI_BLOCK decides
// whether the rows are spread over every block or over one block's threads.
template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT, typename PermT, bool MULTI_BLOCK>
__simt_callee__ inline void SpsvSolveComplexWideLevel(SPSV_COMPLEX_LEVEL_PARAMS(RowPtrT, ColIndT), SpsvComplex64 alpha,
                                                      int32_t levelStart, int32_t levelWidth, int32_t numBlocks)
{
    int32_t start = static_cast<int32_t>(threadIdx.x);
    int32_t stride = static_cast<int32_t>(blockDim.x);
    if constexpr (MULTI_BLOCK) {
        start += static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x);
        stride = numBlocks * static_cast<int32_t>(blockDim.x);
    } else if (static_cast<int64_t>(rowPtr[levelRow[levelStart] + 1]) - rowPtr[levelRow[levelStart]] +
                   static_cast<int64_t>(rowPtr[levelRow[levelStart + levelWidth - 1] + 1]) -
                   rowPtr[levelRow[levelStart + levelWidth - 1]] >=
               32) {
        // Sample the first/last row length uniformly across the block. Short
        // rows keep one thread each; longer dot products use four lanes.
        // This changes work assignment only, never the error certificate.
        constexpr int32_t group = 4;
        for (uint32_t idx = static_cast<uint32_t>(start / group); idx < static_cast<uint32_t>(levelWidth);
             idx = SpsvNextComplexLevelRow(idx, stride / group)) {
            int32_t row = levelRow[levelStart + idx];
            if (!SpsvTryFastComplexRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, true, false, false, group>(
                    rowPtr, colInd, values, vecY, yWide, rowPrep, row) &&
                (static_cast<int32_t>(threadIdx.x) & (group - 1)) == 0) {
                SpsvFinishComplexColdRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, false>(rowPtr, colInd, values, vecX, vecY,
                                                                                     yWide, alpha, row, rowPrep);
            }
        }
        return;
    }
    for (uint32_t idx = static_cast<uint32_t>(start); idx < static_cast<uint32_t>(levelWidth);
         idx = SpsvNextComplexLevelRow(idx, stride)) {
        int32_t row = levelRow[levelStart + idx];
        if (SpsvTryFastComplexRow<RowPtrT, ColIndT, FORWARD, NON_UNIT>(rowPtr, colInd, values, vecY, yWide, rowPrep,
                                                                       row)) {
            continue;
        }
        SpsvFinishComplexColdRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, false>(rowPtr, colInd, values, vecX, vecY, yWide,
                                                                             alpha, row, rowPrep);
    }
}

// One level of the complex schedule.
//
// MULTI_BLOCK picks how the level's rows are spread: across every block, when
// the dispatch launches one level at a time and synchronizes between them, or
// across one block's threads, when a single block owns the whole schedule and
// carries its own barriers.  Everything else is the same, which is why this
// used to exist twice.
//
// Either way a level holding one row is taken by the first warp of block 0 --
// a row of 256+ nonzeros goes to the cooperative reducer, whose atomics are
// block-local and so cannot be spread either.

#endif // OP_KERNEL_SOLVE_SPSV_COMPLEX_FAST_ROW_H_
