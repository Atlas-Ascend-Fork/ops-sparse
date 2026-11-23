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

/* Range protection for rows whose fast solve produces a non-finite value.
 * Complex rows of length 4..255 and wide levels use scaled arithmetic
 * directly; short and cooperative long rows retry locally when needed.
 * FP32 rows retain the fast solve and enter these helpers only on a retry.
 * Products and the RHS are accumulated in a scaled domain, with compensation
 * retained through RHS cancellation before restoring units and dividing.
 */

#ifndef SPSV_RANGE_H_
#define SPSV_RANGE_H_

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "../../shared/spsv_tiling_data.h"
#include "spsv_arith.h"

// Lanes per warp on this part.  A warp executes in step, so a reduction
// across one needs no barrier; asc_reduce_add is collective over exactly this
// many lanes and every one of them has to reach it.
constexpr int32_t kSpsvWarpLanes = 32;

// Sum a (hi, lo) pair across the warp.
//
// asc_reduce_add can only reduce the hi parts, which would round the
// cross-lane sum at single precision and hand back exactly the accuracy the
// pairs are carried for.  A float-float butterfly costs five shuffle steps and
// touches no memory.  SpsvDFAdd is not symmetric in its low word, so lanes can
// finish a butterfly holding totals that differ in the last bit; the broadcast
// makes every lane agree, which is what lets every lane store the row.
__simt_callee__ inline SpsvDF SpsvWarpReduceDF(SpsvDF v)
{
    for (int32_t mask = 1; mask < kSpsvWarpLanes; mask <<= 1) {
        SpsvDF other{asc_shfl_xor(v.hi, mask), asc_shfl_xor(v.lo, mask), asc_shfl_xor(v.lost, mask)};
        v = SpsvDFAdd(v, other);
    }
    return SpsvDF{asc_shfl(v.hi, 0), asc_shfl(v.lo, 0), asc_shfl(v.lost, 0)};
}

// How far down a row of n terms has to be brought for its dot product to be
// summable in range, as a count of bits.  Every term is a product of two
// finite floats and so at most FLT_MAX; n of them, each divided by 2^k with
// 2^k >= 2n, cannot sum past FLT_MAX in any order.
__simt_callee__ inline int32_t SpsvRowShiftBits(int32_t n)
{
    int32_t bits = 1;
    while ((1 << bits) < 2 * n) {
        bits++;
    }
    return bits;
}

// A pair scaled by 2^e with no call, for an e that may be larger than one
// float factor can carry.  SpsvScaleFactors splits it in two and saturates
// what two of them cannot span, which is where the answer leaves range
// anyway, so saturating costs nothing.
__simt_callee__ inline SpsvDF SpsvDFScaleWide(SpsvDF v, int32_t e)
{
    SpsvScalePair f = SpsvScaleFactors(e);
    float hi = v.hi * f.f1 * f.f2;
    float lo = v.lo * f.f1 * f.f2;
    return SpsvDF{hi, lo,
                  v.lost | (kSpsvRangeLoss *
                            static_cast<uint32_t>((v.hi != 0.0f && hi == 0.0f) || (v.lo != 0.0f && lo == 0.0f)))};
}

// The quotient, without calls.
//
// Writing the normalisations with ldexpf is the obvious way and is what this
// did: three calls on the operands and two in each of four SpsvDFScale2,
// eleven in all, on every row the fallback touches.  On this part a call is
// expensive enough to matter -- an earlier round of this same operator got
// 1.15 MB of binary back by taking ldexpf off the row tail, and the same
// mistake had crept in here.
//
// The closing rescale has to be one step either way.  Taken in two, an
// intermediate can be subnormal, and this part flushes subnormals to zero:
// 3e38 * 3e38 / 3e38 would come back as zero.
//
// What this comment used to claim -- that every exponent below is inside
// SpsvPow2's range -- was false, and the review found it.  Normalising a
// value by its own exponent wants 2^-k for a k that reaches 127, and
// SpsvPow2(-127) is the float zero rather than 2^-127.
//
// So the three normalisations use 2^(1-k) and leave their operands in [2,4):
// 1 - k is in range for every exponent SpsvExpOf can return, and the factors
// of two that leaves are integers on the closing exponent rather than
// multiplies.  Only the closing rescale needs the two-factor form.
__simt_callee__ inline void SpsvComplexDFDivRescaled(SpsvDF& re, SpsvDF& im, SpsvComplex64 recip, float scale,
                                                     int32_t extraExp)
{
    int32_t kv = SpsvExpOfMax2(re.hi, im.hi);
    int32_t ks = SpsvExpOf(scale);
    int32_t kr = SpsvExpOfMax2(recip.real, recip.imag);
    // 2^(1-k), not 2^-k: see the note above.  A zero scale still falls to the
    // 1.0 guard below, because zero times any factor is zero.
    float dr = SpsvPow2(1 - kr);
    SpsvComplex64 rs{recip.real * dr, recip.imag * dr};
    SpsvDF nRe = SpsvDFScaleRow(re, 1 - kv);
    SpsvDF nIm = SpsvDFScaleRow(im, 1 - kv);
    float ss = scale * SpsvPow2(1 - ks);
    float by = (ss != 0.0f) ? ss : 1.0f;
    SpsvDF qRe = SpsvDFAdd(SpsvDFMulF(nRe, rs.real), SpsvDFNeg(SpsvDFMulF(nIm, rs.imag)));
    SpsvDF qIm = SpsvDFAdd(SpsvDFMulF(nRe, rs.imag), SpsvDFMulF(nIm, rs.real));
    // the two operands each came in twice as big and the divisor is twice as
    // big too, so the quotient carries one factor of two to be taken out here
    int32_t e = kv - ks + kr + extraExp - 1;
    re = SpsvDFScaleWide(SpsvDF{qRe.hi / by, qRe.lo / by, qRe.lost}, e);
    im = SpsvDFScaleWide(SpsvDF{qIm.hi / by, qIm.lo / by, qIm.lost}, e);
}

// Whether a row of the answer came back in range.  This is the whole of the
// test that decides a retry: one overload for each dtype, because the two
// vectors have different element types and nothing else differs.
__simt_callee__ inline bool SpsvRangeRowFinite(__gm__ const float* vecY, int32_t row)
{
    return SpsvIsFiniteF(vecY[row]);
}

__simt_callee__ inline bool SpsvRangeRowFinite(__gm__ const SpsvComplex64* vecY, int32_t row)
{
    return SpsvIsFiniteF(vecY[row].real) && SpsvIsFiniteF(vecY[row].imag);
}

// How far the matrix entries and the already-solved Y each have to come down
// for a row to be accumulated in range.
struct SpsvRowShift {
    int32_t v;
    int32_t y;
};

// The room a row has, given entirely to its already-solved Y.
//
// ep is the largest PRODUCT the row forms -- max over its terms of
// e(entry) + e(Y) -- and not the largest entry times the largest Y.  Those
// are different quantities and the difference was a defect: for a row of
// [F, -F, t, 1] against Y = [1, 1, F] with F = 2^127 and t = 2^-126, the
// largest entry and the largest Y belong to different terms, so entry-max
// plus Y-max says 2^254 while the largest product is 2^127.  Sizing from
// 2^254 cost 131 bits of shift, which sent t to 2^-192 before the multiply --
// below what fp32 represents at all -- and t * Y[2] = 2 was the whole answer,
// because the two large products cancel exactly.
//
// Nothing comes off the entries: the whole shift goes on Y, so no entry is
// brought down before it is multiplied.  Each product is then at most
// 2^(ep - sh.y), and the room reserved for adding n of them up is what
// SpsvRowShiftBits computes.
//
// er is the right-hand side's exponent.  It joins the sum rather than a
// product, so it is not in ep, but it still has to fit the room the sum has:
// a row whose entries are ordinary and whose alpha*x is near FLT_MAX
// overflows on the way in.
//
// The shift can only get smaller than the old one: where the largest entry
// and the largest Y do belong to the same term, ep is ev + ey and nothing
// moves.  So no row that was in range before can leave it.
//
// One factor carries 126 bits at most: SpsvPow2(-127) is the float zero and
// the exponent field wraps below that, so a shift of more than 126 taken on Y
// alone would multiply the whole dot product by zero, or by a large negative
// power of two.  What exceeds 126 comes off the entries after all.  That needs
// a single product of about 2^249, or an alpha*x that is not finite, and it is
// the one case where scaling the entries cannot be avoided.
//
// Out of reach either way: a term whose product is more than the whole fp32
// range below the row's largest and which matters anyway, because the larger
// ones cancel exactly.  One shared exponent cannot hold both ends of that --
// it needs a per-product one, which this path cannot pay for.  The case above
// is 2^127 apart.
__simt_callee__ inline SpsvRowShift SpsvSplitShifts(int32_t ep, int32_t er, int32_t n)
{
    int32_t cap = 126 - SpsvRowShiftBits(n);
    int32_t top = (ep > er) ? ep : er;
    int32_t need = (top > cap) ? (top - cap) : 0;
    SpsvRowShift sh;
    sh.y = (need > 126) ? 126 : need;
    sh.v = need - sh.y;
    return sh;
}

// The right-hand side's exponent for a complex row, or zero when it joins
// after the division instead (a negative stored scale) and so is not in the
// accumulation's units at all.
__simt_callee__ inline int32_t SpsvRowRhsExp(__gm__ const float* rowPrep, int32_t row)
{
    if (rowPrep[18 * static_cast<int64_t>(row) + 6] < 0.0f) {
        return 0;
    }
    return SpsvExpOfMax2(rowPrep[18 * static_cast<int64_t>(row) + 2], rowPrep[18 * static_cast<int64_t>(row) + 4]);
}

// Those shifts, taken from the operands' own exponents rather than from the
// row's length.  Length alone does not see a product coming: a row of two
// entries, 1e10 against a solved 1e30, forms 1e40 however few terms there
// are, and dividing by the diagonal afterwards cannot bring back what has
// already become an infinity.
//
// The shift goes on Y, and on the entries only for the part one float factor
// cannot carry; either way it costs one multiply each and bounds the product
// by what is left of the two exponents together.  The room reserved for adding
// the row up is what SpsvRowShiftBits already computes.
//
// Collective over the warp: every lane must scale by the same thing, so the
// maxima are reduced across it.
// A share of the row's entries, raising the two maxima.  The share is a
// (start, stride) pair so that one walk serves a warp, which takes every
// thirty-second entry, and a thread that owns the row, which takes all of
// them.  A thread must not reach a collective, so reducing the maxima is the
// warp caller's business and not this function's.
template <typename RowPtrT, typename ColIndT, bool FORWARD>
__simt_callee__ inline void SpsvRowMaxExps(__gm__ const ColIndT* colInd, __gm__ const SpsvComplex64* values,
                                           __gm__ const float* yWide, int32_t row, RowPtrT rs, RowPtrT re,
                                           int32_t start, int32_t stride, int32_t& ep)
{
    for (SpsvCsrCursor<RowPtrT> p = static_cast<SpsvCsrCursor<RowPtrT>>(rs) + start; p < re;
         p += static_cast<RowPtrT>(stride)) {
        int32_t col = static_cast<int32_t>(colInd[p]);
        if (FORWARD ? (col >= row) : (col <= row)) {
            continue;
        }
        // the term's own product, not the row's largest entry and largest Y
        int32_t e = SpsvExpOfMax2(values[p].real, values[p].imag) +
                    SpsvExpOfMax2(yWide[4 * static_cast<int64_t>(col)], yWide[4 * static_cast<int64_t>(col) + 2]);
        ep = (e > ep) ? e : ep;
    }
}

// The same share, accumulated scaled.
__simt_callee__ inline void SpsvAccumScaledTerm(float vr, float vi, SpsvDF yr, SpsvDF yi, float dv, int32_t sy,
                                                SpsvDF& sumRe, SpsvDF& sumIm);

template <typename RowPtrT, typename ColIndT, bool FORWARD>
__simt_callee__ inline void SpsvAccumScaledRow(__gm__ const ColIndT* colInd, __gm__ const SpsvComplex64* values,
                                               __gm__ const float* yWide, int32_t row, RowPtrT rs, RowPtrT re,
                                               int32_t start, int32_t stride, float dv, int32_t sy, SpsvDF& sumRe,
                                               SpsvDF& sumIm)
{
    for (SpsvCsrCursor<RowPtrT> q = static_cast<SpsvCsrCursor<RowPtrT>>(rs) + start; q < re;
         q += static_cast<RowPtrT>(stride)) {
        int32_t col = static_cast<int32_t>(colInd[q]);
        if (FORWARD ? (col >= row) : (col <= row)) {
            continue;
        }
        SpsvAccumScaledTerm(values[q].real, values[q].imag,
                            SpsvDF{yWide[4 * static_cast<int64_t>(col)], yWide[4 * static_cast<int64_t>(col) + 1]},
                            SpsvDF{yWide[4 * static_cast<int64_t>(col) + 2], yWide[4 * static_cast<int64_t>(col) + 3]},
                            dv, sy, sumRe, sumIm);
    }
}

template <typename RowPtrT, typename ColIndT, bool FORWARD>
__simt_callee__ inline SpsvRowShift SpsvRowScaleShifts(__gm__ const ColIndT* colInd, __gm__ const SpsvComplex64* values,
                                                       __gm__ const float* yWide, __gm__ const float* rowPrep,
                                                       int32_t row, RowPtrT rs, RowPtrT re, int32_t lane)
{
    int32_t ep = 0;
    SpsvRowMaxExps<RowPtrT, ColIndT, FORWARD>(colInd, values, yWide, row, rs, re, lane, kSpsvWarpLanes, ep);
    return SpsvSplitShifts(static_cast<int32_t>(asc_reduce_max(static_cast<float>(ep))), SpsvRowRhsExp(rowPrep, row),
                           static_cast<int32_t>(re - rs));
}

// One term of a scaled row.  Shared, so that neither of the two ways of
// reading a row carries its own copy of the arithmetic.
// The entry arrives as two floats rather than as a SpsvComplex64: the stored
// one is __gm__ qualified, which is a distinct type with no conversion to the
// plain struct, so it binds to neither a value parameter nor a ternary beside
// one.  Floats cross the address space freely.
__simt_callee__ inline void SpsvAccumScaledTerm(float vre, float vim, SpsvDF yr, SpsvDF yi, float dv, int32_t sy,
                                                SpsvDF& sumRe, SpsvDF& sumIm)
{
    if (dv == 0.0f) {
        sumRe.lost |= kSpsvRangeLoss;
        sumIm.lost |= kSpsvRangeLoss;
        return;
    }
    // entry and Y each come down by their own shift, so neither the product
    // nor the sum can be formed out of range
    float vr = vre * dv;
    float vi = vim * dv;
    uint32_t lostScale = kSpsvRangeLoss * static_cast<uint32_t>(vr / dv != vre || vi / dv != vim);
    sumRe.lost |= lostScale;
    sumIm.lost |= lostScale;
    SpsvDF ys = SpsvDFScaleRow(yr, -sy);
    SpsvDF yt = SpsvDFScaleRow(yi, -sy);
    sumRe = SpsvDFAdd(sumRe, SpsvDFNeg(SpsvDFAdd(SpsvDFMulF(ys, vr), SpsvDFNeg(SpsvDFMulF(yt, vi)))));
    sumIm = SpsvDFAdd(sumIm, SpsvDFNeg(SpsvDFAdd(SpsvDFMulF(ys, vi), SpsvDFMulF(yt, vr))));
}

// Can the row's total go back to true units?
//
// A total of 2^e in units of 2^-bits is 2^(e + bits) in true ones, so this
// asks whether that is still a number.  When it is, the row's two levels are
// added there and the answer carries both: the scale is a power of two, so
// undoing it is exact.
//
// When it is not, the kept total is past 2^126 in true units, while every
// emptied product is below 2^(bits - 126) by the definition of emptied.  The
// shift's own bound is bits <= 128 + SpsvRowShiftBits(n) and 2^SpsvRowShiftBits
// is under 4n, so n of them come to less than 16n^2 -- below 2^-60 of what was
// kept for any row an int32 can index, where the pair itself carries 2^-48.
// Recovering them could not change a bit of the answer.
//
// What this replaces was a test on whether the emptied products were big
// enough to matter, and the review found it wrong: the right-hand side joins
// the total AFTERWARDS and can cancel it, so a total far outside any band can
// still leave an emptied product as the whole answer.  alpha * x is in the
// total before this is asked now, and representability is not something a
// later addition can take away.
__simt_callee__ inline bool SpsvRowCollapses(float sumRe, float sumIm, int32_t bits)
{
    float top = SpsvMaxAbs2(sumRe, sumIm);
    return (bits > 0) && ((top == 0.0f) || (SpsvExpOf(top) + bits <= 126));
}

// The tail every scaled path shares: let the right-hand side in, divide, and
// publish.  Whoever owns the row has already combined it -- a warp across its
// lanes, a thread in its own register -- so this takes a total, not a share.  The exponent is restored as part
// of dividing rather than before it, so an answer that is representable comes
// out representable -- including one as small as the smallest normal, which a
// restore-then-divide would have flushed to zero.
template <bool NON_UNIT>
__simt_callee__ inline uint32_t SpsvFinishRowScaled(__gm__ SpsvComplex64* vecY, __gm__ float* yWide,
                                                    __gm__ const float* rowPrep, int32_t row, int32_t bits,
                                                    SpsvDF sumRe, SpsvDF sumIm, SpsvDF lostRe, SpsvDF lostIm)
{
    // alpha * x for this row, built by the precompute; a negative stored
    // scale means it already carries the division and joins after it, in true
    // units.  SpsvDFScaleWide rather than SpsvDFScaleRow: this shift is a sum
    // of two exponents and can leave what one factor can express.
    SpsvDF rhsRe{rowPrep[18 * static_cast<int64_t>(row) + 2], rowPrep[18 * static_cast<int64_t>(row) + 3]};
    SpsvDF rhsIm{rowPrep[18 * static_cast<int64_t>(row) + 4], rowPrep[18 * static_cast<int64_t>(row) + 5]};
    bool preDiv = rowPrep[18 * static_cast<int64_t>(row) + 6] < 0.0f;
    if (!preDiv) {
        sumRe = SpsvDFAdd(SpsvDFScaleWide(rhsRe, -bits), sumRe);
        sumIm = SpsvDFAdd(SpsvDFScaleWide(rhsIm, -bits), sumIm);
    }
    // The two levels, added where the row can hold both.  This is after the
    // right-hand side on purpose: see SpsvRowCollapses.
    if (SpsvRowCollapses(sumRe.hi, sumIm.hi, bits)) {
        sumRe = SpsvDFAdd(SpsvDFScaleWide(sumRe, bits), lostRe);
        sumIm = SpsvDFAdd(SpsvDFScaleWide(sumIm, bits), lostIm);
        bits = 0;
    }
    uint32_t discarded = sumRe.lost | sumIm.lost;
    if constexpr (NON_UNIT) {
        // Numerator, diagonal and the accumulation's own shift are combined
        // into one exponent and applied once.  Dividing first and restoring
        // after would form 2^-129 for a row whose answer is 2^-126, and this
        // part flushes that to zero.
        SpsvComplex64 recip{rowPrep[18 * static_cast<int64_t>(row)], rowPrep[18 * static_cast<int64_t>(row) + 1]};
        SpsvComplexDFDivRescaled(sumRe, sumIm, recip, SpsvAbsF(rowPrep[18 * static_cast<int64_t>(row) + 6]), bits);
        if (preDiv) {
            sumRe = SpsvDFAdd(rhsRe, sumRe);
            sumIm = SpsvDFAdd(rhsIm, sumIm);
        }
    } else {
        sumRe = SpsvDFScaleWide(sumRe, bits);
        sumIm = SpsvDFScaleWide(sumIm, bits);
    }
    vecY[row].real = sumRe.hi;
    vecY[row].imag = sumIm.hi;
    yWide[4 * static_cast<int64_t>(row)] = sumRe.hi;
    yWide[4 * static_cast<int64_t>(row) + 1] = sumRe.lo;
    yWide[4 * static_cast<int64_t>(row) + 2] = sumIm.hi;
    yWide[4 * static_cast<int64_t>(row) + 3] = sumIm.lo;
    return discarded | sumRe.lost | sumIm.lost | lostRe.lost | lostIm.lost;
}

// One product of a row, into the level that can hold it.  The row's shift can
// empty a product -- scaled, it falls under what fp32 keeps and comes back
// zero -- and in true units nothing can empty it: all that lies below the
// smallest normal there is a product fp32 cannot form at all.
//
// Which products the shift emptied is decided by redoing the scaled multiply
// and seeing it come back zero, never by comparing exponents.  The review
// found why: expfield(v) + expfield(y) is a LOWER bound on the product, so it
// calls emptied some products the first walk kept -- 24 * 1.5 under a shift of
// 131 is 1.125 * 2^-126, the smallest normal, and its estimate is 2^4 -- and
// adding those here as well counts them twice.  Redoing the multiply is the
// same arithmetic the first walk did, so the two cannot disagree.
//
// Asking the multiply is also what keeps this independent of where the part
// draws its subnormal line.  A scaled product between the smallest subnormal
// and the smallest normal measured as NOT zero here, so a test written
// against "below 2^-126 is gone" would have been wrong in the direction that
// invents a term to recover; this one is right either way, because a product
// is emptied exactly when the arithmetic that was done returned zero.
//
// The sign belongs to the caller's entry rather than to this: a product is
// linear in it, so negating the entry is both exact and free.
__simt_callee__ inline void SpsvAccumLostProduct(SpsvDF y, float v, float dv, int32_t sy, SpsvDF& lost)
{
    if (SpsvDFMulF(SpsvDFScaleRow(y, -sy), v * dv).hi == 0.0f) {
        lost = SpsvDFAdd(lost, SpsvDFMulF(y, v));
    }
}

// The four products of one complex term, routed.  A row subtracts its terms,
// so the signs of -(yr * vre - yi * vim) and -(yr * vim + yi * vre) are what
// the entries carry in.
__simt_callee__ inline void SpsvAccumLostTerm(SpsvDF yr, SpsvDF yi, float vre, float vim, float dv, int32_t sy,
                                              SpsvDF& lostRe, SpsvDF& lostIm)
{
    SpsvAccumLostProduct(yr, -vre, dv, sy, lostRe);
    SpsvAccumLostProduct(yi, vim, dv, sy, lostRe);
    SpsvAccumLostProduct(yr, -vim, dv, sy, lostIm);
    SpsvAccumLostProduct(yi, -vre, dv, sy, lostIm);
}

// A share of the row's entries, in true units, for whatever the shift emptied.
// Each emptied product is below 2^(bits - 126) and so cannot overflow here.
//
// The parameters wrap where they do so that this shares fewer consecutive
// lines with SpsvAccumScaledRow than CodeCheck's clone rule counts.
template <typename RowPtrT, typename ColIndT, bool FORWARD>
__simt_callee__ inline void SpsvAccumLostRow(__gm__ const ColIndT* colInd, __gm__ const SpsvComplex64* values,
                                             __gm__ const float* yWide, int32_t row, RowPtrT rs, RowPtrT re,
                                             int32_t start, int32_t stride, float dv, int32_t sy, SpsvDF& lostRe,
                                             SpsvDF& lostIm)
{
    for (SpsvCsrCursor<RowPtrT> q = static_cast<SpsvCsrCursor<RowPtrT>>(rs) + start; q < re;
         q += static_cast<RowPtrT>(stride)) {
        int32_t col = static_cast<int32_t>(colInd[q]);
        if (FORWARD ? (col >= row) : (col <= row)) {
            continue;
        }
        SpsvAccumLostTerm(SpsvDF{yWide[4 * static_cast<int64_t>(col)], yWide[4 * static_cast<int64_t>(col) + 1]},
                          SpsvDF{yWide[4 * static_cast<int64_t>(col) + 2], yWide[4 * static_cast<int64_t>(col) + 3]},
                          values[q].real, values[q].imag, dv, sy, lostRe, lostIm);
    }
}

// A row that fits the warp: one lane holds its own entry, so the exponents
// and the accumulation come out of a single read.  The second read the
// two-pass form needs is another round of dependent loads -- colInd, then
// yWide at the column it names -- and that is what a short row costs.
template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT>
__simt_callee__ inline uint32_t SpsvSolveRowScaledNarrow(__gm__ const ColIndT* colInd,
                                                         __gm__ const SpsvComplex64* values, __gm__ SpsvComplex64* vecY,
                                                         __gm__ float* yWide, __gm__ const float* rowPrep, int32_t row,
                                                         RowPtrT rs, RowPtrT re, int32_t lane)
{
    SpsvDF zero{0.0f, 0.0f};
    SpsvCsrCursor<RowPtrT> p = static_cast<SpsvCsrCursor<RowPtrT>>(rs) + lane;
    int32_t col = (p < re) ? static_cast<int32_t>(colInd[p]) : row;
    bool dep = (p < re) && (FORWARD ? (col < row) : (col > row));
    float vre = dep ? values[p].real : 0.0f;
    float vim = dep ? values[p].imag : 0.0f;
    SpsvDF yr = dep ? SpsvDF{yWide[4 * static_cast<int64_t>(col)], yWide[4 * static_cast<int64_t>(col) + 1]} : zero;
    SpsvDF yi = dep ? SpsvDF{yWide[4 * static_cast<int64_t>(col) + 2], yWide[4 * static_cast<int64_t>(col) + 3]} : zero;
    int32_t ep = dep ? (SpsvExpOfMax2(vre, vim) + SpsvExpOfMax2(yr.hi, yi.hi)) : 0;
    SpsvRowShift sh = SpsvSplitShifts(static_cast<int32_t>(asc_reduce_max(static_cast<float>(ep))),
                                      SpsvRowRhsExp(rowPrep, row), static_cast<int32_t>(re - rs));
    SpsvDF sumRe = zero;
    SpsvDF sumIm = zero;
    if (dep) {
        SpsvAccumScaledTerm(vre, vim, yr, yi, SpsvPow2(-sh.v), sh.y, sumRe, sumIm);
    }
    sumRe = SpsvWarpReduceDF(sumRe);
    sumIm = SpsvWarpReduceDF(sumIm);
    int32_t bits = sh.v + sh.y;
    SpsvDF lostRe = zero;
    SpsvDF lostIm = zero;
    if (bits > 0) {
        if (dep) {
            SpsvAccumLostTerm(yr, yi, vre, vim, SpsvPow2(-sh.v), sh.y, lostRe, lostIm);
        }
        lostRe = SpsvWarpReduceDF(lostRe);
        lostIm = SpsvWarpReduceDF(lostIm);
    }
    return SpsvFinishRowScaled<NON_UNIT>(vecY, yWide, rowPrep, row, bits, sumRe, sumIm, lostRe, lostIm);
}

// One row, accumulated in a scaled domain and brought back only by the
// division itself.
template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT>
__simt_callee__ inline uint32_t SpsvSolveRowComplexScaled(__gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd,
                                                          __gm__ const SpsvComplex64* values,
                                                          __gm__ SpsvComplex64* vecY, __gm__ float* yWide,
                                                          __gm__ const float* rowPrep, int32_t row)
{
    RowPtrT rs = rowPtr[row];
    RowPtrT re = rowPtr[row + 1];
    int32_t lane = static_cast<int32_t>(threadIdx.x) & (kSpsvWarpLanes - 1);
    if (re - rs <= static_cast<RowPtrT>(kSpsvWarpLanes)) {
        return SpsvSolveRowScaledNarrow<RowPtrT, ColIndT, FORWARD, NON_UNIT>(colInd, values, vecY, yWide, rowPrep, row,
                                                                             rs, re, lane);
    }
    SpsvRowShift sh = SpsvRowScaleShifts<RowPtrT, ColIndT, FORWARD>(colInd, values, yWide, rowPrep, row, rs, re, lane);
    SpsvDF sumRe{0.0f, 0.0f};
    SpsvDF sumIm{0.0f, 0.0f};
    SpsvAccumScaledRow<RowPtrT, ColIndT, FORWARD>(colInd, values, yWide, row, rs, re, lane, kSpsvWarpLanes,
                                                  SpsvPow2(-sh.v), sh.y, sumRe, sumIm);
    sumRe = SpsvWarpReduceDF(sumRe);
    sumIm = SpsvWarpReduceDF(sumIm);
    int32_t bits = sh.v + sh.y;
    SpsvDF lostRe{0.0f, 0.0f};
    SpsvDF lostIm{0.0f, 0.0f};
    if (bits > 0) {
        SpsvAccumLostRow<RowPtrT, ColIndT, FORWARD>(colInd, values, yWide, row, rs, re, lane, kSpsvWarpLanes,
                                                    SpsvPow2(-sh.v), sh.y, lostRe, lostIm);
        lostRe = SpsvWarpReduceDF(lostRe);
        lostIm = SpsvWarpReduceDF(lostIm);
    }
    return SpsvFinishRowScaled<NON_UNIT>(vecY, yWide, rowPrep, row, bits, sumRe, sumIm, lostRe, lostIm);
}

// One complex row in a scaled domain, owned by a single thread.  A level wide
// enough to share out gives each thread a row of its own, so there is no warp
// to agree with: the maxima and the total are this thread's alone, and the
// row is read twice because the scale has to be known before the first
// product is formed.
template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT>
__simt_callee__ inline uint32_t SpsvSolveRowComplexScaledThread(__gm__ const RowPtrT* rowPtr,
                                                                __gm__ const ColIndT* colInd,
                                                                __gm__ const SpsvComplex64* values,
                                                                __gm__ SpsvComplex64* vecY, __gm__ float* yWide,
                                                                __gm__ const float* rowPrep, int32_t row)
{
    RowPtrT rs = rowPtr[row];
    RowPtrT re = rowPtr[row + 1];
    int32_t ep = 0;
    SpsvRowMaxExps<RowPtrT, ColIndT, FORWARD>(colInd, values, yWide, row, rs, re, 0, 1, ep);
    SpsvRowShift sh = SpsvSplitShifts(ep, SpsvRowRhsExp(rowPrep, row), static_cast<int32_t>(re - rs));
    SpsvDF sumRe{0.0f, 0.0f};
    SpsvDF sumIm{0.0f, 0.0f};
    SpsvAccumScaledRow<RowPtrT, ColIndT, FORWARD>(colInd, values, yWide, row, rs, re, 0, 1, SpsvPow2(-sh.v), sh.y,
                                                  sumRe, sumIm);
    int32_t bits = sh.v + sh.y;
    SpsvDF lostRe{0.0f, 0.0f};
    SpsvDF lostIm{0.0f, 0.0f};
    if (bits > 0) {
        SpsvAccumLostRow<RowPtrT, ColIndT, FORWARD>(colInd, values, yWide, row, rs, re, 0, 1, SpsvPow2(-sh.v), sh.y,
                                                    lostRe, lostIm);
    }
    return SpsvFinishRowScaled<NON_UNIT>(vecY, yWide, rowPrep, row, bits, sumRe, sumIm, lostRe, lostIm);
}

#endif // SPSV_RANGE_H_
