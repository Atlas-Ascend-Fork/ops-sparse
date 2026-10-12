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

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "densetosparse_kernel.h"

namespace {

constexpr uint32_t kFormatCoo = 0;
constexpr uint32_t kFormatCsr = 1;
constexpr uint32_t kFormatBell = 3;
constexpr uint32_t kOrderRow = 0;
// Level-0 scan gives one chunk to one thread, so a large chunk leaves most
// threads idle on a long serial accumulation. Measured on P-02 COO: 1024 ->
// 357us, 256 -> 249us, 64 -> 224us, 32 -> 218us, 16 -> 217us.
constexpr uint64_t kScanChunk = 32;
constexpr uint64_t kAlign = 32;
constexpr int32_t kDeviceStatusSuccess = 0;
constexpr int32_t kDeviceStatusInvalid = 1;

template <typename T>
__simt_callee__ __aicore__ inline bool IsNonzero(T bits)
{
    return bits != static_cast<T>(0);
}

template <>
__simt_callee__ __aicore__ inline bool IsNonzero<uint16_t>(uint16_t bits)
{
    return (bits & 0x7FFFU) != 0;
}

template <>
__simt_callee__ __aicore__ inline bool IsNonzero<uint32_t>(uint32_t bits)
{
    return (bits & 0x7FFFFFFFU) != 0;
}

// complex64 is an interleaved (re, im) pair of FP32. Masking the sign bit of
// each half keeps the shared rule "magnitude bits non-zero means non-zero", so
// NaN and Inf survive in either component while (+-0, +-0) stays out.
template <>
__simt_callee__ __aicore__ inline bool IsNonzero<uint64_t>(uint64_t bits)
{
    const uint32_t re = static_cast<uint32_t>(bits) & 0x7FFFFFFFU;
    const uint32_t im = static_cast<uint32_t>(bits >> 32) & 0x7FFFFFFFU;
    return (re | im) != 0;
}

__simt_callee__ __aicore__ inline uint64_t DensePosition(
    uint64_t row, uint64_t col, uint64_t ld, uint32_t order)
{
    return order == kOrderRow ? row * ld + col : col * ld + row;
}

__simt_callee__ __aicore__ inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

__simt_callee__ __aicore__ inline uint64_t CeilDivDevice(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

// Both scan kernels are vector-instruction bound (measured aiv_vec_ratio ~1.0
// with aiv_mte2_ratio ~0), so the hot loops below walk a plain pointer with a
// constant element step instead of recomputing a 2-D address per element. The
// format/order decisions are loop invariant and stay outside the run helpers.

// The scan kernels are limited by load issue rate rather than arithmetic
// (measured: 16-byte loads gave int8 2.8x but left fp32 unchanged), so a window
// is the widest available vector load, 32 bytes. At the target sparsity
// (P scenarios are ~0.2% dense) almost every window is entirely zero.
using WideWord = ulong4;
constexpr uint64_t kWideBytes = 32;

/// Number of non-zero ValueT elements packed into one 64-bit word.
template <typename ValueT>
__simt_callee__ __aicore__ inline uint32_t CountWord(uint64_t word);

template <>
__simt_callee__ __aicore__ inline uint32_t CountWord<uint8_t>(uint64_t word)
{
    // Propagate any set bit of each byte into that byte's bit 7.
    uint64_t marks = ((word & 0x7F7F7F7F7F7F7F7FULL) + 0x7F7F7F7F7F7F7F7FULL) | word;
    marks &= 0x8080808080808080ULL;
    uint32_t count = 0;
    for (uint32_t shift = 7; shift < 64; shift += 8) {
        count += static_cast<uint32_t>((marks >> shift) & 1ULL);
    }
    return count;
}

template <>
__simt_callee__ __aicore__ inline uint32_t CountWord<uint16_t>(uint64_t word)
{
    const uint64_t magnitude = word & 0x7FFF7FFF7FFF7FFFULL;
    uint32_t count = 0;
    for (uint32_t shift = 0; shift < 64; shift += 16) {
        count += ((magnitude >> shift) & 0xFFFFULL) != 0 ? 1U : 0U;
    }
    return count;
}

template <>
__simt_callee__ __aicore__ inline uint32_t CountWord<uint32_t>(uint64_t word)
{
    const uint64_t magnitude = word & 0x7FFFFFFF7FFFFFFFULL;
    return (((magnitude & 0xFFFFFFFFULL) != 0) ? 1U : 0U) +
           (((magnitude >> 32) != 0) ? 1U : 0U);
}

/// complex64 stores one element per 64-bit word: the (re, im) FP32 pair.
template <>
__simt_callee__ __aicore__ inline uint32_t CountWord<uint64_t>(uint64_t word)
{
    return ((word & 0x7FFFFFFF7FFFFFFFULL) != 0) ? 1U : 0U;
}

/// Non-zero elements inside one 32-byte window.
template <typename ValueT>
__simt_callee__ __aicore__ inline uint32_t CountWide(WideWord window)
{
    return CountWord<ValueT>(window.x) + CountWord<ValueT>(window.y) +
           CountWord<ValueT>(window.z) + CountWord<ValueT>(window.w);
}

/// OR-ing raw words then masking is equivalent to OR-ing the per-element
/// magnitudes, which lets one test clear a whole region.
template <typename ValueT>
__simt_callee__ __aicore__ inline uint64_t MagnitudeMask();

template <>
__simt_callee__ __aicore__ inline uint64_t MagnitudeMask<uint8_t>()
{
    return 0xFFFFFFFFFFFFFFFFULL;
}

template <>
__simt_callee__ __aicore__ inline uint64_t MagnitudeMask<uint16_t>()
{
    return 0x7FFF7FFF7FFF7FFFULL;
}

template <>
__simt_callee__ __aicore__ inline uint64_t MagnitudeMask<uint32_t>()
{
    return 0x7FFFFFFF7FFFFFFFULL;
}

template <>
__simt_callee__ __aicore__ inline uint64_t MagnitudeMask<uint64_t>()
{
    return 0x7FFFFFFF7FFFFFFFULL;
}

// Regions are tested 256 bytes at a time: counting each window costs several
// ops, while the emptiness test costs one OR per word.
constexpr uint64_t kCoarseWindows = 8;

template <typename T>
__simt_callee__ __aicore__ inline bool IsWideAligned(__gm__ const T *address)
{
    return (reinterpret_cast<uintptr_t>(address) & (kWideBytes - 1)) == 0;
}

/// True when the kCoarseWindows windows starting at `wide` are all zero.
template <typename ValueT>
__simt_callee__ __aicore__ inline bool CoarseRegionIsZero(
    __gm__ const WideWord *wide)
{
    uint64_t any = 0;
    for (uint64_t k = 0; k < kCoarseWindows; ++k) {
        const WideWord window = wide[k];
        any |= window.x | window.y | window.z | window.w;
    }
    return (any & MagnitudeMask<ValueT>()) == 0;
}

/// Non-zero count over n contiguous elements.
template <typename ValueT>
__simt_callee__ __aicore__ inline uint64_t CountRunContiguous(
    __gm__ const ValueT *ptr, uint64_t n)
{
    constexpr uint64_t perWindow = kWideBytes / sizeof(ValueT);
    uint64_t count = 0;
    uint64_t i = 0;
    while (i < n && !IsWideAligned(ptr + i)) {
        count += IsNonzero<ValueT>(ptr[i]) ? 1 : 0;
        ++i;
    }
    const uint64_t windows = (n - i) / perWindow;
    __gm__ const WideWord *wide =
        reinterpret_cast<__gm__ const WideWord *>(ptr + i);
    uint64_t w = 0;
    for (; w + kCoarseWindows <= windows; w += kCoarseWindows) {
        if (CoarseRegionIsZero<ValueT>(wide + w)) {
            continue;
        }
        for (uint64_t k = 0; k < kCoarseWindows; ++k) {
            count += CountWide<ValueT>(wide[w + k]);
        }
    }
    for (; w < windows; ++w) {
        count += CountWide<ValueT>(wide[w]);
    }
    for (i += windows * perWindow; i < n; ++i) {
        count += IsNonzero<ValueT>(ptr[i]) ? 1 : 0;
    }
    return count;
}

/// Non-zero count over n elements starting at ptr, advancing by step elements.
template <typename ValueT>
__simt_callee__ __aicore__ inline uint64_t CountRun(
    __gm__ const ValueT *ptr, uint64_t step, uint64_t n)
{
    if (step == 1) {
        return CountRunContiguous<ValueT>(ptr, n);
    }
    uint64_t count = 0;
    for (uint64_t i = 0; i < n; ++i, ptr += step) {
        count += IsNonzero<ValueT>(*ptr) ? 1 : 0;
    }
    return count;
}

// One thread per unit makes neighbouring lanes touch addresses a whole chunk
// apart, which the measurements showed capping the scan at ~300 GB/s. For
// contiguous runs the warp cooperates instead so lane L reads window L, the
// same coalescing pattern nnz/arch35 uses for its column direction.
constexpr uint32_t kWarpSize = 32;

/// Warp-cooperative non-zero count over n contiguous elements. Every lane of
/// the warp must call this with identical arguments; the result is valid on
/// all lanes.
// Each lane loads one whole 32-byte window. Narrowing this to a 64-bit word per
// lane (the warp then covering 256 contiguous bytes) measured 1.8x slower, and
// handing each lane several consecutive windows made no difference beyond run
// to run noise, so per-lane load width is what matters here.
/// Bitmap words a run occupies, matching how CountRunWarp lays them out.
template <typename ValueT>
__simt_callee__ __aicore__ inline uint64_t RunBitmapWords(
    __gm__ const ValueT *ptr, uint64_t n)
{
    constexpr uint64_t perWindow = kWideBytes / sizeof(ValueT);
    uint64_t head = 0;
    while (head < n && !IsWideAligned(ptr + head)) {
        ++head;
    }
    return CeilDivDevice((n - head) / perWindow, kWarpSize);
}

template <typename ValueT>
__simt_callee__ __aicore__ inline int32_t CountRunWarp(
    __gm__ const ValueT *ptr, uint64_t n, uint32_t lane,
    __gm__ uint32_t *bitmap, uint64_t bitmapBase)
{
    constexpr uint64_t perWindow = kWideBytes / sizeof(ValueT);
    int32_t partial = 0;
    uint64_t head = 0;
    while (head < n && !IsWideAligned(ptr + head)) {
        ++head;
    }
    const uint64_t windows = (n - head) / perWindow;
    __gm__ const WideWord *wide =
        reinterpret_cast<__gm__ const WideWord *>(ptr + head);
    if (bitmap == nullptr) {
        // No bitmap wanted: keep the tight grid-stride loop, the batch form
        // below costs about 8% more on this path.
        for (uint64_t w = lane; w < windows; w += kWarpSize) {
            partial += static_cast<int32_t>(CountWide<ValueT>(wide[w]));
        }
    } else {
        // Full batches need no bounds check; only the last one does.
        const uint64_t fullBatches = (windows / kWarpSize) * kWarpSize;
        for (uint64_t batch = 0; batch < fullBatches; batch += kWarpSize) {
            const uint32_t mine = CountWide<ValueT>(wide[batch + lane]);
            partial += static_cast<int32_t>(mine);
            // One ballot per 32 windows is exactly one bitmap word.
            const uint32_t occupied = asc_ballot(mine != 0 ? 1 : 0);
            if (lane == 0) {
                bitmap[bitmapBase + batch / kWarpSize] = occupied;
            }
        }
        if (fullBatches < windows) {
            const uint64_t w = fullBatches + lane;
            const uint32_t mine = w < windows ? CountWide<ValueT>(wide[w]) : 0U;
            partial += static_cast<int32_t>(mine);
            const uint32_t occupied = asc_ballot(mine != 0 ? 1 : 0);
            if (lane == 0) {
                bitmap[bitmapBase + fullBatches / kWarpSize] = occupied;
            }
        }
    }
    // The unaligned head and the short tail are a handful of elements; let one
    // lane absorb them rather than pay another reduction. They are always
    // re-read by the emit pass, so they need no bitmap entry.
    if (lane == 0) {
        for (uint64_t i = 0; i < head; ++i) {
            partial += IsNonzero<ValueT>(ptr[i]) ? 1 : 0;
        }
        for (uint64_t i = head + windows * perWindow; i < n; ++i) {
            partial += IsNonzero<ValueT>(ptr[i]) ? 1 : 0;
        }
    }
    return asc_reduce_add(partial);
}

/// Index of the next non-zero at or after i, scanning whole windows while the
/// run stays empty. Returns n when the rest of the run is zero.
template <typename ValueT>
__simt_callee__ __aicore__ inline uint64_t NextNonzero(
    __gm__ const ValueT *ptr, uint64_t i, uint64_t n)
{
    constexpr uint64_t perWindow = kWideBytes / sizeof(ValueT);
    constexpr uint64_t perRegion = perWindow * kCoarseWindows;
    while (i < n) {
        if (IsWideAligned(ptr + i)) {
            __gm__ const WideWord *wide =
                reinterpret_cast<__gm__ const WideWord *>(ptr + i);
            if (i + perRegion <= n && CoarseRegionIsZero<ValueT>(wide)) {
                i += perRegion;
                continue;
            }
            if (i + perWindow <= n && CountWide<ValueT>(*wide) == 0) {
                i += perWindow;
                continue;
            }
        }
        if (IsNonzero<ValueT>(ptr[i])) {
            return i;
        }
        ++i;
    }
    return n;
}

/// Emit the non-zeros of one CSR/CSC run; returns the advanced output cursor.
template <typename ValueT, typename IndexT>
__simt_callee__ __aicore__ inline uint64_t EmitCompressedRun(
    __gm__ const ValueT *ptr, uint64_t step, uint64_t n, uint64_t firstItem,
    uint32_t base, __gm__ IndexT *indices, __gm__ ValueT *values, uint64_t out)
{
    if (step == 1) {
        for (uint64_t i = NextNonzero<ValueT>(ptr, 0, n); i < n;
             i = NextNonzero<ValueT>(ptr, i + 1, n)) {
            indices[out] = static_cast<IndexT>(firstItem + i + base);
            values[out] = ptr[i];
            ++out;
        }
        return out;
    }
    for (uint64_t i = 0; i < n; ++i, ptr += step) {
        const ValueT bits = *ptr;
        if (IsNonzero<ValueT>(bits)) {
            indices[out] = static_cast<IndexT>(firstItem + i + base);
            values[out] = bits;
            ++out;
        }
    }
    return out;
}

/// Emit the non-zeros of one COO run, which always lies inside a single row.
template <typename ValueT, typename IndexT>
__simt_callee__ __aicore__ inline uint64_t EmitCooRun(
    __gm__ const ValueT *ptr, uint64_t step, uint64_t n, uint64_t row,
    uint64_t firstCol, uint32_t base, __gm__ IndexT *rowIndices,
    __gm__ IndexT *colIndices, __gm__ ValueT *values, uint64_t out)
{
    const IndexT encodedRow = static_cast<IndexT>(row + base);
    if (step == 1) {
        for (uint64_t i = NextNonzero<ValueT>(ptr, 0, n); i < n;
             i = NextNonzero<ValueT>(ptr, i + 1, n)) {
            rowIndices[out] = encodedRow;
            colIndices[out] = static_cast<IndexT>(firstCol + i + base);
            values[out] = ptr[i];
            ++out;
        }
        return out;
    }
    for (uint64_t i = 0; i < n; ++i, ptr += step) {
        const ValueT bits = *ptr;
        if (IsNonzero<ValueT>(bits)) {
            rowIndices[out] = encodedRow;
            colIndices[out] = static_cast<IndexT>(firstCol + i + base);
            values[out] = bits;
            ++out;
        }
    }
    return out;
}

/// Exclusive prefix of `value` across the warp; `total` receives the warp sum.
/// Every lane must call this.
__simt_callee__ __aicore__ inline int32_t WarpExclusivePrefix(int32_t value,
                                                              uint32_t lane,
                                                              int32_t *total)
{
    int32_t inclusive = value;
    for (uint32_t delta = 1; delta < kWarpSize; delta <<= 1) {
        const int32_t carried = asc_shfl_up(inclusive, delta);
        if (lane >= delta) {
            inclusive += carried;
        }
    }
    *total = asc_shfl(inclusive, static_cast<int32_t>(kWarpSize) - 1);
    return inclusive - value;
}

/// 头尾不足一个窗口的那几个元素。每个 lane 都走一遍，好让游标不用归约就保持同
/// 步；只有 lane 0 真正落盘。返回推进后的游标。
template <typename ValueT, typename IndexT, bool Coo>
__simt_callee__ __aicore__ inline uint64_t EmitScalarRange(
    __gm__ const ValueT *ptr, uint64_t begin, uint64_t end, uint64_t firstItem,
    uint32_t base, IndexT encodedRow, __gm__ IndexT *indices,
    __gm__ IndexT *rowIndices, __gm__ IndexT *colIndices, __gm__ ValueT *values,
    uint64_t out, uint32_t lane, uint64_t limit)
{
    for (uint64_t i = begin; i < end; ++i) {
        if (!IsNonzero<ValueT>(ptr[i])) {
            continue;
        }
        if (lane == 0 && out < limit) {
            if (Coo) {
                rowIndices[out] = encodedRow;
                colIndices[out] = static_cast<IndexT>(firstItem + i + base);
            } else {
                indices[out] = static_cast<IndexT>(firstItem + i + base);
            }
            values[out] = ptr[i];
        }
        ++out;
    }
    return out;
}

/// 一个 kWideBytes 窗口内的落盘。游标由调用方按 warp 前缀算好，所以这里只管从
/// 它开始往后写；越过 limit 的位置照样计数，只是不写。
template <typename ValueT, typename IndexT, bool Coo>
__simt_callee__ __aicore__ inline void EmitWindow(
    __gm__ const ValueT *ptr, uint64_t first, uint64_t firstItem, uint32_t base,
    IndexT encodedRow, __gm__ IndexT *indices, __gm__ IndexT *rowIndices,
    __gm__ IndexT *colIndices, __gm__ ValueT *values, uint64_t cursor,
    uint64_t limit)
{
    constexpr uint64_t perWindow = kWideBytes / sizeof(ValueT);
    for (uint64_t k = 0; k < perWindow; ++k) {
        const ValueT bits = ptr[first + k];
        if (!IsNonzero<ValueT>(bits)) {
            continue;
        }
        if (cursor < limit) {
            if (Coo) {
                rowIndices[cursor] = encodedRow;
                colIndices[cursor] =
                    static_cast<IndexT>(firstItem + first + k + base);
            } else {
                indices[cursor] =
                    static_cast<IndexT>(firstItem + first + k + base);
            }
            values[cursor] = bits;
        }
        ++cursor;
    }
}

/// Warp-cooperative emit over n contiguous elements. Lane L owns window L of
/// each batch, so the per-lane exclusive prefix keeps the output ascending.
/// Returns the advanced output cursor, identical on every lane.
template <typename ValueT, typename IndexT, bool Coo>
__simt_callee__ __aicore__ inline uint64_t EmitRunWarp(
    __gm__ const ValueT *ptr, uint64_t n, uint64_t firstItem, uint32_t base,
    __gm__ IndexT *indices, __gm__ IndexT *rowIndices, __gm__ IndexT *colIndices,
    __gm__ ValueT *values, uint64_t out, uint32_t lane, uint64_t cooRow,
    uint64_t limit, __gm__ const uint32_t *bitmap, uint64_t bitmapBase)
{
    constexpr uint64_t perWindow = kWideBytes / sizeof(ValueT);
    const IndexT encodedRow = static_cast<IndexT>(cooRow + base);
    uint64_t head = 0;
    while (head < n && !IsWideAligned(ptr + head)) {
        ++head;
    }
    const uint64_t windows = (n - head) / perWindow;
    const uint64_t tailBegin = head + windows * perWindow;

    out = EmitScalarRange<ValueT, IndexT, Coo>(
        ptr, 0, head, firstItem, base, encodedRow, indices, rowIndices,
        colIndices, values, out, lane, limit);

    // A per-lane guard would only be predicated, not skipped, so every skip has
    // to be a branch the whole warp agrees on. Two levels: one bitmap word
    // clears 32 windows, and a warp's worth of words clears 1024 at a time.
    const uint64_t words = CeilDivDevice(windows, kWarpSize);
    for (uint64_t group = 0; group < words; group += kWarpSize) {
        uint32_t held = 0xFFFFFFFFU;
        if (bitmap != nullptr) {
            const uint64_t index = group + lane;
            held = index < words ? bitmap[bitmapBase + index] : 0U;
            if (asc_ballot(held != 0U ? 1 : 0) == 0U) {
                continue;
            }
        }
        const uint64_t span = words - group < kWarpSize ? words - group : kWarpSize;
        for (uint64_t step = 0; step < span; ++step) {
            if (bitmap != nullptr &&
                static_cast<uint32_t>(asc_shfl(static_cast<int32_t>(held),
                                               static_cast<int32_t>(step))) == 0U) {
                continue;
            }
            const uint64_t window = (group + step) * kWarpSize + lane;
            const bool active = window < windows;
            const int32_t mine = active
                ? static_cast<int32_t>(CountWide<ValueT>(
                      reinterpret_cast<__gm__ const WideWord *>(ptr + head)[window]))
                : 0;
            int32_t total = 0;
            const int32_t offset = WarpExclusivePrefix(mine, lane, &total);
            if (mine > 0) {
                EmitWindow<ValueT, IndexT, Coo>(
                    ptr, head + window * perWindow, firstItem, base, encodedRow,
                    indices, rowIndices, colIndices, values,
                    out + static_cast<uint64_t>(offset), limit);
            }
            out += static_cast<uint64_t>(total);
        }
    }

    return EmitScalarRange<ValueT, IndexT, Coo>(
        ptr, tailBegin, n, firstItem, base, encodedRow, indices, rowIndices,
        colIndices, values, out, lane, limit);
}

/// CSR/CSC unit geometry: the run is contiguous when the compressed major
/// dimension matches the dense order, otherwise it steps by ld.
__simt_callee__ __aicore__ inline bool CompressedRunIsContiguous(
    uint32_t format, uint32_t order)
{
    if (format == kFormatCoo) {
        return order == kOrderRow;
    }
    return (format == kFormatCsr) == (order == kOrderRow);
}

__simt_callee__ __aicore__ inline uint64_t CompressedRunStart(
    bool contiguous, uint64_t major, uint64_t begin, uint64_t ld)
{
    return contiguous ? major * ld + begin : begin * ld + major;
}

/// COO unit 内逐段 run 的游标。一个 unit 是一段逻辑区间，可能横跨若干行，而每次
/// 搬运只处理不跨行的一段；把这套推进逻辑收在一处，计数与转换两条路径共用，两者
/// 看到的 run 序列才保证一致。
struct CooRunCursor {
    uint64_t pos;  ///< 下一段 run 的逻辑起点
    uint64_t end;  ///< 本 unit 的逻辑终点
    uint64_t row;  ///< 当前 run 所在行
    uint64_t col;  ///< 当前 run 在行内的起始列
    uint64_t len;  ///< 当前 run 的长度
};

__simt_callee__ __aicore__ inline CooRunCursor CooCursorInit(uint64_t unit,
    uint64_t chunk, uint64_t logical)
{
    CooRunCursor cursor{};
    cursor.pos = unit * chunk;
    cursor.end = cursor.pos + chunk;
    if (cursor.end > logical) {
        cursor.end = logical;
    }
    return cursor;
}

/// 推进到下一段 run；没有剩余时返回 false。
__simt_callee__ __aicore__ inline bool CooCursorNext(CooRunCursor *cursor,
    uint64_t cols)
{
    // cols 为 0 时逻辑元素数也是 0，上面的区间已经是空的，这里只是让除法的
    // 非零前提摆在明面上。
    if (cols == 0 || cursor->pos >= cursor->end) {
        return false;
    }
    cursor->row = cursor->pos / cols;
    cursor->col = cursor->pos - cursor->row * cols;
    uint64_t runEnd = (cursor->row + 1) * cols;
    if (runEnd > cursor->end) {
        runEnd = cursor->end;
    }
    cursor->len = runEnd - cursor->pos;
    cursor->pos = runEnd;
    return true;
}

/// 计数与转换两遍共用的 unit 切分。两遍必须算得一模一样，否则一个 unit 读回的
/// prefix 就不对应它要写出的那段 dense，所以收在一处而不是各算一次。
struct UnitLayout {
    bool contiguous;         ///< run 在内存里是否连续
    bool flatCoo;            ///< 行主序 COO，按逻辑平铺切分
    uint64_t minorDim;       ///< 被切分的那一维
    uint64_t chunks;         ///< 每个 major 切成几段；flatCoo 时无意义
    uint64_t compressedStep; ///< 相邻元素的地址步长
    uint64_t logical;        ///< 逻辑元素总数
};

__simt_callee__ __aicore__ inline UnitLayout PlanUnits(uint64_t rows,
    uint64_t cols, uint64_t ld, uint64_t chunk, uint32_t format, uint32_t order)
{
    UnitLayout plan{};
    plan.contiguous = CompressedRunIsContiguous(format, order);
    // COO emits row-major, the very order CSR units are already walked in, so a
    // column-major dense reuses the CSR split into (row, chunk of columns):
    // consecutive threads then take consecutive rows and their loads at one
    // column are adjacent instead of `ld` apart. Row-major COO keeps the flat
    // logical tiles, whose runs are contiguous.
    plan.flatCoo = format == kFormatCoo && plan.contiguous;
    plan.minorDim = (format == kFormatCsr || format == kFormatCoo) ? cols : rows;
    plan.chunks = 0;
    if (!plan.flatCoo && chunk != 0) {
        plan.chunks = (plan.minorDim + chunk - 1) / chunk;
    }
    plan.compressedStep = plan.contiguous ? 1 : ld;
    plan.logical = rows * cols;
    return plan;
}

/// 非连续路径把 slot 映射成 unit：相邻线程取到相邻的 major，于是一个 warp 在同一
/// minor 位置上的访存落在相邻地址，而不是彼此隔着 ld。顺带给出该 unit 的 major。
__simt_callee__ __aicore__ inline uint64_t StridedUnitOfSlot(uint64_t slot,
    uint64_t majorDim, uint64_t chunks, uint64_t *major)
{
    *major = 0;
    if (majorDim == 0 || chunks == 0) {
        return 0;
    }
    const uint64_t chunkIndex = slot / majorDim;
    const uint64_t unit = (slot - chunkIndex * majorDim) * chunks + chunkIndex;
    *major = unit / chunks;
    return unit;
}

/// 非 COO 的 unit 在 minor 维上的区间 [begin, end)。
__simt_callee__ __aicore__ inline void ChunkSpan(uint64_t unit, uint64_t major,
    uint64_t chunks, uint64_t chunk, uint64_t minorDim, uint64_t *begin,
    uint64_t *end)
{
    *begin = (unit - major * chunks) * chunk;
    *end = *begin + chunk;
    if (*end > minorDim) {
        *end = minorDim;
    }
}

/// 连续 run 的计数路径：整个 warp 走同一批 unit，好让 asc_reduce_add 保持收敛。
template <typename ValueT>
__simt_callee__ __aicore__ inline void CountUnitsWarp(
    __gm__ const ValueT *dense, __gm__ uint64_t *counts, uint64_t cols,
    uint64_t ld, uint64_t unitCount, uint64_t chunk, uint64_t chunks,
    uint64_t minorDim, uint64_t logical, __gm__ uint32_t *bitmap,
    uint64_t bitmapWordsPerUnit, uint32_t format, uint64_t tid, uint64_t stride)
{
    const uint32_t lane = static_cast<uint32_t>(tid % kWarpSize);
    const uint64_t warpId = tid / kWarpSize;
    const uint64_t warpStride = stride / kWarpSize;
    for (uint64_t unit = warpId; unit < unitCount; unit += warpStride) {
        int32_t count = 0;
        __gm__ uint32_t *unitBitmap =
            bitmap == nullptr ? nullptr : bitmap + unit * bitmapWordsPerUnit;
        uint64_t bitmapBase = 0;
        if (format == kFormatCoo) {
            CooRunCursor run = CooCursorInit(unit, chunk, logical);
            while (CooCursorNext(&run, cols)) {
                __gm__ const ValueT *at = dense + run.row * ld + run.col;
                count += CountRunWarp<ValueT>(at, run.len, lane, unitBitmap,
                                              bitmapBase);
                // Window numbering continues across the runs of a unit so the
                // emit pass walks the same sequence.
                bitmapBase += RunBitmapWords<ValueT>(at, run.len);
            }
        } else {
            const uint64_t major = unit / chunks;
            uint64_t begin = 0;
            uint64_t end = 0;
            ChunkSpan(unit, major, chunks, chunk, minorDim, &begin, &end);
            count = CountRunWarp<ValueT>(dense + major * ld + begin,
                                         end - begin, lane, unitBitmap,
                                         bitmapBase);
        }
        if (lane == 0) {
            counts[unit] = static_cast<uint64_t>(count);
        }
    }
}

template <typename ValueT>
__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
CountUnits(__gm__ const ValueT *dense, __gm__ uint64_t *counts,
    uint64_t rows, uint64_t cols, uint64_t ld, uint64_t unitCount,
    uint64_t kMajorChunk, __gm__ uint32_t *bitmap, uint64_t bitmapWordsPerUnit,
    uint32_t format, uint32_t order, uint32_t numBlocks)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    const UnitLayout plan = PlanUnits(rows, cols, ld, kMajorChunk, format, order);
    const uint64_t chunks = plan.chunks;
    const uint64_t minorDim = plan.minorDim;
    const uint64_t compressedStep = plan.compressedStep;
    const bool contiguous = plan.contiguous;

    // Contiguous runs go through the warp-cooperative path; the whole warp must
    // iterate the same units so asc_reduce_add stays convergent.
    if (contiguous) {
        CountUnitsWarp<ValueT>(dense, counts, cols, ld, unitCount, kMajorChunk,
                               chunks, minorDim, plan.logical, bitmap,
                               bitmapWordsPerUnit, format, tid, stride);
        return;
    }
    // 只有非连续路径会走到这里，此时每个 major 都被切成了若干段。
    if (chunks == 0) {
        return;
    }

    // Strided runs keep one unit per thread, but consecutive threads take
    // consecutive majors so a warp's loads at a given minor index land on
    // adjacent addresses (the pattern nnz/arch35 uses for its row direction).
    // Only non-contiguous units reach here, so every format now splits by
    // (major, chunk of minor) -- COO included.
    const uint64_t majorDim = unitCount / chunks;
    for (uint64_t slot = tid; slot < unitCount; slot += stride) {
        uint64_t major = 0;
        const uint64_t unit = StridedUnitOfSlot(slot, majorDim, chunks, &major);
        uint64_t begin = 0;
        uint64_t end = 0;
        ChunkSpan(unit, major, chunks, kMajorChunk, minorDim, &begin, &end);
        counts[unit] = CountRun<ValueT>(
            dense + CompressedRunStart(contiguous, major, begin, ld),
            compressedStep, end - begin);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
ScanInclusive(__gm__ uint64_t *values, __gm__ uint64_t *chunkSums,
    uint64_t count, uint64_t chunkCount, uint32_t numBlocks)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    for (uint64_t chunk = tid; chunk < chunkCount; chunk += stride) {
        const uint64_t begin = chunk * kScanChunk;
        uint64_t end = begin + kScanChunk;
        if (end > count) {
            end = count;
        }
        uint64_t sum = 0;
        for (uint64_t i = begin; i < end; ++i) {
            sum += values[i];
            values[i] = sum;
        }
        if (chunkSums != nullptr) {
            chunkSums[chunk] = sum;
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
AddChunkBases(__gm__ uint64_t *values, __gm__ const uint64_t *chunkPrefix,
    uint64_t count, uint32_t numBlocks)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    for (uint64_t i = tid; i < count; i += stride) {
        const uint64_t chunk = i / kScanChunk;
        if (chunk > 0) {
            values[i] += chunkPrefix[chunk - 1];
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
InitializeHeader(__gm__ int32_t *status, __gm__ uint64_t *total)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        status[0] = kDeviceStatusSuccess;
        total[0] = 0;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
WriteTotal(__gm__ const uint64_t *prefix, __gm__ int32_t *status,
    __gm__ uint64_t *total, uint64_t count, uint64_t expectedNnz,
    uint32_t base, uint32_t offsetType, uint32_t validateExpected)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint64_t nnz = prefix[count - 1];
        total[0] = nnz;
        if ((offsetType == 0 && nnz + base > 0x7FFFFFFFULL) ||
            (validateExpected != 0 && nnz != expectedNnz)) {
            status[0] = kDeviceStatusInvalid;
        }
    }
}

template <typename OffsetT>
__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
WriteOffsets(__gm__ const uint64_t *prefix, __gm__ OffsetT *offsets,
    __gm__ int32_t *status, uint64_t majorDim, uint64_t chunks,
    uint32_t base, uint32_t numBlocks)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    if (tid == 0) {
        offsets[0] = static_cast<OffsetT>(base);
    }
    for (uint64_t major = tid; major < majorDim; major += stride) {
        const uint64_t value = prefix[(major + 1) * chunks - 1] + base;
        if (sizeof(OffsetT) == sizeof(int32_t) && value > 0x7FFFFFFFULL) {
            status[0] = kDeviceStatusInvalid;
        } else {
            offsets[major + 1] = static_cast<OffsetT>(value);
        }
    }
}

template <typename OffsetT>
__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
FillEmptyOffsets(__gm__ OffsetT *offsets, uint64_t count, uint32_t base,
    uint32_t numBlocks)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    for (uint64_t i = tid; i < count; i += stride) {
        offsets[i] = static_cast<OffsetT>(base);
    }
}

template <typename OffsetT>
__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
ValidateOffsets(__gm__ const uint64_t *prefix, __gm__ const OffsetT *offsets,
    __gm__ int32_t *status, uint64_t majorDim, uint64_t chunks,
    uint64_t nnz, uint32_t base, uint32_t numBlocks)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    for (uint64_t major = tid; major < majorDim; major += stride) {
        const int64_t encodedStart = static_cast<int64_t>(offsets[major]);
        const int64_t encodedEnd = static_cast<int64_t>(offsets[major + 1]);
        const uint64_t expectedStart = major == 0 ?
            0 : prefix[major * chunks - 1];
        const uint64_t expectedEnd = prefix[(major + 1) * chunks - 1];
        if (encodedStart < static_cast<int64_t>(base) ||
            encodedEnd < encodedStart) {
            status[0] = kDeviceStatusInvalid;
            continue;
        }
        const uint64_t start = static_cast<uint64_t>(encodedStart - base);
        const uint64_t end = static_cast<uint64_t>(encodedEnd - base);
        if (start != expectedStart || end != expectedEnd || end > nnz) {
            status[0] = kDeviceStatusInvalid;
        }
    }
}

__simt_callee__ __aicore__ inline uint64_t GetUnitOutputPosition(
    __gm__ const uint64_t *prefix, GM_ADDR offsets, uint64_t unit,
    uint64_t major, uint64_t chunks, uint32_t format, uint32_t base,
    uint32_t offsetType)
{
    if (format == kFormatCoo) {
        return unit == 0 ? 0 : prefix[unit - 1];
    }
    const int64_t encodedOffset = offsetType == 0 ?
        static_cast<int64_t>(((__gm__ const int32_t *)offsets)[major]) :
        ((__gm__ const int64_t *)offsets)[major];
    const uint64_t start = static_cast<uint64_t>(encodedOffset - base);
    if (chunks == 1) {
        // One unit per major: the unit prefix and the major prefix are the same
        // value and cancel, so the count pass is not needed to place this unit.
        return start;
    }
    return start + (unit == 0 ? 0 : prefix[unit - 1]) -
        (major == 0 ? 0 : prefix[major * chunks - 1]);
}

__simt_callee__ __aicore__ inline uint64_t NextMajorStart(
    GM_ADDR offsets, uint64_t major, uint32_t base, uint32_t offsetType)
{
    const int64_t encoded = offsetType == 0 ?
        static_cast<int64_t>(((__gm__ const int32_t *)offsets)[major + 1]) :
        ((__gm__ const int64_t *)offsets)[major + 1];
    return static_cast<uint64_t>(encoded - base);
}

/// 连续 run 的转换路径，与 CountUnitsWarp 走同一套 unit 切分，这样一个 unit 读回
/// 的 prefix 仍然对应它即将写出的那段 dense。
template <typename ValueT, typename IndexT>
__simt_callee__ __aicore__ inline void
ConvertUnitsWarp(__gm__ const ValueT *dense, __gm__ const uint64_t *prefix,
    __gm__ int32_t *status, GM_ADDR offsets, __gm__ IndexT *indices,
    __gm__ IndexT *rowIndices, __gm__ IndexT *colIndices, __gm__ ValueT *values,
    uint64_t cols, uint64_t ld, uint64_t unitCount, uint64_t chunk,
    uint64_t chunks, uint64_t minorDim, uint64_t logical, uint64_t tilingNnz,
    __gm__ const uint32_t *bitmap, uint64_t bitmapWordsPerUnit, uint32_t format,
    uint32_t base, uint32_t offsetType, uint32_t singlePass, bool flatCoo,
    uint64_t tid, uint64_t stride)
{
    const uint32_t lane = static_cast<uint32_t>(tid % kWarpSize);
    const uint64_t warpId = tid / kWarpSize;
    const uint64_t warpStride = stride / kWarpSize;
    for (uint64_t unit = warpId; unit < unitCount; unit += warpStride) {
        const uint64_t major = flatCoo ? 0 : unit / chunks;
        uint64_t out = GetUnitOutputPosition(prefix, offsets, unit, major,
            chunks, format, base, offsetType);
        __gm__ const uint32_t *unitBitmap =
            bitmap == nullptr ? nullptr : bitmap + unit * bitmapWordsPerUnit;
        uint64_t bitmapBase = 0;
        if (format == kFormatCoo) {
            CooRunCursor run = CooCursorInit(unit, chunk, logical);
            while (CooCursorNext(&run, cols)) {
                __gm__ const ValueT *at = dense + run.row * ld + run.col;
                out = EmitRunWarp<ValueT, IndexT, true>(
                    at, run.len, run.col, base, indices, rowIndices, colIndices,
                    values, out, lane, run.row, tilingNnz, unitBitmap,
                    bitmapBase);
                bitmapBase += RunBitmapWords<ValueT>(at, run.len);
            }
            continue;
        }
        uint64_t begin = 0;
        uint64_t end = 0;
        ChunkSpan(unit, major, chunks, chunk, minorDim, &begin, &end);
        const uint64_t limit = singlePass
            ? NextMajorStart(offsets, major, base, offsetType)
            : tilingNnz;
        const uint64_t emitted = EmitRunWarp<ValueT, IndexT, false>(
            dense + major * ld + begin, end - begin, begin, base, indices,
            rowIndices, colIndices, values, out, lane, 0, limit, unitBitmap,
            bitmapBase) - out;
        // 单遍转换下每个 major 的产出必须正好填满它那段 offsets，否则说明
        // count 与 emit 两遍看到的非零不一致。
        if (singlePass && lane == 0 && out + emitted != limit) {
            status[0] = kDeviceStatusInvalid;
        }
    }
}

template <typename ValueT, typename IndexT>
__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
ConvertUnits(__gm__ const ValueT *dense, __gm__ const uint64_t *prefix,
    __gm__ int32_t *status, GM_ADDR offsets, __gm__ IndexT *indices,
    __gm__ IndexT *rowIndices, __gm__ IndexT *colIndices,
    __gm__ ValueT *values, uint64_t rows, uint64_t cols, uint64_t ld,
    uint64_t unitCount, uint64_t kMajorChunk, uint64_t tilingNnz,
    __gm__ const uint32_t *bitmap, uint64_t bitmapWordsPerUnit,
    uint32_t format, uint32_t order, uint32_t base, uint32_t offsetType,
    uint32_t singlePass, uint32_t numBlocks)
{
    if (status[0] != kDeviceStatusSuccess) {
        return;
    }
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    const UnitLayout plan = PlanUnits(rows, cols, ld, kMajorChunk, format, order);
    const uint64_t chunks = plan.chunks;
    const uint64_t minorDim = plan.minorDim;
    const uint64_t compressedStep = plan.compressedStep;
    const bool contiguous = plan.contiguous;

    if (plan.contiguous) {
        ConvertUnitsWarp<ValueT, IndexT>(dense, prefix, status, offsets, indices,
            rowIndices, colIndices, values, cols, ld, unitCount, kMajorChunk,
            chunks, minorDim, plan.logical, tilingNnz, bitmap,
            bitmapWordsPerUnit, format, base, offsetType, singlePass,
            plan.flatCoo, tid, stride);
        return;
    }
    if (chunks == 0) {
        return;
    }

    const uint64_t majorDim = unitCount / chunks;
    for (uint64_t slot = tid; slot < unitCount; slot += stride) {
        uint64_t major = 0;
        const uint64_t unit = StridedUnitOfSlot(slot, majorDim, chunks, &major);
        // The count pass already knows this unit is empty, and the strided path
        // has no bitmap to skip empty regions with, so without this it would
        // re-read the unit's whole stretch of dense to emit nothing. An
        // all-zero input reads the dense twice instead of once.
        if (prefix[unit] == (unit == 0 ? 0 : prefix[unit - 1])) {
            continue;
        }
        const uint64_t out = GetUnitOutputPosition(prefix, offsets, unit, major,
            chunks, format, base, offsetType);
        uint64_t begin = 0;
        uint64_t end = 0;
        ChunkSpan(unit, major, chunks, kMajorChunk, minorDim, &begin, &end);
        __gm__ const ValueT *run =
            dense + CompressedRunStart(contiguous, major, begin, ld);
        if (format == kFormatCoo) {
            // The unit is a stretch of a single row, which is exactly the
            // shape EmitCooRun walks.
            (void)EmitCooRun<ValueT, IndexT>(run, compressedStep, end - begin,
                major, begin, base, rowIndices, colIndices, values, out);
        } else {
            (void)EmitCompressedRun<ValueT, IndexT>(run, compressedStep,
                end - begin, begin, base, indices, values, out);
        }
        (void)tilingNnz;
    }
}

template <typename ValueT, typename IndexT>
__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
ConvertBell(__gm__ const ValueT *dense, __gm__ const IndexT *ellColInd,
    __gm__ ValueT *values, uint64_t rows, uint64_t cols, uint64_t ld,
    uint64_t blockSize, uint64_t ellCols, uint32_t order, uint32_t base,
    uint32_t numBlocks)
{
    const uint64_t blockRows = CeilDiv(rows, blockSize);
    const uint64_t slots = CeilDiv(ellCols, blockSize);
    const uint64_t blockColumns = CeilDiv(cols, blockSize);
    const uint64_t tasks = blockRows * slots;
    const uint64_t blockArea = blockSize * blockSize;
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t lane = static_cast<uint32_t>(tid % kWarpSize);
    const uint64_t warpId = tid / kWarpSize;
    const uint64_t totalWarps =
        (static_cast<uint64_t>(numBlocks) * blockDim.x) / kWarpSize;
    // Small slots use one warp. Large slots are split across up to eight
    // warps: P-03 has only 112 64x64 slots, so one-warp-per-slot activated just
    // 112/896 warps and made each active warp execute 128 loop iterations.
    // Eight warps per slot expose exactly 896 warps and cut that loop to 16.
    const uint32_t warpsPerTask = blockArea >= 2048 ? 8 :
        (blockArea >= 1024 ? 4 : (blockArea >= 512 ? 2 : 1));
    const uint64_t taskWarp = warpId / warpsPerTask;
    const uint32_t warpInTask =
        static_cast<uint32_t>(warpId % warpsPerTask);
    const uint64_t taskStride = totalWarps / warpsPerTask;
    // Lanes stripe the slot so stores stay coalesced while the column-strided
    // gathers reuse the few lines the slot spans.
    // Striping the other way (coalesced gathers, blockSize-strided stores) was
    // measured slower on every scenario — P-03 blockSize=64 went 339us -> 453us
    // — because the partial-line stores cost more than the gathers save.
    for (uint64_t task = taskWarp; task < tasks; task += taskStride) {
        const uint64_t blockRow = task / slots;
        const int64_t encoded = static_cast<int64_t>(ellColInd[task]);
        const bool valid = encoded >= static_cast<int64_t>(base) &&
            static_cast<uint64_t>(encoded - base) < blockColumns;
        const uint64_t blockCol = valid ? static_cast<uint64_t>(encoded - base) : 0;
        const uint64_t valueBase = task * blockArea;
        const uint64_t rowBase = blockRow * blockSize;
        const uint64_t colBase = blockCol * blockSize;
        for (uint64_t slot =
                 static_cast<uint64_t>(warpInTask) * kWarpSize + lane;
             slot < blockArea;
             slot += static_cast<uint64_t>(warpsPerTask) * kWarpSize) {
            const uint64_t innerCol = slot / blockSize;
            const uint64_t innerRow = slot - innerCol * blockSize;
            const uint64_t row = rowBase + innerRow;
            const uint64_t col = colBase + innerCol;
            // Tail blocks keep their full blockSize*blockSize slot; the
            // logical elements past rows/cols get positive zero.
            const bool inRange = valid && row < rows && col < cols;
            values[valueBase + slot] = inRange ?
                dense[DensePosition(row, col, ld, order)] :
                static_cast<ValueT>(0);
        }
    }
}

template <typename ValueT>
__aicore__ inline void DispatchCount(
    GM_ADDR dense, GM_ADDR level0, GM_ADDR bitmap,
    const DenseToSparseTilingData &tiling)
{
    asc_vf_call<CountUnits<ValueT>>(dim3{kDenseToSparseThreads},
        (__gm__ const ValueT *)dense, (__gm__ uint64_t *)level0, tiling.rows,
        tiling.cols, tiling.ld, tiling.unitCount, tiling.majorChunk,
        (__gm__ uint32_t *)bitmap, tiling.bitmapWordsPerUnit,
        tiling.format, tiling.order, tiling.numBlocks);
}

template <typename ValueT, typename IndexT>
__aicore__ inline void DispatchConvert(
    GM_ADDR dense, GM_ADDR prefix, GM_ADDR status, GM_ADDR offsets,
    GM_ADDR indices, GM_ADDR rowIndices, GM_ADDR colIndices, GM_ADDR values,
    GM_ADDR bitmap, const DenseToSparseTilingData &tiling)
{
    asc_vf_call<ConvertUnits<ValueT, IndexT>>(dim3{kDenseToSparseThreads},
        (__gm__ const ValueT *)dense, (__gm__ const uint64_t *)prefix,
        (__gm__ int32_t *)status, offsets,
        (__gm__ IndexT *)indices, (__gm__ IndexT *)rowIndices,
        (__gm__ IndexT *)colIndices, (__gm__ ValueT *)values, tiling.rows,
        tiling.cols, tiling.ld, tiling.unitCount, tiling.majorChunk,
        tiling.nnz, (__gm__ const uint32_t *)bitmap, tiling.bitmapWordsPerUnit,
        tiling.format, tiling.order, tiling.base, tiling.offsetType,
        tiling.singlePassConvert, tiling.numBlocks);
}

template <typename ValueT>
__aicore__ inline void DispatchConvertByIndex(
    GM_ADDR dense, GM_ADDR prefix, GM_ADDR status, GM_ADDR offsets,
    GM_ADDR indices, GM_ADDR rowIndices, GM_ADDR colIndices, GM_ADDR values,
    GM_ADDR bitmap, const DenseToSparseTilingData &tiling)
{
    if (tiling.indexType == 0) {
        DispatchConvert<ValueT, int32_t>(dense, prefix, status, offsets,
            indices, rowIndices, colIndices, values, bitmap, tiling);
    } else {
        DispatchConvert<ValueT, int64_t>(dense, prefix, status, offsets,
            indices, rowIndices, colIndices, values, bitmap, tiling);
    }
}

template <typename ValueT, typename IndexT>
__aicore__ inline void DispatchBell(
    GM_ADDR dense, GM_ADDR ellColInd, GM_ADDR values,
    const DenseToSparseTilingData &tiling)
{
    asc_vf_call<ConvertBell<ValueT, IndexT>>(dim3{kDenseToSparseThreads},
        (__gm__ const ValueT *)dense, (__gm__ const IndexT *)ellColInd,
        (__gm__ ValueT *)values, tiling.rows, tiling.cols, tiling.ld,
        tiling.ellBlockSize, tiling.ellCols, tiling.order, tiling.base,
        tiling.numBlocks);
}

template <typename ValueT>
__aicore__ inline void DispatchBellByIndex(
    GM_ADDR dense, GM_ADDR ellColInd, GM_ADDR values,
    const DenseToSparseTilingData &tiling)
{
    if (tiling.indexType == 0) {
        DispatchBell<ValueT, int32_t>(dense, ellColInd, values, tiling);
    } else {
        DispatchBell<ValueT, int64_t>(dense, ellColInd, values, tiling);
    }
}

uint64_t Align32(uint64_t value)
{
    return (value + kAlign - 1) & ~(kAlign - 1);
}

uint64_t NextLevelCount(uint64_t count)
{
    return (count + kScanChunk - 1) / kScanChunk;
}

__aicore__ inline void GetCompressedLayout(
    const DenseToSparseTilingData &tiling, uint64_t &major, uint64_t &chunks)
{
    major = tiling.format == kFormatCsr ? tiling.rows : tiling.cols;
    const uint64_t minor =
        tiling.format == kFormatCsr ? tiling.cols : tiling.rows;
    chunks = (minor + tiling.majorChunk - 1) / tiling.majorChunk;
}

} // namespace

extern "C" __global__ __aicore__ void densetosparse_count_kernel(
    GM_ADDR dense, GM_ADDR level0, GM_ADDR bitmap,
    DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.elementBytes == 1) {
        DispatchCount<uint8_t>(dense, level0, bitmap, tiling);
    } else if (tiling.elementBytes == 2) {
        DispatchCount<uint16_t>(dense, level0, bitmap, tiling);
    } else if (tiling.elementBytes == 8) {
        DispatchCount<uint64_t>(dense, level0, bitmap, tiling);
    } else {
        DispatchCount<uint32_t>(dense, level0, bitmap, tiling);
    }
}

extern "C" __global__ __aicore__ void densetosparse_header_kernel(
    GM_ADDR status, GM_ADDR total)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<InitializeHeader>(dim3{kDenseToSparseThreads},
        (__gm__ int32_t *)status, (__gm__ uint64_t *)total);
}

extern "C" __global__ __aicore__ void densetosparse_scan_kernel(
    GM_ADDR values, GM_ADDR chunkSums, uint64_t count, uint64_t chunkCount,
    uint32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<ScanInclusive>(dim3{kDenseToSparseThreads},
        (__gm__ uint64_t *)values, (__gm__ uint64_t *)chunkSums, count,
        chunkCount, numBlocks);
}

extern "C" __global__ __aicore__ void densetosparse_add_base_kernel(
    GM_ADDR values, GM_ADDR chunkPrefix, uint64_t count, uint32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<AddChunkBases>(dim3{kDenseToSparseThreads},
        (__gm__ uint64_t *)values, (__gm__ const uint64_t *)chunkPrefix,
        count, numBlocks);
}

extern "C" __global__ __aicore__ void densetosparse_offsets_kernel(
    GM_ADDR prefix, GM_ADDR offsets, GM_ADDR status,
    DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint64_t major = 0;
    uint64_t chunks = 0;
    GetCompressedLayout(tiling, major, chunks);
    if (tiling.offsetType == 0) {
        asc_vf_call<WriteOffsets<int32_t>>(dim3{kDenseToSparseThreads},
            (__gm__ const uint64_t *)prefix, (__gm__ int32_t *)offsets,
            (__gm__ int32_t *)status, major, chunks, tiling.base,
            tiling.numBlocks);
    } else {
        asc_vf_call<WriteOffsets<int64_t>>(dim3{kDenseToSparseThreads},
            (__gm__ const uint64_t *)prefix, (__gm__ int64_t *)offsets,
            (__gm__ int32_t *)status, major, chunks, tiling.base,
            tiling.numBlocks);
    }
}

extern "C" __global__ __aicore__ void densetosparse_total_kernel(
    GM_ADDR prefix, GM_ADDR status, GM_ADDR total, uint64_t count,
    uint64_t expectedNnz, uint32_t base, uint32_t offsetType,
    uint32_t validateExpected)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<WriteTotal>(dim3{kDenseToSparseThreads},
        (__gm__ const uint64_t *)prefix, (__gm__ int32_t *)status,
        (__gm__ uint64_t *)total, count, expectedNnz, base, offsetType,
        validateExpected);
}

extern "C" __global__ __aicore__ void densetosparse_validate_offsets_kernel(
    GM_ADDR prefix, GM_ADDR offsets, GM_ADDR status,
    DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint64_t major = 0;
    uint64_t chunks = 0;
    GetCompressedLayout(tiling, major, chunks);
    if (tiling.offsetType == 0) {
        asc_vf_call<ValidateOffsets<int32_t>>(dim3{kDenseToSparseThreads},
            (__gm__ const uint64_t *)prefix, (__gm__ const int32_t *)offsets,
            (__gm__ int32_t *)status, major, chunks, tiling.nnz, tiling.base,
            tiling.numBlocks);
    } else {
        asc_vf_call<ValidateOffsets<int64_t>>(dim3{kDenseToSparseThreads},
            (__gm__ const uint64_t *)prefix, (__gm__ const int64_t *)offsets,
            (__gm__ int32_t *)status, major, chunks, tiling.nnz, tiling.base,
            tiling.numBlocks);
    }
}

/// Single-pass Convert cannot compare offsets against a recount, so it checks
/// what the offsets themselves have to satisfy: monotonic, starting at base,
/// each span within the row, and a total that matches matB.nnz. The per-span
/// element count is then verified while emitting.
template <typename OffsetT>
__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
CheckOffsetSpans(__gm__ const OffsetT *offsets, __gm__ int32_t *status,
    __gm__ uint64_t *total, uint64_t majorDim, uint64_t minorDim, uint64_t nnz,
    uint32_t base, uint32_t numBlocks)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    if (tid == 0) {
        const int64_t last = static_cast<int64_t>(offsets[majorDim]);
        total[0] = static_cast<uint64_t>(last - static_cast<int64_t>(base));
        if (static_cast<int64_t>(offsets[0]) != static_cast<int64_t>(base) ||
            last - static_cast<int64_t>(base) != static_cast<int64_t>(nnz)) {
            status[0] = kDeviceStatusInvalid;
        }
    }
    for (uint64_t major = tid; major < majorDim; major += stride) {
        const int64_t begin = static_cast<int64_t>(offsets[major]);
        const int64_t end = static_cast<int64_t>(offsets[major + 1]);
        const int64_t span = end - begin;
        if (span < 0 || static_cast<uint64_t>(span) > minorDim ||
            begin < static_cast<int64_t>(base) ||
            static_cast<uint64_t>(end - static_cast<int64_t>(base)) > nnz) {
            status[0] = kDeviceStatusInvalid;
        }
    }
}

extern "C" __global__ __aicore__ void densetosparse_span_check_kernel(
    GM_ADDR offsets, GM_ADDR status, GM_ADDR total,
    DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint64_t major = tiling.format == kFormatCsr ? tiling.rows : tiling.cols;
    const uint64_t minor = tiling.format == kFormatCsr ? tiling.cols : tiling.rows;
    if (tiling.offsetType == 0) {
        asc_vf_call<CheckOffsetSpans<int32_t>>(dim3{kDenseToSparseThreads},
            (__gm__ const int32_t *)offsets, (__gm__ int32_t *)status,
            (__gm__ uint64_t *)total, major, minor, tiling.nnz, tiling.base,
            tiling.numBlocks);
    } else {
        asc_vf_call<CheckOffsetSpans<int64_t>>(dim3{kDenseToSparseThreads},
            (__gm__ const int64_t *)offsets, (__gm__ int32_t *)status,
            (__gm__ uint64_t *)total, major, minor, tiling.nnz, tiling.base,
            tiling.numBlocks);
    }
}

extern "C" __global__ __aicore__ void densetosparse_empty_offsets_kernel(
    GM_ADDR offsets, DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint64_t major = tiling.format == kFormatCsr ?
        tiling.rows : tiling.cols;
    if (tiling.offsetType == 0) {
        asc_vf_call<FillEmptyOffsets<int32_t>>(dim3{kDenseToSparseThreads},
            (__gm__ int32_t *)offsets, major + 1, tiling.base,
            tiling.numBlocks);
    } else {
        asc_vf_call<FillEmptyOffsets<int64_t>>(dim3{kDenseToSparseThreads},
            (__gm__ int64_t *)offsets, major + 1, tiling.base,
            tiling.numBlocks);
    }
}

extern "C" __global__ __aicore__ void densetosparse_convert_kernel(
    GM_ADDR dense, GM_ADDR prefix, GM_ADDR status, GM_ADDR offsets,
    GM_ADDR indices, GM_ADDR rowIndices, GM_ADDR colIndices, GM_ADDR values,
    GM_ADDR bitmap, DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.elementBytes == 1) {
        DispatchConvertByIndex<uint8_t>(dense, prefix, status, offsets, indices,
            rowIndices, colIndices, values, bitmap, tiling);
    } else if (tiling.elementBytes == 2) {
        DispatchConvertByIndex<uint16_t>(dense, prefix, status, offsets,
            indices, rowIndices, colIndices, values, bitmap, tiling);
    } else if (tiling.elementBytes == 8) {
        DispatchConvertByIndex<uint64_t>(dense, prefix, status, offsets,
            indices, rowIndices, colIndices, values, bitmap, tiling);
    } else {
        DispatchConvertByIndex<uint32_t>(dense, prefix, status, offsets,
            indices, rowIndices, colIndices, values, bitmap, tiling);
    }
}

extern "C" __global__ __aicore__ void densetosparse_bell_kernel(
    GM_ADDR dense, GM_ADDR ellColInd, GM_ADDR values,
    DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.elementBytes == 1) {
        DispatchBellByIndex<uint8_t>(dense, ellColInd, values, tiling);
    } else if (tiling.elementBytes == 2) {
        DispatchBellByIndex<uint16_t>(dense, ellColInd, values, tiling);
    } else if (tiling.elementBytes == 8) {
        DispatchBellByIndex<uint64_t>(dense, ellColInd, values, tiling);
    } else {
        DispatchBellByIndex<uint32_t>(dense, ellColInd, values, tiling);
    }
}

static void LaunchCountAndScan(GM_ADDR dense, GM_ADDR workspace,
    uint32_t numBlocks, const DenseToSparseTilingData &tiling, void *stream)
{
    GM_ADDR status = workspace + tiling.statusOffset;
    GM_ADDR total = workspace + tiling.nnzOffset;
    densetosparse_header_kernel<<<1, nullptr, stream>>>(status, total);
    GM_ADDR level0 = workspace + tiling.level0Offset;
    GM_ADDR bitmap = tiling.bitmapMode == 1 ?
        workspace + tiling.bitmapOffset : nullptr;
    densetosparse_count_kernel<<<numBlocks, nullptr, stream>>>(
        dense, level0, bitmap, tiling);
    GM_ADDR levels[16] = {};
    uint64_t counts[16] = {};
    uint32_t levelCount = 1;
    levels[0] = level0;
    counts[0] = tiling.unitCount;
    uint64_t offset = tiling.level0Offset +
        Align32(tiling.unitCount * sizeof(uint64_t));
    while (counts[levelCount - 1] > kScanChunk && levelCount < 16) {
        counts[levelCount] = NextLevelCount(counts[levelCount - 1]);
        levels[levelCount] = workspace + offset;
        offset += Align32(counts[levelCount] * sizeof(uint64_t));
        ++levelCount;
    }
    for (uint32_t level = 0; level < levelCount; ++level) {
        const uint64_t chunks = NextLevelCount(counts[level]);
        GM_ADDR next = level + 1 < levelCount ? levels[level + 1] : nullptr;
        densetosparse_scan_kernel<<<numBlocks, nullptr, stream>>>(
            levels[level], next, counts[level], chunks, numBlocks);
    }
    for (uint32_t level = levelCount - 1; level > 0; --level) {
        densetosparse_add_base_kernel<<<numBlocks, nullptr, stream>>>(
            levels[level - 1], levels[level], counts[level - 1], numBlocks);
    }
}

void densetosparse_analysis_kernel_do(
    GM_ADDR dense, GM_ADDR offsets, GM_ADDR workspace, uint32_t numBlocks,
    const DenseToSparseTilingData &tiling, void *stream)
{
    if (tiling.unitCount == 0) {
        if (tiling.format != kFormatCoo) {
            densetosparse_empty_offsets_kernel<<<numBlocks, nullptr, stream>>>(
                offsets, tiling);
        }
        return;
    }
    LaunchCountAndScan(dense, workspace, numBlocks, tiling, stream);
    GM_ADDR level0 = workspace + tiling.level0Offset;
    GM_ADDR status = workspace + tiling.statusOffset;
    GM_ADDR total = workspace + tiling.nnzOffset;
    if (tiling.format != kFormatCoo) {
        densetosparse_offsets_kernel<<<numBlocks, nullptr, stream>>>(
            level0, offsets, status, tiling);
    }
    densetosparse_total_kernel<<<1, nullptr, stream>>>(
        level0, status, total, tiling.unitCount, 0, tiling.base,
        tiling.offsetType, 0);
}

void densetosparse_convert_kernel_do(
    GM_ADDR dense, GM_ADDR workspace, GM_ADDR offsets, GM_ADDR indices,
    GM_ADDR rowIndices, GM_ADDR colIndices, GM_ADDR values, GM_ADDR ellColInd,
    uint32_t numBlocks, const DenseToSparseTilingData &tiling, void *stream)
{
    if (tiling.format == kFormatBell) {
        densetosparse_bell_kernel<<<numBlocks, nullptr, stream>>>(
            dense, ellColInd, values, tiling);
        return;
    }
    GM_ADDR status = workspace + tiling.statusOffset;
    GM_ADDR total = workspace + tiling.nnzOffset;
    GM_ADDR level0 = workspace + tiling.level0Offset;
    if (tiling.singlePassConvert != 0) {
        densetosparse_header_kernel<<<1, nullptr, stream>>>(status, total);
        densetosparse_span_check_kernel<<<numBlocks, nullptr, stream>>>(
            offsets, status, total, tiling);
        densetosparse_convert_kernel<<<numBlocks, nullptr, stream>>>(
            dense, level0, status, offsets, indices, rowIndices,
            colIndices, values, nullptr, tiling);
        return;
    }
    LaunchCountAndScan(dense, workspace, numBlocks, tiling, stream);
    densetosparse_total_kernel<<<1, nullptr, stream>>>(
        level0, status, total, tiling.unitCount, tiling.nnz, tiling.base,
        tiling.offsetType, 1);
    if (tiling.format != kFormatCoo) {
        densetosparse_validate_offsets_kernel<<<numBlocks, nullptr, stream>>>(
            level0, offsets, status, tiling);
    }
    densetosparse_convert_kernel<<<numBlocks, nullptr, stream>>>(
        dense, level0, status, offsets, indices, rowIndices, colIndices, values,
        tiling.bitmapMode == 1 ? workspace + tiling.bitmapOffset : nullptr,
        tiling);
}
