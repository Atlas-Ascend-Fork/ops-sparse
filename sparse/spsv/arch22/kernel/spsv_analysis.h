/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_SPSV_ANALYSIS_H_
#define SPSV_ARCH22_SPSV_ANALYSIS_H_
#include "spsv_kernel_memory.h"

namespace spsv_a2 {
using namespace AscendC;

class Analysis {
public:
    __aicore__ inline Analysis(TPipe &pipe, GM_ADDR ptr, GM_ADDR idx,
        GM_ADDR values, GM_ADDR workspace, SpsvA2Tiling tiling) : t_(tiling)
    {
        int64_t ptrCount = t_.format == 2 ? t_.sourceNnz :
                                            (t_.format == 3 ? static_cast<int64_t>(t_.slices) + 1 : t_.m + 1);
        ptr_.Init(pipe, ptr, ptrCount);
        idx_.Init(pipe, idx, t_.sourceNnz);
        source_.Init(pipe, values, t_.sourceNnz * t_.components);
        row_.Init(pipe, workspace + t_.rowOffset, t_.m + 1);
        col_.Init(pipe, workspace + t_.colOffset, t_.sourceNnz);
        value_.Init(pipe, workspace + t_.valueOffset, t_.sourceNnz * t_.components);
        perm_.Init(pipe, workspace + t_.permOffset, t_.sourceNnz);
        scratch_.Init(pipe, workspace + t_.scratchOffset, t_.m);
        status_.Init(pipe, workspace, 8);
        directRow_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace + t_.rowOffset), t_.m + 1);
        directCol_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace + t_.colOffset), t_.sourceNnz);
        directValue_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + t_.valueOffset),
            t_.sourceNnz * t_.components);
    }
    __aicore__ inline void Run()
    {
        if (t_.format == 0 && TryDirectCsr()) return;
        for (int64_t row = 0; row <= t_.m; ++row) row_.Write(row, 0);
        if (Visit(false)) {
            int32_t running = 0;
            for (int64_t row = 0; row < t_.m; ++row) {
                int32_t count = row_.Read(row + 1);
                row_.Write(row, running);
                scratch_.Write(row, running);
                running += count;
            }
            row_.Write(t_.m, running);
            Visit(true);
            status_.Write(1, running);
        } else {
            status_.Write(1, 0);
        }
        status_.Write(0, error_);
        status_.Write(2, diagonalMatrix_ ? 1 : 0);
        row_.Flush();
        col_.Flush();
        value_.Flush();
        perm_.Flush();
        scratch_.Flush();
        status_.Flush();
    }
private:
    // A common diagonal layout has exactly one entry per row. Verify both
    // arrays element by element, rather than inferring this from m == nnz.
    __aicore__ inline bool ValidateIdentityDiagonal()
    {
        for (int64_t p = 0; p < t_.m;) {
            uint32_t pointerAvailable = 0;
            uint32_t indexAvailable = 0;
            auto ends = ptr_.ReadSpan(p + 1, pointerAvailable);
            auto columns = idx_.ReadSpan(p, indexAvailable);
            uint32_t count = pointerAvailable < indexAvailable ? pointerAvailable : indexAvailable;
            for (uint32_t i = 0; i < count; ++i) {
                int64_t expected = p + i + t_.base;
                if (static_cast<int64_t>(ends.GetValue(i)) != expected + 1 ||
                    static_cast<int64_t>(columns.GetValue(i)) != expected) return false;
            }
            p += count;
        }
        return true;
    }
    // Validate every pointer and column on every analysis. Only a CSR already
    // contained in the requested triangle can retain its row/entry order.
    // Transposition is order-preserving only for a fully diagonal matrix.
    // Filtering, other transposes and malformed inputs use the general path.
    struct CsrIndexWindow {
        LocalTensor<int32_t> indices;
        int64_t start = 0;
        int64_t end = 0;
    };
    __aicore__ inline bool ValidateDirectRow(int64_t row, int64_t end, int64_t &begin,
        bool &diagonal, CsrIndexWindow &window)
    {
        if (end < begin || end > t_.sourceNnz) return false;
        while (begin < end) {
            if (begin >= window.end) {
                uint32_t available = 0;
                window.indices = idx_.ReadSpan(begin, available);
                window.start = begin;
                window.end = begin + available;
            }
            uint32_t count = static_cast<uint32_t>(end - begin);
            if (count > window.end - begin) count = static_cast<uint32_t>(window.end - begin);
            auto span = window.indices[static_cast<uint32_t>(begin - window.start)];
            for (uint32_t i = 0; i < count; ++i) {
                int64_t col = static_cast<int64_t>(span.GetValue(i)) - t_.base;
                if (col < 0 || col >= t_.m || (t_.lower ? col > row : col < row) ||
                    (t_.transpose && col != row)) return false;
                if (col != row) diagonal = false;
            }
            begin += count;
        }
        return true;
    }
    __aicore__ inline bool ValidateDirectCsr(bool &diagonal)
    {
        if (ptr_.Read(0) != t_.base ||
            static_cast<int64_t>(ptr_.Read(t_.m)) - t_.base != t_.sourceNnz) return false;
        if (t_.sourceNnz == t_.m && ValidateIdentityDiagonal()) return true;
        int64_t begin = 0;
        CsrIndexWindow window;
        for (int64_t row = 0; row < t_.m;) {
            uint32_t pointerAvailable = 0;
            auto ends = ptr_.ReadSpan(row + 1, pointerAvailable);
            uint32_t rows = static_cast<uint32_t>(t_.m - row);
            if (rows > pointerAvailable) rows = pointerAvailable;
            for (uint32_t r = 0; r < rows; ++r, ++row) {
                int64_t end = static_cast<int64_t>(ends.GetValue(r)) - t_.base;
                if (!ValidateDirectRow(row, end, begin, diagonal, window)) return false;
            }
        }
        return true;
    }
    __aicore__ inline void CopyDirectRows()
    {
        for (int64_t row = 0; row <= t_.m;) {
            uint32_t srcAvailable = 0;
            uint32_t dstAvailable = 0;
            auto src = ptr_.ReadSpan(row, srcAvailable);
            if (t_.base == 0) {
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopyPad(directRow_[row], src, DataCopyExtParams{
                    1, static_cast<uint32_t>(srcAvailable * sizeof(int32_t)), 0, 0, 0});
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                row += srcAvailable;
                continue;
            }
            auto dst = row_.WritableSpan(row, dstAvailable);
            uint32_t count = srcAvailable < dstAvailable ? srcAvailable : dstAvailable;
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Adds(dst, src, static_cast<int32_t>(-t_.base), static_cast<int32_t>(count));
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            row += count;
        }
        row_.Flush();
    }
    __aicore__ inline void CopyDirectEntries()
    {
        for (int64_t p = 0; p < t_.sourceNnz;) {
            uint32_t idxAvailable = 0;
            uint32_t valueAvailable = 0;
            uint32_t permAvailable = 0;
            auto indices = idx_.ReadSpan(p, idxAvailable);
            auto values = source_.ReadSpan(p * t_.components, valueAvailable);
            auto permutation = perm_.WritableSpan(p, permAvailable);
            uint32_t count = valueAvailable / t_.components;
            if (count > idxAvailable) count = idxAvailable;
            if (count > permAvailable) count = permAvailable;
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            CreateVecIndex(permutation, static_cast<int32_t>(p), count);
            if (t_.base != 0) Adds(indices, indices, static_cast<int32_t>(-t_.base), static_cast<int32_t>(count));
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if (t_.conjugate) {
                for (uint32_t i = 0; i < count; ++i) {
                    values.SetValue(i * 2 + 1, -values.GetValue(i * 2 + 1));
                }
            }
            // Chunk starts are multiples of 1024/2048 entries, hence UB sources
            // are 32-byte aligned. DMA tails write only the valid byte count.
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(directCol_[p], indices, DataCopyExtParams{
                1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0});
            DataCopyPad(directValue_[p * t_.components], values, DataCopyExtParams{
                1, static_cast<uint32_t>(count * t_.components * sizeof(float)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            p += count;
        }
        perm_.Flush();
    }
    __aicore__ inline bool TryDirectCsr()
    {
        bool diagonal = true;
        if (!ValidateDirectCsr(diagonal)) return false;
        CopyDirectRows();
        CopyDirectEntries();
        status_.Write(0, 0);
        status_.Write(1, static_cast<int32_t>(t_.sourceNnz));
        status_.Write(2, diagonal ? 1 : 0);
        status_.Flush();
        return true;
    }
    __aicore__ inline bool Emit(int64_t row, int64_t col, int64_t source, bool scatter)
    {
        if (row < 0 || row >= t_.m || col < 0 || col >= t_.m) {
            error_ = 1;
            return false;
        }
        if ((t_.lower && col > row) || (!t_.lower && col < row)) return true;
        if (row != col) diagonalMatrix_ = false;
        int32_t targetRow = t_.transpose ? col : row;
        int32_t targetCol = t_.transpose ? row : col;
        if (!scatter) {
            row_.Write(targetRow + 1, row_.Read(targetRow + 1) + 1);
            return true;
        }
        int32_t position = scratch_.Read(targetRow);
        scratch_.Write(targetRow, position + 1);
        col_.Write(position, targetCol);
        perm_.Write(position, static_cast<int32_t>(source));
        value_.Write(static_cast<int64_t>(position) * t_.components,
            source_.Read(source * t_.components));
        if (t_.components == 2) {
            float imag = source_.Read(source * 2 + 1);
            value_.Write(static_cast<int64_t>(position) * 2 + 1,
                t_.conjugate ? -imag : imag);
        }
        return true;
    }
    __aicore__ inline bool VisitCoo(bool scatter)
    {
        for (int64_t p = 0; p < t_.sourceNnz; ++p) {
            if (!Emit(static_cast<int64_t>(ptr_.Read(p)) - t_.base,
                static_cast<int64_t>(idx_.Read(p)) - t_.base, p, scatter)) return false;
        }
        return true;
    }
    __aicore__ inline bool VisitSliceRows(int64_t begin, bool scatter)
    {
        // SELL stores m row-major groups of sliceNnz slots; negative columns pad.
        for (int64_t row = 0; row < t_.m; ++row) {
            for (int32_t slot = 0; slot < t_.sliceWidth; ++slot) {
                int64_t p = begin + row * t_.sliceWidth + slot;
                int32_t raw = idx_.Read(p);
                if (raw < t_.base) continue;
                if (!Emit(static_cast<int32_t>(row), raw - t_.base, p, scatter)) return false;
            }
        }
        return true;
    }
    __aicore__ inline bool VisitSlicedEll(bool scatter)
    {
        if (ptr_.Read(0) != t_.base ||
            static_cast<int64_t>(ptr_.Read(t_.slices)) - t_.base != t_.sourceNnz) {
            error_ = 1;
            return false;
        }
        for (int32_t slice = 0; slice < t_.slices; ++slice) {
            int64_t begin = static_cast<int64_t>(ptr_.Read(slice)) - t_.base;
            int64_t end = static_cast<int64_t>(ptr_.Read(slice + 1)) - t_.base;
            if (begin < 0 || end < begin || end > t_.sourceNnz ||
                end - begin != t_.m * t_.sliceWidth) {
                error_ = 1;
                return false;
            }
            if (!VisitSliceRows(begin, scatter)) return false;
        }
        return true;
    }
    __aicore__ inline bool VisitCompressed(bool scatter)
    {
        if (ptr_.Read(0) != t_.base ||
            static_cast<int64_t>(ptr_.Read(t_.m)) - t_.base != t_.sourceNnz) {
            error_ = 1;
            return false;
        }
        for (int64_t compressed = 0; compressed < t_.m; ++compressed) {
            int64_t begin = static_cast<int64_t>(ptr_.Read(compressed)) - t_.base;
            int64_t end = static_cast<int64_t>(ptr_.Read(compressed + 1)) - t_.base;
            if (begin < 0 || end < begin || end > t_.sourceNnz) {
                error_ = 1;
                return false;
            }
            for (int64_t p = begin; p < end; ++p) {
                int64_t coordinate = static_cast<int64_t>(idx_.Read(p)) - t_.base;
                int64_t row = t_.format == 1 ? coordinate : compressed;
                int64_t col = t_.format == 1 ? compressed : coordinate;
                if (!Emit(row, col, p, scatter)) return false;
            }
        }
        return true;
    }
    __aicore__ inline bool Visit(bool scatter)
    {
        if (t_.format == 2) return VisitCoo(scatter);
        if (t_.format == 3) return VisitSlicedEll(scatter);
        return VisitCompressed(scatter);
    }
    SpsvA2Tiling t_;
    int32_t error_ = 0;
    bool diagonalMatrix_ = true;
    UbCache<int32_t, 2048> ptr_, idx_, row_, col_, perm_, scratch_;
    UbCache<float, 2048> source_, value_;
    UbCache<int32_t, 8> status_;
    static_assert(decltype(ptr_)::kBytes + decltype(idx_)::kBytes + decltype(row_)::kBytes +
        decltype(col_)::kBytes + decltype(perm_)::kBytes + decltype(scratch_)::kBytes +
        decltype(source_)::kBytes + decltype(value_)::kBytes + decltype(status_)::kBytes <=
        spsv_a2_ub::kRequiredBytes, "Analysis exceeds the UB budget");
    GlobalTensor<int32_t> directRow_;
    GlobalTensor<int32_t> directCol_;
    GlobalTensor<float> directValue_;
};

} // namespace spsv_a2

#endif // SPSV_ARCH22_SPSV_ANALYSIS_H_
