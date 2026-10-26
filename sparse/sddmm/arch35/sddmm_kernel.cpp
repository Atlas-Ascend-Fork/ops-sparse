/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

// SDDMM SIMD kernel for Ascend 950PR (dav-3510).
//
// Supports two data type combinations via C++ template + if constexpr:
//   FP32: X(fp32) * Y(fp32) -> C(fp32),  ScalarT=float
//   FP16: X(fp16) * Y(fp16) -> C(fp16),  ScalarT=float  (accumulate in fp32)
//
// Pattern: __global__ kernel (SIMD) + TPipe/TBuf/LocalTensor + low-level AscendC API.
// Launch:  sddmm_custom_fp32/fp16<<<blockDim, l2ctrl, stream>>>(GM tensors, workspaceGM, tiling)
//
// For each nonzero element p (row i, col j=colInd[p]) of the CSR matrix C:
//   result = alpha * (X[i,:] . Y[j,:]) + beta * C_values[p]
//   written back to csrValues[p].
//
// Dot product: DataCopyPad(GM->UB) -> Cast(FP16->FP32) -> Mul -> ReduceSum -> scalar accumulate.
// Scaling: Muls(alpha) -> Adds/Muls(beta*C) -> Cast(FP32->FP16 with saturation) -> DataCopyPad(UB->GM).

#include <stdint.h>
#include <type_traits>

#include "kernel_operator.h"
#include "log/log.h"
#include "sddmm.h"
#include "sddmm_kernel.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 3510)
#error "SDDMM SIMD: this TU is only for dav-3510 / Ascend 950PR (__NPU_ARCH__==3510)."
#endif

using namespace AscendC;

namespace {

constexpr float kFp16Max = 65504.0f;
using SddmmBudget::kReduceTmpBytes;
using SddmmBudget::kStridedChunk;
using SddmmBudget::kStridedElemBytes;

template <typename T>
__aicore__ inline void LoadSddmmDeviceScalars(SddmmTilingData& td, TPipe& pipe)
{
    TBuf<TPosition::VECIN> buffer;
    constexpr uint32_t slot = SddmmBudget::kScalarSlotBytes;
    pipe.InitBuffer(buffer, 2 * slot);
    auto local = buffer.Get<T>();
    GlobalTensor<T> alpha, beta;
    alpha.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(td.alphaDevice), 1);
    beta.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(td.betaDevice), 1);
    const DataCopyExtParams copy{1, sizeof(T), 0, 0, 0};
    const DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(local, alpha, copy, pad);
    DataCopyPad(local[slot / sizeof(T)], beta, copy, pad);
    const auto event = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(event);
    WaitFlag<HardEvent::MTE2_S>(event);
    td.alphaHost = static_cast<float>(local.GetValue(0));
    td.betaHost = static_cast<float>(local.GetValue(slot / sizeof(T)));
}

__aicore__ inline SddmmTilingData LoadSddmmTilingData(SddmmTilingData td, TPipe& pipe)
{
    if (td.alphaDevice != 0) {
        if (td.scalarType == SDDMM_DTYPE_FP16) {
            LoadSddmmDeviceScalars<half>(td, pipe);
        } else {
            LoadSddmmDeviceScalars<float>(td, pipe);
        }
    }
    return td;
}

// ============================================================================
// Templated SIMD compute: KernelSddmmSimd<ValT, ScalarT>
//
//   ValT    = float / half   (type of X/Y/C values)
//   ScalarT = float           (type of alpha, beta, and accumulation)
//
// SIMD model: each __global__ kernel instance runs on one AIV core.
// No thread-level parallelism; sequential traversal of CSR rows and nonzeros.
// ============================================================================
template <typename ValT, typename ScalarT>
class KernelSddmmSimd {
    static constexpr bool kIsFp16 = std::is_same_v<ValT, half>;

public:
    __aicore__ inline KernelSddmmSimd() {}

    __aicore__ inline void Init(GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues, GM_ADDR matX, GM_ADDR matY,
                                GM_ADDR workspaceGM, SddmmTilingData tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        td_ = LoadSddmmTilingData(tiling, *pipe);

        // Raw __gm__ pointers for scalar index reads (rowOff, reorder, binEdge)
        rowOffRaw_ = reinterpret_cast<__gm__ int32_t*>(csrRowOffsets);
        wsBase_ = reinterpret_cast<__gm__ uint8_t*>(workspaceGM);
        reorderRaw_ = reinterpret_cast<__gm__ int32_t*>(wsBase_ + static_cast<uint64_t>(td_.reorderOffset));
        binEdgeRaw_ = reinterpret_cast<__gm__ int32_t*>(wsBase_ + static_cast<uint64_t>(td_.binEdgeOffset));

        // Compute nnz from rowOff[m]
        int32_t nnz = rowOffRaw_[td_.m];

        // Cache layout flags first (needed for correct GlobalTensor buffer sizing
        // and reused throughout the kernel).
        k_ = td_.k;
        ldx_ = td_.ldx;
        ldy_ = td_.ldy;
        kTile_ = td_.kTile;
        nTile_ = td_.nTile;
        opX_ = td_.opX;
        opY_ = td_.opY;
        bool xRowMajor = (td_.orderPair == SDDMM_ORDER_RR || td_.orderPair == SDDMM_ORDER_RC);
        yRowMajor_ = (td_.orderPair == SDDMM_ORDER_RR || td_.orderPair == SDDMM_ORDER_CR);
        xContiguous_ = (opX_ == 0) ? xRowMajor : !xRowMajor;

        InitGlobalTensors(csrColInd, csrValues, matX, matY, nnz);

        // Init UB buffers
        const uint32_t kTileBytes = static_cast<uint32_t>(kTile_) * sizeof(ValT);
        const uint32_t kTileFp32Bytes = static_cast<uint32_t>(kTile_) * sizeof(float);
        const uint32_t nTileBytes = static_cast<uint32_t>(nTile_) * sizeof(ValT);
        const uint32_t nTileFp32Bytes = static_cast<uint32_t>(nTile_) * sizeof(float);
        const uint32_t nTileIntBytes = static_cast<uint32_t>(nTile_) * sizeof(int32_t);

        // K-dimension buffers (used in dot product)
        pipe_->InitBuffer(xQue_, 2, kTileBytes);
        pipe_->InitBuffer(yQue_, 2, kTileBytes);
        pipe_->InitBuffer(mulBuf_, kTileFp32Bytes);
        pipe_->InitBuffer(reduceTmpBuf_, static_cast<uint32_t>(kReduceTmpBytes));
        pipe_->InitBuffer(dotTmpBuf_, 32U); // 1 float, 32B aligned

        // Strided GM->UB optimization buffers (always allocated, see design §2.2)
        pipe_->InitBuffer(stridedBuf_, static_cast<uint32_t>(kStridedChunk * kStridedElemBytes));
        pipe_->InitBuffer(offsetBuf_, static_cast<uint32_t>(kStridedChunk * sizeof(int32_t)));
        InitOffsetTable();

        // nTile-dimension buffers (used in accumulation and scaling)
        pipe_->InitBuffer(accBuf_, nTileFp32Bytes);
        pipe_->InitBuffer(cValsBuf_, nTileBytes);
        pipe_->InitBuffer(resultBuf_, nTileFp32Bytes);
        pipe_->InitBuffer(colIndBuf_, nTileIntBytes);

        if constexpr (kIsFp16) {
            pipe_->InitBuffer(xFp32Buf_, kTileFp32Bytes);
            pipe_->InitBuffer(yFp32Buf_, kTileFp32Bytes);
            pipe_->InitBuffer(cValsFp32Buf_, nTileFp32Bytes);
            pipe_->InitBuffer(outBuf_, nTileBytes);
        }
    }

    // Set up GlobalTensor buffers for DataCopyPad sources. Sizing follows the
    // actual access pattern (contiguous vs strided) so the dominant dimension
    // (m or k) is covered; under-sizing causes out-of-bounds GM access.
    __aicore__ inline void InitGlobalTensors(GM_ADDR csrColInd, GM_ADDR csrValues, GM_ADDR matX, GM_ADDR matY,
                                             int32_t nnz)
    {
        colIndGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(csrColInd),
                                  static_cast<uint64_t>(nnz > 0 ? nnz : 1));
        csrValuesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ValT*>(csrValues),
                                     static_cast<uint64_t>(nnz > 0 ? nnz : 1));

        // X buffer size must cover the actual access pattern:
        //   contiguous (X[row*ldx+kk]): spans m rows of length ldx -> m*ldx
        //   strided (X[kk*ldx+row]): spans k rows of length ldx -> k*ldx
        // Using m*ldx for strided under-sizes the buffer when k>m and causes
        // out-of-bounds GM access. Size by the dominant dimension.
        const uint64_t xElemCount = xContiguous_ ? static_cast<uint64_t>(td_.m) * static_cast<uint64_t>(td_.ldx) :
                                                   static_cast<uint64_t>(td_.k) * static_cast<uint64_t>(td_.ldx);
        matXGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ValT*>(matX), xElemCount);

        // Y buffer size must cover the actual access pattern:
        //   contiguous (NON_TRANSPOSE+colMajor or TRANSPOSE+rowMajor): Y[col*ldy+kk]
        //       -> spans n rows of length ldy -> n*ldy
        //   strided  (NON_TRANSPOSE+rowMajor or TRANSPOSE+colMajor): Y[kk*ldy+col]
        //       -> spans k rows of length ldy -> k*ldy
        // Using n*ldy for strided access under-sizes the buffer when k>n.
        const bool yContiguous = (opY_ == 0) ? (!yRowMajor_) : yRowMajor_;
        const uint64_t yElemCount = yContiguous ? static_cast<uint64_t>(td_.n) * static_cast<uint64_t>(td_.ldy) :
                                                  static_cast<uint64_t>(td_.k) * static_cast<uint64_t>(td_.ldy);
        matYGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ValT*>(matY), yElemCount);
    }

    __aicore__ inline void Process()
    {
        const ScalarT alpha = static_cast<ScalarT>(td_.alphaHost);
        const ScalarT beta = static_cast<ScalarT>(td_.betaHost);
        const bool betaZero = (beta == static_cast<ScalarT>(0));

        const int32_t rowBinNum = static_cast<int32_t>(GetBlockNum());
        const int32_t blockId = static_cast<int32_t>(GetBlockIdx());

        // SIMD model: empty bins can early-return safely (no asc_vf_call)
        if (blockId < 0 || blockId >= rowBinNum) {
            return;
        }
        const int32_t rowStart = binEdgeRaw_[blockId];
        const int32_t rowEnd = binEdgeRaw_[blockId + 1];
        if (rowStart >= rowEnd) {
            return;
        }
        if (nTile_ <= 0) {
            return;
        }

        for (int32_t logicalRow = rowStart; logicalRow < rowEnd; ++logicalRow) {
            const int32_t row = reorderRaw_[logicalRow];
            const int32_t s = rowOffRaw_[row];
            const int32_t e = rowOffRaw_[row + 1];

            for (int32_t p = s; p < e; p += nTile_) {
                const int32_t batchEnd = (p + nTile_ < e) ? (p + nTile_) : e;
                const int32_t batchSize = batchEnd - p;
                if (batchSize <= 0) {
                    break;
                }
                ProcessBatch(row, p, batchSize, alpha, beta, betaZero);
            }
        }
    }

private:
    __aicore__ inline void ProcessBatch(int32_t row, int32_t pStart, int32_t batchSize, ScalarT alpha, ScalarT beta,
                                        bool betaZero)
    {
        auto accBuf = accBuf_.Get<float>();
        auto colIndBuf = colIndBuf_.Get<int32_t>();

        // Zero accumulator
        Duplicate<float>(accBuf, 0.0f, static_cast<int32_t>(nTile_));
        PipeBarrier<PIPE_V>();

        // Load colInd batch for this nTile range
        {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(batchSize * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(colIndBuf, colIndGm_[pStart], copyParams, padParams);
            // MTE2->V: DataCopyPad writes colIndBuf (MTE2), GetValue reads it (V)
            PipeBarrier<PIPE_MTE2>();
        }

        ProcessDotProduct(row, batchSize, accBuf, colIndBuf);
        PipeBarrier<PIPE_V>();

        auto resultBuf = resultBuf_.Get<float>();
        ProcessScaling(pStart, batchSize, alpha, beta, betaZero, accBuf, resultBuf);

        ProcessWriteback(pStart, batchSize, resultBuf);
    }

    // Prefetch next Y fragment for double-buffering. Handles both intra-tile
    // (next j within same K-tile) and cross-tile (first j of next K-tile) cases.
    __aicore__ inline void PrefetchNextY(LocalTensor<int32_t>& colIndBuf, int32_t j, int32_t batchSize, int32_t kBase,
                                         int32_t actualKTile)
    {
        const int32_t nextJ = j + 1;
        const bool hasMoreY = (nextJ < batchSize) || (kBase + kTile_ < k_);
        if (!hasMoreY) {
            return;
        }
        int32_t nextCol;
        int32_t nextKBase;
        int32_t nextKT;
        if (nextJ < batchSize) {
            nextCol = colIndBuf.GetValue(nextJ);
            nextKBase = kBase;
            nextKT = actualKTile;
        } else {
            // Cross into next K-tile: prefetch Y[k+1, 0]
            nextCol = colIndBuf.GetValue(0);
            nextKBase = kBase + kTile_;
            nextKT = (nextKBase + kTile_ < k_) ? kTile_ : (k_ - nextKBase);
        }
        auto yNext = yQue_.AllocTensor<ValT>();
        LoadYFragment(yNext, nextCol, nextKBase, nextKT);
        yQue_.EnQue(yNext);
    }

    // Element-wise multiply for one K-tile: FP16 path casts to FP32 before Mul;
    // FP32 path multiplies directly.
    __aicore__ inline void ComputeMulStep(LocalTensor<ValT>& xIn, LocalTensor<ValT>& yIn, LocalTensor<float>& mulBuf,
                                          int32_t actualKTile)
    {
        if constexpr (kIsFp16) {
            auto yFp32Buf = yFp32Buf_.Get<float>();
            auto xFp32Buf = xFp32Buf_.Get<float>();
            Cast<float, half>(yFp32Buf, yIn, RoundMode::CAST_NONE, actualKTile);
            PipeBarrier<PIPE_V>();
            Mul<float>(mulBuf, xFp32Buf, yFp32Buf, actualKTile);
        } else {
            Mul<float>(mulBuf, xIn, yIn, actualKTile);
        }
        PipeBarrier<PIPE_V>();
    }

    // Prefetch next X fragment for K-tile double-buffering.
    __aicore__ inline void PrefetchNextX(int32_t row, int32_t kBase)
    {
        if (kBase + kTile_ >= k_) {
            return;
        }
        const int32_t nextKBase = kBase + kTile_;
        const int32_t nextKT = (nextKBase + kTile_ < k_) ? kTile_ : (k_ - nextKBase);
        auto xNext = xQue_.AllocTensor<ValT>();
        LoadXFragment(xNext, row, nextKBase, nextKT);
        xQue_.EnQue(xNext);
    }

    // K-dimension tiled dot product accumulation with double-buffered X/Y.
    // X is double-buffered at K-tile level: prefetch X[k+1] overlaps with
    // j-loop computation of X[k]. Y is double-buffered at j level: prefetch
    // Y[k,j+1] overlaps with Mul/ReduceSum of Y[k,j].
    __aicore__ inline void ProcessDotProduct(int32_t row, int32_t batchSize, LocalTensor<float>& accBuf,
                                             LocalTensor<int32_t>& colIndBuf)
    {
        if (k_ <= 0) {
            return;
        }
        auto mulBuf = mulBuf_.Get<float>();
        auto reduceTmpBuf = reduceTmpBuf_.Get<float>();
        auto dotTmp = dotTmpBuf_.Get<float>();

        // Prefetch X[0] (first K-tile) before entering K loop
        const int32_t kEnd0 = (kTile_ < k_) ? kTile_ : k_;
        auto xTensor0 = xQue_.AllocTensor<ValT>();
        LoadXFragment(xTensor0, row, 0, kEnd0);
        xQue_.EnQue(xTensor0);

        // Prefetch Y[0, 0] (first K-tile, first j) before entering K loop.
        // For subsequent K-tiles, Y[k,0] is prefetched at the tail of the
        // previous K-tile's last j iteration (cross-tile prefetch).
        const int32_t col0 = colIndBuf.GetValue(0);
        auto yTensor0 = yQue_.AllocTensor<ValT>();
        LoadYFragment(yTensor0, col0, 0, kEnd0);
        yQue_.EnQue(yTensor0);

        if (kTile_ <= 0) {
            return;
        }
        for (int32_t kBase = 0; kBase < k_; kBase += kTile_) {
            const int32_t kEnd = (kBase + kTile_ < k_) ? (kBase + kTile_) : k_;
            const int32_t actualKTile = kEnd - kBase;
            if (actualKTile <= 0) {
                break;
            }

            // DeQue X[k] (blocks until MTE2 load completes)
            auto xIn = xQue_.DeQue<ValT>();
            if constexpr (kIsFp16) {
                auto xFp32Buf = xFp32Buf_.Get<float>();
                Cast<float, half>(xFp32Buf, xIn, RoundMode::CAST_NONE, actualKTile);
                PipeBarrier<PIPE_V>();
            }

            for (int32_t j = 0; j < batchSize; ++j) {
                // DeQue Y[k,j] (blocks until MTE2 load completes)
                auto yIn = yQue_.DeQue<ValT>();

                PrefetchNextY(colIndBuf, j, batchSize, kBase, actualKTile);
                ComputeMulStep(xIn, yIn, mulBuf, actualKTile);

                // ReduceSum to get scalar dot product for this K-tile
                ReduceSum<float, true>(dotTmp, mulBuf, reduceTmpBuf, actualKTile);
                PipeBarrier<PIPE_V>();

                // Accumulate scalar
                accBuf.SetValue(j, accBuf.GetValue(j) + dotTmp.GetValue(0));

                yQue_.FreeTensor(yIn);
            }

            // Free X[k] and prefetch X[k+1] (overlaps with next K-tile setup)
            xQue_.FreeTensor(xIn);
            PrefetchNextX(row, kBase);
        }
    }

    // alpha/beta scaling: result = alpha * acc + (beta != 0 ? beta * C : 0)
    __aicore__ inline void ProcessScaling(int32_t pStart, int32_t batchSize, ScalarT alpha, ScalarT beta, bool betaZero,
                                          LocalTensor<float>& accBuf, LocalTensor<float>& resultBuf)
    {
        // Batch scaling: result = alpha * acc
        Muls<float>(resultBuf, accBuf, alpha, batchSize);
        // V->MTE3: Muls writes resultBuf (V), DataCopyPad reads it (MTE3) when betaZero
        PipeBarrier<PIPE_V>();

        if (betaZero) {
            return;
        }

        // beta * C accumulation
        auto cValsBuf = cValsBuf_.Get<ValT>();
        DataCopyExtParams cpCVals{1, static_cast<uint32_t>(batchSize * sizeof(ValT)), 0, 0, 0};
        DataCopyPadExtParams<ValT> padCVals{false, 0, 0, 0};
        DataCopyPad(cValsBuf, csrValuesGm_[pStart], cpCVals, padCVals);
        // MTE2->V: DataCopyPad writes cValsBuf (MTE2), Cast/Muls reads it (V)
        PipeBarrier<PIPE_MTE2>();

        if constexpr (kIsFp16) {
            auto cValsFp32Buf = cValsFp32Buf_.Get<float>();
            Cast<float, half>(cValsFp32Buf, cValsBuf, RoundMode::CAST_NONE, batchSize);
            PipeBarrier<PIPE_V>();
            Muls<float>(cValsFp32Buf, cValsFp32Buf, beta, batchSize);
            PipeBarrier<PIPE_V>();
            Add<float>(resultBuf, resultBuf, cValsFp32Buf, batchSize);
        } else {
            Muls<float>(cValsBuf, cValsBuf, beta, batchSize);
            PipeBarrier<PIPE_V>();
            Add<float>(resultBuf, resultBuf, cValsBuf, batchSize);
        }
        // V->MTE3: Add writes resultBuf (V), DataCopyPad reads it (MTE3) in FP32 path
        PipeBarrier<PIPE_V>();
    }

    // FP16 saturation + Cast + writeback (FP16) or direct writeback (FP32).
    __aicore__ inline void ProcessWriteback(int32_t pStart, int32_t batchSize, LocalTensor<float>& resultBuf)
    {
        if constexpr (kIsFp16) {
            auto outBuf = outBuf_.Get<half>();
            // Saturate to FP16 range using vectorized Mins/Maxs (replaces the
            // previous scalar clamp loop). Mins clamps the upper bound,
            // Maxs clamps the lower bound; in-place dst==src is supported.
            Mins<float>(resultBuf, resultBuf, kFp16Max, batchSize);
            PipeBarrier<PIPE_V>();
            Maxs<float>(resultBuf, resultBuf, -kFp16Max, batchSize);
            PipeBarrier<PIPE_V>();
            Cast<half, float>(outBuf, resultBuf, RoundMode::CAST_ROUND, batchSize);
            // V->MTE3: Cast writes outBuf (V), DataCopyPad reads it (MTE3)
            PipeBarrier<PIPE_V>();
            DataCopyExtParams cpOut{1, static_cast<uint32_t>(batchSize * sizeof(half)), 0, 0, 0};
            DataCopyPad(csrValuesGm_[pStart], outBuf, cpOut);
            // MTE3->V: ensure MTE3 read of outBuf completes before next iteration overwrites
            PipeBarrier<PIPE_MTE3>();
        } else {
            DataCopyExtParams cpOut{1, static_cast<uint32_t>(batchSize * sizeof(float)), 0, 0, 0};
            DataCopyPad(csrValuesGm_[pStart], resultBuf, cpOut);
            // MTE3->V: ensure MTE3 read of resultBuf completes before next iteration overwrites
            PipeBarrier<PIPE_MTE3>();
        }
    }

    // Generate byte-offset table {0, 32, 64, ..., (kStridedChunk-1)*32} for Gather.
    // Each strided element in stridedBuf is padded to 32B by DataCopyPad blockCount
    // mode, so element i resides at byte offset i*32. ArithProgression generates
    // this arithmetic progression in one vectorized call.
    __aicore__ inline void InitOffsetTable()
    {
        auto offsetInt = offsetBuf_.Get<int32_t>();
        ArithProgression<int32_t>(offsetInt, 0, kStridedElemBytes, kStridedChunk);
        PipeBarrier<PIPE_V>();
    }

    // Strided GM->UB load helper: DataCopyPad strided into stridedBuf (each element
    // padded to 32B), then Gather compactly into dst. Shared by X/Y strided paths.
    // gmBase: GM source base element index; elemStride: element stride (ldx/ldy).
    __aicore__ inline void StridedLoadGather(LocalTensor<ValT>& dst, int32_t dstOffset, const GlobalTensor<ValT>& gm,
                                             uint64_t gmBase, int32_t elemStride, int32_t count)
    {
        auto stridedBuf = stridedBuf_.Get<ValT>();
        auto offsetInt = offsetBuf_.Get<int32_t>();
        auto offsetUInt = offsetInt.ReinterpretCast<uint32_t>();
        uint32_t srcStrideBytes = static_cast<uint32_t>((elemStride - 1) * sizeof(ValT));
        for (int32_t cBase = 0; cBase < count; cBase += kStridedChunk) {
            const int32_t cEnd = (cBase + kStridedChunk < count) ? (cBase + kStridedChunk) : count;
            const int32_t cSize = cEnd - cBase;
            uint64_t chunkGmBase = gmBase + static_cast<uint64_t>(cBase) * static_cast<uint64_t>(elemStride);
            DataCopyExtParams cp{static_cast<uint16_t>(cSize), static_cast<uint32_t>(sizeof(ValT)), srcStrideBytes, 0,
                                 0};
            DataCopyPadExtParams<ValT> pad{false, 0, 0, 0};
            DataCopyPad(stridedBuf, gm[chunkGmBase], cp, pad);
            // MTE2->V: DataCopyPad writes stridedBuf (MTE2), Gather reads it (V)
            PipeBarrier<PIPE_MTE2>();
            Gather(dst[dstOffset + cBase], stridedBuf, offsetUInt, 0, cSize);
            // V->MTE2: Gather reads stridedBuf (V) before next iteration DataCopyPad (MTE2) overwrites;
            // also ensures V completion before outer EnQue on the last iteration.
            PipeBarrier<PIPE_V>();
        }
    }

    // Load X row fragment X[row, kBase:kEnd] into UB.
    // xContiguous determines contiguous vs strided access, derived from opX and
    // the memory layout (order) of X:
    //   NON_TRANSPOSE + row-major -> contiguous (X[row*ldx + kBase])
    //   NON_TRANSPOSE + col-major -> strided  (X[kk*ldx + row])
    //   TRANSPOSE     + row-major -> strided  (X[kk*ldx + row])
    //   TRANSPOSE     + col-major -> contiguous (X[row*ldx + kBase])
    // The destination tensor is obtained from xQue_.AllocTensor() by the caller;
    // synchronization for the contiguous (MTE2) path is handled by xQue_.EnQue/DeQue.
    __aicore__ inline void LoadXFragment(LocalTensor<ValT>& xBuf, int32_t row, int32_t kBase, int32_t actualKTile)
    {
        if (xContiguous_) {
            // Contiguous: X[row*ldx + kBase : +actualKTile]
            uint64_t offset = static_cast<uint64_t>(row) * static_cast<uint64_t>(ldx_) + static_cast<uint64_t>(kBase);
            DataCopyExtParams cp{1, static_cast<uint32_t>(actualKTile * sizeof(ValT)), 0, 0, 0};
            DataCopyPadExtParams<ValT> pad{false, 0, 0, 0};
            DataCopyPad(xBuf, matXGm_[offset], cp, pad);
            // Sync handled by outer xQue_.EnQue/DeQue (MTE2->V)
        } else {
            // Strided: X[kk*ldx + row] for kk in [kBase, kEnd)
            uint64_t gmBase = static_cast<uint64_t>(kBase) * static_cast<uint64_t>(ldx_) + static_cast<uint64_t>(row);
            StridedLoadGather(xBuf, 0, matXGm_, gmBase, ldx_, actualKTile);
            // Last PipeBarrier<PIPE_V> in StridedLoadGather ensures V completion
            // before outer EnQue
        }
    }

    // Load Y row fragment Y[kBase:kEnd, col] (NON_TRANSPOSE) or Y[col, kBase:kEnd] (TRANSPOSE) into UB.
    // The destination tensor is obtained from yQue_.AllocTensor() by the caller;
    // synchronization for the contiguous (MTE2) path is handled by yQue_.EnQue/DeQue.
    __aicore__ inline void LoadYFragment(LocalTensor<ValT>& yBuf, int32_t col, int32_t kBase, int32_t actualKTile)
    {
        // Determine if Y access is contiguous
        // NON_TRANSPOSE: Y[kk, col] -> contiguous if !yRowMajor (Y[col*ldy+kk])
        // TRANSPOSE: Y[col, kk] -> contiguous if yRowMajor (Y[col*ldy+kk])
        const bool yContiguous = (opY_ == 0) ? (!yRowMajor_) : yRowMajor_;

        if (yContiguous) {
            // Contiguous access: base = col*ldy + kBase
            uint64_t offset = static_cast<uint64_t>(col) * static_cast<uint64_t>(ldy_) + static_cast<uint64_t>(kBase);
            DataCopyExtParams cp{1, static_cast<uint32_t>(actualKTile * sizeof(ValT)), 0, 0, 0};
            DataCopyPadExtParams<ValT> pad{false, 0, 0, 0};
            DataCopyPad(yBuf, matYGm_[offset], cp, pad);
            // Sync handled by outer yQue_.EnQue/DeQue (MTE2->V)
        } else {
            // Strided access (NON_TRANSPOSE+colMajor or TRANSPOSE+rowMajor):
            // both share the index formula Y[kk*ldy + col].
            uint64_t gmBase = static_cast<uint64_t>(kBase) * static_cast<uint64_t>(ldy_) + static_cast<uint64_t>(col);
            StridedLoadGather(yBuf, 0, matYGm_, gmBase, ldy_, actualKTile);
        }
    }

private:
    TPipe* pipe_{nullptr};
    SddmmTilingData td_{};

    // GlobalTensor for DataCopyPad sources
    GlobalTensor<int32_t> colIndGm_;
    GlobalTensor<ValT> csrValuesGm_;
    GlobalTensor<ValT> matXGm_;
    GlobalTensor<ValT> matYGm_;

    // Raw __gm__ pointers for scalar index reads
    __gm__ int32_t* rowOffRaw_{nullptr};
    __gm__ int32_t* reorderRaw_{nullptr};
    __gm__ int32_t* binEdgeRaw_{nullptr};
    __gm__ uint8_t* wsBase_{nullptr};

    // UB buffers (K-dimension)
    TQue<TPosition::VECIN, 1> xQue_;
    TQue<TPosition::VECIN, 1> yQue_;
    TBuf<TPosition::VECCALC> mulBuf_;
    TBuf<TPosition::VECCALC> reduceTmpBuf_;
    TBuf<TPosition::VECCALC> dotTmpBuf_;

    // Strided GM->UB optimization buffers (shared by X/Y strided paths)
    TBuf<TPosition::VECCALC> stridedBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_;

    // UB buffers (nTile-dimension)
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> cValsBuf_;
    TBuf<TPosition::VECCALC> resultBuf_;
    TBuf<TPosition::VECCALC> colIndBuf_;

    // FP16-only UB buffers
    TBuf<TPosition::VECCALC> xFp32Buf_;
    TBuf<TPosition::VECCALC> yFp32Buf_;
    TBuf<TPosition::VECCALC> cValsFp32Buf_;
    TBuf<TPosition::VECCALC> outBuf_;

    // Cached tiling fields
    int32_t k_{0};
    int32_t ldx_{0};
    int32_t ldy_{0};
    int32_t kTile_{0};
    int32_t nTile_{0};
    int32_t opX_{0};
    int32_t opY_{0};
    bool yRowMajor_{true};
    bool xContiguous_{true};
};

} // namespace

// ============================================================================
// Kernel entry points — one per data type combination.
// ============================================================================

template <typename OutT, typename AccT>
__aicore__ inline void SddmmKernelEntry(GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues, GM_ADDR matX,
                                        GM_ADDR matY, GM_ADDR workspaceGM, SddmmTilingData tiling)
{
    TPipe pipe;
    KernelSddmmSimd<OutT, AccT> op;
    op.Init(csrRowOffsets, csrColInd, csrValues, matX, matY, workspaceGM, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void sddmm_custom_fp32(GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
                                                        GM_ADDR matX, GM_ADDR matY, GM_ADDR workspaceGM,
                                                        SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SddmmKernelEntry<float, float>(csrRowOffsets, csrColInd, csrValues, matX, matY, workspaceGM, tiling);
}

extern "C" __global__ __aicore__ void sddmm_custom_fp16(GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
                                                        GM_ADDR matX, GM_ADDR matY, GM_ADDR workspaceGM,
                                                        SddmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SddmmKernelEntry<half, float>(csrRowOffsets, csrColInd, csrValues, matX, matY, workspaceGM, tiling);
}

// ============================================================================
// Host-side launch dispatcher — dataType/blockDim passed from sddmm_host.cpp.
// ============================================================================
extern "C" aclsparseStatus_t sddmm_kernel_launch(GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
                                                 GM_ADDR matX, GM_ADDR matY, GM_ADDR workspaceGM,
                                                 const SddmmTilingData& tiling, int32_t dataType, uint32_t blockDim,
                                                 void* stream)
{
    if (dataType == SDDMM_DTYPE_FP32) {
        sddmm_custom_fp32<<<blockDim, nullptr, stream>>>(csrRowOffsets, csrColInd, csrValues, matX, matY, workspaceGM,
                                                         tiling);
        return ACL_SPARSE_STATUS_SUCCESS;
    } else if (dataType == SDDMM_DTYPE_FP16) {
        sddmm_custom_fp16<<<blockDim, nullptr, stream>>>(csrRowOffsets, csrColInd, csrValues, matX, matY, workspaceGM,
                                                         tiling);
        return ACL_SPARSE_STATUS_SUCCESS;
    } else {
        OP_LOGE("aclsparseSDDMM", "unsupported dataType: %d", dataType);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
}
