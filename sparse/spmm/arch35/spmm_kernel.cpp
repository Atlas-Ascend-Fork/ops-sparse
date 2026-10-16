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

// SPMM SIMT kernel for Ascend 950PR (dav-3510).
//
// Supports three data type combinations via C++ template + if constexpr:
//   FP32: A(fp32) * B(fp32) -> C(fp32),  ScalarT=float
//   FP16: A(fp16) * B(fp16) -> C(fp16),  ScalarT=float  (accumulate in fp32)
//   INT8: A(int8) * B(int8) -> C(int32), ScalarT=int32_t (accumulate in int32)
//
// Template parameters:
//   ValT     - element type of sparse A values and dense B
//   CT       - element type of dense C output
//   ScalarT  - type of alpha/beta scalars and accumulation
//
// Pattern: __aicore__ kernel (SPMD outer) + asc_vf_call + __simt_vf__ (SIMT inner).
// Launch:  spmm_custom_fp32/fp16/int8<<<blockDim, l2ctrl, stream>>>(GM tensors, workspaceGM, tilingGM)
//
// Kernel reads SpmmTilingData from tilingGM on device; dtype-specific entry points
// are selected on the host in spmm_kernel_launch by dataType parameter.

#include <stdint.h>
#include <cstdio>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"

#include "spmm.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 3510)
#error "SPMM SIMT: this TU is only for dav-3510 / Ascend 950PR (__NPU_ARCH__==3510)."
#endif

using namespace AscendC;

namespace {

constexpr uint32_t kMaxSimtThreadsPerBlock = 512u;
constexpr int32_t kChunk = 8;

template<typename T, typename U>
struct SpmmIsSame { static constexpr bool value = false; };

template<typename T>
struct SpmmIsSame<T, T> { static constexpr bool value = true; };

__aicore__ inline SpmmTilingData LoadSpmmTilingData(GM_ADDR tilingGM)
{
    __gm__ SpmmTilingData *gmTiling = reinterpret_cast<__gm__ SpmmTilingData *>(tilingGM);
    SpmmTilingData td;
    td.m = gmTiling->m;
    td.n = gmTiling->n;
    td.ldb = gmTiling->ldb;
    td.ldc = gmTiling->ldc;
    td.reorder_offset = gmTiling->reorder_offset;
    td.bin_edge_offset = gmTiling->bin_edge_offset;
    td.high_precision = gmTiling->high_precision;
    td.opB = gmTiling->opB;
    td.order_pair = gmTiling->order_pair;
    td.alpha_host = gmTiling->alpha_host;
    td.beta_host = gmTiling->beta_host;
    return td;
}

} // namespace

// ============================================================================
// Templated SIMT compute: SpmmCsrSimtCompute<ValT, CT, ScalarT, UseKahan>
//
//   ValT    = float / __fp16 / int8_t   (type of A values and B elements)
//   CT      = float / __fp16 / int32_t  (type of C elements)
//   ScalarT = float / int32_t           (type of alpha, beta, and accumulation)
//
// Uses if constexpr to select type-specific logic at compile time:
//   - ScalarT==float  => fp32 accumulation, direct or cast output
//   - ScalarT==int32_t => int32 accumulation, sign-extended input
// ============================================================================
template<typename ValT, typename CT, typename ScalarT, bool UseKahan>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void SpmmCsrSimtCompute(
    __gm__ int32_t *rowOff,
    __gm__ int32_t *colInd,
    __gm__ ValT *values,
    __gm__ ValT *matB,
    __gm__ CT *matC,
    __gm__ int32_t *reorder,
    int32_t n,
    int32_t ldb,
    int32_t ldc,
    int32_t colTiles,
    int32_t logicalRowStart,
    int32_t logicalRowEnd,
    ScalarT alpha,
    ScalarT beta,
    int32_t opB,
    int32_t orderPair)
{
    const int32_t numRows = logicalRowEnd - logicalRowStart;
    if (numRows <= 0 || colTiles <= 0) {
        return;
    }
    const uint64_t totalWork =
        static_cast<uint64_t>(numRows) * static_cast<uint64_t>(colTiles);
    const bool betaZero = (beta == static_cast<ScalarT>(0));
    const uint32_t threadNum = blockDim.x;
    const uint64_t threadIdxX = static_cast<uint64_t>(threadIdx.x);
    const bool bRowMajor = (orderPair == SPMM_ORDER_RR || orderPair == SPMM_ORDER_RC);
    const bool cRowMajor = (orderPair == SPMM_ORDER_RR || orderPair == SPMM_ORDER_CR);

    for (uint64_t work = threadIdxX; work < totalWork; work += static_cast<uint64_t>(threadNum)) {
        const int32_t logicalRow =
            logicalRowStart + static_cast<int32_t>(work / static_cast<uint64_t>(colTiles));
        const int32_t colTile = static_cast<int32_t>(work % static_cast<uint64_t>(colTiles));
        const int32_t colFirst = colTile * kChunk;
        if (colFirst >= n) {
            continue;
        }

        const int32_t row = reorder[logicalRow];
        const int32_t s = rowOff[row];
        const int32_t e = rowOff[row + 1];

        ScalarT acc[kChunk];
        ScalarT comp[kChunk];  // Kahan 补偿量(仅 UseKahan 实例使用)
#pragma unroll
        for (int j = 0; j < kChunk; ++j) {
            if constexpr (SpmmIsSame<ScalarT, float>::value) {
                acc[j] = 0.0f;
                if constexpr (UseKahan) {
                    comp[j] = 0.0f;
                }
            } else {
                acc[j] = 0;
            }
        }

        for (int32_t p = s; p < e; ++p) {
            const int32_t c = colInd[p];

            if constexpr (SpmmIsSame<ScalarT, float>::value) {
                const float v = static_cast<float>(values[p]);
#pragma unroll
                for (int j = 0; j < kChunk; ++j) {
                    if (colFirst + j < n) {
                        int32_t colIdx = colFirst + j;
                        uint64_t bIdx;
                        if (opB == 0) {
                            bIdx = bRowMajor
                                ? static_cast<uint64_t>(c) * static_cast<uint64_t>(ldb) +
                                      static_cast<uint64_t>(colIdx)
                                : static_cast<uint64_t>(colIdx) * static_cast<uint64_t>(ldb) +
                                      static_cast<uint64_t>(c);
                        } else {
                            bIdx = bRowMajor
                                ? static_cast<uint64_t>(colIdx) * static_cast<uint64_t>(ldb) +
                                      static_cast<uint64_t>(c)
                                : static_cast<uint64_t>(c) * static_cast<uint64_t>(ldb) +
                                      static_cast<uint64_t>(colIdx);
                        }
                        const float prod = v * static_cast<float>(matB[bIdx]);
                        if constexpr (UseKahan) {
                            // Kahan 补偿求和(fp32 高精度 alg):压制长行累加的舍入与抵消误差
                            const float y = prod - comp[j];
                            const float t = acc[j] + y;
                            comp[j] = (t - acc[j]) - y;
                            acc[j] = t;
                        } else {
                            // 默认高性能路径(fp32 标准精度 / fp16):在 fp32 里朴素累加
                            acc[j] += prod;
                        }
                    }
                }
            } else {
                const int32_t v = static_cast<int32_t>(values[p]);
#pragma unroll
                for (int j = 0; j < kChunk; ++j) {
                    if (colFirst + j < n) {
                        int32_t colIdx = colFirst + j;
                        uint64_t bIdx;
                        if (opB == 0) {
                            bIdx = bRowMajor
                                ? static_cast<uint64_t>(c) * static_cast<uint64_t>(ldb) +
                                      static_cast<uint64_t>(colIdx)
                                : static_cast<uint64_t>(colIdx) * static_cast<uint64_t>(ldb) +
                                      static_cast<uint64_t>(c);
                        } else {
                            bIdx = bRowMajor
                                ? static_cast<uint64_t>(colIdx) * static_cast<uint64_t>(ldb) +
                                      static_cast<uint64_t>(c)
                                : static_cast<uint64_t>(c) * static_cast<uint64_t>(ldb) +
                                      static_cast<uint64_t>(colIdx);
                        }
                        acc[j] += v * static_cast<int32_t>(matB[bIdx]);
                    }
                }
            }
        }

        if constexpr (SpmmIsSame<ScalarT, float>::value) {
#pragma unroll
            for (int j = 0; j < kChunk; ++j) {
                if (colFirst + j >= n) {
                    break;
                }
                int32_t colIdx = colFirst + j;
                uint64_t cIdx = cRowMajor
                    ? static_cast<uint64_t>(row) * static_cast<uint64_t>(ldc) +
                          static_cast<uint64_t>(colIdx)
                    : static_cast<uint64_t>(colIdx) * static_cast<uint64_t>(ldc) +
                          static_cast<uint64_t>(row);
                float cNew;
                if constexpr (UseKahan) {
                    cNew = alpha * (acc[j] + comp[j]);
                } else {
                    cNew = alpha * acc[j];
                }
                if (!betaZero) {
                    cNew += beta * static_cast<float>(matC[cIdx]);
                }
                if constexpr (SpmmIsSame<CT, float>::value) {
                    matC[cIdx] = cNew;
                } else {
                    constexpr float kFp16Max = 65504.0f;
                    float clamped = cNew;
                    if (clamped > kFp16Max) clamped = kFp16Max;
                    if (clamped < -kFp16Max) clamped = -kFp16Max;
                    matC[cIdx] = static_cast<ValT>(clamped);
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < kChunk; ++j) {
                if (colFirst + j >= n) {
                    break;
                }
                int32_t colIdx = colFirst + j;
                uint64_t cIdx = cRowMajor
                    ? static_cast<uint64_t>(row) * static_cast<uint64_t>(ldc) +
                          static_cast<uint64_t>(colIdx)
                    : static_cast<uint64_t>(colIdx) * static_cast<uint64_t>(ldc) +
                          static_cast<uint64_t>(row);
                // 在 int64 中计算 alpha*acc + beta*C，再饱和到 int32，
                // 避免 int32 相乘/相加溢出导致的回绕（wrap-around）。
                int64_t cWide;
                if (alpha == 1) {
                    cWide = static_cast<int64_t>(acc[j]);
                } else {
                    cWide = static_cast<int64_t>(alpha) * static_cast<int64_t>(acc[j]);
                }
                if (!betaZero) {
                    cWide += static_cast<int64_t>(beta) * static_cast<int64_t>(matC[cIdx]);
                }
                constexpr int64_t kInt32Max = 2147483647LL;
                constexpr int64_t kInt32Min = -2147483648LL;
                if (cWide > kInt32Max) cWide = kInt32Max;
                if (cWide < kInt32Min) cWide = kInt32Min;
                matC[cIdx] = static_cast<int32_t>(cWide);
            }
        }
    }
}

// ============================================================================
// Templated kernel class: KernelSpmmSimt<ValT, CT, ScalarT, UseKahan>
//
// UseKahan=true  => fp32 high-precision (Kahan compensated summation) path.
// UseKahan=false => default high-performance (standard accumulation) path.
//
// Each template instantiation is a fully typed, compile-time resolved class.
// No runtime dataType dispatch — the correct instantiation is selected by
// the per-type kernel entry point (spmm_custom_fp32 / fp16 / int8).
// ============================================================================
template<typename ValT, typename CT, typename ScalarT, bool UseKahan>
class KernelSpmmSimt {
public:
    __aicore__ inline KernelSpmmSimt() {}

    __aicore__ inline void Init(GM_ADDR csrRowOffsets,
        GM_ADDR csrColInd,
        GM_ADDR csrValues,
        GM_ADDR matB,
        GM_ADDR matC,
        GM_ADDR workspaceGM,
        GM_ADDR tilingGM)
    {
        tilingData_ = LoadSpmmTilingData(tilingGM);
        wsBase_ = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        rowOff_ = reinterpret_cast<__gm__ int32_t *>(csrRowOffsets);
        colInd_ = reinterpret_cast<__gm__ int32_t *>(csrColInd);
        values_ = reinterpret_cast<__gm__ ValT *>(csrValues);
        matB_ = reinterpret_cast<__gm__ ValT *>(matB);
        matC_ = reinterpret_cast<__gm__ CT *>(matC);
        reorder_ = reinterpret_cast<__gm__ int32_t *>(
            wsBase_ + static_cast<uint64_t>(tilingData_.reorder_offset));
        binEdge_ = reinterpret_cast<__gm__ int32_t *>(
            wsBase_ + static_cast<uint64_t>(tilingData_.bin_edge_offset));
        n_ = tilingData_.n;
        ldb_ = tilingData_.ldb;
        ldc_ = tilingData_.ldc;
        opB_ = tilingData_.opB;
        orderPair_ = tilingData_.order_pair;
        colTiles_ = static_cast<int32_t>(
            (static_cast<uint64_t>(tilingData_.n) + static_cast<uint64_t>(kChunk) - 1u) /
            static_cast<uint64_t>(kChunk));
    }

    __aicore__ inline void Process()
    {
        const int32_t m = tilingData_.m;
        ScalarT alpha;
        ScalarT beta;
        if constexpr (SpmmIsSame<ScalarT, int32_t>::value) {
            alpha = static_cast<ScalarT>(static_cast<int32_t>(tilingData_.alpha_host));
            beta = static_cast<ScalarT>(static_cast<int32_t>(tilingData_.beta_host));
        } else {
            alpha = static_cast<ScalarT>(tilingData_.alpha_host);
            beta = static_cast<ScalarT>(tilingData_.beta_host);
        }
        const int32_t rowBinNum = static_cast<int32_t>(GetBlockNum());
        const int32_t outerId = static_cast<int32_t>(GetBlockIdx());
        // IMPORTANT: do NOT early-return for out-of-range / empty bins here.
        //
        // On dav-3510 (split AIC/AIV core), asc_vf_call -> cce::async_invoke
        // dispatches the inner SIMT function from each outer AIC core to its
        // paired AIV vector core. That cube->vector dispatch must be issued
        // UNIFORMLY by every outer core. When m < blockDim (e.g. m=2, blockDim=56)
        // most row bins are empty; if those cores returned before asc_vf_call the
        // cube/vector handshake would deadlock and aclrtSynchronizeStream would
        // hang forever (observed as a random mid-batch freeze, all dtypes).
        //
        // Instead, every core falls through to asc_vf_call. Empty bins pass
        // rowStart == rowEnd, which makes SpmmCsrSimtCompute return immediately
        // (numRows <= 0) without touching memory, so they do no work but still
        // participate in the dispatch.
        int32_t rowStart = 0;
        int32_t rowEnd = 0;
        if (outerId >= 0 && outerId < rowBinNum) {
            rowStart = binEdge_[outerId];
            rowEnd = binEdge_[outerId + 1];
        }
        const uint64_t totalWork =
            static_cast<uint64_t>(m) * static_cast<uint64_t>(colTiles_);
        uint32_t simtThreadNum = 1u;
        if (totalWork > 0u && rowBinNum > 0) {
            simtThreadNum = static_cast<uint32_t>((totalWork + static_cast<uint64_t>(rowBinNum) - 1u) /
                                                  static_cast<uint64_t>(rowBinNum));
            if (simtThreadNum > kMaxSimtThreadsPerBlock) {
                simtThreadNum = kMaxSimtThreadsPerBlock;
            }
        }

        asc_vf_call<SpmmCsrSimtCompute<ValT, CT, ScalarT, UseKahan>>(dim3{simtThreadNum},
            rowOff_, colInd_, values_, matB_, matC_,
            reorder_, n_, ldb_, ldc_, colTiles_, rowStart, rowEnd, alpha, beta,
            opB_, orderPair_);
    }

private:
    __gm__ uint8_t *wsBase_{nullptr};
    __gm__ int32_t *rowOff_{nullptr};
    __gm__ int32_t *colInd_{nullptr};
    __gm__ ValT *values_{nullptr};
    __gm__ ValT *matB_{nullptr};
    __gm__ CT *matC_{nullptr};
    __gm__ int32_t *reorder_{nullptr};
    __gm__ int32_t *binEdge_{nullptr};
    int32_t n_{0};
    int32_t ldb_{0};
    int32_t ldc_{0};
    int32_t opB_{0};
    int32_t orderPair_{0};
    int32_t colTiles_{0};
    SpmmTilingData tilingData_{};
};

// ============================================================================
// Kernel entry points — one per data type combination.
//
// All runtime config (data_type, high_precision, shapes, alpha/beta, etc.) is
// read from SpmmTilingData via tilingGM. Signatures are identical across dtypes.
// ============================================================================

extern "C" __global__ __aicore__ void spmm_custom_fp32(
    GM_ADDR csrRowOffsets,
    GM_ADDR csrColInd,
    GM_ADDR csrValues,
    GM_ADDR matB,
    GM_ADDR matC,
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const SpmmTilingData tiling = LoadSpmmTilingData(tilingGM);
    if (tiling.high_precision != 0) {
        KernelSpmmSimt<float, float, float, true> op;
        op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tilingGM);
        op.Process();
    } else {
        KernelSpmmSimt<float, float, float, false> op;
        op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tilingGM);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void spmm_custom_fp16(
    GM_ADDR csrRowOffsets,
    GM_ADDR csrColInd,
    GM_ADDR csrValues,
    GM_ADDR matB,
    GM_ADDR matC,
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelSpmmSimt<__fp16, __fp16, float, false> op;
    op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tilingGM);
    op.Process();
}

extern "C" __global__ __aicore__ void spmm_custom_int8(
    GM_ADDR csrRowOffsets,
    GM_ADDR csrColInd,
    GM_ADDR csrValues,
    GM_ADDR matB,
    GM_ADDR matC,
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelSpmmSimt<int8_t, int32_t, int32_t, false> op;
    op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tilingGM);
    op.Process();
}

// ============================================================================
// Host-side launch dispatcher — dataType/blockDim passed from spmm_host.cpp.
// ============================================================================
extern "C" void spmm_kernel_launch(
    GM_ADDR csrRowOffsets,
    GM_ADDR csrColInd,
    GM_ADDR csrValues,
    GM_ADDR matB,
    GM_ADDR matC,
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM,
    int32_t dataType,
    uint32_t blockDim,
    void *stream)
{
    if (dataType == SPMM_DTYPE_FP32) {
        spmm_custom_fp32<<<blockDim, nullptr, stream>>>(
            csrRowOffsets, csrColInd, csrValues,
            matB, matC, workspaceGM, tilingGM);
    } else if (dataType == SPMM_DTYPE_FP16) {
        spmm_custom_fp16<<<blockDim, nullptr, stream>>>(
            csrRowOffsets, csrColInd, csrValues,
            matB, matC, workspaceGM, tilingGM);
    } else if (dataType == SPMM_DTYPE_INT8) {
        spmm_custom_int8<<<blockDim, nullptr, stream>>>(
            csrRowOffsets, csrColInd, csrValues,
            matB, matC, workspaceGM, tilingGM);
    } else {
        // 未知 dtype：显式报错并拒绝启动，避免静默按 int8 处理导致错误结果。
        fprintf(stderr, "[ERROR] spmm_kernel_launch: unsupported dataType %d "
                        "(expected FP32=%d/FP16=%d/INT8=%d), kernel not launched\n",
                dataType, SPMM_DTYPE_FP32, SPMM_DTYPE_FP16, SPMM_DTYPE_INT8);
    }
}
