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

/**
 * @file sddmm_perf.cpp
 * @brief Standalone performance collector for aclsparseSDDMM (Ascend 950 / arch35).
 *
 * Reuses the golden sparsity generator (sddmm_golden.h) and the three-stage
 * aclsparse workflow, but pre-allocates all device buffers / descriptors once
 * and times ONLY the Execute stage (aclsparseSDDMM) with aclrtEvent across
 * warmup + measured iterations. This isolates the kernel execution time from
 * host-side setup / H2D copy overhead.
 *
 * Per representative shape/dtype it reports:
 *   - kernel time (min / median / nearest-rank p90 / mean, us)
 *   - matched GPU median, GPU/NPU median ratio, and the ratio>=0.3 gate
 *   - FLOPs (2*nnz*k dot-product MADs) and achieved GFLOPS
 *   - algorithmic HBM bytes moved and achieved bandwidth (GB/s)
 *   - bandwidth utilization vs a configurable peak HBM BW
 *
 * This program does NOT modify the operator (sparse/) — it only links against
 * the pre-built libops_sparse.so. Built manually (not via the gtest CMake target).
 */

#include "test_common.h"
#include "sddmm_golden.h"
#include "sddmm_npu_wrapper.h"
#include "sddmm_param.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace sparse_test;

// ============================================================================
// Host data conversion helpers: DoublesToFp32 / DoublesToFp16 are provided by
// sddmm_golden.h (shared with sddmm_test.cpp).
// ============================================================================

// ============================================================================
// Performance case descriptor
// ============================================================================

struct PerfCase {
    std::string label;
    int64_t m;
    int64_t k;
    int64_t n;
    double sparsity_ratio;
    double alpha;
    double beta;
    aclDataType dtype;      // ACL_FLOAT / ACL_FLOAT16 / ACL_BF16 / ACL_COMPLEX64
    std::string dtype_name;
    double value_lo;
    double value_hi;
    uint32_t seed;
    int32_t index_base = 0;
    int32_t nnz_per_row = 0;
    bool y_col_major = false;
    double gpu_baseline_us = 0.0;  // matched GPU Event median; 0 means unavailable
    bool is_bsr = false;
    int32_t block_dim = 1;
    int64_t scalar_nnz = 0;
    aclsparseOrder_t block_order = ACL_SPARSE_ORDER_ROW;
    std::string row_pattern = "uniform";
    int32_t batch = 1;
    aclDataType output_dtype = ACL_DT_UNDEFINED;

    aclDataType OutputDtype() const {
        return output_dtype == ACL_DT_UNDEFINED ? dtype : output_dtype;
    }

    bool UsesNativeBaselineInputs() const {
        return is_bsr || dtype == ACL_COMPLEX64 ||
            (dtype == ACL_FLOAT16 && gpu_baseline_us > 0.0);
    }
};

struct PerfSparsePattern {
    int64_t major_count = 0;
    int64_t stored_entries = 0;
    int64_t value_count = 0;
    std::vector<int32_t> offsets;
    std::vector<int32_t> columns;
    std::vector<double> values;
};

static SddmmCsr MakeFixedNnzPerRowCsr(const PerfCase& pc) {
    SddmmCsr csr;
    csr.m = pc.m;
    csr.n = pc.n;
    csr.rowOffsets.resize(static_cast<size_t>(pc.m + 1));
    csr.nnz = pc.m * static_cast<int64_t>(pc.nnz_per_row);
    csr.colIndices.resize(static_cast<size_t>(csr.nnz));
    csr.values.resize(static_cast<size_t>(csr.nnz));

    std::mt19937 rng(pc.seed);
    std::uniform_real_distribution<double> valuesDist(pc.value_lo, pc.value_hi);
    std::vector<int32_t> rowColumns(static_cast<size_t>(pc.nnz_per_row));
    for (int64_t row = 0; row < pc.m; ++row) {
        csr.rowOffsets[static_cast<size_t>(row)] =
            static_cast<int32_t>(row * pc.nnz_per_row + pc.index_base);
        const int64_t first = (pc.seed + row * 17ULL) % pc.n;
        for (int32_t offset = 0; offset < pc.nnz_per_row; ++offset) {
            rowColumns[static_cast<size_t>(offset)] = static_cast<int32_t>(
                (first + offset) % pc.n);
        }
        std::sort(rowColumns.begin(), rowColumns.end());
        const int64_t rowStart = row * pc.nnz_per_row;
        for (int32_t offset = 0; offset < pc.nnz_per_row; ++offset) {
            const size_t index = static_cast<size_t>(rowStart + offset);
            csr.colIndices[index] = rowColumns[static_cast<size_t>(offset)] + pc.index_base;
            csr.values[index] = valuesDist(rng);
        }
    }
    csr.rowOffsets[static_cast<size_t>(pc.m)] =
        static_cast<int32_t>(csr.nnz + pc.index_base);
    return csr;
}

static PerfSparsePattern MakeCsrPattern(const PerfCase& pc) {
    SddmmCsr csr = pc.nnz_per_row > 0 ? MakeFixedNnzPerRowCsr(pc) :
        MakeSddmmSparsity(pc.m, pc.n, pc.sparsity_ratio,
                          pc.value_lo, pc.value_hi, pc.seed);
    PerfSparsePattern pattern;
    pattern.major_count = pc.m;
    pattern.stored_entries = csr.nnz;
    pattern.value_count = csr.nnz;
    pattern.offsets = std::move(csr.rowOffsets);
    pattern.columns = std::move(csr.colIndices);
    pattern.values = std::move(csr.values);
    return pattern;
}

// Mirrors performance_cases.json -> operator_adapter._bsr ->
// sparse_patterns.csr_structure for the anchor cases. The JSON default
// row_pattern is "uniform": floor(blockNnz/blockRows) blocks go to every row,
// then the first remainder rows receive one additional block. Columns use the
// exact (seed + row*17 + offset) modulo blockCols rule and are sorted per row.
static void ValidateBsrPerfMetadata(const PerfCase &pc)
{
    if (pc.row_pattern != "uniform" || pc.block_dim <= 0 ||
        pc.m % pc.block_dim != 0 || pc.n % pc.block_dim != 0 ||
        pc.batch != 1) {
        throw std::runtime_error("invalid BSR performance case metadata");
    }
}

static PerfSparsePattern MakeBsrUniformPattern(const PerfCase& pc) {
    ValidateBsrPerfMetadata(pc);
    const int64_t blockArea = static_cast<int64_t>(pc.block_dim) * pc.block_dim;
    if (pc.scalar_nnz <= 0 || pc.scalar_nnz % blockArea != 0) {
        throw std::runtime_error("BSR scalar nnz must be a positive multiple of block_dim^2");
    }
    const int64_t blockRows = pc.m / pc.block_dim;
    const int64_t blockCols = pc.n / pc.block_dim;
    const int64_t blockNnz = pc.scalar_nnz / blockArea;
    if (blockNnz > blockRows * blockCols) {
        throw std::runtime_error("BSR block nnz exceeds matrix capacity");
    }

    PerfSparsePattern pattern;
    pattern.major_count = blockRows;
    pattern.stored_entries = blockNnz;
    pattern.value_count = pc.scalar_nnz;
    pattern.offsets.resize(static_cast<size_t>(blockRows + 1));
    pattern.columns.resize(static_cast<size_t>(blockNnz));
    pattern.values.assign(static_cast<size_t>(pc.scalar_nnz), 0.0);

    const int64_t blocksPerRow = blockNnz / blockRows;
    const int64_t remainder = blockNnz % blockRows;
    int64_t blockOffset = 0;
    for (int64_t row = 0; row < blockRows; ++row) {
        pattern.offsets[static_cast<size_t>(row)] =
            static_cast<int32_t>(blockOffset + pc.index_base);
        const int64_t rowCount = blocksPerRow + (row < remainder ? 1 : 0);
        if (rowCount > blockCols) {
            throw std::runtime_error("uniform BSR row exceeds block column count");
        }
        std::vector<int32_t> rowColumns(static_cast<size_t>(rowCount));
        const int64_t first = (static_cast<uint64_t>(pc.seed) +
            static_cast<uint64_t>(row) * 17ULL) % static_cast<uint64_t>(blockCols);
        for (int64_t offset = 0; offset < rowCount; ++offset) {
            rowColumns[static_cast<size_t>(offset)] = static_cast<int32_t>(
                (first + offset) % blockCols);
        }
        std::sort(rowColumns.begin(), rowColumns.end());
        for (int64_t offset = 0; offset < rowCount; ++offset) {
            pattern.columns[static_cast<size_t>(blockOffset + offset)] =
                rowColumns[static_cast<size_t>(offset)] + pc.index_base;
        }
        blockOffset += rowCount;
    }
    pattern.offsets[static_cast<size_t>(blockRows)] =
        static_cast<int32_t>(blockNnz + pc.index_base);
    return pattern;
}

static std::vector<uint16_t> DoublesToBf16(const std::vector<double>& input) {
    std::vector<uint16_t> result(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(input[index]);
        uint32_t bits = 0;
        std::copy_n(reinterpret_cast<const unsigned char *>(&value), sizeof(bits),
                    reinterpret_cast<unsigned char *>(&bits));
        if ((bits & 0x7F800000U) == 0x7F800000U &&
            (bits & 0x007FFFFFU) != 0U) {
            result[index] = static_cast<uint16_t>((bits >> 16) | 0x0040U);
        } else {
            const uint32_t lsb = (bits >> 16) & 1U;
            result[index] = static_cast<uint16_t>((bits + 0x7FFFU + lsb) >> 16);
        }
    }
    return result;
}

// ============================================================================
// Timed three-stage workflow.
//
// Setup (once): allocate device buffers + descriptors, run BufferSize +
// Preprocess. The Preprocess stage writes tiling / reorder / bin_edge into the
// workspace buffer; Execute only launches the kernel, so timing Execute in a
// loop gives a clean kernel-time measurement.
//
// PerfWorkspace aggregates non-copyable RAII handles (HandleManager /
// DnMatManager / SpMatManager / DeviceBuffer). HandleManager has no move ctor,
// so PerfWorkspace is non-movable — therefore it is heap-allocated via
// unique_ptr and returned by pointer (no copy/move needed).
//
// T: float (FP32) or uint16_t (FP16 bit patterns).
// ============================================================================
class PerfBsrMat {
public:
    PerfBsrMat() = default;

    ~PerfBsrMat() {
        if (descr_ != nullptr) {
            (void)aclsparseDestroySpMat(descr_);
        }
    }

    void Create(const PerfCase& pc, const PerfSparsePattern& pattern,
                void* offsets, void* columns, void* values) {
        const auto base = pc.index_base == 0 ? ACL_SPARSE_INDEX_BASE_ZERO :
                                               ACL_SPARSE_INDEX_BASE_ONE;
        const aclsparseStatus_t status = aclsparseCreateBsr(
            &descr_, pc.m / pc.block_dim, pc.n / pc.block_dim,
            pattern.stored_entries, pc.block_dim, pc.block_dim,
            offsets, columns, values,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            base, pc.OutputDtype(), pc.block_order);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseCreateBsr failed: " +
                                     std::to_string(static_cast<int>(status)));
        }
    }

    aclsparseSpMatDescr_t get() const { return descr_; }

    PerfBsrMat(const PerfBsrMat&) = delete;
    PerfBsrMat& operator=(const PerfBsrMat&) = delete;

private:
    aclsparseSpMatDescr_t descr_ = nullptr;
};

template <typename InputT, typename OutputT>
struct PerfWorkspace {
    HandleManager handle;
    DeviceBuffer dRowOff;
    DeviceBuffer dColInd;
    DeviceBuffer dVals;
    DeviceBuffer dX;
    DeviceBuffer dY;
    DeviceBuffer dBuffer;
    DnMatManager matX;
    DnMatManager matY;
    SpMatManager matC;
    PerfBsrMat matBsr;
    bool isBsr;
    aclDataType dtype;
    aclDataType computeType;
    aclsparseSDDMMAlg_t alg;
    aclsparseOperation_t opX;
    aclsparseOperation_t opY;
    float alphaHost;
    float betaHost;
    aclsparseComplex alphaComplexHost;
    aclsparseComplex betaComplexHost;
    size_t bufferSize = 0;
    int64_t nnz;
    int64_t m, k, n;
    bool ok;
    std::string err;

    aclsparseSpMatDescr_t SparseMatrix() {
        return isBsr ? matBsr.get() : matC.get();
    }

    const void* Alpha() const {
        return computeType == ACL_COMPLEX64 ?
            static_cast<const void*>(&alphaComplexHost) :
            static_cast<const void*>(&alphaHost);
    }

    const void* Beta() const {
        return computeType == ACL_COMPLEX64 ?
            static_cast<const void*>(&betaComplexHost) :
            static_cast<const void*>(&betaHost);
    }
};

template <typename InputT, typename OutputT>
static void SetupPerfDescriptors(PerfWorkspace<InputT, OutputT> *ws, aclrtStream stream,
                                 const PerfCase& pc, const PerfSparsePattern& pattern,
                                 const std::vector<OutputT>& hCInit,
                                 const std::vector<InputT>& hX,
                                 const std::vector<InputT>& hY) {
    ws->dtype = pc.dtype;
    // Real low-precision values accumulate with FP32 computeType. Complex64
    // values use complex compute/scalars, matching the public type matrix.
    ws->computeType = pc.dtype == ACL_COMPLEX64 ? ACL_COMPLEX64 : ACL_FLOAT;
    ws->alg = ACL_SPARSE_SDDMM_ALG_DEFAULT;
    ws->opX = ACL_SPARSE_OP_NON_TRANSPOSE;
    ws->opY = ACL_SPARSE_OP_NON_TRANSPOSE;
    ws->alphaHost = static_cast<float>(pc.alpha);
    ws->betaHost = static_cast<float>(pc.beta);
    ws->alphaComplexHost = {static_cast<float>(pc.alpha), 0.0F};
    ws->betaComplexHost = {static_cast<float>(pc.beta), 0.0F};
    ws->m = pc.m;
    ws->k = pc.k;
    ws->n = pc.n;
    ws->nnz = pattern.value_count;
    ws->isBsr = pc.is_bsr;
    ws->handle.setStream(stream);

    ws->dRowOff = DeviceBuffer::copyFrom(pattern.offsets.data(),
                                         pattern.offsets.size() * sizeof(int32_t));
    if (pattern.stored_entries > 0) {
        ws->dColInd = DeviceBuffer::copyFrom(pattern.columns.data(),
                                             pattern.columns.size() * sizeof(int32_t));
        ws->dVals = DeviceBuffer::copyFrom(hCInit.data(),
                                           static_cast<size_t>(pattern.value_count) * sizeof(OutputT));
    }
    ws->dX = DeviceBuffer::copyFrom(hX.data(),
                                    static_cast<size_t>(pc.m) * static_cast<size_t>(pc.k) * sizeof(InputT));
    ws->dY = DeviceBuffer::copyFrom(hY.data(),
                                    static_cast<size_t>(pc.n) * static_cast<size_t>(pc.k) * sizeof(InputT));

    ws->matX = DnMatManager::createConst(pc.m, pc.k, pc.k, ws->dX.raw(), pc.dtype, ACL_SPARSE_ORDER_ROW);
    ws->matY = DnMatManager::createConst(pc.k, pc.n, pc.y_col_major ? pc.k : pc.n,
                                         ws->dY.raw(), pc.dtype,
                                         pc.y_col_major ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW);
    if (pc.is_bsr) {
        ws->matBsr.Create(pc, pattern, ws->dRowOff.get(), ws->dColInd.get(), ws->dVals.get());
    } else {
        ws->matC = SpMatManager::createCsr(
            pc.m, pc.n, pattern.stored_entries,
            ws->dRowOff.get(), ws->dColInd.get(), ws->dVals.get(),
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            pc.index_base == 0 ? ACL_SPARSE_INDEX_BASE_ZERO :
                                 ACL_SPARSE_INDEX_BASE_ONE,
            pc.OutputDtype());
    }
}

template <typename InputT, typename OutputT>
static bool SetupPerfBuffers(PerfWorkspace<InputT, OutputT> *ws, aclrtStream stream) {
    size_t bs = 0;
    auto r1 = aclsparseSDDMMBufferSize(ws->handle.get(), ws->opX, ws->opY, ws->Alpha(),
                                       ws->matX.cget(), ws->matY.cget(), ws->Beta(),
                                       ws->SparseMatrix(),
                                       ws->computeType, ws->alg, &bs);
    if (r1 != ACL_SPARSE_STATUS_SUCCESS) {
        ws->err = "BufferSize failed: " + std::to_string(r1);
        return false;
    }
    ws->bufferSize = bs;
    if (bs > 0) {
        ws->dBuffer = DeviceBuffer::alloc(bs);
    }
    auto r2 = aclsparseSDDMMPreprocess(ws->handle.get(), ws->opX, ws->opY, ws->Alpha(),
                                       ws->matX.cget(), ws->matY.cget(), ws->Beta(),
                                       ws->SparseMatrix(),
                                       ws->computeType, ws->alg, ws->dBuffer.get());
    if (r2 != ACL_SPARSE_STATUS_SUCCESS) {
        ws->err = "Preprocess failed: " + std::to_string(r2);
        return false;
    }
    // Sync after preprocess to ensure workspace is ready before timing and to
    // surface asynchronous setup failures in the per-case status.
    const auto syncStatus = aclrtSynchronizeStream(stream);
    if (syncStatus != ACL_SUCCESS) {
        ws->err = "Preprocess synchronize failed: " +
            std::to_string(static_cast<int>(syncStatus));
        return false;
    }
    return true;
}

template <typename InputT, typename OutputT>
static std::unique_ptr<PerfWorkspace<InputT, OutputT>> SetupPerf(
    aclrtStream stream, const PerfCase& pc, const PerfSparsePattern& pattern,
    const std::vector<OutputT>& hCInit, const std::vector<InputT>& hX,
    const std::vector<InputT>& hY) {
    auto ws = std::make_unique<PerfWorkspace<InputT, OutputT>>();
    ws->ok = false;
    SetupPerfDescriptors<InputT, OutputT>(
        ws.get(), stream, pc, pattern, hCInit, hX, hY);
    ws->ok = SetupPerfBuffers<InputT, OutputT>(ws.get(), stream);
    return ws;
}

struct TimedExecution {
    std::vector<double> times;
    bool ok = false;
    std::string error;
};

static bool CheckTimingStatus(aclError status, const char *phase, TimedExecution &result)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    result.error = std::string(phase) + " failed: " +
        std::to_string(static_cast<int>(status));
    return false;
}

struct PerfEventPair {
    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;

    ~PerfEventPair()
    {
        if (start != nullptr) {
            aclrtDestroyEvent(start);
        }
        if (stop != nullptr) {
            aclrtDestroyEvent(stop);
        }
    }

    bool Create(TimedExecution &result)
    {
        return CheckTimingStatus(aclrtCreateEvent(&start), "Event setup", result) &&
            CheckTimingStatus(aclrtCreateEvent(&stop), "Event setup", result);
    }
};

template <typename ExecFn>
static bool ExecuteTimingCall(ExecFn &exec, const char *phase, int iteration,
                              TimedExecution &result)
{
    const auto status = exec();
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        return true;
    }
    result.error = std::string("Execute ") + phase + " iteration " +
        std::to_string(iteration) + " failed: " +
        std::to_string(static_cast<int>(status));
    return false;
}

template <typename ExecFn>
static bool WarmupExecute(aclrtStream stream, ExecFn &exec, int warmup,
                          TimedExecution &result)
{
    for (int i = 0; i < warmup; ++i) {
        if (!ExecuteTimingCall(exec, "warmup", i, result)) {
            return false;
        }
    }
    return CheckTimingStatus(aclrtSynchronizeStream(stream),
        "Warmup synchronize", result);
}

template <typename ExecFn>
static bool MeasureExecute(aclrtStream stream, ExecFn &exec, int iteration,
                           const PerfEventPair &events, TimedExecution &result)
{
    if (!CheckTimingStatus(aclrtRecordEvent(events.start, stream),
            "Start event record", result) ||
        !ExecuteTimingCall(exec, "measured", iteration, result) ||
        !CheckTimingStatus(aclrtRecordEvent(events.stop, stream),
            "Stop event record", result) ||
        !CheckTimingStatus(aclrtSynchronizeStream(stream),
            "Measured synchronize", result)) {
        return false;
    }
    float ms = 0.0F;
    if (!CheckTimingStatus(aclrtEventElapsedTime(&ms, events.start, events.stop),
            "Event elapsed-time query", result)) {
        return false;
    }
    if (!std::isfinite(ms) || ms < 0.0F) {
        result.error = "Event elapsed-time query returned an invalid value";
        return false;
    }
    result.times.push_back(static_cast<double>(ms) * 1000.0); // us
    return true;
}

// Keep one Event pair around each measured Execute, after all warmup calls.
// Errors stop the case before reporting any successful timing status.
template <typename ExecFn>
static TimedExecution TimeExecute(aclrtStream stream, ExecFn exec, int warmup, int iters)
{
    TimedExecution result;
    result.times.reserve(static_cast<size_t>(iters));
    PerfEventPair events;
    if (!events.Create(result) || !WarmupExecute(stream, exec, warmup, result)) {
        return result;
    }
    for (int i = 0; i < iters; ++i) {
        if (!MeasureExecute(stream, exec, i, events, result)) {
            return result;
        }
    }
    result.ok = true;
    return result;
}

static double MedianOfSorted(const std::vector<double>& sorted) {
    if (sorted.empty()) {
        return 0.0;
    }
    const size_t middle = sorted.size() / 2;
    if ((sorted.size() & 1U) != 0U) {
        return sorted[middle];
    }
    return (sorted[middle - 1] + sorted[middle]) * 0.5;
}

// Nearest-rank percentile: rank=ceil(p*N), with rank starting at one.
static double PercentileNearestRankOfSorted(const std::vector<double>& sorted, double percentile) {
    if (sorted.empty()) {
        return 0.0;
    }
    const size_t rank = static_cast<size_t>(
        std::ceil(percentile * static_cast<double>(sorted.size())));
    return sorted[std::max<size_t>(1, std::min(rank, sorted.size())) - 1];
}

// ============================================================================
// Algorithmic metrics
// ============================================================================
//
// SDDMM: C_out[p] = alpha * (X[i,:] . Y[j,:]) + beta * C[p]  for each nonzero p.
//
// FLOPs: each nonzero needs a length-k dot product (k mul + k-1 add ~ 2k MADs),
//        plus alpha-scale and (optional) beta*fused-add per nonzero (~2 ops).
//        Dominant term: 2 * nnz * k.
//
// HBM bytes (algorithmic lower bound the kernel must move):
//   X read     : m * k * inputElem   (each X element read >= 1x)
//   Y read     : nnz * k * inputElem (kernel loads Y[j,:] per nonzero; dominates)
//   colInd read: stored sparse entries * 4 (CSR nnz or BSR blockNnz)
//   rowOff read: (major rows + 1) * 4 (m or BSR blockRows)
//   C read     : beta != 0 ? nnz * outputElem : 0
//   C write    : nnz * outputElem
//
// Input and output element widths are accounted independently for mixed dtype.

struct AlgoMetrics {
    int64_t nnz;
    double flops;
    double bytes;
};

static AlgoMetrics ComputeAlgoMetrics(const PerfCase& pc, int64_t nnz) {
    AlgoMetrics m;
    m.nnz = nnz;
    const bool isComplex = pc.dtype == ACL_COMPLEX64;
    const double inputElem = isComplex ? 8.0 :
        (pc.dtype == ACL_FLOAT ? 4.0 : 2.0);
    const aclDataType outputType = pc.OutputDtype();
    const double outputElem = outputType == ACL_COMPLEX64 ? 8.0 :
        (outputType == ACL_FLOAT ? 4.0 : 2.0);
    const double dotFlops = isComplex ? 8.0 : 2.0;
    const double scalarFlops = isComplex ? 14.0 : 2.0;
    double flops = dotFlops * static_cast<double>(nnz) * static_cast<double>(pc.k);
    flops += scalarFlops * static_cast<double>(nnz);
    m.flops = flops;

    double bytes = 0.0;
    const double sparseEntries = pc.is_bsr ?
        static_cast<double>(nnz) /
            static_cast<double>(pc.block_dim * pc.block_dim) :
        static_cast<double>(nnz);
    const double majorRows = pc.is_bsr ?
        static_cast<double>(pc.m / pc.block_dim) : static_cast<double>(pc.m);
    bytes += static_cast<double>(pc.m) * static_cast<double>(pc.k) * inputElem; // X
    bytes += static_cast<double>(nnz) * static_cast<double>(pc.k) * inputElem;  // Y
    bytes += sparseEntries * 4.0;                                            // colInd
    bytes += (majorRows + 1.0) * 4.0;                                        // rowOff
    if (pc.beta != 0.0) {
        bytes += static_cast<double>(nnz) * outputElem;                      // C read
    }
    bytes += static_cast<double>(nnz) * outputElem;                          // C write
    m.bytes = bytes;
    return m;
}

// ============================================================================
// HBM bandwidth benchmark (aclrtMemcpy D2D, DMA engine).
//
// Measures the practical peak HBM bandwidth achievable on this device using
// device-to-device memcpy over a large buffer. A D2D copy reads `size` bytes
// from HBM and writes `size` bytes to HBM, so total HBM traffic = 2*size per
// iteration. For large transfers the DMA engine (MTE2/MTE3) approaches the
// device peak HBM bandwidth, giving a defensible practical-peak denominator
// for utilization computation (msprof aic-metrics is unavailable for custom
// ops loaded via shared library on this CANN build).
// ============================================================================
static void PrintBandwidthSummary(std::vector<double> &times, size_t size, size_t sizeMiB)
{
    std::sort(times.begin(), times.end());
    double tMin = times.front();
    double tMed = MedianOfSorted(times);
    // total HBM traffic = 2*size (read + write), bandwidth = 2*size / time
    double bytesTraffic = 2.0 * static_cast<double>(size);
    double bwMin = bytesTraffic / (tMin / 1000.0) / 1e9;   // GB/s
    double bwMed = bytesTraffic / (tMed / 1000.0) / 1e9;

    std::cout << "=== HBM Bandwidth Benchmark (aclrtMemcpy D2D, " << sizeMiB
              << " MiB) ===\n";
    std::cout << "  copy time: min=" << std::fixed << std::setprecision(3) << tMin
              << " ms  median=" << tMed << " ms\n";
    std::cout << "  HBM BW (2*size/time): min=" << bwMin << " GB/s  median=" << bwMed
              << " GB/s\n";
    std::cout << "  -> use median " << bwMed << " GB/s as practical peak HBM BW\n";
    std::cout << std::defaultfloat;

}

static int RunBwBench(aclrtStream stream, int warmup, int iters) {
    // 1 GiB buffer — large enough to saturate HBM, fits in 123 GiB HBM.
    const size_t sizeMiB = 1024;
    const size_t size = sizeMiB * 1024ULL * 1024ULL;

    void* dSrc = nullptr;
    void* dDst = nullptr;
    auto r1 = aclrtMalloc(&dSrc, size, ACL_MEM_MALLOC_HUGE_FIRST);
    auto r2 = aclrtMalloc(&dDst, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (r1 != ACL_SUCCESS || r2 != ACL_SUCCESS) {
        std::cerr << "bwbench: aclrtMalloc failed r1=" << r1 << " r2=" << r2 << std::endl;
        return 1;
    }
    // touch pages (avoid first-touch overhead in measured iters)
    aclrtMemset(dSrc, size, 0, size);
    aclrtMemset(dDst, size, 0, size);
    aclrtSynchronizeStream(stream);

    aclrtEvent startEv = nullptr, stopEv = nullptr;
    aclrtCreateEvent(&startEv);
    aclrtCreateEvent(&stopEv);

    for (int i = 0; i < warmup; ++i) {
        aclrtMemcpy(dDst, size, dSrc, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    }
    aclrtSynchronizeStream(stream);

    std::vector<double> times;
    for (int i = 0; i < iters; ++i) {
        aclrtRecordEvent(startEv, stream);
        aclrtMemcpy(dDst, size, dSrc, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
        aclrtRecordEvent(stopEv, stream);
        aclrtSynchronizeStream(stream);
        float ms = 0.0f;
        aclrtEventElapsedTime(&ms, startEv, stopEv);
        times.push_back(static_cast<double>(ms));  // ms
    }
    PrintBandwidthSummary(times, size, sizeMiB);

    aclrtDestroyEvent(startEv);
    aclrtDestroyEvent(stopEv);
    aclrtFree(dSrc);
    aclrtFree(dDst);
    return 0;
}

// ============================================================================
// Main
// ============================================================================

struct RowOut {
    std::string label;
    std::string dtype_name;
    int64_t nnz = 0;
    size_t workspaceSizeInBytes = 0;
    double tMin = 0.0;
    double tMed = 0.0;
    double tP90 = 0.0;
    double tMean = 0.0;
    double bwGbs = 0.0;
    double gflops = 0.0;
    double bwUtil = -1.0;
    double gpuBaselineUs = 0.0;
    double gpuNpuRatio = -1.0;
    bool ok = false;
    std::string note;
};

static std::string Fmt(double v, int precision = 9) {
    if (v < 0.0) return std::string("-");
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << v;
    return os.str();
}

static std::string JsonEscape(const std::string& value) {
    std::ostringstream escaped;
    for (const unsigned char character : value) {
        switch (character) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\b': escaped << "\\b"; break;
            case '\f': escaped << "\\f"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (character < 0x20U) {
                    escaped << "\\u00" << std::hex << std::setw(2)
                            << std::setfill('0') << static_cast<int>(character)
                            << std::dec << std::setfill(' ');
                } else {
                    escaped << static_cast<char>(character);
                }
                break;
        }
    }
    return escaped.str();
}

struct PerfArgs {
    int warmup = 10;
    int iters = 30;
    double peakHbmGbps = 0.0;  // 0 = unknown
    std::string only;          // if non-empty, run only cases whose label contains this substring
    bool bwbench = false;      // if true, run HBM bandwidth benchmark and exit
};

static PerfArgs ParsePerfArgs(int argc, char** argv) {
    PerfArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto pos = a.find('=');
        if (pos == std::string::npos) {
            if (a == "--bwbench") { args.bwbench = true; }
            continue;
        }
        std::string key = a.substr(0, pos);
        std::string val = a.substr(pos + 1);
        if (key == "--warmup") args.warmup = std::stoi(val);
        else if (key == "--iters") args.iters = std::stoi(val);
        else if (key == "--peak") args.peakHbmGbps = std::stod(val);
        else if (key == "--only") args.only = val;
    }
    return args;
}

template <typename OutputT>
static bool VerifyNativeOutput(const PerfCase &pc, size_t count,
                               const void *deviceValues, RowOut &row)
{
    // The native GPU benchmark uses exactly representable constant inputs.
    // Check every output against K * 0.125 * 0.25, outside the Event interval.
    if (pc.UsesNativeBaselineInputs() && pc.alpha == 1.0 && pc.beta == 0.0) {
        std::vector<OutputT> actual(count);
        const size_t bytes = actual.size() * sizeof(OutputT);
        if (aclrtMemcpy(actual.data(), bytes, deviceValues, bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            row.note = "VERIFY_FAIL:output copy failed";
            return false;
        }
        const float expected = static_cast<float>(pc.k) * 0.03125F;
        const uint16_t expectedBits = pc.OutputDtype() == ACL_BF16 ?
            DoublesToBf16({expected})[0] : DoublesToFp16({expected})[0];
        for (const auto& value : actual) {
            bool match = false;
            if constexpr (std::is_same_v<OutputT, aclsparseComplex>) {
                match = value.x == expected && value.y == 0.0F;
            } else if constexpr (std::is_same_v<OutputT, float>) {
                match = value == expected;
            } else {
                match = value == expectedBits;
            }
            if (!match) {
                row.note = "VERIFY_FAIL:native constant input result mismatch";
                return false;
            }
        }
    }
    return true;
}

template <typename InputT, typename OutputT>
static void RunPerfTyped(aclrtStream stream, const PerfCase& pc,
                         const PerfSparsePattern& pattern,
                         const std::vector<OutputT>& hCInit,
                         const std::vector<InputT>& hX,
                         const std::vector<InputT>& hY,
                         int warmup, int iters, RowOut& row) {
    auto ws = SetupPerf<InputT, OutputT>(stream, pc, pattern, hCInit, hX, hY);
    row.workspaceSizeInBytes = ws->bufferSize;
    if (!ws->ok) {
        row.note = "SETUP_FAIL:" + ws->err;
        return;
    }
    auto exec = [&]() {
        return aclsparseSDDMM(ws->handle.get(), ws->opX, ws->opY, ws->Alpha(),
                              ws->matX.cget(), ws->matY.cget(), ws->Beta(),
                              ws->SparseMatrix(),
                              ws->computeType, ws->alg, ws->dBuffer.get());
    };
    auto timing = TimeExecute(stream, exec, warmup, iters);
    if (!timing.ok || timing.times.empty()) {
        ws->ok = false;
        ws->err = timing.error.empty() ? "no timing samples" : timing.error;
        row.note = "EXECUTE_FAIL:" + ws->err;
        return;
    }
    if (!VerifyNativeOutput<OutputT>(pc, hCInit.size(), ws->dVals.get(), row)) {
        return;
    }
    std::sort(timing.times.begin(), timing.times.end());
    row.tMin = timing.times.front();
    row.tMed = MedianOfSorted(timing.times);
    row.tP90 = PercentileNearestRankOfSorted(timing.times, 0.9);
    double sum = 0;
    for (double t : timing.times) {
        sum += t;
    }
    row.tMean = sum / timing.times.size();
    row.ok = true;
    row.note = "ok";
}

static void PreparePerfInputs(const PerfCase &pc, PerfSparsePattern &pattern,
                              std::vector<double> &Xf64, std::vector<double> &Yf64)
{
    if (pc.UsesNativeBaselineInputs()) {
        // benchmark_cusparse_gpu.cu, which produced the matched TSV baselines,
        // fills both CSR and BSR A with 0.125, B with 0.25, and C with zero.
        std::fill(Xf64.begin(), Xf64.end(), 0.125);
        std::fill(Yf64.begin(), Yf64.end(), 0.25);
        std::fill(pattern.values.begin(), pattern.values.end(), 0.0);
    } else {
        std::mt19937 rngX(pc.seed + 1);
        std::mt19937 rngY(pc.seed + 2);
        std::uniform_real_distribution<double> dist(pc.value_lo, pc.value_hi);
        for (size_t i = 0; i < Xf64.size(); i++) Xf64[i] = dist(rngX);
        for (size_t i = 0; i < Yf64.size(); i++) Yf64[i] = dist(rngY);
    }
}

static void DispatchPerfCase(aclrtStream stream, const PerfCase &pc,
                             const PerfSparsePattern &pattern,
                             const std::vector<double> &Xf64,
                             const std::vector<double> &Yf64,
                             int warmup, int iters, RowOut &row)
{
    if (pc.dtype == ACL_COMPLEX64) {
        // Match benchmark_cusparse_gpu.cu exactly: complex inputs have zero
        // imaginary parts, A=0.125, B=0.25, and initial sparse values are zero.
        const aclsparseComplex zero{0.0F, 0.0F};
        const aclsparseComplex xValue{0.125F, 0.0F};
        const aclsparseComplex yValue{0.25F, 0.0F};
        std::vector<aclsparseComplex> hCInit(static_cast<size_t>(pattern.value_count), zero);
        std::vector<aclsparseComplex> hX(Xf64.size(), xValue);
        std::vector<aclsparseComplex> hY(Yf64.size(), yValue);
        RunPerfTyped<aclsparseComplex, aclsparseComplex>(
            stream, pc, pattern, hCInit, hX, hY, warmup, iters, row);
    } else if (pc.dtype == ACL_FLOAT) {
        RunPerfTyped<float, float>(stream, pc, pattern,
                                   DoublesToFp32(pattern.values),
                                   DoublesToFp32(Xf64), DoublesToFp32(Yf64),
                                   warmup, iters, row);
    } else if (pc.dtype == ACL_BF16) {
        if (pc.OutputDtype() == ACL_FLOAT) {
            RunPerfTyped<uint16_t, float>(stream, pc, pattern,
                                          DoublesToFp32(pattern.values),
                                          DoublesToBf16(Xf64), DoublesToBf16(Yf64),
                                          warmup, iters, row);
        } else {
            RunPerfTyped<uint16_t, uint16_t>(stream, pc, pattern,
                                             DoublesToBf16(pattern.values),
                                             DoublesToBf16(Xf64), DoublesToBf16(Yf64),
                                             warmup, iters, row);
        }
    } else {
        if (pc.OutputDtype() == ACL_FLOAT) {
            RunPerfTyped<uint16_t, float>(stream, pc, pattern,
                                          DoublesToFp32(pattern.values),
                                          DoublesToFp16(Xf64), DoublesToFp16(Yf64),
                                          warmup, iters, row);
        } else {
            RunPerfTyped<uint16_t, uint16_t>(stream, pc, pattern,
                                             DoublesToFp16(pattern.values),
                                             DoublesToFp16(Xf64), DoublesToFp16(Yf64),
                                             warmup, iters, row);
        }
    }
}

static RowOut RunOneCase(aclrtStream stream, const PerfCase& pc, int warmup, int iters,
                         double peakHbmGbps) {
    RowOut row;
    row.label = pc.label;
    row.dtype_name = pc.dtype_name;
    row.gpuBaselineUs = pc.gpu_baseline_us;
    row.note = "NOT_RUN";

    PerfSparsePattern pattern = pc.is_bsr ? MakeBsrUniformPattern(pc) : MakeCsrPattern(pc);
    int64_t nnz = pattern.value_count;
    row.nnz = nnz;

    std::vector<double> Xf64(static_cast<size_t>(pc.m) * static_cast<size_t>(pc.k));
    std::vector<double> Yf64(static_cast<size_t>(pc.n) * static_cast<size_t>(pc.k));
    PreparePerfInputs(pc, pattern, Xf64, Yf64);
    DispatchPerfCase(stream, pc, pattern, Xf64, Yf64, warmup, iters, row);

    auto am = ComputeAlgoMetrics(pc, nnz);
    double tUs = row.tMed;
    if (row.ok && tUs > 0) {
        double tSec = tUs / 1e6;
        row.bwGbs = am.bytes / tSec / 1e9;
        row.gflops = am.flops / tSec / 1e9;
        row.bwUtil = (peakHbmGbps > 0) ? (row.bwGbs / peakHbmGbps * 100.0) : -1.0;
        if (row.gpuBaselineUs > 0) {
            row.gpuNpuRatio = row.gpuBaselineUs / tUs;
        }
    }
    return row;
}

static bool InitAclEnv(aclrtStream &stream) {
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "aclInit failed" << std::endl;
        return false;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed" << std::endl;
        aclFinalize();
        return false;
    }
    const auto streamStatus = aclrtCreateStream(&stream);
    if (streamStatus != ACL_SUCCESS) {
        std::cerr << "aclrtCreateStream failed: " << streamStatus << std::endl;
        aclrtResetDevice(0);
        aclFinalize();
        return false;
    }
    return true;
}

static void CleanupAclEnv(aclrtStream stream) {
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

static bool HasPerformanceBaseline(const RowOut& row) {
    return row.gpuBaselineUs > 0.0;
}

static bool PerformanceGatePassed(const RowOut& row) {
    return HasPerformanceBaseline(row) && row.ok &&
        row.gpuNpuRatio >= 0.3;
}

static void PrintHeader(int64_t arch, size_t totalMem, size_t freeMem,
                        bool hbmMemInfoAvailable,
                        int warmup, int iters, double peakHbmGbps) {
    std::cout << "============================================================\n";
    std::cout << "SDDMM Performance Collection (aclsparseSDDMM, Execute stage)\n";
    std::cout << "Device: Ascend arch=" << arch << " (DAV_3510 / Ascend950)\n";
    if (hbmMemInfoAvailable) {
        std::cout << "HBM total=" << (totalMem >> 20)
                  << " MiB  free=" << (freeMem >> 20) << " MiB\n";
    } else {
        std::cout << "HBM total=N/A  free=N/A\n";
    }
    std::cout << "warmup=" << warmup << " iters=" << iters
              << " peakHbmGbps=" << (peakHbmGbps > 0 ? std::to_string(peakHbmGbps) : std::string("N/A"))
              << "\n";
    std::cout << "Timing: aclrtEvent around aclsparseSDDMM (Execute), per-iter sync\n";
    std::cout << "Percentiles: p90 uses nearest rank ceil(0.90*N)\n";
    std::cout << "Gate: matched GPU median / NPU median >= 0.3\n";
    std::cout << "============================================================\n";

    printf("+------------------------------------+-----------+--------------+-----------------+-----------------+"
           "-----------------+-----------------+-----------------+-----------------+-------+-----------------+"
           "-----------------+-----------------+----------------------------------+\n");
    printf("| %-34s | %-9s | %12s | %15s | %15s | %15s | %15s | %15s | %15s | %5s | %15s | %15s | %15s | %-32s |\n",
           "case", "dtype", "nnz", "min(us)", "med(us)", "p90(us)", "mean(us)",
           "GPUmed(us)", "GPU/NPU", "gate", "BW(GB/s)", "GFLOPS", "BWutil(%)", "status");
    printf("+------------------------------------+-----------+--------------+-----------------+-----------------+"
           "-----------------+-----------------+-----------------+-----------------+-------+-----------------+"
           "-----------------+-----------------+----------------------------------+\n");
}

static void RunAllCases(aclrtStream stream, const std::vector<PerfCase>& cases,
                        const std::string& only, int warmup, int iters, double peakHbmGbps,
                        std::vector<RowOut>& rows, std::vector<PerfCase>& filtered_cases) {
    for (const auto& pc : cases) {
        if (!only.empty() && pc.label.find(only) == std::string::npos) {
            continue;
        }
        RowOut row;
        try {
            row = RunOneCase(stream, pc, warmup, iters, peakHbmGbps);
        } catch (const std::exception& error) {
            row.label = pc.label;
            row.dtype_name = pc.dtype_name;
            row.gpuBaselineUs = pc.gpu_baseline_us;
            row.note = "SETUP_FAIL:" + std::string(error.what());
        } catch (...) {
            row.label = pc.label;
            row.dtype_name = pc.dtype_name;
            row.gpuBaselineUs = pc.gpu_baseline_us;
            row.note = "SETUP_FAIL:unknown exception";
        }
        int64_t nnz = row.nnz;

        rows.push_back(row);
        filtered_cases.push_back(pc);

        std::string bwutilStr = (row.bwUtil < 0) ? "N/A" : Fmt(row.bwUtil);
        std::string baselineStr = !HasPerformanceBaseline(row) ? "N/A" : Fmt(row.gpuBaselineUs);
        std::string ratioStr = (row.gpuNpuRatio < 0) ? "N/A" : Fmt(row.gpuNpuRatio);
        const char* gate = !HasPerformanceBaseline(row) ? "N/A" :
            (PerformanceGatePassed(row) ? "PASS" : "FAIL");
        printf("| %-34s | %-9s | %12lld | %15s | %15s | %15s | %15s | %15s | %15s | %5s | %15s | %15s | %15s | %-32s |\n",
               row.label.c_str(), row.dtype_name.c_str(),
               static_cast<long long>(nnz),
               Fmt(row.tMin).c_str(), Fmt(row.tMed).c_str(), Fmt(row.tP90).c_str(),
               Fmt(row.tMean).c_str(), baselineStr.c_str(), ratioStr.c_str(), gate,
               Fmt(row.bwGbs).c_str(), Fmt(row.gflops).c_str(),
               bwutilStr.c_str(), row.note.c_str());
        fflush(stdout);
    }
    printf("+------------------------------------+-----------+--------------+-----------------+-----------------+"
           "-----------------+-----------------+-----------------+-----------------+-------+-----------------+"
           "-----------------+-----------------+----------------------------------+\n");
}

static void PrintDetailedBreakdown(const std::vector<PerfCase>& filtered_cases,
                                   const std::vector<RowOut>& rows) {
    // Detailed per-case breakdown
    std::cout << "\n=== Detailed breakdown (median time) ===\n";
    for (size_t i = 0; i < filtered_cases.size(); ++i) {
        const auto& pc = filtered_cases[i];
        const auto& row = rows[i];
        auto am = ComputeAlgoMetrics(pc, row.nnz);
        std::cout << "[" << row.label << "]\n";
        std::cout << "  format=" << (pc.is_bsr ? "bsr" : "csr")
                  << " m=" << pc.m << " k=" << pc.k << " n=" << pc.n
                  << " dtype=" << pc.dtype_name
                  << " alpha=" << pc.alpha << " beta=" << pc.beta
                  << " ratio=" << pc.sparsity_ratio << "\n";
        if (pc.is_bsr) {
            std::cout << "  block_dim=" << pc.block_dim
                      << " direction="
                      << (pc.block_order == ACL_SPARSE_ORDER_COL ? "col" : "row")
                      << " batch=" << pc.batch
                      << " row_pattern=" << pc.row_pattern
                      << " dense_layout=row/row"
                      << " seed=" << pc.seed << "\n";
        }
        std::cout << "  nnz=" << row.nnz
                  << " (density=" << std::fixed << std::setprecision(4)
                  << (pc.n > 0 ? static_cast<double>(row.nnz) / static_cast<double>(pc.m * pc.n) : 0.0)
                  << ")\n";
        std::cout << std::defaultfloat;
        std::cout << "  FLOPs=" << static_cast<uint64_t>(am.flops)
                  << "  HBM_bytes=" << static_cast<uint64_t>(am.bytes)
                  << "  workspaceSizeInBytes=" << row.workspaceSizeInBytes << "\n";
        std::cout << "  time: min=" << std::fixed << std::setprecision(9) << row.tMin
                  << "us  median=" << row.tMed << "us  p90=" << row.tP90
                  << "us  mean=" << row.tMean << "us\n";
        if (HasPerformanceBaseline(row)) {
            std::cout << std::fixed << std::setprecision(9)
                      << "  matched GPU median=" << row.gpuBaselineUs
                      << "us  GPU/NPU median ratio=" << row.gpuNpuRatio
                      << "  target>=0.3: "
                      << (PerformanceGatePassed(row) ? "PASS" : "FAIL") << "\n";
        } else {
            std::cout << "  matched GPU median=N/A  GPU/NPU median ratio=N/A  target>=0.3: N/A\n";
        }
        std::cout << "  achieved BW=" << row.bwGbs << " GB/s  GFLOPS=" << row.gflops;
        if (row.bwUtil >= 0) {
            std::cout << "  BWutil=" << row.bwUtil << "%";
        }
        std::cout << "  [" << row.note << "]\n\n";
        std::cout << std::defaultfloat;
    }
}

static void PrintJsonCaseMetrics(const PerfCase &pc, const RowOut &row)
{
    const auto am = ComputeAlgoMetrics(pc, row.nnz);
    std::cout << "{\"label\":\"" << JsonEscape(row.label) << "\""
          << ",\"format\":\"" << (pc.is_bsr ? "bsr" : "csr") << "\""
          << ",\"m\":" << pc.m << ",\"k\":" << pc.k << ",\"n\":" << pc.n
          << ",\"dtype\":\"" << JsonEscape(pc.dtype_name) << "\""
          << ",\"nnz\":" << row.nnz
          << ",\"base\":" << pc.index_base
          << ",\"nnz_per_row\":" << pc.nnz_per_row
          << ",\"seed\":" << pc.seed
          << ",\"alpha\":" << pc.alpha << ",\"beta\":" << pc.beta
          << ",\"ratio\":" << pc.sparsity_ratio
          << ",\"flops\":" << static_cast<uint64_t>(am.flops)
          << ",\"hbm_bytes\":" << static_cast<uint64_t>(am.bytes)
          << ",\"workspaceSizeInBytes\":" << row.workspaceSizeInBytes
          << ",\"t_min_us\":" << row.tMin
          << ",\"t_med_us\":" << row.tMed
          << ",\"t_p90_us\":" << row.tP90
          << ",\"t_mean_us\":" << row.tMean
          << ",\"gpu_baseline_median_us\":" << row.gpuBaselineUs
          << ",\"gpu_npu_median_ratio\":" << row.gpuNpuRatio
          << ",\"performance_target\":0.3"
          << ",\"performance_pass\":";
}

static void PrintJsonCaseContract(const PerfCase &pc, const RowOut &row)
{
    if (!HasPerformanceBaseline(row)) {
        std::cout << "null";
    } else {
        std::cout << (PerformanceGatePassed(row) ? "true" : "false");
    }
    std::cout << ",\"same_output_dtype\":"
          << (pc.OutputDtype() == pc.dtype ? "true" : "false")
          << ",\"setup_and_execute_ok\":"
          << (row.ok ? "true" : "false");
    std::cout << ",\"timing_scope\":\"aclsparseSDDMM_execute\""
          << ",\"gpu_baseline_source\":\""
          << (pc.gpu_baseline_us > 0.0 ? "native_cusparse" : "none") << "\""
          << ",\"input_generator\":\""
          << (pc.UsesNativeBaselineInputs() ? "native_cusparse_constants" : "cpp_mt19937_uniform")
          << "\""
          << ",\"block_dim\":" << pc.block_dim
          << ",\"direction\":\""
          << (pc.block_order == ACL_SPARSE_ORDER_COL ? "col" : "row") << "\""
          << ",\"batch\":" << pc.batch
          << ",\"row_pattern\":\"" << JsonEscape(pc.row_pattern) << "\""
          << ",\"x_layout\":\"row\""
          << ",\"y_layout\":\"" << (pc.y_col_major ? "col" : "row") << "\"";
    if (pc.UsesNativeBaselineInputs()) {
        std::cout << ",\"input_x_real\":0.125,\"input_y_real\":0.25"
              << ",\"input_values_real\":0,\"input_imag\":0";
    }
    std::cout << ",\"p90_definition\":\"nearest_rank\""
          << ",\"bw_gbs\":" << row.bwGbs
          << ",\"gflops\":" << row.gflops
          << ",\"bw_util_pct\":" << row.bwUtil
          << ",\"status\":\"" << JsonEscape(row.note) << "\"}";
}

static void PrintJsonSummary(const std::vector<PerfCase>& filtered_cases,
                             const std::vector<RowOut>& rows,
                             double peakHbmGbps, int warmup, int iters,
                             bool hbmMemInfoAvailable,
                             size_t totalMem, size_t freeMem) {
    // Machine-readable JSON summary
    const auto savedFlags = std::cout.flags();
    const auto savedPrecision = std::cout.precision();
    std::cout << std::defaultfloat
              << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << "=== JSON ===\n";
    std::cout << "{\"device\":\"Ascend950_dav3510\",\"peak_hbm_gbps\":" << peakHbmGbps
              << ",\"warmup\":" << warmup << ",\"iters\":" << iters
              << ",\"hbm_total_bytes\":";
    if (hbmMemInfoAvailable) {
        std::cout << totalMem;
    } else {
        std::cout << "null";
    }
    std::cout << ",\"hbm_free_bytes_at_start\":";
    if (hbmMemInfoAvailable) {
        std::cout << freeMem;
    } else {
        std::cout << "null";
    }
    std::cout << ",\"cases\":[";
    for (size_t i = 0; i < filtered_cases.size(); ++i) {
        const auto& pc = filtered_cases[i];
        const auto& row = rows[i];
        if (i) std::cout << ",";
        PrintJsonCaseMetrics(pc, row);
        PrintJsonCaseContract(pc, row);
    }
    std::cout << "]}\n";
    std::cout.flags(savedFlags);
    std::cout.precision(savedPrecision);
}

// Representative performance cases (aligned with task 4.1 selection).
static const std::vector<PerfCase> kPerfCases = {
    {"case1  128x64x128  FP32 S",   128, 64, 128,   0.5, 1.0, 0.0, ACL_FLOAT,   "FP32", -1.0, 1.0, 1},
    {"case6  64x32x64    FP16 S",   64,  32, 64,    0.5, 1.0, 0.0, ACL_FLOAT16, "FP16", -0.5, 0.5, 6},
    {"case4  1024x512x1024 FP16 M", 1024,512, 1024,  0.5, 0.5, 0.5, ACL_FLOAT16, "FP16", -1.0, 1.0, 4},
    {"case15 640x320x640  FP32 M",  640, 320,640,   0.5, 0.5, 0.5, ACL_FLOAT,   "FP32", -3.0, 3.0, 15},
    {"case5  2048x1024x2048 FP32 L",2048,1024,2048, 0.5, 1.0, 0.0, ACL_FLOAT,   "FP32", -1.0, 1.0, 5},
    {"case16 1280x640x1280 FP16 L", 1280,640, 1280,  0.5, 1.0, 0.0, ACL_FLOAT16, "FP16", -1.0, 1.0, 16},
    {"case20 2048x512x1024 FP32 NF",2048,512, 1024,  0.5, 1.0, 1.0, ACL_FLOAT,   "FP32", -1.0, 1.0, 20},
    // Task P-01/P-02/P-03 CSR anchors. FP16 matches native cuSPARSE
    // (column-major Y). The FP32 TSV contains a PyTorch task-call baseline,
    // which must be compared with all kernels in the same task invocation,
    // not this Execute-only timer. Its gate is measured separately.
    {"sddmm-P-01-csr-fp16-base0", 8192, 128, 28672, 64.0 / 28672.0, 1.0, 0.0,
        ACL_FLOAT16, "FP16", -1.0, 1.0, 20260912, 0, 64, true, 20.768},
    {"sddmm-P-01-csr-fp16-base1", 8192, 128, 28672, 64.0 / 28672.0, 1.0, 0.0,
        ACL_FLOAT16, "FP16", -1.0, 1.0, 20260912, 1, 64, true, 21.152},
    {"sddmm-P-01-csr-fp32-base0", 8192, 128, 28672, 64.0 / 28672.0, 1.0, 0.0,
        ACL_FLOAT, "FP32", -1.0, 1.0, 20260912, 0, 64, false, 0.0},
    {"sddmm-P-01-csr-fp32-base1", 8192, 128, 28672, 64.0 / 28672.0, 1.0, 0.0,
        ACL_FLOAT, "FP32", -1.0, 1.0, 20260912, 1, 64, false, 0.0},
    {"sddmm-P-02-csr-fp16-base0", 4096, 128, 1536, 64.0 / 1536.0, 1.0, 0.0,
        ACL_FLOAT16, "FP16", -1.0, 1.0, 20260912, 0, 64, true, 14.272},
    {"sddmm-P-02-csr-fp16-base1", 4096, 128, 1536, 64.0 / 1536.0, 1.0, 0.0,
        ACL_FLOAT16, "FP16", -1.0, 1.0, 20260912, 1, 64, true, 14.560},
    {"sddmm-P-02-csr-fp32-base0", 4096, 128, 1536, 64.0 / 1536.0, 1.0, 0.0,
        ACL_FLOAT, "FP32", -1.0, 1.0, 20260912, 0, 64, false, 0.0},
    {"sddmm-P-02-csr-fp32-base1", 4096, 128, 1536, 64.0 / 1536.0, 1.0, 0.0,
        ACL_FLOAT, "FP32", -1.0, 1.0, 20260912, 1, 64, false, 0.0},
    {"sddmm-P-03-csr-fp16-base0", 7168, 128, 2048, 64.0 / 2048.0, 1.0, 0.0,
        ACL_FLOAT16, "FP16", -1.0, 1.0, 20260912, 0, 64, true, 19.776},
    {"sddmm-P-03-csr-fp16-base1", 7168, 128, 2048, 64.0 / 2048.0, 1.0, 0.0,
        ACL_FLOAT16, "FP16", -1.0, 1.0, 20260912, 1, 64, true, 19.904},
    {"sddmm-P-03-csr-fp32-base0", 7168, 128, 2048, 64.0 / 2048.0, 1.0, 0.0,
        ACL_FLOAT, "FP32", -1.0, 1.0, 20260912, 0, 64, false, 0.0},
    {"sddmm-P-03-csr-fp32-base1", 7168, 128, 2048, 64.0 / 2048.0, 1.0, 0.0,
        ACL_FLOAT, "FP32", -1.0, 1.0, 20260912, 1, 64, false, 0.0},
    // CSR complex64 also has PyTorch task-call GPU baselines. Keep
    // Execute-only results unpaired; measure all task kernels separately.
    {"sddmm-P-01-csr-complex64-base0", 8192, 128, 28672,
        64.0 / 28672.0, 1.0, 0.0,
        ACL_COMPLEX64, "C64", -1.0, 1.0, 20260912, 0, 64, false, 0.0},
    {"sddmm-P-01-csr-complex64-base1", 8192, 128, 28672,
        64.0 / 28672.0, 1.0, 0.0,
        ACL_COMPLEX64, "C64", -1.0, 1.0, 20260912, 1, 64, false, 0.0},
    {"sddmm-P-02-csr-complex64-base0", 4096, 128, 1536,
        64.0 / 1536.0, 1.0, 0.0,
        ACL_COMPLEX64, "C64", -1.0, 1.0, 20260912, 0, 64, false, 0.0},
    {"sddmm-P-02-csr-complex64-base1", 4096, 128, 1536,
        64.0 / 1536.0, 1.0, 0.0,
        ACL_COMPLEX64, "C64", -1.0, 1.0, 20260912, 1, 64, false, 0.0},
    {"sddmm-P-03-csr-complex64-base0", 7168, 128, 2048,
        64.0 / 2048.0, 1.0, 0.0,
        ACL_COMPLEX64, "C64", -1.0, 1.0, 20260912, 0, 64, false, 0.0},
    {"sddmm-P-03-csr-complex64-base1", 7168, 128, 2048,
        64.0 / 2048.0, 1.0, 0.0,
        ACL_COMPLEX64, "C64", -1.0, 1.0, 20260912, 1, 64, false, 0.0},
    // Exact BSR anchors from performance_cases.json. All use the default
    // uniform block-row pattern, ROW block storage, batch=1, K=128,
    // alpha=1, beta=0, seed=20260912, and row-major dense X/Y. GPU medians
    // are the matching cusparseSDDMM rows in gpu_full_results.tsv.
    {"sddmm-P-01-bsr-float16-base0", 8192, 128, 28672,
        524288.0 / (8192.0 * 28672.0), 1.0, 0.0,
        ACL_FLOAT16, "FP16", -1.0, 1.0, 20260912, 0, 0, false, 24.736,
        true, 16, 524288, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-01-bsr-float16-base1", 8192, 128, 28672,
        524288.0 / (8192.0 * 28672.0), 1.0, 0.0,
        ACL_FLOAT16, "FP16", -1.0, 1.0, 20260912, 1, 0, false, 24.768,
        true, 16, 524288, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-01-bsr-bfloat16-base0", 8192, 128, 28672,
        524288.0 / (8192.0 * 28672.0), 1.0, 0.0,
        ACL_BF16, "BF16", -1.0, 1.0, 20260912, 0, 0, false, 30.592,
        true, 16, 524288, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-01-bsr-bfloat16-base1", 8192, 128, 28672,
        524288.0 / (8192.0 * 28672.0), 1.0, 0.0,
        ACL_BF16, "BF16", -1.0, 1.0, 20260912, 1, 0, false, 30.976,
        true, 16, 524288, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-01-bsr-float32-base0", 8192, 128, 28672,
        524288.0 / (8192.0 * 28672.0), 1.0, 0.0,
        ACL_FLOAT, "FP32", -1.0, 1.0, 20260912, 0, 0, false, 25.344,
        true, 16, 524288, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-01-bsr-float32-base1", 8192, 128, 28672,
        524288.0 / (8192.0 * 28672.0), 1.0, 0.0,
        ACL_FLOAT, "FP32", -1.0, 1.0, 20260912, 1, 0, false, 25.376,
        true, 16, 524288, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-02-bsr-float16-base0", 4096, 128, 1536,
        262144.0 / (4096.0 * 1536.0), 1.0, 0.0,
        ACL_FLOAT16, "FP16", -1.0, 1.0, 20260912, 0, 0, false, 16.672,
        true, 32, 262144, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-02-bsr-float16-base1", 4096, 128, 1536,
        262144.0 / (4096.0 * 1536.0), 1.0, 0.0,
        ACL_FLOAT16, "FP16", -1.0, 1.0, 20260912, 1, 0, false, 16.800,
        true, 32, 262144, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-02-bsr-bfloat16-base0", 4096, 128, 1536,
        262144.0 / (4096.0 * 1536.0), 1.0, 0.0,
        ACL_BF16, "BF16", -1.0, 1.0, 20260912, 0, 0, false, 20.128,
        true, 32, 262144, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-02-bsr-bfloat16-base1", 4096, 128, 1536,
        262144.0 / (4096.0 * 1536.0), 1.0, 0.0,
        ACL_BF16, "BF16", -1.0, 1.0, 20260912, 1, 0, false, 19.968,
        true, 32, 262144, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-02-bsr-float32-base0", 4096, 128, 1536,
        262144.0 / (4096.0 * 1536.0), 1.0, 0.0,
        ACL_FLOAT, "FP32", -1.0, 1.0, 20260912, 0, 0, false, 17.280,
        true, 32, 262144, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-02-bsr-float32-base1", 4096, 128, 1536,
        262144.0 / (4096.0 * 1536.0), 1.0, 0.0,
        ACL_FLOAT, "FP32", -1.0, 1.0, 20260912, 1, 0, false, 17.184,
        true, 32, 262144, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-03-bsr-float16-base0", 7168, 128, 2048,
        458752.0 / (7168.0 * 2048.0), 1.0, 0.0,
        ACL_FLOAT16, "FP16", -1.0, 1.0, 20260912, 0, 0, false, 22.336,
        true, 64, 458752, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-03-bsr-float16-base1", 7168, 128, 2048,
        458752.0 / (7168.0 * 2048.0), 1.0, 0.0,
        ACL_FLOAT16, "FP16", -1.0, 1.0, 20260912, 1, 0, false, 22.720,
        true, 64, 458752, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-03-bsr-bfloat16-base0", 7168, 128, 2048,
        458752.0 / (7168.0 * 2048.0), 1.0, 0.0,
        ACL_BF16, "BF16", -1.0, 1.0, 20260912, 0, 0, false, 27.936,
        true, 64, 458752, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-03-bsr-bfloat16-base1", 7168, 128, 2048,
        458752.0 / (7168.0 * 2048.0), 1.0, 0.0,
        ACL_BF16, "BF16", -1.0, 1.0, 20260912, 1, 0, false, 27.936,
        true, 64, 458752, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-03-bsr-float32-base0", 7168, 128, 2048,
        458752.0 / (7168.0 * 2048.0), 1.0, 0.0,
        ACL_FLOAT, "FP32", -1.0, 1.0, 20260912, 0, 0, false, 23.488,
        true, 64, 458752, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-03-bsr-float32-base1", 7168, 128, 2048,
        458752.0 / (7168.0 * 2048.0), 1.0, 0.0,
        ACL_FLOAT, "FP32", -1.0, 1.0, 20260912, 1, 0, false, 23.296,
        true, 64, 458752, ACL_SPARSE_ORDER_ROW, "uniform"},
    // Complex64 BSR anchors use native cuSPARSE medians from the task TSV.
    {"sddmm-P-01-bsr-complex64-base0", 8192, 128, 28672,
        524288.0 / (8192.0 * 28672.0), 1.0, 0.0,
        ACL_COMPLEX64, "C64", -1.0, 1.0, 20260912, 0, 0, false, 39.840,
        true, 16, 524288, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-01-bsr-complex64-base1", 8192, 128, 28672,
        524288.0 / (8192.0 * 28672.0), 1.0, 0.0,
        ACL_COMPLEX64, "C64", -1.0, 1.0, 20260912, 1, 0, false, 40.000,
        true, 16, 524288, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-02-bsr-complex64-base0", 4096, 128, 1536,
        262144.0 / (4096.0 * 1536.0), 1.0, 0.0,
        ACL_COMPLEX64, "C64", -1.0, 1.0, 20260912, 0, 0, false, 24.256,
        true, 32, 262144, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-02-bsr-complex64-base1", 4096, 128, 1536,
        262144.0 / (4096.0 * 1536.0), 1.0, 0.0,
        ACL_COMPLEX64, "C64", -1.0, 1.0, 20260912, 1, 0, false, 24.128,
        true, 32, 262144, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-03-bsr-complex64-base0", 7168, 128, 2048,
        458752.0 / (7168.0 * 2048.0), 1.0, 0.0,
        ACL_COMPLEX64, "C64", -1.0, 1.0, 20260912, 0, 0, false, 34.784,
        true, 64, 458752, ACL_SPARSE_ORDER_ROW, "uniform"},
    {"sddmm-P-03-bsr-complex64-base1", 7168, 128, 2048,
        458752.0 / (7168.0 * 2048.0), 1.0, 0.0,
        ACL_COMPLEX64, "C64", -1.0, 1.0, 20260912, 1, 0, false, 34.944,
        true, 64, 458752, ACL_SPARSE_ORDER_ROW, "uniform"},
    // The task type matrix also declares mixed real inputs with FP32
    // sparse output. The supplied GPU TSV has no matching mixed-dtype IDs,
    // so these rows collect NPU medians without manufacturing a gate.
    {"sddmm-P-01-csr-fp16-to-fp32-base0", 8192, 128, 28672,
        64.0 / 28672.0, 1.0, 0.0,
        ACL_FLOAT16, "F16>F32", -1.0, 1.0, 20260912, 0, 64, false, 0.0,
        false, 1, 0, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-01-csr-fp16-to-fp32-base1", 8192, 128, 28672,
        64.0 / 28672.0, 1.0, 0.0,
        ACL_FLOAT16, "F16>F32", -1.0, 1.0, 20260912, 1, 64, false, 0.0,
        false, 1, 0, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-02-csr-fp16-to-fp32-base0", 4096, 128, 1536,
        64.0 / 1536.0, 1.0, 0.0,
        ACL_FLOAT16, "F16>F32", -1.0, 1.0, 20260912, 0, 64, false, 0.0,
        false, 1, 0, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-02-csr-fp16-to-fp32-base1", 4096, 128, 1536,
        64.0 / 1536.0, 1.0, 0.0,
        ACL_FLOAT16, "F16>F32", -1.0, 1.0, 20260912, 1, 64, false, 0.0,
        false, 1, 0, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-03-csr-fp16-to-fp32-base0", 7168, 128, 2048,
        64.0 / 2048.0, 1.0, 0.0,
        ACL_FLOAT16, "F16>F32", -1.0, 1.0, 20260912, 0, 64, false, 0.0,
        false, 1, 0, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-03-csr-fp16-to-fp32-base1", 7168, 128, 2048,
        64.0 / 2048.0, 1.0, 0.0,
        ACL_FLOAT16, "F16>F32", -1.0, 1.0, 20260912, 1, 64, false, 0.0,
        false, 1, 0, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-01-bsr-fp16-to-fp32-base0", 8192, 128, 28672,
        524288.0 / (8192.0 * 28672.0), 1.0, 0.0,
        ACL_FLOAT16, "F16>F32", -1.0, 1.0, 20260912, 0, 0, false, 0.0,
        true, 16, 524288, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-01-bsr-fp16-to-fp32-base1", 8192, 128, 28672,
        524288.0 / (8192.0 * 28672.0), 1.0, 0.0,
        ACL_FLOAT16, "F16>F32", -1.0, 1.0, 20260912, 1, 0, false, 0.0,
        true, 16, 524288, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-02-bsr-fp16-to-fp32-base0", 4096, 128, 1536,
        262144.0 / (4096.0 * 1536.0), 1.0, 0.0,
        ACL_FLOAT16, "F16>F32", -1.0, 1.0, 20260912, 0, 0, false, 0.0,
        true, 32, 262144, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-02-bsr-fp16-to-fp32-base1", 4096, 128, 1536,
        262144.0 / (4096.0 * 1536.0), 1.0, 0.0,
        ACL_FLOAT16, "F16>F32", -1.0, 1.0, 20260912, 1, 0, false, 0.0,
        true, 32, 262144, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-03-bsr-fp16-to-fp32-base0", 7168, 128, 2048,
        458752.0 / (7168.0 * 2048.0), 1.0, 0.0,
        ACL_FLOAT16, "F16>F32", -1.0, 1.0, 20260912, 0, 0, false, 0.0,
        true, 64, 458752, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-03-bsr-fp16-to-fp32-base1", 7168, 128, 2048,
        458752.0 / (7168.0 * 2048.0), 1.0, 0.0,
        ACL_FLOAT16, "F16>F32", -1.0, 1.0, 20260912, 1, 0, false, 0.0,
        true, 64, 458752, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-01-bsr-bf16-to-fp32-base0", 8192, 128, 28672,
        524288.0 / (8192.0 * 28672.0), 1.0, 0.0,
        ACL_BF16, "BF16>F32", -1.0, 1.0, 20260912, 0, 0, false, 0.0,
        true, 16, 524288, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-01-bsr-bf16-to-fp32-base1", 8192, 128, 28672,
        524288.0 / (8192.0 * 28672.0), 1.0, 0.0,
        ACL_BF16, "BF16>F32", -1.0, 1.0, 20260912, 1, 0, false, 0.0,
        true, 16, 524288, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-02-bsr-bf16-to-fp32-base0", 4096, 128, 1536,
        262144.0 / (4096.0 * 1536.0), 1.0, 0.0,
        ACL_BF16, "BF16>F32", -1.0, 1.0, 20260912, 0, 0, false, 0.0,
        true, 32, 262144, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-02-bsr-bf16-to-fp32-base1", 4096, 128, 1536,
        262144.0 / (4096.0 * 1536.0), 1.0, 0.0,
        ACL_BF16, "BF16>F32", -1.0, 1.0, 20260912, 1, 0, false, 0.0,
        true, 32, 262144, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-03-bsr-bf16-to-fp32-base0", 7168, 128, 2048,
        458752.0 / (7168.0 * 2048.0), 1.0, 0.0,
        ACL_BF16, "BF16>F32", -1.0, 1.0, 20260912, 0, 0, false, 0.0,
        true, 64, 458752, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
    {"sddmm-P-03-bsr-bf16-to-fp32-base1", 7168, 128, 2048,
        458752.0 / (7168.0 * 2048.0), 1.0, 0.0,
        ACL_BF16, "BF16>F32", -1.0, 1.0, 20260912, 1, 0, false, 0.0,
        true, 64, 458752, ACL_SPARSE_ORDER_ROW, "uniform", 1, ACL_FLOAT},
};

int main(int argc, char** argv) {
    PerfArgs args = ParsePerfArgs(argc, argv);
    if (args.warmup < 0 || args.iters <= 0) {
        std::cerr << "--warmup must be non-negative and --iters must be positive" << std::endl;
        return 2;
    }
    int warmup = args.warmup;
    int iters = args.iters;
    double peakHbmGbps = args.peakHbmGbps;
    const std::string& only = args.only;
    bool bwbench = args.bwbench;


    aclrtStream stream = nullptr;
    if (!InitAclEnv(stream)) {
        return 1;
    }

    if (bwbench) {
        int rc = RunBwBench(stream, warmup, iters);
        CleanupAclEnv(stream);
        return rc;
    }

    int64_t arch = 0;
    aclrtGetDeviceInfo(0, ACL_DEV_ATTR_NPU_ARCH, &arch);
    size_t totalMem = 0, freeMem = 0;
    const bool hbmMemInfoAvailable =
        aclrtGetMemInfo(ACL_HBM_MEM, &freeMem, &totalMem) == ACL_SUCCESS;

    PrintHeader(arch, totalMem, freeMem, hbmMemInfoAvailable,
                warmup, iters, peakHbmGbps);

    std::vector<RowOut> rows;
    std::vector<PerfCase> filtered_cases;
    RunAllCases(stream, kPerfCases, only, warmup, iters, peakHbmGbps, rows, filtered_cases);

    PrintDetailedBreakdown(filtered_cases, rows);
    PrintJsonSummary(filtered_cases, rows, peakHbmGbps, warmup, iters,
                     hbmMemInfoAvailable, totalMem, freeMem);

    bool failed = false;
    for (const auto& row : rows) {
        if (!row.ok ||
            (HasPerformanceBaseline(row) && !PerformanceGatePassed(row))) {
            failed = true;
        }
    }

    CleanupAclEnv(stream);
    return failed ? 1 : 0;
}
