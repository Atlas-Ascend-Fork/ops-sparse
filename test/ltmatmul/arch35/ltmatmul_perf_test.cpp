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
 * @file ltmatmul_perf_test.cpp
 * @brief Performance baseline benchmark for aclsparseLtMatmul (v3: full epilogue + batch + bottleneck).
 *
 * Reuses the existing npu_wrapper (RunMatmulNpu) which records kernel execution time via
 * EventGuard (aclrtEvent). This benchmark:
 *   - Runs each config with WARMUP iterations (excluded from stats)
 *   - Then multi-round MEASURE iterations, reporting mean / min / max / median (ms) + CV
 *   - Passes epilogue (bias/activation/3-vector) and batch params to the NPU chain
 *   - Outputs bottleneck breakdown (compute/H2D/D2H/scalar proportions) + achieved GFLOPS/GB/s
 *
 * Coverage:
 *   §1 Shape×Dtype: 7 shapes × 4 dtypes (FP32/FP16/BF16/INT8_I32), sparse×dense, no epilogue
 *   §2 Epilogue overhead: bias+ReLU / bias+GeLU / 3vec+bias+ReLU vs baseline (FP32/FP16/INT8)
 *   §3 Batch: batch=1/3/4/5 (FP32/FP16), shape 37×151×37
 *   §4 Extreme: m=15744 with 3-vector + bias + ReLU
 *
 * Demand source: no explicit perf requirement; baseline data collection only.
 * No pass/fail threshold; this is reference data for future optimization.
 *
 * Build: standalone executable (own main), linked against the same libs as matmul_test.
 */

#include "test_common.h"

#include "ltmatmul_golden.h"
#include "ltmatmul_npu_wrapper.h"
#include "ltmatmul_param.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace sparse_test;

namespace {

// ============================================================================
// Iteration counts: warmup excluded, then measured.
// Multi-round collection for median + CV (dispersion / fluctuation detection).
// Adaptive: large shapes use fewer iterations to keep total runtime reasonable.
// ============================================================================
constexpr int kWarmupIters = 5;
constexpr int kMeasureIters = 20;
constexpr int kMultiRoundDefault = 5;  // 5 rounds × 20 iters = 100 samples per case

// Adaptive iteration counts for large shapes (m*k*n > threshold).
// Reduces total iterations to keep wall-clock time manageable while preserving
// multi-round median + CV dispersion detection.
struct IterConfig {
    int warmup;
    int measure;
    int rounds;
};

inline IterConfig GetIterConfig(int32_t m, int32_t k, int32_t n) {
    int64_t problemSize = static_cast<int64_t>(m) * k * n;
    if (problemSize > 100000000) {       // > 100M elements (e.g. 15744×160×1792)
        return {2, 5, 3};                 // 2+5×3 = 17 total
    }
    if (problemSize > 10000000) {        // > 10M elements
        return {3, 10, 3};               // 3+10×3 = 33 total
    }
    if (problemSize > 1000000) {         // > 1M elements
        return {3, 10, 5};               // 3+10×5 = 53 total
    }
    return {kWarmupIters, kMeasureIters, kMultiRoundDefault};  // 5+20×5 = 105
}

// ============================================================================
// Stats helpers: mean / min / max / median / stddev / CV.
// ============================================================================
struct Stats {
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    double median = 0.0;
    double stddev = 0.0;   // standard deviation (dispersion)
    double cv = 0.0;       // coefficient of variation = stddev/mean
};

inline Stats ComputeStats(std::vector<double>& samples) {
    Stats s{};
    if (samples.empty()) { return s; }
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    s.min = samples.front();
    s.max = samples.back();
    double sum = 0.0;
    for (double v : samples) { sum += v; }
    s.mean = sum / static_cast<double>(n);
    s.median = (n % 2 == 0)
        ? (samples[n / 2 - 1] + samples[n / 2]) / 2.0
        : samples[n / 2];
    if (n > 1) {
        double sqSum = 0.0;
        for (double v : samples) { sqSum += (v - s.mean) * (v - s.mean); }
        s.stddev = std::sqrt(sqSum / static_cast<double>(n - 1));
        s.cv = (std::fabs(s.mean) > 1e-12) ? s.stddev / std::fabs(s.mean) : 0.0;
    }
    return s;
}

// ============================================================================
// Bottleneck breakdown: estimate compute / H2D / D2H / scalar proportions.
//
// Shape-based theoretical estimate (FLOPs and bytes vs measured kernel time),
// providing first-order bottleneck dimension decomposition. Combined with
// achieved GFLOPS / GB/s, this locks down the measured bottleneck dimension.
//
//   compute_flops = 2 * m * k * n * num_batches * (sparse? 0.5 : 1.0)
//   h2d_bytes     = (m*k + k*n + m*n) * dtype_bytes * num_batches
//   d2h_bytes     = m*n * out_dtype_bytes * num_batches
//   scalar_bytes  = m * sizeof(float) * num_batches * (alpha_vec + beta_vec + bias)
// ============================================================================
struct BottleneckBreakdown {
    double computeFlops = 0.0;
    double h2dBytes = 0.0;
    double d2hBytes = 0.0;
    double scalarBytes = 0.0;
    double computePct = 0.0;
    double h2dPct = 0.0;
    double d2hPct = 0.0;
    double scalarPct = 0.0;
    double achievedGFlops = 0.0;  // based on median kernel time
    double achievedBwGbs = 0.0;   // based on median kernel time
};

inline BottleneckBreakdown EstimateBottleneck(int32_t m, int32_t k, int32_t n,
    int dtypeBytes, int outDtypeBytes, bool isSparse,
    bool alphaVec, bool betaVec, bool hasBias,
    int32_t numBatches, double medianMs)
{
    BottleneckBreakdown b;
    const int32_t nb = (numBatches > 1) ? numBatches : 1;
    const double sparsityFactor = isSparse ? 0.5 : 1.0;
    b.computeFlops = 2.0 * static_cast<double>(m) * k * n * nb * sparsityFactor;
    b.h2dBytes = (static_cast<double>(m) * k + static_cast<double>(k) * n
                  + static_cast<double>(m) * n) * dtypeBytes * nb;
    b.d2hBytes = static_cast<double>(m) * n * outDtypeBytes * nb;
    int scalarCount = (alphaVec ? 1 : 0) + (betaVec ? 1 : 0) + (hasBias ? 1 : 0);
    b.scalarBytes = static_cast<double>(m) * sizeof(float) * nb * scalarCount;

    // Weighted proportions: compute ~GFLOPs, H2D/D2H ~GB, scalar ~MB (tiny).
    const double computeWeight = b.computeFlops / 1e9;
    const double h2dWeight = b.h2dBytes / 1e9;
    const double d2hWeight = b.d2hBytes / 1e9;
    const double scalarWeight = b.scalarBytes / 1e6;
    const double total = computeWeight + h2dWeight + d2hWeight + scalarWeight + 1e-12;
    b.computePct = computeWeight / total;
    b.h2dPct = h2dWeight / total;
    b.d2hPct = d2hWeight / total;
    b.scalarPct = scalarWeight / total;

    // Achieved throughput based on median kernel time.
    if (medianMs > 1e-9) {
        b.achievedGFlops = b.computeFlops / (medianMs * 1e6);  // ms→s, FLOP→GFLOP
        b.achievedBwGbs = (b.h2dBytes + b.d2hBytes) / (medianMs * 1e6);  // bytes→GB
    }
    return b;
}

// ============================================================================
// PerfConfig: descriptor for a single performance configuration.
// ============================================================================
struct PerfConfig {
    std::string tag;
    std::string dtype;     // FP32 / FP16 / BF16 / INT8_I32
    std::string path;      // sparse×dense / dense×dense
    int32_t m;
    int32_t k;
    int32_t n;
    int32_t splitK = 1;
    int32_t splitKMode = 0;  // 0=fused(ONE_KERNEL), 1=TWO_KERNELS
    int dtype_key = 0;         // 0=FP32 1=FP16 2=BF16 3=INT8_I32
    bool dense = false;
    // Epilogue
    bool biasEnabled = false;
    bool alpha_vec = false;
    bool beta_vec = false;
    int32_t activationType = 0;  // 0=none, 1=ReLU, 2=GeLU
    // Batch
    int32_t numBatches = 1;
    int64_t batchStride = 0;     // computed at runtime if >1 batch
    // Section label for grouping in report
    std::string section;
};

// ============================================================================
// Epilogue tag string for output.
// ============================================================================
inline std::string EpilogueTag(const PerfConfig& c) {
    if (!c.biasEnabled && c.activationType == 0 && !c.alpha_vec && !c.beta_vec) {
        return "none";
    }
    std::string s;
    if (c.alpha_vec && c.beta_vec) { s += "3vec+"; }
    if (c.biasEnabled) { s += "bias+"; }
    if (c.activationType == 1) { s += "ReLU"; }
    else if (c.activationType == 2) { s += "GeLU"; }
    if (s.empty()) { s = "none"; }
    return s;
}

// ============================================================================
// Output formatting: header + per-case result line.
// ============================================================================
inline void PrintHeader() {
    std::printf("# aclsparseLtMatmul performance baseline (v3: epilogue + batch + bottleneck)\n");
    std::printf("# adaptive iterations: small=5+20×5, medium=3+10×5, large=2+5×3 (warmup+measure×rounds)\n");
    std::printf("# columns: section|tag|dtype|path|m|k|n|batch|epilogue"
                "|median_ms|min_ms|max_ms|mean_ms|stddev_ms|cv"
                "|gflops|bw_gbs|compute_pct|h2d_pct|d2h_pct|scalar_pct"
                "|warmup|measure|rounds\n");
}

// PrintResultExt: outputs per-case result line with adaptive iteration config
// for methodology transparency.
inline void PrintResultExt(const PerfConfig& c, const Stats& s, const BottleneckBreakdown& b,
                           const IterConfig& ic) {
    std::printf("PERF|%s|%s|%s|%s|%d|%d|%d|%d|%s"
                "|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f"
                "|%.2f|%.2f|%.2f|%.2f|%.2f|%.2f"
                "|%d|%d|%d\n",
                c.section.c_str(), c.tag.c_str(), c.dtype.c_str(), c.path.c_str(),
                c.m, c.k, c.n, c.numBatches, EpilogueTag(c).c_str(),
                s.median, s.min, s.max, s.mean, s.stddev, s.cv,
                b.achievedGFlops, b.achievedBwGbs,
                b.computePct * 100.0, b.h2dPct * 100.0, b.d2hPct * 100.0, b.scalarPct * 100.0,
                ic.warmup, ic.measure, ic.rounds);
    std::fflush(stdout);
}

// ============================================================================
// GenBatchStridedMatrix: generate batch-strided input data.
// Each batch's matrix occupies [b*stride, b*stride+matSize) with zero padding.
// (Mirrors GenBatchStridedMatrix in ltmatmul_test.cpp.)
// ============================================================================
template <typename T>
inline std::vector<T> GenBatchStridedMatrix(int32_t rows, int32_t cols,
    float lo, float hi, uint32_t baseSeed,
    int32_t numBatches, int64_t batchStride)
{
    if (numBatches <= 1) {
        return MatmulGenTestMatrix<T>(rows, cols, lo, hi, baseSeed);
    }
    const size_t matSize = static_cast<size_t>(rows) * cols;
    const size_t totalElems = static_cast<size_t>(numBatches) * static_cast<size_t>(batchStride);
    std::vector<T> data(totalElems, T{0});
    for (int32_t b = 0; b < numBatches; ++b) {
        auto batchData = MatmulGenTestMatrix<T>(rows, cols, lo, hi,
            baseSeed + static_cast<uint32_t>(b) * 100);
        size_t off = static_cast<size_t>(b) * static_cast<size_t>(batchStride);
        std::copy(batchData.begin(), batchData.end(), data.begin() + off);
    }
    return data;
}

// ============================================================================
// TimingVecData + GenTimingVecData: generate vector scaling
// + bias data for a perf run. Extracted from RunOnceTiming to reduce NBNC.
// ============================================================================
struct TimingVecData {
    std::vector<float> alphaVec, betaVec, biasVec;
    const std::vector<float>* alphaPtr = nullptr;
    const std::vector<float>* betaPtr = nullptr;
    const std::vector<float>* biasPtr = nullptr;
};

inline void GenTimingVecData(const PerfConfig& c, uint32_t seed, TimingVecData& d)
{
    if (c.alpha_vec) {
        d.alphaVec = GenScalingVector(c.m, seed + 10);
        d.alphaPtr = &d.alphaVec;
    }
    if (c.beta_vec) {
        d.betaVec = GenScalingVector(c.m, seed + 20);
        d.betaPtr = &d.betaVec;
    }
    if (c.biasEnabled) {
        d.biasVec = GenBiasVector(c.m, seed + 30, -2.0f, 2.0f);
        d.biasPtr = &d.biasVec;
    }
}

// ============================================================================
// RunOnceTiming: single run, returns npuMs (kernel time only) or -1 on failure.
// ============================================================================
template <typename InT, typename OutT>
inline double RunOnceTiming(const PerfConfig& c, aclrtStream /*stream*/)
{
    const uint32_t seed = 7777u;
    const float alpha = 1.0f;
    const float beta = (c.biasEnabled || c.activationType) ? 0.5f : 0.0f;
    const auto order = ACL_SPARSE_ORDER_ROW;
    const auto opA = ACL_SPARSE_OP_NON_TRANSPOSE;
    const auto opB = (c.dtype_key == 3) ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
    const auto pruneAlg = ACLSPARSELT_PRUNE_SPMMA_STRIP;
    const bool isSparseA = !c.dense;
    const float lo = (c.dtype_key == 3) ? -5.0f : -1.0f;
    const float hi = (c.dtype_key == 3) ? 5.0f : 1.0f;

    auto hA = GenBatchStridedMatrix<InT>(c.m, c.k, lo, hi, seed, c.numBatches, c.batchStride);
    auto hB = GenBatchStridedMatrix<InT>(c.k, c.n, lo, hi, seed + 1, c.numBatches, c.batchStride);
    auto hC = GenBatchStridedMatrix<InT>(c.m, c.n, lo, hi, seed + 2, c.numBatches, c.batchStride);
    std::vector<OutT> hD;

    TimingVecData vd;
    GenTimingVecData(c, seed, vd);

    auto npu = RunMatmulNpu<InT, OutT>(c.m, c.k, c.n, hA, hB, hC, hD,
                                        alpha, beta, 0, c.splitK, c.splitKMode,
                                        isSparseA, c.dense, order, opA, opB, pruneAlg,
                                        c.alpha_vec ? 1 : 0, c.beta_vec ? 1 : 0,
                                        vd.alphaPtr, vd.betaPtr,
                                        c.biasEnabled ? 1 : 0, 0,
                                        c.activationType,
                                        FLT_MAX, 0.0f, 1.0f,
                                        c.numBatches, c.batchStride, vd.biasPtr);
    if (npu.matmulRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "[PERF] matmul failed ret=%d tag=%s (m=%d k=%d n=%d batch=%d)\n",
                     static_cast<int>(npu.matmulRet), c.tag.c_str(),
                     c.m, c.k, c.n, c.numBatches);
        return -1.0;
    }
    return npu.npuMs;
}

// ============================================================================
// Dispatch by dtype_key: single run, returns npuMs or -1.0 on failure.
// ============================================================================
inline double RunOnceByDtypeKey(const PerfConfig& c, aclrtStream stream)
{
    switch (c.dtype_key) {
        case 0:
            return RunOnceTiming<float, float>(c, stream);
        case 1:
            return RunOnceTiming<uint16_t, uint16_t>(c, stream);
        case 2:
            return RunOnceTiming<bf16_bits_t, bf16_bits_t>(c, stream);
        case 3:
            return RunOnceTiming<int8_t, int32_t>(c, stream);
        default:
            std::fprintf(stderr, "[PERF] unknown dtype_key=%d\n", c.dtype_key);
            return -1.0;
    }
}

// ============================================================================
// Run a config: warmup + multi-round measure (median + dispersion).
// Each round runs kMeasureIters; final stats aggregate all rounds' medians
// for stable median + CV across rounds (fluctuation detection).
// ============================================================================
inline bool RunConfig(const PerfConfig& c, aclrtStream stream) {
    // Compute batch_stride if not set.
    PerfConfig cRun = c;
    if (cRun.numBatches > 1 && cRun.batchStride == 0) {
        int64_t mk = static_cast<int64_t>(c.m) * c.k;
        int64_t kn = static_cast<int64_t>(c.k) * c.n;
        int64_t mn = static_cast<int64_t>(c.m) * c.n;
        cRun.batchStride = std::max({mk, kn, mn});
    }

    // Adaptive iteration config based on problem size.
    IterConfig ic = GetIterConfig(c.m, c.k, c.n);

    std::vector<double> allSamples;
    allSamples.reserve(static_cast<size_t>(ic.rounds) * ic.measure);
    std::vector<double> roundMedians;
    roundMedians.reserve(static_cast<size_t>(ic.rounds));

    // Warmup (excluded from stats).
    for (int i = 0; i < ic.warmup; ++i) {
        double ms = RunOnceByDtypeKey(cRun, stream);
        if (ms < 0) { return false; }
    }

    // Multi-round measure.
    for (int r = 0; r < ic.rounds; ++r) {
        std::vector<double> roundSamples;
        roundSamples.reserve(ic.measure);
        for (int i = 0; i < ic.measure; ++i) {
            double ms = RunOnceByDtypeKey(cRun, stream);
            if (ms < 0) { return false; }
            roundSamples.push_back(ms);
            allSamples.push_back(ms);
        }
        Stats rs = ComputeStats(roundSamples);
        roundMedians.push_back(rs.median);
    }

    // Aggregate: mean/min/max from allSamples; median/stddev/CV from round medians.
    Stats s = ComputeStats(allSamples);
    Stats roundStats = ComputeStats(roundMedians);
    s.median = roundStats.median;
    s.stddev = roundStats.stddev;
    s.cv = roundStats.cv;

    // Bottleneck breakdown with achieved throughput.
    int dtypeBytes = (c.dtype_key == 0) ? 4 : (c.dtype_key <= 2 ? 2 : 1);
    int outDtypeBytes = (c.dtype_key == 3) ? 4 : dtypeBytes;  // INT8_I32 → 4 bytes output
    auto b = EstimateBottleneck(c.m, c.k, c.n, dtypeBytes, outDtypeBytes,
                                 !c.dense, c.alpha_vec, c.beta_vec, c.biasEnabled,
                                 c.numBatches, s.median);
    // Pass adaptive iter config to PrintResultExt for methodology transparency.
    PrintResultExt(c, s, b, ic);
    return true;
}

// ============================================================================
// Config builders: MakePerfConfig variants + section adders.
// Extracted from BuildConfigList to reduce NBNC.
// ============================================================================
inline PerfConfig MakeBaseConfig(std::string tag, std::string dtype, int dk,
    int32_t m, int32_t k, int32_t n, std::string section)
{
    PerfConfig c;
    c.tag = std::move(tag); c.dtype = std::move(dtype); c.path = "sparse×dense";
    c.m = m; c.k = k; c.n = n; c.dtype_key = dk; c.dense = false;
    c.splitK = 1; c.splitKMode = 0; c.section = std::move(section);
    return c;
}

inline PerfConfig MakeEpiConfig(std::string tag, std::string dtype, int dk,
    int32_t m, int32_t k, int32_t n,
    bool bias, bool aVec, bool bVec, int32_t act, std::string section)
{
    PerfConfig c = MakeBaseConfig(std::move(tag), std::move(dtype), dk, m, k, n, std::move(section));
    c.biasEnabled = bias; c.alpha_vec = aVec; c.beta_vec = bVec;
    c.activationType = act;
    return c;
}

inline void AddShapeDtypeConfigs(std::vector<PerfConfig>& v) {
    struct ShapeDef { int32_t m, k, n; const char* suffix; };
    ShapeDef shapes[] = {
        {64,    64,   64,    "64x64x64"},
        {64,    160,  64,    "64x160x64"},
        {160,   1792, 160,   "160x1792x160"},
        {1792,  64,   160,   "1792x64x160"},
        {15744, 64,   160,   "15744x64x160"},
        {15744, 160,  1792,  "15744x160x1792"},
        {128,   128,  128,   "128x128x128"},
    };
    struct DtypeDef { const char* name; int key; };
    DtypeDef dtypes[] = {{"FP32", 0}, {"FP16", 1}, {"BF16", 2}, {"INT8_I32", 3}};
    for (const auto& sh : shapes) {
        for (const auto& dt : dtypes) {
            std::string tag = "S-" + std::string(dt.name) + "-" + sh.suffix;
            v.push_back(MakeBaseConfig(tag, dt.name, dt.key, sh.m, sh.k, sh.n, "S1-shape-dtype"));
        }
    }
}

inline void AddEpilogueConfigs(std::vector<PerfConfig>& v) {
    struct DtypeDef { const char* name; int key; };
    for (const auto& dt : {DtypeDef{"FP32", 0}, DtypeDef{"FP16", 1}}) {
        v.push_back(MakeEpiConfig("E-" + std::string(dt.name) + "-base", dt.name, dt.key,
                                   160, 1792, 160, false, false, false, 0, "S2-epilogue"));
        v.push_back(MakeEpiConfig("E-" + std::string(dt.name) + "-bias-ReLU", dt.name, dt.key,
                                   160, 1792, 160, true, false, false, 1, "S2-epilogue"));
        v.push_back(MakeEpiConfig("E-" + std::string(dt.name) + "-bias-GeLU", dt.name, dt.key,
                                   160, 1792, 160, true, false, false, 2, "S2-epilogue"));
        v.push_back(MakeEpiConfig("E-" + std::string(dt.name) + "-3vec-bias-ReLU", dt.name, dt.key,
                                   160, 1792, 160, true, true, true, 1, "S2-epilogue"));
    }
    v.push_back(MakeEpiConfig("E-INT8-base", "INT8_I32", 3, 160, 1792, 160, false, false, false, 0, "S2-epilogue"));
    v.push_back(MakeEpiConfig("E-INT8-bias-ReLU", "INT8_I32", 3, 160, 1792, 160, true, false, false, 1, "S2-epilogue"));
}

inline void AddBatchConfigs(std::vector<PerfConfig>& v) {
    struct DtypeDef { const char* name; int key; };
    for (const auto& dt : {DtypeDef{"FP32", 0}, DtypeDef{"FP16", 1}}) {
        for (int32_t nb : {1, 3, 4, 5}) {
            std::string tag = "B-" + std::string(dt.name) + "-b" + std::to_string(nb);
            PerfConfig c = MakeBaseConfig(tag, dt.name, dt.key, 64, 160, 64, "S3-batch");
            c.numBatches = nb;
            v.push_back(std::move(c));
        }
    }
}

inline void AddExtremeConfigs(std::vector<PerfConfig>& v) {
    v.push_back(MakeEpiConfig("X-FP32-15744x64x160-3vec", "FP32", 0, 15744, 64, 160, true, true, true, 1, "S4-extreme"));
    v.push_back(MakeEpiConfig("X-FP16-15744x64x160-3vec", "FP16", 1, 15744, 64, 160, true, true, true, 1, "S4-extreme"));
    v.push_back(MakeEpiConfig("X-BF16-15744x64x160-3vec", "BF16", 2, 15744, 64, 160, true, true, true, 1, "S4-extreme"));
    v.push_back(MakeEpiConfig("X-FP32-15744x160x1792-3vec", "FP32", 0, 15744, 160, 1792, true, true, true, 1, "S4-extreme"));
}

// ============================================================================
// Config list: covers all required dimensions.
// ============================================================================
inline std::vector<PerfConfig> BuildConfigList() {
    std::vector<PerfConfig> v;
    AddShapeDtypeConfigs(v);
    AddEpilogueConfigs(v);
    AddBatchConfigs(v);
    AddExtremeConfigs(v);
    return v;
}

}  // namespace

// ============================================================================
// Standalone main (separate executable, no GTest)
// ============================================================================
int main(int argc, char** argv) {
    // Optional: select a single tag via argv[1] for debugging / msprof single-case
    std::string onlyTag = (argc > 1) ? std::string(argv[1]) : "";

    AclEnvScope env;
    aclrtStream stream = env.stream();

    PrintHeader();
    auto configs = BuildConfigList();
    int okCount = 0, failCount = 0;
    for (const auto& c : configs) {
        if (!onlyTag.empty() && c.tag != onlyTag) { continue; }
        std::printf("# running [%s] %s %s m=%d k=%d n=%d batch=%d epi=%s ...\n",
                    c.section.c_str(), c.tag.c_str(), c.dtype.c_str(),
                    c.m, c.k, c.n, c.numBatches, EpilogueTag(c).c_str());
        std::fflush(stdout);
        if (RunConfig(c, stream)) {
            ++okCount;
        } else {
            ++failCount;
            BottleneckBreakdown bZero;
            Stats sZero;
            IterConfig icZero = GetIterConfig(c.m, c.k, c.n);
            PrintResultExt(c, sZero, bZero, icZero);
        }
    }

    // ================================================================
    // Methodology + summary
    // ================================================================
    std::printf("\n");
    std::printf("# ================================================================\n");
    std::printf("# Performance Collection Methodology\n");
    std::printf("# ================================================================\n");
    std::printf("# Warmup:          %d iterations (excluded from stats)\n", kWarmupIters);
    std::printf("# Measure:         %d iterations per round × %d rounds = %d total samples\n",
                kMeasureIters, kMultiRoundDefault, kMeasureIters * kMultiRoundDefault);
    std::printf("# Preheat:         warmup iterations discarded to eliminate cold-start overhead\n");
    std::printf("# Device isolation: default stream, synchronous (aclrtSynchronizeStream)\n");
    std::printf("# Timing:          aclrtEvent (EventGuard) — kernel-only, excludes H2D/D2H/prune\n");
    std::printf("# Aggregation:     median of per-round medians (cross-round stable median)\n");
    std::printf("# Dispersion:      stddev + CV computed from per-round medians\n");
    std::printf("# Reporting:       stable median as primary metric (not single-run best)\n");
    std::printf("# Bottleneck:      theoretical FLOPs/bytes proportions + achieved GFLOPS/GB/s\n");
    std::printf("# Target chip:     Ascend 950PR (arch35 / DAV_3510), CANN 9.1.0\n");
    std::printf("# ================================================================\n");
    std::printf("# Summary: ok=%d fail=%d total=%d\n", okCount, failCount,
                okCount + failCount);
    std::printf("# Sections:\n");
    std::printf("#   S1-shape-dtype: 7 shapes × 4 dtypes = 28 cases (baseline coverage)\n");
    std::printf("#   S2-epilogue:     bias+activation overhead comparison (10 cases)\n");
    std::printf("#   S3-batch:        batch=1/3/4/5 scaling (8 cases)\n");
    std::printf("#   S4-extreme:      m=15744 with 3-vector+bias+ReLU (4 cases)\n");
    std::printf("# ================================================================\n");

    return (failCount == 0) ? 0 : 1;
}
