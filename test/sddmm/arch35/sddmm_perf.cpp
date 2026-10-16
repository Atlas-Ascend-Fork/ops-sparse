/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
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
 *   - kernel time (min / median / mean, us)
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
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <string>
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
    aclDataType dtype;      // ACL_FLOAT / ACL_FLOAT16
    std::string dtype_name;
    double value_lo;
    double value_hi;
    uint32_t seed;
};

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
template <typename T>
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
    aclDataType dtype;
    aclDataType computeType;
    aclsparseSDDMMAlg_t alg;
    aclsparseOperation_t opX;
    aclsparseOperation_t opY;
    float alphaHost;
    float betaHost;
    size_t bufferSize;
    int64_t nnz;
    int64_t m, k, n;
    bool ok;
    std::string err;
};

template <typename T>
static void SetupPerfDescriptors(PerfWorkspace<T> *ws, aclrtStream stream, const PerfCase& pc,
                                 const SddmmCsr& csrC, const std::vector<T>& hCInit,
                                 const std::vector<T>& hX, const std::vector<T>& hY) {
    ws->dtype = pc.dtype;
    ws->computeType = pc.dtype;  // compute type == value type (FP32/FP16)
    ws->alg = ACL_SPARSE_SDDMM_ALG_DEFAULT;
    ws->opX = ACL_SPARSE_OP_NON_TRANSPOSE;
    ws->opY = ACL_SPARSE_OP_NON_TRANSPOSE;
    ws->alphaHost = static_cast<float>(pc.alpha);
    ws->betaHost = static_cast<float>(pc.beta);
    ws->m = pc.m;
    ws->k = pc.k;
    ws->n = pc.n;
    ws->nnz = csrC.nnz;
    ws->handle.setStream(stream);

    ws->dRowOff = DeviceBuffer::copyFrom(csrC.rowOffsets.data(),
                                         static_cast<size_t>(pc.m + 1) * sizeof(int32_t));
    if (csrC.nnz > 0) {
        ws->dColInd = DeviceBuffer::copyFrom(csrC.colIndices.data(),
                                             static_cast<size_t>(csrC.nnz) * sizeof(int32_t));
        ws->dVals = DeviceBuffer::copyFrom(hCInit.data(),
                                           static_cast<size_t>(csrC.nnz) * sizeof(T));
    }
    ws->dX = DeviceBuffer::copyFrom(hX.data(),
                                    static_cast<size_t>(pc.m) * static_cast<size_t>(pc.k) * sizeof(T));
    ws->dY = DeviceBuffer::copyFrom(hY.data(),
                                    static_cast<size_t>(pc.n) * static_cast<size_t>(pc.k) * sizeof(T));

    ws->matX = DnMatManager::createConst(pc.m, pc.k, pc.k, ws->dX.raw(), pc.dtype, ACL_SPARSE_ORDER_ROW);
    ws->matY = DnMatManager::createConst(pc.k, pc.n, pc.n, ws->dY.raw(), pc.dtype, ACL_SPARSE_ORDER_ROW);
    ws->matC = SpMatManager::createCsr(pc.m, pc.n, csrC.nnz, ws->dRowOff.get(), ws->dColInd.get(),
                                       ws->dVals.get(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_BASE_ZERO, pc.dtype);
}

template <typename T>
static bool SetupPerfBuffers(PerfWorkspace<T> *ws, aclrtStream stream) {
    size_t bs = 0;
    auto r1 = aclsparseSDDMMBufferSize(ws->handle.get(), ws->opX, ws->opY, &ws->alphaHost,
                                       ws->matX.cget(), ws->matY.cget(), &ws->betaHost, ws->matC.get(),
                                       ws->computeType, ws->alg, &bs);
    if (r1 != ACL_SPARSE_STATUS_SUCCESS) {
        ws->err = "BufferSize failed: " + std::to_string(r1);
        return false;
    }
    ws->bufferSize = bs;
    if (bs > 0) {
        ws->dBuffer = DeviceBuffer::alloc(bs);
    }
    auto r2 = aclsparseSDDMMPreprocess(ws->handle.get(), ws->opX, ws->opY, &ws->alphaHost,
                                       ws->matX.cget(), ws->matY.cget(), &ws->betaHost, ws->matC.get(),
                                       ws->computeType, ws->alg, ws->dBuffer.get());
    if (r2 != ACL_SPARSE_STATUS_SUCCESS) {
        ws->err = "Preprocess failed: " + std::to_string(r2);
        return false;
    }
    // sync after preprocess to ensure workspace is ready before timing
    aclrtSynchronizeStream(stream);
    return true;
}

template <typename T>
static std::unique_ptr<PerfWorkspace<T>> SetupPerf(aclrtStream stream, const PerfCase& pc,
                                                   const SddmmCsr& csrC,
                                                   const std::vector<T>& hCInit,
                                                   const std::vector<T>& hX,
                                                   const std::vector<T>& hY) {
    auto ws = std::make_unique<PerfWorkspace<T>>();
    ws->ok = false;
    SetupPerfDescriptors<T>(ws.get(), stream, pc, csrC, hCInit, hX, hY);
    ws->ok = SetupPerfBuffers<T>(ws.get(), stream);
    return ws;
}

// Time Execute (kernel) over warmup + iters with aclrtEvent. Returns times (us).
// exec is called once per iteration (must launch the kernel asynchronously).
template <typename ExecFn>
static std::vector<double> TimeExecute(aclrtStream stream, ExecFn exec, int warmup, int iters) {
    std::vector<double> times;
    times.reserve(static_cast<size_t>(iters));

    aclrtEvent startEv = nullptr;
    aclrtEvent stopEv = nullptr;
    aclrtCreateEvent(&startEv);
    aclrtCreateEvent(&stopEv);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        exec();
    }
    aclrtSynchronizeStream(stream);

    // Measured iterations: one event pair per iteration
    for (int i = 0; i < iters; ++i) {
        aclrtRecordEvent(startEv, stream);
        exec();
        aclrtRecordEvent(stopEv, stream);
        aclrtSynchronizeStream(stream);
        float ms = 0.0f;
        aclrtEventElapsedTime(&ms, startEv, stopEv);
        times.push_back(static_cast<double>(ms) * 1000.0);  // us
    }

    aclrtDestroyEvent(startEv);
    aclrtDestroyEvent(stopEv);
    return times;
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
//   X read     : m * k * elem        (each X element read >= 1x)
//   Y read     : nnz * k * elem      (kernel loads Y[j,:] per nonzero; dominates)
//   colInd read: nnz * 4
//   rowOff read: (m + 1) * 4
//   C read     : beta != 0 ? nnz * elem : 0
//   C write    : nnz * elem
//
// elem = sizeof(T): FP32=4, FP16=2.

struct AlgoMetrics {
    int64_t nnz;
    double flops;
    double bytes;
};

static AlgoMetrics ComputeAlgoMetrics(const PerfCase& pc, int64_t nnz) {
    AlgoMetrics m;
    m.nnz = nnz;
    double elem = (pc.dtype == ACL_FLOAT) ? 4.0 : 2.0;
    double flops = 2.0 * static_cast<double>(nnz) * static_cast<double>(pc.k);
    flops += 2.0 * static_cast<double>(nnz);  // alpha-scale + beta fma per nonzero
    m.flops = flops;

    double bytes = 0.0;
    bytes += static_cast<double>(pc.m) * static_cast<double>(pc.k) * elem;   // X
    bytes += static_cast<double>(nnz) * static_cast<double>(pc.k) * elem;    // Y (per-nonzero, dominates)
    bytes += static_cast<double>(nnz) * 4.0;                                 // colInd
    bytes += static_cast<double>(pc.m + 1) * 4.0;                            // rowOff
    if (pc.beta != 0.0) {
        bytes += static_cast<double>(nnz) * elem;                            // C read
    }
    bytes += static_cast<double>(nnz) * elem;                                // C write
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
    std::sort(times.begin(), times.end());
    double tMin = times.front();
    double tMed = times[times.size() / 2];
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
    int64_t nnz;
    double tMin, tMed, tMean;
    double bwGbs, gflops, bwUtil;
    std::string note;
};

static std::string Fmt(double v, int width) {
    if (v <= 0) return std::string("-");
    std::ostringstream os;
    os << std::fixed << std::setprecision(2) << v;
    std::string s = os.str();
    if (static_cast<int>(s.length()) > width) s = s.substr(0, width);
    return s;
}

struct PerfArgs {
    int warmup = 5;
    int iters = 20;
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

template <typename T>
static void RunPerfTyped(aclrtStream stream, const PerfCase& pc, const SddmmCsr& csrC,
                         const std::vector<T>& hCInit, const std::vector<T>& hX,
                         const std::vector<T>& hY, int warmup, int iters, RowOut& row) {
    auto ws = SetupPerf<T>(stream, pc, csrC, hCInit, hX, hY);
    if (!ws->ok) {
        row.note = "SETUP_FAIL:" + ws->err;
        return;
    }
    auto exec = [&]() {
        return aclsparseSDDMM(ws->handle.get(), ws->opX, ws->opY, &ws->alphaHost,
                              ws->matX.cget(), ws->matY.cget(), &ws->betaHost, ws->matC.get(),
                              ws->computeType, ws->alg, ws->dBuffer.get());
    };
    auto times = TimeExecute(stream, exec, warmup, iters);
    std::sort(times.begin(), times.end());
    row.tMin = times.front();
    row.tMed = times[times.size() / 2];
    double sum = 0; for (double t : times) sum += t;
    row.tMean = sum / times.size();
    row.note = "ok";
}

static RowOut RunOneCase(aclrtStream stream, const PerfCase& pc, int warmup, int iters,
                         double peakHbmGbps) {
    RowOut row;
    row.label = pc.label;
    row.dtype_name = pc.dtype_name;
    row.tMin = row.tMed = row.tMean = 0;
    row.bwGbs = row.gflops = 0;
    row.bwUtil = -1;
    row.note = "";

    SddmmCsr csrC = MakeSddmmSparsity(pc.m, pc.n, pc.sparsity_ratio,
                                      pc.value_lo, pc.value_hi, pc.seed);
    int64_t nnz = csrC.nnz;
    row.nnz = nnz;

    std::mt19937 rngX(pc.seed + 1);
    std::mt19937 rngY(pc.seed + 2);
    std::uniform_real_distribution<double> dist(pc.value_lo, pc.value_hi);
    std::vector<double> Xf64(static_cast<size_t>(pc.m) * static_cast<size_t>(pc.k));
    std::vector<double> Yf64(static_cast<size_t>(pc.n) * static_cast<size_t>(pc.k));
    for (size_t i = 0; i < Xf64.size(); i++) Xf64[i] = dist(rngX);
    for (size_t i = 0; i < Yf64.size(); i++) Yf64[i] = dist(rngY);

    if (pc.dtype == ACL_FLOAT) {
        RunPerfTyped<float>(stream, pc, csrC, DoublesToFp32(csrC.values),
                            DoublesToFp32(Xf64), DoublesToFp32(Yf64), warmup, iters, row);
    } else {
        RunPerfTyped<uint16_t>(stream, pc, csrC, DoublesToFp16(csrC.values),
                               DoublesToFp16(Xf64), DoublesToFp16(Yf64), warmup, iters, row);
    }

    auto am = ComputeAlgoMetrics(pc, nnz);
    double tUs = row.tMed;
    if (tUs > 0) {
        double tSec = tUs / 1e6;
        row.bwGbs = am.bytes / tSec / 1e9;
        row.gflops = am.flops / tSec / 1e9;
        row.bwUtil = (peakHbmGbps > 0) ? (row.bwGbs / peakHbmGbps * 100.0) : -1.0;
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
        return false;
    }
    aclrtCreateStream(&stream);
    return true;
}

static void CleanupAclEnv(aclrtStream stream) {
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

static void PrintHeader(int64_t arch, size_t totalMem, size_t freeMem,
                        int warmup, int iters, double peakHbmGbps) {
    std::cout << "============================================================\n";
    std::cout << "SDDMM Performance Collection (aclsparseSDDMM, Execute stage)\n";
    std::cout << "Device: Ascend arch=" << arch << " (DAV_3510 / Ascend950)\n";
    std::cout << "HBM total=" << (totalMem >> 20) << " MiB  free=" << (freeMem >> 20) << " MiB\n";
    std::cout << "warmup=" << warmup << " iters=" << iters
              << " peakHbmGbps=" << (peakHbmGbps > 0 ? std::to_string(peakHbmGbps) : std::string("N/A"))
              << "\n";
    std::cout << "Timing: aclrtEvent around aclsparseSDDMM (Execute), per-iter sync\n";
    std::cout << "============================================================\n";

    printf("+------------------------+--------+----------+--------+--------+--------+----------+----------+--------+\n");
    printf("| %-22s | %-6s | %8s | %6s | %6s | %6s | %8s | %8s | %6s |\n",
           "case", "dtype", "nnz", "min(us)", "med(us)", "mean(u)", "BW(GB/s)", "GFLOPS", "BWutil");
    printf("+------------------------+--------+----------+--------+--------+--------+----------+----------+--------+\n");
}

static void RunAllCases(aclrtStream stream, const std::vector<PerfCase>& cases,
                        const std::string& only, int warmup, int iters, double peakHbmGbps,
                        std::vector<RowOut>& rows, std::vector<PerfCase>& filtered_cases) {
    for (const auto& pc : cases) {
        if (!only.empty() && pc.label.find(only) == std::string::npos) {
            continue;
        }
        RowOut row = RunOneCase(stream, pc, warmup, iters, peakHbmGbps);
        int64_t nnz = row.nnz;

        rows.push_back(row);
        filtered_cases.push_back(pc);

        std::string bwutilStr = (row.bwUtil < 0) ? "N/A" : Fmt(row.bwUtil, 6);
        printf("| %-22s | %-6s | %8lld | %6s | %6s | %6s | %8s | %8s | %6s |\n",
               row.label.c_str(), row.dtype_name.c_str(),
               static_cast<long long>(nnz),
               Fmt(row.tMin, 6).c_str(), Fmt(row.tMed, 6).c_str(), Fmt(row.tMean, 6).c_str(),
               Fmt(row.bwGbs, 8).c_str(), Fmt(row.gflops, 8).c_str(),
               bwutilStr.c_str());
        fflush(stdout);
    }
    printf("+------------------------+--------+----------+--------+--------+--------+----------+----------+--------+\n");
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
        std::cout << "  m=" << pc.m << " k=" << pc.k << " n=" << pc.n
                  << " dtype=" << pc.dtype_name
                  << " alpha=" << pc.alpha << " beta=" << pc.beta
                  << " ratio=" << pc.sparsity_ratio << "\n";
        std::cout << "  nnz=" << row.nnz
                  << " (density=" << std::fixed << std::setprecision(4)
                  << (pc.n > 0 ? static_cast<double>(row.nnz) / static_cast<double>(pc.m * pc.n) : 0.0)
                  << ")\n";
        std::cout << std::defaultfloat;
        std::cout << "  FLOPs=" << static_cast<uint64_t>(am.flops)
                  << "  HBM_bytes=" << static_cast<uint64_t>(am.bytes) << "\n";
        std::cout << "  time: min=" << std::fixed << std::setprecision(2) << row.tMin
                  << "us  median=" << row.tMed << "us  mean=" << row.tMean << "us\n";
        std::cout << "  achieved BW=" << row.bwGbs << " GB/s  GFLOPS=" << row.gflops;
        if (row.bwUtil >= 0) {
            std::cout << "  BWutil=" << row.bwUtil << "%";
        }
        std::cout << "  [" << row.note << "]\n\n";
        std::cout << std::defaultfloat;
    }
}

static void PrintJsonSummary(const std::vector<PerfCase>& filtered_cases,
                             const std::vector<RowOut>& rows,
                             double peakHbmGbps, int warmup, int iters) {
    // Machine-readable JSON summary
    std::cout << "=== JSON ===\n";
    std::cout << "{\"device\":\"Ascend950_dav3510\",\"peak_hbm_gbps\":" << peakHbmGbps
              << ",\"warmup\":" << warmup << ",\"iters\":" << iters << ",\"cases\":[";
    for (size_t i = 0; i < filtered_cases.size(); ++i) {
        const auto& pc = filtered_cases[i];
        const auto& row = rows[i];
        auto am = ComputeAlgoMetrics(pc, row.nnz);
        if (i) std::cout << ",";
        std::cout << "{\"label\":\"" << row.label << "\""
                  << ",\"m\":" << pc.m << ",\"k\":" << pc.k << ",\"n\":" << pc.n
                  << ",\"dtype\":\"" << pc.dtype_name << "\""
                  << ",\"nnz\":" << row.nnz
                  << ",\"alpha\":" << pc.alpha << ",\"beta\":" << pc.beta
                  << ",\"ratio\":" << pc.sparsity_ratio
                  << ",\"flops\":" << static_cast<uint64_t>(am.flops)
                  << ",\"hbm_bytes\":" << static_cast<uint64_t>(am.bytes)
                  << ",\"t_min_us\":" << row.tMin
                  << ",\"t_med_us\":" << row.tMed
                  << ",\"t_mean_us\":" << row.tMean
                  << ",\"bw_gbs\":" << row.bwGbs
                  << ",\"gflops\":" << row.gflops
                  << ",\"bw_util_pct\":" << row.bwUtil
                  << ",\"status\":\"" << row.note << "\"}";
    }
    std::cout << "]}\n";
}

int main(int argc, char** argv) {
    PerfArgs args = ParsePerfArgs(argc, argv);
    int warmup = args.warmup;
    int iters = args.iters;
    double peakHbmGbps = args.peakHbmGbps;
    const std::string& only = args.only;
    bool bwbench = args.bwbench;

    // Representative performance cases (aligned with task 4.1 selection).
    std::vector<PerfCase> cases = {
        {"case1  128x64x128  FP32 S",   128, 64, 128,   0.5, 1.0, 0.0, ACL_FLOAT,   "FP32", -1.0, 1.0, 1},
        {"case6  64x32x64    FP16 S",   64,  32, 64,    0.5, 1.0, 0.0, ACL_FLOAT16, "FP16", -0.5, 0.5, 6},
        {"case4  1024x512x1024 FP16 M", 1024,512, 1024,  0.5, 0.5, 0.5, ACL_FLOAT16, "FP16", -1.0, 1.0, 4},
        {"case15 640x320x640  FP32 M",  640, 320,640,   0.5, 0.5, 0.5, ACL_FLOAT,   "FP32", -3.0, 3.0, 15},
        {"case5  2048x1024x2048 FP32 L",2048,1024,2048, 0.5, 1.0, 0.0, ACL_FLOAT,   "FP32", -1.0, 1.0, 5},
        {"case16 1280x640x1280 FP16 L", 1280,640, 1280,  0.5, 1.0, 0.0, ACL_FLOAT16, "FP16", -1.0, 1.0, 16},
        {"case20 2048x512x1024 FP32 NF",2048,512, 1024,  0.5, 1.0, 1.0, ACL_FLOAT,   "FP32", -1.0, 1.0, 20},
    };

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
    aclrtGetMemInfo(ACL_HBM_MEM, &freeMem, &totalMem);

    PrintHeader(arch, totalMem, freeMem, warmup, iters, peakHbmGbps);

    std::vector<RowOut> rows;
    std::vector<PerfCase> filtered_cases;
    RunAllCases(stream, cases, only, warmup, iters, peakHbmGbps, rows, filtered_cases);

    PrintDetailedBreakdown(filtered_cases, rows);
    PrintJsonSummary(filtered_cases, rows, peakHbmGbps, warmup, iters);

    CleanupAclEnv(stream);
    return 0;
}
