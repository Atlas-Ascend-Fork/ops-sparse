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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclsparse_fp16_utils.h"
#include "cann_ops_sparse.h"

namespace {

constexpr int32_t kN = 128;
constexpr int32_t kTileM = 16;
constexpr int32_t kTileK = 16;
constexpr int32_t kTileN = 16;

// precision.md: float16 threshold = 2^-10, max threshold = 10 * threshold.
constexpr double kFp16MreThreshold = 1.0 / 1024.0;          // ~9.766e-04
constexpr double kFp16MareThreshold = 10.0 / 1024.0;       // ~9.766e-03

using aclsparse::Float16BitsToFloat32;
using aclsparse::Float32ToFloat16Bits;

struct CooElem {
    int32_t row;
    int32_t col;
    uint16_t val;
};

std::vector<CooElem> ReadMtx(const std::string &path, int64_t &M, int64_t &K, int64_t &nnz)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
        return {};
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        std::istringstream iss(line);
        iss >> M >> K >> nnz;
        break;
    }

    std::vector<CooElem> elems;
    elems.reserve(static_cast<size_t>(nnz));
    for (int64_t i = 0; i < nnz; ++i) {
        if (!std::getline(file, line)) {
            std::cerr << "Unexpected end of file at line " << i << std::endl;
            return {};
        }
        std::istringstream iss(line);
        int32_t r, c;
        float v = 1.0f;
        iss >> r >> c;
        if (!(iss >> v)) {
            v = 1.0f;
        }
        elems.push_back({r - 1, c - 1, Float32ToFloat16Bits(v)});
    }
    return elems;
}

struct SparseMatrix {
    int64_t M = 0;
    int64_t K = 0;
    int64_t nnz = 0;
    std::vector<int32_t> rows;
    std::vector<int32_t> cols;
    std::vector<uint16_t> vals;
};

SparseMatrix GenerateRandomSparseMatrix(
    int64_t M, int64_t K, double density, uint32_t seed)
{
    SparseMatrix sm;
    sm.M = M;
    sm.K = K;

    int64_t maxNnz = M * K;
    int64_t targetNnz = static_cast<int64_t>(static_cast<double>(maxNnz) * density);
    if (targetNnz < 1) {
        targetNnz = 1;
    }
    if (targetNnz > maxNnz) {
        targetNnz = maxNnz;
    }

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int64_t> rowDist(0, M - 1);
    std::uniform_int_distribution<int64_t> colDist(0, K - 1);
    std::uniform_real_distribution<float> valDist(-1.0f, 1.0f);

    std::set<std::pair<int64_t, int64_t>> used;
    sm.rows.reserve(static_cast<size_t>(targetNnz));
    sm.cols.reserve(static_cast<size_t>(targetNnz));
    sm.vals.reserve(static_cast<size_t>(targetNnz));

    while (sm.nnz < targetNnz) {
        int64_t r = rowDist(gen);
        int64_t c = colDist(gen);
        if (!used.insert({r, c}).second) {
            continue;
        }
        float v = valDist(gen);
        if (v == 0.0f) {
            v = 0.5f;
        }
        sm.rows.push_back(static_cast<int32_t>(r));
        sm.cols.push_back(static_cast<int32_t>(c));
        sm.vals.push_back(Float32ToFloat16Bits(v));
        ++sm.nnz;
    }
    return sm;
}

template <typename T>
bool AllocAndCopyToDevice(void **dev, const std::vector<T> &host)
{
    size_t bytes = host.size() * sizeof(T);
    if (aclrtMalloc(dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return false;
    }
    if (host.empty()) {
        return true;
    }
    if (aclrtMemcpy(*dev, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(*dev);
        *dev = nullptr;
        return false;
    }
    return true;
}

std::vector<uint16_t> GenerateDenseB(int64_t K, int64_t N, uint32_t seed = 42)
{
    std::vector<uint16_t> b(static_cast<size_t>(K * N));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < b.size(); ++i) {
        float v = dist(gen);
        if (v == 0.0f) {
            v = 0.5f;
        }
        b[i] = Float32ToFloat16Bits(v);
    }
    return b;
}

std::vector<float> ComputeGoldenFromCoo(
    int64_t actualM, int64_t N,
    const std::vector<int32_t> &cooRows,
    const std::vector<int32_t> &cooCols,
    const std::vector<uint16_t> &cooVals,
    const std::vector<uint16_t> &bBits)
{
    std::vector<float> B(bBits.size());
    for (size_t i = 0; i < bBits.size(); ++i) {
        B[i] = Float16BitsToFloat32(bBits[i]);
    }

    std::vector<float> C(static_cast<size_t>(actualM) * N, 0.0f);
    for (size_t i = 0; i < cooRows.size(); ++i) {
        int64_t r = cooRows[i];
        int64_t k = cooCols[i];
        float a = Float16BitsToFloat32(cooVals[i]);
        if (r >= actualM || a == 0.0f) {
            continue;
        }
        float *cRow = &C[static_cast<size_t>(r) * N];
        const float *bRow = &B[static_cast<size_t>(k) * N];
        for (int64_t n = 0; n < N; ++n) {
            cRow[n] += a * bRow[n];
        }
    }
    return C;
}

bool CheckAccuracy(const std::vector<float> &output, const std::vector<float> &golden,
                   double mreThreshold, double mareThreshold)
{
    if (output.size() != golden.size() || output.empty()) {
        std::cerr << "Size mismatch or empty output" << std::endl;
        return false;
    }

    double sumRel = 0.0;
    double maxRel = 0.0;
    size_t mismatch = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        double diff = std::fabs(static_cast<double>(output[i]) - static_cast<double>(golden[i]));
        double denom = std::fabs(static_cast<double>(golden[i])) + 1e-7;
        double rel = diff / denom;
        sumRel += rel;
        if (rel > maxRel) {
            maxRel = rel;
        }
        if (rel > mareThreshold) {
            if (mismatch < 5) {
                std::cerr << "Mismatch at " << i << ": out=" << output[i]
                          << " golden=" << golden[i] << " rel=" << rel << std::endl;
            }
            ++mismatch;
        }
    }
    double meanRel = sumRel / static_cast<double>(output.size());

    std::cout << "Mean Relative Error: " << meanRel
              << " (threshold " << mreThreshold << ")" << std::endl;
    std::cout << "Max Relative Error:  " << maxRel
              << " (threshold " << mareThreshold << ")" << std::endl;
    std::cout << "Mismatch count: " << mismatch << "/" << output.size() << std::endl;

    return meanRel < mreThreshold && maxRel < mareThreshold;
}

struct TestDeviceContext {
    aclsparseCubeSpmmMatDescr_t matA = nullptr;
    void *cooRowsDev = nullptr;
    void *cooColsDev = nullptr;
    void *cooValsDev = nullptr;
    void *bPad = nullptr;
    void *cDev = nullptr;
    void *workspace = nullptr;
    aclsparseDnMatDescr_t matBPad = nullptr;
    aclsparseDnMatDescr_t matCPad = nullptr;
};

void DestroyTestDeviceContext(TestDeviceContext &ctx)
{
    if (ctx.cDev != nullptr) aclrtFree(ctx.cDev);
    if (ctx.matCPad != nullptr) aclsparseDestroyDnMat(ctx.matCPad);
    if (ctx.matBPad != nullptr) aclsparseDestroyDnMat(ctx.matBPad);
    if (ctx.bPad != nullptr) aclrtFree(ctx.bPad);
    if (ctx.matA != nullptr) aclsparseDestroyCubeSpmmMat(ctx.matA);
    if (ctx.workspace != nullptr) aclrtFree(ctx.workspace);
    if (ctx.cooRowsDev != nullptr) aclrtFree(ctx.cooRowsDev);
    if (ctx.cooColsDev != nullptr) aclrtFree(ctx.cooColsDev);
    if (ctx.cooValsDev != nullptr) aclrtFree(ctx.cooValsDev);
}

struct AclEnvironment {
    bool ok = false;
    aclsparseHandle_t handle = nullptr;
    aclrtStream stream = nullptr;
    int32_t deviceId = -1;
};

AclEnvironment CreateAclEnvironment(int32_t deviceId)
{
    AclEnvironment env;
    env.deviceId = deviceId;
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "aclInit failed" << std::endl;
        return env;
    }
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed" << std::endl;
        aclFinalize();
        return env;
    }
    if (aclsparseCreate(&env.handle) != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "aclsparseCreate failed" << std::endl;
        aclrtResetDevice(deviceId);
        aclFinalize();
        return env;
    }
    if (aclrtCreateStream(&env.stream) != ACL_SUCCESS) {
        std::cerr << "aclrtCreateStream failed" << std::endl;
        aclsparseDestroy(env.handle);
        env.handle = nullptr;
        aclrtResetDevice(deviceId);
        aclFinalize();
        return env;
    }
    aclsparseSetStream(env.handle, env.stream);
    env.ok = true;
    return env;
}

void DestroyAclEnvironment(AclEnvironment &env)
{
    if (env.handle != nullptr) {
        aclsparseDestroy(env.handle);
        env.handle = nullptr;
    }
    if (env.stream != nullptr) {
        aclrtDestroyStream(env.stream);
        env.stream = nullptr;
    }
    if (env.deviceId >= 0) {
        aclrtResetDevice(env.deviceId);
        env.deviceId = -1;
    }
    aclFinalize();
}

bool ExecuteCubeSpmmTest(
    aclsparseHandle_t handle,
    aclrtStream stream,
    int64_t M, int64_t K, int64_t N, int64_t nnz,
    const std::vector<int32_t> &cooRows,
    const std::vector<int32_t> &cooCols,
    const std::vector<uint16_t> &cooVals,
    int32_t numCores)
{
    TestDeviceContext ctx;
    aclsparseStatus_t st;

    // Upload COO to device.
    if (!AllocAndCopyToDevice(&ctx.cooRowsDev, cooRows) ||
        !AllocAndCopyToDevice(&ctx.cooColsDev, cooCols) ||
        !AllocAndCopyToDevice(&ctx.cooValsDev, cooVals)) {
        std::cerr << "Failed to upload COO to device" << std::endl;
        return false;
    }

    // The cube kernel works on M padded to tileM.
    int64_t paddedM = ((M + kTileM - 1) / kTileM) * kTileM;

    // Create sparse descriptor.
    st = aclsparseCreateCubeSpmmMat(
        &ctx.matA, paddedM, K, nnz, kTileM, kTileK, numCores);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "aclsparseCreateCubeSpmmMat failed: " << st << std::endl;
        DestroyTestDeviceContext(ctx);
        return false;
    }

    // Generate deterministic dense B (actual N) on host.
    std::vector<uint16_t> bBits = GenerateDenseB(K, N);

    // 1) Preprocess sparse A: COO -> Cube-BCSR.
    st = aclsparseCubeSpmmPreprocess(
        handle, nullptr, ctx.matA,
        static_cast<const int32_t *>(ctx.cooRowsDev),
        static_cast<const int32_t *>(ctx.cooColsDev),
        static_cast<const uint16_t *>(ctx.cooValsDev),
        ACL_FLOAT);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "aclsparseCubeSpmmPreprocess failed: " << st << std::endl;
        DestroyTestDeviceContext(ctx);
        return false;
    }

    // 2) Pad dense B on host and transfer to device in one H2D copy.
    int64_t nPad = 0;
    st = aclsparseCubeSpmmPadDenseMatrixB(
        handle, K, N, N, bBits.data(),
        ACL_FLOAT16, ACL_SPARSE_ORDER_ROW,
        &nPad, &ctx.bPad);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "aclsparseCubeSpmmPadDenseMatrixB failed: " << st << std::endl;
        DestroyTestDeviceContext(ctx);
        return false;
    }
    std::cout << "N_pad=" << nPad << std::endl;

    // Test with non-trivial leading dimensions for B and C to verify the kernel
    // uses matB->ld / matC->ld instead of assuming compact layout.
    int64_t ldB = nPad + kTileN;
    int64_t ldC = nPad + kTileN;

    // Copy compact bPad into a strided B buffer (ldB > nPad).
    size_t bStrideBytes = static_cast<size_t>(K * ldB) * sizeof(uint16_t);
    void *bStride = nullptr;
    if (aclrtMalloc(&bStride, bStrideBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "aclrtMalloc for strided B failed" << std::endl;
        DestroyTestDeviceContext(ctx);
        return false;
    }
    if (aclrtMemset(bStride, bStrideBytes, 0, bStrideBytes) != ACL_SUCCESS) {
        std::cerr << "aclrtMemset for strided B failed" << std::endl;
        aclrtFree(bStride);
        DestroyTestDeviceContext(ctx);
        return false;
    }
    for (int64_t r = 0; r < K; ++r) {
        size_t dstOff = static_cast<size_t>(r * ldB) * sizeof(uint16_t);
        size_t srcOff = static_cast<size_t>(r * nPad) * sizeof(uint16_t);
        size_t rowBytes = static_cast<size_t>(nPad) * sizeof(uint16_t);
        aclError aclRet = aclrtMemcpy(static_cast<uint8_t *>(bStride) + dstOff, rowBytes,
                                      static_cast<const uint8_t *>(ctx.bPad) + srcOff, rowBytes,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            std::cerr << "aclrtMemcpy for strided B row " << r << " failed" << std::endl;
            aclrtFree(bStride);
            DestroyTestDeviceContext(ctx);
            return false;
        }
    }
    aclrtFree(ctx.bPad);
    ctx.bPad = bStride;

    // Build strided B/C descriptors for the actual SpMM.
    st = aclsparseCreateDnMat(&ctx.matBPad, K, nPad, ldB, ctx.bPad,
                              ACL_FLOAT16, ACL_SPARSE_ORDER_ROW);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "aclsparseCreateDnMat (matBPad) failed: " << st << std::endl;
        DestroyTestDeviceContext(ctx);
        return false;
    }

    size_t cBytes = static_cast<size_t>(paddedM * ldC) * sizeof(float);
    if (aclrtMalloc(&ctx.cDev, cBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemset(ctx.cDev, cBytes, 0, cBytes) != ACL_SUCCESS) {
        std::cerr << "aclrtMalloc/Memset for strided C failed" << std::endl;
        DestroyTestDeviceContext(ctx);
        return false;
    }

    st = aclsparseCreateDnMat(&ctx.matCPad, paddedM, nPad, ldC, ctx.cDev,
                              ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "aclsparseCreateDnMat (matCPad) failed: " << st << std::endl;
        DestroyTestDeviceContext(ctx);
        return false;
    }

    // Query workspace size. The current implementation passes tiling by value
    // with the kernel launch and needs no device workspace, so the size is 0
    // and ctx.workspace stays nullptr.
    size_t workspaceSize = 0;
    st = aclsparseCubeSpmmGetBufferSize(
        handle, nullptr, ctx.matA, ctx.matBPad, nullptr, ctx.matCPad, ACL_FLOAT, &workspaceSize);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "aclsparseCubeSpmmGetBufferSize failed: " << st << std::endl;
        DestroyTestDeviceContext(ctx);
        return false;
    }

    if (workspaceSize > 0 &&
        aclrtMalloc(&ctx.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "aclrtMalloc workspace failed" << std::endl;
        DestroyTestDeviceContext(ctx);
        return false;
    }

    std::vector<float> golden = ComputeGoldenFromCoo(M, N, cooRows, cooCols, cooVals, bBits);

    // First SpMM: tiling is rebuilt on host and passed with the kernel launch.
    st = aclsparseCubeSpmm(
        handle, nullptr, ctx.matA, ctx.matBPad, nullptr, ctx.matCPad, ACL_FLOAT, ctx.workspace);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "aclsparseCubeSpmm (first) failed: " << st << std::endl;
        DestroyTestDeviceContext(ctx);
        return false;
    }

    // Second SpMM: reuse the same Cube-BCSR (workspace stays nullptr).
    // Zero C first so the reuse call does not accumulate on the previous result.
    if (aclrtMemset(ctx.cDev, cBytes, 0, cBytes) != ACL_SUCCESS) {
        std::cerr << "aclrtMemset for C reuse failed" << std::endl;
        DestroyTestDeviceContext(ctx);
        return false;
    }
    st = aclsparseCubeSpmm(
        handle, nullptr, ctx.matA, ctx.matBPad, nullptr, ctx.matCPad, ACL_FLOAT, ctx.workspace);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "aclsparseCubeSpmm (second/reuse) failed: " << st << std::endl;
        DestroyTestDeviceContext(ctx);
        return false;
    }

    // Copy C back and extract the first actual [M, N] region using ldC stride.
    std::vector<float> outputPad(static_cast<size_t>(paddedM * ldC));
    aclError aclRet = aclrtMemcpy(outputPad.data(), cBytes, ctx.cDev, cBytes,
                                  ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        std::cerr << "aclrtMemcpy for C output failed" << std::endl;
        DestroyTestDeviceContext(ctx);
        return false;
    }

    std::vector<float> output;
    output.reserve(static_cast<size_t>(M * N));
    for (int64_t r = 0; r < M; ++r) {
        size_t srcOff = static_cast<size_t>(r * ldC);
        size_t dstOff = output.size();
        output.resize(dstOff + static_cast<size_t>(N));
        std::copy(outputPad.begin() + srcOff,
                  outputPad.begin() + srcOff + N,
                  output.begin() + dstOff);
    }

    bool pass = CheckAccuracy(output, golden, kFp16MreThreshold, kFp16MareThreshold);
    DestroyTestDeviceContext(ctx);
    return pass;
}

}  // namespace

static bool PrepareTestInputs(
    int argc, char **argv,
    int64_t &M, int64_t &K, int64_t &N, int64_t &nnz,
    std::vector<int32_t> &cooRows, std::vector<int32_t> &cooCols,
    std::vector<uint16_t> &cooVals, int32_t &numCores, int32_t &deviceId)
{
    // -------------------------------------------------------------------------
    // Determine input data source.
    //    - No arguments: generate a random sparse matrix for quick verification.
    //    - With arguments: run on an external MTX file (legacy mode).
    // -------------------------------------------------------------------------
    numCores = 20;
    deviceId = 0;
    bool randomMode = (argc <= 1);

    if (randomMode) {
        // Default random case for "bash build.sh --ops=cube_spmm --run".
        M = 512;
        K = 512;
        N = 1024;
        double density = 0.01;
        // Fixed seed keeps the default random case reproducible.
        SparseMatrix sm = GenerateRandomSparseMatrix(M, K, density, 12345u);
        M = sm.M;
        K = sm.K;
        nnz = sm.nnz;
        cooRows = std::move(sm.rows);
        cooCols = std::move(sm.cols);
        cooVals = std::move(sm.vals);
        std::cout << "Random mode: M=" << M << " K=" << K << " nnz=" << nnz
                  << " N=" << N << " numCores=" << numCores << std::endl;
        return true;
    }

    std::string mtxPath = argv[1];
    numCores = (argc > 2) ? std::atoi(argv[2]) : 20;
    deviceId = (argc > 3) ? std::atoi(argv[3]) : 0;
    N = (argc > 4) ? std::atoi(argv[4]) : kN;
    if (N <= 0) {
        std::cerr << "N must be positive" << std::endl;
        return false;
    }

    auto elems = ReadMtx(mtxPath, M, K, nnz);
    if (elems.empty() && nnz > 0) {
        std::cerr << "Failed to read MTX or empty matrix." << std::endl;
        return false;
    }
    cooRows.resize(static_cast<size_t>(nnz));
    cooCols.resize(static_cast<size_t>(nnz));
    cooVals.resize(static_cast<size_t>(nnz));
    for (int64_t i = 0; i < nnz; ++i) {
        cooRows[i] = elems[i].row;
        cooCols[i] = elems[i].col;
        cooVals[i] = elems[i].val;
    }
    std::cout << "M=" << M << " K=" << K << " nnz=" << nnz
              << " N=" << N << " numCores=" << numCores << std::endl;
    return true;
}

int main(int argc, char **argv)
{
    int64_t M = 0;
    int64_t K = 0;
    int64_t N = 0;
    int64_t nnz = 0;
    std::vector<int32_t> cooRows;
    std::vector<int32_t> cooCols;
    std::vector<uint16_t> cooVals;
    int32_t numCores = 20;
    int32_t deviceId = 0;

    if (!PrepareTestInputs(argc, argv, M, K, N, nnz, cooRows, cooCols, cooVals,
                           numCores, deviceId)) {
        return 1;
    }

    AclEnvironment env = CreateAclEnvironment(deviceId);
    if (!env.ok) {
        return 1;
    }

    bool pass = ExecuteCubeSpmmTest(env.handle, env.stream, M, K, N, nnz,
                                    cooRows, cooCols, cooVals, numCores);

    DestroyAclEnvironment(env);

    if (pass) {
        std::cout << "[Success] test case accuracy is verification passed." << std::endl;
        std::cout << "[PASS] cube_spmm_test" << std::endl;
        return 0;
    }
    std::cout << "[Failed] test case accuracy is verification failed." << std::endl;
    std::cout << "[FAIL] cube_spmm_test" << std::endl;
    return 1;
}
