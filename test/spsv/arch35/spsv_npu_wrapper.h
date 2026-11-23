/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPSV_NPU_WRAPPER_H_
#define TEST_SPSV_NPU_WRAPPER_H_

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <complex>
#include <vector>

#include "descriptor_manager.h"
#include "fill.h"
#include "sparse_test.h"
#include "spsv_npu_setup.h"
#include "../../../sparse/common/aclsparse_spsv_descr.h"

namespace sparse_test {

template <typename A>
static inline DeviceBuffer RunAnalysisT(aclsparseHandle_t handle, const SpSVNpuConfig& cfg,
                                        const A& alpha, SpMatManager& matA,
                                        DnVecManager& vecX, DnVecManager& vecY,
                                        SpSVDescrManager& spsvDescr, aclDataType computeType) {
    size_t bufferSize = 0;
    auto ret = aclsparseSpSV_bufferSize(
        handle, cfg.opA, &alpha, matA.cget(),
        cfg.nullVec ? nullptr : vecX.cget(),
        cfg.nullVec ? nullptr : vecY.get(),
        computeType, ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr.get(), &bufferSize);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("aclsparseSpSV_bufferSize failed, status=" + std::to_string(ret));
    }
    DeviceBuffer dWorkspace;
    if (bufferSize > 0) dWorkspace = DeviceBuffer::alloc(bufferSize);
    ret = aclsparseSpSV_analysis(
        handle, cfg.opA, &alpha, matA.cget(),
        cfg.nullVec ? nullptr : vecX.cget(),
        cfg.nullVec ? nullptr : vecY.get(),
        computeType, ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr.get(),
        bufferSize > 0 ? dWorkspace.get() : nullptr);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("aclsparseSpSV_analysis failed, status=" + std::to_string(ret));
    }
    if (cfg.analysisCacheState >= 0) {
        if (aclrtSynchronizeEvent(spsvDescr.get()->analysisEvent) != ACL_SUCCESS) {
            throw std::runtime_error("analysis cache synchronization failed");
        }
        spsvDescr.get()->cachedNumLevels = cfg.analysisCacheState ? *spsvDescr.get()->analysisNumLevels : 0;
        spsvDescr.get()->analysisSnapshotCurrent = false;
    }
    return dWorkspace;
}

static inline DeviceBuffer RunAnalysis(aclsparseHandle_t handle, const SpSVNpuConfig& cfg,
                                        float alpha, SpMatManager& matA,
                                        DnVecManager& vecX, DnVecManager& vecY,
                                        SpSVDescrManager& spsvDescr) {
    return RunAnalysisT(handle, cfg, alpha, matA, vecX, vecY, spsvDescr, ACL_FLOAT);
}

template <typename A, typename V>
static inline void RunSolveT(aclsparseHandle_t handle, aclrtStream stream,
                             const SpSVNpuConfig& cfg, const A& alpha,
                             SpMatManager& matA, DnVecManager& vecX, DnVecManager& vecY,
                             SpSVDescrManager& spsvDescr, DeviceBuffer& dY,
                             std::vector<V>& outY, aclDataType computeType) {
    auto ret = aclsparseSpSV_solve(
        handle, cfg.opA, &alpha, matA.cget(), vecX.cget(), vecY.get(),
        computeType, ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr.get());
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("aclsparseSpSV_solve failed, status=" + std::to_string(ret));
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtSynchronizeStream failed after solve");
    }
    int64_t m = static_cast<int64_t>(outY.size());
    dY.copyToHost(outY.data(), m * sizeof(V));
}

static inline void RunSolve(aclsparseHandle_t handle, aclrtStream stream,
                             const SpSVNpuConfig& cfg, float alpha,
                             SpMatManager& matA, DnVecManager& vecX, DnVecManager& vecY,
                             SpSVDescrManager& spsvDescr, DeviceBuffer& dY,
                             std::vector<float>& outY) {
    RunSolveT(handle, stream, cfg, alpha, matA, vecX, vecY, spsvDescr, dY, outY, ACL_FLOAT);
}

// Upload the new values and hand them to updateMatrix.
//
// Returns the device buffer: on a layout with no workspace copy (zero-base
// CSR, NON_TRANSPOSE) a GENERAL update keeps this pointer rather than
// copying from it, so it has to outlive the next solve.
static inline DeviceBuffer SendUpdate(aclsparseHandle_t handle,
                                      SpSVDescrManager& spsvDescr,
                                      const std::vector<float>& newVals,
                                      const std::string& updateMode) {
    auto dNewVals = DeviceBuffer::copyFrom(newVals.data(), newVals.size() * sizeof(float));
    aclsparseSpSVUpdate_t updatePart = (updateMode == "DIAGONAL")
        ? ACL_SPARSE_SPSV_UPDATE_DIAGONAL : ACL_SPARSE_SPSV_UPDATE_GENERAL;
    auto ret = aclsparseSpSV_updateMatrix(handle, spsvDescr.get(), dNewVals.get(), updatePart);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("aclsparseSpSV_updateMatrix failed, status=" + std::to_string(ret));
    }
    return dNewVals;
}

/**
 * @brief SELL + GENERAL 模式更新：newVals 按 SELL 槽位布局组织
 * @details SELL 槽位布局与 CSR 顺序不同（除非 maxRowNnz=1），
 *          kernel 通过 perm[csrIdx] 映射 CSR 位置 → SELL 槽位位置，
 *          因此 updateMatrix 读取的是 SELL 槽位索引的值.
 */
// Returns SendUpdate's buffer, which the caller must keep alive.
static inline DeviceBuffer RunUpdateSellGeneral(aclsparseHandle_t handle,
                                         SpSVDescrManager& spsvDescr,
                                         int32_t sliceWidth, const CsrMatrix& csr) {
    std::mt19937 rng(9999);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    const int32_t sw = sliceWidth;
    int32_t maxRowNnz = 0;
    for (int64_t i = 0; i < csr.rows; i++) {
        int32_t rNnz = csr.rowOffsets[i + 1] - csr.rowOffsets[i];
        if (rNnz > maxRowNnz) maxRowNnz = rNnz;
    }
    int32_t numSlices = (maxRowNnz + sw - 1) / sw;
    if (numSlices == 0) numSlices = 1;
    // sliceOffsets[s] = s * rows * sw (均匀切片)
    const size_t totalSlots = static_cast<size_t>(numSlices) * csr.rows * sw;
    std::vector<float> newVals(totalSlots, 0.0f);

    for (int64_t i = 0; i < csr.rows; i++) {
        float rowAbsSum = 0.0f;
        int32_t diagSellPos = -1;
        int32_t localIdx = 0;
        for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
            int32_t s = localIdx / sw;
            int32_t off = localIdx % sw;
            int32_t sellPos = s * static_cast<int32_t>(csr.rows) * sw +
                              static_cast<int32_t>(i) * sw + off;
            if (csr.colIndices[k] == static_cast<int32_t>(i)) {
                diagSellPos = sellPos;
            } else {
                float v = dist(rng);
                newVals[sellPos] = v;
                rowAbsSum += std::abs(v);
            }
            localIdx++;
        }
        if (diagSellPos >= 0) {
            newVals[diagSellPos] = rowAbsSum + 1.0f;
        }
    }

    return SendUpdate(handle, spsvDescr, newVals, "GENERAL");
}

/**
 * @brief CSR/COO 格式更新（GENERAL 或 DIAGONAL 模式）
 * @details newVals 按 CSR 顺序索引；GENERAL 模式构造对角占优保证收敛性，
 *          DIAGONAL 模式生成随机值（仅对角线位置被 kernel 读取）.
 */
// Returns SendUpdate's buffer, which the caller must keep alive.
static inline DeviceBuffer RunUpdateCsrLayout(aclsparseHandle_t handle,
                                       SpSVDescrManager& spsvDescr,
                                       const std::string& updateMode,
                                       const CsrMatrix& csr) {
    std::mt19937 rng(9999);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> newVals(csr.values.size());
    if (updateMode == "GENERAL") {
        for (int64_t i = 0; i < csr.rows; i++) {
            float rowAbsSum = 0.0f;
            int32_t diagIdx = -1;
            for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
                if (csr.colIndices[k] == static_cast<int32_t>(i)) {
                    diagIdx = k;
                } else {
                    float v = dist(rng);
                    newVals[k] = v;
                    rowAbsSum += std::abs(v);
                }
            }
            if (diagIdx >= 0) {
                newVals[diagIdx] = rowAbsSum + 1.0f;
            }
        }
    } else {
        // DIAGONAL 模式: 全量生成随机值, 仅对角线位置被 kernel 读取
        for (size_t i = 0; i < newVals.size(); i++) newVals[i] = dist(rng);
    }
    return SendUpdate(handle, spsvDescr, newVals, updateMode);
}

/**
 * @brief CSC 格式更新（GENERAL 或 DIAGONAL 模式）
 * @details newVals 按 CSC 布局（column-major of A^T, 即按列存储 A^T,
 *          等价于按行存储 A 的转置）索引。cuSPARSE 规范要求 CSC 的 newValues
 *          必须按 CSC 列主序排列，才能正确触发 perm 映射路径。
 *          实现采用"列计数 - 前缀和 - scatter"三步法：从 CSR 表示的矩阵遍历，
 *          按列散射到 CSC 顺序的 newVals 数组。
 */
// Returns SendUpdate's buffer, which the caller must keep alive.
static inline DeviceBuffer RunUpdateCsc(aclsparseHandle_t handle,
                                 SpSVDescrManager& spsvDescr,
                                 const std::string& updateMode,
                                 const CsrMatrix& csr) {
    std::mt19937 rng(9999);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    const int64_t m = csr.rows;
    const size_t nnz = csr.nnz;
    std::vector<float> newVals(nnz);

    if (updateMode == "GENERAL") {
        // Pass 1: 统计每列（A^T 的列 = A 的行）的非零元个数
        std::vector<int64_t> colCount(m, 0);
        for (int64_t i = 0; i < m; i++) {
            for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
                colCount[csr.colIndices[k]]++;
            }
        }
        // Pass 2: 计算每列的起始偏移 (exclusive prefix sum)
        std::vector<int64_t> colOffset(m + 1, 0);
        for (int64_t j = 0; j < m; j++) {
            colOffset[j + 1] = colOffset[j] + colCount[j];
        }
        // Pass 3: Scatter - 为每个非零元生成新值，按 CSC 顺序写入 newVals
        // 遍历时维护每行（原始矩阵）的对角占优约束：对非对角元生成随机值，最后设置对角元
        std::vector<int64_t> writePos(colOffset.begin(), colOffset.end() - 1);
        std::vector<float> rowAbsSum(m, 0.0f);
        std::vector<int64_t> diagCscPos(m, -1);

        // 先写入非对角元
        for (int64_t i = 0; i < m; i++) {
            for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
                int64_t j = csr.colIndices[k];
                int64_t pos = writePos[j]++;
                if (static_cast<int32_t>(i) == csr.colIndices[k]) {
                    // 对角元，暂时记录位置
                    diagCscPos[i] = pos;
                } else {
                    float v = dist(rng);
                    newVals[pos] = v;
                    rowAbsSum[i] += std::abs(v);
                }
            }
        }
        // 设置对角元，保证对角占优
        for (int64_t i = 0; i < m; i++) {
            if (diagCscPos[i] >= 0) {
                newVals[diagCscPos[i]] = rowAbsSum[i] + 1.0f;
            }
        }
    } else {
        // DIAGONAL 模式: 全量生成 CSC 顺序的随机值
        for (size_t i = 0; i < nnz; i++) newVals[i] = dist(rng);
    }

    return SendUpdate(handle, spsvDescr, newVals, updateMode);
}

/**
 * @brief RunUpdate 主调度器：根据 format/mode 路由至对应更新实现，
 *        完成后调用 RunSolve 获取更新后的解.
 */
static inline void RunUpdate(aclsparseHandle_t handle, aclrtStream stream,
                              const SpSVNpuConfig& cfg, float alpha,
                              const CsrMatrix& csr, SpMatManager& matA,
                              DnVecManager& vecX, DnVecManager& vecY,
                              SpSVDescrManager& spsvDescr, DeviceBuffer& dY,
                              SpSVNpuResult& result) {
    DeviceBuffer dNewVals;
    if (cfg.updateMode == "GENERAL" && cfg.format == "SLICED_ELL") {
        dNewVals = RunUpdateSellGeneral(handle, spsvDescr, cfg.sliceWidth, csr);
    } else if (cfg.format == "CSC") {
        dNewVals = RunUpdateCsc(handle, spsvDescr, cfg.updateMode, csr);
    } else {
        dNewVals = RunUpdateCsrLayout(handle, spsvDescr, cfg.updateMode, csr);
    }

    int64_t m = csr.rows;
    result.y2.resize(m);
    // dNewVals stays in scope across this call on purpose: for a GENERAL
    // update on a layout with no workspace copy the descriptor is pointing
    // into it, and RunSolve synchronises before returning.
    RunSolve(handle, stream, cfg, alpha, matA, vecX, vecY, spsvDescr, dY, result.y2);
    result.hasSecondSolve = true;
}

struct SpSVNpuResultComplex {
    std::vector<std::complex<float>> y;
    std::vector<std::complex<float>> y2;         // the solve after updateMatrix
    std::vector<std::complex<float>> newValues;  // what the update sent, CSR order
    bool hasSecondSolve = false;
    int32_t analysisLevels = 0;
};

// New complex values for updateMatrix.  GENERAL is parallel to the value
// array and keeps the matrix diagonally dominant so the triangular solve
// stays well conditioned.  DIAGONAL is indexed by ROW -- the kernel does
// values[diagPtr[i]] = newDiag[i] -- so entry i is row i's new diagonal and
// the rest of the buffer is never read.
//
// Unlike the float path, which writes this generator out a second time in the
// test to rebuild the golden, these values are handed back to the caller, so
// the golden cannot drift away from what the device was given.
static inline std::vector<std::complex<float>> MakeComplexUpdateValues(
    const CsrMatrix& csr, const std::string& updateMode) {
    using cf = std::complex<float>;
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<cf> vals(static_cast<size_t>(csr.nnz), cf(0.0f, 0.0f));
    if (updateMode != "GENERAL") {
        // entry i is row i's diagonal; keep it away from zero so the solve
        // stays finite, and give it an imaginary part so H differs from T
        for (int64_t i = 0; i < csr.rows; i++) {
            vals[i] = cf(dist(rng) + (dist(rng) < 0.0f ? -2.5f : 2.5f), dist(rng));
        }
        return vals;
    }
    for (int64_t i = 0; i < csr.rows; i++) {
        float rowAbsSum = 0.0f;
        int32_t diagIdx = -1;
        for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
            if (csr.colIndices[k] == static_cast<int32_t>(i)) {
                diagIdx = k;
            } else {
                cf v(dist(rng), dist(rng));
                vals[k] = v;
                rowAbsSum += std::abs(v);
            }
        }
        // a real dominant diagonal, with an imaginary part small enough to
        // keep |diag| dominant but large enough that T and H differ
        if (diagIdx >= 0) vals[diagIdx] = cf(rowAbsSum + 1.0f, dist(rng) * 0.25f);
    }
    return vals;
}

// updateMatrix for the complex path.  GENERAL values go up in the format's
// own slot order, the same ordering the initial upload used, so CSC and
// SLICED_ELL are covered without a per-format generator; DIAGONAL values are
// indexed by row and go up as they are.
static inline DeviceBuffer RunUpdateComplex(aclsparseHandle_t handle,
                                     SpSVDescrManager& spsvDescr,
                                     const CsrMatrix& csr,
                                     const SpSVNpuConfig& cfg,
                                     const std::vector<std::complex<float>>& newVals) {
    using cf = std::complex<float>;
    std::vector<cf> buf;
    if (cfg.updateMode == "DIAGONAL") {
        // indexed by row, so the format's slot order does not apply
        buf = newVals;
    } else {
        auto order = ValueOrder(csr, cfg);
        buf.assign(order.size(), cf(0.0f, 0.0f));
        for (size_t p = 0; p < order.size(); p++) {
            if (order[p] >= 0) buf[p] = newVals[order[p]];
        }
    }
    auto dNewVals = DeviceBuffer::copyFrom(buf.data(), buf.size() * sizeof(cf));
    aclsparseSpSVUpdate_t updatePart = (cfg.updateMode == "DIAGONAL")
        ? ACL_SPARSE_SPSV_UPDATE_DIAGONAL : ACL_SPARSE_SPSV_UPDATE_GENERAL;
    auto ret = aclsparseSpSV_updateMatrix(handle, spsvDescr.get(), dNewVals.get(), updatePart);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("aclsparseSpSV_updateMatrix (complex64) failed, status="
                                 + std::to_string(ret));
    }
    return dNewVals;
}

// complex64 solve.  Same shape as SpSVNpuWrapper; the structure still comes
// from CsrMatrix and the complex values travel beside it.
inline SpSVNpuResultComplex SpSVNpuWrapperComplex(
    aclsparseHandle_t handle, aclrtStream stream,
    const CsrMatrix& csr,
    const std::vector<std::complex<float>>& values,
    const std::vector<std::complex<float>>& xVec,
    std::complex<float> alpha,
    SpSVNpuConfig cfg)
{
    using cf = std::complex<float>;
    cfg.valueType = ACL_COMPLEX64;
    cfg.complexValues = &values;

    int64_t m = csr.rows;
    SpSVNpuResultComplex result;
    result.y.assign(m, cf(0.0f, 0.0f));
    if (m <= 0) return result;

    auto setup = SetupMatrixDescriptor(csr, cfg);
    aclsparseFillMode_t fillMode = cfg.lower ? ACL_SPARSE_FILL_MODE_LOWER : ACL_SPARSE_FILL_MODE_UPPER;
    aclsparseDiagType_t diagType = cfg.unitDiag ? ACL_SPARSE_DIAG_TYPE_UNIT : ACL_SPARSE_DIAG_TYPE_NON_UNIT;
    setup.matA.setAttribute(ACL_SPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode));
    setup.matA.setAttribute(ACL_SPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType));

    auto dX = DeviceBuffer::copyFrom(xVec.data(), xVec.size() * sizeof(cf));
    DeviceBuffer dYStorage;
    DeviceBuffer* dY = &dX;
    if (!cfg.inPlace) {
        std::vector<cf> yInit(m, cf(0.0f, 0.0f));
        dYStorage = DeviceBuffer::copyFrom(yInit.data(), yInit.size() * sizeof(cf));
        dY = &dYStorage;
    }

    auto vecX = DnVecManager::createConst(m, dX.get(), ACL_COMPLEX64);
    auto vecY = DnVecManager::create(m, dY->get(), ACL_COMPLEX64);
    SpSVDescrManager spsvDescr;

    auto dWorkspace = RunAnalysisT(handle, cfg, alpha, setup.matA, vecX, vecY,
                                   spsvDescr, ACL_COMPLEX64);
    if (cfg.analysisCacheState >= 0) {
        result.analysisLevels = *spsvDescr.get()->analysisNumLevels;
    }
    RunSolveT(handle, stream, cfg, alpha, setup.matA, vecX, vecY, spsvDescr, *dY,
              result.y, ACL_COMPLEX64);

    if (cfg.updateMode == "GENERAL" || cfg.updateMode == "DIAGONAL") {
        result.newValues = MakeComplexUpdateValues(csr, cfg.updateMode);
        DeviceBuffer dNewVals =
            RunUpdateComplex(handle, spsvDescr, csr, cfg, result.newValues);
        if (cfg.inPlace) {
            aclrtMemcpy(dX.get(), xVec.size() * sizeof(cf), xVec.data(),
                        xVec.size() * sizeof(cf), ACL_MEMCPY_HOST_TO_DEVICE);
        }
        result.y2.assign(m, cf(0.0f, 0.0f));
        // dNewVals must outlive this solve -- see RunUpdateComplex.
        RunSolveT(handle, stream, cfg, alpha, setup.matA, vecX, vecY, spsvDescr, *dY,
                  result.y2, ACL_COMPLEX64);
        result.hasSecondSolve = true;
    }
    return result;
}

inline SpSVNpuResult SpSVNpuWrapper(
    aclsparseHandle_t handle, aclrtStream stream,
    const CsrMatrix& csr,
    const std::vector<float>& xVec,
    float alpha,
    const SpSVNpuConfig& cfg)
{
    int64_t m = csr.rows;
    SpSVNpuResult result;
    result.y.resize(m, 0.0f);
    if (m <= 0) return result;

    auto setup = SetupMatrixDescriptor(csr, cfg);
    aclsparseFillMode_t fillMode = cfg.lower ? ACL_SPARSE_FILL_MODE_LOWER : ACL_SPARSE_FILL_MODE_UPPER;
    aclsparseDiagType_t diagType = cfg.unitDiag ? ACL_SPARSE_DIAG_TYPE_UNIT : ACL_SPARSE_DIAG_TYPE_NON_UNIT;
    setup.matA.setAttribute(ACL_SPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode));
    setup.matA.setAttribute(ACL_SPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType));

    auto dX = DeviceBuffer::copyFrom(xVec.data(), xVec.size() * sizeof(float));
    DeviceBuffer dYStorage;
    DeviceBuffer* dY = &dX;
    if (!cfg.inPlace) {
        std::vector<float> yInit(m, 0.0f);
        dYStorage = DeviceBuffer::copyFrom(yInit.data(), yInit.size() * sizeof(float));
        dY = &dYStorage;
    }

    auto vecX = DnVecManager::createConst(m, dX.get(), ACL_FLOAT);
    auto vecY = DnVecManager::create(m, dY->get(), ACL_FLOAT);
    SpSVDescrManager spsvDescr;

    auto dWorkspace = RunAnalysis(handle, cfg, alpha, setup.matA, vecX, vecY, spsvDescr);
    RunSolve(handle, stream, cfg, alpha, setup.matA, vecX, vecY, spsvDescr, *dY, result.y);

    if (cfg.updateMode == "GENERAL" || cfg.updateMode == "DIAGONAL") {
        if (cfg.inPlace) {
            aclrtMemcpy(dX.get(), xVec.size() * sizeof(float), xVec.data(),
                        xVec.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        }
        RunUpdate(handle, stream, cfg, alpha, csr, setup.matA, vecX, vecY, spsvDescr, *dY, result);
    }

    return result;
}

}

#endif
