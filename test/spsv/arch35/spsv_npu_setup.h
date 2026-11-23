/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPSV_NPU_SETUP_H_
#define TEST_SPSV_NPU_SETUP_H_

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <complex>
#include <vector>

#include "descriptor_manager.h"
#include "fill.h"
#include "sparse_test.h"

namespace sparse_test {

struct SpSVNpuResult {
    std::vector<float> y;
    std::vector<float> y2;
    bool hasSecondSolve = false;
};

struct SpSVNpuConfig {
    std::string format = "CSR";
    bool isI64 = false;
    bool isRowPtrI64 = false;  // effective rowPtr (or colOffset for CSC) index width
    bool isColIndI64 = false;  // effective colInd (or rowInd for CSC/COO) index width
    int32_t idxBase = 0;  // 0=INDEX_BASE_ZERO, 1=INDEX_BASE_ONE
    bool lower = true;
    bool unitDiag = false;
    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE;
    bool inPlace = false;
    bool nullVec = false;
    // -1: normal event query; 0/1: deterministic unavailable/ready cache tests.
    int32_t analysisCacheState = -1;
    std::string updateMode = "NONE";
    int32_t sliceWidth = 1;
    // ACL_FLOAT, or ACL_COMPLEX64 with complexValues supplying the nnz values.
    aclDataType valueType = ACL_FLOAT;
    const std::vector<std::complex<float>>* complexValues = nullptr;
};

// Where each slot of a format's value array comes from: the index of the
// source nonzero, or -1 for SLICED_ELL padding.  Derived by running the same
// conversion the descriptor setup runs, over a matrix whose values are
// 0..nnz-1, so the two orderings cannot drift apart.
static inline std::vector<int32_t> ValueOrder(const CsrMatrix& csr, const SpSVNpuConfig& cfg) {
    int64_t m = csr.rows;
    std::vector<int32_t> order;
    if (cfg.format == "CSR" || cfg.format == "COO") {
        order.resize(csr.nnz);
        for (int64_t k = 0; k < csr.nnz; k++) order[k] = static_cast<int32_t>(k);
        return order;
    }
    if (cfg.format == "CSC") {
        std::vector<int32_t> colOff(m + 1, 0);
        for (int32_t c : csr.colIndices) colOff[c + 1]++;
        for (int64_t c = 0; c < m; c++) colOff[c + 1] += colOff[c];
        std::vector<int32_t> counter(colOff.begin(), colOff.end() - 1);
        order.assign(csr.nnz, -1);
        for (int64_t i = 0; i < m; i++) {
            for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
                order[counter[csr.colIndices[k]]++] = k;
            }
        }
        return order;
    }
    // SLICED_ELL: run the frame's own conversion over 0..nnz-1.
    CsrMatrix probe = csr;
    for (int64_t k = 0; k < csr.nnz; k++) probe.values[k] = static_cast<float>(k);
    SlicedEllMatrix ell = csrToSlicedEll(probe, cfg.sliceWidth);
    order.resize(ell.values.size());
    for (size_t p = 0; p < ell.values.size(); p++) {
        order[p] = (ell.colIndices[p] >= 0) ? static_cast<int32_t>(ell.values[p] + 0.5f) : -1;
    }
    return order;
}

// The device value buffer for a format, in that format's slot order.
static inline DeviceBuffer UploadValues(const CsrMatrix& csr, const SpSVNpuConfig& cfg,
                                        const std::vector<int32_t>& order) {
    if (cfg.valueType == ACL_COMPLEX64) {
        const auto& src = *cfg.complexValues;
        std::vector<std::complex<float>> buf(order.size(), std::complex<float>(0.0f, 0.0f));
        for (size_t p = 0; p < order.size(); p++) {
            if (order[p] >= 0) buf[p] = src[order[p]];
        }
        return DeviceBuffer::copyFrom(buf.data(), buf.size() * sizeof(std::complex<float>));
    }
    std::vector<float> buf(order.size(), 0.0f);
    for (size_t p = 0; p < order.size(); p++) {
        if (order[p] >= 0) buf[p] = csr.values[order[p]];
    }
    return DeviceBuffer::copyFrom(buf.data(), buf.size() * sizeof(float));
}

struct SpSVDeviceSetup {
    DeviceBuffer dVals;
    DeviceBuffer dIdx0;
    DeviceBuffer dIdx1;
    SpMatManager matA;
};

static inline SpSVDeviceSetup SetupCsrDescriptor(const CsrMatrix& csr, int64_t m,
                                                   aclsparseIndexType_t rowPtrType,
                                                   aclsparseIndexType_t colIndType,
                                                   aclsparseIndexBase_t idxBase,
                                                   DeviceBuffer&& dVals, aclDataType valueType) {
    SpSVDeviceSetup setup;
    // Row offsets: independently sized by rowPtrType
    if (rowPtrType == ACL_SPARSE_INDEX_64I) {
        std::vector<int64_t> rowOff64(csr.rowOffsets.begin(), csr.rowOffsets.end());
        if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
            for (auto& v : rowOff64) v++;
        }
        setup.dIdx0 = DeviceBuffer::copyFrom(rowOff64.data(), rowOff64.size() * sizeof(int64_t));
    } else {
        std::vector<int32_t> rowOff(csr.rowOffsets);
        if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
            for (auto& v : rowOff) v++;
        }
        setup.dIdx0 = DeviceBuffer::copyFrom(rowOff.data(), rowOff.size() * sizeof(int32_t));
    }
    // Column indices: independently sized by colIndType
    if (colIndType == ACL_SPARSE_INDEX_64I) {
        std::vector<int64_t> colIdx64(csr.colIndices.begin(), csr.colIndices.end());
        if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
            for (auto& v : colIdx64) v++;
        }
        setup.dIdx1 = DeviceBuffer::copyFrom(colIdx64.data(), colIdx64.size() * sizeof(int64_t));
    } else {
        std::vector<int32_t> colIdx(csr.colIndices);
        if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
            for (auto& v : colIdx) v++;
        }
        setup.dIdx1 = DeviceBuffer::copyFrom(colIdx.data(), colIdx.size() * sizeof(int32_t));
    }
    setup.dVals = std::move(dVals);
    setup.matA = SpMatManager::createCsr(m, m, csr.nnz, setup.dIdx0.get(), setup.dIdx1.get(),
                                           setup.dVals.get(), rowPtrType, colIndType,
                                           idxBase, valueType);
    return setup;
}

static inline SpSVDeviceSetup SetupCscDescriptor(const CsrMatrix& csr, int64_t m,
                                                   aclsparseIndexType_t colOffType,
                                                   aclsparseIndexType_t rowIndType,
                                                   aclsparseIndexBase_t idxBase,
                                                   DeviceBuffer&& dVals, aclDataType valueType) {
    CscMatrix csc;
    csc.rows = m; csc.cols = m; csc.nnz = csr.nnz;
    csc.colOffsets.assign(m + 1, 0);
    csc.rowIndices.resize(csr.nnz);
    csc.values.resize(csr.nnz);
    for (int32_t c : csr.colIndices) csc.colOffsets[c + 1]++;
    for (int64_t c = 0; c < m; c++) csc.colOffsets[c + 1] += csc.colOffsets[c];
    std::vector<int32_t> counter(csc.colOffsets.begin(), csc.colOffsets.end() - 1);
    for (int64_t i = 0; i < m; i++) {
        for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
            int32_t c = csr.colIndices[k];
            int32_t dest = counter[c]++;
            csc.rowIndices[dest] = static_cast<int32_t>(i);
            csc.values[dest] = csr.values[k];
        }
    }
    if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
        for (auto& v : csc.colOffsets) v++;
        for (auto& v : csc.rowIndices) v++;
    }
    SpSVDeviceSetup setup;
    // Column offsets: independently sized by colOffType
    if (colOffType == ACL_SPARSE_INDEX_64I) {
        std::vector<int64_t> colOff64(csc.colOffsets.begin(), csc.colOffsets.end());
        setup.dIdx0 = DeviceBuffer::copyFrom(colOff64.data(), colOff64.size() * sizeof(int64_t));
    } else {
        setup.dIdx0 = DeviceBuffer::copyFrom(csc.colOffsets.data(), csc.colOffsets.size() * sizeof(int32_t));
    }
    // Row indices: independently sized by rowIndType
    if (rowIndType == ACL_SPARSE_INDEX_64I) {
        std::vector<int64_t> rowIdx64(csc.rowIndices.begin(), csc.rowIndices.end());
        setup.dIdx1 = DeviceBuffer::copyFrom(rowIdx64.data(), rowIdx64.size() * sizeof(int64_t));
    } else {
        setup.dIdx1 = DeviceBuffer::copyFrom(csc.rowIndices.data(), csc.rowIndices.size() * sizeof(int32_t));
    }
    setup.dVals = std::move(dVals);
    setup.matA = SpMatManager::createCsc(m, m, csc.nnz, setup.dIdx0.get(), setup.dIdx1.get(),
                                           setup.dVals.get(), colOffType, rowIndType,
                                           idxBase, valueType);
    return setup;
}

static inline SpSVDeviceSetup SetupCooDescriptor(const CsrMatrix& csr, int64_t m,
                                                   aclsparseIndexType_t idxType,
                                                   aclsparseIndexBase_t idxBase,
                                                   DeviceBuffer&& dVals, aclDataType valueType) {
    CooMatrix coo;
    coo.rows = m; coo.cols = m; coo.nnz = csr.nnz;
    coo.rowIndices.reserve(csr.nnz);
    coo.colIndices.reserve(csr.nnz);
    coo.values.reserve(csr.nnz);
    for (int64_t i = 0; i < m; i++) {
        for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
            coo.rowIndices.push_back(static_cast<int32_t>(i));
            coo.colIndices.push_back(csr.colIndices[k]);
            coo.values.push_back(csr.values[k]);
        }
    }
    if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
        for (auto& v : coo.rowIndices) v++;
        for (auto& v : coo.colIndices) v++;
    }
    SpSVDeviceSetup setup;
    if (idxType == ACL_SPARSE_INDEX_64I) {
        std::vector<int64_t> rowIdx64(coo.rowIndices.begin(), coo.rowIndices.end());
        std::vector<int64_t> colIdx64(coo.colIndices.begin(), coo.colIndices.end());
        setup.dIdx0 = DeviceBuffer::copyFrom(rowIdx64.data(), rowIdx64.size() * sizeof(int64_t));
        setup.dIdx1 = DeviceBuffer::copyFrom(colIdx64.data(), colIdx64.size() * sizeof(int64_t));
    } else {
        setup.dIdx0 = DeviceBuffer::copyFrom(coo.rowIndices.data(), coo.rowIndices.size() * sizeof(int32_t));
        setup.dIdx1 = DeviceBuffer::copyFrom(coo.colIndices.data(), coo.colIndices.size() * sizeof(int32_t));
    }
    setup.dVals = std::move(dVals);
    setup.matA = SpMatManager::createCoo(m, m, coo.nnz, setup.dIdx0.get(), setup.dIdx1.get(),
                                           setup.dVals.get(), idxType,
                                           idxBase, valueType);
    return setup;
}

static inline SpSVDeviceSetup SetupSlicedEllDescriptor(const CsrMatrix& csr, int64_t m,
                                                       aclsparseIndexType_t idxType,
                                                       aclsparseIndexBase_t idxBase,
                                                       DeviceBuffer&& dVals, aclDataType valueType,
                                                       int32_t sliceWidth = 1) {
    SlicedEllMatrix ell = csrToSlicedEll(csr, sliceWidth);
    if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
        for (auto& v : ell.sliceOffsets) v++;
        for (auto& v : ell.colIndices) {
            if (v >= 0) v++;
        }
    }
    SpSVDeviceSetup setup;
    if (idxType == ACL_SPARSE_INDEX_64I) {
        std::vector<int64_t> sliceOff64(ell.sliceOffsets.begin(), ell.sliceOffsets.end());
        std::vector<int64_t> colIdx64(ell.colIndices.begin(), ell.colIndices.end());
        setup.dIdx0 = DeviceBuffer::copyFrom(sliceOff64.data(), sliceOff64.size() * sizeof(int64_t));
        setup.dIdx1 = DeviceBuffer::copyFrom(colIdx64.data(), colIdx64.size() * sizeof(int64_t));
    } else {
        setup.dIdx0 = DeviceBuffer::copyFrom(ell.sliceOffsets.data(), ell.sliceOffsets.size() * sizeof(int32_t));
        setup.dIdx1 = DeviceBuffer::copyFrom(ell.colIndices.data(), ell.colIndices.size() * sizeof(int32_t));
    }
    setup.dVals = std::move(dVals);
    setup.matA = SpMatManager::createSlicedEll(m, m, ell.nnz, setup.dIdx0.get(), setup.dIdx1.get(),
                                                setup.dVals.get(),
                                                static_cast<int64_t>(ell.sliceWidth),
                                                static_cast<int64_t>(ell.sliceOffsets.size()) - 1,
                                                idxType,
                                                idxBase, valueType);
    return setup;
}

static inline SpSVDeviceSetup SetupMatrixDescriptor(const CsrMatrix& csr, const SpSVNpuConfig& cfg) {
    int64_t m = csr.rows;
    aclsparseIndexType_t rowPtrType = cfg.isRowPtrI64 ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
    aclsparseIndexType_t colIndType = cfg.isColIndI64 ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
    aclsparseIndexBase_t idxBase = (cfg.idxBase == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;
    auto order = ValueOrder(csr, cfg);
    auto dVals = UploadValues(csr, cfg, order);
    if (cfg.format == "CSR") {
        return SetupCsrDescriptor(csr, m, rowPtrType, colIndType, idxBase,
                                  std::move(dVals), cfg.valueType);
    }
    if (cfg.format == "CSC") {
        return SetupCscDescriptor(csr, m, rowPtrType, colIndType, idxBase,
                                  std::move(dVals), cfg.valueType);
    }
    // COO and SLICED_ELL use a single index type for both arrays; map via colIndType
    if (cfg.format == "COO") {
        return SetupCooDescriptor(csr, m, colIndType, idxBase, std::move(dVals), cfg.valueType);
    }
    if (cfg.format == "SLICED_ELL") {
        return SetupSlicedEllDescriptor(csr, m, colIndType, idxBase,
                                        std::move(dVals), cfg.valueType, cfg.sliceWidth);
    }
    throw std::runtime_error("Unsupported format: " + cfg.format);
}

}  // namespace sparse_test

#endif  // TEST_SPSV_NPU_SETUP_H_
