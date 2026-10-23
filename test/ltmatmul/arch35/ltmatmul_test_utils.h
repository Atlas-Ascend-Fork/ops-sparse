/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_LTMATMUL_TEST_UTILS_H_
#define TEST_LTMATMUL_TEST_UTILS_H_

// =============================================================================
// Foundational test utilities for the ltmatmul test binary.
//
// Provides:
//   - NpuDtypeTrait / MatmulComputeTrait
//   - RAII guards for aclsparseLt descriptors / selections / plans / events
//   - SparseLtMatmulChainParams / SparseLtMatmulChainCtx
//   - Shared helpers: SetAlgAttributes, PreparePhysicalA/B,
//     TransposeBPruned (TransposeToRowMajor is in shared transpose_utils.h)
//   - AlgSetAttributeNpu: A-sparse chain (setAttr -> getAttr -> plan -> prune -> matmul)
//     used by alignment-32 end-to-end exception tests
//
// Consumed by ltmatmul_npu_wrapper.h (which holds the RunMatmulNpu chain).
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparseLt.h"
#include "descriptor_manager.h"
#include "aclsparselt_internal.h"  // fill_tiling_dims, compute_ws_layout, ToInternal
#include "transpose_utils.h"  // shared TransposeToRowMajor
#include "layout_utils.h"      // shared ld/order 布局基建（frame 共享层）
#include "event_guard.h"      // shared EventGuard（prune/ltmatmul 计时共用）

#include "../ltmatmul_golden.h"    // for bf16_bits_t, golden types

namespace sparse_test {

// -----------------------------------------------------------------------------
// NpuDtypeTrait: maps C++ storage types to ACL dtype + element size.
// -----------------------------------------------------------------------------
template <typename T>
struct NpuDtypeTrait;

template <>
struct NpuDtypeTrait<float> {
    static constexpr aclDataType kAclDtype = ACL_FLOAT;
    static constexpr size_t kEltSize = sizeof(float);
};

template <>
struct NpuDtypeTrait<uint16_t> {
    static constexpr aclDataType kAclDtype = ACL_FLOAT16;
    static constexpr size_t kEltSize = sizeof(uint16_t);
};

template <>
struct NpuDtypeTrait<bf16_bits_t> {
    static constexpr aclDataType kAclDtype = ACL_BF16;
    static constexpr size_t kEltSize = sizeof(uint16_t);
};

template <>
struct NpuDtypeTrait<int8_t> {
    static constexpr aclDataType kAclDtype = ACL_INT8;
    static constexpr size_t kEltSize = sizeof(int8_t);
};

template <>
struct NpuDtypeTrait<int32_t> {
    static constexpr aclDataType kAclDtype = ACL_INT32;
    static constexpr size_t kEltSize = sizeof(int32_t);
};

// Compute type trait: FP32/FP16/BF16 -> 32F, INT8 -> 32I.
template <typename T>
struct MatmulComputeTrait {
    static constexpr aclsparseComputeType_t kComputeType = ACL_SPARSE_COMPUTE_32F;
};
template <>
struct MatmulComputeTrait<int8_t> {
    static constexpr aclsparseComputeType_t kComputeType = ACL_SPARSE_COMPUTE_32I;
};

// -----------------------------------------------------------------------------
// BiasDtypeTrait: bias dtype follows C dtype, except INT8 uses FP32
// (cuSPARSELt规定 INT8 路径 bias = FP32).
//   FP32(float)        → bias = FP32
//   FP16(uint16_t)     → bias = FP16
//   BF16(bf16_bits_t)  → bias = BF16
//   INT8(int8_t)       → bias = FP32 (exception per cuSPARSELt)
// -----------------------------------------------------------------------------
template <typename T>
struct BiasDtypeTrait {
    static constexpr aclDataType kBiasAclDtype = NpuDtypeTrait<T>::kAclDtype;
    static constexpr size_t kBiasEltSize = NpuDtypeTrait<T>::kEltSize;
};
template <>
struct BiasDtypeTrait<int8_t> {
    static constexpr aclDataType kBiasAclDtype = ACL_FLOAT;
    static constexpr size_t kBiasEltSize = sizeof(float);
};

// -----------------------------------------------------------------------------
// RAII Guards for aclsparseLt descriptors / selections / plans / events.
// -----------------------------------------------------------------------------

class SparseLtHandleGuard {
public:
    SparseLtHandleGuard() {
        auto s = aclsparseLtInit(&handle_);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseLtInit failed");
        }
    }
    ~SparseLtHandleGuard() { if (handle_) aclsparseLtDestroy(&handle_); }
    SparseLtHandleGuard(const SparseLtHandleGuard&) = delete;
    SparseLtHandleGuard& operator=(const SparseLtHandleGuard&) = delete;
    aclsparseLtConstHandle_t get() const { return &handle_; }
private:
    aclsparseLtHandle_t handle_ = nullptr;
};

class SparseLtMatDescGuard {
public:
    SparseLtMatDescGuard(aclsparseLtConstHandle_t handle,
                         int64_t rows, int64_t cols, int64_t ld,
                         uint32_t alignment, aclDataType valueType,
                         aclsparseOrder_t order,
                         aclsparseLtSparsity_t sparsity) {
        auto s = aclsparseLtStructuredDescriptorInit(handle, &descr_, rows, cols, ld,
                                                     alignment, valueType, order, sparsity);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseLtStructuredDescriptorInit failed");
        }
    }
    ~SparseLtMatDescGuard() { aclsparseLtMatDescriptorDestroy(&descr_); }
    SparseLtMatDescGuard(const SparseLtMatDescGuard&) = delete;
    SparseLtMatDescGuard& operator=(const SparseLtMatDescGuard&) = delete;
    aclsparseLtMatDescriptor_t& get() { return descr_; }
    aclsparseLtMatDescriptor_t get() const { return descr_; }
    aclsparseLtConstMatDescriptor_t* cptr() { return const_cast<aclsparseLtConstMatDescriptor_t*>(&descr_); }
private:
    aclsparseLtMatDescriptor_t descr_ = nullptr;
};

class SparseLtDnMatDescGuard {
public:
    SparseLtDnMatDescGuard(aclsparseLtConstHandle_t handle,
                           int64_t rows, int64_t cols, int64_t ld,
                           uint32_t alignment, aclDataType valueType,
                           aclsparseOrder_t order) {
        auto s = aclsparseLtDenseDescriptorInit(handle, &descr_, rows, cols, ld,
                                                alignment, valueType, order);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseLtDenseDescriptorInit failed");
        }
    }
    ~SparseLtDnMatDescGuard() { aclsparseLtMatDescriptorDestroy(&descr_); }
    SparseLtDnMatDescGuard(const SparseLtDnMatDescGuard&) = delete;
    SparseLtDnMatDescGuard& operator=(const SparseLtDnMatDescGuard&) = delete;
    aclsparseLtMatDescriptor_t& get() { return descr_; }
    aclsparseLtMatDescriptor_t get() const { return descr_; }
    aclsparseLtConstMatDescriptor_t* cptr() { return const_cast<aclsparseLtConstMatDescriptor_t*>(&descr_); }
private:
    aclsparseLtMatDescriptor_t descr_ = nullptr;
};

class SparseLtMatmulDescGuard {
public:
    SparseLtMatmulDescGuard(aclsparseLtConstHandle_t handle,
                            aclsparseOperation_t opA, aclsparseOperation_t opB,
                            aclsparseLtMatDescriptor_t A,
                            aclsparseLtMatDescriptor_t B,
                            aclsparseLtMatDescriptor_t C,
                            aclsparseLtMatDescriptor_t D,
                            aclsparseComputeType_t computeType) {
        auto s = aclsparseLtMatmulDescriptorInit(handle, &descr_, opA, opB, &A, &B, &C, &D, computeType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseLtMatmulDescriptorInit failed");
        }
    }
    ~SparseLtMatmulDescGuard() { aclsparseLtMatmulDescriptorDestroy(&descr_); }
    SparseLtMatmulDescGuard(const SparseLtMatmulDescGuard&) = delete;
    SparseLtMatmulDescGuard& operator=(const SparseLtMatmulDescGuard&) = delete;
    aclsparseLtMatmulDescriptor_t& get() { return descr_; }
    aclsparseLtMatmulDescriptor_t get() const { return descr_; }
    aclsparseLtMatmulDescriptor_t* ptr() { return &descr_; }
    aclsparseLtConstMatmulDescriptor_t* cptr() { return const_cast<aclsparseLtConstMatmulDescriptor_t*>(&descr_); }
private:
    aclsparseLtMatmulDescriptor_t descr_ = nullptr;
};

class SparseLtAlgSelectionGuard {
public:
    SparseLtAlgSelectionGuard(aclsparseLtConstHandle_t handle,
                              aclsparseLtMatmulDescriptor_t matmulDesc,
                              aclsparseLtMatmulAlg_t alg = ACL_SPARSE_LT_MATMUL_ALG_DEFAULT) {
        auto s = aclsparseLtMatmulAlgSelectionInit(handle, &sel_, &matmulDesc, alg);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseLtMatmulAlgSelectionInit failed");
        }
    }
    ~SparseLtAlgSelectionGuard() { aclsparseLtMatmulAlgSelectionDestroy(&sel_); }
    SparseLtAlgSelectionGuard(const SparseLtAlgSelectionGuard&) = delete;
    SparseLtAlgSelectionGuard& operator=(const SparseLtAlgSelectionGuard&) = delete;
    aclsparseLtMatmulAlgSelection_t& get() { return sel_; }
    aclsparseLtMatmulAlgSelection_t get() const { return sel_; }
    aclsparseLtMatmulAlgSelection_t* ptr() { return &sel_; }
    aclsparseLtConstMatmulAlgSelection_t* cptr() { return const_cast<aclsparseLtConstMatmulAlgSelection_t*>(&sel_); }
private:
    aclsparseLtMatmulAlgSelection_t sel_ = nullptr;
};

class SparseLtPlanGuard {
public:
    SparseLtPlanGuard(aclsparseLtConstHandle_t handle,
                      aclsparseLtMatmulDescriptor_t matmulDesc,
                      aclsparseLtMatmulAlgSelection_t algSel) {
        auto s = aclsparseLtMatmulPlanInit(handle, &plan_, &matmulDesc, &algSel);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseLtMatmulPlanInit failed");
        }
    }
    ~SparseLtPlanGuard() { aclsparseLtMatmulPlanDestroy(&plan_); }
    SparseLtPlanGuard(const SparseLtPlanGuard&) = delete;
    SparseLtPlanGuard& operator=(const SparseLtPlanGuard&) = delete;
    aclsparseLtMatmulPlan_t& get() { return plan_; }
    aclsparseLtMatmulPlan_t get() const { return plan_; }
    aclsparseLtConstMatmulPlan_t* cptr() { return const_cast<aclsparseLtConstMatmulPlan_t*>(&plan_); }
private:
    aclsparseLtMatmulPlan_t plan_ = nullptr;
};

// EventGuard 已提取至 test/frame/event_guard.h（prune / ltmatmul /
// matmulSearch 计时共用，消除重复定义；经上方 include 引入）。

// -----------------------------------------------------------------------------
// SparseLtMatmulChainParams / SparseLtMatmulChainCtx (from alg_set_attribute).
// Updated chain params with opB support.
// -----------------------------------------------------------------------------
struct SparseLtMatmulChainParams {
    int64_t m;           // logical m
    int64_t k;           // logical k
    int64_t n;           // logical n
    int64_t aLd;         // matA leading dimension (physical)
    uint32_t alignment;
    aclDataType dtype;
    aclsparseComputeType_t computeType;
    aclsparseOrder_t order;
    aclsparseOperation_t opA;
    aclsparseOperation_t opB;
    void* dAPruned;
    void* dB;
    void* dC;
    void* dD;
};

// Updated chain ctx: computes physical matA/matB dimensions
// from logical m/k/n and opA/opB.
struct SparseLtMatmulChainCtx {
    SparseLtHandleGuard handle;
    SparseLtMatDescGuard matA;
    SparseLtDnMatDescGuard matB;
    SparseLtDnMatDescGuard matC;
    SparseLtDnMatDescGuard matD;
    SparseLtMatmulDescGuard matmulDesc;
    SparseLtAlgSelectionGuard algSel;

    explicit SparseLtMatmulChainCtx(const SparseLtMatmulChainParams& p)
        : handle(),
          matA(handle.get(),
               (p.opA == ACL_SPARSE_OP_TRANSPOSE) ? p.k : p.m,
               (p.opA == ACL_SPARSE_OP_TRANSPOSE) ? p.m : p.k,
               p.aLd, p.alignment, p.dtype,
               p.order, ACL_SPARSE_LT_SPARSITY_50_PERCENT),
          matB(handle.get(),
               (p.opB == ACL_SPARSE_OP_TRANSPOSE) ? p.n : p.k,
               (p.opB == ACL_SPARSE_OP_TRANSPOSE) ? p.k : p.n,
               (p.opB == ACL_SPARSE_OP_TRANSPOSE) ? p.k : p.n,
               p.alignment, p.dtype, ACL_SPARSE_ORDER_ROW),
          matC(handle.get(), p.m, p.n, p.n, p.alignment, p.dtype,
               ACL_SPARSE_ORDER_ROW),
          matD(handle.get(), p.m, p.n, p.n, p.alignment, p.dtype,
               ACL_SPARSE_ORDER_ROW),
          matmulDesc(handle.get(), p.opA, p.opB,
                     matA.get(), matB.get(), matC.get(), matD.get(), p.computeType),
          algSel(handle.get(), matmulDesc.get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT) {}
};

inline bool SetAlgAttributes(aclsparseLtConstHandle_t handle,
                              aclsparseLtMatmulAlgSelection_t algSel,
                              int32_t algConfigId, int32_t splitK,
                              aclsparseStatus_t& cfgRet, aclsparseStatus_t& splitRet)
{
    cfgRet = aclsparseLtMatmulAlgSetAttribute(
        handle, &algSel, ACLSPARSELT_MATMUL_ALG_CONFIG_ID,
        &algConfigId, sizeof(int32_t));
    if (cfgRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] AlgSetAttribute(ALG_CONFIG_ID) failed: " << cfgRet << std::endl;
        return false;
    }
    splitRet = aclsparseLtMatmulAlgSetAttribute(
        handle, &algSel, ACLSPARSELT_MATMUL_SPLIT_K,
        &splitK, sizeof(int32_t));
    if (splitRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] AlgSetAttribute(SPLIT_K) failed: " << splitRet << std::endl;
        return false;
    }
    return true;
}

template <typename T>
inline std::vector<T> PreparePhysicalA(const std::vector<T>& hA,
    int32_t m, int32_t k, bool transA, aclsparseOrder_t order, int64_t& aLd,
    int32_t num_batches = 1, int64_t batch_stride = 0)
{
    if (!transA) {
        // For NON_TRANSPOSE, the prune kernel always reads the
        // matrix as row-major (m, k) with stride ld, regardless of order.
        // The order field only selects the pruning direction (alongRow vs
        // alongCol), not the data layout. Setting aLd=k (the actual row-major
        // stride) ensures correct data access when m != k. Previously aLd=m
        // was used for COL order, which only works for square matrices (m==k).
        aLd = static_cast<int64_t>(k);
        return hA;
    }
    aLd = (order == ACL_SPARSE_ORDER_COL)
        ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    if (order == ACL_SPARSE_ORDER_ROW) {
        if (num_batches <= 1) {
            return TransposeToRowMajor<T>(hA, m, k, aLd);
        }
        // Batch: transpose each batch's (m,k) block separately, preserving
        // the batch-stride layout (padding between batches stays as zeros).
        const size_t matSize = static_cast<size_t>(m) * k;
        const size_t totalElems = static_cast<size_t>(num_batches) * static_cast<size_t>(batch_stride);
        std::vector<T> result(totalElems, T{0});
        for (int32_t b = 0; b < num_batches; ++b) {
            size_t off = static_cast<size_t>(b) * static_cast<size_t>(batch_stride);
            std::vector<T> batchData(hA.begin() + off, hA.begin() + off + matSize);
            auto transposed = TransposeToRowMajor<T>(batchData, m, k, aLd);
            std::copy(transposed.begin(), transposed.end(), result.begin() + off);
        }
        return result;
    }
    return hA;
}

template <typename T>
inline std::vector<T> PreparePhysicalB(const std::vector<T>& hB,
    int32_t k, int32_t n, bool transB,
    int32_t num_batches = 1, int64_t batch_stride = 0)
{
    if (!transB) { return hB; }
    if (num_batches <= 1) {
        return TransposeToRowMajor<T>(hB, k, n, static_cast<int64_t>(k));
    }
    // Batch: transpose each batch's (k,n) block separately.
    const size_t matSize = static_cast<size_t>(k) * n;
    const size_t totalElems = static_cast<size_t>(num_batches) * static_cast<size_t>(batch_stride);
    std::vector<T> result(totalElems, T{0});
    for (int32_t b = 0; b < num_batches; ++b) {
        size_t off = static_cast<size_t>(b) * static_cast<size_t>(batch_stride);
        std::vector<T> batchData(hB.begin() + off, hB.begin() + off + matSize);
        auto transposed = TransposeToRowMajor<T>(batchData, k, n, static_cast<int64_t>(k));
        std::copy(transposed.begin(), transposed.end(), result.begin() + off);
    }
    return result;
}

// ld/order 布局基建已提取至 test/frame/layout_utils.h
// （PackPhysicalPadded / UnpackPhysicalToLogical / MatmulLayoutOpts / CdLdOf /
//  CdSpanElems / PackBatchStridedPhysical / ExtractCdLogicalBatch）。

// ComputeErrorMetrics: golden 与 NPU 输出（float 域）的交叉核对统计
// maxAbsErr / maxRelErr（ltmatmul_test.cpp 与 matmulSearch_test.cpp 共用，
// matmulSearch 侧经 include 链复用同一实体）。
inline void ComputeErrorMetrics(const std::vector<float>& goldenF,
                                const std::vector<float>& npuF,
                                double& maxAbsErr, double& maxRelErr)
{
    maxAbsErr = 0.0;
    maxRelErr = 0.0;
    for (size_t i = 0; i < goldenF.size() && i < npuF.size(); ++i) {
        double diff = std::fabs(static_cast<double>(goldenF[i]) - static_cast<double>(npuF[i]));
        if (diff > maxAbsErr) { maxAbsErr = diff; }
        double denom = std::fabs(static_cast<double>(goldenF[i]));
        if (denom > 1e-12) {
            double rel = diff / denom;
            if (rel > maxRelErr) { maxRelErr = rel; }
        }
    }
}

// InitChainCtxBase: 链 ctx 基础字段（batch 属性 + 布局 + C/D span）赋值——
// A-sparse（PrepareMatmulChainContext）与 B-sparse（PrepareBSparseChainContext）
// 两个链设置的共享段（模板 CtxT 覆盖 MatmulChainCtx / MatmulBSparseCtx）。
template <typename CtxT>
inline void InitChainCtxBase(CtxT& ctx, int32_t num_batches, int64_t batch_stride,
    const MatmulLayoutOpts& opts, int32_t m, int32_t n)
{
    ctx.numBatches = num_batches;
    ctx.batchStride = batch_stride;
    ctx.layout = opts;
    ctx.cdSpan = CdSpanElems(m, n, opts);
}

// [Int8CD 共享段] 离散小整数 C 数据生成（[-5,5] 域经 CDT 存储，
// FP16/BF16 精确可表）——ltmatmul PrepareInt8CDCaseInputs 与 matmulSearch
// PrepareInt8CDSearchInputs 共用。
template <typename CDT>
inline void GenInt8CDInputC(int32_t m, int32_t n, std::vector<CDT>& hC)
{
    std::vector<float> hCf(static_cast<size_t>(m) * n);
    for (size_t i = 0; i < hCf.size(); ++i) {
        hCf[i] = static_cast<float>((static_cast<int64_t>(i * 7 + 3) % 11) - 5);
    }
    hC.resize(hCf.size());
    TruncateToOutput<CDT>(hCf, hC, hCf.size());
}

// GenInt8CDInputC 的 batch-strided 版（Int8CD 链 batch 透传）：
// 每 batch 生成一份确定性离散小整数 pattern（batch 偏移区分数据），
// 按 batchStride 铺设（段间 padding 置零）；num_batches<=1 退化为单 batch 版。
template <typename CDT>
inline void GenInt8CDInputCBatch(int32_t m, int32_t n, int32_t numBatches,
    int64_t batchStride, std::vector<CDT>& hC)
{
    if (numBatches <= 1) {
        GenInt8CDInputC<CDT>(m, n, hC);
        return;
    }
    const size_t mn = static_cast<size_t>(m) * n;
    const size_t total = static_cast<size_t>(numBatches) * static_cast<size_t>(batchStride);
    hC.assign(total, CDT{0});
    std::vector<float> hCf(mn);
    std::vector<CDT> hCb;
    for (int32_t b = 0; b < numBatches; ++b) {
        for (size_t i = 0; i < mn; ++i) {
            hCf[i] = static_cast<float>((static_cast<int64_t>(i * 7 + 3 + b) % 11) - 5);
        }
        TruncateToOutput<CDT>(hCf, hCb, mn);
        std::copy(hCb.begin(), hCb.end(),
                  hC.begin() + static_cast<size_t>(b) * static_cast<size_t>(batchStride));
    }
}

// [Int8CD 共享段] golden 三重循环核心：acc = Σ A_pruned·B（A-sparse）/
// Σ A·B_pruned（B-sparse；int32 精确累加）→ D[i,j] = α[i]·acc + β[i]·C[i,j]
// （C 经 CDT 提升；α/β 向量或标量由 useVec 分派）——ltmatmul
// ComputeInt8CDCaseGolden 与 matmulSearch ComputeSearchGoldenInt8CD 共用。
// pruned 传参：A-sparse 传 A_pruned
// 前 m*k 元素；B-sparse 传 B_pruned 全量（(k,n) 行主序，剪枝已转置完成）。
// epilogue（bias/act）与输出量化（cast<CDT>）由调用方按各自参数编排。
template <typename CDT>
inline std::vector<float> ComputeInt8CDGoldenCore(int32_t m, int32_t k, int32_t n,
    bool isSparseA, bool useVec, float alphaScal, float betaScal,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec,
    const std::vector<int8_t>& hA, const std::vector<int8_t>& hB,
    const std::vector<int8_t>& pruned, const std::vector<CDT>& hC)
{
    std::vector<float> Df(static_cast<size_t>(m) * n);
    for (int32_t i = 0; i < m; ++i) {
        const float aV = useVec ? alphaVec[static_cast<size_t>(i)] : alphaScal;
        const float bV = useVec ? betaVec[static_cast<size_t>(i)] : betaScal;
        for (int32_t j = 0; j < n; ++j) {
            int32_t acc = 0;
            if (isSparseA) {
                for (int32_t kk = 0; kk < k; ++kk) {
                    acc += static_cast<int32_t>(pruned[static_cast<size_t>(i) * k + kk])
                         * static_cast<int32_t>(hB[static_cast<size_t>(kk) * n + j]);
                }
            } else {
                for (int32_t kk = 0; kk < k; ++kk) {
                    acc += static_cast<int32_t>(hA[static_cast<size_t>(i) * k + kk])
                         * static_cast<int32_t>(pruned[static_cast<size_t>(kk) * n + j]);
                }
            }
            const std::vector<CDT> cOne{hC[static_cast<size_t>(i) * n + j]};
            const float cVal = MatmulToFloat(cOne)[0];
            Df[static_cast<size_t>(i) * n + j] = aV * static_cast<float>(acc) + bV * cVal;
        }
    }
    return Df;
}

}  // namespace sparse_test


// AlgSetAttributeNpu chain moved to ltmatmul_alg_npu.h
// to reduce header file size. Include it here for backward compat.
#include "ltmatmul_alg_npu.h"

#endif  // TEST_LTMATMUL_TEST_UTILS_H_
