/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file aclsparselt_host.cpp
 * \brief sparseLt shared host-side API implementation.
 * Implements the host-only APIs consumed by the alg_set_attribute /
 * alg_get_attribute / matmul / prune host TUs: GetWorkspaceSize,
 * DescSetAttribute, DescGetAttribute, Version, Property. The host-only
 * helpers (dtype_from_acl / align_up / CubeTiling / WsLayout /
 * push_tiling_to_device / fill_tiling_dims ... live in
 * shared/aclsparselt_internal.h as inline host-only functions).
 * aclsparseLtMatmul (the kernel-launching entry) lives in
 * sparseLt/matmul/arch35/matmul_host.cpp.
 */

// Include order: C standard → C++ → CANN → local.
#include <cstdint>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "log/log.h"

#ifndef __CCE_AICORE__
#include "tiling/platform/platform_ascendc.h"
#endif

#include "cann_ops_sparseLt.h"
#include "aclsparselt_handle_internal.h"
#include "shared/aclsparselt_internal.h"

// ============================================================================
// Per-attribute SetAttribute helpers (extracted to reduce CCN of the
// aclsparseLtMatmulDescSetAttribute switch from 31 to ~10).
// Each helper performs size validation + value assignment for one attribute.
// ============================================================================

static aclsparseStatus_t SetAlphaVectorScaling(aclsparseLtMatmulDescriptor* md,
                                                 const void* data, size_t dataSize)
{
    if (dataSize != sizeof(int)) {
        OP_LOGE(kSparseLtLogTag, "ALPHA_VECTOR_SCALING: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    bool alphaVec = (*static_cast<const int*>(data) != 0);
    // 关闭 ALPHA_VECTOR_SCALING 时联动关闭
    // BETA_VECTOR_SCALING：消除"α 标量 + β 向量"的混合态——NVIDIA 契约
    // 仅定义 "BETA implies ALPHA" 单向语义（开 β 隐含开 α），关闭行为未
    // 文档化；该混合态属未定义域，显式联动关闭以消除（此前"先开 β 再
    // 显式关 α"可构造该态，全仓既有用例零依赖该组合——grep 实证）。
    // 开启 α 不影响 β（维持既有语义）。
    bool betaVec = alphaVec && decode_beta_scaling(md);
    encode_scaling_flags(md, alphaVec, betaVec);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SetBetaVectorScaling(aclsparseLtMatmulDescriptor* md,
                                               const void* data, size_t dataSize)
{
    if (dataSize != sizeof(int)) {
        OP_LOGE(kSparseLtLogTag, "BETA_VECTOR_SCALING: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    bool betaVec = (*static_cast<const int*>(data) != 0);
    // Setting betaVec=true implies alphaVec=true (cuSPARSELt semantics).
    bool alphaVec = betaVec || decode_alpha_scaling(md);
    encode_scaling_flags(md, alphaVec, betaVec);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SetBiasPointer(aclsparseLtMatmulDescriptor* md,
                                         const void* data, size_t dataSize)
{
    if (dataSize != sizeof(void*)) {
        OP_LOGE(kSparseLtLogTag, "BIAS_POINTER: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    md->biasPointer = *static_cast<void* const*>(data);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SetBiasStride(aclsparseLtMatmulDescriptor* md,
                                        const void* data, size_t dataSize)
{
    if (dataSize != sizeof(int64_t)) {
        OP_LOGE(kSparseLtLogTag, "BIAS_STRIDE: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    md->biasStride = *static_cast<const int64_t*>(data);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SetActivationRelu(aclsparseLtMatmulDescriptor* md,
                                            const void* data, size_t dataSize)
{
    if (dataSize != sizeof(int)) {
        OP_LOGE(kSparseLtLogTag, "ACTIVATION_RELU: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int reluVal = *static_cast<const int*>(data);
    // ReLU/GeLU 互斥校验：启用 ReLU 时若 GeLU 已启用则拒绝（设为 0=关闭时不检查）。
    if (reluVal != 0 && md->activationGelu != 0) {
        OP_LOGE(kSparseLtLogTag, "ACTIVATION_RELU: GeLU already enabled, mutual exclusion violation");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    md->activationRelu = reluVal;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SetReluUpperBound(aclsparseLtMatmulDescriptor* md,
                                            const void* data, size_t dataSize)
{
    if (dataSize != sizeof(float)) {
        OP_LOGE(kSparseLtLogTag, "ACTIVATION_RELU_UPPERBOUND: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    md->reluUpperBound = *static_cast<const float*>(data);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SetReluThreshold(aclsparseLtMatmulDescriptor* md,
                                           const void* data, size_t dataSize)
{
    if (dataSize != sizeof(float)) {
        OP_LOGE(kSparseLtLogTag, "ACTIVATION_RELU_THRESHOLD: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    md->reluThreshold = *static_cast<const float*>(data);
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SetActivationGelu(aclsparseLtMatmulDescriptor* md,
                                            const void* data, size_t dataSize)
{
    if (dataSize != sizeof(int)) {
        OP_LOGE(kSparseLtLogTag, "ACTIVATION_GELU: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int geluVal = *static_cast<const int*>(data);
    // ReLU/GeLU 互斥校验：启用 GeLU 时若 ReLU 已启用则拒绝（设为 0=关闭时不检查）。
    if (geluVal != 0 && md->activationRelu != 0) {
        OP_LOGE(kSparseLtLogTag, "ACTIVATION_GELU: ReLU already enabled, mutual exclusion violation");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    md->activationGelu = geluVal;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SetGeluScaling(aclsparseLtMatmulDescriptor* md,
                                         const void* data, size_t dataSize)
{
    if (dataSize != sizeof(float)) {
        OP_LOGE(kSparseLtLogTag, "ACTIVATION_GELU_SCALING: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    md->geluScaling = *static_cast<const float*>(data);
    // cuSPARSELt 语义：设置 GELU_SCALING 隐含启用 GeLU（若 ReLU 未启用）。
    // 若 ReLU 已启用，保持互斥语义（ReLU 优先，GELU_SCALING 仅存储值不启用 GeLU）。
    if (md->activationRelu == 0) {
        md->activationGelu = 1;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Per-attribute GetAttribute helpers (extracted to reduce CCN of the
// aclsparseLtMatmulDescGetAttribute switch from 28 to ~10).
// ============================================================================

static aclsparseStatus_t GetAlphaVectorScaling(const aclsparseLtMatmulDescriptor* md,
                                                void* data, size_t dataSize)
{
    if (dataSize != sizeof(int)) {
        OP_LOGE(kSparseLtLogTag, "ALPHA_VECTOR_SCALING: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<int*>(data) = decode_alpha_scaling(md) ? 1 : 0;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t GetBetaVectorScaling(const aclsparseLtMatmulDescriptor* md,
                                               void* data, size_t dataSize)
{
    if (dataSize != sizeof(int)) {
        OP_LOGE(kSparseLtLogTag, "BETA_VECTOR_SCALING: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<int*>(data) = decode_beta_scaling(md) ? 1 : 0;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t GetBiasPointer(const aclsparseLtMatmulDescriptor* md,
                                         void* data, size_t dataSize)
{
    if (dataSize != sizeof(void*)) {
        OP_LOGE(kSparseLtLogTag, "BIAS_POINTER: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<void**>(data) = md->biasPointer;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t GetBiasStride(const aclsparseLtMatmulDescriptor* md,
                                        void* data, size_t dataSize)
{
    if (dataSize != sizeof(int64_t)) {
        OP_LOGE(kSparseLtLogTag, "BIAS_STRIDE: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<int64_t*>(data) = md->biasStride;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t GetActivationRelu(const aclsparseLtMatmulDescriptor* md,
                                            void* data, size_t dataSize)
{
    if (dataSize != sizeof(int)) {
        OP_LOGE(kSparseLtLogTag, "ACTIVATION_RELU: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<int*>(data) = md->activationRelu;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t GetReluUpperBound(const aclsparseLtMatmulDescriptor* md,
                                            void* data, size_t dataSize)
{
    if (dataSize != sizeof(float)) {
        OP_LOGE(kSparseLtLogTag, "ACTIVATION_RELU_UPPERBOUND: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<float*>(data) = md->reluUpperBound;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t GetReluThreshold(const aclsparseLtMatmulDescriptor* md,
                                           void* data, size_t dataSize)
{
    if (dataSize != sizeof(float)) {
        OP_LOGE(kSparseLtLogTag, "ACTIVATION_RELU_THRESHOLD: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<float*>(data) = md->reluThreshold;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t GetActivationGelu(const aclsparseLtMatmulDescriptor* md,
                                            void* data, size_t dataSize)
{
    if (dataSize != sizeof(int)) {
        OP_LOGE(kSparseLtLogTag, "ACTIVATION_GELU: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<int*>(data) = md->activationGelu;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t GetGeluScaling(const aclsparseLtMatmulDescriptor* md,
                                         void* data, size_t dataSize)
{
    if (dataSize != sizeof(float)) {
        OP_LOGE(kSparseLtLogTag, "ACTIVATION_GELU_SCALING: size mismatch, got %zu", dataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<float*>(data) = md->geluScaling;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseLtMatmulDescSetAttribute(
    const aclsparseLtHandle_t* handle,
    aclsparseLtMatmulDescriptor_t* matmulDescr,
    aclsparseLtMatmulDescAttribute_t matmulAttribute,
    const void* data,
    size_t dataSize)
{
    if (handle == nullptr || *handle == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatmulDescSetAttribute: handle is null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matmulDescr == nullptr || *matmulDescr == nullptr || data == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatmulDescSetAttribute: matmulDescr/data is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* md = *matmulDescr;
    if (md->matD == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatmulDescSetAttribute: matmulDesc not fully initialized (matD is null)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    switch (matmulAttribute) {
        case ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING:
            return SetAlphaVectorScaling(md, data, dataSize);
        case ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING:
            return SetBetaVectorScaling(md, data, dataSize);
        case ACLSPARSELT_MATMUL_BIAS_POINTER:
            return SetBiasPointer(md, data, dataSize);
        case ACLSPARSELT_MATMUL_BIAS_STRIDE:
            return SetBiasStride(md, data, dataSize);
        case ACLSPARSELT_MATMUL_ACTIVATION_RELU:
            return SetActivationRelu(md, data, dataSize);
        case ACLSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND:
            return SetReluUpperBound(md, data, dataSize);
        case ACLSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD:
            return SetReluThreshold(md, data, dataSize);
        case ACLSPARSELT_MATMUL_ACTIVATION_GELU:
            return SetActivationGelu(md, data, dataSize);
        case ACLSPARSELT_MATMUL_ACTIVATION_GELU_SCALING:
            return SetGeluScaling(md, data, dataSize);
        default:
            OP_LOGE(kSparseLtLogTag, "MatmulDescSetAttribute: unsupported attr %d",
                    static_cast<int32_t>(matmulAttribute));
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
}

extern "C" aclsparseStatus_t aclsparseLtMatmulDescGetAttribute(
    const aclsparseLtHandle_t* handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    aclsparseLtMatmulDescAttribute_t matmulAttribute,
    void* data,
    size_t dataSize)
{
    if (handle == nullptr || *handle == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatmulDescGetAttribute: handle is null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matmulDescr == nullptr || *matmulDescr == nullptr || data == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatmulDescGetAttribute: matmulDescr/data is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* md = to_matmul_internal(*matmulDescr);
    if (md->matD == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatmulDescGetAttribute: matmulDesc not fully initialized (matD is null)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    switch (matmulAttribute) {
        case ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING:
            return GetAlphaVectorScaling(md, data, dataSize);
        case ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING:
            return GetBetaVectorScaling(md, data, dataSize);
        case ACLSPARSELT_MATMUL_BIAS_POINTER:
            return GetBiasPointer(md, data, dataSize);
        case ACLSPARSELT_MATMUL_BIAS_STRIDE:
            return GetBiasStride(md, data, dataSize);
        case ACLSPARSELT_MATMUL_ACTIVATION_RELU:
            return GetActivationRelu(md, data, dataSize);
        case ACLSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND:
            return GetReluUpperBound(md, data, dataSize);
        case ACLSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD:
            return GetReluThreshold(md, data, dataSize);
        case ACLSPARSELT_MATMUL_ACTIVATION_GELU:
            return GetActivationGelu(md, data, dataSize);
        case ACLSPARSELT_MATMUL_ACTIVATION_GELU_SCALING:
            return GetGeluScaling(md, data, dataSize);
        default:
            OP_LOGE(kSparseLtLogTag, "MatmulDescGetAttribute: unsupported attr %d",
                    static_cast<int32_t>(matmulAttribute));
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
}

// ============================================================================
// MatDescSetAttribute / MatDescGetAttribute — batch 属性设置/查询
// 对单个矩阵描述符操作，不做跨矩阵一致性校验（一致性在
// aclsparseLtMatmulDescriptorInit 中校验 A/B/C/D numBatches 一致）。
// ============================================================================
extern "C" aclsparseStatus_t aclsparseLtMatDescSetAttribute(
    const aclsparseLtHandle_t* handle,
    aclsparseLtMatDescriptor_t* matDescr,
    aclsparseLtMatDescAttribute_t matAttribute,
    const void* data,
    size_t dataSize)
{
    if (handle == nullptr || *handle == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatDescSetAttribute: handle is null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matDescr == nullptr || *matDescr == nullptr || data == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatDescSetAttribute: matDescr/data is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* md = ToInternal(*matDescr);
    switch (matAttribute) {
        case ACLSPARSELT_MAT_NUM_BATCHES:
            if (dataSize != sizeof(int32_t)) {
                OP_LOGE(kSparseLtLogTag, "MAT_NUM_BATCHES: size mismatch, got %zu", dataSize);
                return ACL_SPARSE_STATUS_INVALID_VALUE;
            }
            {
                int32_t val = *static_cast<const int32_t*>(data);
                if (val < 1) {
                    OP_LOGE(kSparseLtLogTag, "MAT_NUM_BATCHES: value %d < 1", val);
                    return ACL_SPARSE_STATUS_INVALID_VALUE;
                }
                md->numBatches = val;
            }
            return ACL_SPARSE_STATUS_SUCCESS;
        case ACLSPARSELT_MAT_BATCH_STRIDE:
            if (dataSize != sizeof(int64_t)) {
                OP_LOGE(kSparseLtLogTag, "MAT_BATCH_STRIDE: size mismatch, got %zu", dataSize);
                return ACL_SPARSE_STATUS_INVALID_VALUE;
            }
            md->batchStride = *static_cast<const int64_t*>(data);
            return ACL_SPARSE_STATUS_SUCCESS;
        default:
            OP_LOGE(kSparseLtLogTag, "MatDescSetAttribute: unsupported attr %d",
                    static_cast<int32_t>(matAttribute));
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
}

extern "C" aclsparseStatus_t aclsparseLtMatDescGetAttribute(
    const aclsparseLtHandle_t* handle,
    aclsparseLtConstMatDescriptor_t* matDescr,
    aclsparseLtMatDescAttribute_t matAttribute,
    void* data,
    size_t dataSize)
{
    if (handle == nullptr || *handle == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatDescGetAttribute: handle is null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matDescr == nullptr || *matDescr == nullptr || data == nullptr) {
        OP_LOGE(kSparseLtLogTag, "MatDescGetAttribute: matDescr/data is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const auto* md = *matDescr;  // aclsparseLtConstMatDescriptor_t -> const aclsparseLtMatDescriptor*
    switch (matAttribute) {
        case ACLSPARSELT_MAT_NUM_BATCHES:
            if (dataSize != sizeof(int32_t)) {
                OP_LOGE(kSparseLtLogTag, "MAT_NUM_BATCHES: size mismatch, got %zu", dataSize);
                return ACL_SPARSE_STATUS_INVALID_VALUE;
            }
            *static_cast<int32_t*>(data) = md->numBatches;
            return ACL_SPARSE_STATUS_SUCCESS;
        case ACLSPARSELT_MAT_BATCH_STRIDE:
            if (dataSize != sizeof(int64_t)) {
                OP_LOGE(kSparseLtLogTag, "MAT_BATCH_STRIDE: size mismatch, got %zu", dataSize);
                return ACL_SPARSE_STATUS_INVALID_VALUE;
            }
            *static_cast<int64_t*>(data) = md->batchStride;
            return ACL_SPARSE_STATUS_SUCCESS;
        default:
            OP_LOGE(kSparseLtLogTag, "MatDescGetAttribute: unsupported attr %d",
                    static_cast<int32_t>(matAttribute));
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
}

// ============================================================================
// get_cube_core_num implementation — moved here from
// aclsparselt_internal.h so that tiling/platform/platform_ascendc.h (the heavy
// platform header) is only pulled into this TU, not into every host TU that
// includes the shared internal header.
// ============================================================================
uint32_t get_cube_core_num()
{
    auto* plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) { return SPLT_CORE_NUM; }
    uint32_t aic = plat->GetCoreNumAic();
    return (aic > 0) ? aic : static_cast<uint32_t>(SPLT_CORE_NUM);
}

// Query per-core UB capacity from the platform API instead of
// hardcoding 248KB. PlatformAscendC::GetCoreMemSize(CoreMemType::UB) returns
// the actual UB size for the current SoC. Falls back to 248 * 1024 (Ascend950
// UB capacity) when the platform API is unavailable (e.g. null instance or
// zero returned).
uint64_t get_ub_size()
{
    auto* plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) { return 248 * 1024; }
    uint64_t ubSize = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    return (ubSize > 0) ? ubSize : static_cast<uint64_t>(248 * 1024);
}

extern "C" aclsparseStatus_t aclsparseLtMatmulGetWorkspace(
    const aclsparseLtHandle_t* handle,
    aclsparseLtConstMatmulPlan_t* plan, size_t* workspaceSize)
{
    if (handle == nullptr || *handle == nullptr) {
        OP_LOGE(kSparseLtLogTag, "GetWorkspaceSize: handle is null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (plan == nullptr || *plan == nullptr || workspaceSize == nullptr) {
        OP_LOGE(kSparseLtLogTag, "GetWorkspaceSize: plan/workspaceSize is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *workspaceSize = (*plan)->workspaceSize;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Version / property queries — moved from common/aclsparselt_auxiliary.cpp
// to keep common zero-change (aligned with upstream PR#82/#84).
// Version is a compile-time constant derived from ACLSPARSELT_VERSION
// (handle_internal.h: MAJOR*10000 + MINOR*100 + PATCH, e.g. 0.1.0 -> 100).
// ============================================================================
extern "C" aclsparseStatus_t aclsparseLtGetVersion(const aclsparseLtHandle_t* handle, int* version)
{
    // version is compile-time constant, handle not needed beyond null check
    if (handle == nullptr || *handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (version == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *version = static_cast<int32_t>(ACLSPARSELT_VERSION);
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseLtGetProperty(
    aclsparseLtLibraryPropertyType_t propertyType, int* value)
{
    // Aligned with cuSPARSELt cusparseLtGetProperty: returns the requested
    // version component (MAJOR/MINOR/PATCH) into *value. No handle parameter
    // (version is a compile-time constant).
    if (value == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    switch (propertyType) {
        case ACLSPARSELT_MAJOR_VERSION:
            *value = ACLSPARSELT_VERSION_MAJOR;
            break;
        case ACLSPARSELT_MINOR_VERSION:
            *value = ACLSPARSELT_VERSION_MINOR;
            break;
        case ACLSPARSELT_PATCH_LEVEL:
            *value = ACLSPARSELT_VERSION_PATCH;
            break;
        default:
            OP_LOGE(kSparseLtLogTag, "GetProperty: invalid propertyType %d", (int)propertyType);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
