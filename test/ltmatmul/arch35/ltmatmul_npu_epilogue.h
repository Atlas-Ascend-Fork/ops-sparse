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

#ifndef TEST_LTMATMUL_NPU_EPILOGUE_H_
#define TEST_LTMATMUL_NPU_EPILOGUE_H_

// =============================================================================
// Epilogue helpers extracted from ltmatmul_npu_wrapper.h
// to reduce header file size.
//
// Provides:
//   - EpilogueBuffers: device bias buffer holder
//   - ConvertBiasFp32ToBytes: FP32 host bias → target-dtype byte buffer
//   - SetBiasAttr: convert + upload + set BIAS_POINTER/STRIDE
//   - SetActivationAttr: ReLU / GeLU / GELU_SCALING dispatch
//   - SetEpilogueAttrs: orchestrator (bias + activation)
//
// All epilogue calls are gated by LT_TEST_NPU_EPILOGUE_ENABLED.
// =============================================================================

#include "ltmatmul_test_utils.h"  // RAII guards, DeviceBuffer, Fp32ToFp16Bits, etc.

// LT_TEST_NPU_EPILOGUE_ENABLED / LT_TEST_NPU_BATCH_ENABLED are defined in
// ltmatmul_npu_wrapper.h (included by consumers). Define fallbacks here so
// this header can be compiled standalone if needed.
#ifndef LT_TEST_NPU_EPILOGUE_ENABLED
#define LT_TEST_NPU_EPILOGUE_ENABLED 1
#endif
#ifndef LT_TEST_NPU_BATCH_ENABLED
#define LT_TEST_NPU_BATCH_ENABLED 1
#endif

// MatmulNpuResult is defined in ltmatmul_npu_wrapper.h before this header
// is included. The include order is:
//   ltmatmul_npu_wrapper.h -> defines MatmulNpuResult -> includes this header.
// This ensures the full struct definition is available for inline functions.

// -----------------------------------------------------------------------------
// Epilogue attributes: bias + activation.
//
// Sets BIAS_POINTER, RELU / RELU_UPPERBOUND / GELU_SCALING on the matmul
// descriptor. Batch attributes (NUM_BATCHES / BATCH_STRIDE) are set separately
// on matrix descriptors via SetBatchAttrsOnAllMatDescs before
// MatmulDescriptorInit. All epilogue calls are gated by
// LT_TEST_NPU_EPILOGUE_ENABLED — when 0, the function is a no-op and
// result.epilogueApplied stays false.
//
// biasVecHost: per-row bias (host, FP32). When non-null and bias_enabled,
// converted to biasDtype bytes and copied to device; the device pointer is
// set via BIAS_POINTER. When biasStride>0, the full biasVecFlat
// (numBatches * biasStride) is converted and copied.
//
// biasDtype / biasEltSize: bias dtype follows C dtype (FP16 path
// -> FP16 bias, BF16 path -> BF16 bias), except INT8 -> FP32.
// -----------------------------------------------------------------------------
struct EpilogueBuffers {
    sparse_test::DeviceBuffer dBias;  // device bias buffer (per-row or per-batch)
};

// Convert a host FP32 bias vector to target-dtype byte buffer for device upload.
// FP32/INT8 -> FP32 bytes; FP16 -> FP16 bit patterns; BF16 -> BF16 bit patterns.
// Uses std::vector range constructor instead of std::memcpy (unsafe function).
inline std::vector<uint8_t> ConvertBiasFp32ToBytes(const std::vector<float>& biasFp32,
                                                      aclDataType biasDtype)
{
    const auto bitsToBytes = [](const std::vector<uint16_t>& buf) {
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(buf.data());
        return std::vector<uint8_t>(raw, raw + buf.size() * sizeof(uint16_t));
    };
    if (biasDtype == ACL_FLOAT16) {
        std::vector<uint16_t> buf(biasFp32.size());
        for (size_t i = 0; i < biasFp32.size(); ++i) {
            buf[i] = sparse_test::Fp32ToFp16Bits(biasFp32[i]);
        }
        return bitsToBytes(buf);
    } else if (biasDtype == ACL_BF16) {
        std::vector<uint16_t> buf(biasFp32.size());
        for (size_t i = 0; i < biasFp32.size(); ++i) {
            buf[i] = sparse_test::Fp32ToBf16Bits(biasFp32[i]);
        }
        return bitsToBytes(buf);
    }
    // FP32 or INT8 path: bias is FP32
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(biasFp32.data());
    return std::vector<uint8_t>(raw, raw + biasFp32.size() * sizeof(float));
}

// SetBiasAttr: convert + upload + set BIAS_POINTER/STRIDE.
inline EpilogueBuffers SetBiasAttr(aclsparseLtConstHandle_t handle,
    sparse_test::SparseLtMatmulDescGuard& matmulDesc,
    int32_t biasEnabled, int64_t biasStride,
    const std::vector<float>* biasVecHost,
    int32_t m,
    MatmulNpuResult& result, aclDataType biasDtype)
{
    EpilogueBuffers bufs;
    if (biasEnabled && biasVecHost != nullptr && !biasVecHost->empty()) {
        // Validate bias length matches m (per-row broadcast bias).
        if (biasStride > 0) {
            if (static_cast<int64_t>(biasVecHost->size()) < static_cast<int64_t>(m) * 1 &&
                static_cast<int64_t>(biasVecHost->size()) < biasStride) {
                result.biasSetRet = ACL_SPARSE_STATUS_INVALID_VALUE;
                return bufs;
            }
        } else {
            if (static_cast<int64_t>(biasVecHost->size()) < static_cast<int64_t>(m)) {
                result.biasSetRet = ACL_SPARSE_STATUS_INVALID_VALUE;
                return bufs;
            }
        }
        auto biasBytesBuf = ConvertBiasFp32ToBytes(*biasVecHost, biasDtype);
        size_t biasBytes = biasBytesBuf.size();
        bufs.dBias = sparse_test::DeviceBuffer::copyFrom(biasBytesBuf.data(), biasBytes);
        void* biasDevPtr = bufs.dBias.get();
        result.biasSetRet = aclsparseLtMatmulDescSetAttribute(handle, &matmulDesc.get(),
            ACLSPARSELT_MATMUL_BIAS_POINTER, &biasDevPtr, sizeof(void*));
        if (result.biasSetRet != ACL_SPARSE_STATUS_SUCCESS) { return bufs; }
        if (biasStride > 0) {
            int64_t bs = biasStride;
            result.biasSetRet = aclsparseLtMatmulDescSetAttribute(handle, &matmulDesc.get(),
                ACLSPARSELT_MATMUL_BIAS_STRIDE, &bs, sizeof(int64_t));
        }
    }
    return bufs;
}

// SetActivationAttr: ReLU / GeLU / GELU_SCALING dispatch.
inline void SetActivationAttr(aclsparseLtConstHandle_t handle,
    sparse_test::SparseLtMatmulDescGuard& matmulDesc,
    int32_t activationType, float reluUpperBound, float reluThreshold, float geluScaling,
    MatmulNpuResult& result)
{
    if (activationType == 1) {
        int enable = 1;
        result.actSetRet = aclsparseLtMatmulDescSetAttribute(handle, &matmulDesc.get(),
            ACLSPARSELT_MATMUL_ACTIVATION_RELU, &enable, sizeof(int));
        if (result.actSetRet != ACL_SPARSE_STATUS_SUCCESS) { return; }
        result.actSetRet = aclsparseLtMatmulDescSetAttribute(handle, &matmulDesc.get(),
            ACLSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, &reluUpperBound, sizeof(float));
        if (result.actSetRet != ACL_SPARSE_STATUS_SUCCESS) { return; }
        result.actSetRet = aclsparseLtMatmulDescSetAttribute(handle, &matmulDesc.get(),
            ACLSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD, &reluThreshold, sizeof(float));
    } else if (activationType == 2) {
        int enable = 1;
        result.actSetRet = aclsparseLtMatmulDescSetAttribute(handle, &matmulDesc.get(),
            ACLSPARSELT_MATMUL_ACTIVATION_GELU, &enable, sizeof(int));
        if (result.actSetRet != ACL_SPARSE_STATUS_SUCCESS) { return; }
        result.actSetRet = aclsparseLtMatmulDescSetAttribute(handle, &matmulDesc.get(),
            ACLSPARSELT_MATMUL_ACTIVATION_GELU_SCALING, &geluScaling, sizeof(float));
    } else if (activationType == 3) {
        result.actSetRet = aclsparseLtMatmulDescSetAttribute(handle, &matmulDesc.get(),
            ACLSPARSELT_MATMUL_ACTIVATION_GELU_SCALING, &geluScaling, sizeof(float));
    }
}

inline EpilogueBuffers SetEpilogueAttrs(aclsparseLtConstHandle_t handle,
    sparse_test::SparseLtMatmulDescGuard& matmulDesc, int32_t m,
    int32_t biasEnabled, int64_t biasStride,
    int32_t activationType, float reluUpperBound, float reluThreshold, float geluScaling,
    int32_t numBatches, int64_t batchStride,
    const std::vector<float>* biasVecHost,
    MatmulNpuResult& result,
    size_t biasEltSize = sizeof(float),
    aclDataType biasDtype = ACL_FLOAT)
{
    EpilogueBuffers bufs;
    result.epilogueApplied = false;
    (void)m; (void)biasEltSize;
#if LT_TEST_NPU_EPILOGUE_ENABLED
    bufs = SetBiasAttr(handle, matmulDesc, biasEnabled, biasStride,
                       biasVecHost, m, result, biasDtype);
    if (result.biasSetRet != ACL_SPARSE_STATUS_SUCCESS) { return bufs; }
    SetActivationAttr(handle, matmulDesc, activationType,
                       reluUpperBound, reluThreshold, geluScaling, result);
    if (result.actSetRet != ACL_SPARSE_STATUS_SUCCESS) { return bufs; }
    (void)numBatches; (void)batchStride;
    result.batchSetRet = ACL_SPARSE_STATUS_SUCCESS;
    result.epilogueApplied = true;
#else
    (void)handle; (void)matmulDesc; (void)biasEnabled; (void)biasStride;
    (void)activationType; (void)reluUpperBound; (void)reluThreshold; (void)geluScaling;
    (void)numBatches; (void)batchStride; (void)biasVecHost; (void)biasDtype;
    result.biasSetRet = ACL_SPARSE_STATUS_SUCCESS;
    result.actSetRet = ACL_SPARSE_STATUS_SUCCESS;
    result.batchSetRet = ACL_SPARSE_STATUS_SUCCESS;
#endif
    return bufs;
}

#endif  // TEST_LTMATMUL_NPU_EPILOGUE_H_
