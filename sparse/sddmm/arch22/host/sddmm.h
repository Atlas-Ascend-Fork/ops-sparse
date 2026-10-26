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
#ifndef SDDMM_HOST_H_
#define SDDMM_HOST_H_

#include "sddmm_tiling.h"

#include "acl/acl.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"

inline struct aclsparseContext* SddmmToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext*>(handle);
}

inline struct aclsparseSpMatDescr* SddmmToSpMatInner(aclsparseSpMatDescr_t desc)
{
    return reinterpret_cast<struct aclsparseSpMatDescr*>(desc);
}

inline struct aclsparseDnMatDescr* SddmmToDnMatInner(aclsparseConstDnMatDescr_t desc)
{
    return const_cast<struct aclsparseDnMatDescr*>(reinterpret_cast<const struct aclsparseDnMatDescr*>(desc));
}

inline float SddmmReadScalarToF32(const void* ptr, aclDataType computeType)
{
    if (ptr == nullptr) {
        return 0.0f;
    }
    if (computeType == ACL_FLOAT16) {
        uint16_t bits;
        __builtin_memcpy(&bits, ptr, sizeof(uint16_t));
        uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
        uint32_t exp = (bits >> 10) & 0x1Fu;
        uint32_t mant = bits & 0x03FFu;
        uint32_t f;
        if (exp == 0) {
            if (mant == 0) {
                f = sign;
            } else {
                uint32_t shift = __builtin_clz(mant) - 21;
                mant <<= shift;
                exp = 1;
                f = sign | ((exp + 127 - 15 - shift) << 23) | ((mant << 13) & 0x7FFFFFu);
            }
        } else if (exp == 31) {
            f = sign | 0x7F800000u | (mant << 13);
        } else {
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        float v;
        __builtin_memcpy(&v, &f, sizeof(float));
        return v;
    }
    float result;
    __builtin_memcpy(&result, ptr, sizeof(float));
    return result;
}

// `aclsparseComplex` is {float x; float y;} with y the imaginary part, so the
// imaginary half of an alpha/beta host scalar is simply the second float.
inline float SddmmReadScalarImagToF32(const void* ptr, aclDataType computeType)
{
    if (ptr == nullptr || computeType != ACL_COMPLEX64) {
        return 0.0f;
    }
    float result;
    __builtin_memcpy(&result, static_cast<const uint8_t*>(ptr) + sizeof(float), sizeof(float));
    return result;
}

// Internal single-batch CSR routines. The public contract owns validation,
// pattern state and stream ordering before calling these numeric operations.
aclsparseStatus_t SddmmCsrBufferSize(aclsparseHandle_t, aclsparseOperation_t, aclsparseOperation_t, const void*,
                                     aclsparseConstDnMatDescr_t, aclsparseConstDnMatDescr_t, const void*,
                                     aclsparseSpMatDescr_t, aclDataType, aclsparseSDDMMAlg_t, size_t*);
aclsparseStatus_t SddmmCsrPrepare(aclsparseHandle_t, aclsparseOperation_t, aclsparseOperation_t, const void*,
                                  aclsparseConstDnMatDescr_t, aclsparseConstDnMatDescr_t, const void*,
                                  aclsparseSpMatDescr_t, aclDataType, aclsparseSDDMMAlg_t, void*);
aclsparseStatus_t SddmmCsrExecute(aclsparseHandle_t, aclsparseOperation_t, aclsparseOperation_t, const void*,
                                  aclsparseConstDnMatDescr_t, aclsparseConstDnMatDescr_t, const void*,
                                  aclsparseSpMatDescr_t, aclDataType, aclsparseSDDMMAlg_t, void*, void* = nullptr,
                                  void* = nullptr, uint32_t = 0);

// GetAivCoreCount returns 0 when the platform query fails; clamp to 1 so the
// caller never divides by zero when partitioning work across cores.
inline uint32_t SddmmGetBlockDim()
{
    const uint32_t aivNum = GetAivCoreCount();
    return (aivNum > 0) ? aivNum : 1;
}

// Supported dense-input / sparse-output storage pairs.
inline bool SddmmDtypePairSupported(aclDataType xType, aclDataType cType)
{
    return (xType == ACL_FLOAT && cType == ACL_FLOAT) || (xType == ACL_FLOAT16 && cType == ACL_FLOAT16) ||
           (xType == ACL_FLOAT16 && cType == ACL_FLOAT) || (xType == ACL_BF16 && cType == ACL_BF16) ||
           (xType == ACL_BF16 && cType == ACL_FLOAT) || (xType == ACL_COMPLEX64 && cType == ACL_COMPLEX64);
}

// Real products accumulate in FP32; complex products use complex64.
// Pure FP16 also accepts historical half scalars. Mixed FP16/FP32 requires FP32.
inline bool SddmmComputeTypeSupported(aclDataType xType, aclDataType cType, aclDataType computeType)
{
    const aclDataType expectedCompute = cType == ACL_COMPLEX64 ? ACL_COMPLEX64 : ACL_FLOAT;
    const bool legacyFp16Compute = xType == ACL_FLOAT16 && cType == ACL_FLOAT16 && computeType == ACL_FLOAT16;
    return computeType == expectedCompute || legacyFp16Compute;
}

#endif
