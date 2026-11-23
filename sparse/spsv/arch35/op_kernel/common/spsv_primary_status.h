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
#ifndef SPSV_PRIMARY_STATUS_H_
#define SPSV_PRIMARY_STATUS_H_

// This state is published only after every original-equation certificate
// reports no remaining risk and all participating blocks consume that result.
constexpr int32_t kSpsvComplexCertificateCompleted = 4;

__aicore__ inline void SpsvPublishComplexCertificateCompleted(GM_ADDR workspace)
{
    ((__gm__ SpsvTilingData*)workspace)->primaryStatus = kSpsvComplexCertificateCompleted;
    AscendC::GlobalTensor<uint8_t> state;
    state.SetGlobalBuffer(workspace);
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                      AscendC::DcciDst::CACHELINE_OUT>(state);
}

__aicore__ inline bool SpsvComplexCertificateCompleted(GM_ADDR workspace, int64_t m)
{
    if (m < 1024) {
        return false;
    }
    AscendC::GlobalTensor<uint8_t> state;
    state.SetGlobalBuffer(workspace);
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                      AscendC::DcciDst::CACHELINE_OUT>(state);
    return ((__gm__ const SpsvTilingData*)workspace)->primaryStatus == kSpsvComplexCertificateCompleted;
}

__aicore__ inline bool SpsvRealPrimaryCertified(GM_ADDR workspace, const SpsvTilingData& tiling)
{
    const bool scaled = SpsvUseRealScaledPrimary(tiling);
    const bool binary64 = SpsvUseRealBinary64Primary(tiling);
    if (!scaled && !binary64) {
        return false;
    }
    AscendC::GlobalTensor<uint8_t> state;
    state.SetGlobalBuffer(workspace);
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        state);
    int32_t status = ((__gm__ const SpsvTilingData*)workspace)->primaryStatus;
    return status == 1 || (binary64 && status == 3);
}

__aicore__ inline bool SpsvBinary64PrimaryCompleted(GM_ADDR workspace, int64_t m)
{
    if (m < 1024) {
        return false;
    }
    AscendC::GlobalTensor<uint8_t> state;
    state.SetGlobalBuffer(workspace);
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        state);
    return ((__gm__ const SpsvTilingData*)workspace)->primaryStatus == 2;
}

__aicore__ inline bool SpsvComplexPrimaryCompleted(GM_ADDR workspace, int64_t m)
{
    return SpsvBinary64PrimaryCompleted(workspace, m) || SpsvComplexCertificateCompleted(workspace, m);
}

#endif // SPSV_PRIMARY_STATUS_H_
