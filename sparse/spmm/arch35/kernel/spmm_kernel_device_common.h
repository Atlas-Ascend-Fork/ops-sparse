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

#ifndef SPMM_KERNEL_DEVICE_COMMON_H
#define SPMM_KERNEL_DEVICE_COMMON_H

// Shared device types and work partitioning for arch35 SpMM.

#include <stdint.h>
#include <cstdio>

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"

#include "../spmm_config.h"
#include "spmm_kernel_tiling.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 3510)
#error "SPMM SIMT: this TU is only for dav-3510 / Ascend 950PR (__NPU_ARCH__==3510)."
#endif

namespace {

using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using AscendC::WaitFlag;

constexpr uint32_t kMaxSimtThreadsPerBlock = 1024u;
constexpr uint32_t kCubeMaxK = SPMM_CUBE_MAX_K;
constexpr uint32_t kCubeMaxN = SPMM_CUBE_MAX_N;
struct SpmmBlockWorkRange {
    uint64_t first;
    uint64_t end;
};

__aicore__ inline SpmmBlockWorkRange GetSpmmBlockWorkRange(uint64_t total)
{
    const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
    if (blocks == 0u) {
        return {0u, 0u};
    }
    return {total * block / blocks, total * (block + 1) / blocks};
}

template<typename T>
struct SpmmIsFp16 { static constexpr bool value = false; };

template<>
struct SpmmIsFp16<half> { static constexpr bool value = true; };

} // namespace

#endif  // SPMM_KERNEL_DEVICE_COMMON_H
