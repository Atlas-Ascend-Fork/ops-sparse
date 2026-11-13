/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

// Sort 家族（csrsort / cscsort / coosort，arch35）共用 tiling 工具：
// - CalcSortTmpSize：查询 AscendC Sort 共享临时缓冲大小（RADIX_SORT，
//   int32 键 / uint32 索引，hasDstIndex 配置须与 kernel 侧使用一致）；
// - FindMaxRunSize：按 UB 容量与 DataCopyPad 上限（2^21-1 字节）二分
//   最大的单趟 run 段长（5 个 run 尺寸缓冲 + Sort tmp 预算）。
// 此前 csrsort/cscsort/coosort 三处各自维护同口径实现，UB 预算常量
// （kDataCopyPadMaxBytes / kRunSizedBufferCount / kReservedBytes /
// kSimtDcacheBytes）需跨算子双写，收敛为本头文件单点定义。

#ifndef ACLSPARSE_SORT_TILING_UTILS_H_
#define ACLSPARSE_SORT_TILING_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "sort/sort_tiling_intf.h"

namespace sort_tiling {

inline uint32_t CalcSortTmpSize(uint32_t count)
{
    std::vector<int64_t> shapeVec = {static_cast<int64_t>(count)};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = false;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0U;
    uint32_t minValue = 0U;
    AscendC::GetSortMaxMinTmpSize(srcShape, ge::DT_INT32, ge::DT_UINT32, false,
                                  config, maxValue, minValue);
    return maxValue;
}

inline bool FindMaxRunSize(uint64_t ubSize, uint32_t &runSize, uint32_t &sortTmpSize)
{
    // On Ascend 950 (Atlas 350), DataCopyPad with DataCopyExtParams limits
    // blockLen to 2^21 - 1 bytes.
    constexpr uint32_t kDataCopyPadMaxBytes = (1U << 21U) - 1U;
    constexpr uint32_t kDataCopyPadMaxElems = kDataCopyPadMaxBytes / sizeof(int32_t);
    // Five run-sized buffers (col/row indices, pIn, dstKey, dstIdx and outP).
    // The Sort shared temporary buffer is sized by CalcSortTmpSize and counted
    // separately.
    constexpr uint32_t kRunSizedBufferCount = 5U;
    constexpr uint32_t kUbAlignBytes = 32U;
    constexpr uint64_t kReservedBytes = 8192U;
    // Reserve 32 KiB for SIMT DCache so that the short-run Vector Sort path
    // and the long-run SIMT merge path can coexist on the same AIV without UB
    // pressure contention.
    constexpr uint64_t kSimtDcacheBytes = 32U * 1024U;
    constexpr uint64_t kUnavailableBytes = kReservedBytes + kSimtDcacheBytes;
    uint64_t usable = (ubSize > kUnavailableBytes) ? ubSize - kUnavailableBytes : 0U;
    uint64_t maxElems = usable / (kRunSizedBufferCount * sizeof(int32_t));
    uint32_t maxCandidate = static_cast<uint32_t>(
        std::min<uint64_t>(maxElems, kDataCopyPadMaxElems));
    uint32_t low = 1U;
    uint32_t high = maxCandidate;
    uint32_t bestRun = 0U;
    uint32_t bestTmp = 0U;

    while (low <= high) {
        uint32_t mid = low + (high - low) / 2U;
        uint32_t candidate = mid;
        uint32_t tmp = CalcSortTmpSize(candidate);
        uint64_t bufferBytes = static_cast<uint64_t>(candidate) * sizeof(int32_t);
        uint64_t alignedBufferBytes =
            (bufferBytes + kUbAlignBytes - 1U) / kUbAlignBytes * kUbAlignBytes;
        uint64_t required = kRunSizedBufferCount * alignedBufferBytes + tmp;
        if (tmp > 0U && required <= usable) {
            bestRun = candidate;
            bestTmp = tmp;
            low = mid + 1U;
        } else {
            high = mid - 1U;
        }
    }
    if (bestRun == 0U) {
        return false;
    }
    runSize = bestRun;
    sortTmpSize = bestTmp;
    return true;
}


// Sort 家族 SIMT 线程数计算：按每核元素量对齐 warp 大小（32）并夹在
// [32, 2048]。此前 csrsort35 与 cscsort35 各自维护逐字副本（常量
// 32/2048 完全一致，仅常量前缀命名不同），收敛为本头文件单点定义。
inline uint32_t CalcSimtThreads(uint32_t nnz, uint32_t coreNum)
{
    constexpr uint32_t kSortSimtWarpSize = 32U;
    constexpr uint32_t kSortSimtMaxThreads = 2048U;
    if (coreNum == 0U) {
        return kSortSimtWarpSize;
    }
    uint32_t elemsPerCore = (nnz + coreNum - 1U) / coreNum;
    uint32_t aligned = (elemsPerCore + kSortSimtWarpSize - 1U) /
                       kSortSimtWarpSize * kSortSimtWarpSize;
    return std::min(std::max(aligned, kSortSimtWarpSize),
                    kSortSimtMaxThreads);
}

}  // namespace sort_tiling

#endif  // ACLSPARSE_SORT_TILING_UTILS_H_
