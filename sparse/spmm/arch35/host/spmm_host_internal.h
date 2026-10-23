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

#ifndef SPMM_HOST_INTERNAL_H
#define SPMM_HOST_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <iterator>
#include <list>
#include <unordered_map>
#include <vector>

#include "aclsparse_descr_internal.h"
#include "cann_ops_sparse.h"
#include "spmm_host.h"

namespace spmm_host_internal {

struct CsrHostData {
    std::vector<int32_t> rowOffsets;
    std::vector<int32_t> columnIndices;
    int32_t indexBase = 0;
};

// Workspace offsets are shared by buffer sizing, preprocess, and execution.
struct WsOffsets {
    int64_t tilingOff;
    int64_t reorderOff;
    int64_t binEdgeOff;
    int64_t cubeMetaOff;
    int64_t cubeRowIdsOff;
    int64_t cubeSlot1Off;
    int64_t cubeAValueMapOff;
    int64_t cubePackedAOff;
    int64_t cubeComplexStatusOff;
    int64_t cubeComplexBOff;
    int64_t cubeScratchOff;
    int64_t cubeTilingOff;
    int64_t cubeLibWorkspaceOff;
    int64_t totalBytes;
    bool cubeReserved;
    bool complexReserved;
    int64_t transposeRowsOff = 0;
    int64_t transposeColumnsOff = 0;
    int64_t transposePermutationOff = 0;
    int64_t transposeValuesOff = 0;
};

constexpr int32_t kComplexCubeReserveRowsPerGroup = 14;
constexpr size_t kSpmmWorkspaceRegistryCapacity = 1024;

// A missing registration rebuilds preprocessing; eviction never frees device data.
template <typename Identity>
class WorkspaceOwnershipRegistry {
public:
    const Identity *Find(const void *buffer) const
    {
        const auto found = entries_.find(buffer);
        return found == entries_.end() ? nullptr : &found->second.identity;
    }

    void Erase(const void *buffer)
    {
        const auto found = entries_.find(buffer);
        if (found != entries_.end()) {
            order_.erase(found->second.position);
            entries_.erase(found);
        }
    }

    void Publish(const void *buffer, const Identity &identity)
    {
        const auto found = entries_.find(buffer);
        if (found != entries_.end()) {
            found->second.identity = identity;
            return;
        }
        if (entries_.size() >= kSpmmWorkspaceRegistryCapacity) {
            Erase(order_.front());
        }
        order_.push_back(buffer);
        try {
            entries_.emplace(buffer, Entry{identity, std::prev(order_.end())});
        } catch (...) {
            order_.pop_back();
            throw;
        }
    }

    size_t Size() const
    {
        return entries_.size();
    }

private:
    struct Entry {
        Identity identity;
        typename std::list<const void *>::iterator position;
    };
    std::list<const void *> order_;
    std::unordered_map<const void *, Entry> entries_;
};

size_t SpmmWorkspaceRegistrationCount();

template <int32_t Alignment>
inline bool BuildPackedWindow(int64_t columns, int32_t minCol, int32_t maxCol,
                               int32_t packedK,
                               SpmmCubeGroupMeta *meta)
{
    static_assert(Alignment > 0, "packed window alignment must be positive");
    if (minCol < 0 || maxCol < minCol || maxCol >= columns ||
        packedK <= 0 || packedK > columns) {
        return false;
    }
    int64_t colBase = static_cast<int64_t>(minCol / Alignment) * Alignment;
    if (colBase + packedK > columns) {
        colBase = ((columns - packedK) / Alignment) * Alignment;
    }
    if (colBase > minCol || maxCol >= colBase + packedK) {
        return false;
    }
    *meta = SpmmCubeGroupMeta{static_cast<int32_t>(colBase), static_cast<int32_t>(packedK)};
    return true;
}

inline bool BuildCubePrepackWindow(int64_t columns, int32_t minCol,
                                   int32_t maxCol, SpmmCubeGroupMeta *meta)
{
    if (minCol < 0 || maxCol < minCol || maxCol >= columns) return false;
    const int64_t colBase = static_cast<int64_t>(minCol / 4) * 4;
    const int64_t packedK = ((static_cast<int64_t>(maxCol) - colBase + 16) / 16) * 16;
    return packedK <= SPMM_CUBE_MAX_K && BuildPackedWindow<4>(
        columns, minCol, maxCol, static_cast<int32_t>(packedK), meta);
}

aclsparseStatus_t LoadCsrHostData(const aclsparseSpMatDescr *matA,
                                  CsrHostData *data);

bool WorkspaceRegionFits(int64_t begin, int64_t end, size_t bytes);
bool RegBaseWorkspaceFits(const WsOffsets &offsets, size_t batchBytes,
                          size_t rowIdsBytes, size_t slotBytes,
                          size_t packedBytes, size_t controlBytes);
bool IsValidCsrRowRange(int32_t begin, int32_t end, uint64_t nnz);

bool IsOfficialComplexP03Shape(int64_t m, int64_t k, int64_t n,
                               uint64_t nnz);
bool IsComplexRegBaseDegree(int32_t degree);
int32_t ComplexRegBaseRows(int32_t degree);
int32_t ComplexRegBaseWindowRows(int32_t degree);
int32_t ComplexRegBaseMinAverageRows(int32_t degree);

void ResetCubeSelection(bool *eligible, int32_t *selectedRows,
                        int32_t *selectedGroups,
                        int32_t *selectedFallbackCount = nullptr);

} // namespace spmm_host_internal

#endif // SPMM_HOST_INTERNAL_H
