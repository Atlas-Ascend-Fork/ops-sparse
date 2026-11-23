/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPSV_SCHEDULE_CACHE_H_
#define SPSV_SCHEDULE_CACHE_H_

#include <cstddef>
#include "aclsparse_spsv_descr.h"
#include "../shared/spsv_tiling_data.h"

static void SpsvScheduleCacheError(const char* operation, aclError status)
{
    OP_LOGE("aclsparseSpSV", "analysis schedule %s failed: %d", operation, static_cast<int>(status));
}

// Re-analysis never waits for the previous snapshot. If it is still in flight,
// leave its storage alive and use the ordinary recovery dispatch for this analysis.
static aclsparseStatus_t SpsvPrepareScheduleCache(aclsparseSpSVDescr_t descr)
{
    if (descr->cachedM <= 1) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (descr->analysisCopyPending && !descr->analysisEventRecorded) {
        OP_LOGE("aclsparseSpSV_analysis", "a previous schedule snapshot could not be recorded");
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (descr->analysisEvent != nullptr) {
        aclrtEventRecordedStatus ready = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        aclError status = aclrtQueryEventStatus(descr->analysisEvent, &ready);
        if (status != ACL_SUCCESS) {
            SpsvScheduleCacheError("query", status);
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        if (ready != ACL_EVENT_RECORDED_STATUS_COMPLETE) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        descr->analysisCopyPending = false;
        status = aclrtDestroyEvent(descr->analysisEvent);
        if (status != ACL_SUCCESS) {
            SpsvScheduleCacheError("destroy event", status);
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        descr->analysisEvent = nullptr;
        descr->analysisEventRecorded = false;
    }
    if (descr->analysisNumLevels == nullptr) {
        void* storage = nullptr;
        aclError status = aclrtMallocHost(&storage, sizeof(int32_t));
        if (status != ACL_SUCCESS) {
            SpsvScheduleCacheError("allocate", status);
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        descr->analysisNumLevels = static_cast<int32_t*>(storage);
    }
    aclError status = aclrtCreateEvent(&descr->analysisEvent);
    if (status != ACL_SUCCESS) {
        SpsvScheduleCacheError("create event", status);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SpsvRecordScheduleCache(aclsparseSpSVDescr_t descr, void* workspace, aclrtStream stream)
{
    if (descr->cachedM <= 1 || descr->analysisEvent == nullptr || descr->analysisEventRecorded ||
        descr->analysisCopyPending) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const auto* source = static_cast<const uint8_t*>(workspace) + offsetof(SpsvTilingData, numLevels);
    aclError status = aclrtMemcpyAsync(descr->analysisNumLevels, sizeof(int32_t), source, sizeof(int32_t),
                                       ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (status != ACL_SUCCESS) {
        SpsvScheduleCacheError("copy", status);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    descr->analysisCopyStream = stream;
    descr->analysisCopyPending = true;
    status = aclrtRecordEvent(descr->analysisEvent, stream);
    if (status != ACL_SUCCESS) {
        SpsvScheduleCacheError("record event", status);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    descr->analysisEventRecorded = true;
    descr->analysisSnapshotCurrent = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SpsvReadScheduleCache(aclsparseSpSVDescr_t descr, int32_t& levels)
{
    if (descr->cachedNumLevels == 0 && descr->analysisSnapshotCurrent) {
        aclrtEventRecordedStatus ready = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        aclError status = aclrtQueryEventStatus(descr->analysisEvent, &ready);
        if (status != ACL_SUCCESS) {
            SpsvScheduleCacheError("query", status);
            levels = 0;
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        if (ready == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
            descr->analysisCopyPending = false;
            descr->cachedNumLevels = *descr->analysisNumLevels;
            descr->analysisSnapshotCurrent = false;
        }
    }
    levels = descr->cachedNumLevels;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Destruction alone may wait for the small DMA owning the pinned buffer. Solve
// and analysis use nonblocking event queries and never wait for this optimization.
static aclsparseStatus_t SpsvReleaseScheduleCache(aclsparseSpSVDescr_t descr)
{
    if (descr->analysisCopyPending) {
        aclError status = descr->analysisEventRecorded ? aclrtSynchronizeEvent(descr->analysisEvent) :
                                                         aclrtSynchronizeStream(descr->analysisCopyStream);
        if (status != ACL_SUCCESS) {
            SpsvScheduleCacheError("finish copy", status);
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        descr->analysisCopyPending = false;
    }
    if (descr->analysisEvent != nullptr) {
        aclError status = aclrtDestroyEvent(descr->analysisEvent);
        if (status != ACL_SUCCESS) {
            SpsvScheduleCacheError("destroy event", status);
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        descr->analysisEvent = nullptr;
    }
    if (descr->analysisNumLevels != nullptr) {
        aclError status = aclrtFreeHost(descr->analysisNumLevels);
        if (status != ACL_SUCCESS) {
            SpsvScheduleCacheError("free", status);
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        descr->analysisNumLevels = nullptr;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

#endif // SPSV_SCHEDULE_CACHE_H_
