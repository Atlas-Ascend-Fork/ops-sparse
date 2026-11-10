/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file csrsort_host.cpp
 * \brief aclsparseXcsrsort Host 侧实现（Legacy API）。
 *
 * 两个 extern "C" API：
 *   - aclsparseXcsrsort_bufferSizeExt：查询 pBuffer
 * 大小（2*nnz*sizeof(int32_t)）。
 *   - aclsparseXcsrsort：对 CSR 每行的列索引执行原地稳定排序，并输出排列 P。
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "aclsparse_host_utils.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "csrsort_kernel.h"
#include "csrsort_tiling_utils.h"
#include "log/log.h"

namespace {

inline uint32_t CalcSimtThreads(uint32_t nnz, uint32_t coreNum) {
  if (coreNum == 0U) {
    return kCsrsortSimtWarpSize;
  }
  uint32_t elemsPerCore = (nnz + coreNum - 1U) / coreNum;
  uint32_t aligned = (elemsPerCore + kCsrsortSimtWarpSize - 1U) /
                     kCsrsortSimtWarpSize * kCsrsortSimtWarpSize;
  return std::min(std::max(aligned, kCsrsortSimtWarpSize),
                  kCsrsortSimtMaxThreads);
}

inline struct aclsparseContext *ToInternalHandle(aclsparseHandle_t handle) {
  return reinterpret_cast<struct aclsparseContext *>(handle);
}

static aclsparseStatus_t ValidateCsrsortCommonParams(const char *apiName, int m,
                                                     int n, int nnz,
                                                     const int *csrRowPtr,
                                                     const int *csrColInd) {
  if (m < 0 || n < 0 || nnz < 0) {
    OP_LOGE(apiName, "invalid dims: m=%d, n=%d, nnz=%d", m, n, nnz);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if ((m == 0 || n == 0) && nnz != 0) {
    OP_LOGE(apiName, "empty matrix (m=%d, n=%d) with nnz=%d", m, n, nnz);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (m > 0 && csrRowPtr == nullptr) {
    OP_LOGE(apiName, "csrRowPtr is nullptr (m=%d)", m);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (nnz > 0 && csrColInd == nullptr) {
    OP_LOGE(apiName, "csrColInd is nullptr (nnz=%d)", nnz);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ComputeCsrsortTiling(int m, int nnz,
                                              aclsparseIndexBase_t indexBase,
                                              CsrsortTilingData &tiling) {
  uint32_t aivCoreNum = GetAivCoreCount();
  if (aivCoreNum == 0U) {
    OP_LOGE("aclsparseXcsrsort", "GetAivCoreCount returned 0");
    return ACL_SPARSE_STATUS_INTERNAL_ERROR;
  }
  uint64_t ubSize = GetUbSize();
  if (ubSize == 0U) {
    OP_LOGE("aclsparseXcsrsort", "GetUbSize returned 0");
    return ACL_SPARSE_STATUS_INTERNAL_ERROR;
  }

  uint32_t runSize = 0U;
  uint32_t sortTmpBytes = 0U;
  if (!CsrsortTiling::FindMaxRunSize(ubSize, runSize, sortTmpBytes)) {
    OP_LOGE("aclsparseXcsrsort",
            "UB capacity %llu insufficient for minimum run",
            static_cast<unsigned long long>(ubSize));
    return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
  }

  // 每个非空 CSR 行最多形成一个独立任务。限制启动核数不超过 nnz，
  // 避免大量空行矩阵启动无实际工作的 AIV。
  uint32_t taskUpperBound =
      std::min(static_cast<uint32_t>(m), static_cast<uint32_t>(nnz));
  uint32_t coreNum = std::min(aivCoreNum, taskUpperBound);

  tiling.m = static_cast<uint32_t>(m);
  tiling.nnz = static_cast<uint32_t>(nnz);
  tiling.indexBase = static_cast<uint32_t>(indexBase);
  tiling.runSize = runSize;
  tiling.coreNum = coreNum;
  tiling.sortTmpBytes = sortTmpBytes;
  tiling.simtThreads = CalcSimtThreads(tiling.nnz, coreNum);

  OP_LOGI("aclsparseXcsrsort",
          "tiling: m=%u, nnz=%u, runSize=%u, blockDim=%u, sortTmp=%u, "
          "simtThreads=%u",
          tiling.m, tiling.nnz, tiling.runSize, tiling.coreNum,
          tiling.sortTmpBytes, tiling.simtThreads);
  return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t
LaunchCsrsortKernel(aclsparseHandle_t handle, int m, int nnz,
                    aclsparseIndexBase_t indexBase, const int *csrRowPtr,
                    int *csrColInd, int *P, void *pBuffer) {
  auto *h = ToInternalHandle(handle);
  aclrtStream stream = h->stream;
  CsrsortTilingData tiling{};
  aclsparseStatus_t status = ComputeCsrsortTiling(m, nnz, indexBase, tiling);
  if (status != ACL_SPARSE_STATUS_SUCCESS) {
    return status;
  }

  OP_LOGI("aclsparseXcsrsort", "launching kernel: blockDim=%u, stream=%p",
          tiling.coreNum, stream);
    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

  csrsort_kernel_do(
      reinterpret_cast<uint8_t *>(const_cast<int *>(csrRowPtr)),
      reinterpret_cast<uint8_t *>(csrColInd), reinterpret_cast<uint8_t *>(P),
      reinterpret_cast<uint8_t *>(pBuffer), tiling, tiling.coreNum, stream);
  if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
    return ACL_SPARSE_STATUS_EXECUTION_FAILED;
  }

  return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseXcsrsort_bufferSizeExt(aclsparseHandle_t handle,
                                                  int m, int n, int nnz,
                                                  const int *csrRowPtr,
                                                  const int *csrColInd,
                                                  size_t *pBufferSizeInBytes) {
  const char *api = "aclsparseXcsrsort_bufferSizeExt";
  if (handle == nullptr) {
    OP_LOGE(api, "handle is nullptr");
    return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
  }
  if (pBufferSizeInBytes == nullptr) {
    OP_LOGE(api, "pBufferSizeInBytes is nullptr");
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }

  aclsparseStatus_t st =
      ValidateCsrsortCommonParams(api, m, n, nnz, csrRowPtr, csrColInd);
  if (st != ACL_SPARSE_STATUS_SUCCESS) {
    return st;
  }

  if (nnz == 0) {
    *pBufferSizeInBytes = 0U;
    return ACL_SPARSE_STATUS_SUCCESS;
  }

  constexpr size_t kFactor = 2U * sizeof(int32_t);
  if (static_cast<size_t>(nnz) > std::numeric_limits<size_t>::max() / kFactor) {
    OP_LOGE(api, "workspace size overflow: nnz=%d", nnz);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  *pBufferSizeInBytes = static_cast<size_t>(nnz) * kFactor;
  return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseXcsrsort(aclsparseHandle_t handle, int m, int n,
                                    int nnz, const aclsparseMatDescr_t descrA,
                                    const int *csrRowPtr, int *csrColInd,
                                    int *P, void *pBuffer) {
  const char *api = "aclsparseXcsrsort";
  if (handle == nullptr) {
    OP_LOGE(api, "handle is nullptr");
    return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
  }

  aclsparseStatus_t st =
      ValidateCsrsortCommonParams(api, m, n, nnz, csrRowPtr, csrColInd);
  if (st != ACL_SPARSE_STATUS_SUCCESS) {
    return st;
  }

  if (descrA == nullptr) {
    OP_LOGE(api, "descrA is nullptr");
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  aclsparseIndexBase_t indexBase = aclsparseGetMatIndexBase(descrA);
  if (indexBase != ACL_SPARSE_INDEX_BASE_ZERO &&
      indexBase != ACL_SPARSE_INDEX_BASE_ONE) {
    OP_LOGE(api, "invalid indexBase: %d", static_cast<int>(indexBase));
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }

  if (nnz == 0) {
    return ACL_SPARSE_STATUS_SUCCESS;
  }

  if (P == nullptr) {
    OP_LOGE(api, "P is nullptr (nnz=%d)", nnz);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (pBuffer == nullptr) {
    OP_LOGE(api, "pBuffer is nullptr (nnz=%d)", nnz);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }

  return LaunchCsrsortKernel(handle, m, nnz, indexBase, csrRowPtr, csrColInd,
                             P, pBuffer);
}

} // extern "C"
