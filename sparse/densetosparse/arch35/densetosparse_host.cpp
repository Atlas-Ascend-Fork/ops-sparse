/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <limits>

#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "cann_ops_sparse.h"
#include "densetosparse_kernel.h"
#include "log/log.h"

namespace {

constexpr const char *kApi = "aclsparseDenseToSparse";
constexpr uint64_t kWorkPerCore = 4096;
constexpr uint64_t kScanChunk = 32;
// Software scheduling granularity for CSR/CSC count units along the minor
// dimension; it balances parallelism against unit/prefix workspace and scan overhead.
constexpr uint64_t kMaxMajorChunk = 4096;
constexpr uint64_t kMinMajorChunk = 128;
constexpr uint64_t kWarpThreads = 32;
// Re-measured once the emit pass started skipping empty regions, which made
// the per-unit overhead the thing worth cutting. P-01 COO float32:
// 4096 -> 2802us, 8192 -> 2704us, 16384 -> 2518us.
constexpr uint64_t kCooTile = 8192;
// Floor for the adaptive tile below. Under a warp's 32 lanes x 32-byte window
// the extra units stop buying parallelism and only add prefix entries.
constexpr uint64_t kMinCooTile = 256;
constexpr size_t kAlignment = 32;
constexpr size_t kWorkspaceHeaderBytes = 64;
constexpr uint64_t kWindowBytes = 32;
constexpr uint64_t kBitsPerWord = 32;
constexpr int32_t kDeviceStatusSuccess = 0;

bool IsIndexTypeValid(aclsparseIndexType_t type) {
  return type == ACL_SPARSE_INDEX_32I || type == ACL_SPARSE_INDEX_64I;
}

bool IsIndexBaseValid(aclsparseIndexBase_t base) {
  return base == ACL_SPARSE_INDEX_BASE_ZERO ||
         base == ACL_SPARSE_INDEX_BASE_ONE;
}

uint32_t GetElementBytes(aclDataType type) {
  if (type == ACL_INT8) {
    return 1;
  }
  if (type == ACL_FLOAT16 || type == ACL_BF16) {
    return 2;
  }
  if (type == ACL_COMPLEX64) {
    return 8;
  }
  return type == ACL_FLOAT ? 4 : 0;
}

bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t *result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor) {
  return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

/// Elements of the minor dimension handled by one CSR/CSC count unit.
///
/// A fixed chunk cannot serve both extremes: measured on 950PR, an 8192x28672
/// CSR wants the largest chunk (0.30 vs 0.19 of baseline at 512), while a
/// 4096x1536 CSC is starved of units at 4096 and gains 4.8x from 512. So the
/// chunk shrinks only as far as needed to keep every worker busy. Contiguous
/// runs are consumed by whole warps, strided ones by single threads, so the
/// two paths need very different unit counts.
uint64_t SelectMajorChunk(uint64_t major, uint64_t minor, bool contiguous) {
  const uint64_t cores = GetAivCoreCount();
  if (major == 0 || minor == 0 || cores == 0) {
    return kMaxMajorChunk;
  }
  const uint64_t threads = cores * kDenseToSparseThreads;
  const uint64_t targetUnits = contiguous ? threads / kWarpThreads : threads;
  const uint64_t wantedChunks = CeilDiv(targetUnits, major);
  if (wantedChunks <= 1) {
    return kMaxMajorChunk;
  }
  const uint64_t chunk = CeilDiv(minor, wantedChunks);
  if (chunk > kMaxMajorChunk) {
    return kMaxMajorChunk;
  }
  return chunk < kMinMajorChunk ? kMinMajorChunk : chunk;
}

// Blocked-ELL geometry after padding rows/ellCols up to whole blocks. Tail
// blocks keep the invariant that every slot owns exactly blockSize*blockSize
// scalars, so the block-local addressing does not branch on the tail.
bool ComputeBellGeometry(const aclsparseSpMatDescr *sparse, uint64_t *indexCount,
                         uint64_t *valueCount) {
  const uint64_t blockSize = sparse->ellBlockSize;
  uint64_t blockScalars = 0;
  return CheckedMultiply(CeilDiv(sparse->rows, blockSize),
                         CeilDiv(sparse->ellCols, blockSize), indexCount) &&
         CheckedMultiply(blockSize, blockSize, &blockScalars) &&
         CheckedMultiply(*indexCount, blockScalars, valueCount);
}

bool IsSupportedFormat(aclsparseFormat_t format) {
  return format == ACL_SPARSE_FORMAT_CSR || format == ACL_SPARSE_FORMAT_CSC ||
         format == ACL_SPARSE_FORMAT_COO ||
         format == ACL_SPARSE_FORMAT_BLOCKED_ELL;
}

aclsparseStatus_t ValidateDenseSpan(const aclsparseDnMatDescr *dense,
                                    uint32_t elementBytes) {
  if (dense->rows < 0 || dense->cols < 0 || dense->ld < 0 ||
      (dense->order != ACL_SPARSE_ORDER_ROW &&
       dense->order != ACL_SPARSE_ORDER_COL)) {
    OP_LOGE(kApi,
            "invalid dense descriptor: rows=%" PRId64 ", cols=%" PRId64
            ", ld=%" PRId64 ", order=%d",
            dense->rows, dense->cols, dense->ld,
            static_cast<int>(dense->order));
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (dense->rows > 0 && dense->cols > 0 &&
      ((dense->order == ACL_SPARSE_ORDER_ROW && dense->ld < dense->cols) ||
       (dense->order == ACL_SPARSE_ORDER_COL && dense->ld < dense->rows))) {
    const int64_t minLd =
        dense->order == ACL_SPARSE_ORDER_ROW ? dense->cols : dense->rows;
    OP_LOGE(kApi,
            "invalid dense leading dimension: ld=%" PRId64
            ", expected >= %" PRId64 " for order=%d",
            dense->ld, minLd, static_cast<int>(dense->order));
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (dense->rows == 0 || dense->cols == 0) {
    return ACL_SPARSE_STATUS_SUCCESS;
  }
  const uint64_t major = dense->order == ACL_SPARSE_ORDER_ROW
                             ? static_cast<uint64_t>(dense->rows - 1)
                             : static_cast<uint64_t>(dense->cols - 1);
  const uint64_t minor = dense->order == ACL_SPARSE_ORDER_ROW
                             ? static_cast<uint64_t>(dense->cols - 1)
                             : static_cast<uint64_t>(dense->rows - 1);
  uint64_t address = 0;
  if (!CheckedMultiply(major, static_cast<uint64_t>(dense->ld), &address) ||
      address > std::numeric_limits<uint64_t>::max() - minor) {
    OP_LOGE(kApi,
            "dense element offset overflows: major=%" PRIu64 ", minor=%" PRIu64
            ", ld=%" PRId64,
            major, minor, dense->ld);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  address += minor;
  uint64_t bytes = 0;
  if (!CheckedMultiply(address + 1, elementBytes, &bytes) ||
      bytes > std::numeric_limits<size_t>::max()) {
    OP_LOGE(kApi,
            "dense byte span overflows: lastOffset=%" PRIu64
            ", elementBytes=%u",
            address, elementBytes);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateBell(const aclsparseSpMatDescr *sparse) {
  if (sparse->ellBlockSize == 0) {
    OP_LOGE(kApi, "invalid BELL ellBlockSize=0");
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  // Slots are whole block columns; with a column tail the padded ellCols may
  // exceed cols, so the bound is the padded block column count.
  if (CeilDiv(sparse->ellCols, sparse->ellBlockSize) >
      CeilDiv(sparse->cols, sparse->ellBlockSize)) {
    OP_LOGE(kApi,
            "invalid BELL ellCols=%" PRIu64 ", slots exceed block columns of cols=%" PRIu64
            " with ellBlockSize=%" PRIu64,
            sparse->ellCols, sparse->cols, sparse->ellBlockSize);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  uint64_t valueCount = 0;
  uint64_t indexCount = 0;
  if (!ComputeBellGeometry(sparse, &indexCount, &valueCount)) {
    OP_LOGE(kApi,
            "BELL capacity overflows: rows=%" PRIu64 ", ellCols=%" PRIu64
            ", ellBlockSize=%" PRIu64,
            sparse->rows, sparse->ellCols, sparse->ellBlockSize);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (valueCount != sparse->nnz) {
    OP_LOGE(kApi,
            "invalid BELL nnz=%" PRIu64 ", expected padded block grid=%" PRIu64,
            sparse->nnz, valueCount);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (indexCount > 0 && sparse->ellColInd == nullptr) {
    OP_LOGE(kApi, "BELL ellColInd is nullptr while indexCount=%" PRIu64,
            indexCount);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (valueCount > 0 && sparse->values == nullptr) {
    OP_LOGE(kApi, "BELL values is nullptr while valueCount=%" PRIu64,
            valueCount);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateDescriptorConfiguration(
    const aclsparseDnMatDescr *dense, const aclsparseSpMatDescr *sparse,
    aclsparseDenseToSparseAlg_t alg, uint32_t *elementBytes) {
  if (alg != ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT) {
    OP_LOGE(kApi, "invalid algorithm=%d, only default is supported",
            static_cast<int>(alg));
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (!IsSupportedFormat(sparse->format)) {
    OP_LOGE(kApi, "invalid sparse format=%d", static_cast<int>(sparse->format));
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (!IsIndexTypeValid(sparse->ptrType)) {
    OP_LOGE(kApi, "invalid sparse offset type=%d",
            static_cast<int>(sparse->ptrType));
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (!IsIndexTypeValid(sparse->IdxType)) {
    OP_LOGE(kApi, "invalid sparse index type=%d",
            static_cast<int>(sparse->IdxType));
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if ((sparse->format == ACL_SPARSE_FORMAT_CSR ||
       sparse->format == ACL_SPARSE_FORMAT_CSC) &&
      sparse->ptrType != sparse->IdxType) {
    OP_LOGE(kApi,
            "unsupported compressed sparse index type combination: format=%d, "
            "offsetType=%d, indexType=%d; offset and index types must match",
            static_cast<int>(sparse->format),
            static_cast<int>(sparse->ptrType),
            static_cast<int>(sparse->IdxType));
    return ACL_SPARSE_STATUS_NOT_SUPPORTED;
  }
  if (!IsIndexBaseValid(sparse->baseType)) {
    OP_LOGE(kApi, "invalid sparse index base=%d",
            static_cast<int>(sparse->baseType));
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  *elementBytes = GetElementBytes(dense->valueType);
  if (*elementBytes == 0) {
    OP_LOGE(kApi, "unsupported dense value type=%d",
            static_cast<int>(dense->valueType));
    return ACL_SPARSE_STATUS_NOT_SUPPORTED;
  }
  if (GetElementBytes(sparse->valueType) == 0) {
    OP_LOGE(kApi, "unsupported sparse value type=%d",
            static_cast<int>(sparse->valueType));
    return ACL_SPARSE_STATUS_NOT_SUPPORTED;
  }
  return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateDescriptorShape(const aclsparseDnMatDescr *dense,
                                          const aclsparseSpMatDescr *sparse) {
  if (dense->valueType != sparse->valueType) {
    OP_LOGE(kApi, "value type mismatch: dense=%d, sparse=%d",
            static_cast<int>(dense->valueType),
            static_cast<int>(sparse->valueType));
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (dense->rows < 0 || dense->cols < 0) {
    OP_LOGE(kApi, "invalid dense shape: rows=%" PRId64 ", cols=%" PRId64,
            dense->rows, dense->cols);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (sparse->rows != static_cast<uint64_t>(dense->rows) ||
      sparse->cols != static_cast<uint64_t>(dense->cols)) {
    OP_LOGE(kApi,
            "shape mismatch: dense=(%" PRId64 ", %" PRId64 "), sparse=(%" PRIu64
            ", %" PRIu64 ")",
            dense->rows, dense->cols, sparse->rows, sparse->cols);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (sparse->rows >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      sparse->cols >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    OP_LOGE(kApi,
            "sparse shape exceeds INT32_MAX: rows=%" PRIu64 ", cols=%" PRIu64,
            sparse->rows, sparse->cols);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (sparse->rows > 0 && sparse->cols > 0 && dense->values == nullptr) {
    OP_LOGE(kApi, "dense values is nullptr for non-empty matrix");
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSparseStorage(const aclsparseSpMatDescr *sparse) {
  if (sparse->format == ACL_SPARSE_FORMAT_BLOCKED_ELL) {
    return ValidateBell(sparse);
  }
  if ((sparse->format == ACL_SPARSE_FORMAT_CSR ||
       sparse->format == ACL_SPARSE_FORMAT_CSC) &&
      sparse->ptrs == nullptr) {
    OP_LOGE(kApi, "sparse offsets is nullptr for format=%d",
            static_cast<int>(sparse->format));
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateStaticParams(aclsparseHandle_t handle,
                                       aclsparseConstDnMatDescr_t matA,
                                       aclsparseSpMatDescr_t matB,
                                       aclsparseDenseToSparseAlg_t alg) {
  if (handle == nullptr) {
    OP_LOGE(kApi, "handle is nullptr");
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (matA == nullptr) {
    OP_LOGE(kApi, "matA is nullptr");
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (matB == nullptr) {
    OP_LOGE(kApi, "matB is nullptr");
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  const auto *dense = matA;
  auto *sparse = matB;
  uint32_t elementBytes = 0;
  aclsparseStatus_t status =
      ValidateDescriptorConfiguration(dense, sparse, alg, &elementBytes);
  if (status != ACL_SPARSE_STATUS_SUCCESS) {
    return status;
  }
  status = ValidateDescriptorShape(dense, sparse);
  if (status != ACL_SPARSE_STATUS_SUCCESS) {
    return status;
  }
  status = ValidateDenseSpan(dense, elementBytes);
  return status == ACL_SPARSE_STATUS_SUCCESS ? ValidateSparseStorage(sparse)
                                             : status;
}

bool CheckedAddSize(size_t lhs, size_t rhs, size_t *result) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool AlignSize(size_t value, size_t *result) {
  if (value > std::numeric_limits<size_t>::max() - (kAlignment - 1)) {
    return false;
  }
  *result = (value + kAlignment - 1) & ~(kAlignment - 1);
  return true;
}

/// Whether a count unit walks consecutive addresses. COO tiles follow the dense
/// order directly; CSR/CSC are contiguous when the compressed major dimension
/// matches it. Must agree with CompressedRunIsContiguous in the kernel.
bool ScanIsContiguous(const aclsparseSpMatDescr *sparse,
                      aclsparseOrder_t order) {
  if (sparse->format == ACL_SPARSE_FORMAT_COO) {
    return order == ACL_SPARSE_ORDER_ROW;
  }
  return (sparse->format == ACL_SPARSE_FORMAT_CSR) ==
         (order == ACL_SPARSE_ORDER_ROW);
}

/// COO emits row-major, which is the very order CSR units are already walked
/// in, so a column-major dense can reuse the CSR split into (row, chunk of
/// columns). The strided loop then gives consecutive rows to consecutive
/// threads and a warp's loads at one column are adjacent instead of `ld` apart.
/// Row-major COO keeps the flat logical tiles, whose runs are contiguous.
bool CooUsesMajorUnits(const aclsparseSpMatDescr *sparse,
                       aclsparseOrder_t order) {
  return sparse->format == ACL_SPARSE_FORMAT_COO &&
         !ScanIsContiguous(sparse, order);
}

/// A flat COO unit is consumed by one warp, so a tile that leaves fewer units
/// than the machine has warps starves it: 96x192 splits into three units and
/// runs on three warps whatever the core count. Shrink the tile just enough to
/// fill the cores. Inputs big enough to already produce more units than warps
/// keep kCooTile, which is what the large scenarios were tuned at.
uint64_t SelectCooTile(uint64_t logical) {
  const uint64_t cores = GetAivCoreCount();
  if (cores == 0 || logical == 0) {
    return kCooTile;
  }
  const uint64_t warps = cores * kDenseToSparseThreads / kWarpThreads;
  if (warps == 0 || CeilDiv(logical, kCooTile) >= warps) {
    return kCooTile;
  }
  const uint64_t tile = CeilDiv(logical, warps);
  return tile < kMinCooTile ? kMinCooTile : tile;
}

/// The dimension a unit is indexed by, and the one it chunks.
void ScanDims(const aclsparseSpMatDescr *sparse, aclsparseOrder_t order,
              uint64_t *major, uint64_t *minor) {
  const bool overRows = sparse->format == ACL_SPARSE_FORMAT_CSR ||
                        CooUsesMajorUnits(sparse, order);
  *major = overRows ? sparse->rows : sparse->cols;
  *minor = overRows ? sparse->cols : sparse->rows;
}

/// Convert can place a unit from `offsets` alone when the unit is a whole
/// major. That removes one full read of the dense input, but the grid then has
/// only `major` units, so it is only a win where that already fills the cores.
/// Restricted to the contiguous path, which is the one that consumes units per
/// warp and therefore needs far fewer of them.
bool UseSinglePassConvert(const aclsparseDnMatDescr *dense,
                          const aclsparseSpMatDescr *sparse) {
  if (sparse->format != ACL_SPARSE_FORMAT_CSR &&
      sparse->format != ACL_SPARSE_FORMAT_CSC) {
    return false;
  }
  if (!ScanIsContiguous(sparse, dense->order)) {
    return false;
  }
  const uint64_t cores = GetAivCoreCount();
  if (cores == 0) {
    return false;
  }
  const uint64_t major =
      sparse->format == ACL_SPARSE_FORMAT_CSR ? sparse->rows : sparse->cols;
  return major >= cores * kDenseToSparseThreads / kWarpThreads;
}

aclsparseStatus_t ComputeUnitCount(const aclsparseSpMatDescr *sparse,
                                   aclsparseOrder_t order,
                                   uint64_t *unitCount) {
  uint64_t logical = 0;
  if (!CheckedMultiply(sparse->rows, sparse->cols, &logical) ||
      logical > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    OP_LOGE(kApi,
            "logical element count is unsupported: rows=%" PRIu64
            ", cols=%" PRIu64,
            sparse->rows, sparse->cols);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (logical == 0 || sparse->format == ACL_SPARSE_FORMAT_BLOCKED_ELL) {
    *unitCount = 0;
    return ACL_SPARSE_STATUS_SUCCESS;
  }
  if (sparse->format == ACL_SPARSE_FORMAT_COO &&
      !CooUsesMajorUnits(sparse, order)) {
    *unitCount = CeilDiv(logical, SelectCooTile(logical));
    return ACL_SPARSE_STATUS_SUCCESS;
  }
  uint64_t major = 0;
  uint64_t minor = 0;
  ScanDims(sparse, order, &major, &minor);
  const uint64_t chunk =
      SelectMajorChunk(major, minor, ScanIsContiguous(sparse, order));
  const uint64_t chunks = CeilDiv(minor, chunk);
  if (!CheckedMultiply(major, chunks, unitCount)) {
    OP_LOGE(kApi,
            "workspace unit count overflows: major=%" PRIu64
            ", chunks=%" PRIu64,
            major, chunks);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  return ACL_SPARSE_STATUS_SUCCESS;
}

/// Words of window bitmap reserved per count unit. Indexing per unit rather
/// than by absolute position keeps the words 32-bit aligned whatever the run
/// alignment is.
uint64_t BitmapWordsPerUnit(const aclsparseSpMatDescr *sparse,
                            aclsparseOrder_t order, uint32_t elementBytes) {
  if (sparse->format == ACL_SPARSE_FORMAT_BLOCKED_ELL || elementBytes == 0 ||
      !ScanIsContiguous(sparse, order)) {
    return 0;
  }
  const uint64_t perWindow = kWindowBytes / elementBytes;
  if (perWindow == 0) {
    return 0;
  }
  if (sparse->format == ACL_SPARSE_FORMAT_COO) {
    // A COO unit is cut at every row boundary it crosses and each of those runs
    // rounds its window bits up to a whole word, so the reserve has to grow
    // with the run count, not just the element count. Rows too short to fill a
    // word would spend more on the bitmap than the emit pass saves by reading
    // it, so drop the bitmap there instead.
    const uint64_t wordSpan = perWindow * kBitsPerWord;
    if (sparse->cols == 0 || sparse->cols < wordSpan) {
      return 0;
    }
    const uint64_t tile = SelectCooTile(sparse->rows * sparse->cols);
    return CeilDiv(tile, wordSpan) + CeilDiv(tile, sparse->cols) + 1;
  }
  const uint64_t runElements = SelectMajorChunk(
      sparse->format == ACL_SPARSE_FORMAT_CSR ? sparse->rows : sparse->cols,
      sparse->format == ACL_SPARSE_FORMAT_CSR ? sparse->cols : sparse->rows,
      true);
  // A CSR/CSC unit is a single run; one extra window covers a start that is not
  // 32-byte aligned.
  return CeilDiv(CeilDiv(runElements, perWindow) + 1, kBitsPerWord);
}

/// Bytes taken by the multi-level scan, i.e. where the bitmap starts.
size_t ScanLevelsBytes(uint64_t count) {
  size_t bytes = kWorkspaceHeaderBytes;
  while (count > 0) {
    size_t levelBytes = 0;
    if (!AlignSize(static_cast<size_t>(count) * sizeof(uint64_t), &levelBytes) ||
        !CheckedAddSize(bytes, levelBytes, &bytes)) {
      return bytes;
    }
    if (count <= kScanChunk) {
      break;
    }
    count = (count + kScanChunk - 1) / kScanChunk;
  }
  return bytes;
}

aclsparseStatus_t ComputeWorkspace(const aclsparseSpMatDescr *sparse,
                                   aclsparseOrder_t order,
                                   size_t *bufferSize) {
  uint64_t count = 0;
  aclsparseStatus_t status = ComputeUnitCount(sparse, order, &count);
  if (status != ACL_SPARSE_STATUS_SUCCESS || count == 0) {
    *bufferSize = 0;
    return status;
  }
  const uint64_t units = count;
  if (count > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
    OP_LOGE(kApi, "workspace level is too large: count=%" PRIu64, count);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  size_t bytes = ScanLevelsBytes(count);
  uint64_t bitmapBytes = 0;
  if (!CheckedMultiply(units,
                       BitmapWordsPerUnit(sparse, order,
                                          GetElementBytes(sparse->valueType)) *
                           sizeof(uint32_t),
                       &bitmapBytes)) {
    OP_LOGE(kApi, "bitmap size overflows: units=%" PRIu64, units);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  size_t alignedBitmap = 0;
  if (!AlignSize(static_cast<size_t>(bitmapBytes), &alignedBitmap) ||
      !CheckedAddSize(bytes, alignedBitmap, &bytes)) {
    OP_LOGE(kApi, "workspace size overflows adding the bitmap");
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  *bufferSize = bytes;
  return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t BuildTiling(const aclsparseDnMatDescr *dense,
                              const aclsparseSpMatDescr *sparse,
                              uint32_t numBlocks,
                              DenseToSparseTilingData *tiling) {
  tiling->rows = sparse->rows;
  tiling->cols = sparse->cols;
  tiling->ld = static_cast<uint64_t>(dense->ld);
  tiling->nnz = sparse->nnz;
  tiling->ellBlockSize = sparse->ellBlockSize;
  tiling->ellCols = sparse->ellCols;
  const bool contiguous = ScanIsContiguous(sparse, dense->order);
  uint64_t scanMajor = 0;
  uint64_t scanMinor = 0;
  ScanDims(sparse, dense->order, &scanMajor, &scanMinor);
  // A flat COO unit spans logical positions rather than a slice of a major, so
  // it carries the COO tile in the same field.
  tiling->majorChunk =
      sparse->format == ACL_SPARSE_FORMAT_COO && contiguous
          ? SelectCooTile(sparse->rows * sparse->cols)
          : SelectMajorChunk(scanMajor, scanMinor, contiguous);
  aclsparseStatus_t status =
      ComputeUnitCount(sparse, dense->order, &tiling->unitCount);
  if (status != ACL_SPARSE_STATUS_SUCCESS) {
    return status;
  }
  tiling->statusOffset = 0;
  tiling->nnzOffset = kAlignment;
  tiling->level0Offset = kWorkspaceHeaderBytes;
  tiling->bitmapWordsPerUnit =
      BitmapWordsPerUnit(sparse, dense->order, GetElementBytes(dense->valueType));
  tiling->bitmapOffset = ScanLevelsBytes(tiling->unitCount);
  tiling->bitmapMode = 0;
  tiling->format = static_cast<uint32_t>(sparse->format);
  tiling->order = static_cast<uint32_t>(dense->order);
  tiling->base = static_cast<uint32_t>(sparse->baseType);
  tiling->offsetType = static_cast<uint32_t>(sparse->ptrType);
  tiling->indexType = static_cast<uint32_t>(sparse->IdxType);
  tiling->elementBytes = GetElementBytes(dense->valueType);
  tiling->numBlocks = numBlocks;
  return ACL_SPARSE_STATUS_SUCCESS;
}

uint32_t GetLaunchBlocks(uint64_t work) {
  const uint32_t cores = GetAivCoreCount();
  if (cores == 0) {
    return 0;
  }
  const uint64_t needed =
      std::max<uint64_t>(1, (work + kWorkPerCore - 1) / kWorkPerCore);
  return static_cast<uint32_t>(std::min<uint64_t>(cores, needed));
}

aclsparseStatus_t ValidateConvertPointers(const aclsparseSpMatDescr *sparse) {
  if (sparse->format == ACL_SPARSE_FORMAT_COO) {
    if (sparse->nnz > 0 && sparse->rowInds == nullptr) {
      OP_LOGE(kApi, "COO rowInds is nullptr while nnz=%" PRIu64, sparse->nnz);
      return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (sparse->nnz > 0 && sparse->colInds == nullptr) {
      OP_LOGE(kApi, "COO colInds is nullptr while nnz=%" PRIu64, sparse->nnz);
      return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (sparse->nnz > 0 && sparse->values == nullptr) {
      OP_LOGE(kApi, "COO values is nullptr while nnz=%" PRIu64, sparse->nnz);
      return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
  }
  if (sparse->format == ACL_SPARSE_FORMAT_CSR ||
      sparse->format == ACL_SPARSE_FORMAT_CSC) {
    if (sparse->nnz > 0 && sparse->idxs == nullptr) {
      OP_LOGE(kApi, "sparse indices is nullptr for format=%d, nnz=%" PRIu64,
              static_cast<int>(sparse->format), sparse->nnz);
      return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (sparse->nnz > 0 && sparse->values == nullptr) {
      OP_LOGE(kApi, "sparse values is nullptr for format=%d, nnz=%" PRIu64,
              static_cast<int>(sparse->format), sparse->nnz);
      return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
  }
  return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ReadAnalysisResult(aclsparseContext *context,
                                     uint8_t *workspace,
                                     const DenseToSparseTilingData &tiling,
                                     uint64_t logical, uint64_t *actualNnz) {
  int32_t deviceStatus = kDeviceStatusSuccess;
  aclError ret = aclrtMemcpyAsync(
      &deviceStatus, sizeof(deviceStatus), workspace + tiling.statusOffset,
      sizeof(deviceStatus), ACL_MEMCPY_DEVICE_TO_HOST, context->stream);
  if (ret == ACL_SUCCESS) {
    ret = aclrtMemcpyAsync(actualNnz, sizeof(*actualNnz),
                           workspace + tiling.nnzOffset, sizeof(*actualNnz),
                           ACL_MEMCPY_DEVICE_TO_HOST, context->stream);
  }
  if (ret != ACL_SUCCESS ||
      aclrtSynchronizeStream(context->stream) != ACL_SUCCESS) {
    OP_LOGE(kApi, "failed to copy or synchronize analysis result");
    return ACL_SPARSE_STATUS_EXECUTION_FAILED;
  }
  if (deviceStatus != kDeviceStatusSuccess ||
      *actualNnz > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      *actualNnz > logical) {
    OP_LOGE(kApi,
            "invalid analysis result: deviceStatus=%d, nnz=%" PRIu64
            ", logical=%" PRIu64,
            deviceStatus, *actualNnz, logical);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t LaunchAnalysis(aclsparseHandle_t handle,
                                 aclsparseConstDnMatDescr_t matA,
                                 aclsparseSpMatDescr_t matB, void *buffer) {
  const auto *dense = matA;
  auto *sparse = matB;
  if (sparse->format == ACL_SPARSE_FORMAT_BLOCKED_ELL) {
    return ACL_SPARSE_STATUS_SUCCESS;
  }
  uint64_t logical = 0;
  if (!CheckedMultiply(sparse->rows, sparse->cols, &logical)) {
    OP_LOGE(kApi,
            "logical element count overflows: rows=%" PRIu64
            ", cols=%" PRIu64,
            sparse->rows, sparse->cols);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  if (logical == 0 && sparse->format == ACL_SPARSE_FORMAT_COO) {
    sparse->nnz = 0;
    return ACL_SPARSE_STATUS_SUCCESS;
  }
  // Size the grid by the dense scan work, not by the major dimension: the count
  // kernel walks every element in both phases, and sizing by rows starved the
  // analysis launch (measured 2 blocks vs 56 in convert for 8192x28672).
  const uint32_t blocks = GetLaunchBlocks(std::max<uint64_t>(1, logical));
  if (blocks == 0) {
    OP_LOGE(kApi, "GetAivCoreCount returned 0 during analysis");
    return ACL_SPARSE_STATUS_INTERNAL_ERROR;
  }
  auto *context = handle;
  DenseToSparseTilingData tiling{};
  aclsparseStatus_t status = BuildTiling(dense, sparse, blocks, &tiling);
  if (status != ACL_SPARSE_STATUS_SUCCESS) {
    return status;
  }
  densetosparse_analysis_kernel_do(reinterpret_cast<GM_ADDR>(dense->values),
                                   reinterpret_cast<GM_ADDR>(sparse->ptrs),
                                   reinterpret_cast<GM_ADDR>(buffer), blocks,
                                   tiling, context->stream);
  if (logical == 0) {
    sparse->nnz = 0;
    return ACL_SPARSE_STATUS_SUCCESS;
  }
  uint64_t actualNnz = 0;
  status = ReadAnalysisResult(
      context, static_cast<uint8_t *>(buffer), tiling, logical, &actualNnz);
  if (status != ACL_SPARSE_STATUS_SUCCESS) {
    return status;
  }
  sparse->nnz = actualNnz;
  return ACL_SPARSE_STATUS_SUCCESS;
}

/// Convert 的两处 tiling 微调。
///
/// 每个 major 只有一个 unit 时，输出位置就是 offsets[major]，Convert 可以省掉自己
/// 那遍 count+scan；只有在 major 维度仍能喂饱各核时才值得这么做。
void ApplyConvertTilingMode(aclsparseConstDnMatDescr_t dense,
                            const aclsparseSpMatDescr *sparse,
                            DenseToSparseTilingData *tiling) {
  const bool singlePass = UseSinglePassConvert(dense, sparse);
  if (tiling->bitmapWordsPerUnit != 0 && !singlePass) {
    tiling->bitmapMode = 1;
  }
  if (!singlePass) {
    return;
  }
  const bool byRow = sparse->format == ACL_SPARSE_FORMAT_CSR;
  tiling->majorChunk = byRow ? sparse->cols : sparse->rows;
  tiling->unitCount = byRow ? sparse->rows : sparse->cols;
  tiling->singlePassConvert = 1;
}

aclsparseStatus_t LaunchConvert(aclsparseHandle_t handle,
                                aclsparseConstDnMatDescr_t matA,
                                aclsparseSpMatDescr_t matB, void *buffer) {
  const auto *dense = matA;
  auto *sparse = matB;
  uint64_t logical = 0;
  if (!CheckedMultiply(sparse->rows, sparse->cols, &logical)) {
    OP_LOGE(kApi,
            "logical element count overflows: rows=%" PRIu64
            ", cols=%" PRIu64,
            sparse->rows, sparse->cols);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  uint64_t bellTasks = 0;
  uint64_t bellScalars = 0;
  if (sparse->format == ACL_SPARSE_FORMAT_BLOCKED_ELL) {
    ComputeBellGeometry(sparse, &bellTasks, &bellScalars);
  }
  if ((sparse->format == ACL_SPARSE_FORMAT_BLOCKED_ELL ? bellTasks : logical) == 0) {
    return ACL_SPARSE_STATUS_SUCCESS;
  }
  // Grid sizing counts scalars, not BELL tasks: one task writes blockSize^2
  // elements, so sizing by task count left large block sizes on a single core.
  const uint64_t work =
      sparse->format == ACL_SPARSE_FORMAT_BLOCKED_ELL ? bellScalars : logical;
  const uint32_t blocks = GetLaunchBlocks(work);
  if (blocks == 0) {
    OP_LOGE(kApi, "GetAivCoreCount returned 0 during conversion");
    return ACL_SPARSE_STATUS_INTERNAL_ERROR;
  }
  auto *context = handle;
  DenseToSparseTilingData tiling{};
  aclsparseStatus_t status = BuildTiling(dense, sparse, blocks, &tiling);
  if (status != ACL_SPARSE_STATUS_SUCCESS) {
    return status;
  }
  ApplyConvertTilingMode(dense, sparse, &tiling);
  densetosparse_convert_kernel_do(reinterpret_cast<GM_ADDR>(dense->values),
                                  reinterpret_cast<GM_ADDR>(buffer),
                                  reinterpret_cast<GM_ADDR>(sparse->ptrs),
                                  reinterpret_cast<GM_ADDR>(sparse->idxs),
                                  reinterpret_cast<GM_ADDR>(sparse->rowInds),
                                  reinterpret_cast<GM_ADDR>(sparse->colInds),
                                  reinterpret_cast<GM_ADDR>(sparse->values),
                                  reinterpret_cast<GM_ADDR>(sparse->ellColInd),
                                  blocks, tiling, context->stream);
  return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseDenseToSparseGetBufferSize(
    aclsparseHandle_t handle, aclsparseConstDnMatDescr_t matA,
    aclsparseSpMatDescr_t matB, aclsparseDenseToSparseAlg_t alg,
    size_t *bufferSize) {
  if (bufferSize == nullptr) {
    OP_LOGE(kApi, "bufferSize is nullptr");
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  aclsparseStatus_t status = ValidateStaticParams(handle, matA, matB, alg);
  if (status != ACL_SPARSE_STATUS_SUCCESS) {
    return status;
  }
  return ComputeWorkspace(matB, matA->order, bufferSize);
}

aclsparseStatus_t aclsparseDenseToSparseAnalysis(
    aclsparseHandle_t handle, aclsparseConstDnMatDescr_t matA,
    aclsparseSpMatDescr_t matB, aclsparseDenseToSparseAlg_t alg, void *buffer) {
  aclsparseStatus_t status = ValidateStaticParams(handle, matA, matB, alg);
  if (status != ACL_SPARSE_STATUS_SUCCESS) {
    return status;
  }
  size_t required = 0;
  status = ComputeWorkspace(matB, matA->order, &required);
  if (status != ACL_SPARSE_STATUS_SUCCESS) {
    return status;
  }
  if (required > 0 && buffer == nullptr) {
    OP_LOGE(kApi, "buffer is nullptr while required size=%zu", required);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  return LaunchAnalysis(handle, matA, matB, buffer);
}

aclsparseStatus_t aclsparseDenseToSparseConvert(aclsparseHandle_t handle,
                                                aclsparseConstDnMatDescr_t matA,
                                                aclsparseSpMatDescr_t matB,
                                                aclsparseDenseToSparseAlg_t alg,
                                                void *buffer) {
  aclsparseStatus_t status = ValidateStaticParams(handle, matA, matB, alg);
  if (status != ACL_SPARSE_STATUS_SUCCESS) {
    return status;
  }
  auto *sparse = matB;
  size_t required = 0;
  status = ComputeWorkspace(sparse, matA->order, &required);
  if (status != ACL_SPARSE_STATUS_SUCCESS) {
    return status;
  }
  if (required > 0 && buffer == nullptr) {
    OP_LOGE(kApi, "buffer is nullptr while required size=%zu", required);
    return ACL_SPARSE_STATUS_INVALID_VALUE;
  }
  status = ValidateConvertPointers(sparse);
  return status == ACL_SPARSE_STATUS_SUCCESS
             ? LaunchConvert(handle, matA, matB, buffer)
             : status;
}

} // extern "C"
