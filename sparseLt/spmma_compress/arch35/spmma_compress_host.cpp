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

#include <algorithm>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "cann_ops_sparseLt.h"
#include "matmul_plan/aclsparselt_matmul_plan_internal.h"
#include "matmul_descriptor/aclsparselt_matmul_descriptor_internal.h"
#include "../../../sparse/common/aclsparse_host_utils.h"
#include "spmma_compress_kernel.h"

namespace {
constexpr const char* kLogTag = "aclsparseLtSpMMACompress";
constexpr uint64_t kAddressLimit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
// rows、cols 和 ld 的 32 字节对齐要求按元素字节数换算为元素对齐。
constexpr int kMatrixAlignmentBytes = 32;
constexpr uint64_t kVectorBytes = 256;
// 在平台报告的 UB 容量内采用保守工作上限，并为非数据开销保留空间。
constexpr uint64_t kUbReserve = 4096;
constexpr uint64_t kUbLimit = 248 * 1024;

struct CompressParams {
    SpMmaCompressTilingData tiling;
    uint32_t alignment = 0;
    uint32_t blockDim = 0;
};

bool CheckedMulAdd(uint64_t a, uint64_t b, uint64_t c, uint64_t& result)
{
    if (c > kAddressLimit || (b != 0 && a > (kAddressLimit - c) / b)) {
        return false;
    }
    result = a * b + c;
    return true;
}

uint64_t AlignVector(uint64_t bytes)
{
    return (bytes + kVectorBytes - 1) / kVectorBytes * kVectorBytes;
}

aclsparseStatus_t SelectSparseDescriptor(const aclsparseLtMatmulPlan_t* plan,
    const aclsparseLtMatDescriptor*& sparse, aclsparseOperation_t& op)
{
    if (plan == nullptr || *plan == nullptr || (*plan)->matmulDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const auto* md = (*plan)->matmulDescr;
    if (md->matA == nullptr || md->matB == nullptr || md->matC == nullptr || md->matD == nullptr ||
        md->matA->isStructured == md->matB->isStructured) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    sparse = md->matA->isStructured ? md->matA : md->matB;
    op = md->matA->isStructured ? md->opA : md->opB;
    if ((op != ACL_SPARSE_OP_NON_TRANSPOSE && op != ACL_SPARSE_OP_TRANSPOSE) ||
        sparse->sparsity != ACL_SPARSE_LT_SPARSITY_50_PERCENT ||
        (sparse->order != ACL_SPARSE_ORDER_ROW && sparse->order != ACL_SPARSE_ORDER_COL) ||
        sparse->alignment < 16 || sparse->alignment % 16 != 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t DeriveShape(const aclsparseLtMatDescriptor& sparse, aclsparseOperation_t op,
    CompressParams& params)
{
    auto& td = params.tiling;
    switch (sparse.valueType) {
        case ACL_FLOAT: td.dataBytes = 4; td.groupSize = 2; break;
        case ACL_FLOAT16:
        case ACL_BF16: td.dataBytes = 2; td.groupSize = 4; break;
        case ACL_INT8: td.dataBytes = 1; td.groupSize = 4; break;
        default: return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const int64_t divisor = kMatrixAlignmentBytes / td.dataBytes;
    if (sparse.rows <= 0 || sparse.cols <= 0 || sparse.ld <= 0 ||
        sparse.rows > std::numeric_limits<int32_t>::max() ||
        sparse.cols > std::numeric_limits<int32_t>::max() ||
        sparse.rows % divisor != 0 || sparse.cols % divisor != 0 || sparse.ld % divisor != 0 ||
        sparse.numBatches <= 0 || sparse.batchStride < 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const bool rowOrder = sparse.order == ACL_SPARSE_ORDER_ROW;
    td.rows = static_cast<uint64_t>(rowOrder ? sparse.rows : sparse.cols);
    td.cols = static_cast<uint64_t>(rowOrder ? sparse.cols : sparse.rows);
    td.ld = static_cast<uint64_t>(sparse.ld);
    td.batchStride = static_cast<uint64_t>(sparse.batchStride);
    td.numBatches = static_cast<uint64_t>(sparse.numBatches);
    td.alongRow = (op == ACL_SPARSE_OP_NON_TRANSPOSE) == rowOrder;
    params.alignment = sparse.alignment;
    return td.ld >= td.cols ? ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_INVALID_VALUE;
}

aclsparseStatus_t DeriveSizes(SpMmaCompressTilingData& td)
{
    uint64_t matrixSpan = 0;
    uint64_t inputElements = 0;
    uint64_t elements = 0;
    uint64_t outputBytes = 0;
    if (!CheckedMulAdd(td.rows - 1, td.ld, td.cols, matrixSpan) ||
        (td.batchStride != 0 && td.batchStride < matrixSpan) ||
        !CheckedMulAdd(td.numBatches - 1, td.batchStride, matrixSpan, inputElements) ||
        !CheckedMulAdd(inputElements, td.dataBytes, 0, td.inputBytes) ||
        !CheckedMulAdd(td.rows, td.cols, 0, elements) ||
        !CheckedMulAdd(elements, td.numBatches, 0, elements) ||
        !CheckedMulAdd(elements / 2, td.dataBytes, 0, td.valuesBytes)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    td.metadataBytes = elements / (2 * td.groupSize);
    if (!CheckedMulAdd(1, td.valuesBytes, td.metadataBytes, outputBytes) ||
        outputBytes > std::numeric_limits<size_t>::max()) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

uint64_t TileFootprint(const SpMmaCompressTilingData& td, uint64_t width, uint64_t rows)
{
    const uint64_t fragments = td.alongRow ? rows : rows / td.groupSize;
    const uint64_t groups = td.alongRow ? width / td.groupSize : width;
    const uint64_t values = AlignVector(groups * (td.groupSize / 2) * td.dataBytes);
    const uint64_t metadata = AlignVector(groups / 2);
    // 两个输出缓冲区各额外预留一个向量；有效输出不包含这些 UB 余量。
    return rows * AlignVector(width * td.dataBytes) + fragments * (values + metadata) +
        2 * kVectorBytes + kUbReserve;
}

bool CheckedCeilDiv(uint64_t value, uint64_t divisor, uint64_t& result)
{
    if (divisor == 0) {
        return false;
    }
    result = value / divisor + (value % divisor != 0);
    return true;
}

bool CheckedWorstRows(uint64_t totalRows, uint64_t tileRows, uint64_t coreCount, uint64_t& result)
{
    if (tileRows == 0 || coreCount == 0) {
        return false;
    }
    const uint64_t fullTiles = totalRows / tileRows;
    const uint64_t fullRounds = fullTiles / coreCount;
    result = std::max((fullRounds + (fullTiles % coreCount != 0)) * tileRows,
        fullRounds * tileRows + totalRows % tileRows);
    return true;
}

uint64_t BalanceTileRows(const SpMmaCompressTilingData& td, uint64_t width,
    uint64_t rows, uint64_t coreCount)
{
    if (!td.alongRow || td.numBatches != 1 || width != td.cols || rows == 0 || coreCount == 0) {
        return rows;
    }
    uint64_t tiles = 0;
    if (!CheckedCeilDiv(td.rows, rows, tiles) || tiles <= coreCount) {
        return rows;
    }
    uint64_t rounds = 0;
    uint64_t slots = 0;
    if (!CheckedCeilDiv(tiles, coreCount, rounds) || !CheckedMulAdd(rounds, coreCount, 0, slots)) {
        return rows;
    }
    uint64_t candidate = 0;
    if (!CheckedCeilDiv(td.rows, slots, candidate) || candidate == 0 || candidate > rows) {
        return rows;
    }
    uint64_t originalWorst = 0;
    uint64_t candidateWorst = 0;
    if (!CheckedWorstRows(td.rows, rows, coreCount, originalWorst) ||
        !CheckedWorstRows(td.rows, candidate, coreCount, candidateWorst)) {
        return rows;
    }
    // 不增加每核处理的最多分片数，仅在最大轮转行量严格减少时采用新分块。
    return candidateWorst < originalWorst ? candidate : rows;
}

aclsparseStatus_t GetCompressResources(uint32_t& coreCount, uint64_t& ubBytes)
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    if (platform->GetSocVersion() != platform_ascendc::SocVersion::ASCEND950) {
        return ACL_SPARSE_STATUS_ARCH_MISMATCH;
    }
    coreCount = GetAivCoreCount();
    ubBytes = std::min(GetUbSize(), kUbLimit);
    if (coreCount == 0 || ubBytes <= kUbReserve) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t DeriveTiling(CompressParams& params)
{
    uint32_t coreCount = 0;
    uint64_t ubBytes = 0;
    const auto status = GetCompressResources(coreCount, ubBytes);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    auto& td = params.tiling;
    const uint64_t rowStep = td.alongRow ? 1 : td.groupSize;
    const uint64_t widthStep = td.alongRow ? 2 * td.groupSize : 2;
    const uint64_t lanes = kVectorBytes / td.dataBytes;
    uint64_t width = td.cols;
    if (width == 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    while (TileFootprint(td, width, rowStep) > ubBytes && width > widthStep) {
        const uint64_t step = width > 2 * lanes ? lanes : widthStep;
        width = std::max(widthStep, (width / 2 / step) * step);
    }
    if (TileFootprint(td, width, rowStep) > ubBytes) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    const uint64_t fixedBytes = 2 * kVectorBytes + kUbReserve;
    const uint64_t unitBytes = TileFootprint(td, width, rowStep) - fixedBytes;
    const uint64_t capacityRows = std::min(td.rows / rowStep, (ubBytes - fixedBytes) / unitBytes) * rowStep;
    // 全部物理行已放入单个行分段时，无需再次评估核间行量。
    const uint64_t rows = td.rows <= capacityRows ? capacityRows :
        BalanceTileRows(td, width, capacityRows, coreCount);
    if (rows == 0) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    const uint64_t fragments = td.alongRow ? rows : rows / td.groupSize;
    const uint64_t groups = td.alongRow ? width / td.groupSize : width;
    td.tileCols = static_cast<uint32_t>(width);
    td.tileRows = static_cast<uint32_t>(rows);
    td.inputPitch = static_cast<uint32_t>(AlignVector(width * td.dataBytes));
    td.valuePitch = static_cast<uint32_t>(AlignVector(groups * (td.groupSize / 2) * td.dataBytes));
    td.metadataPitch = static_cast<uint32_t>(AlignVector(groups / 2));
    td.inputBufferBytes = static_cast<uint32_t>(rows * td.inputPitch);
    td.valueBufferBytes = static_cast<uint32_t>(fragments * td.valuePitch + kVectorBytes);
    td.metadataBufferBytes = static_cast<uint32_t>(fragments * td.metadataPitch + kVectorBytes);
    td.tilesPerRowBand = (td.cols + width - 1) / width;
    td.rowBandsPerBatch = (td.rows + rows - 1) / rows;
    if (!CheckedMulAdd(td.tilesPerRowBand, td.rowBandsPerBatch, 0, td.totalTiles) ||
        !CheckedMulAdd(td.totalTiles, td.numBatches, 0, td.totalTiles)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    params.blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreCount, td.totalTiles));
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpMmaCompressParams(const aclsparseLtMatmulPlan_t* plan, CompressParams& params)
{
    const aclsparseLtMatDescriptor* sparse = nullptr;
    aclsparseOperation_t op = ACL_SPARSE_OP_NON_TRANSPOSE;
    auto status = SelectSparseDescriptor(plan, sparse, op);
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        status = DeriveShape(*sparse, op, params);
    }
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        status = DeriveSizes(params.tiling);
    }
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        status = DeriveTiling(params);
    }
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE(kLogTag, "Invalid compression descriptor or platform: status=%d", static_cast<int>(status));
    }
    return status;
}

aclsparseStatus_t ValidateBuffers(const void* dense, void* compressed, const CompressParams& params)
{
    const auto input = reinterpret_cast<uintptr_t>(dense);
    const auto output = reinterpret_cast<uintptr_t>(compressed);
    const auto& td = params.tiling;
    const uint64_t outputBytes = td.valuesBytes + td.metadataBytes;
    if (dense == nullptr || compressed == nullptr || input % params.alignment != 0 ||
        output % params.alignment != 0 || td.inputBytes > std::numeric_limits<uintptr_t>::max() - input ||
        outputBytes > std::numeric_limits<uintptr_t>::max() - output) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (input < output + outputBytes && output < input + td.inputBytes) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t LaunchSpMmaCompressKernel(const void* dense, void* compressed,
    const CompressParams& params, aclrtStream stream)
{
    // 保留调用者的线程错误状态。已有错误会妨碍判断本次启动是否失败，
    // 因此在提交任务前返回失败。
    const aclError previousError = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (previousError != ACL_SUCCESS) {
        OP_LOGE(kLogTag, "Pre-existing thread runtime error; compression not enqueued: %d", previousError);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    OP_LOGD(kLogTag, "tileRows=%u tileCols=%u inputPitch=%u valuePitch=%u metadataPitch=%u",
        params.tiling.tileRows, params.tiling.tileCols, params.tiling.inputPitch,
        params.tiling.valuePitch, params.tiling.metadataPitch);
    OP_LOGI(kLogTag, "Launching compression with %u AIV blocks", params.blockDim);
    spmma_compress_kernel_do(static_cast<uint8_t*>(const_cast<void*>(dense)),
        static_cast<uint8_t*>(compressed), params.blockDim, params.tiling, stream);
    // 编译器生成的直接启动函数返回 void，因此读取线程运行时错误判断即时失败。
    // 此处不清除错误，也不等待异步执行完成。
    const aclError launchError = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (launchError != ACL_SUCCESS) {
        OP_LOGE(kLogTag, "Runtime error observed after compression launch: %d", launchError);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
} // namespace

extern "C" aclsparseStatus_t aclsparseLtSpMMACompressedSize(const aclsparseLtHandle_t* handle,
    const aclsparseLtMatmulPlan_t* plan, size_t* compressedSize, size_t* compressBufferSize)
{
    if (handle == nullptr || *handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (compressedSize == nullptr || compressBufferSize == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    CompressParams params{};
    const auto status = ValidateSpMmaCompressParams(plan, params);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    *compressedSize = static_cast<size_t>(params.tiling.valuesBytes + params.tiling.metadataBytes);
    *compressBufferSize = 0;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseLtSpMMACompress(const aclsparseLtHandle_t* handle,
    const aclsparseLtMatmulPlan_t* plan, const void* d_dense, void* d_compressed,
    void* d_compressed_buffer, aclrtStream stream)
{
    if (handle == nullptr || *handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    (void)d_compressed_buffer;
    CompressParams params{};
    auto status = ValidateSpMmaCompressParams(plan, params);
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        status = ValidateBuffers(d_dense, d_compressed, params);
    }
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE(kLogTag, "Compression validation failed: status=%d", static_cast<int>(status));
        return status;
    }
    return LaunchSpMmaCompressKernel(d_dense, d_compressed, params, stream);
}
