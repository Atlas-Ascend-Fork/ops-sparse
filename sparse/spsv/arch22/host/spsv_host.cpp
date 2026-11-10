/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <climits>
#include <algorithm>
#include <limits>
#include <new>
#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_spsv_descr.h"
#include "spsv_host_utils.h"
#include "../kernel/spsv_kernel.h"
#include "spsv_torch_compat.h"

namespace {
using Status = aclsparseStatus_t;
constexpr Status kInvalid = ACL_SPARSE_STATUS_INVALID_VALUE;
constexpr Status kSuccess = ACL_SPARSE_STATUS_SUCCESS;

void WarnTorchInterop(const char* message) { OP_LOGW("aclsparseSpSV", "%s", message); }

struct SpsvA2State : aclsparseSpSVDescr {
    SpsvA2Tiling tiling{};
    size_t bytes = 0;
    aclsparseHandle_t handle = nullptr;
    aclrtStream stream = nullptr;
    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;
    aclsparseConstSpMatDescr_t mat = nullptr;
    aclsparseConstDnVecDescr_t x = nullptr;
    aclsparseDnVecDescr_t y = nullptr;
    void *inputPtr = nullptr;
    void *inputIdx = nullptr;
    void *inputValues = nullptr;
    void *xValues = nullptr;
    void *yValues = nullptr;
    uint64_t sliceWidth = 0;
    int64_t slices = 0;
    aclDataType type = ACL_FLOAT;
    aclsparseOperation_t operation = ACL_SPARSE_OP_NON_TRANSPOSE;
    aclsparseFillMode_t fill = ACL_SPARSE_FILL_MODE_LOWER;
    aclsparseDiagType_t diag = ACL_SPARSE_DIAG_TYPE_NON_UNIT;
    aclsparseIndexBase_t base = ACL_SPARSE_INDEX_BASE_ZERO;
    aclsparseFormat_t format = ACL_SPARSE_FORMAT_CSR;
};

Status DevicePointer(const void* pointer, size_t requiredBytes)
{
    if (pointer == nullptr || reinterpret_cast<uintptr_t>(pointer) % 4 != 0) return kInvalid;
    aclrtPtrAttributes attributes{};
    if (aclrtPointerGetAttributes(pointer, &attributes) != ACL_SUCCESS ||
        attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE) return kInvalid;
    int32_t device = -1;
    if (aclrtGetDevice(&device) != ACL_SUCCESS ||
        attributes.location.id != static_cast<uint32_t>(device)) return kInvalid;
    // Pointer attributes establish location/device, not allocation liveness.
    // Check the allocation span only when the public Runtime query supports it.
    const auto status = spsv_memory::ValidateRange(pointer, requiredBytes);
    if (status == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) return kSuccess;
    if (status != ACL_SUCCESS) {
        OP_LOGE("aclsparseSpSV", "device span validation failed: bytes=%zu, ret=%d", requiredBytes, status);
        return kInvalid;
    }
    return kSuccess;
}

size_t ValueBytes(aclDataType type) { return (type == ACL_COMPLEX64 ? 2 : 1) * sizeof(float); }

Status DeviceArray(const void* pointer, uint64_t count, size_t elementBytes)
{
    if (elementBytes == 0 || count > std::numeric_limits<size_t>::max() / elementBytes)
        return kInvalid;
    return DevicePointer(pointer, static_cast<size_t>(count) * elementBytes);
}

Status MatrixBuffers(aclsparseConstSpMatDescr_t mat)
{
    const uint64_t offsets = mat->format == ACL_SPARSE_FORMAT_COO ?
                                 mat->nnz :
                                 (mat->format == ACL_SPARSE_FORMAT_SLICED_ELL ? mat->numSlices : mat->rows) + 1;
    if (DeviceArray(mat->ptrs, offsets, sizeof(int32_t)) != kSuccess ||
        DeviceArray(mat->idxs, mat->nnz, sizeof(int32_t)) != kSuccess ||
        DeviceArray(mat->values, mat->nnz, ValueBytes(mat->valueType)) != kSuccess)
        return kInvalid;
    return kSuccess;
}

Status DenseVector(aclsparseConstDnVecDescr_t descriptor, const aclsparseSpMatDescr *mat,
    bool required, bool values)
{
    if (descriptor == nullptr) return required ? kInvalid : kSuccess;
    if (descriptor->valueType != mat->valueType || descriptor->nums != mat->rows ||
        descriptor->stride != 1) return kInvalid;
    return values && mat->rows != 0 ? DeviceArray(descriptor->values, mat->rows, ValueBytes(mat->valueType)) : kSuccess;
}

Status ValidateMatrixLayout(aclsparseConstSpMatDescr_t mat, aclDataType type)
{
    if (mat->valueType != type || mat->rows != mat->cols || (mat->rows == 0 && mat->nnz != 0) ||
        mat->rows > INT32_MAX || mat->nnz > INT32_MAX || mat->ptrStride != 1 || mat->idxStride != 1 ||
        mat->valueStride != 1)
        return kInvalid;
    if (mat->format != ACL_SPARSE_FORMAT_CSR && mat->format != ACL_SPARSE_FORMAT_CSC &&
        mat->format != ACL_SPARSE_FORMAT_COO && mat->format != ACL_SPARSE_FORMAT_SLICED_ELL)
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    return kSuccess;
}

Status ValidateMatrixAttributes(aclsparseConstSpMatDescr_t mat)
{
    if ((mat->baseType != ACL_SPARSE_INDEX_BASE_ZERO && mat->baseType != ACL_SPARSE_INDEX_BASE_ONE) ||
        (mat->fillMode != ACL_SPARSE_FILL_MODE_LOWER && mat->fillMode != ACL_SPARSE_FILL_MODE_UPPER) ||
        (mat->diagType != ACL_SPARSE_DIAG_TYPE_UNIT && mat->diagType != ACL_SPARSE_DIAG_TYPE_NON_UNIT))
        return kInvalid;
    if (mat->format == ACL_SPARSE_FORMAT_SLICED_ELL && mat->nnz != 0 &&
        (mat->sliceNnz == 0 || mat->sliceNnz > INT32_MAX || mat->numSlices <= 0 ||
            mat->numSlices > INT32_MAX)) return kInvalid;
    return kSuccess;
}

Status ValidateAlpha(aclsparseContext* context, const void* alpha, aclDataType type)
{
    if (context->pointerMode != ACL_SPARSE_POINTER_MODE_HOST &&
        context->pointerMode != ACL_SPARSE_POINTER_MODE_DEVICE) return kInvalid;
    if (reinterpret_cast<uintptr_t>(alpha) % alignof(float) != 0) return kInvalid;
    if (context->pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE && DevicePointer(alpha, ValueBytes(type)) != kSuccess)
        return kInvalid;
    if (context->pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
        aclrtPtrAttributes attributes{};
        if (aclrtPointerGetAttributes(alpha, &attributes) == ACL_SUCCESS &&
            attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE) return kInvalid;
    }
    return kSuccess;
}

Status Validate(aclsparseHandle_t handle, aclsparseOperation_t op, const void *alpha,
    aclsparseConstSpMatDescr_t mat, aclsparseConstDnVecDescr_t x,
    aclsparseDnVecDescr_t y, aclDataType type, aclsparseSpSVAlg_t alg,
    aclsparseSpSVDescr_t state, bool solve)
{
    if (handle == nullptr) return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    if (mat == nullptr || state == nullptr || alpha == nullptr) return kInvalid;
    if (spsv_common::ValidateOperation(op) != kSuccess) return kInvalid;
    if (!spsv_common::IsValueTypeSupported(type) || alg != ACL_SPARSE_SPSV_ALG_DEFAULT ||
        mat->ptrType != ACL_SPARSE_INDEX_32I || mat->IdxType != ACL_SPARSE_INDEX_32I)
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    Status status = ValidateMatrixLayout(mat, type);
    if (status != kSuccess) return status;
    status = ValidateMatrixAttributes(mat);
    if (status != kSuccess) return status;
    status = ValidateAlpha(reinterpret_cast<aclsparseContext*>(handle), alpha, type);
    if (status != kSuccess) return status;
    status = DenseVector(x, mat, solve, solve);
    return status != kSuccess ? status : DenseVector(y, mat, solve, solve);
}

bool Append(size_t count, size_t elementBytes, size_t &total, uint64_t &offset)
{
    if (elementBytes == 0 || count > (std::numeric_limits<size_t>::max() - 63) / elementBytes) return false;
    size_t bytes = (count * elementBytes + 63) / 64 * 64;
    if (total > std::numeric_limits<size_t>::max() - bytes) return false;
    offset = total;
    total += bytes;
    return true;
}

bool Layout(aclsparseConstSpMatDescr_t mat, aclsparseOperation_t operation,
    SpsvA2Tiling &tiling, size_t &size)
{
    tiling = {};
    tiling.m = mat->rows;
    tiling.nnz = mat->nnz;
    tiling.sourceNnz = mat->nnz;
    tiling.diagonalMatrix = mat->nnz == 0;
    tiling.components = mat->valueType == ACL_COMPLEX64 ? 2 : 1;
    tiling.wideTerms = mat->rows * tiling.components <= spsv_a2_ub::kResidentCapacity ? spsv_a2_ub::kWideCompactTerms :
                                                                                        spsv_a2_ub::kWideRegularTerms;
    tiling.format = mat->format == ACL_SPARSE_FORMAT_CSR ? 0 :
        (mat->format == ACL_SPARSE_FORMAT_CSC ? 1 :
            (mat->format == ACL_SPARSE_FORMAT_COO ? 2 : 3));
    tiling.transpose = operation != ACL_SPARSE_OP_NON_TRANSPOSE;
    tiling.conjugate = mat->valueType == ACL_COMPLEX64 && operation == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    tiling.lower = mat->fillMode == ACL_SPARSE_FILL_MODE_LOWER;
    tiling.unit = mat->diagType == ACL_SPARSE_DIAG_TYPE_UNIT;
    tiling.base = mat->baseType == ACL_SPARSE_INDEX_BASE_ONE;
    tiling.slices = mat->numSlices;
    tiling.sliceWidth = mat->sliceNnz;
    size = 0;
    if (mat->rows == 0 || mat->nnz == 0) return true;
    size = 512;
    return Append(mat->rows + 1, 4, size, tiling.rowOffset) && Append(mat->nnz, 4, size, tiling.colOffset) &&
           Append(mat->nnz, 4 * tiling.components, size, tiling.valueOffset) &&
           Append(mat->nnz, 4, size, tiling.permOffset) && Append(mat->rows, 4, size, tiling.scratchOffset) &&
           Append(mat->rows, sizeof(float) * spsv_a2_ub::SolutionWords(tiling.wideTerms) * tiling.components, size,
                  tiling.solutionOffset) &&
           Append(mat->rows, 4 * tiling.components, size, tiling.inputOffset) &&
           (!tiling.unit || mat->rows * tiling.components > spsv_a2_ub::kResidentCapacity ||
            Append(mat->rows, sizeof(float) * spsv_a2_ub::kUnitPrecisionStride * tiling.components, size,
                   tiling.precisionOffset));
}

void Bind(SpsvA2State &state, aclsparseHandle_t handle, aclsparseOperation_t operation,
    aclsparseConstSpMatDescr_t mat, aclsparseConstDnVecDescr_t x, aclsparseDnVecDescr_t y,
    void *workspace, const SpsvA2Tiling &tiling, size_t bytes)
{
    auto *context = reinterpret_cast<aclsparseContext *>(handle);
    state.tiling = tiling;
    state.bytes = bytes;
    state.workspaceBuffer = workspace;
    state.handle = handle;
    state.stream = context->stream;
    state.pointerMode = context->pointerMode;
    state.mat = mat;
    state.inputPtr = mat->ptrs;
    state.inputIdx = mat->idxs;
    state.inputValues = mat->values;
    state.currentValues = mat->values;
    state.x = x;
    state.y = y;
    state.xValues = x == nullptr ? nullptr : x->values;
    state.yValues = y == nullptr ? nullptr : y->values;
    state.type = mat->valueType;
    state.operation = operation;
    state.fill = mat->fillMode;
    state.diag = mat->diagType;
    state.base = mat->baseType;
    state.format = mat->format;
    state.sliceWidth = mat->sliceNnz;
    state.slices = mat->numSlices;
    state.cachedM = mat->rows;
    state.cachedNnz = mat->nnz;
    state.analysisLaunched = true;
}

bool MatchesBindingContext(const SpsvA2State &state, aclsparseHandle_t handle,
    aclsparseOperation_t operation, aclsparseConstSpMatDescr_t mat)
{
    auto *context = reinterpret_cast<aclsparseContext *>(handle);
    return state.analysisLaunched && handle == state.handle && context->stream == state.stream &&
        context->pointerMode == state.pointerMode && mat == state.mat && operation == state.operation;
}

bool MatchesBindingSource(const SpsvA2State &state, aclsparseConstSpMatDescr_t mat)
{
    return mat->rows == static_cast<uint64_t>(state.cachedM) &&
        mat->nnz == static_cast<uint64_t>(state.cachedNnz) && mat->ptrs == state.inputPtr &&
        mat->idxs == state.inputIdx && mat->values == state.inputValues;
}

bool MatchesBindingAttributes(const SpsvA2State &state, aclsparseConstSpMatDescr_t mat)
{
    return mat->valueType == state.type && mat->baseType == state.base && mat->format == state.format &&
        mat->fillMode == state.fill && mat->diagType == state.diag && mat->sliceNnz == state.sliceWidth &&
        mat->numSlices == state.slices && mat->rows == mat->cols && mat->ptrStride == 1 &&
        mat->idxStride == 1 && mat->valueStride == 1 &&
        mat->ptrType == ACL_SPARSE_INDEX_32I && mat->IdxType == ACL_SPARSE_INDEX_32I;
}

Status CheckBinding(SpsvA2State &state, aclsparseHandle_t handle,
    aclsparseOperation_t operation, aclsparseConstSpMatDescr_t mat)
{
    if (!MatchesBindingContext(state, handle, operation, mat) || !MatchesBindingSource(state, mat) ||
        !MatchesBindingAttributes(state, mat)) return kInvalid;
    if (state.bytes && DevicePointer(state.workspaceBuffer, state.bytes) != kSuccess) return kInvalid;
    return kSuccess;
}
}

extern "C" {
aclsparseStatus_t aclsparseSpSV_createDescr(aclsparseSpSVDescr_t *descriptor)
{
    if (descriptor == nullptr) return kInvalid;
    *descriptor = new (std::nothrow) SpsvA2State();
    return *descriptor == nullptr ? ACL_SPARSE_STATUS_ALLOC_FAILED : kSuccess;
}
aclsparseStatus_t aclsparseSpSV_destroyDescr(aclsparseSpSVDescr_t descriptor)
{
    delete static_cast<SpsvA2State *>(descriptor);
    return kSuccess;
}
aclsparseStatus_t aclsparseSpSV_bufferSize(aclsparseHandle_t handle,
    aclsparseOperation_t op, const void *alpha, aclsparseConstSpMatDescr_t mat,
    aclsparseConstDnVecDescr_t x, aclsparseDnVecDescr_t y, aclDataType type,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t descriptor, size_t *bytes)
{
    if (bytes == nullptr) return kInvalid;
    Status status = Validate(handle, op, alpha, mat, x, y, type, alg, descriptor, false);
    if (status != kSuccess) return status;
    SpsvA2Tiling tiling;
    return Layout(mat, op, tiling, *bytes) ? kSuccess : kInvalid;
}
aclsparseStatus_t aclsparseSpSV_analysis(aclsparseHandle_t handle,
    aclsparseOperation_t op, const void *alpha, aclsparseConstSpMatDescr_t mat,
    aclsparseConstDnVecDescr_t x, aclsparseDnVecDescr_t y, aclDataType type,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t descriptor, void *workspace)
{
    Status status = Validate(handle, op, alpha, mat, x, y, type, alg, descriptor, false);
    if (status != kSuccess) return status;
    auto &state = *static_cast<SpsvA2State *>(descriptor);
    state.analysisLaunched = false;
    SpsvA2Tiling tiling;
    size_t bytes = 0;
    if (!Layout(mat, op, tiling, bytes)) return kInvalid;
    if (bytes != 0) {
        if (reinterpret_cast<uintptr_t>(workspace) % 512 != 0 || DevicePointer(workspace, bytes) != kSuccess ||
            MatrixBuffers(mat) != kSuccess)
            return kInvalid;
        if (GetUbSize() < spsv_a2_ub::kRequiredBytes) return ACL_SPARSE_STATUS_NOT_SUPPORTED;
        auto *context = reinterpret_cast<aclsparseContext *>(handle);
        if (!spsv_torch_compat::Prepare(context->stream, WarnTorchInterop))
            return ACL_SPARSE_STATUS_INTERNAL_ERROR;
        SpsvA2Analysis(mat->ptrs, mat->idxs, mat->values, workspace, tiling, context->stream);
        // Immediate structural errors require a fixed-size status readback.
        // No matrix values or input-sized Host intermediates are copied back.
        if (aclrtSynchronizeStream(context->stream) != ACL_SUCCESS)
            return ACL_SPARSE_STATUS_INTERNAL_ERROR;
        int32_t result[3] = {0, 0, 0};
        if (aclrtMemcpy(result, sizeof(result), workspace, sizeof(result),
            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) return ACL_SPARSE_STATUS_INTERNAL_ERROR;
        if (result[0] != 0 || result[1] < 0 || result[1] > tiling.sourceNnz) return kInvalid;
        tiling.nnz = result[1];
        tiling.diagonalMatrix = result[2] == 1;
    }
    if (tiling.diagonalMatrix && tiling.m != 0) {
        uint32_t cores = GetAivCoreCount();
        if (cores == 0) return ACL_SPARSE_STATUS_NOT_SUPPORTED;
        uint32_t tiles = static_cast<uint32_t>((tiling.m + 127) / 128);
        tiling.blocks = cores < tiles ? cores : tiles;
    }
    Bind(state, handle, op, mat, x, y, workspace, tiling, bytes);
    return kSuccess;
}
aclsparseStatus_t aclsparseSpSV_solve(aclsparseHandle_t handle,
    aclsparseOperation_t op, const void *alpha, aclsparseConstSpMatDescr_t mat,
    aclsparseConstDnVecDescr_t x, aclsparseDnVecDescr_t y, aclDataType type,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t descriptor)
{
    Status status = Validate(handle, op, alpha, mat, x, y, type, alg, descriptor, true);
    if (status != kSuccess) return status;
    auto &state = *static_cast<SpsvA2State *>(descriptor);
    status = CheckBinding(state, handle, op, mat);
    if (status != kSuccess) return status;
    if ((state.x != nullptr && (x != state.x || x->values != state.xValues)) ||
        (state.y != nullptr && (y != state.y || y->values != state.yValues))) return kInvalid;
    if (x->values != y->values && mat->rows != 0) {
        uintptr_t xb = reinterpret_cast<uintptr_t>(x->values);
        uintptr_t yb = reinterpret_cast<uintptr_t>(y->values);
        size_t vectorBytes = mat->rows * 4 * state.tiling.components;
        if ((xb < yb && yb - xb < vectorBytes) || (yb < xb && xb - yb < vectorBytes)) return kInvalid;
    }
    // Only vectors explicitly supplied to Analysis are bound. An analysis
    // performed with null vectors is reusable for fresh solve inputs/outputs.
    if (mat->rows == 0) return kSuccess;
    SpsvA2Tiling tiling = state.tiling;
    if (state.pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        tiling.alphaDevice = reinterpret_cast<uintptr_t>(alpha);
    } else {
        float values[2] = {0, 0};
        const auto *source = static_cast<const unsigned char *>(alpha);
        auto *destination = reinterpret_cast<unsigned char *>(values);
        std::copy_n(source, sizeof(float) * tiling.components, destination);
        tiling.alphaReal = values[0];
        tiling.alphaImag = values[1];
    }
    if (!spsv_torch_compat::Prepare(state.stream, WarnTorchInterop))
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    SpsvA2Solve(x->values, y->values, state.workspaceBuffer, tiling, state.stream);
    return kSuccess;
}
aclsparseStatus_t aclsparseSpSV_updateMatrix(aclsparseHandle_t handle,
    aclsparseSpSVDescr_t descriptor, void *values, aclsparseSpSVUpdate_t part)
{
    if (handle == nullptr) return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    if (descriptor == nullptr || values == nullptr ||
        (part != ACL_SPARSE_SPSV_UPDATE_GENERAL && part != ACL_SPARSE_SPSV_UPDATE_DIAGONAL)) return kInvalid;
    auto &state = *static_cast<SpsvA2State *>(descriptor);
    if (!state.analysisLaunched || state.mat == nullptr) return kInvalid;
    Status status = CheckBinding(state, handle, state.operation, state.mat);
    const uint64_t count = part == ACL_SPARSE_SPSV_UPDATE_DIAGONAL ? state.tiling.m : state.tiling.sourceNnz;
    if (status != kSuccess || DeviceArray(values, count, ValueBytes(state.type)) != kSuccess)
        return kInvalid;
    if (!spsv_torch_compat::Prepare(state.stream, WarnTorchInterop))
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    if (state.tiling.nnz != 0) SpsvA2Update(values, state.workspaceBuffer,
        part == ACL_SPARSE_SPSV_UPDATE_DIAGONAL, state.tiling, state.stream);
    state.currentValues = values;
    state.updateMatrixCalled = true;
    return kSuccess;
}
}
