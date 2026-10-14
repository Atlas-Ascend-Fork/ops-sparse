/**
 * Copyright (c) 2026 liutaw.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "securec.h"
#include <mutex>
#include <new>
#include <tuple>
#include "aclsparse_descr_internal.h"
#include "aclsparse_host_utils.h"
#include "spsm.h"

namespace {
std::mutex gDescriptorMutex;
aclsparseSpSMDescr *gDescriptors = nullptr;
constexpr auto kInvalid = ACL_SPARSE_STATUS_INVALID_VALUE;
constexpr auto kSuccess = ACL_SPARSE_STATUS_SUCCESS;

struct Call {
    aclsparseHandle_t handle;
    aclsparseOperation_t opA, opB;
    const void *alpha;
    aclsparseConstSpMatDescr_t a;
    aclsparseConstDnMatDescr_t b;
    aclsparseDnMatDescr_t c;
    aclDataType type;
    aclsparseSpSMAlg_t alg;
};

bool IsLive(aclsparseSpSMDescr_t descriptor)
{
    for (auto *p = gDescriptors; p != nullptr; p = p->next) {
        if (p == descriptor) { return true; }
    }
    return false;
}

// alignment is rejected when zero: every caller passes a positive literal, but the
// parameter itself carries no such guarantee and the remainder below would otherwise
// be undefined. A zero alignment describes no valid placement, so the pointer fails.
bool IsDevicePointer(const void *pointer, int32_t device, size_t alignment)
{
    // Pointer attributes establish device/location and starting alignment only;
    // they cannot prove the allocation is large enough for descriptor dimensions.
    if (pointer == nullptr || alignment == 0 ||
        reinterpret_cast<uintptr_t>(pointer) % alignment != 0) { return false; }
    aclrtPtrAttributes attr{};
    return aclrtPointerGetAttributes(pointer, &attr) == ACL_SUCCESS &&
        attr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE && attr.location.id == static_cast<uint32_t>(device);
}

bool IsHostPointer(const void *pointer)
{
    if (pointer == nullptr) { return false; }
    aclrtPtrAttributes attr{};
    return aclrtPointerGetAttributes(pointer, &attr) == ACL_SUCCESS &&
        attr.location.type != ACL_MEM_LOCATION_TYPE_DEVICE;
}

bool SameConfig(const spsm_a2::PlanConfig &a, const spsm_a2::PlanConfig &b)
{
    return std::tie(a.rows, a.rhs, a.nnz, a.base, a.format, a.opA, a.opB, a.upper,
        a.unit, a.complex, a.blocks, a.deviceAlpha, a.bRowStride, a.bColStride, a.cRowStride, a.cColStride) ==
        std::tie(b.rows, b.rhs, b.nnz, b.base, b.format, b.opA, b.opB, b.upper,
        b.unit, b.complex, b.blocks, b.deviceAlpha, b.bRowStride, b.bColStride, b.cRowStride, b.cColStride);
}

bool SameBinding(const SpsmA2Binding &a, const SpsmA2Binding &b)
{
    // Analysis binds handle, stream, pointer mode, device, descriptor identities,
    // shape/strides/format/operations and sparse pointers. Changed bindings need
    // Analysis again. Callers must keep descriptors alive; reused addresses are
    // not distinguishable from the original objects by this identity check.
    return a.handle == b.handle && a.matA == b.matA && a.matB == b.matB && a.matC == b.matC &&
        a.stream == b.stream && a.device == b.device && a.first == b.first && a.second == b.second &&
        a.aValues == b.aValues && (a.bValues == nullptr || a.bValues == b.bValues) &&
        (a.cValues == nullptr || a.cValues == b.cValues) && SameConfig(a.config, b.config) &&
        std::tie(a.bRows, a.bCols, a.bLd, a.bOrder, a.cRows, a.cCols, a.cLd, a.cOrder) ==
        std::tie(b.bRows, b.bCols, b.bLd, b.bOrder, b.cRows, b.cCols, b.cLd, b.cOrder);
}

bool ValidDense(const aclsparseDnMatDescr *d)
{
    if (d == nullptr || d->signature != kDnMatSignature || d->rows < 0 || d->cols < 0 || d->ld < 0) { return false; }
    if (d->order != ACL_SPARSE_ORDER_ROW && d->order != ACL_SPARSE_ORDER_COL) { return false; }
    return d->ld >= (d->order == ACL_SPARSE_ORDER_ROW ? d->cols : d->rows);
}

aclsparseStatus_t ValidateSparse(const aclsparseSpMatDescr *a, spsm_a2::PlanConfig &config)
{
    if (a->format == ACL_SPARSE_FORMAT_CSR) { config.format = 0; }
    else if (a->format == ACL_SPARSE_FORMAT_CSC) { config.format = 1; }
    else if (a->format == ACL_SPARSE_FORMAT_COO) { config.format = 2; }
    else { return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED; }
    if (a->ptrType != ACL_SPARSE_INDEX_32I || a->IdxType != ACL_SPARSE_INDEX_32I) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (a->baseType != ACL_SPARSE_INDEX_BASE_ZERO && a->baseType != ACL_SPARSE_INDEX_BASE_ONE) { return kInvalid; }
    if (a->fillMode != ACL_SPARSE_FILL_MODE_LOWER && a->fillMode != ACL_SPARSE_FILL_MODE_UPPER) { return kInvalid; }
    if (a->diagType != ACL_SPARSE_DIAG_TYPE_NON_UNIT && a->diagType != ACL_SPARSE_DIAG_TYPE_UNIT) { return kInvalid; }
    config.base = a->baseType == ACL_SPARSE_INDEX_BASE_ONE;
    if (a->nnz > static_cast<uint64_t>(INT32_MAX - config.base)) { return kInvalid; }
    config.upper = a->fillMode == ACL_SPARSE_FILL_MODE_UPPER;
    config.unit = a->diagType == ACL_SPARSE_DIAG_TYPE_UNIT;
    return kSuccess;
}

// Operation, algorithm and compute type; runs after descriptor presence is checked.
aclsparseStatus_t ValidateCallEnums(const Call &call)
{
    if (call.opA < ACL_SPARSE_OP_NON_TRANSPOSE || call.opA > ACL_SPARSE_OP_CONJUGATE_TRANSPOSE ||
        call.opB < ACL_SPARSE_OP_NON_TRANSPOSE || call.opB > ACL_SPARSE_OP_CONJUGATE_TRANSPOSE ||
        call.alg != ACL_SPARSE_SPSM_ALG_DEFAULT) { return kInvalid; }
    if (call.type != ACL_FLOAT && call.type != ACL_COMPLEX64) { return ACL_SPARSE_STATUS_NOT_SUPPORTED; }
    return kSuccess;
}

// Value types and equation dimensions; descriptors are non-null and B/C signatures were checked.
aclsparseStatus_t ValidateDims(const Call &call)
{
    const auto *a = call.a;
    const auto *b = call.b;
    const auto *c = call.c;
    if (a->valueType != call.type || b->valueType != call.type || c->valueType != call.type) { return kInvalid; }
    if (a->rows != a->cols || a->rows > INT32_MAX || a->nnz > INT32_MAX || c->cols > INT32_MAX) { return kInvalid; }
    int64_t bRows = call.opB == ACL_SPARSE_OP_NON_TRANSPOSE ? b->rows : b->cols;
    int64_t bCols = call.opB == ACL_SPARSE_OP_NON_TRANSPOSE ? b->cols : b->rows;
    if (bRows != static_cast<int64_t>(a->rows) || c->rows != bRows || c->cols != bCols) { return kInvalid; }
    return kSuccess;
}

aclsparseStatus_t ValidateOperands(const Call &call)
{
    if (call.a == nullptr || !ValidDense(call.b) || !ValidDense(call.c)) { return kInvalid; }
    auto status = ValidateCallEnums(call);
    if (status != kSuccess) { return status; }
    return ValidateDims(call);
}

// Copies validated shape, layout and pointer metadata; ValidateSparse has set format.
void FillShapeBinding(const Call &call, SpsmA2Binding &binding)
{
    const auto *a = call.a;
    const auto *b = call.b;
    const auto *c = call.c;
    auto &config = binding.config;
    config.rows = a->rows;
    config.nnz = a->nnz;
    config.rhs = c->cols;
    config.opA = call.opA;
    config.opB = call.opB;
    config.complex = call.type == ACL_COMPLEX64;
    config.bRowStride = b->order == ACL_SPARSE_ORDER_ROW ? b->ld : 1;
    config.bColStride = b->order == ACL_SPARSE_ORDER_ROW ? 1 : b->ld;
    if (config.opB != 0) { std::swap(config.bRowStride, config.bColStride); }
    config.cRowStride = c->order == ACL_SPARSE_ORDER_ROW ? c->ld : 1;
    config.cColStride = c->order == ACL_SPARSE_ORDER_ROW ? 1 : c->ld;
    binding.first = config.format == 2 ? a->rowInds : a->ptrs;
    binding.second = config.format == 2 ? a->colInds : a->idxs;
    binding.aValues = a->values;
    binding.bValues = b->values;
    binding.cValues = c->values;
    binding.bRows = b->rows;
    binding.bCols = b->cols;
    binding.bLd = b->ld;
    binding.bOrder = b->order;
    binding.cRows = c->rows;
    binding.cCols = c->cols;
    binding.cLd = c->ld;
    binding.cOrder = c->order;
}

aclsparseStatus_t ExtractShape(const Call &call, SpsmA2Binding &binding)
{
    auto status = ValidateOperands(call);
    if (status != kSuccess) { return status; }
    status = ValidateSparse(call.a, binding.config);
    if (status != kSuccess) { return status; }
    FillShapeBinding(call, binding);
    return kSuccess;
}

// Handle state is queried in the order stream, pointer mode, current device.
aclsparseStatus_t BindRuntime(const Call &call, SpsmA2Binding &binding)
{
    binding.handle = call.handle;
    binding.matA = call.a;
    binding.matB = call.b;
    binding.matC = call.c;
    aclsparsePointerMode_t mode = ACL_SPARSE_POINTER_MODE_HOST;
    auto status = aclsparseGetStream(call.handle, &binding.stream);
    if (status != kSuccess) { return status; }
    status = aclsparseGetPointerMode(call.handle, &mode);
    if (status != kSuccess) { return status; }
    if (mode != ACL_SPARSE_POINTER_MODE_HOST && mode != ACL_SPARSE_POINTER_MODE_DEVICE) { return kInvalid; }
    if (aclrtGetDevice(&binding.device) != ACL_SUCCESS) { return ACL_SPARSE_STATUS_NOT_INITIALIZED; }
    auto &config = binding.config;
    config.deviceAlpha = mode == ACL_SPARSE_POINTER_MODE_DEVICE;
    config.blocks = GetAivCoreCount();
    if (!spsm_a2::BuildWorkspace(config, binding.layout)) { return kInvalid; }
    return kSuccess;
}

bool ValidDevicePointers(const Call &call, const SpsmA2Binding &binding, bool needValues)
{
    const auto &config = binding.config;
    if (config.deviceAlpha ? !IsDevicePointer(call.alpha, binding.device, 4) : !IsHostPointer(call.alpha)) {
        return false;
    }
    if ((config.format != 2 || config.nnz != 0) && !IsDevicePointer(binding.first, binding.device, 4)) {
        return false;
    }
    if (config.nnz != 0 && (!IsDevicePointer(binding.second, binding.device, 4) ||
        !IsDevicePointer(binding.aValues, binding.device, 4))) { return false; }
    if (needValues && config.rows != 0 && config.rhs != 0 &&
        (!IsDevicePointer(binding.bValues, binding.device, 4) ||
        !IsDevicePointer(binding.cValues, binding.device, 4))) { return false; }
    return true;
}

aclsparseStatus_t Extract(const Call &call, SpsmA2Binding &binding, bool needValues)
{
    if (call.handle == nullptr) { return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR; }
    if (call.alpha == nullptr) { return kInvalid; }
    auto status = ExtractShape(call, binding);
    if (status != kSuccess) { return status; }
    status = BindRuntime(call, binding);
    if (status != kSuccess) { return status; }
    if (!ValidDevicePointers(call, binding, needValues)) { return kInvalid; }
    return kSuccess;
}

size_t DenseBytes(const spsm_a2::PlanConfig &c, bool output)
{
    if (c.rows == 0 || c.rhs == 0) { return 0; }
    const uint64_t row = output ? c.cRowStride : c.bRowStride;
    const uint64_t col = output ? c.cColStride : c.bColStride;
    return ((c.rows - 1) * row + (c.rhs - 1) * col + 1) * (c.complex ? 8 : 4);
}

bool Overlap(const void *a, size_t na, const void *b, size_t nb)
{
    if (a == nullptr || b == nullptr || na == 0 || nb == 0) { return false; }
    uintptr_t left = reinterpret_cast<uintptr_t>(a), right = reinterpret_cast<uintptr_t>(b);
    if (na > UINTPTR_MAX - left || nb > UINTPTR_MAX - right) { return true; }
    return left < right + nb && right < left + na;
}

bool TouchesSparse(const SpsmA2Binding &b, const void *pointer, size_t bytes)
{
    const auto &c = b.config;
    size_t firstCount = c.format == 2 ? c.nnz : static_cast<size_t>(c.rows) + 1;
    return Overlap(pointer, bytes, b.first, firstCount * 4) ||
        Overlap(pointer, bytes, b.second, static_cast<size_t>(c.nnz) * 4) ||
        Overlap(pointer, bytes, b.aValues, static_cast<size_t>(c.nnz) * (c.complex ? 8 : 4));
}

bool ValidWorkspace(const SpsmA2Binding &b, const void *workspace)
{
    // Workspace is current-device memory, aligned to 64 bytes, disjoint from
    // A/B/C and at least BufferSize bytes. Sparse/dense values and device alpha
    // use 4-byte alignment. Solve also rejects overlap with device alpha.
    return IsDevicePointer(workspace, b.device, 64) && !TouchesSparse(b, workspace, b.layout.bytes) &&
        !Overlap(workspace, b.layout.bytes, b.bValues, DenseBytes(b.config, false)) &&
        !Overlap(workspace, b.layout.bytes, b.cValues, DenseBytes(b.config, true));
}

bool ValidOutput(const SpsmA2Binding &b)
{
    // B=C is supported, including different layouts; partial overlap and overlap
    // with A are rejected. Solve copies op(B) to workspace before writing C.
    const size_t bytes = DenseBytes(b.config, true);
    if (TouchesSparse(b, b.cValues, bytes)) { return false; }
    return b.bValues == b.cValues || !Overlap(b.bValues, DenseBytes(b.config, false), b.cValues, bytes);
}

aclsparseStatus_t Fail(aclsparseStatus_t status, const char *stage)
{
    OP_LOGE("aclsparseSpSM", "%s rejected: status=%d", stage, static_cast<int>(status));
    return status;
}
}

aclsparseStatus_t aclsparseSpSMCreateDescr(aclsparseSpSMDescr_t *output)
{
    if (output == nullptr) { return Fail(kInvalid, "Create"); }
    auto *descriptor = new (std::nothrow) aclsparseSpSMDescr();
    if (descriptor == nullptr) { return Fail(ACL_SPARSE_STATUS_ALLOC_FAILED, "Create"); }
    std::lock_guard<std::mutex> lock(gDescriptorMutex);
    descriptor->next = gDescriptors; gDescriptors = descriptor; *output = descriptor;
    return kSuccess;
}

aclsparseStatus_t aclsparseSpSMDestroyDescr(aclsparseSpSMDescr_t descriptor)
{
    // Destroy does not wait for queued work and owns no device allocation.
    // A/B/C, workspace, device alpha and update values must remain valid until
    // their stream work completes, even after this descriptor is destroyed.
    std::lock_guard<std::mutex> lock(gDescriptorMutex);
    auto **link = &gDescriptors;
    while (*link != nullptr && *link != descriptor) { link = &(*link)->next; }
    if (*link == nullptr) { return Fail(kInvalid, "Destroy"); }
    *link = descriptor->next; delete descriptor;
    return kSuccess;
}

aclsparseStatus_t aclsparseSpSMBufferSize(aclsparseHandle_t handle, aclsparseOperation_t opA,
    aclsparseOperation_t opB, const void *alpha, aclsparseConstSpMatDescr_t a,
    aclsparseConstDnMatDescr_t b, aclsparseDnMatDescr_t c, aclDataType type,
    aclsparseSpSMAlg_t alg, aclsparseSpSMDescr_t descriptor, size_t *bytes)
{
    std::lock_guard<std::mutex> lock(gDescriptorMutex);
    // Failure leaves *bytes untouched. B/C values may be supplied after Analysis;
    // alpha is checked for placement here but its numerical value is not read.
    if (!IsLive(descriptor) || bytes == nullptr) { return Fail(kInvalid, "BufferSize"); }
    SpsmA2Binding binding{};
    auto status = Extract({handle, opA, opB, alpha, a, b, c, type, alg}, binding, false);
    if (status != kSuccess) { return Fail(status, "BufferSize"); }
    *bytes = binding.layout.bytes;
    return kSuccess;
}

aclsparseStatus_t aclsparseSpSMAnalysis(aclsparseHandle_t handle, aclsparseOperation_t opA,
    aclsparseOperation_t opB, const void *alpha, aclsparseConstSpMatDescr_t a,
    aclsparseConstDnMatDescr_t b, aclsparseDnMatDescr_t c, aclDataType type,
    aclsparseSpSMAlg_t alg, aclsparseSpSMDescr_t descriptor, void *workspace)
{
    std::lock_guard<std::mutex> lock(gDescriptorMutex);
    // Host rejection queues no work and preserves the prior plan. Device-data
    // rejection invalidates it after preprocessing; C is never written here.
    // The caller must retain the old bound resources when reusing a preserved plan.
    if (!IsLive(descriptor)) { return Fail(kInvalid, "Analysis"); }
    SpsmA2Binding binding{};
    auto status = Extract({handle, opA, opB, alpha, a, b, c, type, alg}, binding, false);
    if (status != kSuccess) { return Fail(status, "Analysis"); }
    if (!ValidWorkspace(binding, workspace)) { return Fail(kInvalid, "Analysis workspace"); }
    spsm_a2::Prepare(binding.config, binding.layout, binding.first, binding.second, binding.aValues, workspace,
        false, binding.stream);
    // The preprocessing kernels record per-block validation results in the workspace
    // status area. Analysis reads them back synchronously so an invalid matrix is
    // reported here with a definite code instead of silently skipping every later
    // stage; Solve stays asynchronous because it only runs on data that passed.
    bool invalid = false;
    if (!spsm_a2::ReadStatus(binding.config, binding.layout, workspace, binding.stream, invalid)) {
        descriptor->analyzed = false;
        return Fail(ACL_SPARSE_STATUS_INTERNAL_ERROR, "Analysis status readback");
    }
    if (invalid) {
        descriptor->analyzed = false;
        return Fail(kInvalid, "Analysis matrix data");
    }
    // A canonical matrix holding only its diagonal has no row-to-row dependency, so
    // Solve can spread rows over every core instead of the single RHS tile the general
    // path is limited to. The structure is fixed for the life of the plan, so this is
    // decided once here; UpdateMatrix only replaces values.
    bool diagonalOnly = false;
    if (!spsm_a2::DetectDiagonalOnly(binding.config, binding.layout, workspace, diagonalOnly)) {
        descriptor->analyzed = false;
        return Fail(ACL_SPARSE_STATUS_INTERNAL_ERROR, "Analysis structure readback");
    }
    binding.config.diagonalOnly = diagonalOnly ? 1 : 0;
    if (!spsm_a2::BuildLevels(binding.config, binding.layout, workspace, binding.stream)) {
        descriptor->analyzed = false;
        return Fail(ACL_SPARSE_STATUS_INTERNAL_ERROR, "Analysis dependency levels");
    }
    // Without duplicate coordinates the canonical layout is a permutation of the sorted
    // entries, so UpdateMatrix can refresh values through it instead of sorting and
    // merging again. The permutation needs saving before any solve borrows that space.
    bool unique = false;
    if (!spsm_a2::DetectNoDuplicates(binding.config, binding.layout, workspace, unique)) {
        descriptor->analyzed = false;
        return Fail(ACL_SPARSE_STATUS_INTERNAL_ERROR, "Analysis duplicate readback");
    }
    if (unique && !spsm_a2::SavePermutation(binding.config, binding.layout, workspace)) {
        descriptor->analyzed = false;
        return Fail(ACL_SPARSE_STATUS_INTERNAL_ERROR, "Analysis permutation copy");
    }
    binding.config.noDuplicates = unique ? 1 : 0;
    descriptor->binding = binding; descriptor->workspace = workspace; descriptor->analyzed = true;
    return kSuccess;
}

aclsparseStatus_t aclsparseSpSM(aclsparseHandle_t handle, aclsparseOperation_t opA,
    aclsparseOperation_t opB, const void *alpha, aclsparseConstSpMatDescr_t a,
    aclsparseConstDnMatDescr_t b, aclsparseDnMatDescr_t c, aclDataType type,
    aclsparseSpSMAlg_t alg, aclsparseSpSMDescr_t descriptor)
{
    std::lock_guard<std::mutex> lock(gDescriptorMutex);
    // SUCCESS means enqueued, not completed: Solve never synchronizes the host.
    // HOST alpha is copied during this call; DEVICE alpha is read by the kernel.
    // Analysis-null B/C pointers bind on the first successful Solve; pointers
    // already present during Analysis must not change. Host rejection preserves
    // the plan and enqueues no work.
    if (!IsLive(descriptor)) { return Fail(kInvalid, "Solve"); }
    if (!descriptor->analyzed) { return Fail(ACL_SPARSE_STATUS_NOT_INITIALIZED, "Solve"); }
    SpsmA2Binding binding{};
    auto status = Extract({handle, opA, opB, alpha, a, b, c, type, alg}, binding, true);
    if (status != kSuccess) { return Fail(status, "Solve"); }
    if (!SameBinding(descriptor->binding, binding) || !ValidWorkspace(binding, descriptor->workspace) ||
        !ValidOutput(binding) || (binding.config.deviceAlpha &&
        Overlap(alpha, binding.config.complex ? 8 : 4, descriptor->workspace, binding.layout.bytes))) {
        return Fail(kInvalid, "Solve binding");
    }
    // The freshly extracted binding describes the same plan (SameBinding just verified
    // it), but only Analysis could inspect the canonical structure. Carry its verdict
    // over so Solve takes the path Analysis decided on.
    binding.config.diagonalOnly = descriptor->binding.config.diagonalOnly;
    binding.config.independentRows = descriptor->binding.config.independentRows;
    binding.config.levelCount = descriptor->binding.config.levelCount;
    binding.config.levelParallelMask = descriptor->binding.config.levelParallelMask;
    spsm_a2::HostScalar hostAlpha{};
    void *deviceAlpha = nullptr;
    if (binding.config.deviceAlpha) { deviceAlpha = const_cast<void *>(alpha); }
    else if (memcpy_s(&hostAlpha, sizeof(hostAlpha), alpha, binding.config.complex ? 8 : 4) != EOK) {
        return Fail(ACL_SPARSE_STATUS_INTERNAL_ERROR, "Solve alpha copy");
    }
    spsm_a2::Solve(binding.config, binding.layout, binding.bValues, binding.cValues,
        deviceAlpha, hostAlpha, descriptor->workspace, binding.stream);
    descriptor->binding.bValues = binding.bValues; descriptor->binding.cValues = binding.cValues;
    return kSuccess;
}

aclsparseStatus_t aclsparseSpSMUpdateMatrix(aclsparseHandle_t handle, aclsparseSpSMDescr_t descriptor,
    const void *newValues, aclsparseSpSMUpdate_t part)
{
    std::lock_guard<std::mutex> lock(gDescriptorMutex);
    // newValues is the full original-order nnz array (before sort/merge); opA=H
    // conjugation happens on device. GENERAL replaces all values, DIAGONAL only
    // row==col entries, preserving the snapshot's off-diagonals. A is not modified.
    // Indices must stay unchanged; duplicate-free updates reuse structural checks,
    // but every update still checks NON_UNIT diagonals for zero. No alpha is read.
    if (!IsLive(descriptor)) { return Fail(kInvalid, "Update"); }
    if (!descriptor->analyzed) { return Fail(ACL_SPARSE_STATUS_NOT_INITIALIZED, "Update"); }
    if (part != ACL_SPARSE_SPSM_UPDATE_GENERAL && part != ACL_SPARSE_SPSM_UPDATE_DIAGONAL) {
        return Fail(kInvalid, "Update mode");
    }
    const auto &old = descriptor->binding;
    float placeholder[2] = {1, 0};
    // Extract validates alpha placement; a valid device scalar address is enough
    // here because Prepare does not read alpha.
    const void *alpha = old.config.deviceAlpha ? descriptor->workspace : placeholder;
    SpsmA2Binding binding{};
    auto status = Extract({handle, static_cast<aclsparseOperation_t>(old.config.opA),
        static_cast<aclsparseOperation_t>(old.config.opB), alpha, old.matA, old.matB, old.matC,
        old.config.complex ? ACL_COMPLEX64 : ACL_FLOAT, ACL_SPARSE_SPSM_ALG_DEFAULT}, binding, false);
    if (status != kSuccess) { return Fail(status, "Update"); }
    if (!SameBinding(old, binding) || !ValidWorkspace(binding, descriptor->workspace) ||
        !IsDevicePointer(newValues, binding.device, 4) || Overlap(newValues,
        static_cast<size_t>(binding.config.nnz) * (binding.config.complex ? 8 : 4), descriptor->workspace,
        binding.layout.bytes)) {
        return Fail(kInvalid, "Update binding/values");
    }
    // Structure is fixed for the life of the plan, so Analysis's findings still hold;
    // the binding rebuilt here does not carry them.
    binding.config.noDuplicates = descriptor->binding.config.noDuplicates;
    binding.config.diagonalOnly = descriptor->binding.config.diagonalOnly;
    binding.config.independentRows = descriptor->binding.config.independentRows;
    binding.config.levelCount = descriptor->binding.config.levelCount;
    binding.config.levelParallelMask = descriptor->binding.config.levelParallelMask;
    spsm_a2::Prepare(binding.config, binding.layout, binding.first, binding.second, newValues,
        descriptor->workspace, part == ACL_SPARSE_SPSM_UPDATE_DIAGONAL, binding.stream);
    // An update can turn a valid matrix invalid (a diagonal entry updated to zero under
    // NON_UNIT, for instance), so the same readback runs here. On failure the descriptor
    // stops being analyzed, which makes the next Solve refuse rather than use stale data.
    bool invalid = false;
    if (!spsm_a2::ReadStatus(binding.config, binding.layout, descriptor->workspace, binding.stream, invalid)) {
        descriptor->analyzed = false;
        return Fail(ACL_SPARSE_STATUS_INTERNAL_ERROR, "Update status readback");
    }
    if (invalid) {
        descriptor->analyzed = false;
        return Fail(kInvalid, "Update matrix data");
    }
    return kSuccess;
}
