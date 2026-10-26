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
#include "sddmm_launch.h"
#include "sddmm.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <new>

namespace {
constexpr uint64_t kWorkspaceLimit = 64u * 1024u * 1024u;
using Key = std::array<uint64_t, 30>;
uint32_t Bytes(aclDataType t) { return t == ACL_COMPLEX64 ? 8 : t == ACL_FLOAT16 || t == ACL_BF16 ? 2 : 4; }
aclsparseStatus_t Runtime(aclError status)
{
    return status == ACL_SUCCESS ? ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_EXECUTION_FAILED;
}
struct Plan {
    uint32_t m = 0, n = 0, k = 0, bd = 1, rows = 0, count = 1, checkCores = 1;
    bool bsr = false, cube = false, packX = false, packY = false, smallBsr = false;
    uint64_t rowBytes = 0, colBytes = 0, patternStride = 0;
    uint64_t flags = 0, flagBytes = 0, expanded = 0, expandedStride = 0;
    uint64_t scalarValues = 0, packedX = 0, packedY = 0, scratch = 0, total = 0;
    uint64_t rowHints = 0;
    Key key{};
};
struct State {
    Plan plan;
    void* buffer = nullptr;
    aclrtEvent event = nullptr;
    aclrtEvent checkReady = nullptr;
    aclrtStream stream = nullptr;
    ~State()
    {
        if (checkReady) {
            (void)aclrtDestroyEvent(checkReady);
        }
        if (event) {
            (void)aclrtDestroyEvent(event);
        }
    }
};
struct Pinned {
    void* data = nullptr;
    size_t bytes = 0;
    ~Pinned()
    {
        if (data) {
            (void)aclrtFreeHost(data);
        }
    }
    bool Reserve(size_t size)
    {
        if (size <= bytes) {
            return true;
        }
        void* next = nullptr;
        if (aclrtMallocHost(&next, size) != ACL_SUCCESS) {
            return false;
        }
        if (data) {
            (void)aclrtFreeHost(data);
        }
        data = next;
        bytes = size;
        return true;
    }
};

bool DenseDimensionsValid(const aclsparseDnMatDescr* d)
{
    if (!d || d->signature != kDnMatSignature || d->rows < 0 || d->cols < 0 || d->rows > INT32_MAX ||
        d->cols > INT32_MAX || d->ld < 0 || (d->order != ACL_SPARSE_ORDER_ROW && d->order != ACL_SPARSE_ORDER_COL) ||
        d->batchCount < 1 || d->batchCount > 65535 || d->batchStride < 0) {
        return false;
    }
    return true;
}

bool DenseValid(const aclsparseDnMatDescr* d)
{
    if (!DenseDimensionsValid(d)) {
        return false;
    }
    const uint64_t minor = d->order == ACL_SPARSE_ORDER_ROW ? d->cols : d->rows;
    const uint64_t major = d->order == ACL_SPARSE_ORDER_ROW ? d->rows : d->cols;
    if (uint64_t(d->ld) < minor || uint64_t(d->ld) > UINT32_MAX / Bytes(d->valueType)) {
        return false;
    }
    const uint64_t span = minor && major ? (major - 1) * uint64_t(d->ld) + minor : 0;
    const uint64_t limit = uint64_t(INT64_MAX) / Bytes(d->valueType);
    if (span > limit || (span && !d->values)) {
        return false;
    }
    if (d->batchCount > 1 &&
        (uint64_t(d->batchStride) < span || uint64_t(d->batchStride) > (limit - span) / (d->batchCount - 1))) {
        return false;
    }
    return true;
}

bool Canonical(const aclsparseDnMatDescr* d, aclsparseOperation_t op)
{
    return (op == ACL_SPARSE_OP_NON_TRANSPOSE && d->order == ACL_SPARSE_ORDER_ROW && d->ld == d->cols) ||
           (op == ACL_SPARSE_OP_TRANSPOSE && d->order == ACL_SPARSE_ORDER_COL && d->ld == d->rows);
}

aclsparseDnMatDescr DenseView(const aclsparseDnMatDescr* d, uint32_t rows, uint32_t cols)
{
    auto view = *d;
    view.rows = rows;
    view.cols = cols;
    view.ld = cols;
    view.order = ACL_SPARSE_ORDER_ROW;
    view.batchCount = 1;
    view.batchStride = 0;
    return view;
}

void CsrView(aclsparseSpMatDescr& view, const aclsparseSpMatDescr* c, const Plan& p)
{
    view.format = ACL_SPARSE_FORMAT_CSR;
    view.rows = p.m;
    view.cols = p.n;
    view.nnz = c->nnz * p.bd * p.bd;
    view.ptrs = c->ptrs;
    view.idxs = c->idxs;
    view.values = c->values;
    view.baseType = p.bsr ? ACL_SPARSE_INDEX_BASE_ZERO : c->baseType;
    view.ptrType = view.IdxType = ACL_SPARSE_INDEX_32I;
    view.valueType = c->valueType;
}

class PlanBuilder {
public:
    PlanBuilder(aclsparseHandle_t handle, aclsparseOperation_t ox, aclsparseOperation_t oy, const void* alpha,
                const aclsparseDnMatDescr* x, const aclsparseDnMatDescr* y, const void* beta, aclsparseSpMatDescr* c,
                aclDataType compute, aclsparseSDDMMAlg_t alg, Plan& p)
        : handle(handle), ox(ox), oy(oy), alpha(alpha), x(x), y(y), beta(beta), c(c), compute(compute), alg(alg), p(p)
    {}
    aclsparseStatus_t Run()
    {
        using Stage = aclsparseStatus_t (PlanBuilder::*)();
        const Stage stages[] = {&PlanBuilder::ValidateHeaders, &PlanBuilder::ValidateSparse,
                                &PlanBuilder::ValidateTypes,   &PlanBuilder::ValidateShape,
                                &PlanBuilder::SelectKernels,   &PlanBuilder::BuildKey,
                                &PlanBuilder::AllocatePattern, &PlanBuilder::AllocateExpanded};
        for (auto stage : stages) {
            const auto status = (this->*stage)();
            if (status != ACL_SPARSE_STATUS_SUCCESS) {
                return status;
            }
        }
        return AllocateNumeric();
    }

private:
    aclsparseStatus_t ValidateHeaders()
    {
        if (!handle) {
            return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
        }
        if (!alpha || !beta || !DenseValid(x) || !DenseValid(y) || !c || c->signature != kSpMatSignature) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        if (alg != ACL_SPARSE_SDDMM_ALG_DEFAULT ||
            (ox != ACL_SPARSE_OP_NON_TRANSPOSE && ox != ACL_SPARSE_OP_TRANSPOSE) ||
            (oy != ACL_SPARSE_OP_NON_TRANSPOSE && oy != ACL_SPARSE_OP_TRANSPOSE) ||
            (c->format != ACL_SPARSE_FORMAT_CSR && c->format != ACL_SPARSE_FORMAT_BSR) ||
            c->ptrType != ACL_SPARSE_INDEX_32I || c->IdxType != ACL_SPARSE_INDEX_32I) {
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
        }
        const auto* h = SddmmToInternalHandle(handle);
        if (!h->stream ||
            (h->pointerMode != ACL_SPARSE_POINTER_MODE_HOST && h->pointerMode != ACL_SPARSE_POINTER_MODE_DEVICE)) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t ValidateSparse()
    {
        p.bsr = c->format == ACL_SPARSE_FORMAT_BSR;
        p.bd = p.bsr ? c->blockSize : 1;
        if (p.bsr && (p.bd < 2 || p.bd > 128 || (p.bd & (p.bd - 1)) ||
                      (c->blockOrder != ACL_SPARSE_ORDER_ROW && c->blockOrder != ACL_SPARSE_ORDER_COL))) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        return ValidateSparseStorage();
    }
    aclsparseStatus_t ValidateSparseStorage()
    {
        if (c->rows > INT32_MAX - 1 || c->cols > INT32_MAX || c->rows % p.bd || c->cols % p.bd ||
            (c->baseType != ACL_SPARSE_INDEX_BASE_ZERO && c->baseType != ACL_SPARSE_INDEX_BASE_ONE) ||
            c->nnz > uint64_t(INT32_MAX - 1) / (p.bd * p.bd) || !c->ptrs ||
            (c->nnz && (!c->idxs || !c->values || !c->rows || !c->cols))) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t ValidateTypes()
    {
        complex = x->valueType == ACL_COMPLEX64;
        if (!SddmmDtypePairSupported(x->valueType, c->valueType) || x->valueType != y->valueType ||
            !SddmmComputeTypeSupported(x->valueType, c->valueType, compute)) {
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t ValidateShape()
    {
        p.m = c->rows;
        p.n = c->cols;
        p.k = ox == ACL_SPARSE_OP_NON_TRANSPOSE ? x->cols : x->rows;
        p.rows = p.m / p.bd;
        p.count = c->batchCount;
        if ((ox == ACL_SPARSE_OP_NON_TRANSPOSE ? x->rows : x->cols) != p.m ||
            (oy == ACL_SPARSE_OP_NON_TRANSPOSE ? y->cols : y->rows) != p.n ||
            (oy == ACL_SPARSE_OP_NON_TRANSPOSE ? y->rows : y->cols) != p.k || p.count < 1 || p.count > 65535 ||
            (!p.bsr && p.count != 1) || (x->batchCount != 1 && x->batchCount != p.count) ||
            (y->batchCount != 1 && y->batchCount != p.count)) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        return ValidateBatchStorage();
    }
    aclsparseStatus_t ValidateBatchStorage()
    {
        if (complex &&
            (p.n > INT32_MAX / 2 || c->nnz * p.bd * p.bd > INT32_MAX / 2 || uint64_t(p.m) * p.k > UINT32_MAX)) {
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
        }
        if (p.count > 1 && (c->offsetsBatchStride < p.rows + 1 || c->columnsBatchStride < c->nnz ||
                            c->valuesBatchStride < c->nnz * p.bd * p.bd)) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    bool UseComplexVector() const
    {
        return p.bsr && complex && p.k > 0 && p.k <= 128 && uint64_t(p.n) * 8 <= UINT32_MAX &&
               ((p.bd == 4 || p.bd == 8) || (p.bd >= 16 && p.bd <= 32 && c->nnz <= 8));
    }
    aclsparseStatus_t SelectKernels()
    {
        const bool complexVector = UseComplexVector();
        p.cube = p.bsr && !complexVector && p.bd >= 16 && p.k > 0 && p.k <= 65472;
        // The direct small-block kernel removes the expanded-CSR workspace. Keep it
        // enabled for every supported real small-block shape; a density-only host
        // threshold is not sufficient because sparse rows can still avoid a large
        // expansion cost. Future dispatch may use preprocessed run-length statistics,
        // but must be based on measured continuity rather than nnz/row alone.
        p.smallBsr = complexVector ||
                     (p.bsr && !p.cube && p.bd <= 8 && p.k > 0 && p.k <= 256 &&
                      uint64_t(p.n) * Bytes(x->valueType) <= UINT32_MAX &&
                      (x->valueType == ACL_FLOAT16 || x->valueType == ACL_BF16 || x->valueType == ACL_FLOAT));
        p.packX = (p.bsr || complex) && !Canonical(x, ox);
        p.packY = (p.bsr || complex) && !Canonical(y, oy);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t BuildKey()
    {
        p.key = {uint64_t(c->format),
                 c->rows,
                 c->cols,
                 c->nnz,
                 c->blockSize,
                 uint64_t(c->blockOrder),
                 uint64_t(c->baseType),
                 uint64_t(c->valueType),
                 reinterpret_cast<uintptr_t>(c->ptrs),
                 reinterpret_cast<uintptr_t>(c->idxs),
                 uint64_t(c->batchCount),
                 uint64_t(c->offsetsBatchStride),
                 uint64_t(c->columnsBatchStride),
                 uint64_t(c->valuesBatchStride),
                 uint64_t(x->rows),
                 uint64_t(x->cols),
                 uint64_t(x->ld),
                 uint64_t(x->order),
                 uint64_t(x->valueType),
                 uint64_t(x->batchCount),
                 uint64_t(x->batchStride),
                 uint64_t(y->rows),
                 uint64_t(y->cols),
                 uint64_t(y->ld),
                 uint64_t(y->order),
                 uint64_t(y->batchCount),
                 uint64_t(y->batchStride),
                 uint64_t(ox),
                 uint64_t(oy),
                 uint64_t(compute)};
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t AllocatePattern()
    {
        p.rowBytes = SddmmAlign32(uint64_t(p.rows + 1) * 4);
        p.colBytes = SddmmAlign32(c->nnz * 4);
        p.patternStride = p.rowBytes + p.colBytes;
        p.total = p.patternStride * p.count;
        const uint64_t chunk = SddmmBudget::kPatternChunk;
        const uint64_t tasks = (p.rows + chunk) / chunk + (c->nnz + chunk - 1) / chunk;
        p.checkCores = std::min<uint64_t>(SddmmGetBlockDim(), tasks);
        p.flags = p.total;
        p.flagBytes = uint64_t(p.checkCores) * SddmmBudget::kFlagStrideBytes * p.count;
        p.total += p.flagBytes;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t AllocateExpanded()
    {
        if (!p.bsr && x->valueType == ACL_FLOAT16 && p.m >= 256 && p.k >= 32 && p.k <= SddmmBudget::kRowHintsKLimit &&
            c->nnz >= uint64_t(p.m) * 8) {
            p.rowHints = p.total;
            // One aligned boundary record per numeric core follows the row hints.
            p.total += SddmmAlign32(uint64_t(p.m) * 4) + uint64_t(SddmmGetBlockDim()) * 32;
        }
        if (p.bsr && !p.cube && !p.smallBsr) {
            p.expanded = p.total;
            p.expandedStride = SddmmAlign32(uint64_t(p.m + 1) * 4) + SddmmAlign32(c->nnz * p.bd * p.bd * 4);
            p.total += p.expandedStride * p.count;
            p.scalarValues = p.total;
            p.total += SddmmAlign32(c->nnz * p.bd * p.bd * Bytes(c->valueType));
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t AllocateNumeric()
    {
        p.packedX = p.total;
        if (p.packX) {
            p.total += SddmmAlign32(uint64_t(p.m) * p.k * Bytes(x->valueType));
        }
        p.packedY = p.total;
        if (p.packY) {
            p.total += SddmmAlign32(uint64_t(p.k) * p.n * Bytes(y->valueType));
        }
        p.scratch = p.total;
        if (c->nnz && p.k) {
            if (p.cube) {
                p.total += sddmm_bsr_cube_workspace(p.m, p.n, p.k, x->valueType);
            } else if (!p.smallBsr) {
                auto dx = p.bsr || complex ? DenseView(x, p.m, p.k) : *x;
                auto dy = p.bsr || complex ? DenseView(y, p.k, p.n) : *y;
                aclsparseSpMatDescr csr;
                CsrView(csr, c, p);
                size_t size = 0;
                const auto status = SddmmCsrBufferSize(handle, p.bsr || complex ? ACL_SPARSE_OP_NON_TRANSPOSE : ox,
                                                       p.bsr || complex ? ACL_SPARSE_OP_NON_TRANSPOSE : oy, alpha, &dx,
                                                       &dy, beta, &csr, compute, alg, &size);
                if (status != ACL_SPARSE_STATUS_SUCCESS) {
                    return status;
                }
                p.total += SddmmAlign32(size);
            }
        }
        return p.total <= kWorkspaceLimit ? ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    aclsparseHandle_t handle;
    aclsparseOperation_t ox, oy;
    const void* alpha;
    const aclsparseDnMatDescr *x, *y;
    const void* beta;
    aclsparseSpMatDescr* c;
    aclDataType compute;
    aclsparseSDDMMAlg_t alg;
    Plan& p;
    bool complex = false;
};

aclsparseStatus_t BuildPlan(aclsparseHandle_t handle, aclsparseOperation_t ox, aclsparseOperation_t oy,
                            const void* alpha, const aclsparseDnMatDescr* x, const aclsparseDnMatDescr* y,
                            const void* beta, aclsparseSpMatDescr* c, aclDataType compute, aclsparseSDDMMAlg_t alg,
                            Plan& p)
{
    return PlanBuilder(handle, ox, oy, alpha, x, y, beta, c, compute, alg, p).Run();
}

bool BufferValid(void* buffer) { return buffer && reinterpret_cast<uintptr_t>(buffer) % 32 == 0; }
uint8_t* Batch(void* ptr, int batch, int count, int64_t stride, uint32_t bytes)
{
    return ptr ? static_cast<uint8_t*>(ptr) + (count == 1 ? 0 : uint64_t(batch) * stride * bytes) : nullptr;
}

aclsparseStatus_t CopyDeviceScalars(aclsparseContext* h, const Plan& p, Pinned& host, const void* alpha,
                                    const void* beta, aclDataType compute, bool requested)
{
    if (requested && h->pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        const uint32_t bytes = Bytes(compute);
        auto status = aclrtMemcpyAsync(static_cast<uint8_t*>(host.data) + p.flagBytes, 16, alpha, bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST, h->stream);
        if (status == ACL_SUCCESS) {
            status = aclrtMemcpyAsync(static_cast<uint8_t*>(host.data) + p.flagBytes + 8, 8, beta, bytes,
                                      ACL_MEMCPY_DEVICE_TO_HOST, h->stream);
        }
        if (status != ACL_SUCCESS) {
            return Runtime(status);
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

void CopyHostScalars(const void* alpha, const void* beta, aclDataType compute, std::array<uint8_t, 16>& scalars)
{
    std::copy_n(static_cast<const uint8_t*>(alpha), Bytes(compute), scalars.data());
    std::copy_n(static_cast<const uint8_t*>(beta), Bytes(compute), scalars.data() + 8);
}

aclsparseStatus_t LaunchPatternChecks(aclsparseContext* h, aclsparseSpMatDescr* c, const Plan& p, void* buffer,
                                      bool compare)
{
    auto* w = static_cast<uint8_t*>(buffer);
    for (uint32_t b = 0; b < p.count; ++b) {
        const auto status = sddmm_pattern_check_launch(
            Batch(c->ptrs, b, p.count, c->offsetsBatchStride, 4), Batch(c->idxs, b, p.count, c->columnsBatchStride, 4),
            w + b * p.patternStride, w + b * p.patternStride + p.rowBytes,
            w + p.flags + b * p.checkCores * SddmmBudget::kFlagStrideBytes, p.rows, p.n / p.bd, c->nnz,
            c->baseType == ACL_SPARSE_INDEX_BASE_ONE, compare, p.checkCores, h->stream);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t WaitForValidation(aclrtStream stream)
{
    // Events belong to the current context. Do not cache one in thread-local
    // pinned storage, which can outlive aclrtResetDevice/aclFinalize.
    aclrtEvent ready = nullptr;
    auto status = aclrtCreateEventWithFlag(&ready, ACL_EVENT_SYNC);
    if (status != ACL_SUCCESS) {
        return Runtime(status);
    }
    status = aclrtRecordEvent(ready, stream);
    if (status == ACL_SUCCESS) {
        status = aclrtSynchronizeEvent(ready);
    }
    const auto destroyed = aclrtDestroyEvent(ready);
    return Runtime(status == ACL_SUCCESS ? destroyed : status);
}

aclsparseStatus_t CheckPatterns(aclsparseContext* h, aclsparseSpMatDescr* c, const Plan& p, void* buffer, bool compare,
                                const void* alpha = nullptr, const void* beta = nullptr,
                                aclDataType compute = ACL_FLOAT, std::array<uint8_t, 16>* scalars = nullptr,
                                aclrtEvent ready = nullptr, const int32_t** pendingFlags = nullptr)
{
    thread_local Pinned host;
    if (!host.Reserve(p.flagBytes + 32)) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    auto* w = static_cast<uint8_t*>(buffer);
    const auto checked = LaunchPatternChecks(h, c, p, buffer, compare);
    if (checked != ACL_SPARSE_STATUS_SUCCESS) {
        return checked;
    }
    auto status = aclrtMemcpyAsync(host.data, host.bytes, w + p.flags, p.flagBytes, ACL_MEMCPY_DEVICE_TO_HOST,
                                   h->stream);
    if (status != ACL_SUCCESS) {
        return Runtime(status);
    }
    const auto copied = CopyDeviceScalars(h, p, host, alpha, beta, compute, scalars != nullptr);
    if (copied != ACL_SPARSE_STATUS_SUCCESS) {
        return copied;
    }
    if (ready) {
        // HOST-scalar real CSR queues a guarded numeric kernel, then
        // waits for this earlier validation event, including without row hints. Thus
        // host wake-up latency overlaps computation without weakening errors.
        if (!scalars || !pendingFlags || h->pointerMode != ACL_SPARSE_POINTER_MODE_HOST) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        CopyHostScalars(alpha, beta, compute, *scalars);
        *pendingFlags = static_cast<const int32_t*>(host.data);
        return Runtime(aclrtRecordEvent(ready, h->stream));
    }
    const auto waited = WaitForValidation(h->stream);
    if (waited != ACL_SPARSE_STATUS_SUCCESS) {
        return waited;
    }
    const auto* flags = static_cast<const int32_t*>(host.data);
    for (uint64_t i = 0; i < p.flagBytes / 4; i += SddmmBudget::kFlagStrideBytes / sizeof(int32_t)) {
        if (flags[i]) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    if (scalars) {
        if (h->pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
            std::copy_n(static_cast<uint8_t*>(host.data) + p.flagBytes, 16, scalars->data());
        } else {
            CopyHostScalars(alpha, beta, compute, *scalars);
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t Pack(const aclsparseDnMatDescr* d, aclsparseOperation_t op, uint32_t rows, uint32_t cols,
                       uint8_t* source, uint8_t* dest, aclrtStream stream)
{
    uint32_t rowStep = d->order == ACL_SPARSE_ORDER_ROW ? d->ld : 1;
    uint32_t colStep = d->order == ACL_SPARSE_ORDER_ROW ? 1 : d->ld;
    if (op == ACL_SPARSE_OP_TRANSPOSE) {
        std::swap(rowStep, colStep);
    }
    return sddmm_pack_dense_launch(source, dest, rows, cols, rowStep, colStep, d->valueType, SddmmGetBlockDim(),
                                   stream);
}
} // namespace

aclsparseStatus_t aclsparseSDDMMBufferSize(aclsparseHandle_t h, aclsparseOperation_t ox, aclsparseOperation_t oy,
                                           const void* alpha, aclsparseConstDnMatDescr_t x,
                                           aclsparseConstDnMatDescr_t y, const void* beta, aclsparseSpMatDescr_t c,
                                           aclDataType compute, aclsparseSDDMMAlg_t alg, size_t* size)
{
    if (!size) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    Plan p;
    const auto status = BuildPlan(h, ox, oy, alpha, x, y, beta, c, compute, alg, p);
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        *size = p.total;
    }
    return status;
}

static aclsparseStatus_t PreparePatternState(aclsparseContext* h, aclsparseSpMatDescr* c, const Plan& p, void* buffer)
{
    aclsparseStatus_t status = ACL_SPARSE_STATUS_SUCCESS;
    try {
        auto state = std::make_shared<State>();
        state->plan = p;
        state->buffer = buffer;
        state->stream = h->stream;
        status = Runtime(aclrtCreateEventWithFlag(&state->event, ACL_EVENT_SYNC));
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        {
            status = Runtime(aclrtCreateEventWithFlag(&state->checkReady, ACL_EVENT_SYNC));
            if (status != ACL_SPARSE_STATUS_SUCCESS) {
                return status;
            }
        }
        if (p.bsr && !p.cube && !p.smallBsr) {
            auto* w = static_cast<uint8_t*>(buffer);
            for (uint32_t b = 0; b < p.count; ++b) {
                auto* expanded = w + p.expanded + b * p.expandedStride;
                status = sddmm_bsr_expand_launch(w + b * p.patternStride, w + b * p.patternStride + p.rowBytes,
                                                 expanded, expanded + SddmmAlign32(uint64_t(p.m + 1) * 4), p.m, c->nnz,
                                                 p.bd, c->baseType == ACL_SPARSE_INDEX_BASE_ONE, SddmmGetBlockDim(),
                                                 h->stream);
                if (status != ACL_SPARSE_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
        status = Runtime(aclrtRecordEvent(state->event, h->stream));
        if (status == ACL_SPARSE_STATUS_SUCCESS) {
            c->sddmmState = state;
        }
        return status;
    } catch (const std::bad_alloc&) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
}

aclsparseStatus_t aclsparseSDDMMPreprocess(aclsparseHandle_t handle, aclsparseOperation_t ox, aclsparseOperation_t oy,
                                           const void* alpha, aclsparseConstDnMatDescr_t x,
                                           aclsparseConstDnMatDescr_t y, const void* beta, aclsparseSpMatDescr_t c,
                                           aclDataType compute, aclsparseSDDMMAlg_t alg, void* buffer)
{
    Plan p;
    auto status = BuildPlan(handle, ox, oy, alpha, x, y, beta, c, compute, alg, p);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (!BufferValid(buffer)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* h = SddmmToInternalHandle(handle);
    auto previous = std::static_pointer_cast<State>(c->sddmmState);
    if (previous && previous->stream != h->stream) {
        status = Runtime(aclrtStreamWaitEvent(h->stream, previous->event));
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
    }
    c->sddmmState.reset();
    status = CheckPatterns(h, c, p, buffer, false);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (p.rowHints) {
        auto* w = static_cast<uint8_t*>(buffer);
        status = sddmm_csr_row_hints_launch(w, w + p.rowBytes, w + p.rowHints, p.m,
                                            c->baseType == ACL_SPARSE_INDEX_BASE_ONE, 32 / Bytes(c->valueType),
                                            SddmmGetBlockDim(), h->stream);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
    }
    return PreparePatternState(h, c, p, buffer);
}

class SddmmExecution {
public:
    SddmmExecution(aclsparseHandle_t handle, aclsparseOperation_t ox, aclsparseOperation_t oy, const void* alpha,
                   aclsparseConstDnMatDescr_t x, aclsparseConstDnMatDescr_t y, const void* beta,
                   aclsparseSpMatDescr_t c, aclDataType compute, aclsparseSDDMMAlg_t alg, void* buffer,
                   bool prepareAndExecute)
        : handle(handle),
          ox(ox),
          oy(oy),
          alpha(alpha),
          x(x),
          y(y),
          beta(beta),
          c(c),
          compute(compute),
          alg(alg),
          buffer(buffer),
          prepareAndExecute(prepareAndExecute)
    {}
    aclsparseStatus_t Run()
    {
        status = Initialize();
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        status = ValidatePattern();
        if (status != ACL_SPARSE_STATUS_SUCCESS || !c->nnz) {
            return status;
        }
        for (uint32_t batch = 0; batch < p.count; ++batch) {
            status = ExecuteBatch(batch);
            if (status != ACL_SPARSE_STATUS_SUCCESS) {
                if (overlapCheck) {
                    (void)aclrtSynchronizeEvent(state->checkReady);
                }
                return status;
            }
        }
        return Finish();
    }

private:
    aclsparseStatus_t Initialize()
    {
        // A call-local operation consumes the snapshot it just validated. No caller
        // can change the descriptors between its preparation and execution phases.
        if (prepareAndExecute) {
            const auto prepared = aclsparseSDDMMPreprocess(handle, ox, oy, alpha, x, y, beta, c, compute, alg, buffer);
            if (prepared != ACL_SPARSE_STATUS_SUCCESS) {
                return prepared;
            }
        }
        status = BuildPlan(handle, ox, oy, alpha, x, y, beta, c, compute, alg, p);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        state = std::static_pointer_cast<State>(c->sddmmState);
        if (!BufferValid(buffer) || !state || state->buffer != buffer || state->plan.key != p.key) {
            c->sddmmState.reset();
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        h = SddmmToInternalHandle(handle);
        if (state->stream != h->stream) {
            status = Runtime(aclrtStreamWaitEvent(h->stream, state->event));
            if (status != ACL_SPARSE_STATUS_SUCCESS) {
                return status;
            }
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t ValidatePattern()
    {
        overlapCheck = !prepareAndExecute && !p.bsr && p.k && c->nnz && x->valueType != ACL_COMPLEX64 &&
                       h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST && SddmmReadScalarToF32(alpha, compute) != 0.0f;
        if (prepareAndExecute && h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
            CopyHostScalars(alpha, beta, compute, scalars);
        } else {
            status = CheckPatterns(h, c, p, buffer, true, alpha, beta, compute, &scalars,
                                   overlapCheck ? state->checkReady : nullptr, &pendingFlags);
        }
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            c->sddmmState.reset();
            return status;
        }
        return status;
    }
    aclsparseStatus_t ExecuteCsr(uint32_t batch, uint8_t* xp, uint8_t* yp, uint8_t* cp, uint8_t* rp, uint8_t* ci)
    {
        const void *a = scalars.data(), *b = scalars.data() + 8;
        const float ar = SddmmReadScalarToF32(a, compute), ai = SddmmReadScalarImagToF32(a, compute);
        const float br = SddmmReadScalarToF32(b, compute), bi = SddmmReadScalarImagToF32(b, compute);
        auto* w = static_cast<uint8_t*>(buffer);
        const bool canonical = p.bsr || x->valueType == ACL_COMPLEX64;
        auto dx = canonical ? DenseView(x, p.m, p.k) : *x;
        auto dy = canonical ? DenseView(y, p.k, p.n) : *y;
        dx.values = xp;
        dy.values = yp;
        aclsparseSpMatDescr csr;
        CsrView(csr, c, p);
        csr.ptrs = rp;
        csr.idxs = ci;
        csr.values = cp;
        if (p.bsr) {
            csr.ptrs = w + p.expanded + batch * p.expandedStride;
            csr.idxs = static_cast<uint8_t*>(csr.ptrs) + SddmmAlign32(uint64_t(p.m + 1) * 4);
            csr.values = w + p.scalarValues;
            if (br != 0 || bi != 0) {
                status = sddmm_bsr_repack_launch(cp, static_cast<uint8_t*>(csr.values), rp, p.rows, c->nnz, p.bd,
                                                 c->baseType == ACL_SPARSE_INDEX_BASE_ONE, c->valueType,
                                                 c->blockOrder == ACL_SPARSE_ORDER_COL, false, SddmmGetBlockDim(),
                                                 h->stream);
                if (status != ACL_SPARSE_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
        const auto nx = canonical ? ACL_SPARSE_OP_NON_TRANSPOSE : ox;
        const auto ny = canonical ? ACL_SPARSE_OP_NON_TRANSPOSE : oy;
        status = SddmmCsrPrepare(handle, nx, ny, a, &dx, &dy, b, &csr, compute, alg, w + p.scratch);
        if (status == ACL_SPARSE_STATUS_SUCCESS) {
            status = SddmmCsrExecute(handle, nx, ny, a, &dx, &dy, b, &csr, compute, alg, w + p.scratch,
                                     p.rowHints ? w + p.rowHints : nullptr, overlapCheck ? w + p.flags : nullptr,
                                     overlapCheck ? p.flagBytes / sizeof(int32_t) : 0);
        }
        if (status == ACL_SPARSE_STATUS_SUCCESS && p.bsr) {
            status = sddmm_bsr_repack_launch(cp, static_cast<uint8_t*>(csr.values), rp, p.rows, c->nnz, p.bd,
                                             c->baseType == ACL_SPARSE_INDEX_BASE_ONE, c->valueType,
                                             c->blockOrder == ACL_SPARSE_ORDER_COL, true, SddmmGetBlockDim(),
                                             h->stream);
        }
        return status;
    }
    aclsparseStatus_t PackOperands(uint8_t*& xp, uint8_t*& yp)
    {
        auto* w = static_cast<uint8_t*>(buffer);
        if (p.packX) {
            status = Pack(x, ox, p.m, p.k, xp, w + p.packedX, h->stream);
            if (status != ACL_SPARSE_STATUS_SUCCESS) {
                return status;
            }
            xp = w + p.packedX;
        }
        if (p.packY) {
            status = Pack(y, oy, p.k, p.n, yp, w + p.packedY, h->stream);
            if (status != ACL_SPARSE_STATUS_SUCCESS) {
                return status;
            }
            yp = w + p.packedY;
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t ExecuteBatch(uint32_t batch)
    {
        const void *a = scalars.data(), *b = scalars.data() + 8;
        const float ar = SddmmReadScalarToF32(a, compute), ai = SddmmReadScalarImagToF32(a, compute);
        const float br = SddmmReadScalarToF32(b, compute), bi = SddmmReadScalarImagToF32(b, compute);
        auto* w = static_cast<uint8_t*>(buffer);
        auto* xp = Batch(x->values, batch, x->batchCount, x->batchStride, Bytes(x->valueType));
        auto* yp = Batch(y->values, batch, y->batchCount, y->batchStride, Bytes(y->valueType));
        auto* cp = Batch(c->values, batch, p.count, c->valuesBatchStride, Bytes(c->valueType));
        if (!p.k || (ar == 0 && ai == 0)) {
            status = sddmm_scale_launch(cp, c->nnz * p.bd * p.bd, c->valueType, br, bi, SddmmGetBlockDim(), h->stream);
            if (status != ACL_SPARSE_STATUS_SUCCESS) {
                return status;
            }
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        status = PackOperands(xp, yp);
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        auto* rp = w + batch * p.patternStride;
        auto* ci = rp + p.rowBytes;
        if (p.cube) {
            status = sddmm_bsr_cube_launch(xp, yp, rp, ci, cp, cp, w + p.scratch, p.m, p.n, p.k, p.bd, c->nnz,
                                           c->baseType == ACL_SPARSE_INDEX_BASE_ONE,
                                           c->blockOrder == ACL_SPARSE_ORDER_COL, x->valueType, c->valueType, ar, ai,
                                           br, bi, h->stream);
        } else if (p.smallBsr && x->valueType == ACL_COMPLEX64) {
            status = sddmm_bsr_complex_vector_launch(xp, yp, rp, ci, cp, p.m, p.n, p.k, p.bd, c->nnz,
                                                     c->baseType == ACL_SPARSE_INDEX_BASE_ONE,
                                                     c->blockOrder == ACL_SPARSE_ORDER_COL, ar, ai, br, bi, h->stream);
        } else if (p.smallBsr) {
            status = sddmm_bsr_small_launch(
                xp, yp, rp, ci, cp, p.m, p.n, p.k, p.bd, c->nnz, c->baseType == ACL_SPARSE_INDEX_BASE_ONE,
                c->blockOrder == ACL_SPARSE_ORDER_COL, x->valueType, c->valueType, ar, br, h->stream);
        } else {
            status = ExecuteCsr(batch, xp, yp, cp, rp, ci);
        }
        return status;
    }
    aclsparseStatus_t Finish()
    {
        state->stream = h->stream;
        status = Runtime(aclrtRecordEvent(state->event, h->stream));
        if (overlapCheck) {
            const auto waited = Runtime(aclrtSynchronizeEvent(state->checkReady));
            if (waited != ACL_SPARSE_STATUS_SUCCESS) {
                return waited;
            }
            for (uint32_t i = 0; i < p.flagBytes / sizeof(int32_t);
                 i += SddmmBudget::kFlagStrideBytes / sizeof(int32_t)) {
                if (pendingFlags[i]) {
                    // A failed synchronous API call must not leave even a guarded
                    // reader in flight against workspace the caller may now free.
                    const auto drained = Runtime(aclrtSynchronizeEvent(state->event));
                    c->sddmmState.reset();
                    return drained == ACL_SPARSE_STATUS_SUCCESS ? ACL_SPARSE_STATUS_INVALID_VALUE : drained;
                }
            }
        }
        return status;
    }
    aclsparseHandle_t handle;
    aclsparseOperation_t ox, oy;
    const void* alpha;
    const aclsparseDnMatDescr *x, *y;
    const void* beta;
    aclsparseSpMatDescr* c;
    aclDataType compute;
    aclsparseSDDMMAlg_t alg;
    Plan p;
    void* buffer;
    bool prepareAndExecute;
    bool overlapCheck = false;
    const int32_t* pendingFlags = nullptr;
    aclsparseContext* h = nullptr;
    std::shared_ptr<State> state;
    alignas(8) std::array<uint8_t, 16> scalars{};
    aclsparseStatus_t status = ACL_SPARSE_STATUS_SUCCESS;
};

static aclsparseStatus_t ExecuteSddmm(aclsparseHandle_t handle, aclsparseOperation_t ox, aclsparseOperation_t oy,
                                      const void* alpha, aclsparseConstDnMatDescr_t x, aclsparseConstDnMatDescr_t y,
                                      const void* beta, aclsparseSpMatDescr_t c, aclDataType compute,
                                      aclsparseSDDMMAlg_t alg, void* buffer, bool prepareAndExecute)
{
    return SddmmExecution(handle, ox, oy, alpha, x, y, beta, c, compute, alg, buffer, prepareAndExecute).Run();
}

aclsparseStatus_t aclsparseSDDMM(aclsparseHandle_t handle, aclsparseOperation_t ox, aclsparseOperation_t oy,
                                 const void* alpha, aclsparseConstDnMatDescr_t x, aclsparseConstDnMatDescr_t y,
                                 const void* beta, aclsparseSpMatDescr_t c, aclDataType compute,
                                 aclsparseSDDMMAlg_t alg, void* buffer)
{
    return ExecuteSddmm(handle, ox, oy, alpha, x, y, beta, c, compute, alg, buffer, false);
}

extern "C" aclsparseStatus_t sddmm_call_local(aclsparseHandle_t handle, aclsparseOperation_t ox,
                                              aclsparseOperation_t oy, const void* alpha, aclsparseConstDnMatDescr_t x,
                                              aclsparseConstDnMatDescr_t y, const void* beta, aclsparseSpMatDescr_t c,
                                              aclDataType compute, aclsparseSDDMMAlg_t alg, void* buffer)
{
    return ExecuteSddmm(handle, ox, oy, alpha, x, y, beta, c, compute, alg, buffer, true);
}
