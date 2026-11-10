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
#include <mutex>
#include <new>
#include <unordered_set>
#include "aclsparse_descr_internal.h"
#include "aclsparse_host_utils.h"
#include "spsm.h"
#include "spsm_plan_kernel.h"

namespace
{
// A registry permits rejecting unknown/destroyed SpSM pointers before reading
// them. Like the other Generic API descriptors, concurrent use of one plan is
// not supported. Independent plans may be used on independent streams.
std::mutex registryMutex;
std::unordered_set<aclsparseSpSMDescr_t> plans;
constexpr auto OK = ACL_SPARSE_STATUS_SUCCESS;
constexpr auto INVALID = ACL_SPARSE_STATUS_INVALID_VALUE;
constexpr auto UNSUPPORTED = ACL_SPARSE_STATUS_NOT_SUPPORTED;

aclsparseStatus_t ReportFailure(aclsparseStatus_t status, const char* function, int line)
{
    OP_LOGE("aclsparseSpSM", "%s:%d rejected the request (status=%d)", function, line, static_cast<int>(status));
    return status;
}

bool IsPlan(aclsparseSpSMDescr_t d)
{
    std::lock_guard<std::mutex> lock(registryMutex);
    return d != nullptr && plans.find(d) != plans.end();
}

uint64_t Address(const void* p)
{
    return reinterpret_cast<uintptr_t>(p);
}
GM_ADDR Gm(const void* p)
{
    return reinterpret_cast<GM_ADDR>(Address(p));
}
uint32_t Blocks()
{
    return std::max(1u, GetAivCoreCount());
}
bool DenseValid(const aclsparseDnMatDescr* d)
{
    if (d->signature != kDnMatSignature || d->rows < 0 || d->cols < 0)
    {
        return false;
    }
    if (d->order != ACL_SPARSE_ORDER_ROW && d->order != ACL_SPARSE_ORDER_COL)
    {
        return false;
    }
    const int64_t n = d->order == ACL_SPARSE_ORDER_ROW ? d->cols : d->rows;
    if (d->ld < std::max<int64_t>(1, n))
    {
        return false;
    }
    if (d->rows == 0 || d->cols == 0)
    {
        return true;
    }
    int64_t minor = d->order == ACL_SPARSE_ORDER_ROW ? d->cols : d->rows;
    int64_t major = d->order == ACL_SPARSE_ORDER_ROW ? d->rows : d->cols;
    int64_t bytes = d->valueType == ACL_COMPLEX64 ? 8 : 4;
    return (major - 1) <= (INT64_MAX / bytes - minor) / d->ld;
}

aclsparseStatus_t ValidateSparse(const aclsparseSpMatDescr* a)
{
    if (a->format != ACL_SPARSE_FORMAT_CSR && a->format != ACL_SPARSE_FORMAT_CSC && a->format != ACL_SPARSE_FORMAT_COO)
    {
        return ReportFailure(ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED, __func__, __LINE__);
    }
    if (a->ptrType != ACL_SPARSE_INDEX_32I || a->IdxType != ACL_SPARSE_INDEX_32I)
    {
        return ReportFailure(UNSUPPORTED, __func__, __LINE__);
    }
    if (a->baseType != ACL_SPARSE_INDEX_BASE_ZERO && a->baseType != ACL_SPARSE_INDEX_BASE_ONE)
    {
        return ReportFailure(UNSUPPORTED, __func__, __LINE__);
    }
    if ((a->format != ACL_SPARSE_FORMAT_COO || a->nnz != 0) && a->ptrs == nullptr)
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    if (a->nnz > 0 && (a->idxs == nullptr || a->values == nullptr))
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    return OK;
}

aclsparseStatus_t ValidateShapes(
    const aclsparseSpMatDescr* a, const aclsparseDnMatDescr* b, const aclsparseDnMatDescr* c, aclsparseOperation_t opB)
{
    if (a->rows != a->cols || !DenseValid(b) || !DenseValid(c))
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    if (a->rows == 0 || c->cols == 0)
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    if (a->rows >= INT32_MAX || c->cols > INT32_MAX || a->nnz > static_cast<uint64_t>(INT32_MAX - a->baseType))
    {
        return ReportFailure(UNSUPPORTED, __func__, __LINE__);
    }
    const int64_t ldb = b->ld;
    const int64_t ldc = c->ld;
    const int64_t bRequired = b->order == ACL_SPARSE_ORDER_ROW ? b->cols : b->rows;
    const int64_t cRequired = c->order == ACL_SPARSE_ORDER_ROW ? c->cols : c->rows;
    if (ldb < std::max<int64_t>(1, bRequired) || ldc < std::max<int64_t>(1, cRequired))
    {
        return ReportFailure(ACL_SPARSE_STATUS_INVALID_VALUE, __func__, __LINE__);
    }
    int64_t br = opB == ACL_SPARSE_OP_NON_TRANSPOSE ? b->rows : b->cols;
    int64_t bc = opB == ACL_SPARSE_OP_NON_TRANSPOSE ? b->cols : b->rows;
    if (br != static_cast<int64_t>(a->rows) || c->rows != br || c->cols != bc)
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    return OK;
}

aclsparseStatus_t ValidateEnums(
    const aclsparseSpMatDescr* a, aclsparseOperation_t opA, aclsparseOperation_t opB)
{
    const aclsparseFillMode_t uplo = a->fillMode;
    const aclsparseOperation_t trans = opA;
    const aclsparseDiagType_t diag = a->diagType;
    if ((uplo != ACL_SPARSE_FILL_MODE_UPPER && uplo != ACL_SPARSE_FILL_MODE_LOWER)
        || (trans != ACL_SPARSE_OP_NON_TRANSPOSE && trans != ACL_SPARSE_OP_TRANSPOSE
            && trans != ACL_SPARSE_OP_CONJUGATE_TRANSPOSE)
        || (diag != ACL_SPARSE_DIAG_TYPE_UNIT && diag != ACL_SPARSE_DIAG_TYPE_NON_UNIT))
    {
        return ReportFailure(ACL_SPARSE_STATUS_INVALID_VALUE, __func__, __LINE__);
    }
    if (opB != ACL_SPARSE_OP_NON_TRANSPOSE && opB != ACL_SPARSE_OP_TRANSPOSE
        && opB != ACL_SPARSE_OP_CONJUGATE_TRANSPOSE)
    {
        return ReportFailure(ACL_SPARSE_STATUS_INVALID_VALUE, __func__, __LINE__);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t Validate(const aclsparseSpMatDescr* a, const aclsparseDnMatDescr* b, const aclsparseDnMatDescr* c,
    aclsparseOperation_t opA, aclsparseOperation_t opB, aclDataType type, aclsparseSpSMAlg_t alg)
{
    if (a == nullptr || b == nullptr || c == nullptr)
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    auto status = ValidateEnums(a, opA, opB);
    if (status != ACL_SPARSE_STATUS_SUCCESS)
    {
        return status;
    }
    if (alg != ACL_SPARSE_SPSM_ALG_DEFAULT)
    {
        return ReportFailure(UNSUPPORTED, __func__, __LINE__);
    }
    if ((type != ACL_FLOAT && type != ACL_COMPLEX64) || a->valueType != type || b->valueType != type
        || c->valueType != type)
    {
        return ReportFailure(UNSUPPORTED, __func__, __LINE__);
    }
    status = ValidateSparse(a);
    return status == OK ? ValidateShapes(a, b, c, opB) : status;
}

std::array<uint64_t, 32> Signature(const aclsparseSpMatDescr* a, const aclsparseDnMatDescr* b,
    const aclsparseDnMatDescr* c, aclsparseOperation_t opA, aclsparseOperation_t opB)
{
    return { Address(a), Address(b), Address(c), a->rows, a->cols, a->nnz, Address(a->ptrs), Address(a->idxs),
        Address(a->values), static_cast<uint64_t>(a->format), static_cast<uint64_t>(a->baseType),
        static_cast<uint64_t>(a->ptrType), static_cast<uint64_t>(a->IdxType), static_cast<uint64_t>(a->valueType),
        static_cast<uint64_t>(a->fillMode), static_cast<uint64_t>(a->diagType), static_cast<uint64_t>(b->rows),
        static_cast<uint64_t>(b->cols), static_cast<uint64_t>(b->ld), static_cast<uint64_t>(b->order),
        static_cast<uint64_t>(b->valueType), static_cast<uint64_t>(c->rows), static_cast<uint64_t>(c->cols),
        static_cast<uint64_t>(c->ld), static_cast<uint64_t>(c->order), static_cast<uint64_t>(c->valueType),
        static_cast<uint64_t>(opA), static_cast<uint64_t>(opB) };
}

bool Reserve(int64_t& end, int64_t count, int64_t bytes, int64_t& offset)
{
    if (count < 0 || bytes <= 0 || end < 0 || end > INT64_MAX - 63)
    {
        return false;
    }
    int64_t required = 0;
    if (__builtin_mul_overflow(count, bytes, &required) || required > INT64_MAX - end - 63)
    {
        return false;
    }
    offset = end;
    end += (required + 63) / 64 * 64;
    return true;
}

bool Layout(const aclsparseSpMatDescr* a, const aclsparseDnMatDescr* b, const aclsparseDnMatDescr* c,
    aclsparseOperation_t opA, aclsparseOperation_t opB, SpsmPlanTiling& t, int64_t& bytes)
{
    t.m = a->rows;
    t.lowRows = t.m;
    t.n = c->cols;
    t.nnz = a->nnz;
    t.format = a->format == ACL_SPARSE_FORMAT_CSR ? 0 : (a->format == ACL_SPARSE_FORMAT_CSC ? 1 : 2);
    t.base = a->baseType;
    t.opA = opA;
    t.opB = opB;
    t.upper = (a->fillMode == ACL_SPARSE_FILL_MODE_UPPER) != (opA != ACL_SPARSE_OP_NON_TRANSPOSE);
    t.unit = a->diagType == ACL_SPARSE_DIAG_TYPE_UNIT;
    t.components = a->valueType == ACL_COMPLEX64 ? 2 : 1;
    t.orderB = b->order;
    t.orderC = c->order;
    t.ldb = b->ld;
    t.ldc = c->ld;
    int64_t m = t.m, q = t.nnz, s = 4 * t.components;
    bytes = sizeof(SpsmDeviceSummary);
    bool valid = Reserve(bytes, m + 1, 4, t.row) && Reserve(bytes, q, 4, t.col) && Reserve(bytes, q, 4, t.perm)
        && Reserve(bytes, q, s, t.val) && Reserve(bytes, m, s, t.diag) && Reserve(bytes, m, 4, t.rowLevel)
        && Reserve(bytes, m + 1, 4, t.levelPtr) && Reserve(bytes, m, 4, t.levelIdx)
        && Reserve(bytes, Blocks(), 128, t.reduce);
    int64_t dense = m * t.n;
    if (!valid || dense > INT64_MAX / s)
    {
        return false;
    }
    int64_t keyBytes = (q * 12 + 63) / 64 * 64;
    // Retain full precision while bounding the extra dense rounding-tail buffer.
    // Independent RHS panels reuse it; the full high component also snapshots B
    // until all panels finish, so transpose and overlapping B/C remain safe.
    constexpr int64_t LOW_BUDGET = 16 * 1024 * 1024;
    t.lowWidth = std::min<int64_t>(t.n, std::max<int64_t>(1, LOW_BUDGET / (m * s)));
    t.lowCapacity = m * t.lowWidth;
    int64_t lowBytes = m * t.lowWidth * s;
    if (s * dense > INT64_MAX - lowBytes)
    {
        return false;
    }
    int64_t scratchBytes = std::max({ 2 * keyBytes, 4 * (m + 1), s * m, s * dense + lowBytes });
    if (!Reserve(bytes, scratchBytes, 1, t.scratch))
    {
        return false;
    }
    t.key0 = t.scratch;
    t.key1 = t.scratch + keyBytes;
    t.candidateDiag = t.scratch;
    t.low = t.scratch + s * dense;
    return true;
}

aclsparseStatus_t Context(aclsparseHandle_t h, aclrtStream& stream, aclsparsePointerMode_t& mode, int32_t& device)
{
    if (h == nullptr)
    {
        return ReportFailure(ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR, __func__, __LINE__);
    }
    if (aclsparseGetStream(h, &stream) != OK || aclsparseGetPointerMode(h, &mode) != OK
        || aclrtGetDevice(&device) != ACL_SUCCESS)
    {
        return ReportFailure(ACL_SPARSE_STATUS_NOT_INITIALIZED, __func__, __LINE__);
    }
    return OK;
}

bool DevicePointer(const void* p, int32_t device)
{
    aclrtPtrAttributes attributes { };
    return p != nullptr && aclrtPointerGetAttributes(p, &attributes) == ACL_SUCCESS
        && attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE
        && attributes.location.id == static_cast<uint32_t>(device);
}

// Query the live ACL allocation containing p, including interior pointers.
// A raw pointer has no tensor dtype or logical suballocation metadata; those
// remain the caller's contract. Never dereference an invalid device address.
bool DeviceSpan(const void* p, uint64_t bytes, int32_t device)
{
    if (!DevicePointer(p, device))
    {
        return false;
    }
    void* base = nullptr;
    size_t allocationBytes = 0;
    if (aclrtMemGetAddressRange(const_cast<void*>(p), &base, &allocationBytes) != ACL_SUCCESS)
    {
        return false;
    }
    uint64_t address = Address(p), start = Address(base);
    return address >= start && address - start <= allocationBytes && bytes <= allocationBytes - (address - start);
}

bool AlphaPointer(const void* p, aclsparsePointerMode_t mode, int32_t device)
{
    if (p == nullptr)
    {
        return false;
    }
    if (mode == ACL_SPARSE_POINTER_MODE_DEVICE)
    {
        return DevicePointer(p, device);
    }
    aclrtPtrAttributes attributes { };
    return mode == ACL_SPARSE_POINTER_MODE_HOST && aclrtPointerGetAttributes(p, &attributes) == ACL_SUCCESS
        && attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE;
}

aclsparseStatus_t Match(aclsparseHandle_t h, aclsparseSpSMDescr_t d, const void* alpha, const aclsparseSpMatDescr* a,
    const aclsparseDnMatDescr* b, const aclsparseDnMatDescr* c, aclsparseOperation_t opA, aclsparseOperation_t opB)
{
    aclrtStream stream = nullptr;
    aclsparsePointerMode_t mode = ACL_SPARSE_POINTER_MODE_HOST;
    int32_t device = -1;
    auto status = Context(h, stream, mode, device);
    if (status != OK)
    {
        return status;
    }
    if (h != d->owner || alpha != d->alpha || stream != d->stream || mode != d->pointerMode || device != d->device
        || Signature(a, b, c, opA, opB) != d->signature)
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    return OK;
}

void Launch(aclsparseSpSMDescr_t d, SpsmPhase phase, const void* values, void* dense = nullptr, int64_t width = 0,
    uint32_t blocks = 0)
{
    spsm_plan_kernel_do(Gm(d->inputA->ptrs), Gm(d->inputA->idxs), Gm(values), Gm(dense), Gm(d->alpha), Gm(d->buffer),
        d->plan, phase, width, blocks ? blocks : Blocks(), d->stream);
}

// Only Analysis/Update wait for a fixed-size validation result. Solve performs
// no host wait or device-to-host transfer. Synchronizing the handle stream also
// surfaces asynchronous kernel failures before accepting the validation result.
aclsparseStatus_t ReadSummary(aclsparseSpSMDescr_t d, SpsmDeviceSummary& summary)
{
    if (aclrtSynchronizeStream(d->stream) != ACL_SUCCESS
        || aclrtMemcpy(&summary, sizeof(summary), d->buffer, sizeof(summary), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
    {
        return ReportFailure(ACL_SPARSE_STATUS_EXECUTION_FAILED, __func__, __LINE__);
    }
    if (summary.error != 0)
    {
        OP_LOGE("aclsparseSpSM", "NPU validation rejected input: summary error=%d", summary.error);
        return summary.error == 2 ? UNSUPPORTED : INVALID;
    }
    return OK;
}

// clear-then-launch-then-peek + summary 回读（评审响应：消重复块 + 统一约定）。
// Peek 只读不清除，陈旧线程级错误会被归因本次下发且一次失败后永久粘滞
// （AlphaPointer 对 host 指针查询属性失败即留粘滞 last error）；
// 每次调用先 aclrtGetLastError 清除，只归因本次下发。
static aclsparseStatus_t LaunchChecked(aclsparseSpSMDescr_t d, SpsmPhase phase,
    const void* values, void* dense, int64_t width, uint32_t blocks)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    Launch(d, phase, values, dense, width, blocks);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS)
    {
        return ReportFailure(ACL_SPARSE_STATUS_EXECUTION_FAILED, __func__, __LINE__);
    }
    return OK;
}

// CHECK 段：下发 + enqueue 查错 + summary 回读（Analysis 尾段与 Update 的
// CHECK 段共用；返回 OK 时 summary 已填）。
static aclsparseStatus_t LaunchCheckAndReadSummary(
    aclsparseSpSMDescr_t d, SpsmPhase phase, const void* values,
    void* dense, int64_t width, uint32_t blocks, SpsmDeviceSummary& summary)
{
    const aclsparseStatus_t status = LaunchChecked(d, phase, values, dense, width, blocks);
    if (status != OK)
    {
        return status;
    }
    return ReadSummary(d, summary);
}

void SetAnalysisMetadata(aclsparseSpSMDescr_t d, const SpsmDeviceSummary& s)
{
    d->plan.levels = s.levels;
    d->plan.maxRowLen = s.maxRowLen;
    auto& plan = d->plan;
    plan.lowRows = plan.m;
    plan.lowWidth = plan.lowCapacity / plan.m;
    bool parallelLevels = plan.levels <= 128 || (plan.levels <= 512 && plan.m >= 4 * plan.levels);
    int64_t window = 1;
    while (window <= s.maxDependencyDistance)
    {
        window *= 2;
    }
    // With natural triangular row order, tails older than the longest edge
    // cannot be read again. A ring then preserves every RHS's low bits while
    // reusing the already-reserved storage and retaining RHS concurrency.
    if (!parallelLevels && plan.lowWidth < plan.n && window < plan.m && window * plan.n <= plan.lowCapacity)
    {
        plan.lowRows = window;
        plan.lowWidth = plan.n;
    }
    const auto& p = d->plan;
    auto& t = d->cachedTiling;
    t = { };
    t.m = p.m;
    t.n = p.n;
    t.L = s.levels;
    t.maxRowLen = std::max(1, s.maxRowLen);
    // Retain the fixed host diagnostics getter used by the existing tests.
    // Each SIMT work item solves one RHS element, or one complete RHS on the
    // serial path; no UB-sized dense RHS tile is required.
    t.kChunkSize = 1;
}

bool OverlapsWorkspace(const void* p, int64_t bytes, const aclsparseSpSMDescr* d)
{
    uint64_t address = Address(p), workspace = Address(d->buffer);
    return bytes > 0
        && (address > UINT64_MAX - static_cast<uint64_t>(bytes)
            || (address < workspace + d->cachedBufferSize && workspace < address + bytes));
}

int64_t DenseBytes(const aclsparseDnMatDescr* d)
{
    if (d->rows == 0 || d->cols == 0)
    {
        return 0;
    }
    int64_t major = d->order == ACL_SPARSE_ORDER_ROW ? d->rows : d->cols;
    int64_t minor = d->order == ACL_SPARSE_ORDER_ROW ? d->cols : d->rows;
    return ((major - 1) * d->ld + minor) * (d->valueType == ACL_COMPLEX64 ? 8 : 4);
}
void BindPlan(aclsparseSpSMDescr_t d, aclsparseHandle_t h, aclsparseConstSpMatDescr_t a, aclsparseConstDnMatDescr_t b,
    aclsparseDnMatDescr_t c, const void* alpha, aclsparseOperation_t opA, aclsparseOperation_t opB,
    aclsparsePointerMode_t mode, aclrtStream stream, int32_t device, const SpsmPlanTiling& tiling, int64_t bytes)
{
    d->analyzed = false;
    d->sized = true;
    d->buffer = nullptr;
    d->signature = Signature(a, b, c, opA, opB);
    d->owner = h;
    d->inputA = a;
    d->inputB = b;
    d->outputC = c;
    d->alpha = alpha;
    d->pointerMode = mode;
    d->stream = stream;
    d->device = device;
    d->boundB = nullptr;
    d->boundC = nullptr;
    d->opA = opA;
    d->opB = opB;
    d->plan = tiling;
    d->plan.deviceAlpha = mode == ACL_SPARSE_POINTER_MODE_DEVICE;
    d->cachedBufferSize = bytes;
}

void BuildPlan(aclsparseSpSMDescr_t d, aclsparseConstSpMatDescr_t a)
{
    d->plan.key0 = d->plan.scratch;
    d->plan.key1 = d->plan.scratch + (static_cast<int64_t>(d->plan.nnz) * 12 + 63) / 64 * 64;
    Launch(d, SpsmPhase::CHECK, a->values, nullptr, 0, 1);
    Launch(d, SpsmPhase::DECODE, a->values);
    for (int64_t width = 1; width < d->plan.nnz; width *= 2)
    {
        Launch(d, SpsmPhase::MERGE, a->values, nullptr, width);
        std::swap(d->plan.key0, d->plan.key1);
    }
    Launch(d, SpsmPhase::CANONICAL, a->values);
    Launch(d, SpsmPhase::ANALYZE, a->values, nullptr, 0, 1);
}

aclsparseStatus_t EnsureDefaultWorkspace(aclsparseHandle_t handle, size_t requiredSize)
{
    void* workspace = nullptr;
    size_t available = 0;
    const aclsparseStatus_t status = aclsparseGetWorkspace(handle, &workspace, &available);
    if (status != ACL_SPARSE_STATUS_SUCCESS)
    {
        return status;
    }
    if (workspace == nullptr || available < requiredSize)
    {
        return ReportFailure(ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES, __func__, __LINE__);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

void* GetEffectiveWorkspace(aclsparseHandle_t handle)
{
    void* workspace = nullptr;
    size_t available = 0;
    if (aclsparseGetWorkspace(handle, &workspace, &available) != ACL_SPARSE_STATUS_SUCCESS)
    {
        return nullptr;
    }
    return workspace;
}

aclsparseStatus_t ResolveWorkspace(
    aclsparseHandle_t handle, aclsparseSpSMDescr_t d, void* requested, void*& workspace)
{
    workspace = requested;
    if (workspace != nullptr)
    {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const aclsparseStatus_t status = EnsureDefaultWorkspace(handle, static_cast<size_t>(d->cachedBufferSize));
    if (status != ACL_SPARSE_STATUS_SUCCESS)
    {
        return status;
    }
    workspace = static_cast<uint8_t*>(GetEffectiveWorkspace(handle));
    return workspace == nullptr ? ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES : ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t BindWorkspaceAndValidateStorage(
    aclsparseSpSMDescr_t d, aclsparseConstSpMatDescr_t a, void* buffer)
{
    // Prefer the caller-owned BufferSize allocation. If Analysis received a
    // null buffer, GetEffectiveWorkspace selected the handle workspace.
    if (Address(buffer) % 64 != 0 || !DeviceSpan(buffer, d->cachedBufferSize, d->device))
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    int64_t ptrCount = d->plan.format == 2 ? a->nnz : a->rows + 1;
    if (ptrCount > 0 && !DeviceSpan(a->ptrs, ptrCount * 4, d->device))
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    if (a->nnz > 0
        && (!DeviceSpan(a->idxs, a->nnz * 4, d->device)
            || !DeviceSpan(a->values, a->nnz * 4 * d->plan.components, d->device)))
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    d->buffer = buffer;
    d->analyzed = false;
    if (OverlapsWorkspace(a->ptrs, ptrCount * 4, d) || OverlapsWorkspace(a->idxs, a->nnz * 4, d)
        || OverlapsWorkspace(a->values, a->nnz * 4 * d->plan.components, d))
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    return OK;
}

aclsparseStatus_t CheckDenseStorage(
    aclsparseSpSMDescr_t d, aclsparseConstDnMatDescr_t b, aclsparseDnMatDescr_t c, const void* alpha)
{
    if (!DeviceSpan(d->buffer, d->cachedBufferSize, d->device) || !DeviceSpan(b->values, DenseBytes(b), d->device)
        || !DeviceSpan(c->values, DenseBytes(c), d->device) || (d->boundB && d->boundB != b->values)
        || (d->boundC && d->boundC != c->values))
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    if (!AlphaPointer(alpha, d->pointerMode, d->device)
        || (d->plan.deviceAlpha && !DeviceSpan(alpha, d->plan.components * 4, d->device)))
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    if (OverlapsWorkspace(b->values, DenseBytes(b), d) || OverlapsWorkspace(c->values, DenseBytes(c), d)
        || (d->plan.deviceAlpha && OverlapsWorkspace(alpha, d->plan.components * 4, d)))
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    d->boundB = b->values;
    d->boundC = c->values;
    return OK;
}

aclsparseStatus_t CheckStage(aclsparseHandle_t h, aclsparseSpSMDescr_t d, bool analyzed, const void* alpha,
    aclsparseConstSpMatDescr_t a, aclsparseConstDnMatDescr_t b, aclsparseDnMatDescr_t c, aclsparseOperation_t opA,
    aclsparseOperation_t opB, aclDataType type, aclsparseSpSMAlg_t alg)
{
    if (h == nullptr)
    {
        return ReportFailure(ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR, __func__, __LINE__);
    }
    if (!IsPlan(d) || (analyzed ? !d->analyzed : !d->sized))
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    auto status = Validate(a, b, c, opA, opB, type, alg);
    return status == OK ? Match(h, d, alpha, a, b, c, opA, opB) : status;
}

void SetHostAlpha(aclsparseSpSMDescr_t d, const void* alpha)
{
    if (!d->plan.deviceAlpha)
    {
        const auto* value = static_cast<const float*>(alpha);
        d->plan.alphaReal = value[0];
        d->plan.alphaImag = d->plan.components == 2 ? value[1] : 0.0f;
    }
}

void SolveDiagonalPlan(aclsparseSpSMDescr_t d, aclsparseConstDnMatDescr_t b, aclsparseDnMatDescr_t c)
{
    bool overlap = Address(b->values) < Address(c->values) + DenseBytes(c)
        && Address(c->values) < Address(b->values) + DenseBytes(b);
    if (overlap)
    {
        Launch(d, SpsmPhase::SNAPSHOT, nullptr, b->values);
    }
    Launch(d, SpsmPhase::SOLVE_DIAGONAL, c->values, b->values, overlap);
}

void SolveGeneralPlan(aclsparseSpSMDescr_t d, aclsparseConstDnMatDescr_t b, aclsparseDnMatDescr_t c)
{
    bool parallelLevels = d->plan.levels <= 128 || (d->plan.levels <= 512 && d->plan.m >= 4 * d->plan.levels);
    Launch(d, parallelLevels ? SpsmPhase::SNAPSHOT : SpsmPhase::SNAPSHOT_ROOTS, nullptr, b->values);
    for (int64_t begin = 0; begin < d->plan.n; begin += d->plan.lowWidth)
    {
        if (d->plan.lowWidth < d->plan.n)
        {
            Launch(
                d, parallelLevels ? SpsmPhase::PREPARE_PANEL : SpsmPhase::PREPARE_PANEL_ROOTS, nullptr, nullptr, begin);
        }
        if (parallelLevels)
        {
            Launch(d, SpsmPhase::SOLVE_LEVELS, nullptr, nullptr, begin);
        }
        else
        {
            uint32_t columns = std::min<int64_t>(d->plan.lowWidth, d->plan.n - begin);
            Launch(d, SpsmPhase::SOLVE, nullptr, nullptr, begin, std::min(Blocks(), columns));
        }
    }
    Launch(d, SpsmPhase::SCATTER, nullptr, c->values);
}

} // namespace

aclsparseStatus_t aclsparseSpSMCreateDescr(aclsparseSpSMDescr_t* out)
{
    if (out == nullptr)
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    auto d = new (std::nothrow) aclsparseSpSMDescr;
    if (d == nullptr)
    {
        return ReportFailure(ACL_SPARSE_STATUS_ALLOC_FAILED, __func__, __LINE__);
    }
    try
    {
        std::lock_guard<std::mutex> lock(registryMutex);
        plans.insert(d);
    }
    catch (const std::bad_alloc&)
    {
        delete d;
        return ReportFailure(ACL_SPARSE_STATUS_ALLOC_FAILED, __func__, __LINE__);
    }
    *out = d;
    return OK;
}

aclsparseStatus_t aclsparseSpSMDestroyDescr(aclsparseSpSMDescr_t d)
{
    if (d == nullptr)
    {
        return OK;
    }
    std::lock_guard<std::mutex> lock(registryMutex);
    if (plans.erase(d) == 0)
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    delete d;
    return OK;
}

aclsparseStatus_t aclsparseSpSMBufferSize(aclsparseHandle_t h, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void* alpha, aclsparseConstSpMatDescr_t a, aclsparseConstDnMatDescr_t b, aclsparseDnMatDescr_t c,
    aclDataType type, aclsparseSpSMAlg_t alg, aclsparseSpSMDescr_t d, size_t* bufferSize)
{
    aclrtStream stream = nullptr;
    aclsparsePointerMode_t mode = ACL_SPARSE_POINTER_MODE_HOST;
    int32_t device = -1;
    auto status = Context(h, stream, mode, device);
    if (status != OK)
    {
        return status;
    }
    if (!IsPlan(d) || bufferSize == nullptr)
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    status = Validate(a, b, c, opA, opB, type, alg);
    if (status != OK)
    {
        return status;
    }
    if (!AlphaPointer(alpha, mode, device))
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    SpsmPlanTiling tiling { };
    int64_t bytes = 0;
    if (!Layout(a, b, c, opA, opB, tiling, bytes))
    {
        return ReportFailure(ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES, __func__, __LINE__);
    }
    BindPlan(d, h, a, b, c, alpha, opA, opB, mode, stream, device, tiling, bytes);
    *bufferSize = bytes;
    return OK;
}

aclsparseStatus_t aclsparseSpSMAnalysis(aclsparseHandle_t h, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void* alpha, aclsparseConstSpMatDescr_t a, aclsparseConstDnMatDescr_t b, aclsparseDnMatDescr_t c,
    aclDataType type, aclsparseSpSMAlg_t alg, aclsparseSpSMDescr_t d, void* buffer)
{
    auto status = CheckStage(h, d, false, alpha, a, b, c, opA, opB, type, alg);
    if (status != OK)
    {
        return status;
    }
    void* workspace = nullptr;
    status = ResolveWorkspace(h, d, buffer, workspace);
    if (status != ACL_SPARSE_STATUS_SUCCESS)
    {
        return status;
    }
    status = BindWorkspaceAndValidateStorage(d, a, workspace);
    if (status != OK)
    {
        return status;
    }
    // 下发失败不得依赖后续 ReadSummary 掩盖：enqueue 期失败时流上无在途任务，
    // 同步会空转成功而读到陈旧 summary。clear-then-launch-then-peek：先清线程级
    // 陈旧错误（先前已正确处理翻译为状态码的校验失败会留下粘滞 last error，如
    // AlphaPointer 对 host 指针查询属性失败），只归因本路径下发（issue #357）。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    BuildPlan(d, a);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS)
    {
        return ReportFailure(ACL_SPARSE_STATUS_EXECUTION_FAILED, __func__, __LINE__);
    }
    SpsmDeviceSummary summary { };
    status = ReadSummary(d, summary);
    if (status != OK)
    {
        return status;
    }
    SetAnalysisMetadata(d, summary);
    d->boundB = b->values;
    d->boundC = c->values;
    d->analyzed = true;
    return OK;
}

aclsparseStatus_t aclsparseSpSM(aclsparseHandle_t h, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void* alpha, aclsparseConstSpMatDescr_t a, aclsparseConstDnMatDescr_t b, aclsparseDnMatDescr_t c,
    aclDataType type, aclsparseSpSMAlg_t alg, aclsparseSpSMDescr_t d)
{
    auto status = CheckStage(h, d, true, alpha, a, b, c, opA, opB, type, alg);
    if (status != OK)
    {
        return status;
    }
    status = CheckDenseStorage(d, b, c, alpha);
    if (status != OK)
    {
        return status;
    }
    SetHostAlpha(d, alpha);
    // Solve 无 host 等待/回读，是三阶段中唯一完全无法经同步观察失败的路径：
    // enqueue 失败必须显式上报。clear-then-launch-then-peek 只归因本路径下发
    // （issue #357）。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    if (d->plan.levels == 1)
    {
        SolveDiagonalPlan(d, b, c);
    }
    else
    {
        SolveGeneralPlan(d, b, c);
    }
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS)
    {
        return ReportFailure(ACL_SPARSE_STATUS_EXECUTION_FAILED, __func__, __LINE__);
    }
    return OK;
}

// Update 入口的句柄/描述符/部件校验与绑定匹配（评审响应：压函数行数至 50 内）。
static aclsparseStatus_t ValidateUpdateCall(
    aclsparseHandle_t h, aclsparseSpSMDescr_t d, const void* newValues, aclsparseSpSMUpdate_t part,
    bool& general)
{
    if (h == nullptr)
    {
        return ReportFailure(ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR, __func__, __LINE__);
    }
    if (!IsPlan(d) || !d->analyzed)
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    if (part != ACL_SPARSE_SPSM_UPDATE_GENERAL && part != ACL_SPARSE_SPSM_UPDATE_DIAGONAL)
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    general = part == ACL_SPARSE_SPSM_UPDATE_GENERAL;
    return Match(h, d, d->alpha, d->inputA, d->inputB, d->outputC, d->opA, d->opB);
}

aclsparseStatus_t aclsparseSpSMUpdateMatrix(
    aclsparseHandle_t h, aclsparseSpSMDescr_t d, const void* newValues, aclsparseSpSMUpdate_t part)
{
    bool general = false;
    auto status = ValidateUpdateCall(h, d, newValues, part, general);
    if (status != OK)
    {
        return status;
    }
    int64_t count = general ? d->plan.nnz : d->plan.m;
    if (!DeviceSpan(d->buffer, d->cachedBufferSize, d->device)
        || (count > 0 && !DeviceSpan(newValues, count * 4 * d->plan.components, d->device))
        || OverlapsWorkspace(newValues, count * 4 * d->plan.components, d))
    {
        return ReportFailure(INVALID, __func__, __LINE__);
    }
    // Unit diagonals are implicit; their supplied values have no effect.
    // A GENERAL update still refreshes every off-diagonal value.
    // 该分支无 ReadSummary 同步兜底，是 Update 中唯一完全无失败观测的路径：
    // 同样走 clear-then-launch-then-peek（评审响应补齐，issue #359）。
    if (d->plan.unit)
    {
        if (general)
        {
            return LaunchChecked(d, SpsmPhase::COMMIT_GENERAL, newValues, nullptr, 0, 0);
        }
        return OK;
    }
    SpsmDeviceSummary summary { };
    status = LaunchCheckAndReadSummary(
        d, general ? SpsmPhase::CHECK_GENERAL : SpsmPhase::CHECK_DIAGONAL,
        newValues, nullptr, 0, 1, summary);
    if (status != OK)
    {
        return status;
    }
    return LaunchChecked(
        d, general ? SpsmPhase::COMMIT_GENERAL : SpsmPhase::COMMIT_DIAGONAL,
        newValues, nullptr, 0, 0);
}
