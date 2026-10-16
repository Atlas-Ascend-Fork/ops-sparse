/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"   // CHECK_ACL 宏
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "scatter.h"
#include "scatter_kernel.h"

// ---------------------------------------------------------------------------
// Scatter execution  (cuSPARSE mode: no D2H, no dedup, no sort)
//
// Data path: raw device indices/values -> Validate -> Tiling -> Kernel launch
// Behavior matches cuSPARSE: deterministic only if indices are distinct.
//
// 能力扩展：
//   Validate（确定参数错误链）
//     - 空 handle                -> ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
//     - 空 vecX / vecY 描述符     -> ACL_SPARSE_STATUS_INVALID_VALUE
//     - idxType ≠ 32I            -> ACL_SPARSE_STATUS_INVALID_VALUE（仅支持 I32）
//     - valueType ∉ 五 dtype 集合 -> ACL_SPARSE_STATUS_INVALID_VALUE
//     - vecX.valueType ≠ vecY     -> ACL_SPARSE_STATUS_INVALID_VALUE
//     - idxBase ∉ {0,1}           -> ACL_SPARSE_STATUS_INVALID_VALUE
//     - nnz>0 且 size=0 / size>nums / 空指针 / 未声明 alias -> ACL_SPARSE_STATUS_INVALID_VALUE
//     - nnz/size > INT32_MAX（规模上界） -> ACL_SPARSE_STATUS_INVALID_VALUE
//   nnz=0 早退：直接 SUCCESS，不下发 kernel、不触碰 vecY 数据指针
//   Tiling：向量核数动态获取（禁止写死），clamp 64；dtype 对齐元素数
//     int8:32 / fp16,bf16:16 / fp32:8 / complex64:4
//   Launch：scatter_kernel_do 按 valueType 分发 5 个模板实例化，经 handle 绑定
//   stream 异步下发，0 workspace。
// ---------------------------------------------------------------------------

namespace {

// dtype -> 元素字节（校验通过后的五 dtype 集合；未支持返回 0）
inline uint32_t ScatterElemBytes(aclDataType t)
{
    switch (t) {
        case ACL_INT8:
            return 1;
        case ACL_FLOAT16:
        case ACL_BF16:
            return 2;
        case ACL_FLOAT:
            return 4;
        case ACL_COMPLEX64:
            return 8;
        default:
            return 0;
    }
}

// dtype -> 对齐元素数（32B / 元素字节，MTE 对齐约束）
inline uint32_t ScatterAlignElems(aclDataType t)
{
    switch (t) {
        case ACL_INT8:
            return 32;
        case ACL_FLOAT16:
        case ACL_BF16:
            return 16;
        case ACL_FLOAT:
            return 8;
        case ACL_COMPLEX64:
            return 4;
        default:
            return 0;
    }
}

// 未声明的输入输出 alias 检测：vecY 与 vecX.values / vecX.indices 内存区间重叠即拒绝
inline bool ScatterRangesOverlap(const void *a, uint64_t aBytes, const void *b, uint64_t bBytes)
{
    if (a == nullptr || b == nullptr || aBytes == 0 || bBytes == 0) {
        return false;
    }
    uintptr_t ua = reinterpret_cast<uintptr_t>(a);
    uintptr_t ub = reinterpret_cast<uintptr_t>(b);
    return (ua < ub + bBytes) && (ub < ua + aBytes);
}

// 校验输出的描述符内部视图与派生标量（主函数 tiling / launch 所需）
struct ScatterValidatedInputs {
    aclsparseContext *handleInner = nullptr;
    struct aclsparseSpVecDescr *spVecInner = nullptr;
    aclsparseDnVecDescr *dnVecInner = nullptr;
    int32_t idxBase = 0;      // kernel 内统一减法的 0/1 基
    uint32_t elemBytes = 0;   // dtype 元素字节（五 dtype 集合）
};

// 校验链第一段：句柄 / 描述符 / 属性：
//   空句柄 → HANDLE_IS_NULLPTR；空描述符 / idxType ≠ 32I / idxBase ∉ {0,1} /
//   valueType ∉ 五 dtype 集合 / vecX.valueType ≠ vecY → INVALID_VALUE。
aclsparseStatus_t ValidateDescriptors(aclsparseHandle_t handle,
                                      aclsparseConstSpVecDescr_t vecX,
                                      aclsparseDnVecDescr_t vecY,
                                      ScatterValidatedInputs &out)
{
    // ---- Validate：空句柄 / 空描述符 ----
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (vecX == nullptr || vecY == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    out.handleInner = reinterpret_cast<aclsparseContext *>(handle);
    out.spVecInner  = ScatterToSpVecInner(vecX);
    out.dnVecInner  = reinterpret_cast<aclsparseDnVecDescr *>(vecY);

    // ---- 索引类型：仅 I32（64I 返回参数错误）----
    if (out.spVecInner->idxType != ACL_SPARSE_INDEX_32I) {
        OP_LOGE("aclsparse", "aclsparseScatter: unsupported idxType %d (only ACL_SPARSE_INDEX_32I)",
                out.spVecInner->idxType);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // ---- idxBase：仅 0/1 ----
    if (out.spVecInner->idxBase != ACL_SPARSE_INDEX_BASE_ZERO &&
        out.spVecInner->idxBase != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE("aclsparse", "aclsparseScatter: invalid idxBase %d (only 0/1)",
                out.spVecInner->idxBase);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    out.idxBase = (out.spVecInner->idxBase == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;

    // ---- dtype：五 dtype 集合 + vecX.values 与 vecY 一致（不做 dtype 提升）----
    out.elemBytes = ScatterElemBytes(out.spVecInner->valueType);
    if (out.elemBytes == 0) {
        OP_LOGE("aclsparse", "aclsparseScatter: unsupported valueType %d",
                out.spVecInner->valueType);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (out.dnVecInner->valueType != out.spVecInner->valueType) {
        OP_LOGE("aclsparse", "aclsparseScatter: dtype mismatch (vecX.values=%d vecY=%d)",
                out.spVecInner->valueType, out.dnVecInner->valueType);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 未声明的输入输出 alias 检测：vecY 与
// vecX.values / vecX.indices 内存区间重叠即拒绝（INVALID_VALUE）
aclsparseStatus_t ValidateNoAlias(const ScatterValidatedInputs &out, int64_t nnz)
{
    auto *spVecInner = out.spVecInner;
    auto *dnVecInner = out.dnVecInner;
    const uint64_t yBytes = static_cast<uint64_t>(dnVecInner->nums) * out.elemBytes;
    const uint64_t valBytes = static_cast<uint64_t>(nnz) * out.elemBytes;
    const uint64_t idxBytes = static_cast<uint64_t>(nnz) * sizeof(int32_t);
    if (ScatterRangesOverlap(dnVecInner->values, yBytes, spVecInner->values, valBytes) ||
        ScatterRangesOverlap(dnVecInner->values, yBytes, spVecInner->indices, idxBytes)) {
        OP_LOGE("aclsparse", "aclsparseScatter: undeclared input/output alias detected");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// nnz 规模守卫：
//   nnz<0 / nnz>INT32_MAX（int64 输入显式拒绝：tiling 链将
//   nnz 收敛为 uint32、Kernel GM 偏移为 uint32 算术，越约静默截断会致错误 tiling /
//   越界 GM 写）→ INVALID_VALUE；
//   nnz==0 早退 → SUCCESS 且 proceed=false：空稀疏向量 indices/values 可合法为
//   nullptr、size 无需校验，早退判定须置于指针/shape 校验之前；
//   nnz>0 → proceed=true 继续 shape/指针/alias 校验。
aclsparseStatus_t ValidateNnzBound(const ScatterValidatedInputs &out, bool &proceed)
{
    const int64_t nnz = static_cast<int64_t>(out.spVecInner->nnz);
    if (nnz < 0) {
        OP_LOGE("aclsparse", "aclsparseScatter: invalid nnz %ld", static_cast<long>(nnz));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE("aclsparse", "aclsparseScatter: nnz %ld exceeds INT32_MAX scale bound",
                static_cast<long>(nnz));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (nnz == 0) {
        proceed = false;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    proceed = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 校验链第二段：规模 / shape / 指针 / alias：
//   nnz 规模守卫（含 nnz=0 早退旗标，见 ValidateNnzBound）；
//   nnz>0 时 nums=0 / size=0 / size > INT32_MAX /
//   size > nums / 空数据指针 / 未声明 alias → INVALID_VALUE。
aclsparseStatus_t ValidateShapesAndAlias(const ScatterValidatedInputs &out, bool &proceed)
{
    const aclsparseStatus_t nnzRet = ValidateNnzBound(out, proceed);
    if (nnzRet != ACL_SPARSE_STATUS_SUCCESS || !proceed) {
        return nnzRet;  // 错误码透传；nnz==0 早退时 proceed=false，返回 SUCCESS
    }
    auto *spVecInner = out.spVecInner;
    auto *dnVecInner = out.dnVecInner;
    const uint64_t size = spVecInner->size;
    const uint64_t nums = dnVecInner->nums;
    const int64_t nnz = static_cast<int64_t>(spVecInner->nnz);

    // ---- shape / 指针 / alias（nnz>0 时）----
    if (nums == 0) {
        OP_LOGE("aclsparse", "aclsparseScatter: nnz=%ld but vecY size is 0", static_cast<long>(nnz));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // vecX size=0 且 nnz>0：拒绝；描述符层 nnz≤size 校验先行拦截时错误码一致
    // （纵深防御）
    if (size == 0) {
        OP_LOGE("aclsparse", "aclsparseScatter: nnz=%ld but vecX size is 0", static_cast<long>(nnz));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // 规模上界（同 nnz：size ≤ INT32_MAX）
    if (size > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE("aclsparse", "aclsparseScatter: size %lu exceeds INT32_MAX scale bound",
                static_cast<unsigned long>(size));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (size > nums) {
        OP_LOGE("aclsparse", "aclsparseScatter: SpVec size=%lu > vecY nums=%lu",
                static_cast<unsigned long>(size), static_cast<unsigned long>(nums));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (spVecInner->indices == nullptr || spVecInner->values == nullptr ||
        dnVecInner->values == nullptr) {
        OP_LOGE("aclsparse", "aclsparseScatter: null data pointer (indices=%p values=%p y=%p)",
                spVecInner->indices, spVecInner->values, dnVecInner->values);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // 未声明的输入输出 alias（vecY 与 vecX.values / vecX.indices 内存重叠）拒绝
    const aclsparseStatus_t aliasRet = ValidateNoAlias(out, nnz);
    if (aliasRet != ACL_SPARSE_STATUS_SUCCESS) {
        return aliasRet;
    }
    proceed = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 参数校验链（串行子函数编排）：
//   1) ValidateDescriptors：句柄/描述符/idxType/idxBase/dtype 白名单与一致性；
//   2) ValidateShapesAndAlias：nnz 规模守卫（ValidateNnzBound，含 nnz=0 早退旗标）/
//      shape/指针/未声明 alias（ValidateNoAlias）。
// 返回值：拒绝时为对应错误码（调用方直接透传）；校验通过返回 SUCCESS 且置 proceed。
aclsparseStatus_t ValidateScatterInputs(aclsparseHandle_t handle,
                                        aclsparseConstSpVecDescr_t vecX,
                                        aclsparseDnVecDescr_t vecY,
                                        ScatterValidatedInputs &out,
                                        bool &proceed)
{
    const aclsparseStatus_t descRet = ValidateDescriptors(handle, vecX, vecY, out);
    if (descRet != ACL_SPARSE_STATUS_SUCCESS) {
        return descRet;
    }
    return ValidateShapesAndAlias(out, proceed);
}

} // namespace

extern "C" aclsparseStatus_t aclsparseScatter(
    aclsparseHandle_t handle,
    aclsparseConstSpVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY)
{
    // ---- Validate：参数校验链 ----
    ScatterValidatedInputs in;
    bool proceed = false;
    const aclsparseStatus_t validateRet = ValidateScatterInputs(handle, vecX, vecY, in, proceed);
    if (validateRet != ACL_SPARSE_STATUS_SUCCESS) {
        return validateRet;
    }
    // nnz=0 早退：成功返回，不触碰 vecY 数据指针、不下发 kernel
    if (!proceed) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    auto *handleInner = in.handleInner;
    auto *spVecInner  = in.spVecInner;
    auto *dnVecInner  = in.dnVecInner;
    const int32_t idxBaseVal = in.idxBase;
    const uint32_t elemBytes = in.elemBytes;

    const int64_t nnz = static_cast<int64_t>(spVecInner->nnz);
    const uint64_t size = spVecInner->size;
    const uint64_t nums = dnVecInner->nums;

    // Raw device pointers — data stays on device, no D2H round-trip
    void *dVal = spVecInner->values;
    void *dIdx = spVecInner->indices;
    void *dY   = dnVecInner->values;

    // -----------------------------------------------------------------------
    // Tiling: power-of-2 tile size for all kernel versions
    // -----------------------------------------------------------------------
    int32_t deviceId = 0;
    CHECK_ACL(aclrtGetDevice(&deviceId));

    int64_t coreCount = 1;
    CHECK_ACL(aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreCount));

    uint32_t availableCores = static_cast<uint32_t>(coreCount);
    uint32_t blockNum = (static_cast<uint32_t>(nnz) < availableCores)
                            ? static_cast<uint32_t>(nnz)
                            : availableCores;

    // Clamp to fixed-array capacity: coreNnzOffset/coreNnzCount in
    // ScatterTilingData are SCATTER_MAX_CORE_NUM-element arrays. Without this
    // guard, devices with >64 vector cores and nnz >64 would yield blockNum >64,
    // causing out-of-bounds writes on the host stack array and out-of-bounds
    // reads on kernel-side blockIdx >= 64.
    if (blockNum > SCATTER_MAX_CORE_NUM) {
        blockNum = SCATTER_MAX_CORE_NUM;
    }

    uint32_t baseCount = static_cast<uint32_t>(nnz) / blockNum;
    uint32_t remainder = static_cast<uint32_t>(nnz) % blockNum;
    uint32_t maxPerCore = baseCount + (remainder > 0 ? 1 : 0);

    // Tile size: smallest power of 2 >= maxPerCore (range [8, 4096]).
    // A floor-sized tile splits small per-core workloads into extra kernel
    // iterations and therefore extra DMA/synchronization rounds.
    uint32_t tileNn = 8;
    while (tileNn < maxPerCore && tileNn < SCATTER_TILE_NN_MAX) {
        tileNn <<= 1;
    }

    // dtype 对齐元素数上取整（AllocTensor 尺寸 / Copy 快路径对齐口径）
    const uint32_t alignElems = ScatterAlignElems(spVecInner->valueType);
    // 防御纵深：上游校验链已拦截未支持 dtype（五 dtype 集合 alignElems 恒非 0），
    // 此处兜底拒绝，杜绝 tileNnAligned 计算的除零路径
    if (alignElems == 0) {
        OP_LOGE("aclsparse", "aclsparseScatter: unsupported valueType %d", spVecInner->valueType);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    ScatterTilingData tilingData{};
    tilingData.nnz           = static_cast<uint32_t>(nnz);
    tilingData.blockNum      = blockNum;
    tilingData.tileNn        = tileNn;
    tilingData.tileNnAligned = ((tileNn + alignElems - 1) / alignElems) * alignElems;
    tilingData.idxBase       = idxBaseVal;
    tilingData.yLen          = nums * elemBytes;

    uint32_t off = 0;
    for (uint32_t c = 0; c < blockNum; c++) {
        tilingData.coreNnzOffset[c] = off;
        uint32_t count = baseCount + (c < remainder ? 1 : 0);
        tilingData.coreNnzCount[c] = count;
        off += count;
    }

    // -----------------------------------------------------------------------
    // Kernel launch with raw device pointers
    //
    // Tiling 随 kernel 启动参数一起下发（const 引用），Host 侧不单独分配
    // device 内存存 tiling。kernel_do 按 valueType 分发 5 个模板实例化。
    // -----------------------------------------------------------------------
    scatter_kernel_do(dVal, dIdx, dY, spVecInner->valueType, tilingData, blockNum,
                      handleInner->stream);

    return ACL_SPARSE_STATUS_SUCCESS;
}
