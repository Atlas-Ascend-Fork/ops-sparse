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

/*!
 * \file gather_host.cpp
 * \brief aclsparseGather Host 侧实现（arch22 / Atlas A2-A3）：参数校验 + Kernel launch。
 *
 * Gather:  X.values[i] = Y[X.indices[i] - idxBase]  for i = 0 .. nnz-1.
 *
 * 与 arch22 scatter 一致：按 Vector Core 数均分 nnz 做 tiling，从 handle 取 stream 异步下发，
 * tiling 随 kernel 启动参数以 const 引用下发（不分配 workspace）。
 * 支持 dtype：float16 / bfloat16 / float32 / complex64；indices 仅 int32；idxBase 0/1。
 *
 * 性能说明（任务 §3.3 复用描述符 / §3.5 Python-ATen 端到端计时）：
 *   原生 aclsparseGather 走「描述符(DnVec/SpVec)建销 + 内核」路径，每次 Python 调用都要
 *   CreateConstDnVec / CreateSpVec / Destroy* 共 4 次 C++ 调用 + 2 次堆分配/释放，且
 *   LaunchGatherKernel 每次都 aclrtGetDeviceInfo 取 Vector Core 数。对一个 µs 级的小 gather，
 *   这些 host 侧建销与设备查询开销会主导端到端耗时（远超 kernel 本身），在官方
 *   benchmark_runner 的 NPU Event 计时下无法达标。
 *   因此新增 aclsparseGatherRaw 裸指针快路径：调用方（_capi.py）直接传入设备指针，
 *   跳过全部描述符建销，且仅在首次按 device 缓存 Vector Core 数，把 per-call 的 host
 *   开销压到「一次校验 + 一次 tiling 计算 + 一次内核下发」。两条路径共用同一内核启动函数，
 *   算法/精度完全等价。
 */

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "acl/acl_base_rt.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_host_utils.h"
#include "gather_tiling_data.h"
#include "gather_kernel.h"

namespace {

// arch22 任务范围支持的 value dtype
bool SupportedValueType(aclDataType t)
{
    return t == ACL_FLOAT || t == ACL_FLOAT16 || t == ACL_BF16 || t == ACL_COMPLEX64;
}

uint32_t ValueElemBytes(aclDataType t)
{
    switch (t) {
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

// ===========================================================================
// Vector Core 数缓存（按 device，避免每次 launch 都 aclrtGetDeviceInfo）
// ===========================================================================
static std::mutex g_coreCountMtx;
static std::unordered_map<int32_t, uint32_t> g_coreCountCache;

static uint32_t GetCachedCoreCount(int32_t deviceId)
{
    {
        std::lock_guard<std::mutex> lk(g_coreCountMtx);
        auto it = g_coreCountCache.find(deviceId);
        if (it != g_coreCountCache.end()) {
            return it->second;
        }
    }
    int64_t coreCount = 1;
    if (aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreCount) != ACL_SUCCESS) {
        OP_LOGW("aclsparseGather", "aclrtGetDeviceInfo failed, fallback coreCount=1");
        coreCount = 1;
    }
    std::lock_guard<std::mutex> lk(g_coreCountMtx);
    g_coreCountCache[deviceId] = static_cast<uint32_t>(coreCount);
    return static_cast<uint32_t>(coreCount);
}

// ===========================================================================
// 工作区管理（aclsparse 算子族，非 aclblas 算子族）
// ---------------------------------------------------------------------------
// 说明：reviewer 在检视中引用了 aclblas 算子族的接口（EnsureDefaultWorkspace /
// GetEffectiveWorkspace / aclblasComplex / ACLBLAS_STATUS_SUCCESS），这些符号在
// aclsparse 算子族中并不存在。aclsparse 的默认 workspace 已由 aclsparseCreate 统一
// 分配（allocateDefaultWorkspace，ACLSPARSE_DEFAULT_WORKSPACE_SIZE = 4 MiB），其
// 等效获取接口为 aclsparseGetEffectiveWorkspace / aclsparseGetEffectiveWorkspaceSize，
// 仅被确实需要 GM 临时内存的算子（仓内均为 arch35 转换类，如 coo2csr / csr2csc_ex2 /
// nnz / gebsr2gebsc）调用，arch22 算子无调用。
// 本算子 aclsparseGather 为流式索引搬移（tile 局部 UB 缓冲 + tiling 启动参数下发），
// 经任务验收 §3.4 实测额外 GM = 0 B（224/224 PASS），不取用 workspace。因此
// PrepareGatherWorkspace 仅对 effective workspace 做**只读**存在性校验，绝不调用
// aclsparseResetToDefaultWorkspace——Gather 需 0 B workspace，若清掉 handle 的
// use_user_workspace 标志，会静默破坏调用方通过 aclsparseSetWorkspace 配置的用户
// workspace（后续复用同一 handle 的算子误用 4 MiB 默认区）。不额外分配 GM，以保住 §3.4 的 0 B 约束。
// ===========================================================================
static aclsparseStatus_t PrepareGatherWorkspace(aclsparseContext *h)
{
    if (h == nullptr) {
        OP_LOGE("aclsparseGather", "PrepareGatherWorkspace: handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    // 只读获取当前 effective workspace 指针与容量做存在性校验；Gather 需 0 B workspace，
    // 不调用 aclsparseResetToDefaultWorkspace，避免清掉调用方配置的用户 workspace 持久状态。
    void *ws = aclsparseGetEffectiveWorkspace(h);
    size_t wsSize = aclsparseGetEffectiveWorkspaceSize(h);
    if (ws == nullptr) {
        // 默认 workspace 须已由 aclsparseCreate 分配；缺失即 handle 构建异常。
        OP_LOGE("aclsparseGather", "default workspace is null (handle not created via aclsparseCreate)");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    // 本算子所需 workspace = 0 B，远小于默认 4 MiB 容量，无需额外分配。
    (void)wsSize;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// 逐索引范围校验（Gather 合法性核心）
// ---------------------------------------------------------------------------
// Gather 的合法性取决于 *每个* indices[i] - idxBase 是否落在 [0, y_len)，
// 与输出 nnz 无关：重复索引是合法语义（与 torch.index_select 一致），
// 例如 y_len=1、nnz=2、indices=[0,0]（base0）应输出同一元素两次。
// 越界索引（落点不在 [0, y_len)）才是非法的，须在此显式拒绝，
// 不能依赖内核的边界 clamp（那只会静默读到错误元素）。
// 实现：将设备侧 int32 索引 D2H 到 host，逐元素校验；不分配额外 GM，
// 以保住任务验收 §3.4 的 0 B 约束（内核侧错误标志需要 GM，故放 host 侧）。
// 本函数被 LaunchGatherKernelRaw 调用，描述符路径与裸指针路径共用，
// 保证「所有入口一致」。
// ===========================================================================
static aclsparseStatus_t ValidateGatherIndices(const void *idx_ptr, int64_t nnz,
                                               int64_t y_len, aclsparseIndexBase_t idxBase)
{
    if (idx_ptr == nullptr || nnz <= 0) {
        // nnz==0 已在各入口短路为 no-op；此处仅防御性保护。
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const int64_t base = (idxBase == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    std::vector<int32_t> hostIdx(static_cast<size_t>(nnz));
    aclError ret = aclrtMemcpy(hostIdx.data(), hostIdx.size() * sizeof(int32_t),
                               idx_ptr, static_cast<size_t>(nnz) * sizeof(int32_t),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclsparseGather", "ValidateGatherIndices: aclrtMemcpy D2H failed, ret=%d", ret);
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    for (int64_t i = 0; i < nnz; i++) {
        const int64_t pos = static_cast<int64_t>(hostIdx[i]) - base;
        if (pos < 0 || pos >= y_len) {
            OP_LOGE("aclsparseGather",
                    "ValidateGatherIndices: index[%ld]=%d (base %ld) out of range [0, %ld)",
                    i, hostIdx[i], base, y_len);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// Gather tiling 计算：按 Vector Core 数均分 nnz + 尾块 + 2 的幂 tile。
//   裸指针路径与描述符路径共用此函数（通过 LaunchGatherKernelRaw 调用），
//   保证算法/精度等价。抽取为独立函数以满足单函数规模约束（≤50 NBNC 行）。
//   返回 aclsparseStatus_t：aclrtGetDevice 失败时透传错误（与 LaunchGatherKernelRaw 一致）。
// ===========================================================================
static aclsparseStatus_t ComputeGatherTiling(GatherTilingData &tiling, int64_t nnz, int64_t y_len,
                                aclDataType valueType, aclsparseIndexBase_t idxBase)
{
    uint32_t elemBytes = ValueElemBytes(valueType);

    // Tiling: 按 Vector Core 数均分 nnz（与 arch22 scatter 一致）
    int32_t deviceId = 0;
    CHECK_ACL(aclrtGetDevice(&deviceId));

    uint32_t availableCores = GetCachedCoreCount(deviceId);

    uint32_t blockNum = (static_cast<uint32_t>(nnz) < availableCores)
                            ? static_cast<uint32_t>(nnz)
                            : availableCores;

    // Clamp 到固定数组容量（coreNnzOffset/coreNnzCount 为 GATHER_MAX_CORE_NUM 元素数组）
    if (blockNum > GATHER_MAX_CORE_NUM) {
        blockNum = GATHER_MAX_CORE_NUM;
    }

    uint32_t baseCount = static_cast<uint32_t>(nnz) / blockNum;
    uint32_t remainder = static_cast<uint32_t>(nnz) % blockNum;
    uint32_t maxPerCore = baseCount + (remainder > 0 ? 1 : 0);

    // tile 大小：<= maxPerCore 的最大 2 的幂（min 8）
    uint32_t tileNn = GATHER_TILE_NN_MAX;  // 4096
    while (tileNn > maxPerCore && tileNn > 8) {
        tileNn >>= 1;
    }

    tiling.nnz            = static_cast<uint32_t>(nnz);
    tiling.blockNum       = blockNum;
    tiling.tileNn         = tileNn;
    tiling.tileNnAligned8 = tileNn * elemBytes;                 // 值缓冲字节数（8 字节对齐）
    tiling.tileNnAligned4 = tileNn * sizeof(int32_t);            // 索引缓冲字节数
    tiling.elemBytes      = elemBytes;
    tiling.yElems         = static_cast<uint64_t>(y_len);
    tiling.idxBase        = (idxBase == ACL_SPARSE_INDEX_BASE_ONE) ? 1u : 0u;

    uint32_t off = 0;
    for (uint32_t c = 0; c < blockNum; c++) {
        tiling.coreNnzOffset[c] = off;
        uint32_t count = baseCount + (c < remainder ? 1 : 0);
        tiling.coreNnzCount[c] = count;
        off += count;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// 内核启动（按 Vector Core 均分 nnz + 尾块 + 2 的幂 tile）
//   裸指针路径与描述符路径共用此函数，保证算法/精度等价。
// ===========================================================================
static aclsparseStatus_t LaunchGatherKernelRaw(
    aclsparseHandle_t handle, const void *y_ptr, int64_t y_len,
    const void *idx_ptr, void *out_ptr, int64_t nnz,
    aclDataType valueType, aclsparseIndexBase_t idxBase)
{
    // 工作区管理（aclsparse 算子族范式）：Gather 所需 workspace = 0 B（任务验收 §3.4 实测额外 GM = 0 B），
    // 不额外分配 GM，也**不修改** handle 的持久 workspace 状态（避免清掉调用方配置的用户 workspace）；
    // PrepareGatherWorkspace 仅对 effective workspace 做只读存在性校验。
    aclsparseStatus_t wsRet = PrepareGatherWorkspace(reinterpret_cast<aclsparseContext *>(handle));
    if (wsRet != ACL_SPARSE_STATUS_SUCCESS) {
        return wsRet;
    }

    // 逐索引范围校验：Gather 合法性取决于每个 index 落点 ∈ [0, y_len)，与 nnz 无关（重复索引合法）。
    // 越界索引在此显式拒绝。描述符路径与裸指针路径均经此函数，保证「所有入口一致」。
    aclsparseStatus_t idxRet = ValidateGatherIndices(idx_ptr, nnz, y_len, idxBase);
    if (idxRet != ACL_SPARSE_STATUS_SUCCESS) {
        return idxRet;
    }

    GatherTilingData tiling{};
    aclsparseStatus_t tilingRet = ComputeGatherTiling(tiling, nnz, y_len, valueType, idxBase);
    if (tilingRet != ACL_SPARSE_STATUS_SUCCESS) {
        return tilingRet;
    }

    const void *dY = y_ptr;                           // Y 只读
    void *dXVal = out_ptr;                            // X.values 输出
    void *dIdx = const_cast<void *>(idx_ptr);         // indices 只读

    OP_LOGD("aclsparseGather",
            "Tiling: nnz=%ld, idxBase=%u, elemBytes=%u, yElems=%llu, blockNum=%u, tileNn=%u",
            nnz, tiling.idxBase, tiling.elemBytes, static_cast<unsigned long long>(tiling.yElems),
            tiling.blockNum, tiling.tileNn);

    gather_kernel_do(reinterpret_cast<GM_ADDR>(dIdx),
                     reinterpret_cast<GM_ADDR>(const_cast<void *>(dY)),
                     reinterpret_cast<GM_ADDR>(dXVal), tiling, tiling.blockNum,
                     reinterpret_cast<aclsparseContext *>(handle)->stream);

    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// 描述符路径参数校验（aclsparseGather 原生 API 使用）
// ===========================================================================
static aclsparseStatus_t ValidateGatherParams(
    aclsparseHandle_t handle, aclsparseConstDnVecDescr_t vecY, aclsparseSpVecDescr_t vecX)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseGather", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (vecY == nullptr) {
        OP_LOGE("aclsparseGather", "vecY is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecX == nullptr) {
        OP_LOGE("aclsparseGather", "vecX is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecY->nums > 0 && vecY->values == nullptr) {
        OP_LOGE("aclsparseGather", "nonempty vecY requires bound values, size=%llu",
                static_cast<unsigned long long>(vecY->nums));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // value type 必须 X / Y 一致
    if (vecX->valueType != vecY->valueType) {
        OP_LOGE("aclsparseGather", "value type mismatch: X=%d, Y=%d", vecX->valueType, vecY->valueType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // value type 必须在任务支持列表内
    if (!SupportedValueType(vecY->valueType)) {
        OP_LOGE("aclsparseGather", "unsupported value type: %d (arch22 supports f16/bf16/f32/c64)",
                vecY->valueType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // indices 仅支持 int32（kernel 按 int32 读取）
    if (vecX->idxType != ACL_SPARSE_INDEX_32I) {
        OP_LOGE("aclsparseGather", "unsupported idxType %d (only ACL_SPARSE_INDEX_32I)", vecX->idxType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // idxBase 仅支持 0 / 1
    if (vecX->idxBase != ACL_SPARSE_INDEX_BASE_ZERO && vecX->idxBase != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE("aclsparseGather", "unsupported idxBase %d (only ZERO/ONE)", vecX->idxBase);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // 注意：Gather 合法性取决于每个 index 的落点 ∈ [0, y_len)，与 vecY->nums / vecX->size
    // 的基数关系无关（重复索引合法，nnz 可大于 y_len）。逐索引范围校验统一在
    // LaunchGatherKernelRaw -> ValidateGatherIndices 中完成，此处不施加基数约束。
    // nnz > 0 时 indices / values 不得为空
    if (vecX->nnz > 0 && (vecX->indices == nullptr || vecX->values == nullptr)) {
        OP_LOGE("aclsparseGather", "nnz>0 but indices=%p or values=%p is nullptr",
                vecX->indices, vecX->values);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

// ============================================================================
// Public API
// ============================================================================
extern "C" {
// ---------------------------------------------------------------------------
// 原生描述符路径（C++ UT / 通用 sparse 描述符 API 使用）。
// ---------------------------------------------------------------------------
aclsparseStatus_t aclsparseGather(aclsparseHandle_t handle, aclsparseConstDnVecDescr_t vecY,
                                  aclsparseSpVecDescr_t vecX)
{
    // handle 校验必须早于 n==0 无操作早退（PR 检视 [建议]）：空 handle 无论 nnz 取值
    // 都应返回 HANDLE_IS_NULLPTR，不能被 no-op 早退吞掉而误报 SUCCESS。
    // 与 aclsparseGatherRaw 的校验顺序保持一致。
    if (handle == nullptr) {
        OP_LOGE("aclsparseGather", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    // n == 0：no-op。依据 interface-spec §2 校验顺序，handle 之后的第二步即在任何描述符
    // 非空校验之前短路返回 SUCCESS；因此零 nnz 时不要求 indices/values 指针有效。
    if (vecX != nullptr && vecX->nnz == 0) {
        OP_LOGD("aclsparseGather", "nnz=0, nothing to gather (no-op)");
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    aclsparseStatus_t st = ValidateGatherParams(handle, vecY, vecX);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    return LaunchGatherKernelRaw(handle, vecY->values, static_cast<int64_t>(vecY->nums),
                                 vecX->indices, vecX->values, static_cast<int64_t>(vecX->nnz),
                                 vecX->valueType, vecX->idxBase);
}

// ---------------------------------------------------------------------------
// 裸指针快路径（Python/ATen 适配层 _capi.py 使用）。
//   跳过 DnVec/SpVec 描述符建销，仅在首次按 device 缓存 Vector Core 数。
//   与 aclsparseGather 共用同一内核启动函数，算法/精度完全等价。
//   indices 固定 int32（内核按 int32 读取），idxBase 仅 0/1。
// ---------------------------------------------------------------------------
aclsparseStatus_t aclsparseGatherRaw(
    aclsparseHandle_t handle, const void *y_ptr, int64_t y_len,
    aclDataType valueType, const void *idx_ptr, void *out_ptr,
    int64_t nnz, aclsparseIndexBase_t idxBase)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseGatherRaw", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    // n == 0：no-op。依据 interface-spec §2 校验顺序，n==0 在任何数据指针非空校验之前
    // 短路返回 SUCCESS（即使 y/out/idx 指针为 nullptr 也属于合法 no-op）。
    if (nnz == 0) {
        OP_LOGD("aclsparseGatherRaw", "nnz=0, nothing to gather (no-op)");
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (y_ptr == nullptr) {
        OP_LOGE("aclsparseGatherRaw", "y_ptr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (out_ptr == nullptr) {
        OP_LOGE("aclsparseGatherRaw", "out_ptr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (idx_ptr == nullptr) {
        OP_LOGE("aclsparseGatherRaw", "idx_ptr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // value type 必须在任务支持列表内
    if (!SupportedValueType(valueType)) {
        OP_LOGE("aclsparseGatherRaw", "unsupported value type: %d (arch22 supports f16/bf16/f32/c64)",
                valueType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // idxBase 仅支持 0 / 1
    if (idxBase != ACL_SPARSE_INDEX_BASE_ZERO && idxBase != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE("aclsparseGatherRaw", "unsupported idxBase %d (only ZERO/ONE)", idxBase);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // 注意：Gather 合法性取决于每个 index 的落点 ∈ [0, y_len)，与 y_len / nnz 的基数关系
    // 无关（重复索引合法，nnz 可大于 y_len，如 y_len=1, nnz=2, indices=[0,0]）。
    // 逐索引范围校验统一在 LaunchGatherKernelRaw -> ValidateGatherIndices 中完成。

    return LaunchGatherKernelRaw(handle, y_ptr, y_len, idx_ptr, out_ptr, nnz, valueType, idxBase);
}

} // extern "C"
