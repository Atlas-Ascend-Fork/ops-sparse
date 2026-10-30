/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_PRUNE_CHECK_NPU_WRAPPER_H_
#define TEST_PRUNE_CHECK_NPU_WRAPPER_H_

// =============================================================================
// PruneCheck NPU wrapper：A/B 稀疏侧与批量执行链（描述符构造 + 执行 + 计时）。
//
// 复用既有件（禁止重复定义）：RAII Guard 与 EventGuard / NpuDtypeTrait /
// PreparePhysicalA / cptr helpers（matmulDescriptorInit 与 prune 家族 wrapper）、
// DeviceBuffer（test/frame/descriptor_manager.h）。
// 执行协议：非批量链 1 次 warm-up + R 轮 EventGuard 计时
//（PRUNE_CHECK_PERF_ROUNDS，默认 5）；批量链为单次调用语义（无计时）。
// padding ld 扩展：SparseSideSpecOf 增加可选 ldOverride
// 参数（缺省 0 = 自动推导不变），>0 时校验合法性（≥物理行长、32B/elem 对齐）
// 并采用；执行链（PruneCheckNpu/BSparse）透传 —— 设备缓冲按 ld 分配行距，
// host 侧数据准备（PrepareSparseSideData）在 padding 区写入非零垃圾。
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparseLt.h"
#include "descriptor_manager.h"
#include "matmulDescriptorInit_npu_wrapper.h"
#include "prune_npu_wrapper.h"  // EventGuard / NpuDtypeTrait / PreparePhysicalA / cptr helpers
#include "prune_test_util.h"
#include "transpose_utils.h"

// -----------------------------------------------------------------------------
// 执行结果：d_valid（D2H 后 host 单元素）+ 逐轮 device 耗时。
// -----------------------------------------------------------------------------
struct PruneCheckNpuResult {
    aclsparseStatus_t checkRet = ACL_SPARSE_STATUS_SUCCESS;
    int hValid = -1;               // d_valid 拷回值（-1 = 未读到）
    std::vector<double> deviceMs;  // 每轮 device 耗时（EventGuard，不含 warm-up）
    size_t scannedBytes = 0;       // 物理矩阵字节数（沿剪枝维度整矩阵必读）
};

// 性能轮数：默认 5，可经 PRUNE_CHECK_PERF_ROUNDS 覆盖（波动用例可调大）。
inline int PruneCheckPerfRounds()
{
    const char* env = std::getenv("PRUNE_CHECK_PERF_ROUNDS");
    if (env != nullptr) {
        const int r = std::atoi(env);
        if (r >= 1 && r <= 1000) { return r; }
    }
    return 5;
}

namespace sparse_test {

// -----------------------------------------------------------------------------
// Sparse-side physical layout spec (shared by the A/B sparse sides; the
// DESC_INVALID interception probe and the normal path reuse the same
// derivation to avoid forking).
// (b) cuSPARSELt semantics (order is the physical memory layout; the kernel
// scan view = the row-major interpretation of the descriptor physical
// buffer's (td.rows, td.cols, ld), bit-exact across the four combinations):
//   RowView (!trans+ROW / trans+COL): physical is row-major under the op
//     view — !trans+ROW: (m,k) ROW ld=k (passed through as-is);
//     trans+COL: (k,m) COL ld=k ≡ (m,k) row-major ld=k (passed through
//     as-is, byte-equivalent).
//   ColView (!trans+COL / trans+ROW): physical is (k,m) row-major ld=m
//     (the transpose of the op view; needTranspose=true: logical (m,k) ->
//     (k,m) transpose before upload).
// -----------------------------------------------------------------------------
struct SparseSidePhysicalSpec {
    int64_t physRows = 0;   // descriptor rows (natural dims: trans ? k : m)
    int64_t physCols = 0;   // descriptor cols (natural dims: trans ? m : k)
    int64_t ld = 0;         // physical row stride (in elements; compact = physical row length)
    size_t physElems = 0;   // physical buffer element count (row-major view rows x ld)
    bool needTranspose = false;  // ColView: needs logical (m,k) -> physical (k,m) transpose before upload
};

// ld 覆盖说明（padding ld 用例）：ldOverride > 0 时采用
// 自定义物理行步长（padding 布局：ld > 物理行长），合法性校验与描述符层
// 同口径 —— ldOverride ≥ 物理行长（RowView k / ColView m，IsLdValid 的
// ROW 序 ld≥cols / COL 序 ld≥rows 约束天然覆盖）且为 32B/elem 元素倍数
//（AllocateAndValidateMatDesc 整除约束，FP32:8 / FP16,BF16:16 / INT8:32）。
// 非法时 ADD_FAILURE 并保持自动推导。
template <typename T>
inline SparseSidePhysicalSpec SparseSideSpecOf(int32_t m, int32_t k, bool trans,
                                               aclsparseOrder_t order, int64_t ldOverride = 0)
{
    SparseSidePhysicalSpec s;
    const bool isColView = (trans == (order == ACL_SPARSE_ORDER_ROW));
    // 描述符自然维（不变）：trans ? (k,m) : (m,k)。物理 buffer 的行主序
    // 扫描视图（kernel DeriveCheckShape 同源）：RowView (m,k) ld=k /
    // ColView (k,m) ld=m（≡ (m,k) COL ld=m 的同一字节，needTranspose 即
    // ColView 的装配转置）。IsLdValid 四组合天然满足。
    s.physRows = trans ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    s.physCols = trans ? static_cast<int64_t>(m) : static_cast<int64_t>(k);
    s.ld = isColView ? static_cast<int64_t>(m) : static_cast<int64_t>(k);
    s.needTranspose = isColView;
    // padding ld 覆盖（ldOverride > 0）：校验后采用（语义见函数头注释）。
    if (ldOverride > 0) {
        const int64_t alignElems = static_cast<int64_t>(32 / NpuDtypeTrait<T>::kEltSize);
        if (ldOverride < s.ld || ldOverride % alignElems != 0) {
            ADD_FAILURE() << "invalid ldOverride=" << ldOverride
                          << " (physical row length=" << s.ld
                          << ", 32B/elem alignment requires multiple of " << alignElems
                          << " elements)";
        } else {
            s.ld = ldOverride;
        }
    }
    // 物理缓冲元素数（行主序扫描视图 rows × ld 的满矩形——kernel 扫描跨度
    // (rows-1)*ld+cols 的紧凑上取整；padding ld 覆盖时缓冲按 ld 行距分配，
    // 数据区 + padding 区全覆盖）。行主序视图行数：RowView m / ColView k。
    s.physElems = static_cast<size_t>(isColView ? k : m) *
                  static_cast<size_t>(s.ld);
    return s;
}

// padding 区垃圾填充值（非零、确定性、全 dtype）：FP32/FP16/BF16 = 1.0 的
// 位模式、INT8 = 1 —— padding 用例要求「若参与判定必然违规」（全非零组），
// 用以佐证 kernel CopyIn 的 srcStride 跳过 padding（padding 数据不参与判定）。
template <typename T>
inline T PruneCheckPaddingGarbage()
{
    if constexpr (std::is_same_v<T, float>) {
        return 1.0f;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return Fp32ToFp16Bits(1.0f);
    } else if constexpr (std::is_same_v<T, bf16_bits_t>) {
        return bf16_bits_t(Fp32ToBf16Bits(1.0f));
    } else {
        return static_cast<T>(1);
    }
}

// 物理数据准备（T+ROW 转置；其余原样），与 prune 家族布局语义一致。
// padding ld（spec.ld > 物理行长）扩展：数据区按行散布（T+ROW 先转置），
// padding 区 [物理行长, ld) 填非零垃圾（PruneCheckPaddingGarbage）——
// 设备缓冲按 ld 行距分配，host 侧数据准备同步写入垃圾。
// 紧致布局（ld == 物理行长，既有全部用例）行为不变。
// 物理视图 = (td.rows, td.cols, ld) 行主序（host DeriveCheckShape 与 kernel
// 扫描同源）：RowView (m,k) / ColView (k,m)，needTranspose 即 ColView。
template <typename T>
inline std::vector<T> PrepareSparseSideData(const std::vector<T>& hLogical,
                                            const SparseSidePhysicalSpec& spec,
                                            int32_t m, int32_t k)
{
    // 物理行数 / 物理行有效元素数（ColView 物理为转置 (k, m)，RowView 为 (m, k)）。
    const int64_t physRowCnt = spec.needTranspose ? static_cast<int64_t>(k)
                                                  : static_cast<int64_t>(m);
    const int64_t physRowLen = spec.needTranspose ? static_cast<int64_t>(m)
                                                  : static_cast<int64_t>(k);
    if (spec.ld <= physRowLen) {
        // 紧致布局（SparseSideSpecOf 覆盖校验保证 ld ≥ 物理行长，此处等号
        // 为唯一可达形态；< 为防御性兜底）：T+ROW 转置，其余原样直传。
        return spec.needTranspose ? TransposeToRowMajor<T>(hLogical, m, k, spec.ld)
                                  : hLogical;
    }
    // padding ld：全缓冲先铺非零垃圾（覆盖 padding 区），再散布数据区
    //（T+ROW 取转置元素 logical[c][r]，其余直传 logical[r][c]）。
    std::vector<T> hPhys(static_cast<size_t>(physRowCnt) * static_cast<size_t>(spec.ld),
                         PruneCheckPaddingGarbage<T>());
    for (int64_t r = 0; r < physRowCnt; ++r) {
        for (int64_t c = 0; c < physRowLen; ++c) {
            const size_t srcIdx = spec.needTranspose
                ? static_cast<size_t>(c) * static_cast<size_t>(k) + static_cast<size_t>(r)
                : static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(c);
            hPhys[static_cast<size_t>(r) * static_cast<size_t>(spec.ld) + static_cast<size_t>(c)] =
                hLogical[srcIdx];
        }
    }
    return hPhys;
}

// 计算类型派生（A/B 稀疏侧与批量三执行链共用）：INT8 → 32I，其余 → 32F
//（prune PruneNpuSetup 同款映射）。
template <typename T>
inline constexpr aclsparseComputeType_t PruneCheckComputeType()
{
    return std::is_same_v<T, int8_t> ? ACL_SPARSE_COMPUTE_32I : ACL_SPARSE_COMPUTE_32F;
}

// 稀疏侧 structured 描述符构造（A/B 稀疏侧与批量三执行链及 DESC_INVALID 探测
// 共用）：rows/cols/ld 取 SparseSideSpecOf 的物理布局规格，sparsity 固定
// 50_PERCENT。返回 initStructured 状态码，调用侧按需检查。
template <typename T>
inline aclsparseStatus_t InitSparseSideDescriptor(const aclsparseLtHandle_t* handle,
                                                  const SparseSidePhysicalSpec& spec,
                                                  aclsparseOrder_t order, MatDescrGuard& mat)
{
    return mat.initStructured(handle, spec.physRows, spec.physCols, spec.ld, 16,
                              NpuDtypeTrait<T>::kAclDtype, order,
                              ACL_SPARSE_LT_SPARSITY_50_PERCENT);
}

// -----------------------------------------------------------------------------
// 描述符层拦截探测（expect_status=DESC_INVALID 用例）：
// 仅构造稀疏侧 StructuredDescriptorInit，返回其状态码。
// 非 32B/elem 整除 shape（rows/cols/ld 非 FP32:8 / FP16,BF16:16 / INT8:32 倍数）
// 预期 ACL_SPARSE_STATUS_INVALID_VALUE（AllocateAndValidateMatDesc 整除拒绝，
// 尾组在描述符可达空间内不可达）。
// -----------------------------------------------------------------------------
template <typename T>
inline aclsparseStatus_t TryInitSparseSideDescriptor(const aclsparseLtHandle_t* handle,
                                                     int32_t m, int32_t k, bool trans,
                                                     aclsparseOrder_t order)
{
    const SparseSidePhysicalSpec spec = SparseSideSpecOf<T>(m, k, trans, order);
    MatDescrGuard matSparse;
    return InitSparseSideDescriptor<T>(handle, spec, order, matSparse);
}

// A 稀疏侧执行链的 dense 占位描述符三件套（matB/matC/matD，PruneCheckNpu 与
// PruneCheckNpuBatch 共用）：
//   - matB：rows = A 逻辑列数 k（md_k = transA ? matA->rows : matA->cols = k，
//     与 prune wrapper 的 logicalK 口径一致；md 维度派生只取自 matA/matB 的
//     rows/cols，占位形状不影响校验结果）；INT8 描述符 op/order 组合规避
//     （matB order 置 COL）沿用 prune 既有处理。
//   - matC/matD：m × kPruneN dense 占位。
// 逐个构造并检查，返回首个失败的 initDense 状态码：PruneCheckNpuBatch 将其
// 透传为 initRet；PruneCheckNpu 忽略返回值，失败由后续链路暴露。
template <typename T>
inline aclsparseStatus_t InitASideDensePlaceholders(const aclsparseLtHandle_t* handle,
                                                    int32_t m, int32_t k,
                                                    MatDescrGuard& matB, MatDescrGuard& matC,
                                                    MatDescrGuard& matD)
{
    constexpr int32_t kPruneN = 128;
    const aclsparseOrder_t matBOrder = std::is_same_v<T, int8_t>
        ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW;
    aclsparseStatus_t ret = matB.initDense(handle, k, kPruneN, kPruneN, 16,
                                           NpuDtypeTrait<T>::kAclDtype, matBOrder);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
    ret = matC.initDense(handle, m, kPruneN, kPruneN, 16,
                         NpuDtypeTrait<T>::kAclDtype, ACL_SPARSE_ORDER_ROW);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
    return matD.initDense(handle, m, kPruneN, kPruneN, 16,
                          NpuDtypeTrait<T>::kAclDtype, ACL_SPARSE_ORDER_ROW);
}

// B 稀疏侧执行链的 dense 占位描述符三件套（matA/matC/matD，PruneCheckNpuBSparse
// 与 PruneCheckNpuBatchBSparse 共用）：
//   - matA：rows = cols = mB 方阵（md->k = matA->cols = mB = B rows，与 prune
//     wrapper PruneNpuBSparse 的占位派生口径一致；占位形状不影响校验结果）。
//   - matC/matD：mB × kB dense 占位（md->m = mB、md->n = kB）。
// 逐个构造并检查，返回首个失败的 initDense 状态码（口径同 InitASideDense
// Placeholders：批量链透传为 initRet，非批量链忽略、失败由后续链路暴露）。
template <typename T>
inline aclsparseStatus_t InitBSideDensePlaceholders(const aclsparseLtHandle_t* handle,
                                                    int32_t mB, int32_t kB,
                                                    MatDescrGuard& matA, MatDescrGuard& matC,
                                                    MatDescrGuard& matD)
{
    aclsparseStatus_t ret = matA.initDense(handle, mB, mB, mB, 16,
                                           NpuDtypeTrait<T>::kAclDtype,
                                           ACL_SPARSE_ORDER_ROW);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
    ret = matC.initDense(handle, mB, kB, kB, 16,
                         NpuDtypeTrait<T>::kAclDtype, ACL_SPARSE_ORDER_ROW);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
    return matD.initDense(handle, mB, kB, kB, 16,
                          NpuDtypeTrait<T>::kAclDtype, ACL_SPARSE_ORDER_ROW);
}

// -----------------------------------------------------------------------------
// E7 辅助：在合法 DeviceBuffer 内构造非 32B 对齐指针（偏移后地址 ≡ 4 mod 16，
// 同时非 16B/32B 对齐）。
// -----------------------------------------------------------------------------
inline void* MisalignedOffsetPtr(DeviceBuffer& buf)
{
    auto* base = static_cast<uint8_t*>(buf.get());
    const uintptr_t addr = reinterpret_cast<uintptr_t>(base);
    const uintptr_t off = ((16u - (addr % 16u)) % 16u) + 4u;  // ∈ [4, 20]
    return base + off;
}

// -----------------------------------------------------------------------------
// E7b 辅助：在合法 DeviceBuffer 内构造 16B 对齐但非 32B 对齐指针（偏移后地址
// ≡ 16 mod 32）—— d_in 作为 DataCopyPad GM 源地址要求 32B 对齐，16B 非 32B
// 场景须被校验表 #8 拦截。
// -----------------------------------------------------------------------------
inline void* HalfAlignedOffsetPtr(DeviceBuffer& buf)
{
    auto* base = static_cast<uint8_t*>(buf.get());
    const uintptr_t addr = reinterpret_cast<uintptr_t>(base);
    const uintptr_t off = ((32u - (addr % 32u)) % 32u) + 16u;  // ≡ 16 mod 32
    return base + off;
}

}  // namespace sparse_test

// -----------------------------------------------------------------------------
// 多轮计时执行段（1 warm-up + R 轮计时 + 末尾一次同步 + D2H 拷回）：
// PruneCheckNpu（A 侧）/ PruneCheckNpuBSparse（B 侧）共用（重复块收敛）；
// 描述符链与输入缓冲由调用侧构造。tag 仅用于失败日志的侧别标识
//（"SpMMAPruneCheck" / "SpMMAPruneCheck(B)"）。
// -----------------------------------------------------------------------------
inline PruneCheckNpuResult RunTimedPruneCheck(sparse_test::SparseLtHandleGuard& handle,
                                              sparse_test::MatmulDescrGuard& md,
                                              sparse_test::DeviceBuffer& dIn,
                                              sparse_test::DeviceBuffer& dValid,
                                              aclrtStream stream, const char* tag)
{
    using namespace sparse_test;
    PruneCheckNpuResult result;

    // ---- 多轮计时执行（1 warm-up + R timed，末尾一次同步）----
    const int rounds = PruneCheckPerfRounds();
    std::vector<EventGuard> ev(static_cast<size_t>(rounds) + 1);

    // warm-up（默认流语义下 stream 可为 nullptr —— E11 路径同样走此函数）
    result.checkRet = aclsparseLtSpMMAPruneCheck(
        handle.ptr(), matmulDescCptr(md), dIn.get(),
        static_cast<int*>(dValid.get()), stream);
    if (result.checkRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] " << tag << " warm-up failed: " << result.checkRet << std::endl;
        return result;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        result.checkRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return result;
    }

    ev[0].record(stream);
    for (int r = 0; r < rounds; ++r) {
        result.checkRet = aclsparseLtSpMMAPruneCheck(
            handle.ptr(), matmulDescCptr(md), dIn.get(),
            static_cast<int*>(dValid.get()), stream);
        if (result.checkRet != ACL_SPARSE_STATUS_SUCCESS) {
            std::cerr << "[NPU] " << tag << " round " << r << " failed: "
                      << result.checkRet << std::endl;
            break;
        }
        ev[static_cast<size_t>(r) + 1].record(stream);
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        result.checkRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return result;
    }
    for (int r = 0; r < rounds; ++r) {
        result.deviceMs.push_back(
            static_cast<double>(EventGuard::elapsedMs(ev[static_cast<size_t>(r)],
                                                      ev[static_cast<size_t>(r) + 1])));
    }

    int hValid = -1;
    dValid.copyToHost(&hValid, sizeof(int));
    result.hValid = hValid;
    return result;
}

// -----------------------------------------------------------------------------
// A 稀疏侧执行链（对齐 PruneNpu 的描述符派生）。
// m/k 为稀疏矩阵 A 自身 (rows, cols)（逻辑 shape）；op/order 为 opA / matA order。
// ldOverride > 0 时透传自定义物理行步长（padding ld 布局：设备缓冲按 ld 行距
// 分配、host 侧 padding 区写入非零垃圾，见 SparseSideSpecOf / PrepareSparse
// SideData；缺省 0 = 按物理行长自动推导，既有调用不变）。
// -----------------------------------------------------------------------------
template <typename T>
inline PruneCheckNpuResult PruneCheckNpu(
    aclrtStream stream,
    const std::vector<T>& hA,
    int32_t m, int32_t k,
    aclsparseOperation_t opA, aclsparseOrder_t order,
    int64_t ldOverride = 0)
{
    using namespace sparse_test;

    const bool transA = (opA == ACL_SPARSE_OP_TRANSPOSE);
    const SparseSidePhysicalSpec spec = SparseSideSpecOf<T>(m, k, transA, order, ldOverride);
    const std::vector<T> hAPhys = PrepareSparseSideData<T>(hA, spec, m, k);

    constexpr size_t kElt = NpuDtypeTrait<T>::kEltSize;
    DeviceBuffer dIn = DeviceBuffer::copyFrom(hAPhys.data(), spec.physElems * kElt);
    DeviceBuffer dValid = DeviceBuffer::alloc(sizeof(int));

    SparseLtHandleGuard handle;
    MatDescrGuard matA;  // 稀疏侧（structured）
    // 正常用例该 init 必须成功（合法 shape/dtype）：失败即测试环境或构造缺陷，
    // 立即失败定位，不进入后续描述符/执行链。
    const aclsparseStatus_t initSparseRet = InitSparseSideDescriptor<T>(handle.ptr(), spec, order, matA);
    if (initSparseRet != ACL_SPARSE_STATUS_SUCCESS) {
        ADD_FAILURE() << "[NPU] SpMMAPruneCheck sparse-side (matA) init failed: "
                      << initSparseRet;
        PruneCheckNpuResult result;
        result.checkRet = initSparseRet;
        return result;
    }
    // matB/matC/matD dense 占位三件套（占位形状派生口径见 helper 注释）。
    MatDescrGuard matB;
    MatDescrGuard matC;
    MatDescrGuard matD;
    InitASideDensePlaceholders<T>(handle.ptr(), m, k, matB, matC, matD);
    constexpr aclsparseComputeType_t kComputeType = PruneCheckComputeType<T>();
    MatmulDescrGuard md;
    md.init(handle.ptr(), opA, ACL_SPARSE_OP_NON_TRANSPOSE,
            matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), kComputeType);

    // ---- 多轮计时执行 + D2H（共用执行段 RunTimedPruneCheck，A/B 侧同协议）----
    PruneCheckNpuResult result =
        RunTimedPruneCheck(handle, md, dIn, dValid, stream, "SpMMAPruneCheck");
    result.scannedBytes = spec.physElems * kElt;
    return result;
}

// -----------------------------------------------------------------------------
// B 稀疏侧执行链（对齐 PruneNpuBSparse 的描述符派生）。
// mB/kB 为 B 自身 (rows, cols) = (逻辑 k, 逻辑 n)；op/order 为 opB / matB order。
// matA 为 dense 占位（md->k = matA->cols = mB）。
// ldOverride 语义同 PruneCheckNpu（padding ld 布局透传，缺省 0 不变）。
// -----------------------------------------------------------------------------
template <typename T>
inline PruneCheckNpuResult PruneCheckNpuBSparse(
    aclrtStream stream,
    const std::vector<T>& hB,
    int32_t mB, int32_t kB,
    aclsparseOperation_t opB, aclsparseOrder_t order,
    int64_t ldOverride = 0)
{
    using namespace sparse_test;

    const bool transB = (opB == ACL_SPARSE_OP_TRANSPOSE);
    const SparseSidePhysicalSpec spec = SparseSideSpecOf<T>(mB, kB, transB, order, ldOverride);
    const std::vector<T> hBPhys = PrepareSparseSideData<T>(hB, spec, mB, kB);

    constexpr size_t kElt = NpuDtypeTrait<T>::kEltSize;
    DeviceBuffer dB = DeviceBuffer::copyFrom(hBPhys.data(), spec.physElems * kElt);
    DeviceBuffer dValid = DeviceBuffer::alloc(sizeof(int));

    SparseLtHandleGuard handle;
    // matA/matC/matD dense 占位三件套（占位形状派生口径见 helper 注释）。
    MatDescrGuard matA;
    MatDescrGuard matC;
    MatDescrGuard matD;
    InitBSideDensePlaceholders<T>(handle.ptr(), mB, kB, matA, matC, matD);
    MatDescrGuard matB;  // 稀疏侧（structured）
    // 正常用例该 init 必须成功（合法 shape/dtype）：失败即测试环境或构造缺陷，
    // 立即失败定位，不进入后续描述符/执行链。
    const aclsparseStatus_t initSparseRet = InitSparseSideDescriptor<T>(handle.ptr(), spec, order, matB);
    if (initSparseRet != ACL_SPARSE_STATUS_SUCCESS) {
        ADD_FAILURE() << "[NPU] SpMMAPruneCheck sparse-side (matB) init failed: "
                      << initSparseRet;
        PruneCheckNpuResult result;
        result.checkRet = initSparseRet;
        return result;
    }
    constexpr aclsparseComputeType_t kComputeType = PruneCheckComputeType<T>();
    MatmulDescrGuard md;
    md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE, opB,
            matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), kComputeType);

    // ---- 多轮计时执行 + D2H（共用执行段 RunTimedPruneCheck，A/B 侧同协议）----
    PruneCheckNpuResult result =
        RunTimedPruneCheck(handle, md, dB, dValid, stream, "SpMMAPruneCheck(B)");
    result.scannedBytes = spec.physElems * kElt;
    return result;
}

// -----------------------------------------------------------------------------
// A 稀疏侧批量执行链（公共 API 路径）：aclsparseLtMatDescSetAttribute 设置
// NUM_BATCHES（A/B/C/D 四侧一致，MatmulDescriptorInit 校验）与 BATCH_STRIDE
//（仅稀疏侧 matA 被 PruneCheck 消费）后单次调用 + 同步 + D2H。
// hPhys 为完整批量物理缓冲：batch b 基址 = b×batchStride 元素，末 batch 尾部
// 覆盖单 batch 物理跨度；单次调用（无 warm-up / 计时轮），供批量语义断言。
// initRet 记录 MatmulDescriptorInit 状态（numBatches 一致性由描述符层承载，
// 失败时 checkRet 不再发起）。
//
// PruneCheckBatchResult：NPU 批量执行链（PruneCheckNpuBatch /
// PruneCheckNpuBatchBSparse）的执行结果结构，与 PruneCheckNpuResult 同族；
// 批量链为单次调用语义（无计时字段），命名不带 Npu 段。
// -----------------------------------------------------------------------------
struct PruneCheckBatchResult {
    aclsparseStatus_t initRet = ACL_SPARSE_STATUS_SUCCESS;   // MatmulDescriptorInit
    aclsparseStatus_t checkRet = ACL_SPARSE_STATUS_SUCCESS;  // SpMMAPruneCheck
    int hValid = -1;                                         // d_valid 拷回值
};

// -----------------------------------------------------------------------------
// 批量属性设置（PruneCheckNpuBatch / PruneCheckNpuBatchBSparse 共用）：对四个
// 描述符（matmul 顺序）设 NUM_BATCHES 一致（一致性校验在 MatmulDescriptorInit）
// + BATCH_STRIDE 仅稀疏侧（sparseIsA ? matA : matB；PruneCheck 只消费稀疏侧
// batchStride）。逐项设置并检查，返回首个失败状态码。
// -----------------------------------------------------------------------------
inline aclsparseStatus_t SetBatchAttributes(const aclsparseLtHandle_t* handle,
                                            sparse_test::MatDescrGuard& matA,
                                            sparse_test::MatDescrGuard& matB,
                                            sparse_test::MatDescrGuard& matC,
                                            sparse_test::MatDescrGuard& matD,
                                            bool sparseIsA,
                                            int32_t numBatches, int64_t batchStride)
{
    for (sparse_test::MatDescrGuard* mat : {&matA, &matB, &matC, &matD}) {
        const int32_t nb = numBatches;
        const aclsparseStatus_t ret = aclsparseLtMatDescSetAttribute(
            handle, mat->ptr(), ACLSPARSELT_MAT_NUM_BATCHES, &nb, sizeof(nb));
        if (ret != ACL_SPARSE_STATUS_SUCCESS) { return ret; }
    }
    sparse_test::MatDescrGuard& matSparse = sparseIsA ? matA : matB;
    return aclsparseLtMatDescSetAttribute(
        handle, matSparse.ptr(), ACLSPARSELT_MAT_BATCH_STRIDE, &batchStride,
        sizeof(batchStride));
}

// -----------------------------------------------------------------------------
// 批量执行段（PruneCheckNpuBatch / PruneCheckNpuBatchBSparse 共用）：hPhys
// 上传 → SpMMAPruneCheck 单次调用（无 warm-up / 计时轮）→ 同步 → D2H。描述
// 符链由调用侧构造完毕后传入（A/B 侧同协议，收敛方式对齐 RunTimedPruneCheck）。
// -----------------------------------------------------------------------------
template <typename T>
inline void RunBatchCheckOnce(sparse_test::SparseLtHandleGuard& handle,
                              sparse_test::MatmulDescrGuard& md,
                              const std::vector<T>& hPhys,
                              aclrtStream stream,
                              PruneCheckBatchResult& result)
{
    using namespace sparse_test;
    constexpr size_t kElt = NpuDtypeTrait<T>::kEltSize;
    DeviceBuffer dIn = DeviceBuffer::copyFrom(hPhys.data(), hPhys.size() * kElt);
    DeviceBuffer dValid = DeviceBuffer::alloc(sizeof(int));
    result.checkRet = aclsparseLtSpMMAPruneCheck(
        handle.ptr(), matmulDescCptr(md), dIn.get(),
        static_cast<int*>(dValid.get()), stream);
    if (result.checkRet != ACL_SPARSE_STATUS_SUCCESS) { return; }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        result.checkRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return;
    }
    dValid.copyToHost(&result.hValid, sizeof(int));
}

template <typename T>
inline PruneCheckBatchResult PruneCheckNpuBatch(
    aclrtStream stream,
    const std::vector<T>& hPhys,
    int32_t m, int32_t k,
    aclsparseOperation_t op, aclsparseOrder_t order,
    int32_t numBatches, int64_t batchStride)
{
    using namespace sparse_test;
    PruneCheckBatchResult result;

    const bool trans = (op == ACL_SPARSE_OP_TRANSPOSE);
    const SparseSidePhysicalSpec spec = SparseSideSpecOf<T>(m, k, trans, order);

    SparseLtHandleGuard handle;
    MatDescrGuard matA;  // 稀疏侧（structured）
    result.initRet = InitSparseSideDescriptor<T>(handle.ptr(), spec, order, matA);
    if (result.initRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }
    // matB/matC/matD dense 占位三件套（形状派生口径与 PruneCheckNpu 一致，
    // 共用 helper）。
    MatDescrGuard matB;
    MatDescrGuard matC;
    MatDescrGuard matD;
    result.initRet = InitASideDensePlaceholders<T>(handle.ptr(), m, k, matB, matC, matD);
    if (result.initRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }
    // 批量属性：NUM_BATCHES 四侧一致 + BATCH_STRIDE 仅稀疏侧 matA（A 侧链，
    // 语义见 SetBatchAttributes 注释）。
    result.initRet = SetBatchAttributes(handle.ptr(), matA, matB, matC, matD,
                                        true, numBatches, batchStride);
    if (result.initRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }

    constexpr aclsparseComputeType_t kComputeType = PruneCheckComputeType<T>();
    MatmulDescrGuard md;
    result.initRet = md.init(handle.ptr(), op, ACL_SPARSE_OP_NON_TRANSPOSE,
                             matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), kComputeType);
    if (result.initRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }
    RunBatchCheckOnce<T>(handle, md, hPhys, stream, result);
    return result;
}

// -----------------------------------------------------------------------------
// B 稀疏侧批量执行链（公共 API 路径）：描述符链与 PruneCheckNpuBSparse 一致
//（matA/matC/matD dense 占位 + matB structured 稀疏），批量属性经
// SetBatchAttributes 装配（BATCH_STRIDE 落稀疏侧 matB —— host DeriveBatchParams
// 读 md->matB 的 numBatches/batchStride）。mB/kB 为 B 自身 (rows, cols) =
// (逻辑 k, 逻辑 n)；op/order 为 opB / matB order；hPhys 装配口径与 A 侧链
// 相同（batch b 基址 = b×batchStride 元素）。
// -----------------------------------------------------------------------------
template <typename T>
inline PruneCheckBatchResult PruneCheckNpuBatchBSparse(
    aclrtStream stream,
    const std::vector<T>& hPhys,
    int32_t mB, int32_t kB,
    aclsparseOperation_t opB, aclsparseOrder_t order,
    int32_t numBatches, int64_t batchStride)
{
    using namespace sparse_test;
    PruneCheckBatchResult result;

    const bool transB = (opB == ACL_SPARSE_OP_TRANSPOSE);
    const SparseSidePhysicalSpec spec = SparseSideSpecOf<T>(mB, kB, transB, order);

    SparseLtHandleGuard handle;
    // matA/matC/matD dense 占位三件套（形状派生口径与 PruneCheckNpuBSparse
    // 一致，共用 helper）。
    MatDescrGuard matA;
    MatDescrGuard matC;
    MatDescrGuard matD;
    result.initRet = InitBSideDensePlaceholders<T>(handle.ptr(), mB, kB, matA, matC, matD);
    if (result.initRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }
    MatDescrGuard matB;  // 稀疏侧（structured）
    result.initRet = InitSparseSideDescriptor<T>(handle.ptr(), spec, order, matB);
    if (result.initRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }
    // 批量属性：NUM_BATCHES 四侧一致 + BATCH_STRIDE 仅稀疏侧 matB（B 侧链，
    // 语义见 SetBatchAttributes 注释）。
    result.initRet = SetBatchAttributes(handle.ptr(), matA, matB, matC, matD,
                                        false, numBatches, batchStride);
    if (result.initRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }

    constexpr aclsparseComputeType_t kComputeType = PruneCheckComputeType<T>();
    MatmulDescrGuard md;
    result.initRet = md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE, opB,
                             matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), kComputeType);
    if (result.initRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }
    RunBatchCheckOnce<T>(handle, md, hPhys, stream, result);
    return result;
}

#endif  // TEST_PRUNE_CHECK_NPU_WRAPPER_H_
