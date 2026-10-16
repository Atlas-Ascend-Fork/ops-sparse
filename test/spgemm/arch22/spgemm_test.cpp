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
 * \file spgemm_test.cpp
 * \brief SpGEMM arch22 C++ UT：多阶段流程、四种 dtype、结构与数值对拍、异常入参。
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

#include "securec.h"
#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "spgemm_ref.h"
#include "spgemm_harness.h"
#include "gtest/gtest.h"
#include "../../frame/presmoke.h"

namespace {

#define SPGEMM_EXPECT(cond, fmt, ...)                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            ADD_FAILURE() << "[FAIL] " << #cond;                               \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define SPGEMM_ACL(x)                                                          \
    do {                                                                       \
        aclError _r = (x);                                                     \
        if (_r != ACL_SUCCESS) {                                               \
            std::printf("  [FAIL] acl error %d at %s:%d\n", _r, __FILE__,      \
                        __LINE__);                                             \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define SPGEMM_ST(x)                                                           \
    do {                                                                       \
        aclsparseStatus_t _r = (x);                                            \
        if (_r != ACL_SPARSE_STATUS_SUCCESS) {                                 \
            std::printf("  [FAIL] aclsparse status %d at %s:%d\n", _r,         \
                        __FILE__, __LINE__);                                   \
            return false;                                                      \
        }                                                                      \
    } while (0)

/** device 内存 RAII，避免异常路径漏 free。 */
class DevBuf {
public:
    DevBuf() = default;
    ~DevBuf() { Reset(); }
    DevBuf(const DevBuf &) = delete;
    DevBuf &operator=(const DevBuf &) = delete;

    bool Alloc(size_t bytes)
    {
        Reset();
        if (bytes == 0) {
            bytes = 64;  // 避免 0 字节申请
        }
        if (aclrtMalloc(&ptr_, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ptr_ = nullptr;
            return false;
        }
        size_ = bytes;
        return true;
    }
    bool Upload(const void *host, size_t bytes)
    {
        if (bytes == 0) {
            return true;
        }
        return aclrtMemcpy(ptr_, size_, host, bytes,
                           ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    }
    bool Download(void *host, size_t bytes) const
    {
        if (bytes == 0) {
            return true;
        }
        return aclrtMemcpy(host, bytes, ptr_, bytes,
                           ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
    }
    void Reset()
    {
        if (ptr_ != nullptr) {
            (void)aclrtFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }
    void *Get() const { return ptr_; }

private:
    void *ptr_ = nullptr;
    size_t size_ = 0;
};

/** 把 host 侧参考 CSR 上传到 device 并建立描述符。 */
struct DeviceCsr {
    DevBuf rowPtr;
    DevBuf colIdx;
    DevBuf values;
    aclsparseSpMatDescr_t descr = nullptr;

    ~DeviceCsr()
    {
        if (descr != nullptr) {
            (void)aclsparseDestroySpMat(descr);
        }
    }
};

bool UploadCsr(const SpgemmRefCsr &m, aclDataType dt, DeviceCsr &out)
{
    const int32_t nnz = m.Nnz();
    if (!out.rowPtr.Alloc((static_cast<size_t>(m.rows) + 1) * sizeof(int32_t))) {
        return false;
    }
    if (!out.rowPtr.Upload(m.rowPtr.data(),
                           (static_cast<size_t>(m.rows) + 1) * sizeof(int32_t))) {
        return false;
    }
    if (!out.colIdx.Alloc(static_cast<size_t>(nnz) * sizeof(int32_t))) {
        return false;
    }
    if (nnz > 0 && !out.colIdx.Upload(m.colIdx.data(),
                                      static_cast<size_t>(nnz) * sizeof(int32_t))) {
        return false;
    }
    const std::vector<uint8_t> packed = SpgemmPackValues(m.valRe, m.valIm, dt);
    if (!out.values.Alloc(packed.size())) {
        return false;
    }
    if (!packed.empty() && !out.values.Upload(packed.data(), packed.size())) {
        return false;
    }
    return aclsparseCreateCsr(&out.descr, m.rows, m.cols, nnz,
                              out.rowPtr.Get(), out.colIdx.Get(), out.values.Get(),
                              ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                              ACL_SPARSE_INDEX_BASE_ZERO, dt) ==
           ACL_SPARSE_STATUS_SUCCESS;
}

/** device 侧算出的 C，取回 host 后的形态。 */
struct HostResult {
    int64_t nnz = 0;
    std::vector<int32_t> rowPtr;
    std::vector<int32_t> colIdx;
    std::vector<double> valRe;
    std::vector<double> valIm;
};

/**
 * C 矩阵在 device 侧的持有者。
 *
 * `cIn != nullptr`（beta != 0）时持有完整的 C_in，descr 由内嵌的 DeviceCsr 负责销毁；
 * `cIn == nullptr` 时只持有一个预分配了 rowOffsets 的空壳，descr 由本 holder 负责销毁。
 * 两种情况下 matC 的生命周期都必须覆盖到 Copy 结束。
 */
struct SpgemmMatCHolder {
    DeviceCsr full;
    DevBuf rowOnly;
    aclsparseSpMatDescr_t descr = nullptr;
    bool ownsDescr = false;

    ~SpgemmMatCHolder()
    {
        if (ownsDescr && descr != nullptr) {
            (void)aclsparseDestroySpMat(descr);
        }
    }
};

/**
 * 建立 matC：初始 nnz 由 beta 决定。
 * beta == 0 时以 nnz=0 + 空 colIdx/values 创建（cuSPARSE 语义）——仅 rowOffsets
 * 需要预分配（长度 M+1），colIndices/values 待 nnz(C) 确定后由 CsrSetPointers 回填。
 */
bool SetupMatC(const SpgemmRefCsr &a, const SpgemmRefCsr &b, const SpgemmRefCsr *cIn,
               aclDataType dt, SpgemmMatCHolder &h)
{
    if (cIn != nullptr) {
        SPGEMM_EXPECT(UploadCsr(*cIn, dt, h.full), "upload C_in failed");
        h.descr = h.full.descr;
        h.ownsDescr = false;
        return true;
    }
    SPGEMM_EXPECT(h.rowOnly.Alloc((static_cast<size_t>(a.rows) + 1) * sizeof(int32_t)),
                  "alloc C rowPtr failed");
    SPGEMM_ST(aclsparseCreateCsr(&h.descr, a.rows, b.cols, 0, h.rowOnly.Get(),
                                 nullptr, nullptr, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                 dt));
    h.ownsDescr = true;
    return true;
}

/**
 * RunSpgemmStages 各阶段共享的运行期状态。
 * workspace 生命周期必须覆盖到对应阶段完成。
 */
struct SpgemmStageRun {
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matA = nullptr;
    aclsparseSpMatDescr_t matB = nullptr;
    aclsparseSpMatDescr_t matC = nullptr;
    aclsparseSpGEMMDescr_t descr = nullptr;
    aclDataType dt = ACL_FLOAT;
    aclsparseSpGEMMAlg_t alg = ACL_SPARSE_SPGEMM_DEFAULT;
    const void *alpha = nullptr;
    const void *beta = nullptr;

    DevBuf buf1;
    DevBuf buf2;
    DevBuf buf3;
    DevBuf outRow;
    DevBuf outCol;
    DevBuf outVal;

    size_t bufSize2 = 0;      // EstimateMemory 回填的 buffer2 尺寸
    int64_t numProds = -1;
    int64_t rowsC = 0;
    int64_t nnzC = 0;
};

/** 阶段 1：WorkEstimation 查询 → 分配 buffer1 → 执行 → 查询中间乘积数。 */
bool RunStageWorkEstimation(SpgemmStageRun &r)
{
    size_t bufSize1 = 0;
    aclsparseStatus_t st = aclsparseSpGEMMWorkEstimation(
        r.handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        r.alpha, r.matA, r.matB, r.beta, r.matC, r.dt, r.alg, r.descr,
        &bufSize1, nullptr);
    if (st != ACL_SPARSE_STATUS_SUCCESS || bufSize1 == 0) {
        std::printf("  [FAIL] WorkEstimation query st=%d size=%zu\n", st, bufSize1);
        return false;
    }
    if (!r.buf1.Alloc(bufSize1)) {
        std::printf("  [FAIL] alloc buffer1 %zu failed\n", bufSize1);
        return false;
    }
    st = aclsparseSpGEMMWorkEstimation(
        r.handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        r.alpha, r.matA, r.matB, r.beta, r.matC, r.dt, r.alg, r.descr,
        &bufSize1, r.buf1.Get());
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::printf("  [FAIL] WorkEstimation exec st=%d\n", st);
        return false;
    }
    st = aclsparseSpGEMMGetNumProducts(r.descr, &r.numProds);
    if (st != ACL_SPARSE_STATUS_SUCCESS || r.numProds < 0) {
        std::printf("  [FAIL] GetNumProducts st=%d n=%ld\n", st, (long)r.numProds);
        return false;
    }
    return true;
}

/** 阶段 2：EstimateMemory，仅 ALG2/ALG3 有此阶段，其余算法直接跳过。 */
bool RunStageEstimateMemory(SpgemmStageRun &r)
{
    if (!(r.alg == ACL_SPARSE_SPGEMM_ALG2 || r.alg == ACL_SPARSE_SPGEMM_ALG3)) {
        return true;
    }
    size_t bufSize3 = 0;
    aclsparseStatus_t st = aclsparseSpGEMMEstimateMemory(
        r.handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        r.alpha, r.matA, r.matB, r.beta, r.matC, r.dt, r.alg, r.descr,
        0.5f, &bufSize3, nullptr, &r.bufSize2);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::printf("  [FAIL] EstimateMemory query st=%d\n", st);
        return false;
    }
    if (!r.buf3.Alloc(bufSize3)) {
        std::printf("  [FAIL] alloc buffer3 failed\n");
        return false;
    }
    st = aclsparseSpGEMMEstimateMemory(
        r.handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        r.alpha, r.matA, r.matB, r.beta, r.matC, r.dt, r.alg, r.descr,
        0.5f, &bufSize3, r.buf3.Get(), &r.bufSize2);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::printf("  [FAIL] EstimateMemory exec st=%d\n", st);
        return false;
    }
    return true;
}

/**
 * 阶段 3：Compute。
 * ALG2/ALG3 使用 EstimateMemory 回填的 bufferSize2；DEFAULT/ALG1 用 Compute 查询。
 */
bool RunStageCompute(SpgemmStageRun &r)
{
    size_t queried2 = r.bufSize2;
    aclsparseStatus_t st = ACL_SPARSE_STATUS_SUCCESS;
    if (!(r.alg == ACL_SPARSE_SPGEMM_ALG2 || r.alg == ACL_SPARSE_SPGEMM_ALG3)) {
        st = aclsparseSpGEMMCompute(
            r.handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            r.alpha, r.matA, r.matB, r.beta, r.matC, r.dt, r.alg, r.descr,
            &queried2, nullptr);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            std::printf("  [FAIL] Compute query st=%d\n", st);
            return false;
        }
    }
    if (!r.buf2.Alloc(queried2)) {
        std::printf("  [FAIL] alloc buffer2 %zu failed\n", queried2);
        return false;
    }
    st = aclsparseSpGEMMCompute(
        r.handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        r.alpha, r.matA, r.matB, r.beta, r.matC, r.dt, r.alg, r.descr,
        &queried2, r.buf2.Get());
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::printf("  [FAIL] Compute exec st=%d\n", st);
        return false;
    }
    // nnz(C) 查询：SpGEMM 的协议性同步点
    int64_t colsC = 0;
    st = aclsparseSpMatGetSize(r.matC, &r.rowsC, &colsC, &r.nnzC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::printf("  [FAIL] SpMatGetSize st=%d\n", st);
        return false;
    }
    return true;
}

/** 按 nnz(C) 分配输出存储，并把三段指针回填给 matC。 */
bool RunStageAllocateOutput(SpgemmStageRun &r)
{
    if (!r.outRow.Alloc((static_cast<size_t>(r.rowsC) + 1) * sizeof(int32_t)) ||
        !r.outCol.Alloc(static_cast<size_t>(r.nnzC) * sizeof(int32_t)) ||
        !r.outVal.Alloc(static_cast<size_t>(r.nnzC) * SpgemmDtypeBytes(r.dt))) {
        std::printf("  [FAIL] alloc output failed\n");
        return false;
    }
    const aclsparseStatus_t st = aclsparseCsrSetPointers(r.matC, r.outRow.Get(),
                                                         r.outCol.Get(), r.outVal.Get());
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::printf("  [FAIL] CsrSetPointers st=%d\n", st);
        return false;
    }
    return true;
}

/** 阶段 4：Copy，随后同步流（下载前必须确保 device 侧写完）。 */
bool RunStageCopy(SpgemmStageRun &r)
{
    const aclsparseStatus_t st = aclsparseSpGEMMCopy(
        r.handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        r.alpha, r.matA, r.matB, r.beta, r.matC, r.dt, r.alg, r.descr);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::printf("  [FAIL] Copy st=%d\n", st);
        return false;
    }
    aclrtStream stream = nullptr;
    (void)aclsparseGetStream(r.handle, &stream);
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::printf("  [FAIL] sync after Copy\n");
        return false;
    }
    return true;
}

/** 把 device 上的 C 取回 host，并按 dtype 解包数值（complex64 拆成 re/im 两路）。 */
bool RunStageDownload(const SpgemmStageRun &r, HostResult &out)
{
    out.nnz = r.nnzC;
    out.rowPtr.assign(static_cast<size_t>(r.rowsC) + 1, 0);
    if (!r.outRow.Download(out.rowPtr.data(),
                           (static_cast<size_t>(r.rowsC) + 1) * sizeof(int32_t))) {
        std::printf("  [FAIL] download rowPtr\n");
        return false;
    }
    out.colIdx.assign(static_cast<size_t>(r.nnzC), 0);
    if (r.nnzC > 0 &&
        !r.outCol.Download(out.colIdx.data(),
                           static_cast<size_t>(r.nnzC) * sizeof(int32_t))) {
        std::printf("  [FAIL] download colIdx\n");
        return false;
    }
    std::vector<uint8_t> raw(static_cast<size_t>(r.nnzC) * SpgemmDtypeBytes(r.dt));
    if (r.nnzC > 0 && !r.outVal.Download(raw.data(), raw.size())) {
        std::printf("  [FAIL] download values\n");
        return false;
    }
    SpgemmUnpackValues(raw, static_cast<size_t>(r.nnzC), r.dt, out.valRe, out.valIm);
    return true;
}

/**
 * 跑完整的 7 接口多阶段流程：
 *   CreateDescr → WorkEstimation(查询/执行) → GetNumProducts
 *   → [EstimateMemory(查询/执行)] → Compute(查询/执行)
 *   → SpMatGetSize → CsrSetPointers → Copy → DestroyDescr
 *
 * 各阶段按 ACL 协议顺序串行，任一阶段失败即短路、后续阶段不再执行。
 */
bool RunSpgemmStages(aclsparseHandle_t handle, const SpgemmRefCsr &a,
                     const SpgemmRefCsr &b, const SpgemmRefCsr *cIn,
                     aclDataType dt, aclsparseSpGEMMAlg_t alg,
                     const void *alpha, const void *beta,
                     HostResult &out, int64_t *numProdsOut)
{
    DeviceCsr dA;
    DeviceCsr dB;
    SPGEMM_EXPECT(UploadCsr(a, dt, dA), "upload A failed");
    SPGEMM_EXPECT(UploadCsr(b, dt, dB), "upload B failed");

    SpgemmMatCHolder cHold;
    SPGEMM_EXPECT(SetupMatC(a, b, cIn, dt, cHold), "setup matC failed");

    bool ok = true;
    {
        // 独立作用域：保证 workspace 先于描述符释放（与拆分前的析构顺序一致）。
        SpgemmStageRun r;
        r.handle = handle;
        r.matA = dA.descr;
        r.matB = dB.descr;
        r.matC = cHold.descr;
        r.dt = dt;
        r.alg = alg;
        r.alpha = alpha;
        r.beta = beta;
        SPGEMM_ST(aclsparseSpGEMMCreateDescr(&r.descr));

        ok = RunStageWorkEstimation(r) && RunStageEstimateMemory(r) &&
             RunStageCompute(r) && RunStageAllocateOutput(r) && RunStageCopy(r) &&
             RunStageDownload(r, out);
        if (numProdsOut != nullptr && r.numProds >= 0) {
            *numProdsOut = r.numProds;
        }
        (void)aclsparseSpGEMMDestroyDescr(r.descr);
    }
    return ok;
}

/**
 * 结构比对：nnz / rowOffsets / colIndices 逐元素精确相等。
 * 额外校验 rowOffsets 单调非降 + 行内列索引严格升序。
 */
bool CompareStructure(const HostResult &got, const SpgemmRefCsr &ref, const char *tag)
{
    SPGEMM_EXPECT(got.nnz == ref.Nnz(), "%s: nnz(C) got=%ld want=%d", tag,
                  (long)got.nnz, ref.Nnz());
    SPGEMM_EXPECT(got.rowPtr.size() == ref.rowPtr.size(),
                  "%s: rowPtr size got=%zu want=%zu", tag, got.rowPtr.size(),
                  ref.rowPtr.size());
    for (size_t i = 0; i < ref.rowPtr.size(); i++) {
        SPGEMM_EXPECT(got.rowPtr[i] == ref.rowPtr[i],
                      "%s: rowOffsets[%zu] got=%d want=%d", tag, i,
                      got.rowPtr[i], ref.rowPtr[i]);
    }
    for (int64_t i = 0; i < got.nnz; i++) {
        SPGEMM_EXPECT(got.colIdx[static_cast<size_t>(i)] ==
                          ref.colIdx[static_cast<size_t>(i)],
                      "%s: colIndices[%ld] got=%d want=%d", tag, (long)i,
                      got.colIdx[static_cast<size_t>(i)],
                      ref.colIdx[static_cast<size_t>(i)]);
    }
    for (size_t r = 0; r + 1 < got.rowPtr.size(); r++) {
        SPGEMM_EXPECT(got.rowPtr[r + 1] >= got.rowPtr[r],
                      "%s: rowOffsets not monotonic at %zu", tag, r);
        for (int32_t p = got.rowPtr[r]; p + 1 < got.rowPtr[r + 1]; p++) {
            SPGEMM_EXPECT(got.colIdx[static_cast<size_t>(p)] <
                              got.colIdx[static_cast<size_t>(p) + 1],
                          "%s: row %zu colIndices not strictly increasing at %d",
                          tag, r, p);
        }
    }
    return true;
}

/**
 * 数值判定的累加器：混合容差（匹配率 ≥ 0.99 + 逐元素硬上限）。
 */
struct CompareValueStats {
    const SpgemmTolerance &tol;
    int64_t total = 0;
    int64_t matched = 0;
    double worstAbs = 0.0;
    int64_t worstIdx = -1;

    /** 返回 false 表示触发了硬上限。 */
    bool CheckOne(double actual, double golden, int64_t idx)
    {
        total++;
        const double absErr = std::fabs(actual - golden);
        if (absErr > worstAbs) {
            worstAbs = absErr;
            worstIdx = idx;
        }
        if (absErr <= tol.atol + tol.rtol * std::fabs(golden)) {
            matched++;
        }
        const double ulp = std::fabs(golden) > 0.0
            ? std::nextafter(std::fabs(golden), 1.0e300) - std::fabs(golden)
            : 0.0;
        return absErr <= std::fmax(tol.hardA, 32.0 * ulp);
    }
};

/**
 * 数值比对：实部与虚部共用同一个统计器。
 */
bool CompareValues(const HostResult &got, const SpgemmRefCsr &ref, aclDataType dt,
                   const char *tag)
{
    const SpgemmTolerance tol = SpgemmToleranceFor(dt);
    CompareValueStats st{tol};
    const bool isComplex = (dt == ACL_COMPLEX64);

    for (int64_t i = 0; i < got.nnz; i++) {
        const size_t u = static_cast<size_t>(i);
        if (!std::isfinite(ref.valRe[u])) {
            SPGEMM_EXPECT(!std::isfinite(got.valRe[u]),
                          "%s: non-finite golden at %ld not propagated (got %g)",
                          tag, (long)i, got.valRe[u]);
        } else {
            SPGEMM_EXPECT(st.CheckOne(got.valRe[u], ref.valRe[u], i),
                          "%s: real hard-limit exceeded at %ld got=%.9g want=%.9g",
                          tag, (long)i, got.valRe[u], ref.valRe[u]);
        }
        if (isComplex) {
            if (!std::isfinite(ref.valIm[u])) {
                SPGEMM_EXPECT(!std::isfinite(got.valIm[u]),
                              "%s: non-finite imag golden at %ld not propagated",
                              tag, (long)i);
            } else {
                SPGEMM_EXPECT(st.CheckOne(got.valIm[u], ref.valIm[u], i),
                              "%s: imag hard-limit exceeded at %ld got=%.9g want=%.9g",
                              tag, (long)i, got.valIm[u], ref.valIm[u]);
            }
        }
    }

    if (st.total > 0) {
        const double rate = static_cast<double>(st.matched) /
                            static_cast<double>(st.total);
        SPGEMM_EXPECT(rate >= 0.99,
                      "%s: match rate %.4f < 0.99 (worst abs err %.6g at %ld)",
                      tag, rate, st.worstAbs, (long)st.worstIdx);
    }
    return true;
}

/**
 * 结构 + 数值比对。
 */
bool CompareResult(const HostResult &got, const SpgemmRefCsr &ref, aclDataType dt,
                   const char *tag)
{
    return CompareStructure(got, ref, tag) && CompareValues(got, ref, dt, tag);
}

}  // namespace

// ===========================================================================
// 测试用例
// ===========================================================================

namespace {

struct SpgemmTestScalars {
    float alphaF[2];
    float betaF[2];
    uint16_t alphaH;
    uint16_t betaH;
    const void *alpha;
    const void *beta;
};

inline void InitAlphaBetaScalars(SpgemmTestScalars &sc, aclDataType dt,
                                  float alphaRe = 1.0f)
{
    sc.alphaF[0] = alphaRe;
    sc.alphaF[1] = 0.0f;
    sc.betaF[0] = 0.0f;
    sc.betaF[1] = 0.0f;
    sc.alphaH = (dt == ACL_BF16) ? SpgemmF32ToBf16Bits(alphaRe)
                                  : SpgemmF32ToF16Bits(alphaRe);
    sc.betaH = 0;
    sc.alpha = (dt == ACL_FLOAT16 || dt == ACL_BF16)
        ? static_cast<const void *>(&sc.alphaH)
        : static_cast<const void *>(sc.alphaF);
    sc.beta = (dt == ACL_FLOAT16 || dt == ACL_BF16)
        ? static_cast<const void *>(&sc.betaH)
        : static_cast<const void *>(sc.betaF);
}

inline void BuildBetaTestMatrices(SpgemmRefCsr &a, SpgemmRefCsr &b, SpgemmRefCsr &cIn)
{
    a.rows = 2;  a.cols = 2;
    a.rowPtr = {0, 1, 2};
    a.colIdx = {0, 1};
    a.valRe = {2.0, 3.0};
    a.valIm = {0.0, 0.0};

    b.rows = 2;  b.cols = 4;
    b.rowPtr = {0, 1, 2};
    b.colIdx = {0, 0};
    b.valRe = {1.5, 2.5};
    b.valIm = {0.0, 0.0};

    cIn.rows = 2;  cIn.cols = 4;
    cIn.rowPtr = {0, 1, 2};
    cIn.colIdx = {3, 3};
    cIn.valRe = {10.0, 20.0};
    cIn.valIm = {0.0, 0.0};
}

inline bool SetupFloat8x8TestDeviceMatrices(
    DeviceCsr &dA, DeviceCsr &dB, DevBuf &cRow, aclsparseSpMatDescr_t &matC)
{
    const SpgemmRefCsr a = SpgemmRefBuildCsr(8, 8, 2, 1, 0, false, 3);
    const SpgemmRefCsr b = SpgemmRefBuildCsr(8, 8, 2, 2, 0, false, 4);
    SPGEMM_EXPECT(UploadCsr(a, ACL_FLOAT, dA), "upload A failed");
    SPGEMM_EXPECT(UploadCsr(b, ACL_FLOAT, dB), "upload B failed");
    SPGEMM_EXPECT(cRow.Alloc(9 * sizeof(int32_t)), "alloc failed");
    matC = nullptr;
    SPGEMM_ST(aclsparseCreateCsr(&matC, 8, 8, 0, cRow.Get(), nullptr, nullptr,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT));
    return true;
}

/** 基础功能：给定 shape/degree，四种 dtype 各跑一遍并与 CPU 参考对拍。 */
bool TestBasicCase(aclsparseHandle_t handle, int32_t M, int32_t K, int32_t N,
                   int32_t degA, int32_t degB, int32_t emptyA, int32_t emptyB,
                   aclDataType dt, aclsparseSpGEMMAlg_t alg, const char *name)
{
    const bool isComplex = (dt == ACL_COMPLEX64);
    SpgemmRefCsr a = SpgemmRefBuildCsr(M, K, degA, 1, emptyA, isComplex, 7);
    SpgemmRefCsr b = SpgemmRefBuildCsr(K, N, degB, degA > 0 ? degA : 1,
                                       emptyB, isComplex, 13);
    // 先把输入量化到被测 dtype，再据此算 golden——保证两侧输入严格同一组值。
    SpgemmQuantizeCsr(a, dt);
    SpgemmQuantizeCsr(b, dt);

    // alpha = 1, beta = 0（Python 路径的固定取值）
    SpgemmTestScalars sc;
    InitAlphaBetaScalars(sc, dt);

    HostResult got;
    int64_t numProds = -1;
    const std::string tag = std::string(name) + "/" + SpgemmDtypeName(dt);
    if (!RunSpgemmStages(handle, a, b, nullptr, dt, alg, sc.alpha, sc.beta, got, &numProds)) {
        return false;
    }

    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 1.0, 0.0, 0.0, 0.0, nullptr,
                                              isComplex);
    if (!CompareResult(got, ref, dt, tag.c_str())) {
        return false;
    }
    // 中间乘积数应等于 Σ_i Σ_{k∈A.cols(i)} nnz(B.row(k))
    int64_t expectProds = 0;
    for (int32_t i = 0; i < a.rows; i++) {
        for (int32_t p = a.rowPtr[i]; p < a.rowPtr[i + 1]; p++) {
            const int32_t k = a.colIdx[p];
            expectProds += b.rowPtr[k + 1] - b.rowPtr[k];
        }
    }
    SPGEMM_EXPECT(numProds == expectProds, "%s: numProds got=%ld want=%ld", tag,
                  (long)numProds, (long)expectProds);
    std::printf("  [ok] %-42s nnz(C)=%-8ld numProds=%ld\n", tag.c_str(), (long)got.nnz,
                (long)numProds);
    return true;
}

/**
 * 显式零必须保留：A = [1, 1], B = [[1], [-1]]，C[0][0] = 0 但 nnz(C) 必须为 1。
 */
bool TestExplicitZero(aclsparseHandle_t handle, aclDataType dt)
{
    const bool isComplex = (dt == ACL_COMPLEX64);
    SpgemmRefCsr a;
    a.rows = 1;
    a.cols = 2;
    a.rowPtr = {0, 2};
    a.colIdx = {0, 1};
    a.valRe = {1.0, 1.0};
    a.valIm = {0.0, 0.0};

    SpgemmRefCsr b;
    b.rows = 2;
    b.cols = 1;
    b.rowPtr = {0, 1, 2};
    b.colIdx = {0, 0};
    b.valRe = {1.0, -1.0};
    b.valIm = {0.0, 0.0};

    SpgemmTestScalars sc;
    InitAlphaBetaScalars(sc, dt);

    HostResult got;
    const std::string tag = std::string("explicit-zero/") + SpgemmDtypeName(dt);
    if (!RunSpgemmStages(handle, a, b, nullptr, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                         sc.alpha, sc.beta, got, nullptr)) {
        return false;
    }
    SPGEMM_EXPECT(got.nnz == 1, "%s: explicit zero dropped, nnz=%ld", tag,
                  (long)got.nnz);
    SPGEMM_EXPECT(got.colIdx[0] == 0, "%s: col got=%d want=0", tag, got.colIdx[0]);
    SPGEMM_EXPECT(got.valRe[0] == 0.0, "%s: value got=%g want=0", tag, got.valRe[0]);
    std::printf("  [ok] %-42s nnz=1 value=0 (retained)\n", tag.c_str());
    return true;
}

/** 无交集乘积：A 选中 B 的空行 → nnz(C) 必须为 0。 */
bool TestNoIntersection(aclsparseHandle_t handle, aclDataType dt)
{
    SpgemmRefCsr a;
    a.rows = 1;
    a.cols = 2;
    a.rowPtr = {0, 1};
    a.colIdx = {0};
    a.valRe = {1.0};
    a.valIm = {0.0};

    SpgemmRefCsr b;  // 第 0 行为空，第 1 行有一个元素
    b.rows = 2;
    b.cols = 1;
    b.rowPtr = {0, 0, 1};
    b.colIdx = {0};
    b.valRe = {1.0};
    b.valIm = {0.0};

    float alphaF[2] = {1.0f, 0.0f};
    float betaF[2] = {0.0f, 0.0f};
    HostResult got;
    const std::string tag = std::string("no-intersection/") + SpgemmDtypeName(dt);
    if (!RunSpgemmStages(handle, a, b, nullptr, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                         alphaF, betaF, got, nullptr)) {
        return false;
    }
    SPGEMM_EXPECT(got.nnz == 0, "%s: expect nnz(C)=0 got=%ld", tag, (long)got.nnz);
    SPGEMM_EXPECT(got.rowPtr.size() == 2 && got.rowPtr[0] == 0 && got.rowPtr[1] == 0,
                  "%s: rowOffsets not all-zero", tag);
    std::printf("  [ok] %-42s nnz(C)=0\n", tag.c_str());
    return true;
}

/**
 * 等步长轮转快路径的负向用例：非等差但首/次/末三点吻合的段。
 * 用精心构造的「三点吻合但非等差」段验证逐元素校验正确性。
 */
bool TestRotationalNonArithmetic(aclsparseHandle_t handle, aclDataType dt)
{
    const bool isComplex = (dt == ACL_COMPLEX64);
    SpgemmRefCsr a;
    a.rows = 1;
    a.cols = 4;
    a.rowPtr = {0, 2};      // da=2 → 落 DA_FIXED=2 特化，快路径生效
    a.colIdx = {0, 1};
    a.valRe = {1.0, 1.0};
    a.valIm = {0.0, 0.0};

    SpgemmRefCsr b;
    b.rows = 4;
    b.cols = 16;
    // 行 0：非等差（5 打破 stride=3），行 1：真等差；两段首 0/1 同在一个 stride 窗口内
    b.rowPtr = {0, 4, 8, 8, 8};
    b.colIdx = {0, 3, 5, 9, 1, 4, 7, 10};
    b.valRe = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    b.valIm.assign(8, 0.0);

    SpgemmTestScalars sc;
    InitAlphaBetaScalars(sc, dt);

    HostResult got;
    const std::string tag = std::string("rotational-non-arithmetic/") +
                                SpgemmDtypeName(dt);
    if (!RunSpgemmStages(handle, a, b, nullptr, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                         sc.alpha, sc.beta, got, nullptr)) {
        return false;
    }
    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 1.0, 0.0, 0.0, 0.0, nullptr,
                                              isComplex);
    if (!CompareResult(got, ref, dt, tag.c_str())) {
        return false;
    }
    std::printf("  [ok] %-42s cols=%d (arith check is per-element)\n", tag.c_str(),
                (int)got.nnz);
    return true;
}

/**
 * 轮转快路径的段首相等专项：B 中不同行拥有相同列模式时，
 * 归并去重后的输出应少于轮转公式给出的数量，且行内保持严格升序。
 */
bool TestRotationalEqualHeads(aclsparseHandle_t handle, aclDataType dt,
                              const char *name, const SpgemmRefCsr &a,
                              const SpgemmRefCsr &b)
{
    const bool isComplex = (dt == ACL_COMPLEX64);
    SpgemmTestScalars sc;
    InitAlphaBetaScalars(sc, dt);

    HostResult got;
    const std::string tag = std::string("rotational-equal-heads/") + name +
                              "/" + SpgemmDtypeName(dt);
    if (!RunSpgemmStages(handle, a, b, nullptr, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                         sc.alpha, sc.beta, got, nullptr)) {
        return false;
    }
    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 1.0, 0.0, 0.0, 0.0, nullptr,
                                              isComplex);
    if (!CompareResult(got, ref, dt, tag.c_str())) {
        return false;
    }
    // 行内严格升序：段首相等缺陷会产生重复列。
    for (size_t r = 0; r + 1 < got.rowPtr.size(); r++) {
        for (int64_t p = got.rowPtr[r] + 1; p < got.rowPtr[r + 1]; p++) {
            SPGEMM_EXPECT(got.colIdx[static_cast<size_t>(p)] >
                              got.colIdx[static_cast<size_t>(p) - 1],
                          "%s: row %zu colIndices not strictly ascending at %ld",
                          tag, r, (long)p);
        }
    }
    std::printf("  [ok] %-58s nnz=%d\n", tag.c_str(), (int)got.nnz);
    return true;
}

/** 构造 CSR：每行给定列表，值全 1（实部）。 */
SpgemmRefCsr SpgemmMakeCsrOnes(int32_t cols,
                               const std::vector<std::vector<int32_t>> &rows)
{
    SpgemmRefCsr m;
    m.rows = static_cast<int32_t>(rows.size());
    m.cols = cols;
    m.rowPtr.push_back(0);
    for (const auto &r : rows) {
        for (int32_t c : r) {
            m.colIdx.push_back(c);
            m.valRe.push_back(1.0);
            m.valIm.push_back(0.0);
        }
        m.rowPtr.push_back(static_cast<int32_t>(m.colIdx.size()));
    }
    return m;
}

/** A 的三行分别覆盖 da=1、da=5 和 da=8 非等差段首。 */
SpgemmRefCsr MakeBandLazyValuesA(aclDataType dt)
{
    SpgemmRefCsr a;
    a.rows = 3;
    a.cols = 10;
    a.rowPtr = {0, 1, 6, 14};
    a.colIdx = {0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 8};
    a.valRe = {0.5, -0.75, 1.25, 0.625, -1.5, 0.875,
               0.25, -0.5, 1.0, 1.5, -0.25, 0.75, -1.25, 0.375};
    a.valIm = {0.125, -0.25, 0.375, -0.125, 0.5, -0.375,
               0.25, 0.125, -0.5, 0.375, -0.25, 0.5, -0.125, 0.25};
    SpgemmQuantizeCsr(a, dt);
    return a;
}

/** B 的各行采用一致的 Band 模式。 */
SpgemmRefCsr MakeBandLazyValuesB(aclDataType dt)
{
    SpgemmRefCsr b;
    b.rows = 10;
    b.cols = 32;
    b.rowPtr.push_back(0);
    for (int32_t k = 0; k < b.rows; ++k) {
        for (int32_t j = 0; j < 3; ++j) {
            b.colIdx.push_back(k + 10 * j);
            b.valRe.push_back(0.25 * (k + 1) + 0.5 * j);
            b.valIm.push_back(0.125 * (j + 1) - 0.0625 * k);
        }
        b.rowPtr.push_back(static_cast<int32_t>(b.colIdx.size()));
    }
    SpgemmQuantizeCsr(b, dt);
    return b;
}

/** beta 非零时参与累加的 C 输入。 */
SpgemmRefCsr MakeBandLazyValuesCInput(aclDataType dt)
{
    SpgemmRefCsr cIn;
    cIn.rows = 3;
    cIn.cols = 32;
    cIn.rowPtr = {0, 2, 4, 6};
    cIn.colIdx = {1, 31, 2, 30, 3, 29};
    cIn.valRe = {0.5, -0.25, 0.75, -0.5, 0.125, 1.0};
    cIn.valIm = {-0.125, 0.25, -0.375, 0.5, 0.25, -0.5};

    SpgemmQuantizeCsr(cIn, dt);
    return cIn;
}

/**
 * Band 值加载回归：同一 B 触发 uniform Band，A 分别覆盖 da=1、da=5 和
 * complex64 da=8 段首非等差。后二者不能共享 C64 vector 发射资格，必须在
 * 标量/General 回退前按需装入 A values。
 */
bool TestBandLazyValues(aclsparseHandle_t handle, aclDataType dt, bool betaNonZero)
{
    const SpgemmRefCsr a = MakeBandLazyValuesA(dt);
    const SpgemmRefCsr b = MakeBandLazyValuesB(dt);
    const SpgemmRefCsr cIn = MakeBandLazyValuesCInput(dt);
    SpgemmTestScalars sc;
    InitAlphaBetaScalars(sc, dt);
    if (betaNonZero) {
        sc.betaF[0] = 1.0f;
        sc.betaH = (dt == ACL_BF16) ? SpgemmF32ToBf16Bits(1.0f) : SpgemmF32ToF16Bits(1.0f);
    }

    HostResult got;
    const std::string tag = std::string("band-lazy-values/") + SpgemmDtypeName(dt) +
                            (betaNonZero ? "/beta1" : "/beta0");
    if (!RunSpgemmStages(handle, a, b, betaNonZero ? &cIn : nullptr, dt,
                         ACL_SPARSE_SPGEMM_DEFAULT, sc.alpha, sc.beta, got, nullptr)) {
        return false;
    }
    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 1.0, 0.0,
                                               betaNonZero ? 1.0 : 0.0, 0.0,
                                               betaNonZero ? &cIn : nullptr, dt == ACL_COMPLEX64);
    if (!CompareResult(got, ref, dt, tag.c_str())) {
        return false;
    }
    std::printf("  [ok] %-42s da={1,5,8-nonAP} nnz=%ld\n", tag.c_str(),
                static_cast<long>(got.nnz));
    return true;
}

/** A 命中空 B 行时，Band 不可用也必须跳过该段，四 dtype 均与参考一致。 */
bool TestBandFallbackEmptyBRow(aclsparseHandle_t handle, aclDataType dt)
{
    const bool isComplex = (dt == ACL_COMPLEX64);
    SpgemmRefCsr a;
    a.rows = 1;
    a.cols = 3;
    a.rowPtr = {0, 3};
    a.colIdx = {0, 1, 2};
    a.valRe = {0.5, -1.0, 0.75};
    a.valIm = {0.25, -0.125, 0.5};
    SpgemmRefCsr b;
    b.rows = 3;
    b.cols = 8;
    b.rowPtr = {0, 2, 2, 4};
    b.colIdx = {0, 4, 2, 6};
    b.valRe = {1.0, 0.5, -0.25, 0.75};
    b.valIm = {0.25, -0.5, 0.125, 0.375};
    SpgemmQuantizeCsr(a, dt);
    SpgemmQuantizeCsr(b, dt);
    SpgemmTestScalars sc;
    InitAlphaBetaScalars(sc, dt);
    HostResult got;
    const std::string tag = std::string("band-fallback-empty-b/") + SpgemmDtypeName(dt);
    if (!RunSpgemmStages(handle, a, b, nullptr, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                         sc.alpha, sc.beta, got, nullptr)) {
        return false;
    }
    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 1.0, 0.0, 0.0, 0.0, nullptr,
                                               isComplex);
    return CompareResult(got, ref, dt, tag.c_str());
}

/**
 * complex64 向量化发射专项：用确定性伪随机复数值构造轮转行，
 * 验证复乘的交叉项（aIm·bIm、aIm·bRe）在非零虚部输入下正确。
 */
bool TestComplexVecEmit(aclsparseHandle_t handle, int32_t n, int32_t d,
                        int32_t shift)
{
    // 确定性 LCG：避免依赖 <random> 的实现差异，保证跨平台可复现。
    uint32_t st = 0x20260821u ^ (static_cast<uint32_t>(n) << 8) ^
                  static_cast<uint32_t>(d);
    auto nextVal = [&st]() -> double {
        st = st * 1664525u + 1013904223u;
        // 映射到 [−2, 2)，避开 0 以免交叉项被偶然抵消
        return (static_cast<double>(st >> 8) / 8388608.0) - 2.0;
    };

    auto mk = [&](int32_t rows, int32_t cols, int32_t sh, int32_t step) {
        if (cols == 0) return SpgemmRefCsr{};
        SpgemmRefCsr m;
        m.rows = rows;
        m.cols = cols;
        m.rowPtr.push_back(0);
        for (int32_t i = 0; i < rows; i++) {
            std::vector<int32_t> cs;
            for (int32_t j = 0; j < d; j++) {
                cs.push_back((i + sh + j * step) % cols);
            }
            std::sort(cs.begin(), cs.end());
            cs.erase(std::unique(cs.begin(), cs.end()), cs.end());
            for (int32_t c : cs) {
                m.colIdx.push_back(c);
                m.valRe.push_back(nextVal());
                m.valIm.push_back(nextVal());
            }
            m.rowPtr.push_back(static_cast<int32_t>(m.colIdx.size()));
        }
        return m;
    };

    const SpgemmRefCsr a = mk(n, n, shift, 1);
    const SpgemmRefCsr b = mk(n, n, 0, d);

    float alphaF[2] = {1.0f, 0.0f};
    float betaF[2] = {0.0f, 0.0f};
    HostResult got;
    const std::string tag = std::string("complex-vec-emit/n=") + std::to_string(n) +
                            ",d=" + std::to_string(d) +
                            ",shift=" + std::to_string(shift);
    if (!RunSpgemmStages(handle, a, b, nullptr, ACL_COMPLEX64,
                         ACL_SPARSE_SPGEMM_DEFAULT, alphaF, betaF, got,
                         nullptr)) {
        return false;
    }
    const SpgemmRefCsr ref =
        SpgemmRefCompute(a, b, 1.0, 0.0, 0.0, 0.0, nullptr, true);
    if (!CompareResult(got, ref, ACL_COMPLEX64, tag.c_str())) {
        return false;
    }
    std::printf("  [ok] %-58s nnz=%d\n", tag.c_str(), (int)got.nnz);
    return true;
}

/**
 * alpha-scaling 专项的 A：8 行 × 每行 3 段（段长 3）。
 */
SpgemmRefCsr BuildAlphaScalingA(bool isComplex)
{
    SpgemmRefCsr a;
    a.rows = 8;
    a.cols = 8;
    a.rowPtr.resize(9);
    for (int i = 0; i <= 8; i++) {
        a.rowPtr[i] = i * 3;
    }
    for (int i = 0; i < 8; i++) {
        std::vector<std::pair<int32_t, std::pair<double, double>>> row;
        for (int j = 0; j < 3; j++) {
            const int32_t c = (i + j) % 8;
            row.push_back({c, {1.0 + 0.25 * j, isComplex ? 0.5 * j : 0.0}});
        }
        std::sort(row.begin(), row.end());
        for (const auto &e : row) {
            a.colIdx.push_back(e.first);
            a.valRe.push_back(e.second.first);
            a.valIm.push_back(e.second.second);
        }
    }
    return a;
}

/** alpha-scaling 专项的 B：段内等差列。 */
SpgemmRefCsr BuildAlphaScalingB(bool isComplex)
{
    SpgemmRefCsr b;
    b.rows = 8;
    b.cols = 32;
    b.rowPtr.resize(9);
    for (int i = 0; i <= 8; i++) {
        b.rowPtr[i] = i * 3;
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 3; j++) {
            b.colIdx.push_back(i + j * 8);        // 段内等差，触发轮转快路径
            b.valRe.push_back(2.0 - 0.5 * j);
            b.valIm.push_back(isComplex ? 0.25 * j : 0.0);
        }
    }
    return b;
}

/** alpha != 1 的融合路径专项。alpha = -2.5 验证缩放正确生效。 */
bool TestAlphaScalingFusedPath(aclsparseHandle_t handle, aclDataType dt)
{
    const bool isComplex = (dt == ACL_COMPLEX64);
    SpgemmRefCsr a = BuildAlphaScalingA(isComplex);
    SpgemmRefCsr b = BuildAlphaScalingB(isComplex);
    SpgemmQuantizeCsr(a, dt);
    SpgemmQuantizeCsr(b, dt);

    const double alphaRe = -2.5;
    SpgemmTestScalars sc;
    InitAlphaBetaScalars(sc, dt, static_cast<float>(alphaRe));

    HostResult got;
    const std::string tag = std::string("alpha-scaling-fused/") + SpgemmDtypeName(dt);
    if (!RunSpgemmStages(handle, a, b, nullptr, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                         sc.alpha, sc.beta, got, nullptr)) {
        return false;
    }
    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, alphaRe, 0.0, 0.0, 0.0,
                                              nullptr, isComplex);
    if (!CompareResult(got, ref, dt, tag.c_str())) {
        return false;
    }
    std::printf("  [ok] %-42s nnz=%d alpha=%.2f\n", tag.c_str(), (int)got.nnz, alphaRe);
    return true;
}

/** beta != 0：C 的结构必须是 A*B 结构与 C 原结构的并集。 */
bool TestBetaStructuralUnion(aclsparseHandle_t handle, aclDataType dt)
{
    const bool isComplex = (dt == ACL_COMPLEX64);
    SpgemmRefCsr a, b, cIn;
    BuildBetaTestMatrices(a, b, cIn);

    float alphaF[2] = {1.0f, 0.0f};
    float betaF[2] = {1.0f, 0.0f};
    HostResult got;
    const std::string tag = std::string("beta-union/") + SpgemmDtypeName(dt);
    if (!RunSpgemmStages(handle, a, b, &cIn, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                         alphaF, betaF, got, nullptr)) {
        return false;
    }
    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 1.0, 0.0, 1.0, 0.0, &cIn,
                                              isComplex);
    SPGEMM_EXPECT(ref.Nnz() == 4, "ref nnz should be 4, got %d", ref.Nnz());
    if (!CompareResult(got, ref, dt, tag.c_str())) {
        return false;
    }
    std::printf("  [ok] %-42s nnz(C)=%ld (union of A*B and C_in)\n", tag.c_str(),
                (long)got.nnz);
    return true;
}

/** K=0 且 beta != 0 时，输出必须保留并缩放 C_in。 */
bool TestKZeroWithNonzeroBetaPreservesCIn(aclsparseHandle_t handle)
{
    SpgemmRefCsr a, b, cIn;
    BuildBetaTestMatrices(a, b, cIn);
    a.cols = 0;
    a.rowPtr = {0, 0, 0};
    a.colIdx.clear();
    a.valRe.clear();
    a.valIm.clear();
    b.rows = 0;
    b.rowPtr = {0};
    b.colIdx.clear();
    b.valRe.clear();
    b.valIm.clear();

    float alpha = 1.0f;
    float beta = 2.0f;
    HostResult got;
    if (!RunSpgemmStages(handle, a, b, &cIn, ACL_FLOAT, ACL_SPARSE_SPGEMM_DEFAULT,
                         &alpha, &beta, got, nullptr)) {
        return false;
    }
    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 1.0, 0.0, 2.0, 0.0, &cIn, false);
    return CompareResult(got, ref, ACL_FLOAT, "k-zero-beta-preserves-c-in");
}

/** 长尾行分布：少数行 P_i 极大，触发列分块兜底路径。 */
bool TestLongTail(aclsparseHandle_t handle, aclDataType dt)
{
    const bool isComplex = (dt == ACL_COMPLEX64);
    const int32_t M = 24;
    const int32_t K = 96;
    const int32_t N = 128;

    // 第 0 行取 B 的全部 96 行（da=96 → 走 T3），其余行 da=2
    SpgemmRefCsr a;
    a.rows = M;
    a.cols = K;
    a.rowPtr.assign(static_cast<size_t>(M) + 1, 0);
    uint32_t rng = 991;
    auto nextVal = [&rng]() -> double {
        rng = rng * 1664525U + 1013904223U;
        return static_cast<double>(static_cast<int32_t>(rng >> 8) % 200 - 100) / 100.0;
    };
    for (int32_t i = 0; i < M; i++) {
        const int32_t da = (i == 0) ? K : 2;
        for (int32_t j = 0; j < da; j++) {
            a.colIdx.push_back((i == 0) ? j : (i * 2 + j) % K);
            a.valRe.push_back(nextVal());
            a.valIm.push_back(isComplex ? nextVal() * 0.25 : 0.0);
        }
        // 保证行内严格升序
        std::sort(a.colIdx.begin() + a.rowPtr[i], a.colIdx.end());
        a.rowPtr[static_cast<size_t>(i) + 1] = static_cast<int32_t>(a.colIdx.size());
    }
    SpgemmRefCsr b = SpgemmRefBuildCsr(K, N, 6, 3, 0, isComplex, 31);
    SpgemmQuantizeCsr(a, dt);
    SpgemmQuantizeCsr(b, dt);

    SpgemmTestScalars sc;
    InitAlphaBetaScalars(sc, dt);

    HostResult got;
    const std::string tag = std::string("long-tail/") + SpgemmDtypeName(dt);
    if (!RunSpgemmStages(handle, a, b, nullptr, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                         sc.alpha, sc.beta, got, nullptr)) {
        return false;
    }
    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 1.0, 0.0, 0.0, 0.0, nullptr,
                                              isComplex);
    if (!CompareResult(got, ref, dt, tag.c_str())) {
        return false;
    }
    std::printf("  [ok] %-42s nnz(C)=%ld (T3 chunk path)\n", tag.c_str(), (long)got.nnz);
    return true;
}

/** 确定性：同一输入连续跑两遍，结构与 values 必须 bit-wise 一致。 */
bool TestDeterminism(aclsparseHandle_t handle, aclDataType dt)
{
    const bool isComplex = (dt == ACL_COMPLEX64);
    SpgemmRefCsr a = SpgemmRefBuildCsr(64, 64, 6, 1, 0, isComplex, 5);
    SpgemmRefCsr b = SpgemmRefBuildCsr(64, 64, 6, 6, 0, isComplex, 9);
    SpgemmQuantizeCsr(a, dt);
    SpgemmQuantizeCsr(b, dt);
    SpgemmTestScalars sc;
    InitAlphaBetaScalars(sc, dt);

    HostResult r1;
    HostResult r2;
    if (!RunSpgemmStages(handle, a, b, nullptr, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                         sc.alpha, sc.beta, r1, nullptr)) {
        return false;
    }
    if (!RunSpgemmStages(handle, a, b, nullptr, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                         sc.alpha, sc.beta, r2, nullptr)) {
        return false;
    }
    const std::string tag = std::string("determinism/") + SpgemmDtypeName(dt);
    SPGEMM_EXPECT(r1.nnz == r2.nnz, "%s: nnz differs %ld vs %ld", tag,
                  (long)r1.nnz, (long)r2.nnz);
    for (size_t i = 0; i < r1.rowPtr.size(); i++) {
        SPGEMM_EXPECT(r1.rowPtr[i] == r2.rowPtr[i], "%s: rowPtr[%zu] differs", tag, i);
    }
    for (int64_t i = 0; i < r1.nnz; i++) {
        const size_t u = static_cast<size_t>(i);
        SPGEMM_EXPECT(r1.colIdx[u] == r2.colIdx[u], "%s: colIdx[%ld] differs", tag,
                      (long)i);
        // bit-wise：直接比 double 相等（两次都是同一条 fp32 计算链的结果）
        SPGEMM_EXPECT(r1.valRe[u] == r2.valRe[u],
                      "%s: values[%ld] not bit-wise equal (%.9g vs %.9g)", tag,
                      (long)i, r1.valRe[u], r2.valRe[u]);
        if (isComplex) {
            SPGEMM_EXPECT(r1.valIm[u] == r2.valIm[u],
                          "%s: imag[%ld] not bit-wise equal", tag, (long)i);
        }
    }
    std::printf("  [ok] %-42s bit-wise reproducible (nnz=%ld)\n", tag.c_str(), (long)r1.nnz);
    return true;
}

// ---------------------------------------------------------------------------
// 异常入参与阶段状态机
// ---------------------------------------------------------------------------

/**
 * 非法入参专项的调用上下文：固定 handle / descr / alpha / beta，
 * 只变 opA、opB、computeType 与描述符形态。
 */
struct InvalidArgCtx {
    aclsparseHandle_t handle = nullptr;
    aclsparseSpGEMMDescr_t descr = nullptr;
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t bs = 0;

    aclsparseStatus_t CallWE(aclsparseOperation_t opA, aclsparseOperation_t opB,
                             aclsparseConstSpMatDescr_t mA,
                             aclsparseConstSpMatDescr_t mB, aclsparseSpMatDescr_t mC,
                             aclDataType ct, aclsparseSpGEMMAlg_t alg)
    {
        return aclsparseSpGEMMWorkEstimation(handle, opA, opB, &alpha, mA, mB,
                                             &beta, mC, ct, alg, descr, &bs, nullptr);
    }
};

/** 一条判定：用例名 + 实得状态码 + 期望状态码。 */
struct InvalidArgCheck {
    const char *name;
    aclsparseStatus_t got;
    aclsparseStatus_t want;
};

/** 转置类 opA/opB、非四选一 computeType、空描述符：与描述符形态无关的固定项。 */
void CollectOpAndTypeChecks(InvalidArgCtx &c, const DeviceCsr &dA,
                            const DeviceCsr &dB, aclsparseSpMatDescr_t matC,
                            std::vector<InvalidArgCheck> &checks)
{
    checks.push_back({"opA=TRANSPOSE",
        c.CallWE(ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
                 dA.descr, dB.descr, matC, ACL_FLOAT, ACL_SPARSE_SPGEMM_DEFAULT),
        ACL_SPARSE_STATUS_NOT_SUPPORTED});
    checks.push_back({"opA=CONJUGATE_TRANSPOSE",
        c.CallWE(ACL_SPARSE_OP_CONJUGATE_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
                 dA.descr, dB.descr, matC, ACL_FLOAT, ACL_SPARSE_SPGEMM_DEFAULT),
        ACL_SPARSE_STATUS_NOT_SUPPORTED});
    checks.push_back({"opB=TRANSPOSE",
        c.CallWE(ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
                 dA.descr, dB.descr, matC, ACL_FLOAT, ACL_SPARSE_SPGEMM_DEFAULT),
        ACL_SPARSE_STATUS_NOT_SUPPORTED});
    checks.push_back({"opB=CONJUGATE_TRANSPOSE",
        c.CallWE(ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_CONJUGATE_TRANSPOSE,
                 dA.descr, dB.descr, matC, ACL_FLOAT, ACL_SPARSE_SPGEMM_DEFAULT),
        ACL_SPARSE_STATUS_NOT_SUPPORTED});
    checks.push_back({"computeType=fp64",
        c.CallWE(ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
                 dA.descr, dB.descr, matC, ACL_DOUBLE, ACL_SPARSE_SPGEMM_DEFAULT),
        ACL_SPARSE_STATUS_NOT_SUPPORTED});
    checks.push_back({"matA=nullptr",
        c.CallWE(ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
                 nullptr, dB.descr, matC, ACL_FLOAT, ACL_SPARSE_SPGEMM_DEFAULT),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR});
}

/**
 * 描述符形态类非法项：int64 索引 / ptrType 与 idxType 不一致 / one-based 索引。
 */
void CollectIndexFormChecks(InvalidArgCtx &c, const DeviceCsr &dA,
                            const SpgemmRefCsr &a, aclsparseSpMatDescr_t matB,
                            aclsparseSpMatDescr_t matC,
                            std::vector<InvalidArgCheck> &checks,
                            std::vector<aclsparseSpMatDescr_t> &extra)
{
    aclsparseSpMatDescr_t m = nullptr;
    if (aclsparseCreateCsr(&m, 8, 8, a.Nnz(), dA.rowPtr.Get(), dA.colIdx.Get(),
                           dA.values.Get(), ACL_SPARSE_INDEX_64I,
                           ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_BASE_ZERO,
                           ACL_FLOAT) == ACL_SPARSE_STATUS_SUCCESS) {
        extra.push_back(m);
        checks.push_back({"index=int64",
            c.CallWE(ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
                     m, matB, matC, ACL_FLOAT, ACL_SPARSE_SPGEMM_DEFAULT),
            ACL_SPARSE_STATUS_NOT_SUPPORTED});
    }
    if (aclsparseCreateCsr(&m, 8, 8, a.Nnz(), dA.rowPtr.Get(), dA.colIdx.Get(),
                           dA.values.Get(), ACL_SPARSE_INDEX_64I,
                           ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                           ACL_FLOAT) == ACL_SPARSE_STATUS_SUCCESS) {
        extra.push_back(m);
        checks.push_back({"index=mixed(I64,I32)",
            c.CallWE(ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
                     m, matB, matC, ACL_FLOAT, ACL_SPARSE_SPGEMM_DEFAULT),
            ACL_SPARSE_STATUS_NOT_SUPPORTED});
    }
    if (aclsparseCreateCsr(&m, 8, 8, a.Nnz(), dA.rowPtr.Get(), dA.colIdx.Get(),
                           dA.values.Get(), ACL_SPARSE_INDEX_32I,
                           ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ONE,
                           ACL_FLOAT) == ACL_SPARSE_STATUS_SUCCESS) {
        extra.push_back(m);
        checks.push_back({"index base=one",
            c.CallWE(ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
                     m, matB, matC, ACL_FLOAT, ACL_SPARSE_SPGEMM_DEFAULT),
            ACL_SPARSE_STATUS_NOT_SUPPORTED});
    }
}

/** 维度不匹配：A 是 8×8，这里造一个 4×4 的 B ⇒ A.cols != B.rows。 */
void CollectDimMismatchCheck(InvalidArgCtx &c, const DeviceCsr &dA,
                             aclsparseSpMatDescr_t matC,
                             std::vector<InvalidArgCheck> &checks)
{
    const SpgemmRefCsr bSmall = SpgemmRefBuildCsr(4, 4, 2, 1, 0, false, 5);
    DeviceCsr dBSmall;
    if (UploadCsr(bSmall, ACL_FLOAT, dBSmall)) {
        checks.push_back({"dim mismatch A.cols != B.rows",
            c.CallWE(ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
                     dA.descr, dBSmall.descr, matC, ACL_FLOAT,
                     ACL_SPARSE_SPGEMM_DEFAULT),
            ACL_SPARSE_STATUS_INVALID_VALUE});
    }
}

/** 逐条打印判定，任一不符即整体失败。 */
bool ReportInvalidArgChecks(const std::vector<InvalidArgCheck> &checks)
{
    bool ok = true;
    for (const InvalidArgCheck &c : checks) {
        if (c.got != c.want) {
            std::printf("  [FAIL] invalid-arg %-34s got=%d want=%d\n", c.name,
                        c.got, c.want);
            ok = false;
        } else {
            std::printf("  [ok] invalid-arg %-36s status=%d\n", c.name, c.got);
        }
    }
    return ok;
}

/**
 * 非法入参必须返回确定错误码而不是崩溃或静默算错。
 */
bool TestInvalidArgs(aclsparseHandle_t handle)
{
    // a 仅用于取 nnz；矩阵本体由 SetupFloat8x8TestDeviceMatrices 上传（同种子同形状）。
    const SpgemmRefCsr a = SpgemmRefBuildCsr(8, 8, 2, 1, 0, false, 3);
    DeviceCsr dA;
    DeviceCsr dB;
    DevBuf cRow;
    aclsparseSpMatDescr_t matC = nullptr;
    SPGEMM_EXPECT(SetupFloat8x8TestDeviceMatrices(dA, dB, cRow, matC),
                  "setup 8x8 float matrices failed");

    InvalidArgCtx c;
    c.handle = handle;
    SPGEMM_ST(aclsparseSpGEMMCreateDescr(&c.descr));

    std::vector<InvalidArgCheck> checks;
    std::vector<aclsparseSpMatDescr_t> extra;
    CollectOpAndTypeChecks(c, dA, dB, matC, checks);
    CollectIndexFormChecks(c, dA, a, dB.descr, matC, checks, extra);
    CollectDimMismatchCheck(c, dA, matC, checks);
    const bool ok = ReportInvalidArgChecks(checks);

    (void)aclsparseSpGEMMDestroyDescr(c.descr);
    (void)aclsparseDestroySpMat(matC);
    for (aclsparseSpMatDescr_t m : extra) {
        (void)aclsparseDestroySpMat(m);
    }
    return ok;
}


/**
 * CSR 内容校验专项用的 device 侧数组持有者。
 * 需要故意构造非法内容（负索引、越界、非升序）。
 */
struct CsrTestData {
    void *dRow = nullptr;
    void *dCol = nullptr;
    void *dVal = nullptr;
    aclsparseSpMatDescr_t descr = nullptr;

    void Free()
    {
        if (descr != nullptr) {
            (void)aclsparseDestroySpMat(descr);
        }
        if (dRow != nullptr) {
            (void)aclrtFree(dRow);
        }
        if (dCol != nullptr) {
            (void)aclrtFree(dCol);
        }
        if (dVal != nullptr) {
            (void)aclrtFree(dVal);
        }
        descr = nullptr;
        dRow = nullptr;
        dCol = nullptr;
        dVal = nullptr;
    }
};

/** host 侧数组 → device + CSR 描述符。nnz == 0 时 colIdx/values 传 nullptr。 */
CsrTestData MakeTestCsr(int32_t rows, int32_t cols, const std::vector<int32_t> &crow,
                        const std::vector<int32_t> &col, int32_t nnz)
{
    CsrTestData d;
    (void)aclrtMalloc(&d.dRow, crow.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    (void)aclrtMemcpy(d.dRow, crow.size() * 4, crow.data(), crow.size() * 4,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (nnz > 0) {
        (void)aclrtMalloc(&d.dCol, std::max<size_t>(col.size(), 1) * 4,
                          ACL_MEM_MALLOC_HUGE_FIRST);
        (void)aclrtMemcpy(d.dCol, col.size() * 4, col.data(), col.size() * 4,
                          ACL_MEMCPY_HOST_TO_DEVICE);
        (void)aclrtMalloc(&d.dVal, static_cast<size_t>(nnz) * 4,
                          ACL_MEM_MALLOC_HUGE_FIRST);
        const std::vector<float> ones(static_cast<size_t>(nnz), 1.0f);
        (void)aclrtMemcpy(d.dVal, static_cast<size_t>(nnz) * 4, ones.data(),
                          static_cast<size_t>(nnz) * 4, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    (void)aclsparseCreateCsr(&d.descr, rows, cols, nnz, d.dRow,
                             nnz > 0 ? d.dCol : nullptr,
                             nnz > 0 ? d.dVal : nullptr,
                             ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                             ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    return d;
}

/** 输出用的空壳 C：只预分配 rowOffsets，nnz = 0。 */
CsrTestData MakeTestOutC(int32_t rows, int32_t cols)
{
    CsrTestData d;
    (void)aclrtMalloc(&d.dRow, (static_cast<size_t>(rows) + 1) * 4,
                      ACL_MEM_MALLOC_HUGE_FIRST);
    (void)aclsparseCreateCsr(&d.descr, rows, cols, 0, d.dRow, nullptr, nullptr,
                             ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                             ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    return d;
}

/**
 * 在给定描述符上跑一次 WorkEstimation（查询 → 分配 → 执行），返回执行期状态码。
 */
aclsparseStatus_t TryWorkEstimationOn(aclsparseHandle_t handle,
                                      aclsparseSpMatDescr_t mA,
                                      aclsparseSpMatDescr_t mB,
                                      aclsparseSpMatDescr_t mC)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    aclsparseSpGEMMDescr_t descr = nullptr;
    (void)aclsparseSpGEMMCreateDescr(&descr);
    size_t bs = 0;
    (void)aclsparseSpGEMMWorkEstimation(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, mA, mB, &beta, mC, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_DEFAULT, descr, &bs, nullptr);
    void *buf = nullptr;
    (void)aclrtMalloc(&buf, bs, ACL_MEM_MALLOC_HUGE_FIRST);
    const aclsparseStatus_t st = aclsparseSpGEMMWorkEstimation(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, mA, mB, &beta, mC, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_DEFAULT, descr, &bs, buf);
    (void)aclrtFree(buf);
    (void)aclsparseSpGEMMDestroyDescr(descr);
    return st;
}

/** 一条判定的输入：一个 CSR 的 host 侧原始内容。 */
struct CsrSpec {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> rowPtr;
    std::vector<int32_t> colIdx;
    int32_t nnz = 0;
};

/** 大多数用例共用的合法 B：4×4、每行 1 个非零、类单位阵。 */
inline CsrSpec ValidTestBSpec()
{
    return CsrSpec{4, 4, {0, 1, 2, 3, 4}, {0, 1, 2, 3}, 4};
}

/** 一条判定：名字 + 期望状态码 + A/B 的内容（C 恒为 (A.rows, A.cols) 的空壳）。 */
struct CsrCase {
    const char *name;
    aclsparseStatus_t want;
    CsrSpec a;
    CsrSpec b;
};

/** 建 A/B/C、跑一次 WorkEstimation、报告，三段 device 内存当场释放。 */
bool RunCsrContentCases(aclsparseHandle_t handle, const std::vector<CsrCase> &cases)
{
    bool ok = true;
    for (const CsrCase &cs : cases) {
        CsrTestData a = MakeTestCsr(cs.a.rows, cs.a.cols, cs.a.rowPtr, cs.a.colIdx,
                                    cs.a.nnz);
        CsrTestData b = MakeTestCsr(cs.b.rows, cs.b.cols, cs.b.rowPtr, cs.b.colIdx,
                                    cs.b.nnz);
        CsrTestData c = MakeTestOutC(cs.a.rows, cs.a.cols);
        const aclsparseStatus_t st =
            TryWorkEstimationOn(handle, a.descr, b.descr, c.descr);
        if (st != cs.want) {
            std::printf("  [FAIL] %-40s got=%d want=%d\n", cs.name, st, cs.want);
            ok = false;
        } else {
            std::printf("  [ok] %-40s\n", cs.name);
        }
        a.Free();
        b.Free();
        c.Free();
    }
    return ok;
}

/**
 * rowOffsets 校验组（用例 1~8）：非法形态返回 INVALID_VALUE，合法边界返回 SUCCESS。
 */
const std::vector<CsrCase> &CsrContentRowOffsetCases()
{
    static const std::vector<CsrCase> cases = {
        {"A.rowOff[0]!=0", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{3, 4, {1, 2, 3, 4}, {0, 1, 2, 3}, 4}, ValidTestBSpec()},
        {"B.rowOff[0]!=0", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{4, 4, {0, 1, 2, 3, 4}, {0, 1, 2, 3}, 4},
         CsrSpec{4, 4, {2, 3, 4, 5, 6}, {0, 1, 2, 3}, 4}},
        {"A.rowOff non-monotone", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{3, 4, {0, 3, 2, 4}, {0, 1, 2, 3}, 4}, ValidTestBSpec()},
        {"B.rowOff non-monotone", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{4, 4, {0, 1, 2, 3, 4}, {0, 1, 2, 3}, 4},
         CsrSpec{4, 4, {0, 2, 1, 3, 4}, {0, 1, 2, 3}, 4}},
        // A.rowPtr[3] = 5 但 nnz = 4
        {"A.rowOff[M]!=nnz", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{3, 4, {0, 1, 2, 5}, {0, 1, 2, 3}, 4}, ValidTestBSpec()},
        // B.rowPtr[4] = 3 但 nnz = 4
        {"B.rowOff[M]!=nnz", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{4, 4, {0, 1, 2, 3, 4}, {0, 1, 2, 3}, 4},
         CsrSpec{4, 4, {0, 1, 2, 3, 3}, {0, 1, 2, 3}, 4}},
        // 第 0、2 行为空行：相邻 rowOffsets 相等是合法 CSR
        {"valid empty rows (no err)", ACL_SPARSE_STATUS_SUCCESS,
         CsrSpec{4, 4, {0, 0, 2, 2, 4}, {0, 1, 2, 3}, 4}, ValidTestBSpec()},
        {"valid nnz=0 (no err)", ACL_SPARSE_STATUS_SUCCESS,
         CsrSpec{4, 4, {0, 0, 0, 0, 0}, {}, 0}, ValidTestBSpec()},
    };
    return cases;
}

/**
 * colIndices 校验组（用例 9~16）：非法内容返回 INVALID_VALUE，合法边界返回 SUCCESS。
 */
const std::vector<CsrCase> &CsrContentColIndexCases()
{
    static const std::vector<CsrCase> cases = {
        {"A neg colIdx", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{2, 4, {0, 2, 4}, {-1, 1, 0, 2}, 4}, ValidTestBSpec()},
        // col = 4 >= K = 4
        {"A colIdx>=K", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{2, 4, {0, 2, 4}, {0, 4, 1, 2}, 4}, ValidTestBSpec()},
        {"B neg colIdx", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{4, 4, {0, 1, 2, 3, 4}, {0, 1, 2, 3}, 4},
         CsrSpec{4, 4, {0, 1, 2, 3, 4}, {-1, 1, 2, 3}, 4}},
        // col = 4 >= N = 4
        {"B colIdx>=N", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{4, 4, {0, 1, 2, 3, 4}, {0, 1, 2, 3}, 4},
         CsrSpec{4, 4, {0, 1, 2, 3, 4}, {0, 1, 2, 4}, 4}},
        // A 第 0 行 [1,0] 降序
        {"A row unsorted", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{2, 4, {0, 2, 4}, {1, 0, 2, 3}, 4}, ValidTestBSpec()},
        {"B row unsorted", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{4, 4, {0, 1, 2, 3, 4}, {0, 1, 2, 3}, 4},
         CsrSpec{4, 4, {0, 2, 4, 6, 8}, {1, 0, 3, 2, 1, 0, 3, 2}, 8}},
        {"valid boundary 0/cols-1 (no err)", ACL_SPARSE_STATUS_SUCCESS,
         CsrSpec{2, 4, {0, 2, 4}, {0, 3, 0, 3}, 4}, ValidTestBSpec()},
        // A 第 0 行 [0,0,1]、第 1 行 [1,2,3]：重复列合法（非严格递增）
        {"valid dup colIdx (no err)", ACL_SPARSE_STATUS_SUCCESS,
         CsrSpec{2, 4, {0, 3, 6}, {0, 0, 1, 1, 2, 3}, 6}, ValidTestBSpec()},
    };
    return cases;
}

/** 与 TryWorkEstimationOn 相同，但以 beta=1.0 跑（触发 C_in 校验路径）。 */
aclsparseStatus_t TryWorkEstimationWithBeta(aclsparseHandle_t handle,
                                            aclsparseSpMatDescr_t mA,
                                            aclsparseSpMatDescr_t mB,
                                            aclsparseSpMatDescr_t mC)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    aclsparseSpGEMMDescr_t descr = nullptr;
    (void)aclsparseSpGEMMCreateDescr(&descr);
    size_t bs = 0;
    (void)aclsparseSpGEMMWorkEstimation(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, mA, mB, &beta, mC, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_DEFAULT, descr, &bs, nullptr);
    void *buf = nullptr;
    (void)aclrtMalloc(&buf, bs, ACL_MEM_MALLOC_HUGE_FIRST);
    const aclsparseStatus_t st = aclsparseSpGEMMWorkEstimation(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, mA, mB, &beta, mC, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_DEFAULT, descr, &bs, buf);
    (void)aclrtFree(buf);
    (void)aclsparseSpGEMMDestroyDescr(descr);
    return st;
}

/** C_in CSR 内容校验的判定结构：A/B 合法，C 按 cIn 构造（作为输入侧 C_in）。 */
struct CsrCInCase {
    const char *name;
    aclsparseStatus_t want;
    CsrSpec cIn;
};

/** 合法的 A（2×4, nnz=2）和 B（4×4, nnz=4, 类单位阵）。 */
inline CsrSpec ValidTestASpec()
{
    return CsrSpec{2, 4, {0, 1, 2}, {0, 1}, 2};
}

/** 建 A/B/C_in、以 beta=1 跑一次 WorkEstimation、报告。 */
bool RunCsrCInCases(aclsparseHandle_t handle, const std::vector<CsrCInCase> &cases)
{
    bool ok = true;
    for (const CsrCInCase &cs : cases) {
        CsrTestData a = MakeTestCsr(2, 4, {0, 1, 2}, {0, 1}, 2);
        CsrTestData b = MakeTestCsr(4, 4, {0, 1, 2, 3, 4}, {0, 1, 2, 3}, 4);
        CsrTestData c = MakeTestCsr(cs.cIn.rows, cs.cIn.cols, cs.cIn.rowPtr,
                                    cs.cIn.colIdx, cs.cIn.nnz);
        const aclsparseStatus_t st =
            TryWorkEstimationWithBeta(handle, a.descr, b.descr, c.descr);
        if (st != cs.want) {
            std::printf("  [FAIL] %-40s got=%d want=%d\n", cs.name, st, cs.want);
            ok = false;
        } else {
            std::printf("  [ok] %-40s\n", cs.name);
        }
        a.Free();
        b.Free();
        c.Free();
    }
    return ok;
}

const std::vector<CsrCInCase> &CsrCInContentCases()
{
    static const std::vector<CsrCInCase> cases = {
        {"C_in rowOff[0]!=0", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{2, 4, {1, 2, 3}, {0, 1}, 2}},
        {"C_in rowOff non-monotone", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{2, 4, {0, 2, 1}, {0, 1}, 2}},
        {"C_in rowOff[M]!=nnz", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{2, 4, {0, 1, 3}, {0, 1}, 2}},
        {"C_in neg colIdx", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{2, 4, {0, 1, 2}, {-1, 1}, 2}},
        {"C_in colIdx>=N", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{2, 4, {0, 1, 2}, {4, 1}, 2}},
        {"C_in row unsorted", ACL_SPARSE_STATUS_INVALID_VALUE,
         CsrSpec{2, 4, {0, 2, 4}, {1, 0, 3, 2}, 4}},
        {"C_in valid (no err)", ACL_SPARSE_STATUS_SUCCESS,
         CsrSpec{2, 4, {0, 1, 2}, {3, 3}, 2}},
        {"C_in valid nnz=0 (no err)", ACL_SPARSE_STATUS_SUCCESS,
         CsrSpec{2, 4, {0, 0, 0}, {}, 0}},
    };
    return cases;
}

/** 把矩阵的三段 device 数组整块拷回 host，用于运行前后逐字节比对。 */
void SnapshotCsrBytes(const DeviceCsr &d, const SpgemmRefCsr &m, aclDataType dt,
                      std::vector<uint8_t> &row, std::vector<uint8_t> &col,
                      std::vector<uint8_t> &val)
{
    const size_t rowBytes = (static_cast<size_t>(m.rows) + 1) * 4;
    const size_t colBytes = static_cast<size_t>(m.Nnz()) * 4;
    const size_t valBytes = static_cast<size_t>(m.Nnz()) * SpgemmDtypeBytes(dt);
    row.assign(rowBytes, 0);
    col.assign(colBytes, 0);
    val.assign(valBytes, 0);
    (void)aclrtMemcpy(row.data(), rowBytes, d.rowPtr.Get(), rowBytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    (void)aclrtMemcpy(col.data(), colBytes, d.colIdx.Get(), colBytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    (void)aclrtMemcpy(val.data(), valBytes, d.values.Get(), valBytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
}

/** 输入只读性：跑完整个流程后，A/B 的三段 device 数组必须逐字节不变。 */
bool TestInputReadonly(aclsparseHandle_t handle)
{
    const SpgemmRefCsr a = SpgemmRefBuildCsr(16, 16, 3, 1, 0, false, 7);
    const SpgemmRefCsr b = SpgemmRefBuildCsr(16, 16, 3, 3, 0, false, 8);
    const aclDataType dts[] = {ACL_FLOAT, ACL_FLOAT16, ACL_BF16, ACL_COMPLEX64};
    bool ok = true;
    for (aclDataType dt : dts) {
        DeviceCsr dA;
        DeviceCsr dB;
        SPGEMM_EXPECT(UploadCsr(a, dt, dA), "upload A");
        SPGEMM_EXPECT(UploadCsr(b, dt, dB), "upload B");
        std::vector<uint8_t> aRowB;
        std::vector<uint8_t> aColB;
        std::vector<uint8_t> aValB;
        std::vector<uint8_t> bRowB;
        std::vector<uint8_t> bColB;
        std::vector<uint8_t> bValB;
        SnapshotCsrBytes(dA, a, dt, aRowB, aColB, aValB);
        SnapshotCsrBytes(dB, b, dt, bRowB, bColB, bValB);

        float alphaF[2] = {1.0f, 0.0f};
        float betaF[2] = {0.0f, 0.0f};
        HostResult out;
        int64_t np = 0;
        (void)RunSpgemmStages(handle, a, b, nullptr, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                              alphaF, betaF, out, &np);

        std::vector<uint8_t> aRowA;
        std::vector<uint8_t> aColA;
        std::vector<uint8_t> aValA;
        std::vector<uint8_t> bRowA;
        std::vector<uint8_t> bColA;
        std::vector<uint8_t> bValA;
        SnapshotCsrBytes(dA, a, dt, aRowA, aColA, aValA);
        SnapshotCsrBytes(dB, b, dt, bRowA, bColA, bValA);

        const bool ro = (aRowB == aRowA) && (aColB == aColA) && (aValB == aValA) &&
                        (bRowB == bRowA) && (bColB == bColA) && (bValB == bValA);
        if (ro) {
            std::printf("  [ok] readonly/%s\n", SpgemmDtypeName(dt));
        } else {
            std::printf("  [FAIL] readonly/%s\n", SpgemmDtypeName(dt));
            ok = false;
        }
    }
    return ok;
}

/** 输出规范性：rowOffsets 单调非降 + 行内列索引严格升序（无越界写的间接判据）。 */
bool IsOutputStructureValid(const HostResult &out)
{
    for (size_t r = 0; r + 1 < out.rowPtr.size(); r++) {
        if (out.rowPtr[r + 1] < out.rowPtr[r]) {
            return false;
        }
        for (int32_t p = out.rowPtr[r] + 1; p < out.rowPtr[r + 1]; p++) {
            if (out.colIdx[static_cast<size_t>(p)] <=
                out.colIdx[static_cast<size_t>(p) - 1]) {
                return false;
            }
        }
    }
    return true;
}

/** 输出缓冲区 canary：结果自身的结构必须合法。 */
bool TestOutputStructure(aclsparseHandle_t handle)
{
    const SpgemmRefCsr a = SpgemmRefBuildCsr(16, 16, 3, 1, 0, false, 11);
    const SpgemmRefCsr b = SpgemmRefBuildCsr(16, 16, 3, 3, 0, false, 12);
    float alphaF[2] = {1.0f, 0.0f};
    float betaF[2] = {0.0f, 0.0f};
    HostResult out;
    int64_t np = 0;
    if (!RunSpgemmStages(handle, a, b, nullptr, ACL_FLOAT, ACL_SPARSE_SPGEMM_DEFAULT,
                         alphaF, betaF, out, &np)) {
        std::printf("  [FAIL] canary setup failed\n");
        return false;
    }
    if (out.nnz <= 0 || !IsOutputStructureValid(out)) {
        std::printf("  [FAIL] output structure invalid\n");
        return false;
    }
    std::printf("  [ok] output structure valid\n");
    return true;
}

/**
 * CSR 内容校验 + 输入只读性 + 输出缓冲区 canary。
 */
/** beta 跨阶段一致性的单条用例：估算阶段传 estBeta，计算阶段换成 compBeta。 */
struct BetaStageCase {
    const char *name;
    float estBeta;
    float compBeta;
    aclsparseStatus_t want;
};

/**
 * 跑一条 beta 跨阶段用例。估算阶段必须成功，计算阶段的返回码与 want 比对。
 */
bool RunBetaCase(aclsparseHandle_t handle, const DeviceCsr &dA, const DeviceCsr &dB,
                 const DeviceCsr &dC, const BetaStageCase &bc)
{
    aclsparseSpGEMMDescr_t descr = nullptr;
    (void)aclsparseSpGEMMCreateDescr(&descr);

    float alpha = 1.0f;
    float estBeta = bc.estBeta;
    size_t bs1 = 0;
    (void)aclsparseSpGEMMWorkEstimation(
        handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, dA.descr, dB.descr, &estBeta, dC.descr, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_DEFAULT, descr, &bs1, nullptr);
    DevBuf buf1;
    bool weOk = buf1.Alloc(bs1);
    if (weOk) {
        aclsparseStatus_t st = aclsparseSpGEMMWorkEstimation(
            handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha, dA.descr, dB.descr, &estBeta, dC.descr, ACL_FLOAT,
            ACL_SPARSE_SPGEMM_DEFAULT, descr, &bs1, buf1.Get());
        weOk = (st == ACL_SPARSE_STATUS_SUCCESS);
    }
    if (!weOk) {
        std::printf("  [FAIL] %-40s WorkEstimation failed\n", bc.name);
        (void)aclsparseSpGEMMDestroyDescr(descr);
        return false;
    }

    float compBeta = bc.compBeta;
    size_t bs2 = 0;
    aclsparseStatus_t st = aclsparseSpGEMMCompute(
        handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, dA.descr, dB.descr, &compBeta, dC.descr, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_DEFAULT, descr, &bs2, nullptr);
    bool ok = true;
    if (st != bc.want) {
        std::printf("  [FAIL] %-40s got=%d want=%d\n", bc.name, st, bc.want);
        ok = false;
    } else {
        std::printf("  [ok] %-40s\n", bc.name);
    }
    (void)aclsparseSpGEMMDestroyDescr(descr);
    return ok;
}

/**
 * beta 跨阶段一致性：WorkEstimation 与 Compute 必须传相同的 beta 值。
 */
bool TestBetaConsistency(aclsparseHandle_t handle)
{
    SpgemmRefCsr a, b, cIn;
    BuildBetaTestMatrices(a, b, cIn);
    DeviceCsr dA;
    DeviceCsr dB;
    DeviceCsr dC;
    SPGEMM_EXPECT(UploadCsr(a, ACL_FLOAT, dA), "upload A");
    SPGEMM_EXPECT(UploadCsr(b, ACL_FLOAT, dB), "upload B");
    SPGEMM_EXPECT(UploadCsr(cIn, ACL_FLOAT, dC), "upload C_in");

    const BetaStageCase cases[] = {
        {"beta 0->1", 0.0f, 1.0f, ACL_SPARSE_STATUS_INVALID_VALUE},
        {"beta 1->0", 1.0f, 0.0f, ACL_SPARSE_STATUS_INVALID_VALUE},
        {"beta 1->2", 1.0f, 2.0f, ACL_SPARSE_STATUS_INVALID_VALUE},
    };

    bool ok = true;
    for (const BetaStageCase &bc : cases) {
        ok = RunBetaCase(handle, dA, dB, dC, bc) && ok;
    }
    return ok;
}

bool TestCsrContentValidation(aclsparseHandle_t handle)
{
    bool ok = RunCsrContentCases(handle, CsrContentRowOffsetCases());
    ok &= RunCsrContentCases(handle, CsrContentColIndexCases());

    std::printf("  -- C_in content (beta!=0) --\n");
    ok &= RunCsrCInCases(handle, CsrCInContentCases());

    std::printf("  -- input readonly --\n");
    ok &= TestInputReadonly(handle);

    std::printf("  -- output canary --\n");
    ok &= TestOutputStructure(handle);
    return ok;
}

/**
 * 阶段状态机的公共入参。
 */
struct StageMachineCtx {
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matA = nullptr;
    aclsparseSpMatDescr_t matB = nullptr;
    aclsparseSpMatDescr_t matC = nullptr;
    float alpha = 1.0f;
    float beta = 0.0f;
};

/** 期望状态码判定 + 报告。name 为标签，与既有输出逐字一致。 */
bool ReportStageStatus(const char *name, aclsparseStatus_t got, aclsparseStatus_t want)
{
    if (got != want) {
        std::printf("  [FAIL] %s got=%d want=%d\n", name, got, want);
        return false;
    }
    std::printf("  [ok] stage %s rejected\n", name);
    return true;
}

/**
 * 第一组：未推进的 descr 上，下游阶段必须被拒绝。
 */
bool TestStageMachineFreshDescr(const StageMachineCtx &c)
{
    aclsparseSpGEMMDescr_t d = nullptr;
    SPGEMM_ST(aclsparseSpGEMMCreateDescr(&d));
    bool ok = true;
    size_t bs2 = 0;
    ok &= ReportStageStatus("Compute-before-WorkEstimation",
        aclsparseSpGEMMCompute(c.handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &c.alpha, c.matA, c.matB, &c.beta, c.matC,
            ACL_FLOAT, ACL_SPARSE_SPGEMM_DEFAULT, d, &bs2, nullptr),
        ACL_SPARSE_STATUS_INVALID_VALUE);

    int64_t np = 0;
    ok &= ReportStageStatus("GetNumProducts-before-WorkEstimation",
                            aclsparseSpGEMMGetNumProducts(d, &np),
                            ACL_SPARSE_STATUS_INVALID_VALUE);
    ok &= ReportStageStatus("Copy-before-Compute",
        aclsparseSpGEMMCopy(c.handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &c.alpha, c.matA, c.matB, &c.beta, c.matC,
            ACL_FLOAT, ACL_SPARSE_SPGEMM_DEFAULT, d),
        ACL_SPARSE_STATUS_INVALID_VALUE);

    size_t bs3 = 0;
    size_t bs2b = 0;
    ok &= ReportStageStatus("EstimateMemory-on-DEFAULT",
        aclsparseSpGEMMEstimateMemory(c.handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &c.alpha, c.matA, c.matB, &c.beta, c.matC,
            ACL_FLOAT, ACL_SPARSE_SPGEMM_DEFAULT, d, 0.5f, &bs3, nullptr, &bs2b),
        ACL_SPARSE_STATUS_NOT_SUPPORTED);

    if (aclsparseSpGEMMDestroyDescr(d) != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseSpGEMMDestroyDescr(nullptr) != ACL_SPARSE_STATUS_SUCCESS) {
        std::printf("  [FAIL] DestroyDescr not idempotent\n");
        ok = false;
    } else {
        std::printf("  [ok] stage DestroyDescr idempotent\n");
    }
    return ok;
}

/**
 * 第二组：ALG2 走完 WorkEstimation 但跳过 EstimateMemory 直接 Compute → INVALID_VALUE。
 */
bool TestStageMachineAlg2NeedsEstimate(const StageMachineCtx &c)
{
    aclsparseSpGEMMDescr_t d = nullptr;
    SPGEMM_ST(aclsparseSpGEMMCreateDescr(&d));
    size_t bs1 = 0;
    aclsparseStatus_t st = aclsparseSpGEMMWorkEstimation(
        c.handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &c.alpha, c.matA, c.matB, &c.beta, c.matC, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_ALG2, d, &bs1, nullptr);
    DevBuf buf1;
    if (st == ACL_SPARSE_STATUS_SUCCESS && buf1.Alloc(bs1)) {
        st = aclsparseSpGEMMWorkEstimation(
            c.handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &c.alpha, c.matA, c.matB, &c.beta, c.matC, ACL_FLOAT,
            ACL_SPARSE_SPGEMM_ALG2, d, &bs1, buf1.Get());
    }
    size_t bs2 = 0;
    const bool ok = ReportStageStatus("ALG2-Compute-before-EstimateMemory",
        aclsparseSpGEMMCompute(c.handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &c.alpha, c.matA, c.matB, &c.beta, c.matC,
            ACL_FLOAT, ACL_SPARSE_SPGEMM_ALG2, d, &bs2, nullptr),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    (void)aclsparseSpGEMMDestroyDescr(d);
    return ok;
}

/** 第三组：workspace 不足 —— 传一个比查询值小的 bufferSize1 → INSUFFICIENT_RESOURCES。 */
bool TestStageMachineInsufficientWorkspace(const StageMachineCtx &c)
{
    aclsparseSpGEMMDescr_t d = nullptr;
    SPGEMM_ST(aclsparseSpGEMMCreateDescr(&d));
    size_t bs1 = 0;
    aclsparseStatus_t st = aclsparseSpGEMMWorkEstimation(
        c.handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &c.alpha, c.matA, c.matB, &c.beta, c.matC, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_DEFAULT, d, &bs1, nullptr);
    bool ok = true;
    DevBuf buf1;
    if (st == ACL_SPARSE_STATUS_SUCCESS && bs1 > 64 && buf1.Alloc(bs1)) {
        size_t tooSmall = bs1 - 64;
        st = aclsparseSpGEMMWorkEstimation(
            c.handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &c.alpha, c.matA, c.matB, &c.beta, c.matC, ACL_FLOAT,
            ACL_SPARSE_SPGEMM_DEFAULT, d, &tooSmall, buf1.Get());
        if (st != ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES) {
            std::printf("  [FAIL] insufficient workspace got=%d want=%d\n", st,
                        ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);
            ok = false;
        } else {
            std::printf("  [ok] stage insufficient-workspace rejected\n");
        }
    }
    (void)aclsparseSpGEMMDestroyDescr(d);
    return ok;
}

/**
 * 阶段状态机：非法阶段跳转必须返回确定错误码。
 */
bool TestStageMachine(aclsparseHandle_t handle)
{
    DeviceCsr dA;
    DeviceCsr dB;
    DevBuf cRow;
    aclsparseSpMatDescr_t matC = nullptr;
    if (!SetupFloat8x8TestDeviceMatrices(dA, dB, cRow, matC)) {
        return false;
    }
    StageMachineCtx c;
    c.handle = handle;
    c.matA = dA.descr;
    c.matB = dB.descr;
    c.matC = matC;

    bool ok = TestStageMachineFreshDescr(c);
    ok &= TestStageMachineAlg2NeedsEstimate(c);
    ok &= TestStageMachineInsufficientWorkspace(c);

    (void)aclsparseDestroySpMat(matC);
    return ok;
}

/** 资源泄漏检查：连续 Create/Execute/Destroy N 次。 */
bool TestNoLeak(aclsparseHandle_t handle, int iters)
{
    const SpgemmRefCsr a = SpgemmRefBuildCsr(32, 32, 4, 1, 0, false, 17);
    const SpgemmRefCsr b = SpgemmRefBuildCsr(32, 32, 4, 4, 0, false, 19);
    float alpha = 1.0f;
    float beta = 0.0f;
    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 1.0, 0.0, 0.0, 0.0, nullptr, false);
    for (int i = 0; i < iters; i++) {
        HostResult got;
        if (!RunSpgemmStages(handle, a, b, nullptr, ACL_FLOAT,
                             ACL_SPARSE_SPGEMM_DEFAULT, &alpha, &beta, got, nullptr)) {
            std::printf("  [FAIL] leak-check iteration %d failed\n", i);
            return false;
        }
        if (got.nnz != ref.Nnz()) {
            std::printf("  [FAIL] leak-check iter %d nnz drift %ld vs %d\n", i,
                        (long)got.nnz, ref.Nnz());
            return false;
        }
    }
    std::printf("  [ok] %-42s %d iterations, nnz stable\n", "no-leak", iters);
    return true;
}

/** alpha/beta 的 DEVICE pointer mode 路径。 */
bool TestDevicePointerMode(aclsparseHandle_t handle)
{
    const SpgemmRefCsr a = SpgemmRefBuildCsr(16, 16, 3, 1, 0, false, 23);
    const SpgemmRefCsr b = SpgemmRefBuildCsr(16, 16, 3, 3, 0, false, 29);

    // alpha = 2.5, beta = 0，标量放在 device 上
    DevBuf dAlpha;
    DevBuf dBeta;
    SPGEMM_EXPECT(dAlpha.Alloc(sizeof(float)), "alloc alpha failed");
    SPGEMM_EXPECT(dBeta.Alloc(sizeof(float)), "alloc beta failed");
    const float hAlpha = 2.5f;
    const float hBeta = 0.0f;
    SPGEMM_EXPECT(dAlpha.Upload(&hAlpha, sizeof(float)), "upload alpha failed");
    SPGEMM_EXPECT(dBeta.Upload(&hBeta, sizeof(float)), "upload beta failed");

    SPGEMM_ST(aclsparseSetPointerMode(handle, ACL_SPARSE_POINTER_MODE_DEVICE));
    HostResult got;
    const bool runOk = RunSpgemmStages(handle, a, b, nullptr, ACL_FLOAT,
                                       ACL_SPARSE_SPGEMM_DEFAULT,
                                       dAlpha.Get(), dBeta.Get(), got, nullptr);
    SPGEMM_ST(aclsparseSetPointerMode(handle, ACL_SPARSE_POINTER_MODE_HOST));
    SPGEMM_EXPECT(runOk, "device-pointer-mode run failed");

    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 2.5, 0.0, 0.0, 0.0, nullptr, false);
    if (!CompareResult(got, ref, ACL_FLOAT, "device-ptr-mode/fp32")) {
        return false;
    }
    std::printf("  [ok] %-42s alpha=2.5 from device\n", "device-ptr-mode/fp32");
    return true;
}

/**
 * 官方精度用例规模的抽样验证（M/K/N ∈ [1000, 20000]，degree ∈ [1,8]）。
 */
bool TestScaleCase(aclsparseHandle_t handle, int32_t M, int32_t K, int32_t N,
                   int32_t degA, int32_t degB, int32_t emptyA, int32_t emptyB,
                   aclDataType dt, const char *name)
{
    const bool isComplex = (dt == ACL_COMPLEX64);
    SpgemmRefCsr a = SpgemmRefBuildCsr(M, K, degA, 1, emptyA, isComplex, 101);
    SpgemmRefCsr b = SpgemmRefBuildCsr(K, N, degB, degA > 0 ? degA : 1,
                                       emptyB, isComplex, 211);
    SpgemmQuantizeCsr(a, dt);
    SpgemmQuantizeCsr(b, dt);
    // alpha/beta 必须按 computeType 编码（cuSPARSE 语义：标量类型 == computeType）。
    // fp16/bf16 传 float 会被按 16 位解释而读成 0，这里按 dtype 选择正确的表示。
    SpgemmTestScalars sc;
    InitAlphaBetaScalars(sc, dt);

    HostResult got;
    int64_t numProds = -1;
    const std::string tag = std::string(name) + "/" + SpgemmDtypeName(dt);
    if (!RunSpgemmStages(handle, a, b, nullptr, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                         sc.alpha, sc.beta, got, &numProds)) {
        return false;
    }
    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 1.0, 0.0, 0.0, 0.0, nullptr,
                                              isComplex);
    if (!CompareResult(got, ref, dt, tag.c_str())) {
        return false;
    }
    std::printf("  [ok] %-42s nnz(C)=%-9ld numProds=%ld\n", tag.c_str(), (long)got.nnz,
                (long)numProds);
    return true;
}

/**
 * T3 宽 N 回归：构造一行输出列数远超 mergeCapacity 且 N 很大的场景，
 * 强制走列分块路径并跨多个窗口。
 */
bool TestWideChunk(aclsparseHandle_t handle, aclDataType dt)
{
    const bool isComplex = (dt == ACL_COMPLEX64);
    const int32_t M = 4;
    const int32_t K = 64;
    const int32_t N = 20000;

    // A 每行取 B 的 64 行；B 每行 128 个非零且跨度大 → 单行输出列数远超 mergeCapacity
    SpgemmRefCsr a;
    a.rows = M;
    a.cols = K;
    a.rowPtr.assign(static_cast<size_t>(M) + 1, 0);
    uint32_t rng = 4243;
    auto nextVal = [&rng]() -> double {
        rng = rng * 1664525U + 1013904223U;
        return static_cast<double>(static_cast<int32_t>(rng >> 8) % 200 - 100) / 100.0;
    };
    for (int32_t i = 0; i < M; i++) {
        for (int32_t j = 0; j < K; j++) {
            a.colIdx.push_back(j);
            a.valRe.push_back(nextVal());
            a.valIm.push_back(isComplex ? nextVal() * 0.25 : 0.0);
        }
        a.rowPtr[static_cast<size_t>(i) + 1] = static_cast<int32_t>(a.colIdx.size());
    }
    // stride=157 让 B 各行命中的列分散到整个 [0, N)，输出行宽 ~64*128
    SpgemmRefCsr b = SpgemmRefBuildCsr(K, N, 128, 157, 0, isComplex, 77);

    SpgemmQuantizeCsr(a, dt);
    SpgemmQuantizeCsr(b, dt);

    SpgemmTestScalars sc;
    InitAlphaBetaScalars(sc, dt);

    HostResult got;
    const std::string tag = std::string("wide-chunk/") + SpgemmDtypeName(dt);
    if (!RunSpgemmStages(handle, a, b, nullptr, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                         sc.alpha, sc.beta, got, nullptr)) {
        return false;
    }
    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 1.0, 0.0, 0.0, 0.0, nullptr,
                                              isComplex);
    if (!CompareResult(got, ref, dt, tag.c_str())) {
        return false;
    }
    // 断言确实触发了 T3：单行输出列数必须超过 T1 的 mergeCapacity 上限 4096
    const int32_t maxRow = got.rowPtr.size() > 1
        ? (got.rowPtr[1] - got.rowPtr[0]) : 0;
    SPGEMM_EXPECT(maxRow > 4096,
                  "%s: row width %d did not exceed mergeCapacity, T3 not exercised",
                  tag, maxRow);
    std::printf("  [ok] %-42s nnz(C)=%-8ld row0 width=%d (T3 wide-N)\n", tag.c_str(),
                (long)got.nnz, maxRow);
    return true;
}

/**
 * beta + ALG2/ALG3 专项的三矩阵构造（M=64, K=64, N=128）：
 * A*B 每行命中列 0，C_in 每行命中列 100（完全不相交），并集后每行 2 个非零。
 */
void BuildBetaChunkedMatrices(SpgemmRefCsr &a, SpgemmRefCsr &b, SpgemmRefCsr &cIn)
{
    const int32_t M = 64;
    const int32_t K = 64;
    const int32_t N = 128;

    a.rows = M;
    a.cols = K;
    a.rowPtr.assign(static_cast<size_t>(M) + 1, 0);
    for (int32_t i = 0; i < M; i++) {
        a.colIdx.push_back(i);
        a.valRe.push_back(1.0 + 0.01 * i);
        a.valIm.push_back(0.0);
        a.rowPtr[static_cast<size_t>(i) + 1] = i + 1;
    }
    b.rows = K;
    b.cols = N;
    b.rowPtr.assign(static_cast<size_t>(K) + 1, 0);
    for (int32_t i = 0; i < K; i++) {
        b.colIdx.push_back(0);
        b.valRe.push_back(2.0 + 0.01 * i);
        b.valIm.push_back(0.0);
        b.rowPtr[static_cast<size_t>(i) + 1] = i + 1;
    }
    cIn.rows = M;
    cIn.cols = N;
    cIn.rowPtr.assign(static_cast<size_t>(M) + 1, 0);
    for (int32_t i = 0; i < M; i++) {
        cIn.colIdx.push_back(100);
        cIn.valRe.push_back(10.0 + i);
        cIn.valIm.push_back(0.0);
        cIn.rowPtr[static_cast<size_t>(i) + 1] = i + 1;
    }
}

/**
 * beta != 0 与 ALG2/ALG3 组合：验证 EstimateMemory 回填的 bufferSize2 计入了 C_in 结构。
 */
bool TestBetaWithChunkedAlg(aclsparseHandle_t handle, aclsparseSpGEMMAlg_t alg,
                            const char *algName)
{
    const int32_t M = 64;
    SpgemmRefCsr a, b, cIn;
    BuildBetaChunkedMatrices(a, b, cIn);

    float alphaF[2] = {1.0f, 0.0f};
    float betaF[2] = {1.0f, 0.0f};
    HostResult got;
    const std::string tag = std::string("beta-union-") + algName + "/fp32";
    if (!RunSpgemmStages(handle, a, b, &cIn, ACL_FLOAT, alg, alphaF, betaF, got,
                         nullptr)) {
        return false;
    }
    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 1.0, 0.0, 1.0, 0.0, &cIn, false);
    SPGEMM_EXPECT(ref.Nnz() == 2 * M, "%s: ref nnz should be %d, got %d", tag,
                  2 * M, ref.Nnz());
    if (!CompareResult(got, ref, ACL_FLOAT, tag.c_str())) {
        return false;
    }
    std::printf("  [ok] %-42s nnz(C)=%ld (A*B %d + C_in %d, disjoint)\n", tag.c_str(),
                (long)got.nnz, M, M);
    return true;
}

/**
 * 复用专项：WorkEstimation 查询 + 执行（fp32，alpha = beta = 1）。
 * buf1 由调用者持有，必须活到 Compute 结束。返回最后一次调用的状态码。
 */
aclsparseStatus_t RunWorkEstimationOnce(aclsparseHandle_t handle, const DeviceCsr &dA,
                                        const DeviceCsr &dB, const DeviceCsr &dC,
                                        aclsparseSpGEMMDescr_t descr, DevBuf &buf1)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    size_t bs1 = 0;
    aclsparseStatus_t st = aclsparseSpGEMMWorkEstimation(
        handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, dA.descr, dB.descr, &beta, dC.descr, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_DEFAULT, descr, &bs1, nullptr);
    if (st == ACL_SPARSE_STATUS_SUCCESS && buf1.Alloc(bs1)) {
        st = aclsparseSpGEMMWorkEstimation(
            handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha, dA.descr, dB.descr, &beta, dC.descr, ACL_FLOAT,
            ACL_SPARSE_SPGEMM_DEFAULT, descr, &bs1, buf1.Get());
    }
    return st;
}

/**
 * 复用专项：单次 Compute（查询 → 分配 buffer2 → 执行）并回读 nnz(C)。
 * buf2 是本函数的局部量，返回即释放——与逐次迭代各自持有 workspace 的语义一致。
 */
bool RunComputeOnceAndQueryNnz(aclsparseHandle_t handle, const DeviceCsr &dA,
                               const DeviceCsr &dB, const DeviceCsr &dC,
                               aclsparseSpGEMMDescr_t descr, int iter, int64_t &nnzC)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    size_t bs2 = 0;
    aclsparseStatus_t st = aclsparseSpGEMMCompute(
        handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, dA.descr, dB.descr, &beta, dC.descr, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_DEFAULT, descr, &bs2, nullptr);
    DevBuf buf2;
    if (st != ACL_SPARSE_STATUS_SUCCESS || !buf2.Alloc(bs2)) {
        std::printf("  [FAIL] repeated-compute iter %d query st=%d\n", iter, st);
        return false;
    }
    st = aclsparseSpGEMMCompute(
        handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, dA.descr, dB.descr, &beta, dC.descr, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_DEFAULT, descr, &bs2, buf2.Get());
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::printf("  [FAIL] repeated-compute iter %d exec st=%d\n", iter, st);
        return false;
    }
    int64_t r = 0;
    int64_t cc = 0;
    (void)aclsparseSpMatGetSize(dC.descr, &r, &cc, &nnzC);
    return true;
}

/**
 * 描述符复用：同一 descr 上重复调用 Compute（beta != 0），验证 nnz(C) 稳定。
 */
bool TestRepeatedComputeWithBeta(aclsparseHandle_t handle)
{
    SpgemmRefCsr a, b, cIn;
    BuildBetaTestMatrices(a, b, cIn);
    DeviceCsr dA;
    DeviceCsr dB;
    DeviceCsr dC;
    SPGEMM_EXPECT(UploadCsr(a, ACL_FLOAT, dA), "upload A failed");
    SPGEMM_EXPECT(UploadCsr(b, ACL_FLOAT, dB), "upload B failed");
    SPGEMM_EXPECT(UploadCsr(cIn, ACL_FLOAT, dC), "upload C_in failed");

    aclsparseSpGEMMDescr_t descr = nullptr;
    SPGEMM_ST(aclsparseSpGEMMCreateDescr(&descr));

    DevBuf buf1;
    const aclsparseStatus_t st =
        RunWorkEstimationOnce(handle, dA, dB, dC, descr, buf1);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::printf("  [FAIL] repeated-compute WorkEstimation st=%d\n", st);
        (void)aclsparseSpGEMMDestroyDescr(descr);
        return false;
    }

    // 连续两次 Compute，第二次必须与第一次得到相同的 nnz(C)
    bool ok = true;
    int64_t firstNnz = -1;
    for (int iter = 0; iter < 2; iter++) {
        int64_t nnzC = 0;
        if (!RunComputeOnceAndQueryNnz(handle, dA, dB, dC, descr, iter, nnzC)) {
            ok = false;
            break;
        }
        if (iter == 0) {
            firstNnz = nnzC;
        } else if (nnzC != firstNnz) {
            std::printf("  [FAIL] repeated-compute nnz drift %ld -> %ld\n",
                        (long)firstNnz, (long)nnzC);
            ok = false;
        }
    }
    if (ok) {
        std::printf("  [ok] %-42s nnz stable across 2 Computes (=%ld)\n",
                    "repeated-compute-beta", (long)firstNnz);
    }
    (void)aclsparseSpGEMMDestroyDescr(descr);
    return ok;
}

/**
 * alpha/beta 标量编码契约：标量的存储类型必须等于 computeType。
 * 对四种 dtype 分别用正确编码的 alpha=2 验证结果确实被放大 2 倍。
 */
bool TestScalarEncoding(aclsparseHandle_t handle, aclDataType dt)
{
    const bool isComplex = (dt == ACL_COMPLEX64);
    SpgemmRefCsr a = SpgemmRefBuildCsr(16, 16, 2, 1, 0, isComplex, 61);
    SpgemmRefCsr b = SpgemmRefBuildCsr(16, 16, 2, 2, 0, isComplex, 67);
    SpgemmQuantizeCsr(a, dt);
    SpgemmQuantizeCsr(b, dt);

    // alpha = 2.0，按 computeType 正确编码
    SpgemmTestScalars sc;
    InitAlphaBetaScalars(sc, dt, 2.0f);

    HostResult got;
    const std::string tag = std::string("scalar-encoding/") + SpgemmDtypeName(dt);
    if (!RunSpgemmStages(handle, a, b, nullptr, dt, ACL_SPARSE_SPGEMM_DEFAULT,
                         sc.alpha, sc.beta, got, nullptr)) {
        return false;
    }
    const SpgemmRefCsr ref = SpgemmRefCompute(a, b, 2.0, 0.0, 0.0, 0.0, nullptr,
                                              isComplex);
    if (!CompareResult(got, ref, dt, tag.c_str())) {
        return false;
    }
    // 反证：alpha 若被误读成 0，输出会全零。
    bool anyNonZero = false;
    for (int64_t i = 0; i < got.nnz; i++) {
        if (got.valRe[static_cast<size_t>(i)] != 0.0 ||
            (isComplex && got.valIm[static_cast<size_t>(i)] != 0.0)) {
            anyNonZero = true;
            break;
        }
    }
    SPGEMM_EXPECT(anyNonZero, "%s: all-zero output means alpha was decoded as 0", tag);
    std::printf("  [ok] %-42s alpha=2 applied (nnz=%ld)\n", tag.c_str(), (long)got.nnz);
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// GoogleTest fixture for SpGEMM arch22 tests.
// ---------------------------------------------------------------------------
class SpGemmArch22Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        std::printf("[SpGEMM] suite=%s\n", SparseTestFullEnabled() ? "full" : "small functional");
        const char *envDev = std::getenv("SPGEMM_TEST_DEVICE");
        int devId = envDev ? std::atoi(envDev) : 0;
        ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
        ASSERT_EQ(aclrtSetDevice(devId), ACL_SUCCESS);
        ASSERT_EQ(aclrtCreateStream(&s_stream_), ACL_SUCCESS);
        ASSERT_EQ(aclsparseCreate(&s_handle_), ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_EQ(aclsparseSetStream(s_handle_, s_stream_), ACL_SPARSE_STATUS_SUCCESS);
    }
    static void TearDownTestSuite() {
        if (s_handle_) { aclsparseDestroy(s_handle_); s_handle_ = nullptr; }
        if (s_stream_) { aclrtDestroyStream(s_stream_); s_stream_ = nullptr; }
        aclrtResetDevice(0);
        aclFinalize();
    }
    static aclsparseHandle_t s_handle_;
    static aclrtStream s_stream_;
};
aclsparseHandle_t SpGemmArch22Test::s_handle_ = nullptr;
aclrtStream SpGemmArch22Test::s_stream_ = nullptr;

// ---------------------------------------------------------------------------
// Dtype-parameterized tests
// ---------------------------------------------------------------------------
struct DtypeParam {
    const char *name;
    aclDataType type;
};

class SpGemmArch22DtypeTest : public SpGemmArch22Test,
                               public ::testing::WithParamInterface<DtypeParam> {};

static const DtypeParam kAllDtypes[] = {
    {"fp32", ACL_FLOAT}, {"fp16", ACL_FLOAT16}, {"bf16", ACL_BF16}, {"c64", ACL_COMPLEX64}
};

INSTANTIATE_TEST_SUITE_P(AllDtypes, SpGemmArch22DtypeTest,
    ::testing::ValuesIn(kAllDtypes),
    [](const ::testing::TestParamInfo<DtypeParam> &info) {
        return std::string(info.param.name);
    });

// ---- Basic cases × 4 dtypes ----
TEST_P(SpGemmArch22DtypeTest, BasicCases) {
    auto dt = GetParam().type;
    auto h = s_handle_;
    EXPECT_TRUE(TestBasicCase(h,   1,   1,   1, 1, 1, 0, 0, dt, ACL_SPARSE_SPGEMM_DEFAULT, "1x1x1"));
    EXPECT_TRUE(TestBasicCase(h,   4,   4,   4, 1, 1, 0, 0, dt, ACL_SPARSE_SPGEMM_DEFAULT, "4x4x4-deg1"));
    EXPECT_TRUE(TestBasicCase(h,  32,  32,  32, 4, 4, 0, 0, dt, ACL_SPARSE_SPGEMM_DEFAULT, "32x32x32-deg4"));
    EXPECT_TRUE(TestBasicCase(h,  64,  32,  48, 3, 6, 0, 0, dt, ACL_SPARSE_SPGEMM_DEFAULT, "64x32x48-deg3/6"));
    EXPECT_TRUE(TestBasicCase(h, 128,  64,  64, 8, 8, 0, 0, dt, ACL_SPARSE_SPGEMM_DEFAULT, "128x64x64-deg8"));
    EXPECT_TRUE(TestBasicCase(h,  64, 128,  32, 6, 2, 0, 0, dt, ACL_SPARSE_SPGEMM_DEFAULT, "tall-64x128x32"));
    EXPECT_TRUE(TestBasicCase(h,  32,  64, 256, 2, 8, 0, 0, dt, ACL_SPARSE_SPGEMM_DEFAULT, "wide-32x64x256"));
    EXPECT_TRUE(TestBasicCase(h,  48,  48,  48, 4, 4, 5, 0, dt, ACL_SPARSE_SPGEMM_DEFAULT, "empty-rows-A-every5"));
    EXPECT_TRUE(TestBasicCase(h,  48,  48,  48, 4, 4, 0, 7, dt, ACL_SPARSE_SPGEMM_DEFAULT, "empty-rows-B-every7"));
    EXPECT_TRUE(TestBasicCase(h,  48,  48,  48, 4, 4, 3, 4, dt, ACL_SPARSE_SPGEMM_DEFAULT, "empty-rows-AB"));
    EXPECT_TRUE(TestBasicCase(h,  16,  16,  16, 0, 4, 0, 0, dt, ACL_SPARSE_SPGEMM_DEFAULT, "nnzA=0"));
    EXPECT_TRUE(TestBasicCase(h,  16,  16,  16, 4, 0, 0, 0, dt, ACL_SPARSE_SPGEMM_DEFAULT, "nnzB=0"));
    EXPECT_TRUE(TestBasicCase(h,   8,   8,   8, 1, 1, 1, 0, dt, ACL_SPARSE_SPGEMM_DEFAULT, "all-rows-empty-A"));
    EXPECT_TRUE(TestBasicCase(h, 200, 200, 200, 8, 8, 0, 0, dt, ACL_SPARSE_SPGEMM_DEFAULT, "200x200x200-deg8-multicore"));
    EXPECT_TRUE(TestBasicCase(h, 512, 256, 256, 4, 4, 0, 0, dt, ACL_SPARSE_SPGEMM_DEFAULT, "512x256x256-deg4"));
}

// ---- Algorithm enums (fp32 only) ----
TEST_F(SpGemmArch22Test, AlgorithmEnums) {
    EXPECT_TRUE(TestBasicCase(s_handle_, 64, 64, 64, 4, 4, 0, 0, ACL_FLOAT, ACL_SPARSE_SPGEMM_ALG1, "alg1"));
    EXPECT_TRUE(TestBasicCase(s_handle_, 64, 64, 64, 4, 4, 0, 0, ACL_FLOAT, ACL_SPARSE_SPGEMM_ALG2, "alg2"));
    EXPECT_TRUE(TestBasicCase(s_handle_, 64, 64, 64, 4, 4, 0, 0, ACL_FLOAT, ACL_SPARSE_SPGEMM_ALG3, "alg3"));
}

// ---- Explicit zero × 4 dtypes ----
TEST_P(SpGemmArch22DtypeTest, ExplicitZero) {
    EXPECT_TRUE(TestExplicitZero(s_handle_, GetParam().type));
}

// ---- No intersection (fp32 + c64) ----
TEST_F(SpGemmArch22Test, NoIntersection) {
    EXPECT_TRUE(TestNoIntersection(s_handle_, ACL_FLOAT));
    EXPECT_TRUE(TestNoIntersection(s_handle_, ACL_COMPLEX64));
}

// ---- Rotational fast-path arithmetic check × 4 dtypes ----
TEST_P(SpGemmArch22DtypeTest, RotationalNonArithmetic) {
    EXPECT_TRUE(TestRotationalNonArithmetic(s_handle_, GetParam().type));
}

TEST_P(SpGemmArch22DtypeTest, BandLazyValues) {
    EXPECT_TRUE(TestBandLazyValues(s_handle_, GetParam().type, false));
    EXPECT_TRUE(TestBandLazyValues(s_handle_, GetParam().type, true));
    EXPECT_TRUE(TestBandFallbackEmptyBRow(s_handle_, GetParam().type));
}

TEST_P(SpGemmArch22DtypeTest, AlphaScalingFusedPath) {
    EXPECT_TRUE(TestAlphaScalingFusedPath(s_handle_, GetParam().type));
}

// ---- Rotational equal-heads (duplicate B rows) × 4 dtypes ----
TEST_P(SpGemmArch22DtypeTest, RotationalEqualHeads) {
    struct EqCase {
        const char *name;
        int32_t bCols;
        std::vector<std::vector<int32_t>> aRows;
        std::vector<std::vector<int32_t>> bRows;
    };
    const std::vector<EqCase> kEq = {
        {"da=2-dup{0,2}", 4, {{0, 1}}, {{0, 2}, {0, 2}}},
        {"da=3-dup{0,3,6}", 9, {{0, 1, 2}}, {{0, 3, 6}, {0, 3, 6}, {0, 3, 6}}},
        {"da=4-paired-dup", 12, {{0, 1, 2, 3}},
         {{0, 4, 8}, {0, 4, 8}, {1, 5, 9}, {1, 5, 9}}},
        {"ctrl-da=2-distinct", 4, {{0, 1}}, {{0, 2}, {1, 3}}},
        {"ctrl-da=4-distinct", 16, {{0, 1, 2, 3}},
         {{0, 4, 8}, {1, 5, 9}, {2, 6, 10}, {3, 7, 11}}},
    };
    auto dt = GetParam().type;
    for (const EqCase &c : kEq) {
        const SpgemmRefCsr a =
            SpgemmMakeCsrOnes(static_cast<int32_t>(c.bRows.size()), c.aRows);
        const SpgemmRefCsr b = SpgemmMakeCsrOnes(c.bCols, c.bRows);
        EXPECT_TRUE(TestRotationalEqualHeads(s_handle_, dt, c.name, a, b));
    }
}

// ---- Complex64 vectorized emission (non-trivial values) ----
TEST_F(SpGemmArch22Test, ComplexVecEmit) {
    struct CvCase { int32_t n, d, shift; };
    const CvCase kCv[] = {
        {512, 8, 1},
        {1024, 8, 1},
        {256, 4, 1},
        {256, 7, 1},
        {300, 6, 3},
        {300, 8, 3},
    };
    for (const CvCase &c : kCv) {
        EXPECT_TRUE(TestComplexVecEmit(s_handle_, c.n, c.d, c.shift));
    }
}

// ---- Beta != 0 structural union (fp32 + c64) ----
TEST_F(SpGemmArch22Test, BetaStructuralUnion) {
    EXPECT_TRUE(TestBetaStructuralUnion(s_handle_, ACL_FLOAT));
    EXPECT_TRUE(TestBetaStructuralUnion(s_handle_, ACL_COMPLEX64));
}

TEST_F(SpGemmArch22Test, KZeroWithNonzeroBetaPreservesCIn) {
    EXPECT_TRUE(TestKZeroWithNonzeroBetaPreservesCIn(s_handle_));
}

// ---- Beta with chunked algorithm ----
TEST_F(SpGemmArch22Test, BetaWithChunkedAlg) {
    EXPECT_TRUE(TestBetaWithChunkedAlg(s_handle_, ACL_SPARSE_SPGEMM_ALG2, "alg2"));
    EXPECT_TRUE(TestBetaWithChunkedAlg(s_handle_, ACL_SPARSE_SPGEMM_ALG3, "alg3"));
}

// ---- Repeated Compute with beta ----
TEST_F(SpGemmArch22Test, RepeatedComputeWithBeta) {
    EXPECT_TRUE(TestRepeatedComputeWithBeta(s_handle_));
}

// ---- Wide-N chunked path × 4 dtypes ----
TEST_P(SpGemmArch22DtypeTest, WideChunk) {
    EXPECT_TRUE(TestWideChunk(s_handle_, GetParam().type));
}

// ---- Long-tail rows (T3 path) × 4 dtypes ----
TEST_P(SpGemmArch22DtypeTest, LongTail) {
    EXPECT_TRUE(TestLongTail(s_handle_, GetParam().type));
}

// ---- Determinism (bit-wise) × 4 dtypes ----
TEST_P(SpGemmArch22DtypeTest, Determinism) {
    EXPECT_TRUE(TestDeterminism(s_handle_, GetParam().type));
}

// ---- Alpha/beta device pointer mode ----
TEST_F(SpGemmArch22Test, DevicePointerMode) {
    EXPECT_TRUE(TestDevicePointerMode(s_handle_));
}

// ---- Scalar encoding contract × 4 dtypes ----
TEST_P(SpGemmArch22DtypeTest, ScalarEncoding) {
    EXPECT_TRUE(TestScalarEncoding(s_handle_, GetParam().type));
}

// ---- Invalid arguments ----
TEST_F(SpGemmArch22Test, InvalidArgs) {
    EXPECT_TRUE(TestInvalidArgs(s_handle_));
}

// ---- CSR content validation & input readonly ----
TEST_F(SpGemmArch22Test, CsrContentValidation) {
    EXPECT_TRUE(TestCsrContentValidation(s_handle_));
}

// ---- Beta cross-stage consistency ----
TEST_F(SpGemmArch22Test, BetaConsistency) {
    EXPECT_TRUE(TestBetaConsistency(s_handle_));
}

// ---- Stage state machine ----
TEST_F(SpGemmArch22Test, StageMachine) {
    EXPECT_TRUE(TestStageMachine(s_handle_));
}

// ---- Resource leak check ----
TEST_F(SpGemmArch22Test, NoLeak) {
    EXPECT_TRUE(TestNoLeak(s_handle_, 50));
}

// ---- Official-manifest scale sampling ----
TEST_F(SpGemmArch22Test, ScaleCases) {
    if (std::getenv("SPGEMM_TEST_SKIP_SCALE")) {
        GTEST_SKIP() << "SPGEMM_TEST_SKIP_SCALE is set";
    }
    auto h = s_handle_;
    struct ScaleCase {
        int32_t m, k, n, degA, degB, emptyA, emptyB;
        const char *name;
    };
    const ScaleCase kScale[] = {
        { 1143, 12705,  3272, 1, 1,  0,  0, "scale-1143x12705x3272-d1"},
        { 7794,  6368,  4231, 4, 4,  0,  0, "scale-7794x6368x4231-d4"},
        { 1000,  1000,  1000, 8, 8,  0,  0, "scale-1000cube-d8"},
        {20000,  1000,  1000, 8, 6, 17, 19, "scale-20000x1000x1000-d8/6-empty"},
        { 2000, 20000,  2000, 2, 3,  0,  0, "scale-2000x20000x2000-d2/3"},
        { 4096,  4096, 16384, 3, 6, 11,  0, "scale-4096x4096x16384-d3/6"},
    };
    // Keep degrees, empty rows and rectangular axes at functional-test sizes.
    const ScaleCase kPresmokeScale[] = {
        { 63, 255, 193, 1, 1,  0,  0, "small-63x255x193-d1"},
        {127, 128,  97, 4, 4,  0,  0, "small-127x128x97-d4"},
        {128, 128, 128, 8, 8,  0,  0, "small-128cube-d8"},
        {256,  64,  96, 8, 6, 17, 19, "small-256x64x96-d8/6-empty"},
        { 96, 256,  96, 2, 3,  0,  0, "small-96x256x96-d2/3"},
        {128, 128, 512, 3, 6, 11,  0, "small-128x128x512-d3/6"},
    };
    const auto &cases = SparseTestFullEnabled() ? kScale : kPresmokeScale;
    for (const ScaleCase &c : cases) {
        EXPECT_TRUE(TestScaleCase(h, c.m, c.k, c.n, c.degA, c.degB, c.emptyA, c.emptyB,
                                  ACL_FLOAT, c.name));
        EXPECT_TRUE(TestScaleCase(h, c.m, c.k, c.n, c.degA, c.degB, c.emptyA, c.emptyB,
                                  ACL_COMPLEX64, c.name));
    }
    const int32_t lowPrecisionSize = SparseTestFullEnabled() ? 4096 : 128;
    EXPECT_TRUE(TestScaleCase(h, lowPrecisionSize, lowPrecisionSize, lowPrecisionSize,
                              4, 4, 0, 0, ACL_FLOAT16, "fp16-cube-d4"));
    EXPECT_TRUE(TestScaleCase(h, lowPrecisionSize, lowPrecisionSize, lowPrecisionSize,
                              4, 4, 0, 0, ACL_BF16, "bf16-cube-d4"));
}
