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
#ifndef SPMV_TEST_STRIDES_H
#define SPMV_TEST_STRIDES_H
#include "../../../sparse/spmv/arch22/spmv_stride_plan.h"

// Host-only layout regressions run before ACL initialization, without accessing fake addresses.
static bool TestSpmvStrideLayout() {
    const uint64_t lengths[5] = {38, 185, 185, 71, 37};
    const int64_t strides[5] = {2, 3, 2, 3, 3};
    const uint32_t widths[5] = {4, 4, 2, 2, 2};
    SpmvStridePlan plan;
    if (!spmv_stride::MakePlan(lengths, strides, widths, 256, plan) || !plan.active) return false;
    uint64_t last = 256;
    for (const auto &copy : plan.copies) {
        if (!copy.enabled || copy.offset % 64 != 0 || copy.offset < last) return false;
        last = copy.offset + copy.length * copy.width;
    }
    if (plan.totalBytes < last + 63) return false;
    const int64_t contiguous[5] = {1, 1, 1, 1, 1};
    if (!spmv_stride::MakePlan(lengths, contiguous, widths, 256, plan) || plan.active || plan.totalBytes != 256) return false;
    return true;
}
static bool TestSpmvStrideOverflow() {
    const uint64_t lengths[5] = {38, 185, 185, 71, 37};
    const int64_t strides[5] = {2, 3, 2, 3, 3};
    const uint32_t widths[5] = {4, 4, 2, 2, 2};
    SpmvStridePlan plan;
    uint64_t huge[5] = {1, std::numeric_limits<uint64_t>::max(), 1, 1, 1};
    if (spmv_stride::MakePlan(huge, strides, widths, 256, plan)) return false;
    const auto high = reinterpret_cast<const void *>(std::numeric_limits<uintptr_t>::max() - 1);
    if (spmv_stride::SpanFits(high, 2, 2, 4)) return false;
    if (spmv_stride::SpanFits(reinterpret_cast<void *>(64), 10, std::numeric_limits<int64_t>::max(), 8)) return false;
    // A representable stride above the DMA uint32 gap limit remains supported by the plan.
    const int64_t largeStride[5] = {1, 1, 1, 1, int64_t(0x100000001ULL)};
    if (!spmv_stride::MakePlan(lengths, largeStride, widths, 0, plan) ||
        plan.copies[4].strideBytes <= 0xffffffffULL) return false;
    return true;
}
static bool TestSpmvStrideDimensions() {
    if (!spmv_stride::DimensionsFit(UINT32_MAX, UINT32_MAX, INT32_MAX, 0) ||
        spmv_stride::DimensionsFit(uint64_t(UINT32_MAX) + 1, 1, 0, 0) ||
        spmv_stride::DimensionsFit(1, uint64_t(UINT32_MAX) + 1, 0, 0) ||
        spmv_stride::DimensionsFit(1, 1, INT32_MAX, 1)) return false;
    return true;
}
static bool TestSpmvStrideSingleton() {
    const uint32_t widths[5] = {4, 4, 2, 2, 2};
    SpmvStridePlan plan;
    const uint64_t singleton[5] = {1, 1, 1, 1, 1};
    const int64_t maxStrides[5] = {INT64_MAX, INT64_MAX, INT64_MAX, INT64_MAX, INT64_MAX};
    if (!spmv_stride::MakePlan(singleton, maxStrides, widths, 0, plan)) return false;
    for (const auto &copy : plan.copies) if (copy.strideBytes != copy.width) return false;
    if (!spmv_stride::SpanFits(reinterpret_cast<void *>(64), 1, INT64_MAX, 8)) return false;
    if (!spmv_stride::SpanFits(nullptr, 0, INT64_MAX, 8)) return false;
    return true;
}
static bool TestSpmvStridePlan() {
    return TestSpmvStrideLayout() && TestSpmvStrideOverflow() &&
           TestSpmvStrideDimensions() && TestSpmvStrideSingleton();
}

struct SpmvStrideComplex { float re, im; };
template<typename T> static SpmvStrideComplex StrideComplex(T value) {
    return {static_cast<float>(value), 0};
}
template<> SpmvStrideComplex StrideComplex<Complex64>(Complex64 value) { return {value.re, value.im}; }
template<typename T> static T StrideNative(float real, float /*imag*/ = 0) { return static_cast<T>(real); }
template<> Complex64 StrideNative<Complex64>(float real, float imag) { return {real, imag}; }
template<typename T> static void StrideStore(std::vector<uint8_t> &bytes, size_t offset, const T &value) {
    std::copy_n(reinterpret_cast<const unsigned char *>(&value), sizeof(T), bytes.data() + offset);
}
template<typename T> static void StridePut(std::vector<uint8_t> &bytes, size_t index, const T &value) {
    StrideStore(bytes, index * sizeof(T), value);
}
template<typename T> static T StrideGet(const std::vector<uint8_t> &bytes, size_t offset) {
    T value;
    std::copy_n(bytes.data() + offset, sizeof(T), reinterpret_cast<unsigned char *>(&value));
    return value;
}

// Query only: fake nonnull device pointers are metadata and must never be dereferenced.
static bool QuerySpmvStrideMetadata(aclsparseHandle_t handle, int64_t rows, int64_t cols,
                                    int64_t nnz, int base, bool accepted, bool transpose = false,
                                    aclsparseSpMVAlg_t alg = ACL_SPARSE_SPMV_ALG_DEFAULT) {
    auto *fake = reinterpret_cast<void *>(0x10000);
    aclsparseSpMatDescr_t mat = nullptr;
    aclsparseDnVecDescr_t x = nullptr, y = nullptr;
    bool created = aclsparseCreateCsr(&mat, rows, cols, nnz, fake, fake, fake,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        base ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT) == ACL_SPARSE_STATUS_SUCCESS &&
        aclsparseCreateDnVec(&x, transpose ? rows : cols, fake, ACL_FLOAT) == ACL_SPARSE_STATUS_SUCCESS &&
        aclsparseCreateDnVec(&y, transpose ? cols : rows, fake, ACL_FLOAT) == ACL_SPARSE_STATUS_SUCCESS &&
        aclsparseDnVecSetStride(x, 2) == ACL_SPARSE_STATUS_SUCCESS &&
        aclsparseDnVecSetStride(y, 2) == ACL_SPARSE_STATUS_SUCCESS;
    float alpha = 1, beta = 0;
    size_t bytes = 123;
    auto status = created ? aclsparseSpMVGetBufferSize(handle,
        transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, mat, x, &beta,
        y, ACL_FLOAT, alg, &bytes) : ACL_SPARSE_STATUS_INTERNAL_ERROR;
    if (mat) aclsparseDestroySpMat(mat);
    if (x) aclsparseDestroyDnVec(x);
    if (y) aclsparseDestroyDnVec(y);
    const bool empty = (transpose ? cols : rows) == 0;
    return created && status == (accepted ? ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_INVALID_VALUE) &&
           (!empty || bytes == 0);
}
static bool TestSpmvStrideMetadata(aclsparseHandle_t handle) {
    const int64_t u32 = std::numeric_limits<uint32_t>::max();
    const int64_t i32 = std::numeric_limits<int32_t>::max();
    return QuerySpmvStrideMetadata(handle, u32 + 1, 1, 0, 0, false) &&
        QuerySpmvStrideMetadata(handle, 1, u32 + 1, 0, 0, false) &&
        QuerySpmvStrideMetadata(handle, 1, 1, i32 + 1, 0, false) &&
        QuerySpmvStrideMetadata(handle, 1, 1, i32, 1, false) &&
        QuerySpmvStrideMetadata(handle, 1, 1, i32, 0, true) &&
        QuerySpmvStrideMetadata(handle, 1, 1, i32 - 1, 1, true) &&
        QuerySpmvStrideMetadata(handle, u32, 1, 0, 0, true) &&
        QuerySpmvStrideMetadata(handle, 1, u32, 0, 0, true) &&
        QuerySpmvStrideMetadata(handle, 0, u32 + 1, i32 + 1, 0, true) &&
        QuerySpmvStrideMetadata(handle, u32 + 1, 0, i32 + 1, 0, true, true) &&
        QuerySpmvStrideMetadata(handle, 0, u32 + 1, 0, 0, true, false, ACL_SPARSE_SPMV_CSR_ALG_CUBE);
}

// New fixture owns native byte buffers, allowing guards, offsets, and const input uploads.
// Existing non-stride fixtures and their complete numerical regression suite stay intact.
template<typename ValT, typename OutT, typename ComputeT>
class SpmvStrideFixture {
public:
    explicit SpmvStrideFixture(aclrtStream stream, size_t workspaceOffset = 1)
        : workspaceOffset_(workspaceOffset), stream_(stream) {}
    ~SpmvStrideFixture() {
        if (mat_) aclsparseDestroySpMat(mat_);
        if (xd_) aclsparseDestroyDnVec(xd_);
        if (yd_) aclsparseDestroyDnVec(yd_);
        if (handle_) aclsparseDestroy(handle_);
        for (auto p : device_) if (p) aclrtFree(p);
        if (workspace_) aclrtFree(workspace_);
    }
    bool Run(const int64_t requested[5], bool transpose, int indexBase, int betaInput, size_t yOffset,
         aclsparseSpMVAlg_t algorithm = ACL_SPARSE_SPMV_ALG_DEFAULT,
         uint32_t rows = 37, uint32_t columns = 71, uint32_t perRow = 5) {
        std::copy(requested, requested + 5, strides);
        m = rows; n = columns; k = perRow; trans = transpose; base = indexBase;
        beta = betaInput; alg = algorithm; prefix = 64 + yOffset;
        nx = trans ? m : n; ny = trans ? n : m; nnz = uint64_t(m) * k;
        op = trans ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
        betaValue = StrideNative<ComputeT>(beta);
        if (!CreateInputs() || !CreateDescriptors() || !CreateWorkspace()) return false;
        for (int generation = 0; generation < 2; ++generation) {
            FillInputs(generation);
            FillExpected();
            if (!ExecuteAndVerify()) return false;
        }
        return VerifyWorkspace();
    }
    bool CheckWorkspaceBounds() {
        if (!TestSpmvStrideMetadata(handle_)) return false;
        if (bytes <= 1 || aclsparseSetWorkspace(handle_, external, bytes - 1) != ACL_SPARSE_STATUS_SUCCESS) return false;
        auto status = aclsparseSpMV(handle_, op, &alpha, mat_, xd_, &betaValue, yd_,
                                   AclTypeOf<ComputeT>(), alg, nullptr);
        if (aclsparseSetWorkspace(handle_, nullptr, 0) != ACL_SPARSE_STATUS_SUCCESS ||
            status != ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES) return false;
        auto nearLimit = reinterpret_cast<void *>(std::numeric_limits<uintptr_t>::max() - 1);
        return aclsparseSpMV(handle_, op, &alpha, mat_, xd_, &betaValue, yd_,
                            AclTypeOf<ComputeT>(), alg, nearLimit) == ACL_SPARSE_STATUS_INVALID_VALUE;
    }
private:
    bool CreateInputs() {
        const size_t widths[5] = {4, 4, sizeof(ValT), sizeof(ValT), sizeof(OutT)};
        const uint64_t lengths[5] = {m + 1U, nnz, nnz, nx, ny};
        for (unsigned i = 0; i < 5; ++i) {
            size_t bytes = (lengths[i] == 0 ? 0 : (lengths[i] - 1) * strides[i] + 1) * widths[i];
            host_[i].assign(bytes + 64 + (i == 4 ? prefix : 0), 0xa5);
            // const void* accepts native const-vector data without casting away const.
            const std::vector<uint8_t> input = host_[i];
            if (CreateDeviceTensor(input.data(), input.size(), &device_[i]) != ACL_SUCCESS) return false;
        }
        return true;
    }
    bool CreateDescriptors() {
        if (aclsparseCreate(&handle_) != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseSetStream(handle_, stream_) != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseCreateCsr(&mat_, m, n, nnz, device_[0], device_[1], device_[2],
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            base ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO, AclTypeOf<ValT>()) != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseCsrSetStrides(mat_, strides[0], strides[1], strides[2]) != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseCreateDnVec(&xd_, nx, device_[3], AclTypeOf<ValT>()) != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseDnVecSetStride(xd_, strides[3]) != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseCreateDnVec(&yd_, ny, device_[4] + prefix, AclTypeOf<OutT>()) != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseDnVecSetStride(yd_, strides[4]) != ACL_SPARSE_STATUS_SUCCESS) return false;
        return true;
    }
    bool CreateWorkspace() {
        if (aclsparseSpMVGetBufferSize(handle_, op, &alpha, mat_, xd_, &betaValue, yd_,
            AclTypeOf<ComputeT>(), alg, &bytes) != ACL_SPARSE_STATUS_SUCCESS) return false;
        // Caller buffer intentionally starts at byte offset 1; query includes alignment slack.
        if (aclrtMalloc(reinterpret_cast<void **>(&workspace_), bytes + 128, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return false;
        std::vector<uint8_t> workspaceInitial(bytes + 128, 0x5a);
        if (aclrtMemcpy(workspace_, workspaceInitial.size(), workspaceInitial.data(), workspaceInitial.size(), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) return false;
        external = bytes ? workspace_ + workspaceOffset_ : nullptr;
        // Strided CUBE must not attempt a D2H row probe here: row data is still canary.
        if (aclsparseSpMVPreprocess(handle_, op, &alpha, mat_, xd_, &betaValue, yd_,
            AclTypeOf<ComputeT>(), alg, external) != ACL_SPARSE_STATUS_SUCCESS) return false;
        return true;
    }
    void FillInputs(int generation) {
        // Mutate all five inputs in place after Preprocess, without changing their pointers.
        gold.assign(ny, {0, 0});
        for (uint32_t r = 0; r <= m; ++r) {
            const uint32_t index = r * k - (generation && r > 0 && r < m ? 1U : 0U);
            StridePut<int32_t>(host_[0], r * strides[0], index + base);
        }
        for (uint64_t i = 0; i < nx; ++i) {
            ValT value = StrideNative<ValT>(static_cast<int>((i + generation) % 7) - 3,
                                            static_cast<int>(i % 3) - 1);
            StridePut<ValT>(host_[3], i * strides[3], value);
        }
        for (uint32_t r = 0; r < m; ++r) {
            const uint32_t begin = r * k - (generation && r > 0 ? 1U : 0U);
            const uint32_t end = (r + 1) * k - (generation && r + 1 < m ? 1U : 0U);
            for (uint32_t j = begin; j < end; ++j) {
                const uint32_t col = (r * 7 + j + generation * 3) % n;
                ValT value = StrideNative<ValT>(static_cast<int>((j + generation) % 5) - 2,
                                                static_cast<int>(j % 3) - 1);
                StridePut<int32_t>(host_[1], j * strides[1], col + base);
                StridePut<ValT>(host_[2], j * strides[2], value);
                auto a = StrideComplex(value);
                auto x = StrideComplex(StrideGet<ValT>(host_[3], (trans ? r : col) * strides[3] * sizeof(ValT)));
                auto &sum = gold[trans ? col : r];
                sum.re += a.re * x.re - a.im * x.im;
                sum.im += a.re * x.im + a.im * x.re;
            }
        }
    }
    void FillExpected() {
        expected = host_[4];
        for (uint64_t i = 0; i < ny; ++i) {
            OutT original;
            // Avoid a NaN-to-integer conversion even when beta=0.
            if constexpr (std::is_integral<OutT>::value) original = beta ? OutT(1) : OutT(17);
            else original = beta ? StrideNative<OutT>(1, -1) :
                StrideNative<OutT>(std::numeric_limits<float>::quiet_NaN());
            StrideStore(host_[4], prefix + i * strides[4] * sizeof(OutT), original);
            auto initial = StrideComplex(original);
            OutT result = StrideNative<OutT>(gold[i].re + (beta ? initial.re : 0),
                                             gold[i].im + (beta ? initial.im : 0));
            StrideStore(expected, prefix + i * strides[4] * sizeof(OutT), result);
        }
    }
    bool ExecuteAndVerify() {
        for (unsigned i = 0; i < 5; ++i)
            if (aclrtMemcpy(device_[i], host_[i].size(), host_[i].data(), host_[i].size(), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) return false;
        if (aclsparseSpMV(handle_, op, &alpha, mat_, xd_, &betaValue, yd_, AclTypeOf<ComputeT>(), alg, external) != ACL_SPARSE_STATUS_SUCCESS ||
            aclrtSynchronizeStream(stream_) != ACL_SUCCESS) return false;
        std::vector<uint8_t> actual(host_[4].size());
        if (aclrtMemcpy(actual.data(), actual.size(), device_[4], actual.size(), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) return false;
        std::vector<uint8_t> active(actual.size(), 0);
        for (uint64_t i = 0; i < ny; ++i) {
            const size_t offset = prefix + i * strides[4] * sizeof(OutT);
            auto result = StrideComplex(StrideGet<OutT>(actual, offset));
            auto wanted = StrideComplex(StrideGet<OutT>(expected, offset));
            if (result.re != wanted.re || result.im != wanted.im) return false;
            for (size_t b = 0; b < sizeof(OutT); ++b) active[offset + b] = 1;
        }
        for (size_t b = 0; b < actual.size(); ++b)
            if (!active[b] && actual[b] != expected[b]) return false;
        return true;
    }
    bool VerifyWorkspace() {
        std::vector<uint8_t> workspaceActual(bytes + 128);
        if (aclrtMemcpy(workspaceActual.data(), workspaceActual.size(), workspace_, workspaceActual.size(), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) return false;
        for (size_t i = 0; i < workspaceOffset_; ++i) if (workspaceActual[i] != 0x5a) return false;
        for (size_t i = bytes + workspaceOffset_; i < workspaceActual.size(); ++i) if (workspaceActual[i] != 0x5a) return false;
        return true;
    }

    int64_t strides[5] = {1, 1, 1, 1, 1};
    uint32_t m = 0, n = 0, k = 0;
    uint64_t nx = 0, ny = 0, nnz = 0;
    int base = 0, beta = 0;
    bool trans = false;
    size_t prefix = 0, bytes = 0;
    aclsparseOperation_t op = ACL_SPARSE_OP_NON_TRANSPOSE;
    aclsparseSpMVAlg_t alg = ACL_SPARSE_SPMV_ALG_DEFAULT;
    ComputeT alpha = StrideNative<ComputeT>(1), betaValue = StrideNative<ComputeT>(0);
    void *external = nullptr;
    std::vector<SpmvStrideComplex> gold;
    std::vector<uint8_t> expected;
    size_t workspaceOffset_ = 1;
    aclrtStream stream_;
    aclsparseHandle_t handle_ = nullptr;
    aclsparseSpMatDescr_t mat_ = nullptr;
    aclsparseDnVecDescr_t xd_ = nullptr, yd_ = nullptr;
    uint8_t *device_[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    uint8_t *workspace_ = nullptr;
    std::vector<uint8_t> host_[5];
};

template<typename ValT, typename OutT, typename ComputeT>
static void RunSpmvStrideType(TestStats &stats, aclrtStream stream, const char *name) {
    const int64_t mixed[5] = {2, 3, 2, 3, 3};
    for (bool trans : {false, true}) for (int beta : {0, 1}) {
        SpmvStrideFixture<ValT, OutT, ComputeT> fixture(stream);
        bool pass = fixture.Run(mixed, trans, 1, beta, sizeof(OutT));
        ++stats.total;
        if (pass) ++stats.passed;
        else { ++stats.failed; stats.failedCases.push_back(std::string("stride-") + name + (trans ? "-T" : "-N") + "-beta" + std::to_string(beta)); }
    }
}
static void RunSpmvStrideRegression(TestStats &stats, aclrtStream stream) {
    RunSpmvStrideType<float, float, float>(stats, stream, "f32");
    RunSpmvStrideType<half, half, float>(stats, stream, "f16");
    RunSpmvStrideType<half, float, float>(stats, stream, "f16-f32");
    RunSpmvStrideType<bfloat16_t, bfloat16_t, float>(stats, stream, "bf16");
    RunSpmvStrideType<bfloat16_t, float, float>(stats, stream, "bf16-f32");
    RunSpmvStrideType<int32_t, int32_t, int32_t>(stats, stream, "i32");
    RunSpmvStrideType<int32_t, float, float>(stats, stream, "i32-f32");
    RunSpmvStrideType<int8_t, int32_t, int32_t>(stats, stream, "i8-i32");
    RunSpmvStrideType<int8_t, float, float>(stats, stream, "i8-f32");
    RunSpmvStrideType<Complex64, Complex64, Complex64>(stats, stream, "c64");
    RunSpmvStrideType<float, Complex64, Complex64>(stats, stream, "f32-c64");
    auto track = [&](bool pass, const std::string &name) {
        ++stats.total;
        if (pass) ++stats.passed;
        else { ++stats.failed; stats.failedCases.push_back(name); }
    };
    for (unsigned field = 0; field < 5; ++field) for (bool trans : {false, true}) {
        int64_t single[5] = {1, 1, 1, 1, 1}; single[field] = 2;
        SpmvStrideFixture<float, float, float> fixture(stream);
        track(fixture.Run(single, trans, 0, 1, sizeof(float), field % 2 ? ACL_SPARSE_SPMV_CSR_ALG1 : ACL_SPARSE_SPMV_CSR_ALG2),
              "stride-single-" + std::to_string(field) + (trans ? "-T" : "-N"));
    }
    const int64_t yOnly[5] = {1, 1, 1, 1, 2};
    { SpmvStrideFixture<half, half, float> fixture(stream, 63);
      track(fixture.Run(yOnly, true, 0, 1, 1), "stride-unaligned-half-y"); }
    { SpmvStrideFixture<float, float, float> fixture(stream);
      track(fixture.Run(yOnly, false, 0, 0, 1), "stride-unaligned-float-y-beta0"); }
    const int64_t singleton[5] = {1, INT64_MAX, INT64_MAX, INT64_MAX, INT64_MAX};
    { SpmvStrideFixture<float, float, float> fixture(stream);
      track(fixture.Run(singleton, false, 0, 1, 4, ACL_SPARSE_SPMV_ALG_DEFAULT, 1, 1, 1), "stride-singleton-int64max"); }
    const int64_t mixed[5] = {2, 3, 2, 3, 2};
    { SpmvStrideFixture<float, float, float> fixture(stream);
      track(fixture.Run(mixed, false, 0, 1, 0, ACL_SPARSE_SPMV_CSR_ALG_CUBE), "stride-cube-vector-fallback"); }
    { SpmvStrideFixture<half, float, float> fixture(stream);
      track(fixture.Run(mixed, false, 0, 0, 0, ACL_SPARSE_SPMV_CSR_ALG_CUBE), "stride-cube-half-vector-fallback"); }
    { SpmvStrideFixture<float, float, float> fixture(stream);
      track(fixture.Run(yOnly, true, 0, 1, 0, ACL_SPARSE_SPMV_ALG_DEFAULT, 40, 100000, 8) && fixture.CheckWorkspaceBounds(), "stride-wide-transpose-workspace-bounds"); }
}
#endif
