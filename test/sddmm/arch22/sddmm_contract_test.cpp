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
#include "cann_ops_sparse.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr auto ok = ACL_SPARSE_STATUS_SUCCESS;
constexpr auto invalid = ACL_SPARSE_STATUS_INVALID_VALUE;
constexpr auto nn = ACL_SPARSE_OP_NON_TRANSPOSE;
constexpr auto tt = ACL_SPARSE_OP_TRANSPOSE;
constexpr auto row = ACL_SPARSE_ORDER_ROW;
constexpr auto col = ACL_SPARSE_ORDER_COL;
constexpr auto alg = ACL_SPARSE_SDDMM_ALG_DEFAULT;
void Check(int status)
{
    if (status) {
        throw std::runtime_error("ACL status " + std::to_string(status));
    }
}

struct Device {
    void* ptr = nullptr;
    size_t bytes = 0;
    explicit Device(size_t size) : bytes(size)
    {
        Check(aclrtMalloc(&ptr, bytes + 64, ACL_MEM_MALLOC_NORMAL_ONLY));
        Check(aclrtMemset(ptr, bytes + 64, 0x5a, bytes + 64));
    }
    ~Device()
    {
        if (ptr) {
            (void)aclrtFree(ptr);
        }
    }
    Device(const Device&) = delete;
    void Put(const void* src, size_t size)
    {
        if (size) {
            Check(aclrtMemcpy(ptr, bytes, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
        }
    }
    template <typename T>
    void Put(const std::vector<T>& src)
    {
        Put(src.data(), src.size() * sizeof(T));
    }
    template <typename T>
    std::vector<T> Get(size_t count) const
    {
        std::vector<T> out(count);
        if (count) {
            Check(aclrtMemcpy(out.data(), count * sizeof(T), ptr, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST));
        }
        return out;
    }
    void Guard() const
    {
        std::vector<unsigned char> tail(64);
        Check(aclrtMemcpy(tail.data(), 64, static_cast<char*>(ptr) + bytes, 64, ACL_MEMCPY_DEVICE_TO_HOST));
        EXPECT_TRUE(std::all_of(tail.begin(), tail.end(), [](auto x) { return x == 0x5a; }));
    }
};

template <typename T>
T Value(int i)
{
    return T((i % 13 - 6) / 16.0);
}
template <>
std::complex<float> Value(int i)
{
    return {(i % 13 - 6) / 16.f, (i % 7 - 3) / 32.f};
}
template <typename T>
struct Problem {
    int bd, m, n, k, batches, xb, yb, rs, cs, vs, xld, yld, xs, ys;
    aclsparseOperation_t ox, oy;
    aclsparseOrder_t xo, yo, bo;
    aclDataType type = sizeof(T) == 8 ? ACL_COMPLEX64 : ACL_FLOAT;
    aclsparseHandle_t h = nullptr;
    aclrtStream stream = nullptr, second = nullptr;
    aclsparseDnMatDescr_t dx = nullptr, dy = nullptr;
    aclsparseSpMatDescr_t dc = nullptr;
    std::vector<int32_t> rp, ci;
    std::vector<T> x, y, c;
    std::unique_ptr<Device> dr, di, dv, dxx, dyy, work;
    T alpha = Value<T>(18), beta = Value<T>(16);
    size_t workspaceBytes = 0;
    int Index(int r, int c, int ld, aclsparseOrder_t order) { return order == row ? r * ld + c : c * ld + r; }
    int XIndex(int b, int r, int t)
    {
        return (xb == 1 ? 0 : b * xs) + Index(ox == nn ? r : t, ox == nn ? t : r, xld, xo);
    }
    int YIndex(int b, int t, int c)
    {
        return (yb == 1 ? 0 : b * ys) + Index(oy == nn ? t : c, oy == nn ? c : t, yld, yo);
    }
    Problem(int block, int base, aclsparseOrder_t blocks, int reduction = 7, int batch = 1, int xBatch = 1,
            int yBatch = 1, aclsparseOperation_t opx = nn, aclsparseOperation_t opy = nn, aclsparseOrder_t orderx = row,
            aclsparseOrder_t ordery = row, bool padded = false)
        : bd(block),
          m(3 * block),
          n(4 * block),
          k(reduction),
          batches(batch),
          xb(xBatch),
          yb(yBatch),
          rs(8),
          cs(8),
          vs(4 * block * block + 16),
          ox(opx),
          oy(opy),
          xo(orderx),
          yo(ordery),
          bo(blocks)
    {
        const int xr = ox == nn ? m : k, xc = ox == nn ? k : m;
        const int yr = oy == nn ? k : n, yc = oy == nn ? n : k;
        xld = (xo == row ? xc : xr) + (padded ? 3 : 0);
        yld = (yo == row ? yc : yr) + (padded ? 5 : 0);
        xs = (xo == row ? xr : xc) * xld + 16;
        ys = (yo == row ? yr : yc) * yld + 16;
        InitValues(base);
        InitDeviceValues();
        Check(aclrtCreateStream(&stream));
        Check(aclrtCreateStream(&second));
        Check(aclsparseCreate(&h));
        Check(aclsparseSetStream(h, stream));
        Check(aclsparseCreateDnMat(&dx, xr, xc, xld, dxx->ptr, type, xo));
        Check(aclsparseCreateDnMat(&dy, yr, yc, yld, dyy->ptr, type, yo));
        Check(aclsparseDnMatSetStridedBatch(dx, xb, xs));
        Check(aclsparseDnMatSetStridedBatch(dy, yb, ys));
        if (bd == 1) {
            Check(aclsparseCreateCsr(&dc, m, n, 4, dr->ptr, di->ptr, dv->ptr, ACL_SPARSE_INDEX_32I,
                                     ACL_SPARSE_INDEX_32I,
                                     base ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO, type));
        } else {
            Check(aclsparseCreateBsr(&dc, 3, 4, 4, bd, bd, dr->ptr, di->ptr, dv->ptr, ACL_SPARSE_INDEX_32I,
                                     ACL_SPARSE_INDEX_32I,
                                     base ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO, type, bo));
            Check(aclsparseBsrSetStridedBatch(dc, batch, rs, cs, vs));
        }
        Check(aclsparseSDDMMBufferSize(h, ox, oy, &alpha, dx, dy, &beta, dc, type, alg, &workspaceBytes));
        work.reset(new Device(workspaceBytes));
    }
    void InitDeviceValues()
    {
        dr.reset(new Device(rp.size() * 4));
        di.reset(new Device(ci.size() * 4));
        dv.reset(new Device(c.size() * sizeof(T)));
        dxx.reset(new Device(x.size() * sizeof(T)));
        dyy.reset(new Device(y.size() * sizeof(T)));
        dr->Put(rp);
        di->Put(ci);
        dv->Put(c);
        dxx->Put(x);
        dyy->Put(y);
    }

    void InitValues(int base)
    {
        rp.resize(rs * batches, 91);
        ci.resize(cs * batches, 93);
        x.resize(xs * xb, T(7));
        y.resize(ys * yb, T(9));
        c.resize(vs * batches, T(11));
        for (int b = 0; b < batches; ++b) {
            for (int i = 0; i < 4; ++i) {
                rp[b * rs + i] = base + (i == 0 ? 0 : i == 1 ? 2 : i == 2 ? 2 : 4);
            }
            for (int i = 0; i < 4; ++i) {
                ci[b * cs + i] = base + (i + b) % 4;
            }
            for (int i = 0; i < 4 * bd * bd; ++i) {
                c[b * vs + i] = Value<T>(i + b);
            }
        }
        for (int b = 0; b < xb; ++b) {
            for (int r = 0; r < m; ++r) {
                for (int t = 0; t < k; ++t) {
                    x[XIndex(b, r, t)] = Value<T>(b * 3 + r + t);
                }
            }
        }
        for (int b = 0; b < yb; ++b) {
            for (int t = 0; t < k; ++t) {
                for (int q = 0; q < n; ++q) {
                    y[YIndex(b, t, q)] = Value<T>(b * 7 + 2 * t + q);
                }
            }
        }
    }
    ~Problem()
    {
        (void)aclrtSynchronizeStream(stream);
        (void)aclrtSynchronizeStream(second);
        if (dc) {
            (void)aclsparseDestroySpMat(dc);
        }
        if (dx) {
            (void)aclsparseDestroyDnMat(dx);
        }
        if (dy) {
            (void)aclsparseDestroyDnMat(dy);
        }
        if (h) {
            (void)aclsparseDestroy(h);
        }
        (void)aclrtDestroyStream(stream);
        (void)aclrtDestroyStream(second);
    }
    auto Prepare(void* buffer = nullptr)
    {
        return aclsparseSDDMMPreprocess(h, ox, oy, &alpha, dx, dy, &beta, dc, type, alg, buffer ? buffer : work->ptr);
    }
    auto Execute(void* buffer = nullptr, const void* a = nullptr, const void* b = nullptr)
    {
        return aclsparseSDDMM(h, ox, oy, a ? a : &alpha, dx, dy, b ? b : &beta, dc, type, alg,
                              buffer ? buffer : work->ptr);
    }
    using Wide = std::conditional_t<sizeof(T) == 8, std::complex<double>, double>;
    Wide ComputeDot(int batch, int matrixRow, int matrixColumn)
    {
        Wide dot = 0;
        if (alpha == T(0)) {
            return dot;
        }
        for (int t = 0; t < k; ++t) {
            dot += Wide(x[XIndex(batch, matrixRow, t)]) * Wide(y[YIndex(batch, t, matrixColumn)]);
        }
        return dot;
    }
    void VerifyBlock(const std::vector<T>& actual, int batch, int blockRow, int blockIndex)
    {
        const int blockColumn = ci[batch * cs + blockIndex] - rp[0];
        for (int i = 0; i < bd; ++i) {
            for (int q = 0; q < bd; ++q) {
                const Wide dot = ComputeDot(batch, blockRow * bd + i, blockColumn * bd + q);
                const int idx = batch * vs + blockIndex * bd * bd + (bo == row ? i * bd + q : q * bd + i);
                const Wide expected = Wide(alpha) * dot + (beta == T(0) ? Wide(0) : Wide(beta) * Wide(c[idx]));
                ASSERT_LE(std::abs(Wide(actual[idx]) - expected), 2e-5 + .001 * std::abs(expected))
                    << "batch=" << batch << " index=" << idx;
            }
        }
    }
    void VerifyBatch(const std::vector<T>& actual, int batch)
    {
        const int base = rp[0];
        for (int r = 0; r < 3; ++r) {
            for (int j = rp[batch * rs + r] - base; j < rp[batch * rs + r + 1] - base; ++j) {
                ASSERT_NO_FATAL_FAILURE(VerifyBlock(actual, batch, r, j));
            }
        }
        for (int i = 4 * bd * bd; i < vs; ++i) {
            EXPECT_EQ(actual[batch * vs + i], c[batch * vs + i]);
        }
    }
    void Verify()
    {
        Check(aclrtSynchronizeStream(stream));
        Check(aclrtSynchronizeStream(second));
        auto actual = dv->Get<T>(c.size());
        for (int b = 0; b < batches; ++b) {
            ASSERT_NO_FATAL_FAILURE(VerifyBatch(actual, b));
        }
        EXPECT_EQ(dr->Get<int32_t>(rp.size()), rp);
        EXPECT_EQ(di->Get<int32_t>(ci.size()), ci);
        EXPECT_EQ(dxx->Get<T>(x.size()), x);
        EXPECT_EQ(dyy->Get<T>(y.size()), y);
        work->Guard();
        dv->Guard();
        dr->Guard();
        di->Guard();
    }
};

class SddmmContract : public testing::Test {
public:
    static void SetUpTestSuite()
    {
        Check(aclInit(nullptr));
        Check(aclrtSetDevice(0));
    }
    static void TearDownTestSuite()
    {
        (void)aclrtResetDevice(0);
        (void)aclFinalize();
    }
};

TEST_F(SddmmContract, BsrBlocksBasesAndOrders)
{
    for (int bd : {2, 4, 8, 16, 32, 64, 128}) {
        for (int base : {0, 1}) {
            for (auto order : {row, col}) {
                SCOPED_TRACE("bd=" + std::to_string(bd) + " base=" + std::to_string(base));
                Problem<float> p(bd, base, order);
                ASSERT_EQ(p.Prepare(), ok);
                ASSERT_EQ(p.Execute(), ok);
                p.Verify();
            }
        }
    }
}
TEST_F(SddmmContract, FourBatchModesWithIndependentPatterns)
{
    for (int bd : {2, 16}) {
        for (int xb : {1, 3}) {
            for (int yb : {1, 3}) {
                Problem<float> p(bd, 1, col, 7, 3, xb, yb, tt, nn, row, col, true);
                ASSERT_EQ(p.Prepare(), ok);
                ASSERT_EQ(p.Execute(), ok);
                p.Verify();
            }
        }
    }
}
TEST_F(SddmmContract, ComplexLayoutsTransposeAndPaddedLd)
{
    for (int bd : {1, 2, 16}) {
        for (auto ox : {nn, tt}) {
            for (auto oy : {nn, tt}) {
                for (auto xo : {row, col}) {
                    for (auto yo : {row, col}) {
                        SCOPED_TRACE("bd=" + std::to_string(bd) + " ops=" + std::to_string(ox) + std::to_string(oy));
                        Problem<std::complex<float>> p(bd, 1, col, 9, 1, 1, 1, ox, oy, xo, yo, true);
                        ASSERT_EQ(p.Prepare(), ok);
                        ASSERT_EQ(p.Execute(), ok);
                        p.Verify();
                    }
                }
            }
        }
    }
}
TEST_F(SddmmContract, LifecycleNumericUpdatesAndStreams)
{
    Problem<float> p(2, 0, row), other(16, 1, col);
    EXPECT_EQ(p.Execute(), invalid);
    EXPECT_EQ(p.Prepare(static_cast<char*>(p.work->ptr) + 1), invalid);
    ASSERT_EQ(p.Prepare(), ok);
    ASSERT_EQ(other.Prepare(), ok);
    for (auto& v : p.x) {
        v += .125f;
    }
    p.dxx->Put(p.x);
    ASSERT_EQ(aclsparseSetStream(p.h, p.second), ok);
    ASSERT_EQ(p.Execute(), ok);
    ASSERT_EQ(other.Execute(), ok);
    p.Verify();
    other.Verify();
    p.dv->Put(p.c);
    Device replacement(p.x.size() * sizeof(float));
    replacement.Put(p.x);
    ASSERT_EQ(aclsparseDnMatSetValues(p.dx, replacement.ptr), ok);
    ASSERT_EQ(p.Execute(), ok);
    p.Verify();
    EXPECT_EQ(p.Execute(other.work->ptr), invalid);
    EXPECT_EQ(p.Execute(), invalid);
    ASSERT_EQ(p.Prepare(), ok);
    p.ci[1] = 3;
    p.di->Put(p.ci);
    EXPECT_EQ(p.Execute(), invalid);
    ASSERT_EQ(p.Prepare(), ok);
    p.dv->Put(p.c);
    ASSERT_EQ(p.Execute(), ok);
    p.Verify();
    p.rp[1] = 1;
    p.dr->Put(p.rp);
    EXPECT_EQ(p.Execute(), invalid);
    ASSERT_EQ(p.Prepare(), ok);
    p.dv->Put(p.c);
    ASSERT_EQ(p.Execute(), ok);
    p.Verify();
}
TEST_F(SddmmContract, InvalidPatternAndDeviceScalars)
{
    Problem<std::complex<float>> p(2, 1, row);
    for (int bad : {-1, 0, 5, INT32_MIN, INT32_MAX}) {
        p.ci[1] = bad;
        p.di->Put(p.ci);
        EXPECT_EQ(p.Prepare(), invalid) << bad;
    }
    p.ci[1] = 2;
    p.di->Put(p.ci);
    for (int bad : {-1, 0, 6, INT32_MIN, INT32_MAX}) {
        p.rp[1] = bad;
        p.dr->Put(p.rp);
        EXPECT_EQ(p.Prepare(), invalid) << bad;
    }
    p.rp[1] = 3;
    p.dr->Put(p.rp);
    ASSERT_EQ(p.Prepare(), ok);
    Device a(sizeof(p.alpha)), b(sizeof(p.beta));
    a.Put(&p.alpha, sizeof(p.alpha));
    b.Put(&p.beta, sizeof(p.beta));
    ASSERT_EQ(aclsparseSetPointerMode(p.h, ACL_SPARSE_POINTER_MODE_DEVICE), ok);
    ASSERT_EQ(p.Execute(nullptr, a.ptr, b.ptr), ok);
    p.Verify();
}
TEST_F(SddmmContract, DeviceScalarUpdatesAtFixedAddresses)
{
    for (int bd : {1, 2, 16}) {
        Problem<float> p(bd, 0, row);
        ASSERT_EQ(p.Prepare(), ok);
        Device a(sizeof(float)), b(sizeof(float));
        ASSERT_EQ(aclsparseSetPointerMode(p.h, ACL_SPARSE_POINTER_MODE_DEVICE), ok);
        for (float alpha : {1.0f, 0.25f, 0.0f, -0.5f}) {
            p.alpha = alpha;
            p.beta = alpha == 0.0f ? 0.5f : -0.25f;
            a.Put(&p.alpha, sizeof(float));
            b.Put(&p.beta, sizeof(float));
            p.dv->Put(p.c);
            ASSERT_EQ(p.Execute(nullptr, a.ptr, b.ptr), ok);
            p.Verify();
        }
    }
}
TEST_F(SddmmContract, CsrWithoutRowHintsChecksPatternBeforeWriting)
{
    Problem<float> p(1, 0, row);
    ASSERT_EQ(p.Prepare(), ok);
    ASSERT_EQ(p.Execute(), ok);
    p.Verify();
    p.dv->Put(p.c);
    p.ci[1] = 0;
    p.di->Put(p.ci);
    EXPECT_EQ(p.Execute(), invalid);
    Check(aclrtSynchronizeStream(p.stream));
    EXPECT_EQ(p.dv->Get<float>(p.c.size()), p.c);
}
TEST_F(SddmmContract, EmptyAndZeroDimensions)
{
    for (int bd : {1, 2, 128}) {
        for (int base : {0, 1}) {
            for (int rows : {0, 3}) {
                Device offsets((rows + 1) * 4), dense(128 * 8);
                std::vector<int32_t> rp(rows + 1, base);
                offsets.Put(rp);
                aclsparseHandle_t h = nullptr;
                aclrtStream stream = nullptr;
                aclsparseDnMatDescr_t dx = nullptr, dy = nullptr;
                aclsparseSpMatDescr_t dc = nullptr;
                Check(aclrtCreateStream(&stream));
                Check(aclsparseCreate(&h));
                Check(aclsparseSetStream(h, stream));
                Check(aclsparseCreateDnMat(&dx, rows * bd, 0, 0, nullptr, ACL_FLOAT, row));
                Check(aclsparseCreateDnMat(&dy, 0, bd, bd, nullptr, ACL_FLOAT, row));
                auto indexBase = base ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;
                if (bd == 1) {
                    Check(aclsparseCreateCsr(&dc, rows, 1, 0, offsets.ptr, nullptr, nullptr, ACL_SPARSE_INDEX_32I,
                                             ACL_SPARSE_INDEX_32I, indexBase, ACL_FLOAT));
                } else {
                    Check(aclsparseCreateBsr(&dc, rows, 1, 0, bd, bd, offsets.ptr, nullptr, nullptr,
                                             ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, indexBase, ACL_FLOAT, col));
                }
                float a = 1, b = 0;
                size_t size = 0;
                EXPECT_EQ(aclsparseSDDMMBufferSize(h, nn, nn, &a, dx, dy, &b, dc, ACL_FLOAT, alg, &size), ok);
                Device work(size);
                EXPECT_EQ(aclsparseSDDMMPreprocess(h, nn, nn, &a, dx, dy, &b, dc, ACL_FLOAT, alg, work.ptr), ok);
                EXPECT_EQ(aclsparseSDDMM(h, nn, nn, &a, dx, dy, &b, dc, ACL_FLOAT, alg, work.ptr), ok);
                Check(aclrtSynchronizeStream(stream));
                work.Guard();
                offsets.Guard();
                Check(aclsparseDestroySpMat(dc));
                Check(aclsparseDestroyDnMat(dx));
                Check(aclsparseDestroyDnMat(dy));
                Check(aclsparseDestroy(h));
                Check(aclrtDestroyStream(stream));
            }
        }
    }
}
TEST_F(SddmmContract, PatternPointerAndValuePointerChanges)
{
    Problem<float> p(1, 0, row);
    ASSERT_EQ(p.Prepare(), ok);
    Device rows(p.rp.size() * 4), columns(p.ci.size() * 4), values(p.c.size() * 4);
    rows.Put(p.rp);
    columns.Put(p.ci);
    values.Put(p.c);
    ASSERT_EQ(aclsparseCsrSetPointers(p.dc, rows.ptr, columns.ptr, p.dv->ptr), ok);
    EXPECT_EQ(p.Execute(), invalid);
    ASSERT_EQ(p.Prepare(), ok);
    ASSERT_EQ(aclsparseSpMatSetValues(p.dc, values.ptr), ok);
    ASSERT_EQ(p.Execute(), ok);
    Check(aclrtSynchronizeStream(p.stream));
    ASSERT_EQ(p.Execute(), ok);
    Check(aclrtSynchronizeStream(p.stream));
    EXPECT_NE(values.Get<float>(p.c.size()), p.c);
    EXPECT_EQ(p.dv->Get<float>(p.c.size()), p.c);
    p.work->Guard();
    values.Guard();
}
TEST_F(SddmmContract, ValuesOnlyCsrRebindKeepsPreparedPattern)
{
    Problem<float> p(1, 0, row);
    ASSERT_EQ(p.Prepare(), ok);
    auto previous = std::move(p.dv);
    p.dv = std::make_unique<Device>(p.c.size() * sizeof(float));
    p.dv->Put(p.c);
    ASSERT_EQ(aclsparseCsrSetPointers(p.dc, p.dr->ptr, p.di->ptr, p.dv->ptr), ok);
    ASSERT_EQ(p.Execute(), ok);
    p.Verify();
    EXPECT_EQ(previous->Get<float>(p.c.size()), p.c);
    previous->Guard();
}
TEST_F(SddmmContract, ZeroReductionAndZeroAlpha)
{
    for (int bd : {1, 2, 16}) {
        for (int k : {0, 7}) {
            Problem<std::complex<float>> p(bd, 1, row, k);
            if (k) {
                p.alpha = 0;
            }
            ASSERT_EQ(p.Prepare(), ok);
            ASSERT_EQ(p.Execute(), ok);
            p.Verify();
            p.beta = 0;
            std::fill(p.c.begin(), p.c.begin() + 4 * bd * bd, std::complex<float>(NAN, NAN));
            p.dv->Put(p.c);
            ASSERT_EQ(p.Execute(), ok);
            p.Verify();
        }
    }
}
TEST_F(SddmmContract, DescriptorValidationAndBatchLimits)
{
    Problem<float> p(2, 0, row);
    int count = 0;
    int64_t stride = 0;
    EXPECT_EQ(aclsparseDnMatGetStridedBatch(p.dx, &count, &stride), ok);
    EXPECT_EQ(count, 1);
    EXPECT_EQ(stride, p.xs);
    EXPECT_EQ(aclsparseDnMatGetStridedBatch(p.dx, nullptr, &stride), invalid);
    for (int bad : {0, -1, 65536}) {
        EXPECT_EQ(aclsparseDnMatSetStridedBatch(p.dx, bad, p.xs), invalid);
        EXPECT_EQ(aclsparseBsrSetStridedBatch(p.dc, bad, p.rs, p.cs, p.vs), invalid);
    }
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(p.dx, 65535, p.xs), ok);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(p.dx, 2, 1), invalid);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(p.dx, 2, INT64_MAX), invalid);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(p.dc, 2, 1, p.cs, p.vs), invalid);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(p.dc, 2, p.rs, p.cs, INT64_MAX), invalid);
    aclsparseConstSpMatDescr_t readOnly = nullptr;
    ASSERT_EQ(aclsparseCreateConstBsr(&readOnly, 3, 4, 4, 2, 2, p.dr->ptr, p.di->ptr, p.dv->ptr, ACL_SPARSE_INDEX_32I,
                                      ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT, row),
              ok);
    EXPECT_EQ(aclsparseDestroySpMat(readOnly), ok);
    aclsparseSpMatDescr_t bad = nullptr;
    EXPECT_EQ(aclsparseCreateBsr(&bad, 3, 4, 4, 3, 3, p.dr->ptr, p.di->ptr, p.dv->ptr, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT, row),
              invalid);
    EXPECT_EQ(aclsparseSDDMMPreprocess(p.h, nn, nn, &p.alpha, p.dx, p.dy, &p.beta, p.dc, ACL_FLOAT, alg, nullptr),
              invalid);
}
TEST_F(SddmmContract, DenseByteStrideOverflow)
{
    Problem<std::complex<float>> p(1, 0, row);
    for (int64_t ld : {int64_t(UINT32_MAX / 8) + 1, int64_t(1) << 61, INT64_MAX}) {
        aclsparseDnMatDescr_t wide = nullptr;
        ASSERT_EQ(aclsparseCreateDnMat(&wide, p.m, p.k, ld, p.dxx->ptr, ACL_COMPLEX64, row), ok);
        size_t bytes = 0;
        EXPECT_EQ(
            aclsparseSDDMMBufferSize(p.h, nn, nn, &p.alpha, wide, p.dy, &p.beta, p.dc, ACL_COMPLEX64, alg, &bytes),
            invalid)
            << ld;
        EXPECT_EQ(aclsparseDestroyDnMat(wide), ok);
    }
}
} // namespace
