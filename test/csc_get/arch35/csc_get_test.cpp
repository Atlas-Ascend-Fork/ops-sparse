/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file csc_get_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseCreateCsc / aclsparseCscGet (arch35).
 *
 * CscGet 对标 cuSPARSE cusparseCscGet：CreateCsc -> CscGet 闭环，读回字段
 * 应与 Create 传入一致。CscGet 是纯 Host 侧描述符访问器，不启动 NPU kernel；
 * 测试侧建设备缓冲存入描述符，CscGet 读回字段后断言与 Create 传入一致。
 *
 *   - TEST_P (CscGetTest)          : 17 个 CSV 驱动的成功路径用例（FP16/FP32,
 *                                    idx0/idx1, 方阵/非方阵/不对齐/大 shape/空矩阵,
 *                                    ptr/idx 索引类型四种组合）
 *   - TEST_F (CscGetExceptionTest) : null descr / 非 CSC 格式 / CreateCsc 与
 *                                    CreateConstCsc 前置校验 / CscSetPointers
 *                                    拒绝路径（nnz>0 空 idx/val、格式不匹配）/
 *                                    NPU wrapper 负维度入参拒绝
 *                                    等异常路径（L2-01..08, L2-15..20）
 *   - TEST_F (CscGetSemanticTest)  : ConstCscGet round-trip / 选择性回写 / 全 null /
 *                                    Set->Get 字段活性等语义边界（L0-04, L2-09..14）
 *   - TEST_F (CscGetWhiteboxTest)  : 依据 aclsparse_descr_guard.h 的
 *                                    ReadCompressedSpMat 实际源码分支生成的白盒用例
 *                                    （W-01..W-18：单字段隔离回写 / Const 变体
 *                                    nullptr 出参与选择性回写 / 错误路径无写出 /
 *                                    SLICED_ELL 格式枚举 / uint64->int64 大值边界 /
 *                                    签名守卫拒绝与恢复复核）
 */

#include "test_common.h"

#include "aclsparse_descr_internal.h"
#include "../csc_get_param.h"
#include "csc_get_npu_wrapper.h"

using namespace sparse_test;

// ============================================================================
// GTest parameterized fixture: CscGetTest
// ============================================================================
class CscGetTest : public testing::TestWithParam<CscGetParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }
    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    CscGetParam param_;

    void SetUp() override { param_ = GetParam(); }
};

// ============================================================================
// Test body: success-path parameterized test（对应 0-黑盒测试设计.md §2.3 口径表）
// ============================================================================
TEST_P(CscGetTest, CscGetRoundTrip) {
    const auto &p = param_;
    std::cout << "==== " << p.case_name << " ==== rows=" << p.rows << " cols=" << p.cols
              << " nnz=" << p.nnz << " dtype=" << p.dtype << " idx_base=" << p.idx_base
              << " ptr_type=" << p.ptr_type << " idx_type=" << p.idx_type << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");

    // 1. 生成 CSC 输入数据（host 侧仅生成输入，不在 CPU 做算子计算）
    // 2. NPU 执行：CreateCsc -> CscGet accessor 验证（wrapper 内完成设备迁移）
    auto npu = CscGetNpu(p.rows, p.cols, p.nnz, p.idx_base, p.isFp16(),
                         p.ptrIndexType(), p.idxIndexType(), p.seed);

    ASSERT_EQ(npu.createRet, ACL_SPARSE_STATUS_SUCCESS)
        << "aclsparseCreateCsc failed";
    ASSERT_EQ(npu.getRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCscGet failed";

    // 3. 断言 Get 读回的标量元数据 == Create 传入（整型/枚举精确相等）
    EXPECT_EQ(npu.rows, p.rows);
    EXPECT_EQ(npu.cols, p.cols);
    EXPECT_EQ(npu.nnz, p.nnz);
    EXPECT_EQ(npu.colOffsetsType, p.ptrIndexType());
    EXPECT_EQ(npu.rowIndType, p.idxIndexType());
    EXPECT_EQ(npu.idxBase, (p.idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE
                                             : ACL_SPARSE_INDEX_BASE_ZERO);
    EXPECT_EQ(npu.valueType, p.isFp16() ? ACL_FLOAT16 : ACL_FLOAT);

    // 4. 断言 Get 读回的设备指针 == Create 传入的设备指针（accessor 语义，无容差）
    EXPECT_TRUE(npu.devPtrMatch) << "CscGet device pointer mismatch";

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    CscGetCases,
    CscGetTest,
    testing::ValuesIn(GetCasesFromCsv<CscGetParam>("csc_get_test.csv")),
    [](const testing::TestParamInfo<CscGetParam> &info) {
        return info.param.case_name;
    });

// ============================================================================
// Exception test fixture: CscGetExceptionTest（L2-01..08 错误路径）
// ============================================================================
class CscGetExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    // 2×2、nnz=2 的合法稀疏输入（accessor/Create 不解引用数据，host 指针即可）
    std::vector<int32_t> offsets_{0, 1, 2};   // CSR rowOffsets / CSC colOffsets，长 dim+1
    std::vector<int32_t> indices_{0, 1};      // CSR colInd / CSC rowInd / COO 索引，长 nnz
    std::vector<float> values_{1.0f, 2.0f};   // values，长 nnz

    void SetUp() override {}
};

// L2-01: CscGet nullptr descriptor -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CscGetExceptionTest, NullSpMatDescrHandle) {
    int64_t r, c, n;
    void *o, *i, *v;
    aclsparseIndexType_t pt, it;
    aclsparseIndexBase_t ib;
    aclDataType vt;
    EXPECT_EQ(aclsparseCscGet(nullptr, &r, &c, &n, &o, &i, &v, &pt, &it, &ib, &vt),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// L2-02: ConstCscGet nullptr descriptor -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CscGetExceptionTest, ConstCscGetNullSpMatDescrHandle) {
    int64_t r, c, n;
    const void *o, *i, *v;
    aclsparseIndexType_t pt, it;
    aclsparseIndexBase_t ib;
    aclDataType vt;
    EXPECT_EQ(aclsparseConstCscGet(nullptr, &r, &c, &n, &o, &i, &v, &pt, &it, &ib, &vt),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// L2-03: CscGet on a CSR descriptor -> ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
TEST_F(CscGetExceptionTest, CscGetOnCsrDescr) {
    try {
        auto mat = SpMatManager::createCsr(2, 2, 2, offsets_.data(), indices_.data(), values_.data(),
                                           ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                           ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        int64_t r, c, n;
        void *o, *i, *v;
        aclsparseIndexType_t pt, it;
        aclsparseIndexBase_t ib;
        aclDataType vt;
        EXPECT_EQ(aclsparseCscGet(mat.get(), &r, &c, &n, &o, &i, &v, &pt, &it, &ib, &vt),
                  ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateCsr threw: " << e.what();
    }
}

// L2-04: CscGet on a COO descriptor -> ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
TEST_F(CscGetExceptionTest, CscGetOnCooDescr) {
    try {
        // indices_={0,1} 同时是合法的 2×2 对角 COO rowInd/colInd（accessor 不解引用）
        auto mat = SpMatManager::createCoo(2, 2, 2, indices_.data(), indices_.data(), values_.data(),
                                           ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                           ACL_FLOAT);
        int64_t r, c, n;
        void *o, *i, *v;
        aclsparseIndexType_t pt, it;
        aclsparseIndexBase_t ib;
        aclDataType vt;
        EXPECT_EQ(aclsparseCscGet(mat.get(), &r, &c, &n, &o, &i, &v, &pt, &it, &ib, &vt),
                  ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateCoo threw: " << e.what();
    }
}

// L2-05: ConstCscGet on a const CSR descriptor -> ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
TEST_F(CscGetExceptionTest, ConstCscGetOnConstCsrDescr) {
    try {
        auto mat = SpMatManager::createConstCsr(2, 2, 2, offsets_.data(), indices_.data(),
                                                values_.data(), ACL_SPARSE_INDEX_32I,
                                                ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                                ACL_FLOAT);
        int64_t r, c, n;
        const void *o, *i, *v;
        aclsparseIndexType_t pt, it;
        aclsparseIndexBase_t ib;
        aclDataType vt;
        EXPECT_EQ(aclsparseConstCscGet(mat.cget(), &r, &c, &n, &o, &i, &v, &pt, &it, &ib, &vt),
                  ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateConstCsr threw: " << e.what();
    }
}

// L2-06: nullptr out handle for CreateCsc -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CscGetExceptionTest, CreateCscNullOutHandle) {
    EXPECT_EQ(aclsparseCreateCsc(nullptr, 2, 2, 2, offsets_.data(), indices_.data(), values_.data(),
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// L2-07: negative dims for CreateCsc -> ACL_SPARSE_STATUS_INVALID_VALUE，descr 保持 nullptr
TEST_F(CscGetExceptionTest, CreateCscNegativeDim) {
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateCsc(&descr, -1, 2, 2, offsets_.data(), indices_.data(), values_.data(),
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr);
}

// L2-08: invalid index type for CreateCsc -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CscGetExceptionTest, CreateCscInvalidIndexType) {
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateCsc(&descr, 2, 2, 2, offsets_.data(), indices_.data(), values_.data(),
                                 static_cast<aclsparseIndexType_t>(99), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr);
}

// L2-15: nullptr out handle for CreateConstCsc -> ACL_SPARSE_STATUS_INVALID_VALUE（const 变体独立覆盖）
TEST_F(CscGetExceptionTest, CreateConstCscNullOutHandle) {
    EXPECT_EQ(aclsparseCreateConstCsc(nullptr, 2, 2, 2, offsets_.data(), indices_.data(),
                                      values_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                      ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// L2-16: negative dims for CreateConstCsc -> ACL_SPARSE_STATUS_INVALID_VALUE，descr 保持 nullptr
TEST_F(CscGetExceptionTest, CreateConstCscNegativeDim) {
    aclsparseConstSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateConstCsc(&descr, 2, -1, 2, offsets_.data(), indices_.data(),
                                      values_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                      ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr);
}

// L2-17: invalid index type for CreateConstCsc -> ACL_SPARSE_STATUS_INVALID_VALUE（覆盖 cscRowIndType 校验分支）
TEST_F(CscGetExceptionTest, CreateConstCscInvalidIndexType) {
    aclsparseConstSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateConstCsc(&descr, 2, 2, 2, offsets_.data(), indices_.data(),
                                      values_.data(), ACL_SPARSE_INDEX_32I,
                                      static_cast<aclsparseIndexType_t>(99),
                                      ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr);
}

// L2-18: CscSetPointers 在 nnz>0 时传空 rowIndices/values -> ACL_SPARSE_STATUS_INVALID_VALUE（延迟绑定拒绝路径）
TEST_F(CscGetExceptionTest, CscSetPointersRejectNullIdxValWithNnz) {
    try {
        auto mat = SpMatManager::createCsc(2, 2, 2, offsets_.data(), indices_.data(),
                                           values_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                           ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        EXPECT_EQ(aclsparseCscSetPointers(mat.get(), offsets_.data(), nullptr, nullptr),
                  ACL_SPARSE_STATUS_INVALID_VALUE);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateCsc threw: " << e.what();
    }
}

// L2-19: CscSetPointers 对非 CSC 格式描述符 -> ACL_SPARSE_STATUS_INVALID_VALUE（格式拒绝路径）
TEST_F(CscGetExceptionTest, CscSetPointersRejectFormatMismatch) {
    try {
        auto mat = SpMatManager::createCsr(2, 2, 2, offsets_.data(), indices_.data(),
                                           values_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                           ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        EXPECT_EQ(aclsparseCscSetPointers(mat.get(), offsets_.data(), indices_.data(),
                                          values_.data()),
                  ACL_SPARSE_STATUS_INVALID_VALUE);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateCsr threw: " << e.what();
    }
}

// L2-20: NPU wrapper 对负维度入参在分配前直接拒绝 -> createRet/getRet 均为 INVALID_VALUE，
// 不进入设备分配路径（负 nnz 经 static_cast<size_t> 会变成超大分配并使异常冲出 wrapper）
TEST_F(CscGetExceptionTest, NpuWrapperRejectNegativeDims) {
    auto rNegRows = CscGetNpu(-1, 2, 2, 0, false, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, 1);
    EXPECT_EQ(rNegRows.createRet, ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(rNegRows.getRet, ACL_SPARSE_STATUS_INVALID_VALUE);

    auto rNegNnz = CscGetNpu(2, 2, -1, 0, false, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, 1);
    EXPECT_EQ(rNegNnz.createRet, ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(rNegNnz.getRet, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ============================================================================
// Semantic test fixture: CscGetSemanticTest（L0-04 + L2-09..14 语义边界）
// ============================================================================
class CscGetSemanticTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    // 2×2、nnz=2 的 well-formed CSC 输入（host 指针即可：accessor 不解引用）
    std::vector<int32_t> offsets_{0, 1, 2};   // cscColOffsets，长 cols+1（CSC 规格）
    std::vector<int32_t> indices_{0, 1};      // cscRowInd，长 nnz
    std::vector<float> values_{1.0f, 2.0f};   // cscValues，长 nnz

    // 选择性回写哨兵值：未请求字段保持不变（0-黑盒测试设计.md §5.3）
    static constexpr int64_t kSentinelScalar = -0x7BADBEEF;
    static constexpr uintptr_t kSentinelAddr = 0xDEAD;
    static constexpr int kSentinelEnum = 0x5A;  // 非法枚举值（I32=0/I64=1、ZERO=0/ONE=1 之外）

    static void *SentinelPtr() { return reinterpret_cast<void *>(kSentinelAddr); }
    static aclsparseIndexType_t SentinelIndexType() {
        return static_cast<aclsparseIndexType_t>(kSentinelEnum);
    }
    static aclsparseIndexBase_t SentinelIndexBase() {
        return static_cast<aclsparseIndexBase_t>(kSentinelEnum);
    }
    static aclDataType SentinelValueType() {
        return static_cast<aclDataType>(kSentinelEnum);
    }

    // 在 2×2 CSC 描述符（I32/I32、idxBase ONE、FP32）上执行 aclsparseCscGet。
    // 传入 nullptr 的出参即"未请求"该字段（仅回写请求字段）。
    aclsparseStatus_t CscGetOn2x2(int64_t *r, int64_t *c, int64_t *n,
                                  void **o, void **i, void **v,
                                  aclsparseIndexType_t *pt, aclsparseIndexType_t *it,
                                  aclsparseIndexBase_t *ib, aclDataType *vt) {
        auto mat = SpMatManager::createCsc(2, 2, 2, offsets_.data(), indices_.data(), values_.data(),
                                           ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                           ACL_SPARSE_INDEX_BASE_ONE, ACL_FLOAT);
        return aclsparseCscGet(mat.get(), r, c, n, o, i, v, pt, it, ib, vt);
    }

    void SetUp() override {}
};

// L0-04: ConstCscGet 全字段 round-trip（const 变体 + host 指针恒等）
TEST_F(CscGetSemanticTest, ConstCscGetRoundTrip) {
    try {
        auto mat = SpMatManager::createConstCsc(2, 2, 2, offsets_.data(), indices_.data(),
                                                values_.data(), ACL_SPARSE_INDEX_32I,
                                                ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ONE,
                                                ACL_FLOAT);
        int64_t r = -1, c = -1, n = -1;
        const void *o = nullptr;
        const void *i = nullptr;
        const void *v = nullptr;
        aclsparseIndexType_t pt, it;
        aclsparseIndexBase_t ib;
        aclDataType vt;
        EXPECT_EQ(aclsparseConstCscGet(mat.cget(), &r, &c, &n, &o, &i, &v, &pt, &it, &ib, &vt),
                  ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(r, 2);
        EXPECT_EQ(c, 2);
        EXPECT_EQ(n, 2);
        EXPECT_EQ(pt, ACL_SPARSE_INDEX_32I);
        EXPECT_EQ(it, ACL_SPARSE_INDEX_32I);
        EXPECT_EQ(ib, ACL_SPARSE_INDEX_BASE_ONE);
        EXPECT_EQ(vt, ACL_FLOAT);
        // 三指针与 Create 传入的 host 指针逐字相等（accessor 不拷贝）
        EXPECT_EQ(o, offsets_.data());
        EXPECT_EQ(i, indices_.data());
        EXPECT_EQ(v, values_.data());
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateConstCsc threw: " << e.what();
    }
}

// L2-09: 选择性回写：仅标量组（rows/cols/nnz）
TEST_F(CscGetSemanticTest, SelectiveWriteScalarsOnly) {
    int64_t r = kSentinelScalar, c = kSentinelScalar, n = kSentinelScalar;
    void *o = SentinelPtr(), *i = SentinelPtr(), *v = SentinelPtr();
    aclsparseIndexType_t pt = SentinelIndexType(), it = SentinelIndexType();
    aclsparseIndexBase_t ib = SentinelIndexBase();
    aclDataType vt = SentinelValueType();
    try {
        EXPECT_EQ(CscGetOn2x2(&r, &c, &n, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr),
                  ACL_SPARSE_STATUS_SUCCESS);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateCsc threw: " << e.what();
    }
    EXPECT_EQ(r, 2);
    EXPECT_EQ(c, 2);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(o, SentinelPtr());
    EXPECT_EQ(i, SentinelPtr());
    EXPECT_EQ(v, SentinelPtr());
    EXPECT_EQ(pt, SentinelIndexType());
    EXPECT_EQ(it, SentinelIndexType());
    EXPECT_EQ(ib, SentinelIndexBase());
    EXPECT_EQ(vt, SentinelValueType());
}

// L2-10: 选择性回写：仅指针组（cscColOffsets/cscRowInd/cscValues）
TEST_F(CscGetSemanticTest, SelectiveWritePointersOnly) {
    int64_t r = kSentinelScalar, c = kSentinelScalar, n = kSentinelScalar;
    void *o = SentinelPtr(), *i = SentinelPtr(), *v = SentinelPtr();
    aclsparseIndexType_t pt = SentinelIndexType(), it = SentinelIndexType();
    aclsparseIndexBase_t ib = SentinelIndexBase();
    aclDataType vt = SentinelValueType();
    try {
        EXPECT_EQ(CscGetOn2x2(nullptr, nullptr, nullptr, &o, &i, &v, nullptr, nullptr, nullptr,
                              nullptr),
                  ACL_SPARSE_STATUS_SUCCESS);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateCsc threw: " << e.what();
    }
    // 三指针与 Create 传入的 host 指针逐字恒等
    EXPECT_EQ(o, offsets_.data());
    EXPECT_EQ(i, indices_.data());
    EXPECT_EQ(v, values_.data());
    EXPECT_EQ(r, kSentinelScalar);
    EXPECT_EQ(c, kSentinelScalar);
    EXPECT_EQ(n, kSentinelScalar);
    EXPECT_EQ(pt, SentinelIndexType());
    EXPECT_EQ(it, SentinelIndexType());
    EXPECT_EQ(ib, SentinelIndexBase());
    EXPECT_EQ(vt, SentinelValueType());
}

// L2-11: 选择性回写：仅类型组（cscColOffsetsType/cscRowIndType/idxBase/valueType）
TEST_F(CscGetSemanticTest, SelectiveWriteTypesOnly) {
    int64_t r = kSentinelScalar, c = kSentinelScalar, n = kSentinelScalar;
    void *o = SentinelPtr(), *i = SentinelPtr(), *v = SentinelPtr();
    aclsparseIndexType_t pt = SentinelIndexType(), it = SentinelIndexType();
    aclsparseIndexBase_t ib = SentinelIndexBase();
    aclDataType vt = SentinelValueType();
    try {
        EXPECT_EQ(CscGetOn2x2(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &pt, &it, &ib,
                              &vt),
                  ACL_SPARSE_STATUS_SUCCESS);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateCsc threw: " << e.what();
    }
    // 描述符为 I32/I32、idxBase ONE、FP32 构造
    EXPECT_EQ(pt, ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(it, ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(ib, ACL_SPARSE_INDEX_BASE_ONE);
    EXPECT_EQ(vt, ACL_FLOAT);
    EXPECT_EQ(r, kSentinelScalar);
    EXPECT_EQ(c, kSentinelScalar);
    EXPECT_EQ(n, kSentinelScalar);
    EXPECT_EQ(o, SentinelPtr());
    EXPECT_EQ(i, SentinelPtr());
    EXPECT_EQ(v, SentinelPtr());
}

// L2-12: 选择性回写：交错混合（仅 rows + cscValues + idxBase + cscRowIndType）
TEST_F(CscGetSemanticTest, SelectiveWriteMixed) {
    int64_t r = kSentinelScalar, c = kSentinelScalar, n = kSentinelScalar;
    void *o = SentinelPtr(), *i = SentinelPtr(), *v = SentinelPtr();
    aclsparseIndexType_t pt = SentinelIndexType(), it = SentinelIndexType();
    aclsparseIndexBase_t ib = SentinelIndexBase();
    aclDataType vt = SentinelValueType();
    try {
        EXPECT_EQ(CscGetOn2x2(&r, nullptr, nullptr, nullptr, nullptr, &v, nullptr, &it, &ib, nullptr),
                  ACL_SPARSE_STATUS_SUCCESS);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateCsc threw: " << e.what();
    }
    // 请求字段正确回写
    EXPECT_EQ(r, 2);
    EXPECT_EQ(v, values_.data());
    EXPECT_EQ(it, ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(ib, ACL_SPARSE_INDEX_BASE_ONE);
    // 未请求字段哨兵不变
    EXPECT_EQ(c, kSentinelScalar);
    EXPECT_EQ(n, kSentinelScalar);
    EXPECT_EQ(o, SentinelPtr());
    EXPECT_EQ(i, SentinelPtr());
    EXPECT_EQ(pt, SentinelIndexType());
    EXPECT_EQ(vt, SentinelValueType());
}

// L2-13: 全部 10 个 OUT 为 nullptr（合法 CSC 描述符）-> SUCCESS，无崩溃
TEST_F(CscGetSemanticTest, AllOutputsNull) {
    try {
        EXPECT_EQ(CscGetOn2x2(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr),
                  ACL_SPARSE_STATUS_SUCCESS);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateCsc threw: " << e.what();
    }
}

// L2-14: Set->Get 字段活性：CscSetPointers 更新后 Get 读回新指针
TEST_F(CscGetSemanticTest, SetPointersThenGet) {
    // 设备缓冲 A：Create 时绑定（SetPointers 要求 nnz>0 时 idx/val 非空，用设备缓冲）
    auto dOffA = DeviceBuffer::alloc(offsets_.size() * sizeof(int32_t));
    auto dIdxA = DeviceBuffer::alloc(indices_.size() * sizeof(int32_t));
    auto dValA = DeviceBuffer::alloc(values_.size() * sizeof(float));
    // 设备缓冲 B：SetPointers 更新后应被 Get 读回
    auto dOffB = DeviceBuffer::alloc(offsets_.size() * sizeof(int32_t));
    auto dIdxB = DeviceBuffer::alloc(indices_.size() * sizeof(int32_t));
    auto dValB = DeviceBuffer::alloc(values_.size() * sizeof(float));
    try {
        auto mat = SpMatManager::createCsc(2, 2, 2, dOffA.get(), dIdxA.get(), dValA.get(),
                                           ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                           ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        ASSERT_EQ(aclsparseCscSetPointers(mat.get(), dOffB.get(), dIdxB.get(), dValB.get()),
                  ACL_SPARSE_STATUS_SUCCESS);
        int64_t r = -1, c = -1, n = -1;
        void *o = nullptr, *i = nullptr, *v = nullptr;
        aclsparseIndexType_t pt, it;
        aclsparseIndexBase_t ib;
        aclDataType vt;
        EXPECT_EQ(aclsparseCscGet(mat.get(), &r, &c, &n, &o, &i, &v, &pt, &it, &ib, &vt),
                  ACL_SPARSE_STATUS_SUCCESS);
        // Get 读回 SetPointers 更新后的新指针（描述符当前状态，而非 Create 时的缓存）
        EXPECT_EQ(o, dOffB.get());
        EXPECT_EQ(i, dIdxB.get());
        EXPECT_EQ(v, dValB.get());
        EXPECT_EQ(r, 2);
        EXPECT_EQ(c, 2);
        EXPECT_EQ(n, 2);
    } catch (const std::exception &e) {
        FAIL() << "descriptor setup threw: " << e.what();
    }
}

// ============================================================================
// Whitebox test fixture: CscGetWhiteboxTest
// （依据 sparse/common/aclsparse_descr_guard.h 的 ReadCompressedSpMat 实际源码
// 分支生成（csc_get_host.cpp 为其 CSC 薄封装），复用 CscGetSemanticTest 的 2×2 描述符构造与哨兵机制：
//   - W-01..W-10 单字段隔离回写：源码 10 个 if (out != nullptr) 回写分支
//     逐一独立为真（其余 9 个为假），验证分支间无串扰；
//   - W-11/W-12 ConstCscGet（PtrOut=const void** 实例化）的 nullptr 出参与
//     选择性回写；
//   - W-13/W-14 错误路径早返回不写出参（无副作用验证）；
//   - W-15/W-16 SLICED_ELL 格式枚举（format != CSC 判真的非 CSR/COO 取值）；
//   - W-17/W-18 描述符 uint64_t 字段经 static_cast<int64_t> 回写的大值边界；
//   - W-19 签名守卫（signature != kSpMatSignature 时拒绝，恢复后复核正常路径）。）
// ============================================================================
class CscGetWhiteboxTest : public CscGetSemanticTest {
public:
    static void SetUpTestSuite() { CscGetSemanticTest::SetUpTestSuite(); }
    static void TearDownTestSuite() { CscGetSemanticTest::TearDownTestSuite(); }

protected:
    // 出参字段编号：对应 ReadCompressedSpMat 源码 10 个出参回写分支的声明顺序
    enum OutField {
        kRows = 0,
        kCols,
        kNnz,
        kColOffsets,
        kRowInd,
        kValues,
        kColOffsetsType,
        kRowIndType,
        kIdxBase,
        kValueType,
        kOutFieldCount
    };

    // CscGetOn2x2 描述符（2×2、nnz=2、I32/I32、idxBase ONE、FP32）的标量期望值
    static constexpr int64_t kExpDim = 2;

    // 10 个出参的聚合载体：ResetBundle 预填哨兵，供单字段隔离与无写出断言复用
    struct OutBundle {
        int64_t rows;
        int64_t cols;
        int64_t nnz;
        void *colOffsets;
        void *rowInd;
        void *values;
        aclsparseIndexType_t colOffsetsType;
        aclsparseIndexType_t rowIndType;
        aclsparseIndexBase_t idxBase;
        aclDataType valueType;
    };

    void ResetBundle(OutBundle &b) {
        b.rows = kSentinelScalar;
        b.cols = kSentinelScalar;
        b.nnz = kSentinelScalar;
        b.colOffsets = SentinelPtr();
        b.rowInd = SentinelPtr();
        b.values = SentinelPtr();
        b.colOffsetsType = SentinelIndexType();
        b.rowIndType = SentinelIndexType();
        b.idxBase = SentinelIndexBase();
        b.valueType = SentinelValueType();
    }

    void ExpectBundleSentinel(const OutBundle &b) {
        EXPECT_EQ(b.rows, kSentinelScalar);
        EXPECT_EQ(b.cols, kSentinelScalar);
        EXPECT_EQ(b.nnz, kSentinelScalar);
        EXPECT_EQ(b.colOffsets, SentinelPtr());
        EXPECT_EQ(b.rowInd, SentinelPtr());
        EXPECT_EQ(b.values, SentinelPtr());
        EXPECT_EQ(b.colOffsetsType, SentinelIndexType());
        EXPECT_EQ(b.rowIndType, SentinelIndexType());
        EXPECT_EQ(b.idxBase, SentinelIndexBase());
        EXPECT_EQ(b.valueType, SentinelValueType());
    }

    // 仅请求 fieldId 一个出参（其余 9 个传 nullptr），在 2×2 CSC 描述符上执行 CscGet
    aclsparseStatus_t RunOneHot(OutBundle &b, int fieldId) {
        return CscGetOn2x2(fieldId == kRows ? &b.rows : nullptr,
                           fieldId == kCols ? &b.cols : nullptr,
                           fieldId == kNnz ? &b.nnz : nullptr,
                           fieldId == kColOffsets ? &b.colOffsets : nullptr,
                           fieldId == kRowInd ? &b.rowInd : nullptr,
                           fieldId == kValues ? &b.values : nullptr,
                           fieldId == kColOffsetsType ? &b.colOffsetsType : nullptr,
                           fieldId == kRowIndType ? &b.rowIndType : nullptr,
                           fieldId == kIdxBase ? &b.idxBase : nullptr,
                           fieldId == kValueType ? &b.valueType : nullptr);
    }

    // 断言仅 fieldId 被回写为描述符期望值，其余 9 个字段保持哨兵不变
    void VerifyOnly(const OutBundle &b, int fieldId) {
        EXPECT_EQ(b.rows, fieldId == kRows ? kExpDim : kSentinelScalar);
        EXPECT_EQ(b.cols, fieldId == kCols ? kExpDim : kSentinelScalar);
        EXPECT_EQ(b.nnz, fieldId == kNnz ? kExpDim : kSentinelScalar);
        EXPECT_EQ(b.colOffsets, fieldId == kColOffsets ? offsets_.data() : SentinelPtr());
        EXPECT_EQ(b.rowInd, fieldId == kRowInd ? indices_.data() : SentinelPtr());
        EXPECT_EQ(b.values, fieldId == kValues ? values_.data() : SentinelPtr());
        EXPECT_EQ(b.colOffsetsType,
                  fieldId == kColOffsetsType ? ACL_SPARSE_INDEX_32I : SentinelIndexType());
        EXPECT_EQ(b.rowIndType,
                  fieldId == kRowIndType ? ACL_SPARSE_INDEX_32I : SentinelIndexType());
        EXPECT_EQ(b.idxBase,
                  fieldId == kIdxBase ? ACL_SPARSE_INDEX_BASE_ONE : SentinelIndexBase());
        EXPECT_EQ(b.valueType, fieldId == kValueType ? ACL_FLOAT : SentinelValueType());
    }
};

// W-01..W-10: 单字段隔离回写——源码 10 个 if (out != nullptr) 回写分支逐一
// 独立为真（其余 9 个为假），验证分支间无串扰；强于黑盒 L2-09..12 的分组选择。
TEST_F(CscGetWhiteboxTest, SelectiveWriteSingleField) {
    for (int fieldId = 0; fieldId < kOutFieldCount; fieldId++) {
        OutBundle b;
        ResetBundle(b);
        try {
            std::cout << "  [one-hot] fieldId=" << fieldId << "\n";
            EXPECT_EQ(RunOneHot(b, fieldId), ACL_SPARSE_STATUS_SUCCESS) << "fieldId=" << fieldId;
        } catch (const std::exception &e) {
            FAIL() << "aclsparseCreateCsc threw: " << e.what();
        }
        VerifyOnly(b, fieldId);
    }
}

// W-11: ConstCscGet 全部 10 个出参为 nullptr——const 实例化下 10 个回写分支
// 全假的路径（黑盒 L2-13 仅对非 const 变体覆盖全 null）。
TEST_F(CscGetWhiteboxTest, ConstCscGetAllOutputsNull) {
    try {
        auto mat = SpMatManager::createConstCsc(2, 2, 2, offsets_.data(), indices_.data(),
                                                values_.data(), ACL_SPARSE_INDEX_32I,
                                                ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ONE,
                                                ACL_FLOAT);
        EXPECT_EQ(aclsparseConstCscGet(mat.cget(), nullptr, nullptr, nullptr, nullptr, nullptr,
                                       nullptr, nullptr, nullptr, nullptr, nullptr),
                  ACL_SPARSE_STATUS_SUCCESS);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateConstCsc threw: " << e.what();
    }
}

// W-12: ConstCscGet 选择性回写（仅 cols/cscColOffsets/cscRowIndType/valueType），
// 描述符取 I64/I64 + idxBase ZERO + FP16（L0-04 之外的 const 实例化字段组合）。
TEST_F(CscGetWhiteboxTest, ConstCscGetSelectiveWriteMixed) {
    // I64 索引数组与 FP16 值占位缓冲（accessor 不解引用数据，指针语义即可）
    std::vector<int64_t> off64 = {0, 1, 2};
    std::vector<int64_t> idx64 = {0, 1};
    std::vector<uint16_t> val16 = {0x3C00, 0x4000};  // FP16 位型 1.0 / 2.0
    int64_t r = kSentinelScalar, c = kSentinelScalar, n = kSentinelScalar;
    const void *o = SentinelPtr();
    const void *i = SentinelPtr();
    const void *v = SentinelPtr();
    aclsparseIndexType_t pt = SentinelIndexType(), it = SentinelIndexType();
    aclsparseIndexBase_t ib = SentinelIndexBase();
    aclDataType vt = SentinelValueType();
    try {
        auto mat = SpMatManager::createConstCsc(2, 2, 2, off64.data(), idx64.data(), val16.data(),
                                                ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_64I,
                                                ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT16);
        EXPECT_EQ(aclsparseConstCscGet(mat.cget(), nullptr, &c, nullptr, &o, nullptr, nullptr,
                                       nullptr, &it, nullptr, &vt),
                  ACL_SPARSE_STATUS_SUCCESS);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateConstCsc threw: " << e.what();
    }
    // 请求字段正确回写
    EXPECT_EQ(c, 2);
    EXPECT_EQ(o, off64.data());
    EXPECT_EQ(it, ACL_SPARSE_INDEX_64I);
    EXPECT_EQ(vt, ACL_FLOAT16);
    // 未请求字段哨兵不变
    EXPECT_EQ(r, kSentinelScalar);
    EXPECT_EQ(n, kSentinelScalar);
    EXPECT_EQ(i, SentinelPtr());
    EXPECT_EQ(v, SentinelPtr());
    EXPECT_EQ(pt, SentinelIndexType());
    EXPECT_EQ(ib, SentinelIndexBase());
}

// W-13: 错误路径无写出——CSR 描述符（format != CSC）在源码判假分支早返回，
// 10 个出参应全部保持哨兵不变（黑盒 L2-03 仅断言返回码，未验证无写出）。
TEST_F(CscGetWhiteboxTest, ErrorPathNoWriteOnCsr) {
    OutBundle b;
    ResetBundle(b);
    try {
        auto mat = SpMatManager::createCsr(2, 2, 2, offsets_.data(), indices_.data(),
                                           values_.data(), ACL_SPARSE_INDEX_32I,
                                           ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                           ACL_FLOAT);
        EXPECT_EQ(aclsparseCscGet(mat.get(), &b.rows, &b.cols, &b.nnz, &b.colOffsets, &b.rowInd,
                                  &b.values, &b.colOffsetsType, &b.rowIndType, &b.idxBase,
                                  &b.valueType),
                  ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateCsr threw: " << e.what();
    }
    ExpectBundleSentinel(b);
}

// W-14: 错误路径无写出——null 描述符在源码判空分支早返回，两变体 10 个出参
// 全部保持哨兵不变（黑盒 L2-01/02 传入未初始化出参，无法验证无写出）。
TEST_F(CscGetWhiteboxTest, ErrorPathNoWriteOnNullDescr) {
    OutBundle b;
    ResetBundle(b);
    EXPECT_EQ(aclsparseCscGet(nullptr, &b.rows, &b.cols, &b.nnz, &b.colOffsets, &b.rowInd,
                              &b.values, &b.colOffsetsType, &b.rowIndType, &b.idxBase,
                              &b.valueType),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectBundleSentinel(b);

    // const 变体同路径（PtrOut=const void** 实例化的判空分支）
    int64_t r = kSentinelScalar, c = kSentinelScalar, n = kSentinelScalar;
    const void *o = SentinelPtr();
    const void *i = SentinelPtr();
    const void *v = SentinelPtr();
    aclsparseIndexType_t pt = SentinelIndexType(), it = SentinelIndexType();
    aclsparseIndexBase_t ib = SentinelIndexBase();
    aclDataType vt = SentinelValueType();
    EXPECT_EQ(aclsparseConstCscGet(nullptr, &r, &c, &n, &o, &i, &v, &pt, &it, &ib, &vt),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(r, kSentinelScalar);
    EXPECT_EQ(c, kSentinelScalar);
    EXPECT_EQ(n, kSentinelScalar);
    EXPECT_EQ(o, SentinelPtr());
    EXPECT_EQ(i, SentinelPtr());
    EXPECT_EQ(v, SentinelPtr());
    EXPECT_EQ(pt, SentinelIndexType());
    EXPECT_EQ(it, SentinelIndexType());
    EXPECT_EQ(ib, SentinelIndexBase());
    EXPECT_EQ(vt, SentinelValueType());
}

// W-15: CscGet 作用于 SLICED_ELL 描述符——format 判假分支的非 CSR/COO 枚举取值
// （ACL_SPARSE_FORMAT_SLICED_ELL），验证判式为相等比较而非区间/集合判断。
TEST_F(CscGetWhiteboxTest, CscGetOnSlicedEllDescr) {
    try {
        // 2×2、nnz=2、sliceNnz=1、numSlices=2 的合法 SLICED_ELL 描述符
        auto mat = SpMatManager::createSlicedEll(2, 2, 2, offsets_.data(), indices_.data(),
                                                 values_.data(), 1, 2, ACL_SPARSE_INDEX_32I,
                                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        int64_t r, c, n;
        void *o, *i, *v;
        aclsparseIndexType_t pt, it;
        aclsparseIndexBase_t ib;
        aclDataType vt;
        EXPECT_EQ(aclsparseCscGet(mat.get(), &r, &c, &n, &o, &i, &v, &pt, &it, &ib, &vt),
                  ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateSlicedEll threw: " << e.what();
    }
}

// W-16: ConstCscGet 作用于 ConstSlicedEll 描述符——const 实例化的 format 判假
// 扩展（黑盒 L2-05 仅覆盖 ConstCsr）。
TEST_F(CscGetWhiteboxTest, ConstCscGetOnConstSlicedEllDescr) {
    try {
        auto mat = SpMatManager::createConstSlicedEll(2, 2, 2, offsets_.data(), indices_.data(),
                                                      values_.data(), 1, 2, ACL_SPARSE_INDEX_32I,
                                                      ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        int64_t r, c, n;
        const void *o, *i, *v;
        aclsparseIndexType_t pt, it;
        aclsparseIndexBase_t ib;
        aclDataType vt;
        EXPECT_EQ(aclsparseConstCscGet(mat.cget(), &r, &c, &n, &o, &i, &v, &pt, &it, &ib, &vt),
                  ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateConstSlicedEll threw: " << e.what();
    }
}

// W-17: uint64->int64 大值回读边界——描述符 rows/cols/nnz 以 uint64_t 存储
// （aclsparse_descr_internal.h），Get 以 static_cast<int64_t> 回写；用超出
// uint32/int32 正表示域的值验证无截断，同时覆盖 Create 延迟绑定（三指针
// nullptr）下的指针恒等回读（黑盒 CSV 最长仅到 2048/32768）。
TEST_F(CscGetWhiteboxTest, LargeDimsRoundTrip) {
    constexpr int64_t kLargeRows = static_cast<int64_t>(1) << 40;        // 2^40
    constexpr int64_t kLargeCols = (static_cast<int64_t>(1) << 31) + 3;  // 2^31+3
    constexpr int64_t kLargeNnz = (static_cast<int64_t>(1) << 35) + 7;   // 2^35+7
    try {
        auto mat = SpMatManager::createCsc(kLargeRows, kLargeCols, kLargeNnz, nullptr, nullptr,
                                           nullptr, ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_64I,
                                           ACL_SPARSE_INDEX_BASE_ONE, ACL_FLOAT);
        int64_t r = -1, c = -1, n = -1;
        void *o = SentinelPtr(), *i = SentinelPtr(), *v = SentinelPtr();
        aclsparseIndexType_t pt, it;
        aclsparseIndexBase_t ib;
        aclDataType vt;
        EXPECT_EQ(aclsparseCscGet(mat.get(), &r, &c, &n, &o, &i, &v, &pt, &it, &ib, &vt),
                  ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(r, kLargeRows);
        EXPECT_EQ(c, kLargeCols);
        EXPECT_EQ(n, kLargeNnz);
        EXPECT_EQ(o, nullptr);  // Create 延迟绑定传入 nullptr，Get 恒等读回 nullptr
        EXPECT_EQ(i, nullptr);
        EXPECT_EQ(v, nullptr);
        EXPECT_EQ(pt, ACL_SPARSE_INDEX_64I);
        EXPECT_EQ(it, ACL_SPARSE_INDEX_64I);
        EXPECT_EQ(ib, ACL_SPARSE_INDEX_BASE_ONE);
        EXPECT_EQ(vt, ACL_FLOAT);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateCsc threw: " << e.what();
    }
}

// W-18: int64 正域上界——rows=INT64_MAX 为合法 Create 入参（非负），Get 的
// static_cast<int64_t> 回读不溢出（描述符字段以 uint64_t 存储的转换边界）。
TEST_F(CscGetWhiteboxTest, Int64MaxRowsRoundTrip) {
    try {
        auto mat = SpMatManager::createCsc(INT64_MAX, 1, 0, nullptr, nullptr, nullptr,
                                           ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                           ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        int64_t r = -1;
        EXPECT_EQ(aclsparseCscGet(mat.get(), &r, nullptr, nullptr, nullptr, nullptr, nullptr,
                                  nullptr, nullptr, nullptr, nullptr),
                  ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(r, INT64_MAX);
    } catch (const std::exception &e) {
        FAIL() << "aclsparseCreateCsc threw: " << e.what();
    }
}

// W-19: 签名守卫——误传非稀疏矩阵描述符句柄（异类句柄 signature 不匹配）时拒绝，
// 返回 ACL_SPARSE_STATUS_INVALID_VALUE。临时破坏 signature 模拟异类句柄，
// 验证后恢复并复核正常路径。两个变体共用 ReadCompressedSpMat，非 const / const
// 实例化各命中一次该分支。
TEST_F(CscGetWhiteboxTest, CscGetOnInvalidSignatureDescr) {
    auto mgr = SpMatManager::createCsc(4, 4, 4, nullptr, nullptr, nullptr);
    const auto savedSignature = mgr.get()->signature;
    mgr.get()->signature = 0;
    EXPECT_EQ(aclsparseCscGet(mgr.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseConstCscGet(mgr.cget(), nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    mgr.get()->signature = savedSignature;

    // 恢复签名后同一句柄可正常读回（守卫不误伤合法描述符）
    int64_t r = -1;
    int64_t c = -1;
    int64_t n = -1;
    EXPECT_EQ(aclsparseCscGet(mgr.get(), &r, &c, &n, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r, 4);
    EXPECT_EQ(c, 4);
    EXPECT_EQ(n, 4);
}
