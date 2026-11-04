/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file csr_get_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseCreateCsr / aclsparseCsrGet (arch35).
 *
 * CsrGet 对标 cuSPARSE cusparseCsrGet：CreateCsr -> CsrGet 闭环，读回字段
 * 应与 Create 传入一致。CsrGet 是纯 Host 侧描述符访问器，不启动 NPU kernel；
 * 测试侧建设备缓冲存入描述符，CsrGet 读回字段后断言与 Create 传入一致。
 * CSR 双索引类型（csrRowOffsetsType / csrColIndType）独立取值，四种组合
 * 经 CSV 参数覆盖。
 *
 *   - TEST_P (CsrGetTest)        : 13 个 CSV 驱动的成功路径用例（L0/L1：
 *                                  索引类型四组合 x dtype x idxBase pairwise
 *                                  + 单行/单列/空矩阵/未对齐/大 shape 边界）
 *   - TEST_F (CsrGetExceptionTest): 14 个非 CSV 用例（null descr / 非 CSR 格式 /
 *                                  Create 前置异常 / OUT 全与部分 nullptr /
 *                                  延迟绑定 / valueType 全集 / const 变体）
 *   - TEST_F (CsrGetWhiteboxTest) : 7 个白盒用例（依据 csr_get_host.cpp 实际
 *                                  控制流：SLICED_ELL / BLOCKED_ELL 格式守卫 /
 *                                  签名守卫 / 10 个 OUT 字段逐一隔离回写 /
 *                                  const 变体散布 nullptr / 0 维与 int64 上界
 *                                  round-trip）
 */

#include "test_common.h"

#include "aclsparse_descr_internal.h"

#include "../csr_get_param.h"
#include "csr_get_npu_wrapper.h"

using namespace sparse_test;

// ============================================================================
// OUT 参数哨兵聚合（收敛各用例重复的 10 行 OUT 声明块）：默认哨兵与既有
// 用例一致——标量 -1、指针 nullptr、枚举零值（ACL_SPARSE_INDEX_32I /
// ACL_SPARSE_INDEX_BASE_ZERO）；需差异化哨兵的用例按字段覆盖（见 E12 /
// E13 / W4 / W5）。const 变体的三个数据指针 OUT 为 const void**。
// ============================================================================
template <typename PtrT>
struct CsrGetOutBlock {
    int64_t rows = -1;
    int64_t cols = -1;
    int64_t nnz = -1;
    PtrT rowOffPtr = nullptr;
    PtrT colIndPtr = nullptr;
    PtrT valuesPtr = nullptr;
    aclsparseIndexType_t rowOffType{};
    aclsparseIndexType_t colIndType{};
    aclsparseIndexBase_t idxBase{};
    aclDataType valueType{};
};
using CsrGetOut = CsrGetOutBlock<void *>;
using ConstCsrGetOut = CsrGetOutBlock<const void *>;

// ============================================================================
// BlockedELL 描述符局部 RAII 守卫：仅 csr_get 白盒用例构造"非 CSR 格式"句柄
// 使用（W2 格式守卫分支），不放入 test/frame 公共框架，避免影响其他算子测试。
// 语义与 SpMatManager 工厂方法一致：Create 失败抛异常，析构统一 DestroySpMat。
// ============================================================================
class BlockedEllDescrGuard {
public:
    BlockedEllDescrGuard(int64_t rows, int64_t cols, int64_t blockSize, int64_t ellCols,
                         void *ellColInd, void *ellValue, aclsparseIndexType_t indexType,
                         aclsparseIndexBase_t idxBase, aclDataType valueType)
    {
        const auto st = aclsparseCreateBlockedEll(&descr_, rows, cols, blockSize, ellCols,
                                                 ellColInd, ellValue, indexType, idxBase,
                                                 valueType);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            descr_ = nullptr;
            throw std::runtime_error("aclsparseCreateBlockedEll failed");
        }
    }
    ~BlockedEllDescrGuard()
    {
        if (descr_ != nullptr) {
            (void)aclsparseDestroySpMat(descr_);
        }
    }
    BlockedEllDescrGuard(const BlockedEllDescrGuard &) = delete;
    BlockedEllDescrGuard &operator=(const BlockedEllDescrGuard &) = delete;
    aclsparseSpMatDescr_t get() const
    {
        return descr_;
    }

private:
    aclsparseSpMatDescr_t descr_ = nullptr;
};

// ============================================================================
// GTest parameterized fixture: CsrGetTest
// ============================================================================
class CsrGetTest : public testing::TestWithParam<CsrGetParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }
    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    CsrGetParam param_;

    void SetUp() override { param_ = GetParam(); }
};

// ============================================================================
// Test body: success-path parameterized test
// ============================================================================
TEST_P(CsrGetTest, CsrGetRoundTrip) {
    const auto &p = param_;
    std::cout << "==== " << p.case_name << " ==== rows=" << p.rows << " cols=" << p.cols
              << " nnz=" << p.nnz << " row_off=" << p.row_off_type
              << " col_ind=" << p.col_ind_type << " dtype=" << p.dtype
              << " idx_base=" << p.idx_base << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");

    // 1. NPU 执行：GenerateCsrData -> CreateCsr -> CsrGet accessor 全字段读回
    auto npu = CsrGetNpu(env_->stream(), p.rows, p.cols, p.nnz, p.idx_base, p.isFp16(), p.seed,
                         p.row_off_type, p.col_ind_type);

    ASSERT_EQ(npu.createRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCreateCsr failed";
    ASSERT_EQ(npu.getRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCsrGet failed";

    // 2. 断言 Get 读回的标量元数据 == Create 传入
    EXPECT_EQ(npu.rows, p.rows);
    EXPECT_EQ(npu.cols, p.cols);
    EXPECT_EQ(npu.nnz, p.nnz);

    // 3. 断言双索引类型独立 round-trip（csrRowOffsetsType / csrColIndType）
    EXPECT_EQ(npu.rowOffsetsType, (p.row_off_type == "64I") ? ACL_SPARSE_INDEX_64I
                                                            : ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(npu.colIndType, (p.col_ind_type == "64I") ? ACL_SPARSE_INDEX_64I
                                                        : ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(npu.idxBase, (p.idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE
                                             : ACL_SPARSE_INDEX_BASE_ZERO);
    EXPECT_EQ(npu.valueType, p.isFp16() ? ACL_FLOAT16 : ACL_FLOAT);

    // 4. 断言 Get 读回的设备指针 == Create 传入的设备指针（accessor 语义）
    EXPECT_TRUE(npu.devPtrMatch) << "CsrGet device pointer mismatch";

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    CsrGetCases,
    CsrGetTest,
    testing::ValuesIn(GetCasesFromCsv<CsrGetParam>("csr_get_test.csv")),
    [](const testing::TestParamInfo<CsrGetParam> &info) {
        return info.param.case_name;
    });

// ============================================================================
// Exception test fixture: CsrGetExceptionTest
// ============================================================================
class CsrGetExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    // 2x2、nnz=2 的公共 host 数据（accessor 不读内容，仅需合法指针）
    std::vector<int32_t> rowOff_{0, 1, 2};   // CSR 行偏移（长度 rows+1）/ CSC 列偏移
    std::vector<int32_t> rowIdx_{0, 1};      // COO 行索引 / CSC 行索引（长度 nnz）
    std::vector<int32_t> colIdx_{0, 1};      // COO/CSR 列索引（长度 nnz）
    std::vector<float> values_{1.0F, 2.0F};  // values（长度 nnz）

    void SetUp() override {}
};

// E1: nullptr descriptor handle -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrGetExceptionTest, NullSpMatDescrHandle) {
    CsrGetOut out;
    EXPECT_EQ(aclsparseCsrGet(nullptr, &out.rows, &out.cols, &out.nnz, &out.rowOffPtr, &out.colIndPtr, &out.valuesPtr,
                              &out.rowOffType, &out.colIndType, &out.idxBase, &out.valueType),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E2: const 变体 nullptr descriptor handle -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrGetExceptionTest, ConstNullSpMatDescrHandle) {
    ConstCsrGetOut out;
    EXPECT_EQ(aclsparseConstCsrGet(nullptr, &out.rows, &out.cols, &out.nnz, &out.rowOffPtr, &out.colIndPtr, &out.valuesPtr,
                                   &out.rowOffType, &out.colIndType, &out.idxBase, &out.valueType),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E3: nullptr out handle for CreateCsr -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrGetExceptionTest, CreateCsrNullOutHandle) {
    EXPECT_EQ(aclsparseCreateCsr(nullptr, 2, 2, 2, rowOff_.data(), colIdx_.data(),
                                 values_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E4: negative dims -> ACL_SPARSE_STATUS_INVALID_VALUE（descr 保持 nullptr，无泄漏）
TEST_F(CsrGetExceptionTest, CreateCsrNegativeDim) {
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateCsr(&descr, -1, 2, 2, rowOff_.data(), colIdx_.data(),
                                 values_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr);
}

// E5: 非法 csrRowOffsetsType（static_cast 非法枚举值）-> ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(CsrGetExceptionTest, CreateCsrInvalidIndexType) {
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateCsr(&descr, 2, 2, 2, rowOff_.data(), colIdx_.data(),
                                 values_.data(), static_cast<aclsparseIndexType_t>(99),
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(descr, nullptr);
}

// E6: 非法 valueType（ACL_DT_UNDEFINED）-> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrGetExceptionTest, CreateCsrInvalidValueType) {
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateCsr(&descr, 2, 2, 2, rowOff_.data(), colIdx_.data(),
                                 values_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_DT_UNDEFINED),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr);
}

// E7: 非法 idxBase（static_cast 非法枚举值）-> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrGetExceptionTest, CreateCsrInvalidIndexBase) {
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateCsr(&descr, 2, 2, 2, rowOff_.data(), colIdx_.data(),
                                 values_.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 static_cast<aclsparseIndexBase_t>(7), ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr);
}

// E8: CsrGet on a COO descriptor -> ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
TEST_F(CsrGetExceptionTest, CsrGetOnCooDescr) {
    auto mgr = SpMatManager::createCoo(2, 2, 2, rowIdx_.data(), colIdx_.data(), values_.data(),
                                       ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                       ACL_FLOAT);
    CsrGetOut out;
    EXPECT_EQ(aclsparseCsrGet(mgr.get(), &out.rows, &out.cols, &out.nnz, &out.rowOffPtr, &out.colIndPtr, &out.valuesPtr,
                             &out.rowOffType, &out.colIndType, &out.idxBase, &out.valueType),
              ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}

// E9: CsrGet on a CSC descriptor -> ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
TEST_F(CsrGetExceptionTest, CsrGetOnCscDescr) {
    auto mgr = SpMatManager::createCsc(2, 2, 2, rowOff_.data(), rowIdx_.data(), values_.data(),
                                       ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CsrGetOut out;
    EXPECT_EQ(aclsparseCsrGet(mgr.get(), &out.rows, &out.cols, &out.nnz, &out.rowOffPtr, &out.colIndPtr, &out.valuesPtr,
                             &out.rowOffType, &out.colIndType, &out.idxBase, &out.valueType),
              ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}

// E10: 全部 10 个 OUT 为 nullptr（合法部分读取的极限形态）-> SUCCESS，不崩溃
TEST_F(CsrGetExceptionTest, AllOutNullptr) {
    auto mgr = SpMatManager::createCsr(2, 2, 2, rowOff_.data(), colIdx_.data(), values_.data(),
                                       ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    EXPECT_EQ(aclsparseCsrGet(mgr.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr),
              ACL_SPARSE_STATUS_SUCCESS);
}

// E11: 部分 OUT 为 nullptr（同一描述符顺序 4 种请求模式）：
//   a) 仅 rows/cols/nnz；b) 仅三个指针；c) 仅类型元数据四元组；d) 仅 nnz。
// 每次均 SUCCESS，且仅请求字段被正确回写（哨兵初值检测写入确实发生）。
TEST_F(CsrGetExceptionTest, PartialOutNullptr) {
    // 4x4、nnz=4、32I/64I、BASE_ONE（字段值可预测）
    std::vector<int32_t> rowOff{1, 2, 3, 4, 5};
    std::vector<int64_t> colInd{1, 2, 3, 4};
    std::vector<float> vals{1.0F, 2.0F, 3.0F, 4.0F};
    auto dRow = DeviceBuffer::copyFrom(rowOff.data(), rowOff.size() * sizeof(int32_t));
    auto dCol = DeviceBuffer::copyFrom(colInd.data(), colInd.size() * sizeof(int64_t));
    auto dVal = DeviceBuffer::copyFrom(vals.data(), vals.size() * sizeof(float));
    auto mgr = SpMatManager::createCsr(4, 4, 4, dRow.get(), dCol.get(), dVal.get(),
                                       ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_64I,
                                       ACL_SPARSE_INDEX_BASE_ONE, ACL_FLOAT);

    // a) 仅标量三元组 rows/cols/nnz
    int64_t rows = -1;
    int64_t cols = -1;
    int64_t nnz = -1;
    EXPECT_EQ(aclsparseCsrGet(mgr.get(), &rows, &cols, &nnz, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(rows, 4);
    EXPECT_EQ(cols, 4);
    EXPECT_EQ(nnz, 4);

    // b) 仅三个指针（读回应与 Create 传入的设备指针逐字一致）
    void *rowOffPtr = nullptr;
    void *colIndPtr = nullptr;
    void *valuesPtr = nullptr;
    EXPECT_EQ(aclsparseCsrGet(mgr.get(), nullptr, nullptr, nullptr, &rowOffPtr, &colIndPtr,
                              &valuesPtr, nullptr, nullptr, nullptr, nullptr),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(rowOffPtr, dRow.get());
    EXPECT_EQ(colIndPtr, dCol.get());
    EXPECT_EQ(valuesPtr, dVal.get());

    // c) 仅类型元数据四元组（哨兵初值 != 期望值，检测写入确实发生）
    aclsparseIndexType_t rowOffType = ACL_SPARSE_INDEX_64I;
    aclsparseIndexType_t colIndType = ACL_SPARSE_INDEX_64I;
    aclsparseIndexBase_t idxBase = ACL_SPARSE_INDEX_BASE_ZERO;
    aclDataType valueType = ACL_DT_UNDEFINED;
    EXPECT_EQ(aclsparseCsrGet(mgr.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              &rowOffType, &colIndType, &idxBase, &valueType),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(rowOffType, ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(colIndType, ACL_SPARSE_INDEX_64I);
    EXPECT_EQ(idxBase, ACL_SPARSE_INDEX_BASE_ONE);
    EXPECT_EQ(valueType, ACL_FLOAT);

    // d) 仅 nnz 单字段
    int64_t nnzOnly = -1;
    EXPECT_EQ(aclsparseCsrGet(mgr.get(), nullptr, nullptr, &nnzOnly, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(nnzOnly, 4);
}

// E12: Create 传三个 nullptr 指针（延迟绑定）后 CsrGet -> SUCCESS，读回 nullptr
TEST_F(CsrGetExceptionTest, DelayedBindingNullPtrs) {
    auto mgr = SpMatManager::createCsr(4, 4, 4, nullptr, nullptr, nullptr, ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                       ACL_FLOAT);
    CsrGetOut out;
    out.rowOffPtr = reinterpret_cast<void *>(0x1);  // 哨兵初值，检测 Get 确实回写 nullptr
    out.colIndPtr = reinterpret_cast<void *>(0x1);
    out.valuesPtr = reinterpret_cast<void *>(0x1);
    EXPECT_EQ(aclsparseCsrGet(mgr.get(), &out.rows, &out.cols, &out.nnz, &out.rowOffPtr, &out.colIndPtr, &out.valuesPtr,
                             &out.rowOffType, &out.colIndType, &out.idxBase, &out.valueType),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(out.rows, 4);
    EXPECT_EQ(out.cols, 4);
    EXPECT_EQ(out.nnz, 4);
    EXPECT_EQ(out.rowOffPtr, nullptr);
    EXPECT_EQ(out.colIndPtr, nullptr);
    EXPECT_EQ(out.valuesPtr, nullptr);
    EXPECT_EQ(out.rowOffType, ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(out.colIndType, ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(out.idxBase, ACL_SPARSE_INDEX_BASE_ZERO);
    EXPECT_EQ(out.valueType, ACL_FLOAT);
}

// E13: valueType 全集 round-trip（IsValidSparseValueType 的 17 个合法类型逐一
// Create+Get，dummy host 缓冲仅需非空以区分，accessor 不读内容）
TEST_F(CsrGetExceptionTest, ValueTypeRoundTripAllValid) {
    const aclDataType kAllValueTypes[] = {
        ACL_FLOAT,         ACL_FLOAT16,       ACL_DOUBLE,      ACL_INT8,
        ACL_INT16,         ACL_INT32,         ACL_INT64,       ACL_UINT8,
        ACL_UINT16,        ACL_UINT32,        ACL_UINT64,      ACL_BF16,
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2,   ACL_FLOAT4_E2M1, ACL_COMPLEX64,
        ACL_COMPLEX128};
    const int kTypeCount = static_cast<int>(sizeof(kAllValueTypes) / sizeof(kAllValueTypes[0]));
    std::vector<float> dummyRowOff(5, 0.0F);
    std::vector<float> dummyColInd(4, 0.0F);
    std::vector<float> dummyVals(4, 0.0F);
    for (int i = 0; i < kTypeCount; i++) {
        const aclDataType want = kAllValueTypes[i];
        auto mgr = SpMatManager::createCsr(4, 4, 4, dummyRowOff.data(), dummyColInd.data(),
                                           dummyVals.data(), ACL_SPARSE_INDEX_32I,
                                           ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                           want);
        CsrGetOut out;
        out.valueType = ACL_DT_UNDEFINED;  // 哨兵异于全部 17 个合法 valueType
        EXPECT_EQ(aclsparseCsrGet(mgr.get(), &out.rows, &out.cols, &out.nnz, &out.rowOffPtr, &out.colIndPtr, &out.valuesPtr,
                                 &out.rowOffType, &out.colIndType, &out.idxBase, &out.valueType),
                  ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(out.valueType, want);
        EXPECT_EQ(out.rows, 4);
        EXPECT_EQ(out.cols, 4);
        EXPECT_EQ(out.nnz, 4);
        EXPECT_EQ(out.rowOffPtr, dummyRowOff.data());
        EXPECT_EQ(out.colIndPtr, dummyColInd.data());
        EXPECT_EQ(out.valuesPtr, dummyVals.data());
    }
}

// E14: const 变体全字段 round-trip（CreateConstCsr -> ConstCsrGet，host 指针对比，
//      混合索引 64I/32I + BASE_ONE + ACL_FLOAT）
TEST_F(CsrGetExceptionTest, ConstCsrGetRoundTrip) {
    std::vector<int64_t> rowOff{1, 2, 3};  // 64I 行偏移
    std::vector<int32_t> colInd{1, 2};     // 32I 列索引
    std::vector<float> vals{1.0F, 2.0F};
    auto mgr = SpMatManager::createConstCsr(2, 2, 2, rowOff.data(), colInd.data(), vals.data(),
                                            ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_32I,
                                            ACL_SPARSE_INDEX_BASE_ONE, ACL_FLOAT);
    ConstCsrGetOut out;
    EXPECT_EQ(aclsparseConstCsrGet(mgr.cget(), &out.rows, &out.cols, &out.nnz, &out.rowOffPtr, &out.colIndPtr, &out.valuesPtr,
                                  &out.rowOffType, &out.colIndType, &out.idxBase, &out.valueType),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(out.rows, 2);
    EXPECT_EQ(out.cols, 2);
    EXPECT_EQ(out.nnz, 2);
    EXPECT_EQ(out.rowOffType, ACL_SPARSE_INDEX_64I);
    EXPECT_EQ(out.colIndType, ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(out.idxBase, ACL_SPARSE_INDEX_BASE_ONE);
    EXPECT_EQ(out.valueType, ACL_FLOAT);
    EXPECT_EQ(out.rowOffPtr, rowOff.data());
    EXPECT_EQ(out.colIndPtr, colInd.data());
    EXPECT_EQ(out.valuesPtr, vals.data());
}

// ============================================================================
// Whitebox fixture: CsrGetWhiteboxTest
// 依据 sparse/csr_get/arch35/csr_get_host.cpp 的实际控制流补充白盒用例：
//   - format 守卫分支的其余可构造非 CSR 枚举值（SLICED_ELL / BLOCKED_ELL，
//     其中 SLICED_ELL 经 const 变体触发，覆盖 CsrGetImpl<const void**> 实例化）
//   - 10 个 OUT 判空回写分支的逐字段隔离（单字段请求 + 哨兵检测未请求字段
//     不被误写）
//   - const 变体的散布 nullptr 与全 nullptr 请求路径
//   - 写回分支的 0 值与 int64 上边界（描述符内部 uint64 存储，Get 侧
//     static_cast<int64_t> 回读）
// ============================================================================
class CsrGetWhiteboxTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;

    // OUT 字段选择位（与 aclsparseCsrGet 的 10 个 OUT 形参顺序一一对应）
    enum OutField : uint16_t {
        kFldRows = 1u << 0,
        kFldCols = 1u << 1,
        kFldNnz = 1u << 2,
        kFldRowOffPtr = 1u << 3,
        kFldColIndPtr = 1u << 4,
        kFldValuesPtr = 1u << 5,
        kFldRowOffType = 1u << 6,
        kFldColIndType = 1u << 7,
        kFldIdxBase = 1u << 8,
        kFldValueType = 1u << 9,
    };

    // Get 回写目标（哨兵初值见 ResetOutSentinels，均异于各白盒用例期望值）
    int64_t rows_ = -1;
    int64_t cols_ = -1;
    int64_t nnz_ = -1;
    void *rowOffPtr_ = nullptr;
    void *colIndPtr_ = nullptr;
    void *valuesPtr_ = nullptr;
    aclsparseIndexType_t rowOffType_ = ACL_SPARSE_INDEX_64I;
    aclsparseIndexType_t colIndType_ = ACL_SPARSE_INDEX_32I;
    aclsparseIndexBase_t idxBase_ = ACL_SPARSE_INDEX_BASE_ZERO;
    aclDataType valueType_ = ACL_DT_UNDEFINED;

    void ResetOutSentinels()
    {
        rows_ = -1;
        cols_ = -1;
        nnz_ = -1;
        rowOffPtr_ = nullptr;
        colIndPtr_ = nullptr;
        valuesPtr_ = nullptr;
        rowOffType_ = ACL_SPARSE_INDEX_64I;
        colIndType_ = ACL_SPARSE_INDEX_32I;
        idxBase_ = ACL_SPARSE_INDEX_BASE_ZERO;
        valueType_ = ACL_DT_UNDEFINED;
    }

    // 按 mask 逐位选择请求的 OUT 字段（未选中位传 nullptr），返回状态码。
    // 逐位驱动 csr_get_host.cpp 中 10 个独立的 OUT 判空回写分支。
    aclsparseStatus_t MaskedCsrGet(aclsparseSpMatDescr_t d, uint16_t mask)
    {
        ResetOutSentinels();
        return aclsparseCsrGet(d, (mask & kFldRows) != 0 ? &rows_ : nullptr,
                               (mask & kFldCols) != 0 ? &cols_ : nullptr,
                               (mask & kFldNnz) != 0 ? &nnz_ : nullptr,
                               (mask & kFldRowOffPtr) != 0 ? &rowOffPtr_ : nullptr,
                               (mask & kFldColIndPtr) != 0 ? &colIndPtr_ : nullptr,
                               (mask & kFldValuesPtr) != 0 ? &valuesPtr_ : nullptr,
                               (mask & kFldRowOffType) != 0 ? &rowOffType_ : nullptr,
                               (mask & kFldColIndType) != 0 ? &colIndType_ : nullptr,
                               (mask & kFldIdxBase) != 0 ? &idxBase_ : nullptr,
                               (mask & kFldValueType) != 0 ? &valueType_ : nullptr);
    }
};

// W1: SLICED_ELL 格式描述符经 const 变体拒绝：补齐 format 守卫分支
// （d->format != ACL_SPARSE_FORMAT_CSR）在 CsrGetImpl<const void**>
// 实例化上的真值路径（黑盒 E08/E09 仅覆盖非 const 实例化的 COO/CSC）。
TEST_F(CsrGetWhiteboxTest, ConstCsrGetOnSlicedEllDescr) {
    std::vector<int32_t> slicePtr{0, 2, 4};   // numSlices + 1 = 3 项
    std::vector<int32_t> colInd{0, 1, 1, 2};  // sliceNnz * numSlices = 4 项
    std::vector<float> vals{1.0F, 2.0F, 3.0F, 4.0F};
    auto mgr = SpMatManager::createConstSlicedEll(3, 4, 4, slicePtr.data(), colInd.data(),
                                                  vals.data(), 2, 2, ACL_SPARSE_INDEX_32I,
                                                  ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    ConstCsrGetOut out;
    EXPECT_EQ(aclsparseConstCsrGet(mgr.cget(), &out.rows, &out.cols, &out.nnz, &out.rowOffPtr, &out.colIndPtr, &out.valuesPtr,
                                  &out.rowOffType, &out.colIndType, &out.idxBase, &out.valueType),
              ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}

// W2: BLOCKED_ELL 格式描述符拒绝：format 守卫分支的第 3 种可构造非 CSR 枚举值。
// 2x4 矩阵、块大小 2、块列数 4（blockRows*slots=1*2 个 slot，每 slot 4 标量）。
TEST_F(CsrGetWhiteboxTest, CsrGetOnBlockedEllDescr) {
    std::vector<int32_t> ellColInd{0, 2};  // block 索引，2 项
    std::vector<float> ellValue(8, 1.0F);  // 2 slot * 2x2 block = 8 项
    BlockedEllDescrGuard guard(2, 4, 2, 4, ellColInd.data(), ellValue.data(),
                               ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CsrGetOut out;
    EXPECT_EQ(aclsparseCsrGet(guard.get(), &out.rows, &out.cols, &out.nnz, &out.rowOffPtr, &out.colIndPtr, &out.valuesPtr,
                             &out.rowOffType, &out.colIndType, &out.idxBase, &out.valueType),
              ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}

// W3: 10 个 OUT 判空回写分支逐字段隔离：同一描述符上 10 次调用，每次仅请求
// 一个 OUT 字段；请求字段回写 Create 传入值，其余 9 个字段保持哨兵（未被误写）。
TEST_F(CsrGetWhiteboxTest, EachOutFieldIsolated) {
    std::vector<int32_t> rowOff{1, 2, 3, 4, 5, 6, 7, 8};  // rows + 1 = 8 项
    std::vector<int64_t> colInd{1, 2, 3};                 // nnz = 3 项
    std::vector<float> vals{1.0F, 2.0F, 3.0F};
    auto mgr = SpMatManager::createCsr(7, 5, 3, rowOff.data(), colInd.data(), vals.data(),
                                       ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_64I,
                                       ACL_SPARSE_INDEX_BASE_ONE, ACL_BF16);
    const uint16_t fields[] = {kFldRows,       kFldCols,       kFldNnz,
                               kFldRowOffPtr,  kFldColIndPtr,  kFldValuesPtr,
                               kFldRowOffType, kFldColIndType, kFldIdxBase,
                               kFldValueType};
    const int fieldCount = static_cast<int>(sizeof(fields) / sizeof(fields[0]));
    for (int i = 0; i < fieldCount; i++) {
        SCOPED_TRACE(testing::Message() << "OutField index " << i);
        EXPECT_EQ(MaskedCsrGet(mgr.get(), fields[i]), ACL_SPARSE_STATUS_SUCCESS);
        // 仅请求字段被回写为期望值，其余字段保持哨兵初值
        EXPECT_EQ(rows_, fields[i] == kFldRows ? 7 : -1);
        EXPECT_EQ(cols_, fields[i] == kFldCols ? 5 : -1);
        EXPECT_EQ(nnz_, fields[i] == kFldNnz ? 3 : -1);
        EXPECT_EQ(rowOffPtr_, fields[i] == kFldRowOffPtr ? static_cast<void *>(rowOff.data()) : nullptr);
        EXPECT_EQ(colIndPtr_, fields[i] == kFldColIndPtr ? static_cast<void *>(colInd.data()) : nullptr);
        EXPECT_EQ(valuesPtr_, fields[i] == kFldValuesPtr ? static_cast<void *>(vals.data()) : nullptr);
        EXPECT_EQ(rowOffType_, fields[i] == kFldRowOffType ? ACL_SPARSE_INDEX_32I : ACL_SPARSE_INDEX_64I);
        EXPECT_EQ(colIndType_, fields[i] == kFldColIndType ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I);
        EXPECT_EQ(idxBase_, fields[i] == kFldIdxBase ? ACL_SPARSE_INDEX_BASE_ONE
                                                : ACL_SPARSE_INDEX_BASE_ZERO);
        EXPECT_EQ(valueType_, fields[i] == kFldValueType ? ACL_BF16 : ACL_DT_UNDEFINED);
    }
}

// W4: const 变体的散布 nullptr 请求模式与全 nullptr 路径：两组散布模式互补，
// 覆盖 CsrGetImpl<const void**> 实例化上 10 个 OUT 的写/跳过交错组合。
TEST_F(CsrGetWhiteboxTest, ConstScatteredOutMask) {
    std::vector<int64_t> rowOff{0, 1, 2};
    std::vector<int32_t> colInd{0, 1};
    std::vector<float> vals{1.0F, 2.0F};
    auto mgr = SpMatManager::createConstCsr(2, 2, 2, rowOff.data(), colInd.data(), vals.data(),
                                            ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_32I,
                                            ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT16);

    // 模式 1（散布）：rows / nnz / csrColInd / csrRowOffsetsType / idxBase
    // 哨兵覆盖：rot/cit/ib/vt 异于期望值（检测写入确实发生）
    ConstCsrGetOut out;
    out.rowOffType = ACL_SPARSE_INDEX_32I;
    out.colIndType = ACL_SPARSE_INDEX_64I;
    out.idxBase = ACL_SPARSE_INDEX_BASE_ONE;
    out.valueType = ACL_DT_UNDEFINED;
    EXPECT_EQ(aclsparseConstCsrGet(mgr.cget(), &out.rows, nullptr, &out.nnz, nullptr, &out.colIndPtr, nullptr,
                                   &out.rowOffType, nullptr, &out.idxBase, nullptr),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(out.rows, 2);
    EXPECT_EQ(out.nnz, 2);
    EXPECT_EQ(out.colIndPtr, colInd.data());
    EXPECT_EQ(out.rowOffType, ACL_SPARSE_INDEX_64I);
    EXPECT_EQ(out.idxBase, ACL_SPARSE_INDEX_BASE_ZERO);

    // 模式 2（互补散布）：cols / csrRowOffsets / csrValues / csrColIndType / valueType
    // （复用模式 1 未请求字段的哨兵初值）
    EXPECT_EQ(aclsparseConstCsrGet(mgr.cget(), nullptr, &out.cols, nullptr, &out.rowOffPtr, nullptr,
                                   &out.valuesPtr, nullptr, &out.colIndType, nullptr, &out.valueType),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(out.cols, 2);
    EXPECT_EQ(out.rowOffPtr, rowOff.data());
    EXPECT_EQ(out.valuesPtr, vals.data());
    EXPECT_EQ(out.colIndType, ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(out.valueType, ACL_FLOAT16);

    // const 变体全 nullptr（全跳过路径，镜像 E10 的非 const 用例）
    EXPECT_EQ(aclsparseConstCsrGet(mgr.cget(), nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr),
              ACL_SPARSE_STATUS_SUCCESS);
}

// W5: 写回分支的 0 值边界：0x0x0 维度合法（Create 仅要求 >= 0），三指针延迟
// 绑定 nullptr；Get 读回全 0 维与 nullptr 指针（哨兵初值检测回写发生）。
TEST_F(CsrGetWhiteboxTest, ZeroDimRoundTrip) {
    auto mgr = SpMatManager::createCsr(0, 0, 0, nullptr, nullptr, nullptr,
                                       ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_BASE_ONE, ACL_DOUBLE);
    CsrGetOut out;
    out.rowOffPtr = reinterpret_cast<void *>(0x1);  // 哨兵初值，检测 Get 确实回写 nullptr
    out.colIndPtr = reinterpret_cast<void *>(0x1);
    out.valuesPtr = reinterpret_cast<void *>(0x1);
    out.colIndType = ACL_SPARSE_INDEX_64I;
    out.valueType = ACL_DT_UNDEFINED;
    EXPECT_EQ(aclsparseCsrGet(mgr.get(), &out.rows, &out.cols, &out.nnz, &out.rowOffPtr, &out.colIndPtr, &out.valuesPtr,
                             &out.rowOffType, &out.colIndType, &out.idxBase, &out.valueType),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(out.rows, 0);
    EXPECT_EQ(out.cols, 0);
    EXPECT_EQ(out.nnz, 0);
    EXPECT_EQ(out.rowOffPtr, nullptr);
    EXPECT_EQ(out.colIndPtr, nullptr);
    EXPECT_EQ(out.valuesPtr, nullptr);
    EXPECT_EQ(out.rowOffType, ACL_SPARSE_INDEX_64I);
    EXPECT_EQ(out.colIndType, ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(out.idxBase, ACL_SPARSE_INDEX_BASE_ONE);
    EXPECT_EQ(out.valueType, ACL_DOUBLE);
}

// W6: int64 上边界 round-trip：描述符内部以 uint64 存储 rows/cols/nnz，Get 侧
// static_cast<int64_t> 回读（Create 入参为 int64 且校验 >= 0，上界即 INT64_MAX，
// 转换恒保值）；延迟绑定三指针 nullptr，不分配设备内存。
TEST_F(CsrGetWhiteboxTest, LargeDimsRoundTrip) {
    constexpr int64_t kRows = INT64_MAX;
    constexpr int64_t kCols = 4611686018427387904LL;  // 2^62
    constexpr int64_t kNnz = 2305843009213693951LL;   // 2^61 - 1
    auto mgr = SpMatManager::createCsr(kRows, kCols, kNnz, nullptr, nullptr, nullptr,
                                       ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_64I,
                                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CsrGetOut out;
    EXPECT_EQ(aclsparseCsrGet(mgr.get(), &out.rows, &out.cols, &out.nnz, &out.rowOffPtr, &out.colIndPtr, &out.valuesPtr,
                             &out.rowOffType, &out.colIndType, &out.idxBase, &out.valueType),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(out.rows, kRows);
    EXPECT_EQ(out.cols, kCols);
    EXPECT_EQ(out.nnz, kNnz);
    EXPECT_EQ(out.rowOffPtr, nullptr);
    EXPECT_EQ(out.colIndPtr, nullptr);
    EXPECT_EQ(out.valuesPtr, nullptr);
    EXPECT_EQ(out.rowOffType, ACL_SPARSE_INDEX_64I);
    EXPECT_EQ(out.colIndType, ACL_SPARSE_INDEX_64I);
    EXPECT_EQ(out.idxBase, ACL_SPARSE_INDEX_BASE_ZERO);
    EXPECT_EQ(out.valueType, ACL_FLOAT);
}

// W7: 签名守卫：误传非稀疏矩阵描述符句柄（异类句柄 signature 不匹配）时拒绝，
// 返回 ACL_SPARSE_STATUS_INVALID_VALUE。参照 gather RejectsInvalidDescriptorObjects
// 先例：临时破坏 signature 模拟异类句柄，验证后恢复并复核正常路径。两个变体
// 共用 CsrGetImpl，非 const / const 实例化各命中一次该分支。
TEST_F(CsrGetWhiteboxTest, CsrGetOnInvalidSignatureDescr) {
    auto mgr = SpMatManager::createCsr(4, 4, 4, nullptr, nullptr, nullptr);
    const auto savedSignature = mgr.get()->signature;
    mgr.get()->signature = 0;
    EXPECT_EQ(aclsparseCsrGet(mgr.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseConstCsrGet(mgr.cget(), nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    mgr.get()->signature = savedSignature;

    // 恢复签名后同一句柄可正常读回（守卫不误伤合法描述符）
    int64_t rows = -1;
    int64_t cols = -1;
    int64_t nnz = -1;
    EXPECT_EQ(aclsparseCsrGet(mgr.get(), &rows, &cols, &nnz, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(rows, 4);
    EXPECT_EQ(cols, 4);
    EXPECT_EQ(nnz, 4);
}
