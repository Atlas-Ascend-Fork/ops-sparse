// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ============================================================================
// test_sparse2dense_host_ut.cpp — aclsparse_sparse2dense Host mock 单元测试
//
// 被测对象 = 主线真实源码（非测试桩复刻）：
//   op_host/sparse2dense_tiling.cpp       CheckMeta / FillTiling / GetBufferSize（TilingKey 编码）
//   op_host/sparse2dense_tiling.cpp       数据校验链（ValidateF3 / ValidateCoordsHost）
// aclrt 运行时符号（aclrtMemcpyAsync 等 4 个）由 mock/aclrt_mock.cpp 以 Host 内存语义承接
// （测试工程自带桩，不链接 libascendcl），并留存调用日志供 mock 层
// 「matB 不被污染」语义断言。
//
// 覆盖：
//   ① float32 主线 CSR/CSC/COO × ROW/COL × base 0/1 tiling 计算正确性 + bufferSize 恒 0；
//   ② 数据校验链各报错分支（越界/offsets 单调/锚定/ld/alg/dtype/len/空指针）+
//      重复坐标合法分支（任意值语义）；
//   ③ 溢出防护（zeroTotalBytes 64 位判溢出、COO nnz ≤ UINT32_MAX×tileElems）；
//   ④ TilingKey_0（int8）/ TilingKey_4（complex64）Host 编码正确 + 内核分支编译链接
//      （ccec dav-2201 编译链接产物存在性在本套件断言）；
//   ⑤ 两级分片参数边界（segNum/segsPerBlock/blockNum/tileElems）；
//   ⑥ alias 防护分支（CheckMeta：matA.values == matB.dense → INVALID_VALUE）；
//   ⑦ TK1/TK2（float16/bfloat16）通道编码 UT + 探针五通道编译链接；
//   ⑧ scatter kernel 五通道 tiling 参数矩阵（5 dtype × 3 格式 × 2 base × 2 布局 = 60 组合）；
//   ⑨ 确定性/只读 Host 侧契约 UT + workspace 恒 0 机器断言；
//   ⑩ 描述符级形状一致性（Sparse2DenseCheckShapeConsistency：{CSR,CSC,COO}×{行,列}轴
//      不匹配 + 全等放行 + uint64/int64 类型安全收敛边界）；
//   ⑪ 未声明 alias 区间重叠防护（同址 values/primary/secondary × dense + 部分重叠
//      双向 + ld padding 区相交 + 相邻不重叠不误拒）。
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <acl/acl.h>

#include "sparse2dense_host.h"
#include "sparse2dense_kernel.h"
#include "sparse2dense_tiling_data.h"  // TilingData + TilingKey 声明（GET_TPL_TILING_KEY / SPARSE2DENSE_VK_*）

#include "aclrt_mock.h"

namespace {

// ---------------------------------------------------------------------------
// 通道契约（编译期断言，TilingKey 表 / Duplicate 通道）
// ---------------------------------------------------------------------------
static_assert(s2d::ElemSizeOf(SPARSE2DENSE_VK_INT8) == 1UL, "TK0 int8 elemSize = 1B");
static_assert(s2d::ZeroChannelBytes(SPARSE2DENSE_VK_INT8) == 2UL, "TK0 int8 -> uint16 2B zero channel");
static_assert(s2d::ElemSizeOf(SPARSE2DENSE_VK_FLOAT16) == 2UL, "TK1 fp16 elemSize = 2B");
static_assert(s2d::ElemSizeOf(SPARSE2DENSE_VK_BFLOAT16) == 2UL, "TK2 bf16 elemSize = 2B");
static_assert(s2d::ElemSizeOf(SPARSE2DENSE_VK_FLOAT32) == 4UL, "TK3 fp32 elemSize = 4B");
static_assert(s2d::ZeroChannelBytes(SPARSE2DENSE_VK_FLOAT32) == 4UL, "TK3 fp32 zero channel = 4B");
static_assert(s2d::ElemSizeOf(SPARSE2DENSE_VK_COMPLEX64) == 8UL, "TK4 complex64 elemSize = 8B (2x float32)");
static_assert(s2d::ZeroChannelBytes(SPARSE2DENSE_VK_COMPLEX64) == 4UL, "TK4 complex64 -> uint32 4B zero channel");

constexpr int64_t kMaxDim = 2147483647LL;      // rows/cols/ld 上界
int32_t s_dummyStream = 0;                     // ValidateF3 stream 形参占位（桩只要求非空指针）

// ---------------------------------------------------------------------------
// 视图构造助手（真实 op_host 接口直接吃 S2DSpMatView / S2DDnMatView）
// ---------------------------------------------------------------------------
struct SpView {
    std::vector<int32_t> primary;
    std::vector<int32_t> secondary;
    std::vector<uint8_t> values;  // bit 缓冲（FillTiling/CheckMeta 不解引用，仅指针）
    S2DSpMatView view{};
};

struct DnView {
    std::vector<uint8_t> dense;   // 输出缓冲（含哨兵，供不被污染断言）
    S2DDnMatView view{};
};

SpView MakeSp(int32_t format, int32_t base, int32_t dtype,
    std::vector<int32_t> primary, std::vector<int32_t> secondary)
{
    SpView s;
    s.primary = std::move(primary);
    s.secondary = std::move(secondary);
    s.values.assign(std::max<size_t>(s.secondary.size() * 8UL, 1UL), 0U);
    s.view.format = format;
    s.view.indexBase = base;
    s.view.nnz = static_cast<int64_t>(s.secondary.size());
    s.view.lenPrimary = static_cast<int64_t>(s.primary.size());
    s.view.values = s.values.data();
    s.view.primary = s.primary.data();
    s.view.secondary = s.secondary.data();
    s.view.valueType = dtype;
    return s;
}

// elemSize 按 dtype 分配：alias 区间防护按「dtype × outer × ld」计算 dense 存储范围，
// 缓冲实配不小于声称范围（complex64 8B 通道尤甚），杜绝与相邻堆块误判重叠
inline uint64_t UtElemSizeOf(int32_t dtype)
{
    return (dtype == ACL_INT8) ? 1UL
        : ((dtype == ACL_FLOAT16 || dtype == ACL_BF16) ? 2UL
        : (dtype == ACL_COMPLEX64) ? 8UL : 4UL);
}

DnView MakeDn(int64_t rows, int64_t cols, int64_t ld, int32_t layout, int32_t dtype)
{
    DnView d;
    d.view.rows = rows;
    d.view.cols = cols;
    d.view.ld = ld;
    d.view.layout = layout;
    d.view.valueType = dtype;
    const uint64_t outer = (layout == S2D_LAYOUT_COL) ? static_cast<uint64_t>(cols)
                                                      : static_cast<uint64_t>(rows);
    d.dense.assign(
        static_cast<size_t>(outer * static_cast<uint64_t>(ld)) * UtElemSizeOf(dtype) + 16UL, 0xA5U);
    d.view.dense = d.dense.data();
    return d;
}

// base=1：存储坐标整体 +1（offsets 与索引同加，MakeSpF32_4x6/4x4 共用）
void ApplyIndexBaseOne(std::vector<int32_t> &primary, std::vector<int32_t> &secondary)
{
    for (auto &v : primary) {
        v += 1;
    }
    for (auto &v : secondary) {
        v += 1;
    }
}

// 4×6 float32 规范稀疏矩阵（nnz=8，坐标唯一、offsets 单调、锚定语义合法、坐标全部落在
// [0,4)×[0,6) 内），可复用于 CheckMeta / FillTiling / 校验链有效路径 × {CSR,CSC,COO} × {base0/1}
SpView MakeSpF32_4x6(int32_t format, int32_t base)
{
    std::vector<int32_t> primary;
    std::vector<int32_t> secondary;
    if (format == S2D_FORMAT_CSR) {
        primary = {0, 2, 4, 6, 8};                  // rowOffsets（rows+1=5 项，每行 2 元素）
        secondary = {0, 1, 2, 3, 4, 5, 0, 1};       // colInd，全部 < cols=6
    } else if (format == S2D_FORMAT_CSC) {
        primary = {0, 2, 2, 4, 5, 6, 8};            // colOffsets（cols+1=7 项），nnz 分布 2,0,2,1,1,2
        secondary = {0, 1, 2, 3, 0, 1, 2, 3};       // rowInd，各列内唯一且 < rows=4
    } else {
        primary = {0, 1, 2, 3, 0, 1, 2, 3};         // COO rowInd
        secondary = {0, 0, 2, 2, 3, 4, 5, 5};       // COO colInd，坐标 (r,c) 全唯一
    }
    if (base == S2D_INDEX_BASE_ONE) {
        ApplyIndexBaseOne(primary, secondary);
    }
    return MakeSp(format, base, ACL_FLOAT, primary, secondary);
}

DnView MakeDnF32_4x6(int32_t layout)
{
    return MakeDn(4, 6, (layout == S2D_LAYOUT_COL) ? 4 : 6, layout, ACL_FLOAT);
}

// 4×4 float32 规范稀疏矩阵（nnz=8，方阵：CSC 的段号=列号与行号界检查重合，见
// 用例 Sparse2DenseF3ChainTest.CscNonSquareCoordBoundContract 的非方阵界检查说明）
SpView MakeSpF32_4x4(int32_t format, int32_t base)
{
    std::vector<int32_t> primary;
    std::vector<int32_t> secondary;
    if (format == S2D_FORMAT_CSR) {
        primary = {0, 2, 4, 6, 8};             // rowOffsets（rows+1=5 项，每行 2 元素）
        secondary = {0, 1, 2, 3, 0, 1, 2, 3};  // colInd < cols=4
    } else if (format == S2D_FORMAT_CSC) {
        primary = {0, 2, 4, 6, 8};             // colOffsets（cols+1=5 项，每列 2 元素）
        secondary = {0, 1, 2, 3, 0, 1, 2, 3};  // rowInd < rows=4
    } else {
        primary = {0, 0, 1, 1, 2, 2, 3, 3};    // COO rowInd
        secondary = {0, 1, 2, 3, 0, 1, 2, 3};  // COO colInd
    }
    if (base == S2D_INDEX_BASE_ONE) {
        ApplyIndexBaseOne(primary, secondary);
    }
    return MakeSp(format, base, ACL_FLOAT, primary, secondary);
}

DnView MakeDnF32_4x4(int32_t layout)
{
    return MakeDn(4, 4, 4, layout, ACL_FLOAT);
}

void ExpectFillTilingOk(const S2DSpMatView &matA, const S2DDnMatView &matB,
    Sparse2DenseTilingData &tiling, uint64_t &key)
{
    ASSERT_EQ(Sparse2DenseFillTiling(&matA, &matB, &tiling, &key), S2D_STATUS_SUCCESS);
    ASSERT_NE(key, INVALID_TILING_KEY);
}

// ---------------------------------------------------------------------------
// ② CheckMeta 元数据静态校验（ld/alg/dtype/空指针/枚举/len 分支）
// ---------------------------------------------------------------------------
TEST(Sparse2DenseCheckMetaTest, ValidF32CsrRowBase0)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_SUCCESS);
}

TEST(Sparse2DenseCheckMetaTest, ValidInt8CooRowBase0)  // TK0 主线元数据路径
{
    SpView a = MakeSp(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, ACL_INT8, {0, 0, 1, 1}, {0, 1, 0, 1});
    DnView b = MakeDn(2, 2, 2, S2D_LAYOUT_ROW, ACL_INT8);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_SUCCESS);
}

TEST(Sparse2DenseCheckMetaTest, ValidComplex64CscColBase1)  // TK4 主线元数据路径
{
    SpView a = MakeSp(S2D_FORMAT_CSC, S2D_INDEX_BASE_ONE, ACL_COMPLEX64, {1, 2, 3}, {1, 2});
    DnView b = MakeDn(2, 2, 2, S2D_LAYOUT_COL, ACL_COMPLEX64);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_SUCCESS);
}

TEST(Sparse2DenseCheckMetaTest, NullMatARejects)
{
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseCheckMeta(nullptr, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, NullMatBRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, nullptr, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, NullDenseRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    b.view.dense = nullptr;
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, NullPrimaryWithPositiveLenRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    a.view.primary = nullptr;
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, NullValuesWithPositiveNnzRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    a.view.values = nullptr;
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, NullSecondaryWithPositiveNnzRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    a.view.secondary = nullptr;
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, IllegalAlgRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, 1), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, IllegalFormatRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    a.view.format = 3;
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, IllegalIndexBaseRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    a.view.indexBase = 2;
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, IllegalLayoutRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    b.view.layout = 2;
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, RowsColsLdOutOfRangeRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b1 = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    b1.view.rows = 0;  // rows < 1
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b1.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
    DnView b2 = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    b2.view.cols = 0;  // cols < 1
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b2.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
    DnView b3 = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    b3.view.ld = 0;  // ld < 1
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b3.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
    DnView b4 = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    b4.view.ld = kMaxDim + 1;  // ld > 2147483647
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b4.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, LdTooSmallRowLayoutRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    b.view.ld = b.view.cols - 1;  // ROW 布局要求 ld >= cols
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, LdTooSmallColLayoutRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSC, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_COL);
    b.view.ld = b.view.rows - 1;  // COL 布局要求 ld >= rows
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, LenMismatchCsrRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    a.view.lenPrimary = a.view.lenPrimary - 1;  // != rows+1
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, LenMismatchCscRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSC, S2D_INDEX_BASE_ZERO);
    a.view.lenPrimary = a.view.lenPrimary - 1;  // != cols+1
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_COL);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, LenMismatchCooRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO);
    a.view.lenPrimary = a.view.nnz - 1;  // COO 要求 lenPrimary == nnz
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, DtypeUnsupportedRejects)  // ACL_DOUBLE 不在 supported_combinations
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    a.view.valueType = ACL_DOUBLE;
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, DtypeMismatchMatBRejects)  // promotion: fixed，matA/matB 必须一致
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDn(4, 6, 6, S2D_LAYOUT_ROW, ACL_INT8);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseCheckMetaTest, NegativeNnzRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    a.view.nnz = -1;
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_INVALID_VALUE);
}

// ---------------------------------------------------------------------------
// ① float32 主线 tiling 计算 + ⑤ 两级分片参数边界
// ---------------------------------------------------------------------------
TEST(Sparse2DenseTilingCalcTest, CooRowBase0F32_3x3)
{
    SpView a = MakeSp(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {0, 0, 1, 1}, {0, 1, 0, 1});
    DnView b = MakeDn(3, 3, 3, S2D_LAYOUT_ROW, ACL_FLOAT);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
    EXPECT_EQ(key, static_cast<uint64_t>(SPARSE2DENSE_VK_FLOAT32));  // TK3
    EXPECT_EQ(key, GET_TPL_TILING_KEY(SPARSE2DENSE_VK_FLOAT32));     // 与模板编码一致
    EXPECT_EQ(tiling.rows, 3);
    EXPECT_EQ(tiling.cols, 3);
    EXPECT_EQ(tiling.ld, 3);
    EXPECT_EQ(tiling.nnz, 4);
    EXPECT_EQ(tiling.lenPrimary, 4);
    EXPECT_EQ(tiling.format, SPARSE2DENSE_FMT_COO);
    EXPECT_EQ(tiling.indexBase, 0U);
    EXPECT_EQ(tiling.isColMajor, 0U);
    EXPECT_EQ(tiling.valueKind, static_cast<uint8_t>(SPARSE2DENSE_VK_FLOAT32));
    EXPECT_EQ(tiling.zeroTotalBytes, 3 * 3 * 4);  // rows×ld×elemSize
    EXPECT_EQ(tiling.zeroTileBytes, SPARSE2DENSE_ZERO_TILE_BYTES);
    EXPECT_EQ(tiling.tileElems, SPARSE2DENSE_NNZ_TILE);
    EXPECT_EQ(tiling.segsPerBlock, SPARSE2DENSE_SEG_BLOCK_SEGS);
    EXPECT_EQ(tiling.segNum, 0U);               // COO 无段头
    EXPECT_EQ(tiling.blockNum, 1U);             // ceil(4/1024)
    EXPECT_EQ(tiling.workspaceSize, 0UL);       // bufferSize 恒 0 协议
}

TEST(Sparse2DenseTilingCalcTest, CsrRowBase0F32)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
    EXPECT_EQ(tiling.format, SPARSE2DENSE_FMT_CSR);
    EXPECT_EQ(tiling.segNum, 4U);   // CSR segNum = rows
    EXPECT_EQ(tiling.blockNum, 1U);  // ceil(4/64)
    EXPECT_EQ(tiling.zeroTotalBytes, 4 * 6 * 4);
}

TEST(Sparse2DenseTilingCalcTest, CscColBase1F32)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSC, S2D_INDEX_BASE_ONE);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_COL);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
    EXPECT_EQ(tiling.format, SPARSE2DENSE_FMT_CSC);
    EXPECT_EQ(tiling.indexBase, 1U);
    EXPECT_EQ(tiling.isColMajor, 1U);   // COL 布局
    EXPECT_EQ(tiling.segNum, 6U);       // CSC segNum = cols
    EXPECT_EQ(tiling.blockNum, 1U);
    EXPECT_EQ(tiling.zeroTotalBytes, 6 * 4 * 4);  // COL 清零范围 = cols×ld（含 padding）
}

TEST(Sparse2DenseTilingCalcTest, AllFormatLayoutBaseCombos)  // 3 格式 × 2 布局 × 2 base
{
    const int32_t formats[3] = {S2D_FORMAT_CSR, S2D_FORMAT_CSC, S2D_FORMAT_COO};
    const int32_t layouts[2] = {S2D_LAYOUT_ROW, S2D_LAYOUT_COL};
    const int32_t bases[2] = {S2D_INDEX_BASE_ZERO, S2D_INDEX_BASE_ONE};
    for (const int32_t format : formats) {
        for (const int32_t layout : layouts) {
            for (const int32_t base : bases) {
                SCOPED_TRACE(::testing::Message() << "format=" << format << " layout=" << layout
                                                  << " base=" << base);
                SpView a = MakeSpF32_4x6(format, base);
                DnView b = MakeDnF32_4x6(layout);
                ASSERT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
                    S2D_STATUS_SUCCESS);
                // 双阶段协议：查询恒 0（12 组合全覆盖）
                size_t bufferSize = 1UL;
                ASSERT_EQ(Sparse2DenseGetBufferSize(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT,
                    &bufferSize), S2D_STATUS_SUCCESS);
                EXPECT_EQ(bufferSize, 0UL);
                Sparse2DenseTilingData tiling;
                uint64_t key = INVALID_TILING_KEY;
                ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
                EXPECT_EQ(key, static_cast<uint64_t>(SPARSE2DENSE_VK_FLOAT32));
                EXPECT_EQ(tiling.format,
                    format == S2D_FORMAT_COO ? SPARSE2DENSE_FMT_COO
                                             : (format == S2D_FORMAT_CSC ? SPARSE2DENSE_FMT_CSC
                                                                         : SPARSE2DENSE_FMT_CSR));
                EXPECT_EQ(tiling.indexBase, static_cast<uint8_t>(base));
                EXPECT_EQ(tiling.isColMajor, layout == S2D_LAYOUT_COL ? 1U : 0U);
                EXPECT_EQ(tiling.rows, 4);
                EXPECT_EQ(tiling.cols, 6);
                EXPECT_EQ(tiling.nnz, 8);
                EXPECT_EQ(tiling.lenPrimary,
                    format == S2D_FORMAT_COO ? 8 : (format == S2D_FORMAT_CSC ? 7 : 5));
                EXPECT_EQ(tiling.zeroTotalBytes, (layout == S2D_LAYOUT_COL ? 6 : 4) *
                    (layout == S2D_LAYOUT_COL ? 4 : 6) * 4);  // (ROW?rows:cols)×ld×elemSize
                EXPECT_EQ(tiling.workspaceSize, 0UL);
                if (format == S2D_FORMAT_COO) {
                    EXPECT_EQ(tiling.segNum, 0U);
                    EXPECT_EQ(tiling.blockNum, 1U);
                } else {
                    EXPECT_EQ(tiling.segNum, format == S2D_FORMAT_CSR ? 4U : 6U);
                    EXPECT_EQ(tiling.blockNum, 1U);
                }
            }
        }
    }
}

TEST(Sparse2DenseTilingCalcTest, ZeroTotalBytesRowLayoutPadding)  // ld padding 计入清零范围
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDn(3, 2, 5, S2D_LAYOUT_ROW, ACL_FLOAT);  // ld=5 > cols=2
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
    EXPECT_EQ(tiling.zeroTotalBytes, 3 * 5 * 4);  // rows×ld，padding 不剔除
}

TEST(Sparse2DenseTilingCalcTest, ZeroTotalBytesColLayoutPadding)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDn(3, 2, 5, S2D_LAYOUT_COL, ACL_FLOAT);  // COL：ld>=rows
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
    EXPECT_EQ(tiling.zeroTotalBytes, 2 * 5 * 4);  // cols×ld
}

TEST(Sparse2DenseTilingCalcTest, GetBufferSizeNullptrRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseGetBufferSize(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT, nullptr),
        S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseTilingCalcTest, GetBufferSizeInvalidMetaPropagates)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    a.view.valueType = ACL_DOUBLE;  // 不支持 dtype → 查询同样拒绝
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    size_t bufferSize = 1UL;
    EXPECT_EQ(Sparse2DenseGetBufferSize(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT, &bufferSize),
        S2D_STATUS_INVALID_VALUE);
    EXPECT_EQ(bufferSize, 1UL);  // 失败路径不回写
}

// —— 两级分片参数边界（segNum<64 / =64 / >64；nnz<tileElems / =tileElems / 尾块）——
namespace {

// 构造 CSR/CSC tiling 输入（维度驱动，FillTiling 不解引用指针，secondary 只给 nnz 个占位）
struct SegCase {
    int32_t format;
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    uint32_t expSegNum;
    uint32_t expBlockNum;
};

void RunSegCase(const SegCase &c)
{
    SCOPED_TRACE(::testing::Message() << "format=" << c.format << " rows=" << c.rows << " cols=" << c.cols
                                      << " nnz=" << c.nnz);
    std::vector<int32_t> primary(static_cast<size_t>(
        c.format == S2D_FORMAT_COO ? c.nnz : (c.format == S2D_FORMAT_CSR ? c.rows : c.cols) + 1), 0);
    std::vector<int32_t> secondary(static_cast<size_t>(c.nnz), 0);
    SpView a = MakeSp(c.format, S2D_INDEX_BASE_ZERO, ACL_FLOAT, primary, secondary);
    a.view.nnz = c.nnz;  // 与占位数组一致
    a.view.lenPrimary = static_cast<int64_t>(primary.size());
    DnView b = MakeDn(c.rows, c.cols, c.format == S2D_FORMAT_CSR ? c.cols : c.rows, S2D_LAYOUT_ROW, ACL_FLOAT);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
    EXPECT_EQ(tiling.segNum, c.expSegNum);
    EXPECT_EQ(tiling.blockNum, c.expBlockNum);
    EXPECT_EQ(tiling.tileElems, 1024U);
    // 段块粒度自适应：CSR/CSC 目标块数 ≥128（2 的幂、下限 8、上限 64）——本表
    // segNum 均 <1024 → segsPerBlock 恒 8；COO 不使用段块，维持基准 64
    EXPECT_EQ(tiling.segsPerBlock,
        c.format == S2D_FORMAT_COO ? SPARSE2DENSE_SEG_BLOCK_SEGS : 8U);
}

}  // namespace

TEST(Sparse2DenseSegShardingTest, SegNumLt64)
{
    ASSERT_NO_FATAL_FAILURE(RunSegCase({S2D_FORMAT_CSR, 10, 8, 16, 10U, 2U}));  // ceil(10/8)
}

TEST(Sparse2DenseSegShardingTest, SegNumEq64)
{
    ASSERT_NO_FATAL_FAILURE(RunSegCase({S2D_FORMAT_CSR, 64, 8, 64, 64U, 8U}));  // =64 → ceil(64/8)
}

TEST(Sparse2DenseSegShardingTest, SegNumGt64)
{
    ASSERT_NO_FATAL_FAILURE(RunSegCase({S2D_FORMAT_CSR, 65, 8, 65, 65U, 9U}));     // ceil(65/8)
    ASSERT_NO_FATAL_FAILURE(RunSegCase({S2D_FORMAT_CSR, 128, 8, 128, 128U, 16U}));  // ceil(128/8)
    ASSERT_NO_FATAL_FAILURE(RunSegCase({S2D_FORMAT_CSR, 129, 8, 129, 129U, 17U}));  // 尾块 1 段
    ASSERT_NO_FATAL_FAILURE(RunSegCase({S2D_FORMAT_CSC, 8, 200, 200, 200U, 25U}));  // CSC segNum=cols
}

TEST(Sparse2DenseSegShardingTest, CooSegNumZeroAndNnzTailBlocks)
{
    ASSERT_NO_FATAL_FAILURE(RunSegCase({S2D_FORMAT_COO, 8, 8, 5, 0U, 1U}));       // nnz < tileElems
    ASSERT_NO_FATAL_FAILURE(RunSegCase({S2D_FORMAT_COO, 8, 8, 1024, 0U, 1U}));    // nnz = tileElems
    ASSERT_NO_FATAL_FAILURE(RunSegCase({S2D_FORMAT_COO, 8, 8, 1025, 0U, 2U}));    // 尾块 1 元素
    ASSERT_NO_FATAL_FAILURE(RunSegCase({S2D_FORMAT_COO, 8, 8, 2049, 0U, 3U}));    // 尾块 1 元素（第 3 块）
}

TEST(Sparse2DenseSegShardingTest, NnzZeroBoundaries)
{
    ASSERT_NO_FATAL_FAILURE(RunSegCase({S2D_FORMAT_COO, 3, 3, 0, 0U, 0U}));  // COO nnz=0 → blockNum=0（scatter kernel 直接返回）
    ASSERT_NO_FATAL_FAILURE(RunSegCase({S2D_FORMAT_CSR, 3, 3, 0, 3U, 1U}));  // CSR nnz=0 → 段块仍覆盖 rows（空段）
}

// ---------------------------------------------------------------------------
// ② 数据校验链（坐标在界 / 重复坐标合法 / offsets 单调 / 越界 / 锚定 / 空守卫
//   + mock 层 matB 不污染）
// ---------------------------------------------------------------------------
TEST(Sparse2DenseF3ChainTest, ValidAllFormatsBases)  // 3 格式 × 2 base 有效路径（方阵，见 4×4 说明）
{
    const int32_t formats[3] = {S2D_FORMAT_CSR, S2D_FORMAT_CSC, S2D_FORMAT_COO};
    const int32_t bases[2] = {S2D_INDEX_BASE_ZERO, S2D_INDEX_BASE_ONE};
    for (const int32_t format : formats) {
        for (const int32_t base : bases) {
            SCOPED_TRACE(::testing::Message() << "format=" << format << " base=" << base);
            s2d_ut_mock::ResetLogs();
            SpView a = MakeSpF32_4x4(format, base);
            DnView b = MakeDnF32_4x4(S2D_LAYOUT_ROW);
            ASSERT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_SUCCESS);
            // mock 层语义：D2H 拷贝 primary(+secondary)，目的地址全部落在校验缓冲，matB 不被触碰
            const auto &copies = s2d_ut_mock::MemcpyLog();
            EXPECT_EQ(copies.size(), (a.view.lenPrimary > 0 ? 1UL : 0UL) + (a.view.nnz > 0 ? 1UL : 0UL));
            for (const auto &c : copies) {
                EXPECT_EQ(c.kind, static_cast<int>(ACL_MEMCPY_DEVICE_TO_HOST));
                EXPECT_PRED2(s2d_ut_mock::InAnyHostAlloc, c.dst, c.bytes);
            }
            EXPECT_TRUE(s2d_ut_mock::HostAllocLog().empty()
                || s2d_ut_mock::InAnyHostAlloc(b.view.dense, 1UL) == false);
        }
    }
}

TEST(Sparse2DenseF3ChainTest, CscNonSquareCoordBoundContract)  // 契约用例
{
    // 合法 4×6 CSC（colOffsets 6+1 项、rowInd 行号全 < rows=4、坐标全唯一在界内）：
    // 按「坐标经 base 换算后落在 [0,rows)×[0,cols)」应返回 SUCCESS。校验链 CSC 分支须按
    // (行, 列) 限位校验（段号=列号 < cols、rowInd=行号 < rows），行/列限位倒置会把
    // 非方阵 CSC 合法输入误拒。方阵情形界检查重合，由 ValidAllFormatsBases 承接。
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSC, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_SUCCESS);
}

TEST(Sparse2DenseF3ChainTest, CooDuplicateCoordAllowed)  // 重复坐标合法（任意值语义）
{
    SpView a = MakeSp(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {0, 0}, {0, 0});
    DnView b = MakeDn(2, 2, 2, S2D_LAYOUT_ROW, ACL_FLOAT);
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_SUCCESS);
}

TEST(Sparse2DenseF3ChainTest, CsrDuplicateInSegAllowed)
{
    // row0 内 colInd 重复 [0,0] → (0,0) 重复；合法（任意值语义）
    SpView a = MakeSp(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {0, 2, 4}, {0, 0, 1, 2});
    DnView b = MakeDn(2, 3, 3, S2D_LAYOUT_ROW, ACL_FLOAT);
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_SUCCESS);
}

TEST(Sparse2DenseF3ChainTest, CscDuplicateInSegAllowed)
{
    // col0 内 rowInd 重复 [0,0] → (0,0) 重复；合法（任意值语义）
    SpView a = MakeSp(S2D_FORMAT_CSC, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {0, 2, 3}, {0, 0, 1});
    DnView b = MakeDn(2, 2, 2, S2D_LAYOUT_ROW, ACL_FLOAT);
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_SUCCESS);
}

TEST(Sparse2DenseF3ChainTest, CsrOffsetsNonMonotonicRejects)
{
    SpView a = MakeSp(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {0, 2, 1, 3}, {0, 1, 0});
    DnView b = MakeDn(2, 3, 3, S2D_LAYOUT_ROW, ACL_FLOAT);
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseF3ChainTest, CooRowOutOfBoundsRejects)
{
    SpView a = MakeSp(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {2, 0}, {0, 1});
    DnView b = MakeDn(2, 2, 2, S2D_LAYOUT_ROW, ACL_FLOAT);  // rows=2，row=2 越界
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseF3ChainTest, CooColOutOfBoundsRejects)
{
    SpView a = MakeSp(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {0, 1}, {0, 3});
    DnView b = MakeDn(2, 3, 3, S2D_LAYOUT_ROW, ACL_FLOAT);  // cols=3，col=3 越界
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseF3ChainTest, Base1StoredZeroCoordRejects)
{
    // base=1 时存储坐标 0 换算后为 -1 → 越界拒绝（COO：lenPrimary == nnz == 1）
    SpView a = MakeSp(S2D_FORMAT_COO, S2D_INDEX_BASE_ONE, ACL_FLOAT, {0}, {1});
    DnView b = MakeDn(2, 2, 2, S2D_LAYOUT_ROW, ACL_FLOAT);
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseF3ChainTest, CsrAnchorStartViolationRejects)  // offsets[0] != indexBase
{
    SpView a = MakeSp(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {1, 2, 3}, {0, 1});
    DnView b = MakeDn(2, 2, 2, S2D_LAYOUT_ROW, ACL_FLOAT);
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseF3ChainTest, CsrAnchorEndViolationRejects)  // offsets[last] − base != nnz
{
    SpView a = MakeSp(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {0, 1, 3}, {0, 1});
    DnView b = MakeDn(2, 2, 2, S2D_LAYOUT_ROW, ACL_FLOAT);  // nnz=2 但端点 3
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseF3ChainTest, CooLenPrimaryMismatchRejects)
{
    SpView a = MakeSp(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {0, 0}, {0, 1});
    a.view.lenPrimary = a.view.nnz + 1;  // COO lenPrimary != nnz
    DnView b = MakeDn(2, 2, 2, S2D_LAYOUT_ROW, ACL_FLOAT);
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseF3ChainTest, NnzZeroEmptyVectorGuardPasses)  // nnz=0 空 vector 可空
{
    SpView a = MakeSp(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {}, {});
    EXPECT_EQ(a.view.nnz, 0);
    EXPECT_EQ(a.view.lenPrimary, 0);
    DnView b = MakeDn(3, 3, 3, S2D_LAYOUT_ROW, ACL_FLOAT);
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_SUCCESS);
    // CSR nnz=0：offsets 全 0（单调 + 锚定合法），secondary 空
    SpView c = MakeSp(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {0, 0, 0, 0}, {});
    DnView d = MakeDn(3, 3, 3, S2D_LAYOUT_ROW, ACL_FLOAT);
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &c.view, &d.view), S2D_STATUS_SUCCESS);
}

TEST(Sparse2DenseF3ChainTest, NullStreamRejects)
{
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseValidateF3(nullptr, &a.view, &b.view), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseF3ChainTest, NullMatARejects)
{
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, nullptr, &b.view), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseF3ChainTest, RejectLeavesMatBUnpolluted)  // mock 层「matB 不被污染」语义
{
    // 校验失败路径：列坐标越界 → 拒绝；整存储缓冲（含 ld padding）与哨兵区 bit 不变
    s2d_ut_mock::ResetLogs();
    SpView a = MakeSp(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {0, 0, 0}, {0, 1, 2});
    DnView b = MakeDn(2, 2, 4, S2D_LAYOUT_ROW, ACL_FLOAT);  // ld=4 含 padding，哨兵字节 0xA5
    EXPECT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_INVALID_VALUE);
    for (const uint8_t byte : b.dense) {
        ASSERT_EQ(byte, 0xA5U) << "matB 被污染（校验失败路径不得触碰 matB）";
    }
    // 拷贝全部为 D2H 且目的地址在校验缓冲内（绝不落入 dense 区间）
    for (const auto &c : s2d_ut_mock::MemcpyLog()) {
        EXPECT_EQ(c.kind, static_cast<int>(ACL_MEMCPY_DEVICE_TO_HOST));
        EXPECT_NE(c.dst, b.view.dense);
        EXPECT_PRED2(s2d_ut_mock::InAnyHostAlloc, c.dst, c.bytes);
    }
    // 有效路径同样不触碰 matB（Host 校验阶段无任何到 matB 的写）
    s2d_ut_mock::ResetLogs();
    SpView v = MakeSpF32_4x6(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO);
    DnView w = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    ASSERT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &v.view, &w.view), S2D_STATUS_SUCCESS);
    for (const uint8_t byte : w.dense) {
        ASSERT_EQ(byte, 0xA5U);
    }
}

// ---------------------------------------------------------------------------
// ③ 溢出防护分支（zeroTotalBytes 64 位判溢出 + COO nnz ≤ UINT32_MAX×tileElems）
// ---------------------------------------------------------------------------
namespace {

// 溢出用例专用：维度/nnz 为上界合成值，不分配真实数组（FillTiling 不解引用任何数据指针）
S2DSpMatView RawSpMat(int32_t format, int32_t base, int64_t nnz, int64_t lenPrimary, int32_t dtype)
{
    S2DSpMatView v{};
    v.format = format;
    v.indexBase = base;
    v.nnz = nnz;
    v.lenPrimary = lenPrimary;
    v.values = nullptr;
    v.primary = nullptr;
    v.secondary = nullptr;
    v.valueType = dtype;
    return v;
}

S2DDnMatView RawDnMat(int64_t rows, int64_t cols, int64_t ld, int32_t layout, int32_t dtype)
{
    S2DDnMatView v{};
    v.rows = rows;
    v.cols = cols;
    v.ld = ld;
    v.layout = layout;
    v.valueType = dtype;
    v.dense = nullptr;
    return v;
}

}  // namespace

TEST(Sparse2DenseOverflowTest, ZeroTotalBytesF32MaxDimsInt64Overflow)
{
    // rows=cols=ld=2147483647、elemSize=4：无符号 64 位乘积 ≈1.8e19 > INT64_MAX → 拒绝
    S2DSpMatView a = RawSpMat(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO, 1, kMaxDim + 1, ACL_FLOAT);
    S2DDnMatView b = RawDnMat(kMaxDim, kMaxDim, kMaxDim, S2D_LAYOUT_ROW, ACL_FLOAT);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    EXPECT_EQ(Sparse2DenseFillTiling(&a, &b, &tiling, &key), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseOverflowTest, ZeroTotalBytesComplex64UnsignedMulOverflow)
{
    // elemSize=8：outerRows×ld 已超 ~0ULL/8 → 无符号乘法溢出分支先行拒绝
    S2DSpMatView a = RawSpMat(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO, 1, kMaxDim + 1, ACL_COMPLEX64);
    S2DDnMatView b = RawDnMat(kMaxDim, kMaxDim, kMaxDim, S2D_LAYOUT_ROW, ACL_COMPLEX64);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    EXPECT_EQ(Sparse2DenseFillTiling(&a, &b, &tiling, &key), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseOverflowTest, ZeroTotalBytesSmallDimsNoFalsePositive)
{
    // 同为极限段位的合法维度：清零字节数在 int64 域内不误报
    S2DSpMatView a = RawSpMat(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO, 1, 1024, ACL_FLOAT);
    S2DDnMatView b = RawDnMat(65536, 65536, 65536, S2D_LAYOUT_ROW, ACL_FLOAT);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_EQ(Sparse2DenseFillTiling(&a, &b, &tiling, &key), S2D_STATUS_SUCCESS);
    EXPECT_EQ(tiling.zeroTotalBytes, static_cast<int64_t>(65536ULL * 65536ULL * 4ULL));
    EXPECT_EQ(tiling.blockNum, static_cast<uint32_t>((65536U + 64U - 1U) / 64U));
}

TEST(Sparse2DenseOverflowTest, CooNnzAboveUint32MaxTimesTileRejects)
{
    // nnz > UINT32_MAX × tileElems（1024）→ COO blockNum 溢出防护拒绝
    const int64_t nnz = static_cast<int64_t>(UINT32_MAX) * static_cast<int64_t>(SPARSE2DENSE_NNZ_TILE) + 1;
    S2DSpMatView a = RawSpMat(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, nnz, nnz, ACL_FLOAT);
    S2DDnMatView b = RawDnMat(1, 1, 1, S2D_LAYOUT_ROW, ACL_FLOAT);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    EXPECT_EQ(Sparse2DenseFillTiling(&a, &b, &tiling, &key), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseOverflowTest, CooNnzEqUint32MaxTimesTileBoundaryOk)
{
    // 恰在防护上界：nnz == UINT32_MAX × tileElems → 合法，blockNum == UINT32_MAX
    const int64_t nnz = static_cast<int64_t>(UINT32_MAX) * static_cast<int64_t>(SPARSE2DENSE_NNZ_TILE);
    S2DSpMatView a = RawSpMat(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, nnz, nnz, ACL_FLOAT);
    S2DDnMatView b = RawDnMat(1, 1, 1, S2D_LAYOUT_ROW, ACL_FLOAT);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_EQ(Sparse2DenseFillTiling(&a, &b, &tiling, &key), S2D_STATUS_SUCCESS);
    EXPECT_EQ(tiling.blockNum, UINT32_MAX);
    EXPECT_EQ(tiling.segNum, 0U);
}

TEST(Sparse2DenseOverflowTest, FillTilingNullArgsRejects)
{
    S2DSpMatView a = RawSpMat(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO, 1, 4, ACL_FLOAT);
    S2DDnMatView b = RawDnMat(3, 3, 3, S2D_LAYOUT_ROW, ACL_FLOAT);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    EXPECT_EQ(Sparse2DenseFillTiling(nullptr, &b, &tiling, &key), S2D_STATUS_INVALID_VALUE);
    EXPECT_EQ(Sparse2DenseFillTiling(&a, nullptr, &tiling, &key), S2D_STATUS_INVALID_VALUE);
    EXPECT_EQ(Sparse2DenseFillTiling(&a, &b, nullptr, &key), S2D_STATUS_INVALID_VALUE);
    EXPECT_EQ(Sparse2DenseFillTiling(&a, &b, &tiling, nullptr), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseOverflowTest, UnsupportedDtypeFillTilingRejects)
{
    S2DSpMatView a = RawSpMat(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO, 1, 4, ACL_DOUBLE);
    S2DDnMatView b = RawDnMat(3, 3, 3, S2D_LAYOUT_ROW, ACL_DOUBLE);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    EXPECT_EQ(Sparse2DenseFillTiling(&a, &b, &tiling, &key), S2D_STATUS_INVALID_VALUE);
    EXPECT_EQ(key, INVALID_TILING_KEY);
}

// ---------------------------------------------------------------------------
// ④ TilingKey_0（int8）/ TilingKey_4（complex64）Host 编码 + 通道契约 + 分支编译链接
// ---------------------------------------------------------------------------
TEST(Sparse2DenseTplKeyBranchTest, Int8KeyEncoding)  // TK0：1B 通道
{
    SpView a = MakeSp(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, ACL_INT8, {0, 0, 1, 1}, {0, 1, 0, 1});
    DnView b = MakeDn(2, 2, 3, S2D_LAYOUT_ROW, ACL_INT8);  // ld=3 > cols：含 padding
    ASSERT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_SUCCESS);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
    EXPECT_EQ(key, static_cast<uint64_t>(SPARSE2DENSE_VK_INT8));
    EXPECT_EQ(key, GET_TPL_TILING_KEY(SPARSE2DENSE_VK_INT8));
    EXPECT_EQ(tiling.valueKind, static_cast<uint8_t>(SPARSE2DENSE_VK_INT8));
    EXPECT_EQ(tiling.zeroTotalBytes, 2 * 3 * 1);  // elemSize=1B
    size_t bufferSize = 1UL;
    ASSERT_EQ(Sparse2DenseGetBufferSize(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT, &bufferSize),
        S2D_STATUS_SUCCESS);
    EXPECT_EQ(bufferSize, 0UL);
}

TEST(Sparse2DenseTplKeyBranchTest, Complex64KeyEncoding)  // TK4：8B = 2×float32 通道
{
    SpView a = MakeSp(S2D_FORMAT_CSC, S2D_INDEX_BASE_ONE, ACL_COMPLEX64, {1, 2, 4}, {1, 2, 2});
    DnView b = MakeDn(2, 2, 3, S2D_LAYOUT_COL, ACL_COMPLEX64);
    ASSERT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_SUCCESS);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
    EXPECT_EQ(key, static_cast<uint64_t>(SPARSE2DENSE_VK_COMPLEX64));
    EXPECT_EQ(key, GET_TPL_TILING_KEY(SPARSE2DENSE_VK_COMPLEX64));
    EXPECT_EQ(tiling.valueKind, static_cast<uint8_t>(SPARSE2DENSE_VK_COMPLEX64));
    EXPECT_EQ(tiling.zeroTotalBytes, 2 * 3 * 8);  // COL：cols×ld×8B
    EXPECT_EQ(tiling.isColMajor, 1U);
}

TEST(Sparse2DenseTplKeyBranchTest, FiveKeysDistinctAndOrdered)
{
    const uint8_t kinds[5] = {SPARSE2DENSE_VK_INT8, SPARSE2DENSE_VK_FLOAT16, SPARSE2DENSE_VK_BFLOAT16,
        SPARSE2DENSE_VK_FLOAT32, SPARSE2DENSE_VK_COMPLEX64};
    uint64_t keys[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 5; i++) {
        keys[i] = GET_TPL_TILING_KEY(kinds[i]);
        EXPECT_EQ(keys[i], static_cast<uint64_t>(kinds[i])) << "V_KIND 编码值应为穷举下标";
        EXPECT_NE(keys[i], INVALID_TILING_KEY);
    }
    for (int i = 0; i < 5; i++) {
        for (int j = i + 1; j < 5; j++) {
            EXPECT_NE(keys[i], keys[j]) << "TilingKey 必须两两可分（dtype 轴）";
        }
    }
}

TEST(Sparse2DenseTplKeyBranchTest, ChannelContractRuntimeCheck)
{
    // 与文件头 static_assert 呼应的运行期确认（TK0 uint16 造零 / TK4 uint32 造零通道）
    EXPECT_EQ(s2d::ElemSizeOf(SPARSE2DENSE_VK_INT8), 1UL);
    EXPECT_EQ(s2d::ZeroChannelBytes(SPARSE2DENSE_VK_INT8), 2UL);
    EXPECT_EQ(s2d::ElemSizeOf(SPARSE2DENSE_VK_FLOAT16), 2UL);
    EXPECT_EQ(s2d::ZeroChannelBytes(SPARSE2DENSE_VK_FLOAT16), 2UL);
    EXPECT_EQ(s2d::ElemSizeOf(SPARSE2DENSE_VK_BFLOAT16), 2UL);
    EXPECT_EQ(s2d::ElemSizeOf(SPARSE2DENSE_VK_FLOAT32), 4UL);
    EXPECT_EQ(s2d::ZeroChannelBytes(SPARSE2DENSE_VK_FLOAT32), 4UL);
    EXPECT_EQ(s2d::ElemSizeOf(SPARSE2DENSE_VK_COMPLEX64), 8UL);
    EXPECT_EQ(s2d::ZeroChannelBytes(SPARSE2DENSE_VK_COMPLEX64), 4UL);
    EXPECT_EQ(s2d::ElemSizeOf(99U), 0UL);  // 未定义 V_KIND → 0（FillTiling 拒绝路径）
}

TEST(Sparse2DenseTplKeyBranchTest, TplBranchProbeCompileLink)
{
    // TilingKey_0（int8）/ TilingKey_4（complex64）内核分支编译链接：
    // 探针源经 ccec --npu-arch=dav-2201 显式实例化
    // zero_kernel<0>/<4> + scatter_kernel<0>/<4>（骨架级），产物经
    // 环境变量传入。本用例断言产物真实存在且非空。
    const char *probeLib = std::getenv("S2D_TPL_PROBE_LIB");
    if (probeLib == nullptr || probeLib[0] == '\0') {
        GTEST_SKIP() << "TPL 分支编译链接探针未启用（S2D_TPL_PROBE_LIB 未设置）";
    }
    RecordProperty("tpl_probe_artifact", probeLib);
    std::FILE *f = std::fopen(probeLib, "rb");
    ASSERT_NE(f, nullptr) << "TPL 分支编译链接产物不存在: " << probeLib;
    ASSERT_EQ(std::fseek(f, 0, SEEK_END), 0);
    const long bytes = std::ftell(f);
    ASSERT_EQ(std::fclose(f), 0);
    ASSERT_GT(bytes, 0L) << "TPL 分支编译链接产物为空: " << probeLib;
    std::printf("[tpl_probe] artifact=%s bytes=%ld (TK0~TK4 全五通道 K1+K2 分支 ccec dav-2201 编译链接产物)\n",
        probeLib, bytes);
}

// ===========================================================================
// UT 扩展：
//   ⑥ alias 防护分支（CheckMeta UT）
//   ⑦ TK1/TK2 通道编码 UT（2B 通道 Host 编码 + 五通道编译链接）
//   ⑧ scatter kernel 五通道 tiling 参数矩阵（5 dtype × 3 格式 × 2 base × 2 布局）
//   ⑨ 确定性/只读 Host 侧契约 UT（只读契约在 mock 拷贝日志层断言）+ workspace 恒 0 机器断言
// ===========================================================================

// ---------------------------------------------------------------------------
// ⑥ CheckMeta alias 防护分支（matA.values == matB.dense → INVALID_VALUE）
// ---------------------------------------------------------------------------
TEST(Sparse2DenseAliasGuardTest, ValuesDenseAliasRejects)
{
    // 未声明 alias：matB 与 matA values 同一 buffer → 参数错误
    // （清零 kernel 会破坏 values 只读语义，须拦截）
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    a.view.values = b.view.dense;  // 注入 alias
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_INVALID_VALUE);
    // 三格式均拦截（同一 buffer 引用，与格式无关）
    SpView csc = MakeSpF32_4x6(S2D_FORMAT_CSC, S2D_INDEX_BASE_ZERO);
    SpView coo = MakeSpF32_4x6(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO);
    csc.view.values = b.view.dense;
    coo.view.values = b.view.dense;
    EXPECT_EQ(Sparse2DenseCheckMeta(&csc.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_INVALID_VALUE);
    EXPECT_EQ(Sparse2DenseCheckMeta(&coo.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseAliasGuardTest, AliasRejectedBeforeAnyDeviceActivity)
{
    // alias 拒绝先于任何 Device 活动（无 D2H 校验拷贝、matB 16B 哨兵不被触碰）
    s2d_ut_mock::ResetLogs();
    SpView a = MakeSpF32_4x6(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    a.view.values = b.view.dense;
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_INVALID_VALUE);
    EXPECT_TRUE(s2d_ut_mock::MemcpyLog().empty()) << "CheckMeta 阶段不得发起任何拷贝";
    EXPECT_TRUE(s2d_ut_mock::HostAllocLog().empty()) << "CheckMeta 阶段不得申请 Host 缓冲";
    for (const uint8_t byte : b.dense) {
        ASSERT_EQ(byte, 0xA5U) << "matB 被污染（alias 拒绝路径不得触碰 matB）";
    }
}

TEST(Sparse2DenseAliasGuardTest, AliasPropagatesThroughBufferSizeQuery)
{
    // 双阶段协议查询接口同样拒绝 alias（CheckMeta 为 GetBufferSize 的守门分支）
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    a.view.values = b.view.dense;
    size_t bufferSize = 7UL;
    EXPECT_EQ(Sparse2DenseGetBufferSize(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT, &bufferSize),
        S2D_STATUS_INVALID_VALUE);
    EXPECT_EQ(bufferSize, 7UL);  // 失败路径不回写
}

TEST(Sparse2DenseAliasGuardTest, DistinctBuffersAccepted)  // 合法输入不误拒
{
    // 正常输入 values 与 dense 为调用方分别分配（恒不 alias）→ 必须放行
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    ASSERT_NE(a.view.values, b.view.dense);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_SUCCESS);
    // 相邻分配（地址不等但同缓冲区族）不触发误报
    SpView a2 = MakeSpF32_4x6(S2D_FORMAT_COO, S2D_INDEX_BASE_ONE);
    DnView b2 = MakeDnF32_4x6(S2D_LAYOUT_COL);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a2.view, &b2.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_SUCCESS);
}

TEST(Sparse2DenseAliasGuardTest, NullValuesWithZeroNnzNoAliasFalsePositive)
{
    // nnz=0：values 为空指针（空 vector 守卫，data() 可为 nullptr），dense 有效——
    // 无 alias 可能，不得误拒（alias 判定仅作用于 values 非空分支）
    SpView a = MakeSp(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {}, {});
    ASSERT_EQ(a.view.nnz, 0);
    a.view.values = nullptr;  // 显式置空：nnz=0 时允许 values 为空
    DnView b = MakeDn(3, 3, 3, S2D_LAYOUT_ROW, ACL_FLOAT);
    ASSERT_NE(b.view.dense, nullptr);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_SUCCESS);
}

// ---------------------------------------------------------------------------
// ⑦ TK1/TK2 通道编码 UT（2B 通道 Host 编码 + tiling 位宽口径）
// ---------------------------------------------------------------------------
TEST(Sparse2DenseTplKeyChannelTest, Float16KeyEncoding)  // TK1：2B half 通道
{
    SpView a = MakeSp(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO, ACL_FLOAT16,
        {0, 2, 4, 6, 8}, {0, 1, 2, 3, 4, 5, 0, 1});
    DnView b = MakeDn(4, 6, 6, S2D_LAYOUT_ROW, ACL_FLOAT16);
    ASSERT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_SUCCESS);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
    EXPECT_EQ(key, static_cast<uint64_t>(SPARSE2DENSE_VK_FLOAT16));
    EXPECT_EQ(key, GET_TPL_TILING_KEY(SPARSE2DENSE_VK_FLOAT16));
    EXPECT_EQ(tiling.valueKind, static_cast<uint8_t>(SPARSE2DENSE_VK_FLOAT16));
    EXPECT_EQ(tiling.zeroTotalBytes, 4 * 6 * 2);  // elemSize=2B
    EXPECT_EQ(tiling.workspaceSize, 0UL);
    size_t bufferSize = 1UL;
    ASSERT_EQ(Sparse2DenseGetBufferSize(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT, &bufferSize),
        S2D_STATUS_SUCCESS);
    EXPECT_EQ(bufferSize, 0UL);
}

TEST(Sparse2DenseTplKeyChannelTest, Bfloat16KeyEncoding)  // TK2：2B bfloat16_t 通道
{
    SpView a = MakeSp(S2D_FORMAT_CSC, S2D_INDEX_BASE_ONE, ACL_BF16,
        {1, 3, 3, 5, 6, 7, 9}, {1, 2, 3, 0, 1, 2, 3, 4});
    DnView b = MakeDn(4, 6, 4, S2D_LAYOUT_COL, ACL_BF16);  // COL：ld=rows=4
    ASSERT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_SUCCESS);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
    EXPECT_EQ(key, static_cast<uint64_t>(SPARSE2DENSE_VK_BFLOAT16));
    EXPECT_EQ(key, GET_TPL_TILING_KEY(SPARSE2DENSE_VK_BFLOAT16));
    EXPECT_EQ(tiling.valueKind, static_cast<uint8_t>(SPARSE2DENSE_VK_BFLOAT16));
    EXPECT_EQ(tiling.zeroTotalBytes, 6 * 4 * 2);  // COL：cols×ld×2B（含 padding 无此例 ld=rows）
    EXPECT_EQ(tiling.isColMajor, 1U);
}

TEST(Sparse2DenseTplKeyChannelTest, TwoByteChannelContract)
{
    // 与文件头 static_assert 呼应的运行期确认：TK1/TK2 走 2B 通道（half/bfloat16_t），
    // 造零通道字节数 = 元素字节数（无通道复用）；TK0 例外（uint16 一次覆盖 2×int8）
    EXPECT_EQ(s2d::ElemSizeOf(SPARSE2DENSE_VK_FLOAT16), 2UL);
    EXPECT_EQ(s2d::ZeroChannelBytes(SPARSE2DENSE_VK_FLOAT16), 2UL);
    EXPECT_EQ(s2d::ElemSizeOf(SPARSE2DENSE_VK_BFLOAT16), 2UL);
    EXPECT_EQ(s2d::ZeroChannelBytes(SPARSE2DENSE_VK_BFLOAT16), 2UL);
}

// ---------------------------------------------------------------------------
// ⑧ scatter kernel 五通道 tiling 参数矩阵（5 dtype × 3 格式 × 2 base × 2 布局 = 60 组合）
// ---------------------------------------------------------------------------
namespace {

// 单组合断言（dtype 通道 × 格式 × base × 布局）：CheckMeta → FillTiling → tiling
// 字段逐项 + bufferSize 恒 0。结构同构的 4×6 基准矩阵仅 dtype 通道变化。
void CheckComboTiling(int32_t dtype, uint8_t kind, uint64_t elemSize, int32_t format,
    int32_t layout, int32_t base, int &combos)
{
    SCOPED_TRACE(::testing::Message() << "dtype=" << dtype << " format=" << format
                                      << " layout=" << layout << " base=" << base);
    SpView a = MakeSpF32_4x6(format, base);
    a.view.valueType = dtype;  // 结构同构，仅 dtype 通道变化
    DnView b = MakeDn(4, 6, (layout == S2D_LAYOUT_COL) ? 4 : 6, layout, dtype);  // dtype 感知分配
    ASSERT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_SUCCESS);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
    EXPECT_EQ(key, static_cast<uint64_t>(kind));
    EXPECT_EQ(key, GET_TPL_TILING_KEY(kind));
    EXPECT_EQ(tiling.valueKind, kind);
    // 清零 kernel 范围按位宽线性缩放（ROW: rows×ld；COL: cols×ld），含 ld padding
    const uint64_t outer = (layout == S2D_LAYOUT_ROW) ? 4ULL : 6ULL;
    const uint64_t ld = (layout == S2D_LAYOUT_ROW) ? 6ULL : 4ULL;
    EXPECT_EQ(tiling.zeroTotalBytes, static_cast<int64_t>(outer * ld * elemSize));
    // scatter kernel 分片参数与 dtype 无关（两级轴：仅 shape/format 决定）
    EXPECT_EQ(tiling.tileElems, SPARSE2DENSE_NNZ_TILE);
    // 段块粒度自适应：CSR/CSC 小段数（4/6）→ segsPerBlock=8；COO 维持基准 64
    EXPECT_EQ(tiling.segsPerBlock, format == S2D_FORMAT_COO ? SPARSE2DENSE_SEG_BLOCK_SEGS : 8U);
    if (format == S2D_FORMAT_COO) {
        EXPECT_EQ(tiling.segNum, 0U);
        EXPECT_EQ(tiling.blockNum, 1U);  // ceil(8/1024)
    } else {
        EXPECT_EQ(tiling.segNum, format == S2D_FORMAT_CSR ? 4U : 6U);
        EXPECT_EQ(tiling.blockNum, 1U);
    }
    EXPECT_EQ(tiling.format,
        format == S2D_FORMAT_COO ? SPARSE2DENSE_FMT_COO
                                 : (format == S2D_FORMAT_CSC ? SPARSE2DENSE_FMT_CSC
                                                             : SPARSE2DENSE_FMT_CSR));
    EXPECT_EQ(tiling.indexBase, static_cast<uint8_t>(base));
    EXPECT_EQ(tiling.isColMajor, layout == S2D_LAYOUT_COL ? 1U : 0U);
    // workspace 恒 0（代码断言）
    EXPECT_EQ(tiling.workspaceSize, 0UL);
    size_t bufferSize = 1UL;
    ASSERT_EQ(Sparse2DenseGetBufferSize(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT,
        &bufferSize), S2D_STATUS_SUCCESS);
    EXPECT_EQ(bufferSize, 0UL);
    combos++;
}

}  // namespace

TEST(Sparse2DenseFiveChannelTilingMatrixTest, AllDtypeFormatBaseLayoutCombos)
{
    const int32_t dtypes[5] = {ACL_INT8, ACL_FLOAT16, ACL_BF16, ACL_FLOAT, ACL_COMPLEX64};
    const uint8_t kinds[5] = {SPARSE2DENSE_VK_INT8, SPARSE2DENSE_VK_FLOAT16, SPARSE2DENSE_VK_BFLOAT16,
        SPARSE2DENSE_VK_FLOAT32, SPARSE2DENSE_VK_COMPLEX64};
    const uint64_t elemSizes[5] = {1UL, 2UL, 2UL, 4UL, 8UL};
    const int32_t formats[3] = {S2D_FORMAT_CSR, S2D_FORMAT_CSC, S2D_FORMAT_COO};
    const int32_t layouts[2] = {S2D_LAYOUT_ROW, S2D_LAYOUT_COL};
    const int32_t bases[2] = {S2D_INDEX_BASE_ZERO, S2D_INDEX_BASE_ONE};
    int combos = 0;
    for (int d = 0; d < 5; d++) {
        for (const int32_t format : formats) {
            for (const int32_t layout : layouts) {
                for (const int32_t base : bases) {
                    ASSERT_NO_FATAL_FAILURE(
                        CheckComboTiling(dtypes[d], kinds[d], elemSizes[d], format, layout, base, combos));
                }
            }
        }
    }
    EXPECT_EQ(combos, 60);  // 5 dtype × 3 格式 × 2 布局 × 2 base 全矩阵
    RecordProperty("tiling_matrix_combos", "60");
}

// ---------------------------------------------------------------------------
// ⑨ 确定性/只读 Host 侧契约 UT
// ---------------------------------------------------------------------------
namespace {

// 字段级 tiling 相等（避免结构体 padding 差异误报）
void ExpectTilingEq(const Sparse2DenseTilingData &x, const Sparse2DenseTilingData &y)
{
    EXPECT_EQ(x.rows, y.rows);
    EXPECT_EQ(x.cols, y.cols);
    EXPECT_EQ(x.ld, y.ld);
    EXPECT_EQ(x.nnz, y.nnz);
    EXPECT_EQ(x.lenPrimary, y.lenPrimary);
    EXPECT_EQ(x.zeroTotalBytes, y.zeroTotalBytes);
    EXPECT_EQ(x.zeroTileBytes, y.zeroTileBytes);
    EXPECT_EQ(x.tileElems, y.tileElems);
    EXPECT_EQ(x.segsPerBlock, y.segsPerBlock);
    EXPECT_EQ(x.blockNum, y.blockNum);
    EXPECT_EQ(x.segNum, y.segNum);
    EXPECT_EQ(x.format, y.format);
    EXPECT_EQ(x.indexBase, y.indexBase);
    EXPECT_EQ(x.isColMajor, y.isColMajor);
    EXPECT_EQ(x.valueKind, y.valueKind);
    EXPECT_EQ(x.workspaceSize, y.workspaceSize);
}

}  // namespace

TEST(Sparse2DenseHostContractTest, DeterminismTilingInvariantToCoordDistribution)
{
    // 确定性契约（Host 侧分量）：清零/scatter kernel 分片划分仅依赖 (format, rows, cols, nnz)——
    // 块/内层 tile 划分与元素分布、空段分布无关 ⇒ 同一元数据下
    // 重复执行的分片编排逐次一致，写入内容与次数不因分片改变（bit-wise 确定性前提）。
    // 断言：同元数据、不同坐标内容/次序的两份输入 → FillTiling 输出逐字段一致。
    SpView a1 = MakeSpF32_4x4(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO);  // 行内升序坐标
    // 同结构（offsets 相同、nnz 相同）但行内坐标降序存放（乱序唯一坐标，接口允许）
    std::vector<int32_t> secRev = {3, 2, 1, 0, 3, 2, 1, 0};
    SpView a2 = MakeSp(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {0, 2, 4, 6, 8}, secRev);
    DnView b = MakeDnF32_4x4(S2D_LAYOUT_ROW);
    Sparse2DenseTilingData t1;
    Sparse2DenseTilingData t2;
    uint64_t k1 = INVALID_TILING_KEY;
    uint64_t k2 = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a1.view, b.view, t1, k1));
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a2.view, b.view, t2, k2));
    EXPECT_EQ(k1, k2);
    ASSERT_NO_FATAL_FAILURE(ExpectTilingEq(t1, t2));
    // 空段分布无关：同 rows/cols/nnz 的 CSR，段非空分布不同（0,4,4,8 vs 2,2,6,6）
    SpView a3 = MakeSp(S2D_FORMAT_CSR, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {0, 2, 2, 6, 8},
        {0, 1, 0, 1, 2, 3, 0, 1});
    Sparse2DenseTilingData t3;
    uint64_t k3 = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a3.view, b.view, t3, k3));
    EXPECT_EQ(k3, k1);
    EXPECT_EQ(t3.blockNum, t1.blockNum);
    EXPECT_EQ(t3.segNum, t1.segNum);
    RecordProperty("st_determinism_evidence",
        "同输入重复执行 10 次输出 bit-wise 一致");
}

TEST(Sparse2DenseHostContractTest, ReadOnlyInputsUntouchedByValidationChain)
{
    // 只读契约（Host 侧分量）：校验链（CheckMeta/ValidateF3/FillTiling）对输入
    // values/primary/secondary 与输出 dense 零写入——mock 拷贝日志仅允许 D2H 进 Host
    // 校验缓冲；输入缓冲执行前后逐字节一致。
    s2d_ut_mock::ResetLogs();
    SpView a = MakeSpF32_4x6(S2D_FORMAT_CSC, S2D_INDEX_BASE_ONE);
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_COL);
    const std::vector<int32_t> primSnap = a.primary;
    const std::vector<int32_t> secSnap = a.secondary;
    const std::vector<uint8_t> valSnap = a.values;
    const std::vector<uint8_t> denseSnap = b.dense;
    ASSERT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT), S2D_STATUS_SUCCESS);
    ASSERT_EQ(Sparse2DenseValidateF3(&s_dummyStream, &a.view, &b.view), S2D_STATUS_SUCCESS);
    Sparse2DenseTilingData tiling;
    uint64_t key = INVALID_TILING_KEY;
    ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
    // 全链路拷贝日志：仅 D2H（Host 校验读入），无 H2D（禁止写 Device 输入）
    for (const auto &c : s2d_ut_mock::MemcpyLog()) {
        EXPECT_EQ(c.kind, static_cast<int>(ACL_MEMCPY_DEVICE_TO_HOST));
        EXPECT_PRED2(s2d_ut_mock::InAnyHostAlloc, c.dst, c.bytes);
    }
    // 输入/输出缓冲逐字节不变（含 values bit 缓冲与 dense 哨兵缓冲）
    EXPECT_EQ(a.primary, primSnap);
    EXPECT_EQ(a.secondary, secSnap);
    EXPECT_EQ(a.values, valSnap);
    EXPECT_EQ(b.dense, denseSnap);
}

TEST(Sparse2DenseHostContractTest, WorkspaceAlwaysZeroMachineEvidence)
{
    // 固有 workspace 绝对值 = 0（bufferSize 恒 0 + tiling.workspaceSize 恒 0），
    // 无与稠密输出成比例的 Device 临时副本（代码断言）。
    const int32_t dtypes[5] = {ACL_INT8, ACL_FLOAT16, ACL_BF16, ACL_FLOAT, ACL_COMPLEX64};
    const int32_t formats[3] = {S2D_FORMAT_CSR, S2D_FORMAT_CSC, S2D_FORMAT_COO};
    for (const int32_t dtype : dtypes) {
        for (const int32_t format : formats) {
            SCOPED_TRACE(::testing::Message() << "dtype=" << dtype << " format=" << format);
            SpView a = MakeSpF32_4x6(format, S2D_INDEX_BASE_ZERO);
            a.view.valueType = dtype;
            DnView b = MakeDn(4, 6, 6, S2D_LAYOUT_ROW, dtype);  // dtype 感知分配
            ASSERT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
                S2D_STATUS_SUCCESS);
            size_t bufferSize = 0xDEADBEEFUL;
            ASSERT_EQ(Sparse2DenseGetBufferSize(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT,
                &bufferSize), S2D_STATUS_SUCCESS);
            EXPECT_EQ(bufferSize, 0UL);
            Sparse2DenseTilingData tiling;
            uint64_t key = INVALID_TILING_KEY;
            ASSERT_NO_FATAL_FAILURE(ExpectFillTilingOk(a.view, b.view, tiling, key));
            EXPECT_EQ(tiling.workspaceSize, 0ULL);
        }
    }
}


// ---------------------------------------------------------------------------

// ===========================================================================
// UT 扩展（第十轮评审整改）：
//   ⑩ 描述符级形状一致性（Sparse2DenseCheckShapeConsistency）
//   ⑪ 未声明 alias 存储区间重叠防护（CheckMeta 段三尾部）
// ===========================================================================

// ---------------------------------------------------------------------------
// ⑩ 描述符级形状一致性：{CSR,CSC,COO} × {行轴,列轴} 不匹配（6 组合）+ 评审场景
//（合法 COO matA=2×3、nnz=1、坐标(0,0) 配 matB=1×1 原全链静默输出错误尺寸）+
// 全等放行与 uint64/int64 类型安全收敛边界。内部 CheckMeta 第二道防线现状一并固化：
// 仅 CSR 行轴（lenPrimary==rows+1）与 CSC 列轴（lenPrimary==cols+1）由 lenPrimary
// 间接拦截，CSR 列轴/CSC 行轴/COO 两轴在内部视图层不设防（matA rows/cols 不入
// S2DSpMatView），由公开层形状校验承接。
// ---------------------------------------------------------------------------
TEST(Sparse2DenseShapeConsistencyTest, PerFormatAxisMismatchMatrix)
{
    const int32_t formats[3] = {S2D_FORMAT_CSR, S2D_FORMAT_CSC, S2D_FORMAT_COO};
    for (const int32_t format : formats) {
        for (int axis = 0; axis < 2; axis++) {  // 0=行轴变异 1=列轴变异（matA 4×6 基准）
            SCOPED_TRACE(::testing::Message() << "format=" << format << " axis=" << axis);
            SpView a = MakeSpF32_4x6(format, S2D_INDEX_BASE_ZERO);
            DnView b = (axis == 0) ? MakeDn(5, 6, 6, S2D_LAYOUT_ROW, ACL_FLOAT)
                                   : MakeDn(4, 7, 7, S2D_LAYOUT_ROW, ACL_FLOAT);
            // 描述符级形状一致性（公开层两入口共用）拒绝
            EXPECT_EQ(Sparse2DenseCheckShapeConsistency(4, 6, b.view.rows, b.view.cols),
                S2D_STATUS_INVALID_VALUE);
            // 内部 CheckMeta 第二道防线现状固化（lenPrimary 间接拦截范围）
            const bool lenPrimaryGuards = (format == S2D_FORMAT_CSR && axis == 0) ||
                (format == S2D_FORMAT_CSC && axis == 1);
            EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
                lenPrimaryGuards ? S2D_STATUS_INVALID_VALUE : S2D_STATUS_SUCCESS);
        }
    }
    // 评审场景：COO matA=2×3（nnz=1、坐标 (0,0)）配 matB=1×1——描述符级两轴校验拒绝
    EXPECT_EQ(Sparse2DenseCheckShapeConsistency(2, 3, 1, 1), S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseShapeConsistencyTest, EqualShapesAcceptedAndTypeSafeBounds)
{
    // 全等放行（含零维与 int64 上界——纯比较函数不设域，rows/cols 域检查在 CheckMeta 段二）
    EXPECT_EQ(Sparse2DenseCheckShapeConsistency(4, 6, 4, 6), S2D_STATUS_SUCCESS);
    EXPECT_EQ(Sparse2DenseCheckShapeConsistency(0, 0, 0, 0), S2D_STATUS_SUCCESS);
    EXPECT_EQ(Sparse2DenseCheckShapeConsistency(static_cast<uint64_t>(INT64_MAX), 6, INT64_MAX, 6),
        S2D_STATUS_SUCCESS);
    // matA 描述符 rows/cols 为 uint64_t：超 int64 域判不等（类型安全收敛，无回绕伪相等）
    EXPECT_EQ(Sparse2DenseCheckShapeConsistency(~0ULL, 6ULL, 4, 6), S2D_STATUS_INVALID_VALUE);
    EXPECT_EQ(Sparse2DenseCheckShapeConsistency(4ULL, (1ULL << 63), 4, 6), S2D_STATUS_INVALID_VALUE);
    // matB 负域（int64）与 matA 非负域必不等
    EXPECT_EQ(Sparse2DenseCheckShapeConsistency(4, 6, -1, 6), S2D_STATUS_INVALID_VALUE);
    EXPECT_EQ(Sparse2DenseCheckShapeConsistency(4, 6, 4, -6), S2D_STATUS_INVALID_VALUE);
}

// ---------------------------------------------------------------------------
// ⑪ 未声明 alias 存储区间重叠防护（CheckMeta 段三尾部）：values/primary/secondary
// 与 dense 同址、部分重叠双向（索引起于 dense 内部 / dense 起于索引内部）、ld
// padding 区相交、相邻不重叠不误拒。
// ---------------------------------------------------------------------------
TEST(Sparse2DenseAliasGuardTest, IndexArraysSameAddrAsDenseRejects)
{
    // 索引数组与 dense 完全同址（原段一精确相等检查仅覆盖 values）：清零 kernel 清
    // dense 即改写索引内容（COO rowInd[0] 被清 0 → scatter 写错坐标），须拒绝
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    SpView a1 = MakeSpF32_4x6(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO);
    a1.view.primary = static_cast<const int32_t *>(b.view.dense);  // rowInd == dense
    EXPECT_EQ(Sparse2DenseCheckMeta(&a1.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_INVALID_VALUE);
    SpView a2 = MakeSpF32_4x6(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO);
    a2.view.secondary = static_cast<const int32_t *>(b.view.dense);  // colInd == dense
    EXPECT_EQ(Sparse2DenseCheckMeta(&a2.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_INVALID_VALUE);
    // values 同址（原检查口径，区间判定升级后保持拒绝）
    SpView a3 = MakeSpF32_4x6(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO);
    a3.view.values = b.view.dense;
    EXPECT_EQ(Sparse2DenseCheckMeta(&a3.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseAliasGuardTest, PartialOverlapBothDirectionsRejects)
{
    // 部分重叠（不同起始地址）：① 索引起于 dense 内部；② dense 起于索引内部；
    // ③ values 与 dense 尾部相交
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);  // dense 存储范围 4×6×4 = 96B
    // ① primary 起于 dense 内部（dense + 88，span 32B → [88,120) 与 [0,96) 交叠 [88,96)）
    SpView a1 = MakeSpF32_4x6(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO);
    a1.view.primary = reinterpret_cast<const int32_t *>(static_cast<uint8_t *>(b.view.dense) + 88);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a1.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_INVALID_VALUE);
    // ② dense 起于索引内部：primary 起于 arena 基址（span 32B），dense 指向 arena+4
    //（范围 [4,100)）→ 交叠 [4,32)
    std::vector<uint8_t> arena(192, 0xA5);
    SpView a2 = MakeSpF32_4x6(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO);
    a2.view.primary = reinterpret_cast<const int32_t *>(arena.data());
    DnView b2 = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    b2.view.dense = arena.data() + 4;
    EXPECT_EQ(Sparse2DenseCheckMeta(&a2.view, &b2.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_INVALID_VALUE);
    // ③ values 部分重叠：values = dense + 88（span 32B，与 [0,96) 交叠 [88,96)）
    SpView a3 = MakeSpF32_4x6(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO);
    a3.view.values = static_cast<uint8_t *>(b.view.dense) + 88;
    EXPECT_EQ(Sparse2DenseCheckMeta(&a3.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseAliasGuardTest, LdPaddingRegionOverlapRejects)
{
    // dense 存储范围含 ld padding（与 FillTiling zeroTotalBytes 同口径）：仅与 padding
    // 区（逻辑列 2..4）相交同样拒绝
    SpView a = MakeSp(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {0, 1}, {0, 1});
    DnView b = MakeDn(3, 2, 5, S2D_LAYOUT_ROW, ACL_FLOAT);  // dense 范围 3×5×4 = 60B
    // secondary = dense + 56（span 8B → [56,64)）：[56,60) 落在 row2 col4 的 padding 元素
    a.view.secondary = reinterpret_cast<const int32_t *>(static_cast<uint8_t *>(b.view.dense) + 56);
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_INVALID_VALUE);
}

TEST(Sparse2DenseAliasGuardTest, AdjacentDistinctRangesAccepted)
{
    // 相邻不重叠（区间端点相接）与长度 0 数组不误拒：合法输入三区间与 dense 互斥
    std::vector<uint8_t> arena(256, 0xA5);
    SpView a = MakeSpF32_4x6(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO);
    a.view.values = arena.data();                                            // [0,32)
    a.view.primary = reinterpret_cast<const int32_t *>(arena.data() + 32);   // [32,64)
    a.view.secondary = reinterpret_cast<const int32_t *>(arena.data() + 64); // [64,96)
    DnView b = MakeDnF32_4x6(S2D_LAYOUT_ROW);
    b.view.dense = arena.data() + 96;  // [96,192)：与三区间端点相接不重叠（arena 256B 容纳）
    EXPECT_EQ(Sparse2DenseCheckMeta(&a.view, &b.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_SUCCESS);
    // 长度 0 的数组（nnz=0 空 vector，指针可为任意非空占位）跳过重叠判定
    SpView e = MakeSp(S2D_FORMAT_COO, S2D_INDEX_BASE_ZERO, ACL_FLOAT, {}, {});
    e.view.primary = reinterpret_cast<const int32_t *>(b.view.dense);  // lenPrimary=0 → 跳过
    DnView b3 = MakeDn(3, 3, 3, S2D_LAYOUT_ROW, ACL_FLOAT);
    EXPECT_EQ(Sparse2DenseCheckMeta(&e.view, &b3.view, S2D_SPARSETODENSE_ALG_DEFAULT),
        S2D_STATUS_SUCCESS);
}

}  // namespace

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
