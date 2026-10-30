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

// =============================================================================
// ltmatmul 白盒负向补充用例（原 ltmatmul_test.cpp WB_L2 段拆分 TU）。
//
// 拆分动机：ltmatmul_test.cpp NBNC 超 2000（codecheck7）；本 TU 承接
// Whitebox supplement (3.3) 的 WB_L2_42~47 负向系。fixture 经
// ltmatmul_fixture.h 与主 TU 共用；batch 布局负向构造（四描述符 + 一致
// batch 属性 + matmulDescr / 溢出基座 / 128³ 执行段）经
// ltmatmul_batch_neg_helper.h 共享（与 matmulSearch X18/X19 系同源）。
// =============================================================================

#include "test_common.h"

#include "ltmatmul_npu_wrapper.h"
#include "ltmatmul_exception_helper.h"  // SparseLtMatmulCtxBase, MatmulExceptionCtx, MatmulRawParams, MatmulNpuRaw, FillMatmulRawParams
#include "ltmatmul_fixture.h"           // MatmulTestFixture（主 TU 共享）
#include "ltmatmul_batch_neg_helper.h"  // LtBatchMatDescKit / MatmulBatchOverflowCtx / RunBatchNegMatmulRaw

#include <gtest/gtest.h>

using namespace sparse_test;

// MatmulExceptionTest fixture 定义于 ltmatmul_fixture.h（主 TU 共享）。

// ============================================================================
// Whitebox supplement (3.3): branches not covered by 3.2 CSV/TEST_F cases.
// ============================================================================

// WB-L2-42: null handle → HANDLE_IS_NULLPTR
// Covers validate_matmul_params branch: `handle == nullptr || *handle == nullptr`.
// (3.2 L2_01 covered null plan, but null handle was not directly exercised.)
TEST_F(MatmulExceptionTest, WB_L2_42_NullHandle) {
    MatmulExceptionCtx ctx;
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = ctx.stream;
    FillMatmulRawParams(p, nullptr, ctx.plan.get(), &alpha,
        ctx.dA.get(), ctx.dB.get(), &beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, 1);
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// WB-L2-43: streams[0]==null → SUCCESS (nullptr is valid default stream)
// streams[0] may be nullptr — the ACL runtime treats 0/nullptr as
// the default stream, which is valid for all aclrtMemcpy / kernel-launch APIs.
// The previous test expected INVALID_VALUE, but the implementation correctly
// accepts nullptr as the default stream and proceeds with kernel launch.
TEST_F(MatmulExceptionTest, WB_L2_43_Streams0Null) {
    MatmulExceptionCtx ctx;
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream nullStream = nullptr;  // streams[0] == null (default stream)
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), &alpha,
        ctx.dA.get(), ctx.dB.get(), &beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &nullStream, 1);  // numStreams > 0, streams[0] is null (valid default)
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// WB-L2-44: matA=null B-sparse → INVALID_VALUE
// Covers prepare_matmul_pointers branch: `!md->isSparseA` (B-sparse) when matA
// is null. 3.2 L2_23 covered A-sparse + matA=null (workspace fallback),
// L2_36 (dense×dense) not supported: upstream MatmulDescriptorInit requires
// exactly one structured descriptor (both-dense returns INVALID_VALUE).
// The B-sparse + matA=null branch is covered here by WB_L2_44.
TEST_F(MatmulExceptionTest, WB_L2_44_MatANullBSparse) {
    // Build a B-sparse plan: matA dense, matB structured (2:4 sparse).
    SparseLtHandleGuard handle;
    SparseLtDnMatDescGuard matA(handle.get(), 8, 8, 8, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    SparseLtMatDescGuard matB(handle.get(), 8, 8, 8, 16, ACL_FLOAT,
                              ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    SparseLtDnMatDescGuard matC(handle.get(), 8, 8, 8, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matD(handle.get(), 8, 8, 8, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    SparseLtMatmulDescGuard md(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, matA.get(), matB.get(), matC.get(), matD.get(),
        ACL_SPARSE_COMPUTE_32F);
    SparseLtAlgSelectionGuard algSel(handle.get(), md.get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    SparseLtPlanGuard plan(handle.get(), md.get(), algSel.get());

    size_t wsSize = 0;
    ASSERT_EQ(aclsparseLtMatmulGetWorkspace(handle.get(), plan.cptr(), &wsSize),
              ACL_SPARSE_STATUS_SUCCESS);
    auto dWorkspace = sparse_test::DeviceBuffer::alloc(wsSize);
    auto dA = sparse_test::DeviceBuffer::alloc(8 * 8 * sizeof(float));
    auto dB = sparse_test::DeviceBuffer::alloc(8 * 8 * sizeof(float));
    auto dC = sparse_test::DeviceBuffer::alloc(8 * 8 * sizeof(float));
    auto dD = sparse_test::DeviceBuffer::alloc(8 * 8 * sizeof(float));

    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = 0;
    FillMatmulRawParams(p, handle.get(), plan.get(), &alpha,
        nullptr, dB.get(), &beta, dC.get(), dD.get(),
        dWorkspace.get(), &stream, 1);  // B-sparse requires explicit dense A → INVALID_VALUE
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// WB-L2-45/46: batch 布局溢出输入 → INVALID_VALUE（非 int64 溢出 UB）。
// batchStride=2^61 × elemSize=4 的乘积 / ld=2^58 的 (rows-1)*ld physSpan
// 乘加链在未受检查的 int64 乘法下均会溢出（UB）；校验层须以受检查
// 计算干净拒绝。batch 属性须在 MatmulDescriptorInit 之前设置于四个
// mat 描述符（一致性校验要求 A/B/C/D 相同）。
// 基座 MatmulBatchOverflowCtx 与共用执行段 RunBatchNegMatmulRaw 见
// ltmatmul_batch_neg_helper.h。

// WB-L2-45: batchStride=2^61（对齐通过 V2，由 V3 字节溢出拦截）
TEST_F(MatmulExceptionTest, WB_L2_45_BatchStrideOverflow) {
    MatmulBatchOverflowCtx ctx(1LL << 61, 0);
    EXPECT_EQ(RunBatchNegMatmulRaw(ctx.handle.get(), ctx.plan->get(),
                                   ctx.dWorkspace.get()),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// WB-L2-46: ld=INT32_MAX 上界值 + batchStride < V1 物理跨度 → INVALID_VALUE。
// 演变说明：原 ld=2^58 形态测 (rows-1)*ld physSpan 乘加链溢出（受检查
// span 计算拦截）；描述符层补 ld > INT32_MAX 拒绝后巨 ld 域被封死，该
// 溢出面不再可达，受检查 span 计算保留为纵深防御。
// 本用例改用上界内巨 ld（2147483644，4 对齐、描述符可创建）验证 V1 的
// span 语义拦截：span = (128-1)*ld+128 ≈ 2.7e11 > batchStride=16384
// → V1 拒绝（batchStride < 物理跨度），不触达 kernel。
TEST_F(MatmulExceptionTest, WB_L2_46_BatchLdSpanOverflow) {
    MatmulBatchOverflowCtx ctx(16384, 2147483644);
    EXPECT_EQ(RunBatchNegMatmulRaw(ctx.handle.get(), ctx.plan->get(),
                                   ctx.dWorkspace.get()),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// WB-L2-47（负向补例，D5 轮 16B 口径改参）：numBatches>1 且 biasStride
// 非 16B 逐批对齐 → INVALID_VALUE。batch≥1 的 bias GM 源
//（biasVecGM[b*biasStride]，DataCopyPad 实测容忍 16B 对齐 GM 源——16B
// 基址探针见 ltmatmul_test_align16.cpp；对齐校验对齐 cuSPARSELt 16B
// 惯例）非 16B 时仍为设备侧错误——本校验把其转化为干净拒绝。构造：四
// mat 合法 batch 布局（numBatches=2、batchStride=16384 穿过 V1/V2/V3）+
// 16B 对齐的 biasPointer（穿过 B1）+ biasStride=21（FP32 bias
// elemSize=4，要求 %4==0；21*4=84B 非 16B 倍数——原 32B 口径的 20
//（80B）在 16B 口径下合法，正向锚点由 align16 的 stride=4 探针承载）
// → B3 拦截。bias 属性在 PlanInit 前设置（正规时序）。四描述符构造经
// LtBatchMatDescKit（batch 属性一致性语义同源）。
TEST_F(MatmulExceptionTest, WB_L2_47_BiasStrideNot16BAlignedPerBatch) {
    SparseLtHandleGuard handle;
    LtBatchMatDescKit kit(handle.get(), 2, 16384, 0, "WB_L2_47_BiasStride");
    // bias：16B 对齐 device 指针（穿过 B1）+ 非对齐 stride（触发 B3）。
    auto dBias = sparse_test::DeviceBuffer::alloc(128 * sizeof(float) * 2);
    void* biasPtr = dBias.get();
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(handle.get(), &kit.matmulDesc->get(),
            ACLSPARSELT_MATMUL_BIAS_POINTER, &biasPtr, sizeof(void*)),
        ACL_SPARSE_STATUS_SUCCESS);
    const int64_t biasStride = 21;  // 21 % 4 != 0 → 84B 非 16B 倍数
    ASSERT_EQ(aclsparseLtMatmulDescSetAttribute(handle.get(), &kit.matmulDesc->get(),
            ACLSPARSELT_MATMUL_BIAS_STRIDE, &biasStride, sizeof(int64_t)),
        ACL_SPARSE_STATUS_SUCCESS);
    SparseLtAlgSelectionGuard algSel(handle.get(), kit.matmulDesc->get(),
        ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    SparseLtPlanGuard plan(handle.get(), kit.matmulDesc->get(), algSel.get());
    size_t wsSize = 0;
    ASSERT_EQ(aclsparseLtMatmulGetWorkspace(handle.get(), plan.cptr(), &wsSize),
              ACL_SPARSE_STATUS_SUCCESS);
    auto dWorkspace = sparse_test::DeviceBuffer::alloc(wsSize);
    EXPECT_EQ(RunBatchNegMatmulRaw(handle.get(), plan.get(), dWorkspace.get()),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// WB-L2-48~51 基座：C/D dtype 配对矩阵负向构造。A/B 主 dtype 与 C/D
// dtype 独立参数化——描述符层保持库基线零 dtype 校验（超集组合在
// DescriptorInit/PlanInit 被接受，与 matmulSearch X20~X23 锁定的行为
// 一致），执行层 validate_launch_dtypes 配对白名单拦截（launch 前任何
// kernel 未启动）。kernel 分派以 cDataType 为键且 OutType 与 CType 绑定，
// 域外组合对应三类风险域：按 C 宽度写 D 的越界写（C=FP16/D=INT8 → 2×；
// C=INT32/D=INT8 → 4×）、落收尾 else 按 1B 写的静默错值（C=INT8/D 非
// INT8/INT32）、非 INT8 路径按 T 读写异宽缓冲（FP16×C/D=INT32）。
// guard 链 / stream / 缓冲骨架与公共描述符构造继承自
// SparseLtMatmulCtxBase（ltmatmul_exception_helper.h，与
// MatmulExceptionCtx 同一基座——codecheck9 消重）。
struct MatmulDtypePairCtx : SparseLtMatmulCtxBase {
    // 128³：满足全 dtype structured 对齐倍数（FP32=8/FP16,BF16=16/INT8=32）。
    // 缓冲按各 dtype 元素宽度分配（校验在 launch 前拦截，kernel 不执行，
    // 内容不消费）；matA 显式传入（A-sparse Honor matA 语义，无需 prune）。
    // op/order 组合：INT8（低精度）路径 DescriptorInit 按
    // IsValidOpLayoutCombination 校验——ROW/ROW 须 (N, T)（X22 同款）；
    // 非 INT8 路径无此约束，用 (N, N)。
    MatmulDtypePairCtx(aclDataType abDtype, aclDataType cDtype, aclDataType dDtype,
                       aclsparseComputeType_t computeType,
                       size_t abEltSize, size_t cEltSize, size_t dEltSize)
        : SparseLtMatmulCtxBase(128, 128, 128, abDtype, cDtype, dDtype,
                                (abDtype == ACL_INT8) ? ACL_SPARSE_OP_TRANSPOSE
                                                      : ACL_SPARSE_OP_NON_TRANSPOSE,
                                computeType)
    {
        CreateStream("MatmulDtypePairCtx: aclrtCreateStream failed");
        constexpr size_t kSq = 128 * 128;
        dA = sparse_test::DeviceBuffer::alloc(kSq * abEltSize);
        dB = sparse_test::DeviceBuffer::alloc(kSq * abEltSize);
        dC = sparse_test::DeviceBuffer::alloc(kSq * cEltSize);
        dD = sparse_test::DeviceBuffer::alloc(kSq * dEltSize);
        if (aclsparseLtMatmulGetWorkspace(handle.get(), plan.cptr(), &workspaceSize)
                != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("MatmulDtypePairCtx: GetWorkspace failed");
        }
        dWorkspace = sparse_test::DeviceBuffer::alloc(workspaceSize);
    }
};

// WB-L2-48~51 共用执行段：合法全链执行 → 期望 NOT_SUPPORTED（执行层
// C/D dtype 配对白名单）。
static aclsparseStatus_t RunDtypePairNegMatmulRaw(MatmulDtypePairCtx& ctx)
{
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), &alpha,
        ctx.dA.get(), ctx.dB.get(), &beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &ctx.stream, 1);
    return MatmulNpuRaw(p);
}

// WB-L2-48: INT8 A/B × C=FP16 / D=INT8 → NOT_SUPPORTED（按 2B/元素写
// 1B/元素声明的 matD——int8_cf16 分派键下 2× 越界写域，执行层拦截）。
TEST_F(MatmulExceptionTest, WB_L2_48_Int8ABWithCf16Dint8Rejected) {
    MatmulDtypePairCtx ctx(ACL_INT8, ACL_FLOAT16, ACL_INT8, ACL_SPARSE_COMPUTE_32I,
                           sizeof(int8_t), sizeof(uint16_t), sizeof(int8_t));
    EXPECT_EQ(RunDtypePairNegMatmulRaw(ctx), ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// WB-L2-49: INT8 A/B × C=INT32 / D=INT8 → NOT_SUPPORTED（int8_ci32 分派
// 键下 4× 越界写域）。
TEST_F(MatmulExceptionTest, WB_L2_49_Int8ABWithCi32Dint8Rejected) {
    MatmulDtypePairCtx ctx(ACL_INT8, ACL_INT32, ACL_INT8, ACL_SPARSE_COMPUTE_32I,
                           sizeof(int8_t), sizeof(int32_t), sizeof(int8_t));
    EXPECT_EQ(RunDtypePairNegMatmulRaw(ctx), ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// WB-L2-50: INT8 A/B × C=INT8 / D=BF16 → NOT_SUPPORTED（落收尾 else 按
// 1B 写 2B 声明缓冲——静默错值域，执行层拦截）。
TEST_F(MatmulExceptionTest, WB_L2_50_Int8ABWithCint8Dbf16Rejected) {
    MatmulDtypePairCtx ctx(ACL_INT8, ACL_INT8, ACL_BF16, ACL_SPARSE_COMPUTE_32I,
                           sizeof(int8_t), sizeof(int8_t), sizeof(uint16_t));
    EXPECT_EQ(RunDtypePairNegMatmulRaw(ctx), ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// WB-L2-51: FP16 A/B × C/D=INT32 → NOT_SUPPORTED（非 INT8 路径无 C/D
// 独立 dtype 支持：SpltEpilogueImpl<__fp16> 默认 CType=OutType=T，按 2B
// 读写 4B/元素声明缓冲——2× 读写越界域）。
TEST_F(MatmulExceptionTest, WB_L2_51_Fp16ABWithInt32CDRejected) {
    MatmulDtypePairCtx ctx(ACL_FLOAT16, ACL_INT32, ACL_INT32, ACL_SPARSE_COMPUTE_32F,
                           sizeof(uint16_t), sizeof(int32_t), sizeof(int32_t));
    EXPECT_EQ(RunDtypePairNegMatmulRaw(ctx), ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// WB-L2-52（D2 负向锁定）：numBatches>1 且 batchStride<0 → INVALID_VALUE。
// D2 广播放行（stride==0 = 跨批广播）不得误伤负值拒绝边界——负 stride 下
// kernel 按 base + b*stride 推进必然回走越界，恒非法。构造经
// MatmulBatchOverflowCtx（四 mat 同设 nb=2 / stride=-8，穿过 desc init 层
// ——SetAttribute 与 MatmulDescriptorInit 均不校验 stride 值），违规由
// validate_matmul_one_batch_side 的 stride<0 分支拦截（A 侧首先被校验）。
TEST_F(MatmulExceptionTest, WB_L2_52_BatchStrideNegative) {
    MatmulBatchOverflowCtx ctx(-8, 0);
    EXPECT_EQ(RunBatchNegMatmulRaw(ctx.handle.get(), ctx.plan->get(),
                                   ctx.dWorkspace.get()),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}
