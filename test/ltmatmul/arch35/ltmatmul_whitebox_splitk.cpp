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

/**
 * @file ltmatmul_whitebox_splitk.cpp
 * @brief PR 检视意见① 回归：fill_splitk_tiling_dims 的 int64 域化
 *        （kSegLen / validSplitK 两处 ceil 加法的 int32 回绕 + k==0 除零防御）。
 *
 * fill_splitk_tiling_dims 是 host 纯函数（经 aclsparselt_internal.h ->
 * aclsparselt_tiling_fill.h include 链可达，公共函数零局部重定义），无需
 * 上板跑 matmul——白盒直接锻造 internal 描述符链注入大 k。合法域依据：
 * 描述符层仅校验 rows/cols > 0 且 <= INT32_MAX（无上限拒绝分支），INT8
 * 极限几何 2^31-32 自证可达。
 *
 * 缺陷背景（修复前）：
 *   - kSegLen 计算 (k + effectiveSplitK - 1)：k 接近 INT32_MAX + splitK>=2
 *     时加法回绕 -> 负 kSegLen；
 *   - validSplitK 计算 (k + kSeg - 1)：k = 1.5*2^30 级几何 + splitK=2 时
 *     和超 INT32_MAX -> 负 validSplitK -> epilogue 防御回退
 *     (validSplitK>0)?validSplitK:splitK 读空段 [validSplitK, splitK)——
 *     B4 已去除 memset，未初始化 temp 被静默累加；
 *   - k==0 时 kSeg==0 -> 除零 SIGFPE（经合法描述符不可达，原实现无本地
 *     防御，现为 kSeg<=0 守卫钳到 1）。
 */

#include "aclsparselt_internal.h"  // fill_splitk_tiling_dims（经 tiling_fill.h）+ internal 描述符结构

#include <gtest/gtest.h>

#include <cstdint>

namespace {

// 锻造 fill_splitk_tiling_dims 读取的最小 internal 描述符链：
// asel -> matmulDescr -> matA（opA=NON_TRANSPOSE 时 md_k 读 matA->cols，
// rows/cols 同填 k 保持稳健）。全部为栈上聚合对象，无所有权/无销毁。
struct ForgedSplitKInput {
    aclsparseLtMatDescriptor matA{};
    aclsparseLtMatmulDescriptor md{};
    aclsparseLtMatmulAlgSelection asel{};
    CubeTiling ct{16, 16, 64};

    ForgedSplitKInput(int64_t k, int32_t baseK)
    {
        matA.rows = k;
        matA.cols = k;
        md.opA = ACL_SPARSE_OP_NON_TRANSPOSE;
        md.matA = &matA;
        asel.matmulDescr = &md;
        ct.baseK = baseK;
    }
};

inline AclsparseltTilingData RunFillSplitK(int64_t k, int32_t splitK,
                                           int32_t splitKMode, int32_t baseK = 64)
{
    ForgedSplitKInput in(k, baseK);
    AclsparseltTilingData td{};
    fill_splitk_tiling_dims(td, &in.asel, in.ct, splitK, splitKMode);
    return td;
}

}  // namespace

// ============================================================================
// 主回归：大 k（1.5*2^30）+ TWO_KERNELS + splitK=2 + baseK=64。
// 修复前 validSplitK 的 ceil 加法 (k + kSeg - 1) = 2415919103 > INT32_MAX
// 回绕为 -1879048193 -> validSplitK = -2（负值即触发 epilogue 空段回退）。
// 修复后：kSegLen = 805306368（ceil(k/2)，64 整除对齐不变），
// validSplitK = min(2, ceil(k/kSegLen)) = 2。
// ============================================================================
TEST(SplitKTilingFillWhitebox, LargeKTwoKernelsSplitK2NoWrap)
{
    constexpr int32_t k = 1610612736;  // 1.5 * 2^30
    const AclsparseltTilingData td =
        RunFillSplitK(k, /*splitK=*/2, ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS);
    EXPECT_EQ(td.splitK, 2);
    EXPECT_GT(td.kSegLen, 0);
    EXPECT_EQ(td.kSegLen, 805306368);
    EXPECT_EQ(td.validSplitK, 2);
}

// kSegLen 的 ceil 加法回绕（:91 系）：k = INT32_MAX（描述符层合法上界，
// <= INT32_MAX 无上限拒绝）。修复前 (k + 2 - 1) 回绕为 -2147483648 ->
// kSegLen 为负；修复后 kSegLen = ceil(INT32_MAX/2) = 2^30（64 整除），
// validSplitK = min(2, ceil(INT32_MAX / 2^30)) = 2。
// ============================================================================
TEST(SplitKTilingFillWhitebox, Int32MaxKCeilAdditionNoWrap)
{
    const AclsparseltTilingData td =
        RunFillSplitK(INT32_MAX, /*splitK=*/2, ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS);
    EXPECT_GT(td.kSegLen, 0);
    EXPECT_EQ(td.kSegLen, 1073741824);
    EXPECT_EQ(td.validSplitK, 2);
}

// ============================================================================
// k==0 除零防御（合法描述符不可达——rows/cols>0 校验；伪造域本地防御）：
// kSegLen 收敛为 0，validSplitK 经 kSeg<=0 守卫钳到 1（退化单段口径，
// 不再 SIGFPE）。
// ============================================================================
TEST(SplitKTilingFillWhitebox, ZeroKDegenerateGuard)
{
    const AclsparseltTilingData td =
        RunFillSplitK(0, /*splitK=*/2, ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS);
    EXPECT_EQ(td.kSegLen, 0);
    EXPECT_EQ(td.validSplitK, 1);
}

// ============================================================================
// 常规语义锚点（防 int64 域化引入行为回归）：
//   - 非整除 + 对齐：k=100, splitK=3 -> kSeg=34 对齐 64，validSplitK=2；
//   - ONE_KERNEL 吸收：splitK=5 -> effectiveSplitK=1，kSeg clamp 到 k；
//   - 整除无空段：k=128, splitK=2 -> kSeg=64，validSplitK=2。
// ============================================================================
TEST(SplitKTilingFillWhitebox, RegularSemanticsUnchanged)
{
    {
        const AclsparseltTilingData td =
            RunFillSplitK(100, 3, ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS);
        EXPECT_EQ(td.splitK, 3);
        EXPECT_EQ(td.kSegLen, 64);
        EXPECT_EQ(td.validSplitK, 2);
    }
    {
        const AclsparseltTilingData td =
            RunFillSplitK(100, 5, ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL);
        EXPECT_EQ(td.splitK, 1);
        EXPECT_EQ(td.kSegLen, 100);
        EXPECT_EQ(td.validSplitK, 1);
    }
    {
        const AclsparseltTilingData td =
            RunFillSplitK(128, 2, ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS);
        EXPECT_EQ(td.kSegLen, 64);
        EXPECT_EQ(td.validSplitK, 2);
    }
}
