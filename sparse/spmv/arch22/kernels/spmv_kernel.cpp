/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

// 统一 host 分发器 —— 根据 SPMV_* 类型 ID 派发到各自的 __global__ kernel
// 各 kernel 定义在 spmv_kernel_*type*.cpp 中

#include "spmv_kernel.h"
#include "spmv_cube_kernel.h"
#include "spmv_trans_finalize.h"

// 注意：spmv.h 会把 GM_ADDR 重定义为普通 host 指针，
// 因此上面的 kernel 声明必须放在它之前（需要 AscendC 的 __gm__ 限定）。
#include "../spmv.h"
#include "../spmv_cube_tiling.h"

namespace {

/**
 * @brief 以 cube 版 kernel 计算非转置 SpMV：y = alpha * A @ x + beta * y。
 *
 * 支持 fp32（A / x / y 均为 fp32）与 fp16（A / x 为 fp16，cube 以 fp32 累加，
 * y 为 fp32）两种输入类型，由 @p isHalfInput 选择对应的 kernel。
 *
 * tiling 由 host 侧（spmv_cube::MakeCubeTiling）算好后按值传入 kernel，
 * workspace 由调用方通过 aclsparseSpMV 的 externalBuffer 提供，
 * 因此本函数只做一次异步 launch：不申请显存、不拷贝 tiling、不同步 stream。
 */
void LaunchSpmvCube(GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal,
					GM_ADDR xVec, GM_ADDR yVec, GM_ADDR workspace,
					tcuscan::SpMVCubeTiling tiling, uint32_t aicBlocks,
					bool isHalfInput, void *stream) {
	// kernel 侧要在完整的 indptr（rows + 1 个元素、升序）上二分查找块边界，
	// 因此这里传入 csrRowPtr 首地址，而不是行结束下标 csrRowPtr + 1。
	if (isHalfInput) {
		spmv_cube_fp16<<<aicBlocks, nullptr, stream>>>(
			csrVal, csrColInd, csrRowPtr, xVec, yVec, workspace, tiling);
	} else {
		spmv_cube_fp32<<<aicBlocks, nullptr, stream>>>(
			csrVal, csrColInd, csrRowPtr, xVec, yVec, workspace, tiling);
	}
}

void LaunchNarrowTranspose(GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal,
                           GM_ADDR xVec, GM_ADDR yVec, GM_ADDR workspace,
                           const SpmvTilingData &tiling, const void *alpha,
                           const void *beta, int32_t outType, uint32_t numBlocks,
                           void *stream)
{
    const float a = *static_cast<const float *>(alpha);
    const float outputBeta = *static_cast<const float *>(beta);
    constexpr float workspaceBeta = 1.0f;
    if (outType == SPMV_OUT_HALF) {
        spmv_kernel_float_half_float<<<numBlocks, nullptr, stream>>>(
            csrRowPtr, csrColInd, csrVal, xVec, workspace,
            tiling.totalRowsNum, tiling.totalColNum, a, workspaceBeta, true);
        spmv_trans_finalize_half<<<numBlocks, nullptr, stream>>>(
            workspace, yVec, tiling.totalColNum, outputBeta);
        return;
    }
    spmv_kernel_float_bfloat16_t_float<<<numBlocks, nullptr, stream>>>(
        csrRowPtr, csrColInd, csrVal, xVec, workspace,
        tiling.totalRowsNum, tiling.totalColNum, a, workspaceBeta, true);
    spmv_trans_finalize_bfloat16_t<<<numBlocks, nullptr, stream>>>(
        workspace, yVec, tiling.totalColNum, outputBeta);
}

void LaunchInt32Spmv(GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal,
                     GM_ADDR xVec, GM_ADDR yVec, const SpmvTilingData &tiling,
                     const void *alpha, const void *beta, bool trans,
                     uint32_t numBlocks, void *stream)
{
    const int32_t a = *static_cast<const int32_t *>(alpha);
    const int32_t b = *static_cast<const int32_t *>(beta);
    spmv_kernel_int32_t_int32_t_int32_t<<<numBlocks, nullptr, stream>>>(
        csrRowPtr, csrColInd, csrVal, xVec, yVec,
        tiling.totalRowsNum, tiling.totalColNum, a, b, trans);
}

void LaunchFloatSpmv(GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal,
                     GM_ADDR xVec, GM_ADDR yVec, const SpmvTilingData &tiling,
                     const void *alpha, const void *beta, int32_t valType,
                     int32_t outType, bool trans, uint32_t numBlocks, void *stream)
{
    const float a = *static_cast<const float *>(alpha);
    const float b = *static_cast<const float *>(beta);
#define SPMV_LAUNCH(ValT, OutT) \
    spmv_kernel_float_##ValT##_##OutT<<<numBlocks, nullptr, stream>>>( \
        csrRowPtr, csrColInd, csrVal, xVec, yVec, \
        tiling.totalRowsNum, tiling.totalColNum, a, b, trans)
    if (valType == SPMV_VAL_HALF && outType == SPMV_OUT_HALF) {
        SPMV_LAUNCH(half, half);
    } else if (valType == SPMV_VAL_HALF) {
        SPMV_LAUNCH(half, float);
    } else if (valType == SPMV_VAL_BF16 && outType == SPMV_OUT_BF16) {
        SPMV_LAUNCH(bfloat16_t, bfloat16_t);
    } else if (valType == SPMV_VAL_BF16) {
        SPMV_LAUNCH(bfloat16_t, float);
    } else if (valType == SPMV_VAL_I32) {
        SPMV_LAUNCH(int32_t, float);
    } else {
        SPMV_LAUNCH(float, float);
    }
#undef SPMV_LAUNCH
}

} // namespace

extern "C" void spmv_kernel_do(GM_ADDR csrRowPtr, GM_ADDR csrColInd,
							   GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
							   SpmvTilingData tiling, const void *alpha,
							   const void *beta, int32_t computeType,
							   int32_t valType, int32_t outType, bool trans,
							   int32_t algType, uint32_t numBlocks,
							   GM_ADDR cubeWorkspace,
							   tcuscan::SpMVCubeTiling cubeTiling,
							   uint32_t cubeBlocks, void *stream) {
	// cube 版实现：fp32 或 fp16 输入 / fp32 输出的非转置 SpMV。
	// host 侧已完成校验与 tiling；tiling 不可用（空矩阵等）时 host 会改派
	// SPMV_ALG_VECTOR，因此这里不再回退。
	if (algType == SPMV_ALG_CUBE) {
		const bool isHalfInput = (valType == SPMV_VAL_HALF);
		LaunchSpmvCube(csrRowPtr, csrColInd, csrVal, xVec, yVec, cubeWorkspace,
					   cubeTiling, cubeBlocks, isHalfInput, stream);
		return;
	}

    if (trans && computeType == SPMV_COMPUTE_F32 &&
        (outType == SPMV_OUT_HALF || outType == SPMV_OUT_BF16)) {
        LaunchNarrowTranspose(csrRowPtr, csrColInd, csrVal, xVec, yVec,
                              cubeWorkspace, tiling, alpha, beta, outType,
                              numBlocks, stream);
        return;
    }

	if (computeType == SPMV_COMPUTE_I32) {
		LaunchInt32Spmv(csrRowPtr, csrColInd, csrVal, xVec, yVec, tiling,
                        alpha, beta, trans, numBlocks, stream);
	} else {
		LaunchFloatSpmv(csrRowPtr, csrColInd, csrVal, xVec, yVec, tiling,
                        alpha, beta, valType, outType, trans, numBlocks, stream);
	}
}
