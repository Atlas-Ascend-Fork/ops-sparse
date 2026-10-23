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

// 统一 host 分发器 —— 根据 SPMV_* 类型 ID 派发到各自的 __global__ kernel
// 基础组合的 kernel 定义在 spmv_kernel_*type*.cpp 中；
// A2/A3 任务新增的 4 种 dtype 组合（c64×c64、c64_f32、f32_i8_f32、i32_i8_i32）
// 按评审意见直接在本文件内实例化（见下方 DEFINE_SPMV_KERNEL* 展开）。

#include "spmv_kernel.h"
#include "tiling_spmv_cube.h"

// cube 版 SpMV kernel（定义在 spmv_cube_kernel.cpp）
extern "C" __global__ __aicore__ void spmv_cube_fp32(
	GM_ADDR vec_in, GM_ADDR cols_in, GM_ADDR indptr, GM_ADDR x_in,
	GM_ADDR vec_out, GM_ADDR workspace, tcuscan::SpMVCubeTiling tiling);

extern "C" __global__ __aicore__ void spmv_cube_fp16(
	GM_ADDR vec_in, GM_ADDR cols_in, GM_ADDR indptr, GM_ADDR x_in,
	GM_ADDR vec_out, GM_ADDR workspace, tcuscan::SpMVCubeTiling tiling);

// 注意：以下 DEFINE 展开必须位于 ../spmv.h 之前——spmv.h 会把 GM_ADDR
// 重定义为普通 host 指针，而 __global__ kernel 定义需要 AscendC 的
// __gm__ 限定。这 4 种组合的 extern 声明已从 spmv_kernel.h 移除
// （同 TU 内声明与定义共存会在 host 编译 pass 形成二义性重载）。
// c64×c64（复数输入复数输出）、c64_f32（实数 f32 输入复数输出）、
// f32_i8_f32 与 i32_i8_i32（int8 值类型两类组合）。
DEFINE_SPMV_KERNEL_C64(c64, true)
DEFINE_SPMV_KERNEL_C64(f32, false)
DEFINE_SPMV_KERNEL(float, int8_t, float)
DEFINE_SPMV_KERNEL(int32_t, int8_t, int32_t)

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

} // namespace

namespace {

#define SPMV_LAUNCH_F32(ValT, OutT)                                            \
	spmv_kernel_float_##ValT##_##OutT<<<numBlocks, nullptr, stream>>>(         \
		csrRowPtr, csrColInd, csrVal, xVec, yVec, ws,                          \
		tiling.totalRowsNum, tiling.totalColNum, a, b, trans,                  \
		tiling.indexBase, tiling.readY)

// f32 计算路径的 kernel 派发（valType/outType -> 具体 kernel）
static void LaunchSpmvF32Kernels(GM_ADDR csrRowPtr, GM_ADDR csrColInd,
								 GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
								 GM_ADDR ws, const SpmvTilingData &tiling,
								 float a, float b, int32_t valType,
								 int32_t outType, bool trans,
								 uint32_t numBlocks, void *stream) {
	if (valType == SPMV_VAL_HALF && outType == SPMV_OUT_HALF) {
		SPMV_LAUNCH_F32(half, half);
	}
	else if (valType == SPMV_VAL_HALF) {
		SPMV_LAUNCH_F32(half, float);
	}
	else if (valType == SPMV_VAL_BF16 && outType == SPMV_OUT_BF16) {
		SPMV_LAUNCH_F32(bfloat16_t, bfloat16_t);
	}
	else if (valType == SPMV_VAL_BF16) {
		SPMV_LAUNCH_F32(bfloat16_t, float);
	}
	else if (valType == SPMV_VAL_I8) {
		SPMV_LAUNCH_F32(int8_t, float);
	}
	else if (valType == SPMV_VAL_I32) {
		SPMV_LAUNCH_F32(int32_t, float);
	}
	else {
		SPMV_LAUNCH_F32(float, float);
	}
}

#undef SPMV_LAUNCH_F32

} // namespace

extern "C" void spmv_kernel_do(GM_ADDR csrRowPtr, GM_ADDR csrColInd,
							   GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
							   GM_ADDR ws,
							   SpmvTilingData tiling, const void *alpha,
							   const void *beta, int32_t computeType,
							   int32_t valType, int32_t outType, bool trans,
							   int32_t algType, uint32_t numBlocks,
							   GM_ADDR cubeWorkspace,
							   tcuscan::SpMVCubeTiling cubeTiling,
							   uint32_t cubeBlocks, void *stream) {
#define SPMV_LAUNCH(CompT, ValT, OutT)                                         \
	spmv_kernel_##CompT##_##ValT##_##OutT<<<numBlocks, nullptr, stream>>>(     \
		csrRowPtr, csrColInd, csrVal, xVec, yVec, ws,                          \
		tiling.totalRowsNum, tiling.totalColNum, a, b, trans,                  \
		tiling.indexBase, tiling.readY)

	// cube 版实现：fp32 或 fp16 输入 / fp32 输出的非转置 SpMV。
	// host 侧已完成校验与 tiling；tiling 不可用（空矩阵等）时 host 会改派
	// SPMV_ALG_VECTOR，因此这里不再回退。
	if (algType == SPMV_ALG_CUBE) {
		const bool isHalfInput = (valType == SPMV_VAL_HALF);
		LaunchSpmvCube(csrRowPtr, csrColInd, csrVal, xVec, yVec, cubeWorkspace,
					   cubeTiling, cubeBlocks, isHalfInput, stream);
		return;
	}

	if (computeType == SPMV_COMPUTE_I32) {
		int32_t a = *static_cast<const int32_t *>(alpha);
		int32_t b = *static_cast<const int32_t *>(beta);
		if (valType == SPMV_VAL_I8) {
			SPMV_LAUNCH(int32_t, int8_t, int32_t);
		}
		else {
			SPMV_LAUNCH(int32_t, int32_t, int32_t);
		}
	}
	else if (computeType == SPMV_COMPUTE_C64) {
		// complex64 标量：按交错 float2 读取（re, im）
		const float *af = static_cast<const float *>(alpha);
		const float *bf = static_cast<const float *>(beta);
		float aRe = af[0], aIm = af[1];
		float bRe = bf[0], bIm = bf[1];
		if (valType == SPMV_VAL_C64) {
			spmv_kernel_c64_c64<<<numBlocks, nullptr, stream>>>(
				csrRowPtr, csrColInd, csrVal, xVec, yVec, ws,
				tiling.totalRowsNum, tiling.totalColNum,
				aRe, aIm, bRe, bIm, trans, tiling.indexBase, tiling.readY);
		}
		else { // SPMV_VAL_F32 -> complex64 输出
			spmv_kernel_c64_f32<<<numBlocks, nullptr, stream>>>(
				csrRowPtr, csrColInd, csrVal, xVec, yVec, ws,
				tiling.totalRowsNum, tiling.totalColNum,
				aRe, aIm, bRe, bIm, trans, tiling.indexBase, tiling.readY);
		}
	}
	else {
		float a = *static_cast<const float *>(alpha);
		float b = *static_cast<const float *>(beta);
		LaunchSpmvF32Kernels(csrRowPtr, csrColInd, csrVal, xVec, yVec, ws,
							 tiling, a, b, valType, outType, trans,
							 numBlocks, stream);
	}

#undef SPMV_LAUNCH
}
