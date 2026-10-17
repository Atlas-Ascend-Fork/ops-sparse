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

#ifndef SPMV_H
#define SPMV_H

#include <cstdint>
#include <iostream>

#include "acl/acl.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"

// -------------------------------------------------------------------
// 便捷宏
// -------------------------------------------------------------------
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

#define CHECK_RET(cond, return_expr) \
	do                               \
	{                                \
		if (!(cond))                 \
		{                            \
			return_expr;             \
		}                            \
	} while (0)

#define LOG_PRINT(message, ...)         \
	do                                  \
	{                                   \
		printf(message, ##__VA_ARGS__); \
	} while (0)

// -------------------------------------------------------------------
// 内部辅助：句柄 / 描述符转换
// -------------------------------------------------------------------
inline struct aclsparseContext *ToInternalHandle(aclsparseHandle_t handle) {
	return reinterpret_cast<struct aclsparseContext *>(handle);
}

inline struct aclsparseSpMatDescr *ToMatInner(aclsparseConstSpMatDescr_t desc) {
	return const_cast<struct aclsparseSpMatDescr *>(
		reinterpret_cast<const struct aclsparseSpMatDescr *>(desc));
}

inline struct aclsparseDnVecDescr *ToVecInner(aclsparseConstDnVecDescr_t desc) {
	return const_cast<struct aclsparseDnVecDescr *>(
		reinterpret_cast<const struct aclsparseDnVecDescr *>(desc));
}

// -------------------------------------------------------------------
// 类型映射：将 ACL 侧类型映射为 kernel 内部 ID，避免 kernel 头文件依赖
// acl/acl.h
// -------------------------------------------------------------------

#define SPMV_COMPUTE_F32 0
#define SPMV_COMPUTE_I32 1

#define SPMV_VAL_F32 0
#define SPMV_VAL_HALF 1
#define SPMV_VAL_BF16 2
#define SPMV_VAL_I32 3

#define SPMV_OUT_F32 0
#define SPMV_OUT_HALF 1
#define SPMV_OUT_BF16 2
#define SPMV_OUT_I32 3

// 算法实现选择：向量版（每核处理若干整行）/ cube 版（矩阵单元段求和）
#define SPMV_ALG_VECTOR 0
#define SPMV_ALG_CUBE 1

static inline void SpmvTypesFromAcl(aclDataType computeType,
									aclDataType valType, aclDataType outType,
									int32_t *cType, int32_t *vType,
									int32_t *oType) {
	*cType = (computeType == ACL_INT32) ? SPMV_COMPUTE_I32 : SPMV_COMPUTE_F32;

	if (valType == ACL_FLOAT16) {
		*vType = SPMV_VAL_HALF;
	} else if (valType == ACL_BF16) {
		*vType = SPMV_VAL_BF16;
	} else if (valType == ACL_INT32) {
		*vType = SPMV_VAL_I32;
	} else {
		*vType = SPMV_VAL_F32;
	}

	if (outType == ACL_FLOAT16) {
		*oType = SPMV_OUT_HALF;
	} else if (outType == ACL_BF16) {
		*oType = SPMV_OUT_BF16;
	} else if (outType == ACL_INT32) {
		*oType = SPMV_OUT_I32;
	} else {
		*oType = SPMV_OUT_F32;
	}
}

#endif // SPMV_H
