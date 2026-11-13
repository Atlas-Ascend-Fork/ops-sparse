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

/*!
 * \file csr2gebsr.h
 * \brief csr2gebsr Host 侧辅助函数：MatDescr 解析、handle 转换。
 */

#ifndef CSR2GEBSR_H_
#define CSR2GEBSR_H_

#include <cstdint>

#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"

/// 将 opaque handle 转为内部结构体指针
inline struct aclsparseContext *Csr2gebsrToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

/// 从 MatDescr 提取 indexBase 的整数值 (0 或 1)
inline int32_t Csr2gebsrGetBase(const aclsparseMatDescr_t descr)
{
    aclsparseIndexBase_t base = aclsparseGetMatIndexBase(descr);
    return (base == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;
}

#endif  // CSR2GEBSR_H_
