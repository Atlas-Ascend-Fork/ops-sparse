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

// TEMPLATE: NPU 封装模板（芯片相关）
// 强制使用 RAII Manager（Handle/SpMat/DnVec/DnMat/DeviceBuffer），禁止裸指针
// 每个 ACL 调用必须校验返回值，失败时立即清理并返回错误码
// 下方提供 Generic API 和 Legacy API 两种变体，按算子所属 API 体系选择

#pragma once

#include <cstdint>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "../frame/sparse_test.h"
#include "../frame/fill.h"
#include "../frame/descriptor_manager.h"

namespace sparse_test {

// ===========================================================================
// 变体 1：Generic API（使用 RAII Manager）
// 适用于 SpMV/SpMM/SpGEMM/SDDMM/SpSM 等 Generic API 算子
// ===========================================================================
template <typename T>
inline std::vector<T> {{Op}}NpuWrapper(
    HandleManager& handle, aclrtStream stream,
    const CsrMatrix& csr,
    const std::vector<T>& xVec,
    const std::vector<T>& yInit,
    T alpha, T beta, bool transpose,
    aclDataType computeType)
{
    // 1. Device 内存（RAII，自动释放）
    auto dRowPtr = DeviceBuffer::copyFrom(csr.rowOffsets.data(), csr.rowOffsets.size() * sizeof(int32_t));
    auto dColIdx = DeviceBuffer::copyFrom(csr.colIndices.data(), csr.colIndices.size() * sizeof(int32_t));
    auto dVals   = DeviceBuffer::copyFrom(csr.values.data(),      csr.values.size() * sizeof(T));
    auto dX      = DeviceBuffer::copyFrom(xVec.data(),            xVec.size() * sizeof(T));
    auto dY      = DeviceBuffer::copyFrom(yInit.data(),           yInit.size() * sizeof(T));

    handle.setStream(stream);

    // 2. 描述符（RAII，自动 Destroy）
    auto matA = SpMatManager::createConstCsr(csr.rows, csr.cols, csr.nnz,
        dRowPtr.get(), dColIdx.get(), dVals.get());
    const auto xLen = transpose ? csr.rows : csr.cols;
    const auto yLen = transpose ? csr.cols : csr.rows;
    auto vecX = DnVecManager::createConst(xLen, dX.get(), aclDataTypeOf<T>());
    auto vecY = DnVecManager::create(yLen, dY.get(), aclDataTypeOf<T>());

    // 3. 算子调用
    auto op = transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
    auto ret = aclsparse{{Op}}(handle.get(), op, &alpha, matA.cget(), vecX.cget(),
                               &beta, vecY.get(), computeType,
                               ACL_SPARSE_{{OP}}_ALG_DEFAULT, nullptr);
    SPARSE_CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, return {});
    SPARSE_CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return {});

    // 4. D2H
    std::vector<T> yNpu(yLen);
    dY.copyToHost(yNpu.data(), yNpu.size() * sizeof(T));
    return yNpu;
}  // exit scope → RAII 自动 Destroy + Free

// ===========================================================================
// 变体 2：Legacy API（使用 DeviceBuffer + HandleManager）
// 适用于 gtsv/gtsv2 三对角、格式转换、排序等 Legacy API 算子
// 无 SpMatDescr/DnVec/DnMat 描述符，数据指针扁平传入
// ===========================================================================
template <typename T>
inline std::vector<T> {{Op}}LegacyNpuWrapper(
    HandleManager& handle, aclrtStream stream,
    int64_t m, int64_t nrhs,
    const std::vector<T>& dl,
    const std::vector<T>& d,
    const std::vector<T>& du,
    const std::vector<T>& xHost)
{
    // 1. Device 内存（RAII）
    auto dDl = DeviceBuffer::copyFrom(dl.data(), (m - 1) * sizeof(T));
    auto dD  = DeviceBuffer::copyFrom(d.data(),  m * sizeof(T));
    auto dDu = DeviceBuffer::copyFrom(du.data(), (m - 1) * sizeof(T));
    auto dX  = DeviceBuffer::copyFrom(xHost.data(), m * nrhs * sizeof(T));

    handle.setStream(stream);

    // TEMPLATE: Legacy 算子可能需要 MatDescr（手动创建/销毁，因 ops-sparse 当前未封装）
    // aclsparseMatDescr_t matDescr;
    // aclsparseCreateMatDescr(&matDescr);
    // aclsparseSetMatType(matDescr, ACL_SPARSE_MATRIX_TYPE_GENERAL);
    // aclsparseSetMatIndexBase(matDescr, ACL_SPARSE_INDEX_BASE_ZERO);
    // ... 算子调用 ...
    // aclsparseDestroyMatDescr(matDescr);

    // 2. 算子调用（扁平参数，无描述符）
    size_t bufferSize = 0;
    auto ret = aclsparseS{{Op}}(handle.get(), static_cast<int>(m),
        dDl.get(), dD.get(), dDu.get(), dX.get(),
        static_cast<int>(nrhs), static_cast<int>(m), &bufferSize);
    SPARSE_CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, return {});
    SPARSE_CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return {});

    // 3. D2H
    std::vector<T> xNpu(m * nrhs);
    dX.copyToHost(xNpu.data(), xNpu.size() * sizeof(T));
    return xNpu;
}

// TEMPLATE: Legacy API 注意事项：
// - 每种精度版本独立测试：S/D 版本必须分别测试
// - workspace 处理：部分 Legacy 算子通过 pBufferSize 返回大小（输入输出语义）
// - 错误处理：RAII 自动清理（DeviceBuffer 析构时释放、HandleManager 析构时 destroy）

}  // namespace sparse_test
