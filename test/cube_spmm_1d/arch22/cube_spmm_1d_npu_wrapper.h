/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUBE_SPMM_1D_NPU_WRAPPER_H_
#define CUBE_SPMM_1D_NPU_WRAPPER_H_
#include "descriptor_manager.h"
#include "cube_spmm_1d_golden.h"
#include <memory>
#include <stdexcept>

inline void CubeCheck(aclsparseStatus_t status)
{
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("cube_spmm_1d status=" + std::to_string(status));
    }
}
struct CubePlanDeleter {
    void operator()(aclsparseCubeSpmm1dPlan* p) const { (void)aclsparseCubeSpmm1dDestroyPlan(p); }
};
using CubePlan = std::unique_ptr<aclsparseCubeSpmm1dPlan, CubePlanDeleter>;

inline void CubePackRow(const Eigen::SparseMatrix<double, Eigen::RowMajor>& canonical, int row, int first,
                        std::map<int, std::vector<uint16_t>>& block)
{
    for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(canonical, row); it; ++it) {
        auto& values = block[it.col()];
        if (values.empty()) {
            values.resize(16, 0);
        }
        values[row - first] = CubeHalf(it.value());
    }
}

inline CubePlan CubePrepare(aclsparseHandle_t handle, const CubeSpmm1dParam& p, const CubeInput& a)
{
    aclsparseCubeSpmm1dPlan_t raw = nullptr;
    aclsparseCubeSpmm1dOptions options{sizeof(options), p.key, 0, 0};
    if (p.csr) {
        CubeCheck(aclsparseCubeSpmm1dCreateCsrPlan(handle, p.m, p.n, p.k, a.offsets.data(), a.offsets.size(),
                                                   a.columns.data(), a.values.data(), a.columns.size(), &options,
                                                   &raw));
    } else {
        const auto canonical = CubeCanonical(a, p.m, p.k);
        std::vector<int32_t> offsets{0}, columns;
        std::vector<uint16_t> values;
        for (int first = 0; first < p.m; first += 16) {
            std::map<int, std::vector<uint16_t>> block;
            for (int row = first; row < first + 16; ++row) {
                CubePackRow(canonical, row, first, block);
            }
            for (const auto& entry : block) {
                columns.push_back(entry.first);
                values.insert(values.end(), entry.second.begin(), entry.second.end());
            }
            offsets.push_back(columns.size());
        }
        CubeCheck(aclsparseCubeSpmm1dCreatePlan(handle, p.m, p.n, p.k, offsets.data(), offsets.size(), columns.data(),
                                                values.data(), columns.size(), &options, &raw));
    }
    return CubePlan(raw);
}
#endif
