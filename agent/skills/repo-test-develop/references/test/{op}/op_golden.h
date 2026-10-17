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

// TEMPLATE: CPU golden 参考实现（与芯片无关）
// 使用 Eigen SparseMatrix<double> FP64 作为唯一 CPU 参考基准
// 签名与算子 API 一致，保留参数校验
// 下方提供 SpMV 和 SpMM 两个变体示例，按算子需求选择/修改

#pragma once

#include <Eigen/Eigen>
#include <Eigen/SparseCore>
#include <vector>
#include "../frame/fill.h"

namespace sparse_test {

// ===========================================================================
// 变体 1：SpMV Eigen Golden（CSR 格式）
// y = alpha * A * x + beta * y（A: sparse m×n CSR, x: dense n, y: dense m）
// ===========================================================================
template <typename T>
std::vector<T> SpMVGolden(
    const CsrMatrix& csr,
    const std::vector<T>& x,
    const std::vector<T>& yInit,
    T alpha, T beta, bool transpose)
{
    using Scalar = double;  // Eigen 用 FP64 避免精度损失

    int rows = static_cast<int>(csr.rows);
    int cols = static_cast<int>(csr.cols);
    int nnz = static_cast<int>(csr.nnz);

    // 1. 构建 Eigen SparseMatrix（CSR = RowMajor）
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> A(rows, cols);
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(nnz);
    for (int i = 0; i < rows; i++) {
        for (int j = static_cast<int>(csr.rowOffsets[i]); j < static_cast<int>(csr.rowOffsets[i + 1]); j++) {
            triplets.emplace_back(i, static_cast<int>(csr.colIndices[j]), static_cast<Scalar>(csr.values[j]));
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    // 2. 构建 Eigen Vector（转换为 FP64）
    int xLen = transpose ? rows : cols;
    int yLen = transpose ? cols : rows;
    Eigen::VectorXd xVec(xLen), yVec(yLen);
    for (int i = 0; i < xLen; i++) xVec(i) = static_cast<Scalar>(x[i]);
    for (int i = 0; i < yLen; i++) yVec(i) = static_cast<Scalar>(yInit[i]);

    // 3. 稀疏矩阵-向量乘法
    Eigen::VectorXd result;
    if (transpose) {
        result = alpha * (A.transpose() * xVec) + beta * yVec;
    } else {
        result = alpha * (A * xVec) + beta * yVec;
    }

    // 4. 转换回 T 类型
    std::vector<T> output(yLen);
    for (int i = 0; i < yLen; i++) {
        output[i] = static_cast<T>(result(i));
    }
    return output;
}

// ===========================================================================
// 变体 2：SpMM Eigen Golden
// C = alpha * A * B + beta * C（A: sparse m×k CSR, B: dense k×n, C: dense m×n）
// ===========================================================================
template <typename T>
std::vector<T> SpMMGolden(
    const CsrMatrix& A,
    const std::vector<T>& B,
    const std::vector<T>& C_init,
    int k, int n,
    T alpha, T beta,
    bool rowMajorB, bool rowMajorC)
{
    using Scalar = double;
    int m = static_cast<int>(A.rows);

    // 1. 构建 Eigen SparseMatrix
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> Amat(m, k);
    std::vector<Eigen::Triplet<Scalar>> triplets;
    for (int i = 0; i < m; i++)
        for (int j = static_cast<int>(A.rowOffsets[i]); j < static_cast<int>(A.rowOffsets[i + 1]); j++)
            triplets.emplace_back(i, static_cast<int>(A.colIndices[j]), static_cast<Scalar>(A.values[j]));
    Amat.setFromTriplets(triplets.begin(), triplets.end());

    // 2. 构建 Eigen DenseMatrix（转换为 FP64）
    Eigen::MatrixXd Bmat(k, n), Cmat(m, n);
    for (int r = 0; r < k; r++)
        for (int c = 0; c < n; c++)
            Bmat(r, c) = rowMajorB ? B[r * n + c] : B[r + c * k];
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++)
            Cmat(r, c) = rowMajorC ? C_init[r * n + c] : C_init[r + c * m];

    // 3. 稀疏矩阵-稠密矩阵乘法
    Cmat = alpha * (Amat * Bmat) + beta * Cmat;

    // 4. 转换回 T 类型
    std::vector<T> output(m * n);
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++) {
            if (rowMajorC) output[r * n + c] = static_cast<T>(Cmat(r, c));
            else           output[r + c * m] = static_cast<T>(Cmat(r, c));
        }
    return output;
}

// TEMPLATE: 按算子需求添加其他 golden 实现（如 SpSM、SDDMM 等）

}  // namespace sparse_test
