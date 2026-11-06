/**
 * Copyright (c) 2026 balckwell
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Host-only regression: g++ -std=c++17 -Itest/frame test/frame/host/test_fill_generator.cpp -o /tmp/test_fill
#include "fill.h"

#include <iostream>
#include <string>

namespace {

bool ValidCsr(const sparse_test::CsrMatrix& csr)
{
    if (csr.rowOffsets.size() != static_cast<size_t>(csr.rows + 1) || csr.rowOffsets.front() != 0 ||
        csr.rowOffsets.back() != csr.nnz || csr.colIndices.size() != static_cast<size_t>(csr.nnz) ||
        csr.values.size() != static_cast<size_t>(csr.nnz)) {
        return false;
    }
    for (int64_t row = 0; row < csr.rows; ++row) {
        const int32_t begin = csr.rowOffsets[row];
        const int32_t end = csr.rowOffsets[row + 1];
        if (begin > end) {
            return false;
        }
        for (int32_t k = begin; k < end; ++k) {
            if (csr.colIndices[k] < 0 || csr.colIndices[k] >= csr.cols || !std::isfinite(csr.values[k]) ||
                (k > begin && csr.colIndices[k - 1] >= csr.colIndices[k])) {
                return false;
            }
        }
    }
    return true;
}

bool CheckDense(const std::string& name, int64_t rows, int64_t cols, uint32_t seed)
{
    sparse_test::SparseFillGenerator generator(seed);
    generator.setSparsity(0.0);
    const auto csr = generator.generateCsr(rows, cols);
    const bool passed = ValidCsr(csr) && csr.nnz == rows * cols;
    std::cout << name << ": expected=" << rows * cols << " actual=" << csr.nnz << " valid_csr=" << ValidCsr(csr) << " "
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool CheckBoundaries()
{
    sparse_test::SparseFillGenerator generator(42);
    generator.setSparsity(1.0);
    const auto empty = generator.generateCsr(9, 17);
    generator.setSparsity(0.0);
    const auto zeroRows = generator.generateCsr(0, 17);
    const auto zeroCols = generator.generateCsr(9, 0);
    generator.setEmptyRowProb(1.0);
    const auto emptyRows = generator.generateCsr(9, 17);
    const bool passed = ValidCsr(empty) && empty.nnz == 0 && ValidCsr(zeroRows) && zeroRows.nnz == 0 &&
                        ValidCsr(zeroCols) && zeroCols.nnz == 0 && ValidCsr(emptyRows) && emptyRows.nnz == 0;
    std::cout << "empty-and-zero-boundaries " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool CheckSparseAndRepeatability()
{
    sparse_test::SparseFillGenerator first(77);
    sparse_test::SparseFillGenerator second(77);
    first.setSparsity(0.85);
    second.setSparsity(0.85);
    const auto a = first.generateCsr(50, 50);
    const auto b = second.generateCsr(50, 50);
    const bool passed = ValidCsr(a) && a.nnz > 0 && a.nnz < 2500 && a.rowOffsets == b.rowOffsets &&
                        a.colIndices == b.colIndices && a.values == b.values;
    std::cout << "sparse-repeatability: nnz=" << a.nnz << " " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool CheckHintAndConversions()
{
    sparse_test::SparseFillGenerator generator(42);
    const auto lowHint = generator.generateCsr(8, 20, 8);
    const auto highHint = generator.generateCsr(8, 20, 152);
    const auto coo = generator.generateCoo(8, 20, 160);
    const auto csc = generator.generateCsc(8, 20, 160);
    const bool passed = ValidCsr(lowHint) && lowHint.nnz == 8 && ValidCsr(highHint) && highHint.nnz == 152 &&
                        coo.nnz == 160 && coo.rowIndices.size() == 160 && coo.colIndices.size() == 160 &&
                        csc.nnz == 160 && csc.colOffsets.front() == 0 && csc.colOffsets.back() == 160 &&
                        csc.rowIndices.size() == 160 && ValidCsr(sparse_test::cooToCsr(coo)) &&
                        ValidCsr(sparse_test::cscToCsr(csc));
    std::cout << "hint-and-conversions: low=" << lowHint.nnz << " high=" << highHint.nnz << " coo=" << coo.nnz
              << " csc=" << csc.nnz << " " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool CheckNearDenseRows()
{
    int failed = 0;
    for (uint32_t seed = 0; seed < 128; ++seed) {
        sparse_test::SparseFillGenerator generator(seed);
        const auto csr = generator.generateCsr(3, 43, 126);
        if (!ValidCsr(csr) || csr.nnz != 126) {
            ++failed;
            continue;
        }
        for (int row = 0; row < 3; ++row) {
            if (csr.rowOffsets[row + 1] - csr.rowOffsets[row] != 42) {
                ++failed;
            }
        }
    }
    std::cout << "near-dense-seed-sweep: seeds=128 failures=" << failed << '\n';
    return failed == 0;
}

} // namespace

int main()
{
    int failed = 0;
    failed += !CheckDense("coo2csr/L1_1x100000_wide", 1, 100000, 105);
    failed += !CheckDense("coo2csr/L1_2048x2048_dense", 2048, 2048, 107);
    failed += !CheckDense("coo2csr/L0_32x32_nnz1024", 32, 32, 118);
    failed += !CheckDense("gebsr2gebsc/s_1x100_direct", 1, 100, 42);
    failed += !CheckDense("one-element", 1, 1, 1);
    failed += !CheckDense("rectangular", 3, 47, 19);
    failed += !CheckBoundaries();
    failed += !CheckSparseAndRepeatability();
    failed += !CheckHintAndConversions();
    failed += !CheckNearDenseRows();
    std::cout << "Checks: 10, failures: " << failed << '\n';
    return failed == 0 ? 0 : 1;
}
