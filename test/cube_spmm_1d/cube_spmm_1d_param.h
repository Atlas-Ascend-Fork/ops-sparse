/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUBE_SPMM_1D_PARAM_H_
#define CUBE_SPMM_1D_PARAM_H_
#include "csv_loader.h"
#include "sparse_test.h"
#include <string>

struct CubeSpmm1dParam {
    int m = 0, n = 0, k = 0, key = -1, seed = 0, mode = 0;
    bool csr = false;
    std::string id;
    void fill(const sparse_test::csv_map& row)
    {
        m = sparse_test::parseInt(row, "m");
        n = sparse_test::parseInt(row, "n");
        k = sparse_test::parseInt(row, "k");
        key = sparse_test::parseInt(row, "key");
        seed = sparse_test::parseInt(row, "seed");
        mode = sparse_test::parseInt(row, "mode");
        csr = sparse_test::parseBool(row, "csr");
        id = sparse_test::parseString(row, "id");
    }
    std::string caseId() const { return id; }
};
#endif
