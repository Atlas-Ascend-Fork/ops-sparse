// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

#ifndef OPS_SPARSE_TEST_PRESMOKE_H
#define OPS_SPARSE_TEST_PRESMOKE_H

#include <cstdlib>
#include <cstring>

// Ordinary build.sh runs use full shapes. Only the presmoke runner opts in
// to small functional shapes through its child-process environment.
inline bool SparseTestFullEnabled()
{
    const char *value = std::getenv("SPARSE_TEST_PRESMOKE");
    return value == nullptr || std::strcmp(value, "1") != 0;
}

#endif
