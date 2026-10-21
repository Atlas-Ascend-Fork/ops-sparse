# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if(SOC_VERSION_LOWER MATCHES "^ascend950")
    ops_sparse_add_gtest_tests(spmv ${OPS_SPARSE} EIGEN)
    if(TARGET spmv_test AND "spmm" IN_LIST OP_LIST)
        target_compile_definitions(spmv_test PRIVATE SPMV_TEST_WITH_SPMM=1)
    endif()
else()
    ops_sparse_add_test(spmv ${OPS_SPARSE})
endif()
