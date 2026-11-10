# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

ops_sparse_add_gtest_tests(cgtsv2 ${OPS_SPARSE} WARN_ON_MISSING_SRC
    EXTRA_INCLUDES ${CMAKE_SOURCE_DIR}/sparse/cgtsv2)

if(TARGET cgtsv2_test)
    set_property(TARGET cgtsv2_test APPEND PROPERTY LINK_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/arch22/cgtsv2_test.csv
        ${CMAKE_CURRENT_SOURCE_DIR}/arch35/cgtsv2_test.csv)
endif()
