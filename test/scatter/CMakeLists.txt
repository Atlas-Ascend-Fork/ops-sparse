# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You should refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# aclsparseScatter 测试：按 SOC 分流。
#   - arch22 (ascend910b): scatter_test.cpp 自带 main，用 add_test（不链 test_main）；
#     scatter_ut_test.cpp 为 A2 全覆盖 gtest UT，链 test_main。
#   - arch35 (ascend950): scatter_test.cpp 为 GTest 风格无 main，用 add_gtest_tests（链 test_main）
# 两个 arch 测试源 main 风格相反，须按 SOC 分流，否则会 main 多重定义或缺 main。
if("arch22" IN_LIST SOC_ARCH_DIRS)
    ops_sparse_add_test(scatter ${OPS_SPARSE})
    # A2 UT：接口级 gtest（NO_SRC_DIR：源码 test/scatter/scatter_ut_test.cpp，不依赖 arch 头）
    ops_sparse_add_gtest_tests(scatter_ut ${OPS_SPARSE} NO_SRC_DIR)
    # 非法输入探测独立二进制（源码 scatter_oob_probe_test.cpp），与主 UT 进程隔离
    ops_sparse_add_gtest_tests(scatter_oob_probe ${OPS_SPARSE} NO_SRC_DIR)
else()
    ops_sparse_add_gtest_tests(scatter ${OPS_SPARSE} WARN_ON_MISSING_SRC)
endif()
