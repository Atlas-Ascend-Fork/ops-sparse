# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# aclsparseCsrSetStridedBatch GTest tests (arch35 / ascend950)
#
# 适配说明（与 coo_set_strided_batch 同范式）：
# - 实现位于 sparse/csr_set_strided_batch/arch35/（Host 侧描述符 setter，无 kernel）。
# - 测试调用 aclsparseSpMatGetStridedBatch（由 sparse/spmat_get_strided_batch/
#   提供，upstream 已合入）；--ops=csr_set_strided_batch 过滤构建时该源不在
#   libops_sparse 中，故将其源码直接编入测试目标。
# - 白盒字段验证：测试通过 aclsparse_descr_internal.h 核对 batch stride 字段
#   （include 路径已含 sparse/common）。
ops_sparse_add_gtest_tests(csr_set_strided_batch ${OPS_SPARSE} WARN_ON_MISSING_SRC)
if(TARGET csr_set_strided_batch_test)
    target_sources(csr_set_strided_batch_test PRIVATE
        ${CMAKE_SOURCE_DIR}/sparse/spmat_get_strided_batch/arch35/spmat_get_strided_batch_host.cpp)
    # 内嵌源码依赖 log/log.h（库目标有此 include，测试目标需补充）
    target_include_directories(csr_set_strided_batch_test PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include/op_common)
endif()
