# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details.
# You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# 描述符 Destroy 系列 null 语义专测（#377）：纯 Host 侧，无独立 kernel 源目录，
# NO_SRC_DIR 布局（源码直接位于 test/destroy_descr_null/，无需 arch 子目录）。
# 仅依赖 common 库符号（Create*/Destroy*），无需设备缓冲。
ops_sparse_add_gtest_tests(destroy_descr_null ${OPS_SPARSE} NO_SRC_DIR)
