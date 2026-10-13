# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Gather operator package.

Importing this package registers the one-dimensional NPU implementation of
``aten::index_select`` and the base-aware ``ops_sparse::gather``. JIT compilation is deferred until the first NPU call.
"""

__all__ = ["gather", "index_select_npu"]

from .gather import TORCH_NPU_SPARSE_FACADE_APIS as TORCH_NPU_SPARSE_FACADE_APIS
from .gather import gather as gather, index_select_npu as index_select_npu
