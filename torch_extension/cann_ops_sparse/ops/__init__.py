# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Discover operator packages collected into the installed wheel."""

import importlib
import os

from .matmul.dispatch import MatmulRegistry
from .matmul.registration import TORCH_NPU_SPARSE_FACADE_APIS as _MATMUL_FACADE_APIS
from .matmul.registration import register_aten

TORCH_NPU_SPARSE_FACADE_APIS = {}
MATMUL_REGISTRY = MatmulRegistry()
_has_matmul_capabilities = False


def _collect_facade_apis(operator_module):
    """Collect the public ``torch_npu.sparse`` APIs declared by one operator."""
    for name, function in getattr(operator_module, "TORCH_NPU_SPARSE_FACADE_APIS", {}).items():
        if not isinstance(name, str) or not name.isidentifier() or not callable(function):
            raise RuntimeError(
                "TORCH_NPU_SPARSE_FACADE_APIS must map Python identifiers to callables"
            )
        if name in TORCH_NPU_SPARSE_FACADE_APIS:
            raise RuntimeError("duplicate torch_npu.sparse API: {}".format(name))
        TORCH_NPU_SPARSE_FACADE_APIS[name] = function


for _category in sorted(os.listdir(os.path.dirname(__file__))):
    if _category.startswith("_"):
        continue
    _category_path = os.path.join(os.path.dirname(__file__), _category)
    if not os.path.isdir(_category_path):
        continue
    for _op_name in sorted(os.listdir(_category_path)):
        if _op_name.startswith("_"):
            continue
        _op_path = os.path.join(_category_path, _op_name)
        if not os.path.isdir(_op_path) or not os.path.isfile(os.path.join(_op_path, "__init__.py")):
            continue
        _operator_module = importlib.import_module("{}.{}.{}".format(__name__, _category, _op_name))
        _collect_facade_apis(_operator_module)
        for _capability in getattr(_operator_module, "MATMUL_CAPABILITIES", ()):
            MATMUL_REGISTRY.register(_capability)
            _has_matmul_capabilities = True

if _has_matmul_capabilities:
    for _name, _function in _MATMUL_FACADE_APIS.items():
        if _name in TORCH_NPU_SPARSE_FACADE_APIS:
            raise RuntimeError("matrix fa\u00e7ade APIs must be declared by the shared framework")
        TORCH_NPU_SPARSE_FACADE_APIS[_name] = _function
    _MATMUL_LIBRARY = register_aten(MATMUL_REGISTRY)

try:
    del _category, _category_path, _op_name, _op_path, _operator_module
except NameError:
    pass
