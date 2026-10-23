# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CPU checks for SpMM capabilities in the repository matrix framework."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[3]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


router = load_source("spmm_matrix_router_test",
                     ROOT / "torch_extension/cann_ops_sparse/ops/matmul/dispatch.py")


def load_spmm(native):
    load = Mock(return_value=native)

    class FakeBuilder:
        def __init__(self, name, category):
            self.name = name
            self.category = category

    FakeBuilder.load = load
    builder = types.ModuleType("cann_ops_sparse.op_builder.builder")
    builder.OpBuilder = FakeBuilder
    backward = load_source("cann_ops_sparse.ops.sparse.spmm.spmm_backward",
                           ROOT / "sparse/spmm/torch_extension/spmm_backward.py")
    modules = {"cann_ops_sparse.op_builder.builder": builder,
               "cann_ops_sparse.ops.matmul.dispatch": router,
               "cann_ops_sparse.ops.sparse.spmm.spmm_backward": backward}
    with patch.dict(sys.modules, modules), patch("torch.library.impl") as register:
        module = load_source("cann_ops_sparse.ops.sparse.spmm.spmm",
                             ROOT / "sparse/spmm/torch_extension/spmm.py")
    register.assert_not_called()
    return module.MATMUL_CAPABILITIES[0], load


class SparseAddmmRegistryTest(unittest.TestCase):
    def test_operator_import_is_lazy_and_declares_supported_capability(self):
        capability, load = load_spmm(types.SimpleNamespace())
        load.assert_not_called()
        self.assertEqual((capability.owner, capability.route, capability.operations),
                         ("spmm", "spmm", ("mm", "addmm")))
        self.assertEqual(capability.device_types, ("npu", "privateuseone"))
        self.assertEqual(capability.dtypes,
                         (torch.float16, torch.bfloat16, torch.float32, torch.complex64))
        self.assertTrue(callable(capability.backward))

    def test_spmm_and_spgemm_share_layout_routing(self):
        native = types.SimpleNamespace(sparse_addmm=Mock(return_value=torch.ones(2, 2)))
        capability, load = load_spmm(native)
        registry = router.MatmulRegistry()
        registry.register(replace(capability, device_types=("cpu",)))
        gemm = Mock(return_value=torch.eye(2).to_sparse_csr())
        registry.register(router.MatmulCapability("spgemm", "spgemm", gemm,
                                                operations=("addmm",)))
        sparse = torch.eye(2).to_sparse_csr()
        dense = torch.eye(2)
        self_tensor = torch.ones(2)
        dense_request = router.MatmulRequest("addmm", sparse, dense, self_tensor, 2, 3)
        result = registry.dispatch(dense_request)
        self.assertIs(result, native.sparse_addmm.return_value)
        native.sparse_addmm.assert_called_once_with(self_tensor, sparse, dense, 2, 3)
        load.assert_called_once_with()
        sparse_request = router.MatmulRequest("addmm", sparse, sparse, sparse, 0, 1)
        registry.dispatch(sparse_request)
        gemm.assert_called_once_with(sparse_request)
        load.assert_called_once_with()

    def test_out_uses_upstream_entry_and_preserves_output_identity(self):
        native = types.SimpleNamespace(addmm_out=Mock(side_effect=lambda *args: args[-1]))
        capability, load = load_spmm(native)
        request = router.MatmulRequest("addmm", torch.eye(2).to_sparse_csr(), torch.eye(2),
                                       torch.ones(2), out=torch.empty(2, 2))
        self.assertIs(capability.forward(request), request.out)
        native.addmm_out.assert_called_once_with(
            request.self, request.mat1, request.mat2, request.beta, request.alpha, request.out)
        load.assert_called_once_with()

    def test_duplicate_provider_does_not_replace_existing_capability(self):
        capability, _ = load_spmm(types.SimpleNamespace())
        registry = router.MatmulRegistry()
        registry.register(capability)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            registry.register(capability)
        self.assertIs(registry.capabilities[("spmm", "addmm")], capability)


if __name__ == "__main__":
    unittest.main(verbosity=2)
