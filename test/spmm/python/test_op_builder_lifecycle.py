# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Offline checks for the public builder's native-resource exit callbacks."""

from contextlib import contextmanager
import importlib.util
import logging
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch


BUILDER_SOURCE = (Path(__file__).resolve().parents[3] / "torch_extension" /
                  "cann_ops_sparse" / "op_builder" / "builder.py")
LOGGER = logging.getLogger(__name__)


@contextmanager
def fake_builder_environment(native_module):
    cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    cpp_extension.load = Mock(return_value=native_module)
    modules = {name: types.ModuleType(name) for name in ("torch", "torch.utils")}
    modules["torch.utils.cpp_extension"] = cpp_extension
    backend_spec = types.SimpleNamespace(submodule_search_locations=["/fake/torch_npu"])
    with patch.dict(sys.modules, modules), \
            patch("importlib.util.find_spec", return_value=backend_spec), \
            patch("os.path.isdir", return_value=True), \
            patch("os.path.isfile", return_value=True):
        spec = importlib.util.spec_from_file_location("op_builder_lifecycle_test", BUILDER_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class FakeBuilder(module.OpBuilder):
            def sources(self):
                return ["csrc/fake.cpp"]

        yield FakeBuilder, cpp_extension.load


def run_exit_order_child():
    import atexit

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    def shutdown_backend():
        LOGGER.info("BACKEND_SHUTDOWN")

    def release_handles():
        LOGGER.info("HANDLES_RELEASED")

    atexit.register(shutdown_backend)
    native = types.SimpleNamespace(release_cached_handles=release_handles)
    with fake_builder_environment(native) as (builder, _):
        builder("exit-order").load()


class OpBuilderLifecycleTest(unittest.TestCase):
    def test_cached_module_registers_one_release_callback(self):
        release = Mock()
        native = types.SimpleNamespace(release_cached_handles=release)
        with fake_builder_environment(native) as (builder, jit_load), \
                patch("atexit.register") as register:
            first = builder("cached")
            self.assertIs(first.load(), native)
            self.assertIs(first.load(), native)
            self.assertIs(builder("cached").load(), native)
            jit_load.assert_called_once()
            register.assert_called_once_with(release)
            release.assert_not_called()

    def test_module_without_release_hook_still_loads(self):
        native = types.SimpleNamespace()
        with fake_builder_environment(native) as (builder, _), \
                patch("atexit.register") as register:
            self.assertIs(builder("no-hook").load(), native)
            register.assert_not_called()

    def test_failed_build_does_not_register_or_cache(self):
        release = Mock()
        native = types.SimpleNamespace(release_cached_handles=release)
        with fake_builder_environment(native) as (builder, jit_load), \
                patch("atexit.register") as register:
            jit_load.side_effect = [RuntimeError("build failed"), native]
            instance = builder("retry")
            with self.assertRaisesRegex(RuntimeError, "build failed"):
                instance.load()
            register.assert_not_called()
            self.assertIs(instance.load(), native)
            self.assertEqual(jit_load.call_count, 2)
            register.assert_called_once_with(release)

    def test_native_resources_release_before_backend_shutdown(self):
        result = subprocess.run([sys.executable, __file__, "--exit-order-child"],
                                capture_output=True, text=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["HANDLES_RELEASED", "BACKEND_SHUTDOWN"])


if __name__ == "__main__":
    if sys.argv[1:] == ["--exit-order-child"]:
        run_exit_order_child()
    else:
        unittest.main(verbosity=2)
