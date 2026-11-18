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

import importlib.util
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
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


def run_host_command(command, directory):
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True,
                            check=False, timeout=60)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)


def create_host_libraries(directory, cxx, ar):
    # Harmless fixture archives stand in only for linkable library files;
    # no vendor API or device operation is implemented or invoked.
    (directory / "empty.cpp").write_text("void fixture_unused() {}\n", encoding="utf-8")
    run_host_command([cxx, "-c", "empty.cpp", "-o", "empty.o"], directory)
    for name in ("ops_sparse", "torch_npu", "c_sec"):
        run_host_command([ar, "rcs", f"lib{name}.a", "empty.o"], directory)


def builder_linker_flags(directory):
    # Observe flags at the public load boundary; keep source/path discovery
    # and the production builder's flag construction active.
    with fake_builder_environment(types.SimpleNamespace()) as (builder, jit_load), \
            patch.dict(os.environ, {"OPS_SPARSE_LIB_DIR": str(directory),
                                    "ASCEND_HOME_PATH": "/fake/cann"}):
        builder("link-path-probe").load()
        return jit_load.call_args.kwargs["extra_ldflags"]


def builder_linker_flags_for_platform(directory, platform_name):
    # Observe platform-specific flags at the same public load boundary: os.name
    # is patched so load() -> _get_linker_flags selects the toolchain flavour,
    # without the test reaching into protected members. Path discovery still
    # runs (isdir/isfile are mocked), so the configured directory stays first.
    with fake_builder_environment(types.SimpleNamespace()) as (builder, jit_load), \
            patch.dict(os.environ, {"OPS_SPARSE_LIB_DIR": str(directory),
                                    "ASCEND_HOME_PATH": "/fake/cann"}), \
            patch.object(os, "name", platform_name):
        builder("link-path-probe").load()
        return jit_load.call_args.kwargs["extra_ldflags"]


class OpBuilderLinkerPathTest(unittest.TestCase):
    def test_library_flags_round_trip_ninja_and_shell(self):
        for directory in ("/plain/lib", "/lib with spaces", "/single'quote", "/cash$directory"):
            with self.subTest(directory=directory):
                flags = builder_linker_flags(directory)
                decoded = shlex.split(" ".join(flags).replace("$$", "$"))
                self.assertEqual(decoded[:2], [f"-L{directory}", f"-Wl,-rpath,{directory}"])
                self.assertEqual(decoded[-3:], ["-lops_sparse", "-ltorch_npu", "-lc_sec"])

    def test_plain_flags_preserve_link_order(self):
        arch_root = f"/fake/cann/{platform.machine()}-linux"
        self.assertEqual(builder_linker_flags("/first"), [
            "-L/first", "-Wl,-rpath,/first",
            "-L/fake/torch_npu/lib", "-Wl,-rpath,/fake/torch_npu/lib",
            "-L/fake/cann/lib64", "-Wl,-rpath,/fake/cann/lib64",
            f"-L{arch_root}/lib64", f"-Wl,-rpath,{arch_root}/lib64",
            "-lops_sparse", "-ltorch_npu", "-lc_sec",
        ])

    def test_windows_flags_use_libpath_and_dot_lib(self):
        # link.exe has no -L/-rpath: it takes /LIBPATH: and <name>.lib, and POSIX
        # single quotes would reach it verbatim, so the flags must differ.
        for directory in ("C:\\plain\\lib", "C:\\lib with spaces", "C:\\cash$directory"):
            with self.subTest(directory=directory):
                flags = builder_linker_flags_for_platform(directory, "nt")
                self.assertEqual(flags[0], '"/LIBPATH:{}"'.format(directory.replace("$", "$$")))
                self.assertEqual(flags[-3:], ["ops_sparse.lib", "torch_npu.lib", "c_sec.lib"])
                joined = " ".join(flags)
                self.assertNotIn("'", joined)
                self.assertNotIn("-Wl,-rpath", joined)
                self.assertNotIn("-lops_sparse", joined)

    def test_posix_flags_unchanged_by_platform_split(self):
        flags = builder_linker_flags_for_platform("/plain/lib", "posix")
        self.assertEqual(flags[:2], ["-L/plain/lib", "-Wl,-rpath,/plain/lib"])
        self.assertEqual(flags[-3:], ["-lops_sparse", "-ltorch_npu", "-lc_sec"])

    def test_get_linker_flags_follows_os_name(self):
        nt_flags = builder_linker_flags_for_platform("/fake/lib", "nt")
        self.assertEqual(nt_flags[0], '"/LIBPATH:/fake/lib"')
        self.assertEqual(nt_flags[-1], "c_sec.lib")
        posix_flags = builder_linker_flags_for_platform("/fake/lib", "posix")
        self.assertEqual(posix_flags[0], "-L/fake/lib")

    def test_real_ninja_link_preserves_library_directory(self):
        tools = [shutil.which(name) for name in ("c++", "ar", "ninja")]
        if not all(tools):
            self.skipTest("Host integration test requires c++, ar and ninja")
        cxx, ar, ninja = tools
        for name in ("plain", "lib with spaces", "single'quote", "cash$directory"):
            with self.subTest(directory=name), tempfile.TemporaryDirectory() as temporary:
                self.link_fixture(Path(temporary), name, cxx, ar, ninja)

    def link_fixture(self, root, name, cxx, ar, ninja):
        libraries = root / name
        libraries.mkdir()
        create_host_libraries(libraries, cxx, ar)
        (root / "main.cpp").write_text("int main() {return 0;}\n", encoding="utf-8")
        flags = " ".join(builder_linker_flags(libraries))
        # Match Torch's POSIX Ninja variable/command boundary. All path names
        # and commands here are controlled Host test fixtures.
        (root / "build.ninja").write_text(
            f"ldflags = {flags}\nrule link\n"
            f"  command = {shlex.quote(cxx)} main.cpp -o program $ldflags\n"
            "build program: link main.cpp\n", encoding="utf-8",
        )
        run_host_command([ninja, "-f", "build.ninja"], root)
        self.assertTrue((root / "program").is_file())

    def test_real_torch_loader_preserves_linker_and_include_paths(self):
        tools = [shutil.which(name) for name in ("c++", "ar", "ninja")]
        if not all(tools):
            self.skipTest("Host integration test requires c++, ar and ninja")
        cxx, ar, _ = tools
        for name in ("plain", "lib with spaces", "single'quote", "cash$directory"):
            with self.subTest(directory=name), tempfile.TemporaryDirectory() as temporary:
                self.torch_link_fixture(Path(temporary), name, cxx, ar)

    def torch_link_fixture(self, root, name, cxx, ar):
        from torch.utils.cpp_extension import load

        libraries = root / name
        libraries.mkdir()
        create_host_libraries(libraries, cxx, ar)
        headers = root / "include with spaces and 'quote"
        headers.mkdir()
        (headers / "fixture.h").write_text("constexpr int fixture_value = 0;\n", encoding="utf-8")
        source = root / "main.cpp"
        source.write_text('#include "fixture.h"\nint main() {return fixture_value;}\n', encoding="utf-8")
        build = root / "build"
        build.mkdir()
        # Exercise the real Torch loader/Ninja writer with unchanged raw
        # include paths. Torch already quotes POSIX include arguments itself.
        executable = load(
            name="ops_sparse_host_link_probe", sources=[str(source)],
            extra_include_paths=[str(headers)],
            extra_ldflags=builder_linker_flags(libraries),
            build_directory=str(build), is_python_module=False,
            is_standalone=True, with_cuda=False, verbose=False,
        )
        run_host_command([executable], root)


if __name__ == "__main__":
    if sys.argv[1:] == ["--exit-order-child"]:
        run_exit_order_child()
    else:
        unittest.main(verbosity=2)
