# Copyright (c) 2026 AscSpMM Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise mixed-kernel tool discovery without requiring a CANN installation."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

EXTENSION = Path(__file__).resolve().parents[2] / "cmake/operator_extensions.cmake"


class ToolchainLayoutTest(unittest.TestCase):
    def configure_layout(self, architecture, compiler_dir, linker_dir, driver="ccec", parsed=False):
        cmake = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")
        self.assertIsNotNone(cmake, "CMake is required for the toolchain layout tests")
        self.assertTrue(Path(cmake).is_absolute(), "CMAKE_COMMAND must be an absolute path")
        cmake = str(Path(cmake).resolve())
        with tempfile.TemporaryDirectory(prefix="cube_spmm_1d_layout_") as directory:
            root = Path(directory)
            package = root / "CANN package"
            compiler = package / compiler_dir / driver
            linker = package / linker_dir / "ld.lld"
            for tool in (compiler, linker):
                tool.parent.mkdir(parents=True, exist_ok=True)
                tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                tool.chmod(0o755)
            definitions = [
                'cmake_minimum_required(VERSION 3.16)',
                'project(layout NONE)',
                f'set(CMAKE_SYSTEM_PROCESSOR "{architecture}")',
                f'set(ASCEND_CANN_PACKAGE_PATH "{package.as_posix()}")',
                f'include("{EXTENSION.as_posix()}")',
            ]
            if parsed:
                definitions.extend([
                    f'set(CMAKE_ASC_COMPILER "{compiler.as_posix()}")',
                    f'set(CMAKE_ASC_LLD_LINKER "{linker.as_posix()}")',
                ])
            definitions.extend([
                'ops_sparse_find_cce_tools(driver linker)',
                f'if(NOT driver STREQUAL "{compiler.as_posix()}" OR NOT linker STREQUAL "{linker.as_posix()}")',
                '  message(FATAL_ERROR "wrong mixed-kernel tools: ${driver};${linker}")',
                'endif()',
            ])
            (root / "CMakeLists.txt").write_text("\n".join(definitions), encoding="utf-8")
            result = subprocess.run([cmake, "-S", str(root), "-B", str(root / "build")],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_legacy_layout(self):
        self.configure_layout("aarch64", "compiler/ccec_compiler/bin", "compiler/ccec_compiler/bin")

    def test_platform_layout_arm(self):
        self.configure_layout("aarch64", "aarch64-linux/ccec_compiler/bin", "aarch64-linux/ccec_compiler/bin")

    def test_platform_layout_x86(self):
        self.configure_layout("x86_64", "x86_64-linux/ccec_compiler/bin", "x86_64-linux/ccec_compiler/bin")

    def test_bisheng_only_layout(self):
        self.configure_layout("x86_64", "x86_64-linux/ccec_compiler/bin",
                              "x86_64-linux/ccec_compiler/bin", driver="bisheng", parsed=True)

    def test_flat_compiler_split_linker(self):
        self.configure_layout("x86_64", "bin", "x86_64-linux/ccec_compiler/bin", driver="bisheng", parsed=True)

    def test_explicit_toolchain_locations(self):
        self.configure_layout("aarch64", "custom/compiler", "custom/linker", driver="bisheng", parsed=True)


if __name__ == "__main__":
    unittest.main()
