# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Observe actual top-level SDK selection at the ASC package lookup boundary."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class SdkSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sparse-sdk-selection-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.probe = self.directory / "asc-probe"
        self.probe.mkdir()
        (self.probe / "ASCConfig.cmake").write_text(
            'file(WRITE "${CMAKE_BINARY_DIR}/selected.txt" "${ASCEND_CANN_PACKAGE_PATH}")\n'
            'message(FATAL_ERROR "SDK_SELECTION_OBSERVED")\n')

    def check_selection(self, explicit, environment, expected):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("ASCEND_", "CMAKE_"))}
        env.update(environment)
        build = self.directory / "build"
        command = ["cmake", "-S", str(ROOT), "-B", str(build), "-DASC_DIR=" + str(self.probe)]
        if explicit is not None:
            command.append("-DASCEND_CANN_PACKAGE_PATH=" + explicit)
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SDK_SELECTION_OBSERVED", result.stdout + result.stderr)
        self.assertEqual((build / "selected.txt").read_text(), expected)

    def test_explicit_without_environment(self):
        self.check_selection("/selected/sdk", {}, "/selected/sdk")

    def test_explicit_overrides_both_environment_paths(self):
        self.check_selection("/selected/sdk", {"ASCEND_HOME_PATH": "/env/home",
                             "ASCEND_TOOLKIT_HOME": "/env/toolkit"}, "/selected/sdk")

    def test_explicit_with_spaces(self):
        self.check_selection("/selected/sdk with spaces", {}, "/selected/sdk with spaces")

    def test_nonempty_false_like_name_is_preserved(self):
        self.check_selection("OFF", {"ASCEND_HOME_PATH": "/env/home"}, "OFF")

    def test_home_fallback_has_priority(self):
        self.check_selection(None, {"ASCEND_HOME_PATH": "/env/home",
                             "ASCEND_TOOLKIT_HOME": "/env/toolkit"}, "/env/home")

    def test_toolkit_fallback(self):
        self.check_selection(None, {"ASCEND_HOME_PATH": "",
                             "ASCEND_TOOLKIT_HOME": "/env/toolkit"}, "/env/toolkit")

    def test_empty_explicit_uses_environment(self):
        self.check_selection("", {"ASCEND_HOME_PATH": "/env/home"}, "/env/home")

    def test_no_sources_is_empty(self):
        self.check_selection(None, {}, "")

    def test_cached_explicit_survives_environment_change(self):
        self.check_selection("/selected/sdk", {}, "/selected/sdk")
        self.check_selection(None, {"ASCEND_HOME_PATH": "/new/environment"}, "/selected/sdk")


if __name__ == "__main__":
    unittest.main()
