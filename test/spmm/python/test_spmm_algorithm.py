# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CPU mock checks for the C++ adapter's native algorithm arguments."""

from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest

import torch
from torch.utils.cpp_extension import include_paths, library_paths


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "algorithm_mock"


class SpmmAlgorithmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="spmm-algorithm-")
        cls.binary = Path(cls.directory.name) / "adapter_algorithm"
        compiler = shutil.which("c++")
        if compiler is None:
            cls.directory.cleanup()
            raise unittest.SkipTest("a C++ compiler is required for the adapter mock")
        include_dirs = [FIXTURES, ROOT / "include", *include_paths(),
                        sysconfig.get_path("include")]
        libraries = library_paths()
        command = [compiler, "-std=c++17", "-O0", "-g0",
                   "-D_GLIBCXX_USE_CXX11_ABI={}".format(int(torch.compiled_with_cxx11_abi())),
                   *["-I{}".format(path) for path in include_dirs],
                   str(FIXTURES / "adapter_algorithm.cpp"), "-o", str(cls.binary),
                   *["-L{}".format(path) for path in libraries],
                   *["-Wl,-rpath,{}".format(path) for path in libraries],
                   "-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_python",
                   "-L{}".format(sysconfig.get_config_var("LIBDIR")),
                   "-lpython{}.{}".format(*sys.version_info[:2]), "-ldl"]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            cls.directory.cleanup()
            raise AssertionError(result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_default_algorithm_across_native_phases(self):
        for mode in ("inplace", "native-out", "private-out", "private-fallback"):
            dtypes = ("float16", "bfloat16", "float32", "complex64")
            if mode.startswith("private"):
                dtypes = ("float32", "complex64")
            for dtype in dtypes:
                with self.subTest(mode=mode, dtype=dtype):
                    result = subprocess.run([str(self.binary), mode, dtype],
                                            capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
