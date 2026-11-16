# Copyright (c) 2026 Xztxzt
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host-only tests of the actual shared header, using struct binary16 as oracle."""
import math
from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class Float16ConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("g++ is required for host header tests")
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        source = Path(cls.directory.name) / "convert.cpp"
        cls.executable = Path(cls.directory.name) / "convert"
        source.write_text(r'''#include <cmath>
#include <cstring>
#include <iostream>
#include "sparse/common/aclsparse_fp16_utils.h"
int main()
{
    uint32_t bits;
    while (std::cin >> std::hex >> bits) {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        std::cout << std::hex << aclsparse::Float32ToFloat16Bits(value) << '\n';
    }
}
''')
        subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-I" + str(ROOT), str(source), "-o", str(cls.executable)], check=True)

    def assert_conversions(self, bits):
        run = subprocess.run([str(self.executable)],
                             input="".join(f"{x:x}\n" for x in bits),
                             text=True, capture_output=True, check=True)
        actual = [int(x, 16) for x in run.stdout.splitlines()]
        self.assertEqual(len(actual), len(bits))
        for value_bits, result in zip(bits, actual):
            value = struct.unpack("<f", struct.pack("<I", value_bits))[0]
            if math.isnan(value):
                self.assertEqual(result & 0x7c00, 0x7c00, hex(value_bits))
                self.assertNotEqual(result & 0x03ff, 0, hex(value_bits))
                self.assertEqual(result >> 15, value_bits >> 31, hex(value_bits))
                continue
            try:
                expected = struct.unpack("<H", struct.pack("<e", value))[0]
            except OverflowError:
                expected = ((value_bits >> 16) & 0x8000) | 0x7c00
            self.assertEqual(result, expected,
                             f"float32=0x{value_bits:08x}: got 0x{result:04x}, expected 0x{expected:04x}")

    def test_all_half_encodings(self):
        values = [struct.unpack("<e", struct.pack("<H", x))[0] for x in range(65536)]
        self.assert_conversions([struct.unpack("<I", struct.pack("<f", x))[0] for x in values])

    def test_every_positive_rounding_midpoint_and_neighbors(self):
        bits = []
        for half in range(0x7bff):
            lower = struct.unpack("<e", struct.pack("<H", half))[0]
            upper = struct.unpack("<e", struct.pack("<H", half + 1))[0]
            midpoint = (lower + upper) / 2
            middle_bits = struct.unpack("<I", struct.pack("<f", midpoint))[0]
            for neighbor in (middle_bits - 1, middle_bits, middle_bits + 1):
                bits.extend((neighbor, neighbor | 0x80000000))
        self.assert_conversions(bits)

    def test_overflow_and_nonfinite_boundaries(self):
        values = [65504.0, 65519.0, 65520.0, 65536.0, float("inf")]
        bits = [struct.unpack("<I", struct.pack("<f", x))[0] for x in values]
        bits += [0x7f800001, 0x7fc00000, 0x7fffffff]
        self.assert_conversions(bits + [x | 0x80000000 for x in bits])

    def test_random_float32_bit_patterns(self):
        rng = random.Random(20261115)
        self.assert_conversions([rng.getrandbits(32) for _ in range(100000)])


if __name__ == "__main__":
    unittest.main()
