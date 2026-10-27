# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Build actual selected-operator wheels without SDK compilation or loading."""

import email
import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def command_path(name):
    path = shutil.which(name)
    assert path is not None, f"Required Host command not found: {name}"
    assert Path(path).is_absolute(), path
    return path


def create_checkout(tmp_path, has_spaces):
    checkout = tmp_path / ("checkout with spaces" if has_spaces else "plain-checkout")
    checkout.mkdir()
    shutil.copyfile(ROOT / "build.sh", checkout / "build.sh")
    # Copy the complete real packaging inputs and operator source trees. No
    # Torch, CANN, subprocess or setuptools API implementation is substituted.
    ignored = shutil.ignore_patterns(
        "build", "build_out", "dist", "*.egg-info", "__pycache__"
    )
    for name in ["torch_extension", "sparse", "include", "test"]:
        shutil.copytree(ROOT / name, checkout / name, ignore=ignored)
    return checkout


def assert_wheel_source_bytes(checkout):
    wheels = list((checkout / "build_out").glob("cann_ops_sparse-*.whl"))
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as wheel:
        members = {
            "cann_ops_sparse/ops/sparse/gather/gather.py": "sparse/gather/torch_extension/gather.py",
            "cann_ops_sparse/csrc/sparse/gather/gather.cpp": "sparse/gather/torch_extension/csrc/gather.cpp",
            "cann_ops_sparse/common/cann_ops_sparse.h": "include/cann_ops_sparse.h",
        }
        for member, source in members.items():
            assert wheel.read(member) == (checkout / source).read_bytes()
        metadata_names = [
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_names) == 1
        metadata = email.message_from_bytes(wheel.read(metadata_names[0]))
        assert metadata["Name"] == "cann_ops_sparse"
        assert metadata["Version"] == "0.1.0"


@pytest.mark.parametrize("has_spaces", [False, True])
@pytest.mark.parametrize("external_cwd", [False, True])
def test_complete_wheel_driver_preserves_checkout_path(
    tmp_path, has_spaces, external_cwd
):
    build_spec = importlib.util.find_spec("build")
    assert build_spec is not None and build_spec.origin, (
        "Install actual torch_extension build requirements first"
    )
    assert Path(sys.executable).is_absolute()
    checkout = create_checkout(tmp_path, has_spaces)
    outside = tmp_path / "external-cwd"
    outside.mkdir()
    # The wheel-only branch checks this string for nonemptiness, then exits
    # before any SDK/CMake work. This existing empty Host directory is explicitly
    # not an installed or simulated CANN SDK and contains no tool/API stand-ins.
    unused_prefix = tmp_path / "unused-host-env-prefix"
    unused_prefix.mkdir()
    environment = dict(os.environ)
    environment["ASCEND_HOME_PATH"] = str(unused_prefix)
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    command = [
        command_path("bash"),
        str(checkout / "build.sh"),
        "--torch_extension",
        "--ops=gather",
    ]
    run = subprocess.run(
        command,
        cwd=outside if external_cwd else checkout,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert_wheel_source_bytes(checkout)
