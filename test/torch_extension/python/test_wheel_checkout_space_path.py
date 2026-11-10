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
import tarfile
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
    shutil.copyfile(ROOT / "LICENSE", checkout / "LICENSE")
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
        assert metadata["License-File"] == "LICENSE"
        license_names = [
            name for name in wheel.namelist()
            if name.startswith("cann_ops_sparse-0.1.0.dist-info/") and name.endswith("/LICENSE")
        ]
        assert license_names, wheel.namelist()
        assert wheel.read(license_names[0]) == (checkout / "LICENSE").read_bytes()


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


def run_package_command(project, output, kind, selected=None, succeed=True):
    environment = dict(os.environ)
    environment.pop("TORCH_EXTENSION_OPS", None)
    if selected is not None:
        environment["TORCH_EXTENSION_OPS"] = selected
    run = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--" + kind,
         "--outdir", str(output), str(project)],
        env=environment, capture_output=True, text=True, timeout=120, check=False,
    )
    if succeed:
        assert run.returncode == 0, run.stdout + run.stderr
    else:
        assert run.returncode != 0, run.stdout + run.stderr
    return run


def extract_source_archive(archive, destination):
    with tarfile.open(archive) as source:
        # This is the archive just built by the test. Check membership before
        # extraction, including links, so the test never writes outside its root.
        for member in source.getmembers():
            path = Path(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            assert member.isfile() or member.isdir(), member.name
        if hasattr(tarfile, "data_filter"):
            source.extractall(destination, filter="data")
        else:
            source.extractall(destination)
    roots = list(destination.iterdir())
    assert len(roots) == 1
    return roots[0]


def wheel_contents(folder):
    wheels = list(folder.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}


def assert_equivalent_wheels(direct, rebuilt):
    first, second = wheel_contents(direct), wheel_contents(rebuilt)
    assert first.keys() == second.keys()
    for name in first:
        # RECORD encodes archive membership/hashes; its row ordering need not
        # identify equivalent payloads. Every other file, including METADATA,
        # WHEEL and registration/native/header bytes, must match exactly.
        if name.endswith(".dist-info/RECORD"):
            assert sorted(first[name].splitlines()) == sorted(second[name].splitlines()), name
        else:
            assert first[name] == second[name], name


def assert_archive_sources(archive, selected):
    expected_ops = {
        path.parent.name for path in (ROOT / "sparse").glob("*/torch_extension")
        if path.is_dir() and (not selected or path.parent.name in selected.split(","))
    }
    with tarfile.open(archive) as source:
        files = {"/".join(Path(member.name).parts[1:]): member
                 for member in source.getmembers() if member.isfile()}
        bundled_ops = {Path(name).parts[2] for name in files
                       if name.startswith("_sdist_sources/sparse/")}
        assert bundled_ops == expected_ops
        for op in expected_ops:
            extension = ROOT / "sparse" / op / "torch_extension"
            sources = list(extension.glob("*.py")) + list((extension / "csrc").glob("*.cpp"))
            for original in sources:
                name = "_sdist_sources/sparse/" + op + "/torch_extension/"
                name += original.relative_to(extension).as_posix()
                assert source.extractfile(files[name]).read() == original.read_bytes()
        for original in (ROOT / "torch_extension/cann_ops_sparse/common").glob("*.h"):
            name = "cann_ops_sparse/common/" + original.name
            assert source.extractfile(files[name]).read() == original.read_bytes()
        for name, original in [("_sdist_sources/include/cann_ops_sparse.h", ROOT / "include/cann_ops_sparse.h"),
                               ("LICENSE", ROOT / "LICENSE"),
                               ("_sdist_sources/LICENSE", ROOT / "LICENSE")]:
            assert source.extractfile(files[name]).read() == original.read_bytes()
        assert not any(Path(name).suffix in (".so", ".o", ".a") for name in files)
        assert "_sdist_sources/.cann_ops_sparse_sdist" in files
        assert "requirements.txt" in files and "README.md" in files


def test_same_named_directory_does_not_change_checkout_wheel(tmp_path):
    checkout = create_checkout(tmp_path, False)
    project = checkout / "torch_extension"
    run_package_command(project, tmp_path / "expected", "wheel", "gather")
    (project / "_sdist_sources").mkdir()
    run_package_command(project, tmp_path / "actual", "wheel", "gather")
    assert_equivalent_wheels(tmp_path / "expected", tmp_path / "actual")


def test_same_named_directory_does_not_change_source_archive(tmp_path):
    # The sdist command is the other consumer of the source root: a checkout
    # directory that merely shares the bundled name must not become the
    # archived sources either.
    checkout = create_checkout(tmp_path, False)
    project = checkout / "torch_extension"
    run_package_command(project, tmp_path / "direct", "wheel", "gather")
    (project / "_sdist_sources").mkdir()
    run_package_command(project, tmp_path / "archives", "sdist", "gather")
    archive, = (tmp_path / "archives").glob("*.tar.gz")
    assert_archive_sources(archive, "gather")
    outside = tmp_path / "outside"
    outside.mkdir()
    unpacked = extract_source_archive(archive, outside)
    run_package_command(unpacked, tmp_path / "rebuilt", "wheel")
    assert_equivalent_wheels(tmp_path / "direct", tmp_path / "rebuilt")


@pytest.mark.parametrize("selected", [None, "gather", "gather,spgemm"])
@pytest.mark.parametrize("has_spaces", [False, True])
def test_sdist_roundtrip_matches_checkout_wheel(tmp_path, selected, has_spaces):
    checkout = create_checkout(tmp_path, has_spaces)
    project = checkout / "torch_extension"
    run_package_command(project, tmp_path / "direct", "wheel", selected)
    run_package_command(project, tmp_path / "archives", "sdist", selected)
    archive, = (tmp_path / "archives").glob("*.tar.gz")
    assert_archive_sources(archive, selected)
    outside = tmp_path / "outside checkout"
    outside.mkdir()
    unpacked = extract_source_archive(archive, outside)
    assert not (unpacked.parent / "sparse").exists()
    assert not (unpacked.parent / "include").exists()
    run_package_command(unpacked, tmp_path / "rebuilt", "wheel")
    assert_equivalent_wheels(tmp_path / "direct", tmp_path / "rebuilt")
    assert not (project / "_sdist_sources").exists()


@pytest.mark.parametrize("kind", ["sdist", "wheel"])
def test_source_archive_rejects_unavailable_operator(tmp_path, kind):
    checkout = create_checkout(tmp_path, False)
    run_package_command(checkout / "torch_extension", tmp_path / "archives", "sdist", "gather")
    archive, = (tmp_path / "archives").glob("*.tar.gz")
    outside = tmp_path / "outside"
    outside.mkdir()
    unpacked = extract_source_archive(archive, outside)
    run = run_package_command(unpacked, tmp_path / "invalid", kind, "spgemm", succeed=False)
    assert "TORCH_EXTENSION_OPS contains operators without torch_extension: spgemm" in run.stdout + run.stderr


def test_source_archive_can_be_recreated_and_filtered(tmp_path):
    checkout = create_checkout(tmp_path, False)
    project = checkout / "torch_extension"
    run_package_command(project, tmp_path / "direct", "wheel", "gather")
    run_package_command(project, tmp_path / "first", "sdist", "gather,spgemm")
    archive, = (tmp_path / "first").glob("*.tar.gz")
    outside = tmp_path / "outside"
    outside.mkdir()
    unpacked = extract_source_archive(archive, outside)
    run_package_command(unpacked, tmp_path / "second", "sdist", "gather")
    second, = (tmp_path / "second").glob("*.tar.gz")
    assert_archive_sources(second, "gather")
    nested = tmp_path / "second outside"
    nested.mkdir()
    final = extract_source_archive(second, nested)
    run_package_command(final, tmp_path / "rebuilt", "wheel")
    assert_equivalent_wheels(tmp_path / "direct", tmp_path / "rebuilt")
