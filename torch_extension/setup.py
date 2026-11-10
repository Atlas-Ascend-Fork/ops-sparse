# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import atexit
import os
import shutil
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    except ImportError:
        _bdist_wheel = None


HERE = Path(__file__).resolve().parent
BUNDLED_SOURCES = "_sdist_sources"
BUNDLED_MARKER = ".cann_ops_sparse_sdist"
OPS_SPARSE_ROOT = (
    HERE / BUNDLED_SOURCES
    if (HERE / BUNDLED_SOURCES / BUNDLED_MARKER).is_file()
    else HERE.parent
)
PACKAGE_NAME = "cann_ops_sparse"
_LICENSE_TARGET = HERE / "LICENSE"
_LICENSE_STAGED = False


def _stage_license():
    """Make the repository LICENSE discoverable without committing a copy."""
    global _LICENSE_STAGED
    source = OPS_SPARSE_ROOT / "LICENSE"
    if _LICENSE_TARGET.exists() or not source.is_file():
        return
    shutil.copy2(source, _LICENSE_TARGET)
    _LICENSE_STAGED = True


def _cleanup_staged_license():
    if _LICENSE_STAGED:
        _LICENSE_TARGET.unlink(missing_ok=True)


_stage_license()
atexit.register(_cleanup_staged_license)


def _operator_extensions():
    """Return ``(category, op, source_dir)`` entries owned by operators."""
    extensions = []
    selected_ops = {
        op.strip() for op in os.environ.get("TORCH_EXTENSION_OPS", "").split(",") if op.strip()
    }
    sparse_root = OPS_SPARSE_ROOT / "sparse"
    for op_dir in sorted(sparse_root.iterdir()):
        extension_dir = op_dir / "torch_extension"
        if extension_dir.is_dir() and (not selected_ops or op_dir.name in selected_ops):
            extensions.append(("sparse", op_dir.name, extension_dir))
    found_ops = {op_name for _, op_name, _ in extensions}
    unknown_ops = selected_ops - found_ops
    if unknown_ops:
        raise RuntimeError(
            "TORCH_EXTENSION_OPS contains operators without torch_extension: "
            + ", ".join(sorted(unknown_ops))
        )
    return extensions


def _copy_extension_sources(source_dir, python_destination, csrc_destination):
    """Copy the same owned source files into wheels and source archives."""
    python_destination.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.py"):
        shutil.copy2(source, python_destination / source.name)
    csrc_sources = list((source_dir / "csrc").glob("*.cpp"))
    if csrc_sources:
        csrc_destination.mkdir(parents=True, exist_ok=True)
    for source in csrc_sources:
        shutil.copy2(source, csrc_destination / source.name)


class SdistWithOps(_sdist):
    """Bundle selected operator sources without modifying the checkout."""

    def make_release_tree(self, base_dir, files):
        extensions = _operator_extensions()
        super().make_release_tree(base_dir, files)
        source_root = Path(base_dir) / BUNDLED_SOURCES
        # A retained release tree must not leak a previous operator selection.
        if source_root.exists():
            shutil.rmtree(source_root)
        source_root.mkdir(parents=True, exist_ok=True)
        (source_root / BUNDLED_MARKER).write_text(
            "Self-contained CANN ops-sparse source archive.\n", encoding="utf-8"
        )
        for category, op_name, source_dir in extensions:
            destination = source_root / category / op_name / "torch_extension"
            _copy_extension_sources(source_dir, destination, destination / "csrc")
        include = source_root / "include"
        include.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OPS_SPARSE_ROOT / "include" / "cann_ops_sparse.h", include)
        shutil.copy2(OPS_SPARSE_ROOT / "LICENSE", source_root / "LICENSE")
        # The sdist is a redistributable source artifact; keep the license at
        # the archive root as well as inside the bundled sources so downstream
        # tools can discover it without knowing the private source layout.
        shutil.copy2(OPS_SPARSE_ROOT / "LICENSE", Path(base_dir) / "LICENSE")


class BuildPyWithOps(_build_py):
    """Collect each operator's torch_extension sources into the wheel package."""

    def run(self):
        super().run()
        package_root = Path(self.build_lib) / PACKAGE_NAME
        # ``build_py`` may reuse an existing build directory.  Remove generated
        # operator artifacts first so deleted wrapper sources cannot leak into a wheel.
        shutil.rmtree(package_root / "csrc", ignore_errors=True)
        extensions = _operator_extensions()
        for category in {item[0] for item in extensions}:
            shutil.rmtree(package_root / "ops" / category, ignore_errors=True)
        categories = {}
        for category, op_name, source_dir in extensions:
            categories.setdefault(category, []).append(op_name)
            destination = package_root / "ops" / category / op_name
            _copy_extension_sources(
                source_dir, destination, package_root / "csrc" / category / op_name
            )

        common_destination = package_root / "common"
        common_destination.mkdir(parents=True, exist_ok=True)
        for source in (HERE / PACKAGE_NAME / "common").glob("*.h"):
            shutil.copy2(source, common_destination / source.name)
        shutil.copy2(
            OPS_SPARSE_ROOT / "include" / "cann_ops_sparse.h",
            common_destination / "cann_ops_sparse.h",
        )

        for category, op_names in categories.items():
            category_init = package_root / "ops" / category / "__init__.py"
            category_init.parent.mkdir(parents=True, exist_ok=True)
            category_init.write_text(
                "".join("from . import {0}\n".format(name) for name in op_names),
                encoding="utf-8",
            )


def _clean_build_artifacts():
    """Remove wheel build artifacts before starting a new wheel build."""
    for path in (HERE / "build", HERE / "dist", *HERE.glob("*.egg-info")):
        if path.is_dir():
            shutil.rmtree(path)


if _bdist_wheel is not None:

    class BdistWheelWithClean(_bdist_wheel):
        """Build a wheel from a clean staging directory."""

        def run(self):
            _clean_build_artifacts()
            super().run()

else:
    BdistWheelWithClean = None


_CMDCLASS = {"build_py": BuildPyWithOps, "sdist": SdistWithOps}
if BdistWheelWithClean is not None:
    _CMDCLASS["bdist_wheel"] = BdistWheelWithClean


setup(
    name="cann_ops_sparse",
    version="0.1.0",
    description="CannOpsSparse PyTorch extension",
    license="CANN Open Software License Agreement Version 2.0",
    license_files=["LICENSE"],
    packages=find_packages(),
    install_requires=["torch>=2.6.0", "torch_npu"],
    cmdclass=_CMDCLASS,
    zip_safe=False,
)
