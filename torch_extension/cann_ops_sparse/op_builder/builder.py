# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Shared JIT builder for ATen operator C++ wrappers.

Native sources are added with their corresponding wrapper PRs.
"""

import os
import platform
import threading
from importlib.util import find_spec
from abc import ABC, abstractmethod
from typing import List

from torch.utils.cpp_extension import load


_LOAD_LOCK = threading.Lock()


class OpBuilder(ABC):
    """JIT-builds an ATen operator's C++ wrapper on first NPU call."""

    _loaded_ops = {}

    def __init__(self, name: str, category: str = None):
        self.name = name
        self.category = category
        self._initialized = False

    @staticmethod
    def _append_existing_paths(destination, candidates) -> None:
        destination.extend(path for path in candidates if os.path.isdir(path))

    @staticmethod
    def _get_linker_flags(library_dirs) -> List[str]:
        flags = []
        for directory in library_dirs:
            flags.extend(["-L{}".format(directory), "-Wl,-rpath,{}".format(directory)])
        # Wrappers call securec functions directly (for example memcpy_s in
        # SpGEMM).  Declare the dependency instead of relying on another loaded
        # CANN library to expose its symbols transitively.
        return flags + ["-lops_sparse", "-ltorch_npu", "-lc_sec"]

    @abstractmethod
    def sources(self) -> List[str]:
        """Relative paths to C++ sources for this op's native C++ wrapper."""

    def load(self):
        # Serialize the complete cache-check/build/store sequence.  Multiple
        # first calls may otherwise start ninja in the same cache directory.
        with _LOAD_LOCK:
            self._ensure_initialized()
            if self.name not in self._loaded_ops:
                self._loaded_ops[self.name] = self._build_extension()
            return self._loaded_ops[self.name]

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        torch_npu_spec = find_spec("torch_npu")
        if torch_npu_spec is None or not torch_npu_spec.submodule_search_locations:
            raise RuntimeError("torch_npu must be installed before loading an ops-sparse C++ wrapper")
        self._torch_npu_path = next(iter(torch_npu_spec.submodule_search_locations))
        self._package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._cann_path = self._get_cann_path()
        self._ops_sparse_lib_dir = self._get_ops_sparse_lib_dir()
        self._initialized = True

    def _get_cann_path(self) -> str:
        """Resolve the CANN root with the same priority as torch_npu extensions."""
        configured_path = os.environ.get("ASCEND_HOME_PATH", "")
        if configured_path and os.path.isdir(configured_path):
            return configured_path
        return os.path.dirname(os.path.dirname(self._torch_npu_path))

    def _get_ops_sparse_lib_dir(self) -> str:
        """Use an explicit development path, then the standard CANN install path."""
        configured_path = os.environ.get("OPS_SPARSE_LIB_DIR", "")
        if configured_path:
            return configured_path
        return os.path.join(self._cann_path, "lib64")

    def _build_extension(self):
        sources = self._get_sources()
        include_paths, library_dirs = self._get_build_paths()
        return load(
            name="cann_ops_sparse_{}".format(self.name),
            sources=sources,
            extra_include_paths=include_paths,
            extra_cflags=["-O2", "-std=c++17"],
            extra_ldflags=self._get_linker_flags(library_dirs),
            verbose=True,
        )

    def _get_sources(self) -> List[str]:
        sources = [os.path.join(self._package_path, path) for path in self.sources()]
        missing = [path for path in sources if not os.path.isfile(path)]
        if missing:
            raise RuntimeError(
                "native C++ wrapper source is missing for {}: {}".format(self.name, ", ".join(missing))
            )

        ops_sparse_library = os.path.join(self._ops_sparse_lib_dir, "libops_sparse.so")
        if not os.path.isfile(ops_sparse_library):
            raise RuntimeError(
                "libops_sparse.so was not found in {}. Install the ops-sparse run package "
                "under ASCEND_HOME_PATH, or set OPS_SPARSE_LIB_DIR to its directory".format(
                    self._ops_sparse_lib_dir
                )
            )
        return sources

    def _get_build_paths(self):
        library_dirs = [self._ops_sparse_lib_dir, os.path.join(self._torch_npu_path, "lib")]
        include_paths = [os.path.join(self._package_path, "common")]
        self._append_cann_paths(include_paths, library_dirs)
        self._append_torch_npu_paths(include_paths)
        return include_paths, library_dirs

    def _append_cann_paths(self, include_paths, library_dirs) -> None:
        if not self._cann_path:
            return
        arch_root = os.path.join(self._cann_path, "{}-linux".format(platform.machine()))
        self._append_existing_paths(
            include_paths,
            (os.path.join(self._cann_path, "include"), os.path.join(arch_root, "include")),
        )
        self._append_existing_paths(
            library_dirs,
            (os.path.join(self._cann_path, "lib64"), os.path.join(arch_root, "lib64")),
        )

    def _append_torch_npu_paths(self, include_paths) -> None:
        # Prefer the active CANN headers.  They must precede torch_npu's bundled
        # ACL copy so both public ACLSparse declarations and torch_npu use one
        # compatible ACL type definition in the wrapper translation unit.
        include_paths.append(os.path.join(self._torch_npu_path, "include"))
        self._append_existing_paths(
            include_paths,
            (os.path.join(self._torch_npu_path, "include", "third_party", "acl", "inc"),),
        )
