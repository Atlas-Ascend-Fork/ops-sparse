# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Layout routing and optional operator capabilities for NPU matrix products.

This module has no ATen registration side effects. The registration layer uses
advertised backward capabilities for sparse requests and preserves existing
ATen autograd for other requests.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable, Optional, Tuple

import torch


_SUPPORTED_SPARSE = (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc)
BACKWARD_HELPER_KEYS = {
    "sparse_mask": ("SparsePrivateUse1", "SparseCsrPrivateUse1"),
    "add.out": ("SparsePrivateUse1", "SparseCsrPrivateUse1"),
    "add_.Tensor": ("SparsePrivateUse1", "SparseCsrPrivateUse1"),
}


@dataclass(frozen=True)
class MatmulRequest:
    """One functional or out matrix-product request."""

    operation: str
    mat1: torch.Tensor
    mat2: torch.Tensor
    self: Optional[torch.Tensor] = None
    beta: object = 1
    alpha: object = 1
    out: Optional[torch.Tensor] = None
    # The public ATen entry point.  ``operation`` describes the mathematical
    # family, while this preserves schema-specific autograd semantics.
    schema: str = "aten::mm"


@dataclass(frozen=True)
class MatmulCapability:
    """An operator-owned lazy implementation, independent of ATen schemas."""

    owner: str
    route: str
    forward: Callable
    operations: Tuple[str, ...] = ("mm", "addmm")
    backward: Optional[Callable] = None
    dtypes: Tuple[torch.dtype, ...] = ()
    device_types: Tuple[str, ...] = ()
    backward_helpers: Tuple[Tuple[str, Callable], ...] = ()


def _kind(tensor):
    if tensor.layout == torch.strided:
        return "dense"
    if tensor.layout in _SUPPORTED_SPARSE:
        if tensor.sparse_dim() != 2 or tensor.dense_dim() != 0:
            raise RuntimeError("matrix routing requires two sparse dimensions and no dense dimensions")
        return "sparse"
    raise RuntimeError("unsupported matrix layout: {}".format(tensor.layout))


def _validate(request):
    if request.operation not in ("mm", "addmm"):
        raise ValueError("unsupported matrix operation: {}".format(request.operation))
    a, b = request.mat1, request.mat2
    if a.dim() != 2 or b.dim() != 2 or a.size(1) != b.size(0):
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied: expected compatible 2-D matrices")
    if a.device != b.device or a.dtype != b.dtype:
        raise RuntimeError("mat1 and mat2 must share dtype and device")
    if request.operation == "mm" and request.self is not None:
        raise ValueError("mm does not accept an accumulated operand")
    if request.operation == "addmm" and request.self is None:
        raise ValueError("addmm requires an accumulated operand")
    for name, tensor in (("self", request.self), ("out", request.out)):
        if tensor is not None and (tensor.device != a.device or tensor.dtype != a.dtype):
            raise RuntimeError("{} must share the dtype and device of the operands".format(name))
    if request.out is not None and torch.is_grad_enabled():
        tensors = (a, b, request.self, request.out)
        if any(t is not None and t.requires_grad for t in tensors):
            raise RuntimeError("out= matrix products do not support automatic differentiation")


def _route(request):
    left, right = _kind(request.mat1), _kind(request.mat2)
    if left == right == "dense":
        route = "dense"
    else:
        route = "spgemm" if left == right else "spmm"
    if route == "spgemm":
        layouts = (request.mat1.layout, request.mat2.layout)
        allowed = ((torch.sparse_csr, torch.sparse_csr), (torch.sparse_coo, torch.sparse_coo))
        if layouts not in allowed:
            raise RuntimeError("sparse products require matching COO or CSR layouts")
        if request.operation == "addmm" and layouts[0] != torch.sparse_csr:
            raise RuntimeError("sparse addmm currently requires CSR operands")
        if request.self is not None:
            if request.self.layout != torch.sparse_csr:
                raise RuntimeError("self must use the sparse CSR layout for a sparse product")
            if tuple(request.self.shape) != (request.mat1.size(0), request.mat2.size(1)):
                raise RuntimeError("self must have shape matching the sparse product")
    elif request.self is not None:
        if request.self.layout != torch.strided:
            raise RuntimeError("self must be dense for a dense output")
        # Validate broadcasting without replacing self: backward must reduce
        # the accumulated gradient to the original operand's shape.
        _ = request.self.expand(request.mat1.size(0), request.mat2.size(1))
    if request.out is not None:
        expected = request.mat1.layout if route == "spgemm" else torch.strided
        if request.out.layout != expected:
            raise RuntimeError("out has an incompatible layout for the matrix product")
    return route, left == "dense" and right == "sparse"


def _csr(tensor):
    if tensor.layout == torch.sparse_csr:
        return tensor
    # COO duplicates must be summed before CSR conversion. These public ATen
    # conversions preserve autograd semantics where the backend supports them.
    if tensor.layout == torch.sparse_coo:
        tensor = tensor.coalesce()
    return tensor.to_sparse_csr()


def _spmm_request(request, transpose):
    if not transpose:
        return replace(request, mat1=_csr(request.mat1))
    accumulated = request.self
    if accumulated is not None:
        accumulated = accumulated.expand(request.mat1.size(0), request.mat2.size(1)).transpose(0, 1)
    output = request.out
    if output is not None:
        shape = (request.mat1.size(0), request.mat2.size(1))
        if tuple(output.shape) != shape:
            output.resize_(shape)
        output = output.transpose(0, 1)
    return replace(
        request,
        mat1=_csr(request.mat2.transpose(0, 1)),
        mat2=request.mat1.transpose(0, 1),
        self=accumulated,
        out=output,
    )


def _dense(request):
    kwargs = {} if request.out is None else {"out": request.out}
    if request.operation == "mm":
        return torch.mm(request.mat1, request.mat2, **kwargs)
    return torch.addmm(request.self, request.mat1, request.mat2,
                       beta=request.beta, alpha=request.alpha, **kwargs)


class MatmulRegistry:
    """Assemble capabilities once during package import, then route requests."""

    def __init__(self):
        self._capabilities = {}
        self.backward_helpers = {}

    @property
    def capabilities(self):
        """Read-only view of registered route/operation capabilities."""
        return MappingProxyType(self._capabilities)

    def register(self, capability):
        if not isinstance(capability, MatmulCapability):
            raise TypeError("matrix capabilities must be MatmulCapability instances")
        if capability.route not in ("spmm", "spgemm") or not capability.owner:
            raise ValueError("a matrix capability requires an owner and a spmm/spgemm route")
        if not callable(capability.forward) or (capability.backward is not None and
                                               not callable(capability.backward)):
            raise TypeError("matrix capability implementations must be callable")
        operations = capability.operations
        if not operations or len(set(operations)) != len(operations):
            raise ValueError("matrix capability operations must be non-empty and unique")
        keys = [(capability.route, operation) for operation in operations]
        for key in keys:
            if key[1] not in ("mm", "addmm"):
                raise ValueError("unsupported matrix operation: {}".format(key[1]))
            if key in self._capabilities:
                raise RuntimeError("duplicate matrix capability: {} {}".format(*key))
        helpers = dict(capability.backward_helpers)
        if len(helpers) != len(capability.backward_helpers):
            raise ValueError("backward helper schemas must be unique")
        for schema, implementation in helpers.items():
            if schema not in BACKWARD_HELPER_KEYS or not callable(implementation):
                raise ValueError("unsupported matrix backward helper: {}".format(schema))
            if schema in self.backward_helpers:
                raise RuntimeError("duplicate matrix backward helper: {}".format(schema))
        for key in keys:
            self._capabilities[key] = capability
        self.backward_helpers.update(helpers)

    def capability_for(self, request):
        """Validate and select a capability without executing or loading it."""
        capability, _, _ = self._resolve(request)
        return capability

    def dispatch(self, request):
        capability, route, transpose = self._resolve(request)
        if route == "dense":
            return _dense(request)
        normalized = _spmm_request(request, transpose) if route == "spmm" else request
        result = capability.forward(normalized)
        if request.out is not None and result is not normalized.out:
            raise RuntimeError("matrix out implementation must return the supplied out tensor")
        if transpose:
            result = result.transpose(0, 1)
            if request.out is None:
                result = result.contiguous()
        if request.out is not None:
            # The operator's out implementation owns resizing and copying.
            return request.out
        return result

    def dispatch_backward(self, request, grad_output):
        """Invoke the VJP supplied by the selected operator capability.

        The callback receives original operands and grad_output and returns
        (grad_self, grad_mat1, grad_mat2). It owns sparse gradient structure,
        scalar conjugation and broadcast reduction.
        """
        if request.out is not None:
            raise RuntimeError("out= matrix products do not support automatic differentiation")
        capability, _, _ = self._resolve(request)
        if capability is None or capability.backward is None:
            raise RuntimeError("no custom backward capability is registered for this matrix request")
        if (tuple(grad_output.shape) != (request.mat1.size(0), request.mat2.size(1)) or
                grad_output.device != request.mat1.device or grad_output.dtype != request.mat1.dtype):
            raise RuntimeError("grad_output must match the matrix product's shape, dtype and device")
        return capability.backward(request, grad_output)

    def _resolve(self, request):
        _validate(request)
        route, transpose = _route(request)
        if route == "dense":
            return None, route, transpose
        capability = self._capabilities.get((route, request.operation))
        if capability is None:
            raise RuntimeError("{} capability is not registered for {}".format(route, request.operation))
        if capability.dtypes and request.mat1.dtype not in capability.dtypes:
            raise RuntimeError("{} does not support dtype {}".format(capability.owner, request.mat1.dtype))
        if capability.device_types and request.mat1.device.type not in capability.device_types:
            raise RuntimeError("{} does not support device {}".format(capability.owner, request.mat1.device))
        return capability, route, transpose
