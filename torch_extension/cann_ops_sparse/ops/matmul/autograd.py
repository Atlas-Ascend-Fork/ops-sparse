# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Backend-specific autograd for explicitly registered matrix capabilities."""

from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable

from .dispatch import MatmulRegistry, MatmulRequest


@dataclass(frozen=True)
class _AutogradOptions:
    """Non-tensor inputs kept separate from differentiable Function arguments."""

    operation: str
    schema: str
    beta: object
    alpha: object
    registry: MatmulRegistry


class _MatmulAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat1, mat2, self, options):
        ctx.save_for_backward(mat1, mat2, self)
        ctx.options = options
        return options.registry.dispatch(MatmulRequest(
            options.operation, mat1, mat2, self, options.beta, options.alpha, schema=options.schema,
        ))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        mat1, mat2, self = ctx.saved_tensors
        options = ctx.options
        request = MatmulRequest(options.operation, mat1, mat2, self, options.beta, options.alpha,
                                schema=options.schema)
        grad_self, grad_mat1, grad_mat2 = options.registry.dispatch_backward(request, grad_output)
        return grad_mat1, grad_mat2, grad_self, None


def register_autograd(library, registry):
    """Specialize sparse requests while preserving shared ATen dense autograd.

    ATen's generated Autograd alias kernel is also registered at AutogradOther.
    Redispatching to that key preserves the tensor backend key and reaches the
    existing formula without recursively entering our PrivateUse1 specialization.
    """
    def implementation(schema, operation):
        op = getattr(torch.ops.aten, schema).default

        def autograd(keyset, *args, **kwargs):
            if operation == "mm":
                mat1, mat2 = args
                self, beta, alpha = None, 0, 1
            else:
                self, mat1, mat2 = args
                beta, alpha = kwargs.get("beta", 1), kwargs.get("alpha", 1)
            request = MatmulRequest(operation, mat1, mat2, self, beta, alpha,
                                    schema="aten::" + schema)
            tensors = (mat1, mat2, self)
            sparse = any(t is not None and t.layout != torch.strided for t in tensors)
            if sparse and torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors):
                capability = registry.capability_for(request)
                if capability is not None and capability.backward is not None:
                    options = _AutogradOptions(operation, request.schema, beta, alpha, registry)
                    return _MatmulAutograd.apply(mat1, mat2, self, options)
            fallback = keyset.remove(torch.DispatchKey.AutogradPrivateUse1)
            fallback = fallback.add(torch.DispatchKey.AutogradOther)
            return op.redispatch(fallback, *args, **kwargs)
        return autograd

    for schema, operation in (("mm", "mm"), ("addmm", "addmm"),
                              ("_sparse_addmm", "addmm"), ("_sparse_sparse_matmul", "mm")):
        library.impl(schema, implementation(schema, operation), "AutogradPrivateUse1", with_keyset=True)
