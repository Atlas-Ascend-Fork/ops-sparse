# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SpGEMM first-order VJP using sparse products and pattern projection."""

import torch
from torch._C import _jit_get_operation as get_aten_operation

from ...matmul.dispatch import MatmulRequest


# Resolve native schemas once; no extension compilation or device execution is
# involved. Raw storage operators retain uncoalesced duplicate coordinates.
_COO_INDICES, _ = get_aten_operation("aten::_indices")
_COO_VALUES, _ = get_aten_operation("aten::_values")
_SPARSE_MASK_PROJECTION, _ = get_aten_operation("aten::_sparse_mask_projection")


def _select(values, indices):
    # Metadata and real/imaginary planes require int64 and multidimensional
    # selection. Keep them independent of Gather's one-dimensional int32
    # aten::index_select registration when both providers share a wheel.
    if values.is_complex():
        return torch.view_as_complex(torch.view_as_real(values.resolve_conj())[indices].contiguous())
    return values[indices]


def _coo_components(tensor):
    # PyTorch's raw COO APIs preserve duplicate coordinates. The public
    # indices()/values() accessors reject an uncoalesced tensor.
    return _COO_INDICES(tensor), _COO_VALUES(tensor)


def _native_sparse_projection(gradient, source):
    # This ATen helper implements the native sparse gradient structure rules;
    # sparse_mask alone has different duplicate/intersection semantics.
    if _SPARSE_MASK_PROJECTION is None:
        raise RuntimeError(
            "SpGEMM COO backward requires aten::_sparse_mask_projection; "
            "this PyTorch build does not provide the required native operator"
        )
    return _SPARSE_MASK_PROJECTION(gradient, source, False)


def as_coo(tensor):
    if tensor.layout == torch.strided:
        mask = tensor.real != 0
        if tensor.is_complex():
            mask = mask | (tensor.imag != 0)
        indices = mask.nonzero().transpose(0, 1)
        offsets = indices[0] * tensor.size(1) + indices[1]
        values = _select(tensor.contiguous().reshape(-1), offsets)
        return torch.sparse_coo_tensor(indices, values, tensor.shape, device=tensor.device,
                                      is_coalesced=True)
    if tensor.layout != torch.sparse_coo:
        return tensor.to_sparse_coo()
    if tensor.is_coalesced():
        return tensor
    raw_indices, raw_values = _coo_components(tensor)
    if not tensor.is_complex():
        if tensor.dtype == torch.bfloat16:
            expanded = torch.sparse_coo_tensor(raw_indices, raw_values.float(),
                                               tensor.shape, device=tensor.device).coalesce()
            return torch.sparse_coo_tensor(expanded.indices(), expanded.values().to(tensor.dtype),
                                           tensor.shape, device=tensor.device, is_coalesced=True)
        return tensor.coalesce()
    # CANN CoalesceSparse does not accept complex values. Sum the two real
    # planes with identical coordinates, then reassemble their common structure.
    planes = torch.view_as_real(raw_values.resolve_conj())
    real = torch.sparse_coo_tensor(raw_indices, planes[:, 0], tensor.shape,
                                  device=tensor.device).coalesce()
    imag = torch.sparse_coo_tensor(raw_indices, planes[:, 1], tensor.shape,
                                  device=tensor.device).coalesce()
    values = torch.view_as_complex(torch.stack((real.values(), imag.values()), dim=-1))
    return torch.sparse_coo_tensor(real.indices(), values, tensor.shape,
                                  device=tensor.device, is_coalesced=True)


def _adjoint(tensor):
    tensor = as_coo(tensor)
    indices = tensor.indices().to(torch.int64).flip(0)
    order = torch.argsort(indices[0] * tensor.size(0) + indices[1])
    values = tensor.values()
    if values.is_complex():
        planes = torch.view_as_real(values.resolve_conj())
        values = torch.view_as_complex(torch.stack((planes[:, 0], -planes[:, 1]), dim=-1))
    return torch.sparse_coo_tensor(indices[:, order], _select(values, order),
                                  (tensor.size(1), tensor.size(0)), device=tensor.device,
                                  is_coalesced=True)


def _scalar_conjugate(value, dtype):
    value = value.conjugate() if hasattr(value, "conjugate") else value
    return value if dtype.is_complex or not isinstance(value, complex) else value.real


def _from_pattern(pattern, coo, values):
    if pattern.layout == torch.sparse_csr:
        return torch.sparse_csr_tensor(pattern.crow_indices(), pattern.col_indices(), values,
                                       size=pattern.shape, device=pattern.device)
    indices, _ = _coo_components(coo)
    return torch.sparse_coo_tensor(indices, values, pattern.shape,
                                   device=pattern.device, is_coalesced=coo.is_coalesced())


def _project(product, pattern, scale):
    target = as_coo(pattern)
    if product.layout == torch.sparse_csr:
        # CSR conversion marks COO coalesced even for unsorted or duplicate
        # columns. Rebuild uncoalesced storage to sort and sum lookup values.
        converted = product.to_sparse_coo()
        indices, values = _coo_components(converted)
        product = torch.sparse_coo_tensor(indices, values, product.shape, device=product.device)
    product = as_coo(product)
    indices = product.indices().to(torch.int64)
    target_indices, target_values = _coo_components(target)
    target_indices = target_indices.to(torch.int64)
    keys = indices[0] * product.size(1) + indices[1]
    target_keys = target_indices[0] * target.size(1) + target_indices[1]
    if keys.numel() == 0:
        values = torch.zeros_like(target_values)
    else:
        positions = torch.searchsorted(keys, target_keys)
        bounded = positions.clamp(max=keys.numel() - 1)
        matched = (positions < keys.numel()) & (keys[bounded] == target_keys)
        selected = _select(product.values(), bounded)
        if selected.is_complex():
            planes = torch.view_as_real(selected)
            planes = torch.where(matched.unsqueeze(1), planes, torch.zeros_like(planes))
            values = torch.view_as_complex(planes)
        else:
            values = torch.where(matched, selected, torch.zeros_like(selected))
    return _from_pattern(pattern, target, values * scale)


def _native_sparse_mask_like_grad(pattern, product):
    """Mirror PyTorch's sparse_mask_like_grad(..., accumulate_matches=False).

    ``_sparse_sparse_matmul`` uses this helper rather than forcing the product
    onto every coordinate of the input.  In particular, the native result may
    retain the product's structure when it is smaller than the input pattern.
    The required ATen projection kernels are not available on NPU, so run the
    structural operation on CPU and move its sparse result back without
    densifying either matrix.
    """
    source = pattern.cpu()
    gradient = product.cpu()
    source_indices, _ = _coo_components(source)
    gradient_indices, _ = _coo_components(gradient)
    source_has_more_entries = source_indices.size(1) >= gradient_indices.size(1)
    if source.is_coalesced() and gradient.is_coalesced():
        result = (_native_sparse_projection(gradient, source)
                  if source_has_more_entries
                  else gradient.sparse_mask(source))
    elif source.is_coalesced():
        result = gradient.sparse_mask(source)
    elif gradient.is_coalesced():
        result = _native_sparse_projection(gradient, source)
    elif source_has_more_entries:
        result = _native_sparse_projection(gradient.coalesce(), source)
    else:
        result = gradient.sparse_mask(source.coalesce())
    return result.to(pattern.device)


def _scales(request):
    """Return ATen-schema-specific scalar multipliers for first derivatives."""
    if request.schema == "aten::_sparse_addmm":
        # PyTorch 2.10's _sparse_addmm derivative differs from
        # addmm: self and sparse mat1 use the raw scalar, mat2 uses conj.
        return request.beta, request.alpha, _scalar_conjugate(request.alpha, request.mat1.dtype)
    conjugated_alpha = _scalar_conjugate(request.alpha, request.mat1.dtype)
    return (_scalar_conjugate(request.beta, request.mat1.dtype),
            conjugated_alpha, conjugated_alpha)


def _maybe_multiply(tensor, scale):
    # Match ATen maybe_multiply: real/integral one is an identity; complex
    # scalars still multiply, including complex one and zero with NaN data.
    if isinstance(scale, (int, float)) and scale == 1:
        return tensor
    if tensor.layout != torch.strided and tensor.device.type == "npu":
        # torch_npu lacks COO scalar multiplication. Preserve the full
        # sparse result and its layout through the native CPU operation.
        return (tensor.cpu() * scale).to(tensor.device)
    return tensor * scale


def _native_dense_matrix_gradient(request, grad_output, side, scale):
    # Dense upstream zeros must remain explicit: 0 * a stored NaN/Inf is
    # observable in native mm/addmm backward. Sparsifying G before multiplication
    # would silently remove those contributions.
    grad = grad_output.cpu()
    other = (request.mat2 if side == 0 else request.mat1).cpu()
    if grad.dtype in (torch.float16, torch.bfloat16):
        # CPU sparse mm lacks these dtypes in this environment; they are
        # additional NPU capabilities and accumulate in float32.
        grad, other = grad.float(), other.float()
    product = grad.mm(other.mH) if side == 0 else other.mH.mm(grad)
    operand = request.mat1 if side == 0 else request.mat2
    return _maybe_multiply(product, scale).to(device=operand.device, dtype=operand.dtype)


def _coordinate_keys(indices, shape):
    keys = torch.zeros(indices.size(1), dtype=torch.int64, device=indices.device)
    for dimension in range(indices.size(0)):
        keys = keys * shape[dimension] + indices[dimension].to(torch.int64)
    return keys


def sparse_mask(tensor, mask):
    if tensor.shape != mask.shape or tensor.device != mask.device:
        raise RuntimeError("sparse_mask requires matching shape and device")
    # Compressed sparse sources use intersection semantics, while COO masks
    # may preserve unmatched coordinates. Use the standard implementation for
    # sparse sources and other compressed layouts rather than approximating
    # their structural/layout rules with the matrix gradient projection.
    compressed_nonmatrix = mask.layout == torch.sparse_csr and mask.dim() != 2
    if (tensor.layout != torch.strided or
            mask.layout not in (torch.sparse_coo, torch.sparse_csr) or
            compressed_nonmatrix):
        return tensor.cpu().sparse_mask(mask.cpu()).to(tensor.device)
    target = mask if mask.layout == torch.sparse_coo else mask.to_sparse_coo()
    indices, _ = _coo_components(target)
    keys = _coordinate_keys(indices, target.shape)
    dense_shape = target.shape[target.sparse_dim():]
    prefix = 1
    for size in tensor.shape[:target.sparse_dim()]:
        prefix *= size
    values = _select(tensor.contiguous().reshape(prefix, *dense_shape), keys)
    if mask.layout == torch.sparse_csr:
        return torch.sparse_csr_tensor(mask.crow_indices(), mask.col_indices(), values,
                                       size=mask.shape, device=tensor.device)
    return torch.sparse_coo_tensor(indices, values, mask.shape,
                                   device=tensor.device, is_coalesced=mask.is_coalesced())


def _format_sparse_matrix_gradient(request, upstream_layout, product, operand, scale):
    """Apply the ATen schema's gradient structure and scalar rules."""
    if request.schema == "aten::_sparse_sparse_matmul" and operand.layout == torch.sparse_coo:
        return _maybe_multiply(_native_sparse_mask_like_grad(operand, product), scale)
    if request.schema in ("aten::mm", "aten::addmm"):
        # Native mm/addmm return the complete matrix derivative.
        # Pattern projection belongs to the sparse schemas.
        # CSR upstream with CSR operands uses native compressed mm
        # derivatives; the internal COO calculation must not leak
        # into the public gradient layout.
        if (upstream_layout == torch.sparse_csr and
                request.mat1.layout == request.mat2.layout == torch.sparse_csr):
            product = product.to_sparse_csr()
        return _maybe_multiply(product, scale)
    return _project(product, operand, scale)


def spgemm_backward(request, grad_output, execute):
    dense_matrix_gradient = (
        request.schema in ("aten::mm", "aten::addmm") and
        grad_output.layout == torch.strided
    )
    grad = None if dense_matrix_gradient else as_coo(grad_output)
    beta, mat1_alpha, mat2_alpha = _scales(request)
    grad_self = None
    if request.self is not None and request.self.requires_grad:
        grad_self = _maybe_multiply(grad_output, beta)
    gradients = []
    for side, operand in enumerate((request.mat1, request.mat2)):
        if not operand.requires_grad:
            gradients.append(None)
        elif dense_matrix_gradient:
            scale = mat1_alpha if side == 0 else mat2_alpha
            gradients.append(_native_dense_matrix_gradient(request, grad_output, side, scale))
        else:
            left, right = (grad, _adjoint(request.mat2)) if side == 0 else (_adjoint(request.mat1), grad)
            product = execute(MatmulRequest("mm", left, right, schema=request.schema))
            scale = mat1_alpha if side == 0 else mat2_alpha
            gradients.append(_format_sparse_matrix_gradient(
                request, grad_output.layout, product, operand, scale))
    return grad_self, gradients[0], gradients[1]


@torch.no_grad()
def sparse_add_out(self, other, *, alpha=1, out):
    """CSR gradient accumulation, including union patterns and complex values."""
    if self.device != other.device or out.device != self.device:
        raise RuntimeError("sparse add requires operands and out on the same device")
    left, right = self.detach().cpu(), other.detach().cpu()
    if left.layout == right.layout == torch.sparse_csr and left.shape == right.shape:
        # COO addition handles complex and union patterns without a CPU MKL
        # dependency; preserve CSR as the public result layout.
        result_dtype = torch.result_type(left, right)
        compute_dtype = torch.float32 if result_dtype in (torch.float16, torch.bfloat16) else result_dtype
        result = torch.add(left.to(compute_dtype).to_sparse_coo(),
                           right.to(compute_dtype).to_sparse_coo(), alpha=alpha).to_sparse_csr()
        result = result.to(result_dtype)
    else:
        result = torch.add(left, right, alpha=alpha)
    if result.layout != out.layout or not torch.can_cast(result.dtype, out.dtype):
        raise RuntimeError("sparse add out has an incompatible layout or dtype")
    result = result.to(device=out.device, dtype=out.dtype)
    if out.layout == torch.strided:
        out.resize_(result.shape)
    else:
        out.resize_as_sparse_(result)
    out.copy_(result)
    return out


def sparse_add_inplace(self, other, *, alpha=1):
    return sparse_add_out(self, other, alpha=alpha, out=self)
