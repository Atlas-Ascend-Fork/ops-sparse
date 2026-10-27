// SPDX-License-Identifier: LicenseRef-CANN-2.0 AND BSD-3-Clause
//
// LicenseRef-CANN-2.0 refers to the repository-root LICENSE and covers original code.
// BSD-3-Clause applies only to the PyTorch-derived input-validation portions
// identified below; their original copyright and full license remain in place.

/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*
 * Input-validation portions are adapted from PyTorch's
 * aten/src/ATen/native/sparse/SparseBlas.cpp (v2.7.1).
 * Those portions retain their original copyrights and BSD license terms;
 * the complete copyright notice and license are reproduced below.
 *
 * From PyTorch:
 *
 * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
 * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
 * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
 * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
 * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
 * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
 * Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
 * Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
 *
 * From Caffe2:
 *
 * Copyright (c) 2016-present, Facebook Inc. All rights reserved.
 *
 * All contributions by Facebook:
 * Copyright (c) 2016 Facebook Inc.
 *
 * All contributions by Google:
 * Copyright (c) 2015 Google Inc.
 * All rights reserved.
 *
 * All contributions by Yangqing Jia:
 * Copyright (c) 2015 Yangqing Jia
 * All rights reserved.
 *
 * All contributions by Kakao Brain:
 * Copyright 2019-2020 Kakao Brain
 *
 * All contributions by Cruise LLC:
 * Copyright (c) 2022 Cruise LLC.
 * All rights reserved.
 *
 * All contributions by Tri Dao:
 * Copyright (c) 2024 Tri Dao.
 * All rights reserved.
 *
 * All contributions by Arm:
 * Copyright (c) 2021, 2023-2024 Arm Limited and/or its affiliates
 *
 * All contributions from Caffe:
 * Copyright(c) 2013, 2014, 2015, the respective contributors
 * All rights reserved.
 *
 * All other contributions:
 * Copyright(c) 2015, 2016 the respective contributors
 * All rights reserved.
 *
 * Caffe2 uses a copyright model similar to Caffe: each contributor holds
 * copyright over their contributions to Caffe2. The project versioning records
 * all such contribution and copyright details. If a contributor wants to further
 * mark their specific copyright on a particular contribution, they should
 * indicate their copyright solely in the commit message of the change when it is
 * committed.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
 *    and IDIAP Research Institute nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>
#include <pybind11/complex.h>

#include "aclsparse_common.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <limits>

namespace ops_sparse::sddmm {
namespace {

using cann_ops_sparse::AclSparseContext;
using cann_ops_sparse::AclSparseWorkspace;

struct HostScalars {
    float alphaReal = 0.0F;
    float betaReal = 0.0F;
    aclsparseComplex alphaComplex{};
    aclsparseComplex betaComplex{};
    bool complex = false;

    const void *Alpha() const
    {
        return complex ? static_cast<const void *>(&alphaComplex) :
                         static_cast<const void *>(&alphaReal);
    }

    const void *Beta() const
    {
        return complex ? static_cast<const void *>(&betaComplex) :
                         static_cast<const void *>(&betaReal);
    }
};

constexpr int64_t kMaxBatchCount = 65535;

aclDataType ToAclDataType(at::ScalarType type)
{
    switch (type) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kComplexFloat:
            return ACL_COMPLEX64;
        default:
            TORCH_CHECK(false, "sampled_addmm: unsupported NPU dtype ", type,
                "; expected float16, float32, or complex64");
    }
}

aclDataType ToComputeType(at::ScalarType type)
{
    return type == at::kComplexFloat ? ACL_COMPLEX64 : ACL_FLOAT;
}

HostScalars ConvertScalars(
    const at::Scalar &alpha, const at::Scalar &beta, at::ScalarType type)
{
    HostScalars result;
    if (type == at::kComplexFloat) {
        result.complex = true;
        const c10::complex<float> alphaValue = alpha.toComplexFloat();
        const c10::complex<float> betaValue = beta.toComplexFloat();
        result.alphaComplex = {alphaValue.real(), alphaValue.imag()};
        result.betaComplex = {betaValue.real(), betaValue.imag()};
    } else {
        result.alphaReal = alpha.toFloat();
        result.betaReal = beta.toFloat();
    }
    return result;
}

int64_t CheckedBatchCount(at::IntArrayRef batchSizes)
{
    int64_t count = 1;
    for (const int64_t size : batchSizes) {
        TORCH_CHECK(size >= 0, "sampled_addmm: negative batch dimension ", size);
        if (size == 0) {
            return 0;
        }
        TORCH_CHECK(count <= std::numeric_limits<int64_t>::max() / size,
            "sampled_addmm: batch dimensions overflow int64");
        count *= size;
    }
    TORCH_CHECK(count <= kMaxBatchCount,
        "sampled_addmm: flattened batch count exceeds ", kMaxBatchCount);
    return count;
}

class DenseDescriptor {
public:
    DenseDescriptor(int64_t rows, int64_t cols, const void *values, aclDataType type)
    {
        const int64_t ld = std::max<int64_t>(cols, 1);
        ACLSPARSE_CHECK(aclsparseCreateConstDnMat(
            &descriptor_, rows, cols, ld, values, type, ACL_SPARSE_ORDER_ROW));
    }

    ~DenseDescriptor()
    {
        if (descriptor_ != nullptr) {
            (void)aclsparseDestroyDnMat(descriptor_);
        }
    }

    DenseDescriptor(const DenseDescriptor &) = delete;
    DenseDescriptor &operator=(const DenseDescriptor &) = delete;

    aclsparseConstDnMatDescr_t Get() const
    {
        return descriptor_;
    }

private:
    aclsparseConstDnMatDescr_t descriptor_ = nullptr;
};

class SparseDescriptor {
public:
    SparseDescriptor(int64_t rows, int64_t cols, int64_t nnz, int32_t *crow,
        int32_t *col, void *values, aclDataType type)
    {
        ACLSPARSE_CHECK(aclsparseCreateCsr(&descriptor_, rows, cols, nnz, crow, col, values,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, type));
    }

    ~SparseDescriptor()
    {
        if (descriptor_ != nullptr) {
            (void)aclsparseDestroySpMat(descriptor_);
        }
    }

    SparseDescriptor(const SparseDescriptor &) = delete;
    SparseDescriptor &operator=(const SparseDescriptor &) = delete;

    aclsparseSpMatDescr_t Get() const
    {
        return descriptor_;
    }

private:
    aclsparseSpMatDescr_t descriptor_ = nullptr;
};

at::Tensor LosslessI32(const at::Tensor &indices, const char *name)
{
    if (indices.scalar_type() == at::kInt) {
        return indices.contiguous();
    }
    TORCH_INTERNAL_ASSERT(indices.scalar_type() == at::kLong);
    at::Tensor converted = indices.to(at::kInt).contiguous();
    TORCH_CHECK(converted.to(at::kLong).equal(indices),
        "sampled_addmm: ", name, " contains a value outside the int32 range");
    return converted;
}

struct CallShape {
    at::DimVector batchPrefix;
    at::DimVector resultSizes;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t reduction = 0;
    int64_t nnz = 0;
    int64_t batchCount = 0;
    int64_t structureBatchCount = 0;
};

void CheckInputLayouts(const at::Tensor &self, const at::Tensor &mat1,
    const at::Tensor &mat2, const at::Tensor &result)
{
    TORCH_CHECK(self.layout() == at::kSparseCsr,
        "sampled_addmm: Expected self to have sparse csr layout, but got ", self.layout());
    TORCH_CHECK(mat1.layout() == at::kStrided,
        "sampled_addmm: Expected mat1 to have strided layout, but got ", mat1.layout());
    TORCH_CHECK(mat2.layout() == at::kStrided,
        "sampled_addmm: Expected mat2 to have strided layout, but got ", mat2.layout());
    TORCH_CHECK(result.layout() == at::kSparseCsr,
        "sampled_addmm: Expected result to have sparse csr layout, but got ", result.layout());
    TORCH_CHECK(self.dense_dim() == 0,
        "sampled_addmm: Expected non-hybrid self tensor");
    TORCH_CHECK(result.dense_dim() == 0,
        "sampled_addmm: Expected non-hybrid result tensor");
}

void CheckInputDevices(const at::Tensor &self, const at::Tensor &mat1,
    const at::Tensor &mat2, const at::Tensor &result)
{
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
        "sampled_addmm: self must be on an NPU device");
    TORCH_CHECK(mat1.device() == self.device() && mat2.device() == self.device() &&
        result.device() == self.device(),
        "sampled_addmm: self, mat1, mat2, and result must be on the same NPU device");
    TORCH_CHECK(self.crow_indices().device() == self.device() &&
        self.col_indices().device() == self.device() && self.values().device() == self.device(),
        "sampled_addmm: all CSR member tensors must be on the same NPU device as self");
}

void CheckInputTypes(const at::Tensor &self, const at::Tensor &mat1,
    const at::Tensor &mat2, const at::Tensor &result)
{
    TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(),
        "sampled_addmm: Expected mat1 and mat2 to have the same dtype, but got ",
        mat1.scalar_type(), " and ", mat2.scalar_type());
    TORCH_CHECK(mat1.scalar_type() == self.scalar_type(),
        "sampled_addmm: Expected mat1 and self to have the same dtype, but got ",
        mat1.scalar_type(), " and ", self.scalar_type());
    TORCH_CHECK(result.scalar_type() == self.scalar_type(),
        "sampled_addmm: Expected result and self to have the same dtype, but got ",
        result.scalar_type(), " and ", self.scalar_type());
    (void)ToAclDataType(self.scalar_type());
}

void CheckInputRanks(const at::Tensor &self, const at::Tensor &mat1,
    const at::Tensor &mat2, const at::Tensor &result)
{
    TORCH_CHECK(mat1.dim() >= 2,
        "sampled_addmm: Expected mat1 to be a matrix, got ", mat1.dim(), "-D tensor");
    TORCH_CHECK(mat2.dim() >= 2,
        "sampled_addmm: Expected mat2 to be a matrix, got ", mat2.dim(), "-D tensor");
    TORCH_CHECK(self.dim() >= 2,
        "sampled_addmm: Expected self to be a matrix, got ", self.dim(), "-D tensor");
    TORCH_CHECK(result.dim() >= 2,
        "sampled_addmm: Expected result to be a matrix, got ", result.dim(), "-D tensor");
}

CallShape CheckInputs(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const at::Tensor &result)
{
    CheckInputLayouts(self, mat1, mat2, result);
    CheckInputDevices(self, mat1, mat2, result);
    CheckInputTypes(self, mat1, mat2, result);
    CheckInputRanks(self, mat1, mat2, result);

    const at::IntArrayRef batch1 = mat1.sizes().slice(0, mat1.dim() - 2);
    const at::IntArrayRef batch2 = mat2.sizes().slice(0, mat2.dim() - 2);
    TORCH_CHECK(batch1 == batch2,
        "sampled_addmm: Expected mat1 and mat2 to have the same batch size, but got ",
        batch1, " and ", batch2);
    if (self.dim() > 2) {
        TORCH_CHECK(self.sizes().slice(0, self.dim() - 2) == batch1,
            "sampled_addmm: Expected self and mat1 to have the same batch size, but got ",
            self.sizes().slice(0, self.dim() - 2), " and ", batch1);
    }

    CallShape shape;
    shape.rows = mat1.size(-2);
    shape.reduction = mat1.size(-1);
    shape.cols = mat2.size(-1);
    TORCH_CHECK(shape.reduction == mat2.size(-2),
        "sampled_addmm: mat1 and mat2 shapes cannot be multiplied (",
        shape.rows, "x", shape.reduction, " and ", mat2.size(-2), "x", shape.cols, ")");
    TORCH_CHECK(self.size(-2) == shape.rows,
        "sampled_addmm: self.shape[-2] must match mat1.shape[-2]");
    TORCH_CHECK(self.size(-1) == shape.cols,
        "sampled_addmm: self.shape[-1] must match mat2.shape[-1]");
    TORCH_CHECK(shape.rows <= std::numeric_limits<int32_t>::max() &&
        shape.cols <= std::numeric_limits<int32_t>::max() &&
        shape.reduction <= std::numeric_limits<int32_t>::max(),
        "sampled_addmm: M, N, and K must fit int32 on the NPU path");

    TORCH_CHECK(self.crow_indices().scalar_type() == self.col_indices().scalar_type(),
        "sampled_addmm: crow_indices and col_indices must have the same dtype");
    TORCH_CHECK(self.crow_indices().scalar_type() == at::kInt ||
        self.crow_indices().scalar_type() == at::kLong,
        "sampled_addmm: CSR indices must be int32 or int64");
    shape.nnz = self.values().size(-1);
    TORCH_CHECK(shape.nnz <= std::numeric_limits<int32_t>::max(),
        "sampled_addmm: nnz must fit int32 on the NPU path");

    shape.batchPrefix.assign(batch1.begin(), batch1.end());
    shape.resultSizes = shape.batchPrefix;
    shape.resultSizes.push_back(shape.rows);
    shape.resultSizes.push_back(shape.cols);
    shape.batchCount = CheckedBatchCount(batch1);
    shape.structureBatchCount = self.dim() == 2 ? 1 : shape.batchCount;
    return shape;
}

at::Tensor CloneForOutput(
    const at::Tensor &source, const at::DimVector &batchPrefix, int64_t trailingSize)
{
    at::DimVector targetSizes = batchPrefix;
    targetSizes.push_back(trailingSize);
    if (source.sizes() == at::IntArrayRef(targetSizes)) {
        return source.clone(at::MemoryFormat::Contiguous);
    }
    return source.expand(targetSizes).clone(at::MemoryFormat::Contiguous);
}

void PrepareDistinctOutput(
    const at::Tensor &self, at::Tensor &result, const CallShape &shape)
{
    at::Tensor crow = CloneForOutput(self.crow_indices(), shape.batchPrefix, shape.rows + 1);
    at::Tensor col = CloneForOutput(self.col_indices(), shape.batchPrefix, shape.nnz);
    at::Tensor values = CloneForOutput(self.values(), shape.batchPrefix, shape.nnz);
    at::sparse_csr::get_sparse_csr_impl(result)->set_member_tensors(
        crow, col, values, shape.resultSizes);
}

void PrepareOutput(const at::Tensor &self, at::Tensor &result, const CallShape &shape)
{
    const bool aliasesInput = result.is_same(self);
    if (aliasesInput) {
        // PyTorch 2.7 does not resize the input object for out=self. In
        // particular, a 2-D pattern cannot be expanded in-place to a batched
        // CSR result because doing so would replace its pattern tensors.
        TORCH_CHECK(result.sizes() == at::IntArrayRef(shape.resultSizes),
            "sampled_addmm: out=self cannot broadcast a 2-D CSR pattern to a batched result");
    } else {
        PrepareDistinctOutput(self, result, shape);
    }
}

void ExecuteBatches(AclSparseContext &context, const at::Tensor &crowI32,
    const at::Tensor &colI32, const CallShape &shape, const at::Tensor &denseXFlat,
    const at::Tensor &denseYFlat, const at::Tensor &valuesFlat,
    const HostScalars &scalars, aclDataType valueType)
{
    const aclsparseHandle_t handle = context.handle();
    const aclDataType computeType = ToComputeType(valuesFlat.scalar_type());
    for (int64_t batch = 0; batch < shape.batchCount; ++batch) {
        at::Tensor xBatch = denseXFlat.select(0, batch);
        at::Tensor yBatch = denseYFlat.select(0, batch);
        at::Tensor valuesBatch = valuesFlat.select(0, batch);
        void *valuesPointer = shape.nnz == 0 ? nullptr : valuesBatch.data_ptr();
        DenseDescriptor matX(shape.rows, shape.reduction, xBatch.data_ptr(), valueType);
        DenseDescriptor matY(shape.reduction, shape.cols, yBatch.data_ptr(), valueType);
        const int64_t structureBatch = shape.structureBatchCount == 1 ? 0 : batch;
        int32_t *crow = crowI32.data_ptr<int32_t>() + structureBatch * (shape.rows + 1);
        int32_t *col = shape.nnz == 0 ? nullptr :
            colI32.data_ptr<int32_t>() + structureBatch * shape.nnz;
        SparseDescriptor matC(shape.rows, shape.cols, shape.nnz, crow, col, valuesPointer, valueType);

        size_t workspaceBytes = 0;
        ACLSPARSE_CHECK(aclsparseSDDMMBufferSize(
            handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            scalars.Alpha(), matX.Get(), matY.Get(), scalars.Beta(), matC.Get(),
            computeType, ACL_SPARSE_SDDMM_ALG_DEFAULT, &workspaceBytes));
        TORCH_CHECK(workspaceBytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
            "sampled_addmm: workspace size exceeds int64");
        AclSparseWorkspace workspace(context, valuesBatch, workspaceBytes);
        ACLSPARSE_CHECK(aclsparseSDDMMPreprocess(
            handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            scalars.Alpha(), matX.Get(), matY.Get(), scalars.Beta(), matC.Get(),
            computeType, ACL_SPARSE_SDDMM_ALG_DEFAULT, workspace.data()));
        ACLSPARSE_CHECK(aclsparseSDDMM(
            handle, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            scalars.Alpha(), matX.Get(), matY.Get(), scalars.Beta(), matC.Get(),
            computeType, ACL_SPARSE_SDDMM_ALG_DEFAULT, workspace.data()));
    }
}

at::Tensor &SampledAddmmOutNpu(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const at::Scalar &beta, const at::Scalar &alpha, at::Tensor &result)
{
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
        "sampled_addmm: self must be on an NPU device");
    c10::OptionalDeviceGuard deviceGuard(self.device());
    const CallShape shape = CheckInputs(self, mat1, mat2, result);
    const at::Tensor sourceCrow = self.crow_indices();
    const at::Tensor sourceCol = self.col_indices();
    PrepareOutput(self, result, shape);
    if (shape.batchCount == 0) {
        return result;
    }

    at::Tensor denseX = mat1.resolve_conj().resolve_neg().contiguous();
    at::Tensor denseY = mat2.resolve_conj().resolve_neg().contiguous();
    at::Tensor values = result.values();
    at::Tensor valuesWork = values.is_contiguous() ? values : values.contiguous();
    if (at::get_overlap_status(valuesWork, denseX) != at::MemOverlapStatus::No ||
        at::get_overlap_status(valuesWork, denseY) != at::MemOverlapStatus::No) {
        // Preserve aliased dense inputs until every block and batch has read them.
        valuesWork = valuesWork.clone(at::MemoryFormat::Contiguous);
    }
    at::Tensor denseXFlat = denseX.reshape({shape.batchCount, shape.rows, shape.reduction});
    at::Tensor denseYFlat = denseY.reshape({shape.batchCount, shape.reduction, shape.cols});
    at::Tensor valuesFlat = valuesWork.reshape({shape.batchCount, shape.nnz});
    at::Tensor crowI32 = LosslessI32(sourceCrow, "crow_indices");
    at::Tensor colI32 = LosslessI32(sourceCol, "col_indices");
    TORCH_CHECK(crowI32.numel() == shape.structureBatchCount * (shape.rows + 1),
        "sampled_addmm: crow_indices has an invalid batched shape");
    TORCH_CHECK(colI32.numel() == shape.structureBatchCount * shape.nnz,
        "sampled_addmm: col_indices has an invalid batched shape");
    const HostScalars scalars = ConvertScalars(alpha, beta, self.scalar_type());
    const aclDataType valueType = ToAclDataType(self.scalar_type());

    // Output copies, dense materialization and index conversion must be queued
    // before Initialize resolves the ACL stream and flushes torch_npu's queue.
    // The common context keeps one locked handle for this device and stream.
    AclSparseContext context(self);
    context.Initialize();
    context.RecordTensors(sourceCrow, sourceCol, crowI32, colI32, denseX, denseY, valuesWork);
    ExecuteBatches(context, crowI32, colI32, shape, denseXFlat, denseYFlat,
        valuesFlat, scalars, valueType);

    if (!valuesWork.is_same(values)) {
        values.copy_(valuesWork);
    }
    return result;
}

at::Tensor SampledAddmmNpu(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const at::Scalar &beta, const at::Scalar &alpha)
{
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
        "sampled_addmm: self must be on an NPU device");
    c10::OptionalDeviceGuard deviceGuard(self.device());
    at::Tensor result = at::empty({0, 0}, self.options());
    SampledAddmmOutNpu(self, mat1, mat2, beta, alpha, result);
    return result;
}

// pybind11 does not provide an at::Scalar caster. Preserve Python scalar
// kinds so the existing real/complex conversion and validation stay unchanged.
at::Scalar PythonScalar(const pybind11::handle &value)
{
    if (PyBool_Check(value.ptr())) {
        return at::Scalar(value.cast<bool>());
    }
    if (PyLong_Check(value.ptr())) {
        return at::Scalar(value.cast<int64_t>());
    }
    if (PyFloat_Check(value.ptr())) {
        return at::Scalar(value.cast<double>());
    }
    if (PyComplex_Check(value.ptr())) {
        const auto scalar = value.cast<std::complex<double>>();
        return at::Scalar(c10::complex<double>(scalar.real(), scalar.imag()));
    }
    TORCH_CHECK(false, "sampled_addmm: alpha and beta must be Python scalars");
}

}  // namespace
}  // namespace ops_sparse::sddmm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def("sampled_addmm",
        [](const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
            const pybind11::object &beta, const pybind11::object &alpha) {
            return ops_sparse::sddmm::SampledAddmmNpu(self, mat1, mat2,
                ops_sparse::sddmm::PythonScalar(beta), ops_sparse::sddmm::PythonScalar(alpha));
        }, "SDDMM through the standard sampled_addmm NPU contract");
    module.def("sampled_addmm_out",
        [](const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
            const pybind11::object &beta, const pybind11::object &alpha, at::Tensor result) {
            ops_sparse::sddmm::SampledAddmmOutNpu(self, mat1, mat2,
                ops_sparse::sddmm::PythonScalar(beta), ops_sparse::sddmm::PythonScalar(alpha), result);
            return result;
        }, "SDDMM with an explicit CSR output tensor");
}
