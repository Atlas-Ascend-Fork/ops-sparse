// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms of conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// A copy of the License is located at
// http://www.huawei.com
// This program is distributed in the hope that it will be useful, but WITHOUT any warranty of any kind.
// ----------------------------------------------------------------------------------------------------------

// Tensor.to_sparse / to_sparse_csr / to_sparse_csc / to_sparse_bsr 的 C++ 封装
// （aten::_to_sparse{,.sparse_dim,_csr,_csc,_bsr}，NPU 后端）。
//
// 每个 wrapper 校验稠密算子（2-D、受支持 dtype、稠密布局），驱动 aclsparse
// 三阶段流程（或 BELL 直转），并以 recordStream 托管输出与 workspace 的
// 生命周期。稀疏输出张量由 Python 注册层组装（见 dense_to_sparse.py 的
// 模块说明）。不支持的参数组合抛出，不回退 CPU。
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <torch/extension.h>
#include <ATen/ATen.h>

#include "aclsparse_common.h"
#include "cann_ops_sparse.h"

namespace {

// Destroys both descriptors on scope exit; without it, an ACLSPARSE_CHECK
// throw between Create and the tail destroys would leak them.
class DescriptorGuard {
public:
    DescriptorGuard() = default;
    DescriptorGuard(const DescriptorGuard &) = delete;
    DescriptorGuard &operator=(const DescriptorGuard &) = delete;
    ~DescriptorGuard()
    {
        if (sp_ != nullptr) {
            (void)aclsparseDestroySpMat(sp_);
        }
        if (dd_ != nullptr) {
            (void)aclsparseDestroyDnMat(dd_);
        }
    }
    void AdoptDn(aclsparseDnMatDescr_t dd) { dd_ = dd; }
    void AdoptSp(aclsparseSpMatDescr_t sp) { sp_ = sp; }

private:
    aclsparseDnMatDescr_t dd_ = nullptr;
    aclsparseSpMatDescr_t sp_ = nullptr;
};


using cann_ops_sparse::AclSparseContext;
using cann_ops_sparse::AclSparseWorkspace;
using cann_ops_sparse::RunAclSparse;

constexpr const char *kApi = "aclsparseDenseToSparse(ATen)";
constexpr aclsparseDenseToSparseAlg_t kAlg = ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT;

aclDataType ToAclType(const torch::Tensor &t)
{
    switch (t.scalar_type()) {
    case torch::kInt8:
        return ACL_INT8;
    case torch::kFloat16:
        return ACL_FLOAT16;
    case torch::kBFloat16:
        return ACL_BF16;
    case torch::kFloat32:
        return ACL_FLOAT;
    case torch::kComplexFloat:
        return ACL_COMPLEX64;
    default:
        return ACL_DT_UNDEFINED;
    }
}

void ValidateDense(const torch::Tensor &self)
{
    if (self.device().type() != c10::DeviceType::PrivateUse1) {
        throw std::runtime_error(
            std::string(kApi) + ": requires an NPU tensor");
    }
    if (self.dim() != 2) {
        throw std::runtime_error(
            std::string(kApi) + ": expected a 2-D dense tensor, got dim " +
            std::to_string(self.dim()));
    }
    if (ToAclType(self) == ACL_DT_UNDEFINED) {
        throw std::runtime_error(
            std::string(kApi) +
            ": unsupported dtype (int8/fp16/bf16/fp32/complex64)");
    }
    if (self.size(0) > 0 && self.size(1) > 0 && !self.data_ptr()) {
        throw std::runtime_error(std::string(kApi) + ": dense is empty");
    }
}

// Dense operand for the descriptor: contiguous row-major storage, or a
// column-major 2-D storage consumed in place; anything else is copied
// (on device) to contiguous.
struct DenseOperand {
    torch::Tensor storage; // keep alive
    void *ptr = nullptr;
    int64_t rows = 0, cols = 0, ld = 0;
    aclsparseOrder_t order = ACL_SPARSE_ORDER_ROW;
};

DenseOperand PrepareDense(const torch::Tensor &self)
{
    DenseOperand d;
    d.rows = self.size(0);
    d.cols = self.size(1);
    d.storage = self;
    const auto strides = self.strides();
    const bool rowMajor = self.is_contiguous();
    // Column-major 2-D views produced by .t() are dense storage with
    // stride (1, rows): feed them to the COL order directly.
    if (!rowMajor && strides[0] == 1 && strides[1] >= self.size(0) &&
        self.storage_offset() == 0) {
        d.order = ACL_SPARSE_ORDER_COL;
        d.ld = strides[1];
        d.ptr = self.data_ptr();
        return d;
    }
    if (!rowMajor) {
        d.storage = self.contiguous();
    }
    d.order = ACL_SPARSE_ORDER_ROW;
    d.ld = d.cols;
    d.ptr = d.storage.data_ptr();
    return d;
}

// Sparse descriptor for the three compressed formats; allocates the
// placeholder (or exact) payload tensors and returns them by reference.
struct TriOut {
    torch::Tensor values;
    torch::Tensor major; // CSR/CSC: offsets (major_dim+1); COO: row ids
    torch::Tensor minor; // CSR/CSC: minor indices; COO: col ids
};

aclsparseSpMatDescr_t CreateCompressedSpMat(aclsparseFormat_t fmt,
    int64_t rows, int64_t cols, int64_t cap, int64_t majorDim,
    aclDataType aclType, torch::TensorOptions i32, torch::TensorOptions u8,
    torch::Tensor &offs, torch::Tensor &rowsT, torch::Tensor &colsT,
    torch::Tensor &valsT)
{
    valsT = torch::empty({1}, u8);
    aclsparseSpMatDescr_t sp = nullptr;
    if (fmt == ACL_SPARSE_FORMAT_COO) {
        rowsT = torch::empty({1}, i32);
        colsT = torch::empty({1}, i32);
        ACLSPARSE_CHECK(aclsparseCreateCoo(
            &sp, rows, cols, cap, rowsT.data_ptr(), colsT.data_ptr(),
            valsT.data_ptr(), ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_BASE_ZERO, aclType));
        return sp;
    }
    offs = torch::empty({majorDim + 1}, i32);
    colsT = torch::empty({1}, i32);
    const auto create = (fmt == ACL_SPARSE_FORMAT_CSR)
                            ? aclsparseCreateCsr
                            : aclsparseCreateCsc;
    ACLSPARSE_CHECK(create(
        &sp, rows, cols, cap, offs.data_ptr(), colsT.data_ptr(),
        valsT.data_ptr(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, aclType));
    return sp;
}

// Output tensor assembly: exact payload views on the non-empty path,
// typed empty tensors on the empty path.
TriOut AssembleTriOut(aclsparseFormat_t fmt, const torch::Tensor &like,
    int64_t majorDim, int64_t nnz, int64_t E, torch::Tensor &valsT,
    torch::Tensor &rowsT, torch::Tensor &colsT, torch::Tensor &offs)
{
    TriOut out;
    if (nnz > 0) {
        out.values = valsT.view(like.scalar_type());
        out.major =
            ((fmt == ACL_SPARSE_FORMAT_COO) ? rowsT : offs).to(torch::kInt64);
        out.minor = colsT.to(torch::kInt64);
        return out;
    }
    // Empty result: typed empty tensors (the placeholder buffers are
    // 1-element uint8/int32 views that cannot be reinterpreted).
    const auto dev = like.device();
    out.values = torch::empty({0}, like.options());
    out.major = torch::empty(
        {fmt == ACL_SPARSE_FORMAT_COO ? (int64_t)0 : majorDim + 1},
        torch::TensorOptions().dtype(torch::kInt64).device(dev));
    if (fmt != ACL_SPARSE_FORMAT_COO) {
        out.major.zero_();
    }
    out.minor = torch::empty(
        {0}, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    return out;
}

// Rebind the sparse descriptor to the (exact-capacity) payload tensors.
void BindCompressedPointers(aclsparseFormat_t fmt, aclsparseSpMatDescr_t sp,
    const torch::Tensor &offs, const torch::Tensor &rowsT,
    const torch::Tensor &colsT, const torch::Tensor &valsT)
{
    if (fmt == ACL_SPARSE_FORMAT_COO) {
        ACLSPARSE_CHECK(aclsparseCooSetPointers(
            sp, rowsT.data_ptr(), colsT.data_ptr(), valsT.data_ptr()));
    } else if (fmt == ACL_SPARSE_FORMAT_CSR) {
        ACLSPARSE_CHECK(aclsparseCsrSetPointers(
            sp, offs.data_ptr(), colsT.data_ptr(), valsT.data_ptr()));
    } else {
        ACLSPARSE_CHECK(aclsparseCscSetPointers(
            sp, offs.data_ptr(), colsT.data_ptr(), valsT.data_ptr()));
    }
}

// Three-stage CSR/CSC/COO conversion. Returns
// (values, offsets-or-rows, indices-or-cols) on device.
TriOut RunCompressed(aclsparseFormat_t fmt, const DenseOperand &d,
                     const torch::Tensor &like)
{
    const auto dev = like.device();
    auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    const int64_t rows = d.rows, cols = d.cols;
    const int64_t E = like.element_size();
    const int64_t majorDim = (fmt == ACL_SPARSE_FORMAT_CSC) ? cols : rows;

    return RunAclSparse(like, [&](AclSparseContext &ctx) {
        if (rows == 0 || cols == 0) {
            // Empty Torch storage has a null data pointer, which the DnMat
            // constructor rejects. The sparse shape/empty payload are known
            // directly; assemble them under the normal device guard.
            torch::Tensor offs, rowsT, colsT, valsT;
            return AssembleTriOut(fmt, like, majorDim, 0, E, valsT, rowsT, colsT, offs);
        }
        const aclsparseHandle_t handle = ctx.handle();
        aclsparseDnMatDescr_t dd = nullptr;
        ACLSPARSE_CHECK(aclsparseCreateDnMat(
            &dd, rows, cols, d.ld, d.ptr, ToAclType(like), d.order));
        DescriptorGuard guard;
        guard.AdoptDn(dd);
        torch::Tensor offs, rowsT, colsT, valsT;
        const int64_t cap = rows * cols;
        aclsparseSpMatDescr_t sp = CreateCompressedSpMat(
            fmt, rows, cols, cap, majorDim, ToAclType(like), i32, u8, offs,
            rowsT, colsT, valsT);
        // Adopt immediately: any later ACLSPARSE_CHECK throw or tensor
        // allocation failure unwinds through the guard, which must free
        // BOTH descriptors (deferring adoption would leak sp).
        guard.AdoptSp(sp);
        size_t bs = 0;
        ACLSPARSE_CHECK(aclsparseDenseToSparseGetBufferSize(
            handle, dd, sp, kAlg, &bs));
        AclSparseWorkspace ws(ctx, like, bs);
        ACLSPARSE_CHECK(aclsparseDenseToSparseAnalysis(handle, dd, sp, kAlg,
                                                       ws.data()));
        int64_t r = 0, c = 0, nnz = 0;
        ACLSPARSE_CHECK(aclsparseSpMatGetSize(sp, &r, &c, &nnz));
        if (nnz > 0) {
            valsT = torch::empty({nnz * E}, u8);
            colsT = torch::empty({nnz}, i32);
        }
        if (fmt == ACL_SPARSE_FORMAT_COO && nnz > 0) {
            rowsT = torch::empty({nnz}, i32);
        }
        BindCompressedPointers(fmt, sp, offs, rowsT, colsT, valsT);
        ACLSPARSE_CHECK(aclsparseDenseToSparseConvert(handle, dd, sp, kAlg,
                                                      ws.data()));
        // Outputs are stream-tracked: the caching allocator will not hand
        // their storage to a later allocation on another stream before the
        // conversion kernel retires.
        ctx.RecordTensors(valsT, colsT, rowsT, offs);

        return AssembleTriOut(fmt, like, majorDim, nnz, E, valsT,
                              rowsT, colsT, offs);
    });
}

// Blocked-ELL occupancy + pattern construction on device. Returns
// (pattern[br, width], width); width == 0 means the empty case.
std::pair<torch::Tensor, int64_t> BuildBellPattern(const DenseOperand &d,
    const torch::Tensor &like, int64_t b, int64_t br, int64_t bc)
{
    const auto dev = like.device();
    // The Python layer validates block >= 1; the clamp guards a stray
    // zero divisor (degrades to unit blocks instead of faulting).
    const int64_t bSafe = b > 0 ? b : 1;
    // Occupancy on a block-aligned scratch copy (conversion itself runs on
    // the original matrix; the kernel handles tail blocks natively).
    torch::Tensor occSrc = like;
    if (d.rows % bSafe != 0 || d.cols % bSafe != 0) {
        occSrc = torch::zeros({br * bSafe, bc * bSafe}, like.options());
        occSrc.narrow(0, 0, d.rows).narrow(1, 0, d.cols).copy_(d.storage);
    }
    torch::Tensor occ;
    if (occSrc.scalar_type() == torch::kComplexFloat) {
        // amax lacks complex64 on NPU; judge the components separately.
        occ = (torch::real(occSrc) != 0) | (torch::imag(occSrc) != 0);
    } else {
        occ = (occSrc != 0);
    }
    occ = occ.view({br, bSafe, bc, bSafe})
              .amax(/*dim=*/std::vector<int64_t>{1, 3})
              .reshape({br, bc});

    // Pattern construction fully on device: slot of the j-th occupied
    // block equals prefix(i, j) - 1, ascending in j by construction
    // (reference ordering); -1 for empty slots. The host only observes
    // the width (single .item() sync) to size the value payload.
    auto counts = occ.sum(/*dim=*/1);  // bool sum promotes to int64
    const int64_t width = counts.max().item<int64_t>();
    if (width == 0) {
        auto emptyP = torch::empty({br, 0},
            torch::TensorOptions().dtype(torch::kInt32).device(dev));
        return {emptyP, 0};
    }
    auto prefix = occ.to(torch::kInt32).cumsum(1);
    auto rowsIdx = torch::arange(br, torch::TensorOptions()
                                         .dtype(torch::kLong).device(dev))
                       .unsqueeze(1);
    auto flat = (prefix.to(torch::kLong) - 1 + rowsIdx * width)
                    .masked_select(occ);
    auto srcIdx = torch::arange(bc, torch::TensorOptions()
                                        .dtype(torch::kLong).device(dev))
                      .unsqueeze(0)
                      .expand({br, bc})
                      .masked_select(occ);
    torch::Tensor pattern =
        torch::full({br * width}, -1,
                    torch::TensorOptions().dtype(torch::kInt32).device(dev));
    pattern.scatter_(0, flat, srcIdx.to(torch::kInt32));
    return {pattern.view({br, width}), width};
}

// Blocked-ELL conversion on the original (possibly tail-blocked) matrix.
// Returns (values[br, width, b, b] row-major tiles, pattern[br, width]).
std::pair<torch::Tensor, torch::Tensor> RunBell(const DenseOperand &d,
                                                const torch::Tensor &like,
                                                int64_t block)
{
    const auto dev = like.device();
    // The Python layer validates block >= 1; the clamp guards a stray
    // zero divisor (degrades to unit blocks instead of faulting).
    const int64_t rows = d.rows, cols = d.cols;
    const int64_t b = block > 0 ? block : 1;
    const int64_t br = (rows + b - 1) / b;
    const int64_t bc = (cols + b - 1) / b;

    auto [pattern, width] = BuildBellPattern(d, like, b, br, bc);
    if (width == 0) {
        auto emptyV = torch::empty({br, 0, b, b}, like.options());
        return {emptyV, pattern};
    }
    // The pattern above is materialized by torch device kernels while
    // Convert is launched through the raw aclsparse handle: without this
    // barrier the externally-launched kernel can read the pattern before
    // the scatter is globally visible (see the hook-side twin fix).
    aclrtSynchronizeStream(
        c10_npu::getCurrentNPUStream(like.device().index()).stream());

    const int64_t E = like.element_size();
    torch::Tensor vals = torch::empty({br * width * b * b * E},
                                      torch::TensorOptions()
                                          .dtype(torch::kUInt8)
                                          .device(dev));
    return RunAclSparse(like, [&](AclSparseContext &ctx) {
        aclsparseDnMatDescr_t dd = nullptr;
        ACLSPARSE_CHECK(aclsparseCreateDnMat(
            &dd, rows, cols, d.ld, d.ptr, ToAclType(like), d.order));
        DescriptorGuard guard;
        guard.AdoptDn(dd);
        aclsparseSpMatDescr_t sp = nullptr;
        ACLSPARSE_CHECK(aclsparseCreateBlockedEll(
            &sp, rows, cols, b, width * b, pattern.data_ptr(),
            vals.data_ptr(), ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_BASE_ZERO, ToAclType(like)));
        guard.AdoptSp(sp);
        ACLSPARSE_CHECK(aclsparseDenseToSparseConvert(
            ctx.handle(), dd, sp, kAlg, nullptr));
        ctx.RecordTensors(vals, pattern);
        // aclsparse lays block interiors column-major; torch wants row-major.
        auto vb = vals.view(like.scalar_type())
                      .view({br, width, b, b})
                      .transpose(-2, -1)
                      .contiguous();
        return std::make_pair(vb, pattern);
    });
}

} // namespace

// Raw operator-chain exports. The ATen dispatcher registrations and torch
// sparse-tensor assembly live in the Python layer (dense_to_sparse.py):
// building sparse tensors from inside a backend kernel re-enters the
// dispatcher with that backend key excluded, which misroutes allocations.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("to_csr", [](torch::Tensor dense) {
        ValidateDense(dense);
        TriOut out = RunCompressed(ACL_SPARSE_FORMAT_CSR,
                                   PrepareDense(dense), dense);
        return py::make_tuple(out.values, out.major, out.minor);
    });
    m.def("to_csc", [](torch::Tensor dense) {
        ValidateDense(dense);
        TriOut out = RunCompressed(ACL_SPARSE_FORMAT_CSC,
                                   PrepareDense(dense), dense);
        return py::make_tuple(out.values, out.major, out.minor);
    });
    m.def("to_coo", [](torch::Tensor dense) {
        ValidateDense(dense);
        TriOut out = RunCompressed(ACL_SPARSE_FORMAT_COO,
                                   PrepareDense(dense), dense);
        return py::make_tuple(out.values, out.major, out.minor);
    });
    m.def("to_bell", [](torch::Tensor dense, int64_t block) {
        ValidateDense(dense);
        return RunBell(PrepareDense(dense), dense, block);
    });
}
