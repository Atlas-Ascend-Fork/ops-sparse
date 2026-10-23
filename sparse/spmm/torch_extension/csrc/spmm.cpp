// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// CSR x dense SpMM 的 C++ 封装（aten::addmm.out）。
//
// 该 wrapper 只做 ATen <-> ACLSparse 的桥接（张量解包、描述符生命周期、布局描述），
// 计算全部交给本仓对应架构实现的 aclsparseSpMM* C API，不重复实现
// 任何 tiling / kernel 逻辑。handle、stream、workspace 与 storage 生命周期按规范
// 交给公共层 aclsparse_common.h。
//
// 覆盖范围：fp32/fp16/bf16/complex64，opA=opB=NON_TRANSPOSE，CSR idxBase=ZERO
// （PyTorch 原生 sparse_csr_tensor 只能产生 0 基索引）。紧凑行/列主序与带 padding
// 的稠密布局就地描述；其余合法 stride 会物化成连续副本。转置与 idxBase=ONE 只能
// 通过裸 C++ API 使用——torch.sparse.addmm 和 PyTorch 的 sparse_csr_tensor 都没有
// 暴露这两个控制项。

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>

#include <dlfcn.h>

#include <torch/extension.h>
// pybind11 没有 c10::Scalar 的类型转换器，因此 beta/alpha 以 std::complex<double>
// 跨越边界，可接受 Python int/float/bool/complex。
#include <pybind11/complex.h>

#include "aclsparse_common.h"

extern "C" aclsparseStatus_t aclsparseSpMMPrivateOutOfPlace(
    aclsparseHandle_t handle, aclsparseOperation_t opA,
    aclsparseOperation_t opB, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseConstDnMatDescr_t initialC,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer);

namespace {

using cann_ops_sparse::AclSparseContext;
using cann_ops_sparse::AclSparseWorkspace;
using cann_ops_sparse::RunAclSparse;

constexpr aclsparseOperation_t kOp = ACL_SPARSE_OP_NON_TRANSPOSE;
constexpr aclsparseSpMMAlg_t kAlg = ACL_SPARSE_SPMM_ALG_DEFAULT;
constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();

aclDataType ToAclDataType(at::ScalarType scalarType)
{
    switch (scalarType) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kComplexFloat:
            return ACL_COMPLEX64;
        default:
            TORCH_CHECK(false, "aclsparseSpMM NPU adapter: unsupported dtype ", scalarType,
                " (torch.sparse.addmm on NPU currently supports float16/bfloat16/float32/complex64)");
    }
}

// aclsparseSpMMGetBufferSize/Preprocess/SpMM 的 computeType 不是简单的"就是那个
// dtype"：spmm_host.cpp 的 IsSupportedSpmmDtypeCombo 要求 float/half/bf16 一律是
// ACL_FLOAT（存储精度可变，累加恒为 fp32），complex64 则必须是 ACL_COMPLEX64
// （复数是唯一存储与计算 dtype 相同的情形）。
aclDataType ComputeTypeFor(aclDataType valueType)
{
    return valueType == ACL_COMPLEX64 ? ACL_COMPLEX64 : ACL_FLOAT;
}

class SpMatGuard {
public:
    SpMatGuard(int64_t rows, int64_t cols, int64_t nnz, void *rowOffsets, void *colIndices, void *values,
        aclDataType valueType)
    {
        // idxBase 恒为 ZERO：torch 的 sparse_csr_tensor 只产生 0 基索引，1 基 CSR
        // 只能走裸 aclsparseCreateCsr 入口，到不了这条 ATen 路径。
        ACLSPARSE_CHECK(aclsparseCreateCsr(&value_, rows, cols, nnz, rowOffsets, colIndices, values,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, valueType));
    }

    ~SpMatGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroySpMat(value_);
        }
    }

    SpMatGuard(const SpMatGuard &) = delete;
    SpMatGuard &operator=(const SpMatGuard &) = delete;

    aclsparseSpMatDescr_t get() const
    {
        return value_;
    }

private:
    aclsparseSpMatDescr_t value_ = nullptr;
};

class DnMatGuard {
public:
    DnMatGuard(int64_t rows, int64_t cols, int64_t ld, void *values, aclDataType valueType,
        aclsparseOrder_t order)
    {
        ACLSPARSE_CHECK(aclsparseCreateDnMat(&value_, rows, cols, ld, values, valueType, order));
    }

    ~DnMatGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroyDnMat(value_);
        }
    }

    DnMatGuard(const DnMatGuard &) = delete;
    DnMatGuard &operator=(const DnMatGuard &) = delete;

    aclsparseDnMatDescr_t get() const
    {
        return value_;
    }

private:
    aclsparseDnMatDescr_t value_ = nullptr;
};

// 把 2-D 稠密 NPU 张量解析成 aclsparseCreateDnMat 需要的 (order, ld, 背后张量)
// 三元组。紧凑行主序缓冲区（含 ld > 逻辑列数的 padding）就地描述、零拷贝；其余
// 情形（比如切片视图里连打包维度本身 stride 都不是 1）物化成一份新的行主序连续
// 缓冲，对应任务书"非连续 Tensor……无法直接描述时生成连续副本"的要求。
struct DenseLayout {
    aclsparseOrder_t order;
    int64_t ld;
    at::Tensor storage; // 持有实际被描述的那个张量（入参本身或物化出的副本）
};

DenseLayout ResolveDenseLayout(const at::Tensor &input)
{
    TORCH_CHECK(input.dim() == 2, "aclsparseSpMM NPU adapter: dense operand must be 2-D, got dim=",
        input.dim());
    // 先解析懒共轭位，再判断布局：紧凑 B.conj() 的 stride 与原 tensor 相同，会命中
    // 下面的快路径；若直接保留 input，DnMatGuard 用 storage.data_ptr() 读到的是未
    // 共轭的底层数据。aten::addmm.out 的 Conjugate 分发键是 fallthrough，不能依赖
    // 通用 fallback 提前物化。不带 conj 位时 resolve_conj() 是 no-op、仍为零拷贝。
    const at::Tensor tensor = input.resolve_conj();
    const int64_t cols = tensor.size(1);
    const int64_t rowStride = tensor.stride(0);
    const int64_t colStride = tensor.stride(1);
    // ATen 这条路径只要能走就走紧凑行主序的优化 kernel；其余合法 stride 按任务书
    // 物化。裸 aclsparse 调用方仍可通过自己的稠密描述符使用通用的行/列主序 +
    // padded-ld 路径。
    if (cols > 0 && colStride == 1 && rowStride == cols) {
        return {ACL_SPARSE_ORDER_ROW, cols, tensor};
    }
    at::Tensor contiguous = tensor.contiguous();
    return {ACL_SPARSE_ORDER_ROW, std::max<int64_t>(contiguous.size(1), 1), contiguous};
}

// 默认 ACL_SPARSE_POINTER_MODE_HOST（公共层建 handle 时设定，本文件从不覆盖）下
// aclsparseSpMM* 期望的 alpha/beta 原始字节：实数一个 fp32，complex64 是 {re, im}
// 两个 fp32——必须与 spmm_host.cpp 的 ReadScalarComplex64 读取布局严格一致。
struct ScalarBuf {
    float data[2] = {0.0f, 0.0f};
};

ScalarBuf ToAclScalar(const std::complex<double> &value, aclDataType valueType, const char *name)
{
    ScalarBuf buf;
    if (valueType == ACL_COMPLEX64) {
        buf.data[0] = static_cast<float>(value.real());
        buf.data[1] = static_cast<float>(value.imag());
    } else {
        TORCH_CHECK(std::fabs(value.imag()) <= 0.0, "aclsparseSpMM NPU adapter: ", name,
            " has a non-zero imaginary part but mat1/mat2/self dtype is real "
            "(task book \"实数dtype不接受虚部非零的复数\")");
        buf.data[0] = static_cast<float>(value.real());
    }
    return buf;
}

bool ScalarIsZero(const ScalarBuf &value, aclDataType valueType)
{
    return value.data[0] == 0.0f && (valueType != ACL_COMPLEX64 || value.data[1] == 0.0f);
}

using SpmmOutOfPlaceFn = aclsparseStatus_t (*)(aclsparseHandle_t, aclsparseOperation_t, aclsparseOperation_t,
    const void *, aclsparseConstSpMatDescr_t, aclsparseConstDnMatDescr_t, const void *, aclsparseDnMatDescr_t,
    void *, aclDataType, aclsparseSpMMAlg_t, void *);

using SpmmPrivateOutOfPlaceFn = decltype(&aclsparseSpMMPrivateOutOfPlace);

SpmmPrivateOutOfPlaceFn GetSpmmPrivateOutOfPlaceFn()
{
    static const auto fn = reinterpret_cast<SpmmPrivateOutOfPlaceFn>(
        dlsym(RTLD_DEFAULT, "aclsparseSpMMPrivateOutOfPlace"));
    return fn;
}


// aclsparseSpMMOutOfPlaceInternal 不在对外头文件里，用 dlsym 取：取不到就退回普通
// 的原地 aclsparseSpMM，功能不受影响，只是多一次 self -> out 的拷贝。
SpmmOutOfPlaceFn GetSpmmOutOfPlaceFn()
{
    static SpmmOutOfPlaceFn fn =
        reinterpret_cast<SpmmOutOfPlaceFn>(dlsym(RTLD_DEFAULT, "aclsparseSpMMOutOfPlaceInternal"));
    return fn;
}

void ValidateAddmmInputs(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const at::Tensor &out)
{
    TORCH_CHECK(self.defined() && mat1.defined() && mat2.defined() && out.defined(),
        "aclsparseSpMM NPU adapter: self/mat1/mat2/out must all be defined tensors");
    TORCH_CHECK(mat1.layout() == c10::kSparseCsr, "aclsparseSpMM NPU adapter: mat1 must be a CSR sparse tensor");
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "aclsparseSpMM NPU adapter: mat1/mat2 must be 2-D matrices");
    TORCH_CHECK(mat1.size(1) == mat2.size(0), "aclsparseSpMM NPU adapter: mat1.size(1) (", mat1.size(1),
        ") must equal mat2.size(0) (", mat2.size(0), ")");
    TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type() && mat1.scalar_type() == self.scalar_type(),
        "aclsparseSpMM NPU adapter: input/mat1/mat2 must share one dtype (no implicit promotion), got self=",
        self.scalar_type(), " mat1=", mat1.scalar_type(), " mat2=", mat2.scalar_type());
    TORCH_CHECK(out.scalar_type() == mat1.scalar_type(),
        "aclsparseSpMM NPU adapter: out dtype must match input dtype, got out=", out.scalar_type(), " input=",
        mat1.scalar_type());
    TORCH_CHECK(mat1.device().type() == c10::DeviceType::PrivateUse1 &&
            mat2.device().type() == c10::DeviceType::PrivateUse1 &&
            self.device().type() == c10::DeviceType::PrivateUse1,
        "aclsparseSpMM NPU adapter: all operands must already be on the same NPU device");
    TORCH_CHECK(mat1.device() == mat2.device() && mat1.device() == self.device() && mat1.device() == out.device(),
        "aclsparseSpMM NPU adapter: input/mat1/mat2/out must be on the same NPU device, got self=", self.device(),
        " mat1=", mat1.device(), " mat2=", mat2.device(), " out=", out.device());
    TORCH_CHECK(mat1.size(0) <= kInt32Max && mat1.size(1) <= kInt32Max && mat1._nnz() <= kInt32Max,
        "aclsparseSpMM NPU adapter: CSR dimensions and nnz must fit signed int32 indices, got rows=",
        mat1.size(0), " cols=", mat1.size(1), " nnz=", mat1._nnz());
}

// 广播过的 self 仍需物化。self 已经是整块连续的 m x n 时，adapter 专用的
// out-of-place kernel 可以直接读它，out 保持全新、省掉一次拷贝。
void PrepareOutput(const at::Tensor &self, int64_t m, int64_t n, bool skipCopy, at::Tensor &out)
{
    if (out.sizes() != c10::IntArrayRef({m, n})) {
        out.resize_({m, n});
    }
    // 实测 composite addmm() wrapper 传进来的一直是全新的 at::empty(...)，所以这条
    // 成立；调用方自带的非连续 out（只能通过裸 .out() overload 到达，torch.sparse.addmm
    // 到不了）留作后续，官方 ATen 测试路径触发不到。
    TORCH_CHECK(out.is_contiguous(), "aclsparseSpMM NPU adapter: `out` must be contiguous row-major");
    at::Tensor src = self.expand({m, n});
    if (skipCopy) {
        // 即便 beta=0 或者 out-of-place kernel 会直接读原始 self，也要 expand 一次，
        // 用来校验广播契约本身是否成立。
        return;
    }
    if (out.is_complex()) {
        // torch_npu 没有 complex64 的 AiCore copy 实现，out.copy_() 会回落到 AI CPU
        // 的 Identity 算子：实测 32MB 要 8ms，比 AiCore 慢约 300 倍，直接盖过 SpMM
        // 本身的耗时。复数在内存里就是实部/虚部交织的 float 对，改走等价的 float
        // 视图搬运，结果逐位相同但回到 AiCore。
        //
        // resolve_conj() 是必须的：view_as_real 拒绝处理带 unresolved conj 位的
        // tensor（比如调用方传了 self.conj() 这种懒惰共轭视图），不 resolve 会直接
        // 抛 RuntimeError。src 不带 conj 位时 resolve_conj() 是 no-op、返回同一块
        // 内存（实测 data_ptr 不变），不引入额外拷贝，不影响上面这条优化路径的性能。
        at::view_as_real(out).copy_(at::view_as_real(src.resolve_conj()));
    } else {
        out.copy_(src);
    }
}

// aclsparseCreateCsr 需要连续的 int32 索引，而 PyTorch 默认给的是 int64 crow/col，
// 所以在这里窄化并压实。返回的张量必须活得比描述符久，因此返回张量而不是裸指针。
struct CsrOperands {
    at::Tensor crow;
    at::Tensor col;
    at::Tensor values;
};

at::Tensor NarrowCsrIndex(const at::Tensor &source, const char *name)
{
    if (source.scalar_type() == at::kInt) {
        return source.contiguous();
    }
    at::Tensor narrowed = source.to(at::kInt);
    TORCH_CHECK(at::equal(source, narrowed.to(at::kLong)), "aclsparseSpMM NPU adapter: CSR ", name,
        " values must fit signed int32");
    return narrowed.contiguous();
}

CsrOperands PrepareCsrOperands(const at::Tensor &crow, const at::Tensor &col, const at::Tensor &values)
{
    TORCH_CHECK((crow.scalar_type() == at::kInt || crow.scalar_type() == at::kLong) &&
            (col.scalar_type() == at::kInt || col.scalar_type() == at::kLong),
        "aclsparseSpMM NPU adapter: CSR row/column indices must be int32 or int64");
    CsrOperands ops;
    ops.crow = NarrowCsrIndex(crow, "row-offset");
    ops.col = NarrowCsrIndex(col, "column-index");
    ops.values = values.contiguous();
    return ops;
}

bool CanUseSeparateOutput(SpmmOutOfPlaceFn outOfPlace, bool skipInput, const at::Tensor &self,
    const at::Tensor &out, int64_t m, int64_t n)
{
    return (outOfPlace != nullptr || (GetSpmmPrivateOutOfPlaceFn() != nullptr &&
        (self.scalar_type() == at::kFloat || self.scalar_type() == at::kComplexFloat))) && !skipInput && self.dim() == 2 &&
        self.size(0) == m && self.size(1) == n && self.is_contiguous() && !self.is_conj() && !self.is_neg() &&
        !out.is_conj() && !out.is_neg();
}

void ValidateEmptySparseStorage(const CsrOperands &csr, int64_t m, int64_t k)
{
    const int64_t nnz = csr.values.numel();
    TORCH_CHECK(csr.crow.dim() == 1 && csr.col.dim() == 1 &&
                csr.values.dim() == 1 && csr.crow.numel() == m + 1 &&
                csr.col.numel() == nnz && nnz <= std::numeric_limits<int32_t>::max(),
        "invalid CSR storage sizes");
    const at::Tensor rows = csr.crow.cpu();
    const int32_t *rowOffsets = rows.data_ptr<int32_t>();
    TORCH_CHECK(rowOffsets[0] == 0 && rowOffsets[m] == nnz,
        "CSR row offsets must start at zero and end at nnz");
    for (int64_t row = 0; row < m; ++row) {
        TORCH_CHECK(rowOffsets[row] >= 0 && rowOffsets[row] <= rowOffsets[row + 1] &&
                    rowOffsets[row + 1] <= nnz, "invalid CSR row offset range");
    }
    if (nnz == 0) return;
    const at::Tensor columns = csr.col.cpu();
    const int32_t *columnIndices = columns.data_ptr<int32_t>();
    for (int64_t position = 0; position < nnz; ++position) {
        TORCH_CHECK(columnIndices[position] >= 0 && columnIndices[position] < k,
            "CSR column index is out of bounds");
    }
}

void ExecuteSeparateOutput(aclsparseHandle_t handle, const ScalarBuf &alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const ScalarBuf &beta, aclsparseDnMatDescr_t initialC, at::Tensor &out,
    aclDataType valueType, aclDataType computeType, void *buffer, const at::Tensor &self)
{
    const auto native = GetSpmmOutOfPlaceFn();
    if (native != nullptr) {
        ACLSPARSE_CHECK(native(handle, kOp, kOp, alpha.data, matA, matB, beta.data,
            initialC, out.data_ptr(), computeType, kAlg, buffer));
        return;
    }
    const DnMatGuard output(out.size(0), out.size(1), out.size(1), out.data_ptr(),
        valueType, ACL_SPARSE_ORDER_ROW);
    const DnMatGuard input(out.size(0), out.size(1), out.size(1), self.data_ptr(),
        valueType, ACL_SPARSE_ORDER_ROW);
    const auto status = GetSpmmPrivateOutOfPlaceFn()(handle, kOp, kOp, alpha.data, matA,
        matB, beta.data, input.get(), output.get(), computeType, kAlg, buffer);
    if (status != ACL_SPARSE_STATUS_NOT_SUPPORTED) {
        ACLSPARSE_CHECK(status);
        return;
    }
    PrepareOutput(self, out.size(0), out.size(1), false, out);
    (void)c10_npu::getCurrentNPUStream(out.device().index()).stream(true);
    ACLSPARSE_CHECK(aclsparseSpMM(handle, kOp, kOp, alpha.data, matA, matB, beta.data,
        output.get(), computeType, kAlg, buffer));
}

bool RunEmptyProduct(const CsrOperands &csr, int64_t m, int64_t k, int64_t n,
    const ScalarBuf &beta, aclDataType valueType, const at::Tensor &self, at::Tensor &out)
{
    if (m != 0 && k != 0 && n != 0 && csr.values.numel() != 0) return false;
    ValidateEmptySparseStorage(csr, m, k);
    if (m == 0 || n == 0) return true;
    if (ScalarIsZero(beta, valueType)) {
        out.zero_();
    } else {
        PrepareOutput(self, m, n, false, out);
        const at::Scalar scale = valueType == ACL_COMPLEX64
            ? at::Scalar(c10::complex<double>(beta.data[0], beta.data[1]))
            : at::Scalar(beta.data[0]);
        out.mul_(scale);
    }
    return true;
}

struct SpmmRequest {
    at::Tensor self;
    at::Tensor mat1;
    at::Tensor mat2;
    at::Tensor out;
    ScalarBuf alpha;
    ScalarBuf beta;
    aclDataType valueType;
};

void LaunchSpmm(AclSparseContext &context, const CsrOperands &csr,
    const DenseLayout &matB, SpmmRequest &request, bool separateOutput)
{
    const auto &mat1 = request.mat1;
    const auto &alpha = request.alpha;
    const auto &beta = request.beta;
    const auto valueType = request.valueType;
    const auto computeType = ComputeTypeFor(valueType);
    const int64_t m = mat1.size(0), k = mat1.size(1), n = request.mat2.size(1);
    const SpMatGuard matA(m, k, csr.values.numel(), csr.crow.data_ptr(), csr.col.data_ptr(),
        csr.values.data_ptr(), valueType);
    const DnMatGuard denseB(k, n, matB.ld, matB.storage.data_ptr(), valueType, matB.order);
    void *initialC = separateOutput && GetSpmmOutOfPlaceFn() != nullptr
        ? request.self.data_ptr() : request.out.data_ptr();
    const DnMatGuard denseC(m, n, n, initialC, valueType, ACL_SPARSE_ORDER_ROW);
    const auto handle = context.handle();
    size_t bufferSize = 0;
    ACLSPARSE_CHECK(aclsparseSpMMGetBufferSize(handle, kOp, kOp, alpha.data, matA.get(),
        denseB.get(), beta.data, denseC.get(), computeType, kAlg, &bufferSize));
    AclSparseWorkspace workspace(context, csr.values, bufferSize);
    ACLSPARSE_CHECK(aclsparseSpMMPreprocess(handle, kOp, kOp, alpha.data, matA.get(),
        denseB.get(), beta.data, denseC.get(), computeType, kAlg, workspace.data()));
    if (separateOutput) {
        ExecuteSeparateOutput(handle, alpha, matA.get(), denseB.get(), beta, denseC.get(),
            request.out, valueType, computeType, workspace.data(), request.self);
    } else {
        ACLSPARSE_CHECK(aclsparseSpMM(handle, kOp, kOp, alpha.data, matA.get(), denseB.get(),
            beta.data, denseC.get(), computeType, kAlg, workspace.data()));
    }
}

void ExecuteSpmmRequest(SpmmRequest &request, AclSparseContext &context)
{
    auto &out = request.out;
    const auto &mat1 = request.mat1;
    const int64_t m = mat1.size(0), k = mat1.size(1), n = request.mat2.size(1);
    request.mat2 = out.is_alias_of(request.mat2) ? request.mat2.clone() : request.mat2;
    request.self = out.is_alias_of(request.self) ? request.self.clone() : request.self;
    const auto crow = out.is_alias_of(mat1.crow_indices()) ? mat1.crow_indices().clone() : mat1.crow_indices();
    const auto col = out.is_alias_of(mat1.col_indices()) ? mat1.col_indices().clone() : mat1.col_indices();
    const auto values = out.is_alias_of(mat1.values()) ? mat1.values().clone() : mat1.values();
    const bool skipInput = ScalarIsZero(request.beta, request.valueType);
    const bool separateOutput = CanUseSeparateOutput(GetSpmmOutOfPlaceFn(), skipInput,
        request.self, out, m, n);
    PrepareOutput(request.self, m, n, skipInput || separateOutput, out);
    const auto csr = PrepareCsrOperands(crow, col, values);
    if (RunEmptyProduct(csr, m, k, n, request.beta, request.valueType, request.self, out)) return;
    const auto matB = ResolveDenseLayout(request.mat2);
    context.RecordTensors(csr.crow, csr.col, csr.values, matB.storage, request.self, out);
    // Submit queued tensor preparation before ACL launches on the same stream.
    (void)c10_npu::getCurrentNPUStream(mat1.device().index()).stream(true);
    LaunchSpmm(context, csr, matB, request, separateOutput);
}

at::Tensor AddmmOut(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const std::complex<double> &beta, const std::complex<double> &alpha, at::Tensor out)
{
    ValidateAddmmInputs(self, mat1, mat2, out);
    const auto valueType = ToAclDataType(mat1.scalar_type());
    SpmmRequest request{self, mat1, mat2, out, ToAclScalar(alpha, valueType, "alpha"),
        ToAclScalar(beta, valueType, "beta"), valueType};
    RunAclSparse(mat1, [&](AclSparseContext &context) { ExecuteSpmmRequest(request, context); });
    return out;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("release_cached_handles", &cann_ops_sparse::AclSparseHandleCache::Clear);
    m.def("sparse_addmm", [](const at::Tensor &self, const at::Tensor &mat1,
              const at::Tensor &mat2, const std::complex<double> &beta,
              const std::complex<double> &alpha) {
        auto out = at::empty({mat1.size(0), mat2.size(1)}, self.options());
        return AddmmOut(self, mat1, mat2, beta, alpha, out);
    }, "beta * self + alpha * (mat1 @ mat2) for CSR x dense on NPU");
    m.def("addmm_out", &AddmmOut, "beta * self + alpha * (mat1 @ mat2) into out, for a CSR mat1 on NPU");
}
