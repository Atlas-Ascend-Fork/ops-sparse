// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// CSR x CSR SpGEMM 的 C++ 封装（aten::_sparse_addmm / aten::_sparse_sparse_matmul）。

#include <array>
#include <complex>
#include <limits>
#include <utility>

#include <torch/extension.h>
// pybind11 没有 c10::Scalar 的类型转换器，因此 beta/alpha 以
// std::complex<double> 跨越边界，可接受 Python int/float/bool/complex。
#include <pybind11/complex.h>

#include "securec.h"
#include "aclsparse_common.h"

namespace {

using cann_ops_sparse::AclSparseContext;
using cann_ops_sparse::AclSparseWorkspace;
using cann_ops_sparse::RunAclSparse;

constexpr aclsparseOperation_t kOp = ACL_SPARSE_OP_NON_TRANSPOSE;
// arch22 实现 DEFAULT/ALG1；ALG2/ALG3 多一个 EstimateMemory 阶段，
// DEFAULT 没有该阶段，bufferSize2 改由 Compute 查询。
constexpr aclsparseSpGEMMAlg_t kAlg = ACL_SPARSE_SPGEMM_DEFAULT;
constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
// 库通过 host 指针读取的最宽标量：complex64 = 2 × fp32。
constexpr size_t kHostScalarBytes = 8;
// _sparse_sparse_matmul 没有自己的 beta/alpha，纯乘积。
const std::complex<double> kZero {0.0, 0.0};
const std::complex<double> kOne {1.0, 0.0};

// ACLSparse 通过 create/destroy 管理描述符而非智能指针，
// 此 RAII 守卫确保异常路径也能正确销毁。
class SpGemmDescrGuard {
public:
    SpGemmDescrGuard()
    {
        ACLSPARSE_CHECK(aclsparseSpGEMMCreateDescr(&value_));
    }

    ~SpGemmDescrGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseSpGEMMDestroyDescr(value_);
        }
    }

    SpGemmDescrGuard(const SpGemmDescrGuard &) = delete;
    SpGemmDescrGuard &operator=(const SpGemmDescrGuard &) = delete;

    aclsparseSpGEMMDescr_t get() const
    {
        return value_;
    }

private:
    aclsparseSpGEMMDescr_t value_ = nullptr;
};

class ConstSpMatGuard {
public:
    ConstSpMatGuard() = default;

    ~ConstSpMatGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroySpMat(value_);
        }
    }

    ConstSpMatGuard(const ConstSpMatGuard &) = delete;
    ConstSpMatGuard &operator=(const ConstSpMatGuard &) = delete;

    aclsparseConstSpMatDescr_t *address()
    {
        return &value_;
    }

    aclsparseConstSpMatDescr_t get() const
    {
        return value_;
    }

private:
    aclsparseConstSpMatDescr_t value_ = nullptr;
};

class SpMatGuard {
public:
    SpMatGuard() = default;

    ~SpMatGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroySpMat(value_);
        }
    }

    SpMatGuard(const SpMatGuard &) = delete;
    SpMatGuard &operator=(const SpMatGuard &) = delete;

    aclsparseSpMatDescr_t *address()
    {
        return &value_;
    }

    aclsparseSpMatDescr_t get() const
    {
        return value_;
    }

private:
    aclsparseSpMatDescr_t value_ = nullptr;
};

aclDataType ToAclDataType(at::ScalarType type)
{
    switch (type) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kComplexFloat:
            return ACL_COMPLEX64;
        default:
            TORCH_CHECK(false, "SpGEMM supports float16, bfloat16, float32 and complex64 values, but got ", type);
    }
}

// alpha/beta 的 host 暂存区。库在 HOST pointer mode 下通过 host 指针读取，
// 且每个阶段都会重新按位比较 beta 与 WorkEstimation 缓存的值。
// 只构建一次并在所有阶段传同一地址，避免重建导致比较不一致。
class HostScalar {
public:
    HostScalar() = default;

    HostScalar(const std::complex<double> &value, at::ScalarType type)
    {
        TORCH_CHECK(type == at::kComplexFloat || value.imag() == 0.0,
            "SpGEMM cannot apply a complex scalar to a real-valued dtype, but got dtype ", type,
            " with imaginary part ", value.imag());
        switch (type) {
            case at::kHalf:
                Store(c10::Half(static_cast<float>(value.real())));
                break;
            case at::kBFloat16:
                Store(c10::BFloat16(static_cast<float>(value.real())));
                break;
            case at::kFloat:
                Store(static_cast<float>(value.real()));
                break;
            case at::kComplexFloat:
                Store(aclsparseComplex{static_cast<float>(value.real()), static_cast<float>(value.imag())});
                break;
            default:
                TORCH_CHECK(false, "SpGEMM cannot represent a scalar in dtype ", type);
        }
    }

    // 不返回 nullptr：SpgemmReadScalar 会将空指针视为 0.0 且不报错，
    // 从而静默丢弃 alpha。
    const void *data() const
    {
        return bytes_.data();
    }

private:
    template <typename T>
    void Store(const T &value)
    {
        static_assert(sizeof(T) <= kHostScalarBytes, "host scalar does not fit the staging buffer");
        static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
        auto ret = memcpy_s(bytes_.data(), kHostScalarBytes, &value, sizeof(value));
        TORCH_CHECK(ret == EOK, "memcpy_s failed in HostScalar::Store with errno ", ret);
    }

    alignas(kHostScalarBytes) std::array<uint8_t, kHostScalarBytes> bytes_ {};
};

// 单个 CSR 操作数，转为 ACLSparse 要求的连续 int32 数组。
struct CsrOperand {
    at::Tensor rowOffsets;
    at::Tensor colIndices;
    at::Tensor values;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
};

void CheckCsrOperand(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.defined(), name, " must be a defined tensor");
    TORCH_CHECK(tensor.layout() == at::kSparseCsr, name, " must use the sparse CSR layout, but got ",
        tensor.layout());
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name,
        " must be on an Ascend NPU, but got device ", tensor.device());
    TORCH_CHECK(tensor.dim() == 2, name, " must be a 2-D matrix, but got ", tensor.dim(), " dimensions");
    TORCH_CHECK(tensor.size(0) <= kInt32Max && tensor.size(1) <= kInt32Max && tensor._nnz() <= kInt32Max,
        name, " exceeds the int32 CSR limit: shape ", tensor.sizes(), " with nnz ", tensor._nnz(),
        " must stay within ", kInt32Max);
}

// PyTorch 默认以 int64 存储 CSR 索引，ACLSparse 要求 int32；
// 窄化在 device 上完成，取裸指针前确保连续。
CsrOperand MakeCsrOperand(const at::Tensor &tensor)
{
    CsrOperand operand;
    operand.rows = tensor.size(0);
    operand.cols = tensor.size(1);
    operand.nnz = tensor._nnz();
    operand.rowOffsets = tensor.crow_indices().to(at::kInt).contiguous();
    operand.colIndices = tensor.col_indices().to(at::kInt).contiguous();
    operand.values = tensor.values().contiguous();
    return operand;
}

void CreateConstCsr(ConstSpMatGuard &descriptor, const CsrOperand &operand, aclDataType valueType)
{
    ACLSPARSE_CHECK(aclsparseCreateConstCsr(descriptor.address(), operand.rows, operand.cols, operand.nnz,
        operand.rowOffsets.const_data_ptr(), operand.colIndices.const_data_ptr(),
        operand.values.const_data_ptr(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, valueType));
}

// 四个阶段共用的全部上下文，alpha/beta 和描述符只构建一次。
struct SpGemmPlan {
    aclDataType valueType = ACL_FLOAT;
    at::ScalarType scalarType = at::kFloat;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    bool betaNonZero = false;
    HostScalar alpha;
    HostScalar beta;
    CsrOperand a;
    CsrOperand b;
    CsrOperand cIn;
    at::Tensor cRowOffsets;
};

SpGemmPlan MakePlan(const c10::optional<at::Tensor> &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const std::complex<double> &beta, const std::complex<double> &alpha)
{
    // 在访问任何数据指针或分配 device 内存之前完成校验。
    CheckCsrOperand(mat1, "mat1");
    CheckCsrOperand(mat2, "mat2");
    TORCH_CHECK(mat1.device() == mat2.device(), "mat1 and mat2 must be on the same NPU, but got ",
        mat1.device(), " and ", mat2.device());
    TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(), "mat1 and mat2 must share one dtype, but got ",
        mat1.scalar_type(), " and ", mat2.scalar_type());
    TORCH_CHECK(mat1.size(1) == mat2.size(0), "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes(),
        " and ", mat2.sizes(), ")");

    SpGemmPlan plan;
    plan.scalarType = mat1.scalar_type();
    plan.valueType = ToAclDataType(plan.scalarType);
    plan.m = mat1.size(0);
    plan.k = mat1.size(1);
    plan.n = mat2.size(1);
    plan.alpha = HostScalar(alpha, plan.scalarType);
    plan.beta = HostScalar(beta, plan.scalarType);
    plan.betaNonZero = beta.real() != 0.0 || beta.imag() != 0.0;
    TORCH_CHECK(!plan.betaNonZero || self.has_value(),
        "SpGEMM needs the accumulated operand when beta is non-zero");
    // aten::addmm 对所有 beta（含零）都会校验 self，因此有 self 时一律检查，
    // 仅在 beta 非零时才将其绑定为 C_in。
    if (self.has_value()) {
        const at::Tensor &accumulated = *self;
        CheckCsrOperand(accumulated, "self");
        TORCH_CHECK(accumulated.device() == mat1.device(), "self must be on the same NPU as mat1, but got ",
            accumulated.device(), " and ", mat1.device());
        TORCH_CHECK(accumulated.scalar_type() == plan.scalarType, "self must share the dtype of mat1, but got ",
            accumulated.scalar_type(), " and ", plan.scalarType);
        TORCH_CHECK(accumulated.size(0) == plan.m && accumulated.size(1) == plan.n,
            "self must have shape [", plan.m, ", ", plan.n, "], but got ", accumulated.sizes());
        if (plan.betaNonZero) {
            plan.cIn = MakeCsrOperand(accumulated);
        }
    }

    plan.a = MakeCsrOperand(mat1);
    plan.b = MakeCsrOperand(mat2);
    // Copy 即使 nnz(C)=0 也会写 M+1 个 row offsets，因此始终按满长分配。
    plan.cRowOffsets = at::empty({plan.m + 1}, plan.a.values.options().dtype(at::kInt));
    return plan;
}

int64_t CheckedWorkspaceSize(size_t size, const char *name)
{
    TORCH_CHECK(size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()), "SpGEMM ", name,
        " of ", size, " bytes exceeds the PyTorch tensor limit");
    return static_cast<int64_t>(size);
}

// beta != 0 时 matC 同时是输入，WorkEstimation 在执行时快照 nnz(C_in)，
// 因此从第一个阶段起就必须携带 C_in 的数据。结果写入新分配的数组，
// 因为 Copy 会覆写 matC 上挂载的 row offsets。
void CreateOutputCsr(SpMatGuard &matC, const SpGemmPlan &plan)
{
    if (plan.betaNonZero) {
        ACLSPARSE_CHECK(aclsparseCreateCsr(matC.address(), plan.m, plan.n, plan.cIn.nnz,
            plan.cIn.rowOffsets.mutable_data_ptr(), plan.cIn.colIndices.mutable_data_ptr(),
            plan.cIn.values.mutable_data_ptr(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_BASE_ZERO, plan.valueType));
    } else {
        ACLSPARSE_CHECK(aclsparseCreateCsr(matC.address(), plan.m, plan.n, 0,
            plan.cRowOffsets.mutable_data_ptr(), nullptr, nullptr, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, plan.valueType));
    }
}

at::Tensor RunSpGemm(const c10::optional<at::Tensor> &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const std::complex<double> &beta, const std::complex<double> &alpha)
{
    return RunAclSparse(mat1, [&](AclSparseContext &context) {
        SpGemmPlan plan = MakePlan(self, mat1, mat2, beta, alpha);
        // Submit torch_npu's queue before calling ACLSparse directly. The
        // library orders CSR validation copies on this same handle stream.
        (void)c10_npu::getCurrentNPUStream(mat1.device().index()).stream(true);
        context.RecordTensors(plan.a.rowOffsets, plan.a.colIndices, plan.a.values, plan.b.rowOffsets,
            plan.b.colIndices, plan.b.values, plan.cIn.rowOffsets, plan.cIn.colIndices, plan.cIn.values,
            plan.cRowOffsets);

        ConstSpMatGuard matA;
        ConstSpMatGuard matB;
        SpMatGuard matC;
        CreateConstCsr(matA, plan.a, plan.valueType);
        CreateConstCsr(matB, plan.b, plan.valueType);
        CreateOutputCsr(matC, plan);

        SpGemmDescrGuard descriptor;
        const aclsparseHandle_t handle = context.handle();

        size_t bufferSize1 = 0;
        ACLSPARSE_CHECK(aclsparseSpGEMMWorkEstimation(handle, kOp, kOp, plan.alpha.data(), matA.get(),
            matB.get(), plan.beta.data(), matC.get(), plan.valueType, kAlg, descriptor.get(), &bufferSize1,
            nullptr));
        AclSparseWorkspace workBuffer(context, plan.a.values,
            static_cast<size_t>(CheckedWorkspaceSize(bufferSize1, "work buffer")));
        ACLSPARSE_CHECK(aclsparseSpGEMMWorkEstimation(handle, kOp, kOp, plan.alpha.data(), matA.get(),
            matB.get(), plan.beta.data(), matC.get(), plan.valueType, kAlg, descriptor.get(), &bufferSize1,
            workBuffer.data()));

        size_t bufferSize2 = 0;
        ACLSPARSE_CHECK(aclsparseSpGEMMCompute(handle, kOp, kOp, plan.alpha.data(), matA.get(), matB.get(),
            plan.beta.data(), matC.get(), plan.valueType, kAlg, descriptor.get(), &bufferSize2, nullptr));
        AclSparseWorkspace computeBuffer(context, plan.a.values,
            static_cast<size_t>(CheckedWorkspaceSize(bufferSize2, "compute buffer")));
        ACLSPARSE_CHECK(aclsparseSpGEMMCompute(handle, kOp, kOp, plan.alpha.data(), matA.get(), matB.get(),
            plan.beta.data(), matC.get(), plan.valueType, kAlg, descriptor.get(), &bufferSize2,
            computeBuffer.data()));

        int64_t rowsC = 0;
        int64_t colsC = 0;
        int64_t nnzC = 0;
        ACLSPARSE_CHECK(aclsparseSpMatGetSize(matC.get(), &rowsC, &colsC, &nnzC));
        TORCH_CHECK(rowsC == plan.m && colsC == plan.n && nnzC >= 0 && nnzC <= kInt32Max,
            "SpGEMM reported inconsistent output metadata: shape [", rowsC, ", ", colsC, "] with nnz ", nnzC,
            " for an expected shape [", plan.m, ", ", plan.n, "]");

        at::Tensor colIndices = at::empty({nnzC}, plan.a.values.options().dtype(at::kInt));
        at::Tensor values = at::empty({nnzC}, plan.a.values.options());
        context.RecordTensors(colIndices, values);
        ACLSPARSE_CHECK(aclsparseCsrSetPointers(matC.get(), plan.cRowOffsets.mutable_data_ptr(),
            colIndices.mutable_data_ptr(), values.mutable_data_ptr()));

        ACLSPARSE_CHECK(aclsparseSpGEMMCopy(handle, kOp, kOp, plan.alpha.data(), matA.get(), matB.get(),
            plan.beta.data(), matC.get(), plan.valueType, kAlg, descriptor.get()));

        return at::sparse_csr_tensor(plan.cRowOffsets, colIndices, values, {plan.m, plan.n},
            values.options().layout(at::kSparseCsr));
    });
}

at::Tensor ToCsr(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.defined(), name, " must be a defined tensor");
    TORCH_CHECK(tensor.layout() == at::kSparse, name, " must use the sparse COO layout, but got ",
        tensor.layout());
    TORCH_CHECK(tensor.dim() == 2, name, " must be a 2-D matrix, but got ", tensor.dim(), " dimensions");
    return tensor.to_sparse_csr();
}

// aten::_sparse_addmm 将 beta/alpha 作为 Scalar 传递；Python 分发器原样转发，
// pybind11 将其转为 std::complex<double>。
at::Tensor SparseAddmm(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const std::complex<double> &beta, const std::complex<double> &alpha)
{
    return RunSpGemm(self, mat1, mat2, beta, alpha);
}

at::Tensor SparseSparseMatmulCsr(const at::Tensor &mat1, const at::Tensor &mat2)
{
    return RunSpGemm(c10::nullopt, mat1, mat2, kZero, kOne);
}

at::Tensor SparseSparseMatmulCoo(const at::Tensor &mat1, const at::Tensor &mat2)
{
    // 布局转换使用 PyTorch 公共算子而非私有 COO/CSR 重实现；
    // 两个方向都保证输出是 coalesced 的。
    at::Tensor result = RunSpGemm(c10::nullopt, ToCsr(mat1, "mat1"), ToCsr(mat2, "mat2"), kZero, kOne);
    return result.to_sparse(at::kSparse);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_addmm", &SparseAddmm, "beta * self + alpha * (mat1 @ mat2) for CSR operands on NPU");
    m.def("sparse_sparse_matmul_csr", &SparseSparseMatmulCsr, "mat1 @ mat2 for CSR operands on NPU");
    m.def("sparse_sparse_matmul_coo", &SparseSparseMatmulCoo, "mat1 @ mat2 for COO operands on NPU");
}
