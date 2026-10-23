// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// CSR x dense vector SpMV 的 C++ 封装（aten::mv 的 SparseCsrPrivateUse1 实现）。
// 除 dispatcher 入口 mv 外，还导出 spmv_npu / spmv_npu_workspace 两个仅供测试的
// 普通 pybind 函数，用于覆盖完整 alpha/beta/trans/indexBase 参数面与 workspace 契约；
// 二者不注册任何 torch schema。

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include <torch/extension.h>
// pybind11 没有 c10::Scalar 的类型转换器，因此 alpha/beta 以
// std::complex<double> 跨越边界，可接受 Python int/float/bool/complex。
#include <pybind11/complex.h>

#include "securec.h"
#include "aclsparse_common.h"

namespace {

using cann_ops_sparse::AclSparseContext;
using cann_ops_sparse::AclSparseWorkspace;
using cann_ops_sparse::RunAclSparse;

constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
// 库通过 host 指针读取的最宽标量：complex64 = 2 × fp32。
constexpr size_t kHostScalarBytes = 8;
constexpr aclsparseSpMVAlg_t kAlg = ACL_SPARSE_SPMV_ALG_DEFAULT;

// ACLSparse 通过 create/destroy 管理描述符而非智能指针，
// RAII 守卫确保异常路径也能正确销毁。
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

class DnVecGuard {
public:
    DnVecGuard() = default;

    ~DnVecGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroyDnVec(value_);
        }
    }

    DnVecGuard(const DnVecGuard &) = delete;
    DnVecGuard &operator=(const DnVecGuard &) = delete;

    aclsparseDnVecDescr_t *address()
    {
        return &value_;
    }

    aclsparseDnVecDescr_t get() const
    {
        return value_;
    }

private:
    aclsparseDnVecDescr_t value_ = nullptr;
};

aclDataType ToAclDataType(at::ScalarType type)
{
    switch (type) {
        case at::kChar:
            return ACL_INT8;
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kInt:
            return ACL_INT32;
        case at::kComplexFloat:
            return ACL_COMPLEX64;
        default:
            TORCH_CHECK(false,
                "SpMV supports int8, float16, bfloat16, float32, int32 and complex64 values, but got ",
                type);
    }
}

// 输出 dtype：INT8 输入累加到 INT32，其余与输入一致
at::ScalarType SpmvOutScalarType(at::ScalarType valueType)
{
    return valueType == at::kChar ? at::kInt : valueType;
}

// 任务书 dtype 组合表：INT8→INT32/FP32、FP16/BF16→FP32 或自身、FP32→FP32/complex64、
// INT32→INT32/FP32、complex64→complex64
bool IsValidSpMvCombo(at::ScalarType valueType, at::ScalarType outType)
{
    switch (valueType) {
        case at::kChar:
            return outType == at::kInt || outType == at::kFloat;
        case at::kHalf:
            return outType == at::kHalf || outType == at::kFloat;
        case at::kBFloat16:
            return outType == at::kBFloat16 || outType == at::kFloat;
        case at::kFloat:
            return outType == at::kFloat || outType == at::kComplexFloat;
        case at::kInt:
            return outType == at::kInt || outType == at::kFloat;
        case at::kComplexFloat:
            return outType == at::kComplexFloat;
        default:
            return false;
    }
}

// 计算精度类型：INT32 输出走整数精确路径，complex64 输出走复数路径，其余 FP32
aclDataType SpmvComputeType(at::ScalarType outType)
{
    if (outType == at::kInt) {
        return ACL_INT32;
    }
    if (outType == at::kComplexFloat) {
        return ACL_COMPLEX64;
    }
    return ACL_FLOAT;
}

// alpha/beta 的 host 暂存区。库在 HOST pointer mode 下通过 host 指针读取。
class HostScalar {
public:
    HostScalar() = default;

    HostScalar(const std::complex<double> &value, aclDataType computeType)
    {
        switch (computeType) {
            case ACL_INT32: {
                TORCH_CHECK(value.imag() == 0.0,
                    "int32 SpMV cannot use a complex scalar, but got imaginary part ", value.imag());
                TORCH_CHECK(value.real() >= static_cast<double>(std::numeric_limits<int32_t>::min()) &&
                    value.real() <= static_cast<double>(kInt32Max),
                    "int32 SpMV scalar ", value.real(), " is out of the int32 range");
                Store(static_cast<int32_t>(value.real()));
                break;
            }
            case ACL_COMPLEX64: {
                Store(aclsparseComplex{static_cast<float>(value.real()),
                    static_cast<float>(value.imag())});
                break;
            }
            default: {
                TORCH_CHECK(value.imag() == 0.0,
                    "real SpMV compute type cannot use a complex scalar, but got imaginary part ",
                    value.imag());
                Store(static_cast<float>(value.real()));
                break;
            }
        }
    }

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

// nnz==0 时 col/values 可能是空张量（data_ptr 为空），用 rowPtr 指针占位，
// kernel 不会读取（nnz=0 时无非零元循环）
const void *NonNullPtr(const at::Tensor &tensor, const at::Tensor &fallback)
{
    const void *ptr = tensor.const_data_ptr();
    return ptr != nullptr ? ptr : fallback.const_data_ptr();
}

void RunSpMv(AclSparseContext &context, aclsparseOperation_t op,
    int64_t rows, int64_t cols,
    const at::Tensor &rowPtr, const at::Tensor &colInd, const at::Tensor &values,
    const at::Tensor &x, const at::Tensor &y,
    const std::complex<double> &alpha, const std::complex<double> &beta,
    aclsparseIndexBase_t indexBase)
{
    const aclDataType valueType = ToAclDataType(values.scalar_type());
    const aclDataType outType = ToAclDataType(y.scalar_type());
    const aclDataType computeType = SpmvComputeType(y.scalar_type());
    const HostScalar alphaScalar(alpha, computeType);
    const HostScalar betaScalar(beta, computeType);

    SpMatGuard matA;
    ACLSPARSE_CHECK(aclsparseCreateCsr(matA.address(), rows, cols, values.numel(),
        const_cast<void *>(rowPtr.const_data_ptr()),
        const_cast<void *>(NonNullPtr(colInd, rowPtr)),
        const_cast<void *>(NonNullPtr(values, rowPtr)),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, indexBase, valueType));

    const int64_t xSize = op == ACL_SPARSE_OP_TRANSPOSE ? rows : cols;
    const int64_t ySize = op == ACL_SPARSE_OP_TRANSPOSE ? cols : rows;
    DnVecGuard vecX;
    ACLSPARSE_CHECK(aclsparseCreateDnVec(vecX.address(), xSize,
        const_cast<void *>(x.const_data_ptr()), valueType));
    DnVecGuard vecY;
    ACLSPARSE_CHECK(aclsparseCreateDnVec(vecY.address(), ySize, y.data_ptr(), outType));

    const aclsparseHandle_t handle = context.handle();
    size_t bufferSize = 0;
    ACLSPARSE_CHECK(aclsparseSpMVGetBufferSize(handle, op, alphaScalar.data(),
        matA.get(), vecX.get(), betaScalar.data(), vecY.get(), computeType, kAlg, &bufferSize));
    // 转置 + 16bit 输出需要 fp32 累加 workspace，其余情形为 0
    AclSparseWorkspace workBuffer(context, values, bufferSize);
    ACLSPARSE_CHECK(aclsparseSpMVPreprocess(handle, op, alphaScalar.data(),
        matA.get(), vecX.get(), betaScalar.data(), vecY.get(), computeType, kAlg,
        workBuffer.data()));
    ACLSPARSE_CHECK(aclsparseSpMV(handle, op, alphaScalar.data(),
        matA.get(), vecX.get(), betaScalar.data(), vecY.get(), computeType, kAlg,
        workBuffer.data()));
}

void ValidateSpMvInputs(const at::Tensor &mat, const at::Tensor &vec)
{
    TORCH_CHECK(mat.defined(), "mat must be a defined tensor");
    TORCH_CHECK(mat.layout() == at::kSparseCsr, "mat must use the sparse CSR layout, but got ",
        mat.layout());
    TORCH_CHECK(mat.device().type() == c10::DeviceType::PrivateUse1,
        "mat must be on an Ascend NPU, but got device ", mat.device());
    TORCH_CHECK(mat.dim() == 2, "mat must be a 2-D sparse matrix, but got ", mat.dim(),
        " dimensions");
    TORCH_CHECK(vec.defined(), "vec must be a defined tensor");
    TORCH_CHECK(vec.layout() == at::kStrided, "vec must be a dense tensor, but got layout ",
        vec.layout());
    TORCH_CHECK(vec.device().type() == c10::DeviceType::PrivateUse1,
        "vec must be on an Ascend NPU, but got device ", vec.device());
    TORCH_CHECK(vec.dim() == 1, "vec must be a 1-D dense vector, but got ", vec.dim(),
        " dimensions");
    TORCH_CHECK(mat.device() == vec.device(), "mat and vec must be on the same NPU, but got ",
        mat.device(), " and ", vec.device());
    TORCH_CHECK(mat.scalar_type() == vec.scalar_type(),
        "mat and vec must have the same dtype, but got ", mat.scalar_type(), " and ",
        vec.scalar_type());
    TORCH_CHECK(mat.size(1) == vec.size(0),
        "mat and vec shapes cannot be multiplied (", mat.sizes(), " and ", vec.sizes(), ")");
    TORCH_CHECK(mat.size(0) <= kInt32Max && mat.size(1) <= kInt32Max && mat._nnz() <= kInt32Max,
        "mat exceeds the int32 CSR limit: shape ", mat.sizes(), " with nnz ", mat._nnz(),
        " must stay within ", kInt32Max);
}

// aten::mv 的 SparseCsrPrivateUse1 实现：语义固定为 alpha=1、beta=0。
at::Tensor SpMvCsr(const at::Tensor &mat, const at::Tensor &vec)
{
    ValidateSpMvInputs(mat, vec);
    return RunAclSparse(mat, [&](AclSparseContext &context) -> at::Tensor {
        const at::ScalarType outType = SpmvOutScalarType(mat.scalar_type());
        // beta=0 时 kernel（N 路径）覆写全部输出且不回读 y，empty 即可，省一次置零任务
        at::Tensor y = at::empty({mat.size(0)},
            mat.values().options().layout(at::kStrided).dtype(outType));
        if (mat.size(0) == 0) {
            return y;
        }
        if (mat.size(1) == 0 || mat._nnz() == 0) {
            // alpha=1, beta=0：y = A*x = 0；y 是 empty 分配的未初始化内存，必须清零
            y.zero_();
            return y;
        }

        // PyTorch 默认以 int64 存储 CSR 索引，ACLSparse 要求 int32；窄化在 device 上完成
        at::Tensor rowPtr = mat.crow_indices().to(at::kInt).contiguous();
        at::Tensor colInd = mat.col_indices().to(at::kInt).contiguous();
        at::Tensor values = mat.values().contiguous();
        at::Tensor x = vec.contiguous();

        // 索引窄化经 PyTorch 入队 torch_npu 任务队列；stream(true) 提交队列任务，
        // 保证随后的直接 aclsparse 启动与之有序。这是提交而非设备同步。
        (void)c10_npu::getCurrentNPUStream(mat.device().index()).stream(true);
        context.RecordTensors(rowPtr, colInd, values, x, y);
        RunSpMv(context, ACL_SPARSE_OP_NON_TRANSPOSE, mat.size(0), mat.size(1),
            rowPtr, colInd, values, x, y, {1.0, 0.0}, {0.0, 0.0},
            ACL_SPARSE_INDEX_BASE_ZERO);
        return y;
    });
}

struct SpMvArgs {
    int64_t rows = 0;
    int64_t cols = 0;
    bool isTrans = false;
    aclsparseIndexBase_t indexBase = ACL_SPARSE_INDEX_BASE_ZERO;
};

// spmv_npu / spmv_npu_workspace 共用的参数校验与推导：rows/cols 由
// row_ptr 长度与 x/y 长度按 trans 推导；trans 为 "N"/"T"/"H"，
// "H"（共轭转置）对实数等价于 "T"，复数时先取共轭再走转置路径。
SpMvArgs ValidateSpMvNpuArgs(const at::Tensor &rowPtr, const at::Tensor &colInd,
    const at::Tensor &values, const at::Tensor &x, const at::Tensor &y,
    const std::string &trans, int64_t base)
{
    TORCH_CHECK(rowPtr.device().type() == c10::DeviceType::PrivateUse1 &&
        colInd.device().type() == c10::DeviceType::PrivateUse1 &&
        values.device().type() == c10::DeviceType::PrivateUse1 &&
        x.device().type() == c10::DeviceType::PrivateUse1 &&
        y.device().type() == c10::DeviceType::PrivateUse1,
        "all tensors must be on an Ascend NPU");
    TORCH_CHECK(rowPtr.device() == values.device() && colInd.device() == values.device() &&
        x.device() == values.device() && y.device() == values.device(),
        "all tensors must be on the same NPU, but got ", rowPtr.device(), ", ", colInd.device(),
        ", ", values.device(), ", ", x.device(), ", ", y.device());
    TORCH_CHECK(rowPtr.scalar_type() == at::kInt && colInd.scalar_type() == at::kInt,
        "row_ptr and col_ind must be int32");
    TORCH_CHECK(colInd.numel() == values.numel(), "col_ind and values length mismatch: ",
        colInd.numel(), " vs ", values.numel());
    TORCH_CHECK(x.scalar_type() == values.scalar_type(),
        "x and values must have the same dtype, but got ", x.scalar_type(), " and ",
        values.scalar_type());
    TORCH_CHECK(base == 0 || base == 1, "base must be 0 or 1, but got ", base);
    TORCH_CHECK(trans == "N" || trans == "T" || trans == "H", "trans must be N, T or H, but got ",
        trans);
    TORCH_CHECK(x.dim() == 1 && y.dim() == 1, "x and y must be one-dimensional");
    TORCH_CHECK(rowPtr.numel() >= 1, "row_ptr must not be empty");
    TORCH_CHECK(rowPtr.numel() - 1 <= kInt32Max && x.numel() <= kInt32Max && y.numel() <= kInt32Max &&
        values.numel() <= kInt32Max,
        "SpMV dims and nnz must stay within the int32 CSR limit ", kInt32Max);
    TORCH_CHECK(IsValidSpMvCombo(values.scalar_type(), y.scalar_type()),
        "unsupported SpMV dtype combination: values ", values.scalar_type(),
        " -> y ", y.scalar_type());

    SpMvArgs args;
    args.rows = rowPtr.numel() - 1;
    args.isTrans = trans != "N";
    args.cols = args.isTrans ? y.size(0) : x.size(0);
    const int64_t xSize = args.isTrans ? args.rows : args.cols;
    const int64_t ySize = args.isTrans ? args.cols : args.rows;
    TORCH_CHECK(x.size(0) >= xSize, "x is too small for the requested op: need ", xSize,
        ", got ", x.size(0));
    TORCH_CHECK(y.size(0) >= ySize, "y is too small for the requested op: need ", ySize,
        ", got ", y.size(0));
    args.indexBase = base == 1 ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;
    return args;
}

// 测试入口：显式给定 CSR 三分量与全部 SpMV 参数，y 不被原地修改，返回计算结果。
at::Tensor SpMvNpu(const at::Tensor &rowPtr, const at::Tensor &colInd,
    const at::Tensor &values, const at::Tensor &x, const at::Tensor &y,
    const std::complex<double> &alpha, const std::complex<double> &beta,
    const std::string &trans, int64_t base)
{
    const SpMvArgs args = ValidateSpMvNpuArgs(rowPtr, colInd, values, x, y, trans, base);
    return RunAclSparse(values, [&](AclSparseContext &context) -> at::Tensor {
        at::Tensor rowPtrC = rowPtr.contiguous();
        at::Tensor colIndC = colInd.contiguous();
        at::Tensor valuesC = values.contiguous();
        at::Tensor xC = x.contiguous();
        // beta==0 且非转置时 kernel 不回读 y 且覆写全部输出，empty 即可，省一次 D2D 拷贝；
        // 转置路径按列窗口原子累加，仍需 clone 保留初值语义
        const bool betaIsZero = beta == std::complex<double>(0.0, 0.0);
        at::Tensor yOut = (betaIsZero && !args.isTrans)
            ? at::empty(y.sizes(), y.options())
            : y.clone(at::MemoryFormat::Contiguous);
        if (trans == "H" && valuesC.is_complex()) {
            valuesC = at::resolve_conj(valuesC.conj()).contiguous();
        }

        (void)c10_npu::getCurrentNPUStream(values.device().index()).stream(true);
        context.RecordTensors(rowPtrC, colIndC, valuesC, xC, yOut);
        RunSpMv(context,
            args.isTrans ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE,
            args.rows, args.cols, rowPtrC, colIndC, valuesC, xC, yOut, alpha, beta,
            args.indexBase);
        return yOut;
    });
}

// workspace 查询入口：与 SpMvNpu 相同的参数推导与描述符创建，只做到
// aclsparseSpMVGetBufferSize 为止，返回真实 bufferSize（供 benchmark 的
// adapter.workspace_bytes 扩展点使用）。
// H 的共轭只改数值不改 dtype/shape，不影响 bufferSize，故无需取共轭。
int64_t SpMvNpuWorkspace(const at::Tensor &rowPtr, const at::Tensor &colInd,
    const at::Tensor &values, const at::Tensor &x, const at::Tensor &y,
    const std::complex<double> &alpha, const std::complex<double> &beta,
    const std::string &trans, int64_t base)
{
    const SpMvArgs args = ValidateSpMvNpuArgs(rowPtr, colInd, values, x, y, trans, base);
    return RunAclSparse(values, [&](AclSparseContext &context) -> int64_t {
        at::Tensor rowPtrC = rowPtr.contiguous();
        at::Tensor colIndC = colInd.contiguous();
        at::Tensor valuesC = values.contiguous();
        at::Tensor xC = x.contiguous();
        at::Tensor yC = y.contiguous();
        (void)c10_npu::getCurrentNPUStream(values.device().index()).stream(true);
        context.RecordTensors(rowPtrC, colIndC, valuesC, xC, yC);

        const aclDataType valueType = ToAclDataType(valuesC.scalar_type());
        const aclDataType outType = ToAclDataType(yC.scalar_type());
        const aclDataType computeType = SpmvComputeType(yC.scalar_type());
        const HostScalar alphaScalar(alpha, computeType);
        const HostScalar betaScalar(beta, computeType);

        SpMatGuard matA;
        ACLSPARSE_CHECK(aclsparseCreateCsr(matA.address(), args.rows, args.cols,
            valuesC.numel(),
            const_cast<void *>(rowPtrC.const_data_ptr()),
            const_cast<void *>(NonNullPtr(colIndC, rowPtrC)),
            const_cast<void *>(NonNullPtr(valuesC, rowPtrC)),
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, args.indexBase, valueType));
        DnVecGuard vecX;
        ACLSPARSE_CHECK(aclsparseCreateDnVec(vecX.address(),
            args.isTrans ? args.rows : args.cols,
            const_cast<void *>(xC.const_data_ptr()), valueType));
        DnVecGuard vecY;
        ACLSPARSE_CHECK(aclsparseCreateDnVec(vecY.address(),
            args.isTrans ? args.cols : args.rows, yC.data_ptr(), outType));

        size_t bufferSize = 0;
        ACLSPARSE_CHECK(aclsparseSpMVGetBufferSize(context.handle(),
            args.isTrans ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE,
            alphaScalar.data(), matA.get(), vecX.get(), betaScalar.data(), vecY.get(),
            computeType, kAlg, &bufferSize));
        TORCH_CHECK(bufferSize <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
            "SpMV workspace of ", bufferSize, " bytes exceeds the int64 limit");
        return static_cast<int64_t>(bufferSize);
    });
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mv", &SpMvCsr, "A @ vec for a CSR sparse matrix and a dense vector on NPU");
    m.def("spmv_npu", &SpMvNpu,
        "y = alpha * op(A) * x + beta * y from explicit CSR arrays (test-only entry)");
    m.def("spmv_npu_workspace", &SpMvNpuWorkspace,
        "query the aclsparseSpMV workspace size in bytes (test-only entry)");
}
