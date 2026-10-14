/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file scatter.cpp
 * \brief aten::index_copy_ 的 NPU C++ kernel wrapper：Tensor.index_copy_(0, index, source) → aclsparseScatter。
 *
 * 语义对齐 PyTorch index_copy_：重复索引 last-write-wins、原地写 self、越界确定报错。
 * 适配约束：
 *   - 仅支持 dim == 0 且 self 为一维连续稠密张量（DnVec 语义）；
 *   - index 仅支持 int32（aclsparse I32 契约，int64 显式报错，不做隐式转换）；
 *   - source 与 self dtype 须一致，且 ∈ {int8, float16, bfloat16, float32, complex64}；
 *   - self / index / source 须位于同一 NPU device，且为非重叠连续布局；
 *   - 越界 index（不在 [0, self.numel())）返回确定错误，不依赖 Device 侧异步错误协议；
 *   - 任何不支持组合显式报错，禁止 CPU fallback。
 *
 * 句柄、stream 与设备上下文由公共层 AclSparseContext 管理（按 device + stream 缓存并
 * 串行化，调用结束自动释放锁与上下文）。越界校验对同一未变更 index 张量做记忆化，
 * 消除稳态重复调用的逐次 Device 同步；缓存键含 version counter（任一侧写入即失效），
 * 缓存持有 source Tensor 强引用排除 data_ptr 复用键冲突。inference tensor（在
 * torch.inference_mode 内创建）版本计数器禁用且原地写入不 bump 版本，无法安全记忆化，
 * 绕过缓存逐次同步校验（见 CheckIndexBounds）。经 ATen 之外的裸内存别名
 * 写入（如 frombuffer 二次包装）不受 version counter 约束，属缓存类机制一致接受的
 * 边界外假设，在算子 README 限制说明中登记。
 */

#include "aclsparse_common.h"

#include <torch/extension.h>

#include <mutex>

namespace {

// ---- 记忆化缓存键：data_ptr + numel + dtype + device + version counter ----
struct TensorKey {
    const void *ptr = nullptr;
    int64_t numel = -1;
    at::ScalarType dtype = at::kFloat;
    c10::Device device{c10::DeviceType::PrivateUse1, 0};
    uint32_t version = 0;
};

TensorKey MakeKey(const at::Tensor &t)
{
    return TensorKey{t.const_data_ptr(), t.numel(), t.scalar_type(), t.device(),
                     t.unsafeGetTensorImpl()->version_counter().current_version()};
}

bool KeyEqual(const TensorKey &a, const TensorKey &b)
{
    return a.ptr == b.ptr && a.numel == b.numel && a.dtype == b.dtype &&
           a.device == b.device && a.version == b.version;
}

// dtype 映射：ATen dtype -> aclDataType（仅五种支持的 values dtype）
bool ToAclDtype(at::ScalarType t, aclDataType &out)
{
    switch (t) {
        case at::kChar:
            out = ACL_INT8;
            return true;
        case at::kHalf:
            out = ACL_FLOAT16;
            return true;
        case at::kBFloat16:
            out = ACL_BF16;
            return true;
        case at::kFloat:
            out = ACL_FLOAT;
            return true;
        case at::kComplexFloat:
            out = ACL_COMPLEX64;
            return true;
        default:
            return false;
    }
}

// 单条目越界校验记忆化：同一（未变更）index 张量重复调用时复用上次 min/max，
// 消除逐次 Device 同步；miss 路径执行两次 .item() 同步校验，语义（越界确定报错）不变。
struct BoundsCache {
    std::mutex mu;
    bool valid = false;
    at::Tensor source;  // 持有存活，防 data_ptr 复用（见 TensorKey 注释）
    TensorKey key{};
    int64_t minv = 0;
    int64_t maxv = 0;
};

// 需在正确的 NPU 设备上下文中调用（.min()/.max() 为同步 Device 读）。
void CheckIndexBounds(const at::Tensor &index, int64_t size)
{
    static BoundsCache cache;
    // inference tensor（inference_mode 内创建）版本计数器禁用：current_version() 抛
    // 异常，且原地写入不 bump 版本、缓存无法失效，故绕过记忆化逐次同步校验（语义不变）
    if (!index.unsafeGetTensorImpl()->version_counter().enabled()) {
        const int64_t idxMin = index.min().item<int64_t>();
        const int64_t idxMax = index.max().item<int64_t>();
        TORCH_CHECK(idxMin >= 0 && idxMax < size,
                    "index_copy_: index out of range [0, ", size, ") for NPU aclsparseScatter (min=",
                    idxMin, ", max=", idxMax, ")");
        return;
    }
    TensorKey key = MakeKey(index);
    int64_t idxMin = 0;
    int64_t idxMax = 0;
    {
        std::lock_guard<std::mutex> lock(cache.mu);
        if (cache.valid && KeyEqual(cache.key, key)) {
            idxMin = cache.minv;
            idxMax = cache.maxv;
        } else {
            idxMin = index.min().item<int64_t>();
            idxMax = index.max().item<int64_t>();
            cache.valid = true;
            cache.source = index;
            cache.key = key;
            cache.minv = idxMin;
            cache.maxv = idxMax;
        }
    }
    TORCH_CHECK(idxMin >= 0 && idxMax < size,
                "index_copy_: index out of range [0, ", size, ") for NPU aclsparseScatter (min=",
                idxMin, ", max=", idxMax, ")");
}

// 参数校验链（不支持组合显式报错，禁止 CPU fallback）。经 aclType 带回 dtype 映射结果。
void ValidateIndexCopyInputs(const at::Tensor &self, int64_t dim, const at::Tensor &index,
                             const at::Tensor &source, aclDataType &aclType)
{
    TORCH_CHECK(dim == 0, "aclsparseScatter adapter only supports index_copy_ with dim == 0, got dim = ",
                dim);
    TORCH_CHECK(self.dim() == 1, "aclsparseScatter adapter only supports 1-D self (DnVec), got dim = ",
                self.dim());
    TORCH_CHECK(index.scalar_type() == at::kInt,
                "aclsparseScatter adapter only supports int32 index (aclsparse I32 contract), got ",
                index.scalar_type());
    TORCH_CHECK(index.dim() == 1, "index must be 1-D, got dim = ", index.dim());
    TORCH_CHECK(self.is_privateuseone() && index.is_privateuseone() && source.is_privateuseone(),
                "self / index / source must be NPU tensors");
    // 同 device 一致性校验：aclsparse 描述符无 device 元数据，落在适配层实施
    const auto selfDevice = self.device();
    TORCH_CHECK(index.device() == selfDevice && source.device() == selfDevice,
                "self / index / source must be on the same NPU device, got ",
                selfDevice, ", ", index.device(), ", ", source.device());
    TORCH_CHECK(self.scalar_type() == source.scalar_type(),
                "aclsparseScatter requires self.dtype == source.dtype, got ",
                self.scalar_type(), " vs ", source.scalar_type());
    TORCH_CHECK(self.layout() == c10::kStrided && index.layout() == c10::kStrided &&
                source.layout() == c10::kStrided,
                "aclsparseScatter adapter requires strided (dense) layout tensors");
    TORCH_CHECK(self.is_contiguous() && index.is_contiguous() && source.is_contiguous(),
                "aclsparseScatter adapter requires contiguous tensors");
    TORCH_CHECK(source.dim() == 1, "aclsparseScatter adapter only supports 1-D source, got dim = ",
                source.dim());
    TORCH_CHECK(index.numel() == source.numel(),
                "index.numel() (", index.numel(), ") must equal source.numel() (", source.numel(), ")");

    // Reject storage overlap before launch (matches aten::index_copy_ contract).
    // Contiguous offset views like source=self[1:] pass pointer-equality checks
    // but still race when the kernel reads source while writing self.
    at::assert_no_internal_overlap(self);
    at::assert_no_overlap(self, source);
    at::assert_no_overlap(self, index);

    aclType = ACL_FLOAT;
    TORCH_CHECK(ToAclDtype(self.scalar_type(), aclType),
                "aclsparseScatter unsupported dtype: ", self.scalar_type());
}

// 越界确定报错 + 描述符转换 + aclsparseScatter 执行（异步，绑定当前 NPU stream）。
// 须在 RunAclSparse 的正确 NPU 设备上下文中调用
at::Tensor ScatterViaAclSparse(cann_ops_sparse::AclSparseContext &context, const at::Tensor &self,
                               const at::Tensor &index, const at::Tensor &source,
                               int64_t size, int64_t nnz, aclDataType aclType)
{
    // 越界确定报错（对齐 PyTorch index_copy_；需设备上下文，故置于 guard 内）
    if (nnz > 0) {
        TORCH_CHECK(size > 0, "index_copy_: non-empty index on empty self");
        CheckIndexBounds(index, size);
    }

    // ---- 描述符转换 + aclsparseScatter（异步，绑定当前 NPU stream）----
    aclsparseSpVecDescr_t vecX = nullptr;
    aclsparseDnVecDescr_t vecY = nullptr;
    // 描述符销毁忽略返回码：销毁失败无需掩盖 scatter 结果或原始错误（对齐公共层风格）
    auto cleanup = [&]() {
        if (vecX != nullptr) {
            (void)aclsparseDestroySpVec(vecX);
        }
        if (vecY != nullptr) {
            (void)aclsparseDestroyDnVec(vecY);
        }
    };

    ACLSPARSE_CHECK(aclsparseCreateSpVec(
        &vecX, size, nnz,
        const_cast<void *>(index.const_data_ptr()),
        const_cast<void *>(source.const_data_ptr()),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, aclType));
    aclsparseStatus_t dnRet = aclsparseCreateDnVec(&vecY, size, self.data_ptr(), aclType);
    if (dnRet != ACL_SPARSE_STATUS_SUCCESS) {
        if (vecX != nullptr) {
            (void)aclsparseDestroySpVec(vecX);
        }
        TORCH_CHECK(false, "aclsparseCreateDnVec failed with ACLSparse status ",
                    static_cast<int>(dnRet));
    }

    aclsparseStatus_t ret = aclsparseScatter(context.handle(), vecX, vecY);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        cleanup();
        TORCH_CHECK(false, "aclsparseScatter failed with ACLSparse status ",
                    static_cast<int>(ret));
    }
    cleanup();

    // 输入 storage 记录到执行 stream，防止异步 kernel 未完成前被缓存分配器复用
    context.RecordTensors(self, index, source);
    return self;
}

} // namespace

/*!
 * \brief aten::index_copy_ 的 NPU 实现：self.index_copy_(0, index, source) → aclsparseScatter。
 *
 * 语义：self[index[i]] = source[i]（原地，无 CPU fallback）。SpVec 索引基取 0
 * （ATen index_copy_ 为 0 基语义）。
 */
at::Tensor IndexCopyNpu(at::Tensor &self, int64_t dim, const at::Tensor &index,
                        const at::Tensor &source)
{
    // ---- 参数校验链（见 ValidateIndexCopyInputs）----
    aclDataType aclType = ACL_FLOAT;
    ValidateIndexCopyInputs(self, dim, index, source, aclType);

    const int64_t size = self.numel();
    const int64_t nnz = index.numel();

    return cann_ops_sparse::RunAclSparse(self,
        [&](cann_ops_sparse::AclSparseContext &context) {
            // 越界确定报错 + 描述符转换 + aclsparseScatter（见 ScatterViaAclSparse）
            return ScatterViaAclSparse(context, self, index, source, size, nnz, aclType);
        });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("index_copy_", &IndexCopyNpu);
}
