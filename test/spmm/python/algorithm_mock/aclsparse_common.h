// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

#pragma once

#include <torch/extension.h>
#include "cann_ops_sparse.h"

#define ACLSPARSE_CHECK(expression)     TORCH_CHECK((expression) == ACL_SPARSE_STATUS_SUCCESS, "mock ACLSparse status")

namespace cann_ops_sparse {
class AclSparseContext {
public:
    aclsparseHandle_t handle() const { return reinterpret_cast<aclsparseHandle_t>(1); }
    template <class... T> void RecordTensors(const T &...) const {}
};
class AclSparseWorkspace {
public:
    AclSparseWorkspace(AclSparseContext &, const at::Tensor &, size_t size) : storage_(size) {}
    void *data() { return storage_.data(); }
private:
    std::vector<uint8_t> storage_;
};
class AclSparseHandleCache {
public:
    static void Clear() {}
};
template <class F> decltype(auto) RunAclSparse(const at::Tensor &, F &&fn)
{
    AclSparseContext context;
    return fn(context);
}
} // namespace cann_ops_sparse

namespace c10_npu {
struct MockStream {
    void *stream(bool) const { return nullptr; }
};
inline MockStream getCurrentNPUStream(int) { return {}; }
} // namespace c10_npu
