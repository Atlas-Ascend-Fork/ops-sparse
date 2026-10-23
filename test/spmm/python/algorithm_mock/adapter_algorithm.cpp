// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

#include <dlfcn.h>
#include <iostream>
#include <string>
#include <vector>

void *AdapterMockDlsym(void *, const char *);
#define dlsym AdapterMockDlsym
#define TORCH_EXTENSION_NAME spmm_algorithm_mock
#include "../../../../sparse/spmm/torch_extension/csrc/spmm.cpp"
#undef dlsym

namespace {
std::string mode;
std::vector<std::string> phases;

aclsparseStatus_t RecordPhase(const char *phase, aclsparseSpMMAlg_t algorithm)
{
    TORCH_CHECK(algorithm == ACL_SPARSE_SPMM_ALG_DEFAULT, phase, " did not use DEFAULT");
    phases.emplace_back(phase);
    return ACL_SPARSE_STATUS_SUCCESS;
}
} // namespace

aclsparseStatus_t aclsparseCreateCsr(aclsparseSpMatDescr_t *output, int64_t, int64_t, int64_t,
    void *, void *, void *, aclsparseIndexType_t, aclsparseIndexType_t,
    aclsparseIndexBase_t, aclDataType)
{
    *output = reinterpret_cast<aclsparseSpMatDescr_t>(1);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroySpMat(aclsparseConstSpMatDescr_t)
{
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateDnMat(aclsparseDnMatDescr_t *output, int64_t, int64_t,
    int64_t, void *, aclDataType, aclsparseOrder_t)
{
    *output = reinterpret_cast<aclsparseDnMatDescr_t>(1);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroyDnMat(aclsparseConstDnMatDescr_t)
{
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMMGetBufferSize(aclsparseHandle_t, aclsparseOperation_t,
    aclsparseOperation_t, const void *, aclsparseConstSpMatDescr_t,
    aclsparseConstDnMatDescr_t, const void *, aclsparseDnMatDescr_t, aclDataType,
    aclsparseSpMMAlg_t algorithm, size_t *size)
{
    *size = 64;
    return RecordPhase("size", algorithm);
}

aclsparseStatus_t aclsparseSpMMPreprocess(aclsparseHandle_t, aclsparseOperation_t,
    aclsparseOperation_t, const void *, aclsparseConstSpMatDescr_t,
    aclsparseConstDnMatDescr_t, const void *, aclsparseDnMatDescr_t, aclDataType,
    aclsparseSpMMAlg_t algorithm, void *)
{
    return RecordPhase("preprocess", algorithm);
}

aclsparseStatus_t aclsparseSpMM(aclsparseHandle_t, aclsparseOperation_t,
    aclsparseOperation_t, const void *, aclsparseConstSpMatDescr_t,
    aclsparseConstDnMatDescr_t, const void *, aclsparseDnMatDescr_t, aclDataType,
    aclsparseSpMMAlg_t algorithm, void *)
{
    return RecordPhase("execute", algorithm);
}

extern "C" aclsparseStatus_t aclsparseSpMMPrivateOutOfPlace(aclsparseHandle_t,
    aclsparseOperation_t, aclsparseOperation_t, const void *, aclsparseConstSpMatDescr_t,
    aclsparseConstDnMatDescr_t, const void *, aclsparseConstDnMatDescr_t,
    aclsparseDnMatDescr_t, aclDataType, aclsparseSpMMAlg_t algorithm, void *)
{
    RecordPhase("private-out", algorithm);
    return mode == "private-fallback" ? ACL_SPARSE_STATUS_NOT_SUPPORTED : ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t MockNativeOutOfPlace(aclsparseHandle_t, aclsparseOperation_t,
    aclsparseOperation_t, const void *, aclsparseConstSpMatDescr_t,
    aclsparseConstDnMatDescr_t, const void *, aclsparseDnMatDescr_t, void *,
    aclDataType, aclsparseSpMMAlg_t algorithm, void *)
{
    return RecordPhase("native-out", algorithm);
}

void *AdapterMockDlsym(void *, const char *symbol)
{
    const std::string name(symbol);
    if (mode == "native-out" && name == "aclsparseSpMMOutOfPlaceInternal") {
        return reinterpret_cast<void *>(&MockNativeOutOfPlace);
    }
    if ((mode == "private-out" || mode == "private-fallback") &&
        name == "aclsparseSpMMPrivateOutOfPlace") {
        return reinterpret_cast<void *>(&aclsparseSpMMPrivateOutOfPlace);
    }
    return nullptr;
}

static at::ScalarType StorageType(const std::string &name)
{
    if (name == "float16") return at::kHalf;
    if (name == "bfloat16") return at::kBFloat16;
    if (name == "complex64") return at::kComplexFloat;
    return at::kFloat;
}

static void RunAdapter(const std::string &dtype)
{
    const auto options = at::TensorOptions().dtype(StorageType(dtype));
    const auto crow = at::tensor({0, 1, 2}, at::TensorOptions().dtype(at::kInt));
    const auto col = at::tensor({0, 2}, at::TensorOptions().dtype(at::kInt));
    const auto values = at::ones({2}, options);
    const auto sparse = at::sparse_csr_tensor(crow, col, values, {2, 3}, options);
    const auto dense = at::ones({3, 2}, options);
    SpmmRequest request{at::ones({2, 2}, options), sparse, dense, at::empty({2, 2}, options),
        {{1.0f, 0.0f}}, {{1.0f, 0.0f}}, ToAclDataType(StorageType(dtype))};
    AclSparseContext context;
    LaunchSpmm(context, {crow, col, values}, ResolveDenseLayout(dense), request, mode != "inplace");
}

int main(int argc, char **argv)
{
    try {
        TORCH_CHECK(argc == 3, "expected mode and dtype");
        mode = argv[1];
        RunAdapter(argv[2]);
        std::vector<std::string> expected{"size", "preprocess"};
        if (mode == "inplace") {
            expected.emplace_back("execute");
        } else if (mode == "native-out") {
            expected.emplace_back("native-out");
        } else {
            expected.emplace_back("private-out");
            if (mode == "private-fallback") expected.emplace_back("execute");
        }
        TORCH_CHECK(phases == expected, "native phase order changed");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
