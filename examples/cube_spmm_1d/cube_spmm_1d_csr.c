/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann_ops_sparse.h"
#include <acl/acl.h>
#include <stdio.h>
#include <stdlib.h>

/* Minimal C consumer. Requires the current public header and installed library.
 * B uses only exactly representable binary16 ones, so no C++ half type is needed. */
int main(void)
{
    aclsparseHandle_t handle = NULL;
    aclsparseCubeSpmm1dPlan_t plan = NULL;
    aclrtStream stream = NULL;
    void *b = NULL, *c = NULL, *workspace = NULL;
    uint16_t* hostB = NULL;
    float* hostC = NULL;
    int result = 1, initialized = 0, deviceSet = 0;
    int64_t offsets[] = {0, 1, 2}, columns[] = {0, 2};
    uint16_t values[] = {0x3c00, 0x4000};
    aclsparseCubeSpmm1dPlanInfo info;
    if (aclInit(NULL) != ACL_SUCCESS) {
        goto cleanup;
    }
    initialized = 1;
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        goto cleanup;
    }
    deviceSet = 1;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS || aclsparseCreate(&handle) != ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseSetStream(handle, stream) != ACL_SPARSE_STATUS_SUCCESS) {
        goto cleanup;
    }
    if (aclsparseCubeSpmm1dCreateCsrPlan(handle, 2, 33, 3, offsets, 3, columns, values, 2, NULL, &plan) !=
            ACL_SPARSE_STATUS_SUCCESS ||
        aclsparseCubeSpmm1dGetPlanInfo(plan, &info) != ACL_SPARSE_STATUS_SUCCESS) {
        goto cleanup;
    }
    hostB = (uint16_t*)calloc(info.bBytes / sizeof(uint16_t), sizeof(uint16_t));
    hostC = (float*)malloc(info.cBytes);
    if (hostB == NULL || hostC == NULL) {
        goto cleanup;
    }
    for (int row = 0; row < info.k; ++row) {
        for (int col = 0; col < info.n; ++col) {
            hostB[(size_t)row * info.paddedN + col] = 0x3c00;
        }
    }
    if (aclrtMalloc(&b, info.bBytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
        aclrtMalloc(&c, info.cBytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
        aclrtMalloc(&workspace, info.workspaceBytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
        aclrtMemcpy(b, info.bBytes, hostB, info.bBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        goto cleanup;
    }
    if (aclsparseCubeSpmm1dExecute(handle, plan, b, info.bBytes, c, info.cBytes, workspace, info.workspaceBytes) !=
            ACL_SPARSE_STATUS_SUCCESS ||
        aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
        aclrtMemcpy(hostC, info.cBytes, c, info.cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        goto cleanup;
    }
    for (int col = 0; col < info.n; ++col) {
        if (hostC[(size_t)col * info.paddedM] != 1.0f || hostC[(size_t)col * info.paddedM + 1] != 2.0f) {
            goto cleanup;
        }
    }
    puts("cube_spmm_1d C CSR example: PASS");
    result = 0;
cleanup:
    if (stream != NULL) {
        (void)aclrtSynchronizeStream(stream);
    }
    if (plan != NULL) {
        (void)aclsparseCubeSpmm1dDestroyPlan(plan);
    }
    if (workspace != NULL) {
        (void)aclrtFree(workspace);
    }
    if (c != NULL) {
        (void)aclrtFree(c);
    }
    if (b != NULL) {
        (void)aclrtFree(b);
    }
    free(hostB);
    free(hostC);
    if (handle != NULL) {
        (void)aclsparseDestroy(handle);
    }
    if (stream != NULL) {
        (void)aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        (void)aclrtResetDevice(0);
    }
    if (initialized) {
        (void)aclFinalize();
    }
    return result;
}
