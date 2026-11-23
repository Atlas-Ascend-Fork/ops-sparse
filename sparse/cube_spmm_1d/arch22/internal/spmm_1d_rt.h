/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSPARSE_SPMM_1D_RT_H_
#define ACLSPARSE_SPMM_1D_RT_H_

#include <cstdint>
#include "acl/acl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RT_DEV_BINARY_MAGIC_ELF 0x43554245

typedef struct tagRtDevBinary {
    uint32_t magic;
    uint32_t version;
    const void *data;
    uint64_t length;
} rtDevBinary_t;

aclError rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle);
aclError rtDevBinaryUnRegister(void *handle);
aclError rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args,
                        uint32_t argsSize, void *smDesc, aclrtStream stream);
aclError rtFunctionRegister(void *binHandle, const void *stubFunc,
                            const char *stubName, const void *devFunc,
                            uint32_t funcMode);
aclError rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len);

#ifdef __cplusplus
}
#endif

#endif  // ACLSPARSE_SPMM_1D_RT_H_
