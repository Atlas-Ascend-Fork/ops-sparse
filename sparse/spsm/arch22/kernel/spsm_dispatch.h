/**
 * Copyright (c) 2026 liutaw.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for details.
 */
#ifndef SPSM_ARCH22_DISPATCH_H
#define SPSM_ARCH22_DISPATCH_H
#include "kernel_operator.h"
#include "../spsm_tiling_data.h"
namespace spsm_a2 {
// Same-stream consumers only: each producer completes all per-core slots before
// a later kernel reads them. Word 0 remains the preparation/error-code contract.
// 0: always; 1: original fallback; 2: retry finite input after legacy overflow;
// 3: restore the original result if that packed retry exceeds its own range.
__aicore__ inline bool DispatchReady(AscendC::GlobalTensor<int32_t> &status,
    int32_t blocks, int32_t condition)
{
    if (condition == 0) { return true; }
    bool rejected = false, failed = false, retry = false;
    for (int32_t block = 0; block < blocks; ++block) {
        rejected = rejected || status.GetValue(block * kSpsmA2StatusStride + 1) != 0;
        failed = failed || status.GetValue(block * kSpsmA2StatusStride + 2) != 0;
        retry = retry || status.GetValue(block * kSpsmA2StatusStride + 3) != 0;
    }
    if (condition == 1) { return rejected || failed; }
    if (condition == 2) { return !rejected && retry; }
    return !rejected && retry && failed;
}
}
#endif
