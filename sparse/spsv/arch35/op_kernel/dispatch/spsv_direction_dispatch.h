/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef SPSV_DIRECTION_DISPATCH_H_
#define SPSV_DIRECTION_DISPATCH_H_

// CALL(FORWARD, NON_UNIT) expands to a strategy-specific template invocation.
// Keep the runtime-to-specialization mapping identical for every strategy.
#define SPSV_DISPATCH_DIRECTION_AND_DIAGONAL(FWD, NON_UNIT, CALL) \
    do {                                                          \
        if ((FWD) && (NON_UNIT)) {                                \
            CALL(true, true);                                     \
        } else if (FWD) {                                         \
            CALL(true, false);                                    \
        } else if (NON_UNIT) {                                    \
            CALL(false, true);                                    \
        } else {                                                  \
            CALL(false, false);                                   \
        }                                                         \
    } while (0)

#endif // SPSV_DIRECTION_DISPATCH_H_
