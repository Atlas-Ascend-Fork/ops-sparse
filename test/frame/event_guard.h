/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_FRAME_EVENT_GUARD_H_
#define TEST_FRAME_EVENT_GUARD_H_

// =============================================================================
// Shared aclrtEvent timing RAII used by prune / ltmatmul / matmulSearch test
// wrappers. Extracted to eliminate duplicate EventGuard definitions
// (prune_npu_wrapper.h vs ltmatmul_test_utils.h, transpose_utils.h 同范式).
//
// record/elapsed 增加返回值检查（QA 静态检视：静默忽略失败会产出
// 0.0 假样本）。失败时 stderr 留痕并抛异常——由调用侧测试框架捕获为用例
// 失败（gtest 异常捕获 / runner try-catch），杜绝静默假数据通道。
// 既有用法兼容：record 返回 aclError，返回值忽略不改变合法调用行为；
// elapsedMs 仅在 aclrt 真实失败时行为变化（原静默 0.0，现显式失败）。
// =============================================================================

#include <cstdio>
#include <stdexcept>

#include "acl/acl.h"

namespace sparse_test {

class EventGuard {
public:
    EventGuard() {
        if (aclrtCreateEvent(&ev_) != ACL_SUCCESS) {
            throw std::runtime_error("aclrtCreateEvent failed");
        }
    }
    ~EventGuard() { if (ev_) aclrtDestroyEvent(ev_); }
    EventGuard(const EventGuard&) = delete;
    EventGuard& operator=(const EventGuard&) = delete;
    aclError record(aclrtStream stream) {
        aclError ret = aclrtRecordEvent(ev_, stream);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[EventGuard] aclrtRecordEvent failed, aclErr=%d\n",
                         static_cast<int>(ret));
            throw std::runtime_error("aclrtRecordEvent failed");
        }
        return ret;
    }
    static float elapsedMs(const EventGuard& start, const EventGuard& stop) {
        float ms = 0.0f;
        aclError ret = aclrtEventElapsedTime(&ms, start.ev_, stop.ev_);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[EventGuard] aclrtEventElapsedTime failed, aclErr=%d\n",
                         static_cast<int>(ret));
            throw std::runtime_error("aclrtEventElapsedTime failed");
        }
        return ms;
    }
private:
    aclrtEvent ev_ = nullptr;
};

}  // namespace sparse_test

#endif  // TEST_FRAME_EVENT_GUARD_H_
