// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ============================================================================
// aclrt_mock.h — UT 专用 aclrt 运行时桩（测试工程自带桩符号，不链接
// libascendcl）。op_host 的数据校验链（op_host/sparse2dense_tiling.cpp）经 aclrtMemcpyAsync
// （D2H）+ aclrtSynchronizeStream 拉取 primary/secondary 后做 Host 校验；UT 无 NPU/驱动，
// 本桩以「Host 内存直拷」语义承接同一调用面，并留存调用日志供 mock 层语义断言：
//   * 所有拷贝必须为 DEVICE_TO_HOST 方向；
//   * 所有拷贝目的地址必须落在桩分配的校验缓冲（aclrtMallocHost）区间内；
//   * 桩绝不写 matB->dense —— 配合测试哨兵验证「校验失败 matB 不被污染」语义。
#ifndef S2D_UT_ACLRT_MOCK_H_
#define S2D_UT_ACLRT_MOCK_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace s2d_ut_mock {

// 一次 aclrtMemcpyAsync 调用记录
struct MemcpyRecord {
    void *dst;        // 拷贝目的地址
    const void *src;  // 拷贝源地址
    size_t bytes;     // 拷贝字节数
    int kind;         // aclrtMemcpyKind 原值
};

// 一次 aclrtMallocHost 分配记录（UT 内 host 校验缓冲）
struct HostAllocRecord {
    void *ptr;
    size_t bytes;
};

// 清空日志（每个用例前置调用）
void ResetLogs();

const std::vector<MemcpyRecord> &MemcpyLog();
const std::vector<HostAllocRecord> &HostAllocLog();

// 判断 [p, p+len) 是否整体落在某次桩分配的 host 校验缓冲区间内
bool InAnyHostAlloc(const void *p, size_t len);

}  // namespace s2d_ut_mock

#endif  // S2D_UT_ACLRT_MOCK_H_
