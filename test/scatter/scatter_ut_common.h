/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You should not use this file except in compliance with the License.
 * You may refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*
 * scatter 测试共用脚手架
 *
 * 主功能 gtest UT（scatter_ut_test.cpp）与非法输入观测探测二进制
 * （scatter_oob_probe_test.cpp）共用的测试工具：securec 安全拷贝封装、
 * 当前 gtest 用例名、JSON 字符串转义、简单设备缓冲（含空缓冲语义）、
 * ACL 设备初始化。卡选择由外部 ASCEND_RT_VISIBLE_DEVICES 指定。
 */

#ifndef TEST_SCATTER_SCATTER_UT_COMMON_H_
#define TEST_SCATTER_SCATTER_UT_COMMON_H_

#include <gtest/gtest.h>
#include <acl/acl.h>
#include "securec.h"  // memcpy_s, EOK

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// securec 拷贝封装：固定长度 bit 搬运，失败即致命
//（不可能路径，防静默产出错误数据）
inline void CheckedMemcpy(void *dst, size_t dstBytes, const void *src, size_t count)
{
    if (memcpy_s(dst, dstBytes, src, count) != EOK) {
        throw std::runtime_error("scatter test: memcpy_s failed");
    }
}

// 当前 gtest 用例名（suite.name），非 gtest 上下文（Environment 阶段）返回 "(env)"
inline std::string CurrentGtestName()
{
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    if (info == nullptr) {
        return "(env)";
    }
    return std::string(info->test_suite_name()) + "." + info->name();
}

// JSON 字符串字面量转义（引号/反斜杠/控制字符）
inline std::string JsonEscape(const std::string &s)
{
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    // \u%04x 控制字符转义（ostringstream 格式化）
                    std::ostringstream b;
                    b << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                      << static_cast<unsigned int>(static_cast<unsigned char>(c));
                    o += b.str();
                } else {
                    o += c;
                }
        }
    }
    return o;
}

// 简单设备缓冲：一次分配即上传；n==0 时保持空指针（nnz=0 / size=0 合法）；
// Upload 复用已分配缓冲重传；Download 整缓冲回读
struct DevBuf {
    uint8_t *ptr = nullptr;
    size_t bytes = 0;

    void Alloc(const void *host, size_t n)
    {
        Free();
        bytes = n;
        if (n == 0) {
            ptr = nullptr;  // nnz=0 / size=0 合法空指针
            return;
        }
        void *p = nullptr;
        aclError r = aclrtMalloc(&p, n, ACL_MEM_MALLOC_HUGE_FIRST);
        if (r != ACL_SUCCESS) {
            fprintf(stderr, "[FATAL] aclrtMalloc(%zu) failed: %d\n", n, (int)r);
            std::exit(1);
        }
        r = aclrtMemcpy(p, n, host, n, ACL_MEMCPY_HOST_TO_DEVICE);
        if (r != ACL_SUCCESS) {
            fprintf(stderr, "[FATAL] aclrtMemcpy H2D failed: %d\n", (int)r);
            std::exit(1);
        }
        ptr = static_cast<uint8_t *>(p);
    }

    void Upload(const void *host, size_t n)
    {
        if (ptr == nullptr || n == 0) {
            return;
        }
        const aclError r = aclrtMemcpy(ptr, n, host, n, ACL_MEMCPY_HOST_TO_DEVICE);
        if (r != ACL_SUCCESS) {
            fprintf(stderr, "[FATAL] aclrtMemcpy H2D failed: %d\n", (int)r);
            std::exit(1);
        }
    }

    std::vector<uint8_t> Download() const
    {
        std::vector<uint8_t> out(bytes);
        if (ptr != nullptr && bytes > 0) {
            const aclError r = aclrtMemcpy(out.data(), bytes, ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (r != ACL_SUCCESS) {
                fprintf(stderr, "[FATAL] aclrtMemcpy D2H failed: %d\n", (int)r);
                std::exit(1);
            }
        }
        return out;
    }

    void Free()
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
            ptr = nullptr;
        }
        bytes = 0;
    }

    ~DevBuf()
    {
        Free();
    }
};

// 进程内 ACL 设备初始化：aclInit 重复初始化容忍（WARN，以 SetDevice 为准出闸），
// aclrtSetDevice(0) 失败即致命退出。设备号固定 0（卡选择由外部
// ASCEND_RT_VISIBLE_DEVICES 钉卡）。
inline void SetupAclDevice0()
{
    aclError r = aclInit(nullptr);
    if (r != ACL_SUCCESS) {
        fprintf(stderr, "[WARN] aclInit returned %d (repeat-init tolerated)\n", (int)r);
    }
    r = aclrtSetDevice(0);
    if (r != ACL_SUCCESS) {
        fprintf(stderr, "[FATAL] aclrtSetDevice failed: %d\n", (int)r);
        std::exit(1);
    }
}

#endif  // TEST_SCATTER_SCATTER_UT_COMMON_H_
