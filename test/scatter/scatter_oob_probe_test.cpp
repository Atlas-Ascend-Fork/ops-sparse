/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You should not use this file except in compliance with the License.
 * You may refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*
 * aclsparseScatter 非法索引观测探测（独立进程二进制）
 *
 * 违反 native API 前置条件的索引内容（负索引 / 上溢 size）无承诺行为
 * （aclsparseScatter 原生 C API 未承诺主动拒绝索引内容），本二进制仅作观测记录：
 * 返回状态、同步结果、哨兵变化、越界槽位、有效写落位、恢复控制组全部记录，
 * 零 pass/fail 判定。独立进程运行，设备异常不波及主测试集（进程隔离仅保护
 * 测试进程，不要求非法调用后设备上下文一定可恢复）。
 *
 * 报告：环境变量 SCATTER_OOB_JSON 指定路径（默认 ./scatter_oob_probe_results.json），
 * schema 字段与主 UT CaseRecord 一致（status 固定 OBSERVED）。
 * main 由 test/frame/test_main.cpp 提供（不得定义 main）。
 */

#include <gtest/gtest.h>
#include <acl/acl.h>
#include "scatter_ut_common.h"  // CheckedMemcpy/CurrentGtestName/JsonEscape/DevBuf/SetupAclDevice0
#include "cann_ops_sparse.h"
#include "securec.h"  // memcpy_s, EOK

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ===================== 观测记录 =====================

// 与主 UT CaseRecord 字段一致（status 固定 OBSERVED，无 pass/fail 语义）
struct ProbeRecord {
    std::string caseId;
    std::string gtest;
    std::string status;
    std::string check;
    std::string dtype;
    size_t size = 0;
    size_t nnz = 0;
    std::string detail;
};

static std::vector<ProbeRecord> g_probeRecords;

// ===================== 探测环境与观测 =====================

// 哨兵缓冲（kGuard | vecY | kGuard）+ idx/val 设备缓冲 + 自建 handle/stream/描述符
struct ProbeEnv {
    DevBuf buf;
    DevBuf idxDev;
    DevBuf valDev;
    aclsparseHandle_t h = nullptr;
    aclrtStream s = nullptr;
    aclsparseSpVecDescr_t sp = nullptr;
    aclsparseDnVecDescr_t dn = nullptr;
};

static void ProbeEnvCreate(ProbeEnv &env, const std::vector<int32_t> &idx,
                           const std::vector<float> &val, uint8_t *yPtr, size_t kSize)
{
    const size_t nnz = idx.size();
    env.idxDev.Alloc(idx.data(), nnz * sizeof(int32_t));
    env.valDev.Alloc(val.data(), nnz * sizeof(float));

    ASSERT_EQ(aclsparseCreate(&env.h), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtCreateStream(&env.s), ACL_SUCCESS);
    ASSERT_EQ(aclsparseSetStream(env.h, env.s), ACL_SPARSE_STATUS_SUCCESS);

    ASSERT_EQ(aclsparseCreateSpVec(&env.sp, static_cast<int64_t>(kSize), static_cast<int64_t>(nnz),
                                   env.idxDev.ptr, env.valDev.ptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseCreateDnVec(&env.dn, static_cast<int64_t>(kSize), yPtr, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
}

// 哨兵区间 / 越界槽位 / 有效写入位观测（全记录，无判定语义）
struct ProbeEval {
    size_t preChanged = 0;
    size_t postChanged = 0;
    float oobSlotVal = 0.0f;
    size_t oobSlotDiff = 0;
    bool validWritesOk = true;
};

static ProbeEval ProbeEvaluate(const std::vector<uint8_t> &out, const std::vector<uint8_t> &host,
                               const std::vector<int32_t> &idx, const std::vector<float> &val,
                               size_t kGuard, size_t kSize, size_t kYBytes, size_t kTotal)
{
    ProbeEval e;
    for (size_t i = 0; i < kGuard; i++) {
        e.preChanged += (out[i] != 0x5A);
    }
    for (size_t i = kGuard + kYBytes; i < kTotal; i++) {
        e.postChanged += (out[i] != 0x5A);
    }
    CheckedMemcpy(&e.oobSlotVal, sizeof(float), out.data() + kGuard + kYBytes, sizeof(float));
    e.oobSlotDiff =
        (std::memcmp(out.data() + kGuard + kYBytes, host.data() + kGuard + kYBytes,
                     sizeof(float)) != 0) ? 4 : 0;
    for (size_t i = 0; i < idx.size(); i++) {
        const int64_t pos = static_cast<int64_t>(idx[i]);
        if (pos >= 0 && pos < static_cast<int64_t>(kSize)) {
            float got = 0.0f;
            CheckedMemcpy(&got, sizeof(float), out.data() + kGuard + pos * sizeof(float),
                          sizeof(float));
            if (got != val[i]) {
                e.validWritesOk = false;
            }
        }
    }
    return e;
}

// 恢复控制组观测：同 env 复位、合法索引重跑，状态/同步/落位全部记录不判定
struct ProbeRecovery {
    int st = -1;
    int sync = -1;
    bool placementOk = false;
};

static ProbeRecovery ProbeControlRun(ProbeEnv &env, uint8_t *yPtr, size_t kGuard, size_t kSize)
{
    const std::vector<int32_t> okIdx = {1, 4, 6};
    const std::vector<float> okVal = {101.0f, 102.0f, 103.0f};
    env.idxDev.Free();
    env.valDev.Free();
    env.idxDev.Alloc(okIdx.data(), okIdx.size() * sizeof(int32_t));
    env.valDev.Alloc(okVal.data(), okVal.size() * sizeof(float));
    ProbeRecovery rec;
    aclsparseSpVecDescr_t sp2 = nullptr;
    aclsparseDnVecDescr_t dn2 = nullptr;
    const bool descOk =
        (aclsparseCreateSpVec(&sp2, static_cast<int64_t>(kSize),
                              static_cast<int64_t>(okIdx.size()), env.idxDev.ptr,
                              env.valDev.ptr, ACL_SPARSE_INDEX_32I,
                              ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT) == ACL_SPARSE_STATUS_SUCCESS) &&
        (aclsparseCreateDnVec(&dn2, static_cast<int64_t>(kSize), yPtr, ACL_FLOAT) ==
         ACL_SPARSE_STATUS_SUCCESS);
    if (descOk) {
        rec.st = static_cast<int>(aclsparseScatter(env.h, sp2, dn2));
        rec.sync = static_cast<int>(aclrtSynchronizeStream(env.s));
        const std::vector<uint8_t> out2 = env.buf.Download();
        rec.placementOk = true;
        for (size_t i = 0; i < okIdx.size(); i++) {
            float got = 0.0f;
            CheckedMemcpy(&got, sizeof(float), out2.data() + kGuard + okIdx[i] * sizeof(float),
                          sizeof(float));
            rec.placementOk = rec.placementOk && (got == okVal[i]);
        }
    }
    aclsparseDestroySpVec(sp2);
    aclsparseDestroyDnVec(dn2);
    return rec;
}

// 单次探测：执行 + 全观测记录（printf 打屏 + JSON），返回拼接的观测明细。
// 观测值不汇总进任何布尔判定；调用方唯一断言为记录非空（探测完成）。
static std::string ProbeRun(const char *variant, const std::vector<int32_t> &idx)
{
    const size_t kGuard = 4096;
    const size_t kSize = 8;
    const size_t kYBytes = kSize * sizeof(float);
    const size_t kTotal = 2 * kGuard + kYBytes;
    std::vector<uint8_t> host(kTotal, 0x5A);
    ProbeEnv env;
    env.buf.Alloc(host.data(), kTotal);
    uint8_t *yPtr = env.buf.ptr + kGuard;

    const size_t nnz = idx.size();
    std::vector<float> val(nnz);
    for (size_t i = 0; i < nnz; i++) {
        val[i] = 10.0f + static_cast<float>(i);
    }
    ProbeEnvCreate(env, idx, val, yPtr, kSize);

    const aclsparseStatus_t st = aclsparseScatter(env.h, env.sp, env.dn);
    const aclError sync = aclrtSynchronizeStream(env.s);
    const std::vector<uint8_t> out = env.buf.Download();
    const ProbeEval ev = ProbeEvaluate(out, host, idx, val, kGuard, kSize, kYBytes, kTotal);

    const ProbeRecovery rec = ProbeControlRun(env, yPtr, kGuard, kSize);

    aclsparseDestroySpVec(env.sp);
    aclsparseDestroyDnVec(env.dn);
    aclsparseDestroy(env.h);
    aclrtDestroyStream(env.s);

    std::ostringstream d;
    d << "variant=" << variant << " (native API precondition violation, no promised behavior): "
      << "scatter=" << static_cast<int>(st) << " sync=" << static_cast<int>(sync)
      << "; sentinel_pre_changed=" << ev.preChanged << " post_changed=" << ev.postChanged
      << " oob_slot_bytes=" << ev.oobSlotDiff << "; valid writes landed="
      << (ev.validWritesOk ? "yes" : "NO")
      << "; recovery control run: scatter=" << rec.st << " sync=" << rec.sync
      << " placement=" << (rec.placementOk ? "OK" : "FAIL")
      << " (observed only, not judged)";
    const std::string detail = d.str();
    printf("[L2_005/%s] %s\n", variant, detail.c_str());

    ProbeRecord r;
    r.caseId = "L2_005";
    r.gtest = CurrentGtestName();
    r.status = "OBSERVED";
    r.check = "illegal_input_probe";
    r.dtype = "float32";
    r.size = kSize;
    r.nnz = nnz;
    r.detail = detail;
    g_probeRecords.push_back(r);
    return detail;
}

TEST(IllegalInputProbe, L2_005_Neg1Probe)
{
    // 唯一断言：探测执行完成并产出观测记录；设备行为（状态/同步/哨兵/恢复）零断言
    EXPECT_FALSE(ProbeRun("neg1", {0, 2, -1, 5}).empty());
}

TEST(IllegalInputProbe, L2_005_OverflowProbe)
{
    EXPECT_FALSE(ProbeRun("overflow", {0, 2, 8, 5}).empty());
}

// ===================== 全局环境与 JSON 报告 =====================

class ScatterOobProbeEnv : public ::testing::Environment {
public:
    void SetUp() override
    {
        SetupAclDevice0();  // aclInit 容错 + SetDevice(0)（卡由外部 ASCEND_RT_VISIBLE_DEVICES 钉卡）
    }

    void TearDown() override
    {
        WriteJsonReport();
        aclrtResetDevice(0);
        aclFinalize();
    }

private:
    static void WriteJsonReport()
    {
        const char *envPath = std::getenv("SCATTER_OOB_JSON");
        const std::string path = (envPath != nullptr) ? envPath : "scatter_oob_probe_results.json";
        std::ofstream f(path);
        if (!f.is_open()) {
            fprintf(stderr, "[WARN] cannot open report file: %s\n", path.c_str());
            return;
        }
        f << "{\n  \"suite\": \"scatter_oob_probe\",\n  \"operator\": \"aclsparseScatter\",\n"
          << "  \"arch\": \"arch22/DAV_2201\",\n  \"soc\": \"ascend910b3\",\n  \"cases\": [\n";
        for (size_t i = 0; i < g_probeRecords.size(); ++i) {
            const auto &r = g_probeRecords[i];
            f << "    {\"case_id\": \"" << JsonEscape(r.caseId) << "\", \"gtest\": \""
              << JsonEscape(r.gtest) << "\", \"status\": \"" << r.status
              << "\", \"check\": \"" << JsonEscape(r.check) << "\", \"dtype\": \""
              << JsonEscape(r.dtype) << "\", \"size\": " << r.size << ", \"nnz\": " << r.nnz
              << ", \"detail\": \"" << JsonEscape(r.detail) << "\"}"
              << (i + 1 < g_probeRecords.size() ? "," : "") << "\n";
        }
        f << "  ],\n  \"summary\": {\"observed\": " << g_probeRecords.size() << "}\n}\n";
        f.close();
        printf("\n==== scatter_oob_probe observed=%zu (%s) ====\n", g_probeRecords.size(),
               path.c_str());
    }
};

// 全局 Environment 经静态初始化注册（main 由 test/frame/test_main.cpp 提供，
// AddGlobalTestEnvironment 须在 RUN_ALL_TESTS 前调用，pre-main 注册为 gtest 支持的用法）
static const bool g_probeEnvRegistered = []() {
    ::testing::AddGlobalTestEnvironment(new ScatterOobProbeEnv());
    return true;
}();
