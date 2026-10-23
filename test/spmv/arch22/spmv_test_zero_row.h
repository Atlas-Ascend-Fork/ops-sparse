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
#ifndef SPMV_TEST_ZERO_ROW_H
#define SPMV_TEST_ZERO_ROW_H

// ===================== TwoProd 零值行漏入快路径回归 =====================

/* 评审反例（P2，评论 id=195311207）：maxAbsX > SPMV_TP_SPLIT_SAFE_MAX 时
 * rowSafeMax 置 0，tile 级（tileMaxV=0<=0）与行级（rowMax=0<=0）判断把零值元/
 * 零值行漏入无守卫快路径；SpmvTwoProdFast(0, 2^116) 中 4097*b=Inf、
 * bh=Inf-Inf=NaN，pErr 污染 accLo，正确结果为 0。
 * 公共 runner：preset CSR + 自定义 x，exactRows 内逐元素精确比较，
 * 其余行按 golden 容差比对。非转置 f32，alpha=1、beta=0、base 0。 */
// exactRows（升序）内逐元素精确比较，其余行按 golden 容差比对；失配打日志返回 1
static int CheckZeroRowResult(const char *tag, const std::vector<float> &result,
                              const std::vector<float> &golden,
                              const std::vector<size_t> &exactRows) {
    size_t nextExact = 0;
    for (size_t i = 0; i < result.size(); ++i) {
        const bool exact = nextExact < exactRows.size() && exactRows[nextExact] == i;
        nextExact += exact ? 1 : 0;
        const float diff = std::fabs(result[i] - golden[i]);
        const bool ok = exact ? (result[i] == golden[i])
                              : (diff <= 1e-5f * std::fabs(golden[i]) + 1e-8f);
        CHECK_RET(ok, LOG_PRINT("[ERROR] %s: row %zu expect %.9e, got %.9e\n", tag, i,
                                golden[i], result[i]);
                  return 1);
    }
    std::cout << "====" << tag << " case pass!====\n";
    return 0;
}

static int RunZeroRowCase(const char *tag, int64_t M, int64_t N,
                          const std::vector<int32_t> &rowPtr,
                          const std::vector<int32_t> &colInd,
                          const std::vector<float> &vals,
                          const std::vector<float> &xVec,
                          const std::vector<float> &golden,
                          const std::vector<size_t> &exactRows,
                          aclrtStream stream) {
    std::vector<float> yVec(M, 0.0f);
    float alpha = 1.0f, beta = 0.0f;
    SpmvRunEnv env;
    int ret = SetupSpmvRunEnv<float, float>(stream, M, N, rowPtr, colInd, vals, xVec, yVec, 0,
                                            ACL_FLOAT, ACL_FLOAT, env);
    CHECK_RET(ret == 0, return ret);
    size_t bufferSize = 0;
    aclsparseStatus_t st = aclsparseSpMVGetBufferSize(env.spHandle, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, env.matDesc, env.vecXDesc, &beta, env.vecYDesc, ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    std::unique_ptr<void, aclError (*)(void *)> wsp(nullptr, aclrtFree);
    if (bufferSize > 0) {
        void *externalBuffer = nullptr;
        ret = aclrtMalloc(&externalBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        wsp.reset(externalBuffer);
    }
    st = aclsparseSpMVPreprocess(env.spHandle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, env.matDesc,
                                 env.vecXDesc, &beta, env.vecYDesc, ACL_FLOAT,
                                 ACL_SPARSE_SPMV_ALG_DEFAULT, wsp.get());
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    st = aclsparseSpMV(env.spHandle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, env.matDesc,
                       env.vecXDesc, &beta, env.vecYDesc, ACL_FLOAT,
                       ACL_SPARSE_SPMV_ALG_DEFAULT, wsp.get());
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<float> result(M, 0.0f);
    ret = aclrtMemcpy(result.data(), M * sizeof(float), env.yDevice, M * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return CheckZeroRowResult(tag, result, golden, exactRows);
}

struct ZeroRowCase {
    const char *tag;
    std::vector<int32_t> rowPtr, colInd;
    std::vector<float> vals, xVec, golden;
    std::vector<size_t> exactRows;
};

// 用例 1-4 构造（shape 均已验证命中 Dot2 短行路径：修复前稳定输出 NaN）
static void BuildZeroRowCases(std::vector<ZeroRowCase> &cases) {
    // 1. 评审原始反例：1×1，val=0，x=2^116
    cases.push_back({"twoprod-zero-row-huge-x-1x1", {0, 1}, {0}, {0.0f},
                     {0x1p+116f}, {0.0f}, {0}});
    // 2. 全零 values tile，单窗口，列 5 的 x=2^116
    {
        ZeroRowCase c{"twoprod-zero-tile-single-win", {}, {}, {}, std::vector<float>(64, 1.0f),
                      std::vector<float>(8, 0.0f), {0, 1, 2, 3, 4, 5, 6, 7}};
        c.rowPtr = {0, 2, 4, 6, 8, 10, 12, 14, 16};
        for (int32_t r = 0; r < 8; ++r) {
            c.colInd.push_back(r == 5 ? 6 : r);
            c.colInd.push_back(5);
            c.vals.insert(c.vals.end(), {0.0f, 0.0f});
        }
        c.xVec[5] = 0x1p+116f;
        cases.push_back(std::move(c));
    }
    // 3. 同 2 但 N=33000 > XWIN_ELEMS(32768)，越限 x 放第二窗口列 32900
    {
        ZeroRowCase c{"twoprod-zero-tile-multi-win", {0, 1, 2, 3, 4}, {100, 32900, 100, 32900},
                      {0.0f, 0.0f, 0.0f, 0.0f}, std::vector<float>(33000, 1.0f),
                      std::vector<float>(4, 0.0f), {0, 1, 2, 3}};
        c.xVec[32900] = 0x1p+116f;
        cases.push_back(std::move(c));
    }
    // 4. 混合 tile：其余行正常小值，行 6 为零值元（val=0）且其列 x=2^116
    {
        ZeroRowCase c{"twoprod-mixed-tile-zero-row", {0, 1, 2, 3, 4, 5, 6, 7, 8},
                      {0, 1, 2, 3, 4, 0, 5, 7},
                      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0x1p+116f, 7.0f, 8.0f},
                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 0.0f, 8.0f}, {6}};
        cases.push_back(std::move(c));
    }
}

// 用例 1-4 登记
static void RunZeroRowRegression(TestStats &stats, aclrtStream stream) {
    std::vector<ZeroRowCase> cases;
    BuildZeroRowCases(cases);
    for (const auto &c : cases) {
        int ret = RunZeroRowCase(c.tag, static_cast<int64_t>(c.golden.size()),
                                 static_cast<int64_t>(c.xVec.size()), c.rowPtr, c.colInd,
                                 c.vals, c.xVec, c.golden, c.exactRows, stream);
        stats.total++;
        if (ret == 0) {
            stats.passed++;
        } else {
            stats.failed++;
            stats.failedCases.push_back(std::string("[") + c.tag + "] exact/tolerance mismatch");
        }
    }
}

#endif // SPMV_TEST_ZERO_ROW_H
