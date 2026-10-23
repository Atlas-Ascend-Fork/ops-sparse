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

// cube 算法测试辅助：由 spmv_test.cpp 在公共工具（SpmvRunEnv、Verification 等）
// 定义之后包含，依赖其 include 与类型定义。
#ifndef SPMV_TEST_CUBE_H
#define SPMV_TEST_CUBE_H

/* 上传 y 初值 → 跑一次 cube SpMV → 同步取回结果。两个阶段共用的公共块。 */
static int RunCubeSpmvOnce(SpmvRunEnv &env, aclrtStream stream, float *alpha, float *beta,
                           void *externalBuffer, const std::vector<float> &yInit,
                           std::vector<float> &yResult) {
    int ret = aclrtMemcpy(env.yDevice, env.totalYByteSize, yInit.data(), env.totalYByteSize,
                          ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    const aclsparseStatus_t st = aclsparseSpMV(env.spHandle, ACL_SPARSE_OP_NON_TRANSPOSE, alpha,
        env.matDesc, env.vecXDesc, beta, env.vecYDesc, ACL_FLOAT,
        ACL_SPARSE_SPMV_CSR_ALG_CUBE, externalBuffer);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(env.yHost, env.totalYByteSize, env.yDevice, env.totalYByteSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    yResult.assign((float *)env.yHost, (float *)(env.yHost + env.totalYByteSize));
    return 0;
}

/* 跑一个验证阶段：两趟 SpMV（第二趟命中缓存）。preprocessFirst=true 在循环前探测并写缓存；
 * false 在第二趟前探测（换绑后缓存失效、先就地重探测的场景）。 */
static int RunCubePhase(SpmvRunEnv &env, aclrtStream stream, float *alpha, float *beta,
                        void *externalBuffer, const std::vector<float> &yInit,
                        const std::vector<float> &golden, const MixedToleranceParams &tol,
                        bool preprocessFirst, const char *errTag,
                        float &MARE, float &MERE, std::string *worstInfo) {
    if (preprocessFirst) {
        const aclsparseStatus_t st = aclsparseSpMVPreprocess(env.spHandle,
            ACL_SPARSE_OP_NON_TRANSPOSE, alpha, env.matDesc, env.vecXDesc, beta, env.vecYDesc,
            ACL_FLOAT, ACL_SPARSE_SPMV_CSR_ALG_CUBE, externalBuffer);
        CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    }
    for (int round = 0; round < 2; ++round) {
        if (!preprocessFirst && round == 1) {
            const aclsparseStatus_t st = aclsparseSpMVPreprocess(env.spHandle,
                ACL_SPARSE_OP_NON_TRANSPOSE, alpha, env.matDesc, env.vecXDesc, beta, env.vecYDesc,
                ACL_FLOAT, ACL_SPARSE_SPMV_CSR_ALG_CUBE, externalBuffer);
            CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
        }
        std::vector<float> yResult;
        const int ret = RunCubeSpmvOnce(env, stream, alpha, beta, externalBuffer, yInit, yResult);
        CHECK_RET(ret == 0, return ret);
        const int vret = Verification<float>(golden, yResult, MARE, MERE, tol, worstInfo);
        CHECK_RET(vret == 0,
                  LOG_PRINT("[ERROR] %s round %d mismatch\n", errTag, round);
                  return vret);
    }
    return 0;
}

/* 构造 nnz 相同、行分布均匀的新 CSR 三数组（换绑用）。M/N/nnz 为零时报错返回，不做除法。 */
static int BuildUniformCsr(size_t M, size_t N, size_t nnz,
                           std::vector<int32_t> &rowPtrB, std::vector<int32_t> &colIndB,
                           std::vector<float> &valsB) {
    if (M == 0 || N == 0 || nnz == 0) {
        LOG_PRINT("[ERROR] cube-maxrow-cache invalid dims M=%zu N=%zu nnz=%zu\n", M, N, nnz);
        return -1;
    }
    rowPtrB.assign(M + 1, 0);
    colIndB.assign(nnz, 0);
    valsB.assign(nnz, 0.0f);
    const size_t per = nnz / M, rem = nnz % M;
    size_t k = 0;
    for (size_t i = 0; i < M; ++i) {
        rowPtrB[i] = static_cast<int32_t>(k);
        const size_t cnt = per + (i < rem ? 1 : 0);
        for (size_t t = 0; t < cnt && k < nnz; ++t) {
            colIndB[k] = static_cast<int32_t>((i + t * 11) % N);
            valsB[k] = 0.5f + 0.25f * static_cast<float>(k % 7);
            ++k;
        }
    }
    rowPtrB[M] = static_cast<int32_t>(k);
    return 0;
}

/* 申请设备侧 CSR 三数组并上传，unique_ptr 接管释放。 */
static int UploadCsrB(size_t M, size_t nnz,
                      const std::vector<int32_t> &rowPtrB, const std::vector<int32_t> &colIndB,
                      const std::vector<float> &valsB,
                      std::unique_ptr<void, aclError (*)(void *)> &rppB,
                      std::unique_ptr<void, aclError (*)(void *)> &cipB,
                      std::unique_ptr<void, aclError (*)(void *)> &vdpB) {
    void *rowPtrBDev = nullptr;
    int ret = aclrtMalloc(&rowPtrBDev, (M + 1) * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    rppB.reset(rowPtrBDev);
    void *colIndBDev = nullptr;
    ret = aclrtMalloc(&colIndBDev, nnz * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    cipB.reset(colIndBDev);
    void *valsBDev = nullptr;
    ret = aclrtMalloc(&valsBDev, nnz * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    vdpB.reset(valsBDev);
    ret = aclrtMemcpy(rowPtrBDev, (M + 1) * sizeof(int32_t), rowPtrB.data(),
                      (M + 1) * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(colIndBDev, nnz * sizeof(int32_t), colIndB.data(),
                      nnz * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(valsBDev, nnz * sizeof(float), valsB.data(),
                      nnz * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}

/* GetBufferSize + 按需申请 cube 外部工作区（bufferSize 为 0 时保持 nullptr）。 */
static int AllocCubeWorkspace(SpmvRunEnv &env, float *alpha, float *beta,
                              void *&externalBuffer,
                              std::unique_ptr<void, aclError (*)(void *)> &wsp) {
    size_t bufferSize = 0;
    const aclsparseStatus_t st = aclsparseSpMVGetBufferSize(env.spHandle,
        ACL_SPARSE_OP_NON_TRANSPOSE, alpha, env.matDesc, env.vecXDesc, beta, env.vecYDesc,
        ACL_FLOAT, ACL_SPARSE_SPMV_CSR_ALG_CUBE, &bufferSize);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    if (bufferSize == 0) {
        return 0;
    }
    const int ret = aclrtMalloc(&externalBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    wsp.reset(externalBuffer);
    return 0;
}

/* LOGIC-ASYNC 回归：cube 最长行探测缓存。同一描述符 Preprocess 后连调两次 SpMV
 *（第二次命中缓存），再经 aclsparseCsrSetPointers 换绑 nnz 相同、行分布不同的
 * 新 rowPtr（缓存必须失效，SpMV 内就地重探测），各阶段结果都须与 golden 一致。
 * 每次调用前把 y 重置为初始值，使 beta 路径的 golden 稳定。 */
int TestCubeMaxRowCache(size_t M, size_t N, float sparsity,
                        float alphaVal, float betaVal, aclrtStream stream,
                        std::string *worstInfo = nullptr) {
    std::vector<int32_t> rowPtr, colInd;
    std::vector<float> vals;
    GenerateCsr<float>(M, N, sparsity, rowPtr, colInd, vals);
    std::mt19937 rng(20261016u);
    std::vector<float> xVec, yInit;
    GenerateDenseVector<float>(N, xVec, rng);
    GenerateDenseVector<float>(M, yInit, rng);
    const std::vector<float> golden =
        SpmvCpu<float, float, float>(rowPtr, colInd, vals, xVec, yInit, alphaVal, betaVal);

    SpmvRunEnv env;
    int ret = SetupSpmvRunEnv<float, float>(stream, M, N, rowPtr, colInd, vals, xVec, yInit, 0,
                                            ACL_FLOAT, ACL_FLOAT, env);
    CHECK_RET(ret == 0, return ret);

    float alpha = alphaVal, beta = betaVal;
    void *externalBuffer = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> wsp(nullptr, aclrtFree);
    ret = AllocCubeWorkspace(env, &alpha, &beta, externalBuffer, wsp);
    CHECK_RET(ret == 0, return ret);

    const MixedToleranceParams tol = GetMixedToleranceParams<float>();
    float MARE = 0.0f, MERE = 0.0f;

    // 阶段 1：Preprocess 探测一次并写缓存，随后两次 SpMV（第二次走缓存）
    ret = RunCubePhase(env, stream, &alpha, &beta, externalBuffer, yInit, golden, tol,
                       true, "cube-maxrow-cache", MARE, MERE, worstInfo);
    CHECK_RET(ret == 0, return ret);

    // 阶段 2：换绑 nnz 相同、行分布不同（均匀）的矩阵 B，缓存必须失效；
    // 先裸调 SpMV（就地重探测），再 Preprocess + SpMV（Preprocess 重探测）
    std::vector<int32_t> rowPtrB, colIndB;
    std::vector<float> valsB;
    ret = BuildUniformCsr(M, N, colInd.size(), rowPtrB, colIndB, valsB);
    CHECK_RET(ret == 0, return ret);
    const std::vector<float> goldenB =
        SpmvCpu<float, float, float>(rowPtrB, colIndB, valsB, xVec, yInit, alphaVal, betaVal);

    std::unique_ptr<void, aclError (*)(void *)> rppB(nullptr, aclrtFree);
    std::unique_ptr<void, aclError (*)(void *)> cipB(nullptr, aclrtFree);
    std::unique_ptr<void, aclError (*)(void *)> vdpB(nullptr, aclrtFree);
    ret = UploadCsrB(M, colInd.size(), rowPtrB, colIndB, valsB, rppB, cipB, vdpB);
    CHECK_RET(ret == 0, return ret);
    const aclsparseStatus_t stb = aclsparseCsrSetPointers(env.matDesc, rppB.get(), cipB.get(),
                                                          vdpB.get());
    CHECK_RET(stb == ACL_SPARSE_STATUS_SUCCESS, return stb);

    ret = RunCubePhase(env, stream, &alpha, &beta, externalBuffer, yInit, goldenB, tol,
                       false, "cube-maxrow-cache rebind", MARE, MERE, worstInfo);
    CHECK_RET(ret == 0, return ret);
    std::cout << "====Test Cube maxRowLength cache (preprocess reuse + rebind) case pass!====\n";
    return 0;
}

#endif // SPMV_TEST_CUBE_H
