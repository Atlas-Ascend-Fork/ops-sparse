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

/*
 * Scatter 算子测试 —— 精度采用 bit-exact 判定。
 *
 * scatter 是非计算类算子（纯数据搬移：y[indices[i]] = values[i]），
 * 按精度标准，非计算类算子要求逐位一致（bit-exact）。
 * 因此 Verification 对 float 做严格逐位匹配（memcmp），任何一位不同即判失败。
 * 同时保留 MARE/MERE 作为信息性指标打印（便于观察），但不参与达标判定。
 *
 * 测试链路：aclsparseCreate -> aclsparseSetStream ->
 *           aclsparseCreateSpVec / aclsparseCreateDnVec ->
 *           aclsparseScatter -> aclsparseDestroy* / aclsparseDestroy。
 *
 * dtype 约束：Host 校验链支持 valueType ∈ {int8, float16, bfloat16,
 *   float32, complex64}（vecX.values 与 vecY 一致），idxType 仅 ACL_SPARSE_INDEX_32I，
 *   idxBase ∈ {0, 1}。float 模板用例覆盖 fp32/base0 基线，"Extended dtype &
 *   idxBase Tests" 段以字节缓冲 + bit-exact golden 覆盖五 dtype ×
 *   base 0/1（含 complex64 2×fp32 交错存储口径），以及 Host 参数校验链错误码用例。
 *
 * 索引约束（互异）：本测试只覆盖互异索引场景，所有用例的 indices 均无重复，确保
 *   kernel 写入的 GM 地址互不冲突，结果具有确定性。这与 cuSPARSE scatter 语义下
 *   "互异索引 → 唯一确定写入" 的约定一致。
 *   含重复索引的场景下，kernel 跨核并发写同一 GM 地址会导致 last-write-wins 顺序
 *   非确定（cuSPARSE 语义下本就未定义），不在本测试覆盖范围内。
 *
 * NaN 处理策略：bit-exact 下 NaN 的位模式可能不同（kernel 不保证 NaN payload 一致）。
 *   对含 NaN 的 value 用例：CPU golden 与 NPU 结果在 NaN 位置只要双方都是 NaN 即视为
 *   通过（isnan 判定），其余元素仍严格 bit-exact。NaN 用例在 worstInfo 中单独标注。
 *   +0.0 / -0.0 仍按严格 bit-exact（区分符号位），因为 scatter 是纯搬移，符号位必须保留。
 *
 * 用例分区：
 *   1. 基础用例：小/中/大 N、nnz=1、nnz=N（全覆盖）、nnz=0（早退）
 *   2. 索引模式：contiguous / reverse / stride / random-unique（均为互异索引）
 *   3. 特殊 value：含 0 / 负数 / 大值 / Inf / NaN
 *   4. 非对齐 nnz（考验 DataCopyPad 边界）
 *   5. 随机参数采样（RunRandomParams，~30 例）
 *   6. 大规模（N=1M+，nnz=1M 随机互异 / 连续）
 *   7. 汇总块（Total/Passed/Failed/Pass rate/失败明细）
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <memory>
#include <sstream>
#include <type_traits>
#include "acl/acl.h"
#include "securec.h"  // memcpy_s, EOK
#include "cann_ops_sparse.h"

// ===================== 工具宏 =====================
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// securec 拷贝封装：bit-exact golden 的固定长度 bit 搬运，失败即致命
//（不可能路径，防静默产出错误 golden）
static void CheckedMemcpy(void *dst, size_t dstBytes, const void *src, size_t count)
{
    if (memcpy_s(dst, dstBytes, src, count) != EOK) {
        throw std::runtime_error("scatter test: memcpy_s failed");
    }
}

// ===================== 索引分布生成 =====================

/*
 * 索引分布模式枚举。
 * scatter 的核心维度是 (N, nnz, 索引分布)，索引分布决定 kernel 内
 * run-length 写入快路径的命中率，是覆盖 kernel 分支的关键。
 */
enum class IndexPattern {
    CONTIGUOUS,    // 连续：0,1,2,...（触发 run-length 快路径）
    REVERSE,       // 逆序：nnz-1,...,1,0（runLen=1 逐元素写）
    STRIDE,        // 等步长：0,stride,2*stride,...（runLen=1）
    RANDOM_UNIQUE  // 随机互异（覆盖一般场景，确定性强）
};

const char *IndexPatternName(IndexPattern p) {
    switch (p) {
        case IndexPattern::CONTIGUOUS:    return "contiguous";
        case IndexPattern::REVERSE:       return "reverse";
        case IndexPattern::STRIDE:        return "stride";
        case IndexPattern::RANDOM_UNIQUE: return "random-unique";
    }
    return "unknown";
}

/*
 * 为 scatter 生成互异索引（升序）：从 [0, N) 中无放回采样 nnz 个。
 */
std::vector<int32_t> GenUniqueIndices(int64_t N, int64_t nnz, std::mt19937 &rng) {
    if (nnz <= 0) return {};
    if (nnz > N) nnz = N;
    std::vector<int32_t> pool(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) pool[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    std::shuffle(pool.begin(), pool.end(), rng);
    std::vector<int32_t> idx(pool.begin(), pool.begin() + nnz);
    std::sort(idx.begin(), idx.end());
    return idx;
}

/*
 * 按指定 IndexPattern 生成索引序列（长度=nnz），所有模式均保证索引互异。
 * - CONTIGUOUS：从 startStart 起连续递增（startStart 用于错开起点）
 * - REVERSE：[startStart+nnz-1, ..., startStart]
 * - STRIDE：startStart, startStart+stride, ...（保证不越界 N）
 * - RANDOM_UNIQUE：随机互异升序
 *
 * 返回的索引顺序即为 kernel 处理顺序（cuSPARSE 模式：不去重不排序）。
 * 所有模式均产生互异索引，确保 kernel 写入地址不冲突，结果确定。
 */
std::vector<int32_t> GenIndicesByPattern(int64_t N, int64_t nnz, IndexPattern pattern,
                                         std::mt19937 &rng, int64_t startStart = 0,
                                         int64_t stride = 8) {
    if (nnz <= 0) return {};
    std::vector<int32_t> idx;
    idx.reserve(static_cast<size_t>(nnz));

    switch (pattern) {
        case IndexPattern::CONTIGUOUS: {
            int64_t start = startStart;
            if (start + nnz > N) start = std::max<int64_t>(0, N - nnz);
            for (int64_t i = 0; i < nnz; ++i) idx.push_back(static_cast<int32_t>(start + i));
            break;
        }
        case IndexPattern::REVERSE: {
            int64_t start = startStart;
            if (start + nnz > N) start = std::max<int64_t>(0, N - nnz);
            for (int64_t i = 0; i < nnz; ++i) idx.push_back(static_cast<int32_t>(start + nnz - 1 - i));
            break;
        }
        case IndexPattern::STRIDE: {
            if (stride < 1) stride = 1;
            int64_t start = startStart;
            // 保证 start + (nnz-1)*stride < N
            int64_t maxStart = N - 1 - (nnz - 1) * stride;
            if (maxStart < 0) { // stride 太大无法容纳 nnz 个，退化为 stride=1
                stride = 1;
                maxStart = N - nnz;
            }
            if (start > maxStart) start = std::max<int64_t>(0, maxStart);
            for (int64_t i = 0; i < nnz; ++i) idx.push_back(static_cast<int32_t>(start + i * stride));
            break;
        }
        case IndexPattern::RANDOM_UNIQUE: {
            idx = GenUniqueIndices(N, nnz, rng);
            break;
        }
    }
    return idx;
}

// ===================== 数据生成（values / yInit） =====================

/*
 * 生成随机 float 值。
 */
template <typename T>
void GenerateValues(size_t nnz, std::vector<T> &vals, std::mt19937 &rng,
                    float lo = -10.0f, float hi = 10.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    vals.assign(nnz, T{});
    for (size_t i = 0; i < nnz; ++i) {
        vals[i] = static_cast<T>(dist(rng));
    }
}

template <typename T>
void GenerateDenseVector(int64_t N, std::vector<T> &y, std::mt19937 &rng,
                         float lo = -5.0f, float hi = 5.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    y.assign(N, T{});
    for (int64_t i = 0; i < N; ++i) {
        y[i] = static_cast<T>(dist(rng));
    }
}

// ===================== CPU 参考实现 =====================

/*
 * scatter CPU golden：y_out[indices[i]] = values[i]（i = 0..nnz-1）。
 * cuSPARSE 模式：不去重不排序。本测试 indices 均互异，无 last-write-wins 歧义。
 * yOut 初始值来自 yInit（模拟 device 上 vecY 的初始内容）。
 */
template <typename ValT>
void ScatterCpu(const std::vector<int32_t> &indices,
                const std::vector<ValT> &values,
                const std::vector<ValT> &yInit,
                std::vector<ValT> &yOut) {
    yOut = yInit;
    for (size_t i = 0; i < indices.size(); ++i) {
        yOut[indices[i]] = values[i];
    }
}

// ===================== bit-exact 精度验证 =====================

/*
 * 非计算类算子：逐位一致（bit-exact）。返回 0=通过，1=失败。
 * 同时计算 MARE/MERE 作为信息性指标打印，
 * 但达标判定严格基于 bit-exact。
 *
 * NaN 处理：双方均为 NaN 视为匹配（isnan 判定），不计入失败；
 *   其余元素（含 +0.0/-0.0）严格 memcmp 逐位比较。
 */
template <typename T>
int32_t Verification(const std::vector<T> &cpuGolden,
                     const std::vector<T> &npuRet,
                     float &MARE, float &MERE,
                     std::string *worstInfo = nullptr) {
    if (npuRet.size() != cpuGolden.size()) {
        std::ostringstream oss;
        oss << "[ERROR] size mismatch: golden=" << cpuGolden.size()
            << " npu=" << npuRet.size();
        if (worstInfo) *worstInfo = oss.str();
        std::cout << oss.str() << "\n";
        return 1;
    }

    int32_t status = 0;
    std::cout << "Verification...\n";
    for (int i = 0; i < std::min(static_cast<int64_t>(npuRet.size()), static_cast<int64_t>(10)); ++i) {
        std::cout << "golden[" << i << "]=" << cpuGolden[i]
                  << " npu_result[" << i << "]=" << npuRet[i] << "\n";
    }

    size_t worstIdx = 0;
    float worstGolden = 0, worstNpu = 0;
    uint32_t bitDiffCount = 0;
    uint32_t nanSkipped = 0;
    bool worstIsNaN = false;

    for (size_t i = 0; i < npuRet.size(); ++i) {
        float npuVal = static_cast<float>(npuRet[i]);
        float cpuVal = static_cast<float>(cpuGolden[i]);

        // 信息性误差度量（NaN 不计入 MARE/MERE）
        if (!std::isnan(npuVal) && !std::isnan(cpuVal)) {
            float aError = std::fabs(npuVal - cpuVal);
            float rError = aError / (std::fabs(cpuVal) + 1e-7f);
            if (rError > MARE) {
                MARE = rError;
                worstIdx = i;
                worstGolden = cpuVal;
                worstNpu = npuVal;
                worstIsNaN = false;
            }
            MERE += rError;
        }

        // ---- bit-exact 判定 ----
        // NaN：双方均为 NaN 视为匹配（payload 可能不同，但语义一致）
        bool goldenIsNaN = std::isnan(cpuVal);
        bool npuIsNaN = std::isnan(npuVal);
        if (goldenIsNaN && npuIsNaN) {
            nanSkipped++;
            continue;
        }
        // 严格逐位比较（区分 +0.0/-0.0/Inf 符号等）
        if (std::memcmp(&npuRet[i], &cpuGolden[i], sizeof(T)) != 0) {
            if (bitDiffCount == 0) {
                worstIdx = i;
                worstGolden = cpuVal;
                worstNpu = npuVal;
                worstIsNaN = (goldenIsNaN != npuIsNaN);
            }
            bitDiffCount++;
            status = 1;
            if (bitDiffCount <= 5) {
                std::cout << "[WARNING] bit-exact FAIL at [" << i
                          << "] golden=" << cpuVal << " npu=" << npuVal << "\n";
            }
        }
    }

    MERE /= (npuRet.size() > 0) ? npuRet.size() : 1;
    std::cout << "Mean Relative Error = " << MERE
              << "; Max Relative Error = " << MARE
              << " (informational only; pass/fail by bit-exact)\n";
    if (nanSkipped > 0) {
        std::cout << "NaN positions skipped (both NaN): " << nanSkipped << "\n";
    }

    if (worstInfo) {
        std::ostringstream oss;
        if (status != 0) {
            oss << "bit-exact FAIL: " << bitDiffCount << "/" << npuRet.size()
                << " elements differ. first mismatch at [" << worstIdx
                << "] golden=" << worstGolden << " npu=" << worstNpu;
            if (worstIsNaN) oss << " (NaN mismatch)";
        } else {
            oss << "bit-exact PASS (all " << npuRet.size() << " elements identical";
            if (nanSkipped > 0) oss << ", " << nanSkipped << " NaN matched";
            oss << ")";
        }
        *worstInfo = oss.str();
    }
    return status;
}

// ===================== 设备资源创建 =====================

int CreateDeviceTensor(uint8_t *hostData, const size_t size, uint8_t **deviceAddr) {
    // size == 0 时不分配（aclrtMalloc 0 字节会返回 ACL_ERROR_INVALID_PARAM），
    // 保持 *deviceAddr = nullptr。scatter 在 nnz==0 时允许 indices/values 为空。
    if (size == 0) {
        *deviceAddr = nullptr;
        return ACL_SUCCESS;
    }
    auto ret = aclrtMalloc((void **)deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }
    return ACL_SUCCESS;
}

// ===================== Init / Finalize =====================

int Init(int32_t deviceId, aclrtStream *stream) {
    auto ret = aclInit(nullptr);
    // aclInit 重复调用返回非 0，测试进程内只 Init 一次；首次失败才报错
    if (ret != ACL_SUCCESS) {
        // 若已初始化，继续；否则报错
    }
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream) {
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

// ===================== 类型映射 =====================

template <typename T>
aclDataType AclTypeOf();
template <>
aclDataType AclTypeOf<float>() { return ACL_FLOAT; }

// ===================== 主测试函数 =====================

/*
 * @brief Scatter 单用例测试（使用 aclsparseScatter 接口）
 *
 * @tparam ValT   值类型（当前仅 float 实例化）
 *
 * @param N        稠密向量 vecY 的长度
 * @param nnz      非零元素个数
 * @param pattern  索引分布模式
 * @param fillVal  value 的填充策略（用于特殊值用例：0/负数/大值/Inf/NaN）
 *                 fillVal == nullptr 时使用随机生成
 * @param MARE     OUT，最大相对误差（信息性）
 * @param MERE     OUT，平均相对误差（信息性）
 * @param stream   ACL stream
 * @param worstInfo OUT，验证详情
 * @return 0=通过，非 0=失败
 */
template <typename ValT>
int Test(const size_t N, const size_t nnz, IndexPattern pattern,
         const std::vector<float> *fillVal,
         float &MARE, float &MERE, aclrtStream stream,
         std::string *worstInfo = nullptr) {
    std::cout << "====Test case: N = " << N
              << " nnz = " << nnz
              << " pattern = " << IndexPatternName(pattern)
              << " type = " << (std::is_same_v<ValT, float> ? "float" : "unknown")
              << "====\n";

    std::mt19937 rng(static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()));

    // 生成索引
    std::vector<int32_t> indices = GenIndicesByPattern(
        static_cast<int64_t>(N), static_cast<int64_t>(nnz), pattern, rng);
    const size_t realNnz = indices.size();

    // 生成 values
    std::vector<ValT> values;
    if (fillVal != nullptr && !fillVal->empty()) {
        // 用 fillVal 填充到 realNnz 长度（循环复用）
        values.assign(realNnz, ValT{});
        for (size_t i = 0; i < realNnz; ++i) {
            values[i] = static_cast<ValT>((*fillVal)[i % fillVal->size()]);
        }
    } else {
        GenerateValues<ValT>(realNnz, values, rng);
    }

    // 生成 yInit
    std::vector<ValT> yInit;
    GenerateDenseVector<ValT>(static_cast<int64_t>(N), yInit, rng);

    // CPU golden
    std::vector<ValT> output_cpu;
    ScatterCpu<ValT>(indices, values, yInit, output_cpu);

    // 设备内存
    size_t idxBytes = realNnz * sizeof(int32_t);
    size_t valBytes = realNnz * sizeof(ValT);
    size_t yBytes   = N * sizeof(ValT);

    uint8_t *idxDev = nullptr, *valDev = nullptr, *yDev = nullptr;

    int aclRet = CreateDeviceTensor((uint8_t *)indices.data(), idxBytes, &idxDev);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, aclError (*)(void *)> idp(idxDev, aclrtFree);

    aclRet = CreateDeviceTensor((uint8_t *)values.data(), valBytes, &valDev);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, aclError (*)(void *)> vdp(valDev, aclrtFree);

    aclRet = CreateDeviceTensor((uint8_t *)yInit.data(), yBytes, &yDev);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, aclError (*)(void *)> ydp(yDev, aclrtFree);

    // handle + set stream
    aclsparseHandle_t spHandle = nullptr;
    aclsparseStatus_t sparseRet = aclsparseCreate(&spHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    sparseRet = aclsparseSetStream(spHandle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 描述符：SpVec（indices+values）+ DnVec（y）
    // idxType 固定 32I，valueType 固定 FLOAT（Host 实现对其它类型返回 NOT_SUPPORTED）
    aclsparseSpVecDescr_t vecXDesc = nullptr;
    sparseRet = aclsparseCreateSpVec(&vecXDesc, static_cast<int64_t>(N),
                                     static_cast<int64_t>(realNnz), idxDev, valDev,
                                     ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                     AclTypeOf<ValT>());
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateSpVec failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    aclsparseDnVecDescr_t vecYDesc = nullptr;
    sparseRet = aclsparseCreateDnVec(&vecYDesc, static_cast<int64_t>(N), yDev,
                                     AclTypeOf<ValT>());
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateDnVec failed. ERROR: %d\n", sparseRet);
              aclsparseDestroySpVec(vecXDesc); return sparseRet);

    // 执行 scatter
    sparseRet = aclsparseScatter(spHandle, vecXDesc, vecYDesc);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseScatter failed. ERROR: %d\n", sparseRet);
              aclsparseDestroySpVec(vecXDesc); aclsparseDestroyDnVec(vecYDesc);
              aclsparseDestroy(spHandle); return sparseRet);

    // 同步 + D2H
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclsparseScatter: aclrtSynchronizeStream failed, ret=%d\n", aclRet);
              aclsparseDestroySpVec(vecXDesc); aclsparseDestroyDnVec(vecYDesc);
              aclsparseDestroy(spHandle); return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    std::vector<ValT> yResult(N);
    aclRet = aclrtMemcpy(yResult.data(), yBytes, yDev, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", aclRet);
              aclsparseDestroySpVec(vecXDesc); aclsparseDestroyDnVec(vecYDesc);
              aclsparseDestroy(spHandle); return aclRet);

    // bit-exact 验证（MARE/MERE 信息性，达标判定严格 bit-exact）
    int verifyRet = Verification<ValT>(output_cpu, yResult, MARE, MERE, worstInfo);

    aclsparseDestroySpVec(vecXDesc);
    aclsparseDestroyDnVec(vecYDesc);
    aclsparseDestroy(spHandle);

    CHECK_RET(verifyRet == 0, LOG_PRINT("====Test case fail!====\n\n"); return verifyRet);
    std::cout << "====Test case pass!====\n\n";
    return 0;
}

// ===================== 测试统计 =====================

struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
    std::vector<std::string> failedCases;
};

/*
 * RunAndTrack：执行单个 Test 用例并统计结果。
 * 适配 scatter 维度：(N, nnz, pattern, fillVal)。
 */
template <typename ValT>
int RunAndTrack(size_t N, size_t nnz, IndexPattern pattern,
                const std::vector<float> *fillVal,
                TestStats &stats, aclrtStream stream, const std::string &tag = "") {
    float MARE = 0, MERE = 0;
    std::string worstInfo;
    int ret = Test<ValT>(N, nnz, pattern, fillVal, MARE, MERE, stream, &worstInfo);

    stats.total++;
    if (ret == 0) {
        stats.passed++;
    } else {
        stats.failed++;
        std::ostringstream oss;
        if (!tag.empty()) oss << "[" << tag << "] ";
        oss << "type=" << (std::is_same_v<ValT, float> ? "float" : "unknown")
            << " N=" << N << " nnz=" << nnz
            << " pattern=" << IndexPatternName(pattern)
            << " MARE=" << MARE << " MERE=" << MERE << "\n"
            << "          " << worstInfo << "\n"
            << "          status=" << ret;
        stats.failedCases.push_back(oss.str());
    }
    return ret;
}

// ===================== 随机测试辅助函数 =====================

/*
 * RunRandomParams：随机采样 (N, nnz, pattern) 执行多例。
 */
template <typename ValT>
void RunRandomParams(int numCases, const std::string &tagPrefix,
                     TestStats &stats, aclrtStream stream) {
    std::cout << "\n======== Float Random Tests ========\n";

    std::mt19937 rng(static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<int64_t> nDist(1024, 65536);
    // 索引模式：偏向 RANDOM_UNIQUE / CONTIGUOUS / STRIDE（覆盖主要分支）
    std::uniform_int_distribution<int> patDist(0, 3);

    for (int i = 0; i < numCases; ++i) {
        size_t N = static_cast<size_t>(nDist(rng));
        std::uniform_int_distribution<int64_t> nnzDist(1, static_cast<int64_t>(N));
        size_t nnz = static_cast<size_t>(nnzDist(rng));
        IndexPattern pat;
        switch (patDist(rng)) {
            case 0: pat = IndexPattern::RANDOM_UNIQUE; break;
            case 1: pat = IndexPattern::CONTIGUOUS; break;
            case 2: pat = IndexPattern::STRIDE; break;
            default: pat = IndexPattern::REVERSE; break;
        }
        std::cout << "--- Random float case " << (i + 1) << "/" << numCases
                  << " (N=" << N << " nnz=" << nnz
                  << " pattern=" << IndexPatternName(pat) << ") ---\n";
        RunAndTrack<ValT>(N, nnz, pat, nullptr, stats, stream,
                          tagPrefix + std::to_string(i + 1));
    }
}

// ===================== 扩展能力测试：五 dtype + idxBase 0/1 =====================

/*
 * 能力扩展 smoke/功能覆盖：
 *   - 五 dtype（int8/float16/bfloat16/float32/complex64）× base 0/1，
 *     字节缓冲 + 纯 bit 搬运 golden（非计算类算子 bit-exact）；
 *   - complex64 建模为 2×fp32 交错存储（8B），验证 T 视角偏移 ×2 口径；
 *   - int8/fp16/bf16 验证非 4B 元素的 MTE3 任意偏移写出与段拷贝路径；
 *   - idxBase=1 验证 run 检测减法路径；
 *   - Host 参数校验链错误码（idxType 64I / 非法 idxBase / dtype 不一致 / 非法 shape /
 *     空句柄描述符）逐项验证。
 */

// float -> dtype bits 转换器（golden 为纯 bit 搬运，转换仅需确定性，无舍入语义要求）
static void FloatToFp32Bytes(float v, uint8_t *dst)
{
    CheckedMemcpy(dst, sizeof(float), &v, sizeof(float));
}

static void FloatToFp16Bytes(float v, uint8_t *dst)
{
    __fp16 h = static_cast<__fp16>(v);  // aarch64 原生半精度
    CheckedMemcpy(dst, sizeof(__fp16), &h, sizeof(__fp16));
}

static void FloatToBf16Bytes(float v, uint8_t *dst)
{
    uint32_t x;
    CheckedMemcpy(&x, sizeof(x), &v, sizeof(x));
    uint16_t bits = static_cast<uint16_t>(x >> 16);  // 截断即可：golden 是 bit 拷贝
    CheckedMemcpy(dst, sizeof(bits), &bits, sizeof(bits));
}

static void FloatToInt8Bytes(float v, uint8_t *dst)
{
    int b = static_cast<int>(v);
    if (b > 127) { b = 127; }
    if (b < -128) { b = -128; }
    int8_t s = static_cast<int8_t>(b);
    CheckedMemcpy(dst, sizeof(int8_t), &s, sizeof(int8_t));
}

static void FloatToC64Bytes(float v, uint8_t *dst)
{
    // complex64 = 2×fp32 交错（re, im），与 Kernel T=float/ELEM_BYTES=8 的存储口径一致
    float re = v;
    float im = -2.0f * v + 0.5f;
    CheckedMemcpy(dst, sizeof(float), &re, sizeof(float));
    CheckedMemcpy(dst + sizeof(float), sizeof(float), &im, sizeof(float));
}

struct ScatterDtypeDesc {
    const char *name;
    aclDataType aclType;
    size_t elemBytes;
    void (*conv)(float, uint8_t *);
};

/*
 * 通用 dtype 单用例：字节缓冲 + bit-exact golden。
 * 索引生成 [0,N) 内目标 targets，描述符 indices = targets + idxBase；
 * golden: golden[target[i]*es .. +es) = values[i*es .. +es)。
 */

// dtype 用例数据：索引 / golden 构造阶段
struct DtypeCaseData {
    std::vector<int32_t> targets;
    std::vector<int32_t> descIdx;
    std::vector<uint8_t> valBytes;
    std::vector<uint8_t> yBytes;
    std::vector<uint8_t> golden;
};

static DtypeCaseData DtypePrepareData(const ScatterDtypeDesc &dt, size_t N, size_t nnz,
                                      IndexPattern pattern, aclsparseIndexBase_t idxBase)
{
    DtypeCaseData d;
    const size_t es = dt.elemBytes;
    std::mt19937 rng(12345u + static_cast<uint32_t>(nnz) + static_cast<uint32_t>(N));
    std::uniform_real_distribution<float> valDist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> yDist(-5.0f, 5.0f);

    d.targets = GenIndicesByPattern(static_cast<int64_t>(N),
                                    static_cast<int64_t>(nnz), pattern, rng);
    const int32_t baseOff = (idxBase == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    d.descIdx.reserve(d.targets.size());
    for (int32_t t : d.targets) {
        d.descIdx.push_back(t + baseOff);
    }
    const size_t realNnz = d.targets.size();

    d.valBytes.assign(realNnz * es, 0);
    d.yBytes.assign(N * es, 0);
    for (size_t i = 0; i < realNnz; ++i) {
        dt.conv(valDist(rng), d.valBytes.data() + i * es);
    }
    for (size_t i = 0; i < N; ++i) {
        dt.conv(yDist(rng), d.yBytes.data() + i * es);
    }
    d.golden = d.yBytes;
    for (size_t i = 0; i < realNnz; ++i) {
        const size_t dstOff = static_cast<size_t>(d.targets[i]) * es;
        CheckedMemcpy(d.golden.data() + dstOff, d.golden.size() - dstOff,
                      d.valBytes.data() + i * es, es);
    }
    return d;
}

// dtype 用例设备缓冲 RAII（aclrtFree 析构释放；声明顺序 idx/val/y，析构序 y/val/idx，
// 寿命覆盖 scatter → 同步 → D2H 全程）
struct DtypeDevBufs {
    std::unique_ptr<void, aclError (*)(void *)> idx{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> val{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> y{nullptr, aclrtFree};
};

// 设备缓冲分配（失败返回错误码，已分配部分随 RAII 释放）
static int DtypeAllocDevice(const ScatterDtypeDesc &dt, size_t N, DtypeCaseData &d,
                            DtypeDevBufs &bufs)
{
    const size_t realNnz = d.targets.size();
    const size_t es = dt.elemBytes;
    uint8_t *idxDev = nullptr, *valDev = nullptr, *yDev = nullptr;
    int aclRet = CreateDeviceTensor(reinterpret_cast<uint8_t *>(d.descIdx.data()),
                                    realNnz * sizeof(int32_t), &idxDev);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    bufs.idx.reset(idxDev);
    aclRet = CreateDeviceTensor(d.valBytes.data(), realNnz * es, &valDev);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    bufs.val.reset(valDev);
    aclRet = CreateDeviceTensor(d.yBytes.data(), N * es, &yDev);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    bufs.y.reset(yDev);
    return 0;
}

// handle 创建/绑流 + 描述符创建 + scatter 提交。
// 返回 0 时 spHandle/vecXDesc/vecYDesc 存活交由调用方销毁；非 0 时已清理本次创建的
// 资源并返回错误码。
static int DtypeSubmitScatter(const ScatterDtypeDesc &dt, size_t N, size_t realNnz,
                              aclsparseIndexBase_t idxBase, const DtypeDevBufs &bufs,
                              aclrtStream stream, aclsparseHandle_t &spHandle,
                              aclsparseSpVecDescr_t &vecXDesc, aclsparseDnVecDescr_t &vecYDesc)
{
    spHandle = nullptr;
    aclsparseStatus_t sparseRet = aclsparseCreate(&spHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet); return sparseRet);
    sparseRet = aclsparseSetStream(spHandle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet);
              aclsparseDestroy(spHandle); return sparseRet);

    vecXDesc = nullptr;
    sparseRet = aclsparseCreateSpVec(&vecXDesc, static_cast<int64_t>(N),
                                     static_cast<int64_t>(realNnz), bufs.idx.get(),
                                     bufs.val.get(), ACL_SPARSE_INDEX_32I, idxBase, dt.aclType);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateSpVec failed. ERROR: %d\n", sparseRet);
              aclsparseDestroy(spHandle); return sparseRet);
    vecYDesc = nullptr;
    sparseRet = aclsparseCreateDnVec(&vecYDesc, static_cast<int64_t>(N), bufs.y.get(), dt.aclType);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateDnVec failed. ERROR: %d\n", sparseRet);
              aclsparseDestroySpVec(vecXDesc); aclsparseDestroy(spHandle); return sparseRet);

    sparseRet = aclsparseScatter(spHandle, vecXDesc, vecYDesc);
    if (sparseRet != ACL_SPARSE_STATUS_SUCCESS) {
        LOG_PRINT("aclsparseScatter failed. ERROR: %d\n", sparseRet);
        aclsparseDestroySpVec(vecXDesc); aclsparseDestroyDnVec(vecYDesc);
        aclsparseDestroy(spHandle);
        return sparseRet;
    }
    return 0;
}

// dtype 用例设备执行阶段（描述符创建 → scatter → 同步 → D2H）。
// 返回 0 且经 result 带回输出；非 0 为错误码。
static int DtypeRunOnDevice(const ScatterDtypeDesc &dt, size_t N, size_t nnz,
                            aclsparseIndexBase_t idxBase, DtypeCaseData &d,
                            aclrtStream stream, std::vector<uint8_t> &result)
{
    const size_t realNnz = d.targets.size();
    const size_t es = dt.elemBytes;

    DtypeDevBufs bufs;
    int aclRet = DtypeAllocDevice(dt, N, d, bufs);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    uint8_t *yDev = static_cast<uint8_t *>(bufs.y.get());

    aclsparseHandle_t spHandle = nullptr;
    aclsparseSpVecDescr_t vecXDesc = nullptr;
    aclsparseDnVecDescr_t vecYDesc = nullptr;
    const int submitRet = DtypeSubmitScatter(dt, N, realNnz, idxBase, bufs, stream,
                                             spHandle, vecXDesc, vecYDesc);
    CHECK_RET(submitRet == 0, return submitRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ret=%d\n", aclRet);
              aclsparseDestroySpVec(vecXDesc); aclsparseDestroyDnVec(vecYDesc);
              aclsparseDestroy(spHandle); return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    result.assign(N * es, 0);
    aclRet = aclrtMemcpy(result.data(), N * es, yDev, N * es, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", aclRet);
              aclsparseDestroySpVec(vecXDesc); aclsparseDestroyDnVec(vecYDesc);
              aclsparseDestroy(spHandle); return aclRet);

    aclsparseDestroySpVec(vecXDesc);
    aclsparseDestroyDnVec(vecYDesc);
    aclsparseDestroy(spHandle);
    return 0;
}

int TestDtype(const ScatterDtypeDesc &dt, size_t N, size_t nnz, IndexPattern pattern,
              aclsparseIndexBase_t idxBase, aclrtStream stream, const std::string &tag)
{
    std::cout << "====Dtype case: " << tag << " type=" << dt.name << " N=" << N
              << " nnz=" << nnz << " pattern=" << IndexPatternName(pattern)
              << " idxBase=" << ((idxBase == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0) << "====\n";

    DtypeCaseData d = DtypePrepareData(dt, N, nnz, pattern, idxBase);

    std::vector<uint8_t> result;
    const int runRet = DtypeRunOnDevice(dt, N, nnz, idxBase, d, stream, result);
    CHECK_RET(runRet == 0, return runRet);

    const size_t es = dt.elemBytes;
    if (std::memcmp(result.data(), d.golden.data(), N * es) != 0) {
        size_t diffCnt = 0, firstDiff = 0;
        for (size_t i = 0; i < N * es; ++i) {
            if (result[i] != d.golden[i]) {
                if (diffCnt == 0) { firstDiff = i; }
                diffCnt++;
            }
        }
        LOG_PRINT("[ERROR] bit-exact FAIL (%s): %zu/%zu bytes differ, first at byte %zu\n",
                  tag.c_str(), diffCnt, N * es, firstDiff);
        LOG_PRINT("====Dtype case fail!====\n\n");
        return 1;
    }
    std::cout << "====Dtype case pass!====\n\n";
    return 0;
}

void RunDtypeAndTrack(const ScatterDtypeDesc &dt, size_t N, size_t nnz, IndexPattern pattern,
                      aclsparseIndexBase_t idxBase, TestStats &stats, aclrtStream stream,
                      const std::string &tag)
{
    int ret = TestDtype(dt, N, nnz, pattern, idxBase, stream, tag);
    stats.total++;
    if (ret == 0) {
        stats.passed++;
    } else {
        stats.failed++;
        stats.failedCases.push_back("[" + std::string(tag) + "] type=" + dt.name +
                                    " N=" + std::to_string(N) + " nnz=" + std::to_string(nnz) +
                                    " base=" + std::to_string(idxBase));
    }
}

// ===================== Host 参数校验链错误码测试 =====================

/*
 * 错误路径逐项验证：
 *   idxType 64I / 非法 idxBase / dtype 不一致 / 不支持 dtype / size>nums /
 *   nnz>0 且 nums=0 / 空描述符 → ACL_SPARSE_STATUS_INVALID_VALUE；
 *   空句柄 → ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR。
 */

// 错误路径共享上下文：设备数据 + handle + 正常 dnVec 描述符（device 缓冲随 ctx 析构释放）
struct ErrorPathCtx {
    std::unique_ptr<void, aclError (*)(void *)> idxDev{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> valDev{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> yDev{nullptr, aclrtFree};
    aclsparseHandle_t spHandle = nullptr;
    aclsparseDnVecDescr_t dnOk = nullptr;
    size_t N = 0;
    size_t nnz = 0;
};

// 错误路径数据与句柄准备（失败时清理已建资源并返回错误码）
static int ErrorPathPrepare(size_t N, size_t nnz, aclrtStream stream, ErrorPathCtx &ctx)
{
    const size_t es = sizeof(float);
    std::vector<int32_t> idx(nnz);
    std::vector<float> val(nnz, 1.5f), y(N, 0.25f);
    for (size_t i = 0; i < nnz; ++i) { idx[i] = static_cast<int32_t>(i * 3); }

    uint8_t *idxDev = nullptr, *valDev = nullptr, *yDev = nullptr;
    int aclRet = CreateDeviceTensor(reinterpret_cast<uint8_t *>(idx.data()), nnz * 4, &idxDev);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    ctx.idxDev.reset(idxDev);
    aclRet = CreateDeviceTensor(reinterpret_cast<uint8_t *>(val.data()), nnz * es, &valDev);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    ctx.valDev.reset(valDev);
    aclRet = CreateDeviceTensor(reinterpret_cast<uint8_t *>(y.data()), N * es, &yDev);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    ctx.yDev.reset(yDev);

    aclsparseHandle_t spHandle = nullptr;
    aclsparseStatus_t sparseRet = aclsparseCreate(&spHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);
    sparseRet = aclsparseSetStream(spHandle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, aclsparseDestroy(spHandle); return sparseRet);
    ctx.spHandle = spHandle;

    aclsparseDnVecDescr_t dnOk = nullptr;
    sparseRet = aclsparseCreateDnVec(&dnOk, static_cast<int64_t>(N), yDev, ACL_FLOAT);
    dnOk = (sparseRet == ACL_SPARSE_STATUS_SUCCESS) ? dnOk : nullptr;
    CHECK_RET(dnOk != nullptr, aclsparseDestroy(spHandle); return 1);
    ctx.dnOk = dnOk;
    return 0;
}

// 错误路径用例判定（三组用例共享）
static void ExpectErrorPath(const char *name, aclsparseStatus_t got, aclsparseStatus_t want,
                            TestStats &stats, int &failed)
{
    bool ok = (got == want);
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name
              << ": got=" << got << " want=" << want << "\n";
    if (!ok) { failed++; }
    stats.total++;
    if (ok) { stats.passed++; } else {
        stats.failed++;
        stats.failedCases.push_back(std::string("[error-path] ") + name);
    }
}

// 错误路径 SpVec 描述符构造（复用共享设备数据指针，构造失败返回 nullptr）
static aclsparseSpVecDescr_t MakeErrSpVec(const ErrorPathCtx &ctx, aclsparseIndexType_t idxType,
                                          aclsparseIndexBase_t idxBase, aclDataType valType,
                                          int64_t size)
{
    aclsparseSpVecDescr_t d = nullptr;
    aclsparseStatus_t r = aclsparseCreateSpVec(&d, size, static_cast<int64_t>(ctx.nnz),
                                               ctx.idxDev.get(), ctx.valDev.get(),
                                               idxType, idxBase, valType);
    return (r == ACL_SPARSE_STATUS_SUCCESS) ? d : nullptr;
}

// 用例准备失败统一清理（dnOk + handle），返回 1 中止
static int ErrorPathAbort(const ErrorPathCtx &ctx)
{
    aclsparseDestroyDnVec(ctx.dnOk);
    aclsparseDestroy(ctx.spHandle);
    return 1;
}

// 错误路径用例组 1：空句柄 / 空 vecX 描述符（null_input）
static void ErrorPathNullCases(const ErrorPathCtx &ctx, TestStats &stats, int &failed)
{
    {   // 空句柄
        aclsparseSpVecDescr_t sp = MakeErrSpVec(ctx, ACL_SPARSE_INDEX_32I,
                                                ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                                static_cast<int64_t>(ctx.N));
        ExpectErrorPath("null-handle", aclsparseScatter(nullptr, sp, ctx.dnOk),
                        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR, stats, failed);
        if (sp != nullptr) { aclsparseDestroySpVec(sp); }
    }
    {   // 空描述符
        ExpectErrorPath("null-vecX", aclsparseScatter(ctx.spHandle, nullptr, ctx.dnOk),
                        ACL_SPARSE_STATUS_INVALID_VALUE, stats, failed);
    }
}

// 错误路径属性负例单条执行：构造 → scatter → 判定 → 销毁；构造失败统一清理返回非 0
static int ErrorPathAttrOne(const ErrorPathCtx &ctx, TestStats &stats, int &failed,
                            const char *name, aclsparseIndexType_t idxType,
                            aclsparseIndexBase_t idxBase, aclDataType valType,
                            aclsparseStatus_t want)
{
    aclsparseSpVecDescr_t sp = MakeErrSpVec(ctx, idxType, idxBase, valType,
                                            static_cast<int64_t>(ctx.N));
    CHECK_RET(sp != nullptr, return ErrorPathAbort(ctx));
    ExpectErrorPath(name, aclsparseScatter(ctx.spHandle, sp, ctx.dnOk), want, stats, failed);
    aclsparseDestroySpVec(sp);
    return 0;
}

// 错误路径用例组 2：描述符属性负例（idxType / idxBase / dtype）。
// 返回非 0 表示用例准备失败
static int ErrorPathAttrCases(const ErrorPathCtx &ctx, TestStats &stats, int &failed)
{
    // idxType = 64I（仅支持 I32）
    CHECK_RET(ErrorPathAttrOne(ctx, stats, failed, "idxType-64I", ACL_SPARSE_INDEX_64I,
                               ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                               ACL_SPARSE_STATUS_INVALID_VALUE) != 0, return 1);
    // 非法 idxBase = 2
    CHECK_RET(ErrorPathAttrOne(ctx, stats, failed, "idxBase-2", ACL_SPARSE_INDEX_32I,
                               static_cast<aclsparseIndexBase_t>(2), ACL_FLOAT,
                               ACL_SPARSE_STATUS_INVALID_VALUE) != 0, return 1);
    // dtype 不一致（values fp16 vs y fp32）
    CHECK_RET(ErrorPathAttrOne(ctx, stats, failed, "dtype-mismatch", ACL_SPARSE_INDEX_32I,
                               ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT16,
                               ACL_SPARSE_STATUS_INVALID_VALUE) != 0, return 1);
    // 不支持 dtype（ACL_INT32）
    CHECK_RET(ErrorPathAttrOne(ctx, stats, failed, "unsupported-dtype-int32", ACL_SPARSE_INDEX_32I,
                               ACL_SPARSE_INDEX_BASE_ZERO, ACL_INT32,
                               ACL_SPARSE_STATUS_INVALID_VALUE) != 0, return 1);
    return 0;
}

// 错误路径用例组 3：shape / 规模负例（size>nums / nnz>0 且 nums=0 / nnz=0 成功早退）
// 返回非 0 表示用例准备失败
// nnz>0 且 vecY nums=0 单条（size=0 仅 nnz=0 合法）。
// 返回非 0 表示用例准备失败
static int ErrorPathNnzGt0Size0Case(const ErrorPathCtx &ctx, TestStats &stats, int &failed)
{
    aclsparseDnVecDescr_t dn0 = nullptr;
    aclsparseStatus_t dn0Ret = aclsparseCreateDnVec(&dn0, 0, nullptr, ACL_FLOAT);
    dn0 = (dn0Ret == ACL_SPARSE_STATUS_SUCCESS) ? dn0 : nullptr;
    if (dn0 == nullptr) {
        std::cout << "  [SKIP] nnz-gt-0-size-0: aclsparseCreateDnVec(0) rejected at create\n";
        return 0;
    }
    aclsparseSpVecDescr_t sp = MakeErrSpVec(ctx, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                            ACL_FLOAT, static_cast<int64_t>(ctx.N));
    if (sp == nullptr) {
        aclsparseDestroyDnVec(dn0);
        return ErrorPathAbort(ctx);
    }
    ExpectErrorPath("nnz-gt-0-size-0", aclsparseScatter(ctx.spHandle, sp, dn0),
                    ACL_SPARSE_STATUS_INVALID_VALUE, stats, failed);
    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn0);
    return 0;
}

// 错误路径用例组 3：shape / 规模负例（size>nums / nnz>0 且 nums=0 / nnz=0 成功早退）
// 返回非 0 表示用例准备失败
static int ErrorPathShapeCases(const ErrorPathCtx &ctx, TestStats &stats, int &failed)
{
    {   // SpVec size > vecY nums
        aclsparseSpVecDescr_t sp = MakeErrSpVec(ctx, ACL_SPARSE_INDEX_32I,
                                                ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT,
                                                static_cast<int64_t>(2 * ctx.N));
        CHECK_RET(sp != nullptr, return ErrorPathAbort(ctx));
        ExpectErrorPath("size-gt-nums", aclsparseScatter(ctx.spHandle, sp, ctx.dnOk),
                        ACL_SPARSE_STATUS_INVALID_VALUE, stats, failed);
        aclsparseDestroySpVec(sp);
    }
    CHECK_RET(ErrorPathNnzGt0Size0Case(ctx, stats, failed) != 0, return 1);
    {   // nnz=0：成功返回且不修改 vecY（数据指针可为 null）
        aclsparseSpVecDescr_t sp = nullptr;
        aclsparseStatus_t sparseRet = aclsparseCreateSpVec(&sp, static_cast<int64_t>(ctx.N), 0,
                                                           nullptr, nullptr,
                                                           ACL_SPARSE_INDEX_32I,
                                                           ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
                  aclsparseDestroyDnVec(ctx.dnOk); aclsparseDestroy(ctx.spHandle); return sparseRet);
        ExpectErrorPath("nnz0-success", aclsparseScatter(ctx.spHandle, sp, ctx.dnOk),
                        ACL_SPARSE_STATUS_SUCCESS, stats, failed);
        aclsparseDestroySpVec(sp);
    }
    return 0;
}

int TestErrorPaths(TestStats &stats, aclrtStream stream)
{
    std::cout << "====Error path tests====\n";
    const size_t N = 16, nnz = 4;
    ErrorPathCtx ctx;
    ctx.N = N;
    ctx.nnz = nnz;
    const int prepRet = ErrorPathPrepare(N, nnz, stream, ctx);
    if (prepRet != 0) {
        return prepRet;
    }

    int failed = 0;
    ErrorPathNullCases(ctx, stats, failed);
    const int attrRet = ErrorPathAttrCases(ctx, stats, failed);
    if (attrRet != 0) {
        return attrRet;
    }
    const int shapeRet = ErrorPathShapeCases(ctx, stats, failed);
    if (shapeRet != 0) {
        return shapeRet;
    }

    aclsparseDestroyDnVec(ctx.dnOk);
    aclsparseDestroy(ctx.spHandle);
    std::cout << "====Error path tests done (failed=" << failed << ")====\n\n";
    return failed == 0 ? 0 : 1;
}

// ===================== 用例套件编排 =====================

/*
 * main 骨架：Init → 分区套件调度 → 汇总打印 → Finalize；各分区用例由 RunXxxSuite 编排。
 */

// 基础用例套件：小/中/大 N、nnz=1、nnz=N（全覆盖）、nnz=0（早退）
static void RunBasicSuite(TestStats &stats, aclrtStream stream)
{
    std::cout << "======== Basic Tests ========\n";
    RunAndTrack<float>(16, 8, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "small-8of16");
    RunAndTrack<float>(16, 12, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "small-12of16");
    RunAndTrack<float>(1024, 1, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nnz1");
    RunAndTrack<float>(2048, 2048, IndexPattern::CONTIGUOUS, nullptr, stats, stream, "full-density-seq");
    RunAndTrack<float>(4096, 4096, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "full-density-rand");
    RunAndTrack<float>(1024, 0, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nnz0-early-return");
    RunAndTrack<float>(65536, 1024, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "medium-1k");
}

// 索引模式覆盖套件（均为互异索引）
static void RunIndexPatternSuite(TestStats &stats, aclrtStream stream)
{
    std::cout << "\n======== Index Pattern Tests ========\n";
    RunAndTrack<float>(4096, 1024, IndexPattern::CONTIGUOUS, nullptr, stats, stream, "contiguous-head");
    RunAndTrack<float>(8192, 4096, IndexPattern::CONTIGUOUS, nullptr, stats, stream, "contiguous-large");
    RunAndTrack<float>(2048, 512, IndexPattern::REVERSE, nullptr, stats, stream, "reverse");
    RunAndTrack<float>(8192, 512, IndexPattern::STRIDE, nullptr, stats, stream, "stride8");
    RunAndTrack<float>(8192, 1000, IndexPattern::STRIDE, nullptr, stats, stream, "stride3");
    RunAndTrack<float>(4096, 2048, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "random-unique");
}

// 特殊 value 套件：含 0/负数/大值/Inf/NaN
static void RunSpecialValueSuite(TestStats &stats, aclrtStream stream)
{
    std::cout << "\n======== Special Values Tests (0/neg/large/Inf/NaN) ========\n";
    {
        // 含 +0/-0/负数/大值/Inf/-Inf/NaN 的混合 value
        std::vector<float> specialVals = {
            0.0f,                                    // +0
            -0.0f,                                   // -0（bit-exact 区分符号位）
            -7.5f,                                   // 负数
            1e30f,                                   // 大值
            std::numeric_limits<float>::infinity(),  // +Inf
            -std::numeric_limits<float>::infinity(), // -Inf
            std::numeric_limits<float>::quiet_NaN(), // NaN（isnan 判定）
            42.0f,
            -1e-10f,
            3.0f
        };
        RunAndTrack<float>(1024, 10, IndexPattern::CONTIGUOUS, &specialVals, stats, stream, "special-mixed");
    }
    {
        // 全 0 值
        std::vector<float> zeroVal = {0.0f};
        RunAndTrack<float>(2048, 256, IndexPattern::RANDOM_UNIQUE, &zeroVal, stats, stream, "all-zero-val");
    }
    {
        // 全负值
        std::vector<float> negVal = {-50.0f, -1.0f, -100.0f, -0.5f};
        RunAndTrack<float>(2048, 256, IndexPattern::RANDOM_UNIQUE, &negVal, stats, stream, "all-negative");
    }
    {
        // 全大值
        std::vector<float> largeVal = {1e30f, -1e30f, 1e35f};
        RunAndTrack<float>(2048, 256, IndexPattern::STRIDE, &largeVal, stats, stream, "all-large");
    }
    {
        // 全 Inf
        std::vector<float> infVal = {std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity()};
        RunAndTrack<float>(2048, 256, IndexPattern::RANDOM_UNIQUE, &infVal, stats, stream, "all-inf");
    }
    {
        // 全 NaN
        std::vector<float> nanVal = {std::numeric_limits<float>::quiet_NaN()};
        RunAndTrack<float>(2048, 256, IndexPattern::RANDOM_UNIQUE, &nanVal, stats, stream, "all-nan");
    }
}

// 非对齐 nnz 套件（考验 DataCopyPad 边界）
static void RunNonAlignedSuite(TestStats &stats, aclrtStream stream)
{
    std::cout << "\n======== Non-aligned nnz Tests ========\n";
    RunAndTrack<float>(4096, 7, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nonalign-7");
    RunAndTrack<float>(4096, 13, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nonalign-13");
    RunAndTrack<float>(4096, 33, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nonalign-33");
    RunAndTrack<float>(4096, 65, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nonalign-65");
    RunAndTrack<float>(4096, 127, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nonalign-127");
}

// 随机参数采样套件
static void RunRandomSuite(TestStats &stats, aclrtStream stream)
{
    RunRandomParams<float>(30, "float-rand-", stats, stream);
}

// 扩展能力套件：五 dtype × base 0/1（bit-exact 字节缓冲）+ Host 校验链错误码
static void RunExtendedDtypeSuite(TestStats &stats, aclrtStream stream)
{
    std::cout << "\n======== Extended dtype & idxBase Tests ========\n";
    const ScatterDtypeDesc dtypes[] = {
        {"float32",   ACL_FLOAT,      4, FloatToFp32Bytes},
        {"float16",   ACL_FLOAT16,    2, FloatToFp16Bytes},
        {"bfloat16",  ACL_BF16,   2, FloatToBf16Bytes},
        {"int8",      ACL_INT8,      1, FloatToInt8Bytes},
        {"complex64", ACL_COMPLEX64,  8, FloatToC64Bytes},
    };
    for (const auto &dt : dtypes) {
        const std::string n = dt.name;
        // base 0 / base 1 各一组（随机互异索引）
        RunDtypeAndTrack(dt, 4096, 1024, IndexPattern::RANDOM_UNIQUE,
                         ACL_SPARSE_INDEX_BASE_ZERO, stats, stream, n + "-base0-rand");
        RunDtypeAndTrack(dt, 4096, 1024, IndexPattern::RANDOM_UNIQUE,
                         ACL_SPARSE_INDEX_BASE_ONE, stats, stream, n + "-base1-rand");
        // 连续段（run 拷贝快路径，base 1 验证 run 检测减法路径）
        RunDtypeAndTrack(dt, 4096, 1000, IndexPattern::CONTIGUOUS,
                         ACL_SPARSE_INDEX_BASE_ONE, stats, stream, n + "-base1-contig");
        // 逆序（逐元素写退化路径）
        RunDtypeAndTrack(dt, 2048, 512, IndexPattern::REVERSE,
                         ACL_SPARSE_INDEX_BASE_ZERO, stats, stream, n + "-base0-reverse");
        // 尾块（非 tile 整数倍）
        RunDtypeAndTrack(dt, 4096, 37, IndexPattern::RANDOM_UNIQUE,
                         ACL_SPARSE_INDEX_BASE_ZERO, stats, stream, n + "-base0-tail37");
        // nnz=0（早退：不触碰 vecY，指针可为 null）
        RunDtypeAndTrack(dt, 64, 0, IndexPattern::RANDOM_UNIQUE,
                         ACL_SPARSE_INDEX_BASE_ZERO, stats, stream, n + "-base0-nnz0");
        // 全密集（nnz = N）
        RunDtypeAndTrack(dt, 64, 64, IndexPattern::CONTIGUOUS,
                         ACL_SPARSE_INDEX_BASE_ONE, stats, stream, n + "-base1-fulldense");
    }
    // 动态大规模（P 场景同源参数规模）
    RunDtypeAndTrack(dtypes[0], 16384, 1024, IndexPattern::RANDOM_UNIQUE,
                     ACL_SPARSE_INDEX_BASE_ZERO, stats, stream, "float32-base0-large");
    RunDtypeAndTrack(dtypes[4], 16384, 1024, IndexPattern::RANDOM_UNIQUE,
                     ACL_SPARSE_INDEX_BASE_ONE, stats, stream, "complex64-base1-large");

    // ---- Host 参数校验链错误码 ----
    std::cout << "\n======== Host Validation / Error Code Tests ========\n";
    TestErrorPaths(stats, stream);
}

// 大规模套件（互异索引）
static void RunLargeScaleSuite(TestStats &stats, aclrtStream stream)
{
    std::cout << "\n======== Large Scale Tests ========\n";
    RunAndTrack<float>(5 * 1024 * 1024, 1024 * 1024, IndexPattern::RANDOM_UNIQUE,
                       nullptr, stats, stream, "large-1M-rand");
    RunAndTrack<float>(4 * 1024 * 1024, 1024 * 1024, IndexPattern::CONTIGUOUS,
                       nullptr, stats, stream, "large-1M-contiguous");
    // 边界回归：fp32 连续 distinct [0,nnz)、N=4M；131072 与 131073 各连跑 3 次，
    // 防范非确定可见性类缺陷单次侥幸通过
    for (int rep = 0; rep < 3; rep++) {
        RunAndTrack<float>(4 * 1024 * 1024, 131072, IndexPattern::CONTIGUOUS,
                           nullptr, stats, stream,
                           ("boundary-131072-r" + std::to_string(rep)).c_str());
        RunAndTrack<float>(4 * 1024 * 1024, 131073, IndexPattern::CONTIGUOUS,
                           nullptr, stats, stream,
                           ("boundary-131073-r" + std::to_string(rep)).c_str());
    }
}

// 汇总打印（Total/Passed/Failed/Pass rate/失败明细）
static void PrintTestSummary(const TestStats &stats)
{
    std::cout << "\n========================================\n";
    std::cout << "              Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "Total cases  : " << stats.total << "\n";
    std::cout << "Passed       : " << stats.passed << "\n";
    std::cout << "Failed       : " << stats.failed << "\n";
    if (stats.total > 0) {
        std::cout << "Pass rate    : "
                  << (100.0 * stats.passed / stats.total) << "%\n";
    }
    if (!stats.failedCases.empty()) {
        std::cout << "----------------------------------------\n";
        std::cout << "Failed case details:\n";
        for (const auto &s : stats.failedCases) {
            std::cout << "  - " << s << "\n";
        }
    }
    std::cout << "========================================\n";
}

// ===================== main =====================

int main(int32_t /*argc*/, char * /*argv*/[]) {
    const int32_t deviceId = 0;  // 使用 device 0
    aclrtStream stream = nullptr;
    int ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    TestStats stats;

    std::cout << "\n"
              << "###########################################################\n"
              << "##            Scatter Test Suite (bit-exact)             ##\n"
              << "###########################################################\n\n";

    // ============ 各分区套件调度 ============
    RunBasicSuite(stats, stream);
    RunIndexPatternSuite(stats, stream);
    RunSpecialValueSuite(stats, stream);
    RunNonAlignedSuite(stats, stream);
    RunRandomSuite(stats, stream);
    RunExtendedDtypeSuite(stats, stream);
    RunLargeScaleSuite(stats, stream);

    PrintTestSummary(stats);

    Finalize(deviceId, stream);
    return (stats.failed == 0) ? 0 : 1;
}
