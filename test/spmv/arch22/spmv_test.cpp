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

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#include <cmath>
#include <string>
#include <memory>
#include <type_traits>
#include <cstdint>
#include <cstring>
#include <limits>
#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "../../frame/presmoke.h"
#include "tiling/platform/platform_ascendc.h"

// host 侧 half/bf16 别名：kernel 侧为 AscendC::half / AscendC::bfloat16_t
// aarch64 host 使用原生 __fp16；其余 host（如 x86_64，GCC 11 无 __fp16/_Float16）使用
// 软件实现的 2 字节 IEEE754 half，仅用于 host 侧数据构造/搬运，位布局与 __fp16 一致
#if defined(__aarch64__) || (defined(__ARM_ARCH) && defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC))
#include <arm_fp16.h>
using half = __fp16;
#else
struct half {
    uint16_t bits;
    half() : bits(0) {}
    half(float f) { *this = f; }
    operator float() const {
        uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
        uint32_t exp = (bits >> 10) & 0x1Fu;
        uint32_t mant = bits & 0x3FFu;
        uint32_t u;
        if (exp == 0) {
            if (mant == 0) {
                u = sign;
            } else {
                exp = 113;
                while ((mant & 0x400u) == 0) {
                    mant <<= 1;
                    exp--;
                }
                u = sign | (exp << 23) | ((mant & 0x3FFu) << 13);
            }
        } else if (exp == 31) {
            u = sign | 0x7F800000u | (mant << 13);
        } else {
            u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        union {
            uint32_t u;
            float f;
        } x = {u};
        return x.f;
    }
    half &operator=(float v) {
        union {
            float f;
            uint32_t u;
        } x = {v};
        uint32_t sign = (x.u >> 16) & 0x8000u;
        uint32_t abs = x.u & 0x7fffffffu;
        if (abs >= 0x7f800000u) {
            uint16_t payload = static_cast<uint16_t>((abs & 0x007fffffu) >> 13);
            bits = static_cast<uint16_t>(sign | 0x7c00u | payload | (payload == 0 ? 0 : 1));
            return *this;
        }
        int32_t exponent = static_cast<int32_t>((abs >> 23) & 0xffu) - 127 + 15;
        uint32_t mantissa = abs & 0x007fffffu;
        if (exponent >= 31) {
            bits = static_cast<uint16_t>(sign | 0x7c00u);
            return *this;
        }
        if (exponent <= 0) {
            if (exponent < -10) {
                bits = static_cast<uint16_t>(sign);
                return *this;
            }
            mantissa |= 0x00800000u;
            uint32_t shift = static_cast<uint32_t>(14 - exponent);
            uint32_t rounded = (mantissa + ((1u << (shift - 1)) - 1u) + ((mantissa >> shift) & 1u)) >> shift;
            bits = static_cast<uint16_t>(sign | rounded);
            return *this;
        }
        uint32_t roundedMantissa = mantissa + 0x00000fffu + ((mantissa >> 13) & 1u);
        if ((roundedMantissa & 0x00800000u) != 0) {
            roundedMantissa = 0;
            ++exponent;
            if (exponent >= 31) {
                bits = static_cast<uint16_t>(sign | 0x7c00u);
                return *this;
            }
        }
        bits = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (roundedMantissa >> 13));
        return *this;
    }
};
static_assert(sizeof(half) == 2, "half must be 2 bytes");
#endif

struct bfloat16_t {
    uint16_t bits;
    bfloat16_t() : bits(0) {}
    bfloat16_t(float f) { *this = f; }
    operator float() const {
        uint32_t u = static_cast<uint32_t>(bits) << 16;
        union {
            uint32_t u;
            float f;
        } x = {u};
        return x.f;
    }
    bfloat16_t &operator=(float f) {
        // round half away from zero，与 kernel 侧 Cast(CAST_ROUND) 舍入一致
        union {
            float f;
            uint32_t u;
        } x = {f};
        x.u += 0x8000u;
        bits = static_cast<uint16_t>(x.u >> 16);
        return *this;
    }
};
static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 2 bytes");

// ===================== 工具宏 =====================
#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do                                  \
    {                                   \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// ===================== 数据生成 =====================

/* 为某一行无重复地从 [0, numCols) 中采样 nnz 个列号, 返回升序列表 */
void SampleRowColumnsUnique(int nnz, uint32_t numCols, uint32_t rowIdx,
                            std::vector<int> &visited, std::mt19937 &rng,
                            std::vector<int> &outCols) {
    std::uniform_int_distribution<int> colDist(0, static_cast<int>(numCols) - 1);
    outCols.clear();
    outCols.reserve(nnz);
    while (static_cast<int>(outCols.size()) < nnz) {
        int c = colDist(rng);
        if (visited[c] != static_cast<int>(rowIdx)) {
            visited[c] = static_cast<int>(rowIdx);
            outCols.push_back(c);
        }
    }
    std::sort(outCols.begin(), outCols.end());
}

template <typename T>
void GenerateDenseVector(uint32_t size, std::vector<T> &x, std::mt19937 &rng) {
    std::uniform_real_distribution<float> valDist(-5.0f, 5.0f);
    x.assign(size, T{});
    for (uint32_t i = 0; i < size; ++i) {
        x[i] = static_cast<T>(valDist(rng));
    }
}

/**
 * @brief 生成 CSR 稀疏矩阵
 *
 * @param sparsity 零元素的比例, [0,1]
 */
template <typename T>
void GenerateCsr(uint32_t numRows, uint32_t numCols, float sparsity,
                 std::vector<int32_t> &csrRowPtr,
                 std::vector<int32_t> &csrColInd,
                 std::vector<T> &csrVal,
                 float emptyRowProb = 0.0f,
                 uint32_t seed = 0) {
    if (sparsity < 0.0f || sparsity > 1.0f) {
        std::cerr << "[ERROR] sparsity must be in [0, 1], got " << sparsity << "\n";
        return;
    }
    const float density = 1.0f - sparsity;

    csrRowPtr.assign(numRows + 1, 0);
    csrColInd.clear();
    csrVal.clear();

    const size_t expectedNNZ = static_cast<size_t>(
        static_cast<float>(numRows) * static_cast<float>(numCols) *
        density * (1.0f - emptyRowProb));
    csrColInd.reserve(expectedNNZ);
    csrVal.reserve(expectedNNZ);

    std::mt19937 rng(seed != 0 ? seed : static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()));

    std::binomial_distribution<int> nnzDist(numCols, density);
    std::uniform_real_distribution<float> valDist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> emptyDist(0.0f, 1.0f);

    std::vector<int> visited(numCols, -1);
    std::vector<int> rowCols;

    for (uint32_t i = 0; i < numRows; ++i) {
        csrRowPtr[i] = static_cast<uint32_t>(csrColInd.size());

        if (emptyRowProb > 0.0f && emptyDist(rng) < emptyRowProb) {
            continue;
        }

        const int nnz = nnzDist(rng);
        if (nnz <= 0)
            continue;

        SampleRowColumnsUnique(nnz, numCols, i, visited, rng, rowCols);
        for (int c : rowCols) {
            csrColInd.push_back(static_cast<uint32_t>(c));
            csrVal.push_back(static_cast<T>(valDist(rng)));
        }
    }
    csrRowPtr[numRows] = static_cast<uint32_t>(csrColInd.size());
}

// ===================== CPU 参考实现 =====================

/**
 * @brief SpMV CPU 参考实现。
 *
 * @param productsInValT 若为 true，每个 `values[j] * x[cols[j]]` 乘积在累加前先舍入回
 * ValT。cube 版算法在输入精度下计算这些乘积（gather 的输出即 ValT），向量版则先升位到
 * CompT 再相乘，故两者需要不同的参考值。
 */
template <typename CompT, typename ValT = CompT, typename OutT = CompT>
std::vector<OutT> SpmvCpu(const std::vector<int32_t> &csrRowPtr,
                          const std::vector<int32_t> &csrColInd,
                          const std::vector<ValT> &csrVal,
                          const std::vector<ValT> &xVec,
                          const std::vector<OutT> &yVec,
                          CompT alpha = static_cast<CompT>(1), CompT beta = static_cast<CompT>(0),
                          bool productsInValT = false) {
    uint32_t M = csrRowPtr.size() - 1;
    std::vector<OutT> z(M);
    for (uint32_t i = 0; i < M; ++i) {
        CompT sum = 0;
        for (uint32_t j = csrRowPtr[i]; j < csrRowPtr[i + 1]; ++j) {
            const CompT prod =
                static_cast<CompT>(csrVal[j]) * static_cast<CompT>(xVec[csrColInd[j]]);
            sum += productsInValT ? static_cast<CompT>(static_cast<ValT>(prod)) : prod;
        }
        CompT zm = alpha * sum + beta * static_cast<CompT>(yVec[i]);
        z[i] = static_cast<OutT>(zm);
    }
    return z;
}

// ===================== CPU 参考实现（转置） =====================

template <typename CompT, typename ValT = CompT, typename OutT = CompT>
std::vector<OutT> SpmvTransCpu(const std::vector<int32_t> &csrRowPtr,
                               const std::vector<int32_t> &csrColInd,
                               const std::vector<ValT> &csrVal,
                               const std::vector<ValT> &xVec,
                               const std::vector<OutT> &yVec,
                               uint32_t numRows, uint32_t numCols,
                               CompT alpha = static_cast<CompT>(1), CompT beta = static_cast<CompT>(0)) {
    // 按 CompT 精度累加，最后一次性转换为 OutT（与 kernel 的 fp32 workspace 累加路径一致；
    // 对 OutT==CompT 的既有用例结果完全不变）
    std::vector<CompT> acc(numCols);
    for (uint32_t j = 0; j < numCols; ++j)
        acc[j] = static_cast<CompT>(beta) * static_cast<CompT>(yVec[j]);
    for (uint32_t i = 0; i < numRows; ++i) {
        CompT xVal = static_cast<CompT>(alpha) * static_cast<CompT>(xVec[i]);
        if (xVal == static_cast<CompT>(0))
            continue;
        for (uint32_t k = csrRowPtr[i]; k < csrRowPtr[i + 1]; ++k)
            acc[csrColInd[k]] = acc[csrColInd[k]] + xVal * static_cast<CompT>(csrVal[k]);
    }
    std::vector<OutT> z(numCols);
    for (uint32_t j = 0; j < numCols; ++j)
        z[j] = static_cast<OutT>(acc[j]);
    return z;
}

// ===================== 精度验证（生态算子开源精度标准：混合容差） =====================

struct MixedToleranceParams {
    float rtol;
    float atol;
    float maxAbsErrorLimit;
    float requiredMatchedRatio;
};

template <typename T>
MixedToleranceParams GetMixedToleranceParams() {
    if constexpr (std::is_same_v<T, float>) {
        return {std::ldexp(1.0f, -10), std::ldexp(1.0f, -16), 1e-2f, 0.99f};
    } else if constexpr (std::is_same_v<T, half>) {
        return {std::ldexp(1.0f, -9), std::ldexp(1.0f, -9), 1e-1f, 0.99f};
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return {std::ldexp(1.0f, -6), std::ldexp(1.0f, -6), 1e-0f, 0.99f};
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return {0.0f, 0.0f, 0.0f, 1.0f};
    }
    return {std::ldexp(1.0f, -10), std::ldexp(1.0f, -16), 1e-2f, 0.99f};
}

/* 浮点结果判定：matchedRatio / maxAbsError 双门限，打印告警与汇总，返回 0/1 */
int32_t CheckMatchedRatioAndAbsError(float matchedRatio, float maxAbsError,
                                     const MixedToleranceParams &params) {
    int32_t status = 0;
    if (matchedRatio < params.requiredMatchedRatio) {
        std::cout << "[WARNING] Matched ratio check fail! matched_ratio=" << matchedRatio
                  << " required>=" << params.requiredMatchedRatio << "\n";
        status = 1;
    }
    if (maxAbsError > params.maxAbsErrorLimit) {
        std::cout << "[WARNING] Max absolute error check fail! max_abs_error=" << maxAbsError
                  << " limit=" << params.maxAbsErrorLimit << "\n";
        status = 1;
    }
    std::cout << "Matched Ratio = " << matchedRatio
              << "; Max Absolute Error = " << maxAbsError
              << " (rtol=" << params.rtol << ", atol=" << params.atol << ")\n";
    return status;
}

template <typename T>
int32_t Verification(const std::vector<T> &cpuGolden,
                     const std::vector<T> &npuRet,
                     float &MARE, float &MERE,
                     const MixedToleranceParams &params,
                     std::string *worstInfo = nullptr) {
    if (npuRet.size() != cpuGolden.size()) {
        std::cout << "[ERROR] The size of npuRet and cpuGolden is not equal!\n";
        return 1;
    }

    int32_t status = 0;
    std::cout << "Verification...\n";
    for (int i = 0; i < std::min(static_cast<int64_t>(npuRet.size()), static_cast<int64_t>(10)); ++i) {
        std::cout << "golden[" << i << "]=" << static_cast<float>(cpuGolden[i])
                  << " npu_result[" << i << "]=" << static_cast<float>(npuRet[i]) << "\n";
    }

    size_t passCount = 0;
    size_t worstIdx = 0;
    float worstGolden = 0.0f;
    float worstNpu = 0.0f;
    float maxAbsError = 0.0f;
    float worstMargin = 0.0f;

    for (size_t i = 0; i < npuRet.size(); ++i) {
        float npuVal = static_cast<float>(npuRet[i]);
        float cpuVal = static_cast<float>(cpuGolden[i]);

        if (!std::isfinite(npuVal) || !std::isfinite(cpuVal)) {
            // Non-finite values must not be excused by the matched-ratio tolerance:
            // fabs(NaN) compares false against both the running max error and the
            // per-element tolerance, so without this check a lone NaN element is
            // silently skipped and a 99%-match result can still report success.
            maxAbsError = std::numeric_limits<float>::infinity();
            worstIdx = i;
            worstGolden = cpuVal;
            worstNpu = npuVal;
            worstMargin = std::numeric_limits<float>::infinity();
            std::cout << "[WARNING] value[" << i << "] is not finite: golden=" << cpuVal
                      << " npu_result=" << npuVal << "\n";
            continue;
        }

        float aError = std::fabs(npuVal - cpuVal);
        if (aError > maxAbsError) {
            maxAbsError = aError;
        }

        if constexpr (std::is_same_v<T, int32_t>) {
            if (npuRet[i] == cpuGolden[i]) {
                passCount++;
            } else {
                std::cout << "[WARNING] value[" << i
                          << "] in result is not equal to golden, the value is: "
                          << npuRet[i] << " while the golden is: " << cpuGolden[i] << "\n";
                const float margin = aError;
                if (margin > worstMargin) {
                    worstMargin = margin;
                    worstIdx = i;
                    worstGolden = cpuVal;
                    worstNpu = npuVal;
                }
            }
        } else {
            const float tolerance = params.atol + params.rtol * std::fabs(cpuVal);
            if (aError <= tolerance) {
                passCount++;
            } else {
                const float margin = aError - tolerance;
                if (margin > worstMargin) {
                    worstMargin = margin;
                    worstIdx = i;
                    worstGolden = cpuVal;
                    worstNpu = npuVal;
                }
            }
        }
    }

    const float matchedRatio = npuRet.empty() ? 1.0f
                                              : static_cast<float>(passCount) / static_cast<float>(npuRet.size());
    MARE = maxAbsError;
    MERE = matchedRatio;

    if constexpr (std::is_same_v<T, int32_t>) {
        if (matchedRatio < params.requiredMatchedRatio) {
            std::cout << "[WARNING] Integer exact match check fail!\n";
            status = 1;
        }
        std::cout << "Matched Ratio = " << matchedRatio
                  << "; Max Absolute Error = " << maxAbsError << "\n";
    } else {
        status = CheckMatchedRatioAndAbsError(matchedRatio, maxAbsError, params);
    }

    if (worstInfo) {
        std::ostringstream oss;
        oss << "worst[" << worstIdx << "] golden=" << worstGolden << " npu=" << worstNpu;
        if constexpr (!std::is_same_v<T, int32_t>) {
            oss << " margin=" << worstMargin;
        }
        *worstInfo = oss.str();
    }

    return status;
}

// ===================== 设备资源创建 =====================

int CreateDeviceTensor(uint8_t *hostData, const size_t size, uint8_t **deviceAddr) {
    // nnz=0（全空矩阵）等场景 size 可能为 0：仍分配最小缓冲，保证设备指针非空
    const size_t allocSize = (size == 0) ? 32 : size;
    auto ret = aclrtMalloc((void **)deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    if (size > 0) {
        ret = aclrtMemcpy(*deviceAddr, allocSize, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }
    return ACL_SUCCESS;
}

// ===================== Init / Finalize =====================

int Init(int32_t deviceId, aclrtStream *stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
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
template <>
aclDataType AclTypeOf<int32_t>() { return ACL_INT32; }
template <>
aclDataType AclTypeOf<half>() { return ACL_FLOAT16; }
template <>
aclDataType AclTypeOf<bfloat16_t>() { return ACL_BF16; }
template <>
aclDataType AclTypeOf<int8_t>() { return ACL_INT8; }

// host 侧 complex64：与 aclsparseComplex / float2 二进制兼容（{re, im} 交错存储）
struct Complex64 {
    float re;
    float im;
};
static_assert(sizeof(Complex64) == sizeof(aclsparseComplex), "Complex64 must match aclsparseComplex");

template <>
aclDataType AclTypeOf<Complex64>() { return ACL_COMPLEX64; }

// ===================== 单用例运行环境（Test / TestC64 共用） =====================

/* indexBase=1 时上传前将 rowPtr/colInd 全部 +1；CPU golden 始终用 base-0 数据 */
void ShiftCsrIndexBase(const std::vector<int32_t> &csrRowPtr,
                       const std::vector<int32_t> &csrColInd, int indexBase,
                       std::vector<int32_t> &csrRowPtrUp,
                       std::vector<int32_t> &csrColIndUp) {
    csrRowPtrUp = csrRowPtr;
    csrColIndUp = csrColInd;
    if (indexBase == 1) {
        for (auto &v : csrRowPtrUp)
            v += 1;
        for (auto &v : csrColIndUp)
            v += 1;
    }
}

/* 依次上传 rowPtr/colInd/vals/x/y 五块设备缓冲，任一失败即返回错误码 */
template <typename ValT, typename OutT>
int CreateBasicTestTensors(const std::vector<int32_t> &csrRowPtr,
                           const std::vector<int32_t> &csrColInd,
                           const std::vector<ValT> &csrVal,
                           const std::vector<ValT> &xVec,
                           const std::vector<OutT> &yVec,
                           uint8_t **rowPtrDev, uint8_t **colIndDev, uint8_t **valsDev,
                           uint8_t **xDev, uint8_t **yDev) {
    int aclRet = CreateDeviceTensor((uint8_t *)csrRowPtr.data(),
                                    csrRowPtr.size() * sizeof(int32_t), rowPtrDev);
    if (aclRet == ACL_SUCCESS)
        aclRet = CreateDeviceTensor((uint8_t *)csrColInd.data(),
                                    csrColInd.size() * sizeof(int32_t), colIndDev);
    if (aclRet == ACL_SUCCESS)
        aclRet = CreateDeviceTensor((uint8_t *)csrVal.data(),
                                    csrVal.size() * sizeof(ValT), valsDev);
    if (aclRet == ACL_SUCCESS)
        aclRet = CreateDeviceTensor((uint8_t *)xVec.data(),
                                    xVec.size() * sizeof(ValT), xDev);
    if (aclRet == ACL_SUCCESS)
        aclRet = CreateDeviceTensor((uint8_t *)yVec.data(),
                                    yVec.size() * sizeof(OutT), yDev);
    return aclRet;
}

/* SpMV 单用例运行环境：handle + 设备/host 缓冲 + 描述符。
 * 缓冲由 unique_ptr 守卫自动释放；handle 与描述符仍由调用点显式 destroy */
struct SpmvRunEnv {
    aclsparseHandle_t spHandle = nullptr;
    uint8_t *rowPtrDevice = nullptr, *colIdxDevice = nullptr;
    uint8_t *valsDevice = nullptr, *xDevice = nullptr, *yDevice = nullptr;
    uint8_t *yHost = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> rpp{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> cip{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> vdp{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> xdp{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> ydp{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> yhp{nullptr, aclrtFreeHost};
    aclsparseSpMatDescr_t matDesc = nullptr;
    aclsparseDnVecDescr_t vecXDesc = nullptr;
    aclsparseDnVecDescr_t vecYDesc = nullptr;
    size_t totalYByteSize = 0;
};

/* 创建 handle、上传数据、分配 host 读回缓冲并创建 CSR/DnVec 描述符 */
template <typename ValT, typename OutT>
int SetupSpmvRunEnv(aclrtStream stream, size_t M, size_t N,
                    const std::vector<int32_t> &csrRowPtrUp,
                    const std::vector<int32_t> &csrColIndUp,
                    const std::vector<ValT> &csrVal,
                    const std::vector<ValT> &xVec,
                    const std::vector<OutT> &yVec,
                    int indexBase, aclDataType valDt, aclDataType outDt,
                    SpmvRunEnv &env) {
    aclsparseStatus_t sparseRet = aclsparseCreate(&env.spHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    sparseRet = aclsparseSetStream(env.spHandle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    int aclRet = CreateBasicTestTensors<ValT, OutT>(csrRowPtrUp, csrColIndUp, csrVal, xVec, yVec,
                                                    &env.rowPtrDevice, &env.colIdxDevice,
                                                    &env.valsDevice, &env.xDevice, &env.yDevice);
    env.rpp.reset(env.rowPtrDevice);
    env.cip.reset(env.colIdxDevice);
    env.vdp.reset(env.valsDevice);
    env.xdp.reset(env.xDevice);
    env.ydp.reset(env.yDevice);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    env.totalYByteSize = yVec.size() * sizeof(OutT);
    aclRet = aclrtMallocHost((void **)(&env.yHost), env.totalYByteSize);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    env.yhp.reset(env.yHost);

    sparseRet = aclsparseCreateCsr(&env.matDesc, M, N, csrColIndUp.size(),
                                   env.rowPtrDevice, env.colIdxDevice, env.valsDevice,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   indexBase == 1 ? ACL_SPARSE_INDEX_BASE_ONE
                                                  : ACL_SPARSE_INDEX_BASE_ZERO,
                                   valDt);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateCsr failed\n");
              return sparseRet);

    sparseRet = aclsparseCreateDnVec(&env.vecXDesc, xVec.size(), env.xDevice, valDt);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    sparseRet = aclsparseCreateDnVec(&env.vecYDesc, yVec.size(), env.yDevice, outDt);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    return 0;
}

// DEVICE 指针模式标量暂存：alpha/beta 先 H2D 写入初始值，再把真实值以异步
// memcpy 挂在 handle stream 上，调用方传设备指针给接口，验证 host 侧
// "先流同步再 D2H 读回"的路径。句柄析构前由 unique_ptr 自动释放。
struct DeviceScalarStaging {
    std::unique_ptr<void, aclError (*)(void *)> dap{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> dbp{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> hap{nullptr, aclrtFreeHost};
    std::unique_ptr<void, aclError (*)(void *)> hbp{nullptr, aclrtFreeHost};
};

template <typename CompT>
int StageDeviceScalars(aclsparseHandle_t spHandle, aclrtStream stream,
                       const CompT &alphaVal, const CompT &betaVal,
                       const void *&alphaPtr, const void *&betaPtr,
                       DeviceScalarStaging &staging) {
    CompT initialAlpha, initialBeta;
    if constexpr (std::is_same_v<CompT, aclsparseComplex>) {
        initialAlpha = CompT{1.0f, 0.0f};
        initialBeta = CompT{0.0f, 0.0f};
    } else {
        initialAlpha = static_cast<CompT>(1);
        initialBeta = static_cast<CompT>(0);
    }
    void *deviceAlpha = nullptr;
    void *deviceBeta = nullptr;
    void *hostAlphaUpdate = nullptr;
    void *hostBetaUpdate = nullptr;
    aclError aclRet = aclrtMalloc(&deviceAlpha, sizeof(CompT), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    staging.dap.reset(deviceAlpha);
    aclRet = aclrtMalloc(&deviceBeta, sizeof(CompT), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    staging.dbp.reset(deviceBeta);
    aclRet = aclrtMemcpy(deviceAlpha, sizeof(CompT), &initialAlpha, sizeof(CompT), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(deviceBeta, sizeof(CompT), &initialBeta, sizeof(CompT), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMallocHost(&hostAlphaUpdate, sizeof(CompT));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    staging.hap.reset(hostAlphaUpdate);
    aclRet = aclrtMallocHost(&hostBetaUpdate, sizeof(CompT));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    staging.hbp.reset(hostBetaUpdate);
    *static_cast<CompT *>(hostAlphaUpdate) = alphaVal;
    *static_cast<CompT *>(hostBetaUpdate) = betaVal;
    aclsparseStatus_t pmRet = aclsparseSetPointerMode(spHandle, ACL_SPARSE_POINTER_MODE_DEVICE);
    CHECK_RET(pmRet == ACL_SPARSE_STATUS_SUCCESS, return pmRet);
    // Leave scalar updates pending on the handle stream. SpMV must observe
    // these values without caller-side synchronization.
    aclRet = aclrtMemcpyAsync(deviceAlpha, sizeof(CompT), hostAlphaUpdate, sizeof(CompT),
                              ACL_MEMCPY_HOST_TO_DEVICE, stream);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpyAsync(deviceBeta, sizeof(CompT), hostBetaUpdate, sizeof(CompT),
                              ACL_MEMCPY_HOST_TO_DEVICE, stream);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    alphaPtr = deviceAlpha;
    betaPtr = deviceBeta;
    return 0;
}

// ===================== 主测试函数 =====================

#include "spmv_test_c64.h"
#include "spmv_test_cube.h"

/**
 * @brief SpMV 单用例测试（使用 aclsparse 新接口）
 *
 * @tparam CompT     计算类型 (float / int32_t)
 * @tparam ValT      输入值类型（默认=CompT）
 * @tparam OutT      输出类型（默认=CompT）
 */
template <typename CompT, typename ValT = CompT, typename OutT = CompT>
int Test(const size_t M, const size_t N, const float sparsity,
         CompT alphaVal, CompT betaVal, bool transpose,
         float &MARE, float &MERE, aclrtStream stream,
         std::string *worstInfo = nullptr, int indexBase = 0,
         float emptyRowProb = 0.0f,
         const std::vector<int32_t> *presetRowPtr = nullptr,
         const std::vector<int32_t> *presetColInd = nullptr,
         const std::vector<ValT> *presetVal = nullptr,
         aclsparseSpMVAlg_t alg = ACL_SPARSE_SPMV_ALG_DEFAULT,
         bool productsInValT = false,
         bool deviceScalarStreamOrder = false,
         uint32_t fixedSeed = 0) {
    const char *label = transpose ? " Transpose"
                                  : (alg == ACL_SPARSE_SPMV_CSR_ALG_CUBE ? " Cube" : "");
    std::cout << "====Test" << label << " case: row num = " << M
              << " col num = " << N
              << " sparsity (zero ratio) = " << sparsity
              << " alpha = " << alphaVal
              << " beta = " << betaVal
              << " indexBase = " << indexBase << "====\n";

    std::vector<ValT> csrVal;
    std::vector<int32_t> csrColInd, csrRowPtr;
    if (presetRowPtr != nullptr) {
        csrRowPtr = *presetRowPtr;
        csrColInd = *presetColInd;
        csrVal = *presetVal;
    } else {
        GenerateCsr<ValT>(M, N, sparsity, csrRowPtr, csrColInd, csrVal, emptyRowProb, fixedSeed);
    }

    uint32_t xSize = transpose ? static_cast<uint32_t>(M) : static_cast<uint32_t>(N);
    uint32_t ySize = transpose ? static_cast<uint32_t>(N) : static_cast<uint32_t>(M);

    std::vector<ValT> xVec;
    std::vector<OutT> yVec; {
        std::mt19937 rng(fixedSeed != 0 ? fixedSeed : static_cast<uint32_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        GenerateDenseVector<ValT>(xSize, xVec, rng);
        GenerateDenseVector<OutT>(ySize, yVec, rng);
    }

    std::vector<OutT> output_cpu;
    if (transpose)
        output_cpu = SpmvTransCpu<CompT, ValT, OutT>(csrRowPtr, csrColInd, csrVal, xVec, yVec, M, N, alphaVal, betaVal);
    else
        output_cpu = SpmvCpu<CompT, ValT, OutT>(csrRowPtr, csrColInd, csrVal, xVec, yVec, alphaVal, betaVal,
                                               productsInValT);

    std::vector<int32_t> csrRowPtrUp, csrColIndUp;
    ShiftCsrIndexBase(csrRowPtr, csrColInd, indexBase, csrRowPtrUp, csrColIndUp);

    aclDataType valDt = AclTypeOf<ValT>();
    aclDataType outDt = AclTypeOf<OutT>();
    aclDataType compDt = std::is_same_v<CompT, float> ? ACL_FLOAT : ACL_INT32;

    SpmvRunEnv env;
    int setupRet = SetupSpmvRunEnv<ValT, OutT>(stream, M, N, csrRowPtrUp, csrColIndUp,
                                               csrVal, xVec, yVec, indexBase, valDt, outDt, env);
    CHECK_RET(setupRet == 0, return setupRet);

    aclsparseStatus_t sparseRet = ACL_SPARSE_STATUS_SUCCESS;
    int aclRet = ACL_SUCCESS;

    aclsparseOperation_t op = transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;

    void *externalBuffer = nullptr;  // 默认不需要额外空间

    // 三阶段流程：GetBufferSize -> Preprocess -> SpMV
    // workspace 契约（与 spmv_host.cpp 的 SpmvWorkspaceSize 一致）：
    // - 转置 + f32 计算（f32/f16/bf16 输出）：double-single 确定性累加平面
    //   groups*roundUp(cols,8)*2*sizeof(float)，groups = min(AIV 核数, max(rows,1))；
    // 其余向量版情形必须为 0。cube 算法（非转置）按 GetBufferSize 报出的
    // gather + scan workspace 大小申请，不做精确值校验。
    const bool expectWs = transpose && compDt == ACL_FLOAT &&
                          (outDt == ACL_FLOAT || outDt == ACL_FLOAT16 || outDt == ACL_BF16);
    size_t expectedWs = 0;
    if (expectWs) {
        uint32_t coreNum = 1;
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (ascendcPlatform != nullptr && ascendcPlatform->GetCoreNumAiv() > 0) {
            coreNum = ascendcPlatform->GetCoreNumAiv();
        }
        const size_t groups = std::min<size_t>(coreNum, std::max<size_t>(M, 1));
        const size_t stride = (N + 7) / 8 * 8;
        expectedWs = groups * stride * 2 * sizeof(float);
    }
    size_t bufferSize = 0;
    sparseRet = aclsparseSpMVGetBufferSize(env.spHandle, op, &alphaVal, env.matDesc, env.vecXDesc,
                                           &betaVal, env.vecYDesc, compDt,
                                           alg, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSpMVGetBufferSize failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    if (alg != ACL_SPARSE_SPMV_CSR_ALG_CUBE) {
        CHECK_RET((bufferSize > 0) == expectWs &&
                      (!expectWs || bufferSize == expectedWs),
                  LOG_PRINT("aclsparseSpMVGetBufferSize: unexpected workspace size %zu (expectWs=%d)\n",
                            bufferSize, static_cast<int>(expectWs));
                  return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    }
    std::unique_ptr<void, aclError (*)(void *)> wsp(nullptr, aclrtFree);
    if (bufferSize > 0) {
        aclRet = aclrtMalloc(&externalBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
        wsp.reset(externalBuffer);
    }

    const void *alphaPtr = &alphaVal;
    const void *betaPtr = &betaVal;
    DeviceScalarStaging staging;
    if (deviceScalarStreamOrder) {
        int stageRet = StageDeviceScalars(env.spHandle, stream, alphaVal, betaVal,
                                          alphaPtr, betaPtr, staging);
        CHECK_RET(stageRet == 0, return stageRet);
    }

    sparseRet = aclsparseSpMVPreprocess(env.spHandle, op, alphaPtr, env.matDesc, env.vecXDesc,
                                        betaPtr, env.vecYDesc, compDt,
                                        alg, externalBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSpMVPreprocess failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    sparseRet = aclsparseSpMV(env.spHandle, op, alphaPtr, env.matDesc, env.vecXDesc,
                              betaPtr, env.vecYDesc, compDt,
                              alg, externalBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    if (deviceScalarStreamOrder) {
        aclsparseSetPointerMode(env.spHandle, ACL_SPARSE_POINTER_MODE_HOST);
    }

    // ==================== 同步 ====================
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclsparseSpMV: aclrtSynchronizeStream failed, ret=%d\n", aclRet);
                return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    aclRet = aclrtMemcpy(env.yHost, env.totalYByteSize, env.yDevice, env.totalYByteSize,
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::vector<OutT> yResult;
    yResult.assign((OutT *)env.yHost, (OutT *)(env.yHost + env.totalYByteSize));

    // 乘积在输入精度下计算时（cube 版 fp16），按输入类型而非输出类型取容差。
    const MixedToleranceParams toleranceParams =
        productsInValT ? GetMixedToleranceParams<ValT>() : GetMixedToleranceParams<OutT>();
    int verifyRet = Verification<OutT>(output_cpu, yResult, MARE, MERE, toleranceParams, worstInfo);

    aclsparseDestroySpMat(env.matDesc);
    aclsparseDestroyDnVec(env.vecXDesc);
    aclsparseDestroyDnVec(env.vecYDesc);
    aclsparseDestroy(env.spHandle);

    CHECK_RET(verifyRet == 0, LOG_PRINT("====Test%s case fail!====\n\n", label); return verifyRet);
    std::cout << "====Test" << label << " case pass!====\n\n";
    return 0;
}

// ===================== 测试统计 =====================

struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
    std::vector<std::string> failedCases;
};

// Keep the presmoke suite focused on functional precision. The complete
// stress shapes are the default for ordinary build.sh runs. Do this at the
// test boundary so every dtype, algorithm, transpose, scalar and regression
// case still runs with its original parameters apart from its regular shape.
static size_t PresmokeSpmvDimension(size_t dimension, const std::string &tag)
{
    // Fixed synchronization and transpose precision regressions retain the
    // exact dimensions that originally exposed their failures.
    if (SparseTestFullEnabled() || dimension <= 64 ||
        tag.find("sync-fixed-") == 0 || tag.find("transpose-") == 0) {
        return dimension;
    }
    // Round upward to retain non-aligned dimensions such as 513 -> 129.
    return (dimension + 3) / 4;
}

template <typename CompT, typename ValT = CompT, typename OutT = CompT>
int RunAndTrack(size_t M, size_t N, float sparsity, CompT alpha, CompT beta,
                bool transpose,
                TestStats &stats, aclrtStream stream, const std::string &tag = "",
                int indexBase = 0, float emptyRowProb = 0.0f,
                const std::vector<int32_t> *presetRowPtr = nullptr,
                const std::vector<int32_t> *presetColInd = nullptr,
                const std::vector<ValT> *presetVal = nullptr,
                aclsparseSpMVAlg_t alg = ACL_SPARSE_SPMV_ALG_DEFAULT,
                bool productsInValT = false,
                bool deviceScalarStreamOrder = false,
                uint32_t fixedSeed = 0) {
    // preset 三数组按原始维度构造，presmoke 缩放必然失配，自带数组的用例豁免缩放
    if (presetRowPtr == nullptr && presetColInd == nullptr && presetVal == nullptr) {
        M = PresmokeSpmvDimension(M, tag);
        N = PresmokeSpmvDimension(N, tag);
    }
    float MARE = 0, MERE = 0;
    std::string worstInfo;
    int ret = Test<CompT, ValT, OutT>(M, N, sparsity, alpha, beta, transpose,
                                      MARE, MERE, stream, &worstInfo, indexBase,
                                      emptyRowProb, presetRowPtr, presetColInd, presetVal,
                                      alg, productsInValT, deviceScalarStreamOrder, fixedSeed);

    stats.total++;
    if (ret == 0) {
        stats.passed++;
    }
    else {
        stats.failed++;
        std::ostringstream oss;
        if (!tag.empty())
            oss << "[" << tag << "] ";
        oss << "type=" << (std::is_same_v<CompT, float> ? "float" : "int32")
            << (std::is_same_v<ValT, CompT> ? "" : " MixPrec")
            << (transpose ? " Transpose" : "")
            << (alg == ACL_SPARSE_SPMV_CSR_ALG_CUBE ? " Cube" : "")
            << " M=" << M << " N=" << N
            << " sparsity=" << sparsity
            << " alpha=" << alpha << " beta=" << beta
            << " indexBase=" << indexBase
            << " max_abs_error=" << MARE << " matched_ratio=" << MERE << "\n"
            << "          " << worstInfo << "\n"
            << "          status=" << ret;
        stats.failedCases.push_back(oss.str());
    }
    return ret;
}

// ===================== 随机测试辅助函数 =====================

template <typename T, typename ADist, typename BDist>
void RunRandomParams(int numCases, const std::string &tagPrefix, bool transpose,
                     TestStats &stats, aclrtStream stream, ADist alphaDist, BDist betaDist,
                     aclsparseSpMVAlg_t alg = ACL_SPARSE_SPMV_ALG_DEFAULT) {
    std::cout << "\n======== " << (std::is_same_v<T, float> ? "Float" : "Int32")
              << (transpose ? " Transpose" : "")
              << (alg == ACL_SPARSE_SPMV_CSR_ALG_CUBE ? " Cube" : "")
              << " Random Tests ========\n";

    std::mt19937 rng(static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    const int maxRows = SparseTestFullEnabled() ? 2048 : 512;
    const int maxCols = SparseTestFullEnabled() ? 4096 : 1024;
    std::uniform_int_distribution<int> rowDist(1, maxRows);
    std::uniform_int_distribution<int> colDist(1, maxCols);
    std::uniform_real_distribution<float> sparsityDist(0.5f, 0.999f);

    for (int i = 0; i < numCases; ++i) {
        size_t M = static_cast<size_t>(rowDist(rng));
        size_t N = static_cast<size_t>(colDist(rng));
        float sp = sparsityDist(rng);
        T a = static_cast<T>(alphaDist(rng));
        T b = static_cast<T>(betaDist(rng));
        std::cout << "--- Random " << (std::is_same_v<T, float> ? "float" : "int32")
                  << (transpose ? " trans" : "")
                  << " case " << (i + 1) << "/" << numCases << " ---\n";
        RunAndTrack<T, T, T>(M, N, sp, a, b, transpose, stats, stream,
                       tagPrefix + std::to_string(i + 1), 0, 0.0f,
                       nullptr, nullptr, nullptr, alg);
    }
}

// ===================== complex64 测试 =====================

/* 手工构造长尾行 CSR（仅索引，values 由调用方按 dtype 生成）：
 * longRow 行有 longRowNnz 个非零元（远超 kernel tile），其余行随机 0~8 个 */
void BuildLongTailCsr(uint32_t numRows, uint32_t numCols, uint32_t longRow,
                      uint32_t longRowNnz,
                      std::vector<int32_t> &csrRowPtr,
                      std::vector<int32_t> &csrColInd) {
    csrRowPtr.assign(numRows + 1, 0);
    csrColInd.clear();
    std::mt19937 rng(20260909u);
    std::uniform_int_distribution<int> fewDist(0, 8);
    std::vector<int> visited(numCols, -1);
    std::vector<int> rowCols;
    for (uint32_t i = 0; i < numRows; ++i) {
        csrRowPtr[i] = static_cast<int32_t>(csrColInd.size());
        int nnz = (i == longRow) ? static_cast<int>(longRowNnz) : fewDist(rng);
        if (nnz <= 0)
            continue;
        SampleRowColumnsUnique(nnz, numCols, i, visited, rng, rowCols);
        for (int c : rowCols)
            csrColInd.push_back(static_cast<int32_t>(c));
    }
    csrRowPtr[numRows] = static_cast<int32_t>(csrColInd.size());
}

/* 转置 + c64 输出的 double-single 确定性累加平面大小（契约同 spmv_host.cpp）；非转置为 0 */
size_t ExpectedC64Workspace(size_t M, size_t N, bool transpose) {
    size_t expectedWs = 0;
    if (transpose) {
        uint32_t coreNum = 1;
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (ascendcPlatform != nullptr && ascendcPlatform->GetCoreNumAiv() > 0) {
            coreNum = ascendcPlatform->GetCoreNumAiv();
        }
        const size_t groups = std::min<size_t>(coreNum, std::max<size_t>(M, 1));
        const size_t stride = (2 * N + 7) / 8 * 8;
        expectedWs = groups * stride * 2 * sizeof(float);
    }
    return expectedWs;
}

/* C64 三阶段执行：GetBufferSize -> Preprocess -> SpMV + 同步，校验 workspace 契约并按需分配 */
int RunC64ThreeStage(SpmvRunEnv &env, bool transpose,
                     const void *alphaPtr, const void *betaPtr,
                     size_t M, size_t N, aclrtStream stream) {
    aclsparseStatus_t sparseRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseOperation_t op = transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;

    void *externalBuffer = nullptr;  // 非转置不需要；转置需要 double-single 平面
    const size_t expectedWs = ExpectedC64Workspace(M, N, transpose);
    size_t bufferSize = 0;
    sparseRet = aclsparseSpMVGetBufferSize(env.spHandle, op, alphaPtr, env.matDesc, env.vecXDesc,
                                           betaPtr, env.vecYDesc, ACL_COMPLEX64,
                                           ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSpMVGetBufferSize failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    CHECK_RET(bufferSize == expectedWs,
              LOG_PRINT("aclsparseSpMVGetBufferSize: expect workspace %zu, got %zu\n",
                        expectedWs, bufferSize);
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    std::unique_ptr<void, aclError (*)(void *)> wsp(nullptr, aclrtFree);
    if (bufferSize > 0) {
        int aclRet = aclrtMalloc(&externalBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
        wsp.reset(externalBuffer);
    }

    sparseRet = aclsparseSpMVPreprocess(env.spHandle, op, alphaPtr, env.matDesc, env.vecXDesc,
                                        betaPtr, env.vecYDesc, ACL_COMPLEX64,
                                        ACL_SPARSE_SPMV_ALG_DEFAULT, externalBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSpMVPreprocess failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    sparseRet = aclsparseSpMV(env.spHandle, op, alphaPtr, env.matDesc, env.vecXDesc,
                              betaPtr, env.vecYDesc, ACL_COMPLEX64,
                              ACL_SPARSE_SPMV_ALG_DEFAULT, externalBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    // ==================== 同步（workspace 守卫在本函数退出时才释放，保证 kernel 已完成） ====================
    int aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclsparseSpMV: aclrtSynchronizeStream failed, ret=%d\n", aclRet);
                return ACL_SPARSE_STATUS_EXECUTION_FAILED);
    return 0;
}

/**
 * @brief complex64 SpMV 单用例测试（三阶段流程）
 *
 * @tparam INPUT_COMPLEX  true: values/x 为 complex64（c64->c64）；
 *                        false: values/x 为 float（f32->c64），y 为 complex64
 * CPU golden 用 double 复数计算后转 float，实/虚部分别按 FP32 容差校验
 */
template <bool INPUT_COMPLEX>
int TestC64(const size_t M, const size_t N, const float sparsity,
            aclsparseComplex alphaVal, aclsparseComplex betaVal, bool transpose,
            int indexBase, float &MARE, float &MERE, aclrtStream stream,
            std::string *worstInfo = nullptr,
            const std::vector<int32_t> *presetRowPtr = nullptr,
            const std::vector<int32_t> *presetColInd = nullptr,
            float valRange = 5.0f,
            bool deviceScalarStreamOrder = false) {
    using ValT = std::conditional_t<INPUT_COMPLEX, Complex64, float>;
    const char *label = transpose ? " Transpose" : "";
    std::cout << "====TestC64" << label << " case: row num = " << M
              << " col num = " << N
              << " sparsity (zero ratio) = " << sparsity
              << " alpha = (" << alphaVal.x << "," << alphaVal.y << ")"
              << " beta = (" << betaVal.x << "," << betaVal.y << ")"
              << " indexBase = " << indexBase
              << " input = " << (INPUT_COMPLEX ? "complex64" : "float32") << "====\n";

    std::vector<int32_t> csrColInd, csrRowPtr;
    std::vector<ValT> csrVal, xVec;
    std::vector<Complex64> yVec;
    std::vector<DComplex> golden;
    PrepareC64Case<ValT>(M, N, sparsity, transpose, presetRowPtr, presetColInd, valRange,
                         alphaVal, betaVal, csrRowPtr, csrColInd, csrVal, xVec, yVec, golden);

    std::vector<int32_t> csrRowPtrUp, csrColIndUp;
    ShiftCsrIndexBase(csrRowPtr, csrColInd, indexBase, csrRowPtrUp, csrColIndUp);

    SpmvRunEnv env;
    int setupRet = SetupSpmvRunEnv<ValT, Complex64>(stream, M, N, csrRowPtrUp, csrColIndUp,
                                                    csrVal, xVec, yVec, indexBase,
                                                    AclTypeOf<ValT>(), ACL_COMPLEX64, env);
    CHECK_RET(setupRet == 0, return setupRet);

    const void *alphaPtr = &alphaVal;
    const void *betaPtr = &betaVal;
    DeviceScalarStaging staging;
    if (deviceScalarStreamOrder) {
        int stageRet = StageDeviceScalars(env.spHandle, stream, alphaVal, betaVal,
                                          alphaPtr, betaPtr, staging);
        CHECK_RET(stageRet == 0, return stageRet);
    }

    int execRet = RunC64ThreeStage(env, transpose, alphaPtr, betaPtr, M, N, stream);
    if (deviceScalarStreamOrder) {
        // env 句柄为用例内新建，显式恢复 HOST 模式避免语义泄漏
        aclsparseSetPointerMode(env.spHandle, ACL_SPARSE_POINTER_MODE_HOST);
    }
    CHECK_RET(execRet == 0, return execRet);

    int verifyRet = ReadbackAndVerifyC64(env, stream, golden, MARE, MERE, worstInfo);

    CHECK_RET(verifyRet == 0, LOG_PRINT("====TestC64%s case fail!====\n\n", label); return verifyRet);
    std::cout << "====TestC64" << label << " case pass!====\n\n";
    return 0;
}

template <bool INPUT_COMPLEX>
int RunC64AndTrack(size_t M, size_t N, float sparsity,
                   aclsparseComplex alpha, aclsparseComplex beta, bool transpose,
                   TestStats &stats, aclrtStream stream, const std::string &tag = "",
                   int indexBase = 0,
                   const std::vector<int32_t> *presetRowPtr = nullptr,
                   const std::vector<int32_t> *presetColInd = nullptr,
                   float valRange = 5.0f,
                   bool deviceScalarStreamOrder = false) {
    float MARE = 0, MERE = 0;
    std::string worstInfo;
    int ret = TestC64<INPUT_COMPLEX>(M, N, sparsity, alpha, beta, transpose, indexBase,
                                     MARE, MERE, stream, &worstInfo, presetRowPtr, presetColInd,
                                     valRange, deviceScalarStreamOrder);

    stats.total++;
    if (ret == 0) {
        stats.passed++;
    }
    else {
        stats.failed++;
        std::ostringstream oss;
        if (!tag.empty())
            oss << "[" << tag << "] ";
        oss << "type=complex64"
            << (INPUT_COMPLEX ? " c64->c64" : " f32->c64")
            << (transpose ? " Transpose" : "")
            << " M=" << M << " N=" << N
            << " sparsity=" << sparsity
            << " alpha=(" << alpha.x << "," << alpha.y << ")"
            << " beta=(" << beta.x << "," << beta.y << ")"
            << " indexBase=" << indexBase
            << " max_abs_error=" << MARE << " matched_ratio=" << MERE << "\n"
            << "          " << worstInfo << "\n"
            << "          status=" << ret;
        stats.failedCases.push_back(oss.str());
    }
    return ret;
}

// ===================== !343 移植场景（重写架构的等价性回归） =====================

// 单次转置三阶段执行（externalBuffer 由调用方决定是否分配），输出原始 y 字节。
// 供确定性连跑与 handle workspace 回退场景复用；成功返回 0。
template <typename CompT, typename ValT, typename OutT>
int RunTransOnce(const std::vector<int32_t> &csrRowPtr, const std::vector<int32_t> &csrColInd,
                 const std::vector<ValT> &csrVal, const std::vector<ValT> &xVec,
                 const std::vector<OutT> &yVec, size_t M, size_t N,
                 CompT alphaVal, CompT betaVal, bool allocWorkspace,
                 aclrtStream stream, std::vector<OutT> &yResult) {
    SpmvRunEnv env;
    int setupRet = SetupSpmvRunEnv<ValT, OutT>(stream, M, N, csrRowPtr, csrColInd,
                                               csrVal, xVec, yVec, 0,
                                               AclTypeOf<ValT>(), AclTypeOf<OutT>(), env);
    CHECK_RET(setupRet == 0, return setupRet);

    const aclDataType compDt = std::is_same_v<CompT, float> ? ACL_FLOAT : ACL_INT32;
    aclsparseStatus_t sparseRet = ACL_SPARSE_STATUS_SUCCESS;
    size_t bufferSize = 0;
    sparseRet = aclsparseSpMVGetBufferSize(env.spHandle, ACL_SPARSE_OP_TRANSPOSE, &alphaVal,
                                           env.matDesc, env.vecXDesc, &betaVal, env.vecYDesc,
                                           compDt, ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSpMVGetBufferSize failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    CHECK_RET(bufferSize > 0,
              LOG_PRINT("transpose case expects a workspace, got 0\n");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    void *externalBuffer = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> wsp(nullptr, aclrtFree);
    if (allocWorkspace) {
        int aclRet = aclrtMalloc(&externalBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
        wsp.reset(externalBuffer);
    }

    sparseRet = aclsparseSpMVPreprocess(env.spHandle, ACL_SPARSE_OP_TRANSPOSE, &alphaVal,
                                        env.matDesc, env.vecXDesc, &betaVal, env.vecYDesc,
                                        compDt, ACL_SPARSE_SPMV_ALG_DEFAULT, externalBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSpMVPreprocess failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    sparseRet = aclsparseSpMV(env.spHandle, ACL_SPARSE_OP_TRANSPOSE, &alphaVal,
                              env.matDesc, env.vecXDesc, &betaVal, env.vecYDesc,
                              compDt, ACL_SPARSE_SPMV_ALG_DEFAULT, externalBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    int aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(env.yHost, env.totalYByteSize, env.yDevice, env.totalYByteSize,
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    yResult.assign((OutT *)env.yHost, (OutT *)(env.yHost + env.totalYByteSize));

    aclsparseDestroySpMat(env.matDesc);
    aclsparseDestroyDnVec(env.vecXDesc);
    aclsparseDestroyDnVec(env.vecYDesc);
    aclsparseDestroy(env.spHandle);
    return 0;
}

// 列碰撞转置确定性（!343 fixedSyncInput==2 场景的重写版移植）：
// 所有非零元落在列 {0,1}、val 固定 0.125、x=1、y=0.5，全部取值在 f32/f16/bf16
// 下精确可表示；同一固定输入连跑 runs 次，要求结果 bit-wise 一致，且与
// double golden（单次舍入）零容差相等——跨核 DS 平面 TwoSum 合并必须确定性。
template <typename CompT, typename ValT, typename OutT>
int RunTransCollisionDeterminism(size_t M, size_t N, CompT beta, int runs,
                                 TestStats &stats, aclrtStream stream, const std::string &tag) {
    std::vector<int32_t> csrRowPtr(M + 1), csrColInd(M * 2);
    std::vector<ValT> csrVal(M * 2, static_cast<ValT>(0.125f));
    for (size_t i = 0; i <= M; ++i)
        csrRowPtr[i] = static_cast<int32_t>(2 * i);
    for (size_t i = 0; i < M; ++i) {
        csrColInd[2 * i] = 0;
        csrColInd[2 * i + 1] = 1;
    }
    std::vector<ValT> xVec(M, static_cast<ValT>(1.0f));
    std::vector<OutT> yVec(N, static_cast<OutT>(0.5f));

    // double golden：转置后仅 y[0]/y[1] 收到全部行的贡献，其余为 beta*y0
    std::vector<double> golden(N);
    for (size_t j = 0; j < N; ++j) {
        const double acc = (j < 2) ? 0.125 * static_cast<double>(M) : 0.0;
        golden[j] = static_cast<double>(beta) * 0.5 + acc;
    }

    std::vector<OutT> yFirst;
    bool bitwiseOk = true;
    bool exactOk = true;
    for (int r = 0; r < runs; ++r) {
        std::vector<OutT> yOut;
        int ret = RunTransOnce<CompT, ValT, OutT>(csrRowPtr, csrColInd, csrVal, xVec, yVec,
                                                  M, N, static_cast<CompT>(1), beta,
                                                  true, stream, yOut);
        CHECK_RET(ret == 0, return ret);
        if (r == 0) {
            yFirst = yOut;
        } else if (yOut.size() != yFirst.size() ||
                   std::memcmp(yOut.data(), yFirst.data(), yOut.size() * sizeof(OutT)) != 0) {
            bitwiseOk = false;
        }
        for (size_t j = 0; j < N; ++j) {
            if (static_cast<double>(static_cast<float>(yOut[j])) != golden[j]) {
                exactOk = false;
                break;
            }
        }
    }

    stats.total++;
    if (bitwiseOk && exactOk) {
        stats.passed++;
        return 0;
    }
    stats.failed++;
    std::ostringstream oss;
    oss << "[" << tag << "] bitwise=" << bitwiseOk << " exact=" << exactOk;
    stats.failedCases.push_back(oss.str());
    return 1;
}

// handle workspace 回退（!343 外部/handle 双 workspace 来源场景的移植）：
// 转置 + 16bit 输出需要 DS 平面 workspace；GetBufferSize 照调但不分配，
// externalBuffer=nullptr 时必须回退句柄 workspace，成功且精度达现有容差。
template <typename ValT>
int RunTransHandleWorkspaceFallback(TestStats &stats, aclrtStream stream, const std::string &tag) {
    const size_t M = 512, N = 512;
    const float sparsity = 0.9f;
    std::vector<int32_t> csrRowPtr, csrColInd;
    std::vector<ValT> csrVal;
    GenerateCsr<ValT>(M, N, sparsity, csrRowPtr, csrColInd, csrVal, 0.0f, 12345u);
    std::vector<ValT> xVec;
    std::vector<ValT> yVec;
    {
        std::mt19937 rng(12345u);
        GenerateDenseVector<ValT>(M, xVec, rng);
        GenerateDenseVector<ValT>(N, yVec, rng);
    }
    std::vector<ValT> golden = SpmvTransCpu<float, ValT, ValT>(csrRowPtr, csrColInd, csrVal,
                                                               xVec, yVec, M, N, 1.0f, 1.0f);

    std::vector<ValT> yResult;
    int ret = RunTransOnce<float, ValT, ValT>(csrRowPtr, csrColInd, csrVal, xVec, yVec,
                                              M, N, 1.0f, 1.0f, false, stream, yResult);
    CHECK_RET(ret == 0, return ret);

    float MARE = 0, MERE = 0;
    std::string worstInfo;
    const MixedToleranceParams toleranceParams = GetMixedToleranceParams<ValT>();
    int verifyRet = Verification<ValT>(golden, yResult, MARE, MERE, toleranceParams, &worstInfo);

    stats.total++;
    if (verifyRet == 0) {
        stats.passed++;
        return 0;
    }
    stats.failed++;
    std::ostringstream oss;
    oss << "[" << tag << "] handle-workspace fallback max_abs_error=" << MARE
        << " matched_ratio=" << MERE << "\n          " << worstInfo;
    stats.failedCases.push_back(oss.str());
    return verifyRet;
}

// fixed-sync 用例的输入守卫（口径同上游 PrepareCsrInput 的 fixedSyncInput==2
// 分支）：N < 2 无法生成列 {0,1} 的 CSR，harness 在调接口前直接判参数错误。
int GuardFixedSyncShape(size_t M, size_t N) {
    if (N < 2) {
        LOG_PRINT("fixed Sync regression requires N >= 2; rejecting invalid M=%zu, N=%zu\n",
                  M, N);
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

// N=0 参数拒绝回归（!343 RunFixedSyncInvalidShapeRegression 的移植）
void RunInvalidShapeN0Regression(TestStats &stats) {
    stats.total++;
    if (GuardFixedSyncShape(1, 0) == ACL_ERROR_INVALID_PARAM) {
        stats.passed++;
        return;
    }
    stats.failed++;
    stats.failedCases.emplace_back("invalid-shape-n0 must be rejected before CSR generation");
}

// ===================== 非法参数 / 资源泄漏测试（单独统计） =====================

struct CheckStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
    std::vector<std::string> failedChecks;
};

void ExpectStatus(CheckStats &cs, const std::string &name,
                  aclsparseStatus_t got, aclsparseStatus_t expect) {
    cs.total++;
    if (got == expect) {
        cs.passed++;
    } else {
        cs.failed++;
        std::ostringstream oss;
        oss << name << ": expect ret=" << expect << " got=" << got;
        cs.failedChecks.push_back(oss.str());
        std::cout << "[FAIL] " << oss.str() << "\n";
    }
}

/* 非法参数测试的合法基准环境：4x8 CSR float */
struct InvalidParamEnv {
    const int64_t M = 4;
    const int64_t N = 8;
    std::vector<int32_t> rowPtr{0, 2, 3, 5, 6};
    std::vector<int32_t> colInd{0, 3, 1, 2, 7, 4};
    std::vector<float> vals{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> xVec;
    std::vector<float> yVec;

    uint8_t *rowPtrDev = nullptr, *colIndDev = nullptr, *valsDev = nullptr;
    uint8_t *xDev = nullptr, *yDev = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> g1{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> g2{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> g3{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> g4{nullptr, aclrtFree};
    std::unique_ptr<void, aclError (*)(void *)> g5{nullptr, aclrtFree};

    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t mat = nullptr;
    aclsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    float alpha = 1.0f, beta = 0.0f;
};

/* 上传基准数据并创建 handle / CSR / DnVec 描述符 */
int SetupInvalidParamEnv(aclrtStream stream, InvalidParamEnv &env) {
    env.xVec.assign(env.N, 1.0f);
    env.yVec.assign(env.M, 0.0f);

    int aclRet = CreateBasicTestTensors<float, float>(env.rowPtr, env.colInd, env.vals,
                                                      env.xVec, env.yVec,
                                                      &env.rowPtrDev, &env.colIndDev, &env.valsDev,
                                                      &env.xDev, &env.yDev);
    env.g1.reset(env.rowPtrDev);
    env.g2.reset(env.colIndDev);
    env.g3.reset(env.valsDev);
    env.g4.reset(env.xDev);
    env.g5.reset(env.yDev);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclsparseStatus_t st = aclsparseCreate(&env.handle);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    st = aclsparseSetStream(env.handle, stream);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);

    st = aclsparseCreateCsr(&env.mat, env.M, env.N, env.colInd.size(),
                            env.rowPtrDev, env.colIndDev, env.valsDev,
                            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                            ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    st = aclsparseCreateDnVec(&env.vecX, env.N, env.xDev, ACL_FLOAT);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    st = aclsparseCreateDnVec(&env.vecY, env.M, env.yDev, ACL_FLOAT);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    return 0;
}

/* handle 为 nullptr（三个阶段接口） */
void CheckNullHandle(InvalidParamEnv &env, CheckStats &cs) {
    size_t bufSize = 0;
    ExpectStatus(cs, "GetBufferSize null handle",
                 aclsparseSpMVGetBufferSize(nullptr, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                                            env.vecX, &env.beta, env.vecY, ACL_FLOAT,
                                            ACL_SPARSE_SPMV_ALG_DEFAULT, &bufSize),
                 ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    ExpectStatus(cs, "Preprocess null handle",
                 aclsparseSpMVPreprocess(nullptr, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                                         env.vecX, &env.beta, env.vecY, ACL_FLOAT,
                                         ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    ExpectStatus(cs, "SpMV null handle",
                 aclsparseSpMV(nullptr, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                               env.vecX, &env.beta, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

/* 描述符 / 标量指针为 nullptr；GetBufferSize 的 bufferSize 为 nullptr */
void CheckNullArgs(InvalidParamEnv &env, CheckStats &cs) {
    ExpectStatus(cs, "SpMV null matA",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, nullptr,
                               env.vecX, &env.beta, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStatus(cs, "SpMV null vecX",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                               nullptr, &env.beta, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStatus(cs, "SpMV null vecY",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                               env.vecX, &env.beta, nullptr, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStatus(cs, "SpMV null alpha",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, nullptr, env.mat,
                               env.vecX, &env.beta, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStatus(cs, "SpMV null beta",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                               env.vecX, nullptr, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectStatus(cs, "GetBufferSize null bufferSize",
                 aclsparseSpMVGetBufferSize(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                                            env.vecX, &env.beta, env.vecY, ACL_FLOAT,
                                            ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_INVALID_VALUE);
}

/* vecX / vecY 长度不足 */
int CheckShortVectors(InvalidParamEnv &env, CheckStats &cs) {
    aclsparseDnVecDescr_t shortX = nullptr, shortY = nullptr;
    CHECK_RET(aclsparseCreateDnVec(&shortX, env.N - 1, env.xDev, ACL_FLOAT) == ACL_SPARSE_STATUS_SUCCESS,
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    ExpectStatus(cs, "SpMV vecX too short",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                               shortX, &env.beta, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_INVALID_VALUE);
    CHECK_RET(aclsparseCreateDnVec(&shortY, env.M - 1, env.yDev, ACL_FLOAT) == ACL_SPARSE_STATUS_SUCCESS,
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    ExpectStatus(cs, "SpMV vecY too short",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                               env.vecX, &env.beta, shortY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroyDnVec(shortX);
    aclsparseDestroyDnVec(shortY);
    return 0;
}

/* 不支持的 alg / opA */
void CheckUnsupportedAlgAndOp(InvalidParamEnv &env, CheckStats &cs) {
    ExpectStatus(cs, "SpMV alg COO_ALG1",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                               env.vecX, &env.beta, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_COO_ALG1, nullptr),
                 ACL_SPARSE_STATUS_NOT_SUPPORTED);
    ExpectStatus(cs, "SpMV alg out-of-range",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                               env.vecX, &env.beta, env.vecY, ACL_FLOAT,
                               static_cast<aclsparseSpMVAlg_t>(99), nullptr),
                 ACL_SPARSE_STATUS_NOT_SUPPORTED);
    ExpectStatus(cs, "SpMV opA conjugate-transpose",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_CONJUGATE_TRANSPOSE, &env.alpha, env.mat,
                               env.vecX, &env.beta, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

/* 非法 indexBase / 索引类型非 I32 / 矩阵非 CSR（CreateCsr 不校验，在 SpMV 校验） */
int CheckBadMatrixVariants(InvalidParamEnv &env, CheckStats &cs) {
    // 950 合入起 aclsparseCreateCsr 在创建期直接拒绝非法 indexBase / valueType
    // （INVALID_VALUE，实机探测确认），不再延迟到 SpMV 期；这两个变体断言新契约。
    aclsparseSpMatDescr_t matBadBase = nullptr;
    ExpectStatus(cs, "CreateCsr invalid indexBase",
                 aclsparseCreateCsr(&matBadBase, env.M, env.N, env.colInd.size(),
                                    env.rowPtrDev, env.colIndDev, env.valsDev,
                                    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                    static_cast<aclsparseIndexBase_t>(5), ACL_FLOAT),
                 ACL_SPARSE_STATUS_INVALID_VALUE);

    aclsparseSpMatDescr_t matBadVal = nullptr;
    ExpectStatus(cs, "CreateCsr invalid valueType",
                 aclsparseCreateCsr(&matBadVal, env.M, env.N, env.colInd.size(),
                                    env.rowPtrDev, env.colIndDev, env.valsDev,
                                    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                    ACL_SPARSE_INDEX_BASE_ZERO,
                                    static_cast<aclDataType>(999)),
                 ACL_SPARSE_STATUS_INVALID_VALUE);

    // I64 索引、CSC 格式创建期仍成功（实机探测确认），不支持断言保留在 SpMV 期。
    aclsparseSpMatDescr_t matI64 = nullptr;
    CHECK_RET(aclsparseCreateCsr(&matI64, env.M, env.N, env.colInd.size(),
                                 env.rowPtrDev, env.colIndDev, env.valsDev,
                                 ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_64I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT) ==
                  ACL_SPARSE_STATUS_SUCCESS,
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    ExpectStatus(cs, "SpMV index type I64",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, matI64,
                               env.vecX, &env.beta, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_NOT_SUPPORTED);
    aclsparseDestroySpMat(matI64);

    aclsparseSpMatDescr_t matCsc = nullptr;
    CHECK_RET(aclsparseCreateCsc(&matCsc, env.M, env.N, env.colInd.size(),
                                 env.rowPtrDev, env.colIndDev, env.valsDev,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT) ==
                  ACL_SPARSE_STATUS_SUCCESS,
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    ExpectStatus(cs, "SpMV CSC format",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, matCsc,
                               env.vecX, &env.beta, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_NOT_SUPPORTED);
    aclsparseDestroySpMat(matCsc);
    return 0;
}

/* dtype 组合不支持 (compute=FLOAT, val=INT8, out=INT32)；x 的 dtype 与 values 不一致 */
int CheckDtypeVariants(InvalidParamEnv &env, CheckStats &cs) {
    aclsparseSpMatDescr_t matI8 = nullptr;
    CHECK_RET(aclsparseCreateCsr(&matI8, env.M, env.N, env.colInd.size(),
                                 env.rowPtrDev, env.colIndDev, env.valsDev,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_INT8) ==
                  ACL_SPARSE_STATUS_SUCCESS,
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    aclsparseDnVecDescr_t xI8 = nullptr, yI32 = nullptr;
    CHECK_RET(aclsparseCreateDnVec(&xI8, env.N, env.xDev, ACL_INT8) == ACL_SPARSE_STATUS_SUCCESS,
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    CHECK_RET(aclsparseCreateDnVec(&yI32, env.M, env.yDev, ACL_INT32) == ACL_SPARSE_STATUS_SUCCESS,
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    ExpectStatus(cs, "SpMV bad dtype combo (FLOAT,int8,int32)",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, matI8,
                               xI8, &env.beta, yI32, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_NOT_SUPPORTED);
    aclsparseDestroySpMat(matI8);
    aclsparseDestroyDnVec(xI8);
    aclsparseDestroyDnVec(yI32);

    aclsparseDnVecDescr_t xI32 = nullptr;
    CHECK_RET(aclsparseCreateDnVec(&xI32, env.N, env.xDev, ACL_INT32) == ACL_SPARSE_STATUS_SUCCESS,
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    ExpectStatus(cs, "SpMV x dtype != value dtype",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                               xI32, &env.beta, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_NOT_SUPPORTED);
    aclsparseDestroyDnVec(xI32);
    return 0;
}

/* 设备指针为 null（矩阵延迟绑定 / SetValues 置空）；handle 未绑定 stream */
int CheckNullDevicePtrsAndStream(InvalidParamEnv &env, CheckStats &cs) {
    aclsparseSpMatDescr_t matNullPtr = nullptr;
    CHECK_RET(aclsparseCreateCsr(&matNullPtr, env.M, env.N, env.colInd.size(),
                                 nullptr, nullptr, nullptr,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT) ==
                  ACL_SPARSE_STATUS_SUCCESS,
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    ExpectStatus(cs, "SpMV null matrix device ptrs",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, matNullPtr,
                               env.vecX, &env.beta, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroySpMat(matNullPtr);

    aclsparseDnVecDescr_t xNull = nullptr;
    CHECK_RET(aclsparseCreateDnVec(&xNull, env.N, env.xDev, ACL_FLOAT) == ACL_SPARSE_STATUS_SUCCESS,
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    CHECK_RET(aclsparseDnVecSetValues(xNull, nullptr) == ACL_SPARSE_STATUS_SUCCESS,
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    ExpectStatus(cs, "SpMV null vecX device ptr",
                 aclsparseSpMV(env.handle, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                               xNull, &env.beta, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroyDnVec(xNull);

    aclsparseHandle_t handleNoStream = nullptr;
    CHECK_RET(aclsparseCreate(&handleNoStream) == ACL_SPARSE_STATUS_SUCCESS,
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    ExpectStatus(cs, "SpMV handle without stream",
                 aclsparseSpMV(handleNoStream, ACL_SPARSE_OP_NON_TRANSPOSE, &env.alpha, env.mat,
                               env.vecX, &env.beta, env.vecY, ACL_FLOAT,
                               ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
                 ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroy(handleNoStream);
    return 0;
}

/* 非法参数：直接调三阶段接口断言返回码，覆盖 spmv_host.cpp 的每类校验 */
int TestInvalidParams(aclrtStream stream, CheckStats &cs) {
    std::cout << "\n======== Invalid-Parameter Tests ========\n";

    InvalidParamEnv env;
    int ret = SetupInvalidParamEnv(stream, env);
    CHECK_RET(ret == 0, return ret);

    CheckNullHandle(env, cs);
    CheckNullArgs(env, cs);
    ret = CheckShortVectors(env, cs);
    CHECK_RET(ret == 0, return ret);
    CheckUnsupportedAlgAndOp(env, cs);
    ret = CheckBadMatrixVariants(env, cs);
    CHECK_RET(ret == 0, return ret);
    ret = CheckDtypeVariants(env, cs);
    CHECK_RET(ret == 0, return ret);
    ret = CheckNullDevicePtrsAndStream(env, cs);
    CHECK_RET(ret == 0, return ret);

    aclsparseDestroySpMat(env.mat);
    aclsparseDestroyDnVec(env.vecX);
    aclsparseDestroyDnVec(env.vecY);
    aclsparseDestroy(env.handle);

    std::cout << "Invalid-parameter checks: " << cs.passed << "/" << cs.total << " passed\n";
    return cs.failed == 0 ? 0 : 1;
}

/* 单次泄漏迭代的三阶段调用 + 同步，返回最终状态码 */
aclsparseStatus_t RunLeakThreeStage(aclsparseHandle_t handle, aclrtStream stream,
                                    aclsparseSpMatDescr_t mat,
                                    aclsparseDnVecDescr_t vecX,
                                    aclsparseDnVecDescr_t vecY) {
    float alpha = 1.2f, beta = 0.3f;
    size_t bufSize = 0;
    aclsparseStatus_t st = aclsparseSpMVGetBufferSize(handle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, mat,
                                                      vecX, &beta, vecY, ACL_FLOAT,
                                                      ACL_SPARSE_SPMV_ALG_DEFAULT, &bufSize);
    if (st == ACL_SPARSE_STATUS_SUCCESS && bufSize != 0)
        st = ACL_SPARSE_STATUS_INTERNAL_ERROR;
    if (st == ACL_SPARSE_STATUS_SUCCESS)
        st = aclsparseSpMVPreprocess(handle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, mat,
                                     vecX, &beta, vecY, ACL_FLOAT,
                                     ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr);
    if (st == ACL_SPARSE_STATUS_SUCCESS)
        st = aclsparseSpMV(handle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, mat,
                           vecX, &beta, vecY, ACL_FLOAT,
                           ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr);
    if (st == ACL_SPARSE_STATUS_SUCCESS && aclrtSynchronizeStream(stream) != ACL_SUCCESS)
        st = ACL_SPARSE_STATUS_EXECUTION_FAILED;
    return st;
}

/* 单次泄漏迭代：create/destroy handle + 描述符 + 三阶段调用，成功路径全部释放 */
aclsparseStatus_t RunLeakIteration(aclrtStream stream,
                                   const std::vector<int32_t> &rowPtr,
                                   const std::vector<int32_t> &colInd,
                                   const std::vector<float> &vals,
                                   const std::vector<float> &xVec,
                                   const std::vector<float> &yVec,
                                   int64_t M, int64_t N) {
    aclsparseHandle_t handle = nullptr;
    aclsparseStatus_t st = aclsparseCreate(&handle);
    if (st == ACL_SPARSE_STATUS_SUCCESS)
        st = aclsparseSetStream(handle, stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS)
        return st;

    uint8_t *rowPtrDev = nullptr, *colIndDev = nullptr, *valsDev = nullptr;
    uint8_t *xDev = nullptr, *yDev = nullptr;
    int aclRet = CreateBasicTestTensors<float, float>(rowPtr, colInd, vals, xVec, yVec,
                                                      &rowPtrDev, &colIndDev, &valsDev,
                                                      &xDev, &yDev);
    if (aclRet != ACL_SUCCESS) {
        aclsparseDestroy(handle);
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    aclsparseSpMatDescr_t mat = nullptr;
    aclsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    st = aclsparseCreateCsr(&mat, M, N, colInd.size(), rowPtrDev, colIndDev, valsDev,
                            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                            ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    if (st == ACL_SPARSE_STATUS_SUCCESS)
        st = aclsparseCreateDnVec(&vecX, N, xDev, ACL_FLOAT);
    if (st == ACL_SPARSE_STATUS_SUCCESS)
        st = aclsparseCreateDnVec(&vecY, M, yDev, ACL_FLOAT);
    if (st == ACL_SPARSE_STATUS_SUCCESS)
        st = RunLeakThreeStage(handle, stream, mat, vecX, vecY);

    aclsparseDestroySpMat(mat);
    aclsparseDestroyDnVec(vecX);
    aclsparseDestroyDnVec(vecY);
    aclsparseDestroy(handle);
    aclrtFree(rowPtrDev);
    aclrtFree(colIndDev);
    aclrtFree(valsDev);
    aclrtFree(xDev);
    aclrtFree(yDev);
    return st;
}

/* 资源泄漏粗测：循环 create/destroy handle + 描述符 + 三阶段调用，确认无错误码、进程不崩 */
int TestResourceLeak(aclrtStream stream, CheckStats &cs) {
    std::cout << "\n======== Resource Leak Test (200 iterations) ========\n";
    const int64_t M = 16, N = 32;
    std::vector<int32_t> rowPtr, colInd;
    std::vector<float> vals;
    GenerateCsr<float>(M, N, 0.7f, rowPtr, colInd, vals);
    std::vector<float> xVec(N, 1.0f), yVec(M, 0.0f);

    aclsparseStatus_t firstErr = ACL_SPARSE_STATUS_SUCCESS;
    int iter = 0;
    for (; iter < 200; ++iter) {
        firstErr = RunLeakIteration(stream, rowPtr, colInd, vals, xVec, yVec, M, N);
        if (firstErr != ACL_SPARSE_STATUS_SUCCESS)
            break;
    }
    std::cout << "Resource leak test: " << iter << "/200 iterations completed, firstErr="
              << firstErr << "\n";
    ExpectStatus(cs, "200x create/destroy handle + descriptors + 3-stage SpMV",
                 (iter == 200) ? ACL_SPARSE_STATUS_SUCCESS : firstErr,
                 ACL_SPARSE_STATUS_SUCCESS);
    return cs.failed == 0 ? 0 : 1;
}

// ===================== TwoProd 积溢出回归 =====================

/* 评审反例（P2）：1×1 CSR，a=2^64、x=2^64-2^40、alpha=0.5、beta=0。
 * 普通 FP32 乘积 a*x = 2^128-2^104 单次舍入为 FLT_MAX，正确输出精确等于 FLT_MAX/2；
 * 若 Dekker 拆分后的高位积 ah*bh=2^128 溢出为 Inf，误差项 e=Inf 会把补偿累加污染为 Inf。
 * 非转置（Dot2 短行标量路径）与转置（DS 窗口累加路径）共用 SpmvTwoProd，各跑一条。
 * 判定用零容差精确比较：double golden 0.5*(2^128-2^104) = 2^127-2^103 在 FP32 精确可表示。 */
static int TestTwoProdProdOverflow(bool transpose, aclrtStream stream) {
    std::vector<int32_t> rowPtr = {0, 1};
    std::vector<int32_t> colInd = {0};
    std::vector<float> vals = {0x1p64f};            // 2^64，FP32 精确
    std::vector<float> xVec = {0x1p64f - 0x1p40f};  // 2^64-2^40 = (2^24-1)*2^40，FP32 精确
    std::vector<float> yVec = {0.0f};
    float alpha = 0.5f, beta = 0.0f;
    const float expected = std::numeric_limits<float>::max() / 2.0f;  // = 2^127-2^103

    SpmvRunEnv env;
    int ret = SetupSpmvRunEnv<float, float>(stream, 1, 1, rowPtr, colInd, vals, xVec, yVec, 0,
                                            ACL_FLOAT, ACL_FLOAT, env);
    CHECK_RET(ret == 0, return ret);
    const aclsparseOperation_t op = transpose ? ACL_SPARSE_OP_TRANSPOSE
                                              : ACL_SPARSE_OP_NON_TRANSPOSE;
    size_t bufferSize = 0;
    aclsparseStatus_t st = aclsparseSpMVGetBufferSize(env.spHandle, op, &alpha, env.matDesc,
        env.vecXDesc, &beta, env.vecYDesc, ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    void *externalBuffer = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> wsp(nullptr, aclrtFree);
    if (bufferSize > 0) {
        ret = aclrtMalloc(&externalBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        wsp.reset(externalBuffer);
    }
    st = aclsparseSpMVPreprocess(env.spHandle, op, &alpha, env.matDesc, env.vecXDesc, &beta,
                                 env.vecYDesc, ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT,
                                 externalBuffer);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    st = aclsparseSpMV(env.spHandle, op, &alpha, env.matDesc, env.vecXDesc, &beta,
                       env.vecYDesc, ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, externalBuffer);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    float result = 0.0f;
    ret = aclrtMemcpy(&result, sizeof(result), env.yDevice, sizeof(result),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    CHECK_RET(result == expected,
              LOG_PRINT("[ERROR] twoprod-prod-overflow-%c: expect %.9e (FLT_MAX/2), got %.9e\n",
                        transpose ? 't' : 'n', expected, result);
              return 1);
    std::cout << "====TwoProd prod-overflow " << (transpose ? "transpose" : "non-transpose")
              << " case pass!====\n";
    return 0;
}

// ---- 整矩阵仅 1 个非零元（固定输入回归；封装以降低 main 圈复杂度） ----
static void RunSingleNonzeroRegression(TestStats &stats, aclrtStream stream) {
    std::vector<int32_t> snRowPtr(65, 0);
    for (size_t i = 11; i < snRowPtr.size(); ++i)
        snRowPtr[i] = 1;  // 仅第 10 行有 1 个非零元
    std::vector<int32_t> snColInd = {77};
    std::vector<float> snVal = {3.5f};
    RunAndTrack<float, float, float>(64, 128, 0.0f, 1.0f, 0.0f, false, stats, stream,
                       "edge-single-nnz", 0, 0.0f, &snRowPtr, &snColInd, &snVal);
    RunAndTrack<float, float, float>(64, 128, 0.0f, 2.0f, 0.5f, true, stats, stream,
                       "edge-trans-single-nnz", 0, 0.0f, &snRowPtr, &snColInd, &snVal);
}

// ---- TwoProd 拆分高位积溢出（P2 评审反例，确定性精确值判定；封装以降低 main 圈复杂度） ----
static void RunTwoProdRegression(TestStats &stats, aclrtStream stream) {
    std::string worstInfo;
    int ret = TestTwoProdProdOverflow(false, stream);
    stats.total++;
    if (ret == 0) {
        stats.passed++;
    } else {
        stats.failed++;
        stats.failedCases.push_back(std::string("[twoprod-prod-overflow-n] ") + worstInfo);
    }
    ret = TestTwoProdProdOverflow(true, stream);
    stats.total++;
    if (ret == 0) {
        stats.passed++;
    } else {
        stats.failed++;
        stats.failedCases.push_back(std::string("[twoprod-prod-overflow-t] ") + worstInfo);
    }
}

// TwoProd 零值行漏入快路径回归（依赖 TestStats / SpmvRunEnv / SetupSpmvRunEnv，
// include 位置须在上述定义之后，参照 spmv_test_c64.h / spmv_test_cube.h 先例）
#include "spmv_test_zero_row.h"

// ===================== main =====================

int main(int32_t /*argc*/, char * /*argv*/[]) {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    TestStats stats;
    std::cout << "[SpMV] " << (SparseTestFullEnabled() ? "full stress shapes" : "small functional shapes")
              << " (set SPARSE_TEST_PRESMOKE=1 for small shapes)\n";

    std::cout << "\n"
              << "###########################################################\n"
              << "##               SpMV Test Suite (aclsparse)              ##\n"
              << "###########################################################\n\n";

    // ============ Float 基础用例 ============
    std::cout << "======== Float Basic Tests (alpha=1.0, beta=0.0) ========\n";
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "float-default");
    RunAndTrack<float, float, float>(512, 1024, 0.0f, 1.0f, 0.0f, false, stats, stream, "float-dense");
    RunAndTrack<float, float, float>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "float-sparse");
    RunAndTrack<float, float, float>(10, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "float-fewrow");
    RunAndTrack<float, float, float>(1, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "float-1row");
    RunAndTrack<float, float, float>(512, 8192, 0.1f, 1.0f, 0.0f, false, stats, stream, "float-wide");

    // ============ Float Alpha/Beta 组合测试 ============
    std::cout << "\n======== Float Alpha/Beta Tests ========\n";
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream, "float-alpha2"); // y = 2*A*x
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 1.0f, 1.0f, false, stats, stream, "float-beta1");  // y = A*x + y
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 0.5f, 0.5f, false, stats, stream, "float-half");   // y = 0.5*A*x + 0.5*y
    RunAndTrack<float, float, float>(512, 1024, 0.9f, -1.0f, 0.0f, false, stats, stream, "float-negA");  // y = -A*x
    RunAndTrack<float, float, float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "float-mix");   // y = 1.5*A*x + 0.2*y

    // ============ Float Alpha=0, beta=1 (纯 pass-through) ============
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 0.0f, 1.0f, false, stats, stream, "float-betaOnly");
    RunAndTrack<float, float, float>(512, 1024, 0.0f, 0.0f, 2.0f, false, stats, stream, "float-beta2");

    // ============ Float 随机参数采样 ============
    RunRandomParams<float>(30, "float-rand-", false, stats, stream,
                           std::uniform_real_distribution<float>(-2.0f, 2.0f),
                           std::uniform_real_distribution<float>(-1.0f, 1.0f));

    // ============ Float TwoProd 拆分溢出回归（极端值：|a| > FLT_MAX/4097 ≈ 8.3e34）============
    // A 含 2^116 / 1e35 极端值，alpha=2^-116 把结果缩回普通量级（保证 max_abs 判定有效）；
    // 修复前拆分乘法溢出导致补偿误差项为 NaN、本组用例失败；修复后极端项回退为普通乘积精度。
    {
        std::vector<int32_t> extremeRowPtr = {0, 1, 3, 3, 5};
        std::vector<int32_t> extremeColInd = {0, 1, 7, 2, 5};
        std::vector<float> extremeVal = {0x1p116f, 1.0e35f, 2.0f, 1.0e35f, -3.0f};
        RunAndTrack<float, float, float>(4, 8, 0.0f, 0x1p-116f, 0.0f, false, stats, stream,
                                         "float-extreme-large-val", 0, 0.0f,
                                         &extremeRowPtr, &extremeColInd, &extremeVal);
        RunAndTrack<float, float, float>(4, 8, 0.0f, 0x1p-116f, 0.0f, true, stats, stream,
                                         "trans-float-extreme-large-val", 0, 0.0f,
                                         &extremeRowPtr, &extremeColInd, &extremeVal);
    }

    // ============ Cube 算法 fp32 用例（ACL_SPARSE_SPMV_CSR_ALG_CUBE，仅非转置）============
    std::cout << "\n======== Float Cube Algorithm Tests ========\n";
    constexpr aclsparseSpMVAlg_t kCube = ACL_SPARSE_SPMV_CSR_ALG_CUBE;
    RunAndTrack<float, float, float>(1, 1, 0.0f, 2.0f, 1.0f, false, stats, stream,
                                     "device-scalar-stream-order-float-cube",
                                     0, 0.0f, nullptr, nullptr, nullptr, kCube, false, true);
    RunAndTrack<int32_t, int32_t, int32_t>(1, 1, 0.0f, 2, 1, false, stats, stream,
                                           "device-scalar-stream-order-int32-vector",
                                           0, 0.0f, nullptr, nullptr, nullptr,
                                           ACL_SPARSE_SPMV_ALG_DEFAULT, false, true);
    {
        // cube 最长行探测缓存：Preprocess 复用 + SetPointers 换绑失效
        std::string worstInfo;
        int ret = TestCubeMaxRowCache(64, 64, 0.9f, 2.0f, 1.0f, stream, &worstInfo);
        stats.total++;
        if (ret == 0) {
            stats.passed++;
        } else {
            stats.failed++;
            stats.failedCases.push_back(std::string("[cube-maxrow-cache] ") + worstInfo);
        }
    }
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "cube-default", 0, 0.0f, nullptr, nullptr, nullptr, kCube);
    RunAndTrack<float, float, float>(512, 1024, 0.0f, 1.0f, 0.0f, false, stats, stream, "cube-dense", 0, 0.0f, nullptr, nullptr, nullptr, kCube);
    RunAndTrack<float, float, float>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "cube-sparse", 0, 0.0f, nullptr, nullptr, nullptr, kCube);
    RunAndTrack<float, float, float>(10, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "cube-fewrow", 0, 0.0f, nullptr, nullptr, nullptr, kCube);
    RunAndTrack<float, float, float>(1, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "cube-1row", 0, 0.0f, nullptr, nullptr, nullptr, kCube);
    RunAndTrack<float, float, float>(512, 8192, 0.1f, 1.0f, 0.0f, false, stats, stream, "cube-wide", 0, 0.0f, nullptr, nullptr, nullptr, kCube);
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream, "cube-alpha2", 0, 0.0f, nullptr, nullptr, nullptr, kCube);
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 0.5f, 0.5f, false, stats, stream, "cube-half", 0, 0.0f, nullptr, nullptr, nullptr, kCube);
    RunAndTrack<float, float, float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "cube-mix", 0, 0.0f, nullptr, nullptr, nullptr, kCube);
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 0.0f, 1.0f, false, stats, stream, "cube-betaOnly", 0, 0.0f, nullptr, nullptr, nullptr, kCube);
    // 全零矩阵（nnz == 0）：cube 版退化为 y = beta * y，由向量版 kernel 接管。
    RunAndTrack<float, float, float>(512, 1024, 1.0f, 1.0f, 0.5f, false, stats, stream, "cube-empty", 0, 0.0f, nullptr, nullptr, nullptr, kCube);
    RunRandomParams<float>(10, "cube-rand-", false, stats, stream,
                           std::uniform_real_distribution<float>(-2.0f, 2.0f),
                           std::uniform_real_distribution<float>(-1.0f, 1.0f), kCube);

    // ============ Cube 算法 fp16 用例（矩阵/x 为 fp16，cube 以 fp32 累加，y 为 fp32）============
    // 乘积在 fp16 下计算，故参考实现同样把乘积舍入回 fp16，并按 fp16 容差校验。
    std::cout << "\n======== Float16 Cube Algorithm Tests ========\n";
    constexpr bool kProductsInF16 = true;
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "cube-f16-default", 0, 0.0f, nullptr, nullptr, nullptr, kCube, kProductsInF16);
    RunAndTrack<float, half, float>(512, 1024, 0.0f, 1.0f, 0.0f, false, stats, stream, "cube-f16-dense", 0, 0.0f, nullptr, nullptr, nullptr, kCube, kProductsInF16);
    RunAndTrack<float, half, float>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "cube-f16-sparse", 0, 0.0f, nullptr, nullptr, nullptr, kCube, kProductsInF16);
    RunAndTrack<float, half, float>(10, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "cube-f16-fewrow", 0, 0.0f, nullptr, nullptr, nullptr, kCube, kProductsInF16);
    RunAndTrack<float, half, float>(1, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "cube-f16-1row", 0, 0.0f, nullptr, nullptr, nullptr, kCube, kProductsInF16);
    RunAndTrack<float, half, float>(512, 8192, 0.1f, 1.0f, 0.0f, false, stats, stream, "cube-f16-wide", 0, 0.0f, nullptr, nullptr, nullptr, kCube, kProductsInF16);
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream, "cube-f16-alpha2", 0, 0.0f, nullptr, nullptr, nullptr, kCube, kProductsInF16);
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 0.5f, 0.5f, false, stats, stream, "cube-f16-half", 0, 0.0f, nullptr, nullptr, nullptr, kCube, kProductsInF16);
    RunAndTrack<float, half, float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "cube-f16-mix", 0, 0.0f, nullptr, nullptr, nullptr, kCube, kProductsInF16);
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 0.0f, 1.0f, false, stats, stream, "cube-f16-betaOnly", 0, 0.0f, nullptr, nullptr, nullptr, kCube, kProductsInF16);

    // ============ Int32 基础用例 ============
    std::cout << "\n======== Int32 Basic Tests (alpha=1, beta=0) ========\n";
    RunAndTrack<int32_t>(512, 1024, 0.9f, 1, 0, false, stats, stream, "int32-default");
    RunAndTrack<int32_t>(512, 1024, 0.0f, 1, 0, false, stats, stream, "int32-dense");
    RunAndTrack<int32_t>(512, 1024, 0.999f, 1, 0, false, stats, stream, "int32-sparse");
    RunAndTrack<int32_t>(10, 1024, 0.999f, 1, 0, false, stats, stream, "int32-fewrow");
    RunAndTrack<int32_t>(1, 1024, 0.9f, 1, 0, false, stats, stream, "int32-1row");

    // ============ Int32 Alpha/Beta 组合测试 ============
    std::cout << "\n======== Int32 Alpha/Beta Tests ========\n";
    RunAndTrack<int32_t>(512, 1024, 0.9f, 2, 0, false, stats, stream, "int32-alpha2"); // y = 2*A*x
    RunAndTrack<int32_t>(512, 1024, 0.9f, 1, 1, false, stats, stream, "int32-beta1");  // y = A*x + y
    RunAndTrack<int32_t>(512, 1024, 0.9f, -1, 0, false, stats, stream, "int32-negA");  // y = -A*x
    RunAndTrack<int32_t>(256, 2048, 0.95f, 3, -2, false, stats, stream, "int32-mix");  // y = 3*A*x - 2*y

    // ============ Int32 随机参数采样 ============
    RunRandomParams<int32_t>(30, "int32-rand-", false, stats, stream,
                             std::uniform_int_distribution<int>(-5, 5),
                             std::uniform_int_distribution<int>(-5, 5));

    // ====================== Transpose 测试 ======================
    std::cout << "\n"
              << "###########################################################\n"
              << "##            Transpose SpMV Test Suite                   ##\n"
              << "###########################################################\n\n";

    // ============ Float Transpose 基础用例 ============
    std::cout << "======== Float Transpose Basic Tests (alpha=1.0, beta=0.0) ========\n";
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-float-default");
    RunAndTrack<float, float, float>(512, 1024, 0.0f, 1.0f, 0.0f, true, stats, stream, "trans-float-dense");
    RunAndTrack<float, float, float>(512, 1024, 0.999f, 1.0f, 0.0f, true, stats, stream, "trans-float-sparse");
    RunAndTrack<float, float, float>(10, 1024, 0.999f, 1.0f, 0.0f, true, stats, stream, "trans-float-fewrow");
    RunAndTrack<float, float, float>(1, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-float-1row");
    RunAndTrack<float, float, float>(512, 8192, 0.1f, 1.0f, 0.0f, true, stats, stream, "trans-float-wide");
    // 方阵
    RunAndTrack<float, float, float>(1024, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-float-square");
    // 行列不等，交换测试 —— 用于覆盖 M < N 和 M > N 场景
    RunAndTrack<float, float, float>(2048, 512, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-float-mgreater");

    // ============ Float Transpose Alpha/Beta 组合 ============
    std::cout << "\n======== Float Transpose Alpha/Beta Tests ========\n";
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 2.0f, 0.0f, true, stats, stream, "trans-float-alpha2");
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 1.0f, 1.0f, true, stats, stream, "trans-float-beta1");
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 0.5f, 0.5f, true, stats, stream, "trans-float-half");
    RunAndTrack<float, float, float>(512, 1024, 0.9f, -1.0f, 0.0f, true, stats, stream, "trans-float-negA");
    RunAndTrack<float, float, float>(256, 2048, 0.95f, 1.5f, 0.2f, true, stats, stream, "trans-float-mix");
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 0.0f, 1.0f, true, stats, stream, "trans-float-betaOnly");
    RunAndTrack<float, float, float>(512, 1024, 0.0f, 0.0f, 2.0f, true, stats, stream, "trans-float-beta2");

    // ============ Float Transpose 随机参数采样 ============
    RunRandomParams<float>(20, "trans-float-rand-", true, stats, stream,
                           std::uniform_real_distribution<float>(-2.0f, 2.0f),
                           std::uniform_real_distribution<float>(-1.0f, 1.0f));

    // ============ Int32 Transpose 基础用例 ============
    std::cout << "\n======== Int32 Transpose Basic Tests (alpha=1, beta=0) ========\n";
    RunAndTrack<int32_t>(512, 1024, 0.9f, 1, 0, true, stats, stream, "trans-int32-default");
    RunAndTrack<int32_t>(512, 1024, 0.0f, 1, 0, true, stats, stream, "trans-int32-dense");
    RunAndTrack<int32_t>(512, 1024, 0.999f, 1, 0, true, stats, stream, "trans-int32-sparse");
    RunAndTrack<int32_t>(10, 1024, 0.999f, 1, 0, true, stats, stream, "trans-int32-fewrow");
    RunAndTrack<int32_t>(1, 1024, 0.9f, 1, 0, true, stats, stream, "trans-int32-1row");
    RunAndTrack<int32_t>(1024, 1024, 0.9f, 1, 0, true, stats, stream, "trans-int32-square");
    RunAndTrack<int32_t>(2048, 512, 0.9f, 1, 0, true, stats, stream, "trans-int32-mgreater");

    // ============ Int32 Transpose Alpha/Beta 组合 ============
    std::cout << "\n======== Int32 Transpose Alpha/Beta Tests ========\n";
    RunAndTrack<int32_t>(512, 1024, 0.9f, 2, 0, true, stats, stream, "trans-int32-alpha2");
    RunAndTrack<int32_t>(512, 1024, 0.9f, 1, 1, true, stats, stream, "trans-int32-beta1");
    RunAndTrack<int32_t>(512, 1024, 0.9f, -1, 0, true, stats, stream, "trans-int32-negA");
    RunAndTrack<int32_t>(256, 2048, 0.95f, 3, -2, true, stats, stream, "trans-int32-mix");

    // ============ Int32 Transpose 随机参数采样 ============
    RunRandomParams<int32_t>(20, "trans-int32-rand-", true, stats, stream,
                             std::uniform_int_distribution<int>(-3, 5),
                             std::uniform_int_distribution<int>(-2, 3));

    // ====================== 混合精度用例 ======================
    std::cout << "\n"
              << "###########################################################\n"
              << "##              Mixed-Precision Test Suite               ##\n"
              << "###########################################################\n\n";

    // ---- Float16→Float32 Non-Transpose ----
    std::cout << "======== Float16→Float32 Non-Transpose ========\n";
    RunAndTrack<float, half, float>(256, 512, 0.9f, 1.0f, 0.0f, false, stats, stream, "f16-f32-default");
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream, "f16-f32-alpha2");
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 1.0f, 1.0f, false, stats, stream, "f16-f32-beta1");
    RunAndTrack<float, half, float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "f16-f32-mix");
    RunAndTrack<float, half, float>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "f16-f32-sparse");
    RunAndTrack<float, half, float>(512, 1024, 0.1f, 1.0f, 0.0f, false, stats, stream, "f16-f32-dense");
    RunAndTrack<float, half, float>(10, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "f16-f32-fewrow");
    RunAndTrack<float, half, float>(1, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "f16-f32-1row");

    // ---- Float16→Float32 Transpose ----
    std::cout << "\n======== Float16→Float32 Transpose ========\n";
    RunAndTrack<float, half, float>(256, 512, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-f16-f32-def");
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 2.0f, 0.0f, true, stats, stream, "trans-f16-f32-alpha2");
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 1.0f, 1.0f, true, stats, stream, "trans-f16-f32-beta1");
    RunAndTrack<float, half, float>(1024, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-f16-f32-square");
    RunAndTrack<float, half, float>(2048, 512, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-f16-f32-mgreater");
    RunAndTrack<float, half, float>(512, 1024, 0.999f, 1.0f, 0.0f, true, stats, stream, "trans-f16-f32-sparse");
    RunAndTrack<float, half, float>(512, 1024, 0.1f, 0.0f, 1.0f, true, stats, stream, "trans-f16-f32-betaOnly");
    RunAndTrack<float, half, float>(10, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-f16-f32-fewrow");

    // ---- BF16→Float32 Non-Transpose ----
    std::cout << "\n======== BF16→Float32 Non-Transpose ========\n";
    RunAndTrack<float, bfloat16_t, float>(256, 512, 0.9f, 1.0f, 0.0f, false, stats, stream, "bf16-f32-default");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream, "bf16-f32-alpha2");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.9f, 1.0f, 1.0f, false, stats, stream, "bf16-f32-beta1");
    RunAndTrack<float, bfloat16_t, float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "bf16-f32-mix");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "bf16-f32-sparse");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.1f, 1.0f, 0.0f, false, stats, stream, "bf16-f32-dense");
    RunAndTrack<float, bfloat16_t, float>(10, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "bf16-f32-fewrow");
    RunAndTrack<float, bfloat16_t, float>(1, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "bf16-f32-1row");

    // ---- BF16→Float32 Transpose ----
    std::cout << "\n======== BF16→Float32 Transpose ========\n";
    RunAndTrack<float, bfloat16_t, float>(256, 512, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-bf16-f32-def");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.9f, 2.0f, 0.0f, true, stats, stream, "trans-bf16-f32-alpha2");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.9f, 1.0f, 1.0f, true, stats, stream, "trans-bf16-f32-beta1");
    RunAndTrack<float, bfloat16_t, float>(1024, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-bf16-f32-square");
    RunAndTrack<float, bfloat16_t, float>(2048, 512, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-bf16-f32-mgreater");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.999f, 1.0f, 0.0f, true, stats, stream, "trans-bf16-f32-sparse");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.1f, 0.0f, 1.0f, true, stats, stream, "trans-bf16-f32-betaOnly");
    RunAndTrack<float, bfloat16_t, float>(10, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-bf16-f32-fewrow");

    // ---- Int32→Float32 Non-Transpose ----
    std::cout << "\n======== Int32→Float32 Non-Transpose ========\n";
    RunAndTrack<float, int32_t, float>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "i32-f32-default");
    RunAndTrack<float, int32_t, float>(512, 1024, 0.0f, 1.0f, 0.0f, false, stats, stream, "i32-f32-dense");
    RunAndTrack<float, int32_t, float>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "i32-f32-sparse");
    RunAndTrack<float, int32_t, float>(10, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "i32-f32-fewrow");
    RunAndTrack<float, int32_t, float>(1, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "i32-f32-1row");
    RunAndTrack<float, int32_t, float>(512, 8192, 0.1f, 1.0f, 0.0f, false, stats, stream, "i32-f32-wide");
    // alpha/beta 组合
    RunAndTrack<float, int32_t, float>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream, "i32-f32-alpha2"); // y = 2*A*x
    RunAndTrack<float, int32_t, float>(512, 1024, 0.9f, 1.0f, 1.0f, false, stats, stream, "i32-f32-beta1");  // y = A*x + y
    RunAndTrack<float, int32_t, float>(512, 1024, 0.9f, 0.5f, 0.5f, false, stats, stream, "i32-f32-half");   // y = 0.5*A*x + 0.5*y
    RunAndTrack<float, int32_t, float>(512, 1024, 0.9f, -1.0f, 0.0f, false, stats, stream, "i32-f32-negA");  // y = -A*x
    RunAndTrack<float, int32_t, float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "i32-f32-mix");   // y = 1.5*A*x + 0.2*y
    RunAndTrack<float, int32_t, float>(512, 1024, 0.9f, 0.0f, 1.0f, false, stats, stream, "i32-f32-betaOnly");
    RunAndTrack<float, int32_t, float>(512, 1024, 0.0f, 0.0f, 2.0f, false, stats, stream, "i32-f32-beta2");

    // ---- Int32→Float32 Transpose ----
    std::cout << "\n======== Int32→Float32 Transpose ========\n";
    RunAndTrack<float, int32_t, float>(512, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-i32-f32-def");
    RunAndTrack<float, int32_t, float>(512, 1024, 0.0f, 1.0f, 0.0f, true, stats, stream, "trans-i32-f32-dense");
    RunAndTrack<float, int32_t, float>(512, 1024, 0.999f, 1.0f, 0.0f, true, stats, stream, "trans-i32-f32-sparse");
    RunAndTrack<float, int32_t, float>(10, 1024, 0.999f, 1.0f, 0.0f, true, stats, stream, "trans-i32-f32-fewrow");
    RunAndTrack<float, int32_t, float>(1, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-i32-f32-1row");
    RunAndTrack<float, int32_t, float>(1024, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-i32-f32-square");
    RunAndTrack<float, int32_t, float>(2048, 512, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-i32-f32-mgreater");
    RunAndTrack<float, int32_t, float>(512, 1024, 0.9f, 2.0f, 0.0f, true, stats, stream, "trans-i32-f32-alpha2");
    RunAndTrack<float, int32_t, float>(512, 1024, 0.9f, 1.0f, 1.0f, true, stats, stream, "trans-i32-f32-beta1");
    RunAndTrack<float, int32_t, float>(512, 1024, 0.9f, -1.0f, 0.0f, true, stats, stream, "trans-i32-f32-negA");
    RunAndTrack<float, int32_t, float>(256, 2048, 0.95f, 1.5f, 0.2f, true, stats, stream, "trans-i32-f32-mix");
    RunAndTrack<float, int32_t, float>(512, 1024, 0.9f, 0.0f, 1.0f, true, stats, stream, "trans-i32-f32-betaOnly");

    // ====================== indexBase=1 用例（float / int32） ======================
    std::cout << "\n"
              << "###########################################################\n"
              << "##              Index-Base-1 Test Suite                  ##\n"
              << "###########################################################\n\n";
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "base1-float-default", 1);
    RunAndTrack<float, float, float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "base1-float-mix", 1);
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "base1-trans-float-def", 1);
    RunAndTrack<float, float, float>(2048, 512, 0.9f, 2.0f, 1.0f, true, stats, stream, "base1-trans-float-mgreater", 1);
    RunAndTrack<int32_t>(512, 1024, 0.9f, 1, 0, false, stats, stream, "base1-int32-default", 1);
    RunAndTrack<int32_t>(256, 2048, 0.95f, 3, -2, false, stats, stream, "base1-int32-mix", 1);
    RunAndTrack<int32_t>(512, 1024, 0.9f, 1, 0, true, stats, stream, "base1-trans-int32-def", 1);
    RunAndTrack<int32_t>(2048, 512, 0.9f, 2, 1, true, stats, stream, "base1-trans-int32-mgreater", 1);

    // ====================== 新 dtype 组合 ======================
    std::cout << "\n"
              << "###########################################################\n"
              << "##              New Dtype-Combo Test Suite               ##\n"
              << "###########################################################\n\n";

    // ---- Int8→Int32 Non-Transpose（exact match） ----
    std::cout << "======== Int8→Int32 Non-Transpose ========\n";
    RunAndTrack<int32_t, int8_t, int32_t>(512, 1024, 0.9f, 1, 0, false, stats, stream, "i8-i32-default");
    RunAndTrack<int32_t, int8_t, int32_t>(512, 1024, 0.0f, 1, 0, false, stats, stream, "i8-i32-dense");
    RunAndTrack<int32_t, int8_t, int32_t>(512, 1024, 0.999f, 1, 0, false, stats, stream, "i8-i32-sparse");
    RunAndTrack<int32_t, int8_t, int32_t>(512, 1024, 0.9f, 2, 0, false, stats, stream, "i8-i32-alpha2");
    RunAndTrack<int32_t, int8_t, int32_t>(512, 1024, 0.9f, 1, 1, false, stats, stream, "i8-i32-beta1");
    RunAndTrack<int32_t, int8_t, int32_t>(256, 2048, 0.95f, 3, -2, false, stats, stream, "i8-i32-mix");
    RunAndTrack<int32_t, int8_t, int32_t>(512, 1024, 0.9f, 0, 1, false, stats, stream, "i8-i32-alpha0beta1");
    RunAndTrack<int32_t, int8_t, int32_t>(512, 1024, 0.9f, 1, 0, false, stats, stream, "i8-i32-base1", 1);
    RunAndTrack<int32_t, int8_t, int32_t>(256, 2048, 0.95f, 3, -2, false, stats, stream, "i8-i32-base1-mix", 1);

    // ---- Int8→Int32 Transpose ----
    std::cout << "\n======== Int8→Int32 Transpose ========\n";
    RunAndTrack<int32_t, int8_t, int32_t>(512, 1024, 0.9f, 1, 0, true, stats, stream, "trans-i8-i32-def");
    RunAndTrack<int32_t, int8_t, int32_t>(512, 1024, 0.9f, 2, 0, true, stats, stream, "trans-i8-i32-alpha2");
    RunAndTrack<int32_t, int8_t, int32_t>(512, 1024, 0.9f, 1, 1, true, stats, stream, "trans-i8-i32-beta1");
    RunAndTrack<int32_t, int8_t, int32_t>(256, 2048, 0.95f, 3, -2, true, stats, stream, "trans-i8-i32-mix");
    RunAndTrack<int32_t, int8_t, int32_t>(1024, 1024, 0.9f, 1, 0, true, stats, stream, "trans-i8-i32-square");
    RunAndTrack<int32_t, int8_t, int32_t>(512, 1024, 0.9f, 1, 0, true, stats, stream, "trans-i8-i32-base1", 1);
    RunAndTrack<int32_t, int8_t, int32_t>(2048, 512, 0.9f, 2, 1, true, stats, stream, "trans-i8-i32-base1-mg", 1);

    // ---- Int8→Float32 Non-Transpose ----
    std::cout << "\n======== Int8→Float32 Non-Transpose ========\n";
    RunAndTrack<float, int8_t, float>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "i8-f32-default");
    RunAndTrack<float, int8_t, float>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "i8-f32-sparse");
    RunAndTrack<float, int8_t, float>(512, 1024, 0.9f, 1.0f, 1.0f, false, stats, stream, "i8-f32-beta1");
    RunAndTrack<float, int8_t, float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "i8-f32-mix");
    RunAndTrack<float, int8_t, float>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "i8-f32-base1", 1);

    // ---- Int8→Float32 Transpose ----
    std::cout << "\n======== Int8→Float32 Transpose ========\n";
    RunAndTrack<float, int8_t, float>(512, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-i8-f32-def");
    RunAndTrack<float, int8_t, float>(256, 2048, 0.95f, 1.5f, 0.2f, true, stats, stream, "trans-i8-f32-mix");
    RunAndTrack<float, int8_t, float>(512, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-i8-f32-base1", 1);

    // ---- Float16→Float16 Non-Transpose ----
    std::cout << "\n======== Float16→Float16 Non-Transpose ========\n";
    RunAndTrack<float, half, half>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "f16-f16-default");
    RunAndTrack<float, half, half>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream, "f16-f16-alpha2");
    RunAndTrack<float, half, half>(512, 1024, 0.9f, 1.0f, 1.0f, false, stats, stream, "f16-f16-beta1");
    RunAndTrack<float, half, half>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "f16-f16-mix");
    RunAndTrack<float, half, half>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "f16-f16-sparse");
    RunAndTrack<float, half, half>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "f16-f16-base1", 1);

    // ---- Float16→Float16 Transpose ----
    std::cout << "\n======== Float16→Float16 Transpose ========\n";
    RunAndTrack<float, half, half>(512, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-f16-f16-def");
    RunAndTrack<float, half, half>(512, 1024, 0.9f, 1.0f, 1.0f, true, stats, stream, "trans-f16-f16-beta1");
    RunAndTrack<float, half, half>(256, 2048, 0.95f, 1.5f, 0.2f, true, stats, stream, "trans-f16-f16-mix");
    RunAndTrack<float, half, half>(512, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-f16-f16-base1", 1);

    // ---- BF16→BF16 Non-Transpose ----
    std::cout << "\n======== BF16→BF16 Non-Transpose ========\n";
    RunAndTrack<float, bfloat16_t, bfloat16_t>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "bf16-bf16-default");
    RunAndTrack<float, bfloat16_t, bfloat16_t>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream, "bf16-bf16-alpha2");
    RunAndTrack<float, bfloat16_t, bfloat16_t>(512, 1024, 0.9f, 1.0f, 1.0f, false, stats, stream, "bf16-bf16-beta1");
    RunAndTrack<float, bfloat16_t, bfloat16_t>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "bf16-bf16-mix");
    RunAndTrack<float, bfloat16_t, bfloat16_t>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "bf16-bf16-base1", 1);

    // ---- BF16→BF16 Transpose ----
    std::cout << "\n======== BF16→BF16 Transpose ========\n";
    RunAndTrack<float, bfloat16_t, bfloat16_t>(512, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-bf16-bf16-def");
    RunAndTrack<float, bfloat16_t, bfloat16_t>(256, 2048, 0.95f, 1.5f, 0.2f, true, stats, stream, "trans-bf16-bf16-mix");
    RunAndTrack<float, bfloat16_t, bfloat16_t>(512, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-bf16-bf16-base1", 1);

    // ====================== complex64 用例 ======================
    std::cout << "\n"
              << "###########################################################\n"
              << "##              Complex64 Test Suite                     ##\n"
              << "###########################################################\n\n";

    // ---- complex64→complex64 Non-Transpose ----
    std::cout << "======== Complex64→Complex64 Non-Transpose ========\n";
    RunC64AndTrack<true>(512, 1024, 0.9f, {1.0f, 0.0f}, {0.0f, 0.0f}, false, stats, stream, "c64-default");
    RunC64AndTrack<true>(512, 1024, 0.9f, {1.5f, -0.5f}, {0.2f, 0.3f}, false, stats, stream, "c64-cplxAB");
    RunC64AndTrack<true>(512, 1024, 0.9f, {0.8f, 0.6f}, {0.0f, 0.0f}, false, stats, stream, "c64-beta0");
    RunC64AndTrack<true>(512, 1024, 0.9f, {0.0f, 0.0f}, {0.5f, -0.25f}, false, stats, stream, "c64-alpha0");
    RunC64AndTrack<true>(512, 1024, 0.999f, {1.0f, 0.0f}, {0.0f, 0.0f}, false, stats, stream, "c64-sparse");
    RunC64AndTrack<true>(512, 1024, 0.9f, {1.0f, 0.0f}, {0.0f, 0.0f}, false, stats, stream, "c64-base1", 1);
    RunC64AndTrack<true>(256, 2048, 0.95f, {1.5f, -0.5f}, {0.2f, 0.3f}, false, stats, stream, "c64-base1-cplxAB", 1);
    RunC64AndTrack<true>(1, 1, 0.0f, {2.0f, -1.0f}, {0.5f, 0.25f}, false, stats, stream,
                         "device-scalar-stream-order-c64", 0, nullptr, nullptr, 5.0f, true);
    RunC64AndTrack<true>(1, 1, 0.0f, {2.0f, -1.0f}, {0.5f, 0.25f}, true, stats, stream,
                         "device-scalar-stream-order-c64-trans", 0, nullptr, nullptr, 5.0f, true);

    // ---- complex64→complex64 Transpose ----
    std::cout << "\n======== Complex64→Complex64 Transpose ========\n";
    RunC64AndTrack<true>(512, 1024, 0.9f, {1.0f, 0.0f}, {0.0f, 0.0f}, true, stats, stream, "trans-c64-default");
    RunC64AndTrack<true>(512, 1024, 0.9f, {1.5f, -0.5f}, {0.2f, 0.3f}, true, stats, stream, "trans-c64-cplxAB");
    RunC64AndTrack<true>(512, 1024, 0.9f, {0.8f, 0.6f}, {0.0f, 0.0f}, true, stats, stream, "trans-c64-beta0");
    RunC64AndTrack<true>(512, 1024, 0.9f, {1.0f, 0.0f}, {0.0f, 0.0f}, true, stats, stream, "trans-c64-base1", 1);

    // ---- FP32→complex64 Non-Transpose ----
    std::cout << "\n======== FP32→Complex64 Non-Transpose ========\n";
    RunC64AndTrack<false>(512, 1024, 0.9f, {1.0f, 0.0f}, {0.0f, 0.0f}, false, stats, stream, "f32-c64-default");
    RunC64AndTrack<false>(512, 1024, 0.9f, {1.5f, -0.5f}, {0.2f, 0.3f}, false, stats, stream, "f32-c64-cplxAB");
    RunC64AndTrack<false>(512, 1024, 0.9f, {0.8f, 0.6f}, {0.0f, 0.0f}, false, stats, stream, "f32-c64-beta0");
    RunC64AndTrack<false>(512, 1024, 0.9f, {1.0f, 0.0f}, {0.0f, 0.0f}, false, stats, stream, "f32-c64-base1", 1);

    // ---- FP32→complex64 Transpose ----
    std::cout << "\n======== FP32→Complex64 Transpose ========\n";
    RunC64AndTrack<false>(512, 1024, 0.9f, {1.0f, 0.0f}, {0.0f, 0.0f}, true, stats, stream, "trans-f32-c64-def");
    RunC64AndTrack<false>(512, 1024, 0.9f, {1.5f, -0.5f}, {0.2f, 0.3f}, true, stats, stream, "trans-f32-c64-cplxAB");
    RunC64AndTrack<false>(512, 1024, 0.9f, {1.0f, 0.0f}, {0.0f, 0.0f}, true, stats, stream, "trans-f32-c64-base1", 1);

    // ====================== 边界用例 ======================
    std::cout << "\n"
              << "###########################################################\n"
              << "##              Boundary / Edge Test Suite               ##\n"
              << "###########################################################\n\n";

    // ---- 长尾行：单行 20000 个非零元（>4096 常规 tile / >2048 复数 tile） ----
    std::cout << "======== Long-tail Row Tests ========\n";
    {
        std::vector<int32_t> ltRowPtr, ltColInd;
        BuildLongTailCsr(128, 20480, 37, 20000, ltRowPtr, ltColInd);
        std::mt19937 rng(42u);
        std::uniform_real_distribution<float> valDist(-5.0f, 5.0f);
        std::vector<float> ltValF(ltColInd.size());
        for (auto &v : ltValF)
            v = valDist(rng);
        RunAndTrack<float, float, float>(128, 20480, 0.0f, 1.0f, 0.0f, false, stats, stream,
                           "longtail-f32", 0, 0.0f, &ltRowPtr, &ltColInd, &ltValF);
        RunAndTrack<float, float, float>(128, 20480, 0.0f, 1.5f, 0.2f, true, stats, stream,
                           "longtail-trans-f32", 0, 0.0f, &ltRowPtr, &ltColInd, &ltValF);
        std::vector<int32_t> ltValI(ltColInd.size());
        for (auto &v : ltValI)
            v = static_cast<int32_t>(valDist(rng));
        RunAndTrack<int32_t>(128, 20480, 0.0f, 1, 0, false, stats, stream,
                             "longtail-i32", 0, 0.0f, &ltRowPtr, &ltColInd, &ltValI);
    }
    {
        // c64 长尾行：单行 10000 个非零元（>2048 复数 tile）；值域收窄到 [-1,1]，
        // 避免 10000 项 fp32 累加的绝对误差超过 maxAbsError=1e-2 上限
        std::vector<int32_t> ltRowPtr, ltColInd;
        BuildLongTailCsr(64, 12000, 13, 10000, ltRowPtr, ltColInd);
        RunC64AndTrack<true>(64, 12000, 0.0f, {1.5f, -0.5f}, {0.2f, 0.3f}, false, stats, stream,
                             "longtail-c64", 0, &ltRowPtr, &ltColInd, 1.0f);
        RunC64AndTrack<true>(64, 12000, 0.0f, {1.0f, 0.0f}, {0.0f, 0.0f}, true, stats, stream,
                             "longtail-trans-c64", 0, &ltRowPtr, &ltColInd, 1.0f);
    }

    // ---- 全空矩阵 nnz=0（sparsity=1.0） ----
    std::cout << "\n======== Empty Matrix (nnz=0) Tests ========\n";
    RunAndTrack<float, float, float>(64, 128, 1.0f, 1.0f, 1.0f, false, stats, stream, "edge-nnz0-beta1");
    RunAndTrack<float, float, float>(64, 128, 1.0f, 2.0f, 0.0f, false, stats, stream, "edge-nnz0-beta0");
    RunAndTrack<float, float, float>(64, 128, 1.0f, 1.0f, 1.0f, true, stats, stream, "edge-trans-nnz0-beta1");
    RunAndTrack<int32_t>(64, 128, 1.0f, 1, 1, false, stats, stream, "edge-nnz0-i32");

    // ---- 整矩阵仅 1 个非零元 ----
    std::cout << "\n======== Single-Nonzero Tests ========\n";
    RunSingleNonzeroRegression(stats, stream);

    // ---- TwoProd 拆分高位积溢出（P2 评审反例，确定性精确值判定） ----
    RunTwoProdRegression(stats, stream);

    // ---- TwoProd 零值行漏入快路径（P2 评审反例，确定性精确值判定） ----
    RunZeroRowRegression(stats, stream);

    // ---- 空行混合（emptyRowProb=0.5） ----
    std::cout << "\n======== Empty-Row Mixed Tests ========\n";
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 1.0f, 0.5f, false, stats, stream, "edge-emptyrow", 0, 0.5f);
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 1.0f, 0.5f, true, stats, stream, "edge-trans-emptyrow", 0, 0.5f);

    // ---- M=1 / N=1 ----
    std::cout << "\n======== M=1 / N=1 Tests ========\n";
    RunAndTrack<float, float, float>(512, 1, 0.9f, 1.0f, 0.0f, false, stats, stream, "edge-1col");
    RunAndTrack<float, float, float>(512, 1, 0.9f, 1.0f, 0.0f, true, stats, stream, "edge-trans-1col");
    RunAndTrack<float, float, float>(1, 1, 0.0f, 2.0f, 1.0f, false, stats, stream, "edge-1x1");

    // ---- alpha=0（y 应等于 beta*y 或 0） ----
    std::cout << "\n======== Alpha=0 Tests ========\n";
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 0.0f, 0.0f, false, stats, stream, "edge-alpha0-beta0");
    RunAndTrack<float, float, float>(512, 1024, 0.9f, 0.0f, 0.0f, true, stats, stream, "edge-trans-alpha0-beta0");
    RunAndTrack<int32_t>(512, 1024, 0.9f, 0, 2, true, stats, stream, "edge-trans-i32-alpha0");

    // ====================== !343 移植场景（重写架构等价性回归） ======================
    std::cout << "\n======== !343 Regression Scenarios ========\n";

    // ---- 全空矩阵转置（nnz=0 时 y = beta*y 仍须真正执行）----
    for (float beta : {0.0f, 1.0f, 2.0f}) {
        RunAndTrack<float, float, float>(512, 513, 1.0f, 1.0f, beta, true, stats, stream,
                                         "mr343-empty-trans-f32");
        RunAndTrack<float, half, half>(512, 513, 1.0f, 1.0f, beta, true, stats, stream,
                                       "mr343-empty-trans-f16");
        RunAndTrack<float, bfloat16_t, bfloat16_t>(512, 513, 1.0f, 1.0f, beta, true, stats, stream,
                                                   "mr343-empty-trans-bf16");
    }

    // ---- 列碰撞转置确定性（固定输入连跑 3 次，bit-wise 一致 + 零容差）----
    for (float beta : {0.0f, 1.0f, 2.0f}) {
        RunTransCollisionDeterminism<float, float, float>(512, 512, beta, 3, stats, stream,
                                                          "mr343-collision-f32");
        RunTransCollisionDeterminism<float, half, half>(512, 512, beta, 3, stats, stream,
                                                        "mr343-collision-f16");
        RunTransCollisionDeterminism<float, bfloat16_t, bfloat16_t>(512, 512, beta, 3, stats, stream,
                                                                    "mr343-collision-bf16");
    }

    // ---- 固定种子转置精度（seed=12345，可复现；sparsity 同上游 !343：每行约 16 个非零元）----
    for (size_t n : {size_t{512}, size_t{4096}}) {
        const float sparsity = 1.0f - 16.0f / n;
        for (float beta : {0.0f, 2.0f}) {
            RunAndTrack<float, half, half>(n, n, sparsity, 1.0f, beta, true, stats, stream,
                                           "mr343-seed-trans-f16", 0, 0.0f, nullptr, nullptr,
                                           nullptr, ACL_SPARSE_SPMV_ALG_DEFAULT, false, false, 12345u);
            RunAndTrack<float, bfloat16_t, bfloat16_t>(n, n, sparsity, 1.0f, beta, true, stats, stream,
                                                       "mr343-seed-trans-bf16", 0, 0.0f, nullptr,
                                                       nullptr, nullptr, ACL_SPARSE_SPMV_ALG_DEFAULT,
                                                       false, false, 12345u);
        }
    }

    // ---- handle workspace 回退（externalBuffer=nullptr）----
    RunTransHandleWorkspaceFallback<half>(stats, stream, "mr343-handlews-f16");
    RunTransHandleWorkspaceFallback<bfloat16_t>(stats, stream, "mr343-handlews-bf16");

    // ---- N=0 参数拒绝（harness 口径，同上游）----
    RunInvalidShapeN0Regression(stats);

    // ====================== 非法参数 / 资源泄漏（单独统计） ======================
    CheckStats paramChecks;
    TestInvalidParams(stream, paramChecks);
    CheckStats leakChecks;
    TestResourceLeak(stream, leakChecks);

    // ====================== 汇总 ======================
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
    std::cout << "----------------------------------------\n";
    std::cout << "Invalid-param checks : " << paramChecks.passed << "/" << paramChecks.total
              << " passed (" << paramChecks.failed << " failed)\n";
    if (!paramChecks.failedChecks.empty()) {
        for (const auto &s : paramChecks.failedChecks)
            std::cout << "  - " << s << "\n";
    }
    std::cout << "Resource-leak checks : " << leakChecks.passed << "/" << leakChecks.total
              << " passed (" << leakChecks.failed << " failed)\n";
    if (!leakChecks.failedChecks.empty()) {
        for (const auto &s : leakChecks.failedChecks)
            std::cout << "  - " << s << "\n";
    }
    std::cout << "========================================\n";

    Finalize(deviceId, stream);
    const int totalFailed = stats.failed + paramChecks.failed + leakChecks.failed;
    return (totalFailed == 0) ? 0 : 1;
}
