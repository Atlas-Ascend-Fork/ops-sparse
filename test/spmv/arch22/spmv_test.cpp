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
#include <limits>
#include "acl/acl.h"
#include "cann_ops_sparse.h"

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
        union {
            float f;
            uint32_t u;
        } x = {f};
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
                 float emptyRowProb = 0.0f) {
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

    std::mt19937 rng(static_cast<uint32_t>(
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
    std::vector<OutT> z = yVec;
    for (uint32_t j = 0; j < numCols; ++j)
        z[j] = static_cast<OutT>(static_cast<CompT>(beta) * static_cast<CompT>(z[j]));
    for (uint32_t i = 0; i < numRows; ++i) {
        CompT xVal = static_cast<CompT>(alpha) * static_cast<CompT>(xVec[i]);
        if (xVal == static_cast<CompT>(0))
            continue;
        for (uint32_t k = csrRowPtr[i]; k < csrRowPtr[i + 1]; ++k)
            z[csrColInd[k]] = static_cast<OutT>(
                static_cast<CompT>(z[csrColInd[k]]) + xVal * static_cast<CompT>(csrVal[k]));
    }
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
        std::cout << "golden[" << i << "]=" << cpuGolden[i]
                  << " npu_result[" << i << "]=" << npuRet[i] << "\n";
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
    // 全零矩阵的 colInd / vals 为空数组，但描述符仍要求非空设备指针，
    // 而 aclrtMalloc(0) 会返回 ACL_ERROR_INVALID_PARAM，故至少申请 1 字节。
    const size_t allocSize = size > 0 ? size : 1;
    auto ret = aclrtMalloc((void **)deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    if (size == 0) {
        return ACL_SUCCESS;
    }
    ret = aclrtMemcpy(*deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
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

// ===================== 主测试函数 =====================

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
         std::string *worstInfo = nullptr,
         aclsparseSpMVAlg_t alg = ACL_SPARSE_SPMV_ALG_DEFAULT,
         bool productsInValT = false,
         bool deviceScalarStreamOrder = false) {
    const char *label = transpose ? " Transpose"
                                  : (alg == ACL_SPARSE_SPMV_CSR_ALG_CUBE ? " Cube" : "");
    std::cout << "====Test" << label << " case: row num = " << M
              << " col num = " << N
              << " sparsity (zero ratio) = " << sparsity
              << " alpha = " << alphaVal
              << " beta = " << betaVal << "====\n";

    std::vector<ValT> csrVal;
    std::vector<int32_t> csrColInd, csrRowPtr;
    GenerateCsr<ValT>(M, N, sparsity, csrRowPtr, csrColInd, csrVal);

    uint32_t xSize = transpose ? static_cast<uint32_t>(M) : static_cast<uint32_t>(N);
    uint32_t ySize = transpose ? static_cast<uint32_t>(N) : static_cast<uint32_t>(M);

    std::vector<ValT> xVec;
    std::vector<OutT> yVec; {
        std::mt19937 rng(static_cast<uint32_t>(
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

    size_t totalRowPtrByteSize = csrRowPtr.size() * sizeof(int32_t);
    size_t totalColIdxByteSize = csrColInd.size() * sizeof(int32_t);
    size_t totalValsByteSize = csrVal.size() * sizeof(ValT);
    size_t totalXByteSize = xSize * sizeof(ValT);
    size_t totalYByteSize = ySize * sizeof(OutT);

    aclsparseHandle_t spHandle = nullptr;
    aclsparseStatus_t sparseRet = aclsparseCreate(&spHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    sparseRet = aclsparseSetStream(spHandle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    uint8_t *rowPtrDevice = nullptr, *colIdxDevice = nullptr;
    uint8_t *valsDevice = nullptr, *xDevice = nullptr, *yDevice = nullptr, *yHost = nullptr;

    int aclRet = CreateDeviceTensor((uint8_t *)csrRowPtr.data(), totalRowPtrByteSize, &rowPtrDevice);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, aclError (*)(void *)> rpp(rowPtrDevice, aclrtFree);

    aclRet = CreateDeviceTensor((uint8_t *)csrColInd.data(), totalColIdxByteSize, &colIdxDevice);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, aclError (*)(void *)> cip(colIdxDevice, aclrtFree);

    aclRet = CreateDeviceTensor((uint8_t *)csrVal.data(), totalValsByteSize, &valsDevice);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, aclError (*)(void *)> vdp(valsDevice, aclrtFree);

    aclRet = CreateDeviceTensor((uint8_t *)xVec.data(), totalXByteSize, &xDevice);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, aclError (*)(void *)> xdp(xDevice, aclrtFree);

    aclRet = CreateDeviceTensor((uint8_t *)yVec.data(), totalYByteSize, &yDevice);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, aclError (*)(void *)> ydp(yDevice, aclrtFree);

    aclRet = aclrtMallocHost((void **)(&yHost), totalYByteSize);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, aclError (*)(void *)> yhp(yHost, aclrtFreeHost);

    aclDataType valDt = AclTypeOf<ValT>();
    aclDataType outDt = AclTypeOf<OutT>();
    aclDataType compDt = std::is_same_v<CompT, float> ? ACL_FLOAT : ACL_INT32;

    aclsparseSpMatDescr_t matDesc = nullptr;
    sparseRet = aclsparseCreateCsr(&matDesc, M, N, csrColInd.size(),
                                   rowPtrDevice, colIdxDevice, valsDevice,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, valDt);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateCsr failed\n");
              return sparseRet);

    aclsparseDnVecDescr_t vecXDesc = nullptr;
    sparseRet = aclsparseCreateDnVec(&vecXDesc, xSize, xDevice, valDt);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    aclsparseDnVecDescr_t vecYDesc = nullptr;
    sparseRet = aclsparseCreateDnVec(&vecYDesc, ySize, yDevice, outDt);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    aclsparseOperation_t op = transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;

    // cube 版按 aclsparseSpMVGetBufferSize 报出的大小申请 workspace；
    // 向量版返回 0，此时 externalBuffer 保持为空。
    size_t bufferSize = 0;
    sparseRet = aclsparseSpMVGetBufferSize(spHandle, op, &alphaVal, matDesc, vecXDesc,
                                           &betaVal, vecYDesc, compDt, alg, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSpMVGetBufferSize failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    void *externalBuffer = nullptr;
    if (bufferSize > 0) {
        aclRet = aclrtMalloc(&externalBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] externalBuffer aclrtMalloc failed, ret=%d\n", aclRet);
                  return aclRet);
    }
    std::unique_ptr<void, aclError (*)(void *)> ebp(externalBuffer, aclrtFree);

    const void *alphaPtr = &alphaVal;
    const void *betaPtr = &betaVal;
    void *deviceAlpha = nullptr;
    void *deviceBeta = nullptr;
    void *hostAlphaUpdate = nullptr;
    void *hostBetaUpdate = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> dap(nullptr, aclrtFree);
    std::unique_ptr<void, aclError (*)(void *)> dbp(nullptr, aclrtFree);
    std::unique_ptr<void, aclError (*)(void *)> hap(nullptr, aclrtFreeHost);
    std::unique_ptr<void, aclError (*)(void *)> hbp(nullptr, aclrtFreeHost);
    if (deviceScalarStreamOrder) {
        const CompT initialAlpha = static_cast<CompT>(1);
        const CompT initialBeta = static_cast<CompT>(0);
        aclRet = aclrtMalloc(&deviceAlpha, sizeof(CompT), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
        dap.reset(deviceAlpha);
        aclRet = aclrtMalloc(&deviceBeta, sizeof(CompT), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
        dbp.reset(deviceBeta);
        aclRet = aclrtMemcpy(deviceAlpha, sizeof(CompT), &initialAlpha, sizeof(CompT), ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
        aclRet = aclrtMemcpy(deviceBeta, sizeof(CompT), &initialBeta, sizeof(CompT), ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
        aclRet = aclrtMallocHost(&hostAlphaUpdate, sizeof(CompT));
        CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
        hap.reset(hostAlphaUpdate);
        aclRet = aclrtMallocHost(&hostBetaUpdate, sizeof(CompT));
        CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
        hbp.reset(hostBetaUpdate);
        *static_cast<CompT *>(hostAlphaUpdate) = alphaVal;
        *static_cast<CompT *>(hostBetaUpdate) = betaVal;
        sparseRet = aclsparseSetPointerMode(spHandle, ACL_SPARSE_POINTER_MODE_DEVICE);
        CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);
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
    }

    sparseRet = aclsparseSpMV(spHandle, op, alphaPtr, matDesc, vecXDesc,
                              betaPtr, vecYDesc, compDt,
                              alg, externalBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return sparseRet);

    // ==================== 同步 ====================
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclsparseSpMV: aclrtSynchronizeStream failed, ret=%d\n", aclRet);
                return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    aclRet = aclrtMemcpy(yHost, totalYByteSize, yDevice, totalYByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::vector<OutT> yResult;
    yResult.assign((OutT *)yHost, (OutT *)(yHost + totalYByteSize));

    // 乘积在输入精度下计算时（cube 版 fp16），按输入类型而非输出类型取容差。
    const MixedToleranceParams toleranceParams =
        productsInValT ? GetMixedToleranceParams<ValT>() : GetMixedToleranceParams<OutT>();
    int verifyRet = Verification<OutT>(output_cpu, yResult, MARE, MERE, toleranceParams, worstInfo);

    aclsparseDestroySpMat(matDesc);
    aclsparseDestroyDnVec(vecXDesc);
    aclsparseDestroyDnVec(vecYDesc);
    aclsparseDestroy(spHandle);

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

template <typename CompT, typename ValT = CompT, typename OutT = CompT>
int RunAndTrack(size_t M, size_t N, float sparsity, CompT alpha, CompT beta,
                bool transpose,
                TestStats &stats, aclrtStream stream, const std::string &tag = "",
                aclsparseSpMVAlg_t alg = ACL_SPARSE_SPMV_ALG_DEFAULT,
                bool productsInValT = false,
                bool deviceScalarStreamOrder = false) {
    float MARE = 0, MERE = 0;
    std::string worstInfo;
    int ret = Test<CompT, ValT, OutT>(M, N, sparsity, alpha, beta, transpose, MARE, MERE, stream,
                                      &worstInfo, alg, productsInValT, deviceScalarStreamOrder);

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
    std::uniform_int_distribution<int> rowDist(1, 2048);
    std::uniform_int_distribution<int> colDist(1, 4096);
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
        RunAndTrack<T>(M, N, sp, a, b, transpose, stats, stream,
                       tagPrefix + std::to_string(i + 1), alg);
    }
}

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

    std::cout << "\n"
              << "###########################################################\n"
              << "##               SpMV Test Suite (aclsparse)              ##\n"
              << "###########################################################\n\n";

    // ============ Float 基础用例 ============
    std::cout << "======== Float Basic Tests (alpha=1.0, beta=0.0) ========\n";
    RunAndTrack<float>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "float-default");
    RunAndTrack<float>(512, 1024, 0.0f, 1.0f, 0.0f, false, stats, stream, "float-dense");
    RunAndTrack<float>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "float-sparse");
    RunAndTrack<float>(10, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "float-fewrow");
    RunAndTrack<float>(1, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "float-1row");
    RunAndTrack<float>(512, 8192, 0.1f, 1.0f, 0.0f, false, stats, stream, "float-wide");

    // ============ Float Alpha/Beta 组合测试 ============
    std::cout << "\n======== Float Alpha/Beta Tests ========\n";
    RunAndTrack<float>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream, "float-alpha2"); // y = 2*A*x
    RunAndTrack<float>(512, 1024, 0.9f, 1.0f, 1.0f, false, stats, stream, "float-beta1");  // y = A*x + y
    RunAndTrack<float>(512, 1024, 0.9f, 0.5f, 0.5f, false, stats, stream, "float-half");   // y = 0.5*A*x + 0.5*y
    RunAndTrack<float>(512, 1024, 0.9f, -1.0f, 0.0f, false, stats, stream, "float-negA");  // y = -A*x
    RunAndTrack<float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "float-mix");   // y = 1.5*A*x + 0.2*y

    // ============ Float Alpha=0, beta=1 (纯 pass-through) ============
    RunAndTrack<float>(512, 1024, 0.9f, 0.0f, 1.0f, false, stats, stream, "float-betaOnly");
    RunAndTrack<float>(512, 1024, 0.0f, 0.0f, 2.0f, false, stats, stream, "float-beta2");

    // ============ Float 随机参数采样 ============
    RunRandomParams<float>(30, "float-rand-", false, stats, stream,
                           std::uniform_real_distribution<float>(-2.0f, 2.0f),
                           std::uniform_real_distribution<float>(-1.0f, 1.0f));

    // ============ Cube 算法 fp32 用例（ACL_SPARSE_SPMV_CSR_ALG_CUBE，仅非转置）============
    std::cout << "\n======== Float Cube Algorithm Tests ========\n";
    constexpr aclsparseSpMVAlg_t kCube = ACL_SPARSE_SPMV_CSR_ALG_CUBE;
    RunAndTrack<float>(1, 1, 0.0f, 2.0f, 1.0f, false, stats, stream,
                       "device-scalar-stream-order-float-cube", kCube, false, true);
    RunAndTrack<int32_t>(1, 1, 0.0f, 2, 1, false, stats, stream,
                         "device-scalar-stream-order-int32-vector",
                         ACL_SPARSE_SPMV_ALG_DEFAULT, false, true);
    RunAndTrack<float>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "cube-default", kCube);
    RunAndTrack<float>(512, 1024, 0.0f, 1.0f, 0.0f, false, stats, stream, "cube-dense", kCube);
    RunAndTrack<float>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "cube-sparse", kCube);
    RunAndTrack<float>(10, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "cube-fewrow", kCube);
    RunAndTrack<float>(1, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "cube-1row", kCube);
    RunAndTrack<float>(512, 8192, 0.1f, 1.0f, 0.0f, false, stats, stream, "cube-wide", kCube);
    RunAndTrack<float>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream, "cube-alpha2", kCube);
    RunAndTrack<float>(512, 1024, 0.9f, 0.5f, 0.5f, false, stats, stream, "cube-half", kCube);
    RunAndTrack<float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "cube-mix", kCube);
    RunAndTrack<float>(512, 1024, 0.9f, 0.0f, 1.0f, false, stats, stream, "cube-betaOnly", kCube);
    // 全零矩阵（nnz == 0）：cube 版退化为 y = beta * y，由向量版 kernel 接管。
    RunAndTrack<float>(512, 1024, 1.0f, 1.0f, 0.5f, false, stats, stream, "cube-empty", kCube);
    RunRandomParams<float>(10, "cube-rand-", false, stats, stream,
                           std::uniform_real_distribution<float>(-2.0f, 2.0f),
                           std::uniform_real_distribution<float>(-1.0f, 1.0f), kCube);

    // ============ Cube 算法 fp16 用例（矩阵/x 为 fp16，cube 以 fp32 累加，y 为 fp32）============
    // 乘积在 fp16 下计算，故参考实现同样把乘积舍入回 fp16，并按 fp16 容差校验。
    std::cout << "\n======== Float16 Cube Algorithm Tests ========\n";
    constexpr bool kProductsInF16 = true;
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "cube-f16-default", kCube, kProductsInF16);
    RunAndTrack<float, half, float>(512, 1024, 0.0f, 1.0f, 0.0f, false, stats, stream, "cube-f16-dense", kCube, kProductsInF16);
    RunAndTrack<float, half, float>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "cube-f16-sparse", kCube, kProductsInF16);
    RunAndTrack<float, half, float>(10, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "cube-f16-fewrow", kCube, kProductsInF16);
    RunAndTrack<float, half, float>(1, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "cube-f16-1row", kCube, kProductsInF16);
    RunAndTrack<float, half, float>(512, 8192, 0.1f, 1.0f, 0.0f, false, stats, stream, "cube-f16-wide", kCube, kProductsInF16);
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream, "cube-f16-alpha2", kCube, kProductsInF16);
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 0.5f, 0.5f, false, stats, stream, "cube-f16-half", kCube, kProductsInF16);
    RunAndTrack<float, half, float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "cube-f16-mix", kCube, kProductsInF16);
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 0.0f, 1.0f, false, stats, stream, "cube-f16-betaOnly", kCube, kProductsInF16);

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
    RunAndTrack<float>(512, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-float-default");
    RunAndTrack<float>(512, 1024, 0.0f, 1.0f, 0.0f, true, stats, stream, "trans-float-dense");
    RunAndTrack<float>(512, 1024, 0.999f, 1.0f, 0.0f, true, stats, stream, "trans-float-sparse");
    RunAndTrack<float>(10, 1024, 0.999f, 1.0f, 0.0f, true, stats, stream, "trans-float-fewrow");
    RunAndTrack<float>(1, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-float-1row");
    RunAndTrack<float>(512, 8192, 0.1f, 1.0f, 0.0f, true, stats, stream, "trans-float-wide");
    // 方阵
    RunAndTrack<float>(1024, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-float-square");
    // 行列不等，交换测试 —— 用于覆盖 M < N 和 M > N 场景
    RunAndTrack<float>(2048, 512, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-float-mgreater");

    // ============ Float Transpose Alpha/Beta 组合 ============
    std::cout << "\n======== Float Transpose Alpha/Beta Tests ========\n";
    RunAndTrack<float>(512, 1024, 0.9f, 2.0f, 0.0f, true, stats, stream, "trans-float-alpha2");
    RunAndTrack<float>(512, 1024, 0.9f, 1.0f, 1.0f, true, stats, stream, "trans-float-beta1");
    RunAndTrack<float>(512, 1024, 0.9f, 0.5f, 0.5f, true, stats, stream, "trans-float-half");
    RunAndTrack<float>(512, 1024, 0.9f, -1.0f, 0.0f, true, stats, stream, "trans-float-negA");
    RunAndTrack<float>(256, 2048, 0.95f, 1.5f, 0.2f, true, stats, stream, "trans-float-mix");
    RunAndTrack<float>(512, 1024, 0.9f, 0.0f, 1.0f, true, stats, stream, "trans-float-betaOnly");
    RunAndTrack<float>(512, 1024, 0.0f, 0.0f, 2.0f, true, stats, stream, "trans-float-beta2");

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
    std::cout << "========================================\n";

    Finalize(deviceId, stream);
    return (stats.failed == 0) ? 0 : 1;
}
