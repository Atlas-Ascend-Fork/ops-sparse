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

/**
 * @file scatter_test.cpp
 * @brief GTest + CSV-driven test cases for aclsparseScatter (Generic API).
 *
 * Tests sparse vector → dense vector scatter:
 *   Y[indices[i] - idxBase] = values[i]   (byte-exact copy)
 *
 * Test structure:
 *   - TEST_P (ScatterTest)          : parameterized success-path tests from CSV
 *   - TEST_F (ScatterExceptionTest) : invalid-param error tests (L2)
 *   - TEST_F (ScatterDuplicateIdxTest) : duplicate-index informational test (L2)
 *
 * Precision: EXACT (atol=0, rtol=0), bitwise comparison via byte-level golden.
 * For special-value cases, an additional memcmp is applied to catch ±0 sign bit
 * and NaN payload differences that EXACT float comparison might miss.
 */

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "test_common.h"
#include "scatter_golden.h"
#include "scatter_npu_wrapper.h"
#include "scatter_param.h"

using sparse_test::AclEnvScope;
using sparse_test::ComputeScatterGolden;
using sparse_test::DeviceBuffer;
using sparse_test::DnVecManager;
using sparse_test::GetCasesFromCsv;
using sparse_test::HandleManager;
using sparse_test::PrecisionMode;
using sparse_test::ScatterAclDataTypeSize;
using sparse_test::ScatterCallFields;
using sparse_test::ScatterNpu;
using sparse_test::ScatterNpuArgs;
using sparse_test::ScatterNpuResult;
using sparse_test::ScatterParam;
using sparse_test::SpVecManager;
using sparse_test::Verifier;
using sparse_test::VerifyConfig;

// ============================================================================
// Bit manipulation helpers (same style as csr2csc_ex2_test.cpp)
//
// TODO: FloatToBits / BitsToFloat / FloatToHalf / FloatToBf16 等位转换工具与
// csr2csc_ex2 等算子重复，属框架级缺失公共位转换工具的问题。待测试框架补齐
// 公共位转换工具后，各算子应统一改用框架公共实现，删除本算子内的重复定义。
// ============================================================================

static uint32_t FloatToBits(float f) {
    union { float f; uint32_t u; } cvt;
    cvt.f = f;
    return cvt.u;
}

static float BitsToFloat(uint32_t u) {
    union { uint32_t u; float f; } cvt;
    cvt.u = u;
    return cvt.f;
}

static void StoreBytes(uint8_t* dst, uint32_t bits, size_t numBytes) {
    for (size_t i = 0; i < numBytes; i++) {
        dst[i] = static_cast<uint8_t>((bits >> (i * 8)) & 0xFFu);
    }
}

static uint32_t LoadBytes(const uint8_t* src, size_t numBytes) {
    uint32_t bits = 0;
    for (size_t i = 0; i < numBytes; i++) {
        bits |= static_cast<uint32_t>(src[i]) << (i * 8);
    }
    return bits;
}

static uint16_t FloatToHalf(float f) {
    uint32_t x = FloatToBits(f);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

static uint16_t FloatToBf16(float f) {
    return static_cast<uint16_t>(FloatToBits(f) >> 16);
}

// Copy POD array raw bytes into a uint8_t buffer (memcpy-free, type-safe).
template <typename T>
static void CopyPodBytes(uint8_t* dst, const T* src, size_t count) {
    const char* srcBegin = reinterpret_cast<const char*>(src);
    const size_t bytes = count * sizeof(T);
    std::copy(srcBegin, srcBegin + bytes, reinterpret_cast<char*>(dst));
}

static float HalfToFloat(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int32_t shifts = 0;
            while ((mant & 0x400u) == 0) { mant <<= 1; shifts++; }
            mant &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(113 - shifts) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    return BitsToFloat(bits);
}

static float Bf16ToFloat(uint16_t b) {
    return BitsToFloat(static_cast<uint32_t>(b) << 16);
}

// ============================================================================
// Type parsing helpers
// ============================================================================

static aclDataType ParseValType(const std::string& s) {
    if (s == "FP32" || s == "FLOAT") return ACL_FLOAT;
    if (s == "FP16" || s == "FLOAT16") return ACL_FLOAT16;
    if (s == "BF16") return ACL_BF16;
    if (s == "INT8" || s == "I8") return ACL_INT8;
    if (s == "COMPLEX64" || s == "C64") return ACL_COMPLEX64;
    return ACL_FLOAT;
}

static aclsparseIndexType_t ParseIdxType(const std::string& s) {
    if (s == "I64") return ACL_SPARSE_INDEX_64I;
    return ACL_SPARSE_INDEX_32I;
}

static aclsparseStatus_t ParseExpectResult(const std::string& s) {
    if (s == "SUCCESS")        return ACL_SPARSE_STATUS_SUCCESS;
    if (s == "INVALID_VALUE")  return ACL_SPARSE_STATUS_INVALID_VALUE;
    if (s == "NOT_SUPPORTED")  return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Special value bit patterns per dtype (from test plan §3.2)
// ============================================================================

struct SpecialValuePattern {
    uint32_t fp32Bits;
    uint16_t fp16Bits;
    uint16_t bf16Bits;
    const char* name;
};

static const SpecialValuePattern kSpecialValues[] = {
    {0x00000000, 0x0000, 0x0000, "+0"},
    {0x80000000, 0x8000, 0x8000, "-0"},
    {0x7F800000, 0x7C00, 0x7F80, "+inf"},
    {0xFF800000, 0xFC00, 0xFF80, "-inf"},
    {0x7FC00000, 0x7E00, 0x7FC0, "nan"},
    {0x7F7FFFFF, 0x7BFF, 0x7F7F, "max_normal"},
    {0x00800000, 0x0400, 0x0080, "min_normal"},
    {0x00000001, 0x0001, 0x0001, "min_subnormal"},
};

static constexpr size_t kNumSpecialValues = sizeof(kSpecialValues) / sizeof(kSpecialValues[0]);

// Extreme value patterns (large/small magnitude mixing)
static const SpecialValuePattern kExtremeValues[] = {
    {0x7F7FFFFF, 0x7BFF, 0x7F7F, "max_normal"},
    {0x00800000, 0x0400, 0x0080, "min_normal"},
    {0x49742400, 0x4A00, 0x4974, "1e5"},
    {0x37800000, 0x0400, 0x3780, "1e-5"},
    {0x00000001, 0x0001, 0x0001, "min_subnormal"},
    {0x7F800000, 0x7C00, 0x7F80, "+inf"},
};
static constexpr size_t kNumExtremeValues = sizeof(kExtremeValues) / sizeof(kExtremeValues[0]);

// ============================================================================
// Data generation: sparse vector indices + values, and dense Y init
// ============================================================================

struct ScatterHostData {
    std::vector<uint8_t> indicesBytes;  // nnz * idxSize bytes
    std::vector<uint8_t> valuesBytes;   // nnz * valSize bytes
    std::vector<uint8_t> yInitBytes;    // dnSize * valSize bytes
    size_t valSize = 0;
    size_t idxSize = 0;
};

// Store a value bit pattern into byte buffer at given offset, little-endian
static void StoreValueBytes(uint8_t* dst, const SpecialValuePattern& bits,
                             aclDataType valType, size_t valSize) {
    switch (valType) {
        case ACL_FLOAT16:
            StoreBytes(dst, bits.fp16Bits, valSize);
            break;
        case ACL_BF16:
            StoreBytes(dst, bits.bf16Bits, valSize);
            break;
        case ACL_INT8: {
            // Map float bits → int8 range for ST coverage.
            int8_t v = static_cast<int8_t>(static_cast<int32_t>(bits.fp32Bits) & 0xFF);
            if (v == 0) {
                v = 1;
            }
            dst[0] = static_cast<uint8_t>(v);
            break;
        }
        case ACL_COMPLEX64: {
            // Pack (re, im) as two float32 words; im uses bf16-expanded float for diversity.
            StoreBytes(dst, bits.fp32Bits, sizeof(float));
            StoreBytes(dst + sizeof(float), FloatToBits(Bf16ToFloat(bits.bf16Bits)), sizeof(float));
            break;
        }
        case ACL_FLOAT:
        default:
            StoreBytes(dst, bits.fp32Bits, valSize);
            break;
    }
}

// Generate unique random indices in [idxBase, idxBase + size - 1]
// If sorted=true, sort ascending; if false, shuffle.
template <typename IdxT, typename RNG>
static std::vector<IdxT> GenerateUniqueIndices(int64_t nnz, int64_t size, int idxBase,
                                                bool sorted, RNG& rng) {
    if (nnz == 0 || size == 0) return {};
    std::vector<IdxT> pool(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; i++) {
        pool[static_cast<size_t>(i)] = static_cast<IdxT>(i + idxBase);
    }
    // Fisher-Yates partial shuffle to pick nnz unique indices
    std::vector<IdxT> selected;
    selected.reserve(static_cast<size_t>(nnz));
    for (int64_t i = 0; i < nnz && i < size; i++) {
        std::uniform_int_distribution<int64_t> dist(i, size - 1);
        int64_t j = dist(rng);
        std::swap(pool[static_cast<size_t>(i)], pool[static_cast<size_t>(j)]);
        selected.push_back(pool[static_cast<size_t>(i)]);
    }
    if (sorted) {
        std::sort(selected.begin(), selected.end());
    } else {
        std::shuffle(selected.begin(), selected.end(), rng);
    }
    return selected;
}

// Check if this is a boundary test case (needs exact min/max indices)
static bool IsBoundaryCase(const std::string& caseName) {
    return caseName.find("idx_boundary") != std::string::npos;
}

// Generate boundary indices: first=min, second=max, pad with unique random values.
template <typename IdxT, typename RNG>
static std::vector<IdxT> GenerateBoundaryIndices(int64_t nnz, int64_t size, int idxBase,
                                                   bool sorted, RNG& rng) {
    // G.EXP.22: guard against division by zero in `rng() % size` when size == 0.
    if (size <= 0) return {};
    std::vector<IdxT> indices = {
        static_cast<IdxT>(idxBase),                    // min
        static_cast<IdxT>(size - 1 + idxBase)          // max
    };
    // Pad with additional unique indices if nnz > 2
    while (static_cast<int64_t>(indices.size()) < nnz) {
        std::uniform_int_distribution<int64_t> dist(0, size - 1);
        IdxT candidate = static_cast<IdxT>(dist(rng)) + idxBase;
        bool dup = false;
        for (auto v : indices) { if (v == candidate) { dup = true; break; } }
        if (!dup) indices.push_back(candidate);
    }
    if (sorted) std::sort(indices.begin(), indices.end());
    return indices;
}

// Generate indices bytes for the sparse vector into `out`.
static void GenerateIndicesBytes(const ScatterParam& p, aclsparseIndexType_t idxType,
                                   size_t idxSize, std::mt19937& rng,
                                   std::vector<uint8_t>& out) {
    out.resize(static_cast<size_t>(p.vec_nnz) * idxSize);
    if (IsBoundaryCase(p.case_name) && p.vec_nnz >= 2) {
        if (idxType == ACL_SPARSE_INDEX_64I) {
            auto indices = GenerateBoundaryIndices<int64_t>(
                p.vec_nnz, p.vec_size, p.idx_base, p.idx_sorted, rng);
            CopyPodBytes(out.data(), indices.data(), indices.size());
        } else {
            auto indices = GenerateBoundaryIndices<int32_t>(
                p.vec_nnz, p.vec_size, p.idx_base, p.idx_sorted, rng);
            CopyPodBytes(out.data(), indices.data(), indices.size());
        }
    } else if (idxType == ACL_SPARSE_INDEX_64I) {
        auto indices = GenerateUniqueIndices<int64_t>(p.vec_nnz, p.vec_size, p.idx_base,
                                                        p.idx_sorted, rng);
        if (!indices.empty()) {
            CopyPodBytes(out.data(), indices.data(), indices.size());
        }
    } else {
        auto indices = GenerateUniqueIndices<int32_t>(p.vec_nnz, p.vec_size, p.idx_base,
                                                       p.idx_sorted, rng);
        if (!indices.empty()) {
            CopyPodBytes(out.data(), indices.data(), indices.size());
        }
    }
}

// Generate values bytes based on value pattern into `out`.
static void GenerateValuesBytes(const ScatterParam& p, aclDataType valType,
                                  size_t valSize, std::mt19937& rng,
                                  std::vector<uint8_t>& out) {
    out.resize(static_cast<size_t>(p.vec_nnz) * valSize);
    if (p.value_pattern == "special") {
        for (int64_t i = 0; i < p.vec_nnz; i++) {
            const auto& sv = kSpecialValues[static_cast<size_t>(i) % kNumSpecialValues];
            StoreValueBytes(&out[static_cast<size_t>(i) * valSize], sv, valType, valSize);
        }
    } else if (p.value_pattern == "extreme") {
        for (int64_t i = 0; i < p.vec_nnz; i++) {
            const auto& sv = kExtremeValues[static_cast<size_t>(i) % kNumExtremeValues];
            StoreValueBytes(&out[static_cast<size_t>(i) * valSize], sv, valType, valSize);
        }
    } else {
        // normal: random [-2, 2] converted to target type
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (int64_t i = 0; i < p.vec_nnz; i++) {
            float fv = dist(rng);
            uint32_t fp32Bits = FloatToBits(fv);
            uint16_t fp16Bits = FloatToHalf(fv);
            uint16_t bf16Bits = FloatToBf16(fv);
            SpecialValuePattern bits{fp32Bits, fp16Bits, bf16Bits, "rnd"};
            StoreValueBytes(&out[static_cast<size_t>(i) * valSize], bits, valType, valSize);
        }
    }
}

// Generate Y init bytes (non-zero random to detect false writes) into `out`.
static void GenerateYInitBytes(const ScatterParam& p, aclDataType valType,
                                 size_t valSize, std::mt19937& rng,
                                 std::vector<uint8_t>& out) {
    out.resize(static_cast<size_t>(p.dn_size) * valSize);
    std::uniform_real_distribution<float> yDist(-5.0f, 10.0f);
    for (int64_t i = 0; i < p.dn_size; i++) {
        float fv = yDist(rng);
        // Avoid 0.0 to ensure unscattered positions are detectable
        if (fv == 0.0f) fv = 1.0f;
        uint32_t fp32Bits = FloatToBits(fv);
        uint16_t fp16Bits = FloatToHalf(fv);
        uint16_t bf16Bits = FloatToBf16(fv);
        SpecialValuePattern bits{fp32Bits, fp16Bits, bf16Bits, "yinit"};
        StoreValueBytes(&out[static_cast<size_t>(i) * valSize], bits, valType, valSize);
    }
}

// Generate scatter host data from test parameters.
// Delegates to GenerateIndicesBytes / GenerateValuesBytes / GenerateYInitBytes
// so that no single function exceeds the NBNC limit.
static ScatterHostData GenerateScatterData(
    const ScatterParam& p, aclDataType valType, aclsparseIndexType_t idxType)
{
    ScatterHostData data;
    data.valSize = ScatterAclDataTypeSize(valType);
    data.idxSize = (idxType == ACL_SPARSE_INDEX_64I) ? sizeof(int64_t) : sizeof(int32_t);

    std::mt19937 rng(p.seed);
    GenerateIndicesBytes(p, idxType, data.idxSize, rng, data.indicesBytes);
    GenerateValuesBytes(p, valType, data.valSize, rng, data.valuesBytes);
    GenerateYInitBytes(p, valType, data.valSize, rng, data.yInitBytes);

    return data;
}

// ============================================================================
// Convert byte buffer to float vector (for Verifier::verifyVector EXACT)
// ============================================================================

static std::vector<float> BytesToFloatVec(const std::vector<uint8_t>& bytes,
                                           aclDataType valType)
{
    size_t valSize = ScatterAclDataTypeSize(valType);
    if (valSize == 0 || bytes.empty()) return {};
    // COMPLEX64: expand to 2 floats per element for Verifier EXACT path.
    const bool isC64 = (valType == ACL_COMPLEX64);
    size_t count = bytes.size() / valSize;
    std::vector<float> out(isC64 ? count * 2 : count);
    for (size_t i = 0; i < count; i++) {
        const uint8_t* src = &bytes[i * valSize];
        switch (valType) {
            case ACL_FLOAT16:
                out[i] = HalfToFloat(static_cast<uint16_t>(LoadBytes(src, sizeof(uint16_t))));
                break;
            case ACL_BF16:
                out[i] = Bf16ToFloat(static_cast<uint16_t>(LoadBytes(src, sizeof(uint16_t))));
                break;
            case ACL_INT8:
                out[i] = static_cast<float>(static_cast<int8_t>(src[0]));
                break;
            case ACL_COMPLEX64:
                out[i * 2] = BitsToFloat(LoadBytes(src, sizeof(float)));
                out[i * 2 + 1] = BitsToFloat(LoadBytes(src + sizeof(float), sizeof(float)));
                break;
            case ACL_FLOAT:
            default:
                out[i] = BitsToFloat(LoadBytes(src, sizeof(float)));
                break;
        }
    }
    return out;
}

// ============================================================================
// GTest parameterized fixture: ScatterTest
// ============================================================================

// Print test case header for console output.
static void PrintCaseHeader(const ScatterParam& p) {
    std::cout << "==== " << p.case_name
              << " ==== size=" << p.vec_size << " nnz=" << p.vec_nnz
              << " dn=" << p.dn_size
              << " valType=" << p.val_type
              << " idxType=" << p.idx_type
              << " idxBase=" << p.idx_base
              << " sorted=" << (p.idx_sorted ? "true" : "false")
              << " pattern=" << p.value_pattern << "\n";
}

// Byte-exact comparison: returns false and sets mismatchAt on first difference.
static bool VerifyBytesExact(const std::vector<uint8_t>& npuBytes,
                              const std::vector<uint8_t>& goldenBytes,
                              size_t& mismatchAt) {
    if (npuBytes.size() != goldenBytes.size()) return false;
    for (size_t i = 0; i < npuBytes.size(); i++) {
        if (npuBytes[i] != goldenBytes[i]) {
            mismatchAt = i;
            return false;
        }
    }
    return true;
}

class ScatterTest : public testing::TestWithParam<ScatterParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    ScatterParam param_;
    aclrtStream stream_ = nullptr;

    void SetUp() override {
        param_ = GetParam();
        stream_ = env_->stream();
    }
};

// ============================================================================
// TEST_P: Success-path parameterized test (CSV-driven)
// ============================================================================

// Byte-exact or EXACT-float check. Kept out of TEST_P so NBNC stays <= 50.
static void CheckScatterOutput(
    const ScatterParam& p,
    aclDataType valType,
    const ScatterNpuResult& npuResult,
    const std::vector<uint8_t>& goldenY)
{
    const bool preferBytes =
        (valType == ACL_INT8 || valType == ACL_COMPLEX64 ||
         p.value_pattern == "special" || p.value_pattern == "extreme");
    if (preferBytes) {
        EXPECT_EQ(npuResult.yBytes.size(), goldenY.size())
            << "[" << p.case_name << "] Y byte size mismatch";
        if (npuResult.yBytes.size() != goldenY.size()) {
            return;
        }
        size_t mismatchAt = 0;
        bool byteExact = VerifyBytesExact(npuResult.yBytes, goldenY, mismatchAt);
        EXPECT_TRUE(byteExact)
            << "[" << p.case_name << "] memcmp mismatch at byte " << mismatchAt
            << " (NPU=0x" << std::hex << static_cast<int>(npuResult.yBytes[mismatchAt])
            << " vs golden=0x" << static_cast<int>(goldenY[mismatchAt]) << std::dec << ")";
        return;
    }
    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::EXACT);
    bool ok = Verifier::verifyVector(
        BytesToFloatVec(npuResult.yBytes, valType),
        BytesToFloatVec(goldenY, valType),
        cfg, p.case_name + "_Y");
    EXPECT_TRUE(ok);
}

TEST_P(ScatterTest, ScatterSuccess) {
    const auto& p = param_;
    PrintCaseHeader(p);

    aclDataType valType = ParseValType(p.val_type);
    aclsparseIndexType_t idxType = ParseIdxType(p.idx_type);
    aclsparseIndexBase_t idxBase = (p.idx_base == 1)
        ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;

    ScatterHostData hostData = GenerateScatterData(p, valType, idxType);
    std::vector<uint8_t> goldenY = ComputeScatterGolden(
        p.dn_size, p.vec_nnz,
        hostData.indicesBytes, hostData.valuesBytes, hostData.yInitBytes,
        p.idx_type, p.idx_base, hostData.valSize);

    HandleManager handle;
    ScatterNpuArgs npuArgs;
    npuArgs.handle = &handle;
    npuArgs.stream = stream_;
    npuArgs.vecSize = p.vec_size;
    npuArgs.nnz = p.vec_nnz;
    npuArgs.dnSize = p.dn_size;
    npuArgs.indicesHost = hostData.indicesBytes.data();
    npuArgs.valuesHost = hostData.valuesBytes.data();
    npuArgs.yInitHost = hostData.yInitBytes.data();
    npuArgs.idxType = idxType;
    npuArgs.idxBase = idxBase;
    npuArgs.valType = valType;
    auto npuResult = ScatterNpu(npuArgs);

    aclsparseStatus_t expectedStatus = ParseExpectResult(p.expect_result);
    ASSERT_EQ(npuResult.execRet, expectedStatus)
        << "[" << p.case_name << "] aclsparseScatter return code mismatch";
    CheckScatterOutput(p, valType, npuResult, goldenY);
    std::cout << "[" << p.case_name << "] PASSED\n";
}

// ============================================================================
// Parameterized test instantiation from CSV
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    ScatterCases,
    ScatterTest,
    testing::ValuesIn(GetCasesFromCsv<ScatterParam>("scatter_test.csv")),
    [](const testing::TestParamInfo<ScatterParam>& info) {
        return info.param.case_name;
    }
);

// ============================================================================
// Part 3: TEST_F — Exception path tests (L2, not in CSV)
// ============================================================================

// Allocate device buffer: copy from host if valid, else allocate empty.
static DeviceBuffer MakeDeviceBuffer(const void* hostPtr, size_t copyBytes,
                                      size_t allocBytes) {
    bool canCopy = (hostPtr != nullptr && copyBytes > 0);
    return canCopy ? DeviceBuffer::copyFrom(hostPtr, copyBytes) : DeviceBuffer::alloc(allocBytes);
}

struct ScatterDevPtrs {
    void* indices = nullptr;
    void* values = nullptr;
    void* y = nullptr;
};

struct ScatterExArgs : ScatterCallFields {
    aclsparseHandle_t handle = nullptr;
    bool nullVecX = false;
    bool nullVecY = false;
};

// E1 helper: create dummy valid descriptors, call scatter with nullptr handle.
static aclsparseStatus_t ScatterWithNullHandle(
    const ScatterExArgs& a, const ScatterDevPtrs& d)
{
    aclsparseConstSpVecDescr_t spVecDescr = nullptr;
    aclsparseDnVecDescr_t dnVecDescr = nullptr;
    auto st = aclsparseCreateConstSpVec(&spVecDescr, a.vecSize, a.nnz,
        d.indices, d.values, a.idxType, a.idxBase, a.valType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    st = aclsparseCreateDnVec(&dnVecDescr, a.dnSize, d.y, a.valType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        aclsparseDestroySpVec(spVecDescr);
        return st;
    }
    st = aclsparseScatter(nullptr, spVecDescr, dnVecDescr);
    aclsparseDestroySpVec(spVecDescr);
    aclsparseDestroyDnVec(dnVecDescr);
    return st;
}

// E2/E3: scatter call with exactly one null descriptor (null vecX or null vecY).
static aclsparseStatus_t ScatterWithOneNullDescriptor(
    const ScatterExArgs& a, const ScatterDevPtrs& d, bool& handled)
{
    handled = true;
    if (a.nullVecX) {
        aclsparseDnVecDescr_t dnVecDescr = nullptr;
        auto st = aclsparseCreateDnVec(&dnVecDescr, a.dnSize, d.y, a.valType);
        if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
        st = aclsparseScatter(a.handle, nullptr, dnVecDescr);
        aclsparseDestroyDnVec(dnVecDescr);
        return st;
    }
    if (a.nullVecY) {
        aclsparseConstSpVecDescr_t spVecDescr = nullptr;
        auto st = aclsparseCreateConstSpVec(&spVecDescr, a.vecSize, a.nnz,
            d.indices, d.values, a.idxType, a.idxBase, a.valType);
        if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
        st = aclsparseScatter(a.handle, spVecDescr, nullptr);
        aclsparseDestroySpVec(spVecDescr);
        return st;
    }
    handled = false;
    return ACL_SPARSE_STATUS_SUCCESS;
}

class ScatterExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    aclrtStream stream_ = nullptr;
    std::unique_ptr<HandleManager> handle_;

    // Small valid 4x4 data for default parameters
    static constexpr int64_t kSize = 16;
    static constexpr int64_t kNnz = 8;
    static constexpr int64_t kDn = 16;

    std::vector<int32_t> indicesHost_;
    std::vector<float> valuesHost_;
    std::vector<float> yInitHost_;

    void SetUp() override {
        stream_ = env_->stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);

        // Prepare small valid data (FP32, I32, ZERO, sorted)
        indicesHost_.resize(kNnz);
        for (int64_t i = 0; i < kNnz; i++) {
            indicesHost_[static_cast<size_t>(i)] = static_cast<int32_t>(i * 2);
        }
        valuesHost_.resize(kNnz, 1.5f);
        yInitHost_.resize(kDn, -3.0f);
    }

    void TearDown() override {
        handle_.reset();
    }

    ScatterExArgs MakeDefaultArgs() {
        ScatterExArgs args;
        args.handle = handle_->get();
        args.stream = stream_;
        args.vecSize = kSize;
        args.nnz = kNnz;
        args.dnSize = kDn;
        args.indicesHost = indicesHost_.data();
        args.valuesHost = valuesHost_.data();
        args.yInitHost = yInitHost_.data();
        return args;
    }

    // Helper: run scatter with packed args (G.FNM.03), return status
    aclsparseStatus_t RunScatter(const ScatterExArgs& a)
    {
        size_t valSize = ScatterAclDataTypeSize(a.valType);
        size_t idxSize = (a.idxType == ACL_SPARSE_INDEX_64I) ? sizeof(int64_t) : sizeof(int32_t);

        size_t idxBytes = std::max(static_cast<size_t>(a.nnz) * idxSize, size_t(1));
        size_t valBytes = std::max(static_cast<size_t>(a.nnz) * valSize, size_t(1));
        size_t yBytes   = std::max(static_cast<size_t>(a.dnSize) * valSize, size_t(1));

        auto dIndices = MakeDeviceBuffer(a.indicesHost,
            static_cast<size_t>(a.nnz) * idxSize, idxBytes);
        auto dValues = MakeDeviceBuffer(a.valuesHost,
            static_cast<size_t>(a.nnz) * valSize, valBytes);
        auto dY = MakeDeviceBuffer(a.yInitHost,
            static_cast<size_t>(a.dnSize) * valSize, yBytes);
        ScatterDevPtrs dptrs{dIndices.get(), dValues.get(), dY.get()};

        if (a.handle != nullptr) {
            auto setRet = aclsparseSetStream(a.handle, a.stream);
            if (setRet != ACL_SPARSE_STATUS_SUCCESS) {
                return setRet;
            }
        }

        if (a.handle == nullptr) {
            return ScatterWithNullHandle(a, dptrs);
        }

        bool nullDescHandled = false;
        auto st = ScatterWithOneNullDescriptor(a, dptrs, nullDescHandled);
        if (nullDescHandled) return st;

        aclsparseConstSpVecDescr_t spVecDescr = nullptr;
        aclsparseDnVecDescr_t dnVecDescr = nullptr;
        st = aclsparseCreateConstSpVec(&spVecDescr, a.vecSize, a.nnz,
            dIndices.get(), dValues.get(), a.idxType, a.idxBase, a.valType);
        if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
        st = aclsparseCreateDnVec(&dnVecDescr, a.dnSize, dY.get(), a.valType);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            aclsparseDestroySpVec(spVecDescr);
            return st;
        }
        st = aclsparseScatter(a.handle, spVecDescr, dnVecDescr);
        if (aclrtSynchronizeStream(a.stream) != ACL_SUCCESS) {
            std::cerr << "[Test] aclrtSynchronizeStream failed (scatter may be incomplete)\n";
        }
        aclsparseDestroySpVec(spVecDescr);
        aclsparseDestroyDnVec(dnVecDescr);
        return st;
    }
};

// E1: null handle
TEST_F(ScatterExceptionTest, NullHandle) {
    auto args = MakeDefaultArgs();
    args.handle = nullptr;
    auto st = RunScatter(args);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// E2: null vecX
TEST_F(ScatterExceptionTest, NullVecX) {
    auto args = MakeDefaultArgs();
    args.indicesHost = nullptr;
    args.nullVecX = true;
    auto st = RunScatter(args);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E3: null vecY
TEST_F(ScatterExceptionTest, NullVecY) {
    auto args = MakeDefaultArgs();
    args.yInitHost = nullptr;
    args.nullVecY = true;
    auto st = RunScatter(args);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E4: vecX.size > vecY.nums
TEST_F(ScatterExceptionTest, SizeGtDn) {
    auto args = MakeDefaultArgs();
    args.vecSize = 20;
    args.dnSize = 16;
    auto st = RunScatter(args);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E5: vecX.nnz > vecX.size
TEST_F(ScatterExceptionTest, NnzGtSize) {
    auto args = MakeDefaultArgs();
    args.vecSize = 16;
    args.nnz = 20;
    args.dnSize = 16;
    auto st = RunScatter(args);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E6: valueType mismatch (X=FP32, Y=FP16) — tested by creating mismatched descriptors
TEST_F(ScatterExceptionTest, TypeMismatch) {
    // Create X as FP32, Y as FP16
    auto dIndices = DeviceBuffer::copyFrom(indicesHost_.data(), kNnz * sizeof(int32_t));
    auto dValues = DeviceBuffer::copyFrom(valuesHost_.data(), kNnz * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yInitHost_.data(), kDn * sizeof(float));

    aclsparseConstSpVecDescr_t spVecDescr = nullptr;
    aclsparseDnVecDescr_t dnVecDescr = nullptr;
    auto st1 = aclsparseCreateConstSpVec(&spVecDescr, kSize, kNnz,
        dIndices.get(), dValues.get(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    ASSERT_EQ(st1, ACL_SPARSE_STATUS_SUCCESS);
    // Create DnVec with FP16 type (mismatch)
    auto st2 = aclsparseCreateDnVec(&dnVecDescr, kDn, dY.get(), ACL_FLOAT16);
    if (st2 == ACL_SPARSE_STATUS_SUCCESS) {
        auto st = aclsparseScatter(handle_->get(), spVecDescr, dnVecDescr);
        // Design doc §7.3 item 6: valueType mismatch → NOT_SUPPORTED
        // (repo-knowledge: unsupported data type → NOT_SUPPORTED)
        EXPECT_EQ(st, ACL_SPARSE_STATUS_NOT_SUPPORTED);
    }
    aclsparseDestroySpVec(spVecDescr);
    if (dnVecDescr) aclsparseDestroyDnVec(dnVecDescr);
}

// E7: INT8 supported on arch35
TEST_F(ScatterExceptionTest, Int8Supported) {
    std::vector<int8_t> valuesHost(kNnz, 7);
    std::vector<int8_t> yInitHost(kDn, -1);
    auto args = MakeDefaultArgs();
    args.valuesHost = valuesHost.data();
    args.yInitHost = yInitHost.data();
    args.valType = ACL_INT8;
    auto st = RunScatter(args);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_SUCCESS) << "INT8 scatter failed with " << st;
}

// E8: FP64 not supported
TEST_F(ScatterExceptionTest, UnsupportedFp64) {
    auto args = MakeDefaultArgs();
    args.valType = ACL_DOUBLE;
    auto st = RunScatter(args);
    bool acceptable = (st == ACL_SPARSE_STATUS_NOT_SUPPORTED ||
                       st == ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_TRUE(acceptable) << "Expected NOT_SUPPORTED or INVALID_VALUE, got " << st;
}

// E9: invalid idxType enum
TEST_F(ScatterExceptionTest, InvalidIdxType) {
    aclsparseIndexType_t invalidIdx = static_cast<aclsparseIndexType_t>(999);
    auto dIndices = DeviceBuffer::copyFrom(indicesHost_.data(), kNnz * sizeof(int32_t));
    auto dValues = DeviceBuffer::copyFrom(valuesHost_.data(), kNnz * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yInitHost_.data(), kDn * sizeof(float));

    aclsparseConstSpVecDescr_t spVecDescr = nullptr;
    auto st = aclsparseCreateConstSpVec(&spVecDescr, kSize, kNnz,
        dIndices.get(), dValues.get(), invalidIdx,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    if (st == ACL_SPARSE_STATUS_SUCCESS) {
        aclsparseDnVecDescr_t dnVecDescr = nullptr;
        auto st2 = aclsparseCreateDnVec(&dnVecDescr, kDn, dY.get(), ACL_FLOAT);
        if (st2 == ACL_SPARSE_STATUS_SUCCESS) {
            st = aclsparseScatter(handle_->get(), spVecDescr, dnVecDescr);
            aclsparseDestroyDnVec(dnVecDescr);
        }
    }
    aclsparseDestroySpVec(spVecDescr);
    // Test plan: NOT_SUPPORTED or INVALID_VALUE (implementation-dependent)
    bool acceptable = (st == ACL_SPARSE_STATUS_NOT_SUPPORTED ||
                       st == ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_TRUE(acceptable) << "Expected NOT_SUPPORTED or INVALID_VALUE, got " << st;
}

// E10: invalid idxBase enum
TEST_F(ScatterExceptionTest, InvalidIdxBase) {
    auto invalidBase = static_cast<aclsparseIndexBase_t>(99);
    auto args = MakeDefaultArgs();
    args.idxBase = invalidBase;
    auto st = RunScatter(args);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E11: index out of bounds (ZERO, indices[i] = dn)
// Design doc §7.3 note + §12: host side does NOT validate index values
// (aligned with cuSPARSE — no runtime bounds check). OOB indices yield
// undefined behavior at kernel level, but the API returns SUCCESS.
TEST_F(ScatterExceptionTest, IdxOobZero) {
    // Create indices with one value == dn (out of bounds for ZERO base)
    std::vector<int32_t> badIndices = {0, 2, 4, 6, 8, 10, 12, static_cast<int32_t>(kDn)};
    auto args = MakeDefaultArgs();
    args.indicesHost = badIndices.data();
    auto st = RunScatter(args);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_SUCCESS);
}

// E12: index out of bounds (ONE, indices[i] = dn+1)
TEST_F(ScatterExceptionTest, IdxOobOne) {
    // idxBase=ONE: valid indices are [1, dn]; dn+1 is out of bounds
    std::vector<int32_t> badIndices = {1, 3, 5, 7, 9, 11, 13, static_cast<int32_t>(kDn + 1)};
    auto args = MakeDefaultArgs();
    args.indicesHost = badIndices.data();
    args.idxBase = ACL_SPARSE_INDEX_BASE_ONE;
    auto st = RunScatter(args);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_SUCCESS);
}

// E13: index negative (ZERO, indices[i] = -1)
TEST_F(ScatterExceptionTest, IdxNegative) {
    std::vector<int32_t> badIndices = {-1, 2, 4, 6, 8, 10, 12, 14};
    auto args = MakeDefaultArgs();
    args.indicesHost = badIndices.data();
    auto st = RunScatter(args);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_SUCCESS);
}

// Helper for E14/E15: create SpVec with one null field, create DnVec, call scatter,
// expect INVALID_VALUE. Eliminates duplicate code between NullIndices and NullValues.
struct NullSpVecFieldArgs {
    aclsparseHandle_t handle = nullptr;
    int64_t size = 0;
    int64_t nnz = 0;
    int64_t dnSize = 0;
    void* dIndices = nullptr;
    void* dValues = nullptr;
    void* dY = nullptr;
};

static void TestScatterNullSpVecField(const NullSpVecFieldArgs& a) {
    aclsparseConstSpVecDescr_t spVecDescr = nullptr;
    aclsparseDnVecDescr_t dnVecDescr = nullptr;
    auto st1 = aclsparseCreateConstSpVec(&spVecDescr, a.size, a.nnz,
        a.dIndices, a.dValues, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    if (st1 == ACL_SPARSE_STATUS_SUCCESS) {
        auto st2 = aclsparseCreateDnVec(&dnVecDescr, a.dnSize, a.dY, ACL_FLOAT);
        if (st2 == ACL_SPARSE_STATUS_SUCCESS) {
            auto st = aclsparseScatter(a.handle, spVecDescr, dnVecDescr);
            EXPECT_EQ(st, ACL_SPARSE_STATUS_INVALID_VALUE);
            aclsparseDestroyDnVec(dnVecDescr);
        }
    }
    if (spVecDescr) aclsparseDestroySpVec(spVecDescr);
}

// E14: null indices (vecX.indices = nullptr, nnz > 0)
TEST_F(ScatterExceptionTest, NullIndices) {
    auto dValues = DeviceBuffer::copyFrom(valuesHost_.data(), kNnz * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yInitHost_.data(), kDn * sizeof(float));
    NullSpVecFieldArgs na;
    na.handle = handle_->get();
    na.size = kSize;
    na.nnz = kNnz;
    na.dnSize = kDn;
    na.dIndices = nullptr;
    na.dValues = dValues.get();
    na.dY = dY.get();
    TestScatterNullSpVecField(na);
}

// E15: null values (vecX.values = nullptr, nnz > 0)
TEST_F(ScatterExceptionTest, NullValues) {
    auto dIndices = DeviceBuffer::copyFrom(indicesHost_.data(), kNnz * sizeof(int32_t));
    auto dY = DeviceBuffer::copyFrom(yInitHost_.data(), kDn * sizeof(float));
    NullSpVecFieldArgs na;
    na.handle = handle_->get();
    na.size = kSize;
    na.nnz = kNnz;
    na.dnSize = kDn;
    na.dIndices = dIndices.get();
    na.dValues = nullptr;
    na.dY = dY.get();
    TestScatterNullSpVecField(na);
}

// E16: null Y values (vecY.values = nullptr)
TEST_F(ScatterExceptionTest, NullYValues) {
    auto dIndices = DeviceBuffer::copyFrom(indicesHost_.data(), kNnz * sizeof(int32_t));
    auto dValues = DeviceBuffer::copyFrom(valuesHost_.data(), kNnz * sizeof(float));
    aclsparseConstSpVecDescr_t spVecDescr = nullptr;
    aclsparseDnVecDescr_t dnVecDescr = nullptr;
    auto st1 = aclsparseCreateConstSpVec(&spVecDescr, kSize, kNnz,
        dIndices.get(), dValues.get(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    if (st1 == ACL_SPARSE_STATUS_SUCCESS) {
        auto st2 = aclsparseCreateDnVec(&dnVecDescr, kDn, nullptr, ACL_FLOAT);
        if (st2 == ACL_SPARSE_STATUS_SUCCESS) {
            auto st = aclsparseScatter(handle_->get(), spVecDescr, dnVecDescr);
            EXPECT_EQ(st, ACL_SPARSE_STATUS_INVALID_VALUE);
            aclsparseDestroyDnVec(dnVecDescr);
        }
    }
    if (spVecDescr) aclsparseDestroySpVec(spVecDescr);
}

// E17: undeclared alias — vecX.values and vecY.values share the same Device pointer
TEST_F(ScatterExceptionTest, ValuesAliasY) {
    auto dIndices = DeviceBuffer::copyFrom(indicesHost_.data(), kNnz * sizeof(int32_t));
    // Allocate one buffer large enough for both values[nnz] and Y[dn] views; reuse as Y.
    auto dShared = DeviceBuffer::copyFrom(yInitHost_.data(), kDn * sizeof(float));
    aclsparseConstSpVecDescr_t spVecDescr = nullptr;
    aclsparseDnVecDescr_t dnVecDescr = nullptr;
    auto st1 = aclsparseCreateConstSpVec(&spVecDescr, kSize, kNnz,
        dIndices.get(), dShared.get(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    ASSERT_EQ(st1, ACL_SPARSE_STATUS_SUCCESS);
    auto st2 = aclsparseCreateDnVec(&dnVecDescr, kDn, dShared.get(), ACL_FLOAT);
    ASSERT_EQ(st2, ACL_SPARSE_STATUS_SUCCESS);
    auto st = aclsparseScatter(handle_->get(), spVecDescr, dnVecDescr);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroyDnVec(dnVecDescr);
    aclsparseDestroySpVec(spVecDescr);
}

// ============================================================================
// L2 duplicate index informational test (not EXACT-verified)
// ============================================================================

class ScatterDuplicateIdxTest : public testing::Test {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }
    static void TearDownTestSuite() {
        env_.reset();
    }
protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

// L2_duplicate_idx_info: construct input with duplicate indices.
// Only assert: (1) aclsparseScatter returns SUCCESS,
//               (2) Y[duplicate_pos] belongs to the set of written values.
// No EXACT comparison (behavior is non-deterministic: last-write-wins).
TEST_F(ScatterDuplicateIdxTest, DuplicateIdxInformational) {
    aclrtStream stream = env_->stream();
    HandleManager handle;
    handle.setStream(stream);

    constexpr int64_t kSize = 16;
    constexpr int64_t kNnz = 8;
    constexpr int64_t kDn = 16;

    // Create indices with duplicates: positions 0,2,4,6,8,10,12,12 (last two duplicate)
    // idxBase = ZERO
    std::vector<int32_t> indices = {0, 2, 4, 6, 8, 10, 12, 12};
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> yInit(kDn, -99.0f);

    auto dIndices = DeviceBuffer::copyFrom(indices.data(), kNnz * sizeof(int32_t));
    auto dValues = DeviceBuffer::copyFrom(values.data(), kNnz * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yInit.data(), kDn * sizeof(float));

    auto vecX = SpVecManager::createConst(kSize, kNnz,
        dIndices.get(), dValues.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    auto vecY = DnVecManager::create(kDn, dY.get(), ACL_FLOAT);

    auto st = aclsparseScatter(handle.get(), vecX.cget(), vecY.get());
    EXPECT_EQ(st, ACL_SPARSE_STATUS_SUCCESS);

    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS) << "aclrtSynchronizeStream failed";

    // Copy Y back
    std::vector<float> yResult(kDn);
    dY.copyToHost(yResult.data(), kDn * sizeof(float));

    // Verify: Y[12] must be either 7.0f or 8.0f (the two values written to index 12)
    // Other positions should be exactly their single written value
    bool dupPosValid = (yResult[12] == 7.0f || yResult[12] == 8.0f);
    EXPECT_TRUE(dupPosValid)
        << "Y[12]=" << yResult[12] << " not in {7.0, 8.0} (duplicate index set)";

    // Verify non-duplicate positions are exactly correct
    EXPECT_FLOAT_EQ(yResult[0], 1.0f);
    EXPECT_FLOAT_EQ(yResult[2], 2.0f);
    EXPECT_FLOAT_EQ(yResult[4], 3.0f);
    EXPECT_FLOAT_EQ(yResult[6], 4.0f);
    EXPECT_FLOAT_EQ(yResult[8], 5.0f);
    EXPECT_FLOAT_EQ(yResult[10], 6.0f);

    // Unscattered positions should keep init value
    EXPECT_FLOAT_EQ(yResult[1], -99.0f);
    EXPECT_FLOAT_EQ(yResult[3], -99.0f);

    std::cout << "[L2_duplicate_idx_info] PASSED (Y[12]=" << yResult[12] << ")\n";
}

// ============================================================================
// L2 whitebox: nnz > UINT32_MAX truncation protection (host-side guard)
//
// The host-side LaunchScatterKernel guards against nnz exceeding UINT32_MAX:
// when xInner->nnz > kScatterNnzUpperLimit (UINT32_MAX), it returns
// ACL_SPARSE_STATUS_NOT_SUPPORTED before casting nnz to uint32_t for CeilDiv.
//
// This guard prevents silent truncation when casting uint64_t nnz to uint32_t
// for CeilDiv. To trigger it, we construct descriptors with nnz = UINT32_MAX+1.
// Since host validation only checks nnz <= size and size <= nums (no upper
// bound), passing size = nums = nnz = UINT32_MAX+1 with dummy non-null device
// pointers should reach the guard.
//
// If aclsparseCreateConstSpVec / aclsparseCreateDnVec internally reject such
// large values, the guard is not reachable via public API and the test
// reports a skip message (branch covered at source level only).
// ============================================================================

TEST_F(ScatterExceptionTest, NnzExceedsUint32Max) {
    // UINT32_MAX + 1 = 4294967296, cast to int64_t (positive, fits)
    constexpr int64_t kHugeNnz = static_cast<int64_t>(1ULL << 32);
    constexpr int64_t kHugeSize = kHugeNnz;
    constexpr int64_t kHugeDn = kHugeNnz;

    // Minimal dummy device buffers (non-null, won't be accessed — guard
    // returns before kernel launch)
    auto dIndices = DeviceBuffer::alloc(16);
    auto dValues = DeviceBuffer::alloc(16);
    auto dY = DeviceBuffer::alloc(16);

    aclsparseConstSpVecDescr_t spVecDescr = nullptr;
    aclsparseDnVecDescr_t dnVecDescr = nullptr;

    auto st1 = aclsparseCreateConstSpVec(&spVecDescr, kHugeSize, kHugeNnz,
        dIndices.get(), dValues.get(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    if (st1 != ACL_SPARSE_STATUS_SUCCESS) {
        std::cout << "[WB_NnzExceedsUint32Max] CreateConstSpVec rejected huge nnz (st="
                  << st1 << "), guard not reachable via public API\n";
        SUCCEED() << "Guard branch not reachable via API (CreateConstSpVec rejects)";
        return;
    }

    auto st2 = aclsparseCreateDnVec(&dnVecDescr, kHugeDn, dY.get(), ACL_FLOAT);
    if (st2 != ACL_SPARSE_STATUS_SUCCESS) {
        std::cout << "[WB_NnzExceedsUint32Max] CreateDnVec rejected huge dn (st="
                  << st2 << ")\n";
        aclsparseDestroySpVec(spVecDescr);
        SUCCEED() << "Guard branch not reachable via API (CreateDnVec rejects)";
        return;
    }

    auto st = aclsparseScatter(handle_->get(), spVecDescr, dnVecDescr);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_NOT_SUPPORTED)
        << "nnz=" << kHugeNnz << " (UINT32_MAX+1) should trigger NOT_SUPPORTED";

    aclsparseDestroySpVec(spVecDescr);
    aclsparseDestroyDnVec(dnVecDescr);
}

// ============================================================================
// AUDIT whitebox: stream not set (host-side guard, scatter_host.cpp:184-187)
//
// LaunchScatterKernel checks `stream == nullptr` and returns
// ACL_SPARSE_STATUS_INVALID_VALUE. All existing TEST_F cases call
// aclsparseSetStream via RunScatter, so this guard branch is never reached.
// This test creates a HandleManager WITHOUT calling setStream, then calls
// aclsparseScatter with otherwise-valid descriptors.
//
// If the guard is missing or broken, the kernel would launch on a nullptr
// stream, causing a crash or undefined behavior.
// ============================================================================
TEST_F(ScatterExceptionTest, StreamNotSet) {
    // Create handle WITHOUT setting stream — stream defaults to nullptr
    HandleManager noStreamHandle;

    auto dIndices = DeviceBuffer::copyFrom(indicesHost_.data(), kNnz * sizeof(int32_t));
    auto dValues = DeviceBuffer::copyFrom(valuesHost_.data(), kNnz * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(yInitHost_.data(), kDn * sizeof(float));

    aclsparseConstSpVecDescr_t spVecDescr = nullptr;
    aclsparseDnVecDescr_t dnVecDescr = nullptr;
    auto st1 = aclsparseCreateConstSpVec(&spVecDescr, kSize, kNnz,
        dIndices.get(), dValues.get(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    ASSERT_EQ(st1, ACL_SPARSE_STATUS_SUCCESS);
    auto st2 = aclsparseCreateDnVec(&dnVecDescr, kDn, dY.get(), ACL_FLOAT);
    ASSERT_EQ(st2, ACL_SPARSE_STATUS_SUCCESS);

    auto st = aclsparseScatter(noStreamHandle.get(), spVecDescr, dnVecDescr);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_INVALID_VALUE)
        << "stream not set should return INVALID_VALUE";

    aclsparseDestroySpVec(spVecDescr);
    aclsparseDestroyDnVec(dnVecDescr);
}

// ============================================================================
// AUDIT whitebox: nnz=0 with null vecY.values (scatter_host.cpp:136-140)
//
// ValidateDnVecAndCompatibility only checks `vecY.values == nullptr` when
// `nnz > 0`. When nnz == 0, vecY.values is allowed to be nullptr because the
// kernel is skipped (nnz == 0 fast-return at scatter_host.cpp:193-196).
// This test verifies that path: nnz=0 + null Y.values should return SUCCESS
// without accessing the null pointer.
// ============================================================================
TEST_F(ScatterExceptionTest, Nnz0NullYValues) {
    HandleManager handle;
    handle.setStream(stream_);

    auto dIndices = DeviceBuffer::alloc(16);
    auto dValues = DeviceBuffer::alloc(16);

    aclsparseConstSpVecDescr_t spVecDescr = nullptr;
    aclsparseDnVecDescr_t dnVecDescr = nullptr;
    // nnz=0: indices/values not accessed, but must be non-null for CreateConstSpVec
    auto st1 = aclsparseCreateConstSpVec(&spVecDescr, kSize, 0,
        dIndices.get(), dValues.get(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    ASSERT_EQ(st1, ACL_SPARSE_STATUS_SUCCESS);
    // Y.values = nullptr, nums = kDn
    auto st2 = aclsparseCreateDnVec(&dnVecDescr, kDn, nullptr, ACL_FLOAT);
    if (st2 != ACL_SPARSE_STATUS_SUCCESS) {
        std::cout << "[AUDIT_Nnz0NullYValues] CreateDnVec rejected null values (st="
                  << st2 << "), skipping\n";
        aclsparseDestroySpVec(spVecDescr);
        SUCCEED() << "CreateDnVec rejects null values — guard not reachable";
        return;
    }

    auto st = aclsparseScatter(handle.get(), spVecDescr, dnVecDescr);
    // nnz=0 → kernel skipped → SUCCESS even with null Y.values
    EXPECT_EQ(st, ACL_SPARSE_STATUS_SUCCESS)
        << "nnz=0 with null Y.values should return SUCCESS (kernel skipped)";

    aclsparseDestroySpVec(spVecDescr);
    aclsparseDestroyDnVec(dnVecDescr);
}

// 禁止定义 main 函数（由 test/frame/test_main.cpp 提供）
