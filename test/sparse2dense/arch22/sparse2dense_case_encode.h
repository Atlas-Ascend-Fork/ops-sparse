// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE for the full text of the License.
// ============================================================================
// sparse2dense_case_encode.h — sparse2dense 用例值编码与位级转换（安全字节
// 存取/dtype 映射/fp32→fp16/bf16 位转换/格式名/量化编码/Rng）；纯 host 计算，
// 无设备/ACL 运行时调用；仅被 sparse2dense_case_gen.h 与 sparse2dense_test.cpp 使用。
// ============================================================================

#ifndef SPARSE2DENSE_CASE_ENCODE_H_
#define SPARSE2DENSE_CASE_ENCODE_H_

#include "sparse2dense_case_types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <acl/acl.h>

// ---- 安全字节存取（替代 memcpy/memset：std::copy/fill 位元等价；本文件全部使用
// 无重叠定长场景，长度由 sizeof/循环上界静态确定）----
template <typename T>
inline void StoreBits(void *dst, const T &src)
{
    const char *p = reinterpret_cast<const char *>(&src);
    std::copy(p, p + sizeof(T), static_cast<char *>(dst));
}

template <typename T>
inline T LoadBits(const void *src)
{
    T dst;
    std::copy(static_cast<const char *>(src), static_cast<const char *>(src) + sizeof(T),
        reinterpret_cast<char *>(&dst));
    return dst;
}

inline void CopyBytes(void *dst, const void *src, size_t n)
{
    std::copy(static_cast<const char *>(src), static_cast<const char *>(src) + n,
        static_cast<char *>(dst));
}

// ============================================================================
// dtype 抽象（存储均为原始 bit 模式，保证 bit-wise 判定）
// ============================================================================

inline const char *DTypeName(DType t)
{
    switch (t) {
        case DType::INT8:
            return "int8";
        case DType::FP16:
            return "float16";
        case DType::BF16:
            return "bfloat16";
        case DType::FP32:
            return "float32";
        case DType::COMPLEX64:
            return "complex64";
    }
    return "?";
}

inline size_t ElemBytes(DType t)
{
    switch (t) {
        case DType::INT8:
            return 1;
        case DType::FP16:
        case DType::BF16:
            return 2;
        case DType::FP32:
            return 4;
        case DType::COMPLEX64:
            return 8;
    }
    return 0;
}

// 描述符 API 值类型为 aclDataType（acl/acl.h），映射见 ToAclDataType
inline aclDataType ToAclDataType(DType t) // 与 op_host Sparse2DenseCheckMeta 支持集一致
{
    switch (t) {
        case DType::INT8:
            return ACL_INT8;
        case DType::FP16:
            return ACL_FLOAT16;
        case DType::BF16:
            return ACL_BF16;
        case DType::FP32:
            return ACL_FLOAT;
        case DType::COMPLEX64:
            return ACL_COMPLEX64;
    }
    return ACL_FLOAT;
}

// ---- fp32 ↔ fp16 / bf16 bit 转换（round-to-nearest-even） ----
inline uint16_t Fp32ToFp16Bits(float v)
{
    const uint32_t x = LoadBits<uint32_t>(&v);
    const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) { // NaN / Inf
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    }
    if (exp > 15) {
        return static_cast<uint16_t>(sign | 0x7C00u); // overflow → Inf
    }
    if (exp >= -14) { // normal
        int32_t e = exp + 15;
        uint32_t m = mant >> 13;
        uint32_t rem = mant & 0x1FFFu;
        uint32_t half = 0x1000u;
        uint32_t frac = static_cast<uint32_t>(e << 10) | m;
        if (rem > half || (rem == half && (m & 1u))) {
            frac += 1; // RNE 进位（可能溢出至下一指数，位模式仍正确）
        }
        return static_cast<uint16_t>(sign | frac);
    }
    if (exp >= -25) { // subnormal
        uint32_t m = mant | 0x800000u;
        const int32_t shift = -14 - exp + 13;
        uint32_t val = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (val & 1u))) {
            val += 1;
        }
        return static_cast<uint16_t>(sign | val);
    }
    return sign; // → 0（含 RNE 舍入到 0 的极小值）
}

inline uint16_t Fp32ToBf16Bits(float v)
{
    uint32_t x = LoadBits<uint32_t>(&v);
    uint32_t lsb = (x >> 16) & 1u;
    x += 0x7FFFu + lsb; // RNE
    return static_cast<uint16_t>(x >> 16);
}

// ============================================================================
// 确定性随机数（splitmix64，seed 取自 CSV value_spec）
// ============================================================================
class Rng {
public:
    explicit Rng(uint64_t seed) : state_(seed)
    {
    }
    uint64_t NextU64()
    {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double NextUniform(double lo, double hi)
    {
        return lo + (hi - lo) * (static_cast<double>(NextU64() >> 11) / 9007199254740992.0);
    }
    int64_t NextInt(int64_t lo, int64_t hi) // [lo, hi]
    {
        return lo + static_cast<int64_t>(NextU64() % static_cast<uint64_t>(hi - lo + 1));
    }

private:
    uint64_t state_;
};

inline const char *FmtName(Fmt f)
{
    switch (f) {
        case Fmt::CSR:
            return "csr";
        case Fmt::CSC:
            return "csc";
        case Fmt::COO:
            return "coo";
    }
    return "?";
}

inline const char *LayoutName(Layout l)
{
    return (l == Layout::ROW) ? "ROW" : "COL";
}

// 将逻辑值（fp32 / 复数对）量化为目标 dtype 的 bit 模式（int8 夹取域取自用例 vLo/vHi）
inline void QuantizeStore(const CaseSpec &cs, double re, double im, uint8_t *dst)
{
    const DType t = cs.dtype;
    switch (t) {
        case DType::INT8: {
            const int lo = static_cast<int>(std::lround(cs.vLo));
            const int hi = static_cast<int>(std::lround(cs.vHi));
            int iv = static_cast<int>(std::lround(re));
            iv = std::max(lo, std::min(hi, iv)); // uniform70 整数域（矩阵 [-1,1] / P-01 [-9,9]）
            const int8_t b = static_cast<int8_t>(iv);
            StoreBits(dst, b);
            return;
        }
        case DType::FP16: {
            const uint16_t b = Fp32ToFp16Bits(static_cast<float>(re));
            StoreBits(dst, b);
            return;
        }
        case DType::BF16: {
            const uint16_t b = Fp32ToBf16Bits(static_cast<float>(re));
            StoreBits(dst, b);
            return;
        }
        case DType::FP32: {
            const float f = static_cast<float>(re);
            StoreBits(dst, f);
            return;
        }
        case DType::COMPLEX64: {
            const float c[2] = {static_cast<float>(re), static_cast<float>(im)};
            CopyBytes(dst, c, 8);
            return;
        }
    }
}


#endif // SPARSE2DENSE_CASE_ENCODE_H_
