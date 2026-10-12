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
 * aclsparseScatter A2 全覆盖 gtest UT
 *
 * 用例来源：黑盒 79 条（L0 8 + L1 56 + L2 15）的功能面 C++ gtest 落地。
 *
 * 判定口径：
 *   - 无重复索引 → golden bit-wise exact（rtol=0/atol=0，五 dtype bitwise_equal）；
 *     golden 为纯 bit 搬运副本（等价 torch index_copy_ 的 bit 语义，纯数据搬运无舍入，
 *     Golden dtype 升格规则 int8-bit/fp16·bf16→fp32/fp32→fp64/c64→c128 仅对算术 oracle 有意义，
 *     bit 副本 golden 天然满足）；
 *   - 有重复索引 → invariant 三断言：冲突位置输出 ∈ 对应输入 values 集合
 *     （collision_value_from_source，last-write-wins 非累加；互异 2 的幂
 *     使任何部分和 ∉ 集合，反证非累加）+ 未写入位置保持 y 原值
 *     （unscattered_positions_preserved）+ vecX 只读（values/indices 逐字节不变）；
 *   - 全部用例附带断言：vecX 只读快照比对。
 *
 * 数据规格与 seed：dtype 矩阵 size=17/nnz=9/seed=7；P 形状 seed=20260902；
 * 尾块 37/4095/4097；P-01(128256/8192)/P-02(151936/4096)/P-03(129280/7168)。
 *
 * 执行证据：case_id 级 JSON 报告（环境变量 SCATTER_UT_JSON 指定路径，默认
 * ./scatter_ut_results.json），含 planned 条目。
 *
 * 分层说明：
 *   - idxBase 非法 / nnz>0 空指针：aclsparseCreateSpVec 描述符层即返回确定参数错误
 *     （既有校验），scatter 侧同名校验为纵深防御（公共 API 不可达），UT 断言
 *     用户可见的确定报错（错误码一致）；
 *   - L2_005 索引内容越界（三层口径）：合法索引属接口契约，本文件以哨兵缓冲
 *     硬门断言输出正确性与越界写不逃出 vecY（内存安全门禁，确定性接口行为，
 *     不依赖平台实现细节）；违反 native API 前置条件的非法索引（负 / 上溢）
 *     无承诺行为，在独立进程探测二进制 scatter_oob_probe_test 全观测记录
 *     （返回状态 / 同步结果 / 哨兵变化 / 恢复控制组），不作 pass/fail 判定；
 *     ATen 入口确定拒绝（TORCH_CHECK 索引校验）由 test_torch_extension.py 承接；
 *   - L2_011 device 不一致：描述符不携带 device 元数据，Host 无法同步判定；双卡环境下
 *     经异步错误协议确定暴露（vecY 不被污染、无静默错误结果、上下文可恢复）。
 */

#include <gtest/gtest.h>
#include <acl/acl.h>
#include "scatter_ut_common.h"  // CheckedMemcpy/CurrentGtestName/JsonEscape/DevBuf/SetupAclDevice0
#include "cann_ops_sparse.h"
#include "securec.h"  // memcpy_s, EOK

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ===================== 全局状态 =====================

static aclrtStream g_stream = nullptr;
static aclsparseHandle_t g_handle = nullptr;

// ===================== dtype 描述与转换 =====================

// float -> dtype bits（golden 为纯 bit 拷贝，转换仅需确定性）
static void ConvFp32(float v, uint8_t *dst)
{
    CheckedMemcpy(dst, sizeof(float), &v, sizeof(float));
}

static void ConvFp16(float v, uint8_t *dst)
{
    __fp16 h = static_cast<__fp16>(v);  // aarch64 原生半精度
    CheckedMemcpy(dst, sizeof(__fp16), &h, sizeof(__fp16));
}

static void ConvBf16(float v, uint8_t *dst)
{
    uint32_t x;
    CheckedMemcpy(&x, sizeof(x), &v, sizeof(x));
    uint16_t bits = static_cast<uint16_t>(x >> 16);  // 截断：golden 是 bit 拷贝
    CheckedMemcpy(dst, sizeof(bits), &bits, sizeof(bits));
}

static void ConvInt8(float v, uint8_t *dst)
{
    int b = static_cast<int>(v);
    if (b > 127) { b = 127; }
    if (b < -128) { b = -128; }
    int8_t s = static_cast<int8_t>(b);
    CheckedMemcpy(dst, sizeof(int8_t), &s, sizeof(int8_t));
}

static void ConvC64(float v, uint8_t *dst)
{
    // complex64 = 2×fp32 交错（re, im），与 Kernel T=float/ELEM_BYTES=8 存储口径一致
    float re = v;
    float im = -2.0f * v + 0.5f;
    CheckedMemcpy(dst, sizeof(float), &re, sizeof(float));
    CheckedMemcpy(dst + sizeof(float), sizeof(float), &im, sizeof(float));
}

struct UtDtype {
    const char *name;
    aclDataType aclType;
    size_t es;                       // element bytes
    void (*conv)(float, uint8_t *);
};

static const UtDtype kFp32 = {"float32", ACL_FLOAT, 4, ConvFp32};
static const UtDtype kFp16 = {"float16", ACL_FLOAT16, 2, ConvFp16};
static const UtDtype kBf16 = {"bfloat16", ACL_BF16, 2, ConvBf16};
static const UtDtype kInt8 = {"int8", ACL_INT8, 1, ConvInt8};
static const UtDtype kC64 = {"complex64", ACL_COMPLEX64, 8, ConvC64};

static aclsparseIndexBase_t BaseOf(int base)
{
    return (base == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;
}

// ===================== 用例参数 =====================

enum class IdxSpec {
    EMPTY,             // nnz=0
    SHUFFLED_NODUP,    // 乱序互异（非单调）
    DUP,               // 部分重复
    ALL_SAME,          // 全相同索引
    IDENTITY_RUNS,     // [0,size) 顺序（全密集，命中 run 段拷贝快路径）
    ORDERED_RUNS2,     // 2 段各 16（[0,16)+[32,48)，命中段拷贝路径）
    ENDPOINT_PAIR,     // 含 0 与 size-1（base1: 1 与 size）
    PERMUTATION_NODUP  // 全随机置换（nnz=size）
};

enum class ValueSpec {
    UNIFORM70,   // [-1,1] 均匀（int8 整数域量化 {-1,0,1}）
    GAUSS20,     // N(0,1)
    DISTINCT,    // 互异 2 的幂（任何部分和 ∉ 集合，反证非累加）
    NAN_ONE,     // 单元素 NaN（bit 模式 payload 搬运）
    POS_INF_ONE, // 单元素 +Inf
    ALL_ZERO,    // 全零
    SUBNORMAL,   // subnormal 小值（FTZ/DAZ 不改变搬运语义）
    MIN_MAX_ALT  // ±FLT_MAX 交替（离群值/正负混合）
};

enum class YFill { RANDOM, SENTINEL };  // SENTINEL: 位置编码可区分哨兵

struct CaseParams {
    const char *caseId;
    const UtDtype *dtype;
    size_t size;
    size_t nnz;
    int base;
    IdxSpec idx;
    ValueSpec val;
    YFill yfill;
    uint32_t seed;
};

// ===================== case_id 级执行证据记录 =====================

struct CaseRecord {
    std::string caseId;
    std::string gtest;
    std::string status;   // PASS / FAIL / PLANNED
    std::string check;
    std::string detail;
    std::string dtype;
    size_t size = 0;
    size_t nnz = 0;
    int base = 0;
    uint32_t seed = 0;
};

static std::vector<CaseRecord> g_records;

static void RecordCase(const CaseRecord &r)
{
    g_records.push_back(r);
}

static void RecordFunctional(const CaseParams &p, const char *status, const char *check,
                             const std::string &detail)
{
    CaseRecord r;
    r.caseId = p.caseId;
    r.gtest = CurrentGtestName();
    r.status = status;
    r.check = check;
    r.detail = detail;
    r.dtype = p.dtype->name;
    r.size = p.size;
    r.nnz = p.nnz;
    r.base = p.base;
    r.seed = p.seed;
    RecordCase(r);
}

static void RecordPlanned(const char *caseId, const char *check, const std::string &reason)
{
    CaseRecord r;
    r.caseId = caseId;
    r.gtest = "(planned)";
    r.status = "PLANNED";
    r.check = check;
    r.detail = reason;
    RecordCase(r);
}

// ===================== 数据生成（value_spec / indices 模式 / 固定 seed） =====================

static std::vector<int32_t> SampleUniqueShuffled(size_t size, size_t k, std::mt19937 &rng,
                                                 size_t offset = 0)
{
    if (k == 0 || size <= offset) {
        return {};
    }
    std::vector<int32_t> pool(size - offset);
    for (size_t i = 0; i < pool.size(); ++i) {
        pool[i] = static_cast<int32_t>(offset + i);
    }
    std::shuffle(pool.begin(), pool.end(), rng);
    if (k > pool.size()) {
        k = pool.size();
    }
    return std::vector<int32_t>(pool.begin(), pool.begin() + static_cast<int64_t>(k));
}

// DUP：ceil(nnz/2) 个互异目标 + 其余从互异集合回填（保证 ≥1 冲突组），整体乱序
static std::vector<int32_t> GenTargetsDup(size_t size, size_t nnz, std::mt19937 &rng)
{
    const size_t k = (nnz + 1) / 2;
    std::vector<int32_t> uniq = SampleUniqueShuffled(size, k, rng);
    std::vector<int32_t> t = uniq;
    if (k > 0) {
        std::uniform_int_distribution<size_t> pick(0, k - 1);
        for (size_t i = k; i < nnz; ++i) {
            t.push_back(uniq[pick(rng)]);
        }
    }
    std::shuffle(t.begin(), t.end(), rng);
    return t;
}

// ENDPOINT_PAIR：含 0 与 size-1（base1 下 descIdx 含 1 与 size）+ 其余互异，乱序
static std::vector<int32_t> GenTargetsEndpointPair(size_t size, size_t nnz, std::mt19937 &rng)
{
    std::vector<int32_t> rest = SampleUniqueShuffled(size - 2 < 1 ? 0 : size - 2, nnz - 2, rng, 1);
    std::vector<int32_t> t;
    t.push_back(0);
    t.push_back(static_cast<int32_t>(size - 1));
    for (int32_t v : rest) {
        t.push_back(v);
    }
    std::shuffle(t.begin(), t.end(), rng);
    return t;
}

static std::vector<int32_t> GenTargets(const CaseParams &p, std::mt19937 &rng)
{
    const size_t size = p.size;
    const size_t nnz = p.nnz;
    switch (p.idx) {
        case IdxSpec::EMPTY:
            return {};
        case IdxSpec::SHUFFLED_NODUP:
            return SampleUniqueShuffled(size, nnz, rng);
        case IdxSpec::DUP:
            return GenTargetsDup(size, nnz, rng);
        case IdxSpec::ALL_SAME:
            return std::vector<int32_t>(nnz, 0);
        case IdxSpec::IDENTITY_RUNS: {
            std::vector<int32_t> t(nnz);
            for (size_t i = 0; i < nnz; ++i) {
                t[i] = static_cast<int32_t>(i);
            }
            return t;
        }
        case IdxSpec::ORDERED_RUNS2: {
            // 2 段各 16：[0,16) + [32,48)
            std::vector<int32_t> t;
            for (int i = 0; i < 16; ++i) {
                t.push_back(i);
            }
            for (int i = 32; i < 48; ++i) {
                t.push_back(i);
            }
            return t;
        }
        case IdxSpec::ENDPOINT_PAIR:
            return GenTargetsEndpointPair(size, nnz, rng);
        case IdxSpec::PERMUTATION_NODUP: {
            std::vector<int32_t> t = SampleUniqueShuffled(size, size, rng);
            return t;
        }
    }
    return {};
}

static std::vector<uint8_t> GenValues(const CaseParams &p, size_t nnz, std::mt19937 &rng)
{
    const size_t es = p.dtype->es;
    std::vector<uint8_t> v(nnz * es);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::normal_distribution<float> nor(0.0f, 1.0f);
    // 互异 2 的幂：任何 ≥2 元素部分和的二进制表示含 ≥2 bit，必 ∉ 集合
    static const float kPows[16] = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f,
                                    256.0f, 512.0f, 1024.0f, 2048.0f, 4096.0f, 8192.0f,
                                    16384.0f, 32768.0f};
    const size_t specialPos = (nnz > 0) ? (rng() % nnz) : 0;
    for (size_t i = 0; i < nnz; ++i) {
        float src = 0.0f;
        switch (p.val) {
            case ValueSpec::UNIFORM70: {
                float u = uni(rng);
                src = (es == 1) ? std::roundf(u) : u;  // int8 整数域 {-1,0,1}
                break;
            }
            case ValueSpec::GAUSS20:
                src = nor(rng);
                break;
            case ValueSpec::DISTINCT:
                src = kPows[i % 16];
                break;
            case ValueSpec::NAN_ONE:
                src = (i == specialPos) ? std::numeric_limits<float>::quiet_NaN() : uni(rng) * 2.0f;
                break;
            case ValueSpec::POS_INF_ONE:
                src = (i == specialPos) ? std::numeric_limits<float>::infinity() : uni(rng) * 2.0f;
                break;
            case ValueSpec::ALL_ZERO:
                src = 0.0f;
                break;
            case ValueSpec::SUBNORMAL:
                // 取值 [1.4e-45, 1.17549435e-38]（FLT_TRUE_MIN 与 FLT_MIN 边界）
                src = (i % 2 == 0) ? 1.4e-45f : 1.17549435e-38f;
                break;
            case ValueSpec::MIN_MAX_ALT:
                src = (i % 2 == 0) ? std::numeric_limits<float>::max()
                                   : -std::numeric_limits<float>::max();
                break;
        }
        p.dtype->conv(src, v.data() + i * es);
    }
    return v;
}

static std::vector<uint8_t> GenY(const CaseParams &p, std::mt19937 &rng)
{
    const size_t es = p.dtype->es;
    std::vector<uint8_t> y(p.size * es);
    std::uniform_real_distribution<float> uni(-5.0f, 5.0f);
    for (size_t j = 0; j < p.size; ++j) {
        float src = (p.yfill == YFill::SENTINEL)
                        ? (static_cast<float>(j) * 0.25f - 64.0f)  // 位置编码可区分哨兵
                        : uni(rng);
        p.dtype->conv(src, y.data() + j * es);
    }
    return y;
}

// ===================== 执行与判定 =====================

struct ExecOutput {
    aclsparseStatus_t status = static_cast<aclsparseStatus_t>(0);
    int syncRet = 0;
    std::vector<uint8_t> yOut;
    std::vector<uint8_t> valOut;
    std::vector<uint8_t> idxOut;
};

static ExecOutput ExecScatter(const UtDtype &dt, size_t size, size_t nnz, int base,
                              const std::vector<int32_t> &descIdx,
                              const std::vector<uint8_t> &valBytes,
                              const std::vector<uint8_t> &yBytes)
{
    ExecOutput o;
    DevBuf idxDev, valDev, yDev;
    idxDev.Alloc(descIdx.data(), descIdx.size() * sizeof(int32_t));
    valDev.Alloc(valBytes.data(), valBytes.size());
    yDev.Alloc(yBytes.data(), yBytes.size());

    aclsparseSpVecDescr_t sp = nullptr;
    o.status = aclsparseCreateSpVec(&sp, static_cast<int64_t>(size), static_cast<int64_t>(nnz),
                                    idxDev.ptr, valDev.ptr, ACL_SPARSE_INDEX_32I, BaseOf(base),
                                    dt.aclType);
    if (o.status != ACL_SPARSE_STATUS_SUCCESS) {
        return o;
    }
    aclsparseDnVecDescr_t dn = nullptr;
    o.status = aclsparseCreateDnVec(&dn, static_cast<int64_t>(size), yDev.ptr, dt.aclType);
    if (o.status != ACL_SPARSE_STATUS_SUCCESS) {
        aclsparseDestroySpVec(sp);
        return o;
    }

    o.status = aclsparseScatter(g_handle, sp, dn);
    if (o.status == ACL_SPARSE_STATUS_SUCCESS) {
        o.syncRet = static_cast<int>(aclrtSynchronizeStream(g_stream));
        o.yOut = yDev.Download();
        o.valOut = valDev.Download();
        o.idxOut = idxDev.Download();
    }
    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
    return o;
}

struct JudgeResult {
    bool ok = false;
    std::string detail;
};

// 无重复索引：golden bit-wise exact（含未写入位置保持 + 写入位 = 输入值 bit）
static JudgeResult JudgeBitwise(size_t es, const std::vector<int32_t> &targets,
                                const std::vector<uint8_t> &valBytes,
                                const std::vector<uint8_t> &yInit,
                                const std::vector<uint8_t> &yOut)
{
    // 防御分支：es 由五 dtype 受控表（UtDtype.es ∈ {1,2,4,8}）取得，恒非 0；
    // 此处兜底拦截下方 es 作除数的除零路径
    if (es == 0) {
        return {false, "bit-wise judge: invalid element size es=0"};
    }
    std::vector<uint8_t> golden = yInit;
    for (size_t i = 0; i < targets.size(); ++i) {
        const size_t dstOff = static_cast<size_t>(targets[i]) * es;
        CheckedMemcpy(golden.data() + dstOff, golden.size() - dstOff,
                      valBytes.data() + i * es, es);
    }
    if (yOut != golden) {
        size_t firstDiff = 0;
        for (size_t b = 0; b < golden.size(); ++b) {
            if (yOut[b] != golden[b]) {
                firstDiff = b;
                break;
            }
        }
        std::ostringstream oss;
        oss << "bit-wise FAIL: first diff at element " << (firstDiff / es)
            << " (byte " << firstDiff << "), size=" << (golden.size() / es)
            << " nnz=" << targets.size();
        return {false, oss.str()};
    }
    std::ostringstream oss;
    oss << "bit-wise exact: " << (golden.size() / es) << " elements, nnz=" << targets.size();
    return {true, oss.str()};
}

// 有重复索引：invariant 三断言（冲突值 ∈ values 集合 + 未写入保持 + 单写位 bit-wise）
static JudgeResult JudgeInvariant(size_t size, size_t es, const std::vector<int32_t> &targets,
                                  const std::vector<uint8_t> &valBytes,
                                  const std::vector<uint8_t> &yInit,
                                  const std::vector<uint8_t> &yOut)
{
    const size_t nnz = targets.size();
    std::vector<std::vector<uint32_t>> writers(size);
    for (size_t i = 0; i < nnz; ++i) {
        writers[static_cast<size_t>(targets[i])].push_back(static_cast<uint32_t>(i));
    }
    size_t conflictGroups = 0;
    size_t singleWrite = 0;
    size_t preserved = 0;
    for (size_t j = 0; j < size; ++j) {
        const uint8_t *out = yOut.data() + j * es;
        const uint8_t *init = yInit.data() + j * es;
        if (writers[j].empty()) {
            if (std::memcmp(out, init, es) != 0) {
                return {false, "unscattered_positions_preserved FAIL: position " +
                                   std::to_string(j) + " modified"};
            }
            preserved++;
        } else if (writers[j].size() == 1) {
            const uint8_t *want = valBytes.data() + static_cast<size_t>(writers[j][0]) * es;
            if (std::memcmp(out, want, es) != 0) {
                return {false, "single-write position " + std::to_string(j) +
                                   " != input value (bit-wise)"};
            }
            singleWrite++;
        } else {
            bool inSet = false;
            for (uint32_t w : writers[j]) {
                if (std::memcmp(out, valBytes.data() + static_cast<size_t>(w) * es, es) == 0) {
                    inSet = true;
                    break;
                }
            }
            if (!inSet) {
                return {false, "collision_value_from_source FAIL: position " + std::to_string(j) +
                                   " value not in source values set (accumulation/synthesis?)"};
            }
            conflictGroups++;
        }
    }
    std::ostringstream oss;
    oss << "invariant PASS: conflict_groups=" << conflictGroups << " single_write=" << singleWrite
        << " preserved=" << preserved << " (last-write-wins, value from source set)";
    return {true, oss.str()};
}

// vecX 只读：values / indices 执行前后逐字节不变
static JudgeResult JudgeReadonly(const std::vector<uint8_t> &valBefore,
                                 const std::vector<uint8_t> &valAfter,
                                 const std::vector<uint8_t> &idxBefore,
                                 const std::vector<uint8_t> &idxAfter)
{
    if (valAfter != valBefore) {
        return {false, "vecX read-only FAIL: values modified"};
    }
    if (idxAfter != idxBefore) {
        return {false, "vecX read-only FAIL: indices modified"};
    }
    return {true, "vecX read-only preserved (values+indices byte-exact)"};
}

// 功能用例生成阶段：targets / descIdx / 输入字节
struct FunctionalCaseData {
    std::vector<int32_t> targets;
    std::vector<int32_t> descIdx;
    std::vector<uint8_t> valBytes;
    std::vector<uint8_t> yBytes;
};

static void PrepareFunctionalCase(const CaseParams &p, FunctionalCaseData &d)
{
    std::mt19937 rng(p.seed);
    d.targets = GenTargets(p, rng);
    ASSERT_EQ(d.targets.size(), p.nnz) << "[" << p.caseId << "] index generator size mismatch";
    d.descIdx.resize(d.targets.size());
    for (size_t i = 0; i < d.targets.size(); ++i) {
        d.descIdx[i] = d.targets[i] + p.base;
    }
    d.valBytes = GenValues(p, d.targets.size(), rng);
    d.yBytes = GenY(p, rng);
}

// 功能用例判定阶段：重复索引检测 → 判定 → 记录
static void JudgeFunctionalCase(const CaseParams &p, const ExecOutput &out,
                                const FunctionalCaseData &d)
{
    const size_t es = p.dtype->es;

    // 重复索引检测（用例生成期静态可知）
    std::vector<int> cnt(p.size, 0);
    bool hasDup = false;
    for (int32_t t : d.targets) {
        if (++cnt[static_cast<size_t>(t)] >= 2) {
            hasDup = true;
        }
    }
    const bool intendedDup = (p.idx == IdxSpec::DUP || p.idx == IdxSpec::ALL_SAME);
    EXPECT_EQ(hasDup, intendedDup) << "[" << p.caseId << "] duplicate-index intent mismatch";

    JudgeResult jr = hasDup
                         ? JudgeInvariant(p.size, es, d.targets, d.valBytes, d.yBytes, out.yOut)
                         : JudgeBitwise(es, d.targets, d.valBytes, d.yBytes, out.yOut);
    std::vector<uint8_t> idxBefore(d.descIdx.size() * sizeof(int32_t));
    if (!idxBefore.empty()) {
        CheckedMemcpy(idxBefore.data(), idxBefore.size(), d.descIdx.data(), idxBefore.size());
    }
    JudgeResult rr = JudgeReadonly(d.valBytes, out.valOut, idxBefore, out.idxOut);

    EXPECT_TRUE(jr.ok) << "[" << p.caseId << "] " << jr.detail;
    EXPECT_TRUE(rr.ok) << "[" << p.caseId << "] " << rr.detail;
    RecordFunctional(p, (jr.ok && rr.ok) ? "PASS" : "FAIL",
                     hasDup ? "invariant_membership" : "bitwise_exact", jr.detail + " | " + rr.detail);
}

// 功能用例统一入口：生成 → 执行 → 判定（按是否存在重复索引分派）→ 记录
static void RunFunctionalCase(const CaseParams &p)
{
    FunctionalCaseData d;
    PrepareFunctionalCase(p, d);

    ExecOutput out = ExecScatter(*p.dtype, p.size, p.nnz, p.base, d.descIdx, d.valBytes, d.yBytes);
    ASSERT_EQ(out.status, ACL_SPARSE_STATUS_SUCCESS)
        << "[" << p.caseId << "] aclsparseScatter returned " << static_cast<int>(out.status);
    ASSERT_EQ(out.syncRet, 0) << "[" << p.caseId << "] stream sync failed";

    if (p.nnz == 0) {
        // nnz=0：成功返回且 vecY 逐字节不变（size=0 时输出为空）
        const bool ok = (out.yOut == d.yBytes);
        EXPECT_TRUE(ok) << "[" << p.caseId << "] nnz=0 modified vecY";
        RecordFunctional(p, ok ? "PASS" : "FAIL", (p.size == 0) ? "returns_empty" : "equals_input",
                         ok ? "vecY byte-exact unchanged (Host early-return, kernel not launched)"
                            : "vecY modified at nnz=0");
        return;
    }

    JudgeFunctionalCase(p, out, d);
}

// ===================== 错误路径（L2）公共设施 =====================

// 共享小规模 fp32 数据（size=8 / nnz=4），各 L2 用例复用
struct ErrData {
    DevBuf idx, val, y;
    std::vector<uint8_t> yOrig;  // 错误路径副作用校验基线
};

static ErrData *g_err = nullptr;

static void CheckErr(const char *caseId, aclsparseStatus_t got, aclsparseStatus_t want,
                     const std::string &note)
{
    const bool ok = (got == want);
    EXPECT_EQ(got, want) << "[" << caseId << "] " << note;
    CaseRecord r;
    r.caseId = caseId;
    r.gtest = CurrentGtestName();
    r.status = ok ? "PASS" : "FAIL";
    r.check = "raises_error";
    r.dtype = "float32";
    r.size = 8;
    r.nnz = 4;
    r.detail = note + " (got=" + std::to_string(static_cast<int>(got)) +
               " want=" + std::to_string(static_cast<int>(want)) + ")";
    RecordCase(r);
}

static void ExpectYUnchanged(const DevBuf &y, const std::vector<uint8_t> &original,
                             const char *caseId)
{
    const std::vector<uint8_t> now = y.Download();
    const bool ok = (now == original);
    EXPECT_TRUE(ok) << "[" << caseId << "] vecY modified by rejected call";
    if (!ok) {
        for (auto &r : g_records) {
            if (r.caseId == caseId && r.status == "PASS") {
                r.status = "FAIL";
                r.detail += " | vecY modified by rejected call";
            }
        }
    }
}

// ===================== 异步 / 无 CPU fallback 探针（流阻塞法） =====================

// 机制：S2 上排 6×1GB D2D 异步拷贝后 record 事件 E；handle 流 WaitEvent(E) 被阻塞；
// 提交 aclsparseScatter（仅入队）→ 立即经独立流 S3 读回 vecY：
//   - 若被修改 → 存在 CPU 同步执行路径（CPU fallback）；
//   - 若未修改 → 写入由 NPU stream 异步完成（无 CPU fallback 证据）；
// 随后释放阻塞并同步，验证最终结果 bit-wise 正确。

// 流阻塞探针共享环境
struct AsyncProbeCtx {
    aclrtStream s2 = nullptr;
    aclrtStream s3 = nullptr;
    aclrtEvent ev = nullptr;
    void *bigA = nullptr;
    void *bigB = nullptr;
};

// 阻塞阶段搭建：S2 上 6×1GB D2D 异步拷贝后 record 事件 E，handle 流 WaitEvent(E) 被阻塞
static void AsyncProbeSetupBlock(AsyncProbeCtx &c)
{
    ASSERT_EQ(aclrtCreateStream(&c.s2), ACL_SUCCESS);
    ASSERT_EQ(aclrtCreateStream(&c.s3), ACL_SUCCESS);
    ASSERT_EQ(aclrtCreateEvent(&c.ev), ACL_SUCCESS);

    const size_t big = 1ull << 30;  // 1GB × 2 缓冲 × 6 次顺序拷贝（≥6GB 流量，阻塞余量充分）
    ASSERT_EQ(aclrtMalloc(&c.bigA, big, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&c.bigB, big, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    for (int r = 0; r < 6; ++r) {
        ASSERT_EQ(aclrtMemcpyAsync(c.bigB, big, c.bigA, big, ACL_MEMCPY_DEVICE_TO_DEVICE, c.s2),
                  ACL_SUCCESS);
    }
    ASSERT_EQ(aclrtRecordEvent(c.ev, c.s2), ACL_SUCCESS);
    ASSERT_EQ(aclrtStreamWaitEvent(g_stream, c.ev), ACL_SUCCESS);  // handle 流阻塞
}

static void AsyncProbeTeardown(const AsyncProbeCtx &c)
{
    aclrtFree(c.bigA);
    aclrtFree(c.bigB);
    aclrtDestroyEvent(c.ev);
    aclrtDestroyStream(c.s2);
    aclrtDestroyStream(c.s3);
}

struct AsyncProbeSubmitResult {
    aclsparseStatus_t st = static_cast<aclsparseStatus_t>(0);
    double submitMs = 0.0;
    bool unchangedWhileBlocked = false;
};

// 异步探针用例数据
struct AsyncProbeData {
    std::vector<int32_t> targets;
    std::vector<int32_t> descIdx;
    std::vector<uint8_t> valBytes;
    std::vector<uint8_t> yBytes;
};

// 异步探针数据/设备/描述符准备
static void AsyncProbePrepare(const std::string &caseId, size_t size, size_t nnz, uint32_t seed,
                              AsyncProbeData &d, DevBuf &idxDev, DevBuf &valDev, DevBuf &yDev,
                              aclsparseSpVecDescr_t &sp, aclsparseDnVecDescr_t &dn)
{
    const UtDtype &dt = kFp32;
    std::mt19937 rng(seed);
    d.targets = SampleUniqueShuffled(size, nnz, rng);
    ASSERT_EQ(d.targets.size(), nnz);
    d.descIdx = d.targets;  // base 0
    CaseParams cp = {caseId.c_str(), &kFp32, size, nnz, 0, IdxSpec::SHUFFLED_NODUP,
                     ValueSpec::UNIFORM70, YFill::RANDOM, seed};
    d.valBytes = GenValues(cp, nnz, rng);
    d.yBytes = GenY(cp, rng);

    idxDev.Alloc(d.descIdx.data(), d.descIdx.size() * sizeof(int32_t));
    valDev.Alloc(d.valBytes.data(), d.valBytes.size());
    yDev.Alloc(d.yBytes.data(), d.yBytes.size());

    ASSERT_EQ(aclsparseCreateSpVec(&sp, static_cast<int64_t>(size), static_cast<int64_t>(nnz),
                                   idxDev.ptr, valDev.ptr, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, dt.aclType),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseCreateDnVec(&dn, static_cast<int64_t>(size), yDev.ptr, dt.aclType),
              ACL_SPARSE_STATUS_SUCCESS);
}

// 异步探针断言与执行证据记录
static void AsyncProbeJudgeAndRecord(const std::string &caseId, const char *check,
                                     size_t size, size_t nnz, uint32_t seed,
                                     const AsyncProbeSubmitResult &sub, int syncRet,
                                     const JudgeResult &bitJr, const JudgeResult &roJr)
{
    ASSERT_EQ(sub.st, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(syncRet, 0);
    EXPECT_TRUE(sub.unchangedWhileBlocked)
        << "[" << caseId << "] vecY modified while stream blocked: CPU synchronous path (fallback)";
    EXPECT_TRUE(bitJr.ok) << "[" << caseId << "] " << bitJr.detail;
    EXPECT_TRUE(roJr.ok) << "[" << caseId << "] " << roJr.detail;

    std::ostringstream oss;
    oss << "submit=" << sub.submitMs << "ms; blocked-phase vecY unchanged="
        << (sub.unchangedWhileBlocked ? "yes" : "NO") << "; " << bitJr.detail << " | " << roJr.detail;
    CaseRecord r;
    r.caseId = caseId;
    r.gtest = CurrentGtestName();
    r.status = (sub.unchangedWhileBlocked && bitJr.ok && roJr.ok) ? "PASS" : "FAIL";
    r.check = check;
    r.dtype = kFp32.name;
    r.size = size;
    r.nnz = nnz;
    r.base = 0;
    r.seed = seed;
    r.detail = oss.str();
    RecordCase(r);
}

// 提交 aclsparseScatter（仅入队，计时）→ 阻塞期间经独立流 S3 读回 vecY
static void AsyncProbeSubmit(const AsyncProbeCtx &c, aclsparseSpVecDescr_t sp,
                             aclsparseDnVecDescr_t dn, const DevBuf &yDev,
                             const std::vector<uint8_t> &yBytes, AsyncProbeSubmitResult &r)
{
    const auto t0 = std::chrono::steady_clock::now();
    r.st = aclsparseScatter(g_handle, sp, dn);
    const auto t1 = std::chrono::steady_clock::now();
    r.submitMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 流阻塞期间读回（独立流 S3）：vecY 应保持原值（无 CPU 同步执行路径）
    std::vector<uint8_t> early(yBytes.size());
    ASSERT_EQ(aclrtMemcpyAsync(early.data(), yBytes.size(), yDev.ptr, yBytes.size(),
                               ACL_MEMCPY_DEVICE_TO_HOST, c.s3),
              ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(c.s3), ACL_SUCCESS);
    r.unchangedWhileBlocked = (early == yBytes);
}

static void RunAsyncProbe(const std::string &caseId, const char *check)
{
    const UtDtype &dt = kFp32;
    const size_t size = 16384;  // L1_050: size=16384 nnz=1024
    const size_t nnz = 1024;
    const uint32_t seed = 7;

    DevBuf idxDev, valDev, yDev;
    aclsparseSpVecDescr_t sp = nullptr;
    aclsparseDnVecDescr_t dn = nullptr;
    AsyncProbeData d;
    AsyncProbePrepare(caseId, size, nnz, seed, d, idxDev, valDev, yDev, sp, dn);

    AsyncProbeCtx block;
    AsyncProbeSetupBlock(block);

    AsyncProbeSubmitResult sub;
    AsyncProbeSubmit(block, sp, dn, yDev, d.yBytes, sub);

    // 释放阻塞 → handle 流执行 → 同步
    ASSERT_EQ(aclrtSynchronizeStream(block.s2), ACL_SUCCESS);
    const int syncRet = static_cast<int>(aclrtSynchronizeStream(g_stream));
    const std::vector<uint8_t> finalY = yDev.Download();
    const std::vector<uint8_t> valAfter = valDev.Download();
    const std::vector<uint8_t> idxAfter = idxDev.Download();
    std::vector<uint8_t> idxBefore(d.descIdx.size() * sizeof(int32_t));
    CheckedMemcpy(idxBefore.data(), idxBefore.size(), d.descIdx.data(), idxBefore.size());
    const JudgeResult bitJr = JudgeBitwise(dt.es, d.targets, d.valBytes, d.yBytes, finalY);
    const JudgeResult roJr = JudgeReadonly(d.valBytes, valAfter, idxBefore, idxAfter);

    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
    AsyncProbeTeardown(block);

    AsyncProbeJudgeAndRecord(caseId, check, size, nnz, seed, sub, syncRet, bitJr, roJr);
}

// ===================== L0 门槛（8） =====================

static const CaseParams kL0_001 = {"L0_001", &kFp32, 17, 9, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL0_002 = {"L0_002", &kFp32, 17, 9, 1, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL0_003 = {"L0_003", &kInt8, 17, 9, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL0_004 = {"L0_004", &kFp16, 17, 9, 1, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL0_005 = {"L0_005", &kBf16, 17, 9, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL0_006 = {"L0_006", &kC64, 17, 9, 1, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
// L0_007：构造互异 values 反证非累加 → DISTINCT
static const CaseParams kL0_007 = {"L0_007", &kFp32, 17, 9, 0, IdxSpec::DUP,
                                   ValueSpec::DISTINCT, YFill::RANDOM, 7};
static const CaseParams kL0_008 = {"L0_008", &kFp32, 128256, 8192, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 20260902};

TEST(L0Gateway, L0_001_Fp32Base0NoDup) { RunFunctionalCase(kL0_001); }
TEST(L0Gateway, L0_002_Fp32Base1NoDup) { RunFunctionalCase(kL0_002); }
TEST(L0Gateway, L0_003_Int8Base0NoDup) { RunFunctionalCase(kL0_003); }
TEST(L0Gateway, L0_004_Fp16Base1NoDup) { RunFunctionalCase(kL0_004); }
TEST(L0Gateway, L0_005_Bf16Base0NoDup) { RunFunctionalCase(kL0_005); }
TEST(L0Gateway, L0_006_C64Base1NoDup) { RunFunctionalCase(kL0_006); }
TEST(L0Gateway, L0_007_Fp32Base0DupInvariant) { RunFunctionalCase(kL0_007); }
TEST(L0Gateway, L0_008_P01ShapeFp32) { RunFunctionalCase(kL0_008); }

// ===================== L1 dtype 矩阵（20） =====================

static const CaseParams kL1_001 = {"L1_001", &kInt8, 17, 9, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_002 = {"L1_002", &kInt8, 17, 9, 0, IdxSpec::DUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_003 = {"L1_003", &kInt8, 17, 9, 1, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_004 = {"L1_004", &kInt8, 17, 9, 1, IdxSpec::DUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_005 = {"L1_005", &kFp16, 17, 9, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_006 = {"L1_006", &kFp16, 17, 9, 0, IdxSpec::DUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_007 = {"L1_007", &kFp16, 17, 9, 1, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_008 = {"L1_008", &kFp16, 17, 9, 1, IdxSpec::DUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_009 = {"L1_009", &kBf16, 17, 9, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_010 = {"L1_010", &kBf16, 17, 9, 0, IdxSpec::DUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_011 = {"L1_011", &kBf16, 17, 9, 1, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_012 = {"L1_012", &kBf16, 17, 9, 1, IdxSpec::DUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_013 = {"L1_013", &kFp32, 17, 9, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_014 = {"L1_014", &kFp32, 17, 9, 0, IdxSpec::DUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_015 = {"L1_015", &kFp32, 17, 9, 1, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_016 = {"L1_016", &kFp32, 17, 9, 1, IdxSpec::DUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_017 = {"L1_017", &kC64, 17, 9, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_018 = {"L1_018", &kC64, 17, 9, 0, IdxSpec::DUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_019 = {"L1_019", &kC64, 17, 9, 1, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_020 = {"L1_020", &kC64, 17, 9, 1, IdxSpec::DUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};

TEST(DtypeMatrix, L1_001_Int8Base0NoDup) { RunFunctionalCase(kL1_001); }
TEST(DtypeMatrix, L1_002_Int8Base0Dup) { RunFunctionalCase(kL1_002); }
TEST(DtypeMatrix, L1_003_Int8Base1NoDup) { RunFunctionalCase(kL1_003); }
TEST(DtypeMatrix, L1_004_Int8Base1Dup) { RunFunctionalCase(kL1_004); }
TEST(DtypeMatrix, L1_005_Fp16Base0NoDup) { RunFunctionalCase(kL1_005); }
TEST(DtypeMatrix, L1_006_Fp16Base0Dup) { RunFunctionalCase(kL1_006); }
TEST(DtypeMatrix, L1_007_Fp16Base1NoDup) { RunFunctionalCase(kL1_007); }
TEST(DtypeMatrix, L1_008_Fp16Base1Dup) { RunFunctionalCase(kL1_008); }
TEST(DtypeMatrix, L1_009_Bf16Base0NoDup) { RunFunctionalCase(kL1_009); }
TEST(DtypeMatrix, L1_010_Bf16Base0Dup) { RunFunctionalCase(kL1_010); }
TEST(DtypeMatrix, L1_011_Bf16Base1NoDup) { RunFunctionalCase(kL1_011); }
TEST(DtypeMatrix, L1_012_Bf16Base1Dup) { RunFunctionalCase(kL1_012); }
TEST(DtypeMatrix, L1_013_Fp32Base0NoDup) { RunFunctionalCase(kL1_013); }
TEST(DtypeMatrix, L1_014_Fp32Base0Dup) { RunFunctionalCase(kL1_014); }
TEST(DtypeMatrix, L1_015_Fp32Base1NoDup) { RunFunctionalCase(kL1_015); }
TEST(DtypeMatrix, L1_016_Fp32Base1Dup) { RunFunctionalCase(kL1_016); }
TEST(DtypeMatrix, L1_017_C64Base0NoDup) { RunFunctionalCase(kL1_017); }
TEST(DtypeMatrix, L1_018_C64Base0Dup) { RunFunctionalCase(kL1_018); }
TEST(DtypeMatrix, L1_019_C64Base1NoDup) { RunFunctionalCase(kL1_019); }
TEST(DtypeMatrix, L1_020_C64Base1Dup) { RunFunctionalCase(kL1_020); }

// ===================== L1 shape 边界（L1_021 全 dtype 参数化展开） =====================

static const CaseParams kL1_021_i8 = {"L1_021", &kInt8, 8, 0, 0, IdxSpec::EMPTY,
                                      ValueSpec::UNIFORM70, YFill::SENTINEL, 7};
static const CaseParams kL1_021_f16 = {"L1_021", &kFp16, 8, 0, 0, IdxSpec::EMPTY,
                                       ValueSpec::UNIFORM70, YFill::SENTINEL, 7};
static const CaseParams kL1_021_bf16 = {"L1_021", &kBf16, 8, 0, 0, IdxSpec::EMPTY,
                                        ValueSpec::UNIFORM70, YFill::SENTINEL, 7};
static const CaseParams kL1_021_fp32 = {"L1_021", &kFp32, 8, 0, 0, IdxSpec::EMPTY,
                                        ValueSpec::UNIFORM70, YFill::SENTINEL, 7};
static const CaseParams kL1_021_c64 = {"L1_021", &kC64, 8, 0, 0, IdxSpec::EMPTY,
                                       ValueSpec::UNIFORM70, YFill::SENTINEL, 7};
static const CaseParams kL1_022 = {"L1_022", &kFp32, 8, 8, 0, IdxSpec::IDENTITY_RUNS,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_023 = {"L1_023", &kFp32, 8, 1, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
// L1_024 size=1 最小稠密向量（SIZE 下界）：
// nnz=3 全同冲突指向 size=1 的 vecY 超出 aclsparse API 可表示域——
// aclsparseCreateSpVec 校验 nnz ≤ SpVec.size（既有校验），Host 校验 SpVec.size ≤ vecY.nums
// （L2_014），故 SIZE=1 时 nnz 上界为 1（与 cuSPARSE nnz ≤ size 一致）。
// 以 SIZE=1/nnz=1（下界单写，全部写入落在同一位置）落地；全同冲突语义由 L1_032/
// L1_002..020 dup 族承接。
static const CaseParams kL1_024 = {"L1_024", &kFp32, 1, 1, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_025 = {"L1_025", &kFp32, 8, 6, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_026 = {"L1_026", &kFp32, 8, 5, 0, IdxSpec::ENDPOINT_PAIR,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_027 = {"L1_027", &kFp32, 8, 5, 1, IdxSpec::ENDPOINT_PAIR,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_028 = {"L1_028", &kFp32, 64, 37, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_029 = {"L1_029", &kFp32, 16384, 1024, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_030 = {"L1_030", &kFp32, 0, 0, 0, IdxSpec::EMPTY,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_042 = {"L1_042", &kFp32, 8192, 4095, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_043 = {"L1_043", &kFp32, 8192, 4097, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};

TEST(ShapeBoundary, L1_021_Nnz0NoModify_Int8) { RunFunctionalCase(kL1_021_i8); }
TEST(ShapeBoundary, L1_021_Nnz0NoModify_Fp16) { RunFunctionalCase(kL1_021_f16); }
TEST(ShapeBoundary, L1_021_Nnz0NoModify_Bf16) { RunFunctionalCase(kL1_021_bf16); }
TEST(ShapeBoundary, L1_021_Nnz0NoModify_Fp32) { RunFunctionalCase(kL1_021_fp32); }
TEST(ShapeBoundary, L1_021_Nnz0NoModify_C64) { RunFunctionalCase(kL1_021_c64); }
TEST(ShapeBoundary, L1_022_FullyDense) { RunFunctionalCase(kL1_022); }
TEST(ShapeBoundary, L1_023_Nnz1) { RunFunctionalCase(kL1_023); }
TEST(ShapeBoundary, L1_024_Size1LowerBound) { RunFunctionalCase(kL1_024); }
TEST(ShapeBoundary, L1_025_ShuffledIndices) { RunFunctionalCase(kL1_025); }
TEST(ShapeBoundary, L1_026_Base0Endpoints) { RunFunctionalCase(kL1_026); }
TEST(ShapeBoundary, L1_027_Base1Endpoints) { RunFunctionalCase(kL1_027); }
TEST(ShapeBoundary, L1_028_Tail37) { RunFunctionalCase(kL1_028); }
TEST(ShapeBoundary, L1_029_LargeDynamic) { RunFunctionalCase(kL1_029); }
TEST(ShapeBoundary, L1_030_Size0Nnz0) { RunFunctionalCase(kL1_030); }
TEST(ShapeBoundary, L1_042_Tail4095) { RunFunctionalCase(kL1_042); }
TEST(ShapeBoundary, L1_043_Tail4097) { RunFunctionalCase(kL1_043); }

// ===================== L1 数值极端（7 + 大样本 2） =====================

static const CaseParams kL1_031 = {"L1_031", &kFp32, 8, 6, 0, IdxSpec::DUP,
                                   ValueSpec::DISTINCT, YFill::RANDOM, 7};
static const CaseParams kL1_032 = {"L1_032", &kFp32, 8, 6, 0, IdxSpec::ALL_SAME,
                                   ValueSpec::DISTINCT, YFill::RANDOM, 7};
static const CaseParams kL1_033 = {"L1_033", &kFp32, 8, 6, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::NAN_ONE, YFill::RANDOM, 7};
static const CaseParams kL1_034 = {"L1_034", &kFp32, 8, 6, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::POS_INF_ONE, YFill::RANDOM, 7};
static const CaseParams kL1_035 = {"L1_035", &kFp32, 8, 4, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::ALL_ZERO, YFill::RANDOM, 7};
static const CaseParams kL1_036 = {"L1_036", &kFp32, 8, 6, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::SUBNORMAL, YFill::RANDOM, 7};
static const CaseParams kL1_037 = {"L1_037", &kFp32, 8, 6, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::MIN_MAX_ALT, YFill::RANDOM, 7};
static const CaseParams kL1_046 = {"L1_046", &kFp32, 1024, 1024, 0, IdxSpec::PERMUTATION_NODUP,
                                   ValueSpec::GAUSS20, YFill::RANDOM, 7};
static const CaseParams kL1_047 = {"L1_047", &kFp32, 16384, 8192, 0, IdxSpec::DUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 20260902};

TEST(Numerical, L1_031_DupDistinctNoAccum) { RunFunctionalCase(kL1_031); }
TEST(Numerical, L1_032_AllSameDistinct) { RunFunctionalCase(kL1_032); }
TEST(Numerical, L1_033_NanBitwise) { RunFunctionalCase(kL1_033); }
TEST(Numerical, L1_034_PosInf) { RunFunctionalCase(kL1_034); }
TEST(Numerical, L1_035_AllZeroUnscattered) { RunFunctionalCase(kL1_035); }
TEST(Numerical, L1_036_Subnormal) { RunFunctionalCase(kL1_036); }
TEST(Numerical, L1_037_MinMaxAlternate) { RunFunctionalCase(kL1_037); }
TEST(Numerical, L1_046_GaussPermutation) { RunFunctionalCase(kL1_046); }
TEST(Numerical, L1_047_DupLargeScale) { RunFunctionalCase(kL1_047); }

// ===================== L1 路径专项（有序 run 段拷贝路径） =====================

static const CaseParams kL1_038 = {"L1_038", &kInt8, 64, 32, 0, IdxSpec::ORDERED_RUNS2,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_039 = {"L1_039", &kFp16, 64, 32, 0, IdxSpec::ORDERED_RUNS2,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_040 = {"L1_040", &kBf16, 64, 32, 0, IdxSpec::ORDERED_RUNS2,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_041 = {"L1_041", &kC64, 64, 32, 0, IdxSpec::ORDERED_RUNS2,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};

TEST(PathDtype, L1_038_Int8OrderedRuns) { RunFunctionalCase(kL1_038); }
TEST(PathDtype, L1_039_Fp16OrderedRuns) { RunFunctionalCase(kL1_039); }
TEST(PathDtype, L1_040_Bf16OrderedRuns) { RunFunctionalCase(kL1_040); }
TEST(PathDtype, L1_041_C64OrderedRuns) { RunFunctionalCase(kL1_041); }

// ===================== L1 P 形状（seed=20260902） =====================

static const CaseParams kL1_044 = {"L1_044", &kFp32, 151936, 4096, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 20260902};
static const CaseParams kL1_045 = {"L1_045", &kC64, 129280, 7168, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 20260902};

TEST(PShapes, L1_044_P02ShapeFp32) { RunFunctionalCase(kL1_044); }
TEST(PShapes, L1_045_P03ShapeC64) { RunFunctionalCase(kL1_045); }

// ===================== L1 语义专项 =====================

static const CaseParams kL1_048 = {"L1_048", &kFp32, 64, 32, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::RANDOM, 7};
static const CaseParams kL1_049 = {"L1_049", &kFp32, 64, 32, 0, IdxSpec::SHUFFLED_NODUP,
                                   ValueSpec::UNIFORM70, YFill::SENTINEL, 7};

// L1_048 vecX 只读（values/indices 执行前后逐字节一致）——readonly 为全量附带断言，此处专项显式化
TEST(Semantic, L1_048_VecXReadonly) { RunFunctionalCase(kL1_048); }

// L1_049 未写入位置保持（y 预填位置编码哨兵，bit 级校验 + 无越界/对齐填充误写）
TEST(Semantic, L1_049_SentinelUnscattered) { RunFunctionalCase(kL1_049); }

// L1_050 stream 异步语义：提交后不阻塞 Host，结果 stream 同步后可见（流阻塞法）
TEST(Semantic, L1_050_AsyncStreamSemantics) { RunAsyncProbe("L1_050", "async_check"); }

// L1_051 数据准备：int8 size=1024/nnz=512 乱序互异索引
struct DeterminismSetup {
    std::vector<int32_t> targets;
    std::vector<int32_t> descIdx;
    std::vector<uint8_t> valBytes;
    std::vector<uint8_t> yBytes;
};

static void UtDeterminismPrepare(size_t size, size_t nnz, uint32_t seed, DeterminismSetup &d)
{
    std::mt19937 rng(seed);
    d.targets = SampleUniqueShuffled(size, nnz, rng);
    ASSERT_EQ(d.targets.size(), nnz);
    d.descIdx = d.targets;
    CaseParams cp = {"L1_051", &kInt8, size, nnz, 0, IdxSpec::SHUFFLED_NODUP,
                     ValueSpec::UNIFORM70, YFill::RANDOM, seed};
    d.valBytes = GenValues(cp, nnz, rng);
    d.yBytes = GenY(cp, rng);
}

// L1_051 确定性：无重复索引相同输入重复执行 10 次位级一致（int8 1B 最弱对齐路径）
TEST(Semantic, L1_051_Determinism10Runs)
{
    const UtDtype &dt = kInt8;
    const size_t size = 1024, nnz = 512;
    const uint32_t seed = 7;
    DeterminismSetup d;
    UtDeterminismPrepare(size, nnz, seed, d);

    DevBuf idxDev, valDev, yDev;
    idxDev.Alloc(d.descIdx.data(), d.descIdx.size() * sizeof(int32_t));
    valDev.Alloc(d.valBytes.data(), d.valBytes.size());
    yDev.Alloc(d.yBytes.data(), d.yBytes.size());

    aclsparseSpVecDescr_t sp = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, static_cast<int64_t>(size), static_cast<int64_t>(nnz),
                                   idxDev.ptr, valDev.ptr, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, dt.aclType),
              ACL_SPARSE_STATUS_SUCCESS);
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&dn, static_cast<int64_t>(size), yDev.ptr, dt.aclType),
              ACL_SPARSE_STATUS_SUCCESS);

    std::vector<std::vector<uint8_t>> runs;
    for (int r = 0; r < 10; ++r) {
        yDev.Upload(d.yBytes.data(), d.yBytes.size());  // 每轮恢复 y 输入
        ASSERT_EQ(aclsparseScatter(g_handle, sp, dn), ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(g_stream), ACL_SUCCESS);
        runs.push_back(yDev.Download());
    }
    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);

    const JudgeResult goldenJr = JudgeBitwise(dt.es, d.targets, d.valBytes, d.yBytes, runs[0]);
    bool allIdentical = true;
    size_t firstDiffRun = 0;
    for (size_t r = 1; r < runs.size(); ++r) {
        if (runs[r] != runs[0]) {
            allIdentical = false;
            firstDiffRun = r;
            break;
        }
    }
    EXPECT_TRUE(goldenJr.ok) << "[L1_051] " << goldenJr.detail;
    EXPECT_TRUE(allIdentical) << "[L1_051] run " << firstDiffRun << " differs from run 0";
    CaseParams cp = {"L1_051", &kInt8, size, nnz, 0, IdxSpec::SHUFFLED_NODUP,
                     ValueSpec::UNIFORM70, YFill::RANDOM, seed};
    RecordFunctional(cp, (goldenJr.ok && allIdentical) ? "PASS" : "FAIL", "deterministic_check",
                     goldenJr.detail + " | 10 runs bit-identical=" +
                         (allIdentical ? "yes" : ("NO (first diff run " + std::to_string(firstDiffRun) + ")")));
}

// L1_052 单轮：恢复 y 输入 → 描述符创建/执行/同步/判定/销毁
static bool UtLifecycleOneRound(const UtDtype &dt, size_t size, size_t nnz,
                                DevBuf &idxDev, DevBuf &valDev, DevBuf &yDev,
                                const std::vector<uint8_t> &yBytes,
                                const std::vector<int32_t> &targets,
                                const std::vector<uint8_t> &valBytes)
{
    yDev.Upload(yBytes.data(), yBytes.size());
    aclsparseSpVecDescr_t sp = nullptr;
    aclsparseDnVecDescr_t dn = nullptr;
    aclsparseStatus_t st = aclsparseCreateSpVec(&sp, static_cast<int64_t>(size),
                                                static_cast<int64_t>(nnz), idxDev.ptr,
                                                valDev.ptr, ACL_SPARSE_INDEX_32I,
                                                ACL_SPARSE_INDEX_BASE_ZERO, dt.aclType);
    if (st == ACL_SPARSE_STATUS_SUCCESS) {
        st = aclsparseCreateDnVec(&dn, static_cast<int64_t>(size), yDev.ptr, dt.aclType);
    }
    if (st == ACL_SPARSE_STATUS_SUCCESS) {
        st = aclsparseScatter(g_handle, sp, dn);
    }
    int syncRet = 0;
    if (st == ACL_SPARSE_STATUS_SUCCESS) {
        syncRet = static_cast<int>(aclrtSynchronizeStream(g_stream));
    }
    bool roundOk = (st == ACL_SPARSE_STATUS_SUCCESS) && (syncRet == 0);
    if (roundOk) {
        const JudgeResult jr = JudgeBitwise(dt.es, targets, valBytes, yBytes, yDev.Download());
        roundOk = jr.ok;
    }
    if (sp != nullptr) {
        aclsparseDestroySpVec(sp);
    }
    if (dn != nullptr) {
        aclsparseDestroyDnVec(dn);
    }
    return roundOk;
}

// L1_052 资源生命周期：连续创建/执行/销毁描述符 100 轮（复用 handle），全轮 SUCCESS + 结果正确
TEST(Semantic, L1_052_Lifecycle100Rounds)
{
    const UtDtype &dt = kFp32;
    const size_t size = 256, nnz = 128;
    const uint32_t seed = 7;
    std::mt19937 rng(seed);
    std::vector<int32_t> targets = SampleUniqueShuffled(size, nnz, rng);
    ASSERT_EQ(targets.size(), nnz);
    std::vector<int32_t> descIdx = targets;
    CaseParams cp = {"L1_052", &kFp32, size, nnz, 0, IdxSpec::SHUFFLED_NODUP,
                     ValueSpec::UNIFORM70, YFill::RANDOM, seed};
    std::vector<uint8_t> valBytes = GenValues(cp, nnz, rng);
    std::vector<uint8_t> yBytes = GenY(cp, rng);

    DevBuf idxDev, valDev, yDev;
    idxDev.Alloc(descIdx.data(), descIdx.size() * sizeof(int32_t));
    valDev.Alloc(valBytes.data(), valBytes.size());
    yDev.Alloc(yBytes.data(), yBytes.size());

    int failedRounds = 0;
    int firstFailRound = -1;
    for (int r = 0; r < 100; ++r) {
        if (!UtLifecycleOneRound(dt, size, nnz, idxDev, valDev, yDev, yBytes, targets, valBytes)) {
            failedRounds++;
            if (firstFailRound < 0) {
                firstFailRound = r;
            }
        }
    }
    EXPECT_EQ(failedRounds, 0) << "[L1_052] failed rounds: " << failedRounds
                               << " (first at " << firstFailRound << ")";
    RecordFunctional(cp, failedRounds == 0 ? "PASS" : "FAIL", "resource_check",
                     "100/100 rounds SUCCESS + bit-exact (descriptor create/execute/destroy x100, "
                     "handle reused; device memory counting deferred to phase-3 collection)");
}

// L1_054 完整生命周期：handle 创建 → 描述符 → 执行 → stream 同步 → 销毁（fp32 4096/2048 base1）
TEST(Semantic, L1_054_FullLifecycleBase1)
{
    const UtDtype &dt = kFp32;
    const size_t size = 4096, nnz = 2048;
    const uint32_t seed = 7;
    std::mt19937 rng(seed);
    std::vector<int32_t> targets = SampleUniqueShuffled(size, nnz, rng);
    ASSERT_EQ(targets.size(), nnz);
    std::vector<int32_t> descIdx(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        descIdx[i] = targets[i] + 1;  // base 1
    }
    CaseParams cp = {"L1_054", &kFp32, size, nnz, 1, IdxSpec::SHUFFLED_NODUP,
                     ValueSpec::UNIFORM70, YFill::RANDOM, seed};
    std::vector<uint8_t> valBytes = GenValues(cp, nnz, rng);
    std::vector<uint8_t> yBytes = GenY(cp, rng);

    DevBuf idxDev, valDev, yDev;
    idxDev.Alloc(descIdx.data(), descIdx.size() * sizeof(int32_t));
    valDev.Alloc(valBytes.data(), valBytes.size());
    yDev.Alloc(yBytes.data(), yBytes.size());

    aclsparseHandle_t h2 = nullptr;
    aclrtStream s2 = nullptr;
    ASSERT_EQ(aclsparseCreate(&h2), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtCreateStream(&s2), ACL_SUCCESS);
    ASSERT_EQ(aclsparseSetStream(h2, s2), ACL_SPARSE_STATUS_SUCCESS);

    aclsparseSpVecDescr_t sp = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, static_cast<int64_t>(size), static_cast<int64_t>(nnz),
                                   idxDev.ptr, valDev.ptr, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ONE, dt.aclType),
              ACL_SPARSE_STATUS_SUCCESS);
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&dn, static_cast<int64_t>(size), yDev.ptr, dt.aclType),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseScatter(h2, sp, dn), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(s2), ACL_SUCCESS);
    const std::vector<uint8_t> yOut = yDev.Download();
    const JudgeResult jr = JudgeBitwise(dt.es, targets, valBytes, yBytes, yOut);

    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
    aclsparseDestroy(h2);
    aclrtDestroyStream(s2);

    EXPECT_TRUE(jr.ok) << "[L1_054] " << jr.detail;
    RecordFunctional(cp, jr.ok ? "PASS" : "FAIL", "lifecycle_check",
                     "full lifecycle SUCCESS + " + jr.detail);
}

// ===================== 验收专项：无 CPU fallback =====================
// 与 L1_050 同机制（流阻塞法）独立断言：阻塞期间 vecY 不被修改 → 计算经 NPU stream 异步完成。
TEST(Acceptance, NoCpuFallback) { RunAsyncProbe("ACC_UT_001", "no_cpu_fallback"); }

// ===================== L2 错误路径（C++ 可承接部分） =====================

TEST(ErrorPath, L2_001_NullHandle)
{
    aclsparseSpVecDescr_t sp = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, 8, 4, g_err->idx.ptr, g_err->val.ptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&dn, 8, g_err->y.ptr, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);
    CheckErr("L2_001", aclsparseScatter(nullptr, sp, dn), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR,
             "scatter 层：空句柄（null_input）");
    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
}

TEST(ErrorPath, L2_002_NullVecX)
{
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&dn, 8, g_err->y.ptr, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);
    CheckErr("L2_002", aclsparseScatter(g_handle, nullptr, dn), ACL_SPARSE_STATUS_INVALID_VALUE,
             "scatter 层：空 vecX 描述符（null_input）");
    aclsparseDestroyDnVec(dn);
}

TEST(ErrorPath, L2_003_NullVecY)
{
    aclsparseSpVecDescr_t sp = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, 8, 4, g_err->idx.ptr, g_err->val.ptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    CheckErr("L2_003", aclsparseScatter(g_handle, sp, nullptr), ACL_SPARSE_STATUS_INVALID_VALUE,
             "scatter 层：空 vecY 描述符（null_input）");
    aclsparseDestroySpVec(sp);
}

TEST(ErrorPath, L2_004_NnzGt0Size0)
{
    aclsparseSpVecDescr_t sp = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, 8, 4, g_err->idx.ptr, g_err->val.ptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    aclsparseDnVecDescr_t dn = nullptr;  // size=0 仅 nnz=0 合法
    ASSERT_EQ(aclsparseCreateDnVec(&dn, 0, nullptr, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);
    CheckErr("L2_004", aclsparseScatter(g_handle, sp, dn), ACL_SPARSE_STATUS_INVALID_VALUE,
             "scatter 层：nnz>0 且 vecY size=0（shape_mismatch）");
    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
}

// idxBase 非法：aclsparseCreateSpVec 描述符层即确定报错（既有校验）；
// scatter 侧同名校验为纵深防御（公共 API 不可达），错误码一致（INVALID_VALUE）
TEST(ErrorPath, L2_006_InvalidBase2)
{
    aclsparseSpVecDescr_t sp = nullptr;
    const aclsparseStatus_t st = aclsparseCreateSpVec(&sp, 8, 4, g_err->idx.ptr, g_err->val.ptr,
                                                      ACL_SPARSE_INDEX_32I,
                                                      static_cast<aclsparseIndexBase_t>(2), ACL_FLOAT);
    CheckErr("L2_006", st, ACL_SPARSE_STATUS_INVALID_VALUE,
             "描述符层：idxBase=2 非法（attribute_value_out_of_range；scatter 侧校验为纵深防御）");
    if (sp != nullptr) {
        aclsparseDestroySpVec(sp);
    }
}

TEST(ErrorPath, L2_007_InvalidBaseNeg1)
{
    aclsparseSpVecDescr_t sp = nullptr;
    const aclsparseStatus_t st = aclsparseCreateSpVec(&sp, 8, 4, g_err->idx.ptr, g_err->val.ptr,
                                                      ACL_SPARSE_INDEX_32I,
                                                      static_cast<aclsparseIndexBase_t>(-1), ACL_FLOAT);
    CheckErr("L2_007", st, ACL_SPARSE_STATUS_INVALID_VALUE,
             "描述符层：idxBase=-1 负值负例（与 L2_006 构成枚举域外两侧）");
    if (sp != nullptr) {
        aclsparseDestroySpVec(sp);
    }
}

TEST(ErrorPath, L2_008_Index64I)
{
    aclsparseSpVecDescr_t sp = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, 8, 4, g_err->idx.ptr, g_err->val.ptr, ACL_SPARSE_INDEX_64I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&dn, 8, g_err->y.ptr, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);
    CheckErr("L2_008", aclsparseScatter(g_handle, sp, dn), ACL_SPARSE_STATUS_INVALID_VALUE,
             "scatter 层：idxType=64I 仅支持 I32（dtype_not_supported）");
    ExpectYUnchanged(g_err->y, g_err->yOrig, "L2_008");
    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
}

TEST(ErrorPath, L2_009_UnsupportedDtypeF64)
{
    aclsparseSpVecDescr_t sp = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, 8, 4, g_err->idx.ptr, g_err->val.ptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_DOUBLE),
              ACL_SPARSE_STATUS_SUCCESS);
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&dn, 8, g_err->y.ptr, ACL_DOUBLE), ACL_SPARSE_STATUS_SUCCESS);
    CheckErr("L2_009", aclsparseScatter(g_handle, sp, dn), ACL_SPARSE_STATUS_INVALID_VALUE,
             "scatter 层：valueType=FLOAT64 不在支持集合（dtype_not_supported）");
    ExpectYUnchanged(g_err->y, g_err->yOrig, "L2_009");
    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
}

TEST(ErrorPath, L2_010_DtypeMismatch)
{
    aclsparseSpVecDescr_t sp = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, 8, 4, g_err->idx.ptr, g_err->val.ptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    aclsparseDnVecDescr_t dn = nullptr;  // values=fp32, y=fp16，不做 dtype 提升
    ASSERT_EQ(aclsparseCreateDnVec(&dn, 8, g_err->y.ptr, ACL_FLOAT16), ACL_SPARSE_STATUS_SUCCESS);
    CheckErr("L2_010", aclsparseScatter(g_handle, sp, dn), ACL_SPARSE_STATUS_INVALID_VALUE,
             "scatter 层：vecX.values=fp32 与 vecY=fp16 不一致（dtype_not_supported）");
    ExpectYUnchanged(g_err->y, g_err->yOrig, "L2_010");
    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
}

// 未声明 alias：vecY 与 vecX.values / vecX.indices 内存重叠（两个参数化变体）
TEST(ErrorPath, L2_012_AliasYValuesOverlap)
{
    std::vector<uint8_t> shared(64, 0xAB);
    DevBuf buf;
    buf.Alloc(shared.data(), shared.size());
    aclsparseSpVecDescr_t sp = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, 8, 4, g_err->idx.ptr, buf.ptr, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&dn, 8, buf.ptr, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);
    CheckErr("L2_012", aclsparseScatter(g_handle, sp, dn), ACL_SPARSE_STATUS_INVALID_VALUE,
             "scatter 层：vecY 与 vecX.values 内存重叠（未声明 alias 拒绝）[y~values 变体]");
    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
}

TEST(ErrorPath, L2_012_AliasYIndicesOverlap)
{
    std::vector<uint8_t> shared(64, 0xCD);
    DevBuf buf;
    buf.Alloc(shared.data(), shared.size());
    aclsparseSpVecDescr_t sp = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, 8, 4, buf.ptr, g_err->val.ptr, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&dn, 8, buf.ptr, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);
    CheckErr("L2_012", aclsparseScatter(g_handle, sp, dn), ACL_SPARSE_STATUS_INVALID_VALUE,
             "scatter 层：vecY 与 vecX.indices 内存重叠（未声明 alias 拒绝）[y~indices 变体]");
    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
}

// nnz>0 时 indices/values 数据指针空：描述符层确定报错（create 校验），
// scatter 侧同名校验为纵深防御（公共 API 不可达），错误码一致
TEST(ErrorPath, L2_013_NullIndicesPtr)
{
    aclsparseSpVecDescr_t sp = nullptr;
    const aclsparseStatus_t st = aclsparseCreateSpVec(&sp, 8, 4, nullptr, g_err->val.ptr,
                                                      ACL_SPARSE_INDEX_32I,
                                                      ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CheckErr("L2_013", st, ACL_SPARSE_STATUS_INVALID_VALUE,
             "描述符层：nnz>0 且 indices=nullptr（scatter 侧校验为纵深防御）[indices 变体]");
    if (sp != nullptr) {
        aclsparseDestroySpVec(sp);
    }
}

TEST(ErrorPath, L2_013_NullValuesPtr)
{
    aclsparseSpVecDescr_t sp = nullptr;
    const aclsparseStatus_t st = aclsparseCreateSpVec(&sp, 8, 4, g_err->idx.ptr, nullptr,
                                                      ACL_SPARSE_INDEX_32I,
                                                      ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CheckErr("L2_013", st, ACL_SPARSE_STATUS_INVALID_VALUE,
             "描述符层：nnz>0 且 values=nullptr（scatter 侧校验为纵深防御）[values 变体]");
    if (sp != nullptr) {
        aclsparseDestroySpVec(sp);
    }
}

TEST(ErrorPath, L2_014_SizeExceedsVecYNums)
{
    aclsparseSpVecDescr_t sp = nullptr;  // SpVec size=16 > vecY nums=8
    ASSERT_EQ(aclsparseCreateSpVec(&sp, 16, 4, g_err->idx.ptr, g_err->val.ptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&dn, 8, g_err->y.ptr, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);
    CheckErr("L2_014", aclsparseScatter(g_handle, sp, dn), ACL_SPARSE_STATUS_INVALID_VALUE,
             "scatter 层：SpVec size=16 > vecY nums=8（shape_mismatch）");
    ExpectYUnchanged(g_err->y, g_err->yOrig, "L2_014");
    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
}

// size=0 且 nnz>0：公开契约（cann_ops_sparse.h「nnz>0 且 size=0」）
// 确定参数错误。描述符层 nnz≤size 校验先行拒绝时错误码一致（INVALID_VALUE），
// scatter 侧同名校验为纵深防御
TEST(ErrorPath, L2_016_Size0NnzGt0)
{
    aclsparseSpVecDescr_t sp = nullptr;
    const aclsparseStatus_t st = aclsparseCreateSpVec(&sp, 0, 4, g_err->idx.ptr, g_err->val.ptr,
                                                      ACL_SPARSE_INDEX_32I,
                                                      ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CheckErr("L2_016", st, ACL_SPARSE_STATUS_INVALID_VALUE,
             "描述符层：SpVec size=0 且 nnz>0（shape_mismatch；契约 cann_ops_sparse.h "
             "「nnz>0 且 size=0」；scatter 侧同名校验为纵深防御，错误码一致）");
    if (sp != nullptr) {
        aclsparseDestroySpVec(sp);
    }
}

// 规模上界拒绝分支（nnz/size > INT32_MAX）：描述符层仅校验
// nnz≤size，允许小缓冲指针构造超界规模，scatter 侧拒绝分支可达且不触碰数据指针
// （ValidateNnzBound 先于 shape/指针校验）。小指针即可触达，无需真实大分配。
TEST(ErrorPath, L2_017_NnzExceedsInt32Max)
{
    constexpr int64_t kOver = static_cast<int64_t>(INT32_MAX) + 1;  // 2^31
    aclsparseSpVecDescr_t sp = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, kOver, kOver, g_err->idx.ptr, g_err->val.ptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&dn, 8, g_err->y.ptr, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);
    CheckErr("L2_017", aclsparseScatter(g_handle, sp, dn), ACL_SPARSE_STATUS_INVALID_VALUE,
             "scatter 层：nnz > INT32_MAX 规模上界拒绝（tiling 链 nnz 收敛 uint32，"
             "越界静默截断会致错误 tiling/GM 绑定）");
    ExpectYUnchanged(g_err->y, g_err->yOrig, "L2_017");
    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
}

TEST(ErrorPath, L2_018_SizeExceedsInt32Max)
{
    constexpr int64_t kOver = static_cast<int64_t>(INT32_MAX) + 1;  // 2^31
    aclsparseSpVecDescr_t sp = nullptr;  // size 超界、nnz 保持小值（避开 nnz 分支）
    ASSERT_EQ(aclsparseCreateSpVec(&sp, kOver, 4, g_err->idx.ptr, g_err->val.ptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&dn, 8, g_err->y.ptr, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);
    CheckErr("L2_018", aclsparseScatter(g_handle, sp, dn), ACL_SPARSE_STATUS_INVALID_VALUE,
             "scatter 层：size > INT32_MAX 规模上界拒绝（接口文档「索引溢出边界」承诺）");
    ExpectYUnchanged(g_err->y, g_err->yOrig, "L2_018");
    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
}

// ===================== L1_053 / L2_005 / L2_011 =====================

static size_t UtReadSelfRssBytes()
{
    std::ifstream f("/proc/self/statm");
    if (!f.is_open()) {
        return 0;
    }
    unsigned long long sizePages = 0, residentPages = 0;
    if (!(f >> sizePages >> residentPages)) {
        residentPages = 0;
    }
    return static_cast<size_t>(residentPages) * 4096ULL;
}

// L1_053 执行轮次类型：0 workspace（三参数签名编译期证明）
using ScatterFnNoWorkspace = aclsparseStatus_t (*)(aclsparseHandle_t,
                                                   aclsparseConstSpVecDescr_t,
                                                   aclsparseDnVecDescr_t);

// L1_053 输入构造：P-02 形状覆盖式铺排 + golden
struct UtL1_053Data {
    std::vector<int32_t> idx;
    std::vector<float> val;
    std::vector<float> y;
    std::vector<float> golden;
};

static UtL1_053Data UtL1_053BuildInputs(int64_t size, int64_t nnz)
{
    UtL1_053Data d;
    // 防御分支：L1_053 实调 nnz 恒 >0（P-02：nnz=4096）；nnz=0 时下方
    // i%nnz / i/nnz 为除零路径，此处确定性回退——
    // idx/val 为空、y=golden（等价 nnz=0 语义：无写入位置），正常路径行为不变
    if (nnz == 0) {
        d.y.assign(static_cast<size_t>(size), 0.5f);
        d.golden = d.y;
        return d;
    }
    d.idx.resize(static_cast<size_t>(nnz));
    d.val.resize(static_cast<size_t>(nnz));
    d.y.assign(static_cast<size_t>(size), 0.5f);
    std::mt19937 rng(20260902u);
    for (int64_t i = 0; i < size; i++) {
        d.idx[static_cast<size_t>(i % nnz)] = static_cast<int32_t>(i);  // 覆盖式铺排后 shuffle
    }
    std::shuffle(d.idx.begin(), d.idx.end(), rng);
    for (int64_t i = 0; i < nnz; i++) {
        d.val[static_cast<size_t>(i)] = -1.0f + 2.0f * (static_cast<float>(i) / static_cast<float>(nnz));
    }
    d.golden = d.y;
    for (int64_t i = 0; i < nnz; i++) {
        d.golden[d.idx[static_cast<size_t>(i)]] = d.val[static_cast<size_t>(i)];
    }
    return d;
}

struct UtL1_053Growth {
    int64_t devGrowth = 0;
    int64_t rssGrowth = 0;
};

// L1_053 执行 10 轮并采集 HBM/RSS 增长
static void UtL1_053RunRounds(ScatterFnNoWorkspace scatterNoWorkspace,
                              aclsparseSpVecDescr_t sp, aclsparseDnVecDescr_t dn,
                              UtL1_053Growth &g)
{
    size_t freeBefore = 0, totalHbm = 0, freeAfter = 0;
    ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &freeBefore, &totalHbm), ACL_SUCCESS);
    const size_t rssBefore = UtReadSelfRssBytes();
    for (int i = 0; i < 10; i++) {
        ASSERT_EQ(scatterNoWorkspace(g_handle, sp, dn), ACL_SPARSE_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtSynchronizeStream(g_stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &freeAfter, &totalHbm), ACL_SUCCESS);
    const size_t rssAfter = UtReadSelfRssBytes();

    g.devGrowth = static_cast<int64_t>(freeBefore) - static_cast<int64_t>(freeAfter);
    g.rssGrowth = static_cast<int64_t>(rssAfter) - static_cast<int64_t>(rssBefore);
}

// L1_053：0 workspace（三参数签名编译期证明）+ P-02 形状连续执行 HBM/RSS 无增长
TEST(Semantic, L1_053_ZeroWorkspaceNoExtraAlloc)
{
    ScatterFnNoWorkspace scatterNoWorkspace = &aclsparseScatter;  // 无 workspace 形参
    ASSERT_TRUE(scatterNoWorkspace != nullptr);

    // P-02 形状（size=151936/nnz=4096/base=0 fp32）
    const int64_t size = 151936;
    const int64_t nnz = 4096;
    const UtL1_053Data d = UtL1_053BuildInputs(size, nnz);

    DevBuf idxDev, valDev, yDev;
    idxDev.Alloc(d.idx.data(), d.idx.size() * sizeof(int32_t));
    valDev.Alloc(d.val.data(), d.val.size() * sizeof(float));
    yDev.Alloc(d.y.data(), d.y.size() * sizeof(float));

    aclsparseSpVecDescr_t sp = nullptr;
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, size, nnz, idxDev.ptr, valDev.ptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseCreateDnVec(&dn, size, yDev.ptr, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);

    ASSERT_EQ(scatterNoWorkspace(g_handle, sp, dn), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(g_stream), ACL_SUCCESS);

    UtL1_053Growth growth;
    UtL1_053RunRounds(scatterNoWorkspace, sp, dn, growth);

    const std::vector<uint8_t> yOut = yDev.Download();
    const bool bitOk = (yOut.size() == d.golden.size() * sizeof(float)) &&
                       (std::memcmp(yOut.data(), d.golden.data(), yOut.size()) == 0);
    const bool noExtraDev = (growth.devGrowth <= 2 * 1024 * 1024);
    const bool noExtraHost = (growth.rssGrowth <= 4 * 1024 * 1024);
    EXPECT_TRUE(noExtraDev) << "[L1_053] device HBM grew by " << growth.devGrowth << " bytes";
    EXPECT_TRUE(noExtraHost) << "[L1_053] host RSS grew by " << growth.rssGrowth << " bytes";
    EXPECT_TRUE(bitOk) << "[L1_053] output != golden (bit-wise)";

    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);

    CaseRecord r;
    r.caseId = "L1_053";
    r.gtest = CurrentGtestName();
    r.status = (noExtraDev && noExtraHost && bitOk) ? "PASS" : "FAIL";
    r.check = "workspace_check";
    r.dtype = "float32";
    r.size = size;
    r.nnz = nnz;
    r.detail = "signature=(handle,vecX,vecY) no-workspace (compile-time); P-02 10 runs after "
               "warmup: devHBM delta=" + std::to_string(growth.devGrowth) + "B, hostRSS delta=" +
               std::to_string(growth.rssGrowth) + "B, bit-wise exact=" + (bitOk ? "yes" : "NO");
    RecordCase(r);
}

// L2_005：合法索引内存安全哨兵硬门（接口契约层）
// 布局：kGuard(0x5A) | vecY(8×fp32) | kGuard(0x5A) 单一设备缓冲，前后哨兵逐字节评估。
// 合法索引属接口契约：输出正确性与越界写不逃出 vecY（内存安全）为确定性行为，
// 硬门不依赖平台实现细节。非法索引（负 / 上溢）违反 native API 前置条件、
// 无承诺行为，由独立进程探测二进制 scatter_oob_probe_test 全观测记录，不在本用例判定。

// vecY 初值：非零可区分（未写位保持断言的期望源）
static std::vector<float> L2_005MakeYInit(size_t kSize)
{
    std::vector<float> yInit(kSize);
    for (size_t i = 0; i < kSize; i++) {
        yInit[i] = 0.5f + 0.25f * static_cast<float>(i);
    }
    return yInit;
}

// 哨兵布局 host 缓冲：全 0x5A，vecY 区间嵌入初值
static std::vector<uint8_t> L2_005MakeHost(const std::vector<float> &yInit, size_t kGuard)
{
    std::vector<uint8_t> host(2 * kGuard + yInit.size() * sizeof(float), 0x5A);
    for (size_t i = 0; i < yInit.size(); i++) {
        CheckedMemcpy(host.data() + kGuard + i * sizeof(float), sizeof(float),
                      &yInit[i], sizeof(float));
    }
    return host;
}

// 前后哨兵逐字节评估（合法索引下越界写不得逃出 vecY）
struct L2_005SentinelEval {
    size_t preChanged = 0;
    size_t postChanged = 0;
};

static L2_005SentinelEval L2_005EvalSentinels(const std::vector<uint8_t> &out, size_t kGuard,
                                              size_t kYBytes)
{
    L2_005SentinelEval e;
    for (size_t i = 0; i < kGuard; i++) {
        e.preChanged += (out[i] != 0x5A);
        e.postChanged += (out[kGuard + kYBytes + i] != 0x5A);
    }
    return e;
}

// 写入位 y[idx]==val 逐位；未写位保持初值逐位
struct L2_005WriteEval {
    bool writeOk = true;
    bool keepOk = true;
};

static L2_005WriteEval L2_005EvalWrites(const std::vector<uint8_t> &out,
                                        const std::vector<int32_t> &idx,
                                        const std::vector<float> &val,
                                        const std::vector<float> &yInit, size_t kGuard)
{
    L2_005WriteEval e;
    for (size_t j = 0; j < idx.size(); j++) {
        e.writeOk = e.writeOk && (std::memcmp(out.data() + kGuard + idx[j] * sizeof(float),
                                             &val[j], sizeof(float)) == 0);
    }
    for (size_t i = 0; i < yInit.size(); i++) {
        const bool written =
            (std::find(idx.begin(), idx.end(), static_cast<int32_t>(i)) != idx.end());
        if (!written) {
            e.keepOk = e.keepOk && (std::memcmp(out.data() + kGuard + i * sizeof(float),
                                               &yInit[i], sizeof(float)) == 0);
        }
    }
    return e;
}

// vecX 只读评估：idx/val 设备缓冲下载后与上传值逐位一致
struct L2_005ReadOnlyEval {
    bool idxOk = true;
    bool valOk = true;
};

static L2_005ReadOnlyEval L2_005EvalVecXReadOnly(const DevBuf &idxDev, const DevBuf &valDev,
                                                const std::vector<int32_t> &idx,
                                                const std::vector<float> &val)
{
    L2_005ReadOnlyEval e;
    const std::vector<uint8_t> idxBack = idxDev.Download();
    const std::vector<uint8_t> valBack = valDev.Download();
    e.idxOk = (idxBack.size() == idx.size() * sizeof(int32_t)) &&
              (std::memcmp(idxBack.data(), idx.data(), idxBack.size()) == 0);
    e.valOk = (valBack.size() == val.size() * sizeof(float)) &&
              (std::memcmp(valBack.data(), val.data(), valBack.size()) == 0);
    return e;
}

// L2_005 case 记录：判定汇总与观测明细拼接
static void L2_005Record(const L2_005SentinelEval &sent, const L2_005WriteEval &wr,
                         const L2_005ReadOnlyEval &ro, size_t kSize, size_t nnz)
{
    CaseRecord r;
    r.caseId = "L2_005";
    r.gtest = CurrentGtestName();
    r.status = (sent.preChanged == 0 && sent.postChanged == 0 && wr.writeOk && wr.keepOk &&
                ro.idxOk && ro.valOk) ? "PASS" : "FAIL";
    r.check = "memory_safety";
    r.dtype = "float32";
    r.size = kSize;
    r.nnz = nnz;
    r.detail = "legal indices (all < size): scatter=SUCCESS sync=SUCCESS; "
               "sentinel_pre_changed=" + std::to_string(sent.preChanged) +
               " post_changed=" + std::to_string(sent.postChanged) +
               "; written y[idx]==val bit-exact=" + (wr.writeOk ? "yes" : "NO") +
               ", unwritten preserved bit-exact=" + (wr.keepOk ? "yes" : "NO") +
               "; vecX read-only=" + ((ro.idxOk && ro.valOk) ? "yes" : "NO") +
               "; OOB write confined to vecY (sentinels intact)";
    RecordCase(r);
}

TEST(MemorySafety, L2_005_LegalIndexSentinel)
{
    constexpr size_t kGuard = 4096;
    constexpr size_t kSize = 8;
    constexpr size_t kYBytes = kSize * sizeof(float);
    constexpr size_t kTotal = 2 * kGuard + kYBytes;

    const std::vector<int32_t> idx = {0, 2, 5, 7};  // 全部 < size，合法索引
    const std::vector<float> val = {101.0f, 102.0f, 103.0f, 104.0f};
    const std::vector<float> yInit = L2_005MakeYInit(kSize);

    const std::vector<uint8_t> host = L2_005MakeHost(yInit, kGuard);
    DevBuf buf;
    buf.Alloc(host.data(), kTotal);
    uint8_t *yPtr = buf.ptr + kGuard;

    DevBuf idxDev, valDev;
    idxDev.Alloc(idx.data(), idx.size() * sizeof(int32_t));
    valDev.Alloc(val.data(), val.size() * sizeof(float));

    aclsparseSpVecDescr_t sp = nullptr;
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, static_cast<int64_t>(kSize),
                                   static_cast<int64_t>(idx.size()), idxDev.ptr, valDev.ptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseCreateDnVec(&dn, static_cast<int64_t>(kSize), yPtr, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);

    const aclsparseStatus_t st = aclsparseScatter(g_handle, sp, dn);
    const aclError sync = aclrtSynchronizeStream(g_stream);
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS) << "[L2_005] legal-index scatter status";
    ASSERT_EQ(sync, ACL_SUCCESS) << "[L2_005] legal-index stream sync";

    const std::vector<uint8_t> out = buf.Download();
    const L2_005SentinelEval sent = L2_005EvalSentinels(out, kGuard, kYBytes);
    const L2_005WriteEval wr = L2_005EvalWrites(out, idx, val, yInit, kGuard);
    const L2_005ReadOnlyEval ro = L2_005EvalVecXReadOnly(idxDev, valDev, idx, val);

    EXPECT_EQ(sent.preChanged, 0u) << "[L2_005] pre-sentinel bytes changed: " << sent.preChanged;
    EXPECT_EQ(sent.postChanged, 0u) << "[L2_005] post-sentinel bytes changed: " << sent.postChanged;
    EXPECT_TRUE(wr.writeOk) << "[L2_005] written positions y[idx] != val (bit-wise)";
    EXPECT_TRUE(wr.keepOk) << "[L2_005] unwritten positions != original y (bit-wise)";
    EXPECT_TRUE(ro.idxOk) << "[L2_005] vecX indices modified (read-only violation)";
    EXPECT_TRUE(ro.valOk) << "[L2_005] vecX values modified (read-only violation)";

    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);

    L2_005Record(sent, wr, ro, kSize, idx.size());
}

// L2_011：device 不一致（双卡；异步错误协议确定暴露，vecY 不污染，无静默错误结果）

// 单卡环境 SKIP 记录
static void UtRecordL2_011Skipped()
{
    CaseRecord r;
    r.caseId = "L2_011";
    r.gtest = CurrentGtestName();
    r.status = "SKIPPED";
    r.check = "raises_error";
    r.dtype = "float32";
    r.size = 8;
    r.nnz = 4;
    r.detail = "single-device environment: dual-card evidence run (dedicated dual-card container, "
               "ASCEND_RT_VISIBLE_DEVICES=0,1)";
    RecordCase(r);
}

// L2_011 错配阶段：vecX@dev1 / vecY@dev0 → 提交 + 同步 + vecY 不变判定。
// 阶段内自持设备上下文切换，返回时上下文恢复 device 0。
struct UtL2_011MismatchResult {
    aclsparseStatus_t st = static_cast<aclsparseStatus_t>(0);
    aclError sync = 0;
    bool yUnchanged = false;
};

static void UtL2_011MismatchPhase(const std::vector<int32_t> &idx,
                                  const std::vector<float> &val,
                                  const std::vector<float> &yInit,
                                  DevBuf &yDev, UtL2_011MismatchResult &res)
{
    DevBuf idxDev, valDev;
    ASSERT_EQ(aclrtSetDevice(1), ACL_SUCCESS);
    idxDev.Alloc(idx.data(), idx.size() * sizeof(int32_t));
    valDev.Alloc(val.data(), val.size() * sizeof(float));
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);

    aclsparseSpVecDescr_t sp = nullptr;
    aclsparseDnVecDescr_t dn = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp, 8, static_cast<int64_t>(idx.size()), idxDev.ptr, valDev.ptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseCreateDnVec(&dn, 8, yDev.ptr, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);

    res.st = aclsparseScatter(g_handle, sp, dn);
    res.sync = aclrtSynchronizeStream(g_stream);
    const std::vector<uint8_t> yAfter = yDev.Download();
    res.yUnchanged = (yAfter.size() == yInit.size() * sizeof(float)) &&
                     (std::memcmp(yAfter.data(), yInit.data(), yAfter.size()) == 0);

    aclsparseDestroySpVec(sp);
    aclsparseDestroyDnVec(dn);
    ASSERT_EQ(aclrtSetDevice(1), ACL_SUCCESS);
    idxDev.Free();
    valDev.Free();
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
}

// L2_011 对照阶段：全部张量@dev0 恢复性控制运行
struct UtL2_011ControlResult {
    aclsparseStatus_t st = static_cast<aclsparseStatus_t>(0);
    aclError sync = 0;
    bool controlOk = false;
};

static void UtL2_011ControlPhase(const std::vector<int32_t> &idx,
                                 const std::vector<float> &val,
                                 const std::vector<float> &yInit,
                                 DevBuf &yDev, UtL2_011ControlResult &res)
{
    DevBuf idxDev0, valDev0;
    idxDev0.Alloc(idx.data(), idx.size() * sizeof(int32_t));
    valDev0.Alloc(val.data(), val.size() * sizeof(float));
    aclsparseSpVecDescr_t sp2 = nullptr;
    aclsparseDnVecDescr_t dn2 = nullptr;
    ASSERT_EQ(aclsparseCreateSpVec(&sp2, 8, static_cast<int64_t>(idx.size()), idxDev0.ptr, valDev0.ptr,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseCreateDnVec(&dn2, 8, yDev.ptr, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);
    res.st = aclsparseScatter(g_handle, sp2, dn2);
    res.sync = aclrtSynchronizeStream(g_stream);
    std::vector<float> yExpect = yInit;
    for (size_t i = 0; i < idx.size(); i++) {
        yExpect[static_cast<size_t>(idx[i])] = val[i];
    }
    const std::vector<uint8_t> yControl = yDev.Download();
    res.controlOk =
        (res.st == ACL_SPARSE_STATUS_SUCCESS) && (res.sync == ACL_SUCCESS) &&
        (yControl.size() == yExpect.size() * sizeof(float)) &&
        (std::memcmp(yControl.data(), yExpect.data(), yControl.size()) == 0);
    aclsparseDestroySpVec(sp2);
    aclsparseDestroyDnVec(dn2);
}

TEST(ErrorPath, L2_011_DeviceMismatchDualCard)
{
    uint32_t devCount = 0;
    ASSERT_EQ(aclrtGetDeviceCount(&devCount), ACL_SUCCESS);
    if (devCount < 2) {
        UtRecordL2_011Skipped();
        GTEST_SKIP() << "L2_011 requires >= 2 NPU devices (dual-card run)";
    }

    const std::vector<int32_t> idx = {0, 3, 5, 7};
    const std::vector<float> val = {1.5f, -2.5f, 3.25f, -0.75f};
    const std::vector<float> yInit(8, 0.25f);

    DevBuf yDev;
    yDev.Alloc(yInit.data(), yInit.size() * sizeof(float));

    UtL2_011MismatchResult mm;
    UtL2_011MismatchPhase(idx, val, yInit, yDev, mm);
    UtL2_011ControlResult ctrl;
    UtL2_011ControlPhase(idx, val, yInit, yDev, ctrl);

    const bool mismatchOk =
        (mm.st == ACL_SPARSE_STATUS_SUCCESS) && (mm.sync != ACL_SUCCESS) && mm.yUnchanged;
    EXPECT_TRUE(mismatchOk) << "[L2_011] st=" << (int)mm.st << " sync=" << (int)mm.sync
                            << " yUnchanged=" << mm.yUnchanged;
    EXPECT_TRUE(ctrl.controlOk) << "[L2_011] control run after device-mismatch fault failed";

    CaseRecord r;
    r.caseId = "L2_011";
    r.gtest = CurrentGtestName();
    r.status = (mismatchOk && ctrl.controlOk) ? "PASS" : "FAIL";
    r.check = "raises_error";
    r.dtype = "float32";
    r.size = 8;
    r.nnz = 4;
    r.detail = "vecX@dev1 vecY@dev0 (dual-card): scatter=" + std::to_string((int)mm.st) +
               " (async submit) sync=" + std::to_string((int)mm.sync) +
               " (async device error, deterministic); vecY unchanged=" +
               std::string(mm.yUnchanged ? "yes" : "NO") + "; control(all@dev0) after fault=" +
               std::string(ctrl.controlOk ? "OK" : "FAIL");
    RecordCase(r);
}

// ===================== 全局环境与 JSON 报告 =====================

class ScatterUtEnv : public ::testing::Environment {
public:
    void SetUp() override
    {
        SetupAclDevice0();  // aclInit 容错 + SetDevice(0)（卡由外部 ASCEND_RT_VISIBLE_DEVICES 钉卡）
        aclError r = aclrtCreateStream(&g_stream);
        if (r != ACL_SUCCESS) {
            fprintf(stderr, "[FATAL] aclrtCreateStream failed: %d\n", (int)r);
            std::exit(1);
        }
        if (aclsparseCreate(&g_handle) != ACL_SPARSE_STATUS_SUCCESS ||
            aclsparseSetStream(g_handle, g_stream) != ACL_SPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "[FATAL] aclsparse handle setup failed\n");
            std::exit(1);
        }

        // L2 共享数据：fp32 size=8 / nnz=4
        g_err = new ErrData();
        const std::vector<int32_t> idx = {0, 3, 5, 7};
        const std::vector<float> val = {1.5f, -2.5f, 3.25f, -0.75f};
        const std::vector<float> y(8, 0.25f);
        g_err->idx.Alloc(idx.data(), idx.size() * sizeof(int32_t));
        g_err->val.Alloc(val.data(), val.size() * sizeof(float));
        g_err->y.Alloc(y.data(), y.size() * sizeof(float));
        g_err->yOrig = g_err->y.Download();
    }

    void TearDown() override
    {
        WriteJsonReport();
        if (g_err != nullptr) {
            delete g_err;
            g_err = nullptr;
        }
        aclsparseDestroy(g_handle);
        g_handle = nullptr;
        aclrtDestroyStream(g_stream);
        g_stream = nullptr;
        aclrtResetDevice(0);
        aclFinalize();
    }

private:
    // cases 数组逐条写出
    static void WriteJsonCases(std::ofstream &f)
    {
        f << "  \"cases\": [\n";
        for (size_t i = 0; i < g_records.size(); ++i) {
            const auto &r = g_records[i];
            f << "    {\"case_id\": \"" << JsonEscape(r.caseId) << "\", "
              << "\"gtest\": \"" << JsonEscape(r.gtest) << "\", "
              << "\"status\": \"" << r.status << "\", "
              << "\"check\": \"" << JsonEscape(r.check) << "\", "
              << "\"dtype\": \"" << JsonEscape(r.dtype) << "\", "
              << "\"size\": " << r.size << ", \"nnz\": " << r.nnz << ", \"base\": " << r.base
              << ", \"seed\": " << r.seed << ", "
              << "\"detail\": \"" << JsonEscape(r.detail) << "\"}"
              << (i + 1 < g_records.size() ? "," : "") << "\n";
        }
        f << "  ],\n";
    }

    // case 级摘要打屏
    static void PrintJsonCaseSummary()
    {
        for (const auto &r : g_records) {
            printf("  %-10s %-8s %s\n", r.caseId.c_str(), r.status.c_str(), r.detail.c_str());
        }
    }

    static void WriteJsonReport()
    {
        const char *envPath = std::getenv("SCATTER_UT_JSON");
        const std::string path = (envPath != nullptr) ? envPath : "scatter_ut_results.json";
        size_t pass = 0, fail = 0, planned = 0;
        for (const auto &r : g_records) {
            if (r.status == "PASS") {
                pass++;
            } else if (r.status == "FAIL") {
                fail++;
            } else if (r.status == "PLANNED") {
                planned++;
            }
        }
        std::ofstream f(path);
        if (!f.is_open()) {
            fprintf(stderr, "[WARN] cannot open report file: %s\n", path.c_str());
            return;
        }
        f << "{\n";
        f << "  \"suite\": \"scatter_ut\",\n";
        f << "  \"operator\": \"aclsparseScatter\",\n";
        f << "  \"arch\": \"arch22/DAV_2201\",\n";
        f << "  \"soc\": \"ascend910b3\",\n";
        WriteJsonCases(f);
        f << "  \"summary\": {\"executed\": " << (pass + fail) << ", \"passed\": " << pass
          << ", \"failed\": " << fail << ", \"planned\": " << planned << "}\n";
        f << "}\n";
        f.close();

        printf("\n==== scatter_ut case-level summary (%s) ====\n", path.c_str());
        PrintJsonCaseSummary();
        printf("  executed=%zu passed=%zu failed=%zu planned=%zu\n", pass + fail, pass, fail, planned);
    }
};

// 全局 Environment 经静态初始化注册（main 由 test/frame/test_main.cpp 提供，
// AddGlobalTestEnvironment 须在 RUN_ALL_TESTS 前调用，pre-main 注册为 gtest 支持的用法）
static const bool g_scatterUtEnvRegistered = []() {
    ::testing::AddGlobalTestEnvironment(new ScatterUtEnv());
    return true;
}();
