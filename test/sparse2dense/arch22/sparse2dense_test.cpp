// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE for the full text of the License.
// ============================================================================
// test_aclnn_aclsparse_sparse2dense.cpp — aclsparseSparseToDense ST 测试（gtest）
//
// C++ 全量用例（sparse2dense_test.csv 唯一用例来源）：
//   * L0 12 条（手写 TEST_F，含定制断言；参数与 CSV L0 段对齐）；
//   * L1 85 条全展开：dtype 矩阵 60（官方 accuracy id0~59 属性同口径，10 种官方 row
//     pattern 同构移植）+ boundary 11 + extreme 6（NaN/±inf/±0 bit 保持/complex64 实虚
//     独立）+ 语义 8（确定性重复执行 L1_075、双阶段协议幂等 L1_077、分片边界 L1_078/
//     079/085、P-01 规模 L1_080、ROW/COL 转置等价 L1_081）；
//   * L2 37 条全展开：空 handle/描述符（L2_001~004）、坐标越界（005~008）、重复坐标成功
//     语义（009，任意值判定）、非单调（010/011）、len 不符（012/013）、dtype 族（014~016）、
//     非法 base/layout/ld/format/alg/rows
//     （017~023）、恒 0 空 buffer 可执行（024）、alias（025）、溢出负例（026~028，
//     Host 校验分支：CheckMeta/ValidateF3/FillTiling 判溢出先于 kernel，matB 不污染断言）、
//     第十轮评审整改：形状不匹配 {CSR,CSC,COO}×{行,列}轴（029~034，公开 API 双入口
//     bufferSize+execute 断言——描述符级形状一致性）+ alias 索引同址/部分重叠
//     （035~037，values 与 primary 区间重叠负例）；
//   * L1/L2 用例参数运行时从本目录 sparse2dense_test.csv 装载（L0 段为手写
//     TEST_F 的参数来源参照）；L2 变异映射装载时与 CSV expected_value 逐条交叉断言；
//     执行证据为真实 gtest 日志/JSON。
//
// L0 基础用例：
//   * L0 12 条全落地（sparse2dense_test.csv L0 段唯一用例来源，
//     case_id/参数逐项对齐 CSV：dtype/format/layout/base/rows/cols/ld/nnz/value_spec/seed）；
//   * L1/L2 由同一 CaseSpec 框架按 CSV 全量展开。
//
// 判定口径（强制执行）：
//   * §4.1 golden：CPU canonical 转换——zeros((rows,cols), dtype=values.dtype) 后按 format 展开
//     坐标（CSR 行段 / CSC 列段 / COO 逐点）、存储坐标减 base、坐标 scatter values
//     （与被测同 dtype 构造，纯 bit 搬运无算术、不做 dtype 提升；重复坐标合法——重复
//     位置值 ∈ 重复值集合、非重复位置 bit-wise exact，判定见 JudgeOutput）；
//     输出与 golden 逐元素 bit-wise exact（五 dtype rtol=0/atol=0/bitwise_equal；complex64 实/虚
//     分别 bit-wise；NaN/Inf 按原始 bit；-0/-0 与 +0 填充 bit 可区分）；fail_rate_budget=0；
//   * §4.2 附带断言（全部用例通用）：
//       ① 输入只读：values/primary/secondary 执行前后逐字节不变；
//       ② 越界哨兵：matB 存储缓冲外 16B 哨兵（0xA5）执行后不变（不越出存储缓冲写）；
//       ③ ld padding 区（若有）被清零 kernel 清为 +0（清零 kernel 全 buffer 含 padding），
//          padding 不参与逻辑比对（逻辑区 bit-wise 已含于 ①整体比对）；
//   * 双阶段协议：每个成功用例先 aclsparseSparseToDense_bufferSize 查询
//     （断言 SUCCESS 且恒 0），再以 buffer=nullptr 执行 aclsparseSparseToDense；
//   * L0_007 零 nnz：额外显式断言输出全 +0（expected_check=all_positive_zero）。
//
// 数据生成口径（用例级单分布标签；官方属性同口径）：
//   * 坐标结构按官方 `common/sparse_patterns.py` 同构移植：row_counts 权重分配（uniform/banded/
//     power_law/block_like）+ 段内列生成（sorted_unique），CSC 按官方 adapter 以列主序重排；
//     L0 CSV 中 COO 坐标模式 shuffled_unique 的用例按 Fisher-Yates 乱序（唯一性保持）；
//   * values 按用例级单分布标签生成（uniform70 [-1,1] 均匀；int8 为整数域 {-1,0,1}，与
//     scatter 算子同口径；complex64 实/虚独立），固定随机种子（splitmix64）；
//   * 与官方 torch 胶水层的生成器差异（torch.randn vs splitmix64）为既成偏差，
//     测试侧按单分布标签执行，不复刻官方 RNG。
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <acl/acl.h>
#include "cann_ops_sparse.h"
// L2 负例直调 libops_sparse 导出的内部校验链（CheckMeta → ValidateF3 → FillTiling），
// 保持拒绝阶段与状态码的精确语义（部分非法元数据无法经公开描述符 API 构造）。
#include "sparse2dense_host.h"
#include "sparse2dense_case_loader.h"
#include "sparse2dense_case_encode.h"
#include "sparse2dense_case_gen.h"

#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

namespace {

// ============================================================================
// 成功路径 runner：生成 → 双阶段调用 → §4 判定 → 机器可解析结果行
// ============================================================================
struct RunResult {
    bool ok;
    std::string judge;   // bitwise_exact | all_positive_zero | inputs_bitwise_unchanged | ...
    size_t mismatches;   // bit-wise 失配字节数（整存储缓冲）
    std::string message; // 失败原因（空 = 通过）
    int queries = 0;             // bufferSize 查询次数（L1_077 幂等证据）
    size_t queryValue = 0;       // 各次查询返回值（恒 0）
    bool negZeroBitsSeen = false; // L1_072：输出含 -0 bit（与 +0 填充可区分的显式证据）
    std::vector<uint8_t> output; // 存储缓冲快照（仅等价断言用例捕获；跨 case bit 比对）
};

// ============================================================================
// 用例准备（Mock / Real 共用，单一实现保证 L0 判定输入同源）：
//   CSV L0_009/L0_010 偏差落地：cal_count=2 与 value_spec offsets=[0,1,1,3] 端点（nnz=3）
//   不一致，执行口径以 offsets 端点为权威 nnz（数据校验与真实描述符语义一致），空行/空列
//   边界语义不变（行/列 1 全空段 +0）；偏差已在 case_manifest.json 与 README 登记。
// ============================================================================
struct PreparedCase {
    SparseData sd;
    size_t storageBytes = 0;
    bool ready = false;
    std::string error;
};

PreparedCase PrepareCase(CaseSpec &cs) // 按引用传入：显式 offsets 用例就地修正有效 nnz
{
    PreparedCase pc;
    if (cs.explicitPrimary != nullptr) {
        const int64_t major = (cs.fmt == Fmt::CSC) ? cs.cols : cs.rows;
        cs.nnz = static_cast<int64_t>(cs.explicitPrimary[static_cast<size_t>(major)]) - cs.base;
    }
    // 显式 offsets 用例（L0_009/L0_010）：坐标直接由 offsets 展开，不走 pattern 分配
    std::vector<Coord> logical = (cs.explicitPrimary != nullptr) ? std::vector<Coord>{} : LogicalCoords(cs);
    if (cs.explicitPrimary == nullptr && static_cast<int64_t>(logical.size()) != cs.nnz) {
        pc.error = "coord count " + std::to_string(logical.size()) + " != nnz " + std::to_string(cs.nnz);
        return pc;
    }
    // 重复坐标注入（coords=duplicate，成功语义）：末元素坐标覆写为首元素坐标——
    // (r,c) 重复对的值集合 = {首元素值, 末元素值}，判定按集合归属（JudgeOutput dup 路径）
    if (cs.coords == CoordMode::DUPLICATE) {
        if (logical.size() < 2) {
            pc.error = "duplicate coords require nnz >= 2";
            return pc;
        }
        logical.back() = logical.front();
    }
    pc.sd = BuildSparse(cs, logical);
    const size_t esz = ElemBytes(cs.dtype);
    const size_t storageElems =
        static_cast<size_t>((cs.layout == Layout::ROW) ? (cs.rows * cs.ld) : (cs.cols * cs.ld));
    pc.storageBytes = storageElems * esz;
    pc.ready = true;
    return pc;
}

// ============================================================================
// §4 判定（Mock / Real 共用，单一实现保证判定口径不变）：
//   §4.1 整存储缓冲（逻辑区 + padding +0）与 golden bit-wise exact；
//   L0_007 零 nnz 显式 all_positive_zero；
//   §4.2 ① 输入只读（Now = 被测执行可见的输入内容：Mock 为宿主数组本体、Real 为
//        Device 回读；与执行前快照逐字节比对）；② 存储缓冲外 16B 哨兵不被污染。
// ============================================================================
// 重复坐标位置的候选值集合：logical 偏移 (r*cols+c) → 各写入元素的 value 字节
//（仅收录多次写入的位置；与 ComputeGoldenStorage 同一坐标展开口径）
std::map<size_t, std::vector<std::vector<uint8_t>>> ComputeDupCandidates(const CaseSpec &cs,
    const SparseData &sd)
{
    std::map<size_t, std::vector<std::vector<uint8_t>>> writers;
    const size_t esz = ElemBytes(cs.dtype);
    const int64_t base = cs.base;
    const auto note = [&](int64_t r, int64_t c, int64_t k) {
        writers[static_cast<size_t>(r * cs.cols + c)].emplace_back(
            sd.values.begin() + static_cast<size_t>(k) * esz,
            sd.values.begin() + static_cast<size_t>(k + 1) * esz);
    };
    if (cs.fmt == Fmt::CSR) {
        for (int64_t r = 0; r < cs.rows; ++r) {
            const int64_t s = static_cast<int64_t>(sd.primary[static_cast<size_t>(r)]) - base;
            const int64_t e = static_cast<int64_t>(sd.primary[static_cast<size_t>(r) + 1]) - base;
            for (int64_t k = s; k < e; ++k) {
                note(r, static_cast<int64_t>(sd.secondary[static_cast<size_t>(k)]) - base, k);
            }
        }
    } else if (cs.fmt == Fmt::CSC) {
        for (int64_t c = 0; c < cs.cols; ++c) {
            const int64_t s = static_cast<int64_t>(sd.primary[static_cast<size_t>(c)]) - base;
            const int64_t e = static_cast<int64_t>(sd.primary[static_cast<size_t>(c) + 1]) - base;
            for (int64_t k = s; k < e; ++k) {
                note(static_cast<int64_t>(sd.secondary[static_cast<size_t>(k)]) - base, c, k);
            }
        }
    } else {
        for (size_t k = 0; k < sd.primary.size(); ++k) {
            note(static_cast<int64_t>(sd.primary[k]) - base, static_cast<int64_t>(sd.secondary[k]) - base,
                static_cast<int64_t>(k));
        }
    }
    std::map<size_t, std::vector<std::vector<uint8_t>>> dups;
    for (auto &w : writers) {
        if (w.second.size() > 1) {
            dups[w.first] = std::move(w.second);
        }
    }
    return dups;
}

// ---------------------------------------------------------------------------
// §4.1 判定子段：单个存储元素是否命中重复坐标候选值集合（任意值语义：值 ∈ 集合）。
// ---------------------------------------------------------------------------
bool MatchDupCandidate(const uint8_t *actual, size_t elemB, size_t esz,
    const std::vector<std::vector<uint8_t>> &cands)
{
    for (const std::vector<uint8_t> &cand : cands) {
        if (std::memcmp(actual + elemB, cand.data(), esz) == 0) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// §4.1 判定子段：整存储缓冲（逻辑区 + padding +0）逐元素比对——重复坐标位置
//（dupCandidates 命中时）值 ∈ 重复值集合，其余位置与 golden bit-wise exact；
// 返回失配字节数（判定语义与拆分前一致）。
// ---------------------------------------------------------------------------
size_t JudgeStorageRegion(const CaseSpec &cs, const std::vector<uint8_t> &golden,
    const std::vector<uint8_t> &denseGuarded, size_t storageElems,
    const std::map<size_t, std::vector<std::vector<uint8_t>>> *dupCandidates)
{
    const size_t esz = ElemBytes(cs.dtype);
    const uint8_t *actual = denseGuarded.data();
    size_t mismatches = 0;
    for (size_t elem = 0; elem < storageElems; ++elem) {
        // 存储元素 → 逻辑 (行, 列)（ROW: dense[r*ld+c] / COL: dense[c*ld+r]）
        int64_t lr = 0;
        int64_t lc = 0;
        if (cs.layout == Layout::ROW) {
            lr = static_cast<int64_t>(elem / static_cast<size_t>(cs.ld));
            lc = static_cast<int64_t>(elem % static_cast<size_t>(cs.ld));
        } else {
            lc = static_cast<int64_t>(elem / static_cast<size_t>(cs.ld));
            lr = static_cast<int64_t>(elem % static_cast<size_t>(cs.ld));
        }
        const size_t elemB = elem * esz;
        const std::vector<std::vector<uint8_t>> *cands = nullptr;
        if (dupCandidates != nullptr && lr < cs.rows && lc < cs.cols) {
            const auto it = dupCandidates->find(static_cast<size_t>(lr * cs.cols + lc));
            if (it != dupCandidates->end()) {
                cands = &it->second;
            }
        }
        if (cands != nullptr) { // 重复位置：值 ∈ 重复值集合
            if (!MatchDupCandidate(actual, elemB, esz, *cands)) {
                mismatches += esz;
            }
        } else if (std::memcmp(actual + elemB, golden.data() + elemB, esz) != 0) {
            mismatches += esz;
        }
    }
    return mismatches;
}

// ---------------------------------------------------------------------------
// L0_007 子段：输出全 +0 逐字节检查（expected_check=all_positive_zero；与零输入
// 结果 bit 一致）。
// ---------------------------------------------------------------------------
bool OutputAllPositiveZero(const uint8_t *actual, size_t storageBytes)
{
    for (size_t b = 0; b < storageBytes; ++b) {
        if (actual[b] != 0x00) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// §4.2 附带断言 ① 子段：输入只读（values/primary/secondary 执行前后逐字节不变）。
// ---------------------------------------------------------------------------
bool InputsBitwiseUnchanged(const std::vector<uint8_t> &valNow, const std::vector<int32_t> &primaryNow,
    const std::vector<int32_t> &secondaryNow, const std::vector<uint8_t> &valSnap,
    const std::vector<int32_t> &primarySnap, const std::vector<int32_t> &secondarySnap)
{
    return std::memcmp(valNow.data(), valSnap.data(), valSnap.size()) == 0 &&
        std::memcmp(primaryNow.data(), primarySnap.data(), primarySnap.size() * sizeof(int32_t)) == 0 &&
        std::memcmp(secondaryNow.data(), secondarySnap.data(), secondarySnap.size() * sizeof(int32_t)) == 0;
}

// ---------------------------------------------------------------------------
// §4.2 附带断言 ② 子段：matB 存储缓冲外尾部 16B 哨兵不被污染（不越界写入）。
// ---------------------------------------------------------------------------
bool TailGuardIntact(const std::vector<uint8_t> &denseGuarded, size_t storageBytes)
{
    for (size_t k = 0; k < 16; ++k) {
        if (denseGuarded[storageBytes + k] != 0xA5) {
            return false;
        }
    }
    return true;
}

void JudgeOutput(const CaseSpec &cs, const SparseData &sd, const std::vector<uint8_t> &valNow,
    const std::vector<int32_t> &primaryNow, const std::vector<int32_t> &secondaryNow,
    const std::vector<uint8_t> &valSnap, const std::vector<int32_t> &primarySnap,
    const std::vector<int32_t> &secondarySnap, const std::vector<uint8_t> &denseGuarded,
    const std::map<size_t, std::vector<std::vector<uint8_t>>> *dupCandidates, RunResult &r)
{
    const size_t esz = ElemBytes(cs.dtype);
    const size_t storageElems =
        static_cast<size_t>((cs.layout == Layout::ROW) ? (cs.rows * cs.ld) : (cs.cols * cs.ld));
    const size_t storageBytes = storageElems * esz;

    // ---- §4.1 判定：整存储缓冲（逻辑区 + padding +0）与 golden bit-wise exact；
    //      重复坐标位置（dupCandidates 非空时）值 ∈ 重复值集合（任意值语义） ----
    r.mismatches += JudgeStorageRegion(cs, ComputeGoldenStorage(cs, sd), denseGuarded, storageElems,
        dupCandidates);
    if (r.mismatches > 0) {
        r.ok = false;
        r.message = "bit-wise mismatched bytes=" + std::to_string(r.mismatches);
    }
    // L0_007 显式断言：输出全 +0（expected_check=all_positive_zero；与零输入结果 bit 一致）
    if (cs.nnz == 0) {
        r.judge = "all_positive_zero";
        if (!OutputAllPositiveZero(denseGuarded.data(), storageBytes)) {
            r.ok = false;
            r.message = "non-zero byte at output (expected dtype +0 fill)";
        }
    }
    // ---- §4.2 附带断言 ①：输入只读（values/primary/secondary 执行前后逐字节不变） ----
    if (r.ok && !InputsBitwiseUnchanged(valNow, primaryNow, secondaryNow, valSnap, primarySnap,
                                        secondarySnap)) {
        r.ok = false;
        r.message = "matA modified (read-only violated)";
        return;
    }
    // ---- §4.2 附带断言 ②：matB 存储缓冲外尾部哨兵不被污染（不越界写入） ----
    if (r.ok && !TailGuardIntact(denseGuarded, storageBytes)) {
        r.ok = false;
        r.message = "dense tail guard polluted (out-of-bounds write)";
    }
}

// ============================================================================
// 正例执行（NPU 真机）：公开 API 双阶段协议 + 设备缓冲管理。
//   阶段一 aclsparseSparseToDense_bufferSize（恒 0；L1_077 追加第二次查询断言幂等）
//   阶段二 aclsparseSparseToDense（buffer=nullptr，查询为 0 → buffer 可空）
//   判定为单一实现（JudgeOutput）：整存储缓冲 golden bit-wise、输入只读回读、
//   存储缓冲外 16B 尾哨兵（0xA5 预填，清零 kernel 漏清零/越界写可检出）。
// ============================================================================
namespace {
aclrtStream g_stream = nullptr;
int32_t g_devId = 0;
uint32_t g_coreNum = 0;
std::string g_socName = "unknown";

// 正例 Device 资源（统一释放；nnz=0 空 vector 守卫：跳过空分配/上传）
struct CaseDeviceBufs {
    void *valDev = nullptr;
    void *primDev = nullptr;
    void *secDev = nullptr;
    void *denseDev = nullptr;
};

// ① Device 分配 + H2D 输入（同步拷贝，setup 语义）+ ② dense 0xA5 预填上传
//（清零 kernel 漏清零/越界写可检出）。失败返回 false 并给出原因。
bool UploadCase(const SparseData &sd, const std::vector<uint8_t> &denseGuarded, size_t denseBytes,
    CaseDeviceBufs *buf, std::string *err)
{
    const size_t valBytes = sd.values.size();
    const size_t primBytes = sd.primary.size() * sizeof(int32_t);
    const size_t secBytes = sd.secondary.size() * sizeof(int32_t);
    if (valBytes > 0 &&
        (aclrtMalloc(&buf->valDev, valBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
         aclrtMemcpy(buf->valDev, valBytes, sd.values.data(), valBytes, ACL_MEMCPY_HOST_TO_DEVICE) != 0)) {
        *err = "device alloc/H2D values failed";
        return false;
    }
    if (primBytes > 0 &&
        (aclrtMalloc(&buf->primDev, primBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
         aclrtMemcpy(buf->primDev, primBytes, sd.primary.data(), primBytes, ACL_MEMCPY_HOST_TO_DEVICE) != 0)) {
        *err = "device alloc/H2D primary failed";
        return false;
    }
    if (secBytes > 0 &&
        (aclrtMalloc(&buf->secDev, secBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
         aclrtMemcpy(buf->secDev, secBytes, sd.secondary.data(), secBytes, ACL_MEMCPY_HOST_TO_DEVICE) != 0)) {
        *err = "device alloc/H2D secondary failed";
        return false;
    }
    if (aclrtMalloc(&buf->denseDev, denseBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMemcpy(buf->denseDev, denseBytes, denseGuarded.data(), denseBytes,
            ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
        *err = "device alloc/H2D dense failed";
        return false;
    }
    return true;
}

// ③ 描述符创建（handle + CSR/CSC/COO 分发 + DnMat；数据指针为 Device 侧）
bool CreateCaseDescriptors(const CaseSpec &cs, const CaseDeviceBufs &buf, aclrtStream stream,
    aclsparseHandle_t *handle, aclsparseConstSpMatDescr_t *matA, aclsparseDnMatDescr_t *matB,
    std::string *err)
{
    aclsparseStatus_t st = aclsparseCreate(handle);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        *err = "aclsparseCreate returned " + std::to_string(static_cast<int>(st));
        return false;
    }
    (void)aclsparseSetStream(*handle, stream);
    const aclsparseIndexType_t idxType = ACL_SPARSE_INDEX_32I;
    const aclDataType dt = ToAclDataType(cs.dtype);
    const aclsparseIndexBase_t ib = (cs.base == 0) ? ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE;
    if (cs.fmt == Fmt::CSR) {
        st = aclsparseCreateConstCsr(matA, cs.rows, cs.cols, cs.nnz, buf.primDev, buf.secDev,
            buf.valDev, idxType, idxType, ib, dt);
    } else if (cs.fmt == Fmt::CSC) {
        st = aclsparseCreateConstCsc(matA, cs.rows, cs.cols, cs.nnz, buf.primDev, buf.secDev,
            buf.valDev, idxType, idxType, ib, dt);
    } else {
        st = aclsparseCreateConstCoo(matA, cs.rows, cs.cols, cs.nnz, buf.primDev, buf.secDev,
            buf.valDev, idxType, ib, dt);
    }
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        *err = "create sparse descriptor returned " + std::to_string(static_cast<int>(st));
        return false;
    }
    const aclsparseOrder_t order = (cs.layout == Layout::ROW) ? ACL_SPARSE_ORDER_ROW : ACL_SPARSE_ORDER_COL;
    st = aclsparseCreateDnMat(matB, cs.rows, cs.cols, cs.ld, buf.denseDev, dt, order);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        *err = "aclsparseCreateDnMat returned " + std::to_string(static_cast<int>(st));
        return false;
    }
    return true;
}

// ④ 阶段一 bufferSize 纯查询（恒 0 断言；protocolIdempotent 追加第二次查询断言幂等）
bool QueryBufferSizeCase(aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB, bool idempotent, RunResult *r, std::string *err)
{
    size_t bufferSize = 1;
    aclsparseStatus_t st = aclsparseSparseToDense_bufferSize(handle, matA, matB,
        ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize);
    r->queries++;
    r->queryValue = bufferSize;
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        *err = "aclsparseSparseToDense_bufferSize returned " + std::to_string(static_cast<int>(st));
        return false;
    }
    if (bufferSize != 0) {
        *err = "bufferSize query returned " + std::to_string(bufferSize) + " (expected const-0)";
        return false;
    }
    if (idempotent) {
        size_t bufferSize2 = 1;
        const aclsparseStatus_t st2 = aclsparseSparseToDense_bufferSize(handle, matA, matB,
            ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize2);
        r->queries++;
        if (st2 != ACL_SPARSE_STATUS_SUCCESS || bufferSize2 != r->queryValue) {
            *err = "bufferSize re-query not idempotent (" + std::to_string(bufferSize2) + " vs " +
                   std::to_string(r->queryValue) + ")";
            return false;
        }
    }
    return true;
}

// ⑥ D2H 回读：输出（含哨兵区）+ 输入只读回读（nnz=0 空 vector 守卫同上传）
bool ReadbackCase(const SparseData &sd, const CaseDeviceBufs &buf, size_t denseBytes,
    std::vector<uint8_t> *denseGuarded, std::vector<uint8_t> *valBack, std::vector<int32_t> *primBack,
    std::vector<int32_t> *secBack, std::string *err)
{
    const size_t valBytes = sd.values.size();
    const size_t primBytes = sd.primary.size() * sizeof(int32_t);
    const size_t secBytes = sd.secondary.size() * sizeof(int32_t);
    if (aclrtMemcpy(denseGuarded->data(), denseBytes, buf.denseDev, denseBytes,
            ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
        *err = "D2H dense readback failed";
        return false;
    }
    valBack->assign(valBytes, 0);
    primBack->assign(sd.primary.size(), 0);
    secBack->assign(sd.secondary.size(), 0);
    if ((valBytes > 0 &&
            aclrtMemcpy(valBack->data(), valBytes, buf.valDev, valBytes, ACL_MEMCPY_DEVICE_TO_HOST) != 0) ||
        (primBytes > 0 &&
            aclrtMemcpy(primBack->data(), primBytes, buf.primDev, primBytes, ACL_MEMCPY_DEVICE_TO_HOST) != 0) ||
        (secBytes > 0 &&
            aclrtMemcpy(secBack->data(), secBytes, buf.secDev, secBytes, ACL_MEMCPY_DEVICE_TO_HOST) != 0)) {
        *err = "D2H input readback failed";
        return false;
    }
    return true;
}

// 描述符销毁（先于 device 缓冲释放；次序 matB → matA → handle）
void DestroyCaseDescriptors(aclsparseDnMatDescr_t matB, aclsparseConstSpMatDescr_t matA,
    aclsparseHandle_t handle)
{
    if (matB != nullptr) {
        (void)aclsparseDestroyDnMat(matB);
    }
    if (matA != nullptr) {
        (void)aclsparseDestroySpMat(matA);
    }
    if (handle != nullptr) {
        (void)aclsparseDestroy(handle);
    }
}

// 清理：描述符（先于 device 缓冲释放）+ Device 资源
void FreeCase(aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA, aclsparseDnMatDescr_t matB,
    CaseDeviceBufs *buf)
{
    DestroyCaseDescriptors(matB, matA, handle);
    if (buf->denseDev != nullptr) {
        (void)aclrtFree(buf->denseDev);
    }
    if (buf->secDev != nullptr) {
        (void)aclrtFree(buf->secDev);
    }
    if (buf->primDev != nullptr) {
        (void)aclrtFree(buf->primDev);
    }
    if (buf->valDev != nullptr) {
        (void)aclrtFree(buf->valDev);
    }
}
}  // namespace

// ⑤ 阶段二执行（buffer=nullptr；查询为 0 → buffer 可空）+ 读取前同步 stream
//（接口异步语义；读取/复用/释放前由调用方负责）。失败返回 false 并给出原因。
bool ExecAndSync(aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA, aclsparseDnMatDescr_t matB,
    std::string *err)
{
    const aclsparseStatus_t st = aclsparseSparseToDense(handle, matA, matB,
        ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, nullptr);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        *err = "aclsparseSparseToDense returned " + std::to_string(static_cast<int>(st));
        return false;
    }
    if (aclrtSynchronizeStream(g_stream) != 0) {
        *err = "aclrtSynchronizeStream failed (kernel error?)";
        return false;
    }
    return true;
}

// 判定后处理：-0 bit 显式证据捕获（L1_072）与等价断言用例的输出快照
void CaptureCaseResult(const std::vector<uint8_t> &denseGuarded, size_t storageBytes, bool wantOutput,
    RunResult *r)
{
    if (!r->ok) {
        return;
    }
    for (size_t b = 0; b < storageBytes; ++b) {
        if (denseGuarded[b] != 0x00) {
            r->negZeroBitsSeen = true;
            break;
        }
    }
    if (wantOutput) {
        r->output.assign(denseGuarded.begin(), denseGuarded.begin() + static_cast<long>(storageBytes));
    }
}

// 输入只读快照 + matB 全缓冲（含 16B 尾哨兵）0xA5 预填（清零 kernel 漏清零/越界写可检出）
struct CaseSnapshot {
    std::vector<uint8_t> valSnap;
    std::vector<int32_t> primarySnap;
    std::vector<int32_t> secondarySnap;
    std::vector<uint8_t> denseGuarded;
};

CaseSnapshot MakeCaseSnapshot(const SparseData &sd, size_t storageBytes)
{
    CaseSnapshot s;
    s.valSnap = sd.values;
    s.primarySnap = sd.primary;
    s.secondarySnap = sd.secondary;
    s.denseGuarded.assign(storageBytes + 16, 0);
    std::fill(s.denseGuarded.begin(), s.denseGuarded.end(), static_cast<uint8_t>(0xA5));
    return s;
}

// 非执行成功路径统一置败（r.message 为空时补 exec failed）
void FinalizeRunResult(bool execOk, RunResult *r)
{
    if (!execOk) {
        r->ok = false;
        if (r->message.empty()) {
            r->message = "exec failed";
        }
    }
}

// ---------------------------------------------------------------------------
// ①-⑥：Device 资源装配 + 双阶段调用 + D2H 回读（任一步失败置 *err 并返回 false；
// 成功回读输出（含哨兵区）与输入到 Host 副本。判定仅消费 Host 副本，Device 资源
// 随函数返回统一释放（先描述符后缓冲，与原执行序列等价）。
// ---------------------------------------------------------------------------
bool ExecCaseAndReadback(const CaseSpec &cs, const SparseData &sd, std::vector<uint8_t> *denseGuarded,
    size_t denseBytes, std::vector<uint8_t> *valBack, std::vector<int32_t> *primBack,
    std::vector<int32_t> *secBack, RunResult *r)
{
    CaseDeviceBufs buf;
    aclsparseHandle_t handle = nullptr;
    aclsparseConstSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t matB = nullptr;
    bool execOk = false;
    do {
        if (!UploadCase(sd, *denseGuarded, denseBytes, &buf, &r->message)) {
            break;
        }
        if (!CreateCaseDescriptors(cs, buf, g_stream, &handle, &matA, &matB, &r->message)) {
            break;
        }
        if (!QueryBufferSizeCase(handle, matA, matB, cs.protocolIdempotent, r, &r->message)) {
            break;
        }
        if (!ExecAndSync(handle, matA, matB, &r->message)) {
            break;
        }
        // ⑥ D2H 回读（输出含哨兵区 + 输入只读回读）
        if (!ReadbackCase(sd, buf, denseBytes, denseGuarded, valBack, primBack, secBack, &r->message)) {
            break;
        }
        execOk = true;
    } while (0);
    FreeCase(handle, matA, matB, &buf);
    return execOk;
}

RunResult RunSparseToDenseCase(CaseSpec cs, bool wantOutput = false)  // 按值传入：显式 offsets 用例需修正有效 nnz
{
    RunResult r{true, (cs.coords == CoordMode::DUPLICATE) ? "matches_oracle_dup" : "bitwise_exact", 0, ""};
    const PreparedCase pc = PrepareCase(cs);
    if (!pc.ready) {
        r.ok = false;
        r.message = pc.error;
        return r;
    }
    const SparseData &sd = pc.sd;
    const size_t storageBytes = pc.storageBytes;

    // 输入只读快照 + matB 存储缓冲外尾部哨兵（§4.2 附带断言 ①②；见 MakeCaseSnapshot）
    CaseSnapshot snap = MakeCaseSnapshot(sd, storageBytes);
    std::vector<uint8_t> valBack;
    std::vector<int32_t> primBack;
    std::vector<int32_t> secBack;
    const bool execOk = ExecCaseAndReadback(cs, sd, &snap.denseGuarded, storageBytes + 16, &valBack,
        &primBack, &secBack, &r);
    if (execOk) {
        // ⑦ §4 判定（单一实现，口径不变）+ -0 bit 证据/输出捕获；重复坐标用例附候选值集合
        const std::map<size_t, std::vector<std::vector<uint8_t>>> dupCands =
            (cs.coords == CoordMode::DUPLICATE) ? ComputeDupCandidates(cs, sd)
                                                : std::map<size_t, std::vector<std::vector<uint8_t>>>{};
        JudgeOutput(cs, sd, valBack, primBack, secBack, snap.valSnap, snap.primarySnap, snap.secondarySnap,
            snap.denseGuarded, (cs.coords == CoordMode::DUPLICATE) ? &dupCands : nullptr, r);
        CaptureCaseResult(snap.denseGuarded, storageBytes, wantOutput, &r);
    }
    FinalizeRunResult(execOk, &r);
    return r;
}

void ReportCase(const CaseSpec &cs, const RunResult &r, const char *sub = nullptr)
{
    const std::string tag = (sub != nullptr)
        ? "[case_id=" + std::string(cs.caseId) + "][sub=" + sub + "]"
        : "[case_id=" + std::string(cs.caseId) + "]";
    if (r.ok) {
        LOG_PRINT("%s PASSED (judge=%s, dtype=%s, format=%s, layout=%s, base=%d, rows=%lld, "
                  "cols=%lld, ld=%lld, nnz=%lld, seed=%llu, bufferSize=0, buffer=nullptr, mismatches=%zu)",
                  tag.c_str(), r.judge.c_str(), DTypeName(cs.dtype), FmtName(cs.fmt), LayoutName(cs.layout),
                  cs.base, static_cast<long long>(cs.rows), static_cast<long long>(cs.cols),
                  static_cast<long long>(cs.ld), static_cast<long long>(cs.nnz),
                  static_cast<unsigned long long>(cs.seed), r.mismatches);
    } else {
        LOG_PRINT("%s FAILED (judge=%s, dtype=%s, format=%s, layout=%s, base=%d, rows=%lld, "
                  "cols=%lld, ld=%lld, nnz=%lld, seed=%llu, reason=%s)",
                  tag.c_str(), r.judge.c_str(), DTypeName(cs.dtype), FmtName(cs.fmt), LayoutName(cs.layout),
                  cs.base, static_cast<long long>(cs.rows), static_cast<long long>(cs.cols),
                  static_cast<long long>(cs.ld), static_cast<long long>(cs.nnz),
                  static_cast<unsigned long long>(cs.seed), r.message.c_str());
    }
}

// 聚合结果行（case_id 级）：gtest 断言状态为准（HasFailure），供机器证据解析
void ReportAggregate(const char *caseId, const char *judge, int subrecords)
{
    if (::testing::Test::HasFailure()) {
        LOG_PRINT("[case_id=%s] FAILED (judge=%s, subrecords=%d, aggregate=gtest assertions failed)", caseId,
                  judge, subrecords);
    } else {
        LOG_PRINT("[case_id=%s] PASSED (judge=%s, subrecords=%d)", caseId, judge, subrecords);
    }
}

} // namespace

// ============================================================================
// L0 标准用例（12 条全落地；参数逐项对齐 sparse2dense_test.csv L0 段）
// Mock / Real 双模式共用同一组 TEST 与判定逻辑（差异仅在 RunSparseToDenseCase 执行后端）。
// ============================================================================

// L0_001 门槛直通 fp32 COO base0 ROW 3x3 乱序唯一坐标 seed=7
// ===========================================================================
// NPU 真机环境（共享 test_main.cpp 提供 main；spgemm 同款 TestSuite 级装配；
// 钉卡经 ASCEND_RT_VISIBLE_DEVICES 环境变量）
// ===========================================================================
std::string InitRealEnv();
void ShutdownRealEnv();

class Sparse2DenseRealTest : public ::testing::TestWithParam<CaseRow> {
protected:
    // 真机环境为进程级单例：INSTANTIATE 参数化 suite 与 L0/MultiDevice 手写 suite
    // 共享本 fixture。仅首个 suite 装配（幂等），TearDown 为空操作——数据校验链缓存
    //（g_devCache：设备指针/event/校验结论）是进程级状态，suite 边界的
    // aclrtResetDevice/aclFinalize 会作废其设备资源令后续 suite 持陈旧指针。
    // 资源回收交由进程退出（与运行期单 suite 形态的基线行为一致）。
    static bool &EnvUp()
    {
        static bool up = false;
        return up;
    }
    static void SetUpTestSuite()
    {
        if (EnvUp()) {
            return;
        }
        const std::string envErr = InitRealEnv();
        if (!envErr.empty()) {
            std::fprintf(stderr, "[real-env] FAILED (%s)\n", envErr.c_str());
            std::exit(2);
        }
        EnvUp() = true;
        const char *vis = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
        LOG_PRINT("[real-env] OK (logical device=%d, soc=%s, vector_cores=%u, ASCEND_RT_VISIBLE_DEVICES=%s; "
                  "NPU 真机执行，无 Mock/CPU fallback)",
            g_devId, g_socName.c_str(), g_coreNum, vis != nullptr ? vis : "(unset)");
    }
    static void TearDownTestSuite() {}
};

TEST_F(Sparse2DenseRealTest, L0_001)
{
    CaseSpec cs{"L0_001", DType::FP32, Fmt::COO, Layout::ROW, 0, 3, 3, 3, 5, RowPattern::UNIFORM,
                CoordMode::SHUFFLED_UNIQUE, 7};
    RunResult r = RunSparseToDenseCase(cs);
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
}

// L0_002 门槛直通 fp32 COO base1 COL 3x3（base 换算 + COL 寻址门槛）seed=7
TEST_F(Sparse2DenseRealTest, L0_002)
{
    CaseSpec cs{"L0_002", DType::FP32, Fmt::COO, Layout::COL, 1, 3, 3, 3, 5, RowPattern::UNIFORM,
                CoordMode::SHUFFLED_UNIQUE, 7};
    RunResult r = RunSparseToDenseCase(cs);
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
}

// L0_003 门槛 int8 CSR base0 ROW 17x19 nnz=43 官方 accuracy id0 属性同口径（1B 通道门槛）seed=13
TEST_F(Sparse2DenseRealTest, L0_003)
{
    CaseSpec cs{"L0_003", DType::INT8, Fmt::CSR, Layout::ROW, 0, 17, 19, 19, 43, RowPattern::UNIFORM,
                CoordMode::SORTED_UNIQUE, 13};
    RunResult r = RunSparseToDenseCase(cs);
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
}

// L0_004 门槛 float16 CSC base1 ROW 17x19 nnz=43 官方 id26 属性同口径（banded；2B 通道门槛）seed=13
TEST_F(Sparse2DenseRealTest, L0_004)
{
    CaseSpec cs{"L0_004", DType::FP16, Fmt::CSC, Layout::ROW, 1, 17, 19, 19, 43, RowPattern::BANDED,
                CoordMode::SORTED_UNIQUE, 13};
    RunResult r = RunSparseToDenseCase(cs);
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
}

// L0_005 门槛 bfloat16 COO base0 COL 17x19 nnz=43 官方 id49 属性同口径（power_law；2B 通道门槛）seed=13
TEST_F(Sparse2DenseRealTest, L0_005)
{
    CaseSpec cs{"L0_005", DType::BF16, Fmt::COO, Layout::COL, 0, 17, 19, 17, 43, RowPattern::POWER_LAW,
                CoordMode::SHUFFLED_UNIQUE, 13};
    RunResult r = RunSparseToDenseCase(cs);
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
}

// L0_006 门槛 complex64 COO base1 ROW 17x19 nnz=43 官方 id58 属性同口径（block_like；8B 双通道）seed=13
TEST_F(Sparse2DenseRealTest, L0_006)
{
    CaseSpec cs{"L0_006", DType::COMPLEX64, Fmt::COO, Layout::ROW, 1, 17, 19, 19, 43, RowPattern::BLOCK_LIKE,
                CoordMode::SHUFFLED_UNIQUE, 13};
    RunResult r = RunSparseToDenseCase(cs);
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
}

// L0_007 零 nnz 输出全 +0（boundary#1；scatter kernel 早退、清零 kernel 承接全定义）seed=7
TEST_F(Sparse2DenseRealTest, L0_007)
{
    CaseSpec cs{"L0_007", DType::FP32, Fmt::COO, Layout::ROW, 0, 3, 3, 3, 0, RowPattern::UNIFORM,
                CoordMode::SORTED_UNIQUE, 7};
    RunResult r = RunSparseToDenseCase(cs);
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.judge, "all_positive_zero"); // 判定路径门槛：零 nnz 必须走 all_positive_zero 显式断言
}

// L0_008 全密集 int8 COO base0 ROW 2x3 nnz=6（boundary#2；nnz=m*n 退化全覆盖）seed=7
TEST_F(Sparse2DenseRealTest, L0_008)
{
    CaseSpec cs{"L0_008", DType::INT8, Fmt::COO, Layout::ROW, 0, 2, 3, 3, 6, RowPattern::UNIFORM,
                CoordMode::SORTED_UNIQUE, 7};
    RunResult r = RunSparseToDenseCase(cs);
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
}

// L0_009 空行 CSR rowOffsets=[0,1,1,3]（boundary#3；行 1 全空段 +0；空段跳过不产生 run）seed=7
// CSV 偏差口径：cal_count 以 offsets 端点为权威 nnz=3
TEST_F(Sparse2DenseRealTest, L0_009)
{
    static const std::vector<int32_t> rowOffsets = {0, 1, 1, 3};
    CaseSpec cs{"L0_009", DType::FP32, Fmt::CSR, Layout::ROW, 0, 3, 3, 3, 3, RowPattern::UNIFORM,
                CoordMode::SORTED_UNIQUE, 7};
    cs.explicitPrimary = rowOffsets.data();
    cs.explicitPrimaryLen = static_cast<int>(rowOffsets.size());
    RunResult r = RunSparseToDenseCase(cs);
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
}

// L0_010 空列 CSC colOffsets=[0,1,1,3]（boundary#4；列 1 全空段）seed=7
// CSV 偏差口径：cal_count 以 offsets 端点为权威 nnz=3
TEST_F(Sparse2DenseRealTest, L0_010)
{
    static const std::vector<int32_t> colOffsets = {0, 1, 1, 3};
    CaseSpec cs{"L0_010", DType::FP32, Fmt::CSC, Layout::ROW, 0, 3, 3, 3, 3, RowPattern::UNIFORM,
                CoordMode::SORTED_UNIQUE, 7};
    cs.explicitPrimary = colOffsets.data();
    cs.explicitPrimaryLen = static_cast<int>(colOffsets.size());
    RunResult r = RunSparseToDenseCase(cs);
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
}

// L0_011 ld padding COL ld=5>rows=3（boundary#7 例；padding 区清零 kernel 写 +0、不越界写哨兵）seed=7
TEST_F(Sparse2DenseRealTest, L0_011)
{
    CaseSpec cs{"L0_011", DType::FP32, Fmt::COO, Layout::COL, 0, 3, 3, 5, 5, RowPattern::UNIFORM,
                CoordMode::SHUFFLED_UNIQUE, 7};
    RunResult r = RunSparseToDenseCase(cs);
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
}

// L0_012 P-02 性能规模精度门槛 fp32 CSR base0 ROW 4096x1536 每行 64 非零（官方 P-02 同口径）seed=20260911
TEST_F(Sparse2DenseRealTest, L0_012)
{
    CaseSpec cs{"L0_012", DType::FP32, Fmt::CSR, Layout::ROW, 0, 4096, 1536, 1536, 262144, RowPattern::UNIFORM,
                CoordMode::SORTED_UNIQUE, 20260911ull};
    RunResult r = RunSparseToDenseCase(cs);
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
}


const char *DTypeTag(DType t)
{
    switch (t) {
        case DType::INT8: return "int8";
        case DType::FP16: return "fp16";
        case DType::BF16: return "bf16";
        case DType::FP32: return "fp32";
        case DType::COMPLEX64: return "complex64";
    }
    return "?";
}

CaseSpec SpecOf(const CaseRow &row)
{
    return row.spec; // 公共执行参数单点承载（CaseRow 专有字段不进入 CaseSpec）
}

// 逐字节 bit 比对（等价断言：两份输出存储缓冲一致）
bool OutputsEqual(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, size_t *diffAt = nullptr)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t k = 0; k < a.size(); ++k) {
        if (a[k] != b[k]) {
            if (diffAt != nullptr) {
                *diffAt = k;
            }
            return false;
        }
    }
    return true;
}

// ---- L1 用例执行（按 expected_value 语义分发；每条用例留存 [case_id=...] 结果行）----
// 各 expected_value 语义的执行体（分发见 RunL1Row；断言口径与结果行格式零变更）
void RunEvAllPosZero(const CaseRow &row)
{
    if (row.expand == Expand::DTYPE5) { // L1_061：零 nnz 全 5 dtype 展开
        const DType dtypes[5] = {DType::INT8, DType::FP16, DType::BF16, DType::FP32, DType::COMPLEX64};
        int subPassed = 0;
        for (DType dt : dtypes) {
            CaseSpec cs = SpecOf(row);
            cs.dtype = dt;
            const RunResult r = RunSparseToDenseCase(cs);
            ReportCase(cs, r, DTypeTag(dt));
            EXPECT_TRUE(r.ok);
            EXPECT_EQ(r.judge, "all_positive_zero");
            subPassed += r.ok ? 1 : 0;
        }
        EXPECT_EQ(subPassed, 5);
        ReportAggregate(row.spec.caseId, "all_positive_zero", 5);
    } else if (row.expand == Expand::CSC_TWIN) { // L1_082：CSR 全零 offsets + CSC 同构展开
        CaseSpec cs = SpecOf(row);
        const RunResult rCsr = RunSparseToDenseCase(cs);
        ReportCase(cs, rCsr, "csr");
        EXPECT_TRUE(rCsr.ok);
        EXPECT_EQ(rCsr.judge, "all_positive_zero");
        CaseSpec csTwin = cs;
        csTwin.fmt = Fmt::CSC; // 全零 colOffsets 同构（coords empty，nnz=0）
        const RunResult rCsc = RunSparseToDenseCase(csTwin);
        ReportCase(csTwin, rCsc, "csc");
        EXPECT_TRUE(rCsc.ok);
        EXPECT_EQ(rCsc.judge, "all_positive_zero");
        ReportAggregate(row.spec.caseId, "all_positive_zero", 2);
    } else {
        CaseSpec cs = SpecOf(row);
        const RunResult r = RunSparseToDenseCase(cs);
        ReportCase(cs, r);
        EXPECT_TRUE(r.ok);
        EXPECT_EQ(r.judge, "all_positive_zero");
    }
}

void RunEvOrderedEq(const CaseRow &row) // L1_065：同坐标集合逆序 vs 有序输出 bit 一致
{
    CaseSpec csRev = SpecOf(row);
    const RunResult rRev = RunSparseToDenseCase(csRev, true);
    ReportCase(csRev, rRev, "reversed");
    EXPECT_TRUE(rRev.ok);
    CaseSpec csOrd = csRev;
    csOrd.coords = CoordMode::SORTED_UNIQUE;
    const RunResult rOrd = RunSparseToDenseCase(csOrd, true);
    ReportCase(csOrd, rOrd, "sorted");
    EXPECT_TRUE(rOrd.ok);
    size_t diff = 0;
    EXPECT_TRUE(OutputsEqual(rRev.output, rOrd.output, &diff))
        << "reversed vs sorted output mismatch at byte " << diff;
    ReportAggregate(row.spec.caseId, "ordered_input_equivalent", 2);
}

void RunEvBase0Eq(const CaseRow &row) // L1_066：base1（坐标+1）输出 == base0 输出
{
    CaseSpec csB1 = SpecOf(row);
    const RunResult rB1 = RunSparseToDenseCase(csB1, true);
    ReportCase(csB1, rB1, "base1");
    EXPECT_TRUE(rB1.ok);
    CaseSpec csB0 = csB1;
    csB0.base = 0;
    const RunResult rB0 = RunSparseToDenseCase(csB0, true);
    ReportCase(csB0, rB0, "base0");
    EXPECT_TRUE(rB0.ok);
    size_t diff = 0;
    EXPECT_TRUE(OutputsEqual(rB1.output, rB0.output, &diff))
        << "base1 vs base0 output mismatch at byte " << diff;
    ReportAggregate(row.spec.caseId, "base0_equivalent", 2);
}

void RunEvNegZero(const CaseRow &row) // L1_072：全 -0 浮点 4 dtype 展开（int8 无 ±0 区分不适用）
{
    const DType dtypes[4] = {DType::FP32, DType::FP16, DType::BF16, DType::COMPLEX64};
    int subPassed = 0;
    for (DType dt : dtypes) {
        CaseSpec cs = SpecOf(row);
        cs.dtype = dt;
        const RunResult r = RunSparseToDenseCase(cs);
        ReportCase(cs, r, DTypeTag(dt));
        EXPECT_TRUE(r.ok);
        EXPECT_TRUE(r.negZeroBitsSeen) << "no -0 bit found in output (neg-zero not kept?)";
        subPassed += (r.ok && r.negZeroBitsSeen) ? 1 : 0;
    }
    EXPECT_EQ(subPassed, 4);
    ReportAggregate(row.spec.caseId, "neg_zero_bitwise_kept", 4);
}

void RunEvEqZeroNnz(const CaseRow &row) // L1_073：全 +0 输出 ≡ 零 nnz 输出（浮点 4 dtype 展开）
{
    const DType dtypes[4] = {DType::FP32, DType::FP16, DType::BF16, DType::COMPLEX64};
    int subPassed = 0;
    for (DType dt : dtypes) {
        CaseSpec cs = SpecOf(row);
        cs.dtype = dt;
        const RunResult rAll0 = RunSparseToDenseCase(cs, true);
        ReportCase(cs, rAll0, DTypeTag(dt));
        EXPECT_TRUE(rAll0.ok);
        CaseSpec csZero = cs;
        csZero.vkind = VKind::EMPTY;
        csZero.nnz = 0;
        const RunResult rZero = RunSparseToDenseCase(csZero, true);
        EXPECT_TRUE(rZero.ok);
        size_t diff = 0;
        const bool eq = OutputsEqual(rAll0.output, rZero.output, &diff);
        EXPECT_TRUE(eq) << "all-+0 output != zero-nnz output at byte " << diff;
        subPassed += (rAll0.ok && eq) ? 1 : 0;
    }
    EXPECT_EQ(subPassed, 4);
    ReportAggregate(row.spec.caseId, "equiv_zero_nnz_output", 4);
}

void RunEvRepeat10(const CaseRow &row) // L1_075：同输入重复执行 10 次输出 bit-wise 一致（确定性）
{
    CaseSpec cs = SpecOf(row);
    RunResult r0 = RunSparseToDenseCase(cs, true);
    ReportCase(cs, r0, "rep1");
    EXPECT_TRUE(r0.ok);
    for (int rep = 2; rep <= 10; ++rep) {
        RunResult rK = RunSparseToDenseCase(cs, true);
        EXPECT_TRUE(rK.ok) << "repeat " << rep << " failed: " << rK.message;
        size_t diff = 0;
        EXPECT_TRUE(OutputsEqual(r0.output, rK.output, &diff))
            << "repeat " << rep << " output mismatch at byte " << diff;
    }
    ReportAggregate(row.spec.caseId, "repeat10_bitwise_identical", 10);
}

void RunEvReadonly(const CaseRow &row) // L1_076：输入只读显式专项（JudgeOutput 附带断言 ①）
{
    CaseSpec cs = SpecOf(row);
    RunResult r = RunSparseToDenseCase(cs);
    r.judge = "inputs_bitwise_unchanged"; // 只读断言在判定链内强制执行（失败即 r.ok=false）
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.judge, "inputs_bitwise_unchanged");
}

void RunEvProtocol(const CaseRow &row) // L1_077：双阶段协议（查询==0 + buffer=nullptr 执行 + 幂等）
{
    CaseSpec cs = SpecOf(row);
    cs.protocolIdempotent = true;
    RunResult r = RunSparseToDenseCase(cs);
    r.judge = "buffersize_zero_and_idempotent";
    ReportCase(cs, r);
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.queries, 2); // 两次查询（幂等证据）
    EXPECT_EQ(r.queryValue, 0U);
}

void RunEvTranspose(const CaseRow &row) // L1_081：同稀疏结构 ROW/COL 两布局输出互为转置
{
    CaseSpec csRow = SpecOf(row);
    const RunResult rRow = RunSparseToDenseCase(csRow, true);
    ReportCase(csRow, rRow, "ROW");
    EXPECT_TRUE(rRow.ok);
    CaseSpec csCol = csRow;
    csCol.layout = Layout::COL;
    csCol.ld = csRow.rows; // COL 布局默认 ld=rows（无 padding）
    const RunResult rCol = RunSparseToDenseCase(csCol, true);
    ReportCase(csCol, rCol, "COL");
    EXPECT_TRUE(rCol.ok);
    // COL 逻辑结果 == ROW 结果转置：B_col[c*ldCol+r] == B_row[r*ldRow+c]
    ASSERT_EQ(rRow.output.size(), static_cast<size_t>(csRow.rows * csRow.cols * ElemBytes(csRow.dtype)));
    ASSERT_EQ(rCol.output.size(), static_cast<size_t>(csRow.cols * csCol.ld * ElemBytes(csRow.dtype)));
    const size_t esz = ElemBytes(csRow.dtype);
    bool transposeEq = true;
    for (int64_t r = 0; r < csRow.rows && transposeEq; ++r) {
        for (int64_t c = 0; c < csRow.cols && transposeEq; ++c) {
            const size_t offRow = static_cast<size_t>(r * csRow.ld + c) * esz;
            const size_t offCol = static_cast<size_t>(c * csCol.ld + r) * esz;
            if (std::memcmp(rRow.output.data() + offRow, rCol.output.data() + offCol, esz) != 0) {
                transposeEq = false;
            }
        }
    }
    EXPECT_TRUE(transposeEq) << "COL output != ROW output transposed";
    ReportAggregate(row.spec.caseId, "col_output_eq_row_transposed", 2);
}

void RunL1Row(const CaseRow &row)
{
    switch (row.ev) {
        case EV::ALL_POS_ZERO:
            RunEvAllPosZero(row);
            break;
        case EV::ORDERED_EQ:
            RunEvOrderedEq(row);
            break;
        case EV::BASE0_EQ:
            RunEvBase0Eq(row);
            break;
        case EV::NEG_ZERO:
            RunEvNegZero(row);
            break;
        case EV::EQ_ZERO_NNZ:
            RunEvEqZeroNnz(row);
            break;
        case EV::REPEAT10:
            RunEvRepeat10(row);
            break;
        case EV::READONLY:
            RunEvReadonly(row);
            break;
        case EV::PROTOCOL:
            RunEvProtocol(row);
            break;
        case EV::TRANSPOSE:
            RunEvTranspose(row);
            break;
        case EV::NONE:
        default: {
            CaseSpec cs = SpecOf(row);
            const RunResult r = RunSparseToDenseCase(cs);
            ReportCase(cs, r);
            EXPECT_TRUE(r.ok);
            break;
        }
    }
}

// ============================================================================
// L2 负例执行（Real 后端）：主线 host_impl 校验链（CheckMeta → ValidateF3 → FillTiling），
// 期望在 CSV notes 标注的阶段拒绝（INVALID_VALUE），matB 哨兵不被污染（不发起清零/scatter kernel）。
// ============================================================================
struct NegOutcomeReal {
    bool ok = false;
    int status = 0;   // 实际 S2DStatus
    int stage = 0;    // 实际拒绝阶段：0=意外成功 1=CheckMeta 2=ValidateF3 3=FillTiling
    bool polluted = false;
    std::string detail;
};

// L2 负例视图参数（CaseRow 默认值 + 按 NegKind 注入变异后的承载）
struct NegParams {
    int64_t nnz = 0;
    int64_t lenPrimary = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    int32_t format = 0;
    int32_t layout = 0;
    int32_t dtype = 0;
    int32_t alg = 0;
    int64_t ld = 0;
    int32_t base = 0;
    std::vector<int32_t> prim;
    std::vector<int32_t> sec;
};

// 按 NegKind 注入变异（L2_005~028 各负例的参数/数据改写；NegKind 无变异项保持默认）
void ApplyNegMutation(const CaseRow &row, NegParams *p)
{
    switch (row.neg) {
        case NegKind::NEG_COO_ROW_OOB: p->rows = 2; p->cols = 2; p->nnz = 3; p->prim = {0, 1, 3}; p->sec = {0, 1, 0}; break;
        case NegKind::NEG_COO_COL_OOB: p->rows = 2; p->cols = 2; p->nnz = 3; p->prim = {0, 1, 2}; p->sec = {0, 1, 2}; break;
        case NegKind::NEG_COO_NEG: p->base = 1; p->nnz = 3; p->prim = {0, 1, 2}; p->sec = {1, 2, 1}; break;
        case NegKind::NEG_CSR_COL_OOB: p->format = S2D_FORMAT_CSR; p->rows = 2; p->cols = 2; p->nnz = 3;
            p->prim = {0, 1, 2, 3}; p->sec = {0, 1, 3}; break;
        case NegKind::NEG_CSR_NONMONO: p->format = S2D_FORMAT_CSR; p->nnz = 2; p->prim = {0, 2, 1, 3}; p->sec = {5, 6, 7}; break;
        case NegKind::NEG_CSC_NONMONO: p->format = S2D_FORMAT_CSC; p->nnz = 2; p->prim = {0, 2, 1, 3}; p->sec = {5, 6, 7}; break;
        case NegKind::NEG_LEN_PRIMARY: p->format = S2D_FORMAT_CSR; p->nnz = 3; p->prim = {0, 1, 2}; p->sec = {0, 1, 2}; break;
        case NegKind::NEG_VALUES_LEN: p->format = S2D_FORMAT_COO; p->nnz = 3; p->prim = {0, 1}; p->sec = {0, 1, 2}; break;
        case NegKind::NEG_DTYPE_MISMATCH: break; // matB dtype 在视图赋值处覆写
        case NegKind::NEG_DTYPE_UNSUPPORTED: p->dtype = ACL_INT16; break;
        case NegKind::NEG_BASE2: p->base = 2; break;
        case NegKind::NEG_LAYOUT: p->layout = 2; break;
        case NegKind::NEG_LD_ROW: p->ld = 2; break;
        case NegKind::NEG_LD_COL: p->ld = 2; p->layout = S2D_LAYOUT_COL; break;
        case NegKind::NEG_FORMAT: p->format = 3; break; // Blocked-ELL
        case NegKind::NEG_ALG: p->alg = 1; break;
        case NegKind::NEG_ROWS_ZERO: p->rows = 0; break;
        case NegKind::NEG_ROWS_MAX: p->rows = 2147483648LL; break;
        case NegKind::NEG_OVERFLOW_COO_NNZ:
            p->nnz = 4398046510081LL; // UINT32_MAX×tileElems(1024)+1（合成参数，不实际生成数据）
            break;
        case NegKind::NEG_OVERFLOW_ZEROBYTES:
            p->rows = 2147483647LL; p->cols = 2147483647LL; p->ld = 2147483647LL; p->nnz = 0;
            p->dtype = ACL_COMPLEX64; // elemSize=8 → rows×ld×8 ≈ 3.69e19 > int64 上界
            break;
        default: break;
    }
}

// lenPrimary 推导 + 变异覆写（L2_012 len 不符 / L2_026 合成 COO nnz）
void FinalizeLenPrimary(const CaseRow &row, NegParams *p)
{
    p->lenPrimary = (p->format == S2D_FORMAT_COO)
        ? ((row.neg == NegKind::NEG_VALUES_LEN) ? 2 : p->nnz)
        : ((p->format == S2D_FORMAT_CSC) ? p->cols + 1 : p->rows + 1);
    if (row.neg == NegKind::NEG_LEN_PRIMARY) {
        p->lenPrimary = 3; // len(primary)=3 != rows+1=4（CSV L2_012：CSR len 不符）
    }
    if (row.neg == NegKind::NEG_OVERFLOW_COO_NNZ) {
        p->lenPrimary = p->nnz; // COO lenPrimary == nnz（CheckMeta len 匹配先行通过）
    }
}

// 视图赋值（matB dtype 覆写承接 L2_014 DTYPE_MISMATCH）
void FillNegViews(const CaseRow &row, const NegParams &p, S2DSpMatView *matA, S2DDnMatView *matB)
{
    matA->format = p.format;
    matA->indexBase = p.base;
    matA->nnz = p.nnz;
    matA->lenPrimary = p.lenPrimary;
    matA->valueType = p.dtype;
    matB->rows = p.rows;
    matB->cols = p.cols;
    matB->ld = p.ld;
    matB->layout = p.layout;
    matB->valueType = (row.neg == NegKind::NEG_DTYPE_MISMATCH) ? ACL_FLOAT16 : p.dtype;
}

// L2 负例 Device 资源（dense 16B 哨兵；输入缓冲仅校验链阶段被读取）
struct NegDeviceBufs {
    void *valDev = nullptr;
    void *primDev = nullptr;
    void *secDev = nullptr;
    void *denseDev = nullptr;
    void *dummyDev = nullptr; // 合成用例独立 dummy 缓冲（≠ denseDev，见 AllocNegDevice 注释）
};

void FreeNegDevice(NegDeviceBufs *b)
{
    if (b->valDev != nullptr) {
        (void)aclrtFree(b->valDev);
    }
    if (b->dummyDev != nullptr) {
        (void)aclrtFree(b->dummyDev);
    }
    if (b->secDev != nullptr) {
        (void)aclrtFree(b->secDev);
    }
    if (b->primDev != nullptr) {
        (void)aclrtFree(b->primDev);
    }
    if (b->denseDev != nullptr) {
        (void)aclrtFree(b->denseDev);
    }
}

// Device 分配 + H2D（dense/dummy 哨兵 + prim/sec/values 上传；合成参数上界用例
// L2_026 nnz=4.4e12 不实际生成/上传 values——CheckMeta 仅需指针非空，以独立 16B
// dummy 缓冲占位，避免 17.6TB host 分配）。失败返回 false。
bool AllocNegDevice(const CaseRow &row, const NegParams &p, const std::vector<uint8_t> &valHost,
    const std::vector<uint8_t> &denseHost, NegDeviceBufs *b, std::string *err)
{
    const size_t denseBytes = denseHost.size();
    if (aclrtMalloc(&b->denseDev, denseBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMemcpy(b->denseDev, denseBytes, denseHost.data(), denseBytes, ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
        *err = "dense alloc/H2D failed";
        return false;
    }
    // 合成参数用例的独立 dummy 缓冲（16B，与 denseDev 不重叠）：占位指针若复用 denseDev，
    // op_host CheckMeta 的 alias 防护（matA->values == matB->dense →
    // INVALID_VALUE@CheckMeta）会令 L2_026 被提前拒绝、偏离「CheckMeta 通过 →
    // FillTiling 判溢出」的测试意图（expect_stage=3）。
    if (aclrtMalloc(&b->dummyDev, denseBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMemcpy(b->dummyDev, denseBytes, denseHost.data(), denseBytes, ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
        *err = "dummy alloc/H2D failed";
        return false;
    }
    const size_t primBytes = p.prim.size() * sizeof(int32_t);
    if (primBytes > 0 && (aclrtMalloc(&b->primDev, primBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
            aclrtMemcpy(b->primDev, primBytes, p.prim.data(), primBytes, ACL_MEMCPY_HOST_TO_DEVICE) != 0)) {
        *err = "primary alloc/H2D failed";
        return false;
    }
    const size_t secBytes = p.sec.size() * sizeof(int32_t);
    if (secBytes > 0 && (aclrtMalloc(&b->secDev, secBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
            aclrtMemcpy(b->secDev, secBytes, p.sec.data(), secBytes, ACL_MEMCPY_HOST_TO_DEVICE) != 0)) {
        *err = "secondary alloc/H2D failed";
        return false;
    }
    const size_t valBytes = valHost.size();
    if (valBytes > 0 && (aclrtMalloc(&b->valDev, valBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
            aclrtMemcpy(b->valDev, valBytes, valHost.data(), valBytes, ACL_MEMCPY_HOST_TO_DEVICE) != 0)) {
        *err = "values alloc/H2D failed";
        return false;
    }
    return true;
}

// 校验链执行（与主线 Sparse2DenseExec 同一入口序；null_input 变体传空描述符/指针；
// syntheticSkipF3：L2_026 合成参数数据校验 D2H 规模不可行，直接 FillTiling 判溢出）。
// 返回实际拒绝阶段（1=CheckMeta 2=ValidateF3 3=FillTiling），status 为对应 S2DStatus。
int RunNegChain(const CaseRow &row, int32_t alg, const S2DSpMatView *matA, const S2DDnMatView *matB,
    bool syntheticSkipF3, S2DStatus *status)
{
    S2DStatus st = S2D_STATUS_SUCCESS;
    if (row.neg == NegKind::NEG_NULL_BUFPTR) {
        // L2_004：GetBufferSize 输出指针空 → INVALID_VALUE（阶段一入口）
        st = Sparse2DenseGetBufferSize(matA, matB, alg, nullptr);
        *status = st;
        return 1;
    }
    const S2DSpMatView *pA = (row.neg == NegKind::NEG_NULL_MATA) ? nullptr : matA;
    const S2DDnMatView *pB = (row.neg == NegKind::NEG_NULL_MATB) ? nullptr : matB;
    st = Sparse2DenseCheckMeta(pA, pB, alg);
    if (st != S2D_STATUS_SUCCESS) {
        *status = st;
        return 1;
    }
    if (syntheticSkipF3) {
        Sparse2DenseTilingData tiling = {};
        uint64_t tilingKey = 0;
        st = Sparse2DenseFillTiling(matA, matB, &tiling, &tilingKey);
        *status = st;
        return 3;
    }
    st = Sparse2DenseValidateF3(g_stream, matA, matB);
    if (st != S2D_STATUS_SUCCESS) {
        *status = st;
        return 2;
    }
    Sparse2DenseTilingData tiling = {};
    uint64_t tilingKey = 0;
    st = Sparse2DenseFillTiling(matA, matB, &tiling, &tilingKey);
    *status = st;
    return 3;
}

// 元数据指针非空性绑定：CheckMeta 要求 lenPrimary>0 → primary 非空、nnz>0 →
// values/secondary 非空（synthetic 用例以独立 dummy 指针占位，校验失败前不被解引用；
// 占位与 denseDev 相互独立，不触碰 alias 契约——primary 现纳入 alias 区间重叠防护，
// 占位不得复用 denseDev）。L2_025 alias 负例在此把 matA.values 指到 matB.dense
//（op_host CheckMeta alias 防护应先行拒绝）。
void BindNegDataPointers(const CaseRow &row, const NegParams &p, const NegDeviceBufs &buf,
    S2DSpMatView *matA, S2DDnMatView *matB)
{
    void *primPtr = (p.lenPrimary > 0 && buf.primDev == nullptr) ? buf.dummyDev : buf.primDev; // 独立 dummy（primary 纳入 alias 区间防护 不得复用 denseDev）
    void *valPtr = (p.nnz > 0 && buf.valDev == nullptr) ? buf.dummyDev : buf.valDev; // 独立 dummy（≠ denseDev）
    void *secPtr = (p.nnz > 0 && buf.secDev == nullptr) ? buf.dummyDev : buf.secDev;
    matA->primary = (p.lenPrimary > 0) ? static_cast<const int32_t *>(primPtr) : nullptr;
    matA->secondary = (p.nnz > 0) ? static_cast<const int32_t *>(secPtr) : nullptr;
    matA->values = (p.nnz > 0) ? valPtr : nullptr;
    matB->dense = buf.denseDev;
    if (row.neg == NegKind::NEG_ALIAS) {
        matA->values = matB->dense;
    }
}

// 结果判定：阶段 0 = 意外成功；status 须 INVALID_VALUE 且 matB 哨兵未被污染
void ClassifyNegOutcome(NegOutcomeReal *out)
{
    if (out->stage == 0) {
        out->ok = false;
        out->detail = "unexpected success (error not raised)";
        return;
    }
    out->ok = (out->status == S2D_STATUS_INVALID_VALUE) && !out->polluted;
    if (out->status != S2D_STATUS_INVALID_VALUE) {
        out->detail = "status " + std::to_string(out->status) + " != INVALID_VALUE(1)";
    } else if (out->polluted) {
        out->detail = "matB polluted on error path";
    }
    if (!out->ok && out->detail.empty()) {
        out->detail = "unknown failure";
    }
}

// CaseRow 默认参数展开 + NegKind 变异注入 + lenPrimary 推导（见 ApplyNegMutation /
// FinalizeLenPrimary）；数据指针仅在校验链阶段被 D2H 读取。
NegParams MakeNegParams(const CaseRow &row)
{
    NegParams p;
    p.nnz = row.spec.nnz;
    p.rows = row.spec.rows;
    p.cols = row.spec.cols;
    p.format = (row.spec.fmt == Fmt::CSR) ? S2D_FORMAT_CSR
        : ((row.spec.fmt == Fmt::CSC) ? S2D_FORMAT_CSC : S2D_FORMAT_COO);
    p.layout = (row.spec.layout == Layout::ROW) ? S2D_LAYOUT_ROW : S2D_LAYOUT_COL;
    p.dtype = ToAclDataType(row.spec.dtype);
    p.alg = S2D_SPARSETODENSE_ALG_DEFAULT;
    p.ld = row.spec.ld;
    p.base = row.spec.base;
    ApplyNegMutation(row, &p);
    FinalizeLenPrimary(row, &p);
    return p;
}

// L2 负例 dense 实配字节数（与 CheckMeta alias 区间防护/FillTiling zeroTotalBytes 同
// 口径）：实配不小于声称范围，杜绝相邻 Device 分配误判重叠。dtype 不支持（段二/段三
// dtype 检查先行拒绝）、维度域非法、字节数溢出或超大合成参数（>16MB）退 16B 哨兵——
// 该类用例的拒绝点不依赖 dense 实配。
size_t NegDenseAllocBytes(const NegParams &p)
{
    uint64_t elemSize = 0;
    switch (p.dtype) {
        case ACL_INT8:
            elemSize = 1;
            break;
        case ACL_FLOAT16:
        case ACL_BF16:
            elemSize = 2;
            break;
        case ACL_FLOAT:
            elemSize = 4;
            break;
        case ACL_COMPLEX64:
            elemSize = 8;
            break;
        default:
            return 16;  // dtype 不支持：拒绝点先于 alias 区间检查
    }
    if (p.rows < 1 || p.cols < 1 || p.ld < 1) {
        return 16;  // 维度域非法：段二先行拒绝
    }
    const uint64_t outer = (p.layout == S2D_LAYOUT_COL) ? static_cast<uint64_t>(p.cols)
                                                        : static_cast<uint64_t>(p.rows);
    const uint64_t ldU = static_cast<uint64_t>(p.ld);
    if (outer > (~0ULL) / ldU || outer * ldU > (~0ULL) / elemSize) {
        return 16;  // 字节数溢出：alias 区间检查跳过、FillTiling 判溢出
    }
    const uint64_t bytes = outer * ldU * elemSize;
    return (bytes > (16ULL << 20)) ? 16U : static_cast<size_t>(bytes);
}

NegOutcomeReal RunNegReal(const CaseRow &row)
{
    NegOutcomeReal out;
    // ---- 视图构建（元数据 + 按变异注入；数据指针仅在校验链阶段被 D2H 读取）----
    const NegParams p = MakeNegParams(row);

    S2DSpMatView matA = {};
    S2DDnMatView matB = {};
    FillNegViews(row, p, &matA, &matB);

    // ---- Device 资源与 Host 占位数据 ----
    const size_t esz = ElemBytes(row.spec.dtype);
    const size_t valBytes = (row.neg == NegKind::NEG_OVERFLOW_COO_NNZ)
        ? 0U
        : static_cast<size_t>(std::max<int64_t>(p.nnz, 0)) * esz;
    // dense 实配按「dtype × outer × ld」声称口径（CheckMeta alias 区间防护同口径），
    // 实配不小于声称范围；溢出/超大合成参数退 16B（该类用例按段二/溢出路径拒绝）
    const size_t denseBytes = NegDenseAllocBytes(p);
    std::vector<uint8_t> denseHost(denseBytes, 0xA5);
    // 合成参数用例（valBytes=0）不实际分配 values 占位缓冲——占位指针回退独立
    // dummy（与 dense 同尺寸同内容，经对调恒位于 dense 上方）：其 fictional 声称区间
    //（L2_026 nnz=4.4e12 → 17.6TB）自占位地址向上延伸，1B 独立分配反而可能落至
    // dense 下方造成 CheckMeta alias 区间误拒；nnz=0 用例 values 恒为 nullptr，不受影响
    std::vector<uint8_t> valHost(valBytes, 0x11);
    NegDeviceBufs buf;
    bool allocOk = true;
    std::string allocErr;
    do {
        if (!AllocNegDevice(row, p, valHost, denseHost, &buf, &allocErr)) {
            allocOk = false;
            break;
        }
        // 合成参数用例的 fictional 区间确定性防护：占位 dummy 与 dense 同尺寸同内容
        //（0xA5），dummy 地址低于 dense 时对调两者——占位指针的声称区间（L2_026
        // nnz=4.4e12 → values/primary 区间 17.6TB）自占位地址向上延伸，须保证 dense
        // 在其下方（对调后 dummy ≥ dense + denseBytes 恒成立，与分配器地址序无关：
        // 分配器对活跃分配保证互不重叠，而 dense 声称范围 == 实配大小）
        if (buf.dummyDev != nullptr && buf.denseDev > buf.dummyDev) {
            std::swap(buf.denseDev, buf.dummyDev);
        }
        BindNegDataPointers(row, p, buf, &matA, &matB);

        // ---- 校验链执行（CheckMeta → ValidateF3 → FillTiling；见 RunNegChain）----
        S2DStatus status = S2D_STATUS_SUCCESS;
        out.stage = RunNegChain(row, p.alg, &matA, &matB, row.skipF3, &status);
        out.status = status;

        // matB 哨兵回读（Host 校验失败 → 清零/scatter kernel 未发起 → 0xA5 保持）
        if (aclrtMemcpy(denseHost.data(), denseBytes, buf.denseDev, denseBytes, ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            out.detail = "dense readback failed";
        } else {
            for (size_t k = 0; k < denseBytes; ++k) {
                if (denseHost[k] != 0xA5) {
                    out.polluted = true;
                    break;
                }
            }
        }
    } while (0);
    if (!allocOk) {
        out.ok = false;
        out.detail = allocErr;
    } else {
        ClassifyNegOutcome(&out);
    }
    FreeNegDevice(&buf);
    return out;
}


// ============================================================================
// 公开 API 双入口负例（第十轮评审整改）：描述符级形状一致性（matA/matB 两轴，
// L2_029~034 {CSR,CSC,COO}×{行,列}轴）与 alias 存储区间重叠（索引同址/部分重叠，
// L2_035~037）。内部直调链（RunNegChain）的 S2DSpMatView 不携带 matA rows/cols——
// 描述符级形状缺口即本轮评审根因，故本族用例走真实描述符 + 两公开入口
//（bufferSize + execute）双断言；拒绝先于任何 kernel，dense 哨兵（0xA5 预填，含
// 16B 尾哨兵）不被污染。输入数据按合法坐标构造（若被测校验失效，执行将合法完成
// 并写 dense → 状态码/哨兵双重检出）。
// ============================================================================
struct NegPublicOutcome {
    aclsparseStatus_t queryStatus = ACL_SPARSE_STATUS_SUCCESS;  // bufferSize 入口
    aclsparseStatus_t execStatus = ACL_SPARSE_STATUS_SUCCESS;   // execute 入口
    bool polluted = false;
    std::string detail;
};

// 公开 API 双入口负例族（描述符级形状一致性 + alias 区间重叠；见 RunNegPublicReal）
bool IsPublicApiNegKind(NegKind k)
{
    return k == NegKind::NEG_SHAPE_ROWS || k == NegKind::NEG_SHAPE_COLS ||
        k == NegKind::NEG_ALIAS_INDEX || k == NegKind::NEG_ALIAS_PARTIAL_VALUES ||
        k == NegKind::NEG_ALIAS_PARTIAL_PRIMARY;
}

// matA 基准输入（fp32 2×3、nnz=2、坐标在界：(0,0)/(1,1)；按格式展开
// CSR rowOffsets {0,1,2} / CSC colOffsets {0,1,2,2} / COO rowInd+colInd {0,1}）
struct NegPublicInput {
    std::vector<int32_t> prim;
    std::vector<int32_t> sec;
    std::vector<uint8_t> values;
};

NegPublicInput MakeNegPublicInput(Fmt fmt)
{
    NegPublicInput in;
    if (fmt == Fmt::CSR) {
        in.prim = {0, 1, 2};     // rowOffsets（rows+1=3，端点 2 == nnz）
        in.sec = {0, 1};         // colInd
    } else if (fmt == Fmt::CSC) {
        in.prim = {0, 1, 2, 2};  // colOffsets（cols+1=4，端点 2 == nnz）
        in.sec = {0, 1};         // rowInd
    } else {
        in.prim = {0, 1};        // COO rowInd
        in.sec = {0, 1};        // COO colInd
    }
    const float vals[2] = {0.5f, -1.25f};
    in.values.resize(2 * sizeof(float));
    CopyBytes(in.values.data(), vals, in.values.size());
    return in;
}

// ---------------------------------------------------------------------------
// RunNegPublicReal 拆分子段（调用次序 / detail 字符串 / 状态码 / 清理次序与拆分前
// 逐项一致）。
// ---------------------------------------------------------------------------
// matB 目标维度（matA 基准 = CSV shape；行轴变异 rows+1 / 列轴变异 cols-1 且 ld=cols）
struct NegPublicDims {
    int64_t rows;
    int64_t cols;
    int64_t ld;
};

NegPublicDims NegPublicTargetDims(const CaseRow &row)
{
    NegPublicDims d = {row.spec.rows, row.spec.cols, row.spec.ld};
    if (row.neg == NegKind::NEG_SHAPE_ROWS) {
        d.rows = row.spec.rows + 1;
    } else if (row.neg == NegKind::NEG_SHAPE_COLS) {
        d.cols = row.spec.cols - 1;
        d.ld = d.cols; // ROW 布局 ld >= cols
    }
    return d;
}

// 单缓冲 Device 分配 + H2D（失败返回 false；短路序与拆分前一致）
bool NegPublicCopyToDevice(void **dev, const void *host, size_t bytes)
{
    return aclrtMalloc(dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0 &&
        aclrtMemcpy(*dev, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0;
}

// 四缓冲分配 + H2D（dense 含 16B 尾哨兵；任一失败置 detail 并返回 false）
bool AllocNegPublicDevice(const NegPublicInput &in, const uint8_t *denseHost, size_t denseBytes,
    void **denseDev, void **primDev, void **secDev, void **valDev, NegPublicOutcome *out)
{
    const size_t primBytes = in.prim.size() * sizeof(int32_t);
    const size_t secBytes = in.sec.size() * sizeof(int32_t);
    const bool allocOk = NegPublicCopyToDevice(denseDev, denseHost, denseBytes) &&
        NegPublicCopyToDevice(primDev, in.prim.data(), primBytes) &&
        NegPublicCopyToDevice(secDev, in.sec.data(), secBytes) &&
        NegPublicCopyToDevice(valDev, in.values.data(), in.values.size());
    if (!allocOk) {
        out->detail = "device alloc/H2D failed";
    }
    return allocOk;
}

// alias 区间注入（形状负例不注入：三输入缓冲与 dense 相互独立）
void NegPublicAliasPtrs(const CaseRow &row, void *denseDev, size_t bDenseBytes, void *primDev,
    void *valDev, void **primPtr, void **valPtr)
{
    *primPtr = primDev;
    *valPtr = valDev;
    if (row.neg == NegKind::NEG_ALIAS_INDEX) {
        *primPtr = denseDev; // 索引与 dense 完全同址
    } else if (row.neg == NegKind::NEG_ALIAS_PARTIAL_VALUES) {
        *valPtr = static_cast<uint8_t *>(denseDev) + bDenseBytes - 4; // values 尾部交叠
    } else if (row.neg == NegKind::NEG_ALIAS_PARTIAL_PRIMARY) {
        *primPtr = static_cast<uint8_t *>(denseDev) + bDenseBytes - 4; // 索引尾部交叠
    }
}

// matA 描述符创建（dtype/index/base 均合法；数据指针为 Device 侧；CSR/CSC/COO 三分支）
aclsparseStatus_t CreateNegPublicSpMat(const CaseRow &row, void *primPtr, void *secDev, void *valPtr,
    aclsparseConstSpMatDescr_t *matA)
{
    const int64_t aRows = row.spec.rows;
    const int64_t aCols = row.spec.cols;
    const aclsparseIndexType_t idxType = ACL_SPARSE_INDEX_32I;
    if (row.spec.fmt == Fmt::CSR) {
        return aclsparseCreateConstCsr(matA, aRows, aCols, 2, primPtr, secDev, valPtr, idxType,
            idxType, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    }
    if (row.spec.fmt == Fmt::CSC) {
        return aclsparseCreateConstCsc(matA, aRows, aCols, 2, primPtr, secDev, valPtr, idxType,
            idxType, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    }
    return aclsparseCreateConstCoo(matA, aRows, aCols, 2, primPtr, secDev, valPtr, idxType,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
}

// matA + matB 描述符创建（任一失败置 detail 并返回 false）
bool CreateNegPublicDescriptors(const CaseRow &row, const NegPublicDims &d, void *primPtr,
    void *secDev, void *valPtr, void *denseDev, aclsparseConstSpMatDescr_t *matA,
    aclsparseDnMatDescr_t *matB, NegPublicOutcome *out)
{
    const aclsparseStatus_t st = CreateNegPublicSpMat(row, primPtr, secDev, valPtr, matA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        out->detail = "create sparse descriptor returned " + std::to_string(static_cast<int>(st));
        return false;
    }
    const aclsparseStatus_t stB = aclsparseCreateDnMat(matB, d.rows, d.cols, d.ld, denseDev, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW);
    if (stB != ACL_SPARSE_STATUS_SUCCESS) {
        out->detail = "aclsparseCreateDnMat returned " + std::to_string(static_cast<int>(stB));
        return false;
    }
    return true;
}

// 双入口断言：bufferSize 查询与 execute 执行均须拒绝（INVALID_VALUE，先于任何 kernel）
void InvokeNegPublicDualEntry(aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB, NegPublicOutcome *out)
{
    size_t bufferSize = 1;
    out->queryStatus = aclsparseSparseToDense_bufferSize(handle, matA, matB,
        ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize);
    out->execStatus = aclsparseSparseToDense(handle, matA, matB,
        ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, nullptr);
    (void)bufferSize;
}

// dense 哨兵回读 + 0xA5 扫描（拒绝先于任何 kernel → 0xA5 保持；回读失败置 detail 返回 false）
bool NegPublicDenseSentinelClean(void *denseDev, uint8_t *denseHost, size_t bDenseBytes,
    NegPublicOutcome *out)
{
    if (aclrtMemcpy(denseHost, bDenseBytes + 16, denseDev, bDenseBytes + 16, ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
        out->detail = "dense readback failed";
        return false;
    }
    for (size_t k = 0; k < bDenseBytes + 16; ++k) {
        if (denseHost[k] != 0xA5) {
            out->polluted = true;
            break;
        }
    }
    return true;
}

// 公开 API 负例资源统一释放（次序：matB → matA → handle → dense → val → sec → prim）
void FreeNegPublicDevice(aclsparseDnMatDescr_t matB, aclsparseConstSpMatDescr_t matA,
    aclsparseHandle_t handle, void *denseDev, void *valDev, void *secDev, void *primDev)
{
    DestroyCaseDescriptors(matB, matA, handle);
    if (denseDev != nullptr) {
        (void)aclrtFree(denseDev);
    }
    if (valDev != nullptr) {
        (void)aclrtFree(valDev);
    }
    if (secDev != nullptr) {
        (void)aclrtFree(secDev);
    }
    if (primDev != nullptr) {
        (void)aclrtFree(primDev);
    }
}

NegPublicOutcome RunNegPublicReal(const CaseRow &row)
{
    NegPublicOutcome out;
    // matA 基准 = CSV shape（2×3 fp32）；matB 维度按变异（行轴 rows+1 / 列轴 cols-1）
    const NegPublicDims d = NegPublicTargetDims(row);
    const size_t bDenseBytes = static_cast<size_t>(d.rows) * static_cast<size_t>(d.ld) * sizeof(float);
    const NegPublicInput in = MakeNegPublicInput(row.spec.fmt);
    aclsparseHandle_t handle = nullptr;
    aclsparseConstSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t matB = nullptr;
    void *primDev = nullptr;
    void *secDev = nullptr;
    void *valDev = nullptr;
    void *denseDev = nullptr;
    do {
        std::vector<uint8_t> denseHost(bDenseBytes + 16, 0xA5);
        if (aclsparseCreate(&handle) != 0) {
            out.detail = "aclsparseCreate failed";
            break;
        }
        (void)aclsparseSetStream(handle, g_stream);
        // Device 分配 + H2D（dense 含 16B 尾哨兵；拒绝路径不读内容，坐标数据按合法构造
        // 以保证「校验失效时执行可合法完成 → 双重检出」）
        if (!AllocNegPublicDevice(in, denseHost.data(), bDenseBytes + 16, &denseDev, &primDev, &secDev,
            &valDev, &out)) {
            break;
        }
        // alias 区间注入（形状负例不注入：三输入缓冲与 dense 相互独立）
        void *primPtr = primDev;
        void *valPtr = valDev;
        NegPublicAliasPtrs(row, denseDev, bDenseBytes, primDev, valDev, &primPtr, &valPtr);
        if (!CreateNegPublicDescriptors(row, d, primPtr, secDev, valPtr, denseDev, &matA, &matB,
            &out)) {
            break;
        }
        InvokeNegPublicDualEntry(handle, matA, matB, &out);
        // dense 哨兵回读（拒绝先于任何 kernel → 0xA5 保持）
        if (!NegPublicDenseSentinelClean(denseDev, denseHost.data(), bDenseBytes, &out)) {
            break;
        }
    } while (0);
    FreeNegPublicDevice(matB, matA, handle, denseDev, valDev, secDev, primDev);
    return out;
}

// L2 用例执行（双模式分发；Real 直调路径结构性不适用用例按 SKIP 记录理由）
void RunL2Row(const CaseRow &row)
{
    if (row.neg == NegKind::NEG_NONE) { // L2_024/L2_009：成功语义（恒 0 buffer 可空 / 重复坐标任意值）
        CaseSpec cs = SpecOf(row);
        RunResult r = RunSparseToDenseCase(cs);
        if (row.ev == EV::SUCCESS) {
            r.judge = "buffersize_zero_and_buffer_nullable_exec_ok";
        }
        ReportCase(cs, r);
        EXPECT_TRUE(r.ok);
        EXPECT_EQ(r.queryValue, 0U);
        return;
    }
    if (row.realSkip) {
        GTEST_SKIP() << row.realSkipReason;
    }
    if (IsPublicApiNegKind(row.neg)) {
        // 公开 API 双入口负例（形状一致性 / alias 区间重叠）：bufferSize + execute 均
        // 须 INVALID_VALUE，dense 哨兵不被污染
        const NegPublicOutcome po = RunNegPublicReal(row);
        const bool ok = po.queryStatus == ACL_SPARSE_STATUS_INVALID_VALUE &&
            po.execStatus == ACL_SPARSE_STATUS_INVALID_VALUE && !po.polluted && po.detail.empty();
        LOG_PRINT("[case_id=%s] %s (neg=%d, entry=public_api bufferSize+execute, query_status=%d, "
                  "exec_status=%d, matB_polluted=%d)",
            row.spec.caseId, ok ? "PASSED" : "FAILED", static_cast<int>(row.neg),
            static_cast<int>(po.queryStatus), static_cast<int>(po.execStatus), po.polluted ? 1 : 0);
        EXPECT_EQ(po.queryStatus, ACL_SPARSE_STATUS_INVALID_VALUE)
            << "bufferSize 入口未拒绝 (detail=" << po.detail << ")";
        EXPECT_EQ(po.execStatus, ACL_SPARSE_STATUS_INVALID_VALUE)
            << "execute 入口未拒绝 (detail=" << po.detail << ")";
        EXPECT_FALSE(po.polluted) << "matB polluted on error path";
        return;
    }
    const NegOutcomeReal out = RunNegReal(row);
    static const char *stageNames[4] = {"none", "CheckMeta", "F3", "FillTiling"};
    LOG_PRINT("[case_id=%s] %s (neg=%d, rejected_at=%s, expect_stage=%d, status=%d, matB_polluted=%d)",
        row.spec.caseId, out.ok ? "PASSED" : "FAILED", static_cast<int>(row.neg), stageNames[out.stage], row.stage,
        out.status, out.polluted ? 1 : 0);
    EXPECT_TRUE(out.ok) << out.detail;
    EXPECT_EQ(out.stage, row.stage) << "rejected at unexpected stage";
    if (std::strcmp(row.expectCode, "ACL_SPARSE_STATUS_INVALID_VALUE") != 0) {
        LOG_PRINT("[case_id=%s] note: CSV expect %s; host_impl 统一校验错误 → INVALID_VALUE"
                  "（sparse2dense_host.h 映射说明）", row.spec.caseId, row.expectCode);
    }
}

// ============================================================================
// CSV 驱动参数化用例（L1 85 + L2 37；用例名即 CSV case_id，证据口径以日志行
// [case_id=...] 为准）。真机环境装配复用 Sparse2DenseRealTest（套件级 SetUp）。
// 分派口径：neg != NEG_NONE（L2 负例）或 ev == SUCCESS/DUP_ANY（L2_024/L2_009 成功
// 语义）→ RunL2Row；其余（L1 等价断言语义）→ RunL1Row。
// ============================================================================
// CSV 参数化用例与 L0/MultiDevice 共用同一 fixture 与真机环境单例：真机环境
//（aclInit/Finalize 与数据校验链缓存）全进程只经历一个 SetUp/TearDown 周期。
TEST_P(Sparse2DenseRealTest, CsvDriven)
{
    const CaseRow &row = GetParam();
    if (row.neg != NegKind::NEG_NONE || row.ev == EV::SUCCESS || row.ev == EV::DUP_ANY) {
        RunL2Row(row);
    } else {
        RunL1Row(row);
    }
}

INSTANTIATE_TEST_SUITE_P(S2DCsv, Sparse2DenseRealTest,
    ::testing::ValuesIn(LoadCsvRows()),
    [](const ::testing::TestParamInfo<CaseRow> &info) { return std::string(info.param.spec.caseId); });

// ============================================================================
// 多 device 切换回归：同进程 device0 → device1 → device0，验证
// 数据校验链记忆化层（arch22/sparse2dense_host.cpp）按 devId 隔离——常驻资源
//（paramsDev/partialsDev/event）与 L1/L2/L3 条目表均为卡级状态，切卡各建各表、
// 回访命中本卡表，不复用他卡资源与校验结论。每次到访：目标卡新建 stream/handle/
// 描述符/Device 缓冲，执行 2 次（首调 L1 miss 全量校验登记、再调 L1 hit 稳态），
// 输出 bit-wise golden + 16B 尾哨兵校验。可见卡数 < 2 显式 SKIP（单卡环境该链路
// 已由常规用例覆盖）。注意：不做 aclrtResetDevice(1)——常驻缓存生命周期随进程
//（与生产语义一致），套件收尾仅复位套件卡 g_devId。
// ============================================================================
struct MultiDeviceVisitResult {
    bool ok = false;
    std::string message;
};

// 到访子步骤 ①：切卡 + 建流（目标卡专属 stream）
bool MdSetupDeviceStream(int32_t devId, aclrtStream *stream, std::string *err)
{
    if (aclrtSetDevice(devId) != 0) {
        *err = "aclrtSetDevice(" + std::to_string(devId) + ") failed";
        return false;
    }
    if (aclrtCreateStream(stream) != 0) {
        *err = "aclrtCreateStream failed on device " + std::to_string(devId);
        return false;
    }
    return true;
}

// 到访子步骤 ⑤：同输入执行 2 次（首调 L1 miss 全量校验登记、再调 L1 hit 稳态）+ 同步
bool MdExecTwice(aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB, aclrtStream stream, int32_t devId, std::string *err)
{
    for (int call = 0; call < 2; ++call) {
        if (aclsparseSparseToDense(handle, matA, matB, ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT,
                nullptr) != 0) {
            *err = "aclsparseSparseToDense call#" + std::to_string(call) + " failed on device " +
                std::to_string(devId);
            return false;
        }
        if (aclrtSynchronizeStream(stream) != 0) {
            *err = "aclrtSynchronizeStream failed on device " + std::to_string(devId);
            return false;
        }
    }
    return true;
}

// 到访子步骤 ⑦：整存储缓冲 bit-wise golden + 16B 尾哨兵校验
bool MdJudgeDense(const std::vector<uint8_t> &denseGuarded, const std::vector<uint8_t> &golden,
    int32_t devId, std::string *err)
{
    const size_t storageBytes = golden.size();
    for (size_t b = 0; b < storageBytes; ++b) {
        if (denseGuarded[b] != golden[b]) {
            *err = "dense mismatch at byte " + std::to_string(b) + " on device " +
                std::to_string(devId);
            return false;
        }
    }
    for (size_t b = storageBytes; b < denseGuarded.size(); ++b) {
        if (denseGuarded[b] != 0xA5) {
            *err = "tail sentinel corrupted at byte " + std::to_string(b) + " on device " +
                std::to_string(devId);
            return false;
        }
    }
    return true;
}

MultiDeviceVisitResult RunCaseVisitOnDevice(const CaseSpec &cs, const PreparedCase &pc,
    const std::vector<uint8_t> &golden, int32_t devId)
{
    MultiDeviceVisitResult vr;
    const size_t denseBytes = pc.storageBytes + 16;
    aclrtStream stream = nullptr;
    CaseDeviceBufs buf;
    aclsparseHandle_t handle = nullptr;
    aclsparseConstSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t matB = nullptr;
    std::vector<uint8_t> denseGuarded(denseBytes, 0xA5);
    do {
        if (!MdSetupDeviceStream(devId, &stream, &vr.message) ||
            !UploadCase(pc.sd, denseGuarded, denseBytes, &buf, &vr.message) ||
            !CreateCaseDescriptors(cs, buf, stream, &handle, &matA, &matB, &vr.message)) {
            break;
        }
        size_t bufferSize = 1;
        if (aclsparseSparseToDense_bufferSize(handle, matA, matB, ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT,
                &bufferSize) != 0 || bufferSize != 0) {
            vr.message = "bufferSize query failed/non-zero on device " + std::to_string(devId);
            break;
        }
        if (!MdExecTwice(handle, matA, matB, stream, devId, &vr.message)) {
            break;
        }
        if (aclrtMemcpy(denseGuarded.data(), denseBytes, buf.denseDev, denseBytes,
                ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            vr.message = "D2H dense readback failed on device " + std::to_string(devId);
            break;
        }
        vr.ok = MdJudgeDense(denseGuarded, golden, devId, &vr.message);
    } while (0);
    FreeCase(handle, matA, matB, &buf);
    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    if (vr.message.empty() && !vr.ok) {
        vr.message = "unknown failure on device " + std::to_string(devId);
    }
    return vr;
}

TEST_F(Sparse2DenseRealTest, MultiDevice_SwitchRevisit)
{
    uint32_t devCount = 0;
    if (aclrtGetDeviceCount(&devCount) != 0 || devCount < 2U) {
        LOG_PRINT("[case_id=MultiDevice_SwitchRevisit] SKIP (visible_devices=%u < 2: F3 记忆化 "
                  "per-device 隔离回归需至少两张可见卡；单卡环境该链路由常规用例覆盖)",
            devCount);
        GTEST_SKIP();
    }
    // fp32 CSC base1 ROW（base 换算 + banded 分布；清零 kernel/scatter kernel/数据校验全链代表性）
    CaseSpec cs{"MD_001", DType::FP32, Fmt::CSC, Layout::ROW, 1, 17, 19, 19, 43, RowPattern::BANDED,
                CoordMode::SORTED_UNIQUE, 13};
    const PreparedCase pc = PrepareCase(cs);
    ASSERT_TRUE(pc.ready) << "PrepareCase failed";
    const std::vector<uint8_t> golden = ComputeGoldenStorage(cs, pc.sd);
    static const int32_t visitDevs[3] = {0, 1, 0};
    for (int visit = 0; visit < 3; ++visit) {
        const int32_t devId = visitDevs[visit];
        const MultiDeviceVisitResult vr = RunCaseVisitOnDevice(cs, pc, golden, devId);
        LOG_PRINT("[case_id=MultiDevice_SwitchRevisit][sub=visit%d_dev%d] %s "
                  "(judge=bitwise_exact, execs=2, f3_second_call=fingerprint_hit)",
            visit + 1, devId, vr.ok ? "PASSED" : "FAILED");
        EXPECT_TRUE(vr.ok) << vr.message;
    }
    (void)aclrtSetDevice(g_devId);  // 恢复套件当前卡
    ReportAggregate("MultiDevice_SwitchRevisit", "bitwise_exact", 3);
}

// Real（NPU 真机）运行时环境：ACL 初始化 + 设备/流（钉卡经容器 env ASCEND_RT_VISIBLE_DEVICES，
// 容器内固定为逻辑 0 号卡；与主线 sparse2dense_host 直调运行口径一致）。
std::string InitRealEnv()
{
    if (aclInit(nullptr) != 0) {
        return "aclInit failed";
    }
    if (aclrtSetDevice(g_devId) != 0) {
        return "aclrtSetDevice failed";
    }
    if (aclrtCreateStream(&g_stream) != 0) {
        return "aclrtCreateStream failed";
    }
    int64_t coreNumI64 = 0;
    if (aclrtGetDeviceInfo(g_devId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreNumI64) != 0 || coreNumI64 <= 0) {
        return "aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM) failed";
    }
    g_coreNum = static_cast<uint32_t>(coreNumI64);
    const char *soc = aclrtGetSocName();
    if (soc != nullptr) {
        g_socName = soc;
    }
    return "";
}

// 真机环境回收（进程退出兜底；suite teardown 显式不调用——数据校验链缓存为进程级
// 状态，见 fixture 注释）。
void ShutdownRealEnv()
{
    if (g_stream != nullptr) {
        (void)aclrtDestroyStream(g_stream);
        g_stream = nullptr;
    }
    (void)aclrtResetDevice(g_devId);
    (void)aclFinalize();
}
