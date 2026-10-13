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


// ============================================================================
// sparse2dense_tiling.cpp — op_host Tiling 计算 + TilingKey 编码
//
// TilingKey 分发：以 ASCENDC_TPL_ARGS_DECL（op_kernel/arch22/
// sparse2dense_tiling_data.h 内的 TilingKey 声明）声明的
// V_KIND 模板参数为准，Host 侧经 GET_TPL_TILING_KEY
// 编译期编码（ASCENDC_TPL_SEL_PARAM(context, V_KIND) 的底层编码接口——aclsparse Legacy
// API 直调协议无 gert::TilingContext，编码结果经 kernel launch 的 func_key 分发对应
// 模板实例，语义与 GE 流程 SetTilingKey 等价）。禁用废弃宏 TILING_KEY_IS /
// BEGIN_TILING_DATA_DEF。
//
// TilingData 计算口径：
//   zeroTotalBytes =（ROW ? rows : cols）× ld × elemSize   —— matB 全 buffer 含 ld padding
//   zeroTileBytes  = 8192（零块基准；维持设计期预算口径、不做运行时收窄——
//                     CANN 9.1.0 GetCoreMemSize 平台访问器 GetInstance 为 host-only（std::mutex），
//                     __aicore__ 上下文不可调用，说明见 kernel 清零 Init 注释）
//   tileElems      = 1024（NNZ_TILE 估算基准）
//   segsPerBlock   = 64（SEG_BLOCK_SEGS 基准，段块主轴）
//   blockNum       = CSR/CSC: ceil(segNum / segsPerBlock)；COO: ceil(nnz / tileElems)
//   segNum         = CSR: rows / CSC: cols / COO: 0
//   Host 不预扫描 offsets、不下发 per-core/per-block 分片——块→核由 kernel 按
//   GetBlockIdx()/GetBlockNum() grid-stride 动态认领。
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>
#include "sparse2dense_host.h"
// 相对包含路径：kernel.h 位于 op_kernel/arch22/（主线 sparse 库 CMake 不提供该
// include 目录，quoted-include 相对解析，两套构建兼容）
#include "../op_kernel/arch22/sparse2dense_kernel.h"

namespace {

constexpr int64_t S2D_MAX_DIM = 2147483647LL;  // rows/cols/ld 上界

// aclDataType → TilingKey V_KIND（与支持的 dtype 组合一一对应）
inline int32_t ValueKindOfDType(int32_t dtype)
{
    switch (dtype) {
        case ACL_INT8:
            return SPARSE2DENSE_VK_INT8;       // TK0，1B
        case ACL_FLOAT16:
            return SPARSE2DENSE_VK_FLOAT16;    // TK1，2B
        case ACL_BF16:
            return SPARSE2DENSE_VK_BFLOAT16;   // TK2，2B
        case ACL_FLOAT:
            return SPARSE2DENSE_VK_FLOAT32;    // TK3，4B
        case ACL_COMPLEX64:
            return SPARSE2DENSE_VK_COMPLEX64;  // TK4，8B = 2×float32
        default:
            return -1;  // dtype_not_supported → INVALID_VALUE
    }
}

inline bool IsSupportedValueType(int32_t dtype)
{
    return ValueKindOfDType(dtype) >= 0;
}

}  // namespace

// ===========================================================================
// CheckMeta 分段（执行顺序与单一返回语义不变：任一段失败即 INVALID_VALUE）
// ===========================================================================
namespace {

// 段一：空指针/空地址防护（values/secondary/primary 在 len=0 时允许为空——nnz=0 空
// vector 的 data() 为 nullptr；dense 须恒有效）。未声明 alias 的存储区间重叠防护在段三
// 尾部（dtype 验证后取 elemSize 方可计算 dense 存储范围，见
// CheckMetaStorageOverlap——原段一 values==dense 精确相等检查由此升级为区间重叠判定）。
S2DStatus CheckMetaPointers(const struct S2DSpMatView *matA, const struct S2DDnMatView *matB)
{
    if (matB->dense == nullptr) {
        return S2D_STATUS_INVALID_VALUE;
    }
    if (matA->primary == nullptr && matA->lenPrimary > 0) {
        return S2D_STATUS_INVALID_VALUE;
    }
    if (matA->nnz > 0 && (matA->values == nullptr || matA->secondary == nullptr)) {
        return S2D_STATUS_INVALID_VALUE;
    }
    return S2D_STATUS_SUCCESS;
}

// 段二：alg/format/index_base/layout 枚举 + rows/cols/ld 域与 ld>=cols(rows) 约束
S2DStatus CheckMetaEnumsAndShape(const struct S2DSpMatView *matA, const struct S2DDnMatView *matB, int32_t alg)
{
    // alg 仅 ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT
    if (alg != S2D_SPARSETODENSE_ALG_DEFAULT) {
        return S2D_STATUS_INVALID_VALUE;
    }
    // format / index_base / layout 非法枚举 → attribute_value_out_of_range（INVALID_VALUE）
    if (matA->format != S2D_FORMAT_CSR && matA->format != S2D_FORMAT_CSC && matA->format != S2D_FORMAT_COO) {
        return S2D_STATUS_INVALID_VALUE;
    }
    if (matA->indexBase != S2D_INDEX_BASE_ZERO && matA->indexBase != S2D_INDEX_BASE_ONE) {
        return S2D_STATUS_INVALID_VALUE;
    }
    if (matB->layout != S2D_LAYOUT_ROW && matB->layout != S2D_LAYOUT_COL) {
        return S2D_STATUS_INVALID_VALUE;
    }
    // rows/cols/ld 越域 → INVALID_VALUE
    if (matB->rows < 1 || matB->rows > S2D_MAX_DIM || matB->cols < 1 || matB->cols > S2D_MAX_DIM ||
        matB->ld < 1 || matB->ld > S2D_MAX_DIM) {
        return S2D_STATUS_INVALID_VALUE;
    }
    // ROW 布局 ld >= cols；COL 布局 ld >= rows
    if (matB->layout == S2D_LAYOUT_ROW && matB->ld < matB->cols) {
        return S2D_STATUS_INVALID_VALUE;
    }
    if (matB->layout == S2D_LAYOUT_COL && matB->ld < matB->rows) {
        return S2D_STATUS_INVALID_VALUE;
    }
    return S2D_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// 未声明 alias 防护——存储区间重叠判定（第十轮评审整改：由段一 values==dense 精确
// 相等升级，覆盖索引数组与不同起始地址的部分重叠）：
//   dense 范围 = [dense, dense + denseBytes)，denseBytes =（ROW ? rows : cols）×ld×
//   elemSize（与 FillTiling 的 zeroTotalBytes 同口径，含 ld padding）；
//   values = nnz×elemSize；primary = lenPrimary×4；secondary = nnz×4；
//   任一非空区间与 dense 范围相交（完全同址、异址部分重叠）→ INVALID_VALUE。
// 动机：清零 kernel 整体清 dense 后 scatter kernel 读索引与 values——values 同址时
// 读到的恒为 0；索引同址时清零把索引内容清 0（如 COO rowInd[0] 被清 0 → scatter
// 写错坐标），且 matA 只读契约被破坏；必须在任何指纹/清零 kernel 提交前拒绝。
// uintptr_t 比较；长度为 0 的数组（空指针或 0 字节）跳过。denseBytes 溢出（与
// FillTiling 同判条件）时本检查不可计算，跳过——溢出本身由 FillTiling 判溢出返回
// 参数错误，保持既有拒绝阶段语义。
// ---------------------------------------------------------------------------
S2DStatus CheckMetaStorageOverlap(const struct S2DSpMatView *matA, const struct S2DDnMatView *matB)
{
    // dtype 已于段三上方验证（本函数由 CheckMetaLensAndDtype 尾部调用），elemSize 恒可取
    const int32_t valueKind = ValueKindOfDType(matA->valueType);
    const uint64_t elemSize = s2d::ElemSizeOf(static_cast<uint32_t>(valueKind));
    if (elemSize == 0UL) {
        return S2D_STATUS_INVALID_VALUE;  // 防御：不可达（dtype 已验证）
    }
    const uint64_t outerRows = (matB->layout == S2D_LAYOUT_ROW) ? static_cast<uint64_t>(matB->rows)
                                                                : static_cast<uint64_t>(matB->cols);
    const uint64_t ldU = static_cast<uint64_t>(matB->ld);
    if (outerRows == 0UL || ldU == 0UL || outerRows > (~0ULL) / ldU ||
        outerRows * ldU > (~0ULL) / elemSize) {
        return S2D_STATUS_SUCCESS;  // 维度域由段二承接、字节数溢出由 FillTiling 承接（本检查不可计算）
    }
    const uintptr_t denseBegin = reinterpret_cast<uintptr_t>(matB->dense);
    const uint64_t denseBytes = outerRows * ldU * elemSize;
    // 三个输入存储区间；长度为 0（空指针或 0 字节）跳过
    const struct {
        const void *ptr;
        uint64_t len;   // 元素数
        uint64_t size;  // 每元素字节数
    } spans[3] = {
        {matA->values, static_cast<uint64_t>(matA->nnz), elemSize},
        {matA->primary, static_cast<uint64_t>(matA->lenPrimary), sizeof(int32_t)},
        {matA->secondary, static_cast<uint64_t>(matA->nnz), sizeof(int32_t)},
    };
    for (const auto &span : spans) {
        if (span.ptr == nullptr || span.len == 0UL) {
            continue;
        }
        if (span.len > (~0ULL) / span.size) {
            return S2D_STATUS_INVALID_VALUE;  // 区间字节数溢出（元数据本身非法）
        }
        const uintptr_t begin = reinterpret_cast<uintptr_t>(span.ptr);
        const uintptr_t end = begin + span.len * span.size;  // 不回绕（溢出已拒）
        if (begin < denseBegin + denseBytes && denseBegin < end) {
            return S2D_STATUS_INVALID_VALUE;
        }
    }
    return S2D_STATUS_SUCCESS;
}

// 段三：nnz/lenPrimary 域 + len 匹配 + dtype（values/matB 一致，固定 dtype）+ 未声明
// alias 存储区间重叠防护（尾部，dtype 验证之后）
S2DStatus CheckMetaLensAndDtype(const struct S2DSpMatView *matA, const struct S2DDnMatView *matB)
{
    // nnz/lenPrimary 域检查（int64 非负）
    if (matA->nnz < 0 || matA->lenPrimary < 0) {
        return S2D_STATUS_INVALID_VALUE;
    }
    // len 匹配：
    //   len(values) == len(secondary) == nnz（结构上由 nnz 单一来源保证）
    //   CSR: len(primary) == rows + 1；CSC: len(primary) == cols + 1；COO: len(primary) == nnz
    if (matA->format == S2D_FORMAT_CSR && matA->lenPrimary != matB->rows + 1) {
        return S2D_STATUS_INVALID_VALUE;
    }
    if (matA->format == S2D_FORMAT_CSC && matA->lenPrimary != matB->cols + 1) {
        return S2D_STATUS_INVALID_VALUE;
    }
    if (matA->format == S2D_FORMAT_COO && matA->lenPrimary != matA->nnz) {
        return S2D_STATUS_INVALID_VALUE;
    }
    // dtype 校验：values ∈ {int8, float16, bfloat16, float32, complex64}（固定 dtype，无 promotion），
    // matA 与 matB dtype 一致（dense.dtype = values.dtype，无类型转换）；索引固定 I32
    //（由 S2DSpMatView 的 int32_t 指针类型承载，此处不重复校验）
    if (!IsSupportedValueType(matA->valueType) || matA->valueType != matB->valueType) {
        return S2D_STATUS_INVALID_VALUE;
    }
    return CheckMetaStorageOverlap(matA, matB);
}

}  // namespace

extern "C" S2DStatus Sparse2DenseCheckMeta(const struct S2DSpMatView *matA, const struct S2DDnMatView *matB,
    int32_t alg)
{
    if (matA == nullptr || matB == nullptr) {
        return S2D_STATUS_INVALID_VALUE;  // 空描述符 → null_input 语义（统一 INVALID_VALUE）
    }
    // 分段校验（顺序：指针 → 枚举/形状域 → len/dtype/alias 区间重叠；任一段失败即返回）
    S2DStatus st = CheckMetaPointers(matA, matB);
    if (st != S2D_STATUS_SUCCESS) {
        return st;
    }
    st = CheckMetaEnumsAndShape(matA, matB, alg);
    if (st != S2D_STATUS_SUCCESS) {
        return st;
    }
    return CheckMetaLensAndDtype(matA, matB);
}

// 描述符级形状一致性纯比较（公开层两入口的 ValidateShapeConsistency 委托至此；
// 调用方负责日志与状态码映射）。matA 描述符 rows/cols 为 uint64_t
// （aclsparse_descr_internal.h）、matB 为 int64_t：matA 维度超 int64 域先判不等
// （类型安全收敛，杜绝 static_cast 回绕伪相等）；matB 负值与 matA 非负域必不等。
extern "C" S2DStatus Sparse2DenseCheckShapeConsistency(uint64_t sparseRows, uint64_t sparseCols,
    int64_t denseRows, int64_t denseCols)
{
    if (sparseRows > static_cast<uint64_t>(INT64_MAX) || sparseCols > static_cast<uint64_t>(INT64_MAX) ||
        denseRows != static_cast<int64_t>(sparseRows) || denseCols != static_cast<int64_t>(sparseCols)) {
        return S2D_STATUS_INVALID_VALUE;
    }
    return S2D_STATUS_SUCCESS;
}

// ===========================================================================
// scatter kernel 分片（两级轴；Host 不预扫描 offsets，不下发 per-core/per-block 分片）：
// 填 tileElems/segsPerBlock/segNum/blockNum。COO：blockNum = ceil(nnz/tileElems)
//（nnz 为 int64 域，显式校验 nnz <= UINT32_MAX×tileElems，溢出返回参数错误）；
// CSR/CSC：segNum = rows / cols，段块粒度自适应（负载均衡）——固定 64
// 行/列段块在「段数小 + 行分布偏斜（many_empty_rows/power_law）」下会把全部元素压
// 进个别块（单核包揽热点）。目标块数 ≥128（≥2×40 vector cores），segsPerBlock 按下限 8、
// 上限 64 的 2 的幂自适应收缩；纯元数据判定（不读 offsets 内容）→ 同输入同划分，
// 确定性前提不变。
// ===========================================================================
namespace {

S2DStatus FillK2Partition(const struct S2DSpMatView *matA, const struct S2DDnMatView *matB,
    struct Sparse2DenseTilingData *tiling)
{
    if (matA->format == S2D_FORMAT_COO) {
        const uint64_t maxCooBlocks = static_cast<uint64_t>(UINT32_MAX) * static_cast<uint64_t>(tiling->tileElems);
        if (static_cast<uint64_t>(matA->nnz) > maxCooBlocks) {
            return S2D_STATUS_INVALID_VALUE;
        }
        tiling->segNum = 0U;  // COO 无段头
        tiling->blockNum = static_cast<uint32_t>(
            (static_cast<uint64_t>(matA->nnz) + tiling->tileElems - 1UL) / tiling->tileElems);
        return S2D_STATUS_SUCCESS;
    }
    // CSR/CSC：segNum = rows / cols（≤ 2147483647，恰在 uint32 域内，无需额外判溢出）；
    // blockNum = ceil(segNum / segsPerBlock) ≤ segNum，uint32 域内天然无溢出
    const uint32_t segNum = static_cast<uint32_t>(
        (matA->format == S2D_FORMAT_CSR) ? matB->rows : matB->cols);
    tiling->segNum = segNum;
    uint32_t target = segNum / 128U;
    if (target < 8U) {
        target = 8U;
    }
    uint32_t spb = 8U;
    while (spb * 2U <= target && spb < SPARSE2DENSE_SEG_BLOCK_SEGS) {
        spb *= 2U;
    }
    tiling->segsPerBlock = spb;
    tiling->blockNum = (segNum + tiling->segsPerBlock - 1U) / tiling->segsPerBlock;
    return S2D_STATUS_SUCCESS;
}

}  // namespace

extern "C" S2DStatus Sparse2DenseFillTiling(const struct S2DSpMatView *matA, const struct S2DDnMatView *matB,
    struct Sparse2DenseTilingData *tiling, uint64_t *tilingKey)
{
    if (matA == nullptr || matB == nullptr || tiling == nullptr || tilingKey == nullptr) {
        return S2D_STATUS_INVALID_VALUE;
    }

    const int32_t valueKind = ValueKindOfDType(matA->valueType);
    if (valueKind < 0) {
        return S2D_STATUS_INVALID_VALUE;
    }
    const uint64_t elemSize = s2d::ElemSizeOf(static_cast<uint32_t>(valueKind));
    if (elemSize == 0UL) {
        return S2D_STATUS_INVALID_VALUE;
    }

    // ---- TilingKey（ASCENDC_TPL_ARGS_DECL 声明 + GET_TPL_TILING_KEY 编码）----
    // V_KIND ∈ {0..4}（ASCENDC_TPL_UI_LIST 穷举），编码值 = 穷举位置下标。
    *tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(valueKind));
    if (*tilingKey == INVALID_TILING_KEY) {
        return S2D_STATUS_INVALID_VALUE;
    }

    // ---- 形状/协议字段 ----
    tiling->rows = matB->rows;
    tiling->cols = matB->cols;
    tiling->ld = matB->ld;
    tiling->nnz = matA->nnz;
    tiling->lenPrimary = matA->lenPrimary;
    tiling->format = static_cast<uint8_t>(matA->format == S2D_FORMAT_COO
        ? SPARSE2DENSE_FMT_COO
        : (matA->format == S2D_FORMAT_CSC ? SPARSE2DENSE_FMT_CSC : SPARSE2DENSE_FMT_CSR));
    tiling->indexBase = static_cast<uint8_t>(matA->indexBase);
    tiling->isColMajor = (matB->layout == S2D_LAYOUT_COL) ? 1U : 0U;
    tiling->valueKind = static_cast<uint8_t>(valueKind);
    tiling->workspaceSize = 0UL;  // bufferSize 恒 0 协议

    // ---- 清零 kernel 范围（全 buffer 含 ld padding；无符号 64 位中间量 + 显式判溢出：
    //      rows=cols=ld=2147483647 且 elemSize=8 时超 int64 上界返回参数错误）----
    const uint64_t outerRows = (matB->layout == S2D_LAYOUT_ROW) ? static_cast<uint64_t>(matB->rows)
                                                                : static_cast<uint64_t>(matB->cols);
    const uint64_t ldU = static_cast<uint64_t>(matB->ld);
    if (outerRows > (~0ULL) / ldU || outerRows * ldU > (~0ULL) / elemSize) {
        return S2D_STATUS_INVALID_VALUE;  // 清零字节数溢出
    }
    const uint64_t zeroTotalBytes = outerRows * ldU * elemSize;
    if (zeroTotalBytes > static_cast<uint64_t>(INT64_MAX)) {
        return S2D_STATUS_INVALID_VALUE;
    }
    tiling->zeroTotalBytes = static_cast<int64_t>(zeroTotalBytes);
    // 零块基准；维持设计期预算口径（清零 kernel 零块双缓冲 16KB + scatter kernel 全缓冲 ≤72KB
    // （complex64 8B 通道）≤ UB 192KB，余量 >2.5×）；GetCoreMemSize 运行时收窄为已知简化项，
    // 不引入（说明见 sparse2dense_kernel.cpp 清零 kernel Init 注释）
    tiling->zeroTileBytes = SPARSE2DENSE_ZERO_TILE_BYTES;

    // ---- scatter kernel 分片（两级轴；溢出/粒度细节见 FillK2Partition）----
    tiling->tileElems = SPARSE2DENSE_NNZ_TILE;
    tiling->segsPerBlock = SPARSE2DENSE_SEG_BLOCK_SEGS;
    return FillK2Partition(matA, matB, tiling);
}

extern "C" S2DStatus Sparse2DenseGetBufferSize(const struct S2DSpMatView *matA, const struct S2DDnMatView *matB,
    int32_t alg, size_t *bufferSize)
{
    if (bufferSize == nullptr) {
        return S2D_STATUS_INVALID_VALUE;
    }
    // aclsparseSparseToDense_bufferSize：arch22 恒返回 0（与 arch35 *bufferSize = 0 语义一致）；
    // 查询不修改描述符、不记录阶段状态；查询为 0 时执行接口 buffer 可空。
    const S2DStatus st = Sparse2DenseCheckMeta(matA, matB, alg);
    if (st != S2D_STATUS_SUCCESS) {
        return st;
    }
    *bufferSize = 0U;
    return S2D_STATUS_SUCCESS;
}

// ============================================================================
// 数据校验链——发起任何 kernel 之前的 Host 坐标校验。
//
// 坐标越界与 offsets 单调性由 Host 在发起任何 kernel 之前校验：
//   1. 执行接口入口处将 primary/secondary（I32）经 aclrtMemcpyAsync（D2H，handle 绑定
//      stream）拷至 Host 校验缓冲（pinned，规模 (lenPrimary + nnz) × 4B）并同步该 stream；
//   2. Host 校验 offsets 单调不减、锚定语义（offsets[0] == indexBase、offsets[len-1] −
//      indexBase == nnz）、坐标经 base 换算后落在 [0,rows)×[0,cols)（重复坐标合法——
//      结果取任一写入值，与 cuSPARSE 一致，不判重）；
//      失败返回 S2D_STATUS_INVALID_VALUE 且不发起清零/scatter kernel。
// 与两条约束不冲突：① 非 CPU fallback——清零与 scatter 仍全部在 NPU 执行，Host 仅做
// 语义要求的输入校验；② stream 异步语义不变——同步点位于执行接口
// 内部、先于清零/scatter kernel 发起。
//
// nnz=0 守卫：nnz=0 时 values/secondary 为空，
// 上传/比对须按 nnz>0 守卫（空 vector data() 为 nullptr，未守卫的 memcpy 会 Host
// Segmentation fault）。
// ============================================================================

// ---------------------------------------------------------------------------
// primary/secondary D2H（aclrtMemcpyAsync，handle 绑定 stream）+ stream 同步：
// 同步点位于执行接口内部、先于清零/scatter kernel 发起（stream 异步语义不变）。nnz=0 空 vector
// 守卫：跳过空数组的 D2H（见上方注释）。成功返回 true 并给出
// primary/secondary 的 Host 指针。
// ---------------------------------------------------------------------------
namespace {

bool CopyInputsAndSync(void *stream, void *hostBuf, const struct S2DSpMatView *matA, uint64_t primaryBytes,
    uint64_t secondaryBytes, const int32_t *&primary, const int32_t *&secondary)
{
    // primary（CSR rowOffsets / CSC colOffsets / COO rowInd）：lenPrimary > 0 恒成立
    //(CSR=rows+1/CSC=cols+1/COO=nnz 且 rows/cols>=1）；nnz=0 且 COO 时 lenPrimary=0 跳过
    if (primaryBytes > 0UL) {
        if (aclrtMemcpyAsync(hostBuf, primaryBytes, matA->primary, primaryBytes, ACL_MEMCPY_DEVICE_TO_HOST,
                static_cast<aclrtStream>(stream)) != 0) {
            return false;
        }
    }
    // secondary：nnz=0 时为空（跳过空 vector 的 D2H）
    void *secHost = nullptr;
    if (secondaryBytes > 0UL) {
        secHost = static_cast<void *>(static_cast<int8_t *>(hostBuf) + primaryBytes);
        if (aclrtMemcpyAsync(secHost, secondaryBytes, matA->secondary, secondaryBytes, ACL_MEMCPY_DEVICE_TO_HOST,
                static_cast<aclrtStream>(stream)) != 0) {
            return false;
        }
    }
    if (aclrtSynchronizeStream(static_cast<aclrtStream>(stream)) != 0) {
        return false;
    }
    primary = static_cast<const int32_t *>(hostBuf);
    secondary = static_cast<const int32_t *>(secHost);
    return true;
}

}  // namespace

namespace {

// ---------------------------------------------------------------------------
// CSR/CSC 分支校验：offsets 单调不减 + 锚定语义 + 段内坐标展开逐元素越界 + 覆盖
// 元素数核对（检查次序与短路语义拆分前后逐项一致，调用方见
// Sparse2DenseValidateCoordsHost）。
// ---------------------------------------------------------------------------
S2DStatus ValidateSegCoords(const int32_t *primary, const int32_t *secondary, int64_t nnz,
    int64_t lenPrimary, int64_t rows, int64_t cols, int32_t format, int32_t indexBase)
{
    // offsets 单调不减（CSR/CSC offsets 单调不减，违反返回参数错误）
    for (int64_t i = 0; i + 1 < lenPrimary; i++) {
        if (primary[i] > primary[i + 1]) {
            return S2D_STATUS_INVALID_VALUE;
        }
    }
    // 锚定语义（差值展开的前缀基不变量）：offsets[0] == indexBase 且
    // offsets[lenPrimary-1] − indexBase == nnz；违反则 kernel 锚定模型失真，必须拒绝
    if (primary[0] != indexBase || static_cast<int64_t>(primary[lenPrimary - 1]) - indexBase != nnz) {
        return S2D_STATUS_INVALID_VALUE;
    }
    // 段内元素坐标展开逐元素越界检查（规模 O(nnz)）：
    // CSR 段号=行号、secondary=列号；CSC 段号=列号、secondary=行号（语义互换）。
    // 重复坐标合法（结果取任一写入值，与 cuSPARSE 一致），不判重。
    // 段号域：执行链 CheckMeta 已校验 CSR lenPrimary == rows+1 / CSC lenPrimary ==
    // cols+1（段号 s < lenPrimary-1 ≤ 行/列数）；本函数为独立入口，段号域显式复查。
    // 限位对应：CSR 段号=行号（< rows）、secondary=列号（< cols）；CSC 段号=列号
    //（< cols）、secondary=行号（< rows）——非方阵 CSC 行/列限位不可倒置。
    const int64_t segLimit = (format == S2D_FORMAT_CSC) ? cols : rows;
    const int64_t secondaryLimit = (format == S2D_FORMAT_CSC) ? rows : cols;
    int64_t k = 0;
    for (int64_t s = 0; s + 1 < lenPrimary; s++) {
        if (s >= segLimit) {
            return S2D_STATUS_INVALID_VALUE;
        }
        const int64_t segBegin = primary[s] - indexBase;
        const int64_t segEnd = primary[s + 1] - indexBase;
        for (int64_t j = segBegin; j < segEnd; j++) {
            const int64_t sec = static_cast<int64_t>(secondary[j]) - indexBase;
            if (sec < 0 || sec >= secondaryLimit) {
                return S2D_STATUS_INVALID_VALUE;
            }
            k++;
        }
    }
    if (k != nnz) {
        return S2D_STATUS_INVALID_VALUE;  // offsets 覆盖元素数与 nnz 不符
    }
    return S2D_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// COO 分支校验：len(primary) == nnz + 坐标经 base 换算后逐元素越界检查
//（检查次序与短路语义拆分前后逐项一致）。
// ---------------------------------------------------------------------------
S2DStatus ValidateCooCoords(const int32_t *primary, const int32_t *secondary, int64_t nnz,
    int64_t lenPrimary, int64_t rows, int64_t cols, int32_t indexBase)
{
    if (lenPrimary != nnz) {
        return S2D_STATUS_INVALID_VALUE;
    }
    // 坐标经 base 换算后逐元素越界检查；重复坐标合法（结果取任一写入值，与
    // cuSPARSE 一致），不判重
    for (int64_t i = 0; i < nnz; i++) {
        const int64_t r = static_cast<int64_t>(primary[i]) - indexBase;
        const int64_t c = static_cast<int64_t>(secondary[i]) - indexBase;
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            return S2D_STATUS_INVALID_VALUE;
        }
    }
    return S2D_STATUS_SUCCESS;
}

}  // namespace

extern "C" S2DStatus Sparse2DenseValidateCoordsHost(const int32_t *primary, const int32_t *secondary,
    int64_t nnz, int64_t lenPrimary, int64_t rows, int64_t cols, int32_t format, int32_t indexBase)
{
    // 参数域守卫（nnz/lenPrimary 非负、rows/cols >= 1）→ 按 format 分派分支校验
    //（CSR/CSC → COO → 其余非法；分派次序与拆分前一致）。
    if (nnz < 0 || lenPrimary < 0 || rows < 1 || cols < 1) {
        return S2D_STATUS_INVALID_VALUE;
    }
    if (format == S2D_FORMAT_CSR || format == S2D_FORMAT_CSC) {
        return ValidateSegCoords(primary, secondary, nnz, lenPrimary, rows, cols, format, indexBase);
    }
    if (format == S2D_FORMAT_COO) {
        return ValidateCooCoords(primary, secondary, nnz, lenPrimary, rows, cols, indexBase);
    }
    return S2D_STATUS_INVALID_VALUE;
}

extern "C" S2DStatus Sparse2DenseValidateF3(void *stream, const struct S2DSpMatView *matA,
    const struct S2DDnMatView *matB)
{
    if (stream == nullptr || matA == nullptr || matB == nullptr) {
        return S2D_STATUS_INVALID_VALUE;
    }
    // 校验缓冲规模 (lenPrimary + nnz) × 4B，无符号 64 位判溢出
    const uint64_t copyElems = static_cast<uint64_t>(matA->lenPrimary) + static_cast<uint64_t>(matA->nnz);
    const uint64_t copyBytes = copyElems * sizeof(int32_t);
    if (copyElems < static_cast<uint64_t>(matA->lenPrimary) || copyBytes / sizeof(int32_t) != copyElems) {
        return S2D_STATUS_INVALID_VALUE;  // 校验缓冲规模溢出
    }

    // pinned Host 校验缓冲（D2H 带宽优化）
    void *hostBuf = nullptr;
    int32_t ret = 0;
    if (copyBytes > 0UL) {
        ret = aclrtMallocHost(&hostBuf, copyBytes);
        if (ret != 0) {
            return S2D_STATUS_INVALID_VALUE;  // 资源不足（映射 INVALID_VALUE）
        }
    }

    S2DStatus status = S2D_STATUS_INVALID_VALUE;
    do {
        const uint64_t primaryBytes = static_cast<uint64_t>(matA->lenPrimary) * sizeof(int32_t);
        const uint64_t secondaryBytes = static_cast<uint64_t>(matA->nnz) * sizeof(int32_t);
        const int32_t *primary = nullptr;
        const int32_t *secondary = nullptr;
        if (!CopyInputsAndSync(stream, hostBuf, matA, primaryBytes, secondaryBytes, primary, secondary)) {
            break;
        }
        status = Sparse2DenseValidateCoordsHost(
            matA->lenPrimary > 0 ? primary : nullptr,
            matA->nnz > 0 ? secondary : nullptr,
            matA->nnz, matA->lenPrimary, matB->rows, matB->cols, matA->format, matA->indexBase);
    } while (0);

    if (hostBuf != nullptr) {
        (void)aclrtFreeHost(hostBuf);
    }
    return status;
}
