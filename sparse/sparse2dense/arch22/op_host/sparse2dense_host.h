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


/*!
 * \file sparse2dense_host.h
 * \brief aclsparse_sparse2dense op_host 层接口（Host 校验 / tiling / bufferSize 协议）。
 *
 * 工程自包含定义：本工作区不含上游 ops-sparse 仓 `include/cann_ops_sparse.h`，
 * 此处以 S2D_* 内部类型展开公开接口所需的最小类型面；合入 ops-sparse（sparse/sparse2dense/
 * arch22/）时按下表映射到上游既有声明（公开签名复用 include/cann_ops_sparse.h L633-679，
 * 不修改公开接口签名）：
 *   S2DStatus              → aclsparseStatus_t（校验失败统一报 ACL_SPARSE_STATUS_INVALID_VALUE）
 *   S2DSparseToDenseAlg    → aclsparseSparseToDenseAlg_t（仅 ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT）
 *   S2DSpMatView           → aclsparseConstSpMatDescr_t（描述符元数据展开）
 *   S2DDnMatView           → aclsparseDnMatDescr_t（描述符元数据展开）
 *   Sparse2DenseGetBufferSize → aclsparseSparseToDense_bufferSize（bufferSize 恒 0 协议）
 *   Sparse2DenseExec       → aclsparseSparseToDense（清零 kernel → scatter kernel 异步执行）
 *
 * Host 分层：
 *   1. CheckMeta    —— 元数据静态校验（空指针/range/format/base/layout/alg/dtype/len 匹配/
 *                      未声明 alias：matB.dense 存储范围与 matA 任一输入存储区间
 *                      （values/primary/secondary，含同址与部分重叠）相交 → 参数错误）
 *   2. ValidateF3   —— 数据校验链：primary/secondary D2H（aclrtMemcpyAsync + 同步）后
 *                      校验 offsets 单调不减、锚定语义、坐标 base 换算后在界内（重复合法）
 *   3. FillTiling   —— TilingData 计算（溢出防护：无符号 64 位中间量 + 显式判溢出）+
 *                      TilingKey 编码（ASCENDC_TPL_ARGS_DECL / GET_TPL_TILING_KEY，
 *                      ASCENDC_TPL_SEL_PARAM 的 GE 上下文变体在 aclsparse 直调协议下
 *                      以其底层编码等价承接，key 经 launch func_key 分发 kernel 实例）
 *   4. GetBufferSize —— aclsparseSparseToDense_bufferSize 语义：恒返回 0
 */

#ifndef SPARSE2DENSE_HOST_H_
#define SPARSE2DENSE_HOST_H_

#include <acl/acl.h>
#include <cstdint>
// 相对包含路径：tiling_data.h 与 kernel 源同目录（op_kernel/arch22/）。主线 sparse 库
// CMake（仓顶 build.sh --ops=sparse2dense）仅提供 ./include、./sparse/common 与 CANN
// 头路径，不含 per-op op_kernel 目录；quoted-include 相对包含文件解析，两套构建兼容。
#include "../op_kernel/arch22/sparse2dense_tiling_data.h"

// ---------------------------------------------------------------------------
// 状态码（统一校验错误 → INVALID_VALUE；合入上游映射 aclsparseStatus_t）
// ---------------------------------------------------------------------------
enum S2DStatus {
    S2D_STATUS_SUCCESS = 0,
    S2D_STATUS_INVALID_VALUE = 1,  // 参数/描述符/坐标/溢出校验失败（对应 ACL_SPARSE_STATUS_INVALID_VALUE）
};

// ---------------------------------------------------------------------------
// 属性枚举（与上游 cann_ops_sparse.h 枚举值同值同序）
// ---------------------------------------------------------------------------
enum S2DSparseFormat {
    S2D_FORMAT_COO = 0,
    S2D_FORMAT_CSR = 1,
    S2D_FORMAT_CSC = 2,
};

enum S2DSparseLayout {
    S2D_LAYOUT_ROW = 0,  // B[r,c] = dense[r*ld + c]
    S2D_LAYOUT_COL = 1,  // B[r,c] = dense[c*ld + r]
};

enum S2DSparseIndexBase {
    S2D_INDEX_BASE_ZERO = 0,
    S2D_INDEX_BASE_ONE = 1,
};

enum S2DSparseToDenseAlg {
    S2D_SPARSETODENSE_ALG_DEFAULT = 0,  // 仅支持该枚举
};

// ---------------------------------------------------------------------------
// 稀疏/稠密矩阵视图（描述符元数据展开；数据指针均为 Device 侧）
// ---------------------------------------------------------------------------
struct S2DSpMatView {
    int32_t format;           // S2DSparseFormat（CSR/CSC/COO；Blocked-ELL 不支持）
    int32_t indexBase;        // S2DSparseIndexBase（0/1）
    int64_t nnz;              // len(values) == len(secondary)
    int64_t lenPrimary;       // len(primary)：CSR=rows+1 / CSC=cols+1 / COO=nnz
    const void *values;       // DEVICE，values 数组（dtype = valueType）
    const int32_t *primary;   // DEVICE，CSR rowOffsets / CSC colOffsets / COO rowInd（固定 I32）
    const int32_t *secondary; // DEVICE，CSR colInd / CSC rowInd / COO colInd（固定 I32）
    int32_t valueType;        // aclDataType（values/dense 共用）
};

struct S2DDnMatView {
    int64_t rows;      // matB 行数 ∈ [1, 2147483647]
    int64_t cols;      // matB 列数 ∈ [1, 2147483647]
    int64_t ld;        // leading dimension（ROW: ld>=cols；COL: ld>=rows）
    int32_t layout;    // S2DSparseLayout
    int32_t valueType; // aclDataType（须与 matA 一致）
    void *dense;       // DEVICE，稠密输出（调用方按 DnMat 布局预分配，含 ld padding）
};

#ifdef __cplusplus
extern "C" {
#endif

// 步骤 1：元数据静态校验（无需 Device 访问；含未声明 alias 防护——matB.dense 存储范围
// （含 ld padding）与 matA 任一输入存储区间（values/primary/secondary，含同址与
// 部分重叠）相交返回 S2D_STATUS_INVALID_VALUE）。
// 失败返回 S2D_STATUS_INVALID_VALUE。
S2DStatus Sparse2DenseCheckMeta(const struct S2DSpMatView *matA, const struct S2DDnMatView *matB, int32_t alg);

// 描述符级形状一致性判定（aclsparseSparseToDense / aclsparseSparseToDense_bufferSize
// 两公开入口共用，对齐上游 arch35 ValidateDenseDescriptorMeta 范式）：matA 与 matB
// 描述符两轴（rows/cols）全等返回 S2D_STATUS_SUCCESS；任一轴不等返回
// S2D_STATUS_INVALID_VALUE。matA 描述符 rows/cols 为 uint64_t、matB 为 int64_t
// （sparse/common/aclsparse_descr_internal.h），比较前类型安全收敛（matA 维度超
// int64 域判不等，无回绕伪相等）。
S2DStatus Sparse2DenseCheckShapeConsistency(uint64_t sparseRows, uint64_t sparseCols,
    int64_t denseRows, int64_t denseCols);

// 步骤 2：数据校验链（发起任何 kernel 之前）——primary/secondary 经
// aclrtMemcpyAsync（D2H，handle 绑定 stream）拷至 Host 校验缓冲并同步，随后校验
// offsets 单调不减、anchor 锚定语义、坐标 base 换算后落在 [0,rows)×[0,cols)（重复合法）。
// nnz=0 时守卫：跳过空 vector 的 D2H/memcpy（values/secondary data 可空）。
S2DStatus Sparse2DenseValidateF3(void *stream, const struct S2DSpMatView *matA, const struct S2DDnMatView *matB);

// 数据校验链记忆化入口：以设备侧 hash 守卫内核
//（sparse2dense_f3_hash_kernel，64-lane 向量化 FNV-1a）对 primary/secondary 全量
// 元素采集内容指纹，按「指针 + 规格 + 指纹 + numBlocks」比对进程内缓存；命中
//（同指针同规格同内容）直接返回缓存的校验结果（免 D2H 与 Host 坐标校验），未命中
//（新输入或同指针原地改写）回退 Sparse2DenseValidateF3 全量链并登记——命中以
// 内容指纹为前提，原地改写索引数组不可能复用陈旧校验结论。
// numBlocks 为指纹内核 launch grid（上限 64 截断），须与缓存键一致以保证折叠确定性。
// onHashEnqueued（poll 形态）：指纹内核 + partials D2H 入队后、任何 event 操作之前
// 触发——调用方在此提交清零 kernel（stream 序 hash → D2H → 清零 → scatter：清零 kernel 先于任何
// event，完成探测以 pinned 盐位自旋替代 event 同步，设备流零空窗）。钩子触发点
// 严格保持在任何 event 操作之前（arch22/CANN 9.1 下 record↔sync 窗口内发射会令
// 流队列停摆）。清零 kernel 不读索引内容、与校验
// 结论无关（合法输入经 CheckMeta alias 防护后 dense 与索引不重叠）；回调内禁止
// 任何同步/等待。采集失败路径不撤回已入队清零 kernel（错误返回时输出内容不在接口契约内）。
// 任何内部资源/启动失败均无感回退全量校验（校验语义不降级）。
using S2DF3OnHashEnqueued = void (*)(void *context);
S2DStatus Sparse2DenseValidateF3Cached(void *stream, uint32_t numBlocks, const struct S2DSpMatView *matA,
    const struct S2DDnMatView *matB, S2DF3OnHashEnqueued onHashEnqueued = nullptr, void *onHashContext = nullptr);

// 步骤 3：TilingData 计算 + TilingKey 编码（前置条件：CheckMeta/ValidateF3 已通过）。
// 溢出防护：zeroTotalBytes 等地址/字节计算用无符号 64 位中间量 + 显式判溢出；
// COO blockNum 显式校验 nnz <= UINT32_MAX * tileElems。失败返回 S2D_STATUS_INVALID_VALUE。
S2DStatus Sparse2DenseFillTiling(const struct S2DSpMatView *matA, const struct S2DDnMatView *matB,
    struct Sparse2DenseTilingData *tiling, uint64_t *tilingKey);

// 步骤 4：aclsparseSparseToDense_bufferSize 语义——arch22 恒返回 0（清零直写 matB、
// 无 Device 临时副本）；查询为 0 时执行接口 buffer 可空。
S2DStatus Sparse2DenseGetBufferSize(const struct S2DSpMatView *matA, const struct S2DDnMatView *matB,
    int32_t alg, size_t *bufferSize);

// 内部纯逻辑校验（数据校验链的 Host 侧核心，供 ValidateF3 与自检入口复用）：
// 输入为已同步到 Host 的 primary/secondary 数组。
S2DStatus Sparse2DenseValidateCoordsHost(const int32_t *primary, const int32_t *secondary,
    int64_t nnz, int64_t lenPrimary, int64_t rows, int64_t cols, int32_t format, int32_t indexBase);

#ifdef __cplusplus
}
#endif

#endif  // SPARSE2DENSE_HOST_H_
