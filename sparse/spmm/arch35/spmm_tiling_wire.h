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

#ifndef SPMM_TILING_WIRE_H
#define SPMM_TILING_WIRE_H

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

// V1 preserves the existing workspace layout; the version uses former padding.
constexpr uint32_t SPMM_TILING_WIRE_VERSION = 1;

struct SpmmTilingWireV1 {
    int32_t m;
    int32_t n;
    int32_t ldb;
    int32_t ldc;
    int32_t reorder_offset;     // byte offset to reorder table in workspace
    int32_t bin_edge_offset;    // byte offset to row-bin edge table in workspace
    int32_t high_precision;     // 0=standard, 1=fp32 Kahan (CSR_FP32_HIGH_PRECISION_ALG)
    int32_t opB;                // 0 = N, 1 = T, 2 = conjugate transpose
    int32_t index_base;         // CSR index base: 0 or 1
    int32_t order_pair;         // SPMM_ORDER_RR / RC / CR / CC
    int32_t sparse_empty;       // nnz == 0: skip CSR/value/B reads
    int32_t row_reorder_identity;
    int32_t uniform_degree;     // exact row degree from preprocess, or -1
    float alpha_host;
    float beta_host;
    float alpha_imag;
    float beta_imag;
    int32_t alpha_int;
    int32_t beta_int;
    uint32_t version;
    uint64_t alpha_ptr;
    uint64_t beta_ptr;
    int32_t pointer_mode;       // 0=HOST, 1=DEVICE
    int32_t cube_mode;
    int32_t cube_rows;
    int32_t cube_max_k;
    int32_t cube_group_count;
    int32_t cube_beta_one;
    int32_t cube_tile_n;
    uint64_t cube_meta_offset;
    uint64_t cube_row_ids_offset;
    uint64_t cube_slot1_offset;
    uint64_t cube_packed_a_offset;
    uint64_t cube_scratch_offset;
    uint64_t cube_tiling_offset;
    uint64_t cube_lib_workspace_offset;
    // Complex64 FP16 high/low Cube path: six A planes are rooted at
    // cube_packed_a_offset, four dense-B planes at this offset, and two
    // FP32 component scratch planes are rooted at cube_scratch_offset.
    uint64_t cube_complex_b_offset;
    uint64_t cube_complex_b_plane_elements;
    uint64_t cube_complex_status_offset;
    // CSR-to-packed positions refresh real and Complex64 A values at execution.
    uint64_t cube_a_value_map_offset;
    uint64_t cube_complex_a_plane_elements;
    uint64_t cube_a_value_nnz;
    // Number of rows handled by the per-row fallback for a mixed Cube path.
    int32_t cube_complex_fallback_count;
};

typedef struct SpmmCubeGroupMeta {
    int32_t col_base;
    int32_t packed_k;
} SpmmCubeGroupMeta;

// FP32 clustered RegBase metadata. The path is selected only for large,
// near-uniform CSR inputs whose consecutive-column rows form dense windows.
// Values are refreshed at execution time; preprocess stores structure only.
typedef struct SpmmFp32RegBaseBatchMeta {
    int32_t window_begin;
    int32_t row_count;
    int32_t row_id_begin;
    int32_t window_rows;
} SpmmFp32RegBaseBatchMeta;

typedef struct SpmmFp32RegBaseRefreshRun {
    int32_t value_begin;
    int32_t row_count;
    int32_t degree;
    int32_t packed_begin;
    int32_t packed_stride;
} SpmmFp32RegBaseRefreshRun;

typedef struct SpmmFp32RegBaseHeader {
    int32_t batch_count;
    // Zero selects the per-candidate SIMT refresh for irregular source order.
    int32_t refresh_run_count;
    int32_t fallback_count;
    int32_t candidate_count;
} SpmmFp32RegBaseHeader;

typedef struct SpmmFp32Degree2BatchMeta {
    int32_t window_begin;
    int32_t row_count;
    int32_t row_meta_begin;
    int32_t window_rows;
} SpmmFp32Degree2BatchMeta;

typedef struct SpmmFp32Degree2RowMeta {
    int32_t local_start;
    int32_t value_offset;
} SpmmFp32Degree2RowMeta;

static_assert(std::is_standard_layout<SpmmTilingWireV1>::value, "tiling wire must have a fixed layout");
static_assert(std::is_trivially_copyable<SpmmTilingWireV1>::value, "tiling wire must be transferable");
static_assert(sizeof(SpmmTilingWireV1) == 240, "V1 tiling size changed");
static_assert(alignof(SpmmTilingWireV1) == 8, "V1 tiling alignment changed");

#define SPMM_CHECK_WIRE_FIELD(name, type, offset) \
    static_assert(std::is_same<decltype(SpmmTilingWireV1::name), type>::value, "V1 field type changed"); \
    static_assert(offsetof(SpmmTilingWireV1, name) == offset, "V1 field offset changed")

SPMM_CHECK_WIRE_FIELD(m, int32_t, 0);
SPMM_CHECK_WIRE_FIELD(n, int32_t, 4);
SPMM_CHECK_WIRE_FIELD(ldb, int32_t, 8);
SPMM_CHECK_WIRE_FIELD(ldc, int32_t, 12);
SPMM_CHECK_WIRE_FIELD(reorder_offset, int32_t, 16);
SPMM_CHECK_WIRE_FIELD(bin_edge_offset, int32_t, 20);
SPMM_CHECK_WIRE_FIELD(high_precision, int32_t, 24);
SPMM_CHECK_WIRE_FIELD(opB, int32_t, 28);
SPMM_CHECK_WIRE_FIELD(index_base, int32_t, 32);
SPMM_CHECK_WIRE_FIELD(order_pair, int32_t, 36);
SPMM_CHECK_WIRE_FIELD(sparse_empty, int32_t, 40);
SPMM_CHECK_WIRE_FIELD(row_reorder_identity, int32_t, 44);
SPMM_CHECK_WIRE_FIELD(uniform_degree, int32_t, 48);
SPMM_CHECK_WIRE_FIELD(alpha_host, float, 52);
SPMM_CHECK_WIRE_FIELD(beta_host, float, 56);
SPMM_CHECK_WIRE_FIELD(alpha_imag, float, 60);
SPMM_CHECK_WIRE_FIELD(beta_imag, float, 64);
SPMM_CHECK_WIRE_FIELD(alpha_int, int32_t, 68);
SPMM_CHECK_WIRE_FIELD(beta_int, int32_t, 72);
SPMM_CHECK_WIRE_FIELD(version, uint32_t, 76);
SPMM_CHECK_WIRE_FIELD(alpha_ptr, uint64_t, 80);
SPMM_CHECK_WIRE_FIELD(beta_ptr, uint64_t, 88);
SPMM_CHECK_WIRE_FIELD(pointer_mode, int32_t, 96);
SPMM_CHECK_WIRE_FIELD(cube_mode, int32_t, 100);
SPMM_CHECK_WIRE_FIELD(cube_rows, int32_t, 104);
SPMM_CHECK_WIRE_FIELD(cube_max_k, int32_t, 108);
SPMM_CHECK_WIRE_FIELD(cube_group_count, int32_t, 112);
SPMM_CHECK_WIRE_FIELD(cube_beta_one, int32_t, 116);
SPMM_CHECK_WIRE_FIELD(cube_tile_n, int32_t, 120);
SPMM_CHECK_WIRE_FIELD(cube_meta_offset, uint64_t, 128);
SPMM_CHECK_WIRE_FIELD(cube_row_ids_offset, uint64_t, 136);
SPMM_CHECK_WIRE_FIELD(cube_slot1_offset, uint64_t, 144);
SPMM_CHECK_WIRE_FIELD(cube_packed_a_offset, uint64_t, 152);
SPMM_CHECK_WIRE_FIELD(cube_scratch_offset, uint64_t, 160);
SPMM_CHECK_WIRE_FIELD(cube_tiling_offset, uint64_t, 168);
SPMM_CHECK_WIRE_FIELD(cube_lib_workspace_offset, uint64_t, 176);
SPMM_CHECK_WIRE_FIELD(cube_complex_b_offset, uint64_t, 184);
SPMM_CHECK_WIRE_FIELD(cube_complex_b_plane_elements, uint64_t, 192);
SPMM_CHECK_WIRE_FIELD(cube_complex_status_offset, uint64_t, 200);
SPMM_CHECK_WIRE_FIELD(cube_a_value_map_offset, uint64_t, 208);
SPMM_CHECK_WIRE_FIELD(cube_complex_a_plane_elements, uint64_t, 216);
SPMM_CHECK_WIRE_FIELD(cube_a_value_nnz, uint64_t, 224);
SPMM_CHECK_WIRE_FIELD(cube_complex_fallback_count, int32_t, 232);

#undef SPMM_CHECK_WIRE_FIELD

#define SPMM_CHECK_META_FIELD(layout, name, offset) \
    static_assert(std::is_same<decltype(layout::name), int32_t>::value, "metadata field type changed"); \
    static_assert(offsetof(layout, name) == offset, "metadata field offset changed")

static_assert(sizeof(SpmmCubeGroupMeta) == 8, "metadata size changed");
static_assert(alignof(SpmmCubeGroupMeta) == 4, "metadata alignment changed");
SPMM_CHECK_META_FIELD(SpmmCubeGroupMeta, col_base, 0);
SPMM_CHECK_META_FIELD(SpmmCubeGroupMeta, packed_k, 4);

static_assert(sizeof(SpmmFp32RegBaseBatchMeta) == 16, "metadata size changed");
static_assert(alignof(SpmmFp32RegBaseBatchMeta) == 4, "metadata alignment changed");
SPMM_CHECK_META_FIELD(SpmmFp32RegBaseBatchMeta, window_begin, 0);
SPMM_CHECK_META_FIELD(SpmmFp32RegBaseBatchMeta, row_count, 4);
SPMM_CHECK_META_FIELD(SpmmFp32RegBaseBatchMeta, row_id_begin, 8);
SPMM_CHECK_META_FIELD(SpmmFp32RegBaseBatchMeta, window_rows, 12);

static_assert(sizeof(SpmmFp32RegBaseRefreshRun) == 20, "metadata size changed");
static_assert(alignof(SpmmFp32RegBaseRefreshRun) == 4, "metadata alignment changed");
SPMM_CHECK_META_FIELD(SpmmFp32RegBaseRefreshRun, value_begin, 0);
SPMM_CHECK_META_FIELD(SpmmFp32RegBaseRefreshRun, row_count, 4);
SPMM_CHECK_META_FIELD(SpmmFp32RegBaseRefreshRun, degree, 8);
SPMM_CHECK_META_FIELD(SpmmFp32RegBaseRefreshRun, packed_begin, 12);
SPMM_CHECK_META_FIELD(SpmmFp32RegBaseRefreshRun, packed_stride, 16);

static_assert(sizeof(SpmmFp32RegBaseHeader) == 16, "metadata size changed");
static_assert(alignof(SpmmFp32RegBaseHeader) == 4, "metadata alignment changed");
SPMM_CHECK_META_FIELD(SpmmFp32RegBaseHeader, batch_count, 0);
SPMM_CHECK_META_FIELD(SpmmFp32RegBaseHeader, refresh_run_count, 4);
SPMM_CHECK_META_FIELD(SpmmFp32RegBaseHeader, fallback_count, 8);
SPMM_CHECK_META_FIELD(SpmmFp32RegBaseHeader, candidate_count, 12);

static_assert(sizeof(SpmmFp32Degree2BatchMeta) == 16, "metadata size changed");
static_assert(alignof(SpmmFp32Degree2BatchMeta) == 4, "metadata alignment changed");
SPMM_CHECK_META_FIELD(SpmmFp32Degree2BatchMeta, window_begin, 0);
SPMM_CHECK_META_FIELD(SpmmFp32Degree2BatchMeta, row_count, 4);
SPMM_CHECK_META_FIELD(SpmmFp32Degree2BatchMeta, row_meta_begin, 8);
SPMM_CHECK_META_FIELD(SpmmFp32Degree2BatchMeta, window_rows, 12);

static_assert(sizeof(SpmmFp32Degree2RowMeta) == 8, "metadata size changed");
static_assert(alignof(SpmmFp32Degree2RowMeta) == 4, "metadata alignment changed");
SPMM_CHECK_META_FIELD(SpmmFp32Degree2RowMeta, local_start, 0);
SPMM_CHECK_META_FIELD(SpmmFp32Degree2RowMeta, value_offset, 4);

#undef SPMM_CHECK_META_FIELD

#endif // SPMM_TILING_WIRE_H
