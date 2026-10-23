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

#ifndef SPMM_KERNEL_TILING_H
#define SPMM_KERNEL_TILING_H

#include "kernel_operator.h"
#include "../spmm_tiling_wire.h"

struct SpmmKernelTilingData {
    // CSR work partitioning and dense tensor address calculation.
    int32_t reorder_offset;
    int32_t bin_edge_offset;
    int32_t row_reorder_identity;
    int32_t uniform_degree;
    int32_t m;
    int32_t n;
    int32_t ldb;
    int32_t ldc;

    // Execution flags and scalar pointer mode.
    int32_t opB;
    int32_t index_base;
    int32_t order_pair;
    int32_t sparse_empty;
    int32_t high_precision;
    int32_t pointer_mode;

    // Real and imaginary scalar components and device pointers.
    float alpha_host;
    float alpha_imag;
    float beta_host;
    float beta_imag;
    int32_t alpha_int;
    int32_t beta_int;
    uint64_t alpha_ptr;
    uint64_t beta_ptr;

    // Cube task geometry and fallback selection.
    int32_t cube_mode;
    int32_t cube_group_count;
    int32_t cube_rows;
    int32_t cube_max_k;
    int32_t cube_tile_n;
    int32_t cube_beta_one;
    int32_t cube_complex_fallback_count;

    // Cube structure, packed inputs, scratch and library workspace.
    uint64_t cube_meta_offset;
    uint64_t cube_row_ids_offset;
    uint64_t cube_slot1_offset;
    uint64_t cube_packed_a_offset;
    uint64_t cube_scratch_offset;
    uint64_t cube_tiling_offset;
    uint64_t cube_lib_workspace_offset;

    // Per-execution packed values refresh.
    uint64_t cube_a_value_map_offset;
    uint64_t cube_a_value_nnz;
    uint64_t cube_complex_a_plane_elements;

    // Complex64 dense component planes and conditional fallback status.
    uint64_t cube_complex_b_offset;
    uint64_t cube_complex_b_plane_elements;
    uint64_t cube_complex_status_offset;
};

constexpr uint32_t SPMM_KERNEL_TILING_VERSION = 1;
static_assert(SPMM_KERNEL_TILING_VERSION == SPMM_TILING_WIRE_VERSION, "unsupported tiling wire version");

#define SPMM_CHECK_KERNEL_FIELD(name) \
    static_assert(std::is_same<decltype(SpmmKernelTilingData::name), \
        decltype(SpmmTilingWireV1::name)>::value, "tiling conversion type changed")

SPMM_CHECK_KERNEL_FIELD(m);
SPMM_CHECK_KERNEL_FIELD(n);
SPMM_CHECK_KERNEL_FIELD(ldb);
SPMM_CHECK_KERNEL_FIELD(ldc);
SPMM_CHECK_KERNEL_FIELD(reorder_offset);
SPMM_CHECK_KERNEL_FIELD(bin_edge_offset);
SPMM_CHECK_KERNEL_FIELD(high_precision);
SPMM_CHECK_KERNEL_FIELD(opB);
SPMM_CHECK_KERNEL_FIELD(index_base);
SPMM_CHECK_KERNEL_FIELD(order_pair);
SPMM_CHECK_KERNEL_FIELD(sparse_empty);
SPMM_CHECK_KERNEL_FIELD(row_reorder_identity);
SPMM_CHECK_KERNEL_FIELD(uniform_degree);
SPMM_CHECK_KERNEL_FIELD(alpha_host);
SPMM_CHECK_KERNEL_FIELD(beta_host);
SPMM_CHECK_KERNEL_FIELD(alpha_imag);
SPMM_CHECK_KERNEL_FIELD(beta_imag);
SPMM_CHECK_KERNEL_FIELD(alpha_int);
SPMM_CHECK_KERNEL_FIELD(beta_int);
SPMM_CHECK_KERNEL_FIELD(alpha_ptr);
SPMM_CHECK_KERNEL_FIELD(beta_ptr);
SPMM_CHECK_KERNEL_FIELD(pointer_mode);
SPMM_CHECK_KERNEL_FIELD(cube_mode);
SPMM_CHECK_KERNEL_FIELD(cube_rows);
SPMM_CHECK_KERNEL_FIELD(cube_max_k);
SPMM_CHECK_KERNEL_FIELD(cube_group_count);
SPMM_CHECK_KERNEL_FIELD(cube_beta_one);
SPMM_CHECK_KERNEL_FIELD(cube_tile_n);
SPMM_CHECK_KERNEL_FIELD(cube_meta_offset);
SPMM_CHECK_KERNEL_FIELD(cube_row_ids_offset);
SPMM_CHECK_KERNEL_FIELD(cube_slot1_offset);
SPMM_CHECK_KERNEL_FIELD(cube_packed_a_offset);
SPMM_CHECK_KERNEL_FIELD(cube_scratch_offset);
SPMM_CHECK_KERNEL_FIELD(cube_tiling_offset);
SPMM_CHECK_KERNEL_FIELD(cube_lib_workspace_offset);
SPMM_CHECK_KERNEL_FIELD(cube_complex_b_offset);
SPMM_CHECK_KERNEL_FIELD(cube_complex_b_plane_elements);
SPMM_CHECK_KERNEL_FIELD(cube_complex_status_offset);
SPMM_CHECK_KERNEL_FIELD(cube_a_value_map_offset);
SPMM_CHECK_KERNEL_FIELD(cube_complex_a_plane_elements);
SPMM_CHECK_KERNEL_FIELD(cube_a_value_nnz);
SPMM_CHECK_KERNEL_FIELD(cube_complex_fallback_count);

#undef SPMM_CHECK_KERNEL_FIELD

namespace {

__aicore__ inline bool IsSpmmTilingVersionSupported(GM_ADDR tilingGM)
{
    const auto wire = reinterpret_cast<__gm__ const SpmmTilingWireV1 *>(tilingGM);
    return wire->version == SPMM_KERNEL_TILING_VERSION;
}

__aicore__ inline SpmmKernelTilingData LoadSpmmKernelTilingData(GM_ADDR tilingGM)
{
    __gm__ const SpmmTilingWireV1 *gmTiling = reinterpret_cast<__gm__ const SpmmTilingWireV1 *>(tilingGM);
    SpmmKernelTilingData td;
    td.m = gmTiling->m;
    td.n = gmTiling->n;
    td.ldb = gmTiling->ldb;
    td.ldc = gmTiling->ldc;
    td.reorder_offset = gmTiling->reorder_offset;
    td.bin_edge_offset = gmTiling->bin_edge_offset;
    td.high_precision = gmTiling->high_precision;
    td.opB = gmTiling->opB;
    td.index_base = gmTiling->index_base;
    td.order_pair = gmTiling->order_pair;
    td.sparse_empty = gmTiling->sparse_empty;
    td.row_reorder_identity = gmTiling->row_reorder_identity;
    td.uniform_degree = gmTiling->uniform_degree;
    td.alpha_host = gmTiling->alpha_host;
    td.beta_host = gmTiling->beta_host;
    td.alpha_imag = gmTiling->alpha_imag;
    td.beta_imag = gmTiling->beta_imag;
    td.alpha_int = gmTiling->alpha_int;
    td.beta_int = gmTiling->beta_int;
    td.alpha_ptr = gmTiling->alpha_ptr;
    td.beta_ptr = gmTiling->beta_ptr;
    td.pointer_mode = gmTiling->pointer_mode;
    td.cube_mode = gmTiling->cube_mode;
    td.cube_rows = gmTiling->cube_rows;
    td.cube_max_k = gmTiling->cube_max_k;
    td.cube_meta_offset = gmTiling->cube_meta_offset;
    td.cube_row_ids_offset = gmTiling->cube_row_ids_offset;
    td.cube_slot1_offset = gmTiling->cube_slot1_offset;
    td.cube_packed_a_offset = gmTiling->cube_packed_a_offset;
    td.cube_complex_b_offset = gmTiling->cube_complex_b_offset;
    td.cube_complex_b_plane_elements = gmTiling->cube_complex_b_plane_elements;
    td.cube_complex_status_offset = gmTiling->cube_complex_status_offset;
    td.cube_a_value_map_offset = gmTiling->cube_a_value_map_offset;
    td.cube_complex_a_plane_elements = gmTiling->cube_complex_a_plane_elements;
    td.cube_a_value_nnz = gmTiling->cube_a_value_nnz;
    td.cube_complex_fallback_count = gmTiling->cube_complex_fallback_count;
    td.cube_scratch_offset = gmTiling->cube_scratch_offset;
    td.cube_tiling_offset = gmTiling->cube_tiling_offset;
    td.cube_lib_workspace_offset = gmTiling->cube_lib_workspace_offset;
    td.cube_group_count = gmTiling->cube_group_count;
    td.cube_beta_one = gmTiling->cube_beta_one;
    td.cube_tile_n = gmTiling->cube_tile_n;
    return td;
}

template<typename CubeTiling>
__aicore__ inline void CopySpmmCubeTiling(CubeTiling &tiling, GM_ADDR tilingGm)
{
    auto dst = reinterpret_cast<uint32_t *>(&tiling);
    auto src = reinterpret_cast<__gm__ const uint32_t *>(tilingGm);
    for (uint32_t i = 0; i < sizeof(CubeTiling) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
}

} // namespace

#endif // SPMM_KERNEL_TILING_H
