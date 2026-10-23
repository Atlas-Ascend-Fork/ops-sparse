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

#ifndef SPMM_HOST_TILING_H
#define SPMM_HOST_TILING_H

#include "../spmm_tiling_wire.h"

struct SpmmHostTilingData {
    // Logical matrix dimensions and dense leading dimensions.
    int32_t m;
    int32_t n;
    int32_t ldb;
    int32_t ldc;

    // Host scalar values used to construct the transfer record.
    float alpha_host;
    float beta_host;
    float alpha_imag;
    float beta_imag;
    int32_t alpha_int;
    int32_t beta_int;

    // Scalar pointer mode and computation precision.
    int32_t pointer_mode;
    uint64_t alpha_ptr;
    uint64_t beta_ptr;
    int32_t high_precision;

    // Operation, storage layout and CSR preprocessing plan.
    int32_t opB;
    int32_t order_pair;
    int32_t index_base;
    int32_t sparse_empty;
    int32_t row_reorder_identity;
    int32_t uniform_degree;
    int32_t reorder_offset;
    int32_t bin_edge_offset;

    // Selected Cube geometry and fallback row count.
    int32_t cube_mode;
    int32_t cube_rows;
    int32_t cube_max_k;
    int32_t cube_group_count;
    int32_t cube_complex_fallback_count;
    int32_t cube_beta_one;
    int32_t cube_tile_n;

    // Packed values, structure mapping and auxiliary workspace regions.
    uint64_t cube_packed_a_offset;
    uint64_t cube_a_value_map_offset;
    uint64_t cube_a_value_nnz;
    uint64_t cube_meta_offset;
    uint64_t cube_row_ids_offset;
    uint64_t cube_slot1_offset;
    uint64_t cube_tiling_offset;
    uint64_t cube_lib_workspace_offset;
    uint64_t cube_scratch_offset;

    // Complex64 component planes and conditional fallback status.
    uint64_t cube_complex_b_offset;
    uint64_t cube_complex_b_plane_elements;
    uint64_t cube_complex_status_offset;
    uint64_t cube_complex_a_plane_elements;
};

constexpr uint32_t SPMM_HOST_TILING_VERSION = 1;
static_assert(SPMM_HOST_TILING_VERSION == SPMM_TILING_WIRE_VERSION, "unsupported tiling wire version");

#define SPMM_CHECK_HOST_FIELD(name) \
    static_assert(std::is_same<decltype(SpmmHostTilingData::name), \
        decltype(SpmmTilingWireV1::name)>::value, "tiling conversion type changed")

SPMM_CHECK_HOST_FIELD(m);
SPMM_CHECK_HOST_FIELD(n);
SPMM_CHECK_HOST_FIELD(ldb);
SPMM_CHECK_HOST_FIELD(ldc);
SPMM_CHECK_HOST_FIELD(reorder_offset);
SPMM_CHECK_HOST_FIELD(bin_edge_offset);
SPMM_CHECK_HOST_FIELD(high_precision);
SPMM_CHECK_HOST_FIELD(opB);
SPMM_CHECK_HOST_FIELD(index_base);
SPMM_CHECK_HOST_FIELD(order_pair);
SPMM_CHECK_HOST_FIELD(sparse_empty);
SPMM_CHECK_HOST_FIELD(row_reorder_identity);
SPMM_CHECK_HOST_FIELD(uniform_degree);
SPMM_CHECK_HOST_FIELD(alpha_host);
SPMM_CHECK_HOST_FIELD(beta_host);
SPMM_CHECK_HOST_FIELD(alpha_imag);
SPMM_CHECK_HOST_FIELD(beta_imag);
SPMM_CHECK_HOST_FIELD(alpha_int);
SPMM_CHECK_HOST_FIELD(beta_int);
SPMM_CHECK_HOST_FIELD(alpha_ptr);
SPMM_CHECK_HOST_FIELD(beta_ptr);
SPMM_CHECK_HOST_FIELD(pointer_mode);
SPMM_CHECK_HOST_FIELD(cube_mode);
SPMM_CHECK_HOST_FIELD(cube_rows);
SPMM_CHECK_HOST_FIELD(cube_max_k);
SPMM_CHECK_HOST_FIELD(cube_group_count);
SPMM_CHECK_HOST_FIELD(cube_beta_one);
SPMM_CHECK_HOST_FIELD(cube_tile_n);
SPMM_CHECK_HOST_FIELD(cube_meta_offset);
SPMM_CHECK_HOST_FIELD(cube_row_ids_offset);
SPMM_CHECK_HOST_FIELD(cube_slot1_offset);
SPMM_CHECK_HOST_FIELD(cube_packed_a_offset);
SPMM_CHECK_HOST_FIELD(cube_scratch_offset);
SPMM_CHECK_HOST_FIELD(cube_tiling_offset);
SPMM_CHECK_HOST_FIELD(cube_lib_workspace_offset);
SPMM_CHECK_HOST_FIELD(cube_complex_b_offset);
SPMM_CHECK_HOST_FIELD(cube_complex_b_plane_elements);
SPMM_CHECK_HOST_FIELD(cube_complex_status_offset);
SPMM_CHECK_HOST_FIELD(cube_a_value_map_offset);
SPMM_CHECK_HOST_FIELD(cube_complex_a_plane_elements);
SPMM_CHECK_HOST_FIELD(cube_a_value_nnz);
SPMM_CHECK_HOST_FIELD(cube_complex_fallback_count);

#undef SPMM_CHECK_HOST_FIELD

inline SpmmTilingWireV1 SerializeSpmmTiling(const SpmmHostTilingData &source)
{
    SpmmTilingWireV1 wire{};
    wire.version = SPMM_HOST_TILING_VERSION;
    wire.m = source.m;
    wire.n = source.n;
    wire.ldb = source.ldb;
    wire.ldc = source.ldc;
    wire.reorder_offset = source.reorder_offset;
    wire.bin_edge_offset = source.bin_edge_offset;
    wire.high_precision = source.high_precision;
    wire.opB = source.opB;
    wire.index_base = source.index_base;
    wire.order_pair = source.order_pair;
    wire.sparse_empty = source.sparse_empty;
    wire.row_reorder_identity = source.row_reorder_identity;
    wire.uniform_degree = source.uniform_degree;
    wire.alpha_host = source.alpha_host;
    wire.beta_host = source.beta_host;
    wire.alpha_imag = source.alpha_imag;
    wire.beta_imag = source.beta_imag;
    wire.alpha_int = source.alpha_int;
    wire.beta_int = source.beta_int;
    wire.alpha_ptr = source.alpha_ptr;
    wire.beta_ptr = source.beta_ptr;
    wire.pointer_mode = source.pointer_mode;
    wire.cube_mode = source.cube_mode;
    wire.cube_rows = source.cube_rows;
    wire.cube_max_k = source.cube_max_k;
    wire.cube_group_count = source.cube_group_count;
    wire.cube_beta_one = source.cube_beta_one;
    wire.cube_tile_n = source.cube_tile_n;
    wire.cube_meta_offset = source.cube_meta_offset;
    wire.cube_row_ids_offset = source.cube_row_ids_offset;
    wire.cube_slot1_offset = source.cube_slot1_offset;
    wire.cube_packed_a_offset = source.cube_packed_a_offset;
    wire.cube_scratch_offset = source.cube_scratch_offset;
    wire.cube_tiling_offset = source.cube_tiling_offset;
    wire.cube_lib_workspace_offset = source.cube_lib_workspace_offset;
    wire.cube_complex_b_offset = source.cube_complex_b_offset;
    wire.cube_complex_b_plane_elements = source.cube_complex_b_plane_elements;
    wire.cube_complex_status_offset = source.cube_complex_status_offset;
    wire.cube_a_value_map_offset = source.cube_a_value_map_offset;
    wire.cube_complex_a_plane_elements = source.cube_complex_a_plane_elements;
    wire.cube_a_value_nnz = source.cube_a_value_nnz;
    wire.cube_complex_fallback_count = source.cube_complex_fallback_count;
    return wire;
}

#endif // SPMM_HOST_TILING_H
