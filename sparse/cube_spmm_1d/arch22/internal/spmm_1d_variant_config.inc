/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

using namespace fp16;
#ifndef SPMM_N_MAJOR
#define SPMM_N_MAJOR 0
#endif

#ifndef SPMM_B_PACK_N_BATCH
#define SPMM_B_PACK_N_BATCH 64
#endif

#ifndef SPMM_AIC_N_BATCH
#define SPMM_AIC_N_BATCH 8
#endif

#ifndef SPMM_B_PACK_PIPE_DEPTH
#define SPMM_B_PACK_PIPE_DEPTH 2
#endif

#ifndef SPMM_B_PACK_SUB_TILE
#define SPMM_B_PACK_SUB_TILE 16
#define SPMM_B_PACK_SUB_TILE_WAS_DEFAULT 1
#endif

#ifndef SPMM_SPARSE_K_TILE
#define SPMM_SPARSE_K_TILE 256
#endif

#ifndef SPMM_ENABLE_AIV_PANEL_PIPELINE
#define SPMM_ENABLE_AIV_PANEL_PIPELINE 0
#endif

#ifndef SPMM_DIRECT_B_GATHER_AIC
#define SPMM_DIRECT_B_GATHER_AIC 0
#endif

#ifndef SPMM_ENABLE_L0C_PIPELINE
#define SPMM_ENABLE_L0C_PIPELINE 0
#endif

static_assert(SPMM_ENABLE_AIV_PANEL_PIPELINE == 0 || SPMM_ENABLE_AIV_PANEL_PIPELINE == 1,
              "SPMM_ENABLE_AIV_PANEL_PIPELINE must be 0 or 1");

static_assert(SPMM_DIRECT_B_GATHER_AIC == 0 || SPMM_DIRECT_B_GATHER_AIC == 1,
              "SPMM_DIRECT_B_GATHER_AIC must be 0 or 1");

static_assert(SPMM_ENABLE_L0C_PIPELINE == 0 || SPMM_ENABLE_L0C_PIPELINE == 1,
              "SPMM_ENABLE_L0C_PIPELINE must be 0 or 1");

static_assert(!(SPMM_ENABLE_AIV_PANEL_PIPELINE && SPMM_DIRECT_B_GATHER_AIC),
              "AIV panel pipeline and direct AIC gather are mutually exclusive");

constexpr int64_t BLOCK_SIZE = 16;
constexpr int64_t VECTOR_LEN = 16;

/* Cube computes C_block^T = B_selected^T * A_block^T. */
constexpr int32_t SPARSE_K_TILE = SPMM_SPARSE_K_TILE;
constexpr int32_t B_PACK_SUB_TILE = SPMM_B_PACK_SUB_TILE;
constexpr int32_t B_PACK_N_BATCH = SPMM_B_PACK_N_BATCH;       // AIV pack super-batch, in 16-column n_blocks
constexpr int32_t AIC_N_BATCH = SPMM_AIC_N_BATCH;             // AIC compute batch, in 16-column n_blocks
constexpr int32_t B_PACK_PIPE_DEPTH = SPMM_B_PACK_PIPE_DEPTH; // GM workspace ring slots
static_assert(B_PACK_N_BATCH > 0, "SPMM_B_PACK_N_BATCH must be positive");
static_assert(AIC_N_BATCH > 0, "SPMM_AIC_N_BATCH must be positive");
static_assert(B_PACK_PIPE_DEPTH > 0 && B_PACK_PIPE_DEPTH <= 8, "SPMM_B_PACK_PIPE_DEPTH must be in [1, 8]");
static_assert(B_PACK_SUB_TILE > 0 && B_PACK_SUB_TILE % BLOCK_SIZE == 0,
              "SPMM_B_PACK_SUB_TILE must be a positive multiple of 16");
static_assert(B_PACK_SUB_TILE <= 32, "SPMM_B_PACK_SUB_TILE must not exceed 32 rows");
static_assert(B_PACK_N_BATCH >= AIC_N_BATCH, "SPMM_B_PACK_N_BATCH must be >= SPMM_AIC_N_BATCH");
static_assert(B_PACK_N_BATCH % AIC_N_BATCH == 0, "SPMM_B_PACK_N_BATCH must be divisible by SPMM_AIC_N_BATCH");
constexpr int32_t B_PACK_BATCH_COLS = B_PACK_N_BATCH * BLOCK_SIZE;
constexpr int32_t AIC_BATCH_COLS = AIC_N_BATCH * BLOCK_SIZE;
constexpr int32_t L0A_CAPACITY_HALF_ELEMS = 64 * 1024 / sizeof(__fp16);
constexpr int32_t L0B_CAPACITY_HALF_ELEMS = 64 * 1024 / sizeof(__fp16);
constexpr int32_t L1_A_REGION_HALF_ELEMS = 64 * 1024 / sizeof(__fp16);
constexpr int32_t L0C_CAPACITY_FLOAT_ELEMS = 64 * 1024 / sizeof(float);
constexpr int32_t L0C_TILE_FLOAT_ELEMS = AIC_BATCH_COLS * VECTOR_LEN;
// Limit the AIV double-panel workspace to 128 KiB.
constexpr int32_t AIV_DOUBLE_PANEL_MAX_FP16_ELEMS = 64 * 1024;
static_assert(!SPMM_ENABLE_AIV_PANEL_PIPELINE ||
                  2 * B_PACK_SUB_TILE * B_PACK_BATCH_COLS <= AIV_DOUBLE_PANEL_MAX_FP16_ELEMS,
              "AIV panel double buffer exceeds 128 KiB");
static_assert(SPARSE_K_TILE > 0 && SPARSE_K_TILE % BLOCK_SIZE == 0,
              "SPMM_SPARSE_K_TILE must be a positive multiple of 16");
static_assert(AIC_BATCH_COLS * SPARSE_K_TILE <= L0A_CAPACITY_HALF_ELEMS,
              "AIC N batch x sparse K tile exceeds the 64KB L0A capacity");
static_assert(AIC_BATCH_COLS * SPARSE_K_TILE <= L1_A_REGION_HALF_ELEMS,
              "AIC N batch x sparse K tile overlaps the L1_B partition");
static_assert(!SPMM_ENABLE_L0C_PIPELINE || 2 * L0C_TILE_FLOAT_ELEMS <= L0C_CAPACITY_FLOAT_ELEMS,
              "two output tiles exceed the 64KB L0C capacity");
constexpr int32_t SYNC_B_PACK_READY_BASE = 0;
constexpr int32_t SYNC_B_PACK_ACK_BASE = 8;

__aicore__ __inline__ int32_t ceil_div_16_i32(int32_t x) { return (x + 15) / 16; }

__aicore__ __inline__ int32_t b_pack_ready_flag(int64_t task_seq)
{
    return SYNC_B_PACK_READY_BASE + (int32_t)(task_seq % B_PACK_PIPE_DEPTH);
}

__aicore__ __inline__ int32_t b_pack_ack_flag(int64_t task_seq)
{
    return SYNC_B_PACK_ACK_BASE + (int32_t)(task_seq % B_PACK_PIPE_DEPTH);
}

#if !SPMM_DIRECT_B_GATHER_AIC
__aicore__ __inline__ void copy_b_pack_row_major_to_l1(__cbuf__ __fp16* L1_A, __gm__ __fp16* B_pack_tile,
                                                       int32_t source_stride, int32_t k_tile_round,
                                                       int32_t k_tile_actual, int32_t n_actual)
{
    int32_t n_round = ceil_div_16_i32(n_actual) * 16;

    /* The workspace stride represents B_selected^T without an AIV transpose. */
    spmm_matrix_gm2cbuf_ND2nZ(L1_A, B_pack_tile, n_round, k_tile_round, n_actual, k_tile_actual, source_stride);
}
#endif

#if SPMM_DIRECT_B_GATHER_AIC
/* Direct sparse B gather for very light rows. */
__aicore__ __inline__ void copy_b_selected_direct_to_l1(__cbuf__ __fp16* L1_A, __gm__ __fp16* B, __gm__ int* row_cols,
                                                        int32_t vec_base, int32_t K, int32_t N, int32_t n_base,
                                                        int32_t k_tile_round, int32_t k_tile_actual, int32_t n_actual)
{
    for (int32_t i = 0; i < k_tile_actual; ++i) {
        int32_t col = row_cols[vec_base + i];
        if (col < 0 || col >= K) {
            continue;
        }

        __gm__ __fp16* row_src = B + (int64_t)col * (int64_t)N + (int64_t)n_base;

        copy_gm_to_cbuf_multi_nd2nz_b16(L1_A + (int64_t)i * CUBE_K0, row_src, static_cast<uint8_t>(0),
                                        static_cast<uint16_t>(1), static_cast<uint16_t>(1),
                                        static_cast<uint16_t>(n_actual), static_cast<uint16_t>(0),
                                        static_cast<uint16_t>(0), static_cast<uint16_t>(k_tile_round),
                                        static_cast<uint16_t>(0), static_cast<uint16_t>(0));
    }
}
#endif
