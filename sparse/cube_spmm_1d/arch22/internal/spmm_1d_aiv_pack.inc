/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

__aicore__ __inline__ void read_selected_b_row_aiv(__ubuf__ __fp16* panel, __gm__ __fp16* source, int32_t columns)
{
    copy_gm_to_ubuf_align_b16(panel, source, 0, 1, columns * (int32_t)sizeof(__fp16), 0, 0, 0, 0);
}

__aicore__ __inline__ void pack_b_selected_panels_nbatch_row_major_aiv(int N, int32_t n_base, int32_t n_actual,
                                                                       int32_t vec_base, __gm__ int* row_cols,
                                                                       __gm__ __fp16* B,
                                                                       __gm__ __fp16* B_pack_batch_dst,
                                                                       __ubuf__ __fp16* panel_ub)
{
    /* Pack the selected rows into one fixed-stride workspace panel. */
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

    for (int32_t inner = 0; inner < B_PACK_SUB_TILE; ++inner) {
        int32_t col = row_cols[vec_base + inner];
        __gm__ __fp16* row_src = B + (int64_t)col * (int64_t)N + n_base;

        read_selected_b_row_aiv(panel_ub + (int64_t)inner * B_PACK_BATCH_COLS, row_src, n_actual);
    }

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    spmm_matrix_ubuf2gm(B_pack_batch_dst + (int64_t)vec_base * B_PACK_BATCH_COLS, panel_ub, n_actual, B_PACK_SUB_TILE,
                        B_PACK_BATCH_COLS, B_PACK_BATCH_COLS);

    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

// Return the selected column, or -1 for a tail/invalid row. Both transfers
// use the same predicate, while retaining their separate event boundaries.
__aicore__ __inline__ int32_t selected_panel_column_aiv(int32_t sparse_idx, int32_t num_vectors, int K,
                                                        __gm__ int* row_cols)
{
    if (sparse_idx >= num_vectors) {
        return -1;
    }
    int32_t col = row_cols[sparse_idx];
    return col < 0 || col >= K ? -1 : col;
}

__aicore__ __inline__ void read_valid_b_panel_aiv(int N, int K, int32_t n_base, int32_t n_actual, int32_t vec_base,
                                                  int32_t num_vectors, __gm__ int* row_cols, __gm__ __fp16* B,
                                                  __gm__ __fp16* B_pack_dst, __ubuf__ __fp16* panel_ub)
{
    for (int32_t inner = 0; inner < B_PACK_SUB_TILE; ++inner) {
        int32_t col = selected_panel_column_aiv(vec_base + inner, num_vectors, K, row_cols);
        if (col < 0) {
            continue;
        }

        __gm__ __fp16* row_src = B + (int64_t)col * (int64_t)N + n_base;

        read_selected_b_row_aiv(panel_ub + (int64_t)inner * B_PACK_BATCH_COLS, row_src, n_actual);
    }
}

__aicore__ __inline__ void write_valid_b_panel_aiv(int N, int K, int32_t n_base, int32_t n_actual, int32_t vec_base,
                                                   int32_t num_vectors, __gm__ int* row_cols, __gm__ __fp16* B,
                                                   __gm__ __fp16* B_pack_dst, __ubuf__ __fp16* panel_ub)
{
    for (int32_t inner = 0; inner < B_PACK_SUB_TILE; ++inner) {
        int32_t col = selected_panel_column_aiv(vec_base + inner, num_vectors, K, row_cols);
        if (col < 0) {
            continue;
        }

        copy_ubuf_to_gm_align_b16(B_pack_dst + (int64_t)inner * B_PACK_BATCH_COLS,
                                  panel_ub + (int64_t)inner * B_PACK_BATCH_COLS, 0, 1,
                                  n_actual * (int32_t)sizeof(__fp16), 0, 0, 0, 0);
    }
}

__aicore__ __inline__ void pack_b_selected_panels_row_major_direct_aiv(int N, int K, int32_t n_base, int32_t n_actual,
                                                                       int32_t vec_base, int32_t num_vectors,
                                                                       __gm__ int* row_cols, __gm__ __fp16* B,
                                                                       __gm__ __fp16* B_pack_dst,
                                                                       __ubuf__ __fp16* panel_ub)
{
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

    read_valid_b_panel_aiv(N, K, n_base, n_actual, vec_base, num_vectors, row_cols, B, B_pack_dst, panel_ub);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    write_valid_b_panel_aiv(N, K, n_base, n_actual, vec_base, num_vectors, row_cols, B, B_pack_dst, panel_ub);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

#if SPMM_ENABLE_AIV_PANEL_PIPELINE
/* Two UB panels overlap GM->UB of panel p+1 with UB->GM of panel p. */
template <event_t EventId>
__aicore__ __inline__ void launch_b_panel_read_aiv(int N, int32_t n_base, int32_t n_actual, int32_t vec_base,
                                                   int32_t panel_rows, __gm__ int* row_cols, __gm__ __fp16* B,
                                                   __ubuf__ __fp16* panel_ub)
{
    set_flag(PIPE_V, PIPE_MTE2, EventId);
    wait_flag(PIPE_V, PIPE_MTE2, EventId);

    for (int32_t inner = 0; inner < panel_rows; ++inner) {
        int32_t col = row_cols[vec_base + inner];
        __gm__ __fp16* row_src = B + (int64_t)col * (int64_t)N + n_base;

        read_selected_b_row_aiv(panel_ub + (int64_t)inner * B_PACK_BATCH_COLS, row_src, n_actual);
    }

    set_flag(PIPE_MTE2, PIPE_V, EventId);
}

template <event_t EventId>
__aicore__ __inline__ void wait_b_panel_read_aiv()
{
    wait_flag(PIPE_MTE2, PIPE_V, EventId);
}

template <event_t EventId>
__aicore__ __inline__ void launch_b_panel_write_aiv(int32_t n_actual, int32_t vec_base, int32_t panel_rows,
                                                    __gm__ __fp16* B_pack_batch_dst, __ubuf__ __fp16* panel_ub)
{
    set_flag(PIPE_V, PIPE_MTE3, EventId);
    wait_flag(PIPE_V, PIPE_MTE3, EventId);

    spmm_matrix_ubuf2gm(B_pack_batch_dst + (int64_t)vec_base * B_PACK_BATCH_COLS, panel_ub, n_actual, panel_rows,
                        B_PACK_BATCH_COLS, B_PACK_BATCH_COLS);

    set_flag(PIPE_MTE3, PIPE_V, EventId);
}

template <event_t EventId>
__aicore__ __inline__ void wait_b_panel_write_aiv()
{
    wait_flag(PIPE_MTE3, PIPE_V, EventId);
}

/* The partial tail participates in the same ping-pong pipeline. */
struct AivPanelPipeline {
    int N;
    int32_t n_base, n_actual, num_vectors;
    __gm__ int* row_cols;
    __gm__ __fp16 *B, *B_pack_batch_dst;
    __ubuf__ __fp16 *panel_ub0, *panel_ub1;
    bool write_pending0 = false, write_pending1 = false;
    int32_t panel_count;
    __aicore__ __attribute__((always_inline)) inline void read_next(int32_t panel)
    {
        int32_t next_panel = panel + 1;
        if (next_panel < panel_count) {
            int32_t next_vec_base = next_panel * B_PACK_SUB_TILE;
            int32_t next_rows = num_vectors - next_vec_base;
            if (next_rows > B_PACK_SUB_TILE) {
                next_rows = B_PACK_SUB_TILE;
            }
            bool next_uses_buf0 = ((next_panel & 1) == 0);

            if (next_uses_buf0) {
                if (write_pending0) {
                    wait_b_panel_write_aiv<EVENT_ID0>();
                    write_pending0 = false;
                }
                launch_b_panel_read_aiv<EVENT_ID0>(N, n_base, n_actual, next_vec_base, next_rows, row_cols, B,
                                                   panel_ub0);
            } else {
                if (write_pending1) {
                    wait_b_panel_write_aiv<EVENT_ID1>();
                    write_pending1 = false;
                }
                launch_b_panel_read_aiv<EVENT_ID1>(N, n_base, n_actual, next_vec_base, next_rows, row_cols, B,
                                                   panel_ub1);
            }
        }
    }

    __aicore__ __attribute__((always_inline)) inline void write_current(bool use_buf0, int32_t vec_base,
                                                                        int32_t panel_rows)
    {
        if (use_buf0) {
            launch_b_panel_write_aiv<EVENT_ID0>(n_actual, vec_base, panel_rows, B_pack_batch_dst, panel_ub0);
            write_pending0 = true;
        } else {
            launch_b_panel_write_aiv<EVENT_ID1>(n_actual, vec_base, panel_rows, B_pack_batch_dst, panel_ub1);
            write_pending1 = true;
        }
    }

    __aicore__ __attribute__((always_inline)) inline void run()
    {
        panel_count = (num_vectors + B_PACK_SUB_TILE - 1) / B_PACK_SUB_TILE;
        if (panel_count <= 0) {
            return;
        }
        int32_t first_rows = num_vectors;
        if (first_rows > B_PACK_SUB_TILE) {
            first_rows = B_PACK_SUB_TILE;
        }
        launch_b_panel_read_aiv<EVENT_ID0>(N, n_base, n_actual, 0, first_rows, row_cols, B, panel_ub0);

        for (int32_t panel = 0; panel < panel_count; ++panel) {
            int32_t vec_base = panel * B_PACK_SUB_TILE;
            int32_t panel_rows = num_vectors - vec_base;
            if (panel_rows > B_PACK_SUB_TILE) {
                panel_rows = B_PACK_SUB_TILE;
            }
            bool use_buf0 = ((panel & 1) == 0);

            if (use_buf0) {
                wait_b_panel_read_aiv<EVENT_ID0>();
            } else {
                wait_b_panel_read_aiv<EVENT_ID1>();
            }

            read_next(panel);
            write_current(use_buf0, vec_base, panel_rows);
        }
        if (write_pending0) {
            wait_b_panel_write_aiv<EVENT_ID0>();
        }
        if (write_pending1) {
            wait_b_panel_write_aiv<EVENT_ID1>();
        }
    }
};

__aicore__ __inline__ void pack_b_selected_panels_nbatch_row_major_aiv_pipelined(
    int N, int32_t n_base, int32_t n_actual, int32_t num_vectors, __gm__ int* row_cols, __gm__ __fp16* B,
    __gm__ __fp16* B_pack_batch_dst, __ubuf__ __fp16* panel_ub0, __ubuf__ __fp16* panel_ub1)
{
    AivPanelPipeline pipeline{N, n_base, n_actual, num_vectors, row_cols, B, B_pack_batch_dst, panel_ub0, panel_ub1};
    pipeline.run();
}
#endif // SPMM_ENABLE_AIV_PANEL_PIPELINE
