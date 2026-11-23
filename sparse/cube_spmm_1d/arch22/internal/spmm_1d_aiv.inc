/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

struct AivState : SpmmKernelInput {
    __ubuf__ __fp16* panel_ub0;
    __ubuf__ __fp16* panel_ub1;
    __ubuf__ __fp16* panel_ub;
    int32_t core_id;
    int32_t core_cnt;
    int32_t num_row_blocks;
    int32_t num_n_blocks_i32;
    int32_t num_n_batches_i32;
    int64_t b_pack_core_stride_elems;
    __gm__ __fp16* core_B_pack;
    int64_t pack_task_seq;
    int32_t row_block;
    int32_t start_vec;
    int32_t end_vec;
    int32_t num_vectors;
    __gm__ int* row_cols;
    int32_t n_batch_idx;
    int32_t n_block_begin;
    int32_t n_base;
    int32_t n_batch_actual;
    int32_t pipe_slot;
    __gm__ __fp16* B_pack_batch_dst;
    __aicore__ __attribute__((always_inline)) inline void initialize()
    {
        (void)values;
        (void)C;
        (void)max_vectors_per_row;
        (void)ldc;

        set_ffts_base_addr((uint64_t)ffts_addr);
        set_vector_mask(-1, -1);
        set_mask_norm();

        panel_ub0 = (__ubuf__ __fp16*)(uintptr_t)0;
        panel_ub1 = panel_ub0 + (int64_t)B_PACK_SUB_TILE * B_PACK_BATCH_COLS;
        panel_ub = panel_ub0;

        core_id = get_block_idx();
        core_cnt = get_block_num();
        num_row_blocks = (int32_t)(((int64_t)M + vec_length - 1) / vec_length);
        num_n_blocks_i32 = (int32_t)num_n_blocks;
        num_n_batches_i32 = (num_n_blocks_i32 + B_PACK_N_BATCH - 1) / B_PACK_N_BATCH;
        b_pack_core_stride_elems = (int64_t)B_PACK_PIPE_DEPTH * b_pack_slot_stride_elems;
        core_B_pack = workspace1 + (int64_t)core_id * b_pack_core_stride_elems;
        pack_task_seq = 0;
    }

#if !SPMM_DIRECT_B_GATHER_AIC
    __aicore__ __attribute__((always_inline)) inline void pack_panels()
    {
        bool fast_grouped_store = ((n_batch_actual % BLOCK_SIZE) == 0);

#if SPMM_ENABLE_AIV_PANEL_PIPELINE
        if (fast_grouped_store) {
            pack_b_selected_panels_nbatch_row_major_aiv_pipelined(N, n_base, n_batch_actual, num_vectors, row_cols, B,
                                                                  B_pack_batch_dst, panel_ub0, panel_ub1);
        } else
#endif
        {
            for (int32_t vec_base = 0; vec_base < num_vectors; vec_base += B_PACK_SUB_TILE) {
                bool full_sparse_tile = fast_grouped_store && (vec_base + B_PACK_SUB_TILE <= num_vectors);

                if (full_sparse_tile) {
                    pack_b_selected_panels_nbatch_row_major_aiv(N, n_base, n_batch_actual, vec_base, row_cols, B,
                                                                B_pack_batch_dst, panel_ub);
                } else {
                    pack_b_selected_panels_row_major_direct_aiv(
                        N, K, n_base, n_batch_actual, vec_base, num_vectors, row_cols, B,
                        B_pack_batch_dst + (int64_t)vec_base * B_PACK_BATCH_COLS, panel_ub);
                }
            }
        }
    }

    __aicore__ __attribute__((always_inline)) inline void pack_batch()
    {
        if (pack_task_seq >= B_PACK_PIPE_DEPTH) {
            wait_flag_dev(b_pack_ack_flag(pack_task_seq - B_PACK_PIPE_DEPTH));
        }

        n_block_begin = n_batch_idx * B_PACK_N_BATCH;
        n_base = n_block_begin * BLOCK_SIZE;

        n_batch_actual = N - n_base;
        if (n_batch_actual > B_PACK_BATCH_COLS) {
            n_batch_actual = B_PACK_BATCH_COLS;
        }
        if (n_batch_actual <= 0) {
            ffts_cross_core_sync(PIPE_MTE3, GET_FFST_MSG(2, b_pack_ready_flag(pack_task_seq)));
            ++pack_task_seq;
            return;
        }

        pipe_slot = (int32_t)(pack_task_seq % B_PACK_PIPE_DEPTH);
        B_pack_batch_dst = core_B_pack + (int64_t)pipe_slot * b_pack_slot_stride_elems;

        pack_panels();
        ffts_cross_core_sync(PIPE_MTE3, GET_FFST_MSG(2, b_pack_ready_flag(pack_task_seq)));
        ++pack_task_seq;
    }

    __aicore__ __attribute__((always_inline)) inline void pack_row(int32_t slab)
    {
        start_vec = row_offsets[row_block];
        end_vec = row_offsets[row_block + 1];
        num_vectors = end_vec - start_vec;

        if (start_vec < 0 || end_vec < start_vec || num_vectors <= 0) {
            return;
        }

        row_cols = col_indices + start_vec;

#if SPMM_N_MAJOR
        const int32_t first_batch = slab;
        const int32_t end_batch = slab + 1;
#else
        const int32_t first_batch = 0;
        const int32_t end_batch = num_n_batches_i32;
#endif
        for (n_batch_idx = first_batch; n_batch_idx < end_batch; ++n_batch_idx) {
            pack_batch();
        }
    }

    __aicore__ __attribute__((always_inline)) inline void drain()
    {
        int64_t drain_begin = pack_task_seq - B_PACK_PIPE_DEPTH;
        if (drain_begin < 0) {
            drain_begin = 0;
        }
        for (int64_t task_seq = drain_begin; task_seq < pack_task_seq; ++task_seq) {
            wait_flag_dev(b_pack_ack_flag(task_seq));
        }
    }

#endif
    __aicore__ __attribute__((always_inline)) inline void run()
    {
        initialize();
#if !SPMM_DIRECT_B_GATHER_AIC
        if (total_b_pack_elems > 0) {
#if SPMM_N_MAJOR
            for (int32_t slab = 0; slab < num_n_batches_i32; ++slab) {
#else
            const int32_t slab = 0;
#endif
                for (row_block = core_id; row_block < num_row_blocks; row_block += core_cnt) {
                    pack_row(slab);
                }
#if SPMM_N_MAJOR
            }
#endif
            drain();
        }
#endif
        pipe_barrier(PIPE_MTE2);
        pipe_barrier(PIPE_MTE3);
    }
};

__aicore__ __inline__ void run_aiv(const SpmmKernelInput& input)
{
    AivState state{input};
    state.run();
}
