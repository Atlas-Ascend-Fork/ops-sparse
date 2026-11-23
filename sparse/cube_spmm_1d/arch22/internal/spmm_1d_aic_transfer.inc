/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

__aicore__ __attribute__((always_inline)) inline void stage_b(int32_t vec_offset, int32_t column_base, int32_t columns,
                                                              __gm__ __fp16* packed, bool release_workspace)
{
#if SPMM_DIRECT_B_GATHER_AIC
    copy_b_selected_direct_to_l1(L1_A, B, row_cols, vec_offset, K, N, column_base, k_tile_round, k_tile_actual,
                                 columns);
#else
    copy_b_pack_row_major_to_l1(L1_A, packed, B_PACK_BATCH_COLS, k_tile_round, k_tile_actual, columns);

    /*
     * Early ACK remains valid in the L0B-resident fast path:
     * after this MTE2 B_pack read completes, this n-batch no
     * longer touches the GM workspace slot.
     */
    if (release_workspace) {
        ffts_cross_core_sync(PIPE_MTE2, GET_FFST_MSG(2, b_pack_ack_flag(pack_task_seq)));
    }
#endif
}

__aicore__ __attribute__((always_inline)) inline void load_l0a()
{
    for (int32_t batch_n_frac = 0; batch_n_frac < batch_n_fracs; ++batch_n_frac) {
        load_cbuf_to_ca(L0A + CUBE_MATRIX_SIZE * k_tile_fracs * batch_n_frac, L1_A,
                        (uint16_t)(batch_n_frac * k_tile_fracs), (uint8_t)k_tile_fracs, 1, 0, 0, true, inc);
    }
}

__aicore__ __attribute__((always_inline)) inline void load_resident_a()
{
    if (l1_release_pending) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        l1_release_pending = false;
    }

    spmm_matrix_gm2cbuf_ND2nN(L1_B, A_src, VECTOR_LEN, row_k_round, VECTOR_LEN, num_vectors, VECTOR_LEN);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    if (l0_release_pending) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        l0_release_pending = false;
    }

    load_cbuf_to_cb(L0B, L1_B, 0, (uint8_t)total_k_fracs, 1, 0, 0, true, inc);

    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    l1_release_pending = true;
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
}

__aicore__ __attribute__((always_inline)) inline void load_single_b()
{
    k_tile_actual = num_vectors;
    k_tile_round = row_k_round;
    k_tile_fracs = total_k_fracs;

    pipe_slot = (int32_t)(pack_task_seq % B_PACK_PIPE_DEPTH);
    B_contig = core_B_pack + (int64_t)pipe_slot * b_pack_slot_stride_elems + (int64_t)local_n_block * BLOCK_SIZE;

    if (b_l1_prefetch_pending) {
        /*
         * The preceding N iteration already queued this GM->L1
         * transfer and its MTE2->MTE1 ready event.
         */
        b_l1_prefetch_pending = false;
    } else {
        if (l1_release_pending) {
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
            l1_release_pending = false;
        }

        stage_b(0, n_base, batch_n_actual, B_contig, last_compute_in_pack);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    }
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    if (l0_release_pending) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        l0_release_pending = false;
    }
}

__aicore__ __attribute__((always_inline)) inline void prefetch_b()
{
    /* Prefetch within the ready pack so MTE2 can overlap MAD/Fixpipe. */
    int32_t next_n_block_idx = n_block_idx + AIC_N_BATCH;
    bool has_next_compute = next_n_block_idx < num_n_blocks_i32;
    bool may_prefetch_next =
#if SPMM_DIRECT_B_GATHER_AIC
        has_next_compute;
#else
        !last_compute_in_pack;
#endif
    if (may_prefetch_next) {
        int32_t next_n_base = next_n_block_idx * BLOCK_SIZE;
        int32_t next_batch_n_actual = N - next_n_base;
        if (next_batch_n_actual > max_batch_n) {
            next_batch_n_actual = max_batch_n;
        }

        int32_t next_local_n_block = local_n_block + AIC_N_BATCH;
        __gm__ __fp16* next_B_contig = core_B_pack + (int64_t)pipe_slot * b_pack_slot_stride_elems +
                                       (int64_t)next_local_n_block * BLOCK_SIZE;

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        l1_release_pending = false;

        bool next_is_last_compute_in_pack = (next_n_block_idx + AIC_N_BATCH >= num_n_blocks_i32) ||
                                            (next_local_n_block + AIC_N_BATCH >= B_PACK_N_BATCH);
        stage_b(0, next_n_base, next_batch_n_actual, next_B_contig, next_is_last_compute_in_pack);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        b_l1_prefetch_pending = true;
    }
}
