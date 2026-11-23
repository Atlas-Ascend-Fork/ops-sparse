/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

struct AicState : SpmmKernelInput {
    int32_t core_id;
    int32_t core_cnt;
    __cbuf__ __fp16* L1_A;
    __cbuf__ __fp16* L1_B;
    __ca__ __fp16* L0A;
    __cb__ __fp16* L0B;
    __cc__ float* L0C;
    __gm__ __fp16* B_pack;
    __gm__ __fp16* core_B_pack;
    int64_t b_pack_core_stride_elems;
    int64_t num_row_blocks_i64;
    int32_t num_n_blocks_i32;
    int64_t row_pack_task_base;
    bool l1_release_pending;
    bool l0_release_pending;
    bool l0c_fix_pending0;
    bool l0c_fix_pending1;
    int64_t m_block_idx;
    int32_t row_block;
    int32_t start_vec;
    int32_t end_vec;
    int32_t num_vectors;
    int32_t total_k_fracs;
    int32_t row_k_round;
    __gm__ __fp16* A_src;
    __gm__ int* row_cols;
    bool resident_a_fast_path;
    bool single_k_tile_fast_path;
    bool b_l1_prefetch_pending;
    int32_t n_block_idx;
    int32_t n_base;
    int32_t batch_n_actual;
    int32_t max_batch_n;
    int32_t batch_n_fracs;
    int32_t n_batch_idx;
    int64_t pack_task_seq;
    int32_t pack_n_block_base;
    int32_t local_n_block;
    bool first_compute_in_pack;
    bool last_compute_in_pack;
    int32_t k_tile_actual;
    int32_t k_tile_round;
    int32_t k_tile_fracs;
    int32_t pipe_slot;
    __gm__ __fp16* B_contig;
    __cc__ float* L0C_tile;
    int32_t l0c_slot;
    int32_t vec_base;
    bool last_k_tile;
    __gm__ __fp16* A_tile_src;
    __cb__ __fp16* A_l0_tile;
    bool first_k_tile;
    __aicore__ __attribute__((always_inline)) inline void initialize()
    {
        (void)total_b_pack_elems;

        set_ffts_base_addr((uint64_t)ffts_addr);

        set_padding(0);
        set_atomic_none();
        set_nd_para(0x1);

        core_id = get_block_idx();
        core_cnt = get_block_num();

        /* L1_A stages B_selected^T; L1_B stages A_block^T. */
        L1_A = (__cbuf__ __fp16*)(uintptr_t)0;
        L1_B = (__cbuf__ __fp16*)(uintptr_t)(64 * 1024);

        L0A = (__ca__ __fp16*)(uintptr_t)0;
        L0B = (__cb__ __fp16*)(uintptr_t)0;
        L0C = (__cc__ float*)(uintptr_t)0;

        /* workspace1 is a per-core GM ring from AIV to AIC. */
        B_pack = workspace1;
        b_pack_core_stride_elems = (int64_t)B_PACK_PIPE_DEPTH * b_pack_slot_stride_elems;
        core_B_pack = B_pack + (int64_t)core_id * b_pack_core_stride_elems;

        num_row_blocks_i64 = ((int64_t)M + vec_length - 1) / vec_length;
        num_n_blocks_i32 = (int32_t)num_n_blocks;

        row_pack_task_base = 0;
        /* MTE1 releases L1 to MTE2; MAD releases L0A/L0B to MTE1. */
        l1_release_pending = false;
        l0_release_pending = false;
#if SPMM_ENABLE_L0C_PIPELINE
        /*
         * Two L0C halves are alternated by N batch. PIPE_FIX releases each half
         * independently, allowing MAD for batch p+1 to overlap Fixpipe for p.
         */
        l0c_fix_pending0 = false;
        l0c_fix_pending1 = false;
#endif
    }

    __aicore__ __attribute__((always_inline)) inline bool prepare_row()
    {
        row_block = (int32_t)m_block_idx;

        start_vec = row_offsets[row_block];
        end_vec = row_offsets[row_block + 1];
        num_vectors = end_vec - start_vec;

        if (start_vec < 0 || end_vec < start_vec) {
            return false;
        }

        if (num_vectors == 0) {
            return false;
        }

        if (num_vectors > (int32_t)max_vectors_per_row) {
            return false;
        }

        total_k_fracs = ceil_div_16_i32(num_vectors);
        row_k_round = total_k_fracs * 16;
        A_src = values + start_vec * VECTOR_LEN;
        row_cols = col_indices + start_vec;

        /* Keep the complete A row block in L0B whenever it fits. */
        resident_a_fast_path = (num_vectors > 0) && (total_k_fracs > 0) && (total_k_fracs <= 255) &&
                               ((int64_t)row_k_round * VECTOR_LEN <= (int64_t)L0B_CAPACITY_HALF_ELEMS);

        single_k_tile_fast_path = resident_a_fast_path && (num_vectors <= SPARSE_K_TILE);

        return true;
    }

    __aicore__ __attribute__((always_inline)) inline bool prepare_batch()
    {
        n_base = n_block_idx * BLOCK_SIZE;

        batch_n_actual = N - n_base;
        max_batch_n = AIC_N_BATCH * BLOCK_SIZE;
        if (batch_n_actual > max_batch_n) {
            batch_n_actual = max_batch_n;
        }
        batch_n_fracs = ceil_div_16_i32(batch_n_actual);
        n_batch_idx = n_block_idx / B_PACK_N_BATCH;
        pack_task_seq =
#if SPMM_N_MAJOR
            row_pack_task_base;
#else
            row_pack_task_base + (int64_t)n_batch_idx;
#endif
        pack_n_block_base = n_batch_idx * B_PACK_N_BATCH;
        local_n_block = n_block_idx - pack_n_block_base;
        first_compute_in_pack = (local_n_block == 0);
        last_compute_in_pack = (n_block_idx + AIC_N_BATCH >= num_n_blocks_i32) ||
                               (local_n_block + AIC_N_BATCH >= B_PACK_N_BATCH);

        if (batch_n_actual <= 0 || batch_n_actual > max_batch_n) {
            return false;
        }

        if (first_compute_in_pack) {
#if !SPMM_DIRECT_B_GATHER_AIC
            wait_flag_dev(b_pack_ready_flag(pack_task_seq));
#endif
        }

        return true;
    }

#include "spmm_1d_aic_transfer.inc"
#include "spmm_1d_aic_compute.inc"
    __aicore__ __attribute__((always_inline)) inline void run_row(int32_t slab)
    {
        if (!prepare_row()) {
            return;
        }
        if (resident_a_fast_path) {
            load_resident_a();
        }
        b_l1_prefetch_pending = false;
#if SPMM_N_MAJOR
        const int32_t first_n_block = slab * B_PACK_N_BATCH;
        const int32_t end_n_block = (slab + 1) * B_PACK_N_BATCH < num_n_blocks_i32 ? (slab + 1) * B_PACK_N_BATCH :
                                                                                     num_n_blocks_i32;
#else
        const int32_t first_n_block = 0;
        const int32_t end_n_block = num_n_blocks_i32;
#endif
        for (n_block_idx = first_n_block; n_block_idx < end_n_block; n_block_idx += AIC_N_BATCH) {
            if (!prepare_batch()) {
                continue;
            }
            if (single_k_tile_fast_path) {
                run_single_tile();
            } else {
                run_multi_tile();
            }
        }
#if SPMM_N_MAJOR
        row_pack_task_base += 1;
#else
        row_pack_task_base += (num_n_blocks_i32 + B_PACK_N_BATCH - 1) / B_PACK_N_BATCH;
#endif
    }

    __aicore__ __attribute__((always_inline)) inline void drain()
    {
        if (l1_release_pending) {
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
            l1_release_pending = false;
        }
        if (l0_release_pending) {
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            l0_release_pending = false;
        }
#if SPMM_ENABLE_L0C_PIPELINE
        if (l0c_fix_pending0) {
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            l0c_fix_pending0 = false;
        }
        if (l0c_fix_pending1) {
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
            l0c_fix_pending1 = false;
        }
#endif
        /*
         * Terminal drain is outside the hot loops. Use per-pipeline barriers
         * instead of PIPE_ALL so kernel completion only waits for pipelines that
         * can still own data from this AIC path.
         */
        pipe_barrier(PIPE_MTE2);
        pipe_barrier(PIPE_MTE1);
        pipe_barrier(PIPE_M);
        pipe_barrier(PIPE_FIX);
    }

    __aicore__ __attribute__((always_inline)) inline void run()
    {
        initialize();
#if SPMM_N_MAJOR
        for (int32_t slab = 0; slab < (num_n_blocks_i32 + B_PACK_N_BATCH - 1) / B_PACK_N_BATCH; ++slab) {
#else
        const int32_t slab = 0;
#endif
            for (m_block_idx = core_id; m_block_idx < num_row_blocks_i64; m_block_idx += core_cnt) {
                run_row(slab);
            }
#if SPMM_N_MAJOR
        }
#endif
        drain();
    }
};

__aicore__ __inline__ void run_aic(const SpmmKernelInput& input)
{
    AicState state{input};
    state.run();
}
