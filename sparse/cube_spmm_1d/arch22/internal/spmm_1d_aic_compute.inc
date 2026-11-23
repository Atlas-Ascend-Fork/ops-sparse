/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

__aicore__ __attribute__((always_inline)) inline void select_output_tile()
{
    L0C_tile = L0C;
#if SPMM_ENABLE_L0C_PIPELINE
    l0c_slot = (n_block_idx / AIC_N_BATCH) & 1;
    if (l0c_slot == 0) {
        if (l0c_fix_pending0) {
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            l0c_fix_pending0 = false;
        }
    } else {
        if (l0c_fix_pending1) {
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
            l0c_fix_pending1 = false;
        }
        L0C_tile = L0C + L0C_TILE_FLOAT_ELEMS;
    }
#endif
}

__aicore__ __attribute__((always_inline)) inline void compute_tile(bool initialize, __cb__ __fp16* right_operand)
{
    mad(L0C_tile, L0A, right_operand, (uint16_t)batch_n_actual, (uint16_t)k_tile_actual, (uint16_t)VECTOR_LEN, 0, 1, 0,
        false, false, false, initialize);
}

__aicore__ __attribute__((always_inline)) inline void begin_fix()
{
#if SPMM_ENABLE_L0C_PIPELINE
    if (l0c_slot == 0) {
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    } else {
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    }
#else
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
}

__aicore__ __attribute__((always_inline)) inline void write_output()
{
    int64_t c_offset = m_block_idx * VECTOR_LEN + (int64_t)n_base * ldc;

    copy_matrix_cc_to_gm(C + c_offset, L0C_tile, 0, VECTOR_LEN, (uint16_t)batch_n_actual, (uint32_t)ldc, VECTOR_LEN, 0,
                         0, 0, false, true);

#if SPMM_ENABLE_L0C_PIPELINE
    if (l0c_slot == 0) {
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        l0c_fix_pending0 = true;
    } else {
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        l0c_fix_pending1 = true;
    }
#else
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
#endif
}

__aicore__ __attribute__((always_inline)) inline void run_single_tile()
{
    load_single_b();
    load_l0a();
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    l1_release_pending = true;
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    select_output_tile();
    compute_tile(true, L0B);
    prefetch_b();
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    l0_release_pending = true;
    begin_fix();
    write_output();
}

__aicore__ __attribute__((always_inline)) inline bool prepare_k_tile()
{
    k_tile_actual = num_vectors - vec_base;
    if (k_tile_actual > SPARSE_K_TILE) {
        k_tile_actual = SPARSE_K_TILE;
    }

    k_tile_fracs = ceil_div_16_i32(k_tile_actual);
    k_tile_round = k_tile_fracs * 16;

    if (k_tile_fracs <= 0 || k_tile_fracs > 255) {
        return false;
    }

    last_k_tile = (vec_base + SPARSE_K_TILE >= num_vectors);

    A_tile_src = A_src + (int64_t)vec_base * VECTOR_LEN;
    A_l0_tile = L0B;
    if (resident_a_fast_path) {
        A_l0_tile = L0B + (int64_t)(vec_base / BLOCK_SIZE) * CUBE_MATRIX_SIZE;
    }
    pipe_slot = (int32_t)(pack_task_seq % B_PACK_PIPE_DEPTH);
    B_contig = core_B_pack + (int64_t)pipe_slot * b_pack_slot_stride_elems + (int64_t)vec_base * B_PACK_BATCH_COLS +
               (int64_t)local_n_block * BLOCK_SIZE;
    if (l1_release_pending) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        l1_release_pending = false;
    }

    return true;
}

__aicore__ __attribute__((always_inline)) inline void stage_k_tile()
{
    /* L1_A stores the left operand B_selected. */
    stage_b(vec_base, n_base, batch_n_actual, B_contig, last_k_tile && last_compute_in_pack);

    /* Stage A per K tile only when the full row does not fit L0B. */
    if (!resident_a_fast_path) {
        spmm_matrix_gm2cbuf_ND2nN(L1_B, A_tile_src, VECTOR_LEN, k_tile_round, VECTOR_LEN, k_tile_actual, VECTOR_LEN);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    if (l0_release_pending) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        l0_release_pending = false;
    }
}

__aicore__ __attribute__((always_inline)) inline void compute_k_tile()
{
    if (!resident_a_fast_path) {
        load_cbuf_to_cb(L0B, L1_B, 0, (uint8_t)k_tile_fracs, 1, 0, 0, true, inc);
    }

    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    l1_release_pending = true;
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    compute_tile(first_k_tile, A_l0_tile);

    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    l0_release_pending = true;
    if (last_k_tile) {
        begin_fix();
    }
    first_k_tile = false;
}

__aicore__ __attribute__((always_inline)) inline void run_k_tile()
{
    if (!prepare_k_tile()) {
        return;
    }
    stage_k_tile();
    load_l0a();
    compute_k_tile();
}

__aicore__ __attribute__((always_inline)) inline void run_multi_tile()
{
    select_output_tile();
    first_k_tile = true;
    for (vec_base = 0; vec_base < num_vectors; vec_base += SPARSE_K_TILE) {
        run_k_tile();
    }
    write_output();
}
