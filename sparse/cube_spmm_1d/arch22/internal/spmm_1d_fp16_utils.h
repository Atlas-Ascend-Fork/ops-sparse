/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSPARSE_SPMM_1D_FP16_UTILS_H
#define ACLSPARSE_SPMM_1D_FP16_UTILS_H

namespace fp16 {
// Minimum Cube tile: (M, N, K) = 16 x 16 x 16.
constexpr int64_t NUM_ELE_PERBLOCK = 32 / sizeof(__fp16);
constexpr int64_t CUBE_N0 = 16;
constexpr int64_t CUBE_K0 = 32 / sizeof(__fp16);
constexpr int64_t CUBE_MATRIX_SIZE = CUBE_K0 * CUBE_N0;
constexpr int64_t UINT16_STRIDE_LIMIT = 65536;

/* Converts a column-major ND matrix from GM to NZ format in L1. */
__aicore__ __inline__ void spmm_matrix_gm2cbuf_ND2nZ(__cbuf__ __fp16* dst, __gm__ __fp16* src, int64_t CBUF_M0,
                                                     int64_t CBUF_N0, int64_t m_actual, int64_t n_actual, size_t stride)
{
    if (stride < UINT16_STRIDE_LIMIT) {
        copy_gm_to_cbuf_multi_nd2nz_b16(
            dst, src, static_cast<uint8_t>(0), static_cast<uint16_t>(1), static_cast<uint16_t>(n_actual),
            static_cast<uint16_t>(m_actual), static_cast<uint16_t>(0), static_cast<uint16_t>(stride),
            static_cast<uint16_t>(CBUF_N0), static_cast<uint16_t>(1), static_cast<uint16_t>(0));
    } else {
        for (int i = 0; i < n_actual; i++) {
            copy_gm_to_cbuf_multi_nd2nz_b16(dst + i * CUBE_K0, src + i * stride, static_cast<uint8_t>(0),
                                            static_cast<uint16_t>(1), static_cast<uint16_t>(1),
                                            static_cast<uint16_t>(m_actual), static_cast<uint16_t>(0),
                                            static_cast<uint16_t>(0), static_cast<uint16_t>(CBUF_N0),
                                            static_cast<uint16_t>(0), static_cast<uint16_t>(0));
        }
    }
}

/* One NN matrix. A source stride wider than the instruction field is
 * represented by issuing one row at a time (dst_n_stride == 0). */
__aicore__ __inline__ void copy_nn_tile(__cbuf__ __fp16* dst, __gm__ __fp16* src, int64_t m_actual, int64_t rows,
                                        size_t stride, uint16_t dst_n_stride)
{
    copy_gm_to_cbuf_multi_nd2nz_b16(dst, src, static_cast<uint8_t>(0), static_cast<uint16_t>(1),
                                    static_cast<uint16_t>(rows), static_cast<uint16_t>(m_actual),
                                    static_cast<uint16_t>(0), static_cast<uint16_t>(stride),
                                    static_cast<uint16_t>(CUBE_N0), dst_n_stride, static_cast<uint16_t>(0));
}

__aicore__ __inline__ void copy_nn_matrices(__cbuf__ __fp16* dst, __gm__ __fp16* src, int64_t cbuf_m, int64_t m_actual,
                                            int matrices, size_t stride)
{
    copy_gm_to_cbuf_multi_nd2nz_b16(
        dst, src, static_cast<uint8_t>(0), static_cast<uint16_t>(matrices), static_cast<uint16_t>(CUBE_N0),
        static_cast<uint16_t>(m_actual), static_cast<uint16_t>(CUBE_N0 * stride), static_cast<uint16_t>(stride),
        static_cast<uint16_t>(CUBE_N0), static_cast<uint16_t>(1), static_cast<uint16_t>(CUBE_N0 * cbuf_m));
}

__aicore__ __inline__ void copy_nn_wide_stride(__cbuf__ __fp16* dst, __gm__ __fp16* src, int64_t cbuf_m,
                                               int64_t m_actual, int64_t n_actual, size_t stride)
{
    for (int i = 0; i < n_actual; ++i) {
        int block = i / CUBE_N0;
        int row = i % CUBE_N0;
        copy_nn_tile(dst + block * CUBE_N0 * cbuf_m + row * CUBE_K0, src + i * stride, m_actual, 1, 0, 0);
    }
}

/* Copies a column-major ND matrix from GM to NN format in L1. */
__aicore__ __inline__ void spmm_matrix_gm2cbuf_ND2nN(__cbuf__ __fp16* dst, __gm__ __fp16* src, int64_t CBUF_M0,
                                                     int64_t CBUF_N0, int64_t m_actual, int64_t n_actual, size_t stride)
{
    int64_t srcNdStride = CUBE_N0 * stride;
    int ndNum = n_actual / CUBE_N0;
    int remains = n_actual % CUBE_N0;
    if (srcNdStride < UINT16_STRIDE_LIMIT) {
        if (ndNum > 0) {
            copy_nn_matrices(dst, src, CBUF_M0, m_actual, ndNum, stride);
        }
    } else if (stride < UINT16_STRIDE_LIMIT) {
        for (int i = 0; i < ndNum; ++i) {
            copy_nn_tile(dst + i * CUBE_N0 * CBUF_M0, src + i * CUBE_N0 * stride, m_actual, CUBE_N0, stride, 1);
        }
    } else {
        copy_nn_wide_stride(dst, src, CBUF_M0, m_actual, n_actual, stride);
        return;
    }
    if (remains > 0) {
        copy_nn_tile(dst + ndNum * CUBE_N0 * CBUF_M0, src + ndNum * CUBE_N0 * stride, m_actual, remains, stride, 1);
    }
}
/**
 * @brief Copies a column-major ND matrix from UB to GM.
 */
__aicore__ __inline__ void spmm_matrix_ubuf2gm(__gm__ __fp16* dst, __ubuf__ __fp16* src, int64_t m_actual,
                                               int64_t n_actual, size_t srcStride, size_t dstStride)
{
    int64_t m_round = ROUND(m_actual, NUM_ELE_PERBLOCK);
    if (m_actual % NUM_ELE_PERBLOCK == 0 && dstStride % NUM_ELE_PERBLOCK == 0 && dstStride < UINT16_STRIDE_LIMIT) {
        copy_ubuf_to_gm(dst, src, 0, n_actual, m_round / NUM_ELE_PERBLOCK, (srcStride - m_round) / NUM_ELE_PERBLOCK,
                        (dstStride - m_round) / NUM_ELE_PERBLOCK);
    } else if (m_actual % NUM_ELE_PERBLOCK == 0 && dstStride * NUM_ELE_PERBLOCK < UINT16_STRIDE_LIMIT) {
        int c0Loops = n_actual / NUM_ELE_PERBLOCK;
        int c0Remain = n_actual % NUM_ELE_PERBLOCK;
        if (c0Loops > 0) {
            for (int i = 0; i < NUM_ELE_PERBLOCK; ++i) {
                copy_ubuf_to_gm(dst + i * dstStride, src + i * srcStride, 0, c0Loops, m_actual / NUM_ELE_PERBLOCK,
                                (srcStride * NUM_ELE_PERBLOCK - m_actual) / NUM_ELE_PERBLOCK,
                                (dstStride * NUM_ELE_PERBLOCK - m_actual) / NUM_ELE_PERBLOCK);
            }
        }
        for (int i = 0; i < c0Remain; ++i) {
            copy_ubuf_to_gm(dst + c0Loops * NUM_ELE_PERBLOCK * dstStride + i * dstStride,
                            src + c0Loops * NUM_ELE_PERBLOCK * srcStride + i * srcStride, 0, 1,
                            m_actual / NUM_ELE_PERBLOCK, 0, 0);
        }
    } else {
        for (int i = 0; i < n_actual; ++i) {
            copy_ubuf_to_gm_align_b16(dst + i * dstStride, src + i * srcStride, 0, 1, m_actual * sizeof(__fp16), 0, 0,
                                      0, 0);
        }
    }
}

} // namespace fp16

#endif // ACLSPARSE_SPMM_1D_FP16_UTILS_H
