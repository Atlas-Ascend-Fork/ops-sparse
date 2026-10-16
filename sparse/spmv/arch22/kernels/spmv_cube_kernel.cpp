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
 
#include "constants.h"
#include "csr_gather_kernel.h"
#include "scale_kernel.h"
#include "row_scan_kernel.h"
#include "seg_sum_vec_revert_kernel.h"
#include "tcuscan_utils.h"
#include "tiling_spmv_cube.h"
#include "spmv_cube_kernel.h"

using namespace AscendC;
using namespace tcuscan;

namespace tcuscan {


/**
 * @brief Run the SpMV using cube unit kernel.
 *
 * @tparam T input data type
 *
 * @param [in] vec_in Pointer to the input vector (sparse matrix non-zero
 * values).
 * @param [in] cols_in Pointer to the CSR column indices array.
 * @param [in] segm_ind_in Pointer to the segment indices vector.
 * @param [in] x_in Pointer to the dense input vector.
 * @param [in] upper_in Pointer to an upper-triangular all-ones square matrix of
 * size \f$\textit{tile\_len} \times \textit{tile\_len}\f$.
 * @param [out] vec_out Pointer to the output vector.
 * @param [in,out] workspace Pointer to a memory region used as workspace.
 * @param [in] vec_len Input vector length (number of non-zeros).
 * @param [in] num_segments Number of segments.
 * @param [in] x_len Length of the dense input vector.
 * @param [in] tile_len Tile size used for the matrix multiplication step.
 * @param [in] block_len Block length assigned to each AI Core group.
 * @param [in] alpha Scaling factor of the SpMV product, applied by the
 * segment reduction as it writes each segment sum.
 * @param [in] beta Scaling factor applied in-place to @p vec_out before the
 * segment reduction accumulates on top of it. Following the BLAS convention,
 * `beta == 0` overwrites @p vec_out without reading it.
 */
template <typename T>
__aicore__ inline void run_spmv_cube(
    GM_ADDR vec_in, GM_ADDR cols_in, GM_ADDR segm_ind_in, GM_ADDR x_in,
    GM_ADDR upper_in, GM_ADDR vec_out,
    GM_ADDR workspace, uint32_t vec_len, uint32_t num_segments, uint32_t x_len,
    uint32_t tile_len, uint32_t block_len, float alpha, float beta) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<T>;

  const uint32_t align_size = tile_len * tile_len;
  const uint32_t padded_vec_len = scalar::AlignUp(vec_len, align_size);
  const uint32_t pad_size = padded_vec_len * sizeof(T);

  GM_ADDR const csr_products_ws = workspace;
  GM_ADDR const spec_block_scan_ws = workspace + pad_size;

  // `y = beta * y` pre-pass. It must complete before the segment reduction
  // atomically accumulates `alpha * A @ x` into `vec_out`; the two `SyncAll`
  // barriers below provide that ordering. No-op when `beta == 1`.
  run_scale_inplace<OutputT, false>(vec_out, num_segments,
                                    static_cast<OutputT>(beta));

  const uint32_t csr_gather_tile_len = align_size > 1024 ? 1024 : align_size;
  run_csr_gather<T, false>(vec_in, cols_in, x_in, csr_products_ws, vec_len,
                           x_len, csr_gather_tile_len);

  sync::SyncGroup<sync::GroupSyncDirection::FULL>();
  AscendC::SyncAll<false /*isAIVOnly*/>();

  if ASCEND_IS_AIC {
    KernelRowScan<T> op_cube(tile_len, tile_len, padded_vec_len);
    op_cube.Init(csr_products_ws, upper_in, spec_block_scan_ws);
    op_cube.Process();
  }

  sync::SyncGroup<sync::GroupSyncDirection::FULL>();
  AscendC::SyncAll<false /*isAIVOnly*/>();
  AscendC::PipeBarrier<PIPE_ALL>();

  if ASCEND_IS_AIV {
    // id is the id of each AI Core (2 AIVs and 1 AIC core)
    const auto id = GetBlockIdx() / GetTaskRation();

    // Fused searchsorted: each group derives its own two per-block segment
    // offsets by binary-searching the full indptr for its block boundaries
    // `sstart[id] = min(id * block_len, vec_len)`
    const uint32_t sstart_id = scalar::Min<uint32_t>(id * block_len, vec_len);
    const uint32_t sstart_next =
        scalar::Min<uint32_t>((id + 1) * block_len, vec_len);
    int32_t segm_ind_offset =
        static_cast<int32_t>(scalar::LowerBoundGM<int32_t>(
            segm_ind_in, num_segments + 1, sstart_id));
    const int32_t next_offset =
        static_cast<int32_t>(scalar::LowerBoundGM<int32_t>(
            segm_ind_in, num_segments + 1, sstart_next));
    const int32_t num_segments_per_block = next_offset - segm_ind_offset;

    // The boundaries of each segment must overlap
    if (id > 0) {
      segm_ind_offset--;
    }

    // Each AI Core group is responsible (offsets) starting from `block_len`
    const uint32_t block_vec_offset = id * block_len;
    if (block_vec_offset >= vec_len) {
      return;
    }
    const bool is_overflow_block = block_vec_offset + block_len > vec_len;
    if (is_overflow_block) {
      block_len = vec_len - block_vec_offset;
    }

    KernelSegSumVecRevert<OutputT, false, true> op(
        block_len, num_segments_per_block, tile_len, block_vec_offset,
        static_cast<OutputT>(alpha));
    // `segm_ind_in` is the full indptr, while the segment reduction consumes
    // segment *end* indices, i.e. `indptr + 1`; hence the extra element skip.
    op.Init(spec_block_scan_ws,
            segm_ind_in + (segm_ind_offset + 1) * sizeof(int32_t),
            vec_out + segm_ind_offset * sizeof(OutputT));
    op.Process();
  }
}

}  // namespace tcuscan

/**
 * @brief Run the SpMV v2 kernel with half/float16 dtype.
 *
 * The segment indices format follows the scipy Compressed Sparse Row Matrix
 * convention
 * (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
 *
 * @param [in] vec_in Pointer to the sparse matrix non-zero values.
 * @param [in] cols_in Pointer to the CSR column indices array.
 * @param [in] indptr Pointer to the segment indices vector (CSR row pointers).
 * @param [in] x_in Pointer to the dense input vector.
 * @param [out] vec_out Pointer to the output vector.
 * @param [in,out] workspace Pointer to workspace. Supplied by the caller
 * through the `externalBuffer` of `aclsparseSpMV`; its size is the one reported
 * by `aclsparseSpMVGetBufferSize`.
 * @param [in] tiling Tiling parameters, passed by value as a kernel argument.
 */
extern "C" __global__ __aicore__ void spmv_cube_fp16(
    GM_ADDR vec_in, GM_ADDR cols_in, GM_ADDR indptr, GM_ADDR x_in,
    GM_ADDR vec_out, GM_ADDR workspace,
    tcuscan::SpMVCubeTiling tiling) {
  // 该 kernel 为 cube + vector 混合形态（1 AIC : 2 AIV）
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  const uint32_t vec_len = tiling.nnz;
  const uint32_t num_segments = tiling.num_segments;
  const uint32_t x_len = tiling.x_len;
  const uint32_t tile_len = tiling.tile_len;
  const uint32_t block_len = tiling.block_len;
  const float alpha = tiling.alpha;
  const float beta = tiling.beta;

  GM_ADDR const upper = tcuscan::load_tril_matrix<half>();
  tcuscan::run_spmv_cube<half>(vec_in, cols_in, indptr, x_in, upper,
                    vec_out, workspace, vec_len, num_segments, x_len, tile_len,
                    block_len, alpha, beta);
}

/**
 * @brief Run the SpMV Cube kernel with float32 dtype.
 *
 * The segment indices format follows the scipy Compressed Sparse Row Matrix
 * convention
 * (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
 *
 * @param [in] vec_in Pointer to the sparse matrix non-zero values.
 * @param [in] cols_in Pointer to the CSR column indices array.
 * @param [in] indptr Pointer to the segment indices vector (CSR row pointers).
 * @param [in] x_in Pointer to the dense input vector.
 * @param [out] vec_out Pointer to the output vector.
 * @param [in,out] workspace Pointer to workspace. Supplied by the caller
 * through the `externalBuffer` of `aclsparseSpMV`; its size is the one reported
 * by `aclsparseSpMVGetBufferSize`.
 * @param [in] tiling Tiling parameters, passed by value as a kernel argument.
 */
extern "C" __global__ __aicore__ void spmv_cube_fp32(
    GM_ADDR vec_in, GM_ADDR cols_in,
    GM_ADDR indptr, GM_ADDR x_in,
    GM_ADDR vec_out, GM_ADDR workspace,
    tcuscan::SpMVCubeTiling tiling) {
  // 该 kernel 为 cube + vector 混合形态（1 AIC : 2 AIV）
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  const uint32_t vec_len = tiling.nnz;
  const uint32_t num_segments = tiling.num_segments;
  const uint32_t x_len = tiling.x_len;
  const uint32_t tile_len = tiling.tile_len;
  const uint32_t block_len = tiling.block_len;
  const float alpha = tiling.alpha;
  const float beta = tiling.beta;

  GM_ADDR const upper = tcuscan::load_tril_matrix<float>();

  tcuscan::run_spmv_cube<float>(vec_in, cols_in, indptr, x_in, upper,
                     vec_out, workspace, vec_len, num_segments, x_len, tile_len,
                     block_len, alpha, beta);
}
