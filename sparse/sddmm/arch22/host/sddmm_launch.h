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
#ifndef SDDMM_HOST_LAUNCH_H_
#define SDDMM_HOST_LAUNCH_H_

#include "sddmm_tiling.h"
#include "cann_ops_sparse.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

extern "C" aclsparseStatus_t sddmm_kernel_launch(GM_ADDR matA, GM_ADDR matB, GM_ADDR csrRowOffsets,
                                                 GM_ADDR csrColIndices, GM_ADDR csrValues,
                                                 const SddmmTilingData& tiling, uint32_t numBlocks, void* stream);

extern "C" aclsparseStatus_t sddmm_y_transpose_launch(GM_ADDR y, GM_ADDR yt, uint32_t rows, uint32_t cols, uint32_t yLd,
                                                      uint32_t pitch, uint32_t xType, uint32_t numBlocks, void* stream);

extern "C" aclsparseStatus_t sddmm_complex_prepare(GM_ADDR matX, GM_ADDR matY, GM_ADDR csrRowOffsets,
                                                   GM_ADDR csrColIndices, uint32_t rows, uint32_t k, uint32_t cols,
                                                   uint32_t nnz, uint32_t idxBase, uint32_t blocks, void* workspace,
                                                   void* stream);

extern "C" aclsparseStatus_t sddmm_complex_launch(GM_ADDR matX, GM_ADDR matY, GM_ADDR csrRowOffsets,
                                                  GM_ADDR csrColIndices, GM_ADDR csrValues, uint32_t rows, uint32_t k,
                                                  uint32_t nnz, uint32_t idxBase, float alphaReal, float alphaImag,
                                                  float betaReal, float betaImag, const SddmmTilingData& innerTiling,
                                                  void* workspace, void* stream);

extern "C" aclsparseStatus_t sddmm_index32_launch(GM_ADDR source, GM_ADDR dest, uint32_t count, uint32_t cores,
                                                  void* stream);

extern "C" aclsparseStatus_t sddmm_csr_row_hints_launch(GM_ADDR rows, GM_ADDR columns, GM_ADDR hints, uint32_t rowCount,
                                                        int32_t base, uint32_t outAlign, uint32_t cores, void* stream);

extern "C" aclsparseStatus_t sddmm_scale_launch(GM_ADDR values, uint32_t count, aclDataType type, float br, float bi,
                                                uint32_t cores, void* stream);

extern "C" aclsparseStatus_t sddmm_pattern_check_launch(GM_ADDR rows, GM_ADDR columns, GM_ADDR savedRows,
                                                        GM_ADDR savedColumns, GM_ADDR status, uint32_t rowCount,
                                                        uint32_t columnCount, uint32_t nnz, int32_t base, bool compare,
                                                        uint32_t cores, void* stream);
extern "C" aclsparseStatus_t sddmm_pack_dense_launch(GM_ADDR source, GM_ADDR dest, uint32_t rows, uint32_t columns,
                                                     uint32_t rowStep, uint32_t columnStep, aclDataType type,
                                                     uint32_t cores, void* stream);
extern "C" aclsparseStatus_t sddmm_bsr_expand_launch(GM_ADDR rows, GM_ADDR columns, GM_ADDR scalarRows,
                                                     GM_ADDR scalarColumns, uint32_t m, uint32_t blocks, uint32_t bd,
                                                     int32_t base, uint32_t cores, void* stream);
extern "C" aclsparseStatus_t sddmm_bsr_repack_launch(GM_ADDR blockValues, GM_ADDR scalarValues, GM_ADDR rows,
                                                     uint32_t blockRows, uint32_t blockNnz, uint32_t blockDim,
                                                     int32_t base, aclDataType type, bool columnMajor, bool unpack,
                                                     uint32_t cores, void* stream);
extern "C" size_t sddmm_bsr_cube_workspace(uint32_t m, uint32_t n, uint32_t k, aclDataType type);
extern "C" aclsparseStatus_t sddmm_bsr_cube_launch(GM_ADDR x, GM_ADDR y, GM_ADDR rows, GM_ADDR columns, GM_ADDR oldC,
                                                   GM_ADDR output, GM_ADDR workspace, uint32_t m, uint32_t n,
                                                   uint32_t k, uint32_t bd, uint32_t blocks, int32_t base,
                                                   bool colMajor, aclDataType xType, aclDataType cType, float ar,
                                                   float ai, float br, float bi, void* stream);
extern "C" aclsparseStatus_t sddmm_bsr_small_launch(GM_ADDR x, GM_ADDR y, GM_ADDR rows, GM_ADDR columns, GM_ADDR values,
                                                    uint32_t m, uint32_t n, uint32_t k, uint32_t bd, uint32_t blocks,
                                                    int32_t base, bool colMajor, aclDataType xType, aclDataType cType,
                                                    float alpha, float beta, void* stream);
extern "C" aclsparseStatus_t sddmm_bsr_complex_vector_launch(GM_ADDR x, GM_ADDR y, GM_ADDR rows, GM_ADDR columns,
                                                             GM_ADDR values, uint32_t m, uint32_t n, uint32_t k,
                                                             uint32_t bd, uint32_t blocks, int32_t base, bool colMajor,
                                                             float ar, float ai, float br, float bi, void* stream);

#endif
