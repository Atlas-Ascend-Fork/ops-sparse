/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSPARSE_SPMM_1D_CATALOG_H_
#define ACLSPARSE_SPMM_1D_CATALOG_H_

#include <cstddef>
#include "../spmm_1d.h"

// One source of truth for legal compiled configurations. Tile dimensions are
// columns/vectors, not core counts. The device's compile-time assertions remain
// the final capacity checks when adding a configuration.
struct SpmmCandidate {
    SpmmTilingConfig tiling;
    int panel_rows;
    bool ub_pipeline;
    bool l0c_pipeline;
    bool direct;
};

inline const SpmmCandidate *SpmmCandidates(size_t *count)
{
    static const SpmmCandidate candidates[] = {
        {MakeWorkspaceTiling(16,128,32,0,"aic16_ktile128_bpack512"),16,false,false,false},
        {MakeWorkspaceTiling(16,128,64,1,"aic16_ktile128_bpack1024"),16,false,false,false},
        {MakeWorkspaceTiling(8,256,64,2,"aic8_ktile256"),16,false,false,false},
        {MakeWorkspaceTiling(4,512,64,3,"aic4_ktile512"),16,false,false,false},
        {MakeWorkspaceTiling(2,1024,64,4,"aic2_ktile1024"),16,false,false,false},
        {MakeWorkspaceTiling(1,2048,64,5,"aic1_ktile2048"),16,false,false,false},
        {MakeWorkspaceTiling(8,256,64,6,"aic8_ktile256_bpack1024_aivpp"),16,true,false,false},
        {MakeWorkspaceTiling(16,128,64,7,"aic16_ktile128_bpack1024_aivpp"),16,true,false,false},
        {MakeWorkspaceTiling(16,128,128,8,"aic16_ktile128_bpack2048_aivpp"),16,true,false,false},
        {MakeDirectTiling(64,32,64,9,"aic64_ktile32_direct"),16,false,false,true},
        {MakeWorkspaceTiling(32,64,64,10,"aic32_ktile64_bpack1024_l0cpp"),16,false,true,false},
        {MakeWorkspaceTiling(16,128,32,15,"aic16_ktile128_bpack512_aivpp"),16,true,false,false},
        {MakeWorkspaceTiling(16,128,64,23,"aic16_ktile128_bpack1024_aivpp_l0cpp"),16,true,true,false},
        {MakeWorkspaceTiling(32,64,32,33,"aic32_ktile64_bpack512_aivpp_l0cpp"),16,true,true,false},
        {MakeWorkspaceTiling(16,128,64,34,"aic16_ktile128_bpack1024_aivpp_l0cpp_p32"),32,true,true,false},
        {MakeWorkspaceTiling(16,128,32,36,"aic16_ktile128_bpack512_aivpp_l0cpp_p32"),32,true,true,false},
        {MakeWorkspaceTiling(32,64,32,37,"aic32_ktile64_bpack512_aivpp_l0cpp_p32"),32,true,true,false},
        {MakeDirectTiling(32,64,64,38,"aic32_ktile64_direct_l0cpp"),16,false,true,true},
        {MakeDirectTiling(16,128,64,39,"aic16_ktile128_direct_l0cpp"),16,false,true,true},
        {MakeFusionTiling(16,128,32,48,"fused2_aic16_ktile128_bpack512"),32,true,true,false},
        {MakeFusionTiling(8,256,32,49,"fused4_aic8_ktile256_bpack512"),32,true,true,false},
        {MakeFusionTiling(4,256,32,50,"fused8_aic4_ktile256_bpack512"),32,true,true,false},
        {MakeWorkspaceTiling(16,128,64,57,"nmajor57"),32,true,true,false},
    };
    if (count) *count = sizeof(candidates) / sizeof(candidates[0]);
    return candidates;
}

inline const SpmmCandidate *SpmmFindCandidate(int key)
{
    size_t count = 0;
    const auto *candidates = SpmmCandidates(&count);
    for (size_t i = 0; i < count; ++i) {
        if (candidates[i].tiling.tiling_key == key) return candidates + i;
    }
    return nullptr;
}

inline bool SpmmCandidateIsLegal(const SpmmCandidate &candidate)
{
    const auto &t = candidate.tiling;
    const int64_t nt = static_cast<int64_t>(t.aic_n_batch) * 16;
    const int64_t pack = static_cast<int64_t>(t.b_pack_n_batch) * 16;
    const int group = SpmmFusionGroup(t.tiling_key);
    if (group != 0) {
        const int64_t height = group * 16;
        return t.sparse_k_tile == SpmmFusionCapacity(group) && nt == 512 / group &&
            2 * nt * t.sparse_k_tile <= 65536 && 2 * height * t.sparse_k_tile <= 65536 &&
            2 * 4 * height * nt <= 65536 && pack == 512 && candidate.panel_rows == 32 &&
            candidate.ub_pipeline && candidate.l0c_pipeline && !candidate.direct;
    }
    return nt > 0 && t.sparse_k_tile >= 32 && t.sparse_k_tile <= 2048 &&
        (t.sparse_k_tile & (t.sparse_k_tile - 1)) == 0 &&
        nt * t.sparse_k_tile * 2 <= 64 * 1024 &&
        static_cast<int64_t>(t.sparse_k_tile) * 16 * 2 <= 64 * 1024 &&
        nt * 16 * 4 * (candidate.l0c_pipeline ? 2 : 1) <= 64 * 1024 &&
        pack >= nt && pack % nt == 0 && candidate.panel_rows > 0 &&
        candidate.panel_rows % 16 == 0 && candidate.panel_rows <= 32 &&
        (!candidate.ub_pipeline || 2 * candidate.panel_rows * pack * 2 <= 128 * 1024) &&
        !(candidate.direct && candidate.ub_pipeline);
}

#endif
