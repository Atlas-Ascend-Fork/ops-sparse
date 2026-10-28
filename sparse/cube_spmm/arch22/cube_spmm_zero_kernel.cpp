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

#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void cube_spmm_zero_kernel(
    GM_ADDR c, uint32_t rows, uint32_t cols, uint32_t ld)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t tile = 4096;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, tile * sizeof(float));
    auto zeros = buf.Get<float>();
    AscendC::Duplicate(zeros, 0.0f, tile);
    const auto event = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    AscendC::SetFlag<HardEvent::V_MTE3>(event);
    AscendC::WaitFlag<HardEvent::V_MTE3>(event);

    AscendC::GlobalTensor<float> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c), static_cast<uint64_t>(rows) * ld);
    for (uint32_t row = GetBlockIdx(); row < rows; row += GetBlockNum()) {
        for (uint32_t col = 0; col < cols; col += tile) {
            const uint32_t count = cols - col < tile ? cols - col : tile;
            AscendC::DataCopyExtParams p{};
            p.blockCount = 1;
            p.blockLen = count * sizeof(float);
            // One logical row at a time preserves arbitrary ld padding.
            AscendC::DataCopyPad(gm[static_cast<uint64_t>(row) * ld + col], zeros, p);
        }
    }
}

extern "C" void cube_spmm_zero_launch(
    GM_ADDR c, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t blocks, void *stream)
{
    if (rows == 0 || cols == 0 || blocks == 0) {
        return;
    }
    blocks = blocks < rows ? blocks : rows;
    cube_spmm_zero_kernel<<<blocks, nullptr, stream>>>(c, rows, cols, ld);
}
