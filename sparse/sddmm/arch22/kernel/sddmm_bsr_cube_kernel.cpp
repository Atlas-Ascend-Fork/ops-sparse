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
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "cann_ops_sparse.h"
#include "sddmm_complex_kernel.h"
#include "../host/sddmm_launch.h"
#include "../host/sddmm_tiling_abi.h"
#include "sddmm_tiling.h"
#include <algorithm>

namespace {
constexpr uint32_t kWaveBytes = SddmmBudget::kCubeWaveBytes;
constexpr uint32_t kGroupMetaBytes = SddmmBudget::kCubeGroupMetaBytes;
constexpr uint32_t kEpilogueValues = SddmmBudget::kCubeEpilogueValues;
} // namespace

namespace {
template <typename CT, bool Complex>
class BlockEpilogue {
public:
    __aicore__ inline void Run(GM_ADDR products, GM_ADDR oldC, GM_ADDR output, SddmmBsrKernelTiling shape)
    {
        Init(products, oldC, output, shape);
        const uint32_t vals = s.bd * s.bd < kEpilogueValues ? s.bd * s.bd : kEpilogueValues;
        const uint32_t words = vals * lanes;
        const uint32_t width = vals / s.bd;
        InitMaps(words, width);
        const uint32_t tiles = s.bd * s.bd / vals;
        const uint64_t plane = uint64_t(s.count) * s.bd * s.bd * lanes;
        for (uint32_t task = AscendC::GetBlockIdx(); task < s.count * tiles; task += AscendC::GetBlockNum()) {
            ProcessTile(task, tiles, words, width, plane);
        }
    }

private:
    struct Views {
        AscendC::LocalTensor<float> z, q, scaled, temp, signs, oldFloat;
        AscendC::LocalTensor<int32_t> index, map, swap, aux, parity;
    };
    static constexpr uint32_t lanes = Complex ? 2 : 1;
    static constexpr uint32_t capacity = kEpilogueValues * lanes;
    __aicore__ inline Views TensorViews()
    {
        auto z = scratch.Get<float>();
        auto q = z[capacity];
        auto scaled = z[2 * capacity];
        auto temp = z[3 * capacity];
        auto signs = z[4 * capacity];
        auto oldFloat = z[5 * capacity];
        auto index = tables.Get<int32_t>();
        auto map = index[capacity];
        auto swap = index[2 * capacity];
        auto aux = index[3 * capacity];
        auto parity = index[4 * capacity];
        return {z, q, scaled, temp, signs, oldFloat, index, map, swap, aux, parity};
    }
    __aicore__ inline void Init(GM_ADDR products, GM_ADDR oldC, GM_ADDR output, SddmmBsrKernelTiling shape)
    {
        s = shape;
        p.SetGlobalBuffer((__gm__ float*)products);
        grouped.SetGlobalBuffer((__gm__ int32_t*)(products + kWaveBytes));
        old.SetGlobalBuffer((__gm__ CT*)oldC);
        out.SetGlobalBuffer((__gm__ CT*)output);
        pipe.InitBuffer(input, 1, capacity * sizeof(float));
        pipe.InitBuffer(oldQueue, 1, capacity * sizeof(CT));
        pipe.InitBuffer(outputQueue, 1, capacity * sizeof(CT));
        pipe.InitBuffer(scratch, capacity * 6 * sizeof(float));
        pipe.InitBuffer(tables, capacity * 5 * sizeof(int32_t));
    }
    __aicore__ inline void InitComplexMaps(uint32_t words)
    {
        auto [z, q, scaled, temp, signs, oldFloat, index, map, swap, aux, parity] = TensorViews();
        AscendC::ShiftRight(aux, index, int32_t(1), words);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(parity, aux, int32_t(2), words);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(parity, index, parity, words);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(swap, parity, int32_t(-2), words);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(swap, swap, index, words);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(swap, swap, int32_t(1), words);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(swap, swap, int32_t(4), words);
        AscendC::Cast(signs, parity, AscendC::RoundMode::CAST_NONE, words);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(signs, signs, 2.0f, words);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(signs, signs, -1.0f, words);
    }
    __aicore__ inline void InitMaps(uint32_t words, uint32_t width)
    {
        auto [z, q, scaled, temp, signs, oldFloat, index, map, swap, aux, parity] = TensorViews();
        AscendC::CreateVecIndex(index, 0, words);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (Complex) {
            InitComplexMaps(words);
        } else {
            AscendC::Muls(aux, index, int32_t(1), words);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (s.colMajor) {
            uint32_t shift = 0;
            for (uint32_t d = s.bd; d > 1; d >>= 1) {
                ++shift;
            }
            AscendC::ShiftRight(map, aux, int32_t(shift), words);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(index, map, int32_t(s.bd), words);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(index, aux, index, words);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(index, index, int32_t(width), words);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(map, map, index, words);
        } else {
            AscendC::Muls(map, aux, int32_t(1), words);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (Complex) {
            AscendC::Muls(map, map, int32_t(2), words);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(map, map, parity, words);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Muls(map, map, int32_t(4), words);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void AddPreviousValues(uint64_t dest, uint32_t words)
    {
        auto [z, q, scaled, temp, signs, oldFloat, index, map, swap, aux, parity] = TensorViews();
        if (s.br != 0.0f || s.bi != 0.0f) {
            auto value = oldQueue.template AllocTensor<CT>();
            AscendC::DataCopy(value, old[dest], words);
            oldQueue.EnQue(value);
            value = oldQueue.template DeQue<CT>();
            if constexpr (AscendC::IsSameType<CT, float>::value) {
                AscendC::DataCopy(oldFloat, value, words);
            } else {
                AscendC::Cast(oldFloat, value, AscendC::RoundMode::CAST_NONE, words);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (s.br != 0.0f) {
                AscendC::Axpy(scaled, oldFloat, s.br, words);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if constexpr (Complex) {
                if (s.bi != 0.0f) {
                    AscendC::Gather(temp, oldFloat, swap.template ReinterpretCast<uint32_t>(), 0, words);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(temp, temp, signs, words);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Axpy(scaled, temp, s.bi, words);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
            oldQueue.FreeTensor(value);
        }
    }
    __aicore__ inline void ProcessTile(uint32_t task, uint32_t tiles, uint32_t words, uint32_t width, uint64_t plane)
    {
        auto [z, q, scaled, temp, signs, oldFloat, index, map, swap, aux, parity] = TensorViews();
        const uint32_t safeTiles = tiles == 0 ? 1u : tiles;
        const uint32_t block = task / safeTiles, tile = task % safeTiles;
        const uint32_t run = s.group > 1 ? grouped.GetValue((block / s.group) * 16) : 1;
        const uint32_t first = run > 1 ? block / s.group * s.group : block;
        const uint32_t pitch = s.bd * run;
        const uint64_t source = uint64_t(first) * s.bd * s.bd * lanes + (block - first) * s.bd * lanes +
                                (s.colMajor ? tile * width * lanes : tile * width * pitch * lanes);
        Load(z, source, words, width, pitch, map);
        if constexpr (Complex) {
            Load(q, plane + source, words, width, pitch, map);
            AscendC::Gather(temp, q, swap.template ReinterpretCast<uint32_t>(), 0, words);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(temp, temp, signs, words);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(z, z, temp, words);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (s.ar == 0.0f) {
            AscendC::Duplicate(scaled, 0.0f, words);
        } else {
            AscendC::Muls(scaled, z, s.ar, words);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (Complex) {
            if (s.ai != 0.0f) {
                AscendC::Gather(temp, z, swap.template ReinterpretCast<uint32_t>(), 0, words);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(temp, temp, signs, words);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Axpy(scaled, temp, s.ai, words);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        const uint64_t dest = uint64_t(s.first + block) * s.bd * s.bd * lanes + tile * words;
        AddPreviousValues(dest, words);
        auto value = outputQueue.template AllocTensor<CT>();
        if constexpr (AscendC::IsSameType<CT, float>::value) {
            AscendC::DataCopy(value, scaled, words);
        } else {
            AscendC::Cast(value, scaled, AscendC::RoundMode::CAST_RINT, words);
        }
        outputQueue.EnQue(value);
        value = outputQueue.template DeQue<CT>();
        AscendC::DataCopy(out[dest], value, words);
        outputQueue.FreeTensor(value);
    }
    __aicore__ inline void Load(AscendC::LocalTensor<float> dest, uint64_t source, uint32_t words, uint32_t width,
                                uint32_t pitch, AscendC::LocalTensor<int32_t> map)
    {
        constexpr uint32_t lanes = Complex ? 2 : 1;
        auto raw = input.AllocTensor<float>();
        if (s.colMajor || pitch != s.bd) {
            const uint32_t columns = s.colMajor ? width : s.bd;
            AscendC::DataCopyExtParams copy{uint16_t(s.colMajor ? s.bd : width), columns * lanes * 4,
                                            (pitch - columns) * lanes * 4, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(raw, p[source], copy, pad);
        } else {
            AscendC::DataCopy(raw, p[source], words);
        }
        input.EnQue(raw);
        raw = input.DeQue<float>();
        AscendC::Gather(dest, raw, map.ReinterpretCast<uint32_t>(), 0, words);
        AscendC::PipeBarrier<PIPE_V>();
        input.FreeTensor(raw);
    }
    SddmmBsrKernelTiling s;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> input, oldQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratch, tables;
    AscendC::GlobalTensor<float> p;
    AscendC::GlobalTensor<int32_t> grouped;
    AscendC::GlobalTensor<CT> old, out;
};

template <typename CT, bool Complex>
__global__ __aicore__ void BsrEpilogue(GM_ADDR products, GM_ADDR oldC, GM_ADDR output, SddmmBsrKernelTiling s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BlockEpilogue<CT, Complex> op;
    op.Run(products, oldC, output, s);
}

} // namespace

namespace {

__aicore__ inline uint32_t SelectCubeGroup(AscendC::GlobalTensor<int32_t> rows, AscendC::GlobalTensor<int32_t> cols,
                                           AscendC::GlobalTensor<int32_t> grouped, const SddmmBsrKernelTiling& s,
                                           uint32_t task, uint32_t& row, bool& contiguous)
{
    const uint32_t block = s.first + task;
    row = 0;
    uint32_t hi = s.m / s.bd;
    while (row < hi) {
        const uint32_t mid = (row + hi) / 2;
        if (rows.GetValue(mid + 1) - s.base <= block) {
            row = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (row >= s.m / s.bd) {
        return 0;
    }
    const int32_t column = (cols.GetValue(block) - s.base) * s.bd;
    if (column < 0 || uint64_t(column) + s.bd > s.n) {
        return 0;
    }
    const uint32_t count = s.count - task < s.group ? s.count - task : s.group;
    contiguous = rows.GetValue(row + 1) - s.base >= block + count;
    for (uint32_t j = 1; j < count && contiguous; ++j) {
        contiguous = cols.GetValue(block + j) == cols.GetValue(block) + int32_t(j);
    }
    if (s.group > 1) {
        // Each work group owns a complete cache line in the metadata plane.
        const uint32_t slot = (task / s.group) * 16;
        grouped.SetValue(slot, contiguous ? int32_t(count) : 1);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(grouped[slot]);
    }
    return count;
}

template <typename T, typename CT>
struct CubeOperands {
    AscendC::GlobalTensor<T> a, b, imag;
    AscendC::GlobalTensor<CT> p;
    AscendC::GlobalTensor<int32_t> rows, cols;
    AscendC::GlobalTensor<int32_t> grouped;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR xi, GM_ADDR y, GM_ADDR products, GM_ADDR rp, GM_ADDR ci)
    {
        a.SetGlobalBuffer((__gm__ T*)x);
        imag.SetGlobalBuffer((__gm__ T*)xi);
        b.SetGlobalBuffer((__gm__ T*)y);
        p.SetGlobalBuffer((__gm__ CT*)products);
        rows.SetGlobalBuffer((__gm__ int32_t*)rp);
        cols.SetGlobalBuffer((__gm__ int32_t*)ci);
        grouped.SetGlobalBuffer((__gm__ int32_t*)(products + kWaveBytes));
    }
};

template <typename T, typename Matmul>
__aicore__ inline void ConfigureCubePrecision(Matmul& mm)
{
    if constexpr (AscendC::IsSameType<T, float>::value) {
        mm.SetHF32(false);
    }
}

// Work is partitioned in sparse blocks, never in individual dot products.
template <typename T, bool Complex, bool TransposedY, typename CT = float>
__global__ __aicore__ void BsrCube(GM_ADDR x, GM_ADDR xi, GM_ADDR y, GM_ADDR rp, GM_ADDR ci, GM_ADDR products,
                                   SddmmBsrKernelTiling s, AscendC::tiling::TCubeTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;
    CubeOperands<T, CT> operands;
    operands.Init(x, xi, y, products, rp, ci);
    auto& [a, b, imag, p, rows, cols, grouped] = operands;
    using Input = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
    using Right = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, TransposedY>;
    using Output = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CT>;
    matmul::Matmul<Input, Right, Output> mm;
    REGIST_MATMUL_OBJ(&pipe, nullptr, mm, &tiling);
    ConfigureCubePrecision<T>(mm);
    constexpr uint32_t lanes = Complex ? 2 : 1;
    const uint32_t blockWords = s.bd * s.bd * lanes;
    const uint32_t bPitch = TransposedY ? s.yPitch : s.k;
    for (uint32_t task = AscendC::GetBlockIdx() * s.group; task < s.count; task += AscendC::GetBlockNum() * s.group) {
        const uint32_t block = s.first + task;
        uint32_t lo = 0;
        bool contiguous = false;
        const uint32_t count = SelectCubeGroup(rows, cols, grouped, s, task, lo, contiguous);
        if (!count) {
            continue;
        }
        const uint32_t run = contiguous ? count : 1;
        mm.SetOrgShape(s.m, s.n * lanes, s.k, bPitch, s.bd * lanes * run);
        mm.SetSingleShape(s.bd, s.bd * lanes * run, s.k);
        for (uint32_t j = 0; j < count; j += run) {
            uint32_t row = lo;
            while (rows.GetValue(row + 1) - s.base <= block + j) {
                ++row;
            }
            const uint64_t aOffset = uint64_t(row) * s.bd * s.k;
            const uint64_t bOffset = uint64_t(cols.GetValue(block + j) - s.base) * s.bd * lanes *
                                     (TransposedY ? bPitch : 1);
            mm.SetTensorA(a[aOffset]);
            mm.SetTensorB(b[bOffset], TransposedY);
            mm.IterateAll(p[uint64_t(task + j) * blockWords]);
            mm.End();
            if constexpr (Complex) {
                mm.SetTensorA(imag[aOffset]);
                mm.SetTensorB(b[bOffset], TransposedY);
                mm.IterateAll(p[uint64_t(s.count + task + j) * blockWords]);
                mm.End();
            }
        }
    }
}

// A = [Yr^T, Yi^T], B = [[Xr^T, Xi^T], [-Xi^T, Xr^T]].
// A*B stores interleaved complex C^T, already in column-major BSR order.
__aicore__ inline void InitComplexPackXMap(AscendC::LocalTensor<int32_t> index, AscendC::LocalTensor<int32_t> temp,
                                           AscendC::LocalTensor<int32_t> lane, AscendC::LocalTensor<int32_t> parity,
                                           AscendC::LocalTensor<int32_t> xmap, AscendC::LocalTensor<int32_t> xswap,
                                           AscendC::LocalTensor<int32_t> ymap, AscendC::LocalTensor<float> signs)
{
    AscendC::CreateVecIndex(index, 0, SddmmBudget::kCubeGatherElements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(temp, index, int32_t(4), SddmmBudget::kCubeGatherElements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(lane, temp, int32_t(16), SddmmBudget::kCubeGatherElements);
    AscendC::Muls(xmap, temp, int32_t(2), SddmmBudget::kCubeGatherElements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(lane, index, lane, SddmmBudget::kCubeGatherElements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(temp, lane, int32_t(1), SddmmBudget::kCubeGatherElements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(parity, temp, int32_t(2), SddmmBudget::kCubeGatherElements);
    AscendC::Muls(temp, temp, int32_t(64), SddmmBudget::kCubeGatherElements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(parity, lane, parity, SddmmBudget::kCubeGatherElements);
    AscendC::Add(xmap, xmap, temp, SddmmBudget::kCubeGatherElements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(xmap, xmap, parity, SddmmBudget::kCubeGatherElements);
    AscendC::Cast(signs, parity, AscendC::RoundMode::CAST_NONE, SddmmBudget::kCubeGatherElements);
    AscendC::Muls(temp, parity, int32_t(-2), SddmmBudget::kCubeGatherElements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(temp, temp, int32_t(1), SddmmBudget::kCubeGatherElements);
    AscendC::Muls(signs, signs, 2.0f, SddmmBudget::kCubeGatherElements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(xswap, xmap, temp, SddmmBudget::kCubeGatherElements);
    AscendC::Adds(signs, signs, -1.0f, SddmmBudget::kCubeGatherElements);
    AscendC::Muls(xmap, xmap, int32_t(4), SddmmBudget::kCubeGatherElements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(xswap, xswap, int32_t(4), SddmmBudget::kCubeGatherElements);
}

__aicore__ inline void InitComplexPackYMap(AscendC::LocalTensor<int32_t> index, AscendC::LocalTensor<int32_t> temp,
                                           AscendC::LocalTensor<int32_t> lane, AscendC::LocalTensor<int32_t> parity,
                                           AscendC::LocalTensor<int32_t> xmap, AscendC::LocalTensor<int32_t> xswap,
                                           AscendC::LocalTensor<int32_t> ymap, AscendC::LocalTensor<float> signs)
{
    AscendC::ShiftRight(temp, index, int32_t(5), 256);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(lane, temp, int32_t(32), 256);
    AscendC::Muls(ymap, temp, int32_t(2), 256);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(lane, index, lane, 256);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(lane, lane, int32_t(16), 256);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(ymap, ymap, lane, 256);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(ymap, ymap, int32_t(4), 256);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(ymap[256], ymap, int32_t(4), 256);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void StoreComplexPackHalves(AscendC::LocalTensor<float> raw,
                                              AscendC::TQue<AscendC::TPosition::VECOUT, 1>& output,
                                              AscendC::GlobalTensor<float> aG, AscendC::GlobalTensor<float> bG,
                                              AscendC::LocalTensor<int32_t> xmap, AscendC::LocalTensor<int32_t> xswap,
                                              AscendC::LocalTensor<int32_t> ymap, AscendC::LocalTensor<float> signs,
                                              bool isX, uint32_t row, uint32_t off, uint32_t m, uint32_t k)
{
    for (uint32_t half = 0; half < 2; ++half) {
        auto result = output.AllocTensor<float>();
        if (isX) {
            auto offsets = half ? xswap : xmap;
            AscendC::Gather(result, raw, offsets.ReinterpretCast<uint32_t>(), 0, SddmmBudget::kCubeGatherElements);
            AscendC::PipeBarrier<PIPE_V>();
            if (half) {
                AscendC::Mul(result, result, signs, SddmmBudget::kCubeGatherElements);
                AscendC::PipeBarrier<PIPE_V>();
            }
        } else {
            AscendC::Gather(result, raw, ymap[half * 256].ReinterpretCast<uint32_t>(), 0, 256);
        }
        output.EnQue(result);
        result = output.DeQue<float>();
        if (isX) {
            AscendC::DataCopyPad(bG[uint64_t(off + half * k) * 2 * m + row * 2], result,
                                 {32, 16 * 4, 0, (2 * m - 16) * 4, 0});
        } else {
            AscendC::DataCopyPad(aG[uint64_t(row) * 2 * k + off + half * k], result,
                                 {8, 32 * 4, 0, (2 * k - 32) * 4, 0});
        }
        output.FreeTensor(result);
    }
}

__global__ __aicore__ void BsrComplexPack(GM_ADDR x, GM_ADDR y, GM_ADDR packed, uint32_t m, uint32_t n, uint32_t k)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> input;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> output;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tables;
    pipe.InitBuffer(input, 1, SddmmBudget::kCubeGatherElements * sizeof(float));
    pipe.InitBuffer(output, 1, SddmmBudget::kCubeGatherElements * sizeof(float));
    pipe.InitBuffer(tables, 8 * SddmmBudget::kCubeGatherElements * sizeof(int32_t));
    auto index = tables.Get<int32_t>();
    auto temp = index[SddmmBudget::kCubeGatherElements], lane = index[2 * SddmmBudget::kCubeGatherElements],
         parity = index[3 * SddmmBudget::kCubeGatherElements];
    auto xmap = index[2048], xswap = index[2560], ymap = index[3072];
    auto signs = index[3584].ReinterpretCast<float>();
    InitComplexPackXMap(index, temp, lane, parity, xmap, xswap, ymap, signs);
    InitComplexPackYMap(index, temp, lane, parity, xmap, xswap, ymap, signs);
    AscendC::GlobalTensor<float> xG, yG, aG, bG;
    xG.SetGlobalBuffer((__gm__ float*)x);
    yG.SetGlobalBuffer((__gm__ float*)y);
    aG.SetGlobalBuffer((__gm__ float*)packed);
    bG.SetGlobalBuffer((__gm__ float*)packed + uint64_t(n) * 2 * k);
    const uint32_t kt = k / 32, xt = (m / 8) * kt, yt = (n / 8) * kt;
    for (uint32_t task = AscendC::GetBlockIdx(); task < xt + yt; task += AscendC::GetBlockNum()) {
        const bool isX = task < xt;
        const uint32_t local = isX ? task : task - xt;
        const uint32_t row = (local / kt) * 8, off = (local % kt) * 32;
        auto raw = input.AllocTensor<float>();
        if (isX) {
            AscendC::DataCopyPad(raw, xG[(uint64_t(row) * k + off) * 2], {8, 64 * 4, (2 * k - 64) * 4, 0, 0},
                                 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        } else {
            AscendC::DataCopyPad(raw, yG[uint64_t(off) * 2 * n + row * 2], {32, 16 * 4, (2 * n - 16) * 4, 0, 0},
                                 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        input.EnQue(raw);
        raw = input.DeQue<float>();
        StoreComplexPackHalves(raw, output, aG, bG, xmap, xswap, ymap, signs, isX, row, off, m, k);
        input.FreeTensor(raw);
    }
}

__global__ __aicore__ void BsrComplexDirect(GM_ADDR packed, GM_ADDR rp, GM_ADDR ci, GM_ADDR output,
                                            SddmmBsrKernelTiling s, AscendC::tiling::TCubeTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> a, b, c;
    AscendC::GlobalTensor<int32_t> rows, cols;
    a.SetGlobalBuffer((__gm__ float*)packed);
    b.SetGlobalBuffer((__gm__ float*)packed + uint64_t(s.n) * 2 * s.k);
    c.SetGlobalBuffer((__gm__ float*)output);
    rows.SetGlobalBuffer((__gm__ int32_t*)rp);
    cols.SetGlobalBuffer((__gm__ int32_t*)ci);
    using Tensor = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    matmul::Matmul<Tensor, Tensor, Tensor> mm;
    REGIST_MATMUL_OBJ(&pipe, nullptr, mm, &tiling);
    mm.SetHF32(false);
    mm.SetOrgShape(s.n, 2 * s.m, 2 * s.k, 2 * s.k, 2 * s.bd);
    const uint32_t perCore = (s.blocks + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
    const uint32_t first = AscendC::GetBlockIdx() * perCore;
    const uint32_t last = first + perCore < s.blocks ? first + perCore : s.blocks;
    if (first >= last) {
        return;
    }
    uint32_t lo = 0, hi = s.m / s.bd;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (uint32_t(rows.GetValue(mid + 1) - s.base) <= first) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    for (uint32_t block = first; block < last;) {
        while (uint32_t(rows.GetValue(lo + 1) - s.base) <= block) {
            ++lo;
        }
        const uint32_t col = cols.GetValue(block) - s.base;
        uint32_t run = 1;
        while (run < s.group && block + run < last && block + run < uint32_t(rows.GetValue(lo + 1) - s.base) &&
               uint32_t(cols.GetValue(block + run) - s.base) == col + run) {
            ++run;
        }
        mm.SetSingleShape(s.bd * run, 2 * s.bd, 2 * s.k);
        mm.SetTensorA(a[uint64_t(col) * s.bd * 2 * s.k]);
        mm.SetTensorB(b[uint64_t(lo) * s.bd * 2]);
        mm.IterateAll(c[uint64_t(block) * s.bd * s.bd * 2]);
        mm.End();
        block += run;
    }
}

__global__ __aicore__ void BsrSplitX(GM_ADDR x, GM_ADDR xr, GM_ADDR xi, uint32_t count)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    SddmmComplexSplitX op;
    op.Init(x, xr, xi, count, &pipe);
    op.Process();
}
} // namespace

extern "C" size_t sddmm_bsr_cube_workspace(uint32_t m, uint32_t n, uint32_t k, aclDataType type)
{
    const bool complex = type == ACL_COMPLEX64;
    const uint64_t lanes = complex ? 2 : 1;
    const uint32_t bytes = type == ACL_FLOAT16 || type == ACL_BF16 ? 2 : 4;
    return kWaveBytes + kGroupMetaBytes + (complex ? SddmmAlign32(uint64_t(m) * k * 4) * 2 : 0) +
           (uint64_t(n) * lanes > 65535 ? uint64_t(n) * lanes * SddmmYtPitch(k) * bytes : 0);
}

namespace {
class BsrCubeLaunch {
public:
    BsrCubeLaunch(GM_ADDR x, GM_ADDR y, GM_ADDR rowPtr, GM_ADDR scalarCols, GM_ADDR oldC, GM_ADDR output,
                  GM_ADDR workspace, uint32_t m, uint32_t n, uint32_t k, uint32_t bd, uint32_t blocks, int32_t base,
                  bool colMajor, aclDataType xType, aclDataType cType, float ar, float ai, float br, float bi,
                  void* stream)
        : x(x),
          y(y),
          rowPtr(rowPtr),
          scalarCols(scalarCols),
          oldC(oldC),
          output(output),
          workspace(workspace),
          m(m),
          n(n),
          k(k),
          bd(bd),
          blocks(blocks),
          base(base),
          colMajor(colMajor),
          xType(xType),
          cType(cType),
          ar(ar),
          ai(ai),
          br(br),
          bi(bi),
          stream(stream),
          complex(xType == ACL_COMPLEX64),
          transposedY(uint64_t(n) * (complex ? 2 : 1) > 65535)
    {}
    aclsparseStatus_t Run()
    {
        if (blocks == 0) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        auto status = Validate();
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        status = InitPlatform();
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        if (UseComplexDirect()) {
            return LaunchComplexDirect();
        }
        status = PrepareTiling();
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        status = PrepareInputs();
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            return status;
        }
        return LaunchWaves();
    }

private:
    aclsparseStatus_t Validate()
    {
        if (!workspace || bd < 16 || bd > 128 || (bd & (bd - 1)) || !k || k > 65472 || m % bd || n % bd) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        return ValidateTypes();
    }
    aclsparseStatus_t ValidateTypes()
    {
        if (!((xType == ACL_FLOAT && cType == ACL_FLOAT) || (complex && cType == ACL_COMPLEX64) ||
              (xType == ACL_FLOAT16 && (cType == ACL_FLOAT16 || cType == ACL_FLOAT)) ||
              (xType == ACL_BF16 && (cType == ACL_BF16 || cType == ACL_FLOAT)))) {
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t InitPlatform()
    {
        platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (!platform) {
            return ACL_SPARSE_STATUS_INTERNAL_ERROR;
        }
        cubeCores = platform->GetCoreNumAic();
        vectorCores = platform->GetCoreNumAiv();
        if (!cubeCores || !vectorCores) {
            return ACL_SPARSE_STATUS_INTERNAL_ERROR;
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    bool UseComplexDirect()
    {
        const uint64_t packedBytes = (uint64_t(m) * 4 + uint64_t(n) * 2) * k * sizeof(float);
        return complex && colMajor && bd >= 64 && k <= 256 && k % 32 == 0 && m <= 32752 && packedBytes <= kWaveBytes &&
               blocks >= cubeCores * 4 && uint64_t(blocks) * bd * bd * 4 >= uint64_t(m) * n && ar == 1.0f &&
               ai == 0.0f && br == 0.0f && bi == 0.0f;
    }
    aclsparseStatus_t LaunchComplexDirect()
    {
        struct DirectCache {
            uint32_t bd = 0, k = 0;
            AscendC::tiling::TCubeTiling tiling{};
        };
        static thread_local DirectCache cache;
        if (cache.bd != bd || cache.k != k) {
            matmul_tiling::MultiCoreMatmulTiling api(*platform);
            api.SetDim(1);
            api.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
                         false);
            api.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
                         false);
            api.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT);
            api.SetShape(bd * 4, 2 * bd, 2 * k);
            api.SetOrgShape(bd * 4, 2 * bd, 2 * k);
            api.EnableBias(false);
            api.SetBufferSpace(-1, -1, -1);
            if (api.GetTiling(cache.tiling) == -1) {
                return ACL_SPARSE_STATUS_NOT_SUPPORTED;
            }
            cache.bd = bd;
            cache.k = k;
        }
        BsrComplexPack<<<vectorCores, nullptr, stream>>>(x, y, workspace, m, n, k);
        SddmmBsrHostTiling s{m, n, k, bd, blocks, 0, blocks, 0, 4, base, colMajor, ar, ai, br, bi};
        BsrComplexDirect<<<std::min(blocks, cubeCores), nullptr, stream>>>(workspace, rowPtr, scalarCols, output,
                                                                           SddmmSerializeBsrTiling(s), cache.tiling);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    struct Cache {
        uint32_t bd = 0, k = 0, group = 0;
        aclDataType type = ACL_DT_UNDEFINED;
        aclDataType outputType = ACL_DT_UNDEFINED;
        bool transposed = false;
        AscendC::tiling::TCubeTiling tiling{};
    };
    bool CacheMatches(const Cache& cache, aclDataType outputType) const
    {
        return !(cache.bd != bd || cache.k != k || cache.group != group || cache.type != xType ||
                 cache.outputType != outputType || cache.transposed != transposedY);
    }
    aclsparseStatus_t PrepareTiling()
    {
        // Exact shape/type tiling cache contains no tensor pointers or pattern state.
        // Thin block rows cannot fill four-block groups. Actual continuity is
        // still checked on device, including row boundaries and mutable indices.
        const bool thinRows = uint64_t(blocks) >= uint64_t(m / bd) * 2 && uint64_t(blocks) < uint64_t(m / bd) * 4;
        group = bd <= 32 && blocks >= cubeCores * 4 ? (thinRows ? 2 : 4) : 1;
        // A single row-major block already has the final sparse values layout.
        // Fixpipe performs the output conversion without a workspace epilogue.
        direct = !complex && !colMajor && group == 1 && ar == 1.0f && br == 0.0f;
        const aclDataType outputType = direct ? cType : ACL_FLOAT;
        static thread_local Cache cache;
        if (!CacheMatches(cache, outputType)) {
            matmul_tiling::MultiCoreMatmulTiling api(*platform);
            auto dtype = xType == ACL_FLOAT16 ? matmul_tiling::DataType::DT_FLOAT16 :
                         xType == ACL_BF16    ? matmul_tiling::DataType::DT_BF16 :
                                                matmul_tiling::DataType::DT_FLOAT;
            api.SetDim(1);
            api.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dtype, false);
            api.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dtype, transposedY);
            api.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         outputType == ACL_FLOAT16 ? matmul_tiling::DataType::DT_FLOAT16 :
                         outputType == ACL_BF16    ? matmul_tiling::DataType::DT_BF16 :
                                                     matmul_tiling::DataType::DT_FLOAT);
            api.SetShape(bd, bd * group * (complex ? 2 : 1), k);
            api.SetOrgShape(bd, bd * group * (complex ? 2 : 1), k);
            api.EnableBias(false);
            api.SetBufferSpace(-1, -1, -1);
            if (api.GetTiling(cache.tiling) == -1) {
                return ACL_SPARSE_STATUS_NOT_SUPPORTED;
            }
            cache.bd = bd;
            cache.group = group;
            cache.k = k;
            cache.type = xType;
            cache.outputType = outputType;
            cache.transposed = transposedY;
        }
        tilingData = cache.tiling;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t PrepareInputs()
    {
        realX = x;
        imagX = x;
        if (complex) {
            realX = workspace + kWaveBytes + kGroupMetaBytes;
            imagX = realX + SddmmAlign32(uint64_t(m) * k * 4);
            BsrSplitX<<<vectorCores, nullptr, stream>>>(x, realX, imagX, m * k);
        }
        effectiveY = y;
        if (transposedY) {
            // ND2NZ srcDValue is 16-bit. Bulk transpose makes the stride depend
            // on K instead of the unbounded dense width, without scalar packing.
            effectiveY = workspace + kWaveBytes + kGroupMetaBytes +
                         (complex ? SddmmAlign32(uint64_t(m) * k * 4) * 2 : 0);
            const uint32_t columns = n * (complex ? 2 : 1);
            const uint32_t code = xType == ACL_FLOAT16 ? SDDMM_DTYPE_FP16 :
                                  xType == ACL_BF16    ? SDDMM_DTYPE_BF16 :
                                                         SDDMM_DTYPE_FP32;
            const auto status = sddmm_y_transpose_launch(y, effectiveY, k, columns, columns, SddmmYtPitch(k), code,
                                                         vectorCores, stream);
            if (status != ACL_SPARSE_STATUS_SUCCESS) {
                return status;
            }
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    template <typename T, bool Complex, typename CT>
    void LaunchTyped(uint32_t cores, GM_ADDR destination, const SddmmBsrHostTiling& s)
    {
        if (transposedY) {
            BsrCube<T, Complex, true, CT><<<cores, nullptr, stream>>>(
                realX, imagX, effectiveY, rowPtr, scalarCols, destination, SddmmSerializeBsrTiling(s), tilingData);
        } else {
            BsrCube<T, Complex, false, CT><<<cores, nullptr, stream>>>(
                realX, imagX, effectiveY, rowPtr, scalarCols, destination, SddmmSerializeBsrTiling(s), tilingData);
        }
    }
    aclsparseStatus_t Dispatch(uint32_t cores, GM_ADDR destination, const SddmmBsrHostTiling& s)
    {
        if (complex) {
            LaunchTyped<float, true, float>(cores, destination, s);
            BsrEpilogue<float, true>
                <<<vectorCores, nullptr, stream>>>(workspace, oldC, output, SddmmSerializeBsrTiling(s));
        } else {
            if (direct && cType == ACL_FLOAT16) {
                LaunchTyped<half, false, half>(cores, destination, s);
            } else if (direct && cType == ACL_BF16) {
                LaunchTyped<bfloat16_t, false, bfloat16_t>(cores, destination, s);
            } else if (xType == ACL_FLOAT16) {
                LaunchTyped<half, false, float>(cores, destination, s);
            } else if (xType == ACL_BF16) {
                LaunchTyped<bfloat16_t, false, float>(cores, destination, s);
            } else if (xType == ACL_FLOAT) {
                LaunchTyped<float, false, float>(cores, destination, s);
            } else {
                return ACL_SPARSE_STATUS_NOT_SUPPORTED;
            }
            if (direct) {
                return ACL_SPARSE_STATUS_SUCCESS;
            }
            if (cType == ACL_FLOAT16) {
                BsrEpilogue<half, false>
                    <<<vectorCores, nullptr, stream>>>(workspace, oldC, output, SddmmSerializeBsrTiling(s));
            } else if (cType == ACL_BF16) {
                BsrEpilogue<bfloat16_t, false>
                    <<<vectorCores, nullptr, stream>>>(workspace, oldC, output, SddmmSerializeBsrTiling(s));
            } else if (cType == ACL_FLOAT) {
                BsrEpilogue<float, false>
                    <<<vectorCores, nullptr, stream>>>(workspace, oldC, output, SddmmSerializeBsrTiling(s));
            } else {
                return ACL_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    aclsparseStatus_t LaunchWaves()
    {
        const uint32_t wave = direct ? blocks : kWaveBytes / (bd * bd * (complex ? 16 : 4));
        for (uint32_t first = 0; first < blocks; first += wave) {
            SddmmBsrHostTiling s{
                m,    n,        k,  bd, blocks, first, std::min(wave, blocks - first), SddmmYtPitch(k), group,
                base, colMajor, ar, ai, br,     bi};
            const uint32_t cores = std::min(cubeCores, (s.count + group - 1) / group);
            const uint32_t outputBytes = cType == ACL_FLOAT16 || cType == ACL_BF16 ? 2 : 4;
            GM_ADDR destination = direct ? output + uint64_t(first) * bd * bd * outputBytes : workspace;
            const auto status = Dispatch(cores, destination, s);
            if (status != ACL_SPARSE_STATUS_SUCCESS) {
                return status;
            }
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    GM_ADDR x;
    GM_ADDR y;
    GM_ADDR rowPtr;
    GM_ADDR scalarCols;
    GM_ADDR oldC;
    GM_ADDR output;
    GM_ADDR workspace;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t bd;
    uint32_t blocks;
    int32_t base;
    bool colMajor;
    aclDataType xType;
    aclDataType cType;
    float ar;
    float ai;
    float br;
    float bi;
    void* stream;
    bool complex, transposedY, direct = false;
    uint32_t cubeCores = 0, vectorCores = 0, group = 1;
    platform_ascendc::PlatformAscendC* platform = nullptr;
    GM_ADDR realX = nullptr;
    GM_ADDR imagX = nullptr;
    GM_ADDR effectiveY = nullptr;
    AscendC::tiling::TCubeTiling tilingData{};
};
} // namespace

extern "C" aclsparseStatus_t sddmm_bsr_cube_launch(GM_ADDR x, GM_ADDR y, GM_ADDR rowPtr, GM_ADDR scalarCols,
                                                   GM_ADDR oldC, GM_ADDR output, GM_ADDR workspace, uint32_t m,
                                                   uint32_t n, uint32_t k, uint32_t bd, uint32_t blocks, int32_t base,
                                                   bool colMajor, aclDataType xType, aclDataType cType, float ar,
                                                   float ai, float br, float bi, void* stream)
{
    return BsrCubeLaunch(x, y, rowPtr, scalarCols, oldC, output, workspace, m, n, k, bd, blocks, base, colMajor, xType,
                         cType, ar, ai, br, bi, stream)
        .Run();
}
