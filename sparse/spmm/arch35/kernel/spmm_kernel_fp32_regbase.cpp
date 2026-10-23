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

#include "spmm_kernel_regbase_common.h"

using namespace AscendC;
using namespace matmul;

namespace {

struct SpmmRegBaseTaskRange {
    int32_t first;
    int32_t end;
    int32_t stride;
};

__aicore__ inline SpmmRegBaseTaskRange GetSpmmRegBaseTaskRange(
    int32_t groups, int32_t tiles)
{
    const int32_t blocks = static_cast<int32_t>(GetBlockNum());
    return {static_cast<int32_t>(GetBlockIdx()),
        blocks > 0 ? groups * tiles : 0, blocks};
}

template<typename Derived>
struct SpmmFp32RegBaseSharedState : SpmmRegBaseTensorState {
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> bQueue_;
    TQue<TPosition::VECIN, 1> accumulatorQueue_;
    TQue<TPosition::VECIN, 1> valueQueue_;
    TBuf<TPosition::VECCALC> rowMetaBuffer_;
    TBuf<TPosition::VECCALC> rowIdsBuffer_;
    event_t rowMetaMte2ToVector_;
    event_t rowIdsMte2ToScalar_;
    event_t accumulatorMte3ToVector_;
    event_t accumulatorVectorToMte3_;

    __aicore__ inline void RunTasks()
    {
        const SpmmRegBaseTaskRange range =
            GetSpmmRegBaseTaskRange(td_.cube_group_count, tileCount_);
        for (int32_t task = range.first; task < range.end; task += range.stride) {
            static_cast<Derived *>(this)->ProcessTile(
                task / tileCount_, task % tileCount_);
        }
    }

    __aicore__ inline void FinishTile(LocalTensor<float> &accumulator,
        LocalTensor<float> &window, LocalTensor<float> &packed)
    {
        SetFlag<HardEvent::MTE3_V>(accumulatorMte3ToVector_);
        WaitFlag<HardEvent::MTE3_V>(accumulatorMte3ToVector_);
        accumulatorQueue_.FreeTensor(accumulator);
        bQueue_.FreeTensor(window);
        valueQueue_.FreeTensor(packed);
    }
};

__aicore__ inline void InitializeSpmmFp32RegBaseEvents(
    TPipe &pipe, event_t *rowMetaMte2ToVector,
    event_t *rowIdsMte2ToScalar, event_t *accumulatorMte3ToVector,
    event_t *accumulatorVectorToMte3)
{
    *rowMetaMte2ToVector = static_cast<event_t>(
        pipe.FetchEventID(HardEvent::MTE2_V));
    *rowIdsMte2ToScalar = static_cast<event_t>(
        pipe.FetchEventID(HardEvent::MTE2_S));
    *accumulatorMte3ToVector = static_cast<event_t>(
        pipe.FetchEventID(HardEvent::MTE3_V));
    *accumulatorVectorToMte3 = static_cast<event_t>(
        pipe.FetchEventID(HardEvent::V_MTE3));
}

__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmFp32RegBaseFallback(
    __gm__ const int32_t *rowOffsets, __gm__ const int32_t *columns,
    __gm__ const float *values, __gm__ const float *matB,
    __gm__ const float *initialC, __gm__ float *matC,
    __gm__ const int32_t *fallbackRows,
    uint64_t first, uint64_t end, int32_t n, int32_t ldb, int32_t ldc,
    int32_t indexBase)
{
    for (uint64_t work = first + static_cast<uint64_t>(threadIdx.x);
         work < end; work += static_cast<uint64_t>(blockDim.x)) {
        const int32_t fallbackIndex = static_cast<int32_t>(work / n);
        const int32_t outputColumn = static_cast<int32_t>(work % n);
        const int32_t row = fallbackRows[fallbackIndex];
        float accumulator = 0.0f;
        for (int32_t position = rowOffsets[row] - indexBase;
             position < rowOffsets[row + 1] - indexBase; ++position) {
            const int32_t denseRow = columns[position] - indexBase;
            accumulator += values[position] *
                matB[static_cast<uint64_t>(denseRow) * ldb + outputColumn];
        }
        const uint64_t outputOffset =
            static_cast<uint64_t>(row) * ldc + outputColumn;
        matC[outputOffset] = initialC[outputOffset] + accumulator;
    }
}

} // namespace

__simd_vf__ inline void SpmmFp32RegBaseMac(
    __ubuf__ float *accumulator, __ubuf__ float *window,
    __ubuf__ float *packedValues, __ubuf__ int32_t *localStarts,
    uint16_t rowCount)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> accumulator0;
    RegTensor<float> accumulator1;
    RegTensor<float> dense0;
    RegTensor<float> dense1;
    RegTensor<float> scalar;
    MaskReg mask = CreateMask<uint8_t, MaskPattern::ALL>();

    for (uint16_t row = 0; row < rowCount; ++row) {
        Duplicate(accumulator0, 0.0f, mask);
        Duplicate(accumulator1, 0.0f, mask);
#pragma unroll
        for (uint16_t ordinal = 0;
             ordinal < SPMM_FP32_REGBASE_DEGREE; ++ordinal) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(
                scalar,
                packedValues + row * SPMM_FP32_REGBASE_DEGREE + ordinal);
            const uint32_t denseOffset =
                (static_cast<uint32_t>(localStarts[row]) + ordinal) *
                SPMM_FP32_REGBASE_N;
            LoadAlign<float, LoadDist::DIST_NORM>(
                dense0, window + denseOffset);
            LoadAlign<float, LoadDist::DIST_NORM>(
                dense1, window + denseOffset + 64);
            MulAddDst(accumulator0, dense0, scalar, mask);
            MulAddDst(accumulator1, dense1, scalar, mask);
        }
        const uint32_t outputOffset =
            static_cast<uint32_t>(row) * SPMM_FP32_REGBASE_N;
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            accumulator + outputOffset, accumulator0, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            accumulator + outputOffset + 64, accumulator1, mask);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmFp32RegBaseRefresh(
    __gm__ const float *values, __gm__ float *packedValues,
    __gm__ const int32_t *valueBegins, __gm__ const int32_t *degrees,
    uint64_t first, uint64_t end)
{
    for (uint64_t work = first + static_cast<uint64_t>(threadIdx.x);
         work < end; work += static_cast<uint64_t>(blockDim.x)) {
        const uint64_t candidate = work / SPMM_FP32_REGBASE_DEGREE;
        const int32_t ordinal = static_cast<int32_t>(
            work % SPMM_FP32_REGBASE_DEGREE);
        packedValues[work] = ordinal < degrees[candidate] ?
            values[static_cast<uint64_t>(valueBegins[candidate]) + ordinal] :
            0.0f;
    }
}

class KernelSpmmFp32RegBaseDmaRefresh {
public:
    __aicore__ inline void Init(
        GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        td_ = LoadSpmmKernelTilingData(tilingGM);
        __gm__ uint8_t *workspace =
            reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        values_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(csrValues));
        packedValues_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + td_.cube_packed_a_offset));
        header_ = reinterpret_cast<__gm__ const SpmmFp32RegBaseHeader *>(
            workspace + td_.cube_scratch_offset);
        runs_ = reinterpret_cast<__gm__ const SpmmFp32RegBaseRefreshRun *>(
            workspace + td_.cube_scratch_offset +
            sizeof(SpmmFp32RegBaseHeader));
        pipe_.InitBuffer(valueBuffer_,
            SPMM_FP32_REGBASE_MAX_REFRESH_ROWS * 32);
        mte2ToMte3_ = static_cast<event_t>(
            pipe_.FetchEventID(HardEvent::MTE2_MTE3));
        mte3ToMte2_ = static_cast<event_t>(
            pipe_.FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        const int32_t block = static_cast<int32_t>(GetBlockIdx());
        const int32_t blocks = static_cast<int32_t>(GetBlockNum());
        for (int32_t runIndex = block;
             runIndex < header_->refresh_run_count; runIndex += blocks) {
            SpmmFp32RegBaseRefreshRun run{};
            run.value_begin = runs_[runIndex].value_begin;
            run.row_count = runs_[runIndex].row_count;
            run.degree = runs_[runIndex].degree;
            run.packed_begin = runs_[runIndex].packed_begin;
            run.packed_stride = runs_[runIndex].packed_stride;
            if (run.value_begin < 0 || run.row_count <= 0 ||
                run.row_count > SPMM_FP32_REGBASE_MAX_REFRESH_ROWS ||
                run.degree <= 0 ||
                run.degree > SPMM_FP32_REGBASE_DEGREE ||
                run.packed_begin < 0 || run.packed_stride <= 0) {
                continue;
            }
            ProcessRun(run);
        }
    }

private:
    __aicore__ inline void ProcessRun(
        const SpmmFp32RegBaseRefreshRun &run)
    {
        LocalTensor<float> values = valueBuffer_.Get<float>();
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
        DataCopyExtParams gather{
            static_cast<uint16_t>(run.row_count),
            static_cast<uint32_t>(run.degree * sizeof(float)),
            0, 0, 0};
        DataCopyPad(values, values_[run.value_begin], gather, noPad);
        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);

        DataCopyExtParams scatter{
            static_cast<uint16_t>(run.row_count),
            static_cast<uint32_t>(run.degree * sizeof(float)),
            0,
            static_cast<int64_t>(
                (static_cast<int64_t>(run.packed_stride) *
                     SPMM_FP32_REGBASE_DEGREE - run.degree) *
                static_cast<int64_t>(sizeof(float))),
            0};
        DataCopyPad(
            packedValues_[static_cast<uint64_t>(run.packed_begin) *
                          SPMM_FP32_REGBASE_DEGREE],
            values, scatter);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> valueBuffer_;
    GlobalTensor<float> values_;
    GlobalTensor<float> packedValues_;
    __gm__ const SpmmFp32RegBaseHeader *header_{nullptr};
    __gm__ const SpmmFp32RegBaseRefreshRun *runs_{nullptr};
    event_t mte2ToMte3_;
    event_t mte3ToMte2_;
    SpmmKernelTilingData td_{};
};

class KernelSpmmFp32RegBaseCompute {
public:
    __aicore__ inline void Init(
        GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        td_ = LoadSpmmKernelTilingData(tilingGM);
        __gm__ uint8_t *workspace =
            reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        matB_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matB));
        matC_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matC));
        packedValues_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + td_.cube_packed_a_offset));
        rowIds_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            workspace + td_.cube_row_ids_offset));
        header_ = reinterpret_cast<__gm__ const SpmmFp32RegBaseHeader *>(
            workspace + td_.cube_scratch_offset);
        __gm__ int32_t *slot1 = reinterpret_cast<__gm__ int32_t *>(
            workspace + td_.cube_slot1_offset);
        localStarts_.SetGlobalBuffer(
            slot1 + header_->fallback_count + 2 * header_->candidate_count);
        batches_ = reinterpret_cast<__gm__ const SpmmFp32RegBaseBatchMeta *>(
            workspace + td_.cube_meta_offset);

        pipe_.InitBuffer(accumulatorQueue_, 1,
            SPMM_FP32_REGBASE_ROWS * SPMM_FP32_REGBASE_N * sizeof(float));
        pipe_.InitBuffer(windowQueue_, 1,
            SPMM_FP32_REGBASE_WINDOW_ROWS *
            SPMM_FP32_REGBASE_N * sizeof(float));
        pipe_.InitBuffer(valueQueue_, 1,
            SPMM_FP32_REGBASE_ROWS * SPMM_FP32_REGBASE_DEGREE *
            sizeof(float));
        pipe_.InitBuffer(rowIdsBuffer_,
            SPMM_FP32_REGBASE_ROWS * sizeof(int32_t));
        pipe_.InitBuffer(localStartsBuffer_,
            SPMM_FP32_REGBASE_ROWS * sizeof(int32_t));
        localStartsReady_ = static_cast<event_t>(
            pipe_.FetchEventID(HardEvent::MTE2_V));
        rowIdsReady_ = static_cast<event_t>(
            pipe_.FetchEventID(HardEvent::MTE2_S));
    }

    __aicore__ inline void Process()
    {
        if (td_.n != SPMM_FP32_REGBASE_N ||
            td_.ldb != SPMM_FP32_REGBASE_N ||
            td_.ldc != SPMM_FP32_REGBASE_N) {
            return;
        }
        const int32_t block = static_cast<int32_t>(GetBlockIdx());
        const int32_t blocks = static_cast<int32_t>(GetBlockNum());
        for (int32_t batch = block; batch < td_.cube_group_count;
             batch += blocks) {
            ProcessBatch(batch);
        }
    }

private:
    __aicore__ inline bool IsValidBatch(const SpmmFp32RegBaseBatchMeta &batch) const
    {
        return batch.window_begin >= 0 && batch.row_count > 0 &&
            batch.row_count <= SPMM_FP32_REGBASE_ROWS && batch.row_id_begin >= 0 &&
            batch.window_rows > 0 &&
            batch.window_rows <= SPMM_FP32_REGBASE_WINDOW_ROWS;
    }

    __aicore__ inline void QueueBatchWindow(
        const SpmmFp32RegBaseBatchMeta &batch, uint32_t windowElements,
        LocalTensor<float> &window)
    {
        DataCopyParams copy{1, static_cast<uint16_t>(windowElements * sizeof(float) / 32), 0, 0};
        DataCopy(window, matB_[static_cast<uint64_t>(batch.window_begin) * td_.ldb], copy);
        windowQueue_.EnQue(window);
    }

    __aicore__ inline void QueueBatchValues(
        const SpmmFp32RegBaseBatchMeta &batch, LocalTensor<float> &packed)
    {
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
        DataCopyExtParams copy{1, static_cast<uint32_t>(batch.row_count *
            SPMM_FP32_REGBASE_DEGREE * sizeof(float)), 0, 0, 0};
        DataCopyPad(packed, packedValues_[static_cast<uint64_t>(batch.row_id_begin) *
            SPMM_FP32_REGBASE_DEGREE], copy, noPad);
        valueQueue_.EnQue(packed);
    }

    __aicore__ inline void LoadBatchMetadata(
        const SpmmFp32RegBaseBatchMeta &batch, LocalTensor<int32_t> &localStarts,
        LocalTensor<int32_t> &rowIds)
    {
        DataCopyPadExtParams<int32_t> noPad{false, 0, 0, 0};
        DataCopyExtParams copy{1, static_cast<uint32_t>(batch.row_count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(localStarts, localStarts_[batch.row_id_begin], copy, noPad);
        SetFlag<HardEvent::MTE2_V>(localStartsReady_);
        DataCopyPad(rowIds, rowIds_[batch.row_id_begin], copy, noPad);
        SetFlag<HardEvent::MTE2_S>(rowIdsReady_);
    }

    __aicore__ inline void StoreBatchOutput(const SpmmFp32RegBaseBatchMeta &batch,
        LocalTensor<int32_t> &rowIds, LocalTensor<float> &accumulator)
    {
        DataCopyExtParams copy{1, static_cast<uint32_t>(SPMM_FP32_REGBASE_N * sizeof(float)), 0, 0, 0};
        SetAtomicAdd<float>();
        for (int32_t localRow = 0; localRow < batch.row_count; ++localRow) {
            const int32_t outputRow = rowIds.GetValue(localRow);
            DataCopyPad(matC_[static_cast<uint64_t>(outputRow) * td_.ldc],
                accumulator[localRow * SPMM_FP32_REGBASE_N], copy);
        }
        SetAtomicNone();
        event_t done = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(done);
        WaitFlag<HardEvent::MTE3_V>(done);
    }

    __aicore__ inline void ProcessBatch(int32_t batchIndex)
    {
        SpmmFp32RegBaseBatchMeta batch{};
        batch.window_begin = batches_[batchIndex].window_begin;
        batch.row_count = batches_[batchIndex].row_count;
        batch.row_id_begin = batches_[batchIndex].row_id_begin;
        batch.window_rows = batches_[batchIndex].window_rows;
        if (!IsValidBatch(batch)) {
            return;
        }

        const uint32_t windowElements = static_cast<uint32_t>(
            batch.window_rows * SPMM_FP32_REGBASE_N);

        LocalTensor<float> accumulator =
            accumulatorQueue_.AllocTensor<float>();
        accumulatorQueue_.EnQue(accumulator);

        LocalTensor<float> window = windowQueue_.AllocTensor<float>();
        QueueBatchWindow(batch, windowElements, window);

        LocalTensor<float> packed = valueQueue_.AllocTensor<float>();
        QueueBatchValues(batch, packed);

        LocalTensor<int32_t> localStarts =
            localStartsBuffer_.Get<int32_t>();
        LocalTensor<int32_t> rowIds = rowIdsBuffer_.Get<int32_t>();
        LoadBatchMetadata(batch, localStarts, rowIds);

        accumulator = accumulatorQueue_.DeQue<float>();
        window = windowQueue_.DeQue<float>();
        packed = valueQueue_.DeQue<float>();
        WaitFlag<HardEvent::MTE2_V>(localStartsReady_);
        SpmmFp32RegBaseMac(
            reinterpret_cast<__ubuf__ float *>(accumulator.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(window.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(packed.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t *>(localStarts.GetPhyAddr()),
            static_cast<uint16_t>(batch.row_count));

        event_t vectorToMte3 = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vectorToMte3);
        WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
        WaitFlag<HardEvent::MTE2_S>(rowIdsReady_);

        StoreBatchOutput(batch, rowIds, accumulator);

        valueQueue_.FreeTensor(packed);
        windowQueue_.FreeTensor(window);
        accumulatorQueue_.FreeTensor(accumulator);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> accumulatorQueue_;
    TQue<TPosition::VECIN, 1> windowQueue_;
    TQue<TPosition::VECIN, 1> valueQueue_;
    TBuf<TPosition::VECCALC> rowIdsBuffer_;
    TBuf<TPosition::VECCALC> localStartsBuffer_;
    GlobalTensor<float> matB_;
    GlobalTensor<float> matC_;
    GlobalTensor<float> packedValues_;
    GlobalTensor<int32_t> rowIds_;
    GlobalTensor<int32_t> localStarts_;
    __gm__ const SpmmFp32RegBaseHeader *header_{nullptr};
    __gm__ const SpmmFp32RegBaseBatchMeta *batches_{nullptr};
    event_t localStartsReady_;
    event_t rowIdsReady_;
    SpmmKernelTilingData td_{};
};

template<int32_t StructuralDegree, int32_t TileN>
__simd_vf__ inline void SpmmFp32RegMac(
    __ubuf__ float *accumulator, __ubuf__ float *window,
    __ubuf__ float *packedValues, __ubuf__ int32_t *rowMeta,
    uint16_t rowCount, uint16_t count)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> sum;
    RegTensor<float> dense;
    RegTensor<float> scalar;
    for (uint16_t row = 0; row < rowCount; ++row) {
        const int32_t localStart = rowMeta[row * 2];
        for (uint16_t column = 0; column < count; column += 64) {
            uint32_t maskCount = count - column < 64 ?
                count - column : 64;
            MaskReg mask = UpdateMask<float>(maskCount);
            LoadAlign(
                sum,
                accumulator + row * TileN + column);
#pragma unroll
            for (int32_t ordinal = 0;
                 ordinal < StructuralDegree; ++ordinal) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(
                    scalar, packedValues + row * StructuralDegree + ordinal);
                LoadAlign(
                    dense,
                    window + (localStart + ordinal) * TileN + column);
                MulAddDst(sum, dense, scalar, mask);
            }
            StoreAlign<float, StoreDist::DIST_NORM_B32>(
                accumulator + row * TileN + column,
                sum, mask);
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmFp32Degree2Refresh(
    __gm__ const int32_t *rowMeta, __gm__ const float *values,
    __gm__ float *packedValues, uint64_t first, uint64_t end)
{
    for (uint64_t work = first + static_cast<uint64_t>(threadIdx.x);
         work < end; work += static_cast<uint64_t>(blockDim.x)) {
        const uint64_t row = work / SPMM_FP32_DEGREE2_DEGREE;
        const uint64_t ordinal = work % SPMM_FP32_DEGREE2_DEGREE;
        const int32_t valueOffset = rowMeta[row * 2 + 1];
        packedValues[work] = values[valueOffset + ordinal];
    }
}

class KernelSpmmFp32Degree2Compute : public SpmmFp32RegBaseSharedState<KernelSpmmFp32Degree2Compute> {
public:
    __aicore__ inline void Init(
        GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
        GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        InitializeSpmmRegBaseTensors(matB, initialC, matC, workspaceGM,
            tilingGM, &td_, matB_, initialC_, matC_, &batches_, rowIds_,
            rowMeta_, packedValues_);
        tileCount_ = (td_.n + SPMM_FP32_DEGREE2_TILE_N - 1) /
            SPMM_FP32_DEGREE2_TILE_N;
        pipe_.InitBuffer(bQueue_, 1,
            SPMM_FP32_DEGREE2_WINDOW_ROWS * SPMM_FP32_DEGREE2_TILE_N *
                sizeof(float));
        pipe_.InitBuffer(accumulatorQueue_, 1,
            SPMM_FP32_DEGREE2_ROWS * SPMM_FP32_DEGREE2_TILE_N *
                sizeof(float));
        pipe_.InitBuffer(valueQueue_, 1,
            SPMM_FP32_DEGREE2_ROWS * SPMM_FP32_DEGREE2_DEGREE *
                sizeof(float));
        pipe_.InitBuffer(rowMetaBuffer_,
            SPMM_FP32_DEGREE2_ROWS * 2 * sizeof(int32_t));
        pipe_.InitBuffer(rowIdsBuffer_,
            SPMM_FP32_DEGREE2_ROWS * sizeof(int32_t));
        InitializeSpmmFp32RegBaseEvents(
            pipe_, &rowMetaMte2ToVector_, &rowIdsMte2ToScalar_,
            &accumulatorMte3ToVector_, &accumulatorVectorToMte3_);
    }

    __aicore__ inline void Process()
    {
        RunTasks();
    }

private:
    __aicore__ inline void LoadTileMetadata(
        const SpmmFp32Degree2BatchMeta &batch, LocalTensor<int32_t> &rowMeta,
        LocalTensor<int32_t> &rowIds)
    {
        DataCopyPadExtParams<int32_t> noPad{false, 0, 0, 0};
        DataCopyExtParams metaCopy{1, static_cast<uint32_t>(batch.row_count * 2 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(rowMeta, rowMeta_[static_cast<uint64_t>(batch.row_meta_begin) * 2], metaCopy, noPad);
        SetFlag<HardEvent::MTE2_V>(rowMetaMte2ToVector_);
        DataCopyExtParams idCopy{1, static_cast<uint32_t>(batch.row_count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(rowIds, rowIds_[batch.row_meta_begin], idCopy, noPad);
        SetFlag<HardEvent::MTE2_S>(rowIdsMte2ToScalar_);
    }

    __aicore__ inline void QueueTileValues(
        const SpmmFp32Degree2BatchMeta &batch, LocalTensor<float> &packed)
    {
        DataCopyExtParams copy{1, static_cast<uint32_t>(batch.row_count *
            SPMM_FP32_DEGREE2_DEGREE * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
        DataCopyPad(packed, packedValues_[static_cast<uint64_t>(batch.row_meta_begin) *
            SPMM_FP32_DEGREE2_DEGREE], copy, noPad);
        valueQueue_.EnQue(packed);
    }

    __aicore__ inline void QueueTileAccumulator(const SpmmFp32Degree2BatchMeta &batch,
        LocalTensor<int32_t> &rowIds, LocalTensor<float> &accumulator,
        int32_t tileBegin, int32_t tileCount, const DataCopyPadExtParams<float> &noPad)
    {
        DataCopyExtParams copy{1, static_cast<uint32_t>(tileCount * sizeof(float)), 0, 0, 0};
        for (int32_t row = 0; row < batch.row_count; ++row) {
            DataCopyPad(accumulator[row * SPMM_FP32_DEGREE2_TILE_N],
                initialC_[static_cast<uint64_t>(rowIds.GetValue(row)) * td_.ldc + tileBegin], copy, noPad);
        }
        accumulatorQueue_.EnQue(accumulator);
    }

    __aicore__ inline void QueueTileWindow(const SpmmFp32Degree2BatchMeta &batch,
        LocalTensor<float> &window, int32_t tileBegin, int32_t tileCount,
        bool wrapBatch, int32_t wrapK, const DataCopyPadExtParams<float> &noPad)
    {
        const uint32_t bytes = static_cast<uint32_t>(tileCount * sizeof(float));
        const uint32_t padded = (bytes + 31U) / 32U * 32U;
        DataCopyExtParams copy{static_cast<uint16_t>(batch.window_rows), bytes,
            static_cast<int64_t>((td_.ldb - tileCount) * sizeof(float)),
            static_cast<int64_t>((SPMM_FP32_DEGREE2_TILE_N * sizeof(float) - padded) / 32U), 0};
        if (wrapBatch) {
            DataCopyExtParams single{1, bytes, 0, 0, 0};
            DataCopyPad(window, matB_[tileBegin], single, noPad);
            DataCopyPad(window[SPMM_FP32_DEGREE2_TILE_N],
                matB_[static_cast<uint64_t>(wrapK - 1) * td_.ldb + tileBegin], single, noPad);
        } else DataCopyPad(window, matB_[static_cast<uint64_t>(batch.window_begin) * td_.ldb + tileBegin], copy, noPad);
        bQueue_.EnQue(window);
    }

    __aicore__ inline void ComputeAndStoreTile(
        const SpmmFp32Degree2BatchMeta &batch, LocalTensor<int32_t> &rowMeta,
        LocalTensor<int32_t> &rowIds, LocalTensor<float> &packed,
        int32_t tileBegin, int32_t tileCount,
        const DataCopyPadExtParams<float> &noPad)
    {
        LocalTensor<float> accumulator = accumulatorQueue_.AllocTensor<float>();
        QueueTileAccumulator(batch, rowIds, accumulator, tileBegin, tileCount,
            noPad);
        LocalTensor<float> window = bQueue_.AllocTensor<float>();
        const bool wrapBatch = batch.window_begin < 0;
        const int32_t wrapK = wrapBatch ? -batch.window_begin : 0;
        QueueTileWindow(batch, window, tileBegin, tileCount, wrapBatch, wrapK,
            noPad);
        window = bQueue_.DeQue<float>();
        accumulator = accumulatorQueue_.DeQue<float>();
        SpmmFp32RegMac<SPMM_FP32_DEGREE2_DEGREE,
            SPMM_FP32_DEGREE2_TILE_N>(
            reinterpret_cast<__ubuf__ float *>(accumulator.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(window.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(packed.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t *>(rowMeta.GetPhyAddr()),
            static_cast<uint16_t>(batch.row_count),
            static_cast<uint16_t>(tileCount));
        SetFlag<HardEvent::V_MTE3>(accumulatorVectorToMte3_);
        WaitFlag<HardEvent::V_MTE3>(accumulatorVectorToMte3_);
        DataCopyExtParams outputCopy{1,
            static_cast<uint32_t>(tileCount * sizeof(float)), 0, 0, 0};
        for (int32_t row = 0; row < batch.row_count; ++row) {
            const int32_t outputRow = rowIds.GetValue(row);
            DataCopyPad(matC_[static_cast<uint64_t>(outputRow) * td_.ldc +
                tileBegin], accumulator[row * SPMM_FP32_DEGREE2_TILE_N],
                outputCopy);
        }
        FinishTile(accumulator, window, packed);
    }

public:
    __aicore__ inline void ProcessTile(
        int32_t groupIndex, int32_t tileIndex)
    {
        const SpmmFp32Degree2BatchMeta batch =
            LoadSpmmRegBaseBatch(batches_, groupIndex);
        const bool wrapBatch = batch.window_begin < 0;
        const int32_t wrapK = wrapBatch ? -batch.window_begin : 0;
        if ((wrapBatch && (wrapK <= 1 ||
                           batch.window_rows != SPMM_FP32_DEGREE2_DEGREE)) ||
            batch.row_count <= 0 ||
            batch.row_count > SPMM_FP32_DEGREE2_ROWS ||
            batch.row_meta_begin < 0 || batch.window_rows <= 0 ||
            batch.window_rows > SPMM_FP32_DEGREE2_WINDOW_ROWS) {
            return;
        }
        const int32_t tileBegin = tileIndex * SPMM_FP32_DEGREE2_TILE_N;
        const int32_t tileCount = td_.n - tileBegin <
            SPMM_FP32_DEGREE2_TILE_N ? td_.n - tileBegin :
            SPMM_FP32_DEGREE2_TILE_N;
        if (tileCount <= 0) {
            return;
        }

        LocalTensor<int32_t> rowMeta = rowMetaBuffer_.Get<int32_t>();
        LocalTensor<int32_t> rowIds = rowIdsBuffer_.Get<int32_t>();
        LoadTileMetadata(batch, rowMeta, rowIds);

        LocalTensor<float> packed = valueQueue_.AllocTensor<float>();
        DataCopyPadExtParams<float> noFloatPad{false, 0, 0, 0};
        QueueTileValues(batch, packed);
        packed = valueQueue_.DeQue<float>();
        WaitFlag<HardEvent::MTE2_V>(rowMetaMte2ToVector_);
        WaitFlag<HardEvent::MTE2_S>(rowIdsMte2ToScalar_);

        ComputeAndStoreTile(batch, rowMeta, rowIds, packed, tileBegin,
            tileCount, noFloatPad);
    }

};

template<int32_t StructuralDegree>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void
SpmmFp32Degree4Refresh(
    __gm__ const int32_t *rowMeta, __gm__ const float *values,
    __gm__ float *packedValues, uint64_t first, uint64_t end)
{
    for (uint64_t work = first + static_cast<uint64_t>(threadIdx.x);
         work < end; work += static_cast<uint64_t>(blockDim.x)) {
        const uint64_t row = work / StructuralDegree;
        const uint64_t ordinal = work % StructuralDegree;
        const int32_t valueOffset = rowMeta[row * 2 + 1];
        packedValues[work] = values[valueOffset + ordinal];
    }
}

template<int32_t StructuralDegree, int32_t Rows, int32_t TileN,
         int32_t WindowRows>
class KernelSpmmFp32Degree4Compute : public SpmmFp32RegBaseSharedState<
    KernelSpmmFp32Degree4Compute<StructuralDegree, Rows, TileN, WindowRows>> {
public:
    __aicore__ inline void Init(
        GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
        GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        InitializeSpmmRegBaseTensors(matB, initialC, matC, workspaceGM,
            tilingGM, &this->td_, this->matB_, this->initialC_, this->matC_,
            &this->batches_, this->rowIds_, this->rowMeta_, this->packedValues_);
        this->tileCount_ = (this->td_.n + TileN - 1) / TileN;
        this->pipe_.InitBuffer(this->bQueue_, 1,
            WindowRows * TileN * sizeof(float));
        this->pipe_.InitBuffer(this->accumulatorQueue_, 1,
            Rows * TileN * sizeof(float));
        this->pipe_.InitBuffer(this->valueQueue_, 1,
            Rows * StructuralDegree * sizeof(float));
        this->pipe_.InitBuffer(this->rowMetaBuffer_,
            Rows * 2 * sizeof(int32_t));
        this->pipe_.InitBuffer(this->rowIdsBuffer_,
            Rows * sizeof(int32_t));
        InitializeSpmmFp32RegBaseEvents(
            this->pipe_, &this->rowMetaMte2ToVector_, &this->rowIdsMte2ToScalar_,
            &this->accumulatorMte3ToVector_, &this->accumulatorVectorToMte3_);
        metadataVectorToMte2_ = static_cast<event_t>(
            this->pipe_.FetchEventID(HardEvent::V_MTE2));
    }

    __aicore__ inline void Process()
    {
        this->RunTasks();
    }

private:
    event_t metadataVectorToMte2_;

    __aicore__ inline bool IsValidBatch(
        const SpmmFp32Degree2BatchMeta &batch, bool wrapBatch,
        int32_t wrapHighStart, int32_t wrapLowRows, int32_t wrapHighRows)
    {
        return this->td_.uniform_degree == StructuralDegree &&
            (!wrapBatch || (wrapHighStart > 0 && wrapLowRows > 0 &&
                            wrapHighRows > 0)) &&
            batch.row_count > 0 && batch.row_count <= Rows &&
            batch.row_meta_begin >= 0 && batch.window_rows > 0 &&
            batch.window_rows <= WindowRows;
    }

    __aicore__ inline void LoadTileMetadata(
        const SpmmFp32Degree2BatchMeta &batch, LocalTensor<int32_t> &rowMeta,
        LocalTensor<int32_t> &rowIds)
    {
        DataCopyPadExtParams<int32_t> noPad{false, 0, 0, 0};
        DataCopyExtParams metaCopy{1,
            static_cast<uint32_t>(batch.row_count * 2 * sizeof(int32_t)),
            0, 0, 0};
        DataCopyPad(rowMeta,
            this->rowMeta_[static_cast<uint64_t>(batch.row_meta_begin) * 2],
            metaCopy, noPad);
        SetFlag<HardEvent::MTE2_V>(this->rowMetaMte2ToVector_);
        DataCopyExtParams idCopy{1,
            static_cast<uint32_t>(batch.row_count * sizeof(int32_t)),
            0, 0, 0};
        DataCopyPad(rowIds, this->rowIds_[batch.row_meta_begin], idCopy, noPad);
        SetFlag<HardEvent::MTE2_S>(this->rowIdsMte2ToScalar_);
    }

    __aicore__ inline void QueueTileValues(
        const SpmmFp32Degree2BatchMeta &batch, LocalTensor<float> &packed,
        const DataCopyPadExtParams<float> &noPad)
    {
        DataCopyExtParams copy{1, static_cast<uint32_t>(batch.row_count *
            StructuralDegree * sizeof(float)), 0, 0, 0};
        DataCopyPad(packed,
            this->packedValues_[static_cast<uint64_t>(batch.row_meta_begin) *
                          StructuralDegree], copy, noPad);
        this->valueQueue_.EnQue(packed);
    }

    __aicore__ inline void QueueTileAccumulator(
        const SpmmFp32Degree2BatchMeta &batch, LocalTensor<int32_t> &rowIds,
        LocalTensor<float> &accumulator, int32_t tileBegin,
        int32_t tileElements, const DataCopyPadExtParams<float> &noPad)
    {
        DataCopyExtParams copy{1,
            static_cast<uint32_t>(tileElements * sizeof(float)), 0, 0, 0};
        for (int32_t row = 0; row < batch.row_count; ++row) {
            const int32_t outputRow = rowIds.GetValue(row);
            DataCopyPad(accumulator[row * TileN],
                this->initialC_[static_cast<uint64_t>(outputRow) * this->td_.ldc +
                          tileBegin], copy, noPad);
        }
        this->accumulatorQueue_.EnQue(accumulator);
    }

    __aicore__ inline void QueueTileWindow(
        const SpmmFp32Degree2BatchMeta &batch, LocalTensor<float> &window,
        int32_t tileBegin, int32_t tileElements, bool wrapBatch,
        int32_t wrapHighStart, int32_t wrapLowRows, int32_t wrapHighRows,
        const DataCopyPadExtParams<float> &noPad)
    {
        const uint32_t bytes = static_cast<uint32_t>(tileElements * sizeof(float));
        const uint32_t padded = (bytes + 31U) / 32U * 32U;
        const int64_t sourceGap =
            static_cast<int64_t>((this->td_.ldb - tileElements) * sizeof(float));
        const int64_t destinationGap =
            static_cast<int64_t>((TileN * sizeof(float) - padded) / 32U);
        if (wrapBatch) {
            DataCopyExtParams lowCopy{static_cast<uint16_t>(wrapLowRows),
                bytes, sourceGap, destinationGap, 0};
            DataCopyPad(window, this->matB_[tileBegin], lowCopy, noPad);
            DataCopyExtParams highCopy{static_cast<uint16_t>(wrapHighRows),
                bytes, sourceGap, destinationGap, 0};
            DataCopyPad(window[wrapLowRows * TileN],
                this->matB_[static_cast<uint64_t>(wrapHighStart) * this->td_.ldb +
                      tileBegin], highCopy, noPad);
        } else {
            DataCopyExtParams copy{static_cast<uint16_t>(batch.window_rows),
                bytes, sourceGap, destinationGap, 0};
            DataCopyPad(window,
                this->matB_[static_cast<uint64_t>(batch.window_begin) * this->td_.ldb +
                      tileBegin], copy, noPad);
        }
        this->bQueue_.EnQue(window);
    }

    __aicore__ inline void ComputeTile(
        const SpmmFp32Degree2BatchMeta &batch, LocalTensor<float> &accumulator,
        LocalTensor<float> &window, LocalTensor<float> &packed,
        LocalTensor<int32_t> &rowMeta, int32_t tileElements)
    {
        SpmmFp32RegMac<StructuralDegree, TileN>(
            reinterpret_cast<__ubuf__ float *>(accumulator.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(window.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(packed.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t *>(rowMeta.GetPhyAddr()),
            static_cast<uint16_t>(batch.row_count),
            static_cast<uint16_t>(tileElements));
    }

    __aicore__ inline void StoreTileOutput(
        const SpmmFp32Degree2BatchMeta &batch, LocalTensor<int32_t> &rowIds,
        LocalTensor<float> &accumulator, int32_t tileBegin,
        int32_t tileElements)
    {
        DataCopyExtParams copy{1,
            static_cast<uint32_t>(tileElements * sizeof(float)), 0, 0, 0};
        for (int32_t row = 0; row < batch.row_count; ++row) {
            const int32_t outputRow = rowIds.GetValue(row);
            DataCopyPad(this->matC_[static_cast<uint64_t>(outputRow) * this->td_.ldc +
                               tileBegin], accumulator[row * TileN], copy);
        }
    }

    __aicore__ inline void ProcessValidatedTile(
        const SpmmFp32Degree2BatchMeta &batch, int32_t tileBegin,
        int32_t tileElements, bool wrapBatch, int32_t wrapHighStart,
        int32_t wrapLowRows, int32_t wrapHighRows)
    {
        LocalTensor<int32_t> rowMeta = this->rowMetaBuffer_.template Get<int32_t>();
        LocalTensor<int32_t> rowIds = this->rowIdsBuffer_.template Get<int32_t>();
        LoadTileMetadata(batch, rowMeta, rowIds);
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
        LocalTensor<float> packed = this->valueQueue_.template AllocTensor<float>();
        QueueTileValues(batch, packed, noPad);
        packed = this->valueQueue_.template DeQue<float>();
        WaitFlag<HardEvent::MTE2_V>(this->rowMetaMte2ToVector_);
        WaitFlag<HardEvent::MTE2_S>(this->rowIdsMte2ToScalar_);
        LocalTensor<float> accumulator = this->accumulatorQueue_.template AllocTensor<float>();
        QueueTileAccumulator(batch, rowIds, accumulator, tileBegin,
            tileElements, noPad);
        LocalTensor<float> window = this->bQueue_.template AllocTensor<float>();
        QueueTileWindow(batch, window, tileBegin, tileElements, wrapBatch,
            wrapHighStart, wrapLowRows, wrapHighRows, noPad);
        window = this->bQueue_.template DeQue<float>();
        accumulator = this->accumulatorQueue_.template DeQue<float>();
        ComputeTile(batch, accumulator, window, packed, rowMeta, tileElements);
        // Keep the next task's metadata DMA behind this VF's TBuf reads.
        SetFlag<HardEvent::V_MTE2>(metadataVectorToMte2_);
        WaitFlag<HardEvent::V_MTE2>(metadataVectorToMte2_);
        SetFlag<HardEvent::V_MTE3>(this->accumulatorVectorToMte3_);
        WaitFlag<HardEvent::V_MTE3>(this->accumulatorVectorToMte3_);
        StoreTileOutput(batch, rowIds, accumulator, tileBegin, tileElements);
        this->FinishTile(accumulator, window, packed);
    }

public:
    __aicore__ inline void ProcessTile(
        int32_t groupIndex, int32_t tileIndex)
    {
        const SpmmFp32Degree2BatchMeta batch =
            LoadSpmmRegBaseBatch(this->batches_, groupIndex);
        const bool wrapBatch = batch.window_begin < 0;
        const int32_t wrapHighStart = wrapBatch ?
            -batch.window_begin - 1 : 0;
        const int32_t wrapLowRows = wrapBatch ? batch.window_rows : 0;
        const int32_t wrapHighRows = wrapBatch ?
            StructuralDegree - wrapLowRows : 0;
        if (!IsValidBatch(batch, wrapBatch, wrapHighStart, wrapLowRows,
                          wrapHighRows)) {
            return;
        }
        const int32_t tileBegin = tileIndex * TileN;
        const int32_t tileElements = this->td_.n - tileBegin <
            TileN ? this->td_.n - tileBegin : TileN;
        if (tileElements <= 0) {
            return;
        }

        ProcessValidatedTile(batch, tileBegin, tileElements, wrapBatch,
            wrapHighStart, wrapLowRows, wrapHighRows);
    }
};

extern "C" __global__ __aicore__ void spmm_custom_fp32_regbase_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    __gm__ uint8_t *workspace =
        reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
    __gm__ const SpmmFp32RegBaseHeader *header =
        reinterpret_cast<__gm__ const SpmmFp32RegBaseHeader *>(
            workspace + td.cube_scratch_offset);
    if (header->refresh_run_count > 0) {
        KernelSpmmFp32RegBaseDmaRefresh op;
        op.Init(csrValues, workspaceGM, tilingGM);
        op.Process();
        return;
    }
    __gm__ const int32_t *slot1 = reinterpret_cast<__gm__ const int32_t *>(
        workspace + td.cube_slot1_offset);
    __gm__ const int32_t *valueBegins = slot1 + header->fallback_count;
    __gm__ const int32_t *degrees = valueBegins + header->candidate_count;
    const uint64_t total = static_cast<uint64_t>(header->candidate_count) *
        SPMM_FP32_REGBASE_DEGREE;
    const SpmmBlockWorkRange range = GetSpmmBlockWorkRange(total);
    const uint64_t first = range.first;
    const uint64_t end = range.end;
    asc_vf_call<SpmmFp32RegBaseRefresh>(dim3{kMaxSimtThreadsPerBlock},
        reinterpret_cast<__gm__ const float *>(csrValues),
        reinterpret_cast<__gm__ float *>(
            workspace + td.cube_packed_a_offset),
        valueBegins, degrees, first, end);
}
extern "C" __global__ __aicore__ void spmm_custom_fp32_degree2_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    __gm__ const SpmmFp32RegBaseHeader *header = nullptr;
    __gm__ const int32_t *rowMeta = nullptr;
    __gm__ float *packedValues = nullptr;
    InitializeSpmmRegBaseRefreshData(workspaceGM, tilingGM, &header, &rowMeta,
        &packedValues);
    const uint64_t total = static_cast<uint64_t>(
        header->refresh_run_count) * SPMM_FP32_DEGREE2_DEGREE;
    const SpmmBlockWorkRange range = GetSpmmBlockWorkRange(total);
    const uint64_t first = range.first;
    const uint64_t end = range.end;
    asc_vf_call<SpmmFp32Degree2Refresh>(dim3{kMaxSimtThreadsPerBlock},
        rowMeta, reinterpret_cast<__gm__ const float *>(csrValues),
        packedValues, first, end);
}
extern "C" __global__ __aicore__ void spmm_custom_fp32_degree2_compute(
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    KernelSpmmFp32Degree2Compute op;
    op.Init(matB, initialC, matC, workspaceGM, tilingGM);
    op.Process();
}

template <int32_t Degree>
__aicore__ inline void LaunchFp32DegreeRefresh(
    __gm__ const int32_t *rowMeta, GM_ADDR csrValues,
    __gm__ float *packedValues, int32_t refreshRunCount)
{
    const uint64_t total = static_cast<uint64_t>(refreshRunCount) * Degree;
    const SpmmBlockWorkRange range = GetSpmmBlockWorkRange(total);
    asc_vf_call<SpmmFp32Degree4Refresh<Degree>>(dim3{kMaxSimtThreadsPerBlock},
        rowMeta, reinterpret_cast<__gm__ const float *>(csrValues), packedValues,
        range.first, range.end);
}

extern "C" __global__ __aicore__ void spmm_custom_fp32_degree4_refresh(
    GM_ADDR csrValues, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    __gm__ const SpmmFp32RegBaseHeader *header = nullptr;
    __gm__ const int32_t *rowMeta = nullptr;
    __gm__ float *packedValues = nullptr;
    const SpmmKernelTilingData td = InitializeSpmmRegBaseRefreshData(
        workspaceGM, tilingGM, &header, &rowMeta, &packedValues);
    if (td.uniform_degree == SPMM_FP32_DEGREE4_DEGREE) {
        LaunchFp32DegreeRefresh<SPMM_FP32_DEGREE4_DEGREE>(
            rowMeta, csrValues, packedValues, header->refresh_run_count);
    } else if (td.uniform_degree == SPMM_FP32_DEGREE4_MAX_DEGREE) {
        LaunchFp32DegreeRefresh<SPMM_FP32_DEGREE4_MAX_DEGREE>(
            rowMeta, csrValues, packedValues, header->refresh_run_count);
    } else if (td.uniform_degree == SPMM_FP32_DEGREE16_DEGREE) {
        LaunchFp32DegreeRefresh<SPMM_FP32_DEGREE16_DEGREE>(
            rowMeta, csrValues, packedValues, header->refresh_run_count);
    } else if (td.uniform_degree == SPMM_FP32_DEGREE32_DEGREE) {
        LaunchFp32DegreeRefresh<SPMM_FP32_DEGREE32_DEGREE>(
            rowMeta, csrValues, packedValues, header->refresh_run_count);
    } else if (td.uniform_degree == SPMM_FP32_DEGREE64_DEGREE) {
        LaunchFp32DegreeRefresh<SPMM_FP32_DEGREE64_DEGREE>(
            rowMeta, csrValues, packedValues, header->refresh_run_count);
    }
}
extern "C" __global__ __aicore__ void spmm_custom_fp32_degree4_compute(
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    if (td.uniform_degree == SPMM_FP32_DEGREE4_DEGREE) {
        KernelSpmmFp32Degree4Compute<
            SPMM_FP32_DEGREE4_DEGREE, SPMM_FP32_DEGREE4_ROWS,
            SPMM_FP32_DEGREE4_TILE_N,
            SPMM_FP32_DEGREE4_WINDOW_ROWS> op;
        op.Init(matB, initialC, matC, workspaceGM, tilingGM);
        op.Process();
    } else if (td.uniform_degree == SPMM_FP32_DEGREE4_MAX_DEGREE) {
        KernelSpmmFp32Degree4Compute<
            SPMM_FP32_DEGREE4_MAX_DEGREE, SPMM_FP32_DEGREE4_ROWS,
            SPMM_FP32_DEGREE4_TILE_N,
            SPMM_FP32_DEGREE4_WINDOW_ROWS> op;
        op.Init(matB, initialC, matC, workspaceGM, tilingGM);
        op.Process();
    } else if (td.uniform_degree == SPMM_FP32_DEGREE16_DEGREE) {
        KernelSpmmFp32Degree4Compute<
            SPMM_FP32_DEGREE16_DEGREE, SPMM_FP32_DEGREE4_ROWS,
            SPMM_FP32_DEGREE4_TILE_N,
            SPMM_FP32_DEGREE4_WINDOW_ROWS> op;
        op.Init(matB, initialC, matC, workspaceGM, tilingGM);
        op.Process();
    } else if (td.uniform_degree == SPMM_FP32_DEGREE32_DEGREE) {
        KernelSpmmFp32Degree4Compute<
            SPMM_FP32_DEGREE32_DEGREE, SPMM_FP32_DEGREE32_ROWS,
            SPMM_FP32_DEGREE32_TILE_N,
            SPMM_FP32_DEGREE32_WINDOW_ROWS> op;
        op.Init(matB, initialC, matC, workspaceGM, tilingGM);
        op.Process();
    } else if (td.uniform_degree == SPMM_FP32_DEGREE64_DEGREE) {
        KernelSpmmFp32Degree4Compute<
            SPMM_FP32_DEGREE64_DEGREE, SPMM_FP32_DEGREE64_ROWS,
            SPMM_FP32_DEGREE64_TILE_N,
            SPMM_FP32_DEGREE64_WINDOW_ROWS> op;
        op.Init(matB, initialC, matC, workspaceGM, tilingGM);
        op.Process();
    }
}
extern "C" __global__ __aicore__ void spmm_custom_fp32_regbase_compute(
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    KernelSpmmFp32RegBaseCompute op;
    op.Init(matB, matC, workspaceGM, tilingGM);
    op.Process();
}

extern "C" __global__ __aicore__ void spmm_custom_fp32_regbase_fallback(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR initialC, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!IsSpmmTilingVersionSupported(tilingGM)) return;
    const SpmmKernelTilingData td = LoadSpmmKernelTilingData(tilingGM);
    __gm__ uint8_t *workspace =
        reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
    __gm__ const SpmmFp32RegBaseHeader *header =
        reinterpret_cast<__gm__ const SpmmFp32RegBaseHeader *>(
            workspace + td.cube_scratch_offset);
    const uint64_t total = static_cast<uint64_t>(header->fallback_count) *
        static_cast<uint32_t>(td.n);
    const uint64_t first = total * GetBlockIdx() / GetBlockNum();
    const uint64_t end = total * (GetBlockIdx() + 1) / GetBlockNum();
    __gm__ const int32_t *fallbackRows =
        (td.cube_rows == SPMM_FP32_DEGREE2_ROWS ||
         td.cube_rows == SPMM_FP32_DEGREE4_ROWS ||
         td.cube_rows == SPMM_FP32_DEGREE32_ROWS ||
         td.cube_rows == SPMM_FP32_DEGREE64_MODE) ?
            reinterpret_cast<__gm__ const int32_t *>(workspace +
                td.cube_scratch_offset + sizeof(SpmmFp32RegBaseHeader)) :
            reinterpret_cast<__gm__ const int32_t *>(
                workspace + td.cube_slot1_offset);
    if (header->fallback_count > 0) {
        asc_vf_call<SpmmFp32RegBaseFallback>(dim3{kMaxSimtThreadsPerBlock},
            reinterpret_cast<__gm__ const int32_t *>(csrRowOffsets),
            reinterpret_cast<__gm__ const int32_t *>(csrColInd),
            reinterpret_cast<__gm__ const float *>(csrValues),
            reinterpret_cast<__gm__ const float *>(matB),
            reinterpret_cast<__gm__ const float *>(initialC),
            reinterpret_cast<__gm__ float *>(matC), fallbackRows,
            first, end, td.n, td.ldb, td.ldc, td.index_base);
    }
}
