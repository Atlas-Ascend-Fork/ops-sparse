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

/*!
 * \file gtsv2_strided_batch_kernel.cpp
 * \brief aclsparseSgtsv2StridedBatch Kernel 实现（CR 紧凑化变体，SIMD AIV only）。
 *
 * 算法：Cyclic Reduction (CR) 紧凑化变体，两级架构
 *   - 内层 UB CR（mPad <= 2048）：正向逐层消元紧凑化，反向逐层回代展开（完全复用现有实现）
 *   - 外层 GM 分块 CR（mPad > 2048）：K = L - 11 层，每层在 GM 中流式逐 tile 处理，
 *     归约到 2048 后交由内层 UB CR，反向对称展开
 *
 * 数据流：
 *   MTE2 (DataCopy GM→UB) → Vector (CR 计算) → MTE3 (DataCopy UB→GM)
 *   跨 Pipe 同步使用 TQue EnQue/DeQue，Vector 内部同步使用 PipeBarrier<PIPE_VECTOR>
 *
 * 性能优化：
 *   gather/scatter 操作使用 Vector Pipe 的 Gather/Scatter API 替代 SetValue/GetValue 标量循环。
 *   偏移表在 Init 阶段一次性预计算，跨所有 batch 复用，将 Scalar Pipe 开销摊薄到可忽略。
 */

#include <cstdint>
#include "kernel_operator.h"
#include "kernel_operator_vec_binary_intf.h"
#include "basic_api/kernel_operator_vec_gather_intf.h"
#include "basic_api/kernel_operator_vec_scatter_intf.h"
// 提供 DivConfig 与 DEFAULT_DIV_CONFIG，二者仅定义于此头文件，未被上方 vec_binary_intf 间接引入
#include "basic_api/reg_compute/kernel_reg_compute_utils.h"
#include "gtsv2_strided_batch_tiling_data.h"
#include "gtsv2_strided_batch_kernel.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
static constexpr DivConfig kDivConfig = DEFAULT_DIV_CONFIG;
static constexpr int32_t kDataBlockSize = 8;

class Gtsv2StridedBatchKernel {
public:
    __aicore__ inline Gtsv2StridedBatchKernel() {}

    // mPad/alignedM read from TilingData (single source from Host)
    __aicore__ inline void InitDerived()
    {
        vecBytes_ = static_cast<uint32_t>(alignedM_) * sizeof(float);
        useVectorGather_ = (mPad_ > kDataBlockSize);
    }

    // 12 个向量工作 buffer（oddA-D、aRight-dRight、k1/k2/t1/t2）统一按 vecBytes_ 分配
    __aicore__ inline void InitVecWorkBuffers()
    {
        pipe_->InitBuffer(oddABuf_, vecBytes_);
        pipe_->InitBuffer(oddBBuf_, vecBytes_);
        pipe_->InitBuffer(oddCBuf_, vecBytes_);
        pipe_->InitBuffer(oddDBuf_, vecBytes_);

        pipe_->InitBuffer(aRightBuf_, vecBytes_);
        pipe_->InitBuffer(bRightBuf_, vecBytes_);
        pipe_->InitBuffer(cRightBuf_, vecBytes_);
        pipe_->InitBuffer(dRightBuf_, vecBytes_);

        pipe_->InitBuffer(k1Buf_, vecBytes_);
        pipe_->InitBuffer(k2Buf_, vecBytes_);
        pipe_->InitBuffer(t1Buf_, vecBytes_);
        pipe_->InitBuffer(t2Buf_, vecBytes_);
    }

    // 外层 GM 分块路径：staging 兼内层输入队列（2P+8 >= 2048），工作 buffer 按内层 2048 常量
    __aicore__ inline void InitOuterPathBuffers(uint32_t saveBufSize)
    {
        uint32_t stagingBytes = static_cast<uint32_t>(2 * outerTileElems_ + GTSV2_GUARD_FLOATS) * sizeof(float);
        pipe_->InitBuffer(aInQue_, BUFFER_NUM, stagingBytes);
        pipe_->InitBuffer(bInQue_, BUFFER_NUM, stagingBytes);
        pipe_->InitBuffer(cInQue_, BUFFER_NUM, stagingBytes);
        pipe_->InitBuffer(dInQue_, BUFFER_NUM, stagingBytes);
        pipe_->InitBuffer(dOutQue_, BUFFER_NUM, static_cast<uint32_t>(2 * outerTileElems_) * sizeof(float));

        uint32_t outerSaveBytes = static_cast<uint32_t>(outerTileElems_) * sizeof(float);
        uint32_t saveBytes = saveBufSize > outerSaveBytes ? saveBufSize : outerSaveBytes;
        pipe_->InitBuffer(aSaveBuf_, saveBytes);
        pipe_->InitBuffer(bSaveBuf_, saveBytes);
        pipe_->InitBuffer(cSaveBuf_, saveBytes);
        pipe_->InitBuffer(dSaveBuf_, saveBytes);

        InitVecWorkBuffers();

        pipe_->InitBuffer(sharedOffsetBuf1_, vecBytes_);
        pipe_->InitBuffer(sharedOffsetBuf2_, vecBytes_);
        pipe_->InitBuffer(sharedOffsetBuf3_, static_cast<uint32_t>(outerTileElems_) * sizeof(uint32_t));
        pipe_->InitBuffer(xPrevSrcBuf_, static_cast<uint32_t>(outerTileElems_ + GTSV2_GUARD_FLOATS) * sizeof(float));
        InitOffsetBuffers();
    }

    // 纯 UB 路径：buffer 尺寸与历史实现完全一致（零回归）
    __aicore__ inline void InitUbPathBuffers(uint32_t saveBufSize)
    {
        pipe_->InitBuffer(aInQue_, BUFFER_NUM, vecBytes_);
        pipe_->InitBuffer(bInQue_, BUFFER_NUM, vecBytes_);
        pipe_->InitBuffer(cInQue_, BUFFER_NUM, vecBytes_);
        pipe_->InitBuffer(dInQue_, BUFFER_NUM, vecBytes_);
        pipe_->InitBuffer(dOutQue_, BUFFER_NUM, vecBytes_);

        pipe_->InitBuffer(aSaveBuf_, saveBufSize);
        pipe_->InitBuffer(bSaveBuf_, saveBufSize);
        pipe_->InitBuffer(cSaveBuf_, saveBufSize);
        pipe_->InitBuffer(dSaveBuf_, saveBufSize);

        InitVecWorkBuffers();

        if (useVectorGather_) {
            // offset buffer 存储 uint32_t 字节偏移（sizeof(uint32_t)==sizeof(float)==4），
            pipe_->InitBuffer(sharedOffsetBuf1_, vecBytes_);
            pipe_->InitBuffer(sharedOffsetBuf2_, vecBytes_);
            InitOffsetBuffers();
        }
    }

    __aicore__ inline void InitBuffers(uint32_t saveBufSize)
    {
        if (useGmWorkspace_) {
            InitOuterPathBuffers(saveBufSize);
            return;
        }
        InitUbPathBuffers(saveBufSize);
    }

    __aicore__ inline void Init(
        GM_ADDR gmDl, GM_ADDR gmD, GM_ADDR gmDu, GM_ADDR gmX,
        GM_ADDR gmBuffer,
        const Gtsv2StridedBatchTilingData *tiling,
        TPipe *pipe)
    {
        // Host guarantees non-null; zero-init safe path
        if (tiling == nullptr || pipe == nullptr) { return; }
        pipe_ = pipe;
        gmDl_ = gmDl;
        gmD_ = gmD;
        gmDu_ = gmDu;
        gmX_ = gmX;
        gmWorkspace_ = gmBuffer;
        m_ = tiling->m;
        batchCount_ = tiling->batchCount;
        batchStride_ = tiling->batchStride;
        batchPerCoreBase_ = tiling->batchPerCoreBase;
        remainder_ = tiling->remainder;
        numLayers_ = tiling->numLayers;
        saveBufSize_ = tiling->saveBufSize;
        useGmWorkspace_ = (tiling->useGmWorkspace != 0);
        outerLevels_ = tiling->outerLevels;
        outerTileElems_ = tiling->outerTileElems;
        mPadOuter_ = tiling->mPadOuter;
        regionBByteOffset_ = tiling->regionBByteOffset;
        saveRegionByteOffset_ = tiling->saveRegionByteOffset;
        wsPerBatch_ = tiling->workspacePerBatchBytes;
        // numLayers/outerLevels guard — CalcBackwardLevelInfo 栈数组与外层层数上限双检查；
        // guard 触发时所有派生成员置安全态（initOk_ 保持 false，Process 直接早退）
        if (numLayers_ > GTSV2_MAX_CR_LEVELS || numLayers_ < 0 ||
            outerLevels_ > GTSV2_MAX_OUTER_CR_LEVELS || outerLevels_ < 0) {
            ResetToSafeState();
            return;
        }
        // mPad/alignedM read from TilingData (single source)
        mPad_ = tiling->mPad;
        alignedM_ = tiling->alignedM;
        InitDerived();
        // 外层路径 P（outerTileElems_）合法性防御：非 0、8 的倍数、2P <= 内层 mPad
        // （外层各层 numTiles = nHalf / P >= 1 的前置条件；tiling 可信，此处为低成本防御）
        if (useGmWorkspace_ &&
            (outerTileElems_ <= 0 || outerTileElems_ % kDataBlockSize != 0 ||
             2 * outerTileElems_ > mPad_)) {
            ResetToSafeState();
            return;
        }
        InitBuffers(static_cast<uint32_t>(saveBufSize_));
        if (useGmWorkspace_) {
            InitOuterMode();
        }
        initOk_ = true;
    }

    __aicore__ inline void Process()
    {
        // Init guard 触发（含 tiling/pipe 为 nullptr 的早退路径）时不以半初始化状态执行
        if (!initOk_) {
            return;
        }
        int32_t blockIdx = GetBlockIdx();
        int32_t batchStart = blockIdx * batchPerCoreBase_ +
                             (blockIdx < remainder_ ? blockIdx : remainder_);
        int32_t myBatchCount = batchPerCoreBase_ + (blockIdx < remainder_ ? 1 : 0);
        int32_t batchEnd = batchStart + myBatchCount;
        if (batchEnd > batchCount_) {
            batchEnd = batchCount_;
        }

        for (int32_t batchIdx = batchStart; batchIdx < batchEnd; batchIdx++) {
            ProcessOneBatch(batchIdx);
        }
    }

private:
    // guard 触发时将全部派生成员置安全态（initOk_ 保持 false，Process 入口直接早退；
    // useGmWorkspace_/batchCount_ 清零保证即使误调 Process 也不会进入任何计算分支）
    __aicore__ inline void ResetToSafeState()
    {
        numLayers_ = 0;
        outerLevels_ = 0;
        mPad_ = 0;
        alignedM_ = 0;
        vecBytes_ = 0;
        useVectorGather_ = false;
        useGmWorkspace_ = false;
        batchCount_ = 0;
        outerTileElems_ = 0;
        mPadOuter_ = 0;
        regionBByteOffset_ = 0;
        saveRegionByteOffset_ = 0;
        wsPerBatch_ = 0;
    }

    // 预计算 Gather/Scatter 偏移表（使用 2 个共享 offsetBuf，分阶段复用）
    // sharedOffsetBuf1_：even 偏移（常驻，整个 CR 过程不变）
    // sharedOffsetBuf2_：分阶段复用：right → odd → leftShift 偏移（外层路径在 InnerSolve 后重建为 oddOff）
    __aicore__ inline void InitOffsetBuffers()
    {
        auto evenOff = sharedOffsetBuf1_.Get<uint32_t>();

        // 外层路径 alignedM_ 恒为内层常量 2048（>= P），纯 UB 路径为自然 alignedM
        for (int32_t j = 0; j < alignedM_; j++) {
            evenOff.SetValue(j, static_cast<uint32_t>(2 * j) * sizeof(float));
        }
    }

    // 外层 GM 分块路径初始化：区域数组步长反推 + xPrevOff 偏移表预计算
    __aicore__ inline void InitOuterMode()
    {
        // Host 为偏移单一来源：区域内单数组步长（floats）由区域字节偏移反推，避免双源计算
        sAFloats_ = regionBByteOffset_ / (GTSV2_NUM_SAVE_ARRAYS * static_cast<int64_t>(sizeof(float)));
        sBFloats_ = (saveRegionByteOffset_ - regionBByteOffset_) /
                    (GTSV2_NUM_SAVE_ARRAYS * static_cast<int64_t>(sizeof(float)));
        // xPrev[j] = xPrevSrc[j + 7]：8 元素左 halo 窗对齐到 j-1
        auto xPrevOff = sharedOffsetBuf3_.Get<uint32_t>();
        for (int32_t j = 0; j < outerTileElems_; j++) {
            xPrevOff.SetValue(j, static_cast<uint32_t>(j + GTSV2_GUARD_FLOATS - 1) * sizeof(float));
        }
        // Scalar SetValue 写偏移表（evenOff/xPrevOff）后 Vector Gather 读，需 PipeBarrier<PIPE_ALL> 同步；
        // 覆盖 readUserDirect（跳过 MaterializeSystem）路径的首个 Gather
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline int64_t SetupBatchGMAddrs(
        int32_t batchIdx,
        GlobalTensor<float> &gmDl, GlobalTensor<float> &gmD,
        GlobalTensor<float> &gmDu, GlobalTensor<float> &gmX)
    {
        auto *gmDlF = reinterpret_cast<__gm__ float *>(gmDl_);
        auto *gmDF  = reinterpret_cast<__gm__ float *>(gmD_);
        auto *gmDuF = reinterpret_cast<__gm__ float *>(gmDu_);
        auto *gmXF  = reinterpret_cast<__gm__ float *>(gmX_);
        int64_t offset = static_cast<int64_t>(batchIdx) * batchStride_;
        gmDl.SetGlobalBuffer(gmDlF + offset);
        gmD.SetGlobalBuffer(gmDF + offset);
        gmDu.SetGlobalBuffer(gmDuF + offset);
        gmX.SetGlobalBuffer(gmXF + offset);
        return offset;
    }

    // 外层路径：计算本 batch RegionA/B/SaveRegion 基址（int64），初始化 cur 乒乓与 x 乒乓 GlobalTensor
    __aicore__ inline void SetupWorkspaceAddrs(int32_t batchIdx)
    {
        GM_ADDR batchBase = gmWorkspace_ + static_cast<int64_t>(batchIdx) * wsPerBatch_;
        auto *regionA = reinterpret_cast<__gm__ float *>(batchBase);
        auto *regionB = reinterpret_cast<__gm__ float *>(batchBase + regionBByteOffset_);
        auto *saveR = reinterpret_cast<__gm__ float *>(batchBase + saveRegionByteOffset_);
        curA_[0].SetGlobalBuffer(regionA);
        curB_[0].SetGlobalBuffer(regionA + sAFloats_);
        curC_[0].SetGlobalBuffer(regionA + 2 * sAFloats_);
        curD_[0].SetGlobalBuffer(regionA + 3 * sAFloats_);
        curA_[1].SetGlobalBuffer(regionB);
        curB_[1].SetGlobalBuffer(regionB + sBFloats_);
        curC_[1].SetGlobalBuffer(regionB + 2 * sBFloats_);
        curD_[1].SetGlobalBuffer(regionB + 3 * sBFloats_);
        // 正向结束后 cur 数据生命周期终止，x ping/pong 直接复用 RegionA/B 首段（别名）
        xBuf_[0].SetGlobalBuffer(regionA);
        xBuf_[1].SetGlobalBuffer(regionB);
        saveRegion_.SetGlobalBuffer(saveR);
    }

    __aicore__ inline void CopyInFromGM(
        GlobalTensor<float> &gmDl, GlobalTensor<float> &gmD,
        GlobalTensor<float> &gmDu, GlobalTensor<float> &gmX,
        LocalTensor<float> &a, LocalTensor<float> &b,
        LocalTensor<float> &c, LocalTensor<float> &d)
    {
        auto aT = aInQue_.AllocTensor<float>();
        auto bT = bInQue_.AllocTensor<float>();
        auto cT = cInQue_.AllocTensor<float>();
        auto dT = dInQue_.AllocTensor<float>();
        DataCopy(aT, gmDl, alignedM_);
        DataCopy(bT, gmD, alignedM_);
        DataCopy(cT, gmDu, alignedM_);
        DataCopy(dT, gmX, alignedM_);
        aInQue_.EnQue(aT);
        bInQue_.EnQue(bT);
        cInQue_.EnQue(cT);
        dInQue_.EnQue(dT);
        a = aInQue_.DeQue<float>();
        b = bInQue_.DeQue<float>();
        c = cInQue_.DeQue<float>();
        d = dInQue_.DeQue<float>();
        SetPaddingRows(a, b, c, d);
    }

    // 4 路源数组 staging 搬运 GM -> UB（各自偏移、统一长度；MTE2→Vector 经 EnQue/DeQue 同步）
    // 外层正向（srcA-D 同偏移）与外层反向（saveRegion 4 段偏移）共用，避免重复块
    __aicore__ inline void CopyFourArraysIn(
        GlobalTensor<float> &srcA, GlobalTensor<float> &srcB,
        GlobalTensor<float> &srcC, GlobalTensor<float> &srcD,
        uint64_t offA, uint64_t offB, uint64_t offC, uint64_t offD,
        uint32_t len,
        LocalTensor<float> &a, LocalTensor<float> &b,
        LocalTensor<float> &c, LocalTensor<float> &d)
    {
        auto aT = aInQue_.AllocTensor<float>();
        auto bT = bInQue_.AllocTensor<float>();
        auto cT = cInQue_.AllocTensor<float>();
        auto dT = dInQue_.AllocTensor<float>();
        DataCopy(aT, srcA[offA], len);
        DataCopy(bT, srcB[offB], len);
        DataCopy(cT, srcC[offC], len);
        DataCopy(dT, srcD[offD], len);
        aInQue_.EnQue(aT);
        bInQue_.EnQue(bT);
        cInQue_.EnQue(cT);
        dInQue_.EnQue(dT);
        a = aInQue_.DeQue<float>();
        b = bInQue_.DeQue<float>();
        c = cInQue_.DeQue<float>();
        d = dInQue_.DeQue<float>();
    }

    __aicore__ inline void WriteBackToGM(GlobalTensor<float> &gmX, LocalTensor<float> &d)
    {
        PipeBarrier<PIPE_V>();
        auto dOut = dOutQue_.AllocTensor<float>();
        DataCopy(dOut, d, alignedM_);
        dOutQue_.EnQue(dOut);
        auto dResult = dOutQue_.DeQue<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(m_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(gmX, dResult, copyParams);
        dOutQue_.FreeTensor(dResult);
    }

    __aicore__ inline void ProcessOneBatch(int32_t batchIdx)
    {
        if (useGmWorkspace_) {
            ProcessOneBatchOuter(batchIdx);
            return;
        }
        GlobalTensor<float> gmDl, gmD, gmDu, gmX;
        SetupBatchGMAddrs(batchIdx, gmDl, gmD, gmDu, gmX);

        LocalTensor<float> a, b, c, d;
        CopyInFromGM(gmDl, gmD, gmDu, gmX, a, b, c, d);
        ForwardCR(a, b, c, d);

        // 奇异矩阵例外：b[0] 可能为 0，产生 Inf/NaN 为 no-pivot 算法预期行为，不做拦截
        // Valve 写 d via Vector Gather + Scalar SetValue d[0]，均需 PipeBarrier<PIPE_ALL> 保证完整同步
        PipeBarrier<PIPE_ALL>();
        d.SetValue(0, d.GetValue(0) / b.GetValue(0));
        BackwardCR(d);
        WriteBackToGM(gmX, d);

        aInQue_.FreeTensor(a);
        bInQue_.FreeTensor(b);
        cInQue_.FreeTensor(c);
        dInQue_.FreeTensor(d);
    }

    // 外层 GM 分块路径单 batch 流程：Materialize -> OuterForward -> InnerSolve -> OuterBackward -> FinalCopy
    __aicore__ inline void ProcessOneBatchOuter(int32_t batchIdx)
    {
        GlobalTensor<float> gmDl, gmD, gmDu, gmX;
        SetupBatchGMAddrs(batchIdx, gmDl, gmD, gmDu, gmX);
        SetupWorkspaceAddrs(batchIdx);

        // m == mPad 且有外层时第 0 层直读用户数组，跳过 Materialize
        bool readUserDirect = (static_cast<int64_t>(m_) == mPadOuter_) && (outerLevels_ > 0);
        if (!readUserDirect) {
            MaterializeSystem(gmDl, gmD, gmDu, gmX);
        }
        OuterForward(gmDl, gmD, gmDu, gmX, readUserDirect);
        InnerSolve();
        OuterBackward();
        FinalCopyToX(gmX);
    }

    // 外层路径阶段 0：用户数组 -> RegionA cur[0, mPad)，[m, mPad) 填单位行（a=0,b=1,c=0,d=0）
    __aicore__ inline void MaterializeSystem(
        GlobalTensor<float> &gmDl, GlobalTensor<float> &gmD,
        GlobalTensor<float> &gmDu, GlobalTensor<float> &gmX)
    {
        int64_t tileElems = 2 * outerTileElems_;
        for (int64_t start = 0; start < mPadOuter_; start += tileElems) {
            int64_t remain = mPadOuter_ - start;
            int32_t tileCount = static_cast<int32_t>(remain < tileElems ? remain : tileElems);
            MaterializeOneArray(aInQue_, gmDl, curA_[0], start, tileCount, 0.0f);
            MaterializeOneArray(bInQue_, gmD, curB_[0], start, tileCount, 1.0f);
            MaterializeOneArray(cInQue_, gmDu, curC_[0], start, tileCount, 0.0f);
            MaterializeOneArray(dInQue_, gmX, curD_[0], start, tileCount, 0.0f);
            // 单缓冲 staging 跨 tile WAR
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void MaterializeOneArray(
        TQue<TPosition::VECIN, BUFFER_NUM> &que,
        GlobalTensor<float> &src, GlobalTensor<float> &dst,
        int64_t start, int32_t tileCount, float fillVal)
    {
        auto t = que.AllocTensor<float>();
        int64_t userCnt64 = static_cast<int64_t>(m_) - start;
        if (userCnt64 > tileCount) {
            userCnt64 = tileCount;
        }
        int32_t userCnt = userCnt64 > 0 ? static_cast<int32_t>(userCnt64) : 0;
        int32_t alignedUser = (userCnt / kDataBlockSize) * kDataBlockSize;
        if (alignedUser > 0) {
            DataCopy(t, src[static_cast<uint64_t>(start)], alignedUser);
        }
        if (userCnt > alignedUser) {
            // 用户数组尾段非对齐（<8 元素）：GM→UB DataCopyPad，dst UB 32B 对齐、src GM 1B 对齐
            DataCopyExtParams copyParams{
                1, static_cast<uint32_t>((userCnt - alignedUser) * static_cast<int32_t>(sizeof(float))), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            DataCopyPad(t[alignedUser], src[static_cast<uint64_t>(start + alignedUser)],
                        copyParams, padParams);
        }
        // [userCnt, tileCount) 填单位行：非对齐头 SetValue（<=7 个），对齐段 Duplicate
        int32_t fillAligned = ((userCnt + kDataBlockSize - 1) / kDataBlockSize) * kDataBlockSize;
        for (int32_t i = userCnt; i < fillAligned && i < tileCount; i++) {
            t.SetValue(i, fillVal);
        }
        if (fillAligned < tileCount) {
            Duplicate<float>(t[fillAligned], fillVal, tileCount - fillAligned);
        }
        PipeBarrier<PIPE_ALL>();
        que.EnQue(t);
        auto out = que.DeQue<float>();
        // DataCopyPad UB→GM：blockLen 字节非对齐允许，GM 侧 1B 对齐（同 WriteBackToGM）
        DataCopyExtParams writeParams{
            1, static_cast<uint32_t>(tileCount * static_cast<int32_t>(sizeof(float))), 0, 0, 0};
        DataCopyPad(dst[static_cast<uint64_t>(start)], out, writeParams);
        que.FreeTensor(out);
    }

    // 外层正向归约：K 层循环，维护 N_k、cur 乒乓、saveRegion 层偏移（int64 累加），逐 tile 处理
    __aicore__ inline void OuterForward(
        GlobalTensor<float> &gmDl, GlobalTensor<float> &gmD,
        GlobalTensor<float> &gmDu, GlobalTensor<float> &gmX, bool readUserDirect)
    {
        int64_t nK = mPadOuter_;
        int64_t saveOff = 0;
        for (int32_t k = 0; k < outerLevels_; k++) {
            int64_t nHalf = nK >> 1;
            int64_t numTiles = nHalf / outerTileElems_;
            int32_t parity = k & 1;
            GlobalTensor<float> &srcA = (k == 0 && readUserDirect) ? gmDl : curA_[parity];
            GlobalTensor<float> &srcB = (k == 0 && readUserDirect) ? gmD : curB_[parity];
            GlobalTensor<float> &srcC = (k == 0 && readUserDirect) ? gmDu : curC_[parity];
            GlobalTensor<float> &srcD = (k == 0 && readUserDirect) ? gmX : curD_[parity];
            for (int64_t t = 0; t < numTiles; t++) {
                OuterForwardTile(srcA, srcB, srcC, srcD,
                                 curA_[parity ^ 1], curB_[parity ^ 1],
                                 curC_[parity ^ 1], curD_[parity ^ 1],
                                 saveOff, nHalf, t * outerTileElems_, t == numTiles - 1);
            }
            saveOff += GTSV2_NUM_SAVE_ARRAYS * nHalf;
            nK = nHalf;
        }
    }

    // 外层正向单 tile 阶段 2：Gather 拆分偶/奇/右邻 + 全局最右 tile halo 边界修复
    __aicore__ inline void GatherSplitForwardTile(
        LocalTensor<float> &a, LocalTensor<float> &b,
        LocalTensor<float> &c, LocalTensor<float> &d, bool isLastTile)
    {
        uint32_t P = static_cast<uint32_t>(outerTileElems_);
        auto evenOff = sharedOffsetBuf1_.Get<uint32_t>();
        LocalTensor<float> aSave, bSave, cSave, dSave, oddA, oddB, oddC, oddD;
        LocalTensor<float> aRight, bRight, cRight, dRight;
        GetGatherBuffers(aSave, bSave, cSave, dSave, oddA, oddB, oddC, oddD,
                         aRight, bRight, cRight, dRight);

        // evenOff 一档三用：base=0 偶 2j，base=4 奇 2j+1，base=8 右邻 2j+2
        Gather(aSave, a, evenOff, 0U, P);
        Gather(bSave, b, evenOff, 0U, P);
        Gather(cSave, c, evenOff, 0U, P);
        Gather(dSave, d, evenOff, 0U, P);
        Gather(oddA, a, evenOff, static_cast<uint32_t>(sizeof(float)), P);
        Gather(oddB, b, evenOff, static_cast<uint32_t>(sizeof(float)), P);
        Gather(oddC, c, evenOff, static_cast<uint32_t>(sizeof(float)), P);
        Gather(oddD, d, evenOff, static_cast<uint32_t>(sizeof(float)), P);
        Gather(aRight, a, evenOff, static_cast<uint32_t>(2 * sizeof(float)), P);
        Gather(bRight, b, evenOff, static_cast<uint32_t>(2 * sizeof(float)), P);
        Gather(cRight, c, evenOff, static_cast<uint32_t>(2 * sizeof(float)), P);
        Gather(dRight, d, evenOff, static_cast<uint32_t>(2 * sizeof(float)), P);
        aInQue_.FreeTensor(a);
        bInQue_.FreeTensor(b);
        cInQue_.FreeTensor(c);
        dInQue_.FreeTensor(d);

        if (isLastTile) {
            // 全局最右 tile 的 halo 元素（索引 = N）不存在，unit-row 修复（同 FixRightBoundaryAndPad 语义）
            PipeBarrier<PIPE_ALL>();
            aRight.SetValue(outerTileElems_ - 1, 0.0f);
            bRight.SetValue(outerTileElems_ - 1, 1.0f);
            cRight.SetValue(outerTileElems_ - 1, 0.0f);
            dRight.SetValue(outerTileElems_ - 1, 0.0f);
            PipeBarrier<PIPE_ALL>();
        }
    }

    // 外层正向单 tile 阶段 3：单层 CR 计算 + 双路写回（下一层 cur' 与本层 save）
    __aicore__ inline void ComputeWriteBackForwardTile(
        GlobalTensor<float> &dstA, GlobalTensor<float> &dstB,
        GlobalTensor<float> &dstC, GlobalTensor<float> &dstD,
        int64_t saveOff, int64_t nHalf, int64_t j0)
    {
        // 满 tile：activeCount == count == P，无需 padding 区初始化
        ComputeForwardLayer(outerTileElems_, 0);

        // V→MTE3：写回下一层 cur' 与本层 save
        // PIPE_V 只同步 Vector 内部，无法保证后续 MTE3（DataCopy UB→GM）读到已完成的
        // Vector 结果；必须用 PIPE_ALL（V→MTE3 依赖）
        PipeBarrier<PIPE_ALL>();
        uint32_t P = static_cast<uint32_t>(outerTileElems_);
        LocalTensor<float> aSave = aSaveBuf_.Get<float>();
        LocalTensor<float> bSave = bSaveBuf_.Get<float>();
        LocalTensor<float> cSave = cSaveBuf_.Get<float>();
        LocalTensor<float> dSave = dSaveBuf_.Get<float>();
        LocalTensor<float> oddA = oddABuf_.Get<float>();
        LocalTensor<float> oddB = oddBBuf_.Get<float>();
        LocalTensor<float> oddC = oddCBuf_.Get<float>();
        LocalTensor<float> oddD = oddDBuf_.Get<float>();
        uint64_t dstIdx = static_cast<uint64_t>(j0);
        DataCopy(dstA[dstIdx], oddA, P);
        DataCopy(dstB[dstIdx], oddB, P);
        DataCopy(dstC[dstIdx], oddC, P);
        DataCopy(dstD[dstIdx], oddD, P);
        uint64_t so = static_cast<uint64_t>(saveOff + j0);
        uint64_t nH = static_cast<uint64_t>(nHalf);
        DataCopy(saveRegion_[so], aSave, P);
        DataCopy(saveRegion_[so + nH], bSave, P);
        DataCopy(saveRegion_[so + 2 * nH], cSave, P);
        DataCopy(saveRegion_[so + 3 * nH], dSave, P);
        // 单缓冲 staging/save 跨 tile WAR
        PipeBarrier<PIPE_ALL>();
    }

    // 外层正向单 tile：搬 2P+halo -> UB，Gather 拆分偶/奇/右邻，复用 ComputeForwardLayer，双路写回
    __aicore__ inline void OuterForwardTile(
        GlobalTensor<float> &srcA, GlobalTensor<float> &srcB,
        GlobalTensor<float> &srcC, GlobalTensor<float> &srcD,
        GlobalTensor<float> &dstA, GlobalTensor<float> &dstB,
        GlobalTensor<float> &dstC, GlobalTensor<float> &dstD,
        int64_t saveOff, int64_t nHalf, int64_t j0, bool isLastTile)
    {
        uint32_t len = static_cast<uint32_t>(2 * outerTileElems_ + (isLastTile ? 0 : GTSV2_GUARD_FLOATS));
        uint64_t srcIdx = static_cast<uint64_t>(2 * j0);
        LocalTensor<float> a, b, c, d;
        CopyFourArraysIn(srcA, srcB, srcC, srcD, srcIdx, srcIdx, srcIdx, srcIdx,
                         len, a, b, c, d);
        GatherSplitForwardTile(a, b, c, d, isLastTile);
        ComputeWriteBackForwardTile(dstA, dstB, dstC, dstD, saveOff, nHalf, j0);
    }

    // 外层路径阶段 2：内层 UB CR（几何恒为内层常量，由 tiling 传入），完全复用现有 ForwardCR/BackwardCR
    __aicore__ inline void InnerSolve()
    {
        int32_t parity = outerLevels_ & 1;
        auto aT = aInQue_.AllocTensor<float>();
        auto bT = bInQue_.AllocTensor<float>();
        auto cT = cInQue_.AllocTensor<float>();
        auto dT = dInQue_.AllocTensor<float>();
        DataCopy(aT, curA_[parity], alignedM_);
        DataCopy(bT, curB_[parity], alignedM_);
        DataCopy(cT, curC_[parity], alignedM_);
        DataCopy(dT, curD_[parity], alignedM_);
        aInQue_.EnQue(aT);
        bInQue_.EnQue(bT);
        cInQue_.EnQue(cT);
        dInQue_.EnQue(dT);
        LocalTensor<float> a = aInQue_.DeQue<float>();
        LocalTensor<float> b = bInQue_.DeQue<float>();
        LocalTensor<float> c = cInQue_.DeQue<float>();
        LocalTensor<float> d = dInQue_.DeQue<float>();

        ForwardCR(a, b, c, d);
        // 奇异矩阵例外：b[0] 可能为 0，产生 Inf/NaN 为 no-pivot 算法预期行为，不做拦截
        PipeBarrier<PIPE_ALL>();
        d.SetValue(0, d.GetValue(0) / b.GetValue(0));
        BackwardCR(d);

        // 内层结果写回 GM xBuf_[0]（RegionA 首段）；PipeBarrier<PIPE_ALL> 保证 MTE3 完成后外层反向再读
        // 同 ComputeWriteBackForwardTile：V→MTE3 依赖需 PIPE_ALL，PIPE_V 不足以同步
        PipeBarrier<PIPE_ALL>();
        DataCopy(xBuf_[0], d, alignedM_);
        PipeBarrier<PIPE_ALL>();
        aInQue_.FreeTensor(a);
        bInQue_.FreeTensor(b);
        cInQue_.FreeTensor(c);
        dInQue_.FreeTensor(d);

        // 内层阶段 sharedOffsetBuf2_ 被挪作他用，OuterBackward 开始前按 P 重建 oddOff
        auto oddOff = sharedOffsetBuf2_.Get<uint32_t>();
        for (int32_t j = 0; j < outerTileElems_; j++) {
            oddOff.SetValue(j, static_cast<uint32_t>(2 * j + 1) * sizeof(float));
        }
        PipeBarrier<PIPE_ALL>();
    }

    // 外层反向回代：K 层逆序循环，x 乒乓（RegionA/B 别名），saveRegion 逆序偏移，逐 tile 处理
    __aicore__ inline void OuterBackward()
    {
        int32_t s = 0;
        for (int32_t k = outerLevels_ - 1; k >= 0; k--, s++) {
            int64_t nK = mPadOuter_ >> k;
            int64_t nHalf = nK >> 1;
            // 层 k save 起始偏移 = sum_{i<k} 4*N_{i+1} = 4*(mPad - N_k)（几何级数闭式）
            int64_t saveOff = GTSV2_NUM_SAVE_ARRAYS * (mPadOuter_ - nK);
            int64_t numTiles = nHalf / outerTileElems_;
            for (int64_t t = 0; t < numTiles; t++) {
                OuterBackwardTile(xBuf_[s & 1], xBuf_[(s + 1) & 1], saveOff, nHalf,
                                  t * outerTileElems_);
            }
        }
    }

    // 外层反向单 tile 阶段 2：搬 x tile 到 xPrevSrc（含 8 元素左 halo，j0=0 时 halo 补 0）
    __aicore__ inline void CopyXHaloWindow(
        GlobalTensor<float> &xIn, int64_t j0, LocalTensor<float> &xPrevSrc)
    {
        if (j0 > 0) {
            DataCopy(xPrevSrc, xIn[static_cast<uint64_t>(j0 - GTSV2_GUARD_FLOATS)],
                     static_cast<uint32_t>(outerTileElems_ + GTSV2_GUARD_FLOATS));
        } else {
            DataCopy(xPrevSrc[GTSV2_GUARD_FLOATS], xIn[0], static_cast<uint32_t>(outerTileElems_));
            Duplicate<float>(xPrevSrc, 0.0f, GTSV2_GUARD_FLOATS);
        }
        // xPrevSrc（MTE2 / Scalar Duplicate）→ Vector（Gather/算术）
        PipeBarrier<PIPE_ALL>();
    }

    // 外层反向单 tile 阶段 3：Scatter 组帧 xNew（偶位 xLeft / 奇位 xCur）并写回 xOut
    __aicore__ inline void ScatterFrameBackwardTile(
        GlobalTensor<float> &xOut, LocalTensor<float> &xLeft,
        LocalTensor<float> &xCur, int64_t j0)
    {
        uint32_t P = static_cast<uint32_t>(outerTileElems_);
        // 组帧 xNew（dOutQue_ 容量 2P）：偶位 2j <- xLeft，奇位 2j+1 <- xCur
        auto xNew = dOutQue_.AllocTensor<float>();
        auto evenOff = sharedOffsetBuf1_.Get<uint32_t>();
        auto oddOff = sharedOffsetBuf2_.Get<uint32_t>();
        Scatter(xNew, xLeft, evenOff, 0U, P);
        Scatter(xNew, xCur, oddOff, 0U, P);
        dOutQue_.EnQue(xNew);
        auto xNewOut = dOutQue_.DeQue<float>();
        DataCopy(xOut[static_cast<uint64_t>(2 * j0)], xNewOut, 2 * P);
        dOutQue_.FreeTensor(xNewOut);
        // 单缓冲 xPrevSrc/save 跨 tile WAR
        PipeBarrier<PIPE_ALL>();
    }

    // 外层反向单 tile：搬 save tile + x tile（含 8 元素左 halo），复用 ComputeBackwardXLeft，Scatter 组帧 2P
    __aicore__ inline void OuterBackwardTile(
        GlobalTensor<float> &xIn, GlobalTensor<float> &xOut,
        int64_t saveOff, int64_t nHalf, int64_t j0)
    {
        uint32_t P = static_cast<uint32_t>(outerTileElems_);
        uint64_t so = static_cast<uint64_t>(saveOff + j0);
        uint64_t nH = static_cast<uint64_t>(nHalf);
        LocalTensor<float> aSave, bSave, cSave, dSave;
        CopyFourArraysIn(saveRegion_, saveRegion_, saveRegion_, saveRegion_,
                         so, so + nH, so + 2 * nH, so + 3 * nH, P,
                         aSave, bSave, cSave, dSave);

        auto xPrevSrc = xPrevSrcBuf_.Get<float>();
        CopyXHaloWindow(xIn, j0, xPrevSrc);

        auto xPrevOff = sharedOffsetBuf3_.Get<uint32_t>();
        auto xPrev = t1Buf_.Get<float>();
        auto xLeft = t2Buf_.Get<float>();
        auto tmp1 = k1Buf_.Get<float>();
        auto tmp2 = k2Buf_.Get<float>();
        Gather(xPrev, xPrevSrc, xPrevOff, 0U, P);
        LocalTensor<float> xCur = xPrevSrc[GTSV2_GUARD_FLOATS];
        // xLeft = (dSave - aSave*xPrev - cSave*xCur)/bSave（现有函数零改动复用）
        ComputeBackwardXLeft(aSave, bSave, cSave, dSave, xPrev, xLeft, tmp1, tmp2, xCur,
                             outerTileElems_, 0);
        aInQue_.FreeTensor(aSave);
        bInQue_.FreeTensor(bSave);
        cInQue_.FreeTensor(cSave);
        dInQue_.FreeTensor(dSave);

        ScatterFrameBackwardTile(xOut, xLeft, xCur, j0);
    }

    // 外层路径阶段 4：最终 x（xBuf[K%2]，长度 mPad）-> gmX[0, m)，尾 tile DataCopyPad 不写用户 padding 区
    __aicore__ inline void FinalCopyToX(GlobalTensor<float> &gmX)
    {
        GlobalTensor<float> &src = xBuf_[outerLevels_ & 1];
        int64_t tileElems = 2 * outerTileElems_;
        for (int64_t off = 0; off < m_; off += tileElems) {
            int64_t remain = static_cast<int64_t>(m_) - off;
            int32_t cnt = static_cast<int32_t>(remain < tileElems ? remain : tileElems);
            int32_t readCnt = ((cnt + kDataBlockSize - 1) / kDataBlockSize) * kDataBlockSize;
            auto t = aInQue_.AllocTensor<float>();
            DataCopy(t, src[static_cast<uint64_t>(off)], readCnt);
            aInQue_.EnQue(t);
            auto in = aInQue_.DeQue<float>();
            // 非冗余：VECIN DeQue 仅保证 MTE2→V 就绪，DataCopyPad（MTE3）读同一 UB 需 MTE2→MTE3 同步
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(cnt * static_cast<int32_t>(sizeof(float))), 0, 0, 0};
            DataCopyPad(gmX[static_cast<uint64_t>(off)], in, copyParams);
            aInQue_.FreeTensor(in);
            // 单缓冲 staging 跨 tile WAR
            PipeBarrier<PIPE_ALL>();
        }
    }

    // 对 m_pad > m 的 padding 位置设置单位行（a=0, b=1, c=0, d=0）
    // 对齐部分使用 Duplicate（Vector Pipe），未对齐头部使用 SetValue（最多 7 个元素）
    __aicore__ inline void SetPaddingRows(
        LocalTensor<float> &a, LocalTensor<float> &b,
        LocalTensor<float> &c, LocalTensor<float> &d)
    {
        if (m_ >= mPad_) {
            return;
        }
        int32_t padAlignedStart = ((m_ + kDataBlockSize - 1) / kDataBlockSize) * kDataBlockSize;
        // 未对齐头部：最多 7 个元素
        for (int32_t i = m_; i < padAlignedStart && i < mPad_; i++) {
            a.SetValue(i, 0.0f);
            b.SetValue(i, 1.0f);
            c.SetValue(i, 0.0f);
            d.SetValue(i, 0.0f);
        }
        // 对齐主体：Duplicate 填充（Vector Pipe）
        if (padAlignedStart < mPad_) {
            int32_t padCount = mPad_ - padAlignedStart;
            Duplicate<float>(a[padAlignedStart], 0.0f, padCount);
            Duplicate<float>(b[padAlignedStart], 1.0f, padCount);
            Duplicate<float>(c[padAlignedStart], 0.0f, padCount);
            Duplicate<float>(d[padAlignedStart], 0.0f, padCount);
        }
    }

    // 计算单层 CR 归约的 count（activeCount 对齐到 8，最小 8）
    __aicore__ inline int32_t CalcLayerCount(int32_t currentSize)
    {
        int32_t activeCount = currentSize / 2;
        int32_t count = ((activeCount + kDataBlockSize - 1) / kDataBlockSize) * kDataBlockSize;
        if (count < kDataBlockSize) {
            count = kDataBlockSize;
        }
        return count;
    }

    // 初始化 padding 区域 [activeCount, count) 为单位行（a=0, b=1, c=0, d=0）
    // Vector 指令读取未初始化 padding 区域的问题
    // padding 最多 7 个元素，使用 SetValue 即可
    __aicore__ inline void InitPaddingRegion(
        LocalTensor<float> &a, LocalTensor<float> &b,
        LocalTensor<float> &c, LocalTensor<float> &d,
        int32_t activeCount, int32_t count)
    {
        for (int32_t i = activeCount; i < count; i++) {
            a.SetValue(i, 0.0f);
            b.SetValue(i, 1.0f);
            c.SetValue(i, 0.0f);
            d.SetValue(i, 0.0f);
        }
    }

    // 准备一层正向归约的输入数据
    // useVectorGather_ 为 true 时使用 Gather API（Vector Pipe），否则使用标量循环
    __aicore__ inline void GatherForwardLayer(
        LocalTensor<float> &a, LocalTensor<float> &b,
        LocalTensor<float> &c, LocalTensor<float> &d,
        int32_t saveOffset, int32_t activeCount, int32_t count)
    {
        if (useVectorGather_) {
            GatherForwardLayerVector(a, b, c, d, saveOffset, activeCount, count);
        } else {
            GatherForwardLayerScalar(a, b, c, d, saveOffset, activeCount, count);
        }
    }

    // GetGatherBuffers：GatherForwardLayer Vector/Scalar 共用的 12 个 buffer 获取
    __aicore__ inline void GetGatherBuffers(
        LocalTensor<float> &aSave, LocalTensor<float> &bSave,
        LocalTensor<float> &cSave, LocalTensor<float> &dSave,
        LocalTensor<float> &oddA, LocalTensor<float> &oddB,
        LocalTensor<float> &oddC, LocalTensor<float> &oddD,
        LocalTensor<float> &aRight, LocalTensor<float> &bRight,
        LocalTensor<float> &cRight, LocalTensor<float> &dRight)
    {
        aSave = aSaveBuf_.Get<float>();
        bSave = bSaveBuf_.Get<float>();
        cSave = cSaveBuf_.Get<float>();
        dSave = dSaveBuf_.Get<float>();
        oddA = oddABuf_.Get<float>();
        oddB = oddBBuf_.Get<float>();
        oddC = oddCBuf_.Get<float>();
        oddD = oddDBuf_.Get<float>();
        aRight = aRightBuf_.Get<float>();
        bRight = bRightBuf_.Get<float>();
        cRight = cRightBuf_.Get<float>();
        dRight = dRightBuf_.Get<float>();
    }

    // FixRightBoundaryAndPad：Vector/Scalar 共用的右边界修复 + padding 初始化
    __aicore__ inline void FixRightBoundaryAndPad(
        LocalTensor<float> &aRight, LocalTensor<float> &bRight,
        LocalTensor<float> &cRight, LocalTensor<float> &dRight,
        LocalTensor<float> &oddA, LocalTensor<float> &oddB,
        LocalTensor<float> &oddC, LocalTensor<float> &oddD,
        int32_t activeCount, int32_t count)
    {
        aRight.SetValue(activeCount - 1, 0.0f);
        bRight.SetValue(activeCount - 1, 1.0f);
        cRight.SetValue(activeCount - 1, 0.0f);
        dRight.SetValue(activeCount - 1, 0.0f);
        InitPaddingRegion(oddA, oddB, oddC, oddD, activeCount, count);
        InitPaddingRegion(aRight, bRight, cRight, dRight, activeCount, count);
    }

    // Vector Pipe 路径：使用 Gather API 替代 SetValue/GetValue 标量循环
    // 使用 2 个共享 offsetBuf（sharedOffsetBuf1_ 常驻 even，sharedOffsetBuf2_ 分阶段复用）
    __aicore__ inline void GatherForwardLayerVector(
        LocalTensor<float> &a, LocalTensor<float> &b,
        LocalTensor<float> &c, LocalTensor<float> &d,
        int32_t saveOffset, int32_t activeCount, int32_t count)
    {
        LocalTensor<float> aSave, bSave, cSave, dSave, oddA, oddB, oddC, oddD;
        LocalTensor<float> aRight, bRight, cRight, dRight;
        GetGatherBuffers(aSave, bSave, cSave, dSave, oddA, oddB, oddC, oddD,
                         aRight, bRight, cRight, dRight);
        auto evenOff = sharedOffsetBuf1_.Get<uint32_t>();
        auto off2 = sharedOffsetBuf2_.Get<uint32_t>();
        uint32_t ac = static_cast<uint32_t>(activeCount);

        Gather(aSave[saveOffset], a, evenOff, 0U, ac);
        Gather(bSave[saveOffset], b, evenOff, 0U, ac);
        Gather(cSave[saveOffset], c, evenOff, 0U, ac);
        Gather(dSave[saveOffset], d, evenOff, 0U, ac);

        for (int32_t i = activeCount; i < count; i++) {
            aSave.SetValue(saveOffset + i, 0.0f);
            bSave.SetValue(saveOffset + i, 1.0f);
            cSave.SetValue(saveOffset + i, 0.0f);
            dSave.SetValue(saveOffset + i, 0.0f);
        }
        // Scalar pad 写 saveBuf → ComputeForwardLayer 通过 Vector 读 saveBuf
        PipeBarrier<PIPE_ALL>();

        for (int32_t j = 0; j < ac; j++) {
            off2.SetValue(j, static_cast<uint32_t>(j + 1) * sizeof(float));
        }
        // Scalar 写 off2 → Vector Gather 读 off2
        PipeBarrier<PIPE_ALL>();

        uint32_t saveBaseBytes = static_cast<uint32_t>(saveOffset) * sizeof(float);
        Gather(aRight, aSave, off2, saveBaseBytes, ac);
        Gather(bRight, bSave, off2, saveBaseBytes, ac);
        Gather(cRight, cSave, off2, saveBaseBytes, ac);
        Gather(dRight, dSave, off2, saveBaseBytes, ac);
        // Vector Gather 读 off2 → Scalar 重写 off2（WAR）
        PipeBarrier<PIPE_ALL>();

        for (int32_t j = 0; j < ac; j++) {
            off2.SetValue(j, static_cast<uint32_t>(2 * j + 1) * sizeof(float));
        }
        // Scalar 写 off2 → Vector Gather 读 off2
        PipeBarrier<PIPE_ALL>();

        Gather(oddA, a, off2, 0U, ac);
        Gather(oddB, b, off2, 0U, ac);
        Gather(oddC, c, off2, 0U, ac);
        Gather(oddD, d, off2, 0U, ac);
        FixRightBoundaryAndPad(aRight, bRight, cRight, dRight,
                               oddA, oddB, oddC, oddD, activeCount, count);
        // Scalar 边界修复/padding 写 odd/right/saveBuf → ComputeForwardLayer 通过 Vector 读这些 buffer
        PipeBarrier<PIPE_ALL>();
    }

    // 标量路径：使用 SetValue/GetValue 循环（小 m 回退，m_pad ≤ 8）
    __aicore__ inline void GatherForwardLayerScalar(
        LocalTensor<float> &a, LocalTensor<float> &b,
        LocalTensor<float> &c, LocalTensor<float> &d,
        int32_t saveOffset, int32_t activeCount, int32_t count)
    {
        LocalTensor<float> aSave, bSave, cSave, dSave, oddA, oddB, oddC, oddD;
        LocalTensor<float> aRight, bRight, cRight, dRight;
        GetGatherBuffers(aSave, bSave, cSave, dSave, oddA, oddB, oddC, oddD,
                         aRight, bRight, cRight, dRight);

        for (int32_t j = 0; j < activeCount; j++) {
            aSave.SetValue(saveOffset + j, a.GetValue(2 * j));
            bSave.SetValue(saveOffset + j, b.GetValue(2 * j));
            cSave.SetValue(saveOffset + j, c.GetValue(2 * j));
            dSave.SetValue(saveOffset + j, d.GetValue(2 * j));
        }
        for (int32_t i = activeCount; i < count; i++) {
            aSave.SetValue(saveOffset + i, 0.0f);
            bSave.SetValue(saveOffset + i, 1.0f);
            cSave.SetValue(saveOffset + i, 0.0f);
            dSave.SetValue(saveOffset + i, 0.0f);
        }

        for (int32_t j = 0; j < activeCount; j++) {
            aRight.SetValue(j, aSave.GetValue(saveOffset + j + 1));
            bRight.SetValue(j, bSave.GetValue(saveOffset + j + 1));
            cRight.SetValue(j, cSave.GetValue(saveOffset + j + 1));
            dRight.SetValue(j, dSave.GetValue(saveOffset + j + 1));
        }

        for (int32_t j = 0; j < activeCount; j++) {
            oddA.SetValue(j, a.GetValue(2 * j + 1));
            oddB.SetValue(j, b.GetValue(2 * j + 1));
            oddC.SetValue(j, c.GetValue(2 * j + 1));
            oddD.SetValue(j, d.GetValue(2 * j + 1));
        }
        FixRightBoundaryAndPad(aRight, bRight, cRight, dRight,
                               oddA, oddB, oddC, oddD, activeCount, count);
        // Scalar 写 saveBuf/oddA/oddB/oddC/oddD → ComputeForwardLayer 通过 Vector 读这些 buffer
        PipeBarrier<PIPE_ALL>();
    }

    // 执行一层正向归约的 Vector 计算（Div/Mul/Sub/Muls）
    // 奇异矩阵例外：bSave/bRight 可能为 0，产生 Inf/NaN 为 no-pivot 算法预期行为，不做拦截
    __aicore__ inline void ComputeForwardLayer(int32_t count, int32_t saveOffset)
    {
        auto aSave = aSaveBuf_.Get<float>();
        auto bSave = bSaveBuf_.Get<float>();
        auto cSave = cSaveBuf_.Get<float>();
        auto dSave = dSaveBuf_.Get<float>();

        auto aRight = aRightBuf_.Get<float>();
        auto bRight = bRightBuf_.Get<float>();
        auto cRight = cRightBuf_.Get<float>();
        auto dRight = dRightBuf_.Get<float>();

        auto oddA = oddABuf_.Get<float>();
        auto oddB = oddBBuf_.Get<float>();
        auto oddC = oddCBuf_.Get<float>();
        auto oddD = oddDBuf_.Get<float>();

        auto k1 = k1Buf_.Get<float>();
        auto k2 = k2Buf_.Get<float>();
        auto t1 = t1Buf_.Get<float>();
        auto t2 = t2Buf_.Get<float>();

        // k1 = oddA / bSave[saveOffset] (左邻居 b，saveOffset 对齐)
        Div<float, kDivConfig>(k1, oddA, bSave[saveOffset], count);
        // k2 = oddC / bRight (右邻居 b，bRight 对齐)
        Div<float, kDivConfig>(k2, oddC, bRight, count);

        // b_new = oddB - k1*cSave[saveOffset] - k2*aRight
        Mul<float>(t1, k1, cSave[saveOffset], count);
        Mul<float>(t2, k2, aRight, count);
        Sub<float>(oddB, oddB, t1, count);
        Sub<float>(oddB, oddB, t2, count);

        // d_new = oddD - k1*dSave[saveOffset] - k2*dRight
        Mul<float>(t1, k1, dSave[saveOffset], count);
        Mul<float>(t2, k2, dRight, count);
        Sub<float>(oddD, oddD, t1, count);
        Sub<float>(oddD, oddD, t2, count);

        // a_new = -k1 * aSave[saveOffset] (左邻居 a)
        Mul<float>(t1, k1, aSave[saveOffset], count);
        Muls<float>(oddA, t1, -1.0f, count);

        // c_new = -k2 * cRight (右邻居 c)
        Mul<float>(t2, k2, cRight, count);
        Muls<float>(oddC, t2, -1.0f, count);
    }

    // 将归约结果从 odd buffer 写回主 tensor（紧凑化，前 count 个元素）
    __aicore__ inline void StoreForwardLayer(
        LocalTensor<float> &a, LocalTensor<float> &b,
        LocalTensor<float> &c, LocalTensor<float> &d,
        int32_t count)
    {
        auto oddA = oddABuf_.Get<float>();
        auto oddB = oddBBuf_.Get<float>();
        auto oddC = oddCBuf_.Get<float>();
        auto oddD = oddDBuf_.Get<float>();

        DataCopy(a, oddA, count);
        DataCopy(b, oddB, count);
        DataCopy(c, oddC, count);
        DataCopy(d, oddD, count);
    }

    // CR 正向归约（逐层消元 + 紧凑化），save 数据驻留 UB
    __aicore__ inline void ForwardCR(
        LocalTensor<float> &a, LocalTensor<float> &b,
        LocalTensor<float> &c, LocalTensor<float> &d)
    {
        int32_t currentSize = mPad_;
        int32_t saveOffset = 0;

        for (int32_t k = 0; k < numLayers_; k++) {
            int32_t activeCount = currentSize / 2;
            int32_t count = CalcLayerCount(currentSize);
            // Scalar SetValue 写 off2 后 Vector Gather 读 off2；Scalar pad 写 after Vector read 均需同步
            PipeBarrier<PIPE_ALL>();
            GatherForwardLayer(a, b, c, d, saveOffset, activeCount, count);
            ComputeForwardLayer(count, saveOffset);
            StoreForwardLayer(a, b, c, d, count);
            saveOffset += count;
            currentSize = activeCount;
        }
    }

    // 计算各层的 save offset 和 count（供反向回代使用）
    __aicore__ inline void CalcBackwardLevelInfo(
        int32_t levelCounts[GTSV2_MAX_CR_LEVELS], int32_t levelOffsets[GTSV2_MAX_CR_LEVELS])
    {
        int32_t currentSize = mPad_;
        int32_t saveOffset = 0;
        for (int32_t k = 0; k < numLayers_; k++) {
            int32_t activeCount = currentSize / 2;
            int32_t count = CalcLayerCount(currentSize);
            levelCounts[k] = count;
            levelOffsets[k] = saveOffset;
            saveOffset += count;
            currentSize = activeCount;
        }
    }

    // 执行单层反向回代
    // useVectorGather_ 为 true 时使用 Gather/Scatter API（Vector Pipe），否则使用标量循环
    // 奇异矩阵例外：bSave[off] 可能为 0，产生 Inf/NaN 为 no-pivot 算法预期行为，不做拦截
    __aicore__ inline void BackwardOneStep(
        LocalTensor<float> &d, int32_t activeCount,
        int32_t count, int32_t off)
    {
        if (useVectorGather_) {
            BackwardOneStepVector(d, activeCount, count, off);
        } else {
            BackwardOneStepScalar(d, activeCount, count, off);
        }
    }

    // GetBackwardSaveAndTmpBuffers：BackwardOneStep Vector/Scalar 共用的 8 个 buffer
    __aicore__ inline void GetBackwardSaveAndTmpBuffers(
        LocalTensor<float> &aSave, LocalTensor<float> &bSave,
        LocalTensor<float> &cSave, LocalTensor<float> &dSave,
        LocalTensor<float> &xPrev, LocalTensor<float> &xLeft,
        LocalTensor<float> &tmp1, LocalTensor<float> &tmp2)
    {
        aSave = aSaveBuf_.Get<float>();
        bSave = bSaveBuf_.Get<float>();
        cSave = cSaveBuf_.Get<float>();
        dSave = dSaveBuf_.Get<float>();
        xPrev = t1Buf_.Get<float>();
        xLeft = t2Buf_.Get<float>();
        tmp1 = k1Buf_.Get<float>();
        tmp2 = k2Buf_.Get<float>();
    }

    // ComputeBackwardXLeft：BackwardOneStep Vector/Scalar 共用的向量代数核心
    // xLeft[j] = (dSave[off+j] - aSave[off+j]*xPrev[j] - cSave[off+j]*d[j]) / bSave[off+j]
    __aicore__ inline void ComputeBackwardXLeft(
        LocalTensor<float> &aSave, LocalTensor<float> &bSave,
        LocalTensor<float> &cSave, LocalTensor<float> &dSave,
        LocalTensor<float> &xPrev, LocalTensor<float> &xLeft,
        LocalTensor<float> &tmp1, LocalTensor<float> &tmp2,
        LocalTensor<float> &d, int32_t count, int32_t off)
    {
        Mul<float>(tmp1, aSave[off], xPrev, count);
        Mul<float>(tmp2, cSave[off], d, count);
        Sub<float>(xLeft, dSave[off], tmp1, count);
        Sub<float>(xLeft, xLeft, tmp2, count);
        Div<float, kDivConfig>(xLeft, xLeft, bSave[off], count);
    }

    // AlignAndCopyBack：BackwardOneStep Vector/Scalar 共用的对齐拷贝回 d
    __aicore__ inline void AlignAndCopyBack(
        LocalTensor<float> &d, LocalTensor<float> &xNew, int32_t activeCount)
    {
        int32_t newCount = activeCount * 2;
        int32_t newCountAligned = ((newCount + kDataBlockSize - 1) / kDataBlockSize) * kDataBlockSize;
        if (newCountAligned < kDataBlockSize) {
            newCountAligned = kDataBlockSize;
        }
        DataCopy(d, xNew, newCountAligned);
    }

    // BackwardOneStepVector：使用 Gather/Scatter API（Vector Pipe）
    // 使用 2 个共享 offsetBuf（sharedOffsetBuf1_ 常驻 even，sharedOffsetBuf2_ 分阶段复用）
    __aicore__ inline void BackwardOneStepVector(
        LocalTensor<float> &d, int32_t activeCount,
        int32_t count, int32_t off)
    {
        LocalTensor<float> aSave, bSave, cSave, dSave, xPrev, xLeft, tmp1, tmp2;
        GetBackwardSaveAndTmpBuffers(aSave, bSave, cSave, dSave, xPrev, xLeft, tmp1, tmp2);
        auto xNew = oddABuf_.Get<float>();
        auto evenOff = sharedOffsetBuf1_.Get<uint32_t>();
        auto off2 = sharedOffsetBuf2_.Get<uint32_t>();

        for (int32_t j = 0; j < count; j++) {
            uint32_t srcIdx = (j > 0) ? static_cast<uint32_t>(j - 1) : 0U;
            off2.SetValue(j, srcIdx * sizeof(float));
        }
        // Scalar 写 off2 → Vector Gather 读 off2（Scalar 写 d via Vector → Vector Gather 读 d）
        PipeBarrier<PIPE_ALL>();

        Gather(xPrev, d, off2, 0U, static_cast<uint32_t>(count));
        // Vector Gather 写 xPrev → Scalar 改写 xPrev[0]
        PipeBarrier<PIPE_ALL>();
        xPrev.SetValue(0, 0.0f);
        // Scalar 写 xPrev → ComputeBackwardXLeft 通过 Vector 读 xPrev
        PipeBarrier<PIPE_ALL>();
        ComputeBackwardXLeft(aSave, bSave, cSave, dSave, xPrev, xLeft, tmp1, tmp2, d, count, off);

        uint32_t ac = static_cast<uint32_t>(activeCount);
        // Vector Gather 读 off2 → Scalar 重写 off2（WAR）
        PipeBarrier<PIPE_ALL>();
        for (int32_t j = 0; j < ac; j++) {
            off2.SetValue(j, static_cast<uint32_t>(2 * j + 1) * sizeof(float));
        }
        PipeBarrier<PIPE_ALL>();

        Scatter(xNew, xLeft, evenOff, 0U, ac);
        Scatter(xNew, d, off2, 0U, ac);
        AlignAndCopyBack(d, xNew, activeCount);
        // Vector Scatter 读 off2 / DataCopy 写 d → 下一层 Scalar 重写 off2 / 读 d（WAR）
        PipeBarrier<PIPE_ALL>();
    }

    // BackwardOneStepScalar：使用 SetValue/GetValue 循环（小 m 回退，m_pad ≤ 8）
    __aicore__ inline void BackwardOneStepScalar(
        LocalTensor<float> &d, int32_t activeCount,
        int32_t count, int32_t off)
    {
        LocalTensor<float> aSave, bSave, cSave, dSave, xPrev, xLeft, tmp1, tmp2;
        GetBackwardSaveAndTmpBuffers(aSave, bSave, cSave, dSave, xPrev, xLeft, tmp1, tmp2);

        xPrev.SetValue(0, 0.0f);
        for (int32_t j = 1; j < count; j++) {
            float val = (j <= activeCount) ? d.GetValue(j - 1) : 0.0f;
            xPrev.SetValue(j, val);
        }
        // Scalar 写 xPrev → ComputeBackwardXLeft 通过 Vector 读 xPrev
        PipeBarrier<PIPE_ALL>();
        ComputeBackwardXLeft(aSave, bSave, cSave, dSave, xPrev, xLeft, tmp1, tmp2, d, count, off);

        auto xNew = oddABuf_.Get<float>();
        // Vector 读 xNew（上一步 AlignAndCopyBack）→ Scalar 重写 xNew（WAR）
        PipeBarrier<PIPE_ALL>();
        for (int32_t j = 0; j < activeCount; j++) {
            xNew.SetValue(2 * j, xLeft.GetValue(j));
            xNew.SetValue(2 * j + 1, d.GetValue(j));
        }
        // Scalar 写 xNew → AlignAndCopyBack 通过 Vector DataCopy 读 xNew
        PipeBarrier<PIPE_ALL>();
        AlignAndCopyBack(d, xNew, activeCount);
        // Vector DataCopy 写 d → 下一层 Scalar GetValue 读 d（WAR）
        PipeBarrier<PIPE_ALL>();
    }

    // CR 反向回代（逐层展开），save 数据驻留 UB
    __aicore__ inline void BackwardCR(LocalTensor<float> &d)
    {
        int32_t levelCounts[GTSV2_MAX_CR_LEVELS];
        int32_t levelOffsets[GTSV2_MAX_CR_LEVELS];
        CalcBackwardLevelInfo(levelCounts, levelOffsets);

        int32_t currentSize = 1;
        for (int32_t k = numLayers_ - 1; k >= 0; k--) {
            int32_t activeCount = currentSize;
            BackwardOneStep(d, activeCount, levelCounts[k], levelOffsets[k]);
            currentSize = activeCount * 2;
        }
    }

    // TPipe 指针（禁止 TPipe 作为成员变量）
    TPipe *pipe_{nullptr};

    // TQue：跨 Pipe 数据流（MTE2→Vector / Vector→MTE3）
    TQue<TPosition::VECIN, BUFFER_NUM> aInQue_, bInQue_, cInQue_, dInQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> dOutQue_;

    // TBuf：save buffer（按层 offset 切片，紧凑化几何求和）
    TBuf<TPosition::VECCALC> aSaveBuf_, bSaveBuf_, cSaveBuf_, dSaveBuf_;

    // TBuf：odd buffer（活跃位置紧凑化）
    TBuf<TPosition::VECCALC> oddABuf_, oddBBuf_, oddCBuf_, oddDBuf_;

    // TBuf：right buffer（右邻居数据，32B 对齐，供 Vector 指令使用）
    TBuf<TPosition::VECCALC> aRightBuf_, bRightBuf_, cRightBuf_, dRightBuf_;

    // TBuf：k1, k2 系数 + 临时计算 buffer
    TBuf<TPosition::VECCALC> k1Buf_, k2Buf_;
    TBuf<TPosition::VECCALC> t1Buf_, t2Buf_;

    // TBuf：Gather/Scatter 偏移表（uint32_t 字节偏移，跨 batch 复用）
    // buf1 常驻 evenOff；buf2 分阶段复用（外层路径在 InnerSolve 后重建为 oddOff）；buf3 xPrevOff（外层路径）
    TBuf<TPosition::VECCALC> sharedOffsetBuf1_;
    TBuf<TPosition::VECCALC> sharedOffsetBuf2_;
    TBuf<TPosition::VECCALC> sharedOffsetBuf3_;

    // TBuf：外层反向 x 左 halo 窗（P+8 floats，仅外层路径分配）
    TBuf<TPosition::VECCALC> xPrevSrcBuf_;

    GM_ADDR gmDl_{nullptr};
    GM_ADDR gmD_{nullptr};
    GM_ADDR gmDu_{nullptr};
    GM_ADDR gmX_{nullptr};
    GM_ADDR gmWorkspace_{nullptr};
    int32_t m_{0};
    int32_t mPad_{0};
    int32_t alignedM_{0};
    uint32_t vecBytes_{0};
    int32_t batchCount_{0};
    int32_t batchStride_{0};
    int32_t batchPerCoreBase_{0};
    int32_t remainder_{0};
    int32_t numLayers_{0};
    int32_t saveBufSize_{0};
    bool useVectorGather_{false};
    bool useGmWorkspace_{false};
    // Init 完整走完（guard 未触发）才置 true，Process 入口据此早退
    bool initOk_{false};

    // ---- 外层 GM 分块路径状态（全部 int64/size_t，防 2^31 附近溢出）----
    // cur 乒乓（RegionA/B 交替）与 x 乒乓（复用 RegionA/B 首段别名）
    GlobalTensor<float> curA_[2], curB_[2], curC_[2], curD_[2];
    GlobalTensor<float> xBuf_[2];
    GlobalTensor<float> saveRegion_;
    int64_t mPadOuter_{0};
    int64_t regionBByteOffset_{0};
    int64_t saveRegionByteOffset_{0};
    int64_t wsPerBatch_{0};
    int64_t sAFloats_{0};
    int64_t sBFloats_{0};
    int32_t outerLevels_{0};
    int32_t outerTileElems_{0};
};

// Kernel 入口函数
extern "C" __global__ __aicore__ void gtsv2_strided_batch_kernel(
    GM_ADDR gmDl, GM_ADDR gmD, GM_ADDR gmDu, GM_ADDR gmX, GM_ADDR gmBuffer,
    const Gtsv2StridedBatchTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    Gtsv2StridedBatchKernel kernel;
    kernel.Init(gmDl, gmD, gmDu, gmX, gmBuffer, &tiling, &pipe);
    kernel.Process();
}

// Kernel 启动器（host 侧调用，异步 launch）
extern "C" void gtsv2_strided_batch_kernel_do(
    GM_ADDR gmDl, GM_ADDR gmD, GM_ADDR gmDu, GM_ADDR gmX, GM_ADDR gmBuffer,
    const Gtsv2StridedBatchTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    gtsv2_strided_batch_kernel<<<numBlocks, nullptr, stream>>>(
        gmDl, gmD, gmDu, gmX, gmBuffer, tiling);
}
