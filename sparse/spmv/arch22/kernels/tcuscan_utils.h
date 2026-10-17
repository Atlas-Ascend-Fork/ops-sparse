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

#pragma once
#include <type_traits>

#include "kernel_operator.h"

namespace tcuscan {

/// @brief Number of bytes for a required alignment in UB.
constexpr uint16_t UB_ALIGNMENT = 32;
/// @brief Number of bytes per data block.
constexpr uint16_t DATA_BLOCK_SIZE = 32;

/**
 * @brief Get the size of the fractal used internally by hardware along the
 * matrix K dimension.
 *
 * @tparam DataType Data type used for matrix multiplication.
 * @return The size of K dimension of the fractal.
 */
template <typename T>
constexpr __aicore__ inline uint16_t GetFractalK() {
  return 32 / sizeof(T);
}

/**
 * @brief Get the size of the fractal used internally by hardware along the
 * matrix M/N dimensions.
 *
 * @tparam DataType Data type used for matrix multiplication.
 * @return The size of both M and N dimensions of the fractal.
 */
template <typename T>
constexpr __aicore__ inline uint16_t GetFractalMN() {
  return 16;
}

/**
 * @brief Copies tiling structure from global memory to registers.
 *
 * @tparam TilingT Structure representing kernel tiling parameters.
 * @param [in] tiling Pointer to the structure allocated in registers.
 * @param [in] tiling_global Pointer to the structure in global memory.
 */
template <typename TilingT>
__aicore__ inline void GetTilingData(TilingT* const tiling,
                                     GM_ADDR tiling_global) {
  uint32_t* const tiling_32b = reinterpret_cast<uint32_t*>(tiling);
  const __gm__ uint32_t* const tiling_global_32b =
      reinterpret_cast<__gm__ uint32_t*>(tiling_global);

  for (uint32_t i = 0; i < sizeof(TilingT) / sizeof(uint32_t); i++) {
    tiling_32b[i] = tiling_global_32b[i];
  }
}

namespace exec_mode {

/**
 * @brief Kernel for cube core that might be used to enable mix mode.
 *
 * This kernel does a single small cube core operation. It can be used to enable
 * mixed mode when only vector operations are used. This is useful when hw
 * synchronization is needed.
 *
 */
class KernelNoOpCube {
 public:
  /**
   * @brief Class constructor.
   *
   */
  __aicore__ inline KernelNoOpCube() {}
  /**
   * @brief Initialize buffers.
   *
   */
  __aicore__ inline void Init() {
    pipe.InitBuffer(a2_q_, 1, tile_size_ * sizeof(half));
    pipe.InitBuffer(b2_q_, 1, tile_size_ * sizeof(half));
    pipe.InitBuffer(co1_q_, 1, tile_size_ * sizeof(float));
  }
  /**
   * @brief Issue a small matrix multiplication instruction.
   *
   */
  __aicore__ inline void Process() {
    AscendC::LocalTensor<half> a2_lt = a2_q_.AllocTensor<half>();
    AscendC::LocalTensor<half> b2_lt = b2_q_.AllocTensor<half>();
    AscendC::LocalTensor<float> co1_lt = co1_q_.AllocTensor<float>();
    AscendC::Mmad(co1_lt, a2_lt, b2_lt,
         {tile_size_, tile_size_, tile_size_, false, 0, false, false, false});
    a2_q_.FreeTensor(a2_lt);
    b2_q_.FreeTensor(b2_lt);
    co1_q_.FreeTensor(co1_lt);
  }

 private:
  AscendC::TPipe pipe;

  AscendC::TQue<AscendC::QuePosition::A2, 1> a2_q_;
  AscendC::TQue<AscendC::QuePosition::B2, 1> b2_q_;
  AscendC::TQue<AscendC::QuePosition::CO1, 1> co1_q_;

  constexpr static uint16_t tile_size_ = 32;
};

/**
 * @brief Executes a small instruction on cube cores.
 *
 * Might be used to enable mix mode in a vector-only kernel.
 */
__aicore__ inline void EnableCubeCores() {
  if ASCEND_IS_AIC {
    KernelNoOpCube op;
    op.Init();
    op.Process();
  }
}

/**
 * @brief Makes the execution fail if the function is called from AIC core.
 *
 * Asserts only on CPU target, the check is done at runtime.
 */
__aicore__ inline void AssertIsAIV() {
#ifdef ASCENDC_CPU_DEBUG
  ASCENDC_ASSERT(g_coreType == AscendC::AIV, {
    KERNEL_LOG(KERNEL_ERROR, "The function can only be called on AIV cores.");
  });
#endif
}

/**
 * @brief Makes the execution fail if the function is called from AIV core.
 *
 * Asserts only on CPU target, the check is done at runtime.
 */
__aicore__ inline void AssertIsAIC() {
#ifdef ASCENDC_CPU_DEBUG
  ASCENDC_ASSERT(g_coreType == AscendC::AIC, {
    KERNEL_LOG(KERNEL_ERROR, "The function can only be called on AIC cores.");
  });
#endif
}

/**
 * @brief  Makes the execution fail if the LocalTensor is not in L1 (A1 or B1).
 *
 * Asserts only on CPU target, the check is done at runtime.
 *
 * @tparam T Local tensor data type.
 * @param lt Local tensor to check.
 */
// clang-format off
template <typename T>
__aicore__ inline void AssertLocalTensorIsInL1(
    [[maybe_unused]] const AscendC::LocalTensor<T> &lt) {
#ifdef ASCENDC_CPU_DEBUG
    const AscendC::TPosition pos = AscendC::TPosition(lt.GetPosition());
    ASCENDC_ASSERT(pos == AscendC::TPosition::A1 || pos == AscendC::TPosition::B1, {
        KERNEL_LOG(KERNEL_ERROR, "LocalTensor is not in L1 (A1 or B1).");
    });
#endif
}
// clang-format on

/**
 * @brief  Makes the execution fail if the LocalTensor is not in L0 (A2 or B2).
 *
 * Asserts only on CPU target, the check is done at runtime.
 *
 * @tparam T Local tensor data type.
 * @param lt Local tensor to check.
 */
// clang-format off
template <typename T>
__aicore__ inline void AssertLocalTensorIsInL0(
    [[maybe_unused]] const AscendC::LocalTensor<T> &lt) {
#ifdef ASCENDC_CPU_DEBUG
    const AscendC::TPosition pos = AscendC::TPosition(lt.GetPosition());
    ASCENDC_ASSERT(pos == AscendC::TPosition::A2 || pos == AscendC::TPosition::B2, {
        KERNEL_LOG(KERNEL_ERROR, "LocalTensor is not in L0 (A2 or B2).");
    });
#endif
}
// clang-format on

}  // namespace exec_mode

namespace copy {

/**
 * @brief Perform a copy from global tensor (ND layout) to local tensor (NZ
 * layout).
 *
 * @tparam DataType Data type of the tensors.
 * @tparam FractalHeight Height of the fractal used internally by hardware.
 * @tparam FractalWidth Width of the fractal used internally by hardware.
 *
 * @param [in] dst Destination local tensor.
 * @param [in] src Source global tensor.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input tensor.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input tensor.
 */
template <typename DataType, uint16_t FractalHeight = 16,
          uint16_t FractalWidth = 16>
__aicore__ inline void CopyND2NZ(const AscendC::LocalTensor<DataType>& dst,
                                 const AscendC::GlobalTensor<DataType>& src,
                                 uint16_t fractals_h, uint16_t fractals_w) {
  exec_mode::AssertLocalTensorIsInL1(dst);
  if (fractals_w == 1) {
    AscendC::DataCopy(dst, src, dst.GetSize());
    return;
  }
  AscendC::Nd2NzParams params;
  params.ndNum = 1;
  params.nValue = fractals_h * FractalHeight;
  params.dValue = fractals_w * FractalWidth;
  params.srcDValue = params.dValue;
  params.dstNzC0Stride = params.nValue;
  params.dstNzNStride = 1;
  // params.dstNzMatrixStride = 1; // This should have no effect when ndNum=1;
  AscendC::DataCopy(dst, src, params);
}

/**
 * @brief Copy a tensor from CO1 queue to global memory.
 *
 * @tparam DataType Data type of the source tensor.
 * @tparam QNumBuffers Depth of the queue.
 *
 * @param [in] global Destination global tensor.
 * @param [in] src_q Source queue. The position must be CO1.
 * @param [in] height Height of the matrix.
 * @param [in] width Width of the matrix.
 */
template <typename DataType, int32_t QNumBuffers>
__aicore__ inline void CopyCL0ToGlobal(
    const AscendC::GlobalTensor<DataType>& global,
    AscendC::TQue<AscendC::QuePosition::CO1, QNumBuffers>& src_q, uint32_t height,
    uint32_t width) {
  exec_mode::AssertIsAIC();
  constexpr uint16_t fractal_size = GetFractalMN<DataType>();

  AscendC::LocalTensor<DataType> lt = src_q.template DeQue<DataType>();

  AscendC::FixpipeParams<DataType> params;
  params.cburstNum = height;
  params.burstLen = width * fractal_size * sizeof(DataType) / DATA_BLOCK_SIZE;
  params.dstStride = height;

  AscendC::Nz2NdParams nz2nd_params;
  nz2nd_params.nz2ndEn = true;
  nz2nd_params.originalNSize = height;
  params.nz2ndParams = nz2nd_params;

  AscendC::Fixpipe(global, lt, params);

  src_q.FreeTensor(lt);
}

/**
 * @brief Copy a tensor from global memory to the local tensor in L1 memory.
 *
 * The queue's position must be either A1 or B1. The data layout is transformed
 * from ND to NZ.
 *
 * @tparam FractalHeight Height of the fractal used internally by hardware.
 * @tparam FractalWidth Width of the fractal used internally by hardware.
 * @tparam DataType Data type of the tensor.
 *
 * @param [in] local Destination local tensor. Must be alocated from either A1
 * or B1 queue.
 * @param [in] global Source global tensor.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input matrix.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input matrix.
 */
template <uint16_t FractalHeight, uint16_t FractalWidth, typename DataType>
__aicore__ inline void CopyGmToL1(const AscendC::LocalTensor<DataType>& local,
                                  const AscendC::GlobalTensor<DataType>& global,
                                  uint16_t fractals_h, uint16_t fractals_w) {
  exec_mode::AssertIsAIC();
  exec_mode::AssertLocalTensorIsInL1(local);
  tcuscan::copy::CopyND2NZ<DataType, FractalHeight, FractalWidth>(
      local, global, fractals_h, fractals_w);
}

/**
 * @brief Copy a tensor from global memory to the queue in A L1 memory.
 *
 * The data layout is transformed from ND to NZ.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam QNumBuffers Depth of the queue.
 *
 * @param [in] q Destination queue. The position must be A1.
 * @param [in] global Source global tensor.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input matrix.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input matrix.
 */
template <typename DataType, int32_t QNumBuffers>
__aicore__ inline void CopyGmToL1A(AscendC::TQue<AscendC::QuePosition::A1, QNumBuffers>& q,
                                   const AscendC::GlobalTensor<DataType>& global,
                                   uint16_t fractals_h, uint16_t fractals_w) {
  exec_mode::AssertIsAIC();
  const AscendC::LocalTensor<DataType> lt = q.template AllocTensor<DataType>();
  tcuscan::copy::CopyGmToL1<GetFractalMN<DataType>(), GetFractalK<DataType>()>(
      lt, global, fractals_h, fractals_w);
  q.EnQue(lt);
}

/**
 * @brief Copy a tensor from global memory to the queue in B L1 memory.
 *
 * The data layout is transformed from ND to NZ.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam QNumBuffers Depth of the queue.
 *
 * @param [in] q Destination queue. The position must be B1.
 * @param [in] global Source global tensor.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input matrix.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input matrix.
 */
template <typename DataType, int32_t QNumBuffers>
__aicore__ inline void CopyGmToL1B(AscendC::TQue<AscendC::QuePosition::B1, QNumBuffers>& q,
                                   const AscendC::GlobalTensor<DataType>& global,
                                   uint16_t fractals_h, uint16_t fractals_w) {
  exec_mode::AssertIsAIC();
  const AscendC::LocalTensor<DataType> lt = q.template AllocTensor<DataType>();
  tcuscan::copy::CopyGmToL1<GetFractalK<DataType>(), GetFractalMN<DataType>()>(
      lt, global, fractals_h, fractals_w);
  q.EnQue(lt);
}

/**
 * @brief Copy a tensor from global memory to the queue in B L0 memory.
 *
 * The data layout is transformed from column-major to NZ.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam QNumBuffersB2 Depth of the B2 queue.
 * @tparam QNumBuffersB1 Depth of the B1 queue.
 *
 * @param [in] b2_q Destination queue. The position must be B2.
 * @param [in] b1_q Intermediate queue. The position must be B1.
 * @param [in] global Source global tensor.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input matrix.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input matrix.
 */
template <typename DataType, int32_t QNumBuffersB2, int32_t QNumBuffersB1>
__aicore__ inline void CopyTransposedGmToL0B(
    AscendC::TQue<AscendC::QuePosition::B2, QNumBuffersB2>& b2_q,
    AscendC::TQue<AscendC::QuePosition::B1, QNumBuffersB1>& b1_q,
    const AscendC::GlobalTensor<DataType>& global, uint16_t fractals_h,
    uint16_t fractals_w) {
  exec_mode::AssertIsAIC();
  // Copy with ND -> NZ transform.
  // But data in GM is transposed (column-major layout) so the resulting
  // layout is ZN (instead of NZ). And ZN is a layout `Mmad` expects for the B
  // operand.
  tcuscan::copy::CopyGmToL1B(b1_q, global, fractals_h, fractals_w);

  // Plain copy from L1 to L0B, because the layout is already correct.
  AscendC::LocalTensor<DataType> src = b1_q.template DeQue<DataType>();
  const AscendC::LocalTensor<DataType> dst = b2_q.template AllocTensor<DataType>();

  AscendC::LoadData2dParams params;
  params.repeatTimes = fractals_w * fractals_h;
  params.srcStride = 1;
  params.ifTranspose = false;

  AscendC::LoadData(dst, src, params);

  b1_q.FreeTensor(src);
  b2_q.EnQue(dst);
}

/**
 * @brief Copy data from tensor allocated in L1 to tensor allocated in L0.
 *
 * @tparam FractalHeight Height of the fractal used internally by hardware.
 * @tparam FractalWidth Width of the fractal used internally by hardware.
 * @tparam DataType Data type of the tensor.
 *
 * @param [in] dst Destination local tensor.
 * @param [in] src Source local tensor.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input tensor.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input tensor.
 * @param [in] transpose Indicates whether or not to transpose the matrix. If
 * set, the function performs an NZ to ZN transformation. Otherwise, it performs
 * NZ to ZZ one.
 */
template <uint16_t FractalHeight, uint16_t FractalWidth, typename DataType>
__aicore__ inline void CopyL1ToL0(const AscendC::LocalTensor<DataType>& dst,
                                  const AscendC::LocalTensor<DataType>& src,
                                  uint16_t fractals_h, uint16_t fractals_w,
                                  bool transpose) {
  exec_mode::AssertIsAIC();
  exec_mode::AssertLocalTensorIsInL1(src);
  exec_mode::AssertLocalTensorIsInL0(dst);

  int src_offset = 0;
  int dst_offset = 0;

  for (uint16_t i = 0; i < fractals_h; ++i) {
    AscendC::LoadData2dParams params;
    params.repeatTimes = fractals_w;
    params.srcStride = fractals_h;
    params.ifTranspose = transpose;

    AscendC::LoadData(dst[dst_offset], src[src_offset], params);

    src_offset += FractalWidth * FractalHeight;
    dst_offset += fractals_w * FractalWidth * FractalHeight;
  }
}

/**
 * @brief Copy a tensor from a given A L1 to A L0 queue.
 *
 * The function performs an NZ to ZZ layout transformation.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam FreeSrc Indicates whether the source tensor should be freed or
 * enqueued back.
 * @tparam L0NumBuffers Depth of the destination queue.
 * @tparam L1NumBuffers Depth of the source queue.
 *
 * @param [in] l0_q Destination queue. The position must be A2.
 * @param [in] l1_q Source queue. The position must be A1.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input matrix.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input matrix.
 */
template <typename DataType, bool FreeSrc, int32_t L0NumBuffers,
          int32_t L1NumBuffers>
__aicore__ inline void CopyL1ToL0A(AscendC::TQue<AscendC::QuePosition::A2, L0NumBuffers>& l0_q,
                                   AscendC::TQue<AscendC::QuePosition::A1, L1NumBuffers>& l1_q,
                                   uint16_t fractals_h, uint16_t fractals_w) {
  exec_mode::AssertIsAIC();
  const AscendC::LocalTensor<DataType> l0_lt = l0_q.template AllocTensor<DataType>();
  AscendC::LocalTensor<DataType> l1_lt = l1_q.template DeQue<DataType>();

  CopyL1ToL0<GetFractalMN<DataType>(), GetFractalK<DataType>()>(
      l0_lt, l1_lt, fractals_h, fractals_w, false /* transpose */);

  if constexpr (FreeSrc) {
    l1_q.FreeTensor(l1_lt);
  } else {
    l1_q.EnQue(l1_lt);
  }
  l0_q.EnQue(l0_lt);
}

/**
 * @brief Copy a tensor from global memory to the VECIN queue.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam QNumBuffers Depth of the queue.
 *
 * @param [in] q Destination queue. The position must be VECIN.
 * @param [in] global Source global tensor.
 * @param [in] num_elems Number of elements to load. By default, the function
 * loads all the elements for the local tensor obtained from the queue. If the
 * value is provided, the function will load \f$num_elems\f$ elements.
 * @param [in] pad_value Value use for padding when \f$num_elems\f$ is not
 * aligned to UB. It defaults to 0.
 */
template <typename DataType, int32_t QNumBuffers>
__aicore__ inline void CopyGmToVec(AscendC::TQue<AscendC::QuePosition::VECIN, QNumBuffers>& q,
                                   const AscendC::GlobalTensor<DataType>& global,
                                   uint32_t num_elems = 0,
                                   DataType pad_value = 0) {
  exec_mode::AssertIsAIV();
  const AscendC::LocalTensor<DataType> lt = q.template AllocTensor<DataType>();
  if (!num_elems) { num_elems = lt.GetSize(); }
  if (num_elems % (UB_ALIGNMENT / sizeof(DataType)) == 0) {
    AscendC::DataCopy(lt, global, num_elems);
  } else {
    const uint32_t align_len =
        static_cast<uint32_t>(UB_ALIGNMENT / sizeof(DataType));
    const uint8_t pad_len =
        static_cast<uint8_t>(align_len - num_elems % align_len);

    AscendC::DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = num_elems * sizeof(DataType);
    params.srcStride = 0;
    params.dstStride = 0;

    AscendC::DataCopyPadExtParams<DataType> pad_params;
    pad_params.isPad = true;
    pad_params.leftPadding = 0;
    pad_params.rightPadding = pad_len;
    pad_params.paddingValue = static_cast<DataType>(pad_value);

    AscendC::DataCopyPad(lt, global, params, pad_params);
  }
  q.EnQue(lt);
}

/**
 * @brief Copy a tensor from a given queue to global memory.
 *
 * The queue's position must be VECOUT.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam VecNumBuffers Depth of the queue.
 *
 * @param [in] global Destination global tensor.
 * @param [in] q Source queue. The position must be VECOUT.
 * @param [in] num_elems Number of elements to store. By default, the function
 * stores all the elements for the local tensor obtained from the queue. If the
 * value is provided, the function will store \f$num_elems\f$ elements.
 */
template <typename DataType, int32_t VecNumBuffers>
__aicore__ inline void CopyVecToGm(const AscendC::GlobalTensor<DataType>& global,
                                   AscendC::TQue<AscendC::QuePosition::VECOUT, VecNumBuffers>& q,
                                   uint32_t num_elems = 0) {
  exec_mode::AssertIsAIV();
  AscendC::LocalTensor<DataType> lt = q.template DeQue<DataType>();
  if (!num_elems) {
    num_elems = lt.GetSize();
  }

  if ((num_elems * sizeof(DataType)) % UB_ALIGNMENT == 0) {
    AscendC::DataCopy(global, lt, num_elems);
  } else {
    AscendC::DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = num_elems * sizeof(DataType);
    params.srcStride = 0;
    params.dstStride = 0;
    AscendC::DataCopyPad(global, lt, params);
  }
  q.FreeTensor(lt);
}

}  // namespace copy

namespace queue {

/**
 * @brief Free a tensor from a given queue.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam Q Type of the queue.
 *
 * @param [in] q Queue from which to free a tensor.
 */
template <typename DataType, typename Q>
__aicore__ inline void FreeFromQ(Q& q) {
  AscendC::LocalTensor<DataType> lt = q.template DeQue<DataType>();
  q.FreeTensor(lt);
}

}  // namespace queue

namespace sync {

/**
 * @brief Used to specifies the direction of synchronization when synchronizing
 * cube and vectors within a single group.
 *
 * A single group consists of one cube core and two vector cores.
 *
 * Can be used to specify either the symmetric or asymetric synchronization.
 */
enum class GroupSyncDirection {
  /// Asymetric synchronization - cube continues execution only after vectors
  /// reach the synchronization point. Can be used when cube consumes the data
  /// produced by vectors from the same group.
  CUBE_WAIT_FOR_VEC,
  /// Asymetric synchronization - vectors continue execution only after cube
  /// reaches the synchronization point. Can be used when vectors consume the
  /// data produced by cube from the same group.
  VEC_WAIT_FOR_CUBE,
  /// Symmetric synchronization - execution continues after cubes and vectors
  /// synchronize at the same time
  FULL
};

/**
 * @brief Returns a synchronization config.
 *
 * @param [in] mode Synchronization mode.
 * @param [in] flag_id Flag to use for synchronization.
 * @return Synchronization config.
 */
__aicore__ inline int GetSyncConf(int mode, int flag_id) {
  return 1 | (mode << 4) | (flag_id << 8);
}

/**
 * @brief Synchronize cube and vector cores within a single group.
 *
 * @tparam Dir Direction of the synchronization.
 */
template <GroupSyncDirection Dir = GroupSyncDirection::FULL>
__aicore__ inline void SyncGroup() {
  const int mode = 2;

  if constexpr (Dir == GroupSyncDirection::CUBE_WAIT_FOR_VEC) {
    const int AIV_SET_FLAG_ID = 11;
    if ASCEND_IS_AIV {
      ffts_cross_core_sync(PIPE_MTE3, GetSyncConf(mode, AIV_SET_FLAG_ID));
    }
    if ASCEND_IS_AIC {
      wait_flag_dev(AIV_SET_FLAG_ID);
    }
    return;
  }
  if constexpr (Dir == GroupSyncDirection::VEC_WAIT_FOR_CUBE) {
    const int AIC_SET_FLAG_ID = 12;
    if ASCEND_IS_AIC {
      ffts_cross_core_sync(PIPE_FIX, GetSyncConf(mode, AIC_SET_FLAG_ID));
    }
    if ASCEND_IS_AIV {
      wait_flag_dev(AIC_SET_FLAG_ID);
    }
    return;
  }
  if constexpr (Dir == GroupSyncDirection::FULL) {
    const int AIV_SET_FLAG_ID = 11;
    const int AIC_SET_FLAG_ID = 12;
    if ASCEND_IS_AIV {
      ffts_cross_core_sync(PIPE_MTE3, GetSyncConf(mode, AIV_SET_FLAG_ID));
      wait_flag_dev(AIC_SET_FLAG_ID);
    }
    if ASCEND_IS_AIC {
      ffts_cross_core_sync(PIPE_FIX, GetSyncConf(mode, AIC_SET_FLAG_ID));
      wait_flag_dev(AIV_SET_FLAG_ID);
    }
    return;
  }
}

}  // namespace sync

namespace data_cache {

/**
 * @brief Invalidates a single cacheline in data cache corresponding to the
 * global memory address.
 *
 * @tparam T Data type of the buffer in global memory.
 *
 * @param global [in] Global tensor which first values are invalidated in data
 * cache.
 */
template <typename T>
__aicore__ inline void InvalidateLine(const AscendC::GlobalTensor<T>& global) {
  AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE,
                           AscendC::DcciDst::CACHELINE_OUT>(global);
}
}  // namespace data_cache

namespace scalar {

/**
 * @brief Rounds an integral value up to the nearest multiple of a given
 * alignment.
 *
 * @tparam T Data type of the integral length.
 * @param [in] length Input length.
 * @param [in] alignment Alignment to use.
 * @return Aligned length.
 */
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
__aicore__ inline T AlignUp(T length, uint32_t alignment) {
  const T tail = length % alignment;
  if (!tail) {
    return length;
  }
  const T padding = alignment - tail;
  return length + padding;
}

/**
 * @brief Performs a division on two integral numbers and rounds the result up
 * to the nearest integer.
 *
 * @tparam T1 Data type of dividend.
 * @tparam T2 Data type of divisor.
 * @param [in] value Dividend.
 * @param [in] divisor Divisor.
 * @return Result of division.
 */
template <typename T1, typename T2,
          typename std::enable_if<std::is_integral<T1>::value &&
                                      std::is_integral<T2>::value,
                                  int>::type = 0>
__aicore__ inline T1 CeilDiv(T1 value, T2 divisor) {
  return (value + divisor - 1) / divisor;
}

/**
 * @brief Performs a division on two integral numbers and rounds the result down
 * to the nearest integer.
 *
 * @tparam T1 Data type of dividend.
 * @tparam T2 Data type of divisor.
 * @param [in] value Dividend.
 * @param [in] divisor Divisor.
 * @return Result of division.
 */
template <typename T1, typename T2,
          typename std::enable_if<std::is_integral<T1>::value &&
                                      std::is_integral<T2>::value,
                                  int>::type = 0>
__aicore__ inline T1 FloorDiv(T1 value, T2 divisor) {
  return value / divisor;
}

/**
 * @brief Returns the next tile length, given the global memory offset. The
 * returned tile length equals typically to `tile_len`, expect the last
 * iteration where the tile length is smaller than `tile_len`.
 *
 * @param tile_len Tile length
 * @param global_offset Global memory offset
 * @param length Total vector length
 * @return Length of "next" tile, given the current global memory offset.
 */
__aicore__ inline uint32_t NextTileLen(uint32_t tile_len,
                                       uint32_t global_offset,
                                       uint32_t length) {
  if (length <= global_offset) {
    return 0;
  }
  const bool full_tile = global_offset + tile_len <= length;
  const uint32_t num_elems_to_process =
      full_tile ? tile_len : length - global_offset;

  return num_elems_to_process;
}

/**
 * @brief Defines how the workload should be distributed among cores.
 *
 * The function returns the number of tiles to be processed by each block so
 * that the depth of execution is minimized. If the workload is not balanced it
 * will greedily assign as many tiles as possible starting from the first block,
 * but keeping the maximum depth optimal. If `vec_len` is not divisible by
 * `tile_size` the last tile will be smaller.
 *
 * @param [in] vec_len Size of the input vector.
 * @param [in] tile_size Tile size.
 * @param [in] block_n Number of blocks.
 *
 * @return Number of tiles assigned to the block calling the function.
 */
__aicore__ inline uint32_t GetWorkDistribution(uint32_t vec_len,
                                               uint32_t tile_size,
                                               uint32_t block_n) {
  const uint32_t num_tiles = scalar::CeilDiv(vec_len, tile_size);
  const uint32_t max_num_tiles_per_block = scalar::CeilDiv(num_tiles, block_n);
  uint32_t num_tiles_to_process = max_num_tiles_per_block;
  const int tiles_left =
      (int)num_tiles - (int)(AscendC::GetBlockIdx() * max_num_tiles_per_block);

  if (tiles_left < 0) {
    num_tiles_to_process = 0;
  } else if (tiles_left < static_cast<int>(max_num_tiles_per_block)) {
    num_tiles_to_process = tiles_left;
  }
  return num_tiles_to_process;
}

/**
 * @brief Reads a value from global memory.
 *
 * @tparam T Data type.
 * @param [in] addr Address in GM.
 * @param [in] offset Offset.
 * @param [in] vec_len Size of the global buffer.
 * @return Value from global memory.
 */
template <typename T>
__aicore__ inline T GetGMValue(GM_ADDR addr, uint32_t offset,
                               uint32_t vec_len) {
  ASCENDC_ASSERT(offset < vec_len, {
    KERNEL_LOG(KERNEL_ERROR,
               "GetGMValue is trying to access data out of bounds.");
  });

  AscendC::GlobalTensor<T> gt;
  gt.SetGlobalBuffer((__gm__ T*)addr, vec_len);

  data_cache::InvalidateLine(gt);

  return gt.GetValue(offset);
}

/**
 * @brief Binary search (lower_bound) of @p value into a sorted GM array.
 *
 * Reproduces `torch.searchsorted(sorted, value, side='left')`: returns the
 * first index `j` such that `sorted[j] >= value` (equivalently, the count of
 * elements `< value`). The haystack is read one element at a time from global
 * memory via `GetGMValue`, so no Unified Buffer staging is required.
 *
 * @tparam T Data type of the sorted array and search value.
 *
 * @param [in] sorted Pointer to the sorted (ascending) array in global memory.
 * @param [in] len Number of elements in @p sorted.
 * @param [in] value Value to search for.
 * @return Insertion index in `[0, len]`.
 */
template <typename T>
__aicore__ inline uint32_t LowerBoundGM(GM_ADDR sorted, uint32_t len, T value) {
  uint32_t lo = 0;
  uint32_t hi = len;
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo) / 2;
    const T pivot = GetGMValue<T>(sorted, mid, len);
    if (pivot < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/**
 * @brief Calculates minimum of two numbers.
 *
 * @tparam T Data type of the values.
 * @param [in] v1 First value.
 * @param [in] v2 Second value.
 * @return Minimum value.
 */
template <typename T>
__aicore__ inline T Min(T v1, T v2) {
  return v1 <= v2 ? v1 : v2;
}

/**
 * @brief Specialization since AI-Core does not support half comparisons.
 *
 * Avoids the compilation message: `error: half precision operation is not
 * allowed in aicore function`
 *
 * @param [in] v1 First value.
 * @param [in] v2 Second value.
 * @return Minimum value.
 */
template <>
__aicore__ inline half Min(half v1, half v2) {
  return static_cast<float>(v1) <= static_cast<float>(v2) ? v1 : v2;
}

}  // namespace scalar

namespace cube_unit {

/**
 * @brief It is true if `DataType` is supported as input for the Cube unit.
 *
 * @tparam DataType Data type to check.
 */
template <typename DataType>
constexpr bool IsCubeSupported =
    std::is_same_v<DataType, half> || std::is_same_v<DataType, float> ||
    std::is_same_v<DataType, int8_t> || std::is_same_v<DataType, uint8_t>;

/**
 * @brief A type metafunction for Cube's input / output types. The following
 * type pairs are supported:(half, float), (float, float), (int8_t, int32_t),
 * (uint8_t, uint32_t)
 *
 * @tparam InputT Input cube type. Must be int8_t, uint8_t, half, or float.
 */
template <typename InputT>
struct CubeOutType {
  /// @brief Type
  using type = InputT;
};

/**
 * @brief Cube data type map int8_t -> int32_t.
 */
template <>
struct CubeOutType<int8_t> {
  /// @brief Type
  using type = int32_t;
};

/**
 * @brief Cube data type map uint8_t -> uint32_t.
 */
template <>
struct CubeOutType<uint8_t> {
  /// @brief Type
  using type = uint32_t;
};

/**
 * @brief Cube data type map half -> float.
 */
template <>
struct CubeOutType<half> {
  /// @brief Type
  using type = float;
};

/**
 * @brief Cube data type map float -> float.
 */
template <>
struct CubeOutType<float> {
  /// @brief Type
  using type = float;
};

/**
 * @brief Syntactic sugar for `CubeOutType`
 * @tparam T Input type
 */
template <typename T>
using CubeOutType_t = typename CubeOutType<T>::type;

/**
 * @brief Multiply two matrices given their queues.
 *
 * The function deques the input matrices from A2 and B2 and computes their
 * product using the cube unit. The output can be accumulated.
 * After execution the input matrices are either freed or enqued again if they
 * need to be reused.
 *
 * @tparam InputT Data type of the input matrices.
 * @tparam accumulate_c If true accumulates the output adding it to the matrix
 * in the CO1 queue.
 * @tparam free_a If true it frees the first input matrix, if false it enques it
 * to reuse it.
 * @tparam free_b If true it frees the second input matrix, if false it enques
 * it to reuse it.
 *
 * @param [in] q_a Input queue for the matrix A with position A2.
 * @param [in] q_b Input queue for the matrix B with position B2.
 * @param [in] q_c Output queue for the matrix C with position CO1.
 * @param [in] M Height of matrix A.
 * @param [in] N Width of matrix B.
 * @param [in] K Width of matrix A and hwight of matrix B.
 */
template <typename InputT, bool accumulate_c = false, bool free_a = true,
          bool free_b = true>
__aicore__ inline void Multiply(AscendC::TQue<AscendC::QuePosition::A2, 1>& q_a,
                                AscendC::TQue<AscendC::QuePosition::B2, 1>& q_b,
                                AscendC::TQue<AscendC::QuePosition::CO1, 1>& q_c, uint16_t M,
                                uint16_t N, uint16_t K) {
  exec_mode::AssertIsAIC();
  static_assert(IsCubeSupported<InputT>,
                "Unsupported input Cube dtype. Please use half or int8_t.");
  using OutputT = CubeOutType_t<InputT>;
  AscendC::LocalTensor<InputT> a2_lt = q_a.DeQue<InputT>();
  AscendC::LocalTensor<InputT> b2_lt = q_b.DeQue<InputT>();
  AscendC::LocalTensor<OutputT> c1_lt =
      accumulate_c ? q_c.DeQue<OutputT>() : q_c.AllocTensor<OutputT>();

  AscendC::Mmad(c1_lt, a2_lt, b2_lt, {M, N, K, accumulate_c, 0, false, false, false});

  q_c.EnQue<OutputT>(c1_lt);

  if constexpr (free_a) {
    q_a.FreeTensor(a2_lt);
  } else {
    q_a.EnQue(a2_lt);
  }

  if constexpr (free_b) {
    q_b.FreeTensor(b2_lt);
  } else {
    q_b.EnQue(b2_lt);
  }
}

}  // namespace cube_unit

}  // namespace tcuscan
