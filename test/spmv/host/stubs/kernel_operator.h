// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// Host-only integer traversal test doubles; no DMA, arithmetic or memory access.
#ifndef SPMV_FINALIZE_TEST_KERNEL_OPERATOR_H
#define SPMV_FINALIZE_TEST_KERNEL_OPERATOR_H

#include <cstdint>
#include <stdexcept>

namespace SpmvFinalizeTest {
inline void CheckCondition(bool condition, const char *message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}
} // namespace SpmvFinalizeTest

#define __global__
#define __aicore__
#define __gm__
using GM_ADDR = uint8_t *;
constexpr int PIPE_V = 0;

namespace AscendC {
enum class TPosition { VECCALC };
enum class HardEvent { MTE2_V, V_MTE3, MTE3_MTE2 };
enum class RoundMode { CAST_NONE, CAST_ROUND };
struct Half { uint16_t bits; Half(uint16_t value = 0) : bits(value) {} };
struct Bfloat16 { uint16_t bits; Bfloat16(uint16_t value = 0) : bits(value) {} };
inline uint32_t blockIdx, blockNum;
inline uint64_t expectedOffset, expectedStride, copies, elements;
inline uint32_t count;
inline uint32_t GetBlockIdx() { return blockIdx; }
inline uint32_t GetBlockNum() { return blockNum; }

template <typename T> struct LocalTensor {};
template <TPosition> struct TBuf {
    template <typename T> LocalTensor<T> Get() { return {}; }
};
struct TPipe {
    template <typename T> void InitBuffer(T &, uint32_t) {}
};
template <typename T> struct GlobalTensor {
    uint64_t offset = 0;
    void SetGlobalBuffer(T *, uint32_t) {}
    GlobalTensor operator[](uint64_t index) const { return {index}; }
};
struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen, srcStride, dstStride, rsv;
};
template <typename T> struct DataCopyPadExtParams {
    bool isPad;
    uint8_t left, right;
    T value;
};
template <typename T>
void DataCopyPad(LocalTensor<T>, GlobalTensor<T> source,
                 DataCopyExtParams params, DataCopyPadExtParams<T>)
{
    SpmvFinalizeTest::CheckCondition(source.offset == expectedOffset, "unexpected source offset");
    SpmvFinalizeTest::CheckCondition(source.offset < count, "source offset out of range");
    const uint64_t remaining = count - source.offset;
    SpmvFinalizeTest::CheckCondition(
        params.blockLen == (remaining < 256 ? remaining : 256) * sizeof(T), "incorrect copy length");
}
template <typename T>
void DataCopyPad(GlobalTensor<T> destination, LocalTensor<T>, DataCopyExtParams params)
{
    // Reject a wrapped offset immediately, so the old kernel fails without hanging.
    SpmvFinalizeTest::CheckCondition(destination.offset == expectedOffset, "unexpected destination offset");
    SpmvFinalizeTest::CheckCondition(destination.offset < count, "destination offset out of range");
    elements += params.blockLen / sizeof(T);
    ++copies;
    expectedOffset += expectedStride;
}
template <typename T, typename U>
void Cast(LocalTensor<T>, LocalTensor<U>, RoundMode, uint32_t) {}
template <int> void PipeBarrier() {}
template <typename T> void Muls(LocalTensor<T>, LocalTensor<T>, float, uint32_t) {}
template <typename T>
void Add(LocalTensor<T>, LocalTensor<T>, LocalTensor<T>, uint32_t) {}
} // namespace AscendC

template <AscendC::HardEvent> void SpmvSyncEvent(AscendC::TPipe &) {}
#endif
