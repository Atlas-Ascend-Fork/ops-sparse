// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// Execute the production loop with host-only AscendC test doubles. This checks
// integer traversal and tail lengths, not device DMA, synchronization or values.
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include "../../../../sparse/spmv/arch22/kernels/spmv_trans_finalize.h"

template <typename T>
void CheckBlock(uint32_t count, uint32_t blocks, uint32_t block, float beta)
{
    using namespace AscendC;
    blockIdx = block;
    blockNum = blocks;
    AscendC::count = count;
    expectedOffset = uint64_t{block} * 256;
    expectedStride = uint64_t{blocks} * 256;
    copies = elements = 0;
    const uint64_t first = expectedOffset;
    const uint64_t expectedCopies = first < count ? (count - 1 - first) / expectedStride + 1 : 0;
    uint64_t expectedElements = expectedCopies * 256;
    if (expectedCopies != 0) {
        const uint64_t last = first + (expectedCopies - 1) * expectedStride;
        const uint64_t remaining = count - last;
        if (remaining < 256) {
            expectedElements -= 256 - remaining;
        }
    }
    SpmvTransFinalize<T>(nullptr, nullptr, count, beta);
    SpmvFinalizeTest::CheckCondition(copies == expectedCopies, "incorrect copy count");
    SpmvFinalizeTest::CheckCondition(elements == expectedElements, "incorrect element count");
    SpmvFinalizeTest::CheckCondition(expectedOffset >= count, "loop stopped before output end");
}

template <typename T> void CheckType()
{
    constexpr uint32_t max = std::numeric_limits<uint32_t>::max();
    for (float beta : {0.0f, 1.0f}) {
        for (uint32_t count : {0u, 1u, 255u, 256u, 257u, max - 255, max - 1, max}) {
            for (uint32_t blocks : {1u, 3u, 32u}) {
                for (uint32_t block = 0; block < blocks; ++block) {
                    CheckBlock<T>(count, blocks, block, beta);
                }
            }
        }
        // Check promotion BEFORE multiplication for both start and stride.
        // These are arithmetic stress cases, not physical core counts.
        CheckBlock<T>(max, (1u << 24) + 1, 0, beta);
        CheckBlock<T>(max, (1u << 24) + 1, 1u << 24, beta);
        CheckBlock<T>(max, max, max - 1, beta);
    }
}

int main()
{
    try {
        CheckType<AscendC::Half>();
        CheckType<AscendC::Bfloat16>();
    } catch (const std::exception &error) {
        std::fprintf(stderr, "SpMV transpose finalize boundary check failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
    std::puts("SpMV transpose finalize integer boundary checks passed");
    return EXIT_SUCCESS;
}
