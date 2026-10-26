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
#ifndef SDDMM_HOST_TILING_ABI_H_
#define SDDMM_HOST_TILING_ABI_H_

#include <cstddef>
#include <type_traits>
#include "sddmm_tiling.h"
#include "../kernel/sddmm_tiling.h"

static_assert(std::is_standard_layout<SddmmTilingData>::value && std::is_standard_layout<SddmmKernelTilingData>::value,
              "tiling must have a stable layout");
static_assert(std::is_trivially_copyable<SddmmTilingData>::value &&
                  std::is_trivially_copyable<SddmmKernelTilingData>::value,
              "tiling must be byte-copyable");
static_assert(!std::is_same<SddmmTilingData, SddmmKernelTilingData>::value, "Host and Kernel tiling types must differ");
static_assert(sizeof(SddmmTilingData) == sizeof(SddmmKernelTilingData), "tiling size mismatch");
static_assert(alignof(SddmmTilingData) == alignof(SddmmKernelTilingData), "tiling alignment mismatch");
static_assert(offsetof(SddmmTilingData, nnz) == offsetof(SddmmKernelTilingData, nnz), "tiling field nnz mismatch");
static_assert(offsetof(SddmmTilingData, rowCount) == offsetof(SddmmKernelTilingData, rowCount),
              "tiling field rowCount mismatch");
static_assert(offsetof(SddmmTilingData, matARows) == offsetof(SddmmKernelTilingData, matARows),
              "tiling field matARows mismatch");
static_assert(offsetof(SddmmTilingData, matACols) == offsetof(SddmmKernelTilingData, matACols),
              "tiling field matACols mismatch");
static_assert(offsetof(SddmmTilingData, matBRows) == offsetof(SddmmKernelTilingData, matBRows),
              "tiling field matBRows mismatch");
static_assert(offsetof(SddmmTilingData, matBCols) == offsetof(SddmmKernelTilingData, matBCols),
              "tiling field matBCols mismatch");
static_assert(offsetof(SddmmTilingData, blockDim) == offsetof(SddmmKernelTilingData, blockDim),
              "tiling field blockDim mismatch");
static_assert(offsetof(SddmmTilingData, rowsPerCore) == offsetof(SddmmKernelTilingData, rowsPerCore),
              "tiling field rowsPerCore mismatch");
static_assert(offsetof(SddmmTilingData, remainderRows) == offsetof(SddmmKernelTilingData, remainderRows),
              "tiling field remainderRows mismatch");
static_assert(offsetof(SddmmTilingData, kBlockLen) == offsetof(SddmmKernelTilingData, kBlockLen),
              "tiling field kBlockLen mismatch");
static_assert(offsetof(SddmmTilingData, nnzPartition) == offsetof(SddmmKernelTilingData, nnzPartition),
              "tiling field nnzPartition mismatch");
static_assert(offsetof(SddmmTilingData, nnzPerCore) == offsetof(SddmmKernelTilingData, nnzPerCore),
              "tiling field nnzPerCore mismatch");
static_assert(offsetof(SddmmTilingData, remainderNnz) == offsetof(SddmmKernelTilingData, remainderNnz),
              "tiling field remainderNnz mismatch");
static_assert(offsetof(SddmmTilingData, alphaHost) == offsetof(SddmmKernelTilingData, alphaHost),
              "tiling field alphaHost mismatch");
static_assert(offsetof(SddmmTilingData, betaHost) == offsetof(SddmmKernelTilingData, betaHost),
              "tiling field betaHost mismatch");
static_assert(offsetof(SddmmTilingData, alphaImagHost) == offsetof(SddmmKernelTilingData, alphaImagHost),
              "tiling field alphaImagHost mismatch");
static_assert(offsetof(SddmmTilingData, betaImagHost) == offsetof(SddmmKernelTilingData, betaImagHost),
              "tiling field betaImagHost mismatch");
static_assert(offsetof(SddmmTilingData, opX) == offsetof(SddmmKernelTilingData, opX), "tiling field opX mismatch");
static_assert(offsetof(SddmmTilingData, opY) == offsetof(SddmmKernelTilingData, opY), "tiling field opY mismatch");
static_assert(offsetof(SddmmTilingData, ldx) == offsetof(SddmmKernelTilingData, ldx), "tiling field ldx mismatch");
static_assert(offsetof(SddmmTilingData, ldy) == offsetof(SddmmKernelTilingData, ldy), "tiling field ldy mismatch");
static_assert(offsetof(SddmmTilingData, orderX) == offsetof(SddmmKernelTilingData, orderX),
              "tiling field orderX mismatch");
static_assert(offsetof(SddmmTilingData, orderY) == offsetof(SddmmKernelTilingData, orderY),
              "tiling field orderY mismatch");
static_assert(offsetof(SddmmTilingData, xType) == offsetof(SddmmKernelTilingData, xType),
              "tiling field xType mismatch");
static_assert(offsetof(SddmmTilingData, cType) == offsetof(SddmmKernelTilingData, cType),
              "tiling field cType mismatch");
static_assert(offsetof(SddmmTilingData, idxBase) == offsetof(SddmmKernelTilingData, idxBase),
              "tiling field idxBase mismatch");
static_assert(offsetof(SddmmTilingData, rowHints) == offsetof(SddmmKernelTilingData, rowHints),
              "tiling field rowHints mismatch");
static_assert(offsetof(SddmmTilingData, patternFlags) == offsetof(SddmmKernelTilingData, patternFlags),
              "tiling field patternFlags mismatch");
static_assert(offsetof(SddmmTilingData, patternFlagWords) == offsetof(SddmmKernelTilingData, patternFlagWords),
              "tiling field patternFlagWords mismatch");

inline SddmmKernelTilingData SddmmSerializeTiling(const SddmmTilingData& host)
{
    SddmmKernelTilingData kernel{};
    kernel.nnz = host.nnz;
    kernel.rowCount = host.rowCount;
    kernel.matARows = host.matARows;
    kernel.matACols = host.matACols;
    kernel.matBRows = host.matBRows;
    kernel.matBCols = host.matBCols;
    kernel.blockDim = host.blockDim;
    kernel.rowsPerCore = host.rowsPerCore;
    kernel.remainderRows = host.remainderRows;
    kernel.kBlockLen = host.kBlockLen;
    kernel.nnzPartition = host.nnzPartition;
    kernel.nnzPerCore = host.nnzPerCore;
    kernel.remainderNnz = host.remainderNnz;
    kernel.alphaHost = host.alphaHost;
    kernel.betaHost = host.betaHost;
    kernel.alphaImagHost = host.alphaImagHost;
    kernel.betaImagHost = host.betaImagHost;
    kernel.opX = host.opX;
    kernel.opY = host.opY;
    kernel.ldx = host.ldx;
    kernel.ldy = host.ldy;
    kernel.orderX = host.orderX;
    kernel.orderY = host.orderY;
    kernel.xType = host.xType;
    kernel.cType = host.cType;
    kernel.idxBase = host.idxBase;
    kernel.rowHints = host.rowHints;
    kernel.patternFlags = host.patternFlags;
    kernel.patternFlagWords = host.patternFlagWords;
    return kernel;
}

static_assert(!std::is_same<SddmmBsrHostTiling, SddmmBsrKernelTiling>::value,
              "Host and Kernel tiling types must differ");
static_assert(sizeof(SddmmBsrHostTiling) == sizeof(SddmmBsrKernelTiling), "BSR tiling size mismatch");
static_assert(alignof(SddmmBsrHostTiling) == alignof(SddmmBsrKernelTiling), "BSR tiling alignment mismatch");
static_assert(std::is_trivially_copyable<SddmmBsrHostTiling>::value &&
                  std::is_trivially_copyable<SddmmBsrKernelTiling>::value,
              "BSR tiling must be byte-copyable");
static_assert(offsetof(SddmmBsrHostTiling, m) == offsetof(SddmmBsrKernelTiling, m), "BSR tiling field m mismatch");
static_assert(offsetof(SddmmBsrHostTiling, n) == offsetof(SddmmBsrKernelTiling, n), "BSR tiling field n mismatch");
static_assert(offsetof(SddmmBsrHostTiling, k) == offsetof(SddmmBsrKernelTiling, k), "BSR tiling field k mismatch");
static_assert(offsetof(SddmmBsrHostTiling, bd) == offsetof(SddmmBsrKernelTiling, bd), "BSR tiling field bd mismatch");
static_assert(offsetof(SddmmBsrHostTiling, blocks) == offsetof(SddmmBsrKernelTiling, blocks),
              "BSR tiling field blocks mismatch");
static_assert(offsetof(SddmmBsrHostTiling, first) == offsetof(SddmmBsrKernelTiling, first),
              "BSR tiling field first mismatch");
static_assert(offsetof(SddmmBsrHostTiling, count) == offsetof(SddmmBsrKernelTiling, count),
              "BSR tiling field count mismatch");
static_assert(offsetof(SddmmBsrHostTiling, yPitch) == offsetof(SddmmBsrKernelTiling, yPitch),
              "BSR tiling field yPitch mismatch");
static_assert(offsetof(SddmmBsrHostTiling, group) == offsetof(SddmmBsrKernelTiling, group),
              "BSR tiling field group mismatch");
static_assert(offsetof(SddmmBsrHostTiling, base) == offsetof(SddmmBsrKernelTiling, base),
              "BSR tiling field base mismatch");
static_assert(offsetof(SddmmBsrHostTiling, colMajor) == offsetof(SddmmBsrKernelTiling, colMajor),
              "BSR tiling field colMajor mismatch");
static_assert(offsetof(SddmmBsrHostTiling, ar) == offsetof(SddmmBsrKernelTiling, ar), "BSR tiling field ar mismatch");
static_assert(offsetof(SddmmBsrHostTiling, ai) == offsetof(SddmmBsrKernelTiling, ai), "BSR tiling field ai mismatch");
static_assert(offsetof(SddmmBsrHostTiling, br) == offsetof(SddmmBsrKernelTiling, br), "BSR tiling field br mismatch");
static_assert(offsetof(SddmmBsrHostTiling, bi) == offsetof(SddmmBsrKernelTiling, bi), "BSR tiling field bi mismatch");

inline SddmmBsrKernelTiling SddmmSerializeBsrTiling(const SddmmBsrHostTiling& host)
{
    SddmmBsrKernelTiling kernel{};
    kernel.m = host.m;
    kernel.n = host.n;
    kernel.k = host.k;
    kernel.bd = host.bd;
    kernel.blocks = host.blocks;
    kernel.first = host.first;
    kernel.count = host.count;
    kernel.yPitch = host.yPitch;
    kernel.group = host.group;
    kernel.base = host.base;
    kernel.colMajor = host.colMajor;
    kernel.ar = host.ar;
    kernel.ai = host.ai;
    kernel.br = host.br;
    kernel.bi = host.bi;
    return kernel;
}

#endif
