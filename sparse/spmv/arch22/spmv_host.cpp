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

#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <vector>
#include <cmath>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "cann_ops_sparse.h"
#include "spmv.h"
#include "spmv_cube_tiling.h"
#include "spmv_tiling_data.h"

// -------------------------------------------------------------------
// 前向声明：AscendC 内核启动分发器（定义在 kernels/spmv_kernel.cpp）
// -------------------------------------------------------------------
extern "C" void spmv_kernel_do(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal,
    GM_ADDR xVec, GM_ADDR yVec, SpmvTilingData tiling,
    const void *alpha, const void *beta,
    int32_t cType, int32_t vType, int32_t oType,
    bool trans, int32_t algType, uint32_t numBlocks,
    GM_ADDR cubeWorkspace, tcuscan::SpMVCubeTiling cubeTiling,
    uint32_t cubeBlocks, void *stream);

// -------------------------------------------------------------------
// 内部辅助函数
// -------------------------------------------------------------------
namespace {

    template <typename T>
    aclsparseStatus_t ReadSpmvScalar(const void *scalar,
                                     aclsparsePointerMode_t pointerMode, T &value,
                                     const char *scalarName)
    {
        if (pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE)
        {
            aclError aclRet = aclrtMemcpy(&value, sizeof(T), scalar, sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
            if (aclRet != ACL_SUCCESS)
            {
                LOG_PRINT("[ERROR] aclsparseSpMV: copying device %s failed, ret=%d\n", scalarName, aclRet);
                return ACL_SPARSE_STATUS_EXECUTION_FAILED;
            }
            return ACL_SPARSE_STATUS_SUCCESS;
        }
        value = *static_cast<const T *>(scalar);
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    bool IsSupportedSpmvDtypeCombo(aclDataType computeType,
                                   aclDataType valType,
                                   aclDataType outType)
    {
        if (computeType == ACL_INT32) {
            return valType == ACL_INT32 && outType == ACL_INT32;
        }
        if (computeType != ACL_FLOAT) {
            return false;
        }
        if (valType == ACL_FLOAT) {
            return outType == ACL_FLOAT;
        }
        if (valType == ACL_FLOAT16) {
            return outType == ACL_FLOAT16 || outType == ACL_FLOAT;
        }
        if (valType == ACL_BF16) {
            return outType == ACL_BF16 || outType == ACL_FLOAT;
        }
        return valType == ACL_INT32 && outType == ACL_FLOAT;
    }

    /**
     * @brief 获取 UB 容量下允许的最大单行长（非零元个数）
     */
    template <typename T>
    uint32_t ComputeMaxRowLength() {
        constexpr uint64_t kYLocalBytes = 32;
        constexpr uint64_t kSystemReserved = 4 * 1024;
        constexpr uint64_t kBufferCount = 5;
        constexpr uint64_t kAlignBytes = 32;
        constexpr uint64_t kAlignSlack = kBufferCount * kAlignBytes;

        auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        CHECK_RET(platform != nullptr,
                  LOG_PRINT("[ERROR] ComputeMaxRowLength: platform instance is nullptr\n");
                  return 0);
        uint64_t ubSize = 0;
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        CHECK_RET(ubSize > 0,
                  LOG_PRINT("[ERROR] GetCoreMemSize(UB) failed\n");
                  return 0);

        constexpr uint64_t perElem =
            sizeof(int32_t)  // colIdx
            + sizeof(T)      // vals
            + sizeof(T)      // xLocal
            + sizeof(float)  // workReduce
            + sizeof(float); // floatTmp

        constexpr uint64_t kFixed = kYLocalBytes + kSystemReserved + kAlignSlack;

        if (ubSize <= kFixed)
            return 0;

        uint64_t maxLen = (ubSize - kFixed) / perElem;

        constexpr uint64_t alignElems =
            kAlignBytes / sizeof(T) > 8 ? kAlignBytes / sizeof(T) : 8;
        maxLen = (maxLen / alignElems) * alignElems;

        return static_cast<uint32_t>(maxLen);
    }

    /**
     * @brief 从 device 侧 rowPtr 计算最大行长
     */
    uint32_t ComputeMaxRowLengthFromDevice(void *csrRowPtrDevice, uint64_t rows, aclrtStream stream) {
        if (rows == 0)
            return 0;

        std::vector<int32_t> rowPtrHost(rows + 1);
        aclError ret = aclrtMemcpy(rowPtrHost.data(),
                                   (rows + 1) * sizeof(int32_t),
                                   csrRowPtrDevice,
                                   (rows + 1) * sizeof(int32_t),
                                   ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] ComputeMaxRowLengthFromDevice: aclrtMemcpy D2H failed, ret=%d\n", ret);
                  return 0);

        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] ComputeMaxRowLengthFromDevice: sync failed, ret=%d\n", ret);
                  return 0);

        uint32_t maxLen = 0;
        for (uint64_t i = 0; i < rows; i++) {
            uint32_t rowLen = rowPtrHost[i + 1] - rowPtrHost[i];
            if (rowLen > maxLen)
                maxLen = rowLen;
        }
        return maxLen;
    }

    /// aclsparseSpMV / aclsparseSpMVGetBufferSize 共同解析出的参数。
    struct SpmvArgs {
        aclsparseContext *handle;
        aclsparseSpMatDescr *mat;
        aclsparseDnVecDescr *x;
        aclsparseDnVecDescr *y;
        aclDataType valType;
        aclDataType outType;
        /// 输出向量 y 的有效长度：非转置为 rows，转置为 cols。
        uint64_t yLength;
        /// 是否走 cube 版实现（算法与数据类型均满足要求）
        bool useCube;
    };

    /**
     * @brief aclsparseSpMV 与 aclsparseSpMVGetBufferSize 的公共参数校验。
     *
     * 两个入口对句柄/描述符的要求完全一致；集中在此可保证 GetBufferSize 报出的
     * workspace 大小与 aclsparseSpMV 实际选择的实现路径一致。
     */
    aclsparseStatus_t ValidateSpmvArgs(aclsparseHandle_t handle,
                                       aclsparseOperation_t opA,
                                       aclsparseConstSpMatDescr_t matA,
                                       aclsparseConstDnVecDescr_t vecX,
                                       aclsparseDnVecDescr_t vecY,
                                       aclDataType computeType,
                                       aclsparseSpMVAlg_t alg,
                                       SpmvArgs *args) {
        CHECK_RET(handle != nullptr,
                  LOG_PRINT("[ERROR] SpMV: handle is nullptr\n");
                  return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
        CHECK_RET(matA != nullptr,
                  LOG_PRINT("[ERROR] SpMV: matA is nullptr\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(vecX != nullptr,
                  LOG_PRINT("[ERROR] SpMV: vecX is nullptr\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(vecY != nullptr,
                  LOG_PRINT("[ERROR] SpMV: vecY is nullptr\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(alg == ACL_SPARSE_SPMV_ALG_DEFAULT || alg == ACL_SPARSE_SPMV_CSR_ALG_CUBE,
                  LOG_PRINT("[ERROR] SpMV: only ACL_SPARSE_SPMV_ALG_DEFAULT and "
                            "ACL_SPARSE_SPMV_CSR_ALG_CUBE are supported currently, got alg=%d\n", alg);
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);

        CHECK_RET(computeType == ACL_FLOAT || computeType == ACL_INT32,
                  LOG_PRINT("[ERROR] SpMV: unsupported computeType %d\n", computeType);
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);

        // ==================== 解包描述符 ====================
        auto *matInner = ToMatInner(matA);
        auto *xInner = ToVecInner(vecX);
        auto *yInner = ToVecInner(vecY);

        CHECK_RET(matInner->format == ACL_SPARSE_FORMAT_CSR,
                  LOG_PRINT("[ERROR] SpMV: unsupported matrix format %d\n", matInner->format);
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);

        CHECK_RET(matInner->baseType == ACL_SPARSE_INDEX_BASE_ZERO,
                  LOG_PRINT("[ERROR] SpMV: 1-based index base not supported\n");
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);
        {
            aclsparseStatus_t idxSt =
                AclsparseValidateSupportedCsrIndexTypes(matInner->ptrType, matInner->IdxType);
            CHECK_RET(idxSt == ACL_SPARSE_STATUS_SUCCESS,
                      LOG_PRINT("[ERROR] SpMV: unsupported index type ptr=%d idx=%d "
                                "(only ACL_SPARSE_INDEX_32I)\n",
                                matInner->ptrType, matInner->IdxType);
                      return idxSt);
        }

        CHECK_RET(opA == ACL_SPARSE_OP_NON_TRANSPOSE || opA == ACL_SPARSE_OP_TRANSPOSE,
                  LOG_PRINT("[ERROR] SpMV: opA conjugate not supported yet\n");
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);

        // 分发器仅包含七种 Kernel 实例。在访问设备前校验完整的
        //（计算类型、输入值类型、输出类型）组合，避免不支持的组合误入其他类型的 Kernel。
        const aclDataType valType = matInner->valueType;
        const aclDataType outType = yInner->valueType;
        CHECK_RET(xInner->valueType == valType,
                  LOG_PRINT("[ERROR] SpMV: matrix value type %d and vecX type %d must match\n",
                            valType, xInner->valueType);
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);
        CHECK_RET(IsSupportedSpmvDtypeCombo(computeType, valType, outType),
                  LOG_PRINT("[ERROR] SpMV: unsupported dtype combination "
                            "computeType=%d valType=%d outType=%d\n",
                            computeType, valType, outType);
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);

        // cube 版实现只覆盖非转置 SpMV，且限于 fp32 计算 / fp32 输出，
        // 矩阵与 x 的值类型为 fp32 或 fp16（后者由 cube 以 fp32 累加）。
        if (alg == ACL_SPARSE_SPMV_CSR_ALG_CUBE) {
            CHECK_RET(opA == ACL_SPARSE_OP_NON_TRANSPOSE,
                      LOG_PRINT("[ERROR] SpMV: ACL_SPARSE_SPMV_CSR_ALG_CUBE requires "
                                "ACL_SPARSE_OP_NON_TRANSPOSE\n");
                      return ACL_SPARSE_STATUS_NOT_SUPPORTED);
            CHECK_RET(computeType == ACL_FLOAT && outType == ACL_FLOAT &&
                          (valType == ACL_FLOAT || valType == ACL_FLOAT16),
                      LOG_PRINT("[ERROR] SpMV: ACL_SPARSE_SPMV_CSR_ALG_CUBE requires "
                                "float32 computeType/outType and float32 or float16 valType, "
                                "got %d/%d/%d\n",
                                computeType, valType, outType);
                      return ACL_SPARSE_STATUS_NOT_SUPPORTED);
        }

        // ==================== 校验设备指针与向量长度 ====================
        CHECK_RET(matInner->ptrs != nullptr && matInner->idxs != nullptr &&
                      matInner->values != nullptr,
                  LOG_PRINT("[ERROR] SpMV: matrix device pointers are null\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(xInner->values != nullptr && yInner->values != nullptr,
                  LOG_PRINT("[ERROR] SpMV: vector device pointers are null\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);

        const bool trans = (opA == ACL_SPARSE_OP_TRANSPOSE);
        const uint64_t xRequired = trans ? matInner->rows : matInner->cols;
        const uint64_t yRequired = trans ? matInner->cols : matInner->rows;
        CHECK_RET(xInner->nums >= xRequired,
                  LOG_PRINT("[ERROR] SpMV: vecX size %lu < required %lu\n",
                            xInner->nums, xRequired);
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(yInner->nums >= yRequired,
                  LOG_PRINT("[ERROR] SpMV: vecY size %lu < required %lu\n",
                            yInner->nums, yRequired);
                  return ACL_SPARSE_STATUS_INVALID_VALUE);

        args->handle = ToInternalHandle(handle);
        args->mat = matInner;
        args->x = xInner;
        args->y = yInner;
        args->valType = valType;
        args->outType = outType;
        args->yLength = yRequired;
        args->useCube = (alg == ACL_SPARSE_SPMV_CSR_ALG_CUBE);
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    /**
     * @brief 计算 cube 版 SpMV 所需的 workspace 布局。
     *
     * @return 该矩阵不走 cube 路径（空矩阵或 nnz 过大，均回退到向量版 kernel）时
     *         返回 false，此时不需要 workspace。
     */
    bool GetCubeWorkspaceLayout(const aclsparseSpMatDescr &mat, aclDataType valType,
                                spmv_cube::CubeWorkspaceLayout *layout) {
        if (mat.rows == 0 || mat.cols == 0 || mat.nnz == 0 ||
            mat.nnz > 0xFFFFFFFFull) {
            return false;
        }
        return spmv_cube::ComputeCubeWorkspaceLayout(
            static_cast<uint32_t>(mat.nnz), valType == ACL_FLOAT16, layout);
    }

    /**
     * @brief 选择 cube 版 SpMV 使用的 workspace。
     *
     * 优先使用调用方通过 externalBuffer 传入的显存（大小由
     * aclsparseSpMVGetBufferSize 给出，调用方自行保证）；externalBuffer 为空时
     * 回退到句柄上的 workspace：先用 aclsparseEnsureDefaultWorkspace 保证容量，
     * 再用 aclsparseGetEffectiveWorkspace 取出实际使用的缓冲区。
     */
    aclsparseStatus_t SelectCubeWorkspace(aclsparseContext *h, void *externalBuffer,
                                          size_t requiredBytes, void **outWorkspace) {
        if (externalBuffer != nullptr) {
            *outWorkspace = externalBuffer;
            return ACL_SPARSE_STATUS_SUCCESS;
        }

        const aclsparseStatus_t status = aclsparseEnsureDefaultWorkspace(h, requiredBytes);
        CHECK_RET(status == ACL_SPARSE_STATUS_SUCCESS,
                  LOG_PRINT("[ERROR] aclsparseSpMV: ACL_SPARSE_SPMV_CSR_ALG_CUBE needs %zu bytes of "
                            "workspace, but the handle workspace (%zu bytes) cannot provide it, "
                            "please pass an externalBuffer sized by aclsparseSpMVGetBufferSize\n",
                            requiredBytes, aclsparseGetEffectiveWorkspaceSize(h));
                  return status);

        auto *workspace = static_cast<uint8_t *>(aclsparseGetEffectiveWorkspace(h));
        CHECK_RET(workspace != nullptr,
                  LOG_PRINT("[ERROR] aclsparseSpMV: handle has no workspace\n");
                  return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);

        *outWorkspace = workspace;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

} // namespace

// -------------------------------------------------------------------
// Public API Implementation
// -------------------------------------------------------------------

extern "C" {
    aclsparseStatus_t aclsparseSpMVGetBufferSize(aclsparseHandle_t handle,
                                                 aclsparseOperation_t opA,
                                                 const void *alpha,
                                                 aclsparseConstSpMatDescr_t matA,
                                                 aclsparseConstDnVecDescr_t vecX,
                                                 const void *beta,
                                                 aclsparseDnVecDescr_t vecY,
                                                 aclDataType computeType,
                                                 aclsparseSpMVAlg_t alg,
                                                 size_t *bufferSize) {
        // workspace 大小只取决于 (nnz, 值类型, 算法)，与 alpha / beta 取值无关。
        (void)alpha;
        (void)beta;

        SpmvArgs args{};
        aclsparseStatus_t st =
            ValidateSpmvArgs(handle, opA, matA, vecX, vecY, computeType, alg, &args);
        CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);


        CHECK_RET(bufferSize != nullptr,
                  LOG_PRINT("[ERROR] aclsparseSpMVGetBufferSize: bufferSize is nullptr\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        *bufferSize = 0;

        // 向量版按行并行，所有中间量都在 UB 内，不需要额外的 device 显存。
        if (!args.useCube) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }

        // cube 版需要 gather 乘积区 + 块内推测扫描结果区；退化输入会回退到
        // 向量版 kernel，同样不需要 workspace。
        spmv_cube::CubeWorkspaceLayout layout{};
        if (GetCubeWorkspaceLayout(*args.mat, args.valType, &layout)) {
            *bufferSize = layout.totalBytes;
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    aclsparseStatus_t aclsparseSpMV(aclsparseHandle_t handle,
                                    aclsparseOperation_t opA,
                                    const void *alpha,
                                    aclsparseConstSpMatDescr_t matA,
                                    aclsparseConstDnVecDescr_t vecX,
                                    const void *beta,
                                    aclsparseDnVecDescr_t vecY,
                                    aclDataType computeType,
                                    aclsparseSpMVAlg_t alg,
                                    void *externalBuffer) {
        // ==================== 参数校验 ====================
        SpmvArgs args{};
        aclsparseStatus_t st =
            ValidateSpmvArgs(handle, opA, matA, vecX, vecY, computeType, alg, &args);
        CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);

        // 输出向量为空（非转置时 rows == 0，转置时 cols == 0）时没有任何元素需要写回，
        // y = alpha * op(A) * x + beta * y 退化为 no-op，直接返回 SUCCESS。
        // 注意不能用 nnz == 0 作为条件：此时 y = beta * y 仍需真正执行。
        if (args.yLength == 0) {
            return ACL_SPARSE_STATUS_SUCCESS;
        }

        auto *h = args.handle;
        auto *matInner = args.mat;
        auto *xInner = args.x;
        auto *yInner = args.y;
        const aclDataType valType = args.valType;
        const aclDataType outType = args.outType;

        // ==================== Alpha / Beta ====================
        // alpha/beta 类型必须与 computeType 一致（任务书要求），调用方负责保证
        CHECK_RET(alpha != nullptr,
                  LOG_PRINT("[ERROR] aclsparseSpMV: alpha is nullptr\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(beta != nullptr,
                  LOG_PRINT("[ERROR] aclsparseSpMV: beta is nullptr\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);

        // Device scalars can be written by work already queued on the handle
        // stream. aclrtMemcpy(D2H) does not wait for that work, so establish
        // the stream dependency before reading alpha/beta on the host.
        aclrtStream stream = h->stream;
        CHECK_RET(stream != nullptr,
                  LOG_PRINT("[ERROR] aclsparseSpMV: stream is nullptr, please call aclsparseSetStream first\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        if (h->pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
            aclError syncRet = aclrtSynchronizeStream(stream);
            CHECK_RET(syncRet == ACL_SUCCESS,
                      LOG_PRINT("[ERROR] aclsparseSpMV: synchronizing handle stream before scalar read failed, ret=%d\n",
                                syncRet);
                      return ACL_SPARSE_STATUS_EXECUTION_FAILED);
        }

        float alphaFloat = 1.0f;
        float betaFloat = 0.0f;
        int32_t alphaInt = 1;
        int32_t betaInt = 0;
        aclsparseStatus_t scalarStatus = ACL_SPARSE_STATUS_SUCCESS;
        if (computeType == ACL_FLOAT) {
            scalarStatus = ReadSpmvScalar(alpha, h->pointerMode, alphaFloat, "alpha");
            CHECK_RET(scalarStatus == ACL_SPARSE_STATUS_SUCCESS, return scalarStatus);
            scalarStatus = ReadSpmvScalar(beta, h->pointerMode, betaFloat, "beta");
        } else {
            scalarStatus = ReadSpmvScalar(alpha, h->pointerMode, alphaInt, "alpha");
            CHECK_RET(scalarStatus == ACL_SPARSE_STATUS_SUCCESS, return scalarStatus);
            scalarStatus = ReadSpmvScalar(beta, h->pointerMode, betaInt, "beta");
        }
        CHECK_RET(scalarStatus == ACL_SPARSE_STATUS_SUCCESS, return scalarStatus);

        // ==================== 平台参数 ====================
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        CHECK_RET(ascendcPlatform != nullptr,
                  LOG_PRINT("[ERROR] aclsparseSpMV: platform instance is nullptr\n");
                  return ACL_SPARSE_STATUS_INTERNAL_ERROR);
        uint32_t blockDim = ascendcPlatform->GetCoreNumAiv();
        CHECK_RET(blockDim > 0,
                  LOG_PRINT("[ERROR] aclsparseSpMV: GetCoreNumAiv returned 0\n");
                  return ACL_SPARSE_STATUS_INTERNAL_ERROR);

        // ==================== 提取矩阵参数 ====================
        uint64_t rows = matInner->rows;
        uint64_t cols = matInner->cols;
        uint64_t nnz = matInner->nnz;

        void *csrRowPtrDevice = matInner->ptrs;
        void *csrColIndDevice = matInner->idxs;
        void *csrValDevice = matInner->values;

        // ==================== cube 版 tiling / workspace ====================
        // tiling 全部在 host 侧算好，随 launch 按值传给 kernel；workspace 优先取
        // 调用方的 externalBuffer（大小见 aclsparseSpMVGetBufferSize），未提供时
        // 回退到句柄上的 workspace（aclsparseGetEffectiveWorkspace）。因此整个
        // 派发过程只往 stream 上排任务，不做任何同步，保持 aclsparseSpMV 异步语义。
        int32_t algType = SPMV_ALG_VECTOR;
        tcuscan::SpMVCubeTiling cubeTiling{};
        uint32_t cubeBlocks = 0;
        void *cubeWorkspace = nullptr;

        if (args.useCube) {
            spmv_cube::CubeWorkspaceLayout layout{};
            if (GetCubeWorkspaceLayout(*matInner, valType, &layout) &&
                spmv_cube::MakeCubeTiling(static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
                                          static_cast<uint32_t>(nnz), alphaFloat, betaFloat,
                                          blockDim, &cubeTiling, &cubeBlocks)) {
                void *workspace = nullptr;
                st = SelectCubeWorkspace(h, externalBuffer, layout.totalBytes, &workspace);
                CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);

                // gather 只写前 nnz 个乘积，尾部补零以免 cube 扫描到未初始化数据。
                // 异步 memset 排在同一条 stream 上，天然早于随后的 kernel。
                const size_t usedBytes = static_cast<size_t>(nnz) * layout.valElemSize;
                const size_t tailBytes = layout.productsBytes - usedBytes;
                if (tailBytes > 0) {
                    aclError ret = aclrtMemsetAsync(
                        static_cast<uint8_t *>(workspace) + usedBytes,
                        tailBytes, 0, tailBytes, stream);
                    CHECK_RET(ret == ACL_SUCCESS,
                              LOG_PRINT("[ERROR] aclsparseSpMV: workspace memset failed, ret=%d\n", ret);
                              return ACL_SPARSE_STATUS_EXECUTION_FAILED);
                }

                cubeWorkspace = workspace;
                algType = SPMV_ALG_CUBE;
            } else {
                LOG_PRINT("[WARN] aclsparseSpMV: cube algorithm unavailable, "
                          "falling back to the vector kernel\n");
            }
        }

        // ==================== UB 容量检查（仅 vector 版）====================
        // vector 版按行切分，整行必须放进 UB，因此需要读回 rowPtr 求最大行长。
        // cube 版（分段求和）按 nnz 切分，行长不受 UB 限制，故跳过该检查以及
        // 随之而来的 rowPtr D2H 拷贝与 stream 同步。
        if (algType != SPMV_ALG_CUBE) {
            uint32_t maxRowLength = ComputeMaxRowLengthFromDevice(csrRowPtrDevice, rows, stream);
            uint32_t maxTileLength =
                (computeType == ACL_FLOAT) ? ComputeMaxRowLength<float>() : ComputeMaxRowLength<int32_t>();

            CHECK_RET(maxTileLength > 0,
                      LOG_PRINT("[ERROR] aclsparseSpMV: failed to compute max tile length\n");
                      return ACL_SPARSE_STATUS_INTERNAL_ERROR);
            CHECK_RET(maxRowLength <= maxTileLength,
                      LOG_PRINT("[ERROR] aclsparseSpMV: max row length %u exceeds UB capacity %u\n",
                                maxRowLength, maxTileLength);
                      return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);
        }

        // ==================== Tiling 数据准备 ====================
        SpmvTilingData tilingHost = {
            static_cast<uint32_t>(rows),
            static_cast<uint32_t>(cols),
            static_cast<uint32_t>(nnz)};

        // ==================== 启动内核 ====================
        bool trans = (opA == ACL_SPARSE_OP_TRANSPOSE);

        int32_t cType, vType, oType;
        SpmvTypesFromAcl(computeType, valType, outType, &cType, &vType, &oType);

        const void *alphaPtr = (computeType == ACL_FLOAT) ?
                                   static_cast<const void *>(&alphaFloat) :
                                   static_cast<const void *>(&alphaInt);
        const void *betaPtr = (computeType == ACL_FLOAT) ?
                                  static_cast<const void *>(&betaFloat) :
                                  static_cast<const void *>(&betaInt);

        spmv_kernel_do(
            static_cast<GM_ADDR>(csrRowPtrDevice), static_cast<GM_ADDR>(csrColIndDevice),
            static_cast<GM_ADDR>(csrValDevice), static_cast<GM_ADDR>(xInner->values),
            static_cast<GM_ADDR>(yInner->values), tilingHost,
            alphaPtr, betaPtr,
            cType, vType, oType, trans, algType, blockDim,
            static_cast<GM_ADDR>(cubeWorkspace), cubeTiling, cubeBlocks, stream);

        return ACL_SPARSE_STATUS_SUCCESS;
    }

} // extern "C"
