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

#ifndef TEST_SDDMM_SDDMM_ARCH35_SDDMM_TEST_SUPPORT_H_
#define TEST_SDDMM_SDDMM_ARCH35_SDDMM_TEST_SUPPORT_H_

// Support declarations for the cases registered in the existing sddmm_test.cpp.
#include "test_common.h"
#include "sddmm_golden.h"
#include "sddmm_npu_wrapper.h"
#include "sddmm_param.h"
#include "aclsparse_descr_internal.h"
#include "sddmm_complex_bsr_cube_common.h"
#include "sddmm_csr_cube_common.h"
#include "sddmm_tiling_data.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sparse_test {
namespace sddmm_extended {

void CheckComplexBsrL0CapacityBoundaries();
void CheckComplexBsrGroupedL0Requirement();
void CheckComplexBsrL0RejectsUnsupportedTiles();

void CheckWarmWorkspaceOwners(aclrtStream stream, bool cube);
#ifdef SDDMM_TEST_WITH_SPMM
void CheckWarmWorkspaceSpmm(aclrtStream stream, bool cube, bool alias);
#endif

void CheckCsrComplexBetaOneInfinity(aclrtStream stream, bool deviceScalars, int64_t reduction);
void CheckBsrComplexBetaOneInfinity(aclrtStream stream, int64_t blockSize,
    int64_t reduction, aclsparseOrder_t order, bool deviceScalars);

class BsrMatGuard {
public:
    BsrMatGuard() = default;

    ~BsrMatGuard()
    {
        if (descr_ != nullptr) {
            (void)aclsparseDestroySpMat(descr_);
        }
    }

    aclsparseSpMatDescr_t *put() { return &descr_; }
    aclsparseSpMatDescr_t get() const { return descr_; }

    BsrMatGuard(const BsrMatGuard &) = delete;
    BsrMatGuard &operator=(const BsrMatGuard &) = delete;

private:
    aclsparseSpMatDescr_t descr_ = nullptr;
};

struct SddmmStageResult {
    size_t workspaceSize = 0;
    bool workspaceSizeAligned = true;
    bool workspacePointerAligned = true;
    aclsparseStatus_t bufferSizeRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t preprocessRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t executeRet = ACL_SPARSE_STATUS_SUCCESS;
    aclError synchronizeRet = ACL_SUCCESS;
};

struct RealRunResult {
    SddmmStageResult stages;
    std::vector<float> actual;
    std::vector<float> golden;
    bool valuesPointer64Aligned = false;
    uint32_t patternFlags = SDDMM_PATTERN_FLAG_NONE;
    int32_t preparedLogicalFirstColumn = -1;
};

struct ComplexRunResult {
    SddmmStageResult stages;
    std::vector<float> actualParts;
    std::vector<float> goldenParts;
};

void CheckSetupStatus(aclsparseStatus_t status, const char *api);

size_t RealTypeBytes(aclDataType dataType);

float QuantizeReal(float value, aclDataType dataType);

std::vector<float> QuantizeRealVector(
    const std::vector<float> &values, aclDataType dataType);

std::vector<uint8_t> PackRealValues(
    const std::vector<float> &values, aclDataType dataType);

std::vector<float> UnpackRealValues(
    const std::vector<uint8_t> &bytes, size_t count, aclDataType dataType);

std::vector<float> MakeRealValues(
    int batchCount, int64_t elementsPerBatch, float phase);

class SddmmGuardedBuffer {
public:
    static SddmmGuardedBuffer copyFrom(
        const void *source, size_t bytes, size_t extraOffset = 0);

    void *get() const;

    void copyToHost(void *destination, size_t bytes) const;

    std::vector<uint8_t> snapshot() const;

    bool unchanged() const;

    bool guardsIntact() const;

    void restore() const;

private:
    DeviceBuffer storage_;
    std::vector<uint8_t> initial_;
    size_t bytes_ = 0;
    size_t prefix_ = 64U;
};

void CheckGuardedBuffers(
    const SddmmGuardedBuffer &values, const SddmmGuardedBuffer &workspace,
    const std::vector<const SddmmGuardedBuffer *> &inputs);

void SetHostTestStream(HandleManager &handle, aclrtStream stream);

bool QueryTestWorkspace(SddmmStageResult &result, HandleManager &handle,
    const void *alpha, const void *beta, DnMatManager &matX, DnMatManager &matY,
    aclsparseSpMatDescr_t matC, aclDataType computeType);

SddmmStageResult RunThreeStages(
    HandleManager &handle, const void *alpha, const void *beta,
    DnMatManager &matX, DnMatManager &matY, aclsparseSpMatDescr_t matC,
    aclDataType computeType, aclrtStream stream, DeviceBuffer &workspace);

struct BsrTestResources {
    SddmmGuardedBuffer dOffsets;
    SddmmGuardedBuffer dColumns;
    SddmmGuardedBuffer dX;
    SddmmGuardedBuffer dY;
    SddmmGuardedBuffer dValues;
    DnMatManager matX;
    DnMatManager matY;
    BsrMatGuard matC;
    HandleManager handle;
    DeviceBuffer dAlpha;
    DeviceBuffer dBeta;
    DeviceBuffer workspace;
    const void *alpha = nullptr;
    const void *beta = nullptr;

    template <typename Scalar>
    void SetScalars(aclrtStream stream, bool deviceScalars,
                    const Scalar &hostAlpha, const Scalar &hostBeta);
};

struct BsrRealCase {
    std::string name;
    int64_t blockRows = 1;
    int64_t blockCols = 2;
    int64_t blockSize = 2;
    int64_t k = 3;
    std::vector<int32_t> zeroBasedOffsets = {0, 2};
    std::vector<int32_t> zeroBasedColumns = {0, 1};
    aclsparseIndexBase_t indexBase = ACL_SPARSE_INDEX_BASE_ZERO;
    aclsparseOrder_t blockOrder = ACL_SPARSE_ORDER_ROW;
    aclDataType inputType = ACL_FLOAT;
    aclDataType outputType = ACL_FLOAT;
    int outputBatchCount = 1;
    int xBatchCount = 1;
    int yBatchCount = 1;
    float alpha = 0.75F;
    float beta = -0.25F;
    bool deviceScalars = false;
    bool specialDenseValues = false;
    bool randomValues = false;
    bool guardedRepeat = false;
    bool subnormalDenseValues = false;
    std::vector<float> initialValuesOverride;
};

float RealDotGolden(const float *x, const float *y, int64_t k,
                           int64_t yStride, aclDataType inputType,
                           float alpha, float beta, float oldValue);

std::vector<float> BsrRealGolden(
    const BsrRealCase &testCase, const std::vector<float> &x,
    const std::vector<float> &y, const std::vector<float> &cInitial);

RealRunResult RunBsrRealCase(aclrtStream stream, const BsrRealCase &testCase);

bool VerifyRealRun(
    const RealRunResult &result, aclDataType outputType,
    const std::string &caseName);

std::vector<aclsparseComplex> MakeComplexValues(
    size_t count, float phase);

struct BsrComplexCase {
    std::string name;
    int64_t blockRows = 2;
    int64_t blockCols = 3;
    int64_t blockSize = 16;
    int64_t k = sddmm_arch35::kSddmmComplexBsrCubeReduction;
    std::vector<int32_t> zeroBasedOffsets = {0, 2, 3};
    std::vector<int32_t> zeroBasedColumns = {2, 0, 1};
    aclsparseIndexBase_t indexBase = ACL_SPARSE_INDEX_BASE_ZERO;
    aclsparseOrder_t blockOrder = ACL_SPARSE_ORDER_ROW;
    aclsparseComplex alpha{0.75F, -0.25F};
    aclsparseComplex beta{-0.5F, 0.125F};
    bool offsetDensePointersByOneElement = false;
    bool randomValues = false;
    bool guardedRepeat = false;
    bool specialDenseValues = false;
    bool columnMajorY = false;
    bool deviceScalars = false;
};

std::vector<float> BsrComplexGoldenParts(
    const BsrComplexCase &testCase,
    const std::vector<aclsparseComplex> &x,
    const std::vector<aclsparseComplex> &y,
    const std::vector<aclsparseComplex> &cInitial);

ComplexRunResult RunBsrComplexCase(aclrtStream stream, const BsrComplexCase &testCase);

ComplexRunResult RunCsrBaseOneComplex(aclrtStream stream);

bool VerifyComplexRun(
    const ComplexRunResult &result, const std::string &caseName);

RealRunResult RunCsrFp16ColumnMajorCase(
    aclrtStream stream, aclsparseIndexBase_t indexBase,
    int64_t m = 4, int64_t n = 96,
    int64_t rowStride = sddmm_arch35::kSddmmCsrCubeRowColumnStride);

struct CsrCubeRealCase {
    std::string name;
    aclDataType dataType = ACL_FLOAT16;
    aclsparseIndexBase_t indexBase = ACL_SPARSE_INDEX_BASE_ZERO;
    int32_t logicalFirstColumn = 64;
    bool perturbOneColumn = false;
    float alpha = 0.75F;
    float beta = -0.25F;
    bool fp32Output = false;
    int64_t m = 16;
    int64_t n = 512;
    bool specialDenseValues = false;
    int64_t rowStride = sddmm_arch35::kSddmmCsrCubeRowColumnStride;
};

struct CsrCubeRealRunResult {
    RealRunResult run;
    uint32_t patternFlags = SDDMM_PATTERN_FLAG_NONE;
    int32_t preparedLogicalFirstColumn = -1;
    size_t expectedPackedYBytes = 0;
    size_t expectedCubeScratchBytes = 0;
    size_t expectedStaticWorkspace = 0;
};

CsrCubeRealRunResult RunCsrCubeRealCase(aclrtStream stream, const CsrCubeRealCase &testCase);

struct PreparedInvalidationResult {
    SddmmStageResult stages;
    aclsparseStatus_t setterRet = ACL_SPARSE_STATUS_SUCCESS;
    bool preparedBefore = false;
    bool preparedAfterSetter = true;
    bool preparedAfterExecute = false;
    bool workspaceClearedAfterSetter = false;
    bool rebuiltForNewOffsets = false;
    uint64_t generationBefore = 0;
    uint64_t generationAfterSetter = 0;
    uint64_t epochBefore = 0;
    uint64_t epochAfter = 0;
    std::vector<float> actual;
    std::vector<float> golden;
};

PreparedInvalidationResult RunPreparedInvalidationCase(
    aclrtStream stream);

struct ActiveBufferIsolationResult {
    aclsparseStatus_t bufferSizeRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t distinctWorkspaceRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t aliasedWorkspaceRet = ACL_SPARSE_STATUS_SUCCESS;
    bool distinctOwnerPreserved = false;
    bool aliasedOwnerCleared = false;
    bool distinctSpmvOwnerPreserved = false;
    bool aliasedSpmvOwnerCleared = false;
    bool sddmmPrepared = false;
};

ActiveBufferIsolationResult RunActiveBufferIsolationCase(
    aclrtStream stream);

bool StagesSucceeded(const SddmmStageResult &stages);

void AddComplexCubeBlock16Cases(std::vector<BsrComplexCase> &cases);

void AddComplexCubeBlock32Cases(std::vector<BsrComplexCase> &cases);

void AddComplexCubeBlock64Cases(std::vector<BsrComplexCase> &cases);

BsrComplexCase MakeComplexColumnIdentityCase(int64_t reduction);

BsrComplexCase MakeComplexFourGroupCase(int variant);

std::vector<size_t> BuildDescendingComplexReference(
    const BsrComplexCase &testCase, BsrComplexCase &fallback);

struct SparsePackVariant {
    int32_t blocks;
    int32_t blockColumns;
    int32_t blockSize;
    int32_t reduction;
    bool sparsePack;
};

BsrComplexCase MakeSparsePackCase(const SparsePackVariant &variant, int ordinal);

struct DeviceIdentityCase {
    bool bsr;
    aclDataType input;
    aclDataType output;
    aclsparseOrder_t order;
    aclsparseIndexBase_t base;
};

void RunDeviceIdentityNoOpCase(aclrtStream stream, const DeviceIdentityCase &testCase);

void CheckCsrCubeNumerics(const CsrCubeRealRunResult &result,
    const CsrCubeRealCase &testCase, bool scratchDominates);

void SetGuardedBsrPattern(
    int kind, std::vector<int32_t> &offsets, std::vector<int32_t> &columns);

BsrRealCase MakeBlock8IdentityCase(int &ordinal, aclDataType inputType,
    aclDataType outputType, aclsparseOrder_t order, aclsparseIndexBase_t base, int pattern);

void CheckBlock4GenericBroadcast(aclrtStream stream,
    const std::pair<aclDataType, aclDataType> &type, int layout, int base,
    int scalars, int &ordinal);

void CheckBsr4TailCase(aclrtStream stream,
    const std::pair<aclDataType, aclDataType> &type, int32_t blockNnz,
    int ordinal, bool deviceScalars, int scalarMode = 1);

enum class CsrWorkspacePattern {
    REGULAR,
    ALIGNED_SHIFTED,
    PERTURBED,
    WEAK,
    UNALIGNED,
};

class CsrWorkspaceFixture {
public:
    static constexpr int64_t m = 128;
    static constexpr int64_t n = 512;
    static constexpr int64_t k = 128;
    // For this fixed shape, packed FP16 Y occupies 128 KiB while Cube needs
    // 160 KiB. A fallback query must remove the excess 32 KiB reservation.
    static constexpr size_t genericBytes = kSddmmWorkspaceBytes + 128U * 1024U;
    static constexpr size_t cubeBytes = kSddmmWorkspaceBytes + 160U * 1024U;

    float alpha = 0.75F;
    float beta = -0.25F;
    SddmmStageResult stages;
    SddmmGuardedBuffer workspace;

    CsrWorkspaceFixture(aclrtStream callStream, CsrWorkspacePattern pattern,
        int64_t rows = m, aclsparseIndexBase_t base = ACL_SPARSE_INDEX_BASE_ZERO,
        int32_t lastRowEntries = 64);

    bool Query();

    void AllocateWorkspace();

    aclsparseStatus_t Preprocess();

    aclsparseStatus_t Execute();

    bool Run(bool preprocess = true);

    void ChangePattern(CsrWorkspacePattern pattern, bool replacePointer = false);

    void PerturbRow(int64_t row);

    void ShiftRows(int64_t firstRow, int32_t columnShift);

    int32_t Column(size_t position) const;
    int32_t Offset(size_t position) const;
    size_t ColumnCount() const;

    void SetColumn(size_t position, int32_t value);

    void SetOffset(size_t position, int32_t value);

    bool Prepare();
    bool IsPrepared() const;
    uint32_t PatternFlags() const;

    void CheckRejectedPattern();

    void SetDeviceScalars(bool deviceMode);

    void ChangeYToColumnMajor();

    void CheckStorage() const;

private:
    aclrtStream stream;
    int64_t rowCount;
    aclsparseIndexBase_t indexBase;
    int32_t lastRowNnz;
    bool deviceScalars = false;
    std::vector<int32_t> offsets;
    std::vector<int32_t> columns;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> cInitial;
    SddmmGuardedBuffer dOffsets;
    SddmmGuardedBuffer dColumns;
    SddmmGuardedBuffer dX;
    SddmmGuardedBuffer dY;
    SddmmGuardedBuffer dValues;
    DeviceBuffer dAlpha;
    DeviceBuffer dBeta;
    DnMatManager matX;
    DnMatManager matY;
    SpMatManager matC;
    HandleManager handle;

    const void *Alpha() const;
    const void *Beta() const;

    void MakePattern(CsrWorkspacePattern pattern);

    static void UploadIndices(SddmmGuardedBuffer &device, const std::vector<int32_t> &host);

    void PrepareInputs();

    void CreateDescriptors();

    bool PrepareAndCheck();

    std::vector<float> Golden() const;
};

struct BsrWorkspaceFixture : BsrTestResources {
    SddmmStageResult stages;
    SddmmGuardedBuffer guardedWorkspace;
    aclrtStream stream;
    aclDataType computeType = ACL_FLOAT;

    explicit BsrWorkspaceFixture(aclrtStream callStream);

    template <typename Case, typename Value>
    void Initialize(const Case &testCase, aclDataType valueType,
        const std::vector<Value> &x, const std::vector<Value> &y,
        const std::vector<Value> &initial, size_t denseOffset = 0);

    bool Query();

    bool Run(bool preprocess = true);
};

struct SddmmInputs {
    SddmmCsr csrC;
    std::vector<double> Xf64;
    std::vector<double> Yf64;
    int64_t nnz = 0;
};

SddmmInputs GenerateSddmmInputs(const SddmmTestParam& p);

std::vector<double> ComputeGoldenExpect(const SddmmTestParam& p,
                                                const SddmmCsr& csrC,
                                                const std::vector<double>& Xf64,
                                                const std::vector<double>& Yf64,
                                                aclDataType dtype,
                                                aclsparseOperation_t opX,
                                                aclsparseOperation_t opY);

SddmmNpuResult RunNpuSddmm(const SddmmTestParam& p, aclrtStream stream,
                                   const SddmmCsr& csrC,
                                   const std::vector<double>& Xf64,
                                   const std::vector<double>& Yf64,
                                   aclDataType dtype, aclDataType computeType,
                                   aclsparseSDDMMAlg_t alg,
                                   aclsparseOperation_t opX, aclsparseOperation_t opY,
                                   aclsparseOrder_t orderX, aclsparseOrder_t orderY,
                                   int64_t nnz);

void VerifySddmmPrecision(const std::string& caseName,
                                  const SddmmNpuResult& npuResult,
                                  const std::vector<double>& expectFp64,
                                  aclDataType dtype, int64_t nnz);

void CheckDenseBatchStrideRequiresFullPaddedSpan(
    HandleManager *handle_, float &alpha_, float &beta_, DeviceBuffer &dBuffer_);

void CheckZeroOutputDimension(aclrtStream stream, bool zeroRows);

void CheckZeroReduction(aclrtStream stream, float beta);

class MutableSpMatGuard {
public:
    MutableSpMatGuard() = default;

    ~MutableSpMatGuard() {
        if (descr_ != nullptr) {
            (void)aclsparseDestroySpMat(descr_);
        }
    }

    aclsparseSpMatDescr_t* put() { return &descr_; }
    aclsparseSpMatDescr_t get() const { return descr_; }

    MutableSpMatGuard(const MutableSpMatGuard&) = delete;
    MutableSpMatGuard& operator=(const MutableSpMatGuard&) = delete;

private:
    aclsparseSpMatDescr_t descr_ = nullptr;
};

class ConstSpMatGuard {
public:
    ConstSpMatGuard() = default;

    ~ConstSpMatGuard() {
        if (descr_ != nullptr) {
            (void)aclsparseDestroySpMat(descr_);
        }
    }

    aclsparseConstSpMatDescr_t* put() { return &descr_; }
    aclsparseConstSpMatDescr_t get() const { return descr_; }

    ConstSpMatGuard(const ConstSpMatGuard&) = delete;
    ConstSpMatGuard& operator=(const ConstSpMatGuard&) = delete;

private:
    aclsparseConstSpMatDescr_t descr_ = nullptr;
};

struct BsrCreateParams {
    int64_t blockRows = 2;
    int64_t blockCols = 3;
    int64_t blockNnz = 3;
    int64_t rowBlockSize = 2;
    int64_t colBlockSize = 2;
    void* rowOffsets = nullptr;
    void* colIndices = nullptr;
    void* values = nullptr;
    aclsparseIndexType_t rowOffsetsType = ACL_SPARSE_INDEX_32I;
    aclsparseIndexType_t colIndicesType = ACL_SPARSE_INDEX_32I;
    aclsparseIndexBase_t indexBase = ACL_SPARSE_INDEX_BASE_ZERO;
    aclDataType valueType = ACL_FLOAT;
    aclsparseOrder_t order = ACL_SPARSE_ORDER_ROW;
};

aclsparseStatus_t CreateMutableBsr(
    const BsrCreateParams& p, aclsparseSpMatDescr_t* descr);

aclsparseStatus_t CreateConstBsr(
    const BsrCreateParams& p, aclsparseConstSpMatDescr_t* descr);

void ExpectMutableAndConstBsrCreate(
    const BsrCreateParams& p, aclsparseStatus_t expected);

void ExpectDnMatBatchState(
    aclsparseConstDnMatDescr_t descr, int expectedCount,
    int64_t expectedStride);
} // namespace sddmm_extended
} // namespace sparse_test

#endif // TEST_SDDMM_SDDMM_ARCH35_SDDMM_TEST_SUPPORT_H_
