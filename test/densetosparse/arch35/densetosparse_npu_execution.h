/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_DENSETOSPARSE_ARCH35_NPU_EXECUTION_H_
#define TEST_DENSETOSPARSE_ARCH35_NPU_EXECUTION_H_

namespace sparse_test {

class DenseToSparseConstDnMat {
public:
  ~DenseToSparseConstDnMat() {
    if (value_ != nullptr)
      aclsparseDestroyDnMat(value_);
  }
  aclsparseStatus_t create(int64_t rows, int64_t cols, int64_t ld,
                           const void *values, aclDataType type,
                           aclsparseOrder_t order) {
    return aclsparseCreateConstDnMat(&value_, rows, cols, ld, values, type,
                                     order);
  }
  aclsparseConstDnMatDescr_t get() const { return value_; }

private:
  aclsparseConstDnMatDescr_t value_ = nullptr;
};

struct DenseToSparseRunContext {
  HandleManager &handle;
  aclrtStream stream;
  const DenseToSparseParam &param;
  const DenseToSparseHostInput &host;
  DenseToSparseRunResult result;
  aclDataType valueType;
  size_t valueWidth;
  aclsparseIndexType_t offsetType;
  aclsparseIndexType_t indexType;
  size_t offsetWidth;
  size_t indexWidth;
  size_t offsetCount;
  int64_t bellValueCount;
  std::vector<uint8_t> patternBytes;
  DeviceBuffer denseBuffer;
  std::unique_ptr<DeviceBuffer> offsets, pattern, bellValues;
  std::unique_ptr<DeviceBuffer> index0, index1, values;
  std::unique_ptr<DeviceBuffer> analysisWorkspace, convertWorkspace;
  DenseToSparseConstDnMat dense;
  DenseToSparseSpMat sparse;

  DenseToSparseRunContext(HandleManager &h, aclrtStream s,
                          const DenseToSparseParam &p,
                          const DenseToSparseHostInput &input)
      : handle(h), stream(s), param(p), host(input),
        valueType(DenseToSparseValueType(p.value_type)),
        valueWidth(DenseToSparseTypeSize(valueType)),
        offsetType(DenseToSparseIndexType(p.offset_type)),
        indexType(DenseToSparseIndexType(p.index_type)),
        offsetWidth(offsetType == ACL_SPARSE_INDEX_64I ? 8 : 4),
        indexWidth(indexType == ACL_SPARSE_INDEX_64I ? 8 : 4),
        offsetCount(p.format == "CSR"   ? static_cast<size_t>(p.m + 1)
                    : p.format == "CSC" ? static_cast<size_t>(p.n + 1)
                                        : 0),
        bellValueCount(p.format == "BELL" ? DenseToSparseBellValueCount(
                                                p.m, p.block_size, p.ell_cols)
                                          : 0),
        patternBytes(PackDenseToSparseIndices(input.ellPattern, indexType)),
        denseBuffer(CopyDense(input)) {}

private:
  static DeviceBuffer CopyDense(const DenseToSparseHostInput &input) {
    const uint8_t empty = 0;
    const void *source = input.dense.empty()
                             ? static_cast<const void *>(&empty)
                             : static_cast<const void *>(input.dense.data());
    return DeviceBuffer::copyFrom(source,
                                  std::max<size_t>(input.dense.size(), 1));
  }
};

inline bool PrepareDenseToSparseRun(DenseToSparseRunContext *ctx) {
  ctx->handle.setStream(ctx->stream);
  if (ctx->offsetCount != 0) {
    ctx->offsets = std::make_unique<DeviceBuffer>(
        DeviceBuffer::alloc(ctx->offsetCount * ctx->offsetWidth));
  }
  ctx->pattern = DenseToSparseCopy(ctx->patternBytes);
  if (ctx->param.format == "BELL" && ctx->bellValueCount > 0) {
    ctx->bellValues = std::make_unique<DeviceBuffer>(DeviceBuffer::alloc(
        static_cast<size_t>(ctx->bellValueCount) * ctx->valueWidth));
  }
  ctx->result.descriptorStatus = ctx->dense.create(
      ctx->param.m, ctx->param.n, ctx->param.ld, ctx->denseBuffer.get(),
      ctx->valueType, DenseToSparseOrder(ctx->param.order));
  if (ctx->result.descriptorStatus != ACL_SPARSE_STATUS_SUCCESS)
    return false;
  ctx->result.descriptorStatus = ctx->sparse.create(
      ctx->param, ctx->offsets ? ctx->offsets->get() : nullptr, nullptr,
      nullptr, ctx->bellValues ? ctx->bellValues->get() : nullptr,
      ctx->pattern ? ctx->pattern->get() : nullptr);
  return ctx->result.descriptorStatus == ACL_SPARSE_STATUS_SUCCESS;
}

inline bool QueryAndAnalyzeDenseToSparse(DenseToSparseRunContext *ctx) {
  if (ctx->param.format == "BELL") {
    int64_t rows = 0, cols = 0;
    aclsparseSpMatGetSize(ctx->sparse.get(), &rows, &cols,
                          &ctx->result.bellNnzAfterQuery);
  }
  ctx->result.queryStatus = aclsparseDenseToSparseGetBufferSize(
      ctx->handle.get(), ctx->dense.get(), ctx->sparse.get(),
      ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &ctx->result.workspaceSize);
  if (ctx->result.queryStatus != ACL_SPARSE_STATUS_SUCCESS)
    return false;
  if (ctx->result.workspaceSize != 0) {
    ctx->analysisWorkspace = std::make_unique<DeviceBuffer>(
        DeviceBuffer::alloc(ctx->result.workspaceSize));
    ctx->convertWorkspace = std::make_unique<DeviceBuffer>(
        DeviceBuffer::alloc(ctx->result.workspaceSize));
    ctx->result.analysisWorkspaceAddress =
        reinterpret_cast<uintptr_t>(ctx->analysisWorkspace->get());
    ctx->result.convertWorkspaceAddress =
        reinterpret_cast<uintptr_t>(ctx->convertWorkspace->get());
  }
  ctx->result.analysisStatus = aclsparseDenseToSparseAnalysis(
      ctx->handle.get(), ctx->dense.get(), ctx->sparse.get(),
      ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
      ctx->analysisWorkspace ? ctx->analysisWorkspace->get() : nullptr);
  if (ctx->result.analysisStatus != ACL_SPARSE_STATUS_SUCCESS)
    return false;
  int64_t rows = 0, cols = 0;
  aclsparseSpMatGetSize(ctx->sparse.get(), &rows, &cols,
                        &ctx->result.queriedNnz);
  if (ctx->param.format == "BELL")
    ctx->result.bellNnzAfterAnalysis = ctx->result.queriedNnz;
  return true;
}

inline void AllocateDenseToSparseRunOutput(DenseToSparseRunContext *ctx) {
  if (ctx->param.format == "BELL" || ctx->result.queriedNnz == 0)
    return;
  ctx->index0 = std::make_unique<DeviceBuffer>(DeviceBuffer::alloc(
      static_cast<size_t>(ctx->result.queriedNnz) * ctx->indexWidth));
  if (ctx->param.format == "COO") {
    ctx->index1 = std::make_unique<DeviceBuffer>(DeviceBuffer::alloc(
        static_cast<size_t>(ctx->result.queriedNnz) * ctx->indexWidth));
  }
  ctx->values = std::make_unique<DeviceBuffer>(DeviceBuffer::alloc(
      static_cast<size_t>(ctx->result.queriedNnz) * ctx->valueWidth));
}

inline bool BindDenseToSparseRunOutput(DenseToSparseRunContext *ctx) {
  AllocateDenseToSparseRunOutput(ctx);
  if (ctx->param.format == "CSR") {
    ctx->result.setPointersStatus = aclsparseCsrSetPointers(
        ctx->sparse.get(), ctx->offsets ? ctx->offsets->get() : nullptr,
        ctx->index0 ? ctx->index0->get() : nullptr,
        ctx->values ? ctx->values->get() : nullptr);
  } else if (ctx->param.format == "CSC") {
    ctx->result.setPointersStatus = aclsparseCscSetPointers(
        ctx->sparse.get(), ctx->offsets ? ctx->offsets->get() : nullptr,
        ctx->index0 ? ctx->index0->get() : nullptr,
        ctx->values ? ctx->values->get() : nullptr);
  } else if (ctx->param.format == "COO") {
    ctx->result.setPointersStatus = aclsparseCooSetPointers(
        ctx->sparse.get(), ctx->index0 ? ctx->index0->get() : nullptr,
        ctx->index1 ? ctx->index1->get() : nullptr,
        ctx->values ? ctx->values->get() : nullptr);
  }
  return ctx->result.setPointersStatus == ACL_SPARSE_STATUS_SUCCESS;
}

inline bool ConvertDenseToSparseRun(DenseToSparseRunContext *ctx) {
  ctx->result.convertStatus = aclsparseDenseToSparseConvert(
      ctx->handle.get(), ctx->dense.get(), ctx->sparse.get(),
      ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
      ctx->convertWorkspace ? ctx->convertWorkspace->get() : nullptr);
  if (ctx->result.convertStatus != ACL_SPARSE_STATUS_SUCCESS)
    return false;
  ctx->result.syncStatus = aclrtSynchronizeStream(ctx->stream);
  if (ctx->result.syncStatus != ACL_SUCCESS)
    return false;
  if (ctx->param.format == "BELL") {
    int64_t rows = 0, cols = 0;
    aclsparseSpMatGetSize(ctx->sparse.get(), &rows, &cols,
                          &ctx->result.bellNnzAfterConvert);
  }
  return true;
}

inline void CopyDenseToSparseRunOutput(DenseToSparseRunContext *ctx) {
  ctx->result.offsets.resize(ctx->offsetCount * ctx->offsetWidth);
  if (ctx->offsets)
    ctx->offsets->copyToHost(ctx->result.offsets.data(),
                             ctx->result.offsets.size());
  ctx->result.indices0.resize(static_cast<size_t>(ctx->result.queriedNnz) *
                              ctx->indexWidth);
  ctx->result.indices1.resize(
      ctx->param.format == "COO"
          ? static_cast<size_t>(ctx->result.queriedNnz) * ctx->indexWidth
          : 0);
  ctx->result.values.resize(
      ctx->param.format == "BELL"
          ? static_cast<size_t>(ctx->bellValueCount) * ctx->valueWidth
          : static_cast<size_t>(ctx->result.queriedNnz) * ctx->valueWidth);
  if (ctx->index0)
    ctx->index0->copyToHost(ctx->result.indices0.data(),
                            ctx->result.indices0.size());
  if (ctx->index1)
    ctx->index1->copyToHost(ctx->result.indices1.data(),
                            ctx->result.indices1.size());
  if (ctx->values)
    ctx->values->copyToHost(ctx->result.values.data(),
                            ctx->result.values.size());
  if (ctx->bellValues)
    ctx->bellValues->copyToHost(ctx->result.values.data(),
                                ctx->result.values.size());
  ctx->result.ellPattern.resize(ctx->patternBytes.size());
  if (ctx->pattern)
    ctx->pattern->copyToHost(ctx->result.ellPattern.data(),
                             ctx->result.ellPattern.size());
}

inline DenseToSparseRunResult
RunDenseToSparse(HandleManager &handle, aclrtStream stream,
                 const DenseToSparseParam &p,
                 const DenseToSparseHostInput &host) {
  DenseToSparseRunContext ctx(handle, stream, p, host);
  if (!PrepareDenseToSparseRun(&ctx))
    return ctx.result;
  if (!QueryAndAnalyzeDenseToSparse(&ctx))
    return ctx.result;
  if (!BindDenseToSparseRunOutput(&ctx))
    return ctx.result;
  if (!ConvertDenseToSparseRun(&ctx))
    return ctx.result;
  CopyDenseToSparseRunOutput(&ctx);
  return ctx.result;
}

} // namespace sparse_test
#endif
