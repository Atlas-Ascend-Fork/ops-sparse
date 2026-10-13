/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FRAME_DESCRIPTOR_MANAGER_H_
#define TEST_FRAME_DESCRIPTOR_MANAGER_H_

#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "acl/acl.h"
#include "cann_ops_sparse.h"

namespace sparse_test {

class SpMatManager {
public:
    SpMatManager() = default;

    static SpMatManager createCsr(int64_t rows, int64_t cols, int64_t nnz,
                                   void* rowOffsets, void* colIndices, void* values,
                                   aclsparseIndexType_t rowOffsetType = ACL_SPARSE_INDEX_32I,
                                   aclsparseIndexType_t colIdxType = ACL_SPARSE_INDEX_32I,
                                   aclsparseIndexBase_t idxBase = ACL_SPARSE_INDEX_BASE_ZERO,
                                   aclDataType valueType = ACL_FLOAT) {
        SpMatManager m;
        auto s = aclsparseCreateCsr(&m.descr_, rows, cols, nnz, rowOffsets, colIndices,
                                    values, rowOffsetType, colIdxType, idxBase, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateCsr failed");
        return m;
    }

    static SpMatManager createConstCsr(int64_t rows, int64_t cols, int64_t nnz,
                                        const void* rowOffsets, const void* colIndices, const void* values,
                                        aclsparseIndexType_t rowOffsetType = ACL_SPARSE_INDEX_32I,
                                        aclsparseIndexType_t colIdxType = ACL_SPARSE_INDEX_32I,
                                        aclsparseIndexBase_t idxBase = ACL_SPARSE_INDEX_BASE_ZERO,
                                        aclDataType valueType = ACL_FLOAT) {
        SpMatManager m;
        aclsparseConstSpMatDescr_t constDescr = nullptr;
        auto s = aclsparseCreateConstCsr(&constDescr, rows, cols, nnz, rowOffsets, colIndices,
                                         values, rowOffsetType, colIdxType, idxBase, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateConstCsr failed");
        m.descr_ = const_cast<aclsparseSpMatDescr_t>(constDescr);
        return m;
    }

    static SpMatManager createCsc(int64_t rows, int64_t cols, int64_t nnz,
                                   void* colOffsets, void* rowIndices, void* values,
                                   aclsparseIndexType_t colOffsetType = ACL_SPARSE_INDEX_32I,
                                   aclsparseIndexType_t rowIdxType = ACL_SPARSE_INDEX_32I,
                                   aclsparseIndexBase_t idxBase = ACL_SPARSE_INDEX_BASE_ZERO,
                                   aclDataType valueType = ACL_FLOAT) {
        SpMatManager m;
        auto s = aclsparseCreateCsc(&m.descr_, rows, cols, nnz, colOffsets, rowIndices,
                                    values, colOffsetType, rowIdxType, idxBase, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateCsc failed");
        return m;
    }

    static SpMatManager createConstCsc(int64_t rows, int64_t cols, int64_t nnz,
                                        const void* colOffsets, const void* rowIndices, const void* values,
                                        aclsparseIndexType_t colOffsetType = ACL_SPARSE_INDEX_32I,
                                        aclsparseIndexType_t rowIdxType = ACL_SPARSE_INDEX_32I,
                                        aclsparseIndexBase_t idxBase = ACL_SPARSE_INDEX_BASE_ZERO,
                                        aclDataType valueType = ACL_FLOAT) {
        SpMatManager m;
        aclsparseConstSpMatDescr_t constDescr = nullptr;
        auto s = aclsparseCreateConstCsc(&constDescr, rows, cols, nnz, colOffsets, rowIndices,
                                         values, colOffsetType, rowIdxType, idxBase, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateConstCsc failed");
        m.descr_ = const_cast<aclsparseSpMatDescr_t>(constDescr);
        return m;
    }

    static SpMatManager createCoo(int64_t rows, int64_t cols, int64_t nnz,
                                   void* rowInd, void* colInd, void* values,
                                   aclsparseIndexType_t cooIdxType = ACL_SPARSE_INDEX_32I,
                                   aclsparseIndexBase_t idxBase = ACL_SPARSE_INDEX_BASE_ZERO,
                                   aclDataType valueType = ACL_FLOAT) {
        SpMatManager m;
        auto s = aclsparseCreateCoo(&m.descr_, rows, cols, nnz, rowInd, colInd,
                                    values, cooIdxType, idxBase, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateCoo failed");
        return m;
    }

    static SpMatManager createConstCoo(int64_t rows, int64_t cols, int64_t nnz,
                                        const void* rowInd, const void* colInd, const void* values,
                                        aclsparseIndexType_t cooIdxType = ACL_SPARSE_INDEX_32I,
                                        aclsparseIndexBase_t idxBase = ACL_SPARSE_INDEX_BASE_ZERO,
                                        aclDataType valueType = ACL_FLOAT) {
        SpMatManager m;
        aclsparseConstSpMatDescr_t constDescr = nullptr;
        auto s = aclsparseCreateConstCoo(&constDescr, rows, cols, nnz, rowInd, colInd,
                                         values, cooIdxType, idxBase, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateConstCoo failed");
        m.descr_ = const_cast<aclsparseSpMatDescr_t>(constDescr);
        return m;
    }

    static SpMatManager createSlicedEll(int64_t rows, int64_t cols, int64_t nnz,
                                         void* slicePtr, void* colInd, void* values,
                                         int64_t sliceNnz, int64_t numSlices,
                                         aclsparseIndexType_t sellIdxType = ACL_SPARSE_INDEX_32I,
                                         aclsparseIndexBase_t idxBase = ACL_SPARSE_INDEX_BASE_ZERO,
                                         aclDataType valueType = ACL_FLOAT) {
        SpMatManager m;
        auto s = aclsparseCreateSlicedEll(&m.descr_, rows, cols, nnz, sliceNnz, numSlices,
                                          slicePtr, colInd, values,
                                          sellIdxType, idxBase, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateSlicedEll failed");
        return m;
    }

    static SpMatManager createConstSlicedEll(int64_t rows, int64_t cols, int64_t nnz,
                                              const void* slicePtr, const void* colInd, const void* values,
                                              int64_t sliceNnz, int64_t numSlices,
                                              aclsparseIndexType_t sellIdxType = ACL_SPARSE_INDEX_32I,
                                              aclsparseIndexBase_t idxBase = ACL_SPARSE_INDEX_BASE_ZERO,
                                              aclDataType valueType = ACL_FLOAT) {
        SpMatManager m;
        aclsparseConstSpMatDescr_t constDescr = nullptr;
        auto s = aclsparseCreateConstSlicedEll(&constDescr, rows, cols, nnz, sliceNnz, numSlices,
                                               slicePtr, colInd, values,
                                               sellIdxType, idxBase, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateConstSlicedEll failed");
        m.descr_ = const_cast<aclsparseSpMatDescr_t>(constDescr);
        return m;
    }

    void setAttribute(aclsparseSpMatAttribute_t attr, const void* data, size_t dataSize) {
        auto s = aclsparseSpMatSetAttribute(descr_, attr, data, dataSize);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseSpMatSetAttribute failed");
    }

    aclsparseSpMatDescr_t get() { return descr_; }
    aclsparseConstSpMatDescr_t cget() const { return descr_; }

    ~SpMatManager() {
        if (descr_) aclsparseDestroySpMat(descr_);
    }

    SpMatManager(const SpMatManager&) = delete;
    SpMatManager& operator=(const SpMatManager&) = delete;
    SpMatManager(SpMatManager&& other) noexcept : descr_(other.descr_) {
        other.descr_ = nullptr;
    }
    SpMatManager& operator=(SpMatManager&& other) noexcept {
        if (this != &other) {
            if (descr_) aclsparseDestroySpMat(descr_);
            descr_ = other.descr_;
            other.descr_ = nullptr;
        }
        return *this;
    }

private:
    aclsparseSpMatDescr_t descr_ = nullptr;
};

class DnVecManager {
public:
    DnVecManager() = default;

    static DnVecManager create(int64_t size, void* values, aclDataType valueType) {
        DnVecManager m;
        auto s = aclsparseCreateDnVec(&m.descr_, size, values, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateDnVec failed");
        return m;
    }

    static DnVecManager createConst(int64_t size, const void* values, aclDataType valueType) {
        DnVecManager m;
        aclsparseConstDnVecDescr_t constDescr = nullptr;
        auto s = aclsparseCreateConstDnVec(&constDescr, size, values, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateConstDnVec failed");
        m.descr_ = const_cast<aclsparseDnVecDescr_t>(constDescr);
        return m;
    }

    aclsparseDnVecDescr_t get() { return descr_; }
    aclsparseConstDnVecDescr_t cget() const { return descr_; }

    ~DnVecManager() {
        if (descr_) aclsparseDestroyDnVec(descr_);
    }

    DnVecManager(const DnVecManager&) = delete;
    DnVecManager& operator=(const DnVecManager&) = delete;
    DnVecManager(DnVecManager&& other) noexcept : descr_(other.descr_) { other.descr_ = nullptr; }
    DnVecManager& operator=(DnVecManager&& other) noexcept {
        if (this != &other) {
            if (descr_) aclsparseDestroyDnVec(descr_);
            descr_ = other.descr_;
            other.descr_ = nullptr;
        }
        return *this;
    }

private:
    aclsparseDnVecDescr_t descr_ = nullptr;
};

// ============================================================================
// SpVecManager: RAII wrapper for aclsparseCreateSpVec / CreateConstSpVec /
// DestroySpVec. Follows the same style as DnVecManager / SpMatManager.
// Used by SpVec-DnVec operators (e.g. aclsparseScatter / aclsparseGather).
// ============================================================================
class SpVecManager {
public:
    SpVecManager() = default;

    static SpVecManager create(int64_t size, int64_t nnz,
                                void* indices, void* values,
                                aclsparseIndexType_t idxType,
                                aclsparseIndexBase_t idxBase,
                                aclDataType valueType) {
        SpVecManager m;
        auto s = aclsparseCreateSpVec(&m.descr_, size, nnz, indices, values,
                                       idxType, idxBase, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateSpVec failed");
        return m;
    }

    static SpVecManager createConst(int64_t size, int64_t nnz,
                                     const void* indices, const void* values,
                                     aclsparseIndexType_t idxType,
                                     aclsparseIndexBase_t idxBase,
                                     aclDataType valueType) {
        SpVecManager m;
        aclsparseConstSpVecDescr_t constDescr = nullptr;
        auto s = aclsparseCreateConstSpVec(&constDescr, size, nnz, indices, values,
                                            idxType, idxBase, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateConstSpVec failed");
        m.descr_ = const_cast<aclsparseSpVecDescr_t>(constDescr);
        return m;
    }

    aclsparseSpVecDescr_t get() { return descr_; }
    aclsparseConstSpVecDescr_t cget() const { return descr_; }

    ~SpVecManager() {
        if (descr_) aclsparseDestroySpVec(descr_);
    }

    SpVecManager(const SpVecManager&) = delete;
    SpVecManager& operator=(const SpVecManager&) = delete;
    SpVecManager(SpVecManager&& other) noexcept : descr_(other.descr_) {
        other.descr_ = nullptr;
    }
    SpVecManager& operator=(SpVecManager&& other) noexcept {
        if (this != &other) {
            if (descr_) aclsparseDestroySpVec(descr_);
            descr_ = other.descr_;
            other.descr_ = nullptr;
        }
        return *this;
    }

private:
    aclsparseSpVecDescr_t descr_ = nullptr;
};

class DnMatManager {
public:
    DnMatManager() = default;

    static DnMatManager create(int64_t rows, int64_t cols, int64_t ld,
                                void* values, aclDataType valueType, aclsparseOrder_t order) {
        DnMatManager m;
        auto s = aclsparseCreateDnMat(&m.descr_, rows, cols, ld, values, valueType, order);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateDnMat failed");
        return m;
    }

    static DnMatManager createConst(int64_t rows, int64_t cols, int64_t ld,
                                     const void* values, aclDataType valueType, aclsparseOrder_t order) {
        DnMatManager m;
        aclsparseConstDnMatDescr_t constDescr = nullptr;
        auto s = aclsparseCreateConstDnMat(&constDescr, rows, cols, ld, values, valueType, order);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateConstDnMat failed");
        m.descr_ = const_cast<aclsparseDnMatDescr_t>(constDescr);
        return m;
    }

    aclsparseDnMatDescr_t get() { return descr_; }
    aclsparseConstDnMatDescr_t cget() const { return descr_; }

    ~DnMatManager() {
        if (descr_) aclsparseDestroyDnMat(descr_);
    }

    DnMatManager(const DnMatManager&) = delete;
    DnMatManager& operator=(const DnMatManager&) = delete;
    DnMatManager(DnMatManager&& other) noexcept : descr_(other.descr_) { other.descr_ = nullptr; }
    DnMatManager& operator=(DnMatManager&& other) noexcept {
        if (this != &other) {
            if (descr_) aclsparseDestroyDnMat(descr_);
            descr_ = other.descr_;
            other.descr_ = nullptr;
        }
        return *this;
    }

private:
    aclsparseDnMatDescr_t descr_ = nullptr;
};

class HandleManager {
public:
    HandleManager() {
        auto s = aclsparseCreate(&handle_);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreate failed");
    }

    void setStream(aclrtStream stream) {
        auto s = aclsparseSetStream(handle_, stream);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseSetStream failed");
    }

    aclsparseHandle_t get() { return handle_; }

    ~HandleManager() {
        if (handle_) aclsparseDestroy(handle_);
    }

    HandleManager(const HandleManager&) = delete;
    HandleManager& operator=(const HandleManager&) = delete;

private:
    aclsparseHandle_t handle_ = nullptr;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    static DeviceBuffer alloc(size_t size) {
        DeviceBuffer b;
        if (size == 0) return b;
        auto ret = aclrtMalloc(reinterpret_cast<void**>(&b.ptr_), size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) throw std::runtime_error("aclrtMalloc failed");
        b.size_ = size;
        return b;
    }

    static DeviceBuffer copyFrom(const void* hostPtr, size_t size) {
        if (size == 0) return DeviceBuffer();
        auto b = alloc(size);
        auto ret = aclrtMemcpy(b.ptr_, size, hostPtr, size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) throw std::runtime_error("aclrtMemcpy H2D failed");
        return b;
    }

    void copyToHost(void* hostPtr, size_t size) const {
        if (size == 0) {
            return;
        }
        auto ret = aclrtMemcpy(hostPtr, size, ptr_, size, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) throw std::runtime_error("aclrtMemcpy D2H failed");
    }

    void* get() { return ptr_; }
    void* raw() const { return ptr_; }
    size_t size() const { return size_; }

    ~DeviceBuffer() {
        if (ptr_) aclrtFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) aclrtFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

class SpSVDescrManager {
public:
    SpSVDescrManager() {
        auto s = aclsparseSpSV_createDescr(&descr_);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseSpSV_createDescr failed");
    }

    aclsparseSpSVDescr_t get() { return descr_; }

    ~SpSVDescrManager() {
        if (descr_) aclsparseSpSV_destroyDescr(descr_);
    }

    SpSVDescrManager(const SpSVDescrManager&) = delete;
    SpSVDescrManager& operator=(const SpSVDescrManager&) = delete;
    SpSVDescrManager(SpSVDescrManager&& other) noexcept : descr_(other.descr_) {
        other.descr_ = nullptr;
    }
    SpSVDescrManager& operator=(SpSVDescrManager&& other) noexcept {
        if (this != &other) {
            if (descr_) aclsparseSpSV_destroyDescr(descr_);
            descr_ = other.descr_;
            other.descr_ = nullptr;
        }
        return *this;
    }

private:
    aclsparseSpSVDescr_t descr_ = nullptr;
};

}

#endif
