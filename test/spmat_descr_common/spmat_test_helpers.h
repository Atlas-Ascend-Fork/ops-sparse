/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FRAME_SPMAT_TEST_HELPERS_H_
#define TEST_FRAME_SPMAT_TEST_HELPERS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cann_ops_sparse.h"

#include <gtest/gtest.h>

namespace sparse_test {

// ============================================================================
// 描述符 RAII 守卫（沿用 test/spmat_get_indexbase 的守卫形态，含 release 语义：
// 显式 Destroy 前 release 释放所有权，避免双重释放）。mutable/const 创建入口
// 共用：底层是同一 aclsparseSpMatDescr 对象，const 变体句柄仅在创建签名上
// 不同（spmat_get_attribute / destroy_spmat wrapper 共享）。
// ============================================================================
class SpMatGuard {
public:
    SpMatGuard() = default;
    ~SpMatGuard()
    {
        if (p_ != nullptr) {
            (void)aclsparseDestroySpMat(p_);
        }
    }
    // 禁止拷贝（CG-15.6）：析构销毁句柄，隐式浅拷贝将双重销毁
    //（对齐 descriptor_manager.h SpMatManager 先例；无按值返回需求，移动语义不加）
    SpMatGuard(const SpMatGuard&) = delete;
    SpMatGuard& operator=(const SpMatGuard&) = delete;
    aclsparseSpMatDescr_t *put()
    {
        if (p_ != nullptr) {
            (void)aclsparseDestroySpMat(p_);
            p_ = nullptr;
        }
        return &p_;
    }
    // 释放所有权：调用方接管句柄后须自行 Destroy（本守卫不再释放）
    aclsparseSpMatDescr_t release()
    {
        aclsparseSpMatDescr_t handle = p_;
        p_ = nullptr;
        return handle;
    }
    aclsparseSpMatDescr_t get() const { return p_; }
    aclsparseConstSpMatDescr_t cget() const { return p_; }

private:
    aclsparseSpMatDescr_t p_ = nullptr;
};

// ============================================================================
// 入口注册表原语（注册表 kEntries 数组本体仍定义在各 test.cpp 内，对齐先例
// spmat_get_indexbase 的位置约定——编译单元私有数据，不在头文件中定义非
// inline 数组；共享的条目结构与查找逻辑在此收敛）。
// ============================================================================
struct FormatEntry {
    const char *name;
    bool isConst;
};

/// CSV entry 名 -> 注册表查找；未注册名返回 nullptr（测试侧 ASSERT 拦截）
template <size_t N>
inline const FormatEntry *FindEntry(const FormatEntry (&entries)[N], const std::string &name)
{
    for (const auto &e : entries) {
        if (name == e.name) {
            return &e;
        }
    }
    return nullptr;
}

// Create/Destroy 均不解引用设备数据，缓冲内容任意，仅要求指针有效
inline constexpr int64_t kRows = 4;
inline constexpr int64_t kCols = 4;
inline constexpr int64_t kNnz = 5;

// ============================================================================
// 统一创建模板：Create 成功/失败都把句柄挂到 guard 上。
// ============================================================================
template <typename CreateFn, typename... Args>
aclsparseStatus_t CreateSpMatEntry(SpMatGuard *guard, CreateFn create, Args... args)
{
    aclsparseSpMatDescr_t descr = nullptr;
    const aclsparseStatus_t status = create(&descr, args...);
    *guard->put() = descr;
    return status;
}

// const 变体创建模板：写 const 句柄后转存为 guard 的 mutable 视图
//（底层为同一 aclsparseSpMatDescr 对象，与 SpMatManager::createConst* 同法）
template <typename CreateFn, typename... Args>
aclsparseStatus_t CreateConstSpMatEntry(SpMatGuard *guard, CreateFn create, Args... args)
{
    aclsparseConstSpMatDescr_t descr = nullptr;
    const aclsparseStatus_t status = create(&descr, args...);
    *guard->put() = const_cast<aclsparseSpMatDescr_t>(descr);
    return status;
}

// ============================================================================
// 描述符句柄 unique_ptr RAII 包装（aclsparseDestroySpMat 删除器）。BSR/COO/CSR
// 等无 SpMatManager 工厂的直调用例共用（先例 csr_set_strided_batch wrapper）。
// ============================================================================
inline std::unique_ptr<aclsparseSpMatDescr, decltype(&aclsparseDestroySpMat)> WrapSpMatHandle(
    aclsparseSpMatDescr_t descr)
{
    return std::unique_ptr<aclsparseSpMatDescr, decltype(&aclsparseDestroySpMat)>(
        descr, &aclsparseDestroySpMat);
}

// ============================================================================
// CooGet/CsrGet 全字段回读的出参哨兵组：预置明确非法初值（-1 / nullptr /
// 非缺省枚举），实现漏写任一字段时回读值保持哨兵、可被调用方断言发现。
// 字段命名取中性 ptrA/ptrB/ptrC：COO 语义为 rowInd/colInd/values，
// CSR 语义为 offsets/colInd/values（ptrType 仅 CSR 的 CsrGet 回读使用）。
// ============================================================================
struct SpMatGetOutParams {
    int64_t rows{-1};
    int64_t cols{-1};
    int64_t nnz{-1};
    void *ptrA{nullptr};
    void *ptrB{nullptr};
    void *ptrC{nullptr};
    aclsparseIndexType_t ptrType{ACL_SPARSE_INDEX_64I};
    aclsparseIndexType_t idxType{ACL_SPARSE_INDEX_64I};
    aclsparseIndexBase_t idxBase{ACL_SPARSE_INDEX_BASE_ONE};
    aclDataType valueType{ACL_BF16};
};

// ============================================================================
// Create 系 wrapper 结果公共回读段：Create + 四个通用 Get 访问器
//（Format/Size/IndexBase/StridedBatch）的返回码与回读值。未执行步骤保持
// NOT_INITIALIZED 哨兵，不误报 SUCCESS；白盒字段由各 wrapper 结果结构追加
//（create_bsr / create_sliced_ell 共享）。
// ============================================================================
struct SpMatCreateGetResult {
    aclsparseStatus_t createRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t getFormatRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t getSizeRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t getIndexBaseRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};
    aclsparseStatus_t getBatchRet{ACL_SPARSE_STATUS_NOT_INITIALIZED};

    // 通用 Get 访问器回读值（GetSize 为块展开后的标量维度）
    aclsparseFormat_t format{static_cast<aclsparseFormat_t>(-1)};
    int64_t rows{-1};
    int64_t cols{-1};
    int64_t nnz{-1};
    aclsparseIndexBase_t idxBase{static_cast<aclsparseIndexBase_t>(-1)};
    int batchCount{-1};
};

/// 通用 Get 访问器回读（四步）：填充 r 公共段字段的返回码与回读值，任一
/// 失败返回 false（调用方负责错误输出与提前返回）。纯 Host 元数据操作，
/// 不解引用设备数据。
inline bool ReadBackGenericGets(aclsparseConstSpMatDescr_t descr, SpMatCreateGetResult &r)
{
    r.getFormatRet = aclsparseSpMatGetFormat(descr, &r.format);
    r.getSizeRet = aclsparseSpMatGetSize(descr, &r.rows, &r.cols, &r.nnz);
    r.getIndexBaseRet = aclsparseSpMatGetIndexBase(descr, &r.idxBase);
    r.getBatchRet = aclsparseSpMatGetStridedBatch(descr, &r.batchCount);
    return r.getFormatRet == ACL_SPARSE_STATUS_SUCCESS &&
        r.getSizeRet == ACL_SPARSE_STATUS_SUCCESS &&
        r.getIndexBaseRet == ACL_SPARSE_STATUS_SUCCESS &&
        r.getBatchRet == ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// 语义用例基线 BSR 描述符（2×2 块网格、bs=2、nnzb=2，host 占位数据，
// Create 不解引用）：直调 aclsparseCreateBsr + unique_ptr RAII。
// create_bsr BSR-S02~S04 与 bsr_set_strided_batch BSRB-S01~S02 语义用例共用
//（BSR 无 SpMatManager 工厂，沿用 csr_set_strided_batch 先例形态）。
// ============================================================================
inline std::unique_ptr<aclsparseSpMatDescr, decltype(&aclsparseDestroySpMat)> MakeValidBsr()
{
    // static 复用占位数据；非 const 以匹配 aclsparseCreateBsr 的 void* 入参
    //（Create 仅地址算术不解引用）
    static std::vector<int32_t> hostOffsets{0, 1, 2};
    static std::vector<int32_t> hostColInd{0, 1};
    static std::vector<float> hostValues{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateBsr(&descr, 2, 2, 2, 2, 2, hostOffsets.data(), hostColInd.data(),
                                 hostValues.data(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT, ACL_SPARSE_ORDER_ROW),
              ACL_SPARSE_STATUS_SUCCESS);
    return WrapSpMatHandle(descr);
}

}  // namespace sparse_test

#endif  // TEST_FRAME_SPMAT_TEST_HELPERS_H_
