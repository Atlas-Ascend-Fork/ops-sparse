// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

#pragma once

using aclrtStream = void *;
enum aclDataType {
    ACL_FLOAT = 0, ACL_FLOAT16 = 1, ACL_INT8 = 2, ACL_INT32 = 3, ACL_UINT8 = 4,
    ACL_INT16 = 6, ACL_UINT16 = 7, ACL_UINT32 = 8, ACL_INT64 = 9, ACL_UINT64 = 10,
    ACL_DOUBLE = 11, ACL_BOOL = 12, ACL_COMPLEX64 = 16, ACL_BF16 = 27
};
