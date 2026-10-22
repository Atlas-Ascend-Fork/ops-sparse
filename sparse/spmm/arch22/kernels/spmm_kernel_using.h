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

#include "kernel_operator.h"

// 集中的 AscendC 具体名字引入（using-declaration，而非 using-directive），供
// spmm_kernel.h 与 spmm_kernel_group.h 共用：
//   - G.INC.08：头文件内不用 `using namespace` 导入整个命名空间，仅按名引入所需符号；
//     与 spmm_kernel_common.h 的 AscendC:: 全限定同源。
//   - 抽成单独一份被两处 #include，避免在两个头文件里重复该清单（重复代码），也控制
//     单个头文件的 NBNC 行数（超大头文件）。
// 注：event_t / GetTPipePtr / PIPE_V 属全局命名空间（::），本就可见，无需在此引入。
using AscendC::Add;
using AscendC::Adds;
using AscendC::Axpy;
using AscendC::Cast;
using AscendC::DataCopy;
using AscendC::DataCopyPad;
using AscendC::DataCopyParams;
using AscendC::DataCopyPadParams;
using AscendC::Duplicate;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::Muls;
using AscendC::PipeBarrier;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using AscendC::WaitFlag;
