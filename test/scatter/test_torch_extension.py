# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""aclsparseScatter Torch Extension 的端到端测试。"""

import torch
import torch_npu  # noqa: F401  # NPU backend 注册
import cann_ops_sparse  # noqa: F401  # 触发算子包导入与 dispatch 注册

DTYPES = (torch.int8, torch.float16, torch.bfloat16, torch.float32, torch.complex64)


def _make_case(size, nnz, dtype, device, seed=7):
    """构造 (y, idx, src) 测试数据：idx 为无重复乱序索引，数值含正负混合。"""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    y = torch.randn(size, generator=generator, dtype=torch.float32).to(dtype).to(device)
    idx = torch.randperm(size, generator=generator)[:nnz].to(torch.int32).to(device)
    src = torch.randn(nnz, generator=generator, dtype=torch.float32).to(dtype).to(device)
    return y, idx, src


def test_scatter_torch_extension_five_dtypes():
    """验证五种 dtype 的 index_copy_ NPU 路径与 CPU 参考逐位一致。"""
    device = "npu:{}".format(torch.npu.current_device())
    for dtype in DTYPES:
        y, idx, src = _make_case(4096, 1024, dtype, device)
        reference = y.cpu().clone()
        reference.index_copy_(0, idx.cpu().long(), src.cpu())

        y.index_copy_(0, idx, src)  # NPU 路径 → aclsparseScatter
        torch.npu.synchronize()

        torch.testing.assert_close(y.cpu(), reference, rtol=0, atol=0)


def test_scatter_duplicate_indices_last_write_wins():
    """重复索引为 last-write-wins 语义族（结果取对应 values 之一）。"""
    device = "npu:{}".format(torch.npu.current_device())
    y = torch.zeros(8, dtype=torch.float32, device=device)
    idx = torch.tensor([1, 1, 1, 5], dtype=torch.int32, device=device)
    src = torch.tensor([10.0, 20.0, 30.0, 4.0], dtype=torch.float32, device=device)

    y.index_copy_(0, idx, src)
    torch.npu.synchronize()
    result = y.cpu()

    # 重复索引结果为对应输入 values 之一（非确定 last-write-wins），
    # 禁止承诺确定输出；重复位置不设逐位参考，其余位置保持初值。
    assert int(result[1]) in (10.0, 20.0, 30.0)
    assert int(result[5]) == 4
    others = [i for i in range(8) if i not in (1, 5)]
    assert all(int(result[i]) == 0 for i in others)


def test_scatter_empty_index_unchanged():
    """nnz=0 成功返回且不修改 y。"""
    device = "npu:{}".format(torch.npu.current_device())
    y, _, _ = _make_case(256, 0, torch.float32, device)
    idx = torch.empty(0, dtype=torch.int32, device=device)
    src = torch.empty(0, dtype=torch.float32, device=device)

    y_ref = y.clone()
    y.index_copy_(0, idx, src)
    torch.npu.synchronize()
    assert y.numel() == 256
    assert torch.equal(y, y_ref)


def test_scatter_error_paths():
    """非法输入确定拒绝且输出逐位未修改；合法路径不受影响。

    入口校验（TORCH_CHECK，含索引越界检查 CheckIndexBounds）先于任何
    kernel/stream 提交，以输出逐位不变为未启动证据（不引入 mock/调用计数）。
    每组非法输入（int64 index / 越界正索引 / 负索引 / 2-D source）断言
    RuntimeError 抛出 + y 逐位未修改；全部拒绝后追加一次合法 index_copy_
    同 y bit-exact 验证（上下文干净、合法路径不受影响）。
    """
    device = "npu:{}".format(torch.npu.current_device())
    y = torch.randn(16, dtype=torch.float32, device=device)
    src = torch.randn(4, dtype=torch.float32, device=device)

    illegal_groups = [
        ("int64 index", torch.arange(4, dtype=torch.int64, device=device), src),
        ("out-of-range positive index",
         torch.tensor([0, 16], dtype=torch.int32, device=device), src[:2]),
        ("negative index",
         torch.tensor([0, -1], dtype=torch.int32, device=device), src[:2]),
        ("2-D source", torch.arange(4, dtype=torch.int32, device=device), src.reshape(2, 2)),
    ]
    for label, idx, source in illegal_groups:
        y_ref = y.clone()
        try:
            y.index_copy_(0, idx, source)
            raise AssertionError("{} should raise".format(label))
        except RuntimeError:
            pass
        torch.npu.synchronize()
        assert torch.equal(y, y_ref), "{}: y modified on rejection".format(label)

    # 全部拒绝后合法路径复验：同 y 一次合法 index_copy_ 与 CPU 参考逐位一致
    legal_idx = torch.tensor([3, 1, 7, 0], dtype=torch.int32, device=device)
    reference = y.cpu().clone()
    reference.index_copy_(0, legal_idx.cpu().long(), src.cpu())
    y.index_copy_(0, legal_idx, src)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), reference, rtol=0, atol=0)


def test_scatter_non_default_stream():
    """非默认 stream 下调用与队列顺序正确。"""
    device = "npu:{}".format(torch.npu.current_device())
    stream = torch.npu.Stream(device=device)
    y, idx, src = _make_case(1024, 256, torch.float32, device, seed=11)
    reference = y.cpu().clone()
    reference.index_copy_(0, idx.cpu().long(), src.cpu())

    with torch.npu.stream(stream):
        y.index_copy_(0, idx, src)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), reference, rtol=0, atol=0)


def test_scatter_inference_mode():
    """inference_mode 内创建并调用的张量正常执行（index 版本计数器禁用路径）。

    inference tensor 的 version_counter 处于 disabled 态，越界校验须绕过
    记忆化逐次同步校验，不得读取 current_version()（否则 RuntimeError）。
    """
    device = "npu:{}".format(torch.npu.current_device())
    with torch.inference_mode():
        y = torch.zeros(16, dtype=torch.float32, device=device)
        idx = torch.tensor([3, 1, 7], dtype=torch.int32, device=device)
        src = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        y.index_copy_(0, idx, src)
        torch.npu.synchronize()

    result = y.cpu()
    assert int(result[1]) == 2
    assert int(result[3]) == 1
    assert int(result[7]) == 3
    assert all(int(result[i]) == 0 for i in range(16) if i not in (1, 3, 7))

    # 越界确定报错在绕过记忆化的逐次校验路径下同样成立
    with torch.inference_mode():
        y2 = torch.zeros(8, dtype=torch.float32, device=device)
        bad = torch.tensor([0, 8], dtype=torch.int32, device=device)
        src2 = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
        try:
            y2.index_copy_(0, bad, src2)
            raise AssertionError("out-of-range inference index should raise")
        except RuntimeError:
            pass
