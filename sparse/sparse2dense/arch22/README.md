# sparse2dense arch22（Atlas A2/A3，DAV_2201）

aclsparseSparseToDense 的 arch22 实现（Atlas A2 训练系列 910B3/910B4、Atlas A3 训练系列
910_93）：`op_host/`（Host 校验链 CheckMeta → 坐标校验（设备侧指纹记忆化）→ FillTiling）
+ `op_kernel/`（清零 + scatter 双 kernel，桶转置 / 批量散布直写快路径）+
`sparse2dense_host.cpp`（公开 API 描述符映射层）+ `torch_extension/`（`aten::_to_dense`
NPU 稀疏后端）。

## 构建

```bash
# 仓顶主线（libops_sparse.so，含本算子公开入口）
bash build.sh --ops=sparse2dense                # 默认 ascend910b3
bash build.sh --ops=sparse2dense --soc=ascend910_93

# 编译并运行测试（C++ UT + ST，NPU 真机）
bash build.sh --ops=sparse2dense --run

# Torch Extension wheel（aten::_to_dense NPU 注册，JIT 延迟编译）
bash build.sh --torch_extension --ops=sparse2dense
python3 -m pip install build_out/cann_ops_sparse-*.whl --force-reinstall --no-deps
```

## 测试

- C++ UT（`test/sparse2dense/arch22/ut/`）：Host 校验链 / tiling / 溢出防护 / 分片边界
  （x86 可跑，aclrt 桩，无 NPU 依赖）。
- C++ ST（`test/sparse2dense/arch22/sparse2dense_test.cpp`）：公开 API 双阶段协议 +
  bit-wise golden（含 ±0/±inf/nan 位元、输入只读、存储缓冲哨兵）+ L2 校验链负例
  （拒绝阶段与状态码精确断言）。
- Python 端到端（`test/sparse2dense/python/test_sparse2dense_torch.py`）：
  `aten::_to_dense` dispatcher 注册断言 + CSR/CSC/COO × 五 dtype bit-wise + int64→I32
  Device 侧 cast + 乱序唯一坐标 + COO 重复坐标求和语义（aclsparse API 层对重复坐标为任一写入值契约，torch 适配层预归并求和） + 重复执行确定性 + 异常矩阵 + CPU 不拦截。

## torch 适配（`torch_extension/`）

`Tensor.to_dense` → `aten::_to_dense`（SparsePrivateUse1 / SparseCsrPrivateUse1）NPU 注册，
元数据映射后经 aclsparseSparseToDense 公开 API 执行，全程 NPU 无 CPU fallback。
`import cann_ops_sparse` 须在 `import torch_npu` 之后（Dispatcher 同 (op, key) 后注册
覆盖接管）。
