# SparseToDense Torch Extension 测试

## 测试说明

本目录验证 SparseToDense 通过统一 `torch_extension` 注册的 `aten::_to_dense`
（CSR/CSC/COO → `Tensor.to_dense()`）。算子接口与规格见
[SparseToDense 算子说明](../../../sparse/sparse2dense/README.md) 与
[Torch Extension 接口说明](../../../torch_extension/cann_ops_sparse/docs/zh/sparse2dense.md)。

## 前置环境

在已安装 CANN、匹配的 PyTorch / `torch_npu` 的 Ascend 设备上执行前，先配置环境：

```bash
source /usr/local/Ascend/cann/set_env.sh
# 或：source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

可选环境变量：

| 变量 | 含义 | 默认 |
| --- | --- | --- |
| `SOC` | `build.sh --soc` | `ascend950` |
| `OPS_SPARSE_LIB_DIR` | `libops_sparse.so` 所在目录 | 脚本自动探测 `build_out/lib64` |
| `TORCH_EXTENSIONS_DIR` | JIT 扩展缓存目录 | `$PWD/.torch_extensions` |

## 运行

在仓库根目录执行以下构建和测试步骤：

```bash
CMAKE_BUILD_TYPE=Release bash build.sh --ops=sparse2dense --soc="${SOC:-ascend950}"
python3 -m pip install -r torch_extension/requirements.txt
bash build.sh --torch_extension --ops=sparse2dense
python3 -m pip install --force-reinstall --no-deps build_out/cann_ops_sparse-*.whl
export OPS_SPARSE_LIB_DIR=$PWD/build_out/lib64
export TORCH_EXTENSIONS_DIR=$PWD/.torch_extensions
python3 -m pytest -q test/sparse2dense/python/test_sparse2dense_torch_extension.py -v
```
