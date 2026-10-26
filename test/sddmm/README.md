# SDDMM 测试

`arch22/` 包含 Atlas A2/A3 的 C++ 测试和 CSV 用例，`arch35/` 包含 A5 的对应测试。
共享参数和 CPU 参考计算位于本目录，Torch 回归测试位于 `python/`。

使用目标芯片和已安装的 CANN 配置 `-DOP_LIST=sddmm -DBUILD_TEST=ON`，构建并运行 `sddmm_test`。
`SddmmContract.*` 验证三阶段公共接口，包括空维度、零归约、布局、标量指针模式和错误输入。

Python 测试需安装匹配的 PyTorch、torch_npu 和 cann_ops_sparse。
将 `OPS_SPARSE_LIB_DIR` 指向本次构建的算子库目录，然后执行：

```sh
python3 -m pytest test/sddmm/python -v
```

测试覆盖 CSR/BSR、批次广播、复数标量、输入视图、重复执行、输出所有权和转换溢出。
高动态范围复数用例分别比较实部和虚部，避免参考误差计算溢出。
CPU 结果仅用于精度对照。
