# Gather测试

## 测试说明

Gather测试覆盖`aclsparseGather`低层C API与`torch.index_select`公开入口，验证四种任务要求的
value dtype、I32索引、base 0/1、动态规模、重复/乱序索引、尾块、`nnz=0/1`、异常参数、
输入只读、逐位一致性和非默认stream。接口语义、支持范围和限制见
[Gather算子说明](../../sparse/gather/README.md)与
[Torch Extension接口说明](../../torch_extension/cann_ops_sparse/docs/zh/gather.md)。

## 目录结构

```text
test/gather/
├── CMakeLists.txt
├── README.md
├── gather_golden.h
├── gather_param.h
├── arch35/
│   ├── gather_npu_wrapper.h
│   ├── gather_test.cpp
│   └── gather_test.csv
└── python/
    └── test_gather_torch_extension.py
```

## C++测试

在已安装CANN开发环境的 Ascend 950PR系列产品 设备上，从仓根执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
CMAKE_BUILD_TYPE=Release bash build.sh --ops=gather --soc=ascend950 --run
```

参数化用例覆盖FP16、BF16、FP32、Complex64和既有FP64兼容路径，同时覆盖I32/I64及
base 0/1。新增契约测试专项验证任务要求的四种dtype逐位一致、INF/NAN payload、输入只读、
重复执行一致、空向量以及非法描述符、dtype、shape、base、index type、alias和device指针。

## PyTorch端到端测试

使用相互匹配的PyTorch 2.7+与torch_npu 26.0.0+环境。先构建主库和最小wheel，再从安装后的
包执行测试：

```bash
CMAKE_BUILD_TYPE=Release bash build.sh --ops=gather --soc=ascend950
bash build.sh --torch_extension --ops=gather
python3 -m pip install --force-reinstall --no-deps build_out/cann_ops_sparse-*.whl

OPS_SPARSE_LIB_DIR=$PWD/build \
TORCH_EXTENSIONS_DIR=$PWD/.torch_extensions \
python3 -m pytest -q test/gather/python/test_gather_torch_extension.py
```

C++ wrapper在首次NPU调用时JIT编译，需要链接`libops_sparse.so`。若主库位于
`build_out/lib64`，相应地将`OPS_SPARSE_LIB_DIR`设为该目录。

端到端测试覆盖注册与延迟JIT、四种dtype逐位精度、公开`aten::index_select`分发、动态边界、
乱序/重复索引（包括长度超过输入的重复索引及跨分段尾块）、空输出、输入只读、alias、
非默认stream的分段调用及不支持的dtype、layout、shape、
stride、index type、dim与autograd，并验证正、负越界索引在Gather Kernel下发前被拒绝，
以及索引原地修改或输入长度改变后会使已验证范围缓存失效。
测试只接受NPU输出，不存在CPU结果回填路径。
