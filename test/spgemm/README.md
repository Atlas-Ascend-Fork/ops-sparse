# SpGEMM测试

## 测试说明

SpGEMM测试覆盖`aclsparseSpGEMM*`多阶段接口、支持的数据类型与算法、CSR输出结构、
状态机和异常参数，并通过`torch_extension`注册的PyTorch公开接口验证NPU端到端调用。算子接口、支持规格和调用
流程见[SpGEMM算子说明](../../sparse/spgemm/README.md)。

## 目录结构

```text
test/spgemm/
├── CMakeLists.txt
├── README.md
├── spgemm_golden.h
├── spgemm_param.h
├── arch22/                              # Atlas A2/A3
│   ├── spgemm_harness.h
│   ├── spgemm_ref.h
│   └── spgemm_test.cpp                  # C++功能与接口测试
├── arch35/                              # Ascend 950PR
│   ├── spgemm_npu_wrapper.h
│   ├── spgemm_test.cpp                  # C++功能与接口测试
│   ├── spgemm_test.csv                  # 仓库原有参数化回归用例
│   └── spgemm_perf.cpp                  # 独立性能测试程序
└── python/
    └── test_spgemm_torch_extension.py   # Torch Extension注册与端到端测试
```

## C++测试

在已安装CANN开发环境的Ascend 950PR设备上执行：

```bash
source /usr/local/Ascend/cann/set_env.sh

cmake -S . -B build \
  -DSOC_VERSION=ascend950 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TEST=ON \
  -DOP_LIST=spgemm
cmake --build build --target spgemm_test spgemm_perf --parallel

./build/test/spgemm/spgemm_test --gtest_color=no
```

`spgemm_test`覆盖FP16、BF16、FP32、Complex64，DEFAULT、ALG1、ALG2、ALG3，
非零beta与预置C累加，以及workspace不足、重复执行和非法阶段调用等场景。

性能程序可通过命令行指定矩阵规模、每行非零元数量、数据类型、预热次数和采样次数。例如：

```bash
./build/test/spgemm/spgemm_perf \
  --n 19717 --d 4 --dtype float32 --warmup 10 --samples 30
```

## PyTorch端到端测试

使用相互匹配的PyTorch和torch_npu环境。先构建`libops_sparse.so`主库（`--torch_extension`流程只构建wheel，不编译主库），再构建并安装`cann_ops_sparse` wheel，
最后用pytest执行：

```bash
bash build.sh --ops=spgemm --soc=<SOC型号>
bash build.sh --torch_extension --ops=spgemm
python3 -m pip install build_out/cann_ops_sparse-*.whl --force-reinstall --no-deps

OPS_SPARSE_LIB_DIR=$PWD/build_out/lib64 \
python3 -m pytest test/spgemm/python/ -v
```

C++侧wrapper由PyTorch在首次NPU调用时JIT编译，需要链接`libops_sparse.so`。
`OPS_SPARSE_LIB_DIR`用于指向构建产物目录，未设置时回退到`$ASCEND_HOME_PATH/lib64`
下已安装的run包。

该测试覆盖接口注册（含“导入不触发JIT编译”）、CSR/COO公开路径、int32/int64索引转换、
FP16/BF16/FP32/Complex64、非零beta的`torch.sparse.addmm`、结构边界与非法输入、
多设备与非默认stream、wheel打包内容以及`torch_npu.sparse`命名空间。
