# CreateSlicedEll 测试

## 测试说明

`aclsparseCreateSlicedEll` / `aclsparseCreateConstSlicedEll` 对标 cuSPARSE
`cusparseCreateSlicedEll`：创建 SELL（Sliced-Ellpack）格式稀疏矩阵描述符，
签名见 `include/cann_ops_sparse.h`。验证方式为元数据 roundtrip——期望值即
Create 调用方传入的参数本身（EXACT，零容差）：经通用访问器
`aclsparseSpMatGetFormat`（== `ACL_SPARSE_FORMAT_SLICED_ELL`）、
`aclsparseSpMatGetSize`（rows/cols/nnz 回读）、`aclsparseSpMatGetIndexBase`、
`aclsparseSpMatGetStridedBatch`（新建描述符默认 1）回读核对。

该接口为纯 Host 侧描述符创建，不启动 NPU kernel、无浮点数值计算；实现在
`sparse/common/aclsparse_descr.cpp`（随公共库构建，无独立算子源目录），故走
NO_SRC_DIR 注册（源码在 `arch35/`，cmake/test.cmake 回退查找命中 arch 源）。
仓内无 SlicedEllGet 访问器，`sliceNnz` / `numSlices` / 索引与值类型经内部结构
白盒核对（先例 test/csr_set_strided_batch 的 stride 字段）；实现不校验数据指针
（延迟绑定语义），测试侧建设备缓冲仅作指针占位、内容不被访问。

测试为 NO_SRC_DIR 纯 GTest，共 20 用例：

- `arch35/create_sliced_ell_test.csv` 驱动 12 个成功用例：基础 4×4 / tall 32×4 /
  wide 4×64 / 大 shape 1024 / 0 维 (0,0,0,0,0) / nnz=0+null 指针 / 单 slice /
  sliceNnz>nnz（无交叉校验行为锁定）/ I64 索引 / fp16 值类型 / const 变体；
- 语义 TEST_F 1 条（SE-S01）：nnz>0 且三个数组指针全 null 创建成功（延迟绑定）；
- 异常 TEST_F 7 条（SE-E01~E07）：OUT 句柄 null（mutable+const）、rows/cols/nnz/
  sliceNnz/numSlices 负值、非法 sellIdxType，失败后断言 OUT 保持 nullptr。

注：TEST_P 断言 `aclsparseSpMatGetStridedBatch`（实现在
`sparse/spmat_get_strided_batch/arch35/`，非公共库），`--ops` 过滤构建时该符号
不在 libops_sparse 中，CMakeLists 将其源码直接编入测试目标（先例
test/csr_set_strided_batch）。

## 目录结构

```text
test/create_sliced_ell/
├── CMakeLists.txt                            # ops_sparse_add_gtest_tests(... NO_SRC_DIR) + 内嵌 getter 源
└── arch35/
    ├── create_sliced_ell_param.h             # CSV 参数结构
    ├── create_sliced_ell_npu_wrapper.h       # SpMatManager 工厂封装 + 白盒字段回读
    ├── create_sliced_ell_test.cpp            # CSV 驱动往返 + 语义/异常路径
    └── create_sliced_ell_test.csv            # 用例清单（12 成功用例）
```

## C++测试

在已安装 CANN 开发环境的设备上（如 Ascend 950PR），从仓根执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --ops=create_sliced_ell --soc=ascend950 --run
```

预期输出 20 项测试全部通过（12 CSV 参数化 + 1 语义 + 7 异常）：

```txt
[==========] 20 tests from 3 test suites ran.
[  PASSED  ] 20 tests.
```
