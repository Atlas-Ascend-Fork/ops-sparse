# SpMatGetIndexBase 测试

## 测试说明

`aclsparseSpMatGetIndexBase` 对标 cuSPARSE `cusparseSpMatGetIndexBase`：读取稀疏矩阵
描述符创建时传入的索引基值（`ACL_SPARSE_INDEX_BASE_ZERO` / `ACL_SPARSE_INDEX_BASE_ONE`）。
该接口为纯 Host 侧描述符访问器，不启动 NPU kernel；实现在
`sparse/common/aclsparse_descr.cpp`（随公共库构建，无独立算子源目录），接口语义见
[API 文档](../../docs/zh/api_list.md#aclsparsespmatgetindexbase)。

测试为 NO_SRC_DIR 纯 GTest（arch 无关），采用仓标准 GTest + CSV 驱动形态，覆盖：

- `spmat_get_indexbase_test.csv` 驱动 22 个往返用例：11 个带 `idxBase` 入参的 SpMat 创建入口
  （CSR/CSC/COO/BSR/BlockedELL/SlicedEll，mutable/const 变体）× base 0/1 的 Create-Get
  一致性，每例含二次 Get 幂等断言；
- `aclsparseCsrSetPointers` / `aclsparseSpMatSetValues` 后 idxBase 保持不变
  （Set 系列不触碰该字段，黑盒活性验证）；
- 异常路径：null 描述符、null 输出指针（容忍口径与 `aclsparseSpMatGetFormat` /
  `aclsparseSpVecGetIndexBase` 一致，不回写返回成功）、非 SpMat 描述符（SpVec 签名不符）；
  错误路径均断言 OUT 出参不被污染。

## 目录结构

```text
test/spmat_get_indexbase/
├── CMakeLists.txt                    # ops_sparse_add_gtest_tests(... NO_SRC_DIR)
├── README.md
├── spmat_get_indexbase_param.h       # CSV 参数结构
├── spmat_get_indexbase_test.cpp      # CSV 驱动往返 + 活性/异常路径
└── spmat_get_indexbase_test.csv      # 用例清单（11 入口 × base 0/1 = 22 用例）
```

## C++测试

在已安装 CANN 开发环境的设备上（如 Ascend 950PR），从仓根执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --ops=spmat_get_indexbase --soc=ascend950 --run
```

预期输出 26 项测试全部通过（22 CSV 参数化 + 1 活性 + 3 异常）：

```txt
[==========] 26 tests from 3 test suites ran.
[  PASSED  ] 26 tests.
```
