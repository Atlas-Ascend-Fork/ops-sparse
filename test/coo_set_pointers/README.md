# CooSetPointers 测试

## 测试说明

`aclsparseCooSetPointers` 对标 cuSPARSE `cusparseCooSetPointers`：重绑 COO 稀疏
矩阵描述符的 rowIndices / colIndices / values 三指针（签名见
`include/cann_ops_sparse.h`）。期望值即调用方传入指针本身（roundtrip oracle，
EXACT 零容差）：Set(P1) → `aclsparseCooGet` 回读九字段（指针 == P1），并经内部
结构白盒核对复用字段同步（`ptrs==rowInds`、`idxs==colInds`——CooGet 读回的正是
复用字段）与 `structureGeneration` 不变（COO 路径不调用
`AclsparseMarkSpMatStructureChanged`，与 CSR 行为差异锁定）。

该接口为纯 Host 侧描述符 setter，不启动 NPU kernel、无浮点数值计算；实现在
`sparse/common/aclsparse_descr.cpp`（随公共库构建，无独立算子源目录），故走
NO_SRC_DIR 注册（源码在 `arch35/`）。守卫：null / 签名不符 / 非 COO 格式 /
nnz>0 且任一指针 null → INVALID_VALUE（nnz=0 时允许全 null，合法补集成对覆盖）；
指针任一变化触发 `ResetSpmvActivePlan`，全同不触发。

测试为 NO_SRC_DIR 纯 GTest，共 15 用例：

- `arch35/coo_set_pointers_test.csv` 驱动 9 个成功用例：基础 / idxBase 1 / I64
  索引 / fp16 / nnz=0 全 null（合法补集）/ nnz=0 全有效 / tall / wide / large，
  每例含 CooGet 九字段 roundtrip + 白盒 ptrs/idxs 复用同步 + generation 不变；
- 语义 TEST_F 2 条：指针变化触发 ResetSpmvActivePlan（activeBuffer/
  activeSpmvBuffer/activeRowOffsets 三哨兵清空，COP-S01）/ 同参二次调用不触发
  （哨兵保持，COP-S02），双向断言 generation 不变；
- 异常 TEST_F 4 条（COP-E01~E04）：null descr / DnMat 错型句柄 / 非 COO 格式
  （CSR）/ nnz>0 任一指针 null（三断言），失败后断言 CooGet 回读仍为旧指针组
  P0（失败不改状态）。

注：roundtrip 载体 `aclsparseCooGet` 实现在 `sparse/coo_get/arch35/`（非公共
库），`--ops` 过滤构建时该符号不在 libops_sparse 中，CMakeLists 将其源码直接编入
测试目标（先例 test/csr_set_strided_batch 内嵌 getter 源）。

## 目录结构

```text
test/coo_set_pointers/
├── CMakeLists.txt                            # ops_sparse_add_gtest_tests(... NO_SRC_DIR) + 内嵌 CooGet 载体源
└── arch35/
    ├── coo_set_pointers_param.h              # CSV 参数结构
    ├── coo_set_pointers_npu_wrapper.h        # CreateCoo(P0) -> SetPointers(P1) -> CooGet 回读 + 白盒
    ├── coo_set_pointers_test.cpp             # CSV 驱动往返 + Reset 双态白盒 + 异常路径
    └── coo_set_pointers_test.csv             # 用例清单（9 成功用例）
```

## C++测试

在已安装 CANN 开发环境的设备上（如 Ascend 950PR），从仓根执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --ops=coo_set_pointers --soc=ascend950 --run
```

预期输出 15 项测试全部通过（9 CSV 参数化 + 2 语义 + 4 异常）：

```txt
[==========] 15 tests from 3 test suites ran.
[  PASSED  ] 15 tests.
```
