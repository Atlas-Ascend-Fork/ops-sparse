# CsrSetPointers 测试

## 测试说明

`aclsparseCsrSetPointers` 对标 cuSPARSE `cusparseCsrSetPointers`：重绑 CSR 稀疏
矩阵描述符的 rowOffsets / colIndices / values 三指针（签名见
`include/cann_ops_sparse.h`）。期望值即调用方传入指针本身（roundtrip oracle，
EXACT 零容差）：Set(P1) → `aclsparseCsrGet` 回读十字段（指针 == P1）+ 白盒
`structureGeneration+1`（经共享 `SetCompressedPointers` 的 structureChanged 分支，
descr.cpp:1006-1008；与 `aclsparseCscSetPointers` 共用代码路径但入口/格式守卫
不同）。

该接口为纯 Host 侧描述符 setter，不启动 NPU kernel、无浮点数值计算；实现在
`sparse/common/aclsparse_descr.cpp`（随公共库构建，无独立算子源目录），故走
NO_SRC_DIR 注册（源码在 `arch35/`）。守卫：null / 签名不符 / 非 CSR 格式 /
offsets==null 且 rows>0 且 cols>0 / nnz>0 且 indices 或 values null →
INVALID_VALUE（零维矩阵允许 offsets=null、nnz=0 允许 indices/values null，合法
补集成对覆盖）。写入语义分态：structureChanged（ptrs 或 idxs 变化）→
`AclsparseMarkSpMatStructureChanged`；任一指针变化 → `ResetSpmvActivePlan`；
ptrs 变化 → DenseToSparse 分析失效；尾部无条件清空 activeBuffer/
activeSpmvBuffer + 滚动 spmmWorkspaceOwnerId + spmvCubeMaxRowLengthValid=false。

测试为 NO_SRC_DIR 纯 GTest，共 19 用例：

- `arch35/csr_set_pointers_test.csv` 驱动 10 个成功用例：基础 / idxBase 1 /
  I64 索引 / fp16 / nnz=0 idx+val null（合法补集）/ rows=0、cols=0 offsets null
  （零维合法补集）/ tall / wide / large，每例含 CsrGet 十字段 roundtrip +
  白盒 generation+1；
- 语义 TEST_F 4 条（变化五值分态白盒，全变已由 CSV 覆盖）：offsets-only
  （generation+1 + DenseToSparse 失效，ptrs 轴隔离，CSP-S01）/ idxs-only
  （generation+1 + DenseToSparse 保持，CSP-S02）/ values-only（generation 不变 +
  Reset 走，CSP-S03）/ 全同（generation 不变 + Reset 不走 + 尾部四效应无条件
  全断言：activeBuffer/activeSpmvBuffer 清空 + ownerId 滚动 +
  maxRowLengthValid=false，CSP-S04）；
- 异常 TEST_F 5 条（CSP-E01~E05）：null descr / DnMat 错型句柄 / 非 CSR 格式
  （COO）/ 非零维 offsets null / nnz>0 idx·val null，失败后断言 CsrGet 回读
  仍为旧指针组 P0（失败不改状态）。

注：roundtrip 载体 `aclsparseCsrGet` 实现在 `sparse/csr_get/arch35/`（非公共
库），`--ops` 过滤构建时该符号不在 libops_sparse 中，CMakeLists 将其源码直接编入
测试目标（先例 test/csr_set_strided_batch）；op_common 为传递依赖
（csr_get_host.cpp → aclsparse_descr_guard.h → log/log.h）。

## 目录结构

```text
test/csr_set_pointers/
├── CMakeLists.txt                            # ops_sparse_add_gtest_tests(... NO_SRC_DIR) + 内嵌 CsrGet 载体源
└── arch35/
    ├── csr_set_pointers_param.h              # CSV 参数结构
    ├── csr_set_pointers_npu_wrapper.h        # CreateCsr(P0) -> SetPointers(P1) -> CsrGet 回读 + 白盒代际
    ├── csr_set_pointers_test.cpp             # CSV 驱动往返 + 变化五值分态白盒 + 异常路径
    └── csr_set_pointers_test.csv             # 用例清单（10 成功用例）
```

## C++测试

在已安装 CANN 开发环境的设备上（如 Ascend 950PR），从仓根执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --ops=csr_set_pointers --soc=ascend950 --run
```

预期输出 19 项测试全部通过（10 CSV 参数化 + 4 语义 + 5 异常）：

```txt
[==========] 19 tests from 3 test suites ran.
[  PASSED  ] 19 tests.
```
