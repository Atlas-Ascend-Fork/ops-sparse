# DestroySpMat 测试

## 测试说明

`aclsparseDestroySpMat` 对标 cuSPARSE `cusparseDestroySpMat`：销毁稀疏矩阵
描述符（签名见 `include/cann_ops_sparse.h`，`aclsparseConstSpMatDescr_t` 同签名
可传）。分支语义：null → **SUCCESS**（null 容忍，对齐 cuSPARSE）；签名不符
（错型句柄）→ INVALID_VALUE 且**不 delete**；合法 → 置零签名 + delete →
SUCCESS。判定口径为返回码位精确相等（EXACT，零容差）。

该接口为纯 Host 侧描述符销毁操作，不启动 NPU kernel、无浮点数值计算；实现在
`sparse/common/aclsparse_descr.cpp`（随公共库构建，无独立算子源目录），故走
NO_SRC_DIR 注册（源码在 `arch35/`），无需内嵌其他算子源。

测试为 NO_SRC_DIR 纯 GTest，共 14 用例：

- `arch35/destroy_spmat_test.csv` 驱动 11 个成功用例：11 个创建入口
  （coo / csr / csc / bsr / blocked_ell / sliced_ell 六个 mutable + const_csr /
  const_coo / const_csc / const_bsr / const_sliced_ell 五个 const 变体，
  blocked_ell 无 const 变体）的 Create → 手动 Destroy（guard release 语义：
  RAII 管理器释放所有权后显式调用，避免双重释放）→ SUCCESS；
- 异常 TEST_F 3 条：null 容忍（DS-E01，SUCCESS 不崩溃）/ SpVec + DnMat 错型
  句柄拒绝且随后各自经 `aclsparseDestroySpVec` / `aclsparseDestroyDnMat` 正常
  销毁（DS-E02，错误路径不 delete、无泄漏的可观测证明）/ 白盒损坏签名
  （0xDEADBEEF）→ INVALID_VALUE → 恢复签名 → 正常 Destroy SUCCESS（DS-E03，
  失败路径不 delete 的可观测证明；save→corrupt→assert→restore，先例
  spmat_get_strided_batch E7）。

注：双 destroy（合法销毁后句柄悬垂）不设用例——Destroy 置零签名后即 delete，
二次调用属 use-after-free UB，不可注入；其签名拒绝分支已由 DS-E02/E03 等价
覆盖。

## 目录结构

```text
test/destroy_spmat/
├── CMakeLists.txt                            # ops_sparse_add_gtest_tests(... NO_SRC_DIR)
└── arch35/
    ├── destroy_spmat_param.h                 # CSV 参数结构
    ├── destroy_spmat_npu_wrapper.h           # 11 入口创建 + guard release 语义 Destroy 封装
    ├── destroy_spmat_test.cpp                # CSV 驱动销毁往返 + null/错型/损坏签名异常
    └── destroy_spmat_test.csv                # 用例清单（11 成功用例）
```

## C++测试

在已安装 CANN 开发环境的设备上（如 Ascend 950PR），从仓根执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --ops=destroy_spmat --soc=ascend950 --run
```

预期输出 14 项测试全部通过（11 CSV 参数化 + 3 异常）：

```txt
[==========] 14 tests from 2 test suites ran.
[  PASSED  ] 14 tests.
```
