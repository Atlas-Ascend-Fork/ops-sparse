# SpMatGetAttribute 测试

## 测试说明

`aclsparseSpMatGetAttribute` / `aclsparseSpMatSetAttribute` 对标 cuSPARSE
`cusparseSpMatGetAttribute` / `cusparseSpMatSetAttribute`：读写稀疏矩阵描述符的
`FILL_MODE` / `DIAG_TYPE` 属性（`ACL_SPARSE_SPMAT_FILL_MODE` /
`ACL_SPARSE_SPMAT_DIAG_TYPE`，见 `include/cann_ops_sparse.h`）。Set(值) → Get
回读构成元数据 roundtrip：期望值即调用方写入值本身，判定口径为枚举位精确相等
（EXACT，零容差）。

该接口为纯 Host 侧描述符属性操作，不启动 NPU kernel、无浮点数值计算；实现在
`sparse/common/aclsparse_descr.cpp`（随公共库构建，无独立算子源目录），共享守卫
`ValidateAttributeAccess` 校验 null 描述符 / null data / `dataSize` 与对齐 /
非法 attribute 枚举。新建描述符默认值为 `fillMode=ACL_SPARSE_FILL_MODE_LOWER`、
`diagType=ACL_SPARSE_DIAG_TYPE_NON_UNIT`。

测试为 NO_SRC_DIR 纯 GTest（源码在 `arch35/`，cmake/test.cmake 的 NO_SRC_DIR
回退查找命中 arch 源），全部符号（Create*/Get/Set/Destroy）来自公共库，无需内嵌
其他算子源。用例构成（共 22 用例）：

- `arch35/spmat_get_attribute_test.csv` 驱动 12 个成功用例：3 个 mutable 创建入口
  （csr / bsr / sliced_ell）× fill{LOWER,UPPER} × diag{NON_UNIT,UNIT} 全组合，
  每例含默认值断言（Set 前 Get）、Set→Get 回读一致、二次 Get 幂等；
- 语义 TEST_F 4 条：新建描述符默认值（GA-S01）、const_csr/const_bsr 签名路径
  Get（GA-S02）、属性隔离与覆盖写 latest-wins（GA-S03）、非法值 Set 被拒后
  Get 回读旧值不变（GA-S04，与 test/spsm 的返回码断言去重，仅新增状态保持断言）；
- 异常 TEST_F 6 条（GA-E01~E06）：null 描述符 / null data / dataSize 2·8 /
  错位 1 字节指针 / 非法 attribute 枚举{2,99,-1} / Set 侧共享守卫对称性抽样，
  错误返回后断言 OUT 哨兵值不被污染。

## 目录结构

```text
test/spmat_get_attribute/
├── CMakeLists.txt                            # ops_sparse_add_gtest_tests(... NO_SRC_DIR)
└── arch35/
    ├── spmat_get_attribute_param.h           # CSV 参数结构
    ├── spmat_get_attribute_npu_wrapper.h     # 入口创建 + roundtrip 封装（RAII + DeviceBuffer）
    ├── spmat_get_attribute_test.cpp          # CSV 驱动往返 + 语义/异常路径（kEntries 注册表在此）
    └── spmat_get_attribute_test.csv          # 用例清单（3 入口 × fill × diag = 12 用例）
```

## C++测试

在已安装 CANN 开发环境的设备上（如 Ascend 950PR），从仓根执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --ops=spmat_get_attribute --soc=ascend950 --run
```

预期输出 22 项测试全部通过（12 CSV 参数化 + 4 语义 + 6 异常）：

```txt
[==========] 22 tests from 3 test suites ran.
[  PASSED  ] 22 tests.
```
