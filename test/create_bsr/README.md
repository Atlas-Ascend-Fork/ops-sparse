# CreateBsr 测试

## 测试说明

`aclsparseCreateBsr` / `aclsparseCreateConstBsr` 对标 cuSPARSE
`cusparseCreateBsr`：创建 BSR 格式稀疏矩阵描述符，接口语义见
[API 文档](../../docs/zh/api_list.md#aclsparsecreatebsr)。验证方式为元数据
roundtrip——期望值按块展开公式计算（EXACT，零容差）：`GetSize` 回读
rows=blockRows×bs、cols=blockCols×bs、nnz=blockNnz（非零块数），另经
`aclsparseSpMatGetFormat`（== `ACL_SPARSE_FORMAT_BSR`）、
`aclsparseSpMatGetIndexBase`、`aclsparseSpMatGetStridedBatch`（新建描述符
默认 1）核对。

该接口为纯 Host 侧描述符创建，不启动 NPU kernel、无浮点数值计算；实现在
`sparse/common/aclsparse_descr.cpp`（随公共库构建，无独立算子源目录），故走
NO_SRC_DIR 注册（源码在 `arch35/`，cmake/test.cmake 回退查找命中 arch 源）。
仓内无 BsrGet 访问器，块几何字段（blockRows/blockCols/blockNnz/rowBlockSize/
colBlockSize/bsrValuesCount/bsrOrder）经内部结构白盒核对（先例
test/csr_set_strided_batch）；校验链要求块大小方形 ∈ {2,4,8,16,32,64,128}、
索引仅 32I、valueType ∈ {FP16,BF16,FLOAT,COMPLEX64}、offsets 恒非空
（含 nnzb=0）、INT32 几何上界，Create 仅地址算术不解引用指针。

测试为 NO_SRC_DIR 纯 GTest，共 31 用例：

- `arch35/create_bsr_test.csv` 驱动 14 个成功用例：四种值类型全覆盖（FP32/
  FP16/BF16/COMPLEX64）、ROW+COL、块大小 2~128（含最大 128）、0 维 (0,0,0) +
  offsets 单端点、nnzb=0+非零几何（colInd/values 可 null）、tall/wide/1×1 块
  网格/rect、idxBase 0/1、const 变体；
- 语义 TEST_F 4 条：INT32 几何边界内极大值 (blockRows=INT32_MAX-1,
  blockCols=INT32_MAX, nnzb=0) 创建成功且 GetSize 块展开回读（BSR-S01，指针
  不解引用）；batch roundtrip——`aclsparseBsrSetStridedBatch(4,3,2,8)` 非广播
  最小步长 + GetStridedBatch 回读 + 白盒三 stride（BSR-S02，与
  spmat_get_strided_batch E8 的广播 (0,0,8) 场景区分）；覆盖写 latest-wins
  （BSR-S03）；单 batch 关闭批量语义（BSR-S04）；
- 异常 TEST_F 13 条（BSR-E01~E13）：OUT 句柄 null（mutable+const）、负块维、
  非正/非方/不支持的块大小、64I 索引（NOT_SUPPORTED）、非法 idxBase/order、
  DOUBLE/INT32 值类型（NOT_SUPPORTED）、INT32 几何越界四断言、零维几何非零块、
  null 存储指针三断言（offsets 恒非空 / nnzb>0 时 colInd/values 非空）、
  `UINTPTR_MAX` 高位指针注入触发 CheckedAddressExtent 地址回绕三断言
  （同 csr/coo_set_strided_batch E9 的受检溢出注入思想），失败后断言 OUT
  保持 nullptr。

注：TEST_P / BSR-S02~S04 断言 `aclsparseSpMatGetStridedBatch`（实现在
`sparse/spmat_get_strided_batch/arch35/`，非公共库），`--ops` 过滤构建时该符号
不在 libops_sparse 中，CMakeLists 将其源码直接编入测试目标（先例
test/csr_set_strided_batch）。

## 目录结构

```text
test/create_bsr/
├── CMakeLists.txt                            # ops_sparse_add_gtest_tests(... NO_SRC_DIR) + 内嵌 getter 源
└── arch35/
    ├── create_bsr_param.h                    # CSV 参数结构
    ├── create_bsr_npu_wrapper.h              # unique_ptr+Destroy RAII + 白盒块几何回读 + dtype/order 解析
    ├── create_bsr_test.cpp                   # CSV 驱动往返 + batch roundtrip 语义/异常路径
    └── create_bsr_test.csv                   # 用例清单（14 成功用例）
```

## C++测试

在已安装 CANN 开发环境的设备上（如 Ascend 950PR），从仓根执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --ops=create_bsr --soc=ascend950 --run
```

预期输出 31 项测试全部通过（14 CSV 参数化 + 4 语义 + 13 异常）：

```txt
[==========] 31 tests from 3 test suites ran.
[  PASSED  ] 31 tests.
```
