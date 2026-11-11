# BsrSetStridedBatch 测试

## 测试说明

`aclsparseBsrSetStridedBatch` 对标 cuSPARSE `cusparseBsrSetStridedBatch`：设置 BSR
描述符的 strided-batch 属性（接口语义见
[API 文档](../../docs/zh/api_list.md#aclsparsebsrsetstridedbatch)）。将批量矩阵个数
与相邻 batch 间三个数组的地址间隔写入描述符；三 stride 按元素个数计（span 语义：
`offsetsSpan=blockRows+1`、`columnsSpan=blockNnz`、`valuesSpan=blockNnz×bs²`），
offsets / columns 的 0 表示 batch 间广播结构，`batchCount>1` 且块值非空时 values
不允许广播。Set(值) → `aclsparseSpMatGetStridedBatch` 回读构成元数据 roundtrip：
期望值即调用方写入值本身，判定口径为 int64 / 返回码位精确相等（EXACT，零容差）。

该接口为纯 Host 侧描述符 setter，不启动 NPU kernel、无浮点数值计算；实现在
`sparse/common/aclsparse_descr.cpp`（随公共库构建，无独立算子源目录），故走
NO_SRC_DIR 注册（源码在 `arch35/`，cmake/test.cmake 回退查找命中 arch 源）。校验链：
入口守卫（null / 签名不符 / 非 BSR 格式）→ `ValidateBsrBatchLayout`（batchCount
∈ [1,65535]、stride 非负、`blockRows+1` 受检加法、offsets/columns 重叠步长、
values 广播与重叠禁令）→ `ValidateBsrBatchStorage`（`CheckedStridedExtent` 三分量
溢出、byte extent 受检乘、`CheckedAddressExtent` 地址回绕）。公共 getter 只回读
batchCount，三个 stride 经内部结构白盒核对（先例 test/csr_set_strided_batch）。

测试为 NO_SRC_DIR 纯 GTest，共 23 用例：

- `arch35/bsr_set_strided_batch_test.csv` 驱动 10 个成功用例：batchCount
  1/2/4/8/65535（上边界）、最小非广播（stride==span）、间隔排布（stride>span）、
  offsets+columns 广播 + values 非广播、nnzb=0 空值矩阵的 bc>1 全 0 广播与部分
  非广播、fp16/bf16、bs=8/128 大 valuesSpan；每例含 Get 默认值断言（Set 前
  batchCount==1）、Set/Get 回读 + 白盒 batch 四字段逐字段断言（防 int64 截断）；
- 语义 TEST_F 2 条：覆盖写 latest-wins 白盒三 stride 逐字段（BSRB-S01，升级
  create_bsr BSR-S03 的仅 Get 断言）；changed 条件化缓存失效白盒（BSRB-S02：
  changed=true 时 structureGeneration+1、activeBuffer/activeSpmvBuffer 清空、
  sddmmPrepared.prepared 置 false，changed=false 时全部保持）；
- 异常 TEST_F 11 条（BSRB-E01~E11）：null / DnMat 错型句柄 / 非 BSR 格式
  （COO/CSR）/ batchCount 0·负·65536 越界 / 负 stride / 重叠步长 / 空 values
  广播禁令 / `blockRows=INT64_MAX` 白盒构造跨度溢出（save→corrupt→assert→
  restore，先例 spmat_get_strided_batch E7）/ 受检跨度溢出三分量
  （2^50 乘法、INT64_MAX+span 加法，先例 csr E9）/ byte extent 溢出三调用点
  （2^61×4B）/ `UINTPTR_MAX` 高位指针注入（ptrs/idxs/values 逐项，其余指针保持
  有效设备指针），失败后均断言描述符状态不被污染（GetStridedBatch 回读仍为
  既有值）。

注：读回验证载体 `aclsparseSpMatGetStridedBatch` 实现在
`sparse/spmat_get_strided_batch/arch35/`（非公共库），`--ops` 过滤构建时该符号不在
libops_sparse 中，CMakeLists 将其源码直接编入测试目标（先例
test/csr_set_strided_batch / test/coo_set_strided_batch）。

## 目录结构

```text
test/bsr_set_strided_batch/
├── CMakeLists.txt                            # ops_sparse_add_gtest_tests(... NO_SRC_DIR) + 内嵌 getter 源
└── arch35/
    ├── bsr_set_strided_batch_param.h         # CSV 参数结构
    ├── bsr_set_strided_batch_npu_wrapper.h   # CreateBsr -> Get(默认) -> Set -> Get 回读 + 白盒三 stride
    ├── bsr_set_strided_batch_test.cpp        # CSV 驱动往返 + 覆盖写/缓存失效语义 + 异常路径（白盒构造+恢复）
    └── bsr_set_strided_batch_test.csv        # 用例清单（10 成功用例）
```

## C++测试

在已安装 CANN 开发环境的设备上（如 Ascend 950PR），从仓根执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --ops=bsr_set_strided_batch --soc=ascend950 --run
```

预期输出 23 项测试全部通过（10 CSV 参数化 + 2 语义 + 11 异常）：

```txt
[==========] 23 tests from 3 test suites ran.
[  PASSED  ] 23 tests.
```
