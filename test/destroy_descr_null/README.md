# 描述符 Destroy 系列 null 语义专测

## 测试内容

覆盖 `aclsparseDestroySpMat` / `aclsparseDestroySpVec` / `aclsparseDestroyDnVec` /
`aclsparseDestroyDnMat` 四个描述符销毁接口的 null 语义契约（issue #377）：

- **DN-E01**：`Destroy*(nullptr)` 一律返回 `ACL_SPARSE_STATUS_SUCCESS`（null 容忍，
  对齐 cuSPARSE `cusparseDestroy*`、仓内 `aclsparseSpGEMMDestroyDescr` 与
  `test/destroy_spmat` DS-E01 断言的既有契约）。
- **DN-E02**：正常创建后 `Destroy*` 返回 SUCCESS——确认 null 分支调整不影响非 null 路径。
- **DN-E03**：SpMat 句柄伪装为 SpVec/DnVec/DnMat 时各 Destroy 返回
  `ACL_SPARSE_STATUS_INVALID_VALUE` 且不 `delete`；原对象随后经 `DestroySpMat`
  正常释放（错误路径无泄漏）。

## 测试性质

纯 Host 侧描述符操作（NO_SRC_DIR 布局，无 arch 子目录），不启动 NPU kernel；
缓冲仅占位（描述符只借用指针、不解引用内容）。仅依赖 common 库符号。

## 编译与运行

```bash
bash build.sh --soc=ascend950 --ops=destroy_descr_null
cd build/test/destroy_descr_null && ./destroy_descr_null_test
```
