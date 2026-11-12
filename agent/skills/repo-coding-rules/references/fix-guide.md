# 常见违规修复方法对照表

根据违规类型找到对应修复方法。按优先级排序：致命 → 严重 → 一般。

---

## 致命级 (Fatal)

| 违规规则 | 修复方法 |
|---------|---------|
| **G.STD.15** 外部可控数据启动进程 | 使用参数化 API（如 `execvp` 而非 `system`）或白名单映射 |
| **CIP.01** 不安全 IPSI 算法 | 替换为 SM2/SM3/SM4 或 AES/SHA-256/SHA-512 等合规算法 |

---

## 严重级 (Severe)

| 违规规则 | 修复方法 |
|---------|---------|
| **G.PRE.05** 跨文件分段预处理 | 合入同一文件，或重构条件编译逻辑 |
| **G.PRE.07** 宏名与关键字冲突 | 重命名宏，添加前缀或使用宏名规范（如 `OP_`/`SPARSE_`） |
| **G.INC.05** extern "C" 包含头文件 | 把 `#include` 移到 `extern "C"` 块之外 |
| **G.INC.08** #include 之前使用 using | 把 `using` 移到所有 `#include` 之后 |
| **G.EXP.33** 自增/自减重复引用 | 拆分为独立语句，或使用临时变量 |
| **G.FUU.09** 使用 realloc() | 改用 `malloc` + `memcpy` + `free` 的安全组合，或使用安全版本函数 |
| **G.FUU.10** 使用 alloca() | 改用静态 buffer（栈上）或 `malloc`/`new`（堆上） |
| **G.FUU.12** destMax 设置错误 | 使用 `sizeof(buf)` 或明确的 size 常量 |
| **G.FUU.13** 封装安全函数 | 直接调用原安全函数，不要二次封装 |
| **G.FUU.15** 未使用安全函数库 | 查找对应安全函数版本（如 `memcpy` → `memcpy_s`） |
| **G.MEM.04** 敏感信息未清零 | 使用 `memset_s`/`explicit_bzero` 等安全版本清除 |
| **G.STD.05** 字符串缓冲区不足 | 重新评估缓冲区大小，确保包含 `\0` |
| **G.STD.07** std::string 存敏感信息 | 改用 `char[]` 配合 `memset_s` 清除 |
| **G.STD.13** 格式化字符串不匹配 | 检查 `%d`/`%lld`/`%f` 等与参数类型匹配 |
| **G.STD.16/17** 使用 exit/atexit/kill | 改为 `return` 或抛出自定义异常 |
| **G.RES.02** 内存申请未校验大小 | 加 `if (size > 0 && size <= MAX_SIZE)` 校验 |
| **G.OTH.05** 代码含公网地址 | 删除或替换为占位符，测试数据标注例外 |
| **OAT.1** 二进制文件混入 | 删除二进制文件并加入 `.gitignore` |

---

## 一般级 (Normal)

| 违规规则 | 修复方法 |
|---------|---------|
| **R5** 圈复杂度 > 20 | 提取条件块为独立函数 |
| **R6** 嵌套深度 > 5 | 提取内层循环/分支为独立函数 |
| **R7** NBNC > 50 | 按功能拆分为多个函数 |
| **R8** 除零风险 | 在除法/取模前添加被除数非零校验 |
| **R10** extern 函数声明 | 替换为头文件 `#include`（kernel 入口除外） |
| **G.EXP.26** 整型表达式溢出 | 先转型为更大类型（`static_cast<long long>()`）再运算 |
| **G.AST.03** 断言检测运行时错误 | 替换为 `if + OP_LOGE + return` 错误处理 |
| **G.FUU.11** 安全函数未检查返回值 | 添加 `if (ret != EOK)` 错误处理 |
| **G.FUU.14** 用 macro 重命名安全函数 | 直接调用原安全函数名 |
| **G.ARR.03** sizeof(ptr) 求数组大小 | 显式传递 size 参数 |
| **G.STD.18** 库函数竞争条件 | 多线程场景改用线程安全版本（如 `strtok` → `strtok_r`） |
| **G.OTH.03** rand 做安全用途 | 改用 `/dev/urandom` 或 `std::random_device` |
| **CQ.04/05** 源文件/头文件过大 | 按功能拆分模块，提取为独立文件 |
| **CQ.06** 目录文件过多 | 分层组织，按子模块建子目录 |
| **CQ.07** 重复代码块 | 提取为公共函数，放 `src/common/` |
| **CQ.08** 重复文件 | 合并为同一文件 |
| **CQ.09** 未使用的代码 | 直接删除未使用的 `#include` / 变量 / 函数 / 死代码 |
| **CQ.10** 不安全函数 | 替换为安全函数版本 |
| **CQ.11** 无理由压制告警 | 修复告警的根本原因，删除 `#pragma diagnostic ignored` |

---

## 建议级 (Advisory)

| 违规规则 | 修复方法 |
|---------|---------|
| **G.CTL.03** 死循环 / 无安全退出 | 在循环中添加明确的终止条件（counter / timeout / flag） |

---

## 典型修复模板

### 模板 A：除零防御

```cpp
// 修复 R8
uint32_t coreNum = GetAivCoreCount();
if (coreNum == 0) {
    OP_LOGE("aclsparseXxx", "GetAivCoreCount failed");
    return ACL_SPARSE_STATUS_INTERNAL_ERROR;
}
uint32_t perCore = totalN / coreNum;   // ← 此时 coreNum 必不为 0
```

### 模板 B：安全函数调用

```cpp
// 修复 CQ.10 / G.FUU.15
char buf[64];
errno_t ret = strncpy_s(buf, sizeof(buf), src, sizeof(buf) - 1);
if (ret != 0) {
    OP_LOGE("ReadConfig", "strncpy_s failed with %d", ret);
    return false;
}
```

### 模板 C：敏感信息清除

```cpp
// 修复 G.MEM.04
char password[32];
ReadPassword(password, sizeof(password));
Authenticate(password);
errno_t ret = memset_s(password, sizeof(password), 0, sizeof(password));   // ← 强制清零
if (ret != EOK) {
    OP_LOGE("ClearPassword", "memset_s failed with %d", ret);
    return;
}
```