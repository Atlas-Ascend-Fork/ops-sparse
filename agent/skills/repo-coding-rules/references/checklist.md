# 代码检查流程 (8 步自查)

在 2.x.1.A 算子初稿完成后、2.x.2 联调前、2.x.3 验收前、CP2.x 打点前，**必须**按以下 8 步流程逐条自查。

---

## 步骤 1：文件级检查

- [ ] 无二进制文件混入（OAT.1）
- [ ] 所有源码文件含许可证头（R9）
- [ ] 无公网地址（G.OTH.05）
- [ ] 无重复文件（CQ.08）
- [ ] 目录文件数在合理范围（CQ.06）

**快速检查命令**：
```bash
# 检查二进制文件
find . \( -name "*.o" -o -name "*.so" -o -name "*.a" -o -name "*.exe" \) -type f
# 检查无许可证头
grep -L "Copyright (c)" $(find . -name "*.cpp" -o -name "*.h" -o -name "*.c")
# 检查公网 URL（常见域名）
grep -rE "(gitcode|gitee|github|google|huggingface)\.(com|cn)" .
```

---

## 步骤 2：头文件检查

- [ ] 无 `extern "C"` 包含头文件（G.INC.05）
- [ ] `#include` 在 `using` 之前（G.INC.08）
- [ ] 无 extern 函数/变量声明（R10）
- [ ] 头文件大小在合理范围（CQ.05）

**快速检查命令**：
```bash
# 检查 include 顺序（using 应在 include 之后）
for f in $(find . -name "*.cpp" -o -name "*.h"); do
  awk '/^using / {u=NR; next} /^#include / && u {print FILENAME ":" NR ": using before include"; exit}' "$f"
done

# 检查 extern "C" 中 #include
grep -B5 "^#include" $(find . -name "*.cpp" -o -name "*.h") | grep -B5 'extern "C"'
```

---

## 步骤 3：函数级检查

- [ ] 圈复杂度 ≤ 20（R5）
- [ ] 嵌套深度 ≤ 5（R6）
- [ ] NBNC ≤ 50（R7）
- [ ] 函数功能单一，无重复逻辑（CQ.07）
- [ ] 循环有安全退出条件（G.CTL.03）
- [ ] 断言不用于运行时错误检测（G.AST.03）

**圈复杂度手动估算**：1（基础）+ if/else if/for/while/case/&&/|| 数量

**嵌套深度统计**：用 IDE 的 "Show Indent Guide" 或 `awk` 数 `{` 层级。

---

## 步骤 4：表达式与运算检查

- [ ] 除法/取模前校验被除数非零（R8）
- [ ] 整型表达式比较前已转型为更大类型（G.EXP.26）
- [ ] 自增/自减表达式无重复引用（G.EXP.33）
- [ ] 无 `sizeof(ptr)` 代替 `sizeof(array)`（G.ARR.03）

**grep 快速扫描**：
```bash
grep -nE "[^a-zA-Z_](/|%)[ ]*[a-zA-Z_]+" $(find . -name "*.cpp") | grep -v "_s"
```

---

## 步骤 5：安全函数检查

- [ ] 无不安全函数调用（realloc/alloca）（G.FUU.09/10）
- [ ] 安全函数返回值已检查（G.FUU.11）
- [ ] `destMax` 参数正确设置（G.FUU.12）
- [ ] 未封装/重命名安全函数（G.FUU.13/14）
- [ ] 使用华为安全函数库（G.FUU.15）

**快速检查命令**：
```bash
# 检查是否使用了 realloc / alloca
grep -rnE "\brealloc\b|\balloca\b" $(find . -name "*.cpp" -o -name "*.c")
```

---

## 步骤 6：内存与资源检查

- [ ] 内存申请前校验大小合法性（G.RES.02）
- [ ] 敏感信息使用后已清零（G.MEM.04）
- [ ] 字符串缓冲区足够容纳数据和结束符（G.STD.05）
- [ ] 不使用 `std::string` 存敏感信息（G.STD.07）

---

## 步骤 7：标准库与安全检查

- [ ] 格式化字符串与参数类型匹配（G.STD.13）
- [ ] 不使用外部可控数据启动进程（G.STD.15）
- [ ] 不使用 exit/atexit/kill 等危险函数（G.STD.16/17）
- [ ] 多线程场景使用线程安全版本（G.STD.18）
- [ ] 不使用 `rand` 做安全用途（G.OTH.03）
- [ ] 不使用不安全 IPSI 算法（CIP.01）

**快速检查命令**：
```bash
grep -rEn "\b(exit|_exit|abort|atexit|kill|TerminateProcess)\b" $(find . -name "*.cpp")
```

---

## 步骤 8：冗余与告警检查

- [ ] 无未使用的 include/变量/函数（CQ.09）
- [ ] 无死代码（CQ.09）
- [ ] 无重复代码块（CQ.07）
- [ ] 编译告警已修复而非抑制（CQ.11）

**OAT 自动化扫描**（推荐，自动检查许可证头和文件类型）：
```bash
sh scripts/oat_check.sh <变更文件列表>
```

---

## 检查结果记录

每完成一处检查记录：
- ✅ **PASS**：无违规
- ⚠️ **WARNING**：有建议级规则违反（可记录后继续）
- 🔴 **FAIL**：有严重/致命级规则违反（**必须先修复再继续**）

所有 FAIL 项修复后，**重新执行步骤 1-8 整轮流程**，直至全 PASS。