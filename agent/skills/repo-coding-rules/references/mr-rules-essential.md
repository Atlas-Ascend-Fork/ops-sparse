# MR 安全编码规则 - 严重/致命级（提交前必查）

本文件汇总所有**严重级（Severe）**和**致命级（Fatal）**规则。MR 提交前必须确保无违规。

---

## 1. 预处理器规则 (G.PRE)

### G.PRE.05 [严重] 预处理分段不能跨文件
`#else` / `#elif` / `#endif` 必须与对应的 `#if` / `#ifdef` / `#ifndef` 在同一文件中。

**错误**：分散在多个文件中的条件编译
```cpp
// file1.h
#ifdef ENABLE_FEATURE
    void foo();

// file2.h
#else
    void bar();
#endif
```

**正确**：
```cpp
#ifdef ENABLE_FEATURE
    void foo();
#else
    void bar();
#endif
```

### G.PRE.07 [严重] 宏名不能与关键字相同
使用 `#define` 时避开 C/C++ 关键字。

**错误**：
```cpp
#define max 100
#define class MyClass
```

**正确**：
```cpp
#define MAX_SIZE 100
#define CLASS_NAME MyClass
```

---

## 2. 头文件规则 (G.INC)

### G.INC.05 [严重] extern "C" 块内禁止包含头文件
`extern "C"` 块内只能有函数/变量声明，不能有 `#include`。

**错误**：
```cpp
extern "C" {
    #include "helper.h"      // ← 违规
    void my_function();
}
```

**正确**：
```cpp
#include "helper.h"
extern "C" {
    void my_function();
}
```

### G.INC.08 [严重] using 必须在 include 之后
`#include` 必须位于所有 `using` 之前。

**错误**：
```cpp
using namespace std;
#include <vector>
#include <memory>
```

**正确**：
```cpp
#include <vector>
#include <memory>
using namespace std;
```

---

## 3. 表达式规则 (G.EXP)

### G.EXP.33 [严重] 自增/自减表达式禁止重复引用
含 `++` / `--` 的表达式中禁止在相同表达式中再次引用同一变量。

**错误**：
```cpp
a = b[i++] + b[i];    // ← b[i] 重复引用
x = y++ * y;          // ← y 重复引用
```

**正确**：
```cpp
a = b[i++] + b[i];    // 拆分为两条语句后不再违规，或者用临时变量
int idx = i;
i++;
a = b[idx] + b[i];
```

---

## 4. 安全函数规则 (G.FUU)

### G.FUU.09 [严重] 禁止使用 realloc()
使用安全内存管理替代。

### G.FUU.10 [严重] 禁止使用 alloca() 栈上动态分配
栈上动态分配易导致栈溢出，应使用堆上分配或静态缓冲。

### G.FUU.12 [严重] destMax 必须正确设置
安全函数的 `destMax` 参数必须反映真实缓冲区大小，避免写入越界。

**错误**：
```cpp
char buf[64];
strcpy_s(buf, 1024, src);   // ← destMax 大于实际 buf 大小
```

**正确**：
```cpp
char buf[64];
strcpy_s(buf, sizeof(buf), src);
```

### G.FUU.13 [严重] 禁止封装安全函数
不要对安全函数再封装一层。

**错误**：
```cpp
// 封装 strcpy_s 为 custom_strncpy
int custom_strncpy(char* dst, size_t size, const char* src) {
    return strcpy_s(dst, size, src);
}
```

**正确**：直接调用 `strcpy_s`。

### G.FUU.15 [严重] 只能使用安全函数库中的函数
禁止使用不安全替代品，必须使用华为安全函数库。

---

## 5. 内存规则 (G.MEM)

### G.MEM.04 [严重] 敏感信息使用后清零
内存中敏感信息（密码、密钥、token）使用完毕后立即用安全 `memset` 清除。

**错误**：
```cpp
char password[32];
ReadPassword(password);
Authenticate(password);
// ← 忘记清除
return;
```

**正确**：
```cpp
char password[32];
ReadPassword(password);
Authenticate(password);
errno_t ret = memset_s(password, sizeof(password), 0, sizeof(password));   // ← 安全清除
if (ret != EOK) {
    return;   // 清除失败：不能假定密码已擦除，走失败路径
}
return;
```

---

## 6. 标准库规则 (G.STD)

### G.STD.05 [严重] 字符串缓冲区必须足够
确保缓冲区能容纳数据和结束符 `\0`。

### G.STD.07 [严重] 禁止用 std::string 存储敏感信息
敏感信息应使用安全内存管理而非 `std::string`。

### G.STD.13 [严重] 格式化输入/输出字符串匹配
调用 `printf` / `scanf` 等函数时，格式字符串必须与参数类型匹配。

### G.STD.15 [致命] 禁止外部可控数据启动进程
禁止将外部可控数据作为 `system` / `popen` / `exec` 等函数的参数，防止命令注入。

**错误**：
```cpp
system(user_input_cmd);   // ← 外部可控数据，命令注入风险
```

**正确**：使用参数化 API 或白名单。

### G.STD.16 [严重] 禁用程序退出函数
禁止使用 `exit` / `_exit` / `abort` / `atexit`。

### G.STD.17 [严重] 禁用 kill / TerminateProcess
禁止直接终止其他进程，防止资源泄漏。

---

## 7. 资源规则 (G.RES)

### G.RES.02 [严重] 内存申请前校验大小合法性
申请内存前必须校验大小 > 0 且不超过合理上限。

**错误**：
```cpp
void* buf = malloc(requested_size);   // ← 未校验 size 合法性
```

**正确**：
```cpp
if (requested_size == 0 || requested_size > MAX_ALLOC_SIZE) {
    return nullptr;
}
void* buf = malloc(requested_size);
```

---

## 8. 其他安全规则 (G.OTH)

### G.OTH.05 [严重] 禁止代码中包含公网地址
代码中不得包含公网 IP / URL（测试数据除外，需明确标注）。

---

## 9. 算法安全 (CIP)

### CIP.01 [致命] 禁止使用不安全 IPSI 算法
禁止使用 MD5 / SHA1 / DES 等已被证明不安全的密码算法，应使用合规算法（如 SM2/SM3/SM4）。

---

## 10. 文件完整性 (OAT)

### OAT.1 [严重] 不允许包含二进制文件
仓库中禁止提交 `.o` / `.so` / `.exe` / `.dll` 等二进制文件。

---

## MR 提交前自查清单

- [ ] 无 G.INC.05 / G.INC.08 违规
- [ ] 无 G.EXP.33 自增/自减重复引用
- [ ] 无 G.FUU.09 / G.FUU.10 不安全内存函数
- [ ] G.FUU.12 destMax 正确
- [ ] 未封装安全函数 / 重命名安全函数 (G.FUU.13/14)
- [ ] 使用安全函数库 (G.FUU.15)
- [ ] 敏感信息使用后清零 (G.MEM.04)
- [ ] 字符串缓冲区足够 (G.STD.05) / 不用 std::string 存敏感信息 (G.STD.07)
- [ ] 格式化字符串与参数匹配 (G.STD.13)
- [ ] 无命令注入风险 (G.STD.15)
- [ ] 不用 exit/atexit/kill (G.STD.16 / G.STD.17)
- [ ] 内存申请前校验大小 (G.RES.02)
- [ ] 无公网地址 (G.OTH.05)
- [ ] 使用合规密码算法 (CIP.01)
- [ ] 无二进制文件 (OAT.1)