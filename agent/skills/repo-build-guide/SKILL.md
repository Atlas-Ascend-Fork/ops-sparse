---
name: repo-build-guide
description: Sparse 算子仓编译与验证命令参考。编译算子、运行测试、打包的常用命令。触发：需要编译算子、运行测试、查看 build.sh 用法时。
---

# ops-sparse 构建命令

## 环境要求

- CANN 开发套件（含 Ascend C 编译器）
- CMake >= 3.16
- GCC（与 CANN 版本匹配）

## 编译命令

### 编译指定算子

```bash
bash build.sh --ops={op_name} --soc={soc_version}
```

参数说明：
- `--ops`：算子目录名（snake_case），如 `spmv`、`spmm`。支持逗号分隔多算子：`--ops=spmv,spmm`
- `--soc`：芯片版本号（小写），如 `ascend910b`、`ascend950`
- `--device=<id>`：指定测试运行的 NPU 设备 ID（默认 0），如 `--device=1`
- `--run`：编译后运行测试
- `--pkg`：编译后打包 run 包
- `--torch_extension`：构建 Torch Extension wheel，可配合 `--ops` 指定算子；不能与 `--run` 或 `--pkg` 同用。

### 编译全量

```bash
bash build.sh --soc={soc_version}
```

### 编译并运行测试

```bash
bash build.sh --ops={op_name} --soc={soc_version} --run
```

### 打包

```bash
bash build.sh --pkg --soc={soc_version}
```

## 常用芯片版本

| 芯片 | --soc 参数 | NPU_ARCH | 架构目录 |
|------|-----------|----------|----------|
| Ascend910B | ascend910b | dav-2201 | arch22 |
| Ascend910_93 | ascend910_93 | dav-2201 | arch22 |
| Ascend950 | ascend950 | dav-3510 | arch35 |
| Ascend310P | ascend310p | dav-2002 | arch20 |

> `build.sh` 内部做大小写无关匹配，支持前缀匹配（如 `ascend910b3` 匹配到 `ascend910b`），推荐使用小写形式。

### 架构默认反选

根据 SOC 参数确定目标 arch 目录的默认值：

| SOC | 默认 arch |
|-----|----------|
| `ascend950` | `arch35` |
| `ascend910b` / `ascend910_93` | `arch22` |
| `ascend310p` | `arch20` |

此映射为默认值，开发者若有对应板子可根据实际目标调整。

## 算子名解析

`--ops` 参数支持两种格式：

1. **具体算子名**（如 `spmv`、`csrmv`）→ 直接编译该算子
2. **家族名**（如 `spmv`）→ 自动展开为家族下所有有当前 SOC arch 实现的子算子

## 输出目录

| 产物 | 路径 | 说明 |
|------|------|------|
| 动态库 | `build_out/lib64/libcann_ops_sparse.so` | 安装后的算子库 |
| 头文件 | `build_out/include/cann_ops_sparse.h` | 安装后的公共头文件 |
| 测试二进制 | `build/test/{op}/{op}_test` | 编译后的测试程序 |

## 测试运行

```bash
# 方式一：通过 build.sh 编译并运行
bash build.sh --ops={op_name} --soc={soc_version} --run

# 方式二：直接运行已编译的测试程序
./build/test/{op_name}/{op_name}_test
```

## 测试失败诊断

### 快速排查

1. 检查编译日志确认无警告
2. 检查 NPU 设备状态：`npu-smi info`
3. 检查 ACL 运行日志（stdout 输出）
4. 对比 golden 与 NPU 输出的最大误差和位置

### 基线对比（判断是否为本次修改引入）

当测试用例执行失败时，需要判断是否为**本次修改引入的问题**：

1. **获取最新基准分支代码**：
   ```bash
   git fetch cann
   ```

2. **切换到基准分支**，重新编译并运行相同算子的测试：
   ```bash
   git checkout cann/master
   bash build.sh --ops={op_name} --soc={soc_version} --run
   ```

3. **对比结果**：
   - 若基准分支上测试**通过** → 本次修改引入了问题，需要排查
   - 若基准分支上测试**同样失败** → 这是算子原有的问题，非本次修改导致

4. **切回开发分支**继续工作：
   ```bash
   git checkout <你的分支名>
   ```

**注意**：切换分支前确保当前修改已 commit 或 stash，避免丢失工作进度。

## 提交前检查

> **提醒**：提交 PR 前检查并移除工作流中产生的临时文件（如 `*_perf.cpp`、`*_perf.csv`、`run_perf.py` 等性能迭代产物），这些文件不应带入 PR。
