# 贡献指南

本项目欢迎广大开发者体验并参与贡献。在参与社区贡献之前，请参见 [cann-community](https://gitcode.com/cann/community) 了解行为准则，完成 CLA 协议签署，并熟悉源码仓的贡献流程。

## 准备工作

开发者在准备本地代码与提交 PR 时，需重点关注以下事项：

1. 提交 PR 时，请按照 PR 模板仔细填写本次 PR 的业务背景、目的、方案等信息。
2. 若您的修改涉及新增特性、新增接口、新增配置参数或修改代码流程等（非简单 bug 修复），请务必先通过 Issue 进行方案讨论，以避免代码被拒绝合入。若不确定修改是否属于"简单 bug 修复"，建议提交 Issue 进行讨论。

## 贡献场景

开发者贡献场景主要包括：

- [贡献新算子](#一贡献新算子)
- [算子 Bug 修复](#二算子-bug-修复)
- [算子优化](#三算子优化)
- [文档纠错](#四文档纠错)
- [帮助解决他人 Issue](#五帮助解决他人-issue)

---

## 一、贡献新算子

算子开发贡献流程如下：

![算子开发贡献流程](./docs/zh/figures/算子开发贡献流程.png "算子开发贡献流程图")

如果您有全新的稀疏算子希望基于 NPU 设计与实现，欢迎在 Issue 中提出您的想法与设计方案。完整的贡献过程如下：

### 1. 创建 Issue 需求

新建 `Requirement|需求建议` 类 Issue，并阐明新增算子的设计方案。Issue 一般需包含以下内容：

- **背景信息**
- **价值/作用**
- **设计方案**

具体操作步骤请参阅：[Issue 操作指南](https://gitcode.com/cann/community/blob/master/contributor/issue-operation.md)。

### 2. 需求评审

创建需求类 Issue 后，需申报 SIG 议题，并在指定时间参加 Ops-linear-algebra SIG 例会评审。

- [会议时间](https://etherpad-cann.meeting.osinfra.cn/p/sig-ops-linear-algebra)
- [议题申请](https://etherpad-cann.meeting.osinfra.cn/p/sig-ops-linear-algebra)
- [会议链接](https://meeting.osinfra.cn/cann?sig=sig-ops-linear-algebra)

**紧急需求处理：**

若需求紧急，可申请临时 SIG 会议：

1. 填写议题申请：[议题申请](https://etherpad-cann.meeting.osinfra.cn/p/sig-ops-linear-algebra)
2. 发送邮件给 [maintainer](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-linear-algebra/README.md)，建议邮件主题注明：申请临时 SIG 议题及议题内容

**需求接纳：**

若需求被接纳，[SIG 成员](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-linear-algebra/sig-info.yaml)将为您分配合适的算子分类路径，请将贡献算子提交至 `sparse/${operator}/` 对应目录。

### 3. PR 提交

#### PR 提交须知

**提交前准备：**

- 建议在需求评审通过后，再提交 PR。
- 提交 PR 前需完成开发环境准备，并仔细了解项目特定的开发规范和版权声明要求，确保您的贡献符合项目标准，签署 CLA。具体操作步骤请参阅：[PR 操作指南](https://gitcode.com/cann/community/blob/master/contributor/pull_request_operation.md)。

**交付件要求：**

| 类型 | 要求 | 参考资料 |
|------|------|---------|
| 代码交付件 | 需提供算子 Host 侧实现、Device 侧 Kernel 实现、算子测试文件 | [快速入门](docs/QUICKSTART.md) |
| 文档交付件 | 算子说明文档为必选，需放置在 `sparse/${operator}/README.md`；测试与复现说明放置在 `test/${operator}/README.md`，其余文档可视情况提供 | 参考已有算子 README（如 [sparse/spmv/README.md](sparse/spmv/README.md)、[test/spmv/README.md](test/spmv/README.md)） |
| 精度要求 | 新贡献算子需满足精度标准 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |

**合规检查：**

- [ ] 代码是否符合《[C++ 编程规范](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)》
- [ ] 代码是否编译通过（`bash build.sh --ops=${operator} --soc=${soc_version}`）
- [ ] 测试是否全部通过（`bash build.sh --ops=${operator} --soc=${soc_version} --run`）
- [ ] Markdown 文档语法是否符合规范

**提交规范：**

- **接口声明**：新增算子需在 `include/cann_ops_sparse.h` 中添加 API 声明。
- **贡献目录**：按 SIG 成员意见提交至 `sparse/${operator}/` 目录下，可参考已有算子文件放置规则。
- **PR 提交**：通过 `git` 命令提交目标分支 PR，检查 PR 标题是否清晰、PR 描述是否规范（指明更改内容和原因、是否关联对应 Issue）。

> **注意**：如果您希望贡献项目标准算子，其交付件和开发过程比生态算子复杂，包括多架构支持等，具体贡献指导参考 [附录](#附录)。

### 4. CI 门禁

通过评论 `compile` 指令触发开源仓门禁，并依据 CI 检测结果进行修改。目前 CI 门禁包含以下检查项：

- 代码编译
- 静态检查（如涉及 codecheck 误报，请提交给 SIG 成员屏蔽）

门禁通过后，请在关联的 Issue 中 @ Committer。

### 5. Committer 检视

Committer 检视后将反馈检视意见，请根据意见修改。建议您在 PR 修改过程中，在 PR 标题中增加 `[WIP]` 标签；修改完成后去掉该标签，并 @ 指派的 Committer 进行二次检视。

### 6. Maintainer 合入

Committer 检视通过后，将标注 `/lgtm` 标签。Maintainer 最终审核，确认无问题后，将标注 `/approve` 标签合入 PR。

---

## 二、算子 Bug 修复

如果您在本项目中发现了某些算子 Bug，希望对其进行修复，欢迎您新建 Issue 进行反馈和跟踪处理。

您可以按照 [提交 Issue/处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务) 指引新建 `Bug-Report|缺陷反馈` 类 Issue 对 Bug 进行描述。

---

## 三、算子优化

如果您对本项目中某些算子实现有泛化性增强或性能优化思路，希望着手实现这些优化点，欢迎您对算子进行优化贡献。

您可以按照 [提交 Issue/处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务) 指引新建 `Requirement|需求建议` 类 Issue 对优化点进行说明，并提供您的设计方案。

---

## 四、文档纠错

如果您在本项目中发现某些算子文档描述错误，欢迎您新建 Issue 进行反馈和修复。

您可以按照 [提交 Issue/处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务) 指引新建 `Documentation|文档反馈` 类 Issue 指出对应文档的问题。README 文档规范参考已有算子 README（如 [sparse/spmv/README.md](sparse/spmv/README.md)）。

---

## 五、帮助解决他人 Issue

如果社区中他人遇到的问题您有合适的解决方法，欢迎您在 Issue 中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

如果对应 Issue 需要进行代码修改，您可以在 Issue 评论框中输入 `/assign` 或 `/assign @yourself`，将该 Issue 分配给您，跟踪协助解决问题。

---

## 附录

### 项目标准算子交付件

```text
sparse/${operator}/                            # 算子源码目录
├── README.md                               # 必选，算子说明：语义、接口、参数约束、支持的稀疏格式
└── ${arch_dir}/                            # 面向特定架构的实现（如 arch22、arch35）
    ├── ${operator}_host.cpp                # Host 侧：参数校验、任务下发
    ├── ${operator}_kernel.cpp              # Device 侧：Ascend C 核函数实现
    ├── ${operator}_csr_mat.h               # 可选，CSR 矩阵数据结构头文件
    ├── ${operator}_csr_mat.cpp             # 可选，CSR 矩阵预处理实现
    └── ${operator}.h                       # 算子内部头文件

test/${operator}/                           # 算子测试目录
├── CMakeLists.txt                          # 测试编译配置，调用 ops_sparse_add_test()
├── README.md                               # 可选，测试说明：测试入口、运行命令、复现步骤、平台支持情况
└── ${arch_dir}/                            # 面向特定架构的测试
    └── ${operator}_test.cpp                # 测试主文件
```

> **说明**：
> - `${operator}` 为算子名，如 `spmv`、`spmm`、`sddmm`。
> - `${arch_dir}` 为架构子目录，如 `arch22`（Atlas A2系列产品、Atlas A3系列产品）、`arch35`（Ascend 950PR&950DT系列产品），由编译时 `--soc` 参数决定：
>   - `arch22`：Ascend 910B / 910_93（NPU 架构 `dav-2201`）
>   - `arch35`：Ascend 950PR&950DT系列产品（NPU 架构 `dav-3510`）
>   - `arch20`：Ascend 310P（NPU 架构 `dav-2002`）
> - 算子源码目录（`sparse/`）下**无需**提供 `CMakeLists.txt`，由 `sparse/CMakeLists.txt` 自动收集。
> - 算子说明文档 `README.md` 放置在**算子源码目录**（`sparse/${operator}/README.md`），承载算子语义、接口、参数约束与支持的稀疏格式；测试与复现说明放置在**测试目录**（`test/${operator}/README.md`），承载测试入口、运行命令与复现步骤。两者职责不同，**现有算子 README 无需搬迁**。
> - 测试目录（`test/`）下**必须**提供 `CMakeLists.txt`，调用 `ops_sparse_add_test()` 注册测试可执行文件。
> - 目录结构详细说明参考 [目录结构说明](docs/zh/install/dir_structure.md)。

### 多架构并存规则

本仓采用**多架构并存**模式：各平台实现分别放入对应的 `${arch_dir}/` 子目录，编译时由 `--soc` 自动选择，同一时刻仅编入当前 SOC 对应目录下的源文件。

| SOC 版本 | NPU 架构 | 架构目录 `${arch_dir}` | 当前算子示例 |
|----------|----------|------------------------|--------------|
| `ascend910b*`、`ascend910_93*` | `dav-2201` | `arch22` | `spmv` |
| `ascend950*` | `dav-3510` | `arch35` | `spmm` |
| `ascend310p*` | `dav-2002` | `arch20` | — |

**放置原则**：

1. Host/Kernel 实现放入 `sparse/${operator}/${arch_dir}/`，不要在算子根目录与多个 `${arch_dir}/` 下同时放置同名入口文件，否则会产生符号冲突。
2. 同一算子在不同芯片上的实现，分别维护在各自的 `${arch_dir}/` 目录（如 910B 的 `arch22/`、Ascend 950PR&950DT系列产品的 `arch35/`）。
3. 测试文件放入 `test/${operator}/${arch_dir}/${operator}_test.cpp`，与 `sparse/${operator}/${arch_dir}/` 一一对应；`CMakeLists.txt` 调用 `ops_sparse_add_test()`，按 `SOC_ARCH_DIRS` 自动选择当前 SOC 的测试源文件，无匹配则跳过。

**测试 CMakeLists 示例**：

```cmake
ops_sparse_add_test(spmm ${OPS_SPARSE})
```

**具体示例**（`spmm`：当前实现面向 Ascend 950PR&950DT系列产品，位于 `arch35/`）：

```text
sparse/spmm/
└── arch35/
    ├── spmm_host.cpp
    ├── spmm_kernel.cpp
    ├── spmm.h
    ├── spmm_csr_mat.h
    └── spmm_csr_mat.cpp

test/spmm/
├── CMakeLists.txt                  # ops_sparse_add_test(spmm ${OPS_SPARSE})
└── arch35/
    └── spmm_test.cpp
```
