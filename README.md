# ops-sparse

## 🔥Latest News

- [2026/09] ops-sparse 相关[开源任务](https://www.hiascend.com/activities/task-center?activeCategory=Cann)已发布，欢迎开发者参与，共同完善 CANN 稀疏算子能力。
- [2026/06] 新增稀疏矩阵-稠密矩阵乘法（SpMM）算子实现。
- [2026/05] ops-sparse项目上线，提供稀疏矩阵计算的API以及优化的稀疏矩阵向量乘法实现。

## 🚀概述

ops-sparse是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供高性能稀疏矩阵计算的算子库，专注于优化稀疏矩阵的计算效率。

## 🛠️环境准备

[环境部署](docs/zh/install/quick_install.md)是体验本项目能力的前提，请先完成NPU驱动、CANN包安装等，确保环境正常。CANN 软件包（toolkit/ops 包）请从 [CANN 官网下载中心](https://www.hiascend.com/cann/download) 获取，并按产品系列、CPU 架构、操作系统选择对应版本。

> **说明**：本项目仅支持CANN 9.0.0及后续版本，源码版本与CANN版本配套关系参见[release仓库](https://gitcode.com/cann/release-management)。

## ⬇️源码下载

环境准备好后，下载与CANN版本配套的分支源码，命令如下，\$\{tag\_version\}替换为分支标签名。

> 说明：若环境中已存在配套分支源码，**可跳过本步骤**，例如CANNLab默认已提供最新商发版CANN对应的源码。

```bash
git clone -b ${tag_version} https://gitcode.com/cann/ops-sparse.git
```

说明：对于CANNLab云开发环境，已默认提供最新商发CANN版本配套的源码，如需获取其他版本源码，参考上述命令获取。

## 📖学习教程

- [快速入门](docs/QUICKSTART.md)：从零开始快速体验项目核心基础能力，涵盖源码编译、算子调用、开发与调试等操作。
- [接口列表](docs/zh/api_list.md)：提供全量API信息，方便您查看aclsparse接口的分类和功能。

## 🔍目录结构

ops-sparse仓关键目录结构请参见[目录结构](docs/zh/install/dir_structure.md)。

> **说明**：当前算子调用样例位于`test/`目录下（如`test/spmv/arch22/spmv_test.cpp`），`examples/`为预留目录。各算子的详细调用说明可参考`test/<算子名>/README.md`。

## 💬相关信息

- [目录结构](docs/zh/install/dir_structure.md)
- [接口列表](docs/zh/api_list.md)
- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-linear-algebra)

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/ops-sparse/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/ops-sparse/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/ops-sparse/wiki)获取技术文章，如系列化教程、优秀实践等。
