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
- [算子支持情况](#算子支持情况)
- [接口列表](docs/zh/api_list.md)：提供全量API信息，方便您查看aclsparse接口的分类和功能。

## 算子支持情况

下表汇总 **已实现的计算/转换类算子** 在 Atlas A2/A3、Ascend 950PR/950DT 上的支持情况（按源码 `arch22` / `arch35` 与各算子 README 产品说明整理）。**不包含** handle / 描述符创建销毁等通用 API。已写入接口文档的条目见 [接口列表](docs/zh/api_list.md)（按通用 API / 算子 API 分类；本表中部分算子的详细接口说明仍以各算子目录 `README.md` 为准，接口列表将持续补齐）。

图例：`√` 已支持，`×` 当前未支持。A2/A3 对应 `arch22`，950PR/950DT 对应 `arch35`（同架构实现共享，合并为一列展示）。

| 算子 | 代表接口 | A2/A3 | 950PR/950DT |
|------|----------|:-----:|:----------:|
| [SpMV](sparse/spmv/README.md) | `aclsparseSpMV` | √ | √ |
| [SpMVOp](sparse/spmv_op/README.md) | `aclsparseSpMVOp` | × | √ |
| [SpMM](sparse/spmm/README.md) | `aclsparseSpMM` | √ | √ |
| [SpMMOp](sparse/spmm_op/README.md) | `aclsparseSpMMOp` | × | √ |
| [CubeSpMM](sparse/cube_spmm/README.md) | `aclsparseCubeSpmm` | √ | × |
| [SDDMM](sparse/sddmm/README.md) | `aclsparseSDDMM` | √ | √ |
| [SpGEMM](sparse/spgemm/README.md) | `aclsparseSpGEMM` | √ | √ |
| [SpSM](sparse/spsm/README.md) | `aclsparseSpSM` | √ | √ |
| [SpSV](sparse/spsv/README.md) | `aclsparseSpSV_solve` | √ | √ |
| [SpVV](sparse/spvv/README.md) | `aclsparseSpvv` | √ | × |
| [Gather](sparse/gather/README.md) | `aclsparseGather` | √ | √ |
| [Scatter](sparse/scatter/README.md) | `aclsparseScatter` | √ | √ |
| [DenseToSparse](sparse/densetosparse/README.md) | `aclsparseDenseToSparseConvert` | √ | √ |
| [SparseToDense](sparse/sparse2dense/README.md) | `aclsparseSparseToDense` | √ | √ |
| [gtsv2](sparse/gtsv2/README.md) | `aclsparseSgtsv2` | √ | √ |
| [Cgtsv2](sparse/cgtsv2/README.md) | `aclsparseCgtsv2` | √ | √ |
| [gtsv2_nopivot](sparse/gtsv2_nopivot/README.md) | `aclsparseSgtsv2Nopivot` | × | √ |
| [gtsv2_strided_batch](sparse/gtsv2_strided_batch/README.md) | `aclsparseSgtsv2StridedBatch` | × | √ |
| [gtsv_interleaved_batch](sparse/gtsv_interleaved_batch/README.md) | `aclsparseSgtsvInterleavedBatch` | × | √ |
| [gpsv_interleaved_batch](sparse/gpsv_interleaved_batch/README.md) | `aclsparseSgpsvInterleavedBatch` | × | √ |
| [CSR 排序](sparse/csrsort/README.md) | `aclsparseXcsrsort` | × | √ |
| [CSC 排序](sparse/cscsort/README.md) | `aclsparseXcscsort` | √ | √ |
| [COO 排序](sparse/coosort/README.md) | `aclsparseXcoosortByRow` | × | √ |
| [COO→CSR](sparse/coo2csr/README.md) | `aclsparseXcoo2csr` | × | √ |
| [CSR→COO](sparse/csr2coo/README.md) | `aclsparseXcsr2coo` | × | √ |
| [CSR→CSC](sparse/csr2csc_ex2/README.md) | `aclsparseCsr2cscEx2` | × | √ |
| [CSR→GEBSR](sparse/csr2gebsr/README.md) | `aclsparseScsr2gebsr` | × | √ |
| [GEBSR→GEBSC](sparse/gebsr2gebsc/README.md) | `aclsparseGebsr2gebsc` | × | √ |
| [CSRGEAM2](sparse/csrgeam2/README.md) | `aclsparseScsrgeam2` | × | √ |
| [NNZ](sparse/nnz/README.md) | `aclsparseSnnz` | × | √ |

> 更细的 dtype / 算法组合以各算子目录下 `README.md` 为准；已文档化的接口说明见 `docs/zh/api_list.md`。结构化稀疏（`aclsparseLt*`）见 [api_list_sparseLt.md](docs/zh/api_list_sparseLt.md)。

## 🔍目录结构

ops-sparse仓关键目录结构请参见[目录结构](docs/zh/install/dir_structure.md)。

> **说明**：当前算子调用样例位于`test/`目录下（如`test/spmv/arch22/spmv_test.cpp`），`examples/`为预留目录。各算子的详细调用说明可参考`test/<算子名>/README.md`。

## 💬相关信息

- [算子支持情况](#算子支持情况)
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
