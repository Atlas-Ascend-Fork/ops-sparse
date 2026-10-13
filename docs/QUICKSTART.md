# 快速入门：基于ops-sparse仓

## 使用须知

本指南旨在帮助您快速上手CANN和`ops-sparse`算子仓的使用。为方便快速了解算子开发全流程，将以**Spmv**算子为实践对象，其源文件位于`ops-sparse/sparse/spmv/arch22`（Ascend 910B 实现），具体操作流程如下：

1. **[环境部署](zh/install/quick_install.md)**：完成软件包安装和源码下载，此处不再赘述。快速入门场景下，**推荐WebIDE或Docker环境**，安装操作简单。

   > **说明**：当前WebIDE或Docker环境默认安装最新商发版CANN包；如需体验master分支最新能力，可手动安装CANN包，注意软件与源码版本配套。

2. **[编译运行](#一编译运行)**：编译算子包并安装，实现快速调用算子。

3. **[算子开发](#二算子开发)**：通过修改现有算子Kernel，体验开发、编译、验证的完整闭环。

4. **[算子调试](#三算子调试)**：掌握算子打印和性能采集方法。

## 环境准备

**[环境部署](zh/install/quick_install.md)**：完成软件包安装和源码下载，此处不再赘述。快速入门场景下，**推荐WebIDE或Docker环境**，安装操作简单。

> **说明**：当前WebIDE或Docker环境默认安装最新商发版CANN包；如需体验master分支最新能力，可手动安装CANN包，注意软件与源码版本配套。

---

## 一、编译运行

本阶段目的是**快速体验项目标准流程**，验证环境能否成功进行算子源码编译、打包、安装和运行。

### 1. 编译Spmv算子

环境准备好后（注意软件与源码版本配套），进入环境并访问项目源码根目录，编译指定算子。

通用编译命令格式：`bash build.sh --pkg --soc=<芯片版本> --ops=<算子名>`。以Spmv算子为例，编译命令如下：

```bash
bash build.sh --pkg --soc=ascend910b --ops=spmv
```

> **说明**：run 包名中的芯片字段使用短名（见 `cmake/package.cmake`）：`ascend910b` 对应 `910b`、`ascend950` 对应 `950`、`ascend910_93` 对应 `A3`、`ascend310p` 对应 `310p`，完整包名为 `cann-<芯片短名>-ops-sparse-<版本>_linux-<arch>.run`。例如 `--soc=ascend910b` 生成 `cann-910b-ops-sparse-<版本>_linux-<arch>.run`。

若提示如下信息，说明编译成功。

```bash
Self-extractable archive "cann-910b-ops-sparse-${cann_version}_linux-${arch}.run" successfully created.
```

编译成功后，run包存放于项目根目录的build_out目录下。

### 2. 安装Spmv算子包

```bash
./build_out/cann-910b-ops-sparse-*.run --install --install-path=/usr/local/Ascend/
```

注：安装算子包之后，每次调用会从算子包中加载算子，若需快速调试验证算子功能，可直接按照 [3. 快速验证：运行算子样例](#3-快速验证运行算子样例) 步骤进行，无需安装算子包。

### 3. 快速验证：运行算子样例

通用的运行命令格式：`bash build.sh --soc=<芯片版本> --ops=<算子名> --run`。

以Spmv为例，其提供了算子样例`test/spmv/arch22/spmv_test.cpp`，覆盖 float / int32 / fp16 / bf16 等多种 dtype 组合及转置场景，运行该样例验证算子功能是否正常。

```bash
bash build.sh --soc=ascend910b --ops=spmv --run
```

预期输出：各测试用例精度比对通过，并最终打印汇总信息。

```txt
[PASS] spmv_test

========================================
Test Summary:
  Passed: 1 - spmv
  Skipped: 0 -
  Failed: 0 -
========================================
```

## 二、算子开发

本阶段通过**改 kernel 代码 → 重新编译 → 跑测试**，体验算子修改与验证的闭环。

### 1. 修改 Kernel

打开 `sparse/spmv/arch22/kernels/spmv_kernel.h`，在 `Compute` 函数的 float 路径中，将 `alpha` 缩放临时改为 2 倍（**仅用于学习，勿合入**）：

```cpp
AscendC::Muls(floatTmpBuffer, floatTmpBuffer, this->alpha * 2.0f, 1);
```

### 2. 编译与验证

```bash
bash build.sh --soc=ascend910b --ops=spmv --run
```

- 修改后：精度比对失败，`spmv_test` 退出码非 0
- 改回正确实现后：各用例输出 `====Test case pass!====`，build.sh 打印 `[PASS] spmv_test`

如需打包安装，步骤同[一、编译运行](#一编译运行)。

## 三、算子调试

本阶段以Spmv为例，在算子中添加打印并采集算子性能数据，以便后续问题分析定位。

### 1. 打印

算子如果出现执行失败、精度异常等问题，添加打印进行问题分析和定位。

请在 `sparse/spmv/arch22/kernels/spmv_kernel.h`（或具体 dtype 对应的 `kernels/spmv_kernel_*.cpp`）中进行代码修改。

* **printf**

  该接口支持打印Scalar类型数据，如整数、字符型、布尔型等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > printf”。

  在 `Init` 中打印当前核负责的行范围与 tile 大小，便于确认多核切分是否正确：

  ```cpp
  __aicore__ inline void Init(
      GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
      uint32_t totalRowsNum, uint32_t totalColsNum,
      CompT alpha, CompT beta) {
      // ... 行切分与 GlobalBuffer 绑定 ...

      AscendC::PRINTF("blockIdx=%u startRow=%u blockRowNum=%u tileLength=%u nnzRange=[%u,%u)\n",
                      AscendC::GetBlockIdx(), this->startRow, this->blockRowNum,
                      this->tileLength, this->startValIdx, this->endValIdx);

      pipe.InitBuffer(inQueueColIdx, BUFFER_NUM, this->tileLength * sizeof(int32_t));
      // ...
  }
  ```

* **DumpTensor**

  该接口支持Dump指定Tensor的内容，同时支持打印自定义附加信息，比如当前行号等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > DumpTensor”。

  在 `Compute` 中，`xLocal` 与 `valsLocal` 逐元素相乘后，Dump 局部向量检查中间结果：

  ```cpp
  __aicore__ inline void Compute(int32_t currentRow, int32_t validNum) {
      // ... xLocal 已与 valsLocal 完成 Mul ...

      AscendC::Mul(xLocal, xLocal, valsLocal, validNum);
      AscendC::PipeBarrier<PIPE_V>();

      // 打印当前行、有效非零元个数及 xLocal 前若干元素
      AscendC::PRINTF("Compute row=%d validNum=%d\n", currentRow, validNum);
      AscendC::DumpTensor(xLocal, 0, validNum);

      sharedTmpBuffer = workQueueReduce.AllocTensor<float>();
      // ...
  }
  ```

### 2. 性能采集

当算子功能验证正确后，可通过`msprof`工具采集算子性能数据。

- **生成可执行文件**

    调用Spmv算子的test样例，生成可执行文件（spmv_test），该文件位于项目`ops-sparse/build/test/spmv`目录。

    ```bash
    bash build.sh --soc=ascend910b --ops=spmv
    ```

- **采集性能数据**

    进入Spmv算子可执行文件目录`ops-sparse/build/test/spmv`，执行如下命令：

    ```bash
    msprof --application="./spmv_test"
    ```

采集结果在项目`ops-sparse/build/test/spmv`目录，msprof命令执行完后会自动解析并导出性能数据结果文件，详细内容请参见[msprof](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/T&ITools/Profiling/atlasprofiling_16_0110.html#ZH-CN_TOPIC_0000002504160251)。
