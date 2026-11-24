# SpMMACompress 测试

测试覆盖压缩大小查询、结构化稀疏压缩、参数校验和异步执行。接口规格与输出格式见[算子说明](../../sparseLt/spmma_compress/README.md)。

## 功能测试

CSV 驱动测试覆盖四种数据类型、行列方向、A/B 稀疏侧、op/order 组合、ld 填充、批次与广播，以及完整宽度、尾部和多分块。特殊值覆盖负零、NaN 载荷位、Inf、非规格化数和 INT8 极值。

独立 CPU 参考实现按字节比较 values 和 metadata，并检查输入、填充区和输出保护区。组内非零数量超限（overfull）的输入仅检查输入不变和输出边界，不要求确定的压缩数值结果或同步报错。

按仓库[环境部署说明](../../docs/zh/install/quick_install.md)加载 CANN 环境，并确保 CMake 能找到 GTest。以下命令均在仓库根目录执行，完整功能测试需要 Ascend 950PR&950DT系列产品设备：

```bash
bash build.sh --ops=spmma_compress --soc=ascend950 --run
```

脚本在测试程序所在目录运行，并读取同目录的 `spmma_compress_test.csv`。构建完成后，可单独运行无需初始化设备的 CPU 参考实现与空句柄检查：

```bash
(
    cd build/test/spmma_compress
    ./spmma_compress_test --gtest_filter='CompressOracleTest.*:CompressHostNullTest.*'
)
```

## 性能测试

[spmma_compress_perf.cpp](arch35/spmma_compress_perf.cpp) 由本目录的 [CMakeLists.txt](CMakeLists.txt) 注册为独立目标，需要显式构建。完成上述配置和构建后执行：

```bash
cmake --build build --target spmma_compress_perf -j
mkdir -p build/spmma-compress-results
./build/test/spmma_compress/spmma_compress_perf build/spmma-compress-results/compress
```

参数为输出路径前缀，父目录须已存在。程序生成 `compress-samples.csv` 和 `compress-summary.csv`。功能和性能目标均依赖 GTest 可用；运行时须能加载本工程的 sparseLt 库及 CANN 依赖。

性能测试覆盖四种类型、行列方向、大小矩阵和完整/尾部宽度，共 32 个场景；输入为 mixed 模式、单批次、连续存储。每个场景预热 5 次、采样 30 次，预热后和采样后均检查完整输出、输入、填充区和保护区。执行、校验或结果文件写入失败时返回非零退出码。

耗时单位为微秒，每条原始样本对应一次 Compress 调用。event 耗时为两个设备事件之间的间隔，可能包含 Host 提交间隙；Host 耗时仅包含接口调用，不含 stream 同步。Plan 创建、内存分配和参考结果计算均不计入采样。

汇总文件记录 event 的中位数、P95、最小值、最大值、总体变异系数（CV），以及 Host 的中位数和 P95。中位数取中间两数的均值，P95 采用 nearest-rank 方法。程序不比较其他实现的性能。
