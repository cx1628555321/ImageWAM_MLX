# 前缀瓶颈后续验证与版本入口（2026-11-06）

## 版本入口验证

真实推理用法与速度总表见 [版本索引](../../README_INFERENCE_VERSIONS.md)。已在m5pro实际运行四种入口，10步、同样三图、同一noise、warmup=2；下面是单次CLI结果，不能代替50轮ABBA均值。

| 版本 | 单次 request ms | 输出形状 | 与此前同版本标准输出逐位相同 | 最大差 |
|---|---:|---|---|---:|
| stock | 169.072 | [16, 14] | True | 0.00000000 |
| runtime | 148.790 | [16, 14] | True | 0.00000000 |
| native-w1 | 146.483 | [16, 14] | True | 0.00000000 |
| native-w4 | 151.483 | [16, 14] | True | 0.00000000 |

## 前缀候选

新运行时 route4796 + 原生W1动作网络固定不变。每轮10预热/50正式/50独立模块计时，保留51个案例和H=1/H=64/长文本输出。无图像缓存，不固定文本长度。

| 候选 | E2E ms | 前缀 ms | 动作 ms | 51案例最大差 |
|---|---:|---:|---:|---:|
| baseline | 148.260 | 67.462 | 47.584 | 0.00000000 |
| fuse_bias | 153.533 | 67.451 | 47.382 | 0.00000000 |
| fuse_norm | 149.867 | 67.394 | 47.503 | 0.01600134 |
| fuse_both | 145.002 | 67.402 | 47.488 | 0.01600134 |
| slim_baseline | 149.935 | 67.517 | 47.568 | 0.00000000 |
| slim_quant | 150.290 | 67.518 | 47.412 | 0.00000000 |

### 组合融合 ABBA
基线两轮均值 146.937ms；组合融合 148.593ms；差值 -1.656ms。初筛145.002ms没有在ABBA中复现，不推荐启用。

### 无用指数缓冲裁剪
`x_span=0` 时指数数据没有实际读取。新版激活量化kernel不再输出指数缓冲，mm兼容参数使用动态INT8激活的uint8视图，不新增拷贝；在mm入口断言 x_span=0。
六种矩阵形状的量化值与scale逐位相同；完整模型标准输出、51案例及动态输入通过。此方案仍约150.290ms，没有超过对应基线149.935ms，因此仅保留实验入口。
修复过程保留：第一次字符串匹配误删 max_exp；修复后静态占位张量被编译成Metal constant地址空间，与device指针不匹配；最终采用动态激活缓冲视图解决。前两版失败日志不删除，也不计为有效性能结果。

## 结论及下一方向

本轮四个前缀方向均已适配和验证，未得到可推广的端到端收益。现有推荐版本保持不变。
融合候选前缀计时仍约67.4ms，删除指数写入后仍约67.5ms；说明这些小型中间操作不是当前已证实的主要可优化开销。下一步应以真实前缀线性层形状和GPU kernel时间为依据，定位主矩阵乘法的tile/布局/带宽，而非继续盲目减少小算子。
此前Metal System Trace默认未启用Shader Timeline，尚不能给出逐kernel占比；不能将上述推断写成已经测出的硬件瓶颈。

## 复现候选测试

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
BASE_RUN="$PWD/records/prefix_followup_20261106_repaired/baseline_attempt1" \
records/background_20261106_xcode/builds/route4796/venv/bin/python \
  experiments/prefix_followup_20261106/worker.py slim_quant \
  "records/recheck_slim_quant_$(date +%Y%m%d_%H%M%S)"
```

其他候选把slim_quant替换为fuse_bias/fuse_norm/fuse_both；它们均包含原生W1动作路径。测试失败、中断与修复分开保存于 records/prefix_followup_20261106*，最终汇总在 records/prefix_followup_20261106/FINAL_SUMMARY.json。
