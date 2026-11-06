# 当前保留的 ImageWAM 推理版本

更新：2026-11-06。设备：Apple M5 Pro，24GiB统一内存；三张RoboTwin输入、H16、10步动作去噪。下表列出可使用的精度、速度和对照入口，不把失败候选算作新增推荐版本。

| 保留版本 | 入口 | 文本缓存 | 稳态端到端 ms | 标准样本参考 RMSE | 用途 |
|---|---|---|---:|---:|---|
| 全模型 accuracy | 原 infer.py --profile accuracy | 无 | 281.650（历史） | 不与新批次数值混比 | 原BF16精度对照；本轮未重测 |
| stock | infer_version.py --variant stock | 命中 | 160.747 | — | 原MLX 0.32.4速度对照 |
| runtime | infer_version.py --variant runtime | 命中 | 149.187 | — | 仅切换新运行时的对照 |
| native-w1 | infer_version.py --variant native-w1 | 命中 | 146.903；另批144.773 | 0.51806 | 速度优先 |
| native-w4 | infer_version.py --variant native-w4 | 命中 | 147.885 | 0.00755 | 精度/速度折中 |
| **action-bf16** | **infer_version.py --variant action-bf16** | 命中 | **179.828** | **0.00220** | **原始BF16动作权重，保留当前INT8前缀的精度选项** |

这些是不同批次已有实测，不是六版本同轮统一复测。新运行时action-bf16与同期W1分别179.828/150.171ms；不得用历史最低W1值计算其同轮收益。最新W1在不同批次约145–150ms。除accuracy历史记录外，新版测量每轮预热10次、正式50次，并做独立进程交叉复测。

缓存只复用同一文本的特征，三图和状态每次计算。时间不包含加载、磁盘读取和输出保存。精度值为标准样本数值误差，未代表机器人闭环成功率。action-bf16 **不是全模型BF16**；视觉/多模态前缀仍是当前优化的all_int8路径。

## 启动方式

在M5Pro执行：

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
NEWPY="$PWD/records/background_20261106_xcode/builds/route4796/venv/bin/python"
SAMPLE="$PWD/examples/robotwin_ep0_frame0"
# 可选 action-bf16 / native-w1 / native-w4 / runtime
VARIANT=action-bf16
"$NEWPY" scripts/infer_version.py --variant "$VARIANT" \
  --inputs "$SAMPLE/inputs.npz" --prompt-file "$SAMPLE/prompt.txt" \
  --noise "$SAMPLE/initial_noise_fp32.npy" --steps 10 --warmup 10 \
  --output "records/infer_${VARIANT}_$(date +%Y%m%d_%H%M%S)"
```

stock使用 `.venv/bin/python scripts/infer_version.py --variant stock`，其余输入参数相同。

原accuracy使用：

```bash
.venv/bin/python scripts/infer.py --profile accuracy \
  --inputs "$SAMPLE/inputs.npz" --prompt-file "$SAMPLE/prompt.txt" \
  --noise "$SAMPLE/initial_noise_fp32.npy" --steps 10 --warmup 10 \
  --output "records/infer_accuracy_$(date +%Y%m%d_%H%M%S)"
```

默认行为不改变；新版精度路径显式传 `--variant action-bf16`。结果包含action.npy、normalized.npy、action.csv及run.json，后者记录实际版本和action权重类型。每次启动使用新输出目录。

## 其他配置与归档

旧speed无文本缓存193.399ms、缓存169.379ms属于此前实现/窗口；旧4步缓存136.390ms改变去噪步数，不与本表10步版本混排，也不是当前版本速度承诺。详见 [历史汇总](docs/optimization/README_INFERENCE_QUANTIZATION_SUMMARY_20261105.md)。

FP16 action（179.216ms）与W1展开BF16（180.000ms）保留实验代码及结果，但没有速度优势，不增加推荐启动版本。分块、融合等无收益候选同样仅留作实验记录。原始权重未覆盖。

证据：[当前版本与优化索引](README_INFERENCE_VERSIONS.md)、[Action浮点对照](docs/optimization/README_ACTION_FLOAT_COMPARE_20261106.md)。

## 新入口验证

`action-bf16` 已执行真实三图CLI，10步、2次预热，输出16×14；action和normalized均与此前BF16实验标准输出逐位一致（max_abs=0）。本次单请求181.835ms仅作入口验证，正式性能引用179.828ms交叉测试均值。远端记录：`records/action_bf16_retained_cli_20261106/verification.json`。

## RoboTwin 多样本精度更新

六版本已完成同批12个episode、36个真实帧、两份噪声共432次推理。相对全模型accuracy的RMSE：native-w1 0.571768、native-w4 0.007701、action-bf16 0.003633。精度/速度折中优先native-w4，更低平均数值偏差用action-bf16；W1仅保留速度极限用途。详情与边界见 [RoboTwin精度报告](README_ROBOTWIN_ACCURACY_20261106.md)，不要用此前单样本误差替代本轮结论。
