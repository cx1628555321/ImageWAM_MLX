# RoboTwin 六个保留版本精度实测

日期：2026-11-06。M5Pro执行，432次推理全部完成、输出均为有限值。本轮不重新测性能，性能仍引用各版本原报告。

## 数据与比较方式

- 12个episode：0、2499、4999、7499、9999、12499、14999、17499、19999、22499、24999、27499。
- 每个episode选起始、中间、最后一个完整16步窗口，共36个真实帧；三相机RGB、状态、指令均取同一数据行，验证视频时间戳和帧索引。36个task_index不代表36个不同任务类别。
- 每帧使用两份固定FP32噪声（NumPy default_rng seed42/123），每版本72组，共六版本432次。保持H16、14维动作、10步，不重采样噪声，不修改文本。
- 真实文本token范围36–51。此次不冒充31–79 token的全范围验证；之前的长度压力测试是另一套。
- 数值参照是本轮全模型MLX accuracy（MLX0.32.4），不是PyTorch真值。accuracy与stock用旧环境，其余版本用已验证新环境。
- 原始动作按当前帧开始的连续16行对齐，数据配置action_state_transforms=null。直接计算反归一化动作误差，同时保存归一化及每维统计。关节与夹爪单位不同，混合RMSE仅用于相同输出定义下的对照。

## 结果

| 版本 | 相对accuracy动作RMSE | MAE | 最大绝对差 | 归一化动作RMSE | 相对示范动作RMSE |
|---|---:|---:|---:|---:|---:|
| accuracy | 0（参照定义） | 0 | 0 | 0 | 0.030532 |
| stock | 0.571783 | 0.421106 | 3.008241 | 0.839831 | 0.574726 |
| runtime | 0.571789 | 0.421129 | 3.008241 | 0.839808 | 0.574733 |
| native-w1 | 0.571768 | 0.421103 | 3.008241 | 0.839807 | 0.574711 |
| native-w4 | 0.007701 | 0.005543 | 0.043815 | 0.011597 | 0.031396 |
| action-bf16 | 0.003633 | 0.001865 | 0.045226 | 0.005189 | 0.029726 |

| 版本 | 示范关节RMSE（12维） | 示范夹爪RMSE（2维） |
|---|---:|---:|
| accuracy | 0.032427 | 0.014713 |
| stock | 0.597556 | 0.411984 |
| runtime | 0.597568 | 0.411949 |
| native-w1 | 0.597539 | 0.411977 |
| native-w4 | 0.033261 | 0.016189 |
| action-bf16 | 0.031543 | 0.014682 |

## 结论与版本选择

1. 三个W1系列（stock/runtime/native-w1）误差相近、明显大于W4和BF16。当前数据不支持把W1称为精度合格版本；可继续保留速度极限实验用途，但不作为兼顾精度的首选。
2. **兼顾速度与数值精度，优先native-w4**。本批相对accuracy RMSE为0.007701，最大差0.043815；历史性能147.885ms，不能把这一历史性能误称为本轮同批测量。
3. **更低平均数值误差选择action-bf16**，RMSE为0.003633；历史性能179.828ms。它不是全模型BF16，仍有INT8前缀误差。最大差0.045226略大于W4，不能声称每个样本/元素都更好。
4. accuracy保留作全浮点实现参照。其相对自身误差0不代表任务误差0。action-bf16示范RMSE略低于accuracy，不能由此断言量化提高模型真实精度或成功率。
5. 模型对示范动作的离线误差不等于机器人闭环成功率；生成动作可能有多个有效解。数据元信息标记train，本次未核验训练留出划分，**不是独立测试集泛化或RoboTwin成功率评测**。尚未设定任务允许误差阈值，因此不做“全场景无损/合格”的结论。

最差案例（按相对accuracy逐案例RMSE）：W1为ep19999_frame165_s123（RMSE0.877997）；W4为ep17499_frame152_s123（0.010767）；BF16 action为ep19999_frame165_s123（0.008423）。最差RMSE案例与全局最大单元素差不一定相同。

## 文件与复现

远端根目录：`/Users/qklj/work/cx/ImageWAM_MLX`。

- `records/robotwin_accuracy_20261106/manifest.json`：采样、原视频/表格路径、指令、输入SHA256、噪声定义。
- `records/robotwin_accuracy_20261106/inputs/`：36份真实三图输入、状态、示范动作和噪声。
- `records/robotwin_accuracy_20261106/*_attempt1/`：每版本72组action/normalized输出及版本信息。
- `records/robotwin_accuracy_20261106/accuracy_summary.json`：汇总、逐维、逐时间步和逐案例误差。
- `records/robotwin_accuracy_20261106/REPORT.md`：远端自动汇总表。
- `experiments/robotwin_accuracy_20261106/`：prepare.py、worker.py、runner.py、summarize.py。

所有完整输入与输出保留远端，不批量复制到本地。本地此README记录本轮读取的汇总结果。

单版本复测（先cd项目根目录，输出目录必须不存在）：

```bash
NEWPY="$PWD/records/background_20261106_xcode/builds/route4796/venv/bin/python"
"$NEWPY" experiments/robotwin_accuracy_20261106/worker.py native-w4 \
  "records/robotwin_accuracy_w4_recheck_$(date +%Y%m%d_%H%M%S)"
```

其他版本替换名称。accuracy/stock必须用`.venv/bin/python`。汇总脚本读取已完成的本次runner索引；新一轮独立输出不能自动混入旧汇总。
