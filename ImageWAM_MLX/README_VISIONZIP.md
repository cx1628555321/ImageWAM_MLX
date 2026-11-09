# ImageWAM_MLX VisionZip适配与固化版本

更新：2026-11-09。本文件独立汇总VisionZip的适配原理、维度变化、启动方法和精度/性能依据。推荐 `native-w4-zip252-20261109`（native-w4＋merge252），未压缩native-w4用于对照与回退。模型权重和统计沿用项目现有文件。

## 适配位置与原理

ImageWAM使用VAE产生视觉token，没有直接可套用的ViT CLS注意力。这里沿用VisionZip“保留重要token＋合并剩余上下文”的思路，在多模态前缀网络第5个双流块完成后压缩视觉特征。它是ImageWAM适配实现，不是原版ViT插件；π0.5的VisionZip实现和参数也不适用于这里。

1. **评分**：取第5块的旋转后Q、K，用FP32计算联合文本、状态、视觉序列的attention softmax。只取视觉key的列，对所有query和head求均值，衡量视觉token收到的注意力。softmax分母保留完整序列。
2. **三视角配额**：原视觉token为主相机192、左腕48、右腕48。压缩后分别168、42、42，避免全局排序耗尽某个腕相机的信息。
3. **重要token**：按分数降序选择240个，配额160、40、40。分数相同时按原索引排序，保证确定性。
4. **上下文合并**：剩余特征按原顺序均匀选12个代表，配额8、2、2。用归一化特征的余弦相似度分组，代表强制包含自身，每组FP32求均值后转回原dtype。没有加attention质量修正或按分数加权，因此不能宣称无损。
5. **位置与KV**：合并特征沿用代表token的原4D RoPE坐标，不重新编号。文本、状态不压缩。前5层已经生成的完整KV保留，后20层使用缩短序列；动作专家逐层读取对应长度的KV。
6. **工程接入**：只替换编译后的prefill函数，保留原prefill供关闭压缩时恢复。动作W4、VAE、状态归一化、动作还原、16步输出和10轮去噪不变。每次重新计算图像和状态，不复用上一帧视觉特征。

## 明确的维度变化

以batch=1、文本T=42为例：

| 阶段 | 形状或token数 |
|---|---|
| 三图拼图，NHWC | `[1,288,256,3]`，H=288、W=256 |
| VAE输出 | `[1,36,32,32]` |
| 2×2 latent打包 | `[1,18,16,128]` |
| 展平、视觉投影 | `[1,288,128] → [1,288,3072]` |
| 文本＋状态 | `[1,42+1,3072]` |
| 完整前缀 | 42＋1＋288＝331 tokens |
| 第5块后视觉压缩 | `[1,288,3072] → [1,252,3072]` |
| 后20层前缀 | 42＋1＋252＝295 tokens |
| 前5层每个K/V | `[1,24,331,128]` |
| 后20层每个K/V | `[1,24,295,128]` |
| 动作输出 | 每次16×14，10轮去噪 |

因此原前缀较短，是因为三图先拼成一张图，再由VAE和2×2打包压成288个视觉token；不是三张图分别产生288个。文本长度随提示词改变，331不是固定长度。

视觉token减少12.5%，本例后续前缀总长度减少10.88%；VAE、前5层和其他开销仍在，因此端到端收益不会等于12.5%。选择第5块后压缩，在获得联合语义评分的同时，让后续20层受益。

## 固定配置与启动

新入口 `scripts/infer_native_w4_zip252.py` 固定speed＋all_int8、native W4动作、第5块后merge252、10轮去噪、16步输出、文本缓存1条、VAE asymmetric关闭、MLX默认allocator。它校验源码SHA256并拒绝覆盖固定推理参数。MLX为 `0.32.5.dev20261106`，`MLX_MAX_OPS_PER_BUFFER=256`指算子数，不能解释成256MiB内存限额。

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
NEWPY="$PWD/records/background_20261106_xcode/builds/route4796/venv/bin/python"
SAMPLE="$PWD/examples/robotwin_ep0_frame0"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"

# 推荐固定ZIP版本
"$NEWPY" scripts/infer_native_w4_zip252.py \
  --inputs "$SAMPLE/inputs.npz" --prompt-file "$SAMPLE/prompt.txt" \
  --noise "$SAMPLE/initial_noise_fp32.npy" \
  --output "records/${RUN_TAG}_zip252"

# 未压缩对照；该原始入口保留可配置参数
"$NEWPY" scripts/infer_version.py --variant native-w4 \
  --inputs "$SAMPLE/inputs.npz" --prompt-file "$SAMPLE/prompt.txt" \
  --noise "$SAMPLE/initial_noise_fp32.npy" --steps 10 --horizon 16 \
  --warmup 10 --output "records/${RUN_TAG}_native_w4"
```

完整选型表见 [README_VERSION_SELECTION.md](README_VERSION_SELECTION.md)。固定入口的实际输出已经与参考逐位对齐，记录见 `records/native_w4_zip252_20261109_release_check/verification.json`。配置/源码清单为 `versions/native-w4-zip252-20261109.json`，不包含大权重文件的内容锁定。

## 原项目完整测试：选型依据

以下来自ImageWAM_MLX默认内存策略完整复测，不冒充本次embody_edge重测。每方案400组AB/BA端到端配对，8个输入、两轮；每方案360组动作精度比较，108帧、36个episode。端到端为文本缓存命中，不含加载、磁盘I/O和网络。

| 方案 | 配对基线→压缩均值 ms | 延迟降低 | 归一化动作RMSE | 最大绝对偏差 |
|---|---:|---:|---:|---:|
| merge252，固化 | 155.61→148.00 | 4.89% | 0.003649 | 0.144531 |
| late_align32 | 152.72→150.01 | 1.77%，区间跨0 | 0.001938 | 0.031250 |
| prune252 | 154.60→146.46 | 5.26% | 0.003625 | 0.144531 |
| prune216 | 155.37→138.75 | 10.70% | 0.004974 | 0.128906 |

merge252的P50为141.28ms，P95由187.15降至182.53ms；节省时间95%区间4.30～10.87ms。相比prune252，本次尾延迟更好；相比prune216，总体误差更小；late_align32未证明稳定提速。因此选merge252作为折中，不能宣称所有输入上都最优。

独立模块配对：前缀71.06→66.43ms，完整动作专家50.62→47.76ms。ZIP净新增开销0.76～0.87ms，包含评分、选取/合并、必要拼接和位置处理；端到端已包含该开销，不重复扣除。模块隔离/profile的同步方式不同，不能简单相加。VAE没有压缩收益。

相对各自参考输出，两个进程共800次重复均逐位一致，其中merge252为160次。它说明数值可重复，不能等同任务精度无损；最大局部误差不能忽略。样本元数据来自train，未确认独立留出；没有闭环成功率结论。冷文本另测出现秒级长尾，上表不能泛化到频繁切换指令的场景。

## embody_edge集成结果

两个配置已接入 `/Users/qklj/projects/embody_edge` 内置的ImageWAM加载器、离线推理和常驻服务，命令分别选择 `--version native-w4` / `--version native-w4-zip252`。两版本共用该仓库的 `models/imagewam/native-w4` 离线包，正式推理不调用本项目脚本，也不导入本项目源码。完整启动及测试说明在该仓库 `docs/README_IMAGEWAM_VISIONZIP.md`。

迁移快速实测80组配对：均值150.04→143.98ms（降低4.04%），P50 149.00→142.74ms，P95 158.08→151.36ms。每版本17组动作对照与本项目对应参考逐位一致，新增迁移误差0；该困难子集的压缩RMSE为0.007549、最大差0.144531，与上面360组总体统计不是同一分布。两CLI和ZIP真实WebSocket服务均通过输出一致性检查。原始记录保存在embody_edge的`artifacts/imagewam/versions_20261109/`。
