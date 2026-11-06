# 启动预计算与当前指令文本缓存

已适配到 M5 Pro 正式 runtime 和 infer.py。speed 默认 text_cache_size=1：首次计算并缓存，同一原始文本复用；文本变化清除上一条并重新计算。A→B→A 会计算三次。当前默认 accuracy 仍不启用缓存，benchmark.py 默认显式传0，保留原无缓存测量口径。

## 持续服务使用

```python
from imagewam_mlx import ImageWAM
model = ImageWAM("weights/bf16", "assets", profile="speed")
ready = model.prepare_text(prompt)  # 收到初始指令后调用；同步完成文本计算
# 三图到达后：
action, normalized = model.predict(images=images, state=state,
                                   prompt=prompt, noise=noise, steps=10)
# 下次 predict 的 prompt 相同：命中；不同：重新计算并替换。
# 新指令提前到达时也可再次调用 model.prepare_text(new_prompt)。
```

构造模型时尚无 prompt，因此 API 用 prepare_text 显式完成首次准备；即使没有调用 prepare_text，首次 predict 也会自动缓存。

prepare_text 返回 tokens、prepare_ms、text_cache 状态，并执行 mx.eval + mx.synchronize，确保不是只创建惰性图。文本特征仍是 Qwen 9/18/27 层输出；图像、状态、多模态 KV 每次根据实际观察计算。缓存存活于当前模型实例，重启进程会重新计算，不写磁盘权重。

## 命令

```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
.venv/bin/python scripts/infer.py --profile speed   --inputs examples/robotwin_ep0_frame0/inputs.npz   --prompt-file examples/robotwin_ep0_frame0/prompt.txt   --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy   --steps 10 --output records/text_cached_run_01
```

infer.py 启动加载后自动 prepare_text，随后进行已有 warmup 和推理。`run.json` 单列 text_preparation，新增 startup_total_ms 包含加载、文本准备、warmup和请求。request_ms 不含准备成本；不能据此称为无缓存性能。既有 load_warmup_request_ms 字段仍仅表示其名称中的三项，完整启动成本看 startup_total_ms。

传 `--text-cache-size 0` 关闭；显式 >1 保持既有多条 LRU 行为。benchmark.py 仍默认0，需要 `--profile speed --text-cache-size 1` 才测命中路径，它的首次/预热阶段会自然填充缓存。

## 边界与验证

- 缓存默认容量1、特征上限64 MiB。超大文本特征照常推理但不缓存，不截断文本；因此超大项可能每次重算。
- 原始指令字符串改变时单条模式强制清理，即使 token 序列碰巧相同；完整 token ID 仍用于校验命中。
- 权重/量化配置改变后调用 clear_text_cache 或新建模型。模型实例仍按串行请求使用，不新增多线程并发保证。
- 5项CPU回归覆盖首次/命中/变更/回到旧指令、同token不同原文、clear/禁用、超大项、非法文本。M5 Pro 另用真实 MLX CPU 张量检查准备和复用；不是全模型GPU精度或性能测试。
- 已在空闲窗口完成全模型回归和正式 CLI 启动验证，结果见下文；后台 Split-K 任务已结束。

测试：`python tests/test_text_prepare.py`；远端记录 `records/text_prepare_20261105/`。原文件备份在该记录目录。后台 Split-K 任务显式 text_cache_size=0，不受新默认影响。

## 空闲窗口全模型复测

2026-11-05 后续复测，同一 speed 权重与量化路径，三图、10步、H16、39-token 标准指令。缓存和无缓存模型共享只读权重，无缓存对照直接调用原编码器，不改变动作编译图。每组预热5次，30组随机顺序交替，显式同步至 CPU 结果。

|模式|稳态均值 ms|P95 ms|
|---|---:|---:|
|无缓存|193.399|193.669|
|相同文本缓存命中|169.379|169.791|

平均节省24.020 ms，约12.42%。这是文本特征复用后的条件性性能，不是无缓存基线提速。

7个完整推理对照包括 A→A→B→B→A→长文本→相同长文本，并在重复指令时改变图像与机器人状态；action/normalized 均与无缓存对照逐值一致。命中/未命中计数逐项符合预期。输出文件保存在 full/case_*.npz，含参考值。

本次加载3.939秒，首次文本准备5078.979 ms，包含首次执行/编译等成本，不能忽略或当作稳态文本耗时。单独正式 CLI（warmup=0）文本准备2792.738 ms、首次完整请求611.617 ms、startup_total_ms=6681.497 ms；记录1 miss / 1 hit，输出与无缓存参考逐值一致。两次首次准备波动明显，不是严格冷启动比较，也不用于承诺固定启动延迟。

正式 CLI 命令：

```bash
.venv/bin/python scripts/infer.py --profile speed \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --steps 10 --warmup 0 --output records/text_prepare_20261105/cli
```

此输出目录已经存在，重跑须换新目录。

原始数据：[全模型结果](../../records/text_prepare_20261105/full/results.json)、[CLI记录](../../records/text_prepare_20261105/cli/run.json)。脚本：[full_test.py](../../experiments/text_prepare_20261105/full_test.py)。不涉及机器人任务成功率重新评估。
