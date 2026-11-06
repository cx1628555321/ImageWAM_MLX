# INT8 分块交替顺序复测

环境：M5 Pro，15 CPU / 16 GPU，24 GiB 统一内存，macOS 27.2，MLX 0.32.4。后台 mediaanalysisd、视频编解码及虚拟机仍活动，没有暂停其他服务。各轮之前保存进程负载快照，不能称为空闲环境测试。

本轮不再仅比较旧的历史数字，加入原 INT8 融合视觉路径作为同轮控制。顺序：speed → 原 INT8 → tile32k512 → tile64n64 → tile64n64 → tile32k512 → 原 INT8 → speed。每配置两个独立进程，各首次前向、3 次预热、30 次稳态、3 次模块剖析；共 240 次正式稳态。

|配置|60 次平均 ms|两轮均值 ms|P95 ms|视觉前缀 ms（独立同步）|
|---|---:|---|---:|---:|
|current_speed|290.586|280.992 / 300.180|328.181|134.714|
|speed_native_visual|285.983|283.577 / 288.389|321.450|131.571|
|tile32k512|286.176|282.655 / 289.697|320.982|139.233|
|tile64n64|286.453|284.598 / 288.308|320.290|148.106|

三个相机输入、在线分词、状态处理、文本/图像编码、视觉前缀、10 步动作去噪、CPU 输出均包含在端到端内；不包含模型加载、图像文件解码和文件保存。仍为 16×14 动作，不固定或截断文本长度。

与本轮原 INT8 输出比较：{'02_tile32k512': {'output.npz': True, 'dynamic_0.npz': True, 'dynamic_1.npz': True, 'dynamic_2.npz': True}, '03_tile64n64': {'output.npz': True, 'dynamic_0.npz': True, 'dynamic_1.npz': True, 'dynamic_2.npz': True}, '04_tile64n64': {'output.npz': True, 'dynamic_0.npz': True, 'dynamic_1.npz': True, 'dynamic_2.npz': True}, '05_tile32k512': {'output.npz': True, 'dynamic_0.npz': True, 'dynamic_1.npz': True, 'dynamic_2.npz': True}}

动态测试包含 39-token/horizon8/3步、193-token/horizon16/10步、39-token/horizon32/3步。逐值一致仅表示分块改动没有再改变对应 INT8 路径的输出，不代表其与 accuracy 相同或机器人任务精度已验证。

启动：
```bash
cd /Users/qklj/work/cx/ImageWAM_MLX
RUN_TAG=$(date +%Y%m%d_%H%M%S)
.venv/bin/python experiments_tiles_20261104/e2e.py --case tile32k512 \
  --iterations 30 --output records/$RUN_TAG-tile-retest
```

--case 也可取 current_speed、speed_native_visual、tile64n64；对应耗时见上表。完整逐次耗时、加载、首次、模块和输出文件位于 `records/tiles_retest_20261104/<round_case>/`。复测脚本 `experiments_tiles_20261104/retest.py` 拒绝覆盖本次固定记录路径，重新使用需更换目录。

默认 accuracy / speed 保持不变。背景负载和独立进程之间的漂移仍限制小幅收益判断；不会将单算子最快或单轮最快直接升级成默认。
