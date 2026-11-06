# ImageWAM MLX 本轮全方向适配与汇总（2026-11-05）

本轮已完成剩余可执行方案的集中适配、真实输入验证和性能复核：两种 KV 改写、文本缓存命中/未命中、8/6/4 步、缓存＋4 步、KV 缓冲所有权审计、并发负载记录及 CPU 栈诊断。先前尾块与双缓冲结果一并汇总。没有把候选微基准当作正式端到端收益。

**默认保持无文本缓存、10 步，以及已固化的尾块优化。** 文本缓存通过显式 `--text-cache-size 8` 开启；`benchmark.py` 新增 `--steps`，`infer.py` 原有步数参数继续适用。没有修改或覆盖模型权重。缓存不用于 accuracy 配置，默认 accuracy 行为不变。

## 环境、口径与测试规模

远端 `/Users/qklj/work/cx/ImageWAM_MLX`。Apple M5 Pro，15 核 CPU、16 核 GPU，24 GiB 统一内存（无独立显存），macOS 27.2，Python 3.12.13，MLX 0.32.4，`MLX_MAX_OPS_PER_BUFFER=256`。

标准输入为真实 RoboTwin 三图＋状态＋39-token 指令，H16。端到端计时从已解码 RGB/状态/文本到 CPU 动作，包含在线图像编码和状态处理，排除磁盘 I/O、模型加载。缓存命中模式仅跳过重复文本编码；少步模式改变去噪轨迹。首次与加载另行记录。

* 8 配置同进程随机配对，各 50 次。
* 9 配置正反顺序各一次独立进程，共 18 个进程；每进程预热 8 次、稳态 60 次、模块诊断 3 次，共 1080 次稳态测量。
* 全方案 51 组真实输入验证；等价候选另有动态形状、步数、图像状态变化回归。
* 集成后正式 CLI 5 个进程、300 次稳态，逐次核对 action/normalized 与相应实验配置一致。
* CPU 栈采样另用独立诊断进程，结果不并入性能表。所有慢样本保留。

## 正式 CLI：可直接使用的结果

顺序为基线、缓存、4 步、缓存＋4 步、基线；基线两块为 196.586 / 200.744 ms，存在时间漂移。下表基线合并 120 次，其余各 60 次；百分比仅比较本组实测，不宣称已排除负载影响。

|配置|均值 ms|P95 ms|相对同轮基线节省|精度/使用条件|
|---|---:|---:|---:|---|
|默认：无缓存、10 步|198.665|203.567|—|当前 speed 基线|
|文本缓存命中、10 步|172.664|173.014|26.000 ms / 13.09%|重复 token 序列；与 speed 逐值一致|
|无缓存、4 步|159.888|160.653|38.777 ms / 19.52%|动作数值改变|
|文本缓存命中＋4 步|136.390|137.938|62.274 ms / 31.35%|同时要求重复文本与接受少步误差|

**136 ms 不是原始无缓存、10 步的推理速度。** 新进程或新指令首次不命中缓存；正式缓存测试均包含首次 miss，再单独计命中的稳态。

|配置|模型加载 s|首次推理 ms|
|---|---:|---:|
|默认（两进程均值）|3.097|832.782|
|缓存、10 步|3.049|819.923|
|4 步|3.057|775.342|
|缓存＋4 步|3.026|778.353|

首次为本进程首次前向，不代表清空系统/编译缓存后的冷启动。未以加载差异计入优化收益。

模块诊断（ms，带同步；会改变重叠关系，不能相加替代端到端）：

|模块|默认（两进程均值）|缓存10步|4步|缓存＋4步|
|---|---:|---:|---:|---:|
|文本编码|27.552|0.028|27.531|0.027|
|图像预处理|2.469|2.614|2.545|2.607|
|图像编码|27.957|27.917|27.949|27.941|
|视觉 prefill|83.464|81.116|81.925|83.309|
|动作去噪|63.071|62.677|25.785|25.772|

## 全方向独立进程筛选

下表属于前一组 18 进程测试，与上面的正式 CLI 组分开比较。每种配置合并 120 次；基线两块为 196.335 / 209.239 ms。

|方案|端到端均值 ms|结果与处理|
|---|---:|---|
|当前基线|202.787|保留|
|固定 padded 前缀、不回传更新 KV|203.558|逐值一致，但更慢；仅实验|
|每层 K/V 打包，一次 slice_update|211.743|逐值一致，但更慢；仅实验|
|缓存命中|176.448|有效，显式可选|
|缓存每次强制 miss|210.628|不保证 miss 提速；保持默认关闭|
|8 步|194.548|少步误差；显式选择|
|6 步|177.870|少步误差；显式选择|
|4 步|166.425|少步误差；显式选择|
|缓存命中＋4 步|139.806|业务条件成立时可选|

缓存 miss 的同进程配对只比基线略慢，独立进程却进入约 101–103 ms CPU 时间的状态。不能把独立差值全部归因于缓存代码，也不能承诺缓存对全新指令零代价。缓存命中率低的业务应使用默认无缓存路径。

此前方向的最终状态：

|方向|状态|
|---|---|
|视觉 INT8 32 行尾块|已固化；上一轮同轮正式对照 199.177 → 197.488 ms，约 0.85%|
|双缓冲权重流水|已适配并复核；追加 ABBA 201.336 → 204.679 ms，未合入|
|增宽 BN、改变遍历顺序、减少提交频率、QoS 等|此前已测试，无稳定部署收益；未因本轮请求重新启用|
|调度慢态诊断|本轮完成观测和栈采样，尚未确定根因；没有调度修复收益|

## 精度、动态输入与缓存语义

51 组真实输入（RoboTwin 代表文本长度及不同观察）中，static KV、packed KV、缓存 hit/miss 的 action 与 normalized 均逐值一致。额外覆盖 39/189 tokens、H1/8/16/64、1/3/10/11 步及相同指令下改变图像/状态；等价候选均通过。

少步方案相对当前 **10 步量化 speed** 的动作差异：

|步数|51组平均逐样本 RMSE|最坏逐样本 RMSE|全组最大绝对误差|
|---|---:|---:|---:|
|8|0.219616|0.243617|0.776062|
|6|0.181692|0.191791|0.608049|
|4|0.382030|0.391251|1.392112|

均为有限值；6 步在此样本集误差小于 8 步，不代表任务成功率更高。当前量化 speed 本身就不同于 PyTorch/accuracy，上述一致性不是原始模型无损声明。没有完成 RoboTwin 闭环成功率评估，也没有开展少步蒸馏训练。

缓存位于 `imagewam_mlx/text_cache.py`：每模型实例独立 LRU，键为完整 token ID、shape、dtype，不截断文本、不跨实例复用。权重和编码器配置固定时缓存有效；模板/分词变化产生不同 token 会自然 miss。改变权重或模型策略后必须调用 `model.clear_text_cache()`，或创建新实例。默认最多 8 项（可配置），特征总量上限 64 MiB，超大特征不缓存；标准样本一项 599040 字节。原有模型实例不提供并发调用保证，缓存同样按顺序请求使用。

已验证 LRU 淘汰、byte 上限、超大项 bypass、dtype 区分和 clear 失效。正式缓存组统计 1 miss / 72 hits，涵盖首次、预热、计时、模块诊断及 trace。

## KV 审计与调度诊断

MLX 0.32.4 的 [DynamicSliceUpdate 实现](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/gpu/primitives.cpp) 先尝试复制或移交缓冲，再更新目标区间；[Metal copy 实现](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/copy.cpp) 在缓冲可移交且 dtype 相同时跳过复制。因此逻辑更新次数不等于物理全量复制次数。

独立所有权实验覆盖 P320/328/368/478。以 P328 为例，单 K/V 缓冲 2113536 字节：释放旧数组引用时 active allocation 变化约 -4 字节；保留旧数组时增加约一个完整缓冲，且旧数组内容未被修改。这支持所有权影响缓冲复用，但不是整网每层每步的物理复制计数。

10 次更新的小型计时为可复用 2.545 ms、保留旧引用 2.265 ms，受分配、同步和微基准顺序影响，不能据此推断复制更快或整网可节省多少。真正决策依据是两种 KV 改写的端到端退化，故保留原 workspace。

18 进程期间每 5 秒保存进程快照，共 70 次，没有看到 Pi0.5 进程；仍有 CPU 高耗时状态，不能将慢态简单归因于该任务。进程快照不保证 GPU 独占。

独立 `/usr/bin/sample` 栈含条件变量等待、IOKit、MLX eval、Metal 编码、缓冲管理及 Pillow resize。等待线程样本数不是 CPU 热点耗时，更不是 GPU 时间。服务器 `xcrun --find xctrace` 无可用结果；完整 GPU 时间线和慢态唯一根因尚未获取。本轮没有安装完整 Xcode、停止其他服务，或宣称修复系统调度问题。

## 启动命令

在 `/Users/qklj/work/cx/ImageWAM_MLX` 执行，每次使用新的输出目录。下列耗时来自本轮正式 CLI 组，不是每次运行的保证值。

```bash
# 默认，无缓存10步：198.665 ms
.venv/bin/python scripts/benchmark.py --profile speed \
  --reference examples/robotwin_ep0_frame0 --output records/all_default_01 \
  --warmup 8 --iterations 60 --profile-iterations 3

# 重复指令缓存命中，10步：172.664 ms
.venv/bin/python scripts/benchmark.py --profile speed --text-cache-size 8 \
  --reference examples/robotwin_ep0_frame0 --output records/all_cached_01 \
  --warmup 8 --iterations 60 --profile-iterations 3

# 少步，输出改变：4步159.888 ms；也可显式选择 --steps 8 或 --steps 6
.venv/bin/python scripts/benchmark.py --profile speed --steps 4 \
  --reference examples/robotwin_ep0_frame0 --output records/all_steps4_01 \
  --warmup 8 --iterations 60 --profile-iterations 3

# 缓存命中＋4步：136.390 ms；不代表原始10步无缓存性能
.venv/bin/python scripts/benchmark.py --profile speed --text-cache-size 8 --steps 4 \
  --reference examples/robotwin_ep0_frame0 --output records/all_fast_01 \
  --warmup 8 --iterations 60 --profile-iterations 3

# 实际输入启动；warmup=1 会先执行一次 miss，随后测命中请求
.venv/bin/python scripts/infer.py --profile speed --text-cache-size 8 --steps 4 \
  --inputs examples/robotwin_ep0_frame0/inputs.npz \
  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \
  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \
  --output records/all_infer_01 --warmup 1
```

缓存需要同一模型进程连续服务才能跨请求获益，退出 CLI 后缓存消失；第一次调用不能预先命中。去掉 `--text-cache-size` / `--steps` 即恢复默认。`--no-tail-visual` 仍可回退上一轮尾块优化。

记录全部位于 `records/remaining_20261105/`：screen、independent、precision_summary、process_snapshots、kv_audit、dynamic、cli_*（包含每次输出和源码快照）、CPU sample、infer_smoke。实验代码位于 `experiments/remaining_20261105/`；正式缓存不依赖实验代码。主要后续未解决事项为慢态根因、整网 GPU 物理拷贝计数及少步闭环精度，不计入本轮已获收益。
