from pathlib import Path
import json,hashlib,shutil,tarfile
import numpy as np
root=Path('records/split_tune_20261104');src=Path('development/legacy_experiments/experiments_split_tune_20261104')
sweep=json.loads((root/'sweep.json').read_text());tiles=json.loads((root/'tiles/sweep.json').read_text());r=json.loads((root/'confirm_v2/results.json').read_text())
lines=['# Split-K 并行度与 NAX 布局调优（2026-11-04）','','## 范围','','在上一轮 W1 Split-K 门控残差融合版基础上，适配目标线程组数量与矩阵块布局。没有改变模型权重、三图预处理、文本 token、去噪步数；没有增加整层权重反量化。生产默认代码和上一轮实验均未修改。','','基准是上一轮 Split-K 融合版（target=512，BM32/BN64/WM2/WN2），不是生产 speed，也不是 PyTorch。不能将本轮数值与其他系统状态下的历史速度直接相减。','','M5 Pro / 15 CPU 核 / 16 GPU 核 / 24 GiB 统一内存，macOS 27.2，Python 3.12.13，MLX 0.32.4。','','## 动作模块筛选','','使用真实三图生成一次前缀 KV，仅重复动作去噪以隔离模块。默认 16×14 动作、10 步；每个配置预热 4 次，随机顺序交叉执行。以下不是完整请求耗时。target 是目标线程组数：按实际 M/N 计算分块，并向下调整到满足 K 与 group64 整除的分块数。','','|target（每项50次）|均值 ms|中位 ms|相对512归一化动作 max abs|','|---:|---:|---:|---:|']
for k,v in sweep.items():lines.append(f"|{k}|{v['mean']:.3f}|{v['median']:.3f}|{v['max_abs']:.8f}|")
lines+=['','单独提高 target 没有得到可靠优势，较多分块增加部分和缓冲及归约开销；这是实现结构解释，未通过 GPU 时间线分解。保留 target512。','','|布局 BM/BN/WM/WN（每项40次）|均值 ms|中位 ms|归一化动作 max abs|','|---|---:|---:|---:|']
for v in tiles.values():lines.append(f"|{'/'.join(map(str,v['tile']))}|{v['mean']:.3f}|{v['median']:.3f}|{v['max_abs']:.8f}|")
lines+=['','16/64/1/2 使用 64 线程、较短 M 块。在默认 M16 下减少不使用的行计算；布局仍按实际动作长度计算网格，不固定文本长度或动作长度。所有布局保留 NAX fragment 合法性检查。','','## 完整请求确认','','首轮测试在汇总时因参数变量被输出数组覆盖而退出，没有完整逐次计时，因此不纳入结论。已修正变量名并添加逐对落盘，保留失败日志。','','100 对 AB/BA 交替；共用模型权重，双方各首次运行及3次预热后计时。每次重新计算三图编码、文本、视觉前缀与动作，不复用跨请求特征。包括 CPU 动作输出，不包括模型加载、磁盘解码和输出保存。首次调用共享编译缓存，不能用它推导冷启动收益。','',f"原 Split-K 平均 **{r['base_ms']:.3f} ms**，compact 布局 **{r['fused_ms']:.3f} ms**，平均节省 **{r['saving_ms']:.3f} ms**（{100*r['saving_ms']/r['base_ms']:.2f}%）。配对 bootstrap 95% 区间 **[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] ms**，只反映本轮样本，不覆盖跨运行负载变化。",'',f"默认样本最终动作 max abs={r['max_abs']:.8f}（相对上一轮融合版）。",'','## 动态输入与模块计时','','|tokens|动作长度|步数|max abs|RMSE|','|---:|---:|---:|---:|---:|']
for v in r['dynamic']:lines.append(f"|{v['tokens']}|{v['horizon']}|{v['steps']}|{v['max_abs']:.8f}|{v['rmse']:.8f}|")
lines+=['','不同动作长度会改变网格和分块数，可能改变浮点归约顺序；不宣称任意形状逐值一致。此表是强制 compact 布局的诊断；推荐 adaptive 路径的额外覆盖见后文。没有闭环任务精度验证。','','|模块（独立同步3次）|原融合版 ms|compact ms|','|---|---:|---:|']
for k in r['modules']['base'][0]:
 a=np.mean([v[k] for v in r['modules']['base']]);b=np.mean([v[k] for v in r['modules']['fused']]);lines.append(f'|{k}|{a:.3f}|{b:.3f}|')
lines+=['','模块同步会改变调度，不应相加代替端到端测量。','','## 结论','']
if r['ci95'][0]>0:
 lines+=['本轮端到端配对呈正向收益，但优势有限且更长动作序列可能存在数值差异。已增加 adaptive 布局以消除长动作长度上的新增差异，未替换默认生产路径或上一轮入口。']
else:lines+=['完整请求的收益区间未排除零，不能认定 compact 稳定更快。原 Split-K 布局保持不变，新布局仅保留为实验选项。']
lines+=['','## 启动命令','','```bash','cd /Users/qklj/work/cx/ImageWAM_MLX','RUN_TAG=$(date +%Y%m%d_%H%M%S)','.venv/bin/python development/legacy_experiments/experiments_split_tune_20261104/infer.py \\', '  --layout adaptive --split-target 512 \\', '  --inputs examples/robotwin_ep0_frame0/inputs.npz \\', '  --prompt-file examples/robotwin_ep0_frame0/prompt.txt \\', '  --noise examples/robotwin_ep0_frame0/initial_noise_fp32.npy \\', '  --warmup 3 --output records/$RUN_TAG-split-compact','```','','此入口默认 --epilogue split、--profile speed。--layout original 对应上一轮融合版；--epilogue baseline 对应生产 speed。端到端耗时见完整请求确认。输出目录拒绝覆盖，记录参数、噪声和动作。','','筛选脚本为 sweep.py、tile_sweep.py；配对确认使用 tile_confirm.py --tile 1。它们拒绝覆盖已有 records 目录；重复运行需在脚本副本中设置新的 root。所有源码、生成头文件、许可证、原始逐次计时及输出均保留。','','下一步应优先减少动作层的激活拼接或归约读写，并先做同形状算子探针；不继续盲目增加 Split-K 分块数。']
validation=json.loads((root/'validation/completed.json').read_text());extra=json.loads((root/'validation/extra.json').read_text())
lines+=['','## 推荐 adaptive 路径的最终验证','',f"{validation['cases']} 组样本（31–79 token 长度覆盖及真实图像案例）全部与上一版 Split-K 逐值一致，max abs={validation['max_abs']}。额外 horizon={[v['horizon'] for v in extra]} 也逐值一致。长文本 193 token 的 horizon16 配对检查也一致。",'','adaptive 对 eligible Linear 在动作长度14–16时选择16/64/1/2布局，长度>16保留32/64/2/2；长度<=13沿用原官方路径。文本长度不参与此路由，无截断或固定文本要求。默认16步动作的内核与上述100对compact确认相同。','', '推荐命令已使用 --layout adaptive；--layout compact 仍可用于诊断，但长动作序列会改变数值。此处无新增误差指相对上一版量化融合路径，不代表与BF16/PyTorch一致。']
Path('README_SPLIT_TUNE_20261104.md').write_text('\n'.join(lines)+'\n')
readme=Path('README.md');backup=root/'README.before_tune.md'
if not backup.exists():shutil.copy2(readme,backup)
link='[Split-K 并行度与布局调优](README_SPLIT_TUNE_20261104.md)'
if link not in readme.read_text():readme.write_text(readme.read_text()+'\n- '+link+'：五种并行度、四种布局、100 对端到端确认与动态输入输出。\n')
(root/'source_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir() if p.is_file()},indent=2))
with tarfile.open(root/'review.tar.gz','w:gz') as t:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.npy','.npz','.log','.md'):t.add(p,arcname=str(p))
 for p in src.iterdir():
  if p.is_file():t.add(p,arcname=str(root/'source'/p.name))
print({k:v for k,v in r.items() if k not in ('pairs','modules','dynamic')})
