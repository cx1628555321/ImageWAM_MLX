from pathlib import Path
import shutil
out=Path('records/speed_final_20261103/docs_before_profile_update');out.mkdir(exist_ok=False)
notice='**量化分支更新：**不传参数或 `--profile accuracy` 使用原 BF16 精度路径；`--profile speed` 使用允许精度退化的混合量化配置。同期各 200 次稳态平均为 **291.082 / 277.018 ms**；默认 162 个数组逐字节一致。启动命令、算子类型/反量化审计、全部候选和输出误差以 [精度与速度分支报告](README_SPEED_QUANTIZATION_20261103.md) 为准。以下保留此前各轮结果，不能直接跨轮相减归因。\n\n'
for name in ('README.md','README_CURRENT_RUNTIME_20261103.md'):
 p=Path(name);shutil.copy2(p,out/name);s=p.read_text();first,rest=s.split('\n',1);p.write_text(first+'\n\n'+notice+rest.lstrip('\n'))
