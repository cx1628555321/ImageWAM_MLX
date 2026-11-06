import json,hashlib
from pathlib import Path
import numpy as np
r=Path('records/robotwin_accuracy_20261106');state=json.loads((r/'status.json').read_text());assert state['status']=='completed'
manifest=json.loads((r/'manifest.json').read_text());dirs={j['name']:Path(j['path']) for j in state['jobs'] if j['status']=='completed'}
variants=['accuracy','stock','runtime','native-w1','native-w4','action-bf16'];assert set(dirs)==set(variants)
def metric(diff):
 d=np.asarray(diff,np.float64);return dict(rmse=float(np.sqrt(np.mean(d*d))),mae=float(np.mean(np.abs(d))),max_abs=float(np.max(np.abs(d))),p95_abs=float(np.percentile(np.abs(d),95)),per_dimension_rmse=np.sqrt(np.mean(d*d,axis=(0,1))).tolist())
summary={};tokens=[]
for v in variants:
 report=json.loads((dirs[v]/'results.json').read_text());assert report['count']==72
 errs=[];normerrs=[];gt=[];case_rows=[];perstep=[]
 for c in manifest['cases']:
  path=r/'inputs'/(c['label']+'.npz');assert hashlib.sha256(path.read_bytes()).hexdigest()==c['sha256'];raw=np.load(path)
  for seed in manifest['seeds']:
   key=c['label']+f'_s{seed}';a=np.load(dirs[v]/(key+'.npz'));b=np.load(dirs['accuracy']/(key+'.npz'))
   d=a['action'].astype(np.float64)-b['action'];dn=a['normalized'].astype(np.float64)-b['normalized'];dg=a['action'].astype(np.float64)-raw['target']
   assert np.isfinite(d).all() and d.shape==(16,14)
   errs.append(d);normerrs.append(dn);gt.append(dg)
   case_rows.append(dict(key=key,rmse=float(np.sqrt(np.mean(d*d))),max_abs=float(np.max(np.abs(d))),demo_rmse=float(np.sqrt(np.mean(dg*dg)))))
 e=np.stack(errs);g=np.stack(gt);joints=[i for i in range(14) if i not in (6,13)]
 summary[v]=dict(reference=metric(e),normalized_reference=metric(np.stack(normerrs)),demonstration=metric(g),demo_joint_rmse=float(np.sqrt(np.mean(g[:,:,joints]**2))),demo_gripper_rmse=float(np.sqrt(np.mean(g[:,:,[6,13]]**2))),per_horizon_reference_rmse=np.sqrt(np.mean(e*e,axis=(0,2))).tolist(),worst_case=max(case_rows,key=lambda x:x['rmse']),case_rmse_median=float(np.median([c['rmse'] for c in case_rows])),cases=case_rows)
 tokens.extend(x['tokens'] for x in report['cases'])
result=dict(status='completed',reference='MLX full-model accuracy, MLX0.32.4; not ground-truth action',frames=36,episodes=12,seeds=2,per_variant=72,total_predictions=432,token_range=[min(tokens),max(tokens)],variants=summary)
(r/'accuracy_summary.json').write_text(json.dumps(result,indent=2))
lines=['# RoboTwin 六版本离线精度验证','','12个episode、36个真实帧、两份固定噪声；每版本72次，共432次。实际文本token范围：'+str(result['token_range'])+'。','参考：同输入、同噪声的全模型 MLX accuracy。所有输出均为有限值。','','| 版本 | 相对accuracy动作RMSE | MAE | 最大绝对差 | 归一化动作RMSE | 示范动作RMSE |','|---|---:|---:|---:|---:|---:|']
for v in variants:
 d=summary[v];a=d['reference'];lines.append(f"| {v} | {a['rmse']:.6f} | {a['mae']:.6f} | {a['max_abs']:.6f} | {d['normalized_reference']['rmse']:.6f} | {d['demonstration']['rmse']:.6f} |")
lines+=['','## 口径','', '- 相对accuracy误差反映实现/量化偏差；accuracy自身误差为0是参照定义，不能据此称模型完美。','- 示范动作按当前帧起连续16步对齐，不做相对动作变换；12个关节和2个夹爪量纲不同，分维数据见JSON。','- 数据元信息标记train；本次不是独立留出集，也未测RoboTwin仿真闭环成功率。','- 单次固定噪声的生成动作不一定唯一匹配示范，离线误差不能替代成功率。','- 输入来自真实episode，不用同图配随机指令充当独立样本；两个seed复用同一帧，不作为独立场景。','- 当前是精度测试，没有用异步执行或编译开销混杂的单请求时间更新性能排名。','', '## 复现','', '代码：`experiments/robotwin_accuracy_20261106/{prepare,worker,runner,summarize}.py`。完整输入、manifest、版本输出和逐案例统计均保存在本目录。','输入文件SHA256在汇总时重新核验；原始视频路径、episode、帧、时间戳、指令在manifest中。', '', '独立运行某版本：先设置项目根目录，使用旧Python运行accuracy/stock，新Python运行其余版本，再执行 `worker.py VARIANT NEW_OUTPUT_DIR`。']
(r/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:13]))
