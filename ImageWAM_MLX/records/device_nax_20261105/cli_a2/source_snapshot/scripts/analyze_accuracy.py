"""Aggregate saved cross-framework actions; no new inference or chosen tolerance."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
base=Path('records/validation_v1');reference=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full')
v=json.loads((base/'validation.json').read_text());r=json.loads((reference/'comparison.json').read_text());result={}
for mode in ('dynamic','baseline'):
 predictions=[];expected=[];labels=[]
 for case in v['cases']:
  label=case['label'];row=next(x for x in r['records'] if x['case']==label and x['mode']==mode)
  predictions.append(np.load(base/(label+'_action.npy')));expected.append(np.load(reference/f"{row['index']:04d}_{label}_{mode}_action.npy"));labels.append(label)
 x=np.stack(predictions).astype(np.float64);y=np.stack(expected).astype(np.float64);delta=x-y;absolute=abs(delta)
 def summary(d,ref):
  ab=abs(d);return {'mae':float(ab.mean()),'rmse':float(np.sqrt(np.mean(d*d))),'max_abs':float(ab.max()),'p50_abs':float(np.percentile(ab,50)),'p95_abs':float(np.percentile(ab,95)),'p99_abs':float(np.percentile(ab,99)),'relative_l2':float(np.linalg.norm(d)/np.linalg.norm(ref))}
 idx=np.unravel_index(absolute.argmax(),absolute.shape)
 result[mode]={'aggregate':summary(delta,y),'per_action_dimension':{str(i):summary(delta[:,:,i],y[:,:,i]) for i in range(14)},'per_horizon_step':{str(i):summary(delta[:,i,:],y[:,i,:]) for i in range(16)},'worst':{'case':labels[idx[0]],'horizon_step':int(idx[1]),'action_dimension':int(idx[2]),'mlx_value':float(x[idx]),'mps_value':float(y[idx])},'exact_element_fraction':float((delta==0).mean()),'per_case':{label:summary(delta[i],y[i]) for i,label in enumerate(labels)}}
result['scope']={'cases':51,'elements':51*16*14,'comparison':'saved BF16 action outputs, original and dynamic MPS references; no CUDA/FP32 gold reference and no closed-loop success evaluation','threshold':'No application acceptance tolerance has been specified; numerical differences are reported without an accuracy pass claim.'}
(a.output/'accuracy.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v['aggregate'] for k,v in result.items() if 'aggregate' in v},indent=2))
