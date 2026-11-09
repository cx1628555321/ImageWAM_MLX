"""All observed RoboTwin text lengths and three additional real observations."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('../ImageWAM'));p.add_argument('--weights',type=Path,default=Path('weights/bf16'));p.add_argument('--output',type=Path,required=True)
p.add_argument('--profile',choices=['accuracy','speed'],default='accuracy')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
ref=a.source/'evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full';data=json.loads((ref/'comparison.json').read_text())
base=a.source/'evaluate_results/m5pro_robotwin/20261103_dynamic_profile';noise=np.load(base/'initial_noise_fp32.npy')
model=ImageWAM(a.weights,a.source,profile=a.profile);results=[]
def error(out,expected):
    delta=out.astype(np.float64)-expected.astype(np.float64)
    return {'max_abs':float(np.max(np.abs(delta))),'mae':float(np.mean(np.abs(delta))),'rmse':float(np.sqrt(np.mean(delta**2)))}
def persist():
    (a.output/'validation.json').write_text(json.dumps({'status':'running','cases':results},indent=2))
for case in data['cases']:
    label=case['label'];raw=np.load(ref/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'))
    images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};ids=model.tokenize(case['prompt'])
    if 'tokens' in case:assert ids.shape[1]==case['tokens']
    t=time.perf_counter();out,norm=model.predict(images,raw['state'],case['prompt'],noise);mx.synchronize();ms=(time.perf_counter()-t)*1000
    assert out.shape==(16,14) and np.isfinite(out).all()
    np.save(a.output/(label+'_action.npy'),out);np.save(a.output/(label+'_normalized.npy'),norm);np.save(a.output/(label+'_token_ids.npy'),ids)
    record={'label':label,'prompt':case['prompt'],'tokens':ids.shape[1],'wall_ms':ms,'input_sha256':{k:hashlib.sha256(raw[k].tobytes()).hexdigest() for k in raw.files},'comparison':{}}
    for mode in ('baseline','dynamic'):
        item=next(r for r in data['records'] if r['case']==label and r['mode']==mode)
        path=ref/f"{item['index']:04d}_{label}_{mode}"
        record['comparison'][mode]={'action':error(out,np.load(str(path)+'_action.npy')),'normalized':error(norm,np.load(str(path)+'_normalized.npy'))}
    results.append(record);persist();print(label,ids.shape[1],round(ms,2),record['comparison']['dynamic']['action'],flush=True)
# Exercise non-monotonic length changes, including lengths beyond the dataset.
raw=np.load(base/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')}
changes=[]
for prompt in [data['cases'][-4]['prompt'],data['cases'][0]['prompt'],'Move carefully. '*60,data['cases'][0]['prompt']]:
    ids=model.tokenize(prompt);out,norm=model.predict(images,raw['state'],prompt,noise)
    assert np.isfinite(out).all();np.save(a.output/f'length_transition_{len(changes)}.npy',out)
    changes.append({'tokens':ids.shape[1],'prompt':prompt})
# Numerical sensitivity checks ensure all raw camera/state/text inputs are consumed.
prompt=json.loads((base/'input_manifest.json').read_text())['prompt'];original,_=model.predict(images,raw['state'],prompt,noise);sensitivity={}
for key in (*images,'state','text'):
    im=dict(images);st=raw['state'].copy();pr=prompt
    if key in images:im[key]=np.zeros_like(im[key])
    elif key=='state':st=st+np.float32(.25)
    else:pr='A video recorded from a robot\'s point of view executing the following instruction: Move both arms to the left.'
    out,_=model.predict(im,st,pr,noise);delta=float(np.max(np.abs(out-original)));assert delta>0
    sensitivity[key]=delta;np.save(a.output/f'changed_{key}.npy',out)
summary={'status':'completed','inference_profile':model.inference_profile,'quantization':model.quantization_info,'speed_options':model.speed_options,'cases':results,'text_lengths':sorted({r['tokens'] for r in results if r['label'].startswith('length_')}),'length_transitions':changes,'input_sensitivity_max_abs':sensitivity,'summary':{mode:{'max_action_abs':max(r['comparison'][mode]['action']['max_abs'] for r in results),'mean_case_action_mae':float(np.mean([r['comparison'][mode]['action']['mae'] for r in results])),'max_normalized_abs':max(r['comparison'][mode]['normalized']['max_abs'] for r in results)} for mode in ('baseline','dynamic')},'note':'51 offline cases, not the entire dataset and not closed-loop task success. Per-case timings include first-use shape costs; not steady-state benchmarks.'}
(a.output/'validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary['summary']),flush=True)
