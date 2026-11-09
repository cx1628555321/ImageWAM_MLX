"""Decode new real observations with the existing data Python; no MLX needed.

24 episodes not used in the previous 12-episode selection, three complete
16-action windows each, four fixed noise seeds. Train metadata is retained;
this is new-observation validation, not a claimed training hold-out split.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import av

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'records/visionzip_20261109/dataset'
OUT.mkdir(parents=True,exist_ok=False)
inputs=OUT/'inputs';inputs.mkdir()
data=ROOT.parent/'ImageWAM/data/robotwin2.0/robotwin2.0'
old=json.loads((ROOT/'records/robotwin_accuracy_20261106/manifest.json').read_text())
excluded={c['episode'] for c in old['cases']}
files=[p for p in sorted((data/'data').glob('chunk-*/*.parquet')) if int(p.stem.split('_')[-1]) not in excluded]
chosen=np.floor((np.arange(24)+.5)*len(files)/24).astype(int)
cams=('cam_high','cam_left_wrist','cam_right_wrist');cases=[]
for ix in chosen:
    p=files[ix];ep=int(p.stem.split('_')[-1]);assert ep not in excluded
    rows=pq.read_table(p).to_pydict();length=len(rows['action']);assert length>=32
    frames=[0,(length-16)//2,length-16];assert len(set(frames))==3
    videos={c:data/'videos'/p.parent.name/('observation.images.'+c)/(p.stem+'.mp4') for c in cams}
    images={f:{} for f in frames}
    for c,video in videos.items():
        with av.open(str(video)) as container:
            for i,frame in enumerate(container.decode(video=0)):
                if i in images:
                    assert abs(float(frame.time)-float(rows['timestamp'][i]))<.011
                    images[i][c]=frame.to_ndarray(format='rgb24')
                if i>=max(frames):break
    for f in frames:
        assert int(rows['frame_index'][f])==f and len(images[f])==3
        label=f'ep{ep}_frame{f}'
        state=np.asarray(rows['observation.state'][f],np.float32)
        target=np.asarray(rows['action'][f:f+16],np.float32)
        assert state.shape==(14,) and target.shape==(16,14)
        dest=inputs/(label+'.npz');np.savez_compressed(dest,**images[f],state=state,target=target)
        cases.append(dict(label=label,episode=ep,frame=f,task_index=int(rows['task_index'][f]),
            timestamp=float(rows['timestamp'][f]),parquet=str(p),videos={k:str(v) for k,v in videos.items()},
            sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    print('PREPARED',ep,frames,flush=True)
needed={c['task_index'] for c in cases};tasks={}
with (data/'meta/tasks.jsonl').open() as f:
    for line in f:
        row=json.loads(line)
        if row['task_index'] in needed:tasks[row['task_index']]=row['task']
for c in cases:
    c['instruction']=tasks[c['task_index']]
    c['prompt']="A video recorded from a robot's point of view executing the following instruction: "+c['instruction']
seeds=[42,123,2026,1109]
for seed in seeds:np.save(inputs/f'noise_{seed}.npy',np.random.default_rng(seed).standard_normal((1,16,14)).astype(np.float32))
(OUT/'manifest.json').write_text(json.dumps(dict(status='complete',dataset=str(data),
    split='train metadata; training hold-out status not verified',sampling='24 midpoint-spaced episodes after excluding original 12; start/middle/last full window',
    previous_episodes=sorted(excluded),cases=cases,seeds=seeds,steps=10,horizon=16),indent=2)+'\n')
print('COMPLETE',len(cases),'observations',len(cases)*len(seeds),'noise-observation pairs',flush=True)
