"""Interleaved original/optimized tests on unchanged real RoboTwin inputs."""
import argparse,json,time,sys,hashlib,datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT/"src",ROOT/"third_party/flux2/src"):sys.path.insert(0,str(path))
from optimizations import optimized
from robotwin_io import prepare_observation,denormalize_action
from timing import Recorder,stats,write
from audit import compare,save_provenance
from imagewam.optimized_action_loader import create_imagewam_flux2_action_only
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from hydra.utils import instantiate
from omegaconf import OmegaConf

p=argparse.ArgumentParser();p.add_argument("--full",action="store_true");args=p.parse_args()
run=ROOT/"evaluate_results/m5pro_robotwin"/(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+"_dynamic_"+("full" if args.full else "smoke"))
run.mkdir(parents=True,exist_ok=False)
old=ROOT/"evaluate_results/m5pro_robotwin/20261103_102606_verified"
base=json.loads((old/"benchmark.json").read_text())
inputs=np.load(old/"inputs.npz");images={k:inputs[k] for k in ("cam_high","cam_left_wrist","cam_right_wrist")};state=inputs["state"]
cfg=OmegaConf.create({"data":OmegaConf.load(ROOT/"configs/data/robotwin_omnigen2.yaml")})
proc=instantiate(cfg.data.train.processor).eval()
proc.set_normalizer_from_stats(load_dataset_stats_from_json(str(old/"dataset_stats.json")))
rec=Recorder()
with rec.span("load",category="load"):
 model=create_imagewam_flux2_action_only(**base["model_config"],model_dtype=torch.bfloat16,recorder=rec).eval()
records=[];cases=[];result={"status":"running","load_spans":rec.records,"path":str(run),"configuration":base["model_config"]}
def persist():result.update(records=records,cases=cases);write(run/"comparison.json",result)
def infer(mode,prompt,images=images,state=state,label="base",phase="test"):
 with optimized(model,mode),torch.no_grad():
  torch.mps.synchronize();start=time.perf_counter()
  image,proprio=prepare_observation(images,state,proc)
  pred=model.infer_action_flux2(prompt=prompt,input_image=image,proprio=proprio,action_horizon=16,num_inference_steps=10,seed=42,rand_device="cpu")["action"]
  action=denormalize_action(pred,proc)
  torch.mps.synchronize();ms=(time.perf_counter()-start)*1000
 assert action.shape==(16,14) and np.isfinite(action).all()
 index=len(records)
 np.save(run/f"{index:04d}_{label}_{mode}_normalized.npy",pred.numpy())
 np.save(run/f"{index:04d}_{label}_{mode}_action.npy",action)
 records.append({"mode":mode,"case":label,"phase":phase,"ms":ms,"index":index})
 persist();print(label,phase,mode,round(ms,3),flush=True)
 return action,pred.numpy()
try:
 if not args.full:
  results={}
  for mode in ["baseline","tail","trim","dynamic"]:
   for i in range(3):
    x,n=infer(mode,base["input"]["prompt"],phase="smoke")
   results[mode]={"action":x,"normalized":n}
  result["errors"]={mode:{"action":compare(v["action"],results["baseline"]["action"]),"normalized":compare(v["normalized"],results["baseline"]["normalized"])} for mode,v in results.items()}
 else:
  inventory=json.loads((ROOT/"development_records/m5pro_robotwin/20261103_dynamic_optimization/text_lengths.json").read_text())
  # Validate every distinct observed task length, not only a handpicked short prompt.
  for length,example in sorted(inventory["examples_by_length"].items(),key=lambda kv:int(kv[0])):
   label="length_"+str(length);prompt=example["prompt"]
   a,n=infer("baseline",prompt,label=label)
   b,m=infer("dynamic",prompt,label=label)
   cases.append({"label":label,"task_index":example["task_index"],"prompt":prompt,"tokens":int(length),
                 "action":compare(b,a),"normalized":compare(m,n)})
   persist()
  # Different real observations and naturally associated task text.
  import av,pandas as pd
  data=ROOT/"data/robotwin2.0/robotwin2.0"
  info=json.loads((data/"meta/info.json").read_text())
  for episode,frame_index in [(0,20),(1000,0),(10000,0)]:
   row=pd.read_parquet(data/info["data_path"].format(episode_chunk=episode//1000,episode_index=episode)).iloc[frame_index]
   task_id=int(row["task_index"])
   with (data/"meta/tasks.jsonl").open() as f:
    task=next(json.loads(line) for line in f if json.loads(line)["task_index"]==task_id)
   prompt="A video recorded from a robot's point of view executing the following instruction: "+task["task"]
   im={}
   for cam in images:
    path=data/info["video_path"].format(episode_chunk=episode//1000,episode_index=episode,video_key="observation.images."+cam)
    with av.open(str(path)) as container:
     for idx,frm in enumerate(container.decode(video=0)):
      if idx==frame_index:im[cam]=frm.to_ndarray(format="rgb24");break
   st=np.asarray(row["observation.state"],dtype=np.float32);label=f"ep{episode}_frame{frame_index}"
   np.savez(run/(label+"_inputs.npz"),**im,state=st)
   a,n=infer("baseline",prompt,im,st,label)
   b,m=infer("dynamic",prompt,im,st,label)
   cases.append({"label":label,"task_index":task_id,"prompt":prompt,"action":compare(b,a),"normalized":compare(m,n)})
   persist()
  # Alternating long/short input already exercised shape transitions above.
  prompt=base["input"]["prompt"]
  for i in range(6):
   for mode in ("baseline","dynamic"):infer(mode,prompt,phase="warmup")
  for i in range(20):
   for mode in (("baseline","dynamic") if i%2==0 else ("dynamic","baseline")):
    infer(mode,prompt,phase="steady")
  result["steady"]={mode:stats(r["ms"] for r in records if r["mode"]==mode and r["phase"]=="steady") for mode in ("baseline","dynamic")}
  result["max_action_abs"]=max(c["action"]["max_abs"] for c in cases)
  result["max_normalized_abs"]=max(c["normalized"]["max_abs"] for c in cases)
  np.savez(run/"base_inputs.npz",**images,state=state)
 result["status"]="completed";persist()
 print(json.dumps({k:v for k,v in result.items() if k not in ["records","cases","configuration","load_spans"]}),flush=True)
except BaseException:
 import traceback
 result["status"]="failed";result["error"]=traceback.format_exc();persist();raise
