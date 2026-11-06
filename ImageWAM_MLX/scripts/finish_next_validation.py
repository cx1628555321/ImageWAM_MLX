"""Serialize validation and diagnostics with bounded experimental timeouts."""
import subprocess,json,sys
from pathlib import Path
root=Path('records/next_optimizations_20261103')
tasks=[('visual_fusion_diagnosis',[sys.executable,'scripts/diagnose_visual_fusion.py'],90),('visual_operator_profile',[sys.executable,'scripts/profile_visual_operators.py'],120),('vae_validation',[sys.executable,'scripts/run_next_candidate.py','--mode','vae_gn_silu','--entry','validate_cases','--output',str(root/'vae_validation')],240),('rms_validation',[sys.executable,'scripts/run_next_candidate.py','--mode','rms_action','--entry','validate_cases','--output',str(root/'rms_validation')],240),('gate_validation',[sys.executable,'scripts/run_next_candidate.py','--mode','visual_gate_concat','--entry','validate_cases','--output',str(root/'gate_validation')],240)]
r=[]
for name,cmd,timeout in tasks:
 print('START',name,flush=True)
 with (root/(name+'.log')).open('w') as log:
  try:status={'exit_code':subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode}
  except subprocess.TimeoutExpired:status={'timeout_s':timeout}
 r.append({'name':name,**status});(root/'validation_jobs.json').write_text(json.dumps(r,indent=2));print('DONE',r[-1],flush=True)
