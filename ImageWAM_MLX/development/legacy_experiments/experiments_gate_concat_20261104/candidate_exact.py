import copy,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'development/legacy_experiments/experiments_split_tune_20261104'))
import mlx.core as mx
from tile_candidates import Epilogue,install as baseline
from imagewam_mlx.workspace import workspace_step
from gate_exact import gate_concat
class FusedGate(Epilogue):
 def single_post(self,x,att,mlp,p,m):
  return self.residual_linear(gate_concat(att,mlp),p+'.linear2',x,m[2])
def install(original,enabled=False):
 m=baseline(original,512,(16,64,1,2),adaptive=True)
 if enabled:
  m.action.__class__=FusedGate
  m._compiled_workspace_step=mx.compile(lambda cache,x,t,delta,pe:workspace_step(m,cache,x,t,delta,pe))
 return m
