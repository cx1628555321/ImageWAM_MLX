"""Capture only relevant device/software fields, excluding serial identifiers."""
import json,platform,subprocess,sys,importlib.metadata,datetime
from pathlib import Path
import mlx.core as mx
hw=json.loads(subprocess.check_output(['system_profiler','SPHardwareDataType','SPDisplaysDataType','-json']))
h=hw['SPHardwareDataType'][0];gpu=hw['SPDisplaysDataType'][0]
out={'captured_at':datetime.datetime.now().astimezone().isoformat(),'hardware':{k:h.get(k) for k in ('machine_name','machine_model','chip_type','number_processors','physical_memory')},'physical_memory_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'])),'gpu':{k:gpu.get(k) for k in ('sppci_model','sppci_cores','spdisplays_metal')},'mlx_device':mx.device_info(),'software':{'python':sys.version,'platform':platform.platform(),'macos':platform.mac_ver()[0],'packages':{p:importlib.metadata.version(p) for p in ('mlx','mlx-metal','numpy','transformers','pillow','safetensors','tokenizers')}},'memory_note':'Apple Silicon unified memory shared by CPU/GPU; no independent VRAM. MLX memory metrics are not total system memory or directly comparable to MPS allocator counters.'}
p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)
with p.open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
