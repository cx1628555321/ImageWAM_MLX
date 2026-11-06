import os,sys
from pathlib import Path
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
root=Path.cwd();sys.path.insert(0,str(root));sys.path.insert(0,str(root/'experiments/background_20261105'));sys.path.insert(0,str(root/'experiments/prefix_tiles_20261106'))
import imagewam_mlx,imagewam_mlx.runtime as rt
source=Path(rt.__file__).read_text();assert "version('mlx')=='0.32.4'" in source
exec(compile(source.replace("version('mlx')=='0.32.4'",'True'),rt.__file__,'exec'),rt.__dict__)
imagewam_mlx.ImageWAM=rt.ImageWAM
source=Path('experiments/background_20261105/worker_template.py').read_text().replace('from candidates import install,action_int8','from candidate import install\nfrom candidates import action_int8')
source=source.replace("profile='speed',text_cache_size=1","profile='speed',text_cache_size=1,prefix_precision='all_int8'")
exec(compile(source,'prefix_tile_worker','exec'))
