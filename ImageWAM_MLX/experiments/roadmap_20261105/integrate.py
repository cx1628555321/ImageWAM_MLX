"""Integrate an opt-in, shape-guarded candidate; default speed remains unchanged."""
from pathlib import Path
import shutil,hashlib,json
r=Path('records/roadmap_20261105/integration_before');r.mkdir(exist_ok=False)
files=['imagewam_mlx/runtime.py','scripts/infer.py','scripts/benchmark.py']
for name in files:
 dst=r/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dst)
Path('imagewam_mlx/shape_visual.py').write_text('''"""Shape-guarded two-SIMD candidate; safe four-SIMD fallback for other inputs."""
from .device_visual import DeviceVisual
from .group_visual import GroupVisual

class ShapeVisual(GroupVisual):
    SHAPES = frozenset(((328,27648,3072),(328,3072,12288)))
    def __init__(self, original):
        super().__init__(original,64)
        self.info = dict(self.info, visual_shape_dispatch=True, visual_group_rows=0,
                         selected_shapes=sorted(self.SHAPES))
    def mm(self, **kwargs):
        template = dict(kwargs['template'])
        shape = (template['ROWS'],kwargs['inputs'][0].shape[0],template['XK'])
        method = GroupVisual.mm if shape in self.SHAPES else DeviceVisual.mm
        return method(self, **kwargs)
''')
p=Path(files[0]);s=p.read_text();old='device_visual=None,visual_group_rows=None):';assert old in s;s=s.replace(old,'device_visual=None,visual_group_rows=None,visual_shape_dispatch=False):',1)
s=s.replace('        self.visual_group_rows=0','        self.visual_shape_dispatch=False\n        if visual_shape_dispatch and visual_group_rows not in (None,0):raise ValueError("shape dispatch and visual_group_rows are mutually exclusive")\n        self.visual_group_rows=0',1)
needle='                                            self.visual_group_rows=visual_group_rows';assert needle in s
s=s.replace(needle,needle+'\n                                        elif visual_shape_dispatch:\n                                            from .shape_visual import ShapeVisual\n                                            self.visual=ShapeVisual(self.visual)\n                                            self.visual_shape_dispatch=True',1);p.write_text(s)
for name in files[1:]:
 p=Path(name);s=p.read_text();needle="p.add_argument('--visual-group-rows'";i=s.index(needle)
 s=s[:i]+"p.add_argument('--visual-shape-dispatch',action='store_true',help='Experimental shape-guarded two-SIMD visual projections; other shapes use original path')\n"+s[i:]
 s=s.replace('visual_group_rows=a.visual_group_rows,','visual_group_rows=a.visual_group_rows,visual_shape_dispatch=a.visual_shape_dispatch,',1)
 if name.endswith('infer.py'):s=s.replace("{'visual_group_rows':model.visual_group_rows", "{'visual_shape_dispatch':model.visual_shape_dispatch,'visual_group_rows':model.visual_group_rows",1)
 else:s=s.replace("report['visual_group_rows']=model.visual_group_rows", "report['visual_shape_dispatch']=model.visual_shape_dispatch;report['visual_group_rows']=model.visual_group_rows",1)
 p.write_text(s)
(r/'hashes.json').write_text(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in files},indent=2))
print('Integrated opt-in only')
