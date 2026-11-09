"""Shape-guarded two-SIMD candidate; safe four-SIMD fallback for other inputs."""
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
