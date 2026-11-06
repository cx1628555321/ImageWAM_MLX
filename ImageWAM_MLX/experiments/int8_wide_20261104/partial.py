from candidate import Candidate as Wide
from imagewam_mlx import prepacked_visual as pv
class Candidate(Wide):
    def mm(self,**kwargs):
        if kwargs['inputs'][0].shape[0]!=3072:
            return pv.kernel(**kwargs)
        return super().mm(**kwargs)
