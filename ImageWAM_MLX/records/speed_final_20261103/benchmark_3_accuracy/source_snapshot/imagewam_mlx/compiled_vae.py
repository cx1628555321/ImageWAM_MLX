"""Compile residual tails while leaving GN reduction arithmetic unchanged."""
import mlx.core as mx
from .fused_vae import FusedVAE

class CompiledResidualVAE(FusedVAE):
    @classmethod
    def from_vae(cls,original):
        obj=super().from_vae(original)
        obj._tails={}
        return obj

    def _tail(self,p):
        # The closure owns this layer name; MLX caches actual input shapes.
        def finish(x,y):
            y=self.conv(y,p+'.conv2',pad=1)
            if p+'.nin_shortcut.weight' in self.w:
                x=self.conv(x,p+'.nin_shortcut')
            return x+y
        return mx.compile(finish)

    def res(self,x,p):
        y=self.conv(self.gn_silu(x,p+'.norm1'),p+'.conv1',pad=1)
        y=self.gn_silu(y,p+'.norm2')
        if p not in self._tails:self._tails[p]=self._tail(p)
        return self._tails[p](x,y)
