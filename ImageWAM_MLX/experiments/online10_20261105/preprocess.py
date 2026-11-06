import numpy as np
from PIL import Image
import mlx.core as mx
from imagewam_mlx.ops import D
kernel=mx.fast.metal_kernel(name='wam_rgb_normalize_exact',input_names=['rgb'],output_names=['out'],source='''
uint i=thread_position_in_grid.x;
if(i>=COUNT)return;
volatile float scaled=float(rgb[i])*(2.0f/255.0f);
T rounded=T(scaled);
out[i]=T(float(rounded)-1.0f);
''')
def normalize(rgb):
    return kernel(inputs=[rgb],template=[('T',D),('COUNT',rgb.size)],grid=(rgb.size,1,1),threadgroup=(256,1,1),output_shapes=[rgb.shape],output_dtypes=[D])[0]
def preprocess(self,images,state):
    if np.asarray(state).shape != (14,) or not np.isfinite(state).all():raise ValueError('state must be finite [14]')
    for key in ('cam_high','cam_left_wrist','cam_right_wrist'):
        im=np.asarray(images[key])
        if im.ndim!=3 or im.shape[-1]!=3 or im.dtype!=np.uint8 or min(im.shape[:2])<1:raise ValueError(f'{key} must be nonempty uint8 HWC RGB')
    names=('cam_high','cam_left_wrist','cam_right_wrist');sizes=((256,192),(128,96),(128,96))
    imgs=[np.asarray(Image.fromarray(images[k]).resize(size,Image.Resampling.BILINEAR)) for k,size in zip(names,sizes)]
    rgb=np.concatenate([imgs[0],np.concatenate(imgs[1:],axis=1)],axis=0)
    x=normalize(mx.array(rgb[None]))
    scale,offset=self.norms['state'];st=np.clip(np.asarray(state,dtype=np.float32)*scale+offset,-5,5)
    return x,mx.array(st.reshape(1,1,14)).astype(D)
