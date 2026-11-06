"""RoboTwin compact three-camera input and native normalization."""
import numpy as np
import torch
from PIL import Image

def prepare_observation(images, state, proc, device='mps', dtype=torch.bfloat16):
    head = np.asarray(Image.fromarray(images['cam_high']).resize((256, 192), Image.Resampling.BILINEAR))
    left = np.asarray(Image.fromarray(images['cam_left_wrist']).resize((128, 96), Image.Resampling.BILINEAR))
    right = np.asarray(Image.fromarray(images['cam_right_wrist']).resize((128, 96), Image.Resampling.BILINEAR))
    image = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    tensor = tensor * (2.0 / 255.0) - 1.0
    batch = {'state': {'default': torch.from_numpy(state.copy()).unsqueeze(0)}}
    batch = proc.action_state_transform(batch)
    batch = proc.normalizer.forward(batch)
    return (tensor, batch['state']['default'])

def denormalize_action(action, proc):
    return proc.normalizer.normalizers['action']['default'].backward(action.float().cpu().unsqueeze(0))[0].numpy()
