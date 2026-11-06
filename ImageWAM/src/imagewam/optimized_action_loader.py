"""Strict, action-only FLUX.2 loader. No base visual file or second checkpoint read.

The original policy methods and kernels are retained. Qwen returns hidden_states
0..27 without the original terminal norm; ImageWAM consumes indices 9/18/27.
This module deliberately does not support training or image/video generation.
"""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from safetensors import safe_open

from imagewam.models.backbones.imagewam import ImageWAM


class UnsupportedBranch(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("The action-only loader did not load image/video generation weights")


class ActionOnlyImageWAM(ImageWAM):
    def load_checkpoint(self, path, optimizer=None):
        """Normal eval calls this after construction; weights are already restored."""
        p = Path(path).resolve()
        signature = (str(p), p.stat().st_size, p.stat().st_mtime_ns)
        if signature != self._restored_checkpoint or optimizer is not None:
            raise ValueError("Action-only model already restored; create a new model for another checkpoint")
        self.loading_metadata['normal_entry_restore_checks'] += 1
        return {'step': self.loading_metadata['checkpoint_step'], 'already_restored': True}

    def infer_joint(self, *args, **kwargs):
        raise RuntimeError("Action-only loader cannot generate future images/video")

    def infer_video(self, *args, **kwargs):
        raise RuntimeError("Action-only loader cannot generate future images/video")


def _plain(value):
    from omegaconf import DictConfig, OmegaConf
    return OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else dict(value or {})


def _restore(module, state, device, dtype):
    expected = module.state_dict()
    if set(expected) != set(state):
        raise ValueError(f"Strict key mismatch: missing={sorted(set(expected)-set(state))}, extra={sorted(set(state)-set(expected))}")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(f"Shape mismatch: {name}: {tensor.shape} != {expected[name].shape}")
    tensors = {name: tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)
               for name, tensor in state.items()}
    module.load_state_dict(tensors, strict=True, assign=True)
    return {'tensor_count': len(tensors), 'elements': sum(t.numel() for t in tensors.values()),
            'bytes': sum(t.numel()*t.element_size() for t in tensors.values())}


def _safe_state(path, names):
    with safe_open(str(path), framework='pt', device='cpu') as stream:
        return {name: stream.get_tensor(name) for name in names}


def create_imagewam_flux2_action_only(
    checkpoint_path, flux2_src_path, ae_model_path, qwen3_model_spec,
    flux2_model_path=None, variant='klein-base-4b', action_dit_config=None,
    action_dit_pretrained_path=None, proprio_dim=8, load_text_encoder=True,
    qwen_context_len=512, video_scheduler=None, action_scheduler=None, loss=None,
    mot_checkpoint_mixed_attn=True, mot_gqa_implementation='repeat',
    mot_force_flash_attention=False, pack_proprio_after_text=True,
    flux2_lora_config=None, model_dtype=torch.bfloat16, device='cuda', recorder=None,
):
    from imagewam.models.backbones.flux2_imports import ensure_flux2_importable
    ensure_flux2_importable(flux2_src_path)
    from flux2.model import Flux2, Klein4BParams
    from flux2.autoencoder import AutoEncoder, AutoEncoderParams
    from imagewam.models.backbones.flux2_video_expert import Flux2VideoExpert
    from imagewam.models.backbones.action_dit_flux2 import ActionDiTFlux2
    from imagewam.models.backbones.mot import MoT
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RotaryEmbedding

    if variant != 'klein-base-4b' or model_dtype != torch.bfloat16:
        raise ValueError('Validated loader supports klein-base-4b BF16 only')
    if _plain(flux2_lora_config).get('enabled') or action_dit_pretrained_path or not load_text_encoder:
        raise ValueError('Requires full policy checkpoint, online Qwen, no active LoRA or independent action checkpoint')
    span = lambda name: recorder.span(name, category='load') if recorder else nullcontext()
    source = Path(checkpoint_path).resolve(strict=True)
    metadata = {'loader': 'action_only', 'checkpoint_reads': 0, 'visual_initialization_file_reads': 0,
                'normal_entry_restore_checks': 0, 'checkpoint_path': str(source),
                'removed': ['visual.final_layer', 'ae.decoder', 'qwen.layers.27..35', 'qwen.norm', 'qwen.lm_head'],
                'qwen_hidden_indices': [9, 18, 27], 'components': {}}
    with span('optimized.policy.map_checkpoint'):
        payload = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        metadata['checkpoint_reads'] += 1
        if payload.get('checkpoint_format', 'full') != 'full':
            raise ValueError('Requires a full checkpoint, not LoRA deltas')
        state = payload['mot']
        metadata['checkpoint_step'] = int(payload['step'])
        # Wrapper aliases must point at the same source storage, not conflicting weights.
        visual_aliases = 0
        for key, tensor in state.items():
            for alias in ('mixtures.video.double_blocks.', 'mixtures.video.single_blocks.'):
                if key.startswith(alias):
                    canonical = key.replace('mixtures.video.', 'mixtures.video.transformer.', 1)
                    other = state[canonical]
                    if tensor.shape != other.shape or tensor.data_ptr() != other.data_ptr():
                        raise ValueError(f'Conflicting visual alias: {key}')
                    visual_aliases += 1
        metadata['checkpoint_visual_aliases_verified'] = visual_aliases
        allowed = ('mixtures.video.transformer.', 'mixtures.video.double_blocks.',
                   'mixtures.video.single_blocks.', 'mixtures.action.')
        if any(not key.startswith(allowed) for key in state):
            raise ValueError('Checkpoint contains unknown MoT parameter groups')

    with span('optimized.visual.construct_and_load'):
        with torch.device('meta'):
            transformer = Flux2(Klein4BParams()).to(dtype=model_dtype)
        removed_visual = {k: v for k,v in transformer.state_dict().items() if k.startswith('final_layer.')}
        transformer.final_layer = UnsupportedBranch()
        if getattr(transformer, 'use_guidance_embed', False):
            raise ValueError('Unexpected guidance embedding for current variant')
        prefix = 'mixtures.video.transformer.'
        visual_state = {k[len(prefix):]:v for k,v in state.items()
                        if k.startswith(prefix) and not k[len(prefix):].startswith('final_layer.')}
        metadata['components']['visual'] = _restore(transformer, visual_state, device, model_dtype)
        metadata['components']['visual']['removed_elements'] = sum(t.numel() for t in removed_visual.values())
        video = Flux2VideoExpert(transformer, variant)
        video.flux2_lora_enabled = False
        del visual_state

    with span('optimized.action.construct_and_load'):
        action_cfg = _plain(action_dit_config)
        action_cfg.update(num_heads=video.num_heads, attn_head_dim=video.attn_head_dim,
                          num_layers_double=video.double_layers, num_layers_single=video.single_layers)
        with torch.device('meta'):
            action = ActionDiTFlux2(**action_cfg).to(dtype=model_dtype)
        prefix = 'mixtures.action.'
        action_state = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        metadata['components']['action'] = _restore(action, action_state, device, model_dtype)
        del action_state

    with span('optimized.ae.construct_and_load'):
        with torch.device('meta'):
            ae = AutoEncoder(AutoEncoderParams()).to(dtype=model_dtype)
        removed_ae = sum(p.numel() for p in ae.decoder.parameters())
        ae.decoder = UnsupportedBranch()
        with safe_open(str(ae_model_path), framework='pt', device='cpu') as stream:
            names = [k for k in stream.keys() if not k.startswith('decoder.')]
        metadata['components']['ae'] = _restore(ae, _safe_state(ae_model_path, names), device, model_dtype)
        metadata['components']['ae']['removed_elements'] = removed_ae

    with span('optimized.qwen.construct_and_load'):
        directory = Path(qwen3_model_spec).resolve(strict=True)
        config = AutoConfig.from_pretrained(directory, local_files_only=True)
        metadata['qwen_original_tied_embedding_head'] = bool(config.tie_word_embeddings)
        if config.model_type != 'qwen3' or config.num_hidden_layers != 36 or config.hidden_size != 2560:
            raise ValueError('Requires the validated Qwen3-4B architecture')
        config.num_hidden_layers = 27
        if hasattr(config, 'layer_types'):
            config.layer_types = config.layer_types[:27]
        config._attn_implementation = 'sdpa'
        with torch.device('meta'):
            qwen = Qwen3Model(config).to(dtype=model_dtype)
        # Hidden state 27 must be raw layer-26 output, not the original final norm.
        qwen.norm = nn.Identity()
        # RoPE is a nonpersistent FP32 buffer in the native pretrained model.
        qwen.rotary_emb = Qwen3RotaryEmbedding(config, device='cpu').to(device=device)
        mapping = json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
        wanted = {'model.'+key: key for key in qwen.state_dict()}
        missing = set(wanted)-set(mapping)
        if missing:
            raise ValueError(f'Missing Qwen tensors: {sorted(missing)}')
        by_shard = {}
        for key in wanted:
            by_shard.setdefault(mapping[key], []).append(key)
        loaded = {}
        for shard, keys in sorted(by_shard.items()):
            with safe_open(str(directory/shard), framework='pt', device='cpu') as stream:
                for key in keys:
                    loaded[wanted[key]] = stream.get_tensor(key).to(device=device, dtype=model_dtype)
        metadata['components']['qwen'] = _restore(qwen, loaded, device, model_dtype)
        metadata['components']['qwen'].update(layers=27, attention_implementation=qwen.config._attn_implementation,
                                             rope_dtype=str(qwen.rotary_emb.inv_freq.dtype),
                                             skipped_checkpoint_keys=sorted(set(mapping)-set(wanted)))
        del loaded
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        text_encoder = SimpleNamespace(model=qwen.eval(), tokenizer=tokenizer, max_length=int(qwen_context_len))

    with span('optimized.assemble_and_proprio'):
        mot = MoT({'video': video, 'action': action}, mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
                  gqa_implementation=mot_gqa_implementation, force_flash_attention=mot_force_flash_attention)
        vs, ac, losses = _plain(video_scheduler), _plain(action_scheduler), _plain(loss)
        # Avoid allocating/randomly initializing proprio before its actual weights.
        model = ActionOnlyImageWAM(video_expert=video, action_expert=action, mot=mot, vae=ae,
            text_encoder=text_encoder, text_dim=7680, proprio_dim=None, device=device, torch_dtype=model_dtype,
            stack='flux2', qwen_context_len=qwen_context_len, pack_proprio_after_text=pack_proprio_after_text,
            video_train_shift=vs.get('train_shift',5), video_infer_shift=vs.get('infer_shift',5),
            video_num_train_timesteps=vs.get('num_train_timesteps',1000),
            action_train_shift=ac.get('train_shift',5), action_infer_shift=ac.get('infer_shift',5),
            action_num_train_timesteps=ac.get('num_train_timesteps',1000),
            loss_lambda_video=losses.get('lambda_video',1), loss_lambda_action=losses.get('lambda_action',1))
        if proprio_dim is None:
            raise ValueError('Current policy requires proprio encoder')
        with torch.device('meta'):
            proprio = nn.Linear(int(proprio_dim), 7680).to(dtype=model_dtype)
        metadata['components']['proprio'] = _restore(proprio, payload['proprio_encoder'], device, model_dtype)
        model.proprio_dim = int(proprio_dim)
        model.proprio_encoder = proprio
        model.model_paths = {'flux2': str(source), 'action_dit': str(source), 'ae': str(ae_model_path),
                             'qwen3': str(directory), 'flux2_src': str(flux2_src_path)}
        model._restored_checkpoint = (str(source), source.stat().st_size, source.stat().st_mtime_ns)
        model.loading_metadata = metadata
        metadata['shared_module_checks'] = {
            'visual_same_as_mot': model.video_expert is model.mot.mixtures['video'],
            'action_same_as_mot': model.action_expert is model.mot.mixtures['action'],
            'dit_same_as_mot': model.dit is model.mot,
            'visual_double_block_alias': video.double_blocks is transformer.double_blocks,
            'visual_single_block_alias': video.single_blocks is transformer.single_blocks,
            'complete_qwen_embedding_retained': list(qwen.embed_tokens.weight.shape) == [config.vocab_size, config.hidden_size],
        }
        if not all(metadata['shared_module_checks'].values()):
            raise RuntimeError('Shared module/embedding preservation check failed')
        metadata['qwen_embedding_shape'] = list(qwen.embed_tokens.weight.shape)
        metadata['qwen_embedding_source'] = 'model.embed_tokens.weight (shared with native LM head; retained in full)'
        for owner in (model, qwen):
            if any(t.is_meta for t in list(owner.parameters())+list(owner.buffers())):
                raise RuntimeError('Unmaterialized meta tensor remains')
        metadata['resident_parameter_elements'] = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in qwen.parameters())
        metadata['resident_parameter_bytes'] = sum(p.numel()*p.element_size() for p in model.parameters()) + sum(p.numel()*p.element_size() for p in qwen.parameters())
    del state, payload
    return model.eval()
