"""Action-only FLUX.2 prefill: last-layer K/V retained, unused final output skipped.
Derived from MoT.prefill_flux2_video_cache; no image generation contract.
"""
import torch

def action_only_prefill(self, video_tokens: dict[str, torch.Tensor], video_freqs: dict[str, torch.Tensor], video_t_mod: dict[str, object], attention_mask: dict[str, torch.Tensor]) -> dict[str, object]:
    if self.block_protocol != 'flux2':
        raise ValueError("`prefill_flux2_video_cache` requires block_protocol='flux2'.")
    video_expert = self.mixtures['video']
    txt = video_tokens['txt']
    img = video_tokens['img']
    txt_pe = video_freqs['txt']
    img_pe = video_freqs['img']
    from flux2.model import apply_rope
    double_cache = []
    for layer_idx in range(int(getattr(video_expert, 'double_layers'))):
        block = video_expert.double_blocks[layer_idx]
        (q, k, v, pe_full, num_txt_tokens, mods) = block._prepare_qkv(img, txt, img_pe, txt_pe, video_t_mod['double_img'], video_t_mod['double_txt'])
        (q, k) = apply_rope(q, k, pe_full)
        flat_q = self._flux2_flatten_heads(q)
        flat_k = self._flux2_flatten_heads(k)
        flat_v = self._flux2_flatten_heads(v)
        double_cache.append({'k': flat_k, 'v': flat_v})
        mixed = self._mixed_attention(flat_q, flat_k, flat_v, attention_mask['double_joint'])
        (txt_attn, img_attn) = torch.split(mixed, [num_txt_tokens, img.shape[1]], dim=1)
        (img, txt) = block._apply_residuals(img, txt, img_attn, txt_attn, mods)
    video_stream = torch.cat([txt, img], dim=1)
    stream_pe = torch.cat([txt_pe, img_pe], dim=2)
    single_cache = []
    for layer_idx in range(int(getattr(video_expert, 'single_layers'))):
        block = video_expert.single_blocks[layer_idx]
        state = self._flux2_video_single_io(block, video_stream, stream_pe, video_t_mod['single'])
        single_cache.append({'k': state['k'], 'v': state['v']})
        if layer_idx == int(video_expert.single_layers) - 1:
            break
        mixed = self._mixed_attention(state['q'], state['k'], state['v'], attention_mask['single'])
        video_stream = block._out(state['residual_x'], mixed, state['mlp'], state['gate'])
    return {'double': double_cache, 'single': single_cache, 'txt_len': int(txt.shape[1]), 'img_len': int(img.shape[1]), 'unused_final_video_before_last_output': {'txt': video_stream[:, :txt.shape[1]], 'img': video_stream[:, txt.shape[1]:]}}
