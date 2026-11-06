"""Validation-only hashes and parameter-owner coverage, outside measured latency."""
from contextlib import ExitStack, contextmanager
import hashlib


def tensor_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        data = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': hashlib.sha256(data).hexdigest()}
    if isinstance(value, dict):
        return {str(k): tensor_tree(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_tree(v) for v in value]
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    raise TypeError(type(value))


@contextmanager
def capture(model, stages, coverage=None):
    targets = [('text_features', model, '_encode_flux2_prompts'),
               ('proprio_context', model, '_append_proprio_to_context_if_enabled'),
               ('ae_latents', model.vae, 'encode'),
               ('visual_kv', model.mot, 'prefill_flux2_video_cache'),
               ('action_transformer', model.mot, 'forward_flux2_action_with_video_cache')]
    with ExitStack() as stack:
        for name, owner, method in targets:
            own, prior, original = method in vars(owner), vars(owner).get(method), getattr(owner, method)
            def wrapped(*args, _name=name, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                stages.setdefault(_name, []).append(tensor_tree(result))
                return result
            def restore(_owner=owner, _method=method, _own=own, _prior=prior):
                if _own:
                    setattr(_owner, _method, _prior)
                else:
                    delattr(_owner, _method)
            setattr(owner, method, wrapped)
            stack.callback(restore)
        if coverage is not None:
            seen = set()
            for prefix, root in [('policy', model), ('qwen', model.text_encoder.model)]:
                for name, module in root.named_modules():
                    if id(module) in seen:
                        continue
                    seen.add(id(module))
                    params = list(module.named_parameters(recurse=False))
                    if not params:
                        continue
                    key = prefix + '.' + name
                    coverage[key] = {'calls': 0, 'parameter_elements': sum(p.numel() for _,p in params),
                                     'parameters': [k for k,_ in params]}
                    def record(_module, _args, _key=key):
                        coverage[_key]['calls'] += 1
                    handle = module.register_forward_pre_hook(record)
                    stack.callback(handle.remove)
        yield
