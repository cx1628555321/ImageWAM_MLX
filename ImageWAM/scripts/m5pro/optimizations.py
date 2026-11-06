"""Opt-in inference-only optimizations. No text cache or image-size constants."""
from contextlib import contextmanager
import functools
import types
import torch

@contextmanager
def optimized(model, mode="dynamic"):
    """Modes: baseline, tail (bitwise candidate), trim, dynamic.
    trim retains original Qwen computation then removes masked trailing tokens.
    dynamic tokenizes without truncation and executes exactly the batch max length.
    All shape-dependent quantities are derived anew; no cross-input caches.
    """
    if mode not in {"baseline","tail","trim","dynamic"}:
        raise ValueError(mode)
    if mode=="baseline":
        yield
        return
    from action_prefill import action_only_prefill
    new_prefill=types.MethodType(action_only_prefill,model.mot)
    old_prefill=model.mot.prefill_flux2_video_cache
    old_encode=model._encode_flux2_prompts
    model.mot.prefill_flux2_video_cache=new_prefill
    @torch.no_grad()
    def encode(prompts):
        tok=model.text_encoder.tokenizer
        if tok.padding_side!="right":
            raise ValueError("Current padding-removal proof requires right padding")
        texts=[]
        for prompt in prompts:
            texts.append(tok.apply_chat_template([{"role":"user","content":str(prompt)}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False))
        inputs=tok(texts,return_tensors="pt",padding=True,truncation=False)
        count=inputs["input_ids"].shape[1]
        if count>model.text_encoder.model.config.max_position_embeddings:
            raise ValueError("Text exceeds Qwen positional capacity; refusing truncation")
        if mode=="trim":
            if count>model.text_encoder.max_length:
                raise ValueError("Reference padded mode would truncate this input; use dynamic")
            hidden,mask=old_encode(prompts)
            return hidden[:,:count],mask[:,:count]
        qwen=model.text_encoder.model
        outputs=qwen(input_ids=inputs["input_ids"].to(qwen.device),
            attention_mask=inputs["attention_mask"].to(qwen.device),
            output_hidden_states=True,use_cache=False)
        hidden=torch.stack([outputs.hidden_states[k] for k in [9,18,27]],dim=1)
        hidden=hidden.permute(0,2,1,3).reshape(hidden.shape[0],count,-1)
        return hidden.to(device=model.device,dtype=model.torch_dtype),inputs["attention_mask"].to(device=model.device,dtype=torch.bool)
    if mode in {"trim","dynamic"}:model._encode_flux2_prompts=encode
    try:
        yield
    finally:
        model.mot.prefill_flux2_video_cache=old_prefill
        model._encode_flux2_prompts=old_encode
