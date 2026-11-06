"""Invert HF Diffusers' official FLUX2 VAE key/shape mapping.

Source mapping:
https://github.com/huggingface/diffusers/blob/main/scripts/convert_flux2_to_diffusers.py
Compatible native model:
https://github.com/black-forest-labs/flux2/blob/50fe516/src/flux2/autoencoder.py
Input asset (public official BFL):
black-forest-labs/FLUX.2-klein-base-4B @ a3b4f4849157f664bdbc776fd7453c2783562f4d
vae/diffusion_pytorch_model.safetensors

This changes names and attention kernel shapes only, preserving published dtype.
"""
import argparse
import re


def diffusers_to_native_name(key):
    key = key.replace("encoder.conv_norm_out.", "encoder.norm_out.")
    key = key.replace("decoder.conv_norm_out.", "decoder.norm_out.")
    if key.startswith("quant_conv."):
        key = "encoder." + key
    elif key.startswith("post_quant_conv."):
        key = "decoder." + key
    key = re.sub(r"encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.",
                 lambda m: f"encoder.down.{m[1]}.block.{m[2]}.", key)
    key = re.sub(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0\.",
                 lambda m: f"encoder.down.{m[1]}.downsample.", key)
    key = re.sub(r"decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.",
                 lambda m: f"decoder.up.{3-int(m[1])}.block.{m[2]}.", key)
    key = re.sub(r"decoder\.up_blocks\.(\d+)\.upsamplers\.0\.",
                 lambda m: f"decoder.up.{3-int(m[1])}.upsample.", key)
    key = re.sub(r"\.mid_block\.resnets\.(\d+)\.",
                 lambda m: f".mid.block_{int(m[1])+1}.", key)
    key = key.replace(".conv_shortcut.", ".nin_shortcut.")
    key = key.replace(".mid_block.attentions.0.group_norm.", ".mid.attn_1.norm.")
    key = key.replace(".mid_block.attentions.0.to_out.0.", ".mid.attn_1.proj_out.")
    for qkv in "qkv":
        key = key.replace(f".mid_block.attentions.0.to_{qkv}.", f".mid.attn_1.{qkv}.")
    return key


def convert_state_dict(source, expected):
    converted = {}
    for source_name, value in source.items():
        target_name = diffusers_to_native_name(source_name)
        if target_name in converted:
            raise ValueError(f"Duplicate mapping: {source_name} -> {target_name}")
        if target_name not in expected:
            raise ValueError(f"Unrecognized mapping: {source_name} -> {target_name}")
        shape = tuple(expected[target_name].shape)
        if tuple(value.shape) != shape:
            if (re.search(r"\.mid\.attn_1\.(?:q|k|v|proj_out)\.weight$", target_name)
                    and value.ndim == 2 and shape == (*value.shape, 1, 1)):
                value = value[:, :, None, None]
            else:
                raise ValueError(f"Shape mismatch {source_name}: {value.shape} vs {shape}")
        converted[target_name] = value.contiguous()
    missing = sorted(set(expected) - set(converted))
    if missing:
        raise ValueError(f"Missing native parameters: {missing}")
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file, save_file
    from flux2.autoencoder import AutoEncoder, AutoEncoderParams

    # Meta allocation validates all names/shapes without allocating full weights.
    with torch.device("meta"):
        native = AutoEncoder(AutoEncoderParams())
    source = load_file(args.source, device="cpu")
    state = convert_state_dict(source, native.state_dict())
    native.load_state_dict(state, strict=True, assign=True)
    save_file(state, args.destination, metadata={
        "source_repo": "black-forest-labs/FLUX.2-klein-base-4B",
        "source_revision": "a3b4f4849157f664bdbc776fd7453c2783562f4d",
        "source_file": "vae/diffusion_pytorch_model.safetensors",
        "conversion": "Inverse official HF Diffusers FLUX2 VAE mapping; dtype preserved",
    })
    print(f"Converted {len(state)} tensors; strict native load passed: {args.destination}")


if __name__ == "__main__":
    main()
