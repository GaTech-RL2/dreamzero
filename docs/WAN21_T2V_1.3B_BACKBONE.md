# Training DreamZero with the Wan2.1-T2V-1.3B Backbone

This guide explains how to train DreamZero on DROID using **Wan2.1-T2V-1.3B** — a
small, pure **text-to-video** DiT — as the backbone, instead of the default
Wan2.1-I2V-14B or Wan2.2-TI2V-5B.

## Architecture differences

| Component | Wan2.1-I2V-14B | Wan2.2-TI2V-5B | Wan2.1-T2V-1.3B |
|-----------|----------------|----------------|-----------------|
| DiT dim | 5120 | 3072 | **1536** |
| DiT layers | 32 | 30 | **30** |
| DiT heads | 16 | 24 | **12** |
| FFN dim | 13824 | 14336 | **8960** |
| VAE latent channels | 16 | 48 | **16** |
| VAE spatial stride | 8× | 16× | **8×** |
| Model type | i2v | ti2v | **t2v** |
| in_dim / out_dim | 36 / 16 | 48 / 48 | **16 / 16** |
| First-frame conditioning | latent concat (`y`) | CLIP | **none (intrinsic clean_x)** |
| Image encoder (CLIP) | yes | yes (from 2.1) | **no** |
| Video resolution | 320×176 | 320×160 | **320×176** |
| Frame seqlen | 880 | 50 | **880** |

The same `CausalWanModel` class supports all three via configuration — no new
model class is required. T2V-1.3B reuses the **Wan2.1 VAE** (z=16, 8×), patch size
`(1,2,2)`, and 320×176 resolution of the 14B path, so `frame_seqlen` stays **880**
and the data config (`data=dreamzero/droid_relative`) is reused unchanged.

## How a text-to-video backbone "accounts for" having no image input

A pure T2V model has no image encoder and no first-frame input channels. DreamZero
nevertheless conditions on the current observation through an **intrinsic,
backbone-agnostic** path: the clean (un-noised) observed latents `clean_x` are fed
through the causal DiT with a **blockwise causal mask**, so each noisy block only
attends to *past clean frames* (`is_tf=True` in
`modules/wan_video_dit_action_casual_chunk.py`). This teacher-forcing path is
always on, independent of `model_type`.

The two I2V-only signals are simply **disabled** for T2V:

- **First-frame latent concat** (`y`, gated by `concat_first_frame_latent`) → set
  `concat_first_frame_latent: false`, `in_dim: 16`. The pretrained 16-channel
  `patch_embedding` then loads **exactly** (no weight surgery).
- **CLIP first-frame embedding** (`img_emb`, only built for `i2v`/`ti2v`) → the DiT
  builds no `img_emb`, and the action head sets `image_encoder_cfg: null` so no CLIP
  is instantiated, downloaded, or used.

So observation conditioning rides `clean_x` (autoregressive, blockwise-causal) plus
text cross-attention — exactly what a text-to-video world model should do. The
action/state registers, RoPE layout, KV cache, and closed-loop inference are
unchanged from the other backbones (see `WAN22_BACKBONE.md` for that shared logic).

## Prerequisites

```bash
# Wan2.1-T2V-1.3B bundles the DiT, the umt5-xxl T5 encoder, and the Wan2.1 VAE — no CLIP needed.
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./checkpoints/Wan2.1-T2V-1.3B

# DROID dataset in LeRobot format
huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./data/droid_lerobot
```

## Quick start

```bash
export WAN_CKPT_DIR=./checkpoints/Wan2.1-T2V-1.3B
export DROID_DATA_ROOT=./data/droid_lerobot

bash scripts/train/droid_training_wan21_t2v_1.3b.sh
```

The script runs a **full fine-tune** (`train_architecture=full`); at 1.3B this is
cheap and gives the strongest adaptation. It defaults to ZeRO-2 (no offload); set
`DEEPSPEED_CFG=zero2_offload` to trade speed for memory.

## Configuration details

The T2V config (`wan_flow_matching_action_tf_wan21_t2v.yaml`) extends the base
Wan2.1 action-head config and overrides only:

- **diffusion_model_cfg**: `model_type: t2v`, `dim: 1536`, `in_dim: 16`,
  `out_dim: 16`, `ffn_dim: 8960`, `num_heads: 12`, `num_layers: 30`,
  `freq_dim: 256`, `eps: 1e-6`, `concat_first_frame_latent: false`.
- **image_encoder_cfg**: `null` (no CLIP).
- VAE stays `WanVideoVAE` (z=16) and `frame_seqlen` stays 880 (inherited).

The action head (`wan_flow_matching_action_tf.py`) auto-selects the
`Wan-AI/Wan2.1-T2V-1.3B` HuggingFace repo for the DiT (single safetensors file),
VAE, and T5 weights when `model_type == 't2v'`, and skips the CLIP image encoder
entirely (instantiation, loading, and all forward/inference call sites are guarded
on `model_type`/`image_encoder is None`).

One model-code fix was needed because no prior DreamZero backbone used the `t2v`
cross-attention path (14B is i2v, 5B is ti2v): `WanT2VCrossAttention.forward`
(`modules/wan2_1_submodule.py`) made `context_lens` optional (default `None`), so it
matches the `self.cross_attn(norm3(x), context)` call in `CausalWanModel`'s block —
exactly like `WanI2VCrossAttention`. With `context_lens=None` the cross-attention
attends to all text tokens (padding is already zeroed in the action head's
`encode_prompt`). Validated end-to-end by `scripts/smoke_t2v_1.3b.py` (builds the
backbone, loads the real 1.3B weights, runs a forward → finite video + action output).

## File layout

```
dreamzero/
├── groot/vla/configs/model/dreamzero/action_head/
│   ├── wan_flow_matching_action_tf.yaml          # Wan2.1-I2V-14B (default)
│   ├── wan_flow_matching_action_tf_wan22.yaml    # Wan2.2-TI2V-5B
│   └── wan_flow_matching_action_tf_wan21_t2v.yaml # Wan2.1-T2V-1.3B  (this backbone)
├── scripts/train/
│   └── droid_training_wan21_t2v_1.3b.sh          # full-finetune script for this backbone
└── docs/
    ├── WAN22_BACKBONE.md
    └── WAN21_T2V_1.3B_BACKBONE.md                # this file
```
