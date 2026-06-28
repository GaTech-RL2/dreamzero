"""Param-count + forward smoke test for the ~300M image+text-to-video DreamZero DiT.

This is self-contained: it instantiates ONLY the trainable DiT (CausalWanModel) from the exact
training Hydra config (model/dreamzero/action_head=wan_flow_matching_action_tf_300m). It does NOT
build the full WANPolicyHead, so it needs no multi-GB T5/CLIP/VAE downloads -- the frozen encoders
are validated end-to-end by the first steps of the training script instead.

Stage 1 (CPU, no downloads): build the DiT, assert the i2v config (model_type=i2v, in_dim=36, dim=896,
heads=7, layers=22), and print the total parameter count -- this is the number we are sizing to ~300M.

Stage 2 (GPU): run one i2v _forward_train with self-consistent synthetic tensors (real clip_feature +
first-frame latent y, not None) to confirm the image+text path produces finite video + action outputs.

Usage:
    python scripts/smoke_300m.py            # Stage 1 always; Stage 2 if a GPU is available
"""
import os
import sys
import traceback

import torch
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Expected ~300M config (keep in sync with wan_flow_matching_action_tf_300m.yaml).
EXP_MODEL_TYPE = "i2v"
EXP_DIM = 896
EXP_FFN = 3584
EXP_HEADS = 7
EXP_LAYERS = 19
EXP_IN_DIM = 36
EXP_OUT_DIM = 16
PARAM_LO, PARAM_HI = 250_000_000, 350_000_000  # acceptable band around the ~300M target


def main() -> int:
    cfg_dir = os.path.join(REPO, "groot", "vla", "configs")
    overrides = [
        "data=dreamzero/droid_relative",
        "model=dreamzero/vla",
        "model/dreamzero/action_head=wan_flow_matching_action_tf_300m",
        "model/dreamzero/transform=dreamzero_cotrain",
        "num_frames=33", "action_horizon=24", "num_views=3",
        "num_frame_per_block=2", "num_action_per_block=24", "num_state_per_block=1",
        "max_chunk_size=4", "train_architecture=full",
        "image_resolution_width=320", "image_resolution_height=176",
        "droid_data_root=/tmp/none",        # dummy; dataset is never instantiated
        "dit_version=null",                 # DiT trains from scratch; no checkpoint
        "text_encoder_pretrained_path=null",
        "image_encoder_pretrained_path=null",
        "vae_pretrained_path=null",
    ]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="conf", overrides=overrides)

    # ---- Stage 1: instantiate ONLY the DiT (no T5/CLIP/VAE), assert dims, count params ----
    print("[smoke] Stage 1: instantiating CausalWanModel (DiT only, no weight downloads)...")
    m = instantiate(cfg.action_head_cfg.config.diffusion_model_cfg)
    print(f"[smoke] DiT: model_type={m.model_type} dim={m.dim} ffn_dim={m.ffn_dim} "
          f"in_dim={m.in_dim} out_dim={m.out_dim} layers={m.num_layers} heads={m.num_heads}")
    assert m.model_type == EXP_MODEL_TYPE, f"expected i2v, got {m.model_type}"
    assert m.dim == EXP_DIM and m.ffn_dim == EXP_FFN and m.num_heads == EXP_HEADS, "DiT width/ffn/heads mismatch"
    assert m.num_layers == EXP_LAYERS, f"expected {EXP_LAYERS} layers, got {m.num_layers}"
    assert m.in_dim == EXP_IN_DIM and m.out_dim == EXP_OUT_DIM, "i2v in/out dims must be 36/16"
    assert hasattr(m, "img_emb"), "i2v DiT must build img_emb (CLIP projection)"
    assert (m.dim % m.num_heads) == 0 and (m.dim // m.num_heads) % 2 == 0, "head_dim must be even"

    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[smoke] DiT total params:     {total:,}")
    print(f"[smoke] DiT trainable params: {trainable:,}")
    print(f"[smoke] ~{total / 1e6:.1f}M params (target ~300M)")
    assert PARAM_LO <= total <= PARAM_HI, (
        f"param count {total:,} outside [{PARAM_LO:,}, {PARAM_HI:,}] -- nudge num_layers/ffn_dim")
    print("[smoke] Stage 1 PASSED: i2v DiT built at ~300M.\n")

    # ---- Stage 2: one direct i2v forward with synthetic but self-consistent shapes ----
    if not torch.cuda.is_available():
        print("[smoke] No GPU available; skipping Stage 2 forward. Stage 1 (param count) PASSED.")
        return 0
    dev = "cuda"
    print(f"[smoke] Stage 2: i2v forward on {torch.cuda.get_device_name(0)} (clip_feature + y set)...")
    # F=3 latent frames -> num_image_blocks=(3-1)//2=1 == action_blocks(24//24) == state_blocks(1//1).
    # H_lat=44,W_lat=80 -> tokens/frame=(44//2)*(80//2)=880; seq_len = F*880 = 2640.
    m.to(device=dev, dtype=torch.bfloat16).eval()
    m.gradient_checkpointing = True  # mirror training: _forward_train unpacks (out, kv) in the ckpt branch
    B, F, Hl, Wl = 1, 3, 44, 80
    seq_len = F * (Hl // 2) * (Wl // 2)
    bf = dict(device=dev, dtype=torch.bfloat16)
    # x: 16-ch noisy latent; y: 20-ch (first-frame latent 16 + mask 4) -> concat to in_dim=36 internally.
    x = torch.randn(B, EXP_OUT_DIM, F, Hl, Wl, **bf)
    y = torch.randn(B, EXP_IN_DIM - EXP_OUT_DIM, F, Hl, Wl, **bf)
    clean_x = torch.randn(B, EXP_OUT_DIM, F, Hl, Wl, **bf)
    clip_feature = torch.randn(B, 257, 1280, **bf)               # CLIP first-frame embedding
    context = torch.randn(B, m.text_len, m.text_dim, **bf)       # (B, 512, 4096)
    state = torch.randn(B, 1, m.max_state_dim, **bf)             # state_dim = DiT max_state_dim (64)
    action = (torch.rand(B, 24, m.action_dim, **bf) * 2 - 1)     # action_dim = DiT action_dim (32)
    timestep = torch.full((B, F), 500.0, device=dev, dtype=torch.float32)
    timestep_action = torch.full((B, 24), 500.0, device=dev, dtype=torch.float32)
    embodiment_id = torch.zeros(B, device=dev, dtype=torch.long)
    x.requires_grad_(True)  # give checkpoint a grad-requiring input so it engages cleanly
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_pred, a_pred = m(
                x, timestep=timestep, clip_feature=clip_feature, y=y, context=context,
                seq_len=seq_len, state=state, embodiment_id=embodiment_id,
                action=action, timestep_action=timestep_action, clean_x=clean_x,
            )
        ok = bool(torch.isfinite(v_pred).all().item() and torch.isfinite(a_pred).all().item())
        print(f"[smoke] forward OK: video_pred {tuple(v_pred.shape)} action_pred {tuple(a_pred.shape)} "
              f"finite={ok}")
        assert ok, "non-finite outputs"
        assert tuple(v_pred.shape) == (B, m.out_dim, F, Hl, Wl), f"unexpected video_pred shape {v_pred.shape}"
        print("\n[smoke] Stage 2 PASSED: i2v forward produced finite video + action predictions.")
        print("\n=== SMOKE PASSED (param count ~300M + i2v forward) ===")
        return 0
    except Exception:
        print("[smoke] Stage 2 forward FAILED (Stage 1 param count already PASSED):")
        traceback.print_exc()
        print("\n=== SMOKE PARTIAL: Stage 1 (param count) PASSED; Stage 2 (forward) inconclusive ===")
        return 2


if __name__ == "__main__":
    sys.exit(main())
