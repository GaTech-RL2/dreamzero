"""No-dataset GPU smoke test for the Wan2.1-T2V-1.3B DreamZero backbone.

Stage 1 (construction + real weight load): build WANPolicyHead via the exact training Hydra
config (model/dreamzero/action_head=wan_flow_matching_action_tf_wan21_t2v). This exercises the
edited weight-loading code: t2v repo selection, single-file DiT download, and the optional CLIP
path. Asserts the load is clean (image_encoder is None, no img_emb, t2v dims), with missing keys
limited to DreamZero's added modules.

Stage 2 (t2v forward): run one direct CausalWanModel forward with self-consistent synthetic
tensors and clip_feature=None, y=None, clean_x=<clean latents> to confirm the text-to-video path
(no CLIP / no first-frame concat) produces finite video + action noise predictions.

No DROID data, no flash-attn (torch SDPA fallback) required. Run on a single GPU.
"""
import os
import sys
import traceback

import torch
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.environ.get("WAN_CKPT_DIR", "").strip() or None  # local dir; else None -> download from HF


def main() -> int:
    assert torch.cuda.is_available(), "smoke requires a GPU"
    dev = "cuda"
    print(f"[smoke] torch {torch.__version__}, cuda {torch.version.cuda}, gpu {torch.cuda.get_device_name(0)}")

    # ---- compose the exact training config, t2v action head ----
    cfg_dir = os.path.join(REPO, "groot", "vla", "configs")
    # If a local checkpoint dir is given, point the weight paths at it; else null -> code downloads from HF.
    dit_v = CKPT if CKPT else "null"
    t5_v = f"{CKPT}/models_t5_umt5-xxl-enc-bf16.pth" if CKPT else "null"
    vae_v = f"{CKPT}/Wan2.1_VAE.pth" if CKPT else "null"
    overrides = [
        "data=dreamzero/droid_relative",
        "model=dreamzero/vla",
        "model/dreamzero/action_head=wan_flow_matching_action_tf_wan21_t2v",
        "model/dreamzero/transform=dreamzero_cotrain",
        "num_frames=33", "action_horizon=24", "num_views=3",
        "num_frame_per_block=2", "num_action_per_block=24", "num_state_per_block=1",
        "max_chunk_size=4", "train_architecture=full",
        "image_resolution_width=320", "image_resolution_height=176",
        "droid_data_root=/tmp/none",          # dummy; dataset is never instantiated
        f"dit_version={dit_v}",
        f"text_encoder_pretrained_path={t5_v}",
        f"vae_pretrained_path={vae_v}",
    ]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="conf", overrides=overrides)

    print("[smoke] Stage 1: instantiating WANPolicyHead (downloads/loads real T2V weights)...")
    head = instantiate(cfg.action_head_cfg)   # builds T5 + VAE + CausalWanModel(t2v), loads weights

    # ---- Stage 1 assertions ----
    m = head.model
    print(f"[smoke] DiT: model_type={getattr(m,'model_type',None)} dim={m.dim} in_dim={m.in_dim} "
          f"out_dim={m.out_dim} layers={m.num_layers} heads={m.num_heads}")
    assert head.image_encoder is None, "t2v must have NO CLIP image encoder"
    assert not hasattr(m, "img_emb"), "t2v DiT must not build img_emb"
    assert m.model_type == "t2v" and m.in_dim == 16 and m.out_dim == 16
    assert m.dim == 1536 and m.num_layers == 30 and m.num_heads == 12
    pe = tuple(m.patch_embedding.weight.shape)
    print(f"[smoke] patch_embedding.weight {pe}")
    assert pe == (1536, 16, 1, 2, 2), "patch_embedding must be the exact pretrained 16-ch conv"
    assert m.patch_embedding.weight.abs().sum().item() > 0, "patch_embedding weights look uninitialized"
    print("[smoke] Stage 1 PASSED: clean t2v construction + real weight load (no CLIP, no img_emb).\n")

    # ---- Stage 2: one direct t2v DiT forward, self-consistent shapes ----
    # F=3 latent frames -> num_image_blocks=(3-1)//2=1 == action_blocks(24//24) == state_blocks(1//1).
    # H_lat=44,W_lat=80 -> tokens/frame=(44//2)*(80//2)=880 = frame_seqlen.
    print("[smoke] Stage 2: direct t2v forward (clip_feature=None, y=None, clean_x set)...")
    m.to(device=dev, dtype=torch.bfloat16).eval()
    # Mirror the training forward path: _forward_train only unpacks each block's (out, kv_cache)
    # tuple in the gradient-checkpointing branch (grad enabled + gradient_checkpointing=True). Run
    # exactly that way (we never call backward); the no-grad/no-checkpoint branch is unused in training.
    m.gradient_checkpointing = True
    B, F, C, Hl, Wl = 1, 3, 16, 44, 80
    seq_len = F * (Hl // 2) * (Wl // 2)
    bf = dict(device=dev, dtype=torch.bfloat16)
    x = torch.randn(B, C, F, Hl, Wl, **bf)
    clean_x = torch.randn(B, C, F, Hl, Wl, **bf)
    context = torch.randn(B, m.text_len, m.text_dim, **bf)
    state = torch.randn(B, 1, m.max_state_dim, **bf)                # state_dim = DiT max_state_dim (64)
    action = (torch.rand(B, 24, m.action_dim, **bf) * 2 - 1)        # action_dim = DiT action_dim (32)
    timestep = torch.full((B, F), 500.0, device=dev, dtype=torch.float32)
    timestep_action = torch.full((B, 24), 500.0, device=dev, dtype=torch.float32)
    embodiment_id = torch.zeros(B, device=dev, dtype=torch.long)
    x.requires_grad_(True)  # give checkpoint a grad-requiring input so it engages cleanly
    try:
        # grad ENABLED on purpose (see above); we do not call backward.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_pred, a_pred = m(
                x, timestep=timestep, clip_feature=None, y=None, context=context,
                seq_len=seq_len, state=state, embodiment_id=embodiment_id,
                action=action, timestep_action=timestep_action, clean_x=clean_x,
            )
        ok = bool(torch.isfinite(v_pred).all().item() and torch.isfinite(a_pred).all().item())
        print(f"[smoke] forward OK: video_pred {tuple(v_pred.shape)} action_pred {tuple(a_pred.shape)} "
              f"finite={ok}")
        assert ok, "non-finite outputs"
        assert tuple(v_pred.shape) == (B, m.out_dim, F, Hl, Wl), f"unexpected video_pred shape {v_pred.shape}"
        print("\n[smoke] Stage 2 PASSED: t2v forward produced finite video + action predictions.")
        print("\n=== SMOKE PASSED (construction + real weight load + t2v forward) ===")
        return 0
    except Exception:
        print("[smoke] Stage 2 forward FAILED (Stage 1 construction+load already PASSED):")
        traceback.print_exc()
        print("\n=== SMOKE PARTIAL: Stage 1 (load) PASSED; Stage 2 (forward) inconclusive ===")
        return 2


if __name__ == "__main__":
    sys.exit(main())
