"""GPU train-step test for the ~300M i2v DreamZero DiT.

Proves the TRAINING computation works on real hardware: builds the from-scratch ~293M CausalWanModel,
then runs forward -> MSE loss (video + action) -> backward -> optimizer step, twice, on synthetic but
self-consistent i2v inputs. No dataset, no checkpoint, no flash_attn needed:
  * self-attention uses the SDPA backend  (ATTENTION_BACKEND=torch)
  * cross-attention uses attention.py's SDPA fallback (auto when flash_attn is absent on CUDA)

Requires a CUDA GPU (the cross-attention asserts q.device.type == 'cuda'). Run via Slurm, e.g.:
  srun --partition=overcap --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=00:20:00 \
       env ATTENTION_BACKEND=torch .venv/bin/python scripts/smoke_train_step_300m.py
"""
import os
os.environ.setdefault("ATTENTION_BACKEND", "torch")  # force SDPA self-attn (must be set before build)
import sys
import time
import torch
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    if not torch.cuda.is_available():
        print("[train-step] ERROR: needs a CUDA GPU (cross-attention is CUDA-only). Submit via Slurm.")
        return 1
    dev = "cuda"
    print(f"[train-step] torch {torch.__version__} | gpu {torch.cuda.get_device_name(0)} "
          f"| ATTENTION_BACKEND={os.environ.get('ATTENTION_BACKEND')}")

    cfg_dir = os.path.join(REPO, "groot", "vla", "configs")
    ov = ["data=dreamzero/droid_relative", "model=dreamzero/vla",
          "model/dreamzero/action_head=wan_flow_matching_action_tf_300m",
          "model/dreamzero/transform=dreamzero_cotrain",
          "num_frames=33", "action_horizon=24", "num_views=3",
          "num_frame_per_block=2", "num_action_per_block=24", "num_state_per_block=1",
          "max_chunk_size=4", "train_architecture=full",
          "image_resolution_width=320", "image_resolution_height=176",
          "droid_data_root=/tmp/none", "dit_version=null",
          "text_encoder_pretrained_path=null", "image_encoder_pretrained_path=null",
          "vae_pretrained_path=null"]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="conf", overrides=ov)

    print("[train-step] building ~293M i2v DiT (from scratch, no weight load)...")
    m = instantiate(cfg.action_head_cfg.config.diffusion_model_cfg)
    m.to(device=dev, dtype=torch.bfloat16).train()
    m.gradient_checkpointing = True
    nP = sum(p.numel() for p in m.parameters())
    print(f"[train-step] params: {nP:,} ({nP / 1e6:.1f}M), all trainable={all(p.requires_grad for p in m.parameters())}")
    assert m.model_type == "i2v" and m.in_dim == 36 and m.num_layers == 19

    # i2v inputs (frame_seqlen=880/frame: Hl=44,Wl=80). F=3 -> 1 image/action/state block each.
    B, F, Hl, Wl = 1, 3, 44, 80
    seq_len = F * (Hl // 2) * (Wl // 2)
    bf = dict(device=dev, dtype=torch.bfloat16)
    x         = torch.randn(B, 16, F, Hl, Wl, **bf, requires_grad=True)
    y         = torch.randn(B, 20, F, Hl, Wl, **bf)        # first-frame latent(16)+mask(4) -> in_dim 36
    clean_x   = torch.randn(B, 16, F, Hl, Wl, **bf)
    clip_feat = torch.randn(B, 257, 1280, **bf)
    context   = torch.randn(B, m.text_len, m.text_dim, **bf)
    state     = torch.randn(B, 1, m.max_state_dim, **bf)
    action    = torch.rand(B, 24, m.action_dim, **bf) * 2 - 1
    ts        = torch.full((B, F), 500.0, device=dev)
    ts_a      = torch.full((B, 24), 500.0, device=dev)
    emb       = torch.zeros(B, device=dev, dtype=torch.long)
    v_tgt     = torch.randn(B, 16, F, Hl, Wl, **bf)
    a_tgt     = torch.randn(B, 24, m.action_dim, **bf)

    opt = torch.optim.SGD(m.parameters(), lr=1e-3)

    def step(i):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        v, a = m(x, timestep=ts, clip_feature=clip_feat, y=y, context=context, seq_len=seq_len,
                 state=state, embodiment_id=emb, action=action, timestep_action=ts_a, clean_x=clean_x)
        loss = torch.nn.functional.mse_loss(v.float(), v_tgt.float()) + \
               torch.nn.functional.mse_loss(a.float(), a_tgt.float())
        loss.backward()
        gp = [p for p in m.parameters() if p.grad is not None]
        finite = all(torch.isfinite(p.grad).all() for p in gp)
        gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9)
        opt.step()
        torch.cuda.synchronize()
        print(f"[train-step] step {i}: loss={loss.item():.4f} grad_norm={float(gnorm):.3f} "
              f"params_with_grad={len(gp)}/{sum(1 for _ in m.parameters())} grads_finite={finite} "
              f"v{tuple(v.shape)} a{tuple(a.shape)} {time.time()-t0:.1f}s "
              f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        assert finite, "non-finite gradients"
        return loss.item()

    print("[train-step] running 2 optimizer steps...")
    l0 = step(0)
    l1 = step(1)
    print(f"\n[train-step] RESULT: two real grad steps; loss {l0:.4f} -> {l1:.4f}")
    print("=== TRAIN-STEP TEST PASSED: fwd+bwd+optimizer step works on the ~293M i2v DiT (GPU) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
