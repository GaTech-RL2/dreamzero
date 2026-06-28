"""GPU equivalence gate for per-device batching.

Builds the ~300M i2v DiT (pusht config: frame_seqlen=256), then checks that a BATCHED forward over B
distinct samples produces the SAME per-sample outputs as running each sample individually at batch 1.
This proves batching introduces no cross-sample leakage (the core correctness concern of enabling
per_device_batch_size>1). Run in eval() so there's no dropout randomness.

  srun --partition=overcap --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=00:20:00 \
       bash -c "cd $PWD && ATTENTION_BACKEND=torch .venv/bin/python scripts/smoke_batch_equiv.py"
"""
import os
os.environ.setdefault("ATTENTION_BACKEND", "torch")
import sys
import torch
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    if not torch.cuda.is_available():
        print("[equiv] ERROR: needs a CUDA GPU. Submit via Slurm.")
        return 1
    dev = "cuda"
    cfg_dir = os.path.join(REPO, "groot", "vla", "configs")
    ov = ["data=dreamzero/droid_relative", "model=dreamzero/vla",
          "model/dreamzero/action_head=wan_flow_matching_action_tf_300m",
          "model/dreamzero/transform=dreamzero_cotrain",
          "num_frames=33", "action_horizon=24", "num_views=1",
          "num_frame_per_block=2", "num_action_per_block=24", "num_state_per_block=1",
          "max_chunk_size=4", "train_architecture=full",
          "frame_seqlen=256", "image_resolution_width=256", "image_resolution_height=256",
          "droid_data_root=/tmp/none", "dit_version=null",
          "text_encoder_pretrained_path=null", "image_encoder_pretrained_path=null",
          "vae_pretrained_path=null"]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="conf", overrides=ov)

    import torch.nn as nn
    m = instantiate(cfg.action_head_cfg.config.diffusion_model_cfg)
    m.to(device=dev, dtype=torch.bfloat16).train()
    # _forward_train only unpacks block output on the grad-checkpointing path (torch.is_grad_enabled()
    # and gradient_checkpointing); the plain else-branch is buggy. So we must run grad-enabled +
    # gradient_checkpointing. Disable dropout for determinism; freeze params so no graph/memory builds.
    m.gradient_checkpointing = True
    m.requires_grad_(False)
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
    print(f"[equiv] built DiT frame_seqlen={m.frame_seqlen} model_type={m.model_type} in_dim={m.in_dim}")

    # pusht latent: frame_seqlen=256 = (Hl/2)*(Wl/2) -> Hl=Wl=32. F frames.
    B, F, Hl, Wl = 4, 3, 32, 32
    seq_len = F * (Hl // 2) * (Wl // 2)
    bf = dict(device=dev, dtype=torch.bfloat16)
    torch.manual_seed(0)
    # distinct per-sample inputs
    x         = torch.randn(B, 16, F, Hl, Wl, **bf)
    y         = torch.randn(B, 20, F, Hl, Wl, **bf)
    clean_x   = torch.randn(B, 16, F, Hl, Wl, **bf)
    clip_feat = torch.randn(B, 257, 1280, **bf)
    context   = torch.randn(B, m.text_len, m.text_dim, **bf)
    state     = torch.randn(B, 1, m.max_state_dim, **bf)
    action    = torch.rand(B, 24, m.action_dim, **bf) * 2 - 1
    ts        = torch.full((B, F), 500.0, device=dev)
    ts_a      = torch.full((B, 24), 500.0, device=dev)
    emb       = torch.zeros(B, device=dev, dtype=torch.long)

    def fwd(sl):
        return m(x[sl], timestep=ts[sl], clip_feature=clip_feat[sl], y=y[sl], context=context[sl],
                 seq_len=seq_len, state=state[sl], embodiment_id=emb[sl], action=action[sl],
                 timestep_action=ts_a[sl], clean_x=clean_x[sl])

    # Must run grad-ENABLED so _forward_train takes the checkpointing path that unpacks block output.
    # Params are frozen, so no graph/memory is built. Detach outputs.
    with torch.enable_grad():
        v_batch, a_batch = fwd(slice(0, B))
        v_batch, a_batch = v_batch.detach(), a_batch.detach()
        v_each, a_each = [], []
        for i in range(B):
            vi, ai = fwd(slice(i, i + 1))
            v_each.append(vi.detach()); a_each.append(ai.detach())
    v_each = torch.cat(v_each, 0).float()
    a_each = torch.cat(a_each, 0).float()
    v_batch = v_batch.float(); a_batch = a_batch.float()

    v_err = (v_batch - v_each).abs().max().item()
    a_err = (a_batch - a_each).abs().max().item()
    v_scale = v_each.abs().mean().item(); a_scale = a_each.abs().mean().item()
    print(f"[equiv] video: max|batch-individual|={v_err:.4e} (mean|v|={v_scale:.3e}) shape={tuple(v_batch.shape)}")
    print(f"[equiv] action: max|batch-individual|={a_err:.4e} (mean|a|={a_scale:.3e}) shape={tuple(a_batch.shape)}")
    tol = 5e-2  # bf16 numerics; cross-sample leakage would be O(scale), far above this
    ok = v_err < tol and a_err < tol
    print(f"\n=== BATCH EQUIVALENCE {'PASSED' if ok else 'FAILED'} (tol={tol}) ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
