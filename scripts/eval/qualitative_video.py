"""Qualitative per-task dump for DreamZero RoboTwin RICL: for one sample frame of each task,
generate the model's predicted future VIDEO (lazy_joint_video_action -> VAE-decode to pixels)
and save it alongside the RETRIEVED demo frames that conditioned it.

Mirrors the rollout adapter's inference prep (egomimic.ricl.dreamzero_robotwin_policy.get_action):
repeat the obs frame over T, attach retrieved_video, slice to one warm-up frame, reset the KV
cache, sample. Per task writes under <out>/<task>/:
  generated.mp4        - decoded predicted video (the world model's rollout for the obs)
  retrieved.png        - montage of the K demos x X frames (head view) that were retrieved
  obs.png              - the observation (head/left/right)

Run on a GPU node (a100/l40s, bf16):
  PYTHONPATH=external/dreamzero/_dzdeps:external/dreamzero:. TORCH_COMPILE_DISABLE=1 HF_HUB_OFFLINE=1 \
  emimic/bin/python external/dreamzero/scripts/eval/qualitative_video.py \
    --checkpoint <ckpt> --robotwin-root <root> --quantiles <train quantiles.json> --out <dir>
"""
import argparse
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("ATTENTION_BACKEND", "FA2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from heldout_loss import build_model  # same Hydra build (treatment)


def _save_mp4(frames, path, fps=4):
    """frames: [T,H,W,3] uint8 RGB."""
    import cv2
    if frames.shape[0] == 0:
        return
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def _grid(frames, path, ncol):
    """frames: [N,H,W,3] uint8 -> a row-major grid PNG."""
    import cv2
    if len(frames) == 0:
        return
    n = len(frames); nrow = (n + ncol - 1) // ncol
    h, w = frames[0].shape[:2]
    canvas = np.zeros((nrow * h, ncol * w, 3), np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, ncol)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def decode_video(model, video_pred):
    """video_pred from lazy_joint_video_action -> [T,H,W,3] uint8 (head/grid as generated)."""
    vae = model.action_head.vae
    v = video_pred
    # Expect a 5D latent [B, C, T, H, W]; lazy_joint returns output.transpose(1,2). If the
    # temporal axis ended up before channels, put channels first (C == vae.z_dim, e.g. 48).
    zc = getattr(vae, "z_dim", 48)
    if v.dim() == 5 and v.shape[1] != zc and v.shape[2] == zc:
        v = v.transpose(1, 2).contiguous()
    with torch.no_grad():
        px = vae.decode(v.to(model.action_head._device, dtype=torch.bfloat16))
    if isinstance(px, (list, tuple)):
        px = px[0]
    px = px.float()
    # [B,3,T,H,W] -> [T,H,W,3], scale [-1,1]->[0,255]
    px = ((px.clamp(-1, 1) + 1) / 2 * 255).round().byte().cpu().numpy()[0]
    px = np.transpose(px, (1, 2, 3, 0))
    return px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--robotwin-root", required=True)
    ap.add_argument("--quantiles", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--x", type=int, default=4)
    ap.add_argument("--num-frames", type=int, default=9)
    ap.add_argument("--action-horizon", type=int, default=24)
    ap.add_argument("--num-frame-per-block", type=int, default=2)
    ap.add_argument("--max-chunk-size", type=int, default=8)
    ap.add_argument("--embed", default="dinov2", choices=["dinov2", "fake"])
    ap.add_argument("--frame", type=int, default=5, help="sample query frame index per episode")
    ap.add_argument("--steps", type=int, default=8, help="num diffusion inference steps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from egomimic.ricl import robotwin_data as R

    model, transform, collator = build_model(
        args.checkpoint, no_incontext=False, K=args.k, X=args.x,
        num_frames=args.num_frames, action_horizon=args.action_horizon,
        num_frame_per_block=args.num_frame_per_block, max_chunk_size=args.max_chunk_size,
        robotwin_root=args.robotwin_root)
    model.action_head.num_inference_steps = args.steps
    dev = model.action_head._device

    q = R.load_quantiles(args.quantiles)
    corpus = R.RoboTwinCorpus(args.robotwin_root, mode="joint", quantiles=q)
    make_embed = R.make_fake_embedding_provider if args.embed == "fake" else R.make_dinov2_embedding_provider
    cache = R.build_robotwin_retrieval_cache(corpus, args.k, make_embed(corpus))
    clip = R.make_robotwin_clip_provider(corpus, frames_per_demo=args.x)
    CAM = R.CAM_KEYS

    os.makedirs(args.out, exist_ok=True)
    for task, hashes in corpus.group_to_hashes.items():
        h = hashes[0]
        fi = min(args.frame, corpus.num_frames(h) - 1)
        obs_views = np.stack([corpus.image(h, fi, c) for c in CAM], 0)        # [V,H,W,3]
        video = np.repeat(obs_views[None], args.num_frames, 0)                # [T,V,H,W,3]
        state = np.clip(corpus.quantile_norm(corpus.state(h, fi), "state"), -1, 1)[None].astype(np.float32)
        bh, bf, _ = cache.neighbors(h, fi)
        rv, mask = [], []
        empty = np.zeros((args.x, len(CAM), *corpus.image_hw, 3), np.uint8)
        for j in range(args.k):
            hj = str(bh[j]) if j < len(bh) else ""; fj = int(bf[j]) if j < len(bf) else -1
            if hj and fj >= 0:
                rv.append(clip(hj, fj)["frames"].astype(np.uint8)); mask.append(True)
            else:
                rv.append(empty); mask.append(False)
        retrieved = np.stack(rv, 0)                                            # [K,X,V,H,W,3]
        data = {
            "video": video, "state": state,
            "action": np.zeros((args.action_horizon, corpus.state_dim), np.float32),
            "annotation.task": corpus.prompt(h),
            "retrieved_video": retrieved, "retrieved_mask": np.asarray(mask, bool),
        }
        batch = collator([transform(data)])
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["images"] = batch["images"][:, :1].contiguous()
        model.action_head.language = None
        model.action_head.current_start_frame = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.lazy_joint_video_action(batch)
        gen = decode_video(model, out["video_pred"])                          # [T,H,W,3]

        d = os.path.join(args.out, task); os.makedirs(d, exist_ok=True)
        _save_mp4(gen, os.path.join(d, "generated.mp4"))
        _grid(list(gen), os.path.join(d, "generated_strip.png"), ncol=len(gen))
        # retrieved head-view montage: K rows x X cols
        head = retrieved[:, :, 0].reshape(-1, *retrieved.shape[3:])           # [K*X,H,W,3]
        _grid(list(head), os.path.join(d, "retrieved.png"), ncol=args.x)
        _grid(list(obs_views), os.path.join(d, "obs.png"), ncol=len(CAM))
        print(f"[{task}] gen={gen.shape} retrieved={retrieved.shape} -> {d}", flush=True)
    print(f"[done] wrote per-task video dumps under {args.out}", flush=True)


if __name__ == "__main__":
    main()
