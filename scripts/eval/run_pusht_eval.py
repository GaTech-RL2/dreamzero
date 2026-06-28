"""Closed-loop PushT evaluation for DreamZero (single-process, gym-pusht).

Resets gym-pusht, observes the top-down RGB frame, runs the DreamZero policy to predict an
action chunk (+ a latent video rollout), executes the first n_action_steps actions, repeats.
Computes the Diffusion-Policy-style coverage success metric and saves, per episode:
  * <out>/seed{seed}_env.mp4   -- the rendered rollout
  * <out>/seed{seed}_pred.mp4  -- the model's predicted video (VAE-decoded), if --save_pred_video

Run on a GPU (cross-attention is CUDA-only), e.g.:
  srun --partition=overcap --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=01:00:00 \
    bash -c "cd $PWD && .venv/bin/python scripts/eval/run_pusht_eval.py \
       --model_path checkpoints/dreamzero_pusht_300m/checkpoint-XXXX --num_episodes 50"
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("ATTENTION_BACKEND", "torch")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.distributed as dist
import gymnasium as gym
import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
import imageio
from einops import rearrange
from tianshou.data import Batch

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema.embodiment_tags import EmbodimentTag

TASK_TEXT = "push the T-shaped block to the target goal"


def _bump_recompile_limit():
    # The UniPC inference sampler is @torch.compile(fullgraph=True, dynamic=False), so each new
    # shape recompiles; the default limit (8) trips FailOnRecompileLimitHit. socket_test uses 800.
    try:
        torch._dynamo.config.recompile_limit = 800
        torch._dynamo.config.cache_size_limit = 800
    except Exception:  # noqa: BLE001
        pass


def init_distributed():
    _bump_recompile_limit()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=0, world_size=1)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)


class PushTAdapter:
    """Maps gym-pusht obs <-> DreamZero PushT modality keys (single view, 2D target action)."""

    def reset(self):
        pass

    def to_model_obs(self, obs: dict) -> dict:
        pixels = np.asarray(obs["pixels"]).astype(np.uint8)          # (96,96,3) RGB
        agent_pos = np.asarray(obs["agent_pos"], dtype=np.float64).reshape(1, 2)
        return {
            "video.rgb_cam_primary": pixels[None],                   # (T=1, 96, 96, 3)
            "state.agent_pos": agent_pos,                            # (T=1, 2)
            "annotation.language.action_text": TASK_TEXT,
        }

    @staticmethod
    def from_model_action(act) -> np.ndarray:
        """Extract action.target_pos -> (horizon, 2) absolute pixel-coord targets."""
        value = None
        if isinstance(act, dict):
            for k, v in act.items():
                if isinstance(k, str) and k.startswith("action."):
                    value = v
                    break
        else:  # tianshou Batch
            for k in dir(act):
                if isinstance(k, str) and k.startswith("action."):
                    value = getattr(act, k)
                    break
        if value is None:
            raise RuntimeError(f"No action.* key in policy output: {act}")
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        return value.reshape(-1, value.shape[-1])[:, :2]


def decode_pred_video(policy, pred_latents):
    """VAE-decode accumulated latent video preds -> list of uint8 RGB frames (or None)."""
    if not pred_latents:
        return None
    try:
        ah = policy.trained_model.action_head
        with torch.inference_mode():
            lat = torch.cat(pred_latents, dim=2)
            frames = ah.vae.decode(
                lat,
                tiled=ah.tiled,
                tile_size=(ah.tile_size_height, ah.tile_size_width),
                tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
            )
        frames = rearrange(frames, "B C T H W -> B T H W C")[0]
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        return list(frames)
    except Exception as e:  # noqa: BLE001
        print(f"[eval] WARNING: predicted-video decode failed: {e}")
        return None


def rollout(policy, env, seed, n_action_steps, max_steps, n_pred_chunks):
    adapter = PushTAdapter()
    obs, info = env.reset(seed=seed)
    adapter.reset()
    render_frames = [env.render()]
    pred_latents = []
    max_cov = float(info.get("coverage", 0.0))
    steps, done = 0, False
    lo, hi = env.action_space.low, env.action_space.high

    while steps < max_steps and not done:
        rb, video_pred = policy.lazy_joint_forward_causal(Batch(obs=adapter.to_model_obs(obs)))
        if len(pred_latents) < n_pred_chunks and video_pred is not None:
            pred_latents.append(video_pred)
        actions = adapter.from_model_action(rb.act)  # (horizon, 2)
        for a in actions[:n_action_steps]:
            a = np.clip(a, lo, hi).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(a)
            render_frames.append(env.render())
            max_cov = max(max_cov, float(info.get("coverage", 0.0)))
            steps += 1
            if terminated or truncated or steps >= max_steps:
                done = terminated or truncated
                break
    return max_cov, render_frames, pred_latents


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="checkpoint dir with experiment_cfg/")
    p.add_argument("--num_episodes", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="explicit seeds (default 0..N-1)")
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save_pred_video", action="store_true")
    p.add_argument("--n_pred_chunks", type=int, default=6, help="cap predicted-video inferences to decode")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[eval] ERROR: needs a CUDA GPU (cross-attention is CUDA-only).")

    init_distributed()
    out_dir = args.output_dir or os.path.join(args.model_path, "pusht_eval")
    os.makedirs(out_dir, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else list(range(args.num_episodes))

    # GrootSimPolicy reads model_path/experiment_cfg/{conf.yaml,metadata.json}. The trainer's
    # CheckpointFormatCallback copies it into each checkpoint-* on save. We must NOT create it
    # ourselves (a symlink/copy here races the trainer's copytree and crashes training). Just
    # require it -- the watcher only submits once experiment_cfg/metadata.json exists.
    exp_cfg = os.path.join(args.model_path, "experiment_cfg")
    if not os.path.isfile(os.path.join(exp_cfg, "conf.yaml")):
        raise SystemExit(
            f"[eval] {exp_cfg}/conf.yaml missing -- wait for the checkpoint's experiment_cfg copy "
            "to finish before evaluating (do not symlink it in; that races the trainer)."
        )

    print(f"[eval] loading policy from {args.model_path}")
    policy = GrootSimPolicy(EmbodimentTag.PUSHT, args.model_path, device=args.device)
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
        render_mode="rgb_array", max_episode_steps=args.max_steps,
    )
    policy.on_env_init(env)
    threshold = float(getattr(env.unwrapped, "success_threshold", 0.95))

    results = []
    for seed in seeds:
        max_cov, render_frames, pred_latents = rollout(
            policy, env, seed, args.n_action_steps, args.max_steps, args.n_pred_chunks
        )
        success = max_cov >= threshold
        score = min(max_cov / threshold, 1.0)
        results.append({"seed": seed, "max_coverage": max_cov, "score": score, "success": bool(success)})
        print(f"[eval] seed {seed}: max_coverage={max_cov:.3f} score={score:.3f} success={success}")

        imageio.mimsave(os.path.join(out_dir, f"seed{seed}_env.mp4"),
                        render_frames, fps=10, codec="libx264", macro_block_size=1)
        if args.save_pred_video:
            pred = decode_pred_video(policy, pred_latents)
            if pred:
                imageio.mimsave(os.path.join(out_dir, f"seed{seed}_pred.mp4"),
                                pred, fps=5, codec="libx264", macro_block_size=1)

    success_rate = float(np.mean([r["success"] for r in results]))
    mean_score = float(np.mean([r["score"] for r in results]))
    mean_cov = float(np.mean([r["max_coverage"] for r in results]))
    summary = {
        "model_path": args.model_path, "num_episodes": len(seeds),
        "success_threshold": threshold, "n_action_steps": args.n_action_steps,
        "max_steps": args.max_steps, "success_rate": success_rate,
        "mean_score": mean_score, "mean_max_coverage": mean_cov, "episodes": results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] ===== PushT RESULTS ({len(seeds)} eps) =====")
    print(f"[eval] success_rate (cov>={threshold}) = {success_rate:.3f}")
    print(f"[eval] mean_score                       = {mean_score:.3f}")
    print(f"[eval] mean_max_coverage                = {mean_cov:.3f}")
    print(f"[eval] summary -> {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
