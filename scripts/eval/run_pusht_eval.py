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
import re

os.environ.setdefault("ATTENTION_BACKEND", "torch")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import cv2
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


def derive_wandb_run_id(model_path):
    """Map a checkpoint path back to the training run's wandb id.

    sbatch_pusht.sh sets WANDB_RUN_ID=dzpusht_${SIZE//./p}, where the output dir is
    checkpoints/dreamzero_pusht_<SIZE>. So .../dreamzero_pusht_300m/checkpoint-N -> dzpusht_300m
    and .../dreamzero_pusht_1.3b/checkpoint-N -> dzpusht_1p3b.
    """
    for part in os.path.normpath(model_path).split(os.sep):
        if part.startswith("dreamzero_pusht_"):
            size = part[len("dreamzero_pusht_"):]
            return "dzpusht_" + size.replace(".", "p")
    return None


def derive_step(model_path):
    """checkpoint-12000 -> 12000 (the trainer global_step), else None."""
    m = re.match(r"checkpoint-(\d+)", os.path.basename(os.path.normpath(model_path)))
    return int(m.group(1)) if m else None


def log_eval_to_wandb(summary, step, run_id, entity=None, project=None):
    """Append eval metrics to the training run on a custom `eval/step` x-axis.

    Eval lags training (it runs on a checkpoint while training has moved on), so we DON'T pass a
    wandb step — we let wandb auto-increment its internal step and plot eval/* against the custom
    `eval/step` metric (the checkpoint's global_step). This avoids any "step must increase" conflict
    with the trainer logging concurrently to the same run, and places each point at its true step.
    Resuming the (live) training run from this separate process is the standard async-eval pattern;
    finishing here only flushes our handle — the trainer keeps logging to the same run unaffected.
    """
    import wandb

    entity = entity or os.environ.get("WANDB_ENTITY", "rl2-group")
    project = project or os.environ.get("WANDB_PROJECT", "world-value")
    wandb.init(entity=entity, project=project, id=run_id, resume="allow",
               settings=wandb.Settings(silent=True))
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.log({
        "eval/step": int(step),
        "eval/success_rate": float(summary["success_rate"]),
        "eval/mean_score": float(summary["mean_score"]),
        "eval/mean_coverage": float(summary["mean_max_coverage"]),
        "eval/num_episodes": int(summary["num_episodes"]),
    })
    wandb.finish()
    print(f"[eval] logged to wandb {entity}/{project} run={run_id} at eval/step={step}")


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


def _resample_indices(src_len, out_len):
    """Nearest-neighbor index map from an `out_len` timeline onto a `src_len` track (both span [0,1])."""
    if src_len <= 1 or out_len <= 1:
        return [0] * out_len
    return [round(i * (src_len - 1) / (out_len - 1)) for i in range(out_len)]


def _resize_h(frame, height):
    """Resize an (H,W,3) uint8 frame to `height`, preserving aspect ratio."""
    h, w = frame.shape[:2]
    width = max(1, int(round(w * height / h)))
    interp = cv2.INTER_AREA if height < h else cv2.INTER_NEAREST
    return cv2.resize(frame, (width, height), interpolation=interp)


def make_side_by_side(dream_frames, env_frames, height=512, labels=("DREAM (imagined)", "ACTUAL")):
    """Align the dreamed and actual rollouts on a common start->end timeline and h-stack them.

    The tracks have different frame counts (the dream is 4x temporal-upsampled latent; the env is one
    render per sim step) but both span the whole episode, so each is resampled to a shared length by
    nearest-neighbor on normalized time -- so column i shows "what the model imagined at time t" beside
    "what actually happened at time t". Panels are resized to a common height with a white divider and a
    text label. Returns a list of (height, W_dream+4+W_env, 3) uint8 RGB frames, or None.
    """
    if not dream_frames or not env_frames:
        return None
    out_len = max(len(dream_frames), len(env_frames))
    di = _resample_indices(len(dream_frames), out_len)
    ei = _resample_indices(len(env_frames), out_len)
    sep = np.full((height, 4, 3), 255, dtype=np.uint8)  # white divider
    out = []
    for i in range(out_len):
        d = _resize_h(np.ascontiguousarray(dream_frames[di[i]]), height)
        e = _resize_h(np.ascontiguousarray(env_frames[ei[i]]), height)
        if labels:
            # frames are RGB (imageio order): (255,255,0)=yellow, (0,255,0)=green
            cv2.putText(d, labels[0], (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(e, labels[1], (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        out.append(np.concatenate([d, sep, e], axis=1))
    return out


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
        # n_pred_chunks < 0 => keep every imagined chunk (the full dreamed rollout); >=0 caps for speed.
        if video_pred is not None and (n_pred_chunks < 0 or len(pred_latents) < n_pred_chunks):
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
    p.add_argument("--n_pred_chunks", type=int, default=-1,
                   help="cap imagined chunks to decode; -1 = all (the full dreamed rollout)")
    p.add_argument("--save_compare", action=argparse.BooleanOptionalAction, default=True,
                   help="write seedN_compare.mp4 (dream | actual, aligned side-by-side) -- the default output")
    p.add_argument("--save_env", action="store_true", help="also write the standalone actual-rollout mp4 (off by default)")
    p.add_argument("--save_pred", action="store_true", help="also write the standalone dreamed mp4 (off by default)")
    p.add_argument("--compare_height", type=int, default=512, help="panel height for the compare video")
    p.add_argument("--wandb", action="store_true", help="log eval/* metrics to the training wandb run")
    p.add_argument("--wandb_run_id", default=None, help="override (default: derived from model_path)")
    p.add_argument("--wandb_step", type=int, default=None, help="override (default: checkpoint step)")
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

        # Video viz: by default ONLY the aligned dream|actual compare is written; the standalone
        # env/pred mp4s are opt-in (--save_env/--save_pred) to save disk.
        if args.save_pred_video:
            pred = decode_pred_video(policy, pred_latents)
            if pred:
                print(f"[eval] seed {seed}: dreamed {len(pred)} frames over {len(render_frames)} env frames")
                if args.save_compare:
                    comp = make_side_by_side(pred, render_frames, height=args.compare_height)
                    if comp:
                        imageio.mimsave(os.path.join(out_dir, f"seed{seed}_compare.mp4"),
                                        comp, fps=10, codec="libx264", macro_block_size=1)
                if args.save_pred:  # fps 10 matches env so the dream plays on the rollout timeline
                    imageio.mimsave(os.path.join(out_dir, f"seed{seed}_pred.mp4"),
                                    pred, fps=10, codec="libx264", macro_block_size=1)
        if args.save_env:
            imageio.mimsave(os.path.join(out_dir, f"seed{seed}_env.mp4"),
                            render_frames, fps=10, codec="libx264", macro_block_size=1)

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

    if args.wandb:
        run_id = args.wandb_run_id or derive_wandb_run_id(args.model_path)
        step = args.wandb_step if args.wandb_step is not None else derive_step(args.model_path)
        if run_id is None or step is None:
            print(f"[eval] WARN: could not derive wandb run_id/step from {args.model_path}; skipping wandb log")
        else:
            try:
                log_eval_to_wandb(summary, step, run_id)
            except Exception as e:  # noqa: BLE001  — never let wandb break eval
                print(f"[eval] WARN: wandb logging failed: {e}")


if __name__ == "__main__":
    main()
