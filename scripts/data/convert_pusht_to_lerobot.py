"""
Convert the Diffusion-Policy PushT zarr replay buffer to a LeRobot-v2.0 dataset
laid out exactly like DreamZero expects (see data/droid_mini for the reference shape).

Source zarr (pusht_cchi_v7_replay.zarr):
  data/img    (N, 96, 96, 3) float32 in [0, 255]
  data/state  (N, 5) float32   -> agent_pos = state[:, :2] (pixel coords in [0, 512])
  data/action (N, 2) float32   -> target agent position (pixel coords in [0, 512])
  meta/episode_ends (E,) int64

This writes per-episode parquet + per-episode mp4 + meta/{info.json, tasks.jsonl,
episodes.jsonl}. It does NOT write modality.json / stats.json -- run
scripts/data/convert_lerobot_to_gear.py afterwards (it builds those from the parquet),
e.g.:

  python scripts/data/convert_pusht_to_lerobot.py \
      --zarr-path /path/pusht_cchi_v7_replay.zarr --output-path ./data/pusht_lerobot --fps 10
  python scripts/data/convert_lerobot_to_gear.py \
      --dataset-path ./data/pusht_lerobot --embodiment-tag pusht \
      --state-keys '{"agent_pos":[0,2]}' --action-keys '{"target_pos":[0,2]}' \
      --task-key annotation.language.action_text --fps 10

The annotation column is stored as an int64 task-index (0) and resolved to a sentence
through tasks.jsonl, mirroring the DROID dataset (see lerobot.py:get_language).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import zarr

TASK_TEXT = "push the T-shaped block to the target goal"
VIDEO_KEY = "observation.images.rgb_cam_primary"
CHUNKS_SIZE = 1000


def episode_slices(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return [(int(s), int(e)) for s, e in zip(starts, episode_ends)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr-path", required=True, type=str)
    p.add_argument("--output-path", required=True, type=str)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--max-episodes", type=int, default=None, help="Cap episodes (for smoke tests)")
    p.add_argument("--crf", type=int, default=10, help="h264 quality (lower = better)")
    args = p.parse_args()

    out = Path(args.output_path).resolve()
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "videos" / "chunk-000" / VIDEO_KEY).mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    z = zarr.open(args.zarr_path, "r")
    img = z["data"]["img"]          # (N,96,96,3) float32 0..255
    state = np.asarray(z["data"]["state"])   # (N,5)
    action = np.asarray(z["data"]["action"]) # (N,2)
    episode_ends = np.asarray(z["meta"]["episode_ends"])
    H, W = img.shape[1], img.shape[2]

    slices = episode_slices(episode_ends)
    if args.max_episodes is not None:
        slices = slices[: args.max_episodes]

    print(f"Converting {len(slices)} episodes -> {out}")
    print(f"  action range: min={action.min(0)} max={action.max(0)}")
    print(f"  agent_pos range: min={state[:, :2].min(0)} max={state[:, :2].max(0)}")

    episodes_meta = []
    total_frames = 0
    for ep_idx, (s, e) in enumerate(slices):
        length = e - s
        frames = np.asarray(img[s:e]).clip(0, 255).astype(np.uint8)  # (T,96,96,3)
        agent_pos = state[s:e, :2].astype(np.float64)                # (T,2)
        act = action[s:e].astype(np.float64)                         # (T,2)

        df = pd.DataFrame(
            {
                "observation.state": list(agent_pos),
                "action": list(act),
                "timestamp": (np.arange(length) / args.fps).astype(np.float64),
                "episode_index": np.full(length, ep_idx, dtype=np.int64),
                "frame_index": np.arange(length, dtype=np.int64),
                "annotation.language.action_text": np.zeros(length, dtype=np.int64),
                "task_index": np.zeros(length, dtype=np.int64),
            }
        )
        df.to_parquet(out / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")

        vid_path = out / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{ep_idx:06d}.mp4"
        imageio.mimsave(
            vid_path, list(frames), fps=args.fps, codec="libx264",
            ffmpeg_params=["-crf", str(args.crf), "-pix_fmt", "yuv420p"],
        )

        episodes_meta.append({"episode_index": ep_idx, "tasks": [TASK_TEXT], "length": int(length)})
        total_frames += length
        if (ep_idx + 1) % 20 == 0:
            print(f"  ...{ep_idx + 1}/{len(slices)} episodes")

    n_ep = len(slices)
    info = {
        "codebase_version": "v2.0",
        "robot_type": "pusht",
        "total_episodes": n_ep,
        "total_frames": int(total_frames),
        "total_tasks": 1,
        "total_videos": n_ep,
        "total_chunks": (n_ep - 1) // CHUNKS_SIZE + 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": args.fps,
        "splits": {"train": f"0:{n_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [H, W, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": args.fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {"dtype": "float64", "shape": [2], "names": ["agent_x", "agent_y"]},
            "action": {"dtype": "float64", "shape": [2], "names": ["target_x", "target_y"]},
            "timestamp": {"dtype": "float64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "annotation.language.action_text": {"dtype": "int64", "shape": [1]},
        },
    }
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_TEXT}) + "\n")
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    print(f"Done. {n_ep} episodes, {total_frames} frames -> {out}")
    print("Next: run scripts/data/convert_lerobot_to_gear.py to build modality.json + stats.json")


if __name__ == "__main__":
    main()
