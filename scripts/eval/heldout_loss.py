"""Held-out teacher-forced loss eval for DreamZero RoboTwin RICL (baseline vs treatment).

Loads a trained LoRA checkpoint, builds a held-out dataset over a RoboTwin task root using the
SAME quantiles the model trained on, runs the training forward (flow-matching) over a balanced
per-task subset of frames, and reports mean dynamics_loss / action_loss per task + overall.

Two scenarios (the two experiments in the plan):
  --scenario train_heldout : queries = held-out 20% of each TRAIN task's episodes; retrieval bank
                             = that task's TRAIN episodes (mirrors training). Eval 1 (in-dist).
  --scenario newtask_loo   : queries = ALL episodes of each (NEW) task; retrieval = within-task
                             leave-one-out over the task's own demos. Eval 2 (new-skill test).

The forward is stochastic (flow-matching samples noise+timestep); we fix torch.manual_seed per
sample so baseline and treatment see identical draws on identical frames (paired comparison).

Run on a GPU node (a100/l40s, bf16 ~ fits 48-80GB):
  PYTHONPATH=external/dreamzero/_dzdeps:external/dreamzero:. TORCH_COMPILE_DISABLE=1 HF_HUB_OFFLINE=1 \
  emimic/bin/python external/dreamzero/scripts/eval/heldout_loss.py \
    --checkpoint <ckpt> --robotwin-root <root> --quantiles <train quantiles.json> \
    --scenario newtask_loo --mode treatment --out <out.json>
"""
import argparse
import collections
import json
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("ATTENTION_BACKEND", "FA2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = "/storage/project/r-dxu345-0/rco3/EgoVerse2"
DZ = os.path.join(REPO, "external", "dreamzero")
DZ_CKPTS = os.path.join(DZ, "checkpoints")


def build_model(checkpoint, no_incontext, K, X, num_frames, action_horizon,
                num_frame_per_block, max_chunk_size, robotwin_root):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from groot.vla.data.schema import EmbodimentTag

    wan = os.path.join(DZ_CKPTS, "Wan2.2-TI2V-5B")
    img_enc = os.path.join(DZ_CKPTS, "Wan2.1-I2V-14B-480P")
    tok = os.path.join(DZ_CKPTS, "umt5-xxl")
    ricl = (["enable_retrieved_context=false", "num_retrieved_demos=0"] if no_incontext
            else ["enable_retrieved_context=true",
                  f"num_retrieved_demos={K}", f"frames_per_demo={X}"])
    overrides = [
        "data=dreamzero/robotwin_wan22",
        f"robotwin_root={robotwin_root}",          # only to satisfy the data cfg ??? (unused here)
        "model=dreamzero/vla",
        "model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_ricl",
        "model/dreamzero/transform=dreamzero_cotrain",
        "train_architecture=lora",
        f"num_frames={num_frames}", f"action_horizon={action_horizon}", "state_horizon=1",
        f"num_frame_per_block={num_frame_per_block}", "num_action_per_block=24",
        "num_state_per_block=1", f"max_chunk_size={max_chunk_size}",
        *ricl,
        f"dit_version={wan}",
        f"vae_pretrained_path={wan}/Wan2.2_VAE.pth",
        f"text_encoder_pretrained_path={wan}/models_t5_umt5-xxl-enc-bf16.pth",
        f"image_encoder_pretrained_path={img_enc}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        f"tokenizer_path={tok}",
    ]
    cfgdir = os.path.join(DZ, "groot", "vla", "configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="conf", overrides=overrides)
    model = instantiate(cfg.model)
    if checkpoint:
        model.load_lora_weight(checkpoint)
    else:
        print("[eval] WARNING: no checkpoint — UNTRAINED base (smoke only)", flush=True)
    model.action_head.post_initialize()          # cast heavy modules to bf16 on cuda
    model = model.eval()
    transform = instantiate(cfg.model_specific_transform)
    transform.embodiment_tag = EmbodimentTag("xdof")
    collator = instantiate(cfg.data_collator)
    return model, transform, collator


def build_dataset(scenario, root, quantiles_path, transform, no_incontext, K, X,
                  num_frames, action_horizon, embed, seed):
    from egomimic.ricl import robotwin_data as R
    from groot.vla.data.dataset.robotwin_dreamzero import (
        RoboTwinDreamZeroDataset, split_hashes_per_group, _subset)

    q = R.load_quantiles(quantiles_path)          # TRAIN quantiles — must match training
    corpus = R.RoboTwinCorpus(root, mode="joint", quantiles=q)
    make_embed = (R.make_fake_embedding_provider if embed == "fake"
                  else R.make_dinov2_embedding_provider)
    cache = clip = None
    if scenario == "train_heldout":
        train_h, eval_h = split_hashes_per_group(corpus, 0.2, seed)
        query_h = eval_h
        if not no_incontext:
            ep = make_embed(corpus)
            cache = R.build_cross_embodiment_retrieval_cache(
                _subset(corpus, eval_h), _subset(corpus, train_h), K, ep, ep)
            clip = R.make_robotwin_clip_provider(corpus, frames_per_demo=X)
    else:  # newtask_loo
        query_h = [h for hs in corpus.group_to_hashes.values() for h in hs]
        if not no_incontext:
            ep = make_embed(corpus)
            cache = R.build_robotwin_retrieval_cache(corpus, K, ep)  # within-task LOO over all
            clip = R.make_robotwin_clip_provider(corpus, frames_per_demo=X)
    ds = RoboTwinDreamZeroDataset(
        corpus, transform, num_frames, action_horizon,
        retrieval_cache=cache, clip_provider=clip,
        num_retrieved_demos=(0 if no_incontext else K), frames_per_demo=X, hashes=query_h)
    return corpus, ds


def balanced_subset(ds, corpus, per_task):
    """Pick up to `per_task` frames per task, evenly spaced, so the mean isn't dominated by
    long episodes / the first task. Returns (selected positions, hash->task map)."""
    hash2task = {h: g for g, hs in corpus.group_to_hashes.items() for h in hs}
    by_task = collections.defaultdict(list)
    for pos, (h, _fi) in enumerate(ds.index):
        by_task[hash2task.get(h, h.split("__")[0])].append(pos)
    sel = []
    for _g, positions in by_task.items():
        if len(positions) <= per_task:
            sel += positions
        else:
            idx = np.linspace(0, len(positions) - 1, per_task).round().astype(int)
            sel += [positions[i] for i in idx]
    sel.sort()
    return sel, hash2task


def _to_cuda(batch):
    return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--robotwin-root", required=True)
    ap.add_argument("--quantiles", required=True, help="train quantiles.json")
    ap.add_argument("--scenario", choices=["train_heldout", "newtask_loo"], default="newtask_loo")
    ap.add_argument("--mode", choices=["baseline", "treatment"], required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--x", type=int, default=4)
    ap.add_argument("--num-frames", type=int, default=9)
    ap.add_argument("--action-horizon", type=int, default=24)
    ap.add_argument("--num-frame-per-block", type=int, default=2)
    ap.add_argument("--max-chunk-size", type=int, default=8)
    ap.add_argument("--embed", default="dinov2", choices=["dinov2", "fake"])
    ap.add_argument("--per-task", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    no_ic = args.mode == "baseline"
    print(f"[eval] mode={args.mode} scenario={args.scenario} root={args.robotwin_root}", flush=True)
    model, transform, collator = build_model(
        args.checkpoint, no_ic, args.k, args.x, args.num_frames, args.action_horizon,
        args.num_frame_per_block, args.max_chunk_size, args.robotwin_root)
    corpus, ds = build_dataset(
        args.scenario, args.robotwin_root, args.quantiles, transform, no_ic, args.k, args.x,
        args.num_frames, args.action_horizon, args.embed, args.seed)
    sel, hash2task = balanced_subset(ds, corpus, args.per_task)
    print(f"[eval] {len(sel)} frames over {len(set(hash2task.values()))} tasks", flush=True)
    dl = DataLoader(Subset(ds, sel), batch_size=1, shuffle=False, num_workers=2,
                    collate_fn=collator)

    per_task = collections.defaultdict(lambda: {"dyn": 0.0, "act": 0.0, "n": 0})
    overall = {"dyn": 0.0, "act": 0.0, "n": 0}
    for i, batch in enumerate(dl):
        h, _fi = ds.index[sel[i]]
        task = hash2task.get(h, h.split("__")[0])
        batch = _to_cuda(batch)
        torch.manual_seed(args.seed + i)                 # paired noise across baseline/treatment
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
        dyn = float(out["dynamics_loss"]); act = float(out["action_loss"])
        for d, key in ((per_task[task], None), (overall, None)):
            d["dyn"] += dyn; d["act"] += act; d["n"] += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(sel)}] running dyn={overall['dyn']/overall['n']:.4f} "
                  f"act={overall['act']/overall['n']:.4f}", flush=True)

    def _mean(d):
        n = max(1, d["n"])
        return {"dynamics_loss": d["dyn"] / n, "action_loss": d["act"] / n, "n": d["n"]}

    results = {
        "mode": args.mode, "scenario": args.scenario, "checkpoint": args.checkpoint,
        "robotwin_root": args.robotwin_root, "k": args.k, "x": args.x,
        "per_task": {t: _mean(d) for t, d in sorted(per_task.items())},
        "overall": _mean(overall),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== RESULT ===", flush=True)
    print(f"OVERALL  dyn={results['overall']['dynamics_loss']:.4f}  "
          f"act={results['overall']['action_loss']:.4f}  n={results['overall']['n']}", flush=True)
    for t, m in results["per_task"].items():
        print(f"  {t:24s} dyn={m['dynamics_loss']:.4f} act={m['action_loss']:.4f} n={m['n']}",
              flush=True)
    print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
