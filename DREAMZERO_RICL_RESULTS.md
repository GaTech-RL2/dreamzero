# DreamZero RICL on RoboTwin — experiment results

**Question.** Does feeding a DreamZero world model (Wan2.2-TI2V-5B, joint video+action) *retrieved
demonstration frames* in-context (RICL-style) improve its predictions vs. a plain finetune —
in-distribution and on unseen tasks?

**Setup.** One training corpus → two LoRA checkpoints from identical hyperparameters (4000 steps,
gpu-a100 80GB):
- **plain** (`rt_dz_baseline`): stock DreamZero finetune (`enable_retrieved_context=false`).
- **RICL** (`rt_dz_treatment`): retrieved-frame in-context, K=2 demos × X=4 frames, DINOv2 kNN,
  within-task leave-one-out at train time.

10 train tasks / 10 disjoint eval tasks (aloha-agilex, 20 episodes each), RoboTwin clean setting.
Train quantiles reused everywhere (incl. unseen-task eval — joint ranges are shared).

## Primary metric — held-out teacher-forced loss (lower = better)

Flow-matching loss on held-out frames, paired noise seed per frame, balanced 25 frames/task.
`Δ = RICL − plain` (negative = retrieval helps). `dyn` = video dynamics loss, `act` = action loss.

### Eval 1 — train tasks, held-out episodes (in-distribution)
| | plain | RICL | Δ |
|---|---|---|---|
| **action loss** | 0.8152 | **0.4786** | **−0.3366 (~41%)** |
| dynamics loss | 1.3133 | 1.2992 | −0.0140 |

Action loss improves on **all 10 train tasks** (Δ −0.25 … −0.44).

### Eval 2 — 10 unseen tasks (new-skill test, within-task retrieval from the new task's demos)
| | plain | RICL | Δ |
|---|---|---|---|
| **action loss** | 0.7660 | **0.6437** | **−0.1223 (~16%)** |
| dynamics loss | 1.2645 | 1.3047 | +0.0403 |

Action loss improves on **every** unseen task. By difficulty bucket (mean Δ action loss):

| bucket | mean Δdyn | mean Δact |
|---|---|---|
| near-duplicate (2) | +0.048 | −0.119 |
| shared-primitive (5) | +0.029 | **−0.134** |
| novel (3) | +0.054 | −0.105 |

**Finding.** Retrieved-frame grounding substantially improves the **action** prediction — strongly
in-distribution (~41%) and consistently on novel tasks (~16%, every task) — at a small cost to video
dynamics loss (≈neutral in-dist, +3% on unseen). The shared-primitive bucket benefits most on action
(−0.134) and novel least (−0.105), the hypothesized ordering: retrieval transfers best when a related
motion primitive was seen in training. Action loss is the policy-relevant output, so this is a clear
positive signal for RICL on the world model.

Raw per-task JSON: `outputs/eval_results/eval{1,2}_{baseline,treatment}.json`; table: `eval_results/RESULTS.md`.

## Qualitative videos
Per task (`outputs/eval_results/videos/{train,eval}/<task>/`), from the RICL model with real DINOv2
retrieval: `generated.mp4` (the world model's predicted future, VAE-decoded), `retrieved.png`
(the K×X demo frames it retrieved), `obs.png`, `generated_strip.png`. 20 tasks.

## Closed-loop (sim success rate) — set up, blocked on a 3-way integration segfault
The RoboTwin SAPIEN sim is **set up in the same `emimic` env** (no separate env needed): sapien+mplib
were already present; built **curobo v0.7.8** (RoboTwin's pinned version) + **warp-lang 1.12.0** against
torch 2.7.1 via uv (`--no-build-isolation`, torch/numpy pinned). Verified individually: SAPIEN renders
on a100, and curobo `MotionGen` works standalone with torch 2.7.1.

**Blocker:** `record_rollout.py` (dz_plain/dz_ricl arms) segfaults (SIGSEGV) during curobo `MotionGen`
warmup **only when the 5B model + SAPIEN's Vulkan context are already resident** — a SAPIEN(Vulkan) +
curobo(warp) + torch(CUDA) context interaction. Each component works alone; the 3-way combination in one
process crashes.

**Path forward:** run the sim+planner in a **separate process** from the policy (policy-server bridge —
`eval_utils/serve_dreamzero_wan22.py` exists), or debug the Vulkan/warp/CUDA init ordering. The held-out
loss is the delivered metric; closed-loop success rate is the natural follow-up.

## Infra notes / fixes (this run)
- This cluster's `gpu-a100` nodes are **80 GB** (not 40 GB as CLAUDE.md states) and schedule instantly —
  fit 5B LoRA training (~47 GB resident) and finished in ~50 min vs. days waiting for h200/h100.
- `emimic` is a **uv-managed venv** (no pip; use `/storage/project/r-dxu345-0/rco3/uv pip ... --python emimic/bin/python`).
- Fixed 5 latent bugs surfaced by first-ever end-to-end runs of the RoboTwin DreamZero harness +
  no-grad eval: GR00T `cfg.transforms`/`merged_metadata` asserts (empty placeholders), `transformers`
  `_get_train_sampler(dataset)` signature, final-save tokenizer crash (use LoRA-aware `save_model`),
  DiT `_forward_train` tuple-unpack in the no-grad path, and the closed-loop quantiles wiring
  (use train quantiles). All in this branch.

## Key files
- Training: `scripts/train/robotwin_dreamzero.sbatch`, `groot/vla/experiment/train_robotwin_dreamzero.py`
  (quantile hook + LoRA final save), data shim `groot/vla/data/dataset/robotwin_dreamzero.py`.
- Eval: `scripts/eval/heldout_loss.py`, `aggregate_results.py`, `qualitative_video.py`, `run_all_heldout.sh`.
- Closed-loop: `egomimic/ricl/dreamzero_robotwin_policy.py`, `egomimic/ricl/scripts/record_rollout.py`
  (dz arms), `scripts/eval/run_closed_loop.sh`, curobo stub in `external/RoboTwin/envs/robot/planner.py`.
- Data: `egomimic/ricl/outputs/robotwin_10x10/{train,eval}` (10+10 tasks).
