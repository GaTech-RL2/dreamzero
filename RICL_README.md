# DreamZero + RICL: retrieved-frame in-context grounding

Grounds DreamZero's (Wan2.2-TI2V-5B) video+action generation in real demos, RICL-style:
for each query, retrieve **K** nearest demos (DINOv2 kNN) and feed **X** frames per demo
as **clean context frames** prepended to the model's video sequence. Goal: does showing the
world-model real demo frames improve generation vs a plain finetune?

**Training is from the Wan2.2-5B base** (pretrained DiT + frozen VAE/T5/CLIP) — NOT from a
DreamZero DROID/AgiBot checkpoint (`pretrained_model_path` stays `null`).

## Design (the key idea)

DreamZero is already an autoregressive video model with a CausVid teacher-forcing sequence
`[clean_video | noisy_video | action_register]`. Retrieved frames are just **more leading
clean context frames**: VAE-encode K·X retrieved frames (each independently, T=1 → 1 clean
latent frame each), prepend them to the **clean** half only, give them early RoPE positions,
and let the generated/action blocks attend to them. The noisy half is unchanged, so the
action/state register block alignment is untouched.

- In the teacher-forcing kernel, generated blocks already attend **globally** to all clean
  context (no sliding window on noisy→clean), so the prefix grounds generation directly.
- For RoboTwin's short episodes we set `max_chunk_size=8` (`local_attn_size=17 ≥ frames`) so
  the window never evicts the prefix.
- `enable_retrieved_context=false` (K=0) → bit-identical to stock DreamZero (the baseline).

## What changed

**Model (training path) — `groot/vla/model/dreamzero/`** ✅ CPU-verified
- `modules/wan_video_dit_action_casual_chunk.py`: `CausalWanModel` config
  (`enable_retrieved_context`, `num_retrieved_demos` K, `frames_per_demo` X); asymmetric
  clean/noisy split + dual RoPE (`freqs_clean` full, `freqs` = noisy tail) in
  `CausalWanSelfAttention.forward`; `clean_context_ends` offset by R in
  `_process_noisy_image_blocks` / `_process_noisy_action_blocks`; `num_retrieved_frames`
  threaded through `_forward_train` + `CausalWanAttentionBlock.forward`.
- `action_head/wan_flow_matching_action_tf.py`: `WANPolicyHeadConfig` retrieval fields;
  `_encode_retrieved_latents` (independent T=1 VAE encode, mask invalid demos) prepended to
  `clean_x`; `num_retrieved_frames` passed to the DiT. **Inference path** (`lazy_joint_video_action`,
  warm-up branch): encodes the R=K·X retrieved latents and populates them into BOTH KV caches
  (pos+neg, via `prompt_embs`/`kv_caches` lists) at positions 0..R-1 before the obs frame, with
  `_num_retrieved_kv`-aware warm-up guards. Three inference fixes were needed: pass PIXEL (not
  latent) dims to the retrieved VAE encode; default `trt_engine=None` in `__init__` (set only in
  `post_initialize`); the generation loop's CFG combine requires `cfg_scale!=1.0` (the real path).
- `transform/dreamzero_cotrain.py`: `DreamTransform` composes each retrieved demo's X frames
  into the same multiview grid as the observation (passthrough of `retrieved_video`/`_mask`).

**Data — `groot/vla/data/dataset/robotwin_dreamzero.py`** ✅ CPU-verified
- `RoboTwinDreamZeroDataset` reuses `egomimic.ricl.robotwin_data.RoboTwinCorpus` (native HDF5,
  no LeRobot conversion) + `DreamTransform`/`DefaultDataCollator`; yields the obs clip,
  normalized state/action (clipped to [-1,1]), and `retrieved_video [K,X,V,H,W,3]` + `_mask`.
- Train queries use within-train LOO retrieval; held-out eval queries retrieve from the TRAIN
  bank (no leakage). Clip provider: `egomimic.ricl.robotwin_data.make_robotwin_clip_provider`.

**Configs / launch** ✅ Hydra-compose-verified
- `configs/model/dreamzero/action_head/wan_flow_matching_action_tf_wan22_ricl.yaml`
- `configs/data/dreamzero/robotwin_wan22.yaml`
- `groot/vla/experiment/train_robotwin_dreamzero.py` (adds held-out eval set)
- `scripts/train/robotwin_dreamzero.sh`/`.sbatch`

## Frame/action alignment (validated on the real VAE)

The Wan VAE does **4× temporal** downsample: `num_frames` pixel frames → `T_lat = (num_frames-1)/4 + 1`
latent frames. The action register block math then fixes the action horizon:
`action_horizon = (T_lat-1)/num_frame_per_block * num_action_per_block`. With
`num_frame_per_block=2, num_action_per_block=24`:

| num_frames | T_lat | blocks | action_horizon |
|---|---|---|---|
| **9** | 3 | 1 | **24** (default) |
| 17 | 5 | 2 | 48 |
| 33 | 9 | 4 | 96 |

Use `num_frames = 8m+1`. Default is **num_frames=9, action_horizon=24**, `max_state_dim=64`.

## Env setup (one-time; emimic untouched)

Weights (~37 GB) → `external/dreamzero/checkpoints/` (Wan2.2-TI2V-5B = DiT+VAE+T5, Wan2.1 CLIP file,
umt5-xxl tokenizer): `huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ...` etc.
DreamZero deps that emimic lacks go in a PYTHONPATH **overlay** (`_dzdeps`), leaving emimic intact:

```bash
uv pip install --python emimic/bin/python --target external/dreamzero/_dzdeps \
   peft accelerate ftfy regex dm-tree "albumentations==1.4.18" sentencepiece
```

Run with `PYTHONPATH=external/dreamzero/_dzdeps:external/dreamzero:.` + `emimic/bin/python`.

## Run (Phase 1: training + held-out val-loss comparison)

Needs a RoboTwin task dir + the env above. Train fits on one h100/h200 (single-GPU, LoRA, no
DeepSpeed). **Always run on a salloc'd node, never the login node.**

```bash
# baseline (no retrieval) and treatment (K=2, X=4 frames/demo) — same data, same hyperparams
MODE=baseline  ROBOTWIN_ROOT=/.../robotwin_raw_mt/extracted/<task>__aloha-agilex_clean \
   sbatch external/dreamzero/scripts/train/robotwin_dreamzero.sbatch
MODE=treatment ROBOTWIN_ROOT=/.../<task>__aloha-agilex_clean K=2 X=4 \
   sbatch external/dreamzero/scripts/train/robotwin_dreamzero.sbatch
```

Compare `dynamics_loss` / `action_loss` on the **held-out eval** episodes (judge on val loss,
not train loss). Tiny smoke first: add `MAX_STEPS=4 EMBED=fake` (fake embeddings skip DINOv2).

## GPU train smoke ✅ PASSED (h100)

`_ricl_gpu_train_smoke.py` loads the real Wan2.2-5B base, builds a RoboTwin batch, and runs
forward+backward+optimizer.step for BOTH baseline and treatment (K=2,X=4) with finite loss:
```
ROBOTWIN_ROOT=<task dir> NUM_FRAMES=9 ACTION_HORIZON=24 \
PYTHONPATH=external/dreamzero/_dzdeps:external/dreamzero:. TORCH_COMPILE_DISABLE=1 HF_HUB_OFFLINE=1 \
emimic/bin/python external/dreamzero/_ricl_gpu_train_smoke.py
# [treatment] loss=4.20 ... backward OK ... step OK   [baseline] loss=2.25 ... OK
```

## GPU inference smoke ✅ PASSED (h100)

`_ricl_gpu_infer_smoke.py` runs the closed-loop INFERENCE path (`lazy_joint_video_action`) for
BOTH arms in the re-warm protocol the adapter uses:
```
[treatment] step0 action_pred=(1,24,32) cs=11 retr_kv=8 out_keys=['action_pred','video_pred']
[baseline]  step0 action_pred=(1,24,32) cs=3  retr_kv=0 out_keys=['action_pred','video_pred']
```
`retr_kv=8` = the K·X=8 retrieved frames were VAE-encoded into the KV cache (treatment); both
arms produce an action chunk. **Receding-horizon design:** each `get_action()` resets
`current_start_frame=0` (re-warm) and feeds ONE obs frame — each step independently encodes
(retrieved prefix + current obs) → fresh chunk, mirroring the PIRicl contract. (The
autoregressive multi-call generation path is a future enhancement: it needs ≥4 pixel frames per
generation step — `videos.shape[2]//4` must be ≥1 — plus KV-sink preservation on truncation.)

## CPU smokes (re-runnable; run on a salloc'd node)

```bash
salloc -A gts-dxu345-rl2 -N1 -q inferno -t 1:00:00 --mem=16G --cpus-per-task=4 --no-shell
J=<jobid>
srun --jobid=$J bash -lc 'cd <repo> && ATTENTION_BACKEND=torch PYTHONPATH=external/dreamzero:. \
   emimic/bin/python external/dreamzero/_ricl_dit_smoke.py'      # kernel: grounds video+action
srun --jobid=$J bash -lc 'cd <repo> && PYTHONPATH=external/dreamzero:. \
   emimic/bin/python external/dreamzero/_ricl_data_smoke.py'     # dataset shapes
srun --jobid=$J bash -lc 'cd <repo> && PYTHONPATH=external/dreamzero:. \
   emimic/bin/python external/dreamzero/_ricl_config_smoke.py'   # Hydra wiring
```

## Phase 2 ✅ DONE: closed-loop RoboTwin rollout

**Inference-path prefix** (above): retrieved latents populate both KV caches during warm-up.
**Adapter** `egomimic/ricl/dreamzero_robotwin_policy.py` — `DreamZeroRoboTwinPolicy` mirrors
`robotwin_policy.PIRiclPolicy`'s RoboTwin deploy contract (`get_model`/`eval`/`reset_model` +
`set_language`/`update_observation_window`/`get_action`/`reset_obsrvationwindows`), so the SAME
eval driver + `record_rollout.py` machinery work. It builds Wan2.2-5B via Hydra (same overrides
as training) → `load_lora_weight(ckpt)` → `post_initialize()` (bf16, fits a 40-48GB eval node);
per step it runs live DINOv2 retrieval (`DreamZeroOnlineRetriever`, returns
`retrieved_video`/`_mask`), re-warms, and `lazy_joint_video_action` → un-normalizes the action
to RoboTwin qpos. Views are mapped BY NAME to CAM_KEYS order (`_RGB_ORDER` is [head,RIGHT,LEFT]
but the grid is [base,LEFT,RIGHT] — a positional copy would swap wrists). `control_mode="joint"`
(joint-space v1; EE is a follow-on).

**Wiring** — `record_rollout.py` gains `dz_plain` / `dz_ricl` arms (`build_policy` dispatches on
`dz_` → `_build_dz_policy`; `RetrievalTap`/`run_episode`/`encode_obs` are reused unchanged — the
DreamZero retriever exposes the same `.index.query`/`.bank_provider`/`.retrieve`/`.last` surface).
Checkpoints resolve from `external/dreamzero/outputs/rt_dz_{baseline,treatment}/checkpoint-N`
(override per arm with `$DZ_PLAIN_DIR` / `$DZ_RICL_DIR`).

Run (after training both arms + a DINOv2 bank index per task via
`egomimic/ricl/scripts/build_robotwin_bank_index.py`; needs the RoboTwin submodule + sim deps):
```bash
DZ_PLAIN_DIR=external/dreamzero/outputs/rt_dz_baseline \
DZ_RICL_DIR=external/dreamzero/outputs/rt_dz_treatment \
PYTHONPATH=external/dreamzero/_dzdeps:external/dreamzero:. TORCH_COMPILE_DISABLE=1 HF_HUB_OFFLINE=1 \
emimic/bin/python egomimic/ricl/scripts/record_rollout.py \
  --task_name beat_block_hammer --task_config demo_clean --models dz_plain dz_ricl
```
Compare success rate on the SAME held-out seeds for both arms.

### Phase 2 smokes ✅
- `_ricl_policy_gpu_smoke.py` (a100/l40s, bf16) — builds the REAL `DreamZeroRoboTwinPolicy`
  (untrained base) and runs `get_action` end-to-end for baseline + treatment (stub retriever
  over the real bank, no DINOv2 index needed) → action chunk `(pi0_step, state_dim)`, treatment
  `_num_retrieved_kv==K·X`.
- `_ricl_adapter_cpu_smoke.py` (CPU) — model-free glue: `DreamZeroOnlineRetriever` shapes +
  masking + `RetrievalTap` surface, CAM_KEYS ordering, `record_rollout` `dz_*` dispatch +
  `_latest_hf_checkpoint`.
