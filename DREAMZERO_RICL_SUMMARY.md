# DreamZero 5B + RICL retrieved-frame grounding — work summary

Status as of 2026-06-21. Branch `ryanco/dreamzero-in-context`. Companion design/run doc:
`external/dreamzero/RICL_README.md` (architecture + alignment table + commands). This file is
the **project-state** summary: context, what was built, findings, and next steps.

---

## 1. Goal & context

Make DreamZero (Wan2.2-TI2V-5B "world action model" that jointly generates future **video +
actions**) condition on **retrieved real demonstrations**, mirroring this repo's RICL work on
pi0.5. For each query, retrieve **K** nearest demos (DINOv2 kNN) and feed **X frames per demo**
as in-context grounding. Hypothesis: showing the world model real demo frames of a similar task
improves generated video + predicted actions vs a plain finetune — measured by held-out loss and
RoboTwin closed-loop success.

**Locked decisions:** (1) inject retrieved frames as a **VAE-latent clean prefix** (same latent
space the DiT denoises); (2) **frames only** for v1 (retrieved actions/states = optional later
add-on via the text path); (3) first comparison = **same tasks, held-out episodes**; (4) train
from the **Wan2.2-5B base** (pretrained DiT + frozen VAE/T5/CLIP) — NOT a DreamZero DROID/AgiBot
checkpoint (`pretrained_model_path` stays `null`); (5) weights downloaded locally (no existing
checkpoints).

**Key realization:** DreamZero is already autoregressive over a CausVid teacher-forcing sequence
`[clean_video | noisy_video | action_register]` with an always-visible first-frame "sink".
Retrieved frames are just **more leading clean context frames** — VAE-encode them, prepend to the
clean half, give them early RoPE positions, let generated/action blocks attend back. The noisy
half (and thus action/state-register alignment) is untouched. `enable_retrieved_context=false`
(K=0) → bit-identical to stock DreamZero (the baseline).

---

## 2. Frame/action alignment (validated on the real VAE)

Wan VAE does **4× temporal** downsample: `num_frames` pixel → `T_lat=(num_frames-1)/4+1` latent.
Action horizon then follows the register-block math:
`action_horizon = (T_lat-1)/num_frame_per_block * num_action_per_block`.

| num_frames | T_lat | blocks | action_horizon | (num_frame_per_block=2, num_action_per_block=24) |
|---|---|---|---|---|
| **9** | 3 | 1 | **24** (default) | |
| 17 | 5 | 2 | 48 | |
| 33 | 9 | 4 | 96 | |

Use `num_frames = 8m+1`. Default **num_frames=9, action_horizon=24, max_state_dim=64**,
`max_chunk_size=8` (→ `local_attn_size=17 ≥ frames`, so the sliding window never evicts the
retrieved prefix). Retrieval default **K=2, X=4** (8 context frames).

---

## 3. What was implemented

### Model (training path) — `groot/vla/model/dreamzero/`
- `modules/wan_video_dit_action_casual_chunk.py` (`CausalWanModel`): retrieval config
  (`enable_retrieved_context`, `num_retrieved_demos` K, `frames_per_demo` X); asymmetric
  clean/noisy split + **dual RoPE** (`freqs_clean` over the full clean half incl. the R-frame
  prefix, `freqs` over the noisy tail) in `CausalWanSelfAttention.forward`; `clean_context_ends`
  offset by R in `_process_noisy_image_blocks` / `_process_noisy_action_blocks`;
  `num_retrieved_frames` threaded through `_forward_train` + `CausalWanAttentionBlock.forward`.
- `action_head/wan_flow_matching_action_tf.py`: `WANPolicyHeadConfig` retrieval fields;
  `_encode_retrieved_latents` (VAE-encode K·X frames, mask invalid demos) prepended to `clean_x`;
  `num_retrieved_frames` passed to the DiT.
- `transform/dreamzero_cotrain.py`: `DreamTransform` composes each retrieved demo's X frames into
  the same multiview grid as the observation (`retrieved_video` / `retrieved_mask` passthrough).

### Model (inference path) — same `action_head` file
- `lazy_joint_video_action` warm-up: encodes the R=K·X retrieved latents and populates them into
  **both** KV caches (pos+neg) at positions 0..R-1 before the obs frame; warm-up guards made
  `_num_retrieved_kv`-aware. Three inference bugs fixed (see §4).

### Data — `groot/vla/data/dataset/robotwin_dreamzero.py` (new)
- `RoboTwinDreamZeroDataset` reuses `egomimic.ricl.robotwin_data.RoboTwinCorpus` (native RoboTwin
  HDF5, no LeRobot conversion) + `DreamTransform`/`DefaultDataCollator`; yields obs clip,
  quantile-normalized+clipped state/action, `retrieved_video [K,X,V,H,W,3]` + `_mask`.
- Train queries use within-train LOO retrieval; held-out eval retrieves from the TRAIN bank (no
  leakage). Builders: `build_robotwin_dreamzero_data` / `_split` (Hydra `_target_`, memoized so
  the DINOv2 caches build once).
- `egomimic/ricl/robotwin_data.py`: added `make_robotwin_clip_provider` (`(hash,frame) → X
  multiview frames`) + `_clip_frame_indices` (forward/centered/even, bound-clamped).

### Rollout adapter — `egomimic/ricl/dreamzero_robotwin_policy.py` (new)
- `DreamZeroRoboTwinPolicy` mirrors `robotwin_policy.PIRiclPolicy`'s RoboTwin deploy contract
  (`get_model`/`eval`/`reset_model` + `set_language`/`update_observation_window`/`get_action`/
  `reset_obsrvationwindows`), so the SAME eval driver + `record_rollout.py` work.
- Build: Hydra `instantiate` with the SAME overrides as training → `load_lora_weight(ckpt)` →
  `post_initialize()` (bf16; fits a 40-48GB eval node). Per step: live DINOv2 retrieval
  (`DreamZeroOnlineRetriever`) → re-warm (`current_start_frame=0`) → `lazy_joint_video_action` →
  `unnormalize_action` to RoboTwin qpos.
- **Receding-horizon (re-warm) inference:** each `get_action()` resets the KV cache and feeds one
  obs frame, so every step independently encodes (retrieved prefix + current obs) → fresh chunk —
  the PIRicl contract. The autoregressive multi-call path is a deliberate future enhancement.
- View ordering: maps `img_arr` BY NAME to CAM_KEYS grid order (`_RGB_ORDER`=[head,RIGHT,LEFT] but
  CAM_KEYS=[base,LEFT,RIGHT] — a positional copy would swap wrists). `control_mode="joint"` (v1).
- `DreamZeroOnlineRetriever` exposes the same `.index.query`/`.bank_provider`/`.retrieve`/`.last`
  surface as `OnlineRetriever`, so `record_rollout.RetrievalTap` records the strip unchanged.

### Wiring — `egomimic/ricl/scripts/record_rollout.py`
- New `dz_plain` / `dz_ricl` arms (`build_policy` dispatches `dz_` → `_build_dz_policy`).
  `run_episode`/`RetrievalTap`/`encode_obs` reused unchanged. Checkpoints resolve from
  `external/dreamzero/outputs/rt_dz_{baseline,treatment}/checkpoint-N` (override `$DZ_PLAIN_DIR`/
  `$DZ_RICL_DIR`); bank/quantiles from `$DZ_BANK_ROOT` (MUST match training), index from
  `$DZ_BANK_INDEX`.

### Configs / launch (new)
- `configs/model/dreamzero/action_head/wan_flow_matching_action_tf_wan22_ricl.yaml`
- `configs/data/dreamzero/robotwin_wan22.yaml`
- `groot/vla/experiment/train_robotwin_dreamzero.py` (adds held-out eval set; asserts
  `pretrained_model_path is None`)
- `scripts/train/robotwin_dreamzero.sbatch` (MODE=baseline|treatment single knob)

### Environment (one-time; emimic untouched)
- Weights (~37 GB) in `external/dreamzero/checkpoints/`: Wan2.2-TI2V-5B (DiT+VAE+T5), Wan2.1 CLIP
  file, umt5-xxl tokenizer.
- DreamZero deps emimic lacks → PYTHONPATH **overlay** `external/dreamzero/_dzdeps`
  (peft, accelerate, ftfy, regex, dm-tree, albumentations==1.4.18, sentencepiece), via
  `uv pip install --target`. Run with
  `PYTHONPATH=external/dreamzero/_dzdeps:external/dreamzero:.` + `emimic/bin/python`.

---

## 4. Inference-path bugs fixed (this session)

The training path was already verified (CPU + GPU). Making the closed-loop **inference** path
work surfaced three issues (all in `action_head/wan_flow_matching_action_tf.py`):

1. **Retrieved VAE encode got LATENT dims, not PIXEL dims.** In `lazy_joint_video_action`,
   `height/width` are reused for the latent dims (`noise_obs.shape`) before the retrieved encode;
   passing those (10×20) made the VAE tiling reshape crash. Fix: capture `pixel_h, pixel_w`
   right after `videos.shape` and pass them (training already passed pixel dims — that's why it
   worked there).
2. **`trt_engine` attribute missing.** Set only in `post_initialize`; a bare Hydra-instantiated
   model hit `AttributeError` in `_run_diffusion_steps`. Fix: default `self.trt_engine = None`
   in `__init__`.
3. **Generation CFG combine requires `cfg_scale != 1.0`.** The loop always reads `predictions[0]`
   (cond) + `predictions[1]` (uncond), but `_prepare_text_inputs` returns a single context when
   `cfg_scale==1.0` → IndexError. The real eval path uses `cfg_scale=5.0` (2-pass CFG) and the
   transform already emits `text_negative`; the retrieved-prefix warm-up correctly populates both
   pos+neg KV caches (it passes the full `prompt_embs`/`kv_caches` lists). Fix: smoke uses the
   real CFG path (don't force cfg_scale=1).

**Design decision (re-warm vs autoregressive):** the autoregressive generation call needs ≥4
pixel frames (`videos.shape[2]//4 ≥ 1`; `FRAMES_PER_CHUNK=4` in the serve path) and KV-sink
preservation on truncation. For v1 we use **receding-horizon re-warm** (reset each step) — simpler,
matches PIRicl, fully validated by the warm-up path, and orthogonal to the RICL question.

---

## 5. Findings / verification (all smokes)

| Smoke | Where | Result |
|---|---|---|
| DiT kernel (grounds video+action) | `_ricl_dit_smoke.py` (CPU) | ✅ PASS |
| Dataset shapes | `_ricl_data_smoke.py` (CPU) | ✅ PASS |
| Hydra config wiring | `_ricl_config_smoke.py` (CPU) | ✅ PASS |
| Train fwd+bwd+step (baseline+treatment) | `_ricl_gpu_train_smoke.py` (h100) | ✅ PASS (finite loss both arms) |
| Inference re-warm (baseline+treatment) | `_ricl_gpu_infer_smoke.py` (h100) | ✅ PASS — treatment `retr_kv=8`, both produce `action_pred=(1,24,32)` |
| Adapter model-free glue | `_ricl_adapter_cpu_smoke.py` (CPU) | ✅ PASS (retriever shapes/masking, tap surface, dispatch) |
| Policy `get_action` end-to-end | `_ricl_policy_gpu_smoke.py` (a100) | ✅ baseline → `(24,14)` qpos; treatment → `(24,14)` + `retr_kv=8` |

Notable: the 5B model in **bf16 (`post_initialize`) fits a 40 GB a100** for eval (fp32 needs the
80 GB h100). The retrieved frames are confirmed to reach the KV cache at inference (`retr_kv=K·X=8`).

**No training comparison has been run yet** (per instruction: don't start full training). So there
is **no result yet** on whether retrieved-frame grounding helps held-out loss or RoboTwin success.

---

## 6. How to run

### Train (both arms, from the Wan2.2-5B base, single h100/h200, LoRA, no DeepSpeed)
```bash
MODE=baseline  ROBOTWIN_ROOT=/.../<task>__aloha-agilex_clean \
   sbatch external/dreamzero/scripts/train/robotwin_dreamzero.sbatch
MODE=treatment ROBOTWIN_ROOT=/.../<task>__aloha-agilex_clean K=2 X=4 \
   sbatch external/dreamzero/scripts/train/robotwin_dreamzero.sbatch
```
Compare `dynamics_loss`/`action_loss` on the **held-out eval** episodes (judge on val, not train).
Tiny smoke: add `MAX_STEPS=4 EMBED=fake`.

### Closed-loop rollout (after training + a per-task DINOv2 bank index)
```bash
# build the bank index once per task (train split):
emimic/bin/python egomimic/ricl/scripts/build_robotwin_bank_index.py ...   # -> _index/

DZ_PLAIN_DIR=external/dreamzero/outputs/rt_dz_baseline \
DZ_RICL_DIR=external/dreamzero/outputs/rt_dz_treatment \
DZ_BANK_ROOT=/.../<task>__aloha-agilex_clean DZ_BANK_INDEX=/.../<task>/_index \
PYTHONPATH=external/dreamzero/_dzdeps:external/dreamzero:. TORCH_COMPILE_DISABLE=1 HF_HUB_OFFLINE=1 \
emimic/bin/python egomimic/ricl/scripts/record_rollout.py \
  --task_name <task> --task_config demo_clean --models dz_plain dz_ricl
```

---

## 7. Next steps / open items

1. **Run the training comparison** (baseline vs treatment) on one or more RoboTwin tasks; compare
   held-out `dynamics_loss`/`action_loss`. This is the first real signal — currently unrun.
2. **Closed-loop rollout** needs the RoboTwin submodule + sim deps (sapien/mplib/curobo/pytorch3d;
   install selectively — do NOT run RoboTwin's `_install.sh`, it pins torch 2.4.1 and breaks
   emimic 2.7.1) and a per-task DINOv2 bank index. Then run `record_rollout.py --models dz_plain
   dz_ricl` and compare success on the SAME held-out seeds.
3. **Bank/quantiles correctness:** the rollout bank_root MUST be the task dir DreamZero trained on
   (quantiles are recomputed from it and must match training); build the index over the TRAIN
   split only. Wired via `$DZ_BANK_ROOT`/`$DZ_BANK_INDEX`.
4. **Autoregressive rollout (optional upgrade):** carry video history across steps instead of
   re-warming — needs ≥4 pixel frames/step + KV-sink preservation on truncation. Only if re-warm
   underperforms.
5. **Follow-ons:** retrieved actions/states via the T5 text path (RICL-style, zero attention-mask
   surgery); EE-pose control mode; sweep K/X.

---

## 8. Gotchas worth remembering
- pi0.5 weights ≈ 3.6 B; **DreamZero is 5.6 B** — training needs ≥80 GB (h200/h100); eval fits
  40-48 GB in **bf16** (a100/l40s). `gpu-rtxpro-blackwell` is torch-incompatible (no sm_120) —
  exclude it even though it schedules first.
- `num_frames` is PIXEL frames; the VAE 4× temporal downsample means `action_horizon` is fixed by
  the alignment table — get it wrong and the action-register assertion fails.
- Both `lazy_joint_video_action` and `lazy_joint_video_action_causal` call the SAME action-head
  method, so the RICL surgery covers both entry points.
- The identity backbone means all heavy params live in the action_head → `post_initialize()` casts
  everything to bf16.
