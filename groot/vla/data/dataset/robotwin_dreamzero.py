"""RoboTwin -> DreamZero dataset (thin shim, no LeRobot conversion).

Feeds RoboTwin 2.0 bimanual expert demos (native HDF5) into DreamZero's training
loop by REUSING EgoVerse's RoboTwin reader (``egomimic.ricl.robotwin_data``) and
DreamZero's own ``DreamTransform`` / ``DefaultDataCollator``. Mirrors the proven
"thin shim, not a port" pattern used for pi0.5 RICL (``train_robotwin_ricl.py``).

What this yields per item (after ``DreamTransform``):
  images [T,Hg,Wg,3] uint8 (multiview grid), text (str), state/action (+masks),
  embodiment_id, and — when retrieval is enabled — RICL grounding keys
  ``retrieved_video [K,X,V,H,W,3]`` + ``retrieved_mask [K]`` (composed into the
  same grid by DreamTransform, VAE-encoded as the clean prefix by the action head).

Requires ``egomimic`` on PYTHONPATH (same env that runs pi0.5 RICL). Embodiment is
masqueraded as ``xdof`` (head/left/right -> the 2x2 head/right/left/black grid that
DreamZero's collate + _prepare_video already handle); the DiT forces category 0
internally, so only the collate TEXT template depends on the tag.
"""
from __future__ import annotations

import os

import numpy as np
import yaml
from torch.utils.data import Dataset

# RoboTwin reader + retrieval live in egomimic (shared with pi0.5 RICL).
from egomimic.ricl.robotwin_data import (
    CAM_KEYS,
    RoboTwinCorpus,
    make_robotwin_clip_provider,
    make_dinov2_embedding_provider,
    make_fake_embedding_provider,
    build_robotwin_retrieval_cache,
    build_cross_embodiment_retrieval_cache,
)

DEFAULT_EMBODIMENT = "xdof"  # head/right/left/black 2x2 grid template in DreamZero's collate


class RoboTwinDreamZeroDataset(Dataset):
    """Per-frame RoboTwin dataset for DreamZero training.

    Args:
        query_corpus:    RoboTwinCorpus the OBSERVATION clip / state / action come from.
        transform:       a DreamTransform (already configured with embodiment_tag +
                         mapping). If None, __getitem__ returns the RAW data dict
                         (for a tokenizer-free shape smoke).
        num_frames:      pixel frames in the observation clip (mirror DROID: 33).
        action_horizon:  action steps per sample (mirror DROID: 24).
        retrieval_cache: a RICL RetrievalCache (neighbors per query frame) or None
                         (baseline / no grounding).
        clip_provider:   make_robotwin_clip_provider over the BANK corpus (resolves
                         neighbor hashes -> X multiview frames). Required if cache set.
        num_retrieved_demos / frames_per_demo: K / X.
        hashes:          restrict to these query episodes (train/eval split).
    """

    def __init__(
        self,
        query_corpus: RoboTwinCorpus,
        transform=None,
        num_frames: int = 33,
        action_horizon: int = 24,
        retrieval_cache=None,
        clip_provider=None,
        num_retrieved_demos: int = 0,
        frames_per_demo: int = 0,
        hashes: list[str] | None = None,
    ):
        self.corpus = query_corpus
        # BaseExperiment asserts train_dataset.merged_metadata is not None and dumps it to
        # metadata.json (GR00T normalization metadata). The RoboTwin shim normalizes in the
        # dataset (quantile_norm) and persists quantiles.json instead, so this is an empty
        # no-op placeholder ({} is not None; .items() dumps to an empty metadata.json).
        self.merged_metadata = {}
        self.transform = transform
        self.num_frames = int(num_frames)
        self.action_horizon = int(action_horizon)
        self.cache = retrieval_cache
        self.clip_provider = clip_provider
        self.K = int(num_retrieved_demos)
        self.X = int(frames_per_demo)
        if self.cache is not None:
            assert self.clip_provider is not None and self.K > 0, \
                "retrieval_cache set but clip_provider/num_retrieved_demos missing"
        hashes = hashes if hashes is not None else self.corpus.hashes
        self.index: list[tuple[str, int]] = [
            (h, fi) for h in hashes for fi in range(self.corpus.num_frames(h))
        ]
        self._cam_hw = self.corpus.image_hw

    def __len__(self) -> int:
        return len(self.index)

    def _observation_video(self, h: str, fi: int) -> np.ndarray:
        """[T, V, H, W, C] uint8 observation clip (num_frames consecutive frames)."""
        T = self.corpus.num_frames(h)
        fis = [min(fi + t, T - 1) for t in range(self.num_frames)]
        return np.stack(
            [
                np.stack([self.corpus.image(h, f, cam) for cam in CAM_KEYS], axis=0)
                for f in fis
            ],
            axis=0,
        )  # [T, V, H, W, C]

    def _retrieved(self, h: str, fi: int) -> tuple[np.ndarray, np.ndarray]:
        """[K, X, V, H, W, C] uint8 retrieved demo clips + [K] bool validity."""
        bh, bf, _ = self.cache.neighbors(h, fi)
        rv, mask = [], []
        empty = np.zeros((self.X, len(CAM_KEYS), *self._cam_hw, 3), dtype=np.uint8)
        for j in range(self.K):
            hj = str(bh[j]) if j < len(bh) else ""
            fj = int(bf[j]) if j < len(bf) else -1
            if hj and fj >= 0:
                rv.append(self.clip_provider(hj, fj)["frames"].astype(np.uint8))
                mask.append(True)
            else:
                rv.append(empty)
                mask.append(False)
        return np.stack(rv, axis=0), np.asarray(mask, dtype=bool)

    def __getitem__(self, idx: int) -> dict:
        h, fi = self.index[idx]
        c = self.corpus
        # quantile-norm to [-1,1] and CLIP (values beyond q01/q99 exceed +/-1; the action
        # head hard-asserts actions in [-1,1]).
        state = np.clip(c.quantile_norm(c.state(h, fi), "state"), -1.0, 1.0)[None, :]
        action = np.clip(
            c.quantile_norm(c.action_chunk(h, fi, self.action_horizon), "actions"), -1.0, 1.0
        )  # [action_horizon, state_dim]
        data = {
            "video": self._observation_video(h, fi),
            "state": state.astype(np.float32),
            "action": action.astype(np.float32),
            "annotation.task": c.prompt(h),
        }
        if self.cache is not None:
            data["retrieved_video"], data["retrieved_mask"] = self._retrieved(h, fi)
        if self.transform is None:
            return data
        return self.transform(data)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _embodiment_mapping() -> dict:
    """DreamZero's embodiment_tag -> projector index (read from the transform base config)."""
    base = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "configs", "model", "dreamzero", "transform", "base.yaml",
    )
    with open(os.path.abspath(base)) as f:
        cfg = yaml.safe_load(f)
    return cfg["embodiment_tag_to_projector_index"]


def build_dreamzero_transform(
    action_horizon: int = 24,
    state_horizon: int = 1,
    max_state_dim: int = 44,
    max_action_dim: int = 32,
    max_length: int = 512,
    num_views: int = 3,
    tokenizer_path: str = "google/umt5-xxl",
    embodiment: str = DEFAULT_EMBODIMENT,
    default_instruction: str = "Perform the default behavior.",
):
    """Build a DreamTransform + DefaultDataCollator for RoboTwin (xdof masquerade)."""
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
        DreamTransform,
        DefaultDataCollator,
    )

    mapping = _embodiment_mapping()
    transform = DreamTransform(
        default_instruction=default_instruction,
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        max_length=max_length,
        state_horizon=state_horizon,
        action_horizon=action_horizon,
        num_views=num_views,
        embodiment_tag_mapping=mapping,
        tokenizer_path=tokenizer_path,
        training=True,
    )
    transform.embodiment_tag = EmbodimentTag(embodiment)
    collator = DefaultDataCollator(
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        num_views=num_views,
        embodiment_tag_mapping=mapping,
    )
    return transform, collator


def split_hashes_per_group(corpus: RoboTwinCorpus, eval_frac: float = 0.2, seed: int = 0):
    """Hold out the last `eval_frac` of each task group's episodes for eval."""
    rng = np.random.default_rng(seed)
    train, eval = [], []
    for _g, hs in corpus.group_to_hashes.items():
        hs = list(hs)
        rng.shuffle(hs)
        n_eval = max(1, int(round(len(hs) * eval_frac))) if len(hs) > 1 else 0
        eval.extend(hs[:n_eval])
        train.extend(hs[n_eval:])
    return train, eval


def build_robotwin_dreamzero_data(
    root: str,
    num_retrieved_demos: int = 0,
    frames_per_demo: int = 4,
    clip_stride: int = 1,
    clip_mode: str = "forward",
    num_frames: int = 33,
    action_horizon: int = 24,
    state_horizon: int = 1,
    embed: str = "dinov2",
    eval_frac: float = 0.2,
    transform=None,
    seed: int = 0,
):
    """Build (train_ds, eval_ds, collator) for DreamZero on RoboTwin.

    - Baseline (num_retrieved_demos==0): no retrieval; plain DreamZero finetune.
    - Treatment (K>0): DINOv2 kNN retrieval. Train queries use within-train LOO;
      eval queries (held-out episodes) retrieve from the TRAIN bank (no leakage).
    """
    corpus = RoboTwinCorpus(root, mode="joint")
    train_hashes, eval_hashes = split_hashes_per_group(corpus, eval_frac, seed)

    if transform is None:
        transform, collator = build_dreamzero_transform(
            action_horizon=action_horizon, state_horizon=state_horizon
        )
    else:
        collator = None  # Hydra path supplies the collator via cfg.data_collator

    train_cache = eval_cache = clip_provider = None
    if num_retrieved_demos > 0:
        make_embed = (
            make_fake_embedding_provider if embed == "fake" else make_dinov2_embedding_provider
        )
        # Bank = TRAIN episodes only (held-out eval must not retrieve itself).
        train_corpus = RoboTwinCorpus(root, mode="joint", quantiles=corpus.quantiles)
        # restrict the bank corpus index to train hashes by building caches over subsets
        embed_provider = make_embed(corpus)
        train_cache = build_robotwin_retrieval_cache(
            _subset(corpus, train_hashes), num_retrieved_demos, embed_provider
        )
        if eval_hashes:
            eval_cache = build_cross_embodiment_retrieval_cache(
                _subset(corpus, eval_hashes), _subset(corpus, train_hashes),
                num_retrieved_demos, embed_provider, embed_provider,
            )
        clip_provider = make_robotwin_clip_provider(
            corpus, frames_per_demo=frames_per_demo, stride=clip_stride, mode=clip_mode
        )

    train_ds = RoboTwinDreamZeroDataset(
        corpus, transform, num_frames, action_horizon,
        retrieval_cache=train_cache, clip_provider=clip_provider,
        num_retrieved_demos=num_retrieved_demos, frames_per_demo=frames_per_demo,
        hashes=train_hashes,
    )
    eval_ds = RoboTwinDreamZeroDataset(
        corpus, transform, num_frames, action_horizon,
        retrieval_cache=eval_cache, clip_provider=clip_provider,
        num_retrieved_demos=num_retrieved_demos, frames_per_demo=frames_per_demo,
        hashes=eval_hashes,
    ) if eval_hashes else None
    return train_ds, eval_ds, collator


# Memoized shared state so Hydra instantiating train + eval datasets builds the
# (expensive) DINOv2 retrieval caches only ONCE per config.
_SPLIT_CACHE: dict = {}


def build_robotwin_dreamzero_split(
    split: str,
    root: str,
    transform=None,
    embodiment: str = DEFAULT_EMBODIMENT,
    num_retrieved_demos: int = 0,
    frames_per_demo: int = 4,
    clip_stride: int = 1,
    clip_mode: str = "forward",
    num_frames: int = 33,
    action_horizon: int = 24,
    state_horizon: int = 1,
    embed: str = "dinov2",
    eval_frac: float = 0.2,
    seed: int = 0,
):
    """Return ONE split ('train'|'eval') Dataset for use as a Hydra ``_target_``.

    Reuses build_robotwin_dreamzero_data (memoized by config) so train+eval share the
    same corpus + retrieval caches. ``transform`` is the DreamZero model_specific_transform
    (a DreamTransform); its embodiment_tag is set to ``embodiment`` (xdof) here.
    """
    from groot.vla.data.schema import EmbodimentTag

    if transform is not None and getattr(transform, "embodiment_tag", None) is None:
        transform.embodiment_tag = EmbodimentTag(embodiment)

    key = (os.path.abspath(root), num_retrieved_demos, frames_per_demo, clip_stride,
           clip_mode, num_frames, action_horizon, state_horizon, embed, eval_frac, seed)
    if key not in _SPLIT_CACHE:
        train_ds, eval_ds, _ = build_robotwin_dreamzero_data(
            root=root, num_retrieved_demos=num_retrieved_demos, frames_per_demo=frames_per_demo,
            clip_stride=clip_stride, clip_mode=clip_mode, num_frames=num_frames,
            action_horizon=action_horizon, state_horizon=state_horizon, embed=embed,
            eval_frac=eval_frac, transform=transform, seed=seed,
        )
        _SPLIT_CACHE[key] = {"train": train_ds, "eval": eval_ds}
    return _SPLIT_CACHE[key][split]


def _subset(corpus: RoboTwinCorpus, hashes: list[str]) -> RoboTwinCorpus:
    """A lightweight view of `corpus` restricted to `hashes` (shares caches/quantiles)."""
    import copy

    sub = copy.copy(corpus)
    keep = set(hashes)
    sub.hash_to_path = type(corpus.hash_to_path)(
        (h, p) for h, p in corpus.hash_to_path.items() if h in keep
    )
    sub.group_to_hashes = type(corpus.group_to_hashes)(
        (g, [h for h in hs if h in keep])
        for g, hs in corpus.group_to_hashes.items()
        if any(h in keep for h in hs)
    )
    return sub
