"""DreamZero training entrypoint for RoboTwin (RICL retrieved-frame grounding).

Identical to ``experiment.py`` except it also builds a HELD-OUT eval dataset from
``cfg.eval_dataset`` (BaseExperiment.create_val_dataset returns None by default), so
runs can be judged on validation loss over new episodes (not just train loss).

Baseline vs treatment is a single knob (see configs/.../wan_flow_matching_action_tf_wan22_ricl):
  baseline  : enable_retrieved_context=false num_retrieved_demos=0
  treatment : enable_retrieved_context=true  num_retrieved_demos=2 frames_per_demo=4

Launch via scripts/train/robotwin_dreamzero.sh (mirrors droid_training_wan22.sh; keeps
pretrained_model_path=null so we train the Wan2.2-5B base, NOT a DreamZero DROID/AgiBot ckpt).
"""
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from groot.vla.experiment.experiment import VLAExperiment


class RoboTwinVLAExperiment(VLAExperiment):
    def create_val_dataset(self, cfg, model):
        eval_cfg = OmegaConf.select(cfg, "eval_dataset")
        if eval_cfg is None:
            return None
        return instantiate(cfg.eval_dataset)

    def train(self):
        # Mirror BaseExperiment.train() but do the FINAL save via the LoRA-aware
        # trainer.save_model (respects save_lora_only, skips the tokenizer) instead of
        # safe_save_model_for_hf_trainer -> trainer._save. The latter crashes here because
        # transformers' _save falls back to data_collator.tokenizer.save_pretrained when
        # processing_class is None, and DreamZero's HuggingfaceTokenizer has no
        # save_pretrained; it would also dump the full 5B instead of LoRA-only. Using
        # save_model makes the root save match the checkpoint-N format the eval adapter loads.
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.trainer.save_state()
        if getattr(self.trainer, "deepspeed", None):
            from groot.vla.experiment.utils import safe_save_model_for_hf_trainer
            safe_save_model_for_hf_trainer(self.trainer, self.training_args.output_dir)
        else:
            self.trainer.save_model(self.training_args.output_dir, _internal_call=True)
        self._save_quantiles()

    def _save_quantiles(self):
        """Persist the training corpus's quantile norm stats beside the checkpoints.

        Eval (held-out + closed-loop) MUST normalize states / un-normalize actions with
        the SAME q01/q99 the model trained on — including Eval 2 on unseen tasks, where
        new-task observations are normalized with the TRAIN quantiles. Mirrors the pi0.5
        RICL trainer (egomimic/ricl/scripts/train_robotwin_ricl.py).
        """
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        corpus = getattr(self.train_dataset, "corpus", None)
        if corpus is None or not hasattr(corpus, "save_quantiles"):
            print(
                "[post-train] train_dataset has no .corpus.save_quantiles; skipping "
                "quantiles dump (reconstruct manually: "
                "RoboTwinCorpus(<train_root>).save_quantiles(<out>/quantiles.json))",
                flush=True,
            )
            return
        out = self.exp_cfg_dir.parent / "quantiles.json"
        corpus.save_quantiles(str(out))
        print(f"[post-train] saved quantiles -> {out}", flush=True)


@hydra.main(config_path="../configs", config_name="conf", version_base=None)
def main(cfg):
    assert getattr(cfg, "pretrained_model_path", None) is None, (
        "pretrained_model_path must stay null: train the Wan2.2-5B base, not a DreamZero "
        "DROID/AgiBot checkpoint."
    )
    experiment = RoboTwinVLAExperiment(cfg)
    experiment.train()


if __name__ == "__main__":
    main()
