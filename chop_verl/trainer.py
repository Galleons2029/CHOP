"""verl RayPPOTrainer subclass running the CHOP step instead of PPO's advantage/update path.

Everything outside the update is verl: data loading, vLLM rollouts through the AgentLoop manager
(multi-turn OpenHands loop for SWE-bench, rolling-truncation loop for long math), reward computation,
checkpointing, FSDP -> vLLM weight synchronisation, validation and logging.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.trainer.ppo.reward import extract_reward
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking

from chop.monitors import CollapseGuard
from chop.router import CellRouter
from chop_verl.driver import build_payload, chop_step
from chop_verl.worker import method_config


class ChopRayTrainer(RayPPOTrainer):
    # ---------------------------------------------------------------- state
    def _chop_setup(self) -> None:
        chop = self.config.actor_rollout_ref.chop
        self.chop_cfg = method_config(chop.method, chop.get("method_file"))
        self.chop_router = CellRouter(self.chop_cfg.router, seed=int(chop.get("seed", 0)))
        self.chop_guard = CollapseGuard(self.chop_cfg.segment.rank_fallback_threshold)
        self.chop_native_context = int(chop.native_context)
        self.chop_rollout_chunk = int(chop.rollout_chunk)
        self.chop_need_pi = bool(self.chop_cfg.pi.sources) and self.chop_cfg.router.kappa_gate != "none"
        calib = chop.get("calibration_states")
        if calib:  # pre-fill the streaming buffers from the 1024-state calibration corpus
            state = torch.load(calib)
            self.chop_router.rho_q.initialize(state["rho"])
            if state.get("kappa") is not None:
                self.chop_router.kappa_q.initialize(state["kappa"])

    def _chop_state_path(self, step: int) -> str:
        return os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}", "chop_state.pt")

    def _save_checkpoint(self):
        super()._save_checkpoint()
        path = self._chop_state_path(self.global_steps)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"router": self.chop_router.state_dict(), "guard": self.chop_guard.state_dict()}, path)

    def _load_chop_state(self) -> None:
        path = self._chop_state_path(self.global_steps)
        if self.global_steps and os.path.exists(path):
            state = torch.load(path, weights_only=False)
            self.chop_router.load_state_dict(state["router"])
            self.chop_guard.load_state_dict(state["guard"])

    # -------------------------------------------------------------- batches
    def _payloads(self, batch: DataProto, rewards: torch.Tensor):
        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        resp_mask = batch.batch["response_mask"]
        attn = batch.batch["attention_mask"]
        plen = prompts.shape[1]
        extras = batch.non_tensor_batch.get("extra_info", [{}] * len(batch))
        out, group_of = [], {}
        for i in range(len(batch)):
            p_ids = prompts[i][attn[i, :plen].bool()].tolist()
            n_resp = int(attn[i, plen:].sum())
            uid = f"{batch.non_tensor_batch['uid'][i]}-{i}"
            group_of[uid] = str(batch.non_tensor_batch["uid"][i])
            extra = dict(extras[i] or {})
            per_chunk = batch.non_tensor_batch.get("pi_prefix_ids_per_chunk")
            if per_chunk is not None and per_chunk[i]:
                extra["pi_prefix_ids_per_chunk"] = per_chunk[i]
            out.append(build_payload(uid, p_ids, responses[i, :n_resp].tolist(), resp_mask[i, :n_resp].tolist(),
                                     float(rewards[i].sum()), extra, self.chop_cfg,
                                     self.chop_native_context, self.chop_rollout_chunk))
        return out, group_of

    # ------------------------------------------------------------------ fit
    def fit(self):
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))
        self._chop_setup()
        self.global_steps = 0
        self._load_checkpoint()
        self._load_chop_state()
        self.checkpoint_manager.update_weights(self.global_steps)
        if self.config.trainer.get("val_before_train", True):
            logger.log(data=self._validate(), step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        self.global_steps += 1
        world = self.actor_rollout_wg.world_size
        n = self.config.actor_rollout_ref.rollout.n
        for _epoch in range(self.global_steps // max(len(self.train_dataloader), 1), self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics, timing = {}, {}
                is_last = self.global_steps >= self.total_training_steps
                batch = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=n, interleave=True)
                with marked_timer("step", timing):
                    with marked_timer("gen", timing):
                        gen_out = self.async_rollout_manager.generate_sequences(gen_batch)
                        self.checkpoint_manager.sleep_replicas()
                        gen_out.meta_info.pop("timing", None)
                    batch = batch.repeat(repeat_times=n, interleave=True).union(gen_out)
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    with marked_timer("reward", timing):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch = batch.union(self._compute_reward_colocate(batch))
                        reward_tensor, _ = extract_reward(batch)
                    payloads, group_of = self._payloads(batch, reward_tensor)
                    with marked_timer("chop", timing):
                        metrics.update(chop_step(self.actor_rollout_wg, self.chop_router, self.chop_guard, payloads,
                                                 self.chop_cfg, self.global_steps, world, self.chop_need_pi, group_of))
                    if self.config.trainer.save_freq > 0 and (is_last or
                                                              self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing):
                            self._save_checkpoint()
                    with marked_timer("update_weights", timing):
                        self.checkpoint_manager.update_weights(self.global_steps)
                if self.config.trainer.test_freq > 0 and (is_last or
                                                          self.global_steps % self.config.trainer.test_freq == 0):
                    metrics.update(self._validate())
                metrics.update({f"timing_s/{k}": v for k, v in timing.items()})
                metrics["training/global_step"] = self.global_steps
                logger.log(data=metrics, step=self.global_steps)
                if is_last:
                    return
                self.global_steps += 1
