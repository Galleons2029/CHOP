"""Entry point: ``python -m chop_verl.main --config-name chop_math_60k [hydra overrides]``.

Mirrors verl's ``main_ppo_v0`` task runner with two substitutions: the actor/rollout worker is
:class:`chop_verl.worker.ChopActorRolloutRefWorker` and the trainer is
:class:`chop_verl.trainer.ChopRayTrainer`.
"""

from __future__ import annotations

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.config import validate_config


@ray.remote
class ChopTaskRunner(BaseTaskRunner):
    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        from chop_verl.worker import ChopActorRolloutRefWorker

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(ChopActorRolloutRefWorker)
        self.mapping[Role.ActorRollout] = "global_pool"
        return ChopActorRolloutRefWorker, RayWorkerGroup

    def run(self, config):
        from verl.utils.config import omega_conf_to_dataclass
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.workers.config import HFModelConfig

        from chop_verl.trainer import ChopRayTrainer

        print(f"ChopTaskRunner on {socket.gethostname()} (pid {os.getpid()})")
        OmegaConf.resolve(config)
        _, worker_group_cls = self.add_actor_rollout_worker(config)
        validate_config(config=config, use_reference_policy=need_reference_policy(config),
                        use_critic=need_critic(config))
        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer, processor = model_config.tokenizer, model_config.processor
        resource_pool_manager = self.init_resource_pool_mgr(config)
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True,
                                          max_samples=config.data.get("train_max_samples", -1))
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False,
                                        max_samples=config.data.get("val_max_samples", -1))
        trainer = ChopRayTrainer(config=config, tokenizer=tokenizer, processor=processor,
                                 role_worker_mapping=self.role_worker_mapping,
                                 resource_pool_manager=resource_pool_manager, ray_worker_group_cls=worker_group_cls,
                                 train_dataset=train_dataset, val_dataset=val_dataset, collate_fn=collate_fn,
                                 train_sampler=create_rl_sampler(config.data, train_dataset))
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="chop_math", version_base=None)
def main(config):
    run_ppo(config, task_runner_class=ChopTaskRunner)


if __name__ == "__main__":
    main()
