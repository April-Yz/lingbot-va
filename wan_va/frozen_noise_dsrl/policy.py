import logging
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wan_va.action_only_dsrl.modules import (
    AlphaTemperature,
    SimpleReplayBuffer,
    clone_module,
    soft_update,
)
from wan_va.configs import VA_CONFIGS
from wan_va.wan_va_server import VA_Server

from .modules import (
    CompactStateEncoder,
    FutureConditionedCritic,
    FutureConditionedNoisePolicy,
    FutureLatentAdapter,
    LightweightImageEncoder64,
)


LOGGER = logging.getLogger("lingbot_frozen_noise_dsrl_v2")


def _count_params(parameters):
    return sum(param.numel() for param in parameters)


class LingBotFrozenNoiseV2Policy(nn.Module):
    """Frozen LingBot backbone with future-latent-conditioned action-noise policy."""

    camera_keys = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.policy_cfg = config["policy"]
        self.algorithm_cfg = config["algorithm"]

        self.freeze_future_video_module = bool(
            self.policy_cfg.get("freeze_future_video_module", True)
        )
        self.freeze_inverse_or_action_flow = bool(
            self.policy_cfg.get("freeze_inverse_or_action_flow", True)
        )
        self.train_noise_policy_only = bool(
            self.policy_cfg.get("train_noise_policy_only", True)
        )
        self.future_latent_as_condition = bool(
            self.policy_cfg.get("future_latent_as_condition", True)
        )
        self.use_rlinf_or_v1_style_training = bool(
            self.policy_cfg.get("use_rlinf_or_v1_style_training", True)
        )
        self.detach_future_latent = bool(
            self.policy_cfg.get("detach_future_latent", True)
        )
        self.use_value_critic = bool(self.policy_cfg.get("use_value_critic", True))
        self.use_dsrl = bool(self.policy_cfg.get("use_dsrl", True))
        self.dsrl_mode = self.policy_cfg.get("dsrl_mode", "frozen_noise_v2")
        self.injection_mode = self.policy_cfg.get(
            "dsrl_noise_injection_mode",
            "initial_noise",
        )

        lingbot_cfg_name = self.policy_cfg.get("lingbot_config_name", "robotwin")
        job_config = deepcopy(VA_CONFIGS[lingbot_cfg_name])
        job_config.local_rank = int(self.policy_cfg.get("local_rank", 0))
        job_config.rank = int(self.policy_cfg.get("rank", 0))
        job_config.world_size = int(self.policy_cfg.get("world_size", 1))
        if self.policy_cfg.get("lingbot_model_path"):
            job_config.wan22_pretrained_model_name_or_path = self.policy_cfg[
                "lingbot_model_path"
            ]
        if self.policy_cfg.get("save_root"):
            job_config.save_root = self.policy_cfg["save_root"]

        self.server = VA_Server(job_config)
        self.device = self.server.device
        self.action_noise_shape = self.server._action_noise_shape()
        self.action_noise_dim = int(
            self.policy_cfg.get("dsrl_action_noise_dim")
            or math.prod(self.action_noise_shape[1:])
        )
        expected_noise_dim = math.prod(self.action_noise_shape[1:])
        if self.action_noise_dim != expected_noise_dim:
            raise ValueError(
                f"dsrl_action_noise_dim={self.action_noise_dim} must equal "
                f"{expected_noise_dim} for initial_noise injection."
            )

        self.raw_state_dim = int(self.policy_cfg.get("dsrl_state_dim", 30))
        self.raw_future_dim = int(
            self.policy_cfg.get("dsrl_future_summary_input_dim", 48)
        )
        self.hidden_dims = tuple(
            self.policy_cfg.get("dsrl_hidden_dims", [128, 128, 128])
        )
        self.image_latent_dim = int(self.policy_cfg.get("dsrl_image_latent_dim", 64))
        self.state_latent_dim = int(self.policy_cfg.get("dsrl_state_latent_dim", 64))
        self.future_latent_dim = int(
            self.policy_cfg.get("dsrl_future_latent_dim", 64)
        )
        self.num_q_heads = int(self.policy_cfg.get("dsrl_num_q_heads", 10))
        self.actor_input_dim = (
            self.image_latent_dim + self.state_latent_dim + self.future_latent_dim
        )

        self._freeze_lingbot()

        self.actor_image_encoder = LightweightImageEncoder64(
            num_images=len(self.camera_keys),
            latent_dim=self.image_latent_dim,
        ).to(self.device)
        self.actor_state_encoder = CompactStateEncoder(
            state_dim=self.raw_state_dim,
            hidden_dim=self.state_latent_dim,
        ).to(self.device)
        self.actor_future_encoder = FutureLatentAdapter(
            input_dim=self.raw_future_dim,
            hidden_dim=self.future_latent_dim,
            output_dim=self.future_latent_dim,
        ).to(self.device)
        self.noise_policy = FutureConditionedNoisePolicy(
            input_dim=self.actor_input_dim,
            output_dim=self.action_noise_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        self.critic_image_encoder = LightweightImageEncoder64(
            num_images=len(self.camera_keys),
            latent_dim=self.image_latent_dim,
        ).to(self.device)
        self.critic_state_encoder = CompactStateEncoder(
            state_dim=self.raw_state_dim,
            hidden_dim=self.state_latent_dim,
        ).to(self.device)
        self.critic_future_encoder = FutureLatentAdapter(
            input_dim=self.raw_future_dim,
            hidden_dim=self.future_latent_dim,
            output_dim=self.future_latent_dim,
        ).to(self.device)
        self.q_head = FutureConditionedCritic(
            state_dim=self.state_latent_dim,
            image_dim=self.image_latent_dim,
            future_dim=self.future_latent_dim,
            action_dim=self.action_noise_dim,
            hidden_dims=self.hidden_dims,
            num_q_heads=self.num_q_heads,
        ).to(self.device)

    def _freeze_lingbot(self):
        # The current LingBot transformer serves both future-latent generation
        # and action decoding, so V2 freezes the full loaded backbone.
        if self.freeze_future_video_module or self.freeze_inverse_or_action_flow:
            self.server.transformer.requires_grad_(False)
            self.server.transformer.eval()
        self.server.vae.requires_grad_(False)
        self.server.vae.eval()
        self.server.text_encoder.requires_grad_(False)
        self.server.text_encoder.eval()

    def freeze_report(self):
        lingbot_params = list(self.server.transformer.parameters())
        lingbot_params += list(self.server.vae.parameters())
        lingbot_params += list(self.server.text_encoder.parameters())
        return {
            "frozen_lingbot_params": _count_params(lingbot_params),
            "trainable_v2_params": _count_params(self.trainable_parameters()),
            "freeze_future_video_module": self.freeze_future_video_module,
            "freeze_inverse_or_action_flow": self.freeze_inverse_or_action_flow,
            "train_noise_policy_only": self.train_noise_policy_only,
            "future_latent_as_condition": self.future_latent_as_condition,
            "use_rlinf_or_v1_style_training": self.use_rlinf_or_v1_style_training,
            "detach_future_latent": self.detach_future_latent,
            "use_value_critic": self.use_value_critic,
            "use_dsrl": self.use_dsrl,
            "dsrl_mode": self.dsrl_mode,
            "injection_mode": self.injection_mode,
        }

    def trainable_parameters(self):
        modules = [
            self.actor_image_encoder,
            self.actor_state_encoder,
            self.actor_future_encoder,
            self.noise_policy,
            self.critic_image_encoder,
            self.critic_state_encoder,
            self.critic_future_encoder,
            self.q_head,
        ]
        for module in modules:
            yield from module.parameters()

    def optimizer_groups(self):
        actor_modules = [
            self.actor_image_encoder,
            self.actor_state_encoder,
            self.actor_future_encoder,
            self.noise_policy,
        ]
        critic_modules = [
            self.critic_image_encoder,
            self.critic_state_encoder,
            self.critic_future_encoder,
            self.q_head,
        ]
        actor_params = []
        critic_params = []
        for module in actor_modules:
            actor_params.extend(list(module.parameters()))
        for module in critic_modules:
            critic_params.extend(list(module.parameters()))
        return actor_params, critic_params

    def optimizer_report(self):
        actor_params, critic_params = self.optimizer_groups()
        return {
            "actor_param_count": _count_params(actor_params),
            "critic_param_count": _count_params(critic_params),
            "optimizes_lingbot": False,
        }

    def reset(self, prompt):
        self.server._reset(prompt=prompt)

    def compute_kv_cache(self, key_frame_list, action_chunk):
        self.server._compute_kv_cache({"obs": key_frame_list, "state": action_chunk})

    def _prepare_images(self, formatted_obs):
        image_tensors = []
        for key in self.camera_keys:
            image = np.asarray(formatted_obs[key], dtype=np.float32) / 255.0
            tensor = torch.from_numpy(image).permute(2, 0, 1)[None]
            tensor = F.interpolate(
                tensor,
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )
            image_tensors.append(tensor[0])
        return torch.stack(image_tensors, dim=0)

    def _prepare_state(self, formatted_obs):
        raw_state = np.asarray(
            formatted_obs.get("observation.state", np.zeros(self.raw_state_dim)),
            dtype=np.float32,
        ).reshape(-1)
        state = np.zeros(self.raw_state_dim, dtype=np.float32)
        copy_dim = min(self.raw_state_dim, raw_state.shape[0])
        state[:copy_dim] = raw_state[:copy_dim]
        return torch.from_numpy(state)

    def _summarize_future_latents(self, future_latents):
        summary = future_latents.float().mean(dim=(2, 3, 4))
        if summary.shape[1] < self.raw_future_dim:
            summary = F.pad(summary, (0, self.raw_future_dim - summary.shape[1]))
        elif summary.shape[1] > self.raw_future_dim:
            summary = summary[:, : self.raw_future_dim]
        return summary

    def build_step_batch(self, formatted_obs, future_latents):
        future_condition = (
            future_latents.detach() if self.detach_future_latent else future_latents
        )
        future_summary = self._summarize_future_latents(future_condition)
        return {
            "images": self._prepare_images(formatted_obs)[None].to(self.device),
            "states": self._prepare_state(formatted_obs)[None].to(self.device),
            "future_summary": future_summary.to(self.device),
        }

    def _actor_features(self, batch):
        return torch.cat(
            [
                self.actor_image_encoder(batch["images"]),
                self.actor_state_encoder(batch["states"]),
                self.actor_future_encoder(batch["future_summary"]),
            ],
            dim=-1,
        )

    def _critic_features(self, batch):
        return (
            self.critic_state_encoder(batch["states"]),
            self.critic_image_encoder(batch["images"]),
            self.critic_future_encoder(batch["future_summary"]),
        )

    def _target_critic_features(self, batch, target_modules):
        target_state, target_image, target_future = target_modules
        return (
            target_state(batch["states"]),
            target_image(batch["images"]),
            target_future(batch["future_summary"]),
        )

    def _reshape_noise(self, noise_flat):
        return noise_flat.reshape(noise_flat.shape[0], *self.action_noise_shape[1:])

    def zero_step_batch(self, batch_size=1):
        return {
            "images": torch.zeros(
                batch_size, len(self.camera_keys), 3, 64, 64, device=self.device
            ),
            "states": torch.zeros(batch_size, self.raw_state_dim, device=self.device),
            "future_summary": torch.zeros(
                batch_size,
                self.raw_future_dim,
                device=self.device,
            ),
        }

    def act(self, formatted_obs, deterministic=False, use_dsrl=None):
        use_dsrl = self.use_dsrl if use_dsrl is None else use_dsrl
        model_obs = {"obs": formatted_obs}
        future_latents = self.server.sample_future_latents(
            model_obs,
            frame_st_id=self.server.frame_st_id,
        )
        step_batch = self.build_step_batch(formatted_obs, future_latents)

        steer_noise = None
        log_prob = torch.zeros(1, device=self.device)
        if use_dsrl:
            actor_features = self._actor_features(step_batch)
            steer_noise_flat, log_prob = self.noise_policy.sample(
                actor_features,
                deterministic=deterministic,
            )
            steer_noise = self._reshape_noise(steer_noise_flat)

        action_tensor = self.server.sample_actions(
            frame_st_id=self.server.frame_st_id,
            initial_noise=steer_noise,
        )
        action_chunk = self.server.postprocess_action(action_tensor)
        diagnostics = {
            "use_dsrl": bool(use_dsrl),
            "dsrl_mode": self.dsrl_mode,
            "injection_mode": self.injection_mode if use_dsrl else "disabled",
            "future_latent_as_condition": self.future_latent_as_condition,
            "future_latent_detached": self.detach_future_latent,
            "steer_noise_shape": tuple(steer_noise.shape) if steer_noise is not None else None,
            "action_output_shape": tuple(action_chunk.shape),
            "log_prob": float(log_prob.detach().cpu()[0]),
        }
        return {
            "action_chunk": action_chunk,
            "step_batch": {k: v.detach() for k, v in step_batch.items()},
            "steer_noise_flat": (
                steer_noise.reshape(steer_noise.shape[0], -1).detach()
                if steer_noise is not None
                else torch.zeros(1, self.action_noise_dim, device=self.device)
            ),
            "log_prob": log_prob.detach(),
            "diagnostics": diagnostics,
        }


class LingBotFrozenNoiseV2Trainer:
    def __init__(self, config):
        self.config = config
        self.policy = LingBotFrozenNoiseV2Policy(config)
        self.device = self.policy.device
        self.use_dsrl = self.policy.use_dsrl
        self.gamma = float(config["algorithm"].get("gamma", 0.99))
        self.tau = float(config["algorithm"].get("tau", 0.005))
        self.batch_size = int(config["algorithm"].get("batch_size", 4))
        self.warmup_steps = int(config["algorithm"].get("warmup_steps", 4))
        self.updates_per_step = int(config["algorithm"].get("updates_per_step", 1))
        self.actor_agg_q = config["algorithm"].get("actor_agg_q", "mean")
        self.target_entropy = float(
            config["algorithm"].get(
                "target_entropy",
                -0.25 * self.policy.action_noise_dim,
            )
        )
        self.global_step = 0

        self.replay_buffer = SimpleReplayBuffer(
            capacity=int(config["algorithm"].get("replay_size", 4096))
        )

        actor_params, critic_params = self.policy.optimizer_groups()
        self.actor_optimizer = torch.optim.Adam(
            actor_params,
            lr=float(config["algorithm"].get("actor_lr", 1e-4)),
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=float(config["algorithm"].get("critic_lr", 1e-4)),
        )
        self.alpha_temperature = AlphaTemperature(
            initial_alpha=float(config["algorithm"].get("initial_alpha", 0.01))
        ).to(self.device)
        self.alpha_optimizer = torch.optim.Adam(
            self.alpha_temperature.parameters(),
            lr=float(config["algorithm"].get("alpha_lr", 1e-4)),
        )

        self.target_critic_image_encoder = clone_module(self.policy.critic_image_encoder)
        self.target_critic_state_encoder = clone_module(self.policy.critic_state_encoder)
        self.target_critic_future_encoder = clone_module(self.policy.critic_future_encoder)
        self.target_q_head = clone_module(self.policy.q_head)

    def startup_report(self):
        report = {}
        report.update(self.policy.freeze_report())
        report.update(self.policy.optimizer_report())
        report["algorithm"] = "embodied_sac"
        report["target_entropy"] = self.target_entropy
        return report

    def _aggregate_q(self, q_values):
        if self.actor_agg_q == "min":
            return q_values.min(dim=-1).values
        return q_values.mean(dim=-1)

    def add_transition(self, step_batch, steer_noise_flat, reward, done, next_step_batch):
        transition = {
            "images": step_batch["images"][0].detach().cpu(),
            "states": step_batch["states"][0].detach().cpu(),
            "future_summary": step_batch["future_summary"][0].detach().cpu(),
            "steer_noise": steer_noise_flat[0].detach().cpu(),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "done": torch.tensor([float(done)], dtype=torch.float32),
            "next_images": next_step_batch["images"][0].detach().cpu(),
            "next_states": next_step_batch["states"][0].detach().cpu(),
            "next_future_summary": next_step_batch["future_summary"][0].detach().cpu(),
        }
        self.replay_buffer.add(transition)

    def update(self):
        if not self.use_dsrl or len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return None

        metrics = None
        for _ in range(self.updates_per_step):
            batch = self.replay_buffer.sample(self.batch_size, self.device)
            current_batch = {
                "images": batch.images,
                "states": batch.states,
                "future_summary": batch.future_summary,
            }
            next_batch = {
                "images": batch.next_images,
                "states": batch.next_states,
                "future_summary": batch.next_future_summary,
            }

            with torch.no_grad():
                next_actor_features = self.policy._actor_features(next_batch)
                next_noise_flat, next_log_prob = self.policy.noise_policy.sample(
                    next_actor_features,
                    deterministic=False,
                )
                next_state_features = self.policy._target_critic_features(
                    next_batch,
                    (
                        self.target_critic_state_encoder,
                        self.target_critic_image_encoder,
                        self.target_critic_future_encoder,
                    ),
                )
                next_q = self.target_q_head(
                    next_state_features[0],
                    next_state_features[1],
                    next_state_features[2],
                    next_noise_flat,
                )
                next_q = self._aggregate_q(next_q)
                target_q = batch.reward.squeeze(-1) + (1.0 - batch.done.squeeze(-1)) * self.gamma * (
                    next_q - self.alpha_temperature.alpha.detach() * next_log_prob
                )

            critic_state_features = self.policy._critic_features(current_batch)
            current_q = self.policy.q_head(
                critic_state_features[0],
                critic_state_features[1],
                critic_state_features[2],
                batch.steer_noise,
            )
            critic_loss = F.mse_loss(
                current_q,
                target_q.unsqueeze(-1).expand_as(current_q),
            )
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_features = self.policy._actor_features(current_batch)
            sampled_noise_flat, log_prob = self.policy.noise_policy.sample(
                actor_features,
                deterministic=False,
            )
            critic_state_features = self.policy._critic_features(current_batch)
            actor_q = self.policy.q_head(
                critic_state_features[0],
                critic_state_features[1],
                critic_state_features[2],
                sampled_noise_flat,
            )
            actor_q = self._aggregate_q(actor_q)
            actor_loss = (
                self.alpha_temperature.alpha.detach() * log_prob - actor_q
            ).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            alpha_loss = -(
                self.alpha_temperature.alpha * (log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

            soft_update(
                [
                    self.policy.critic_image_encoder,
                    self.policy.critic_state_encoder,
                    self.policy.critic_future_encoder,
                    self.policy.q_head,
                ],
                [
                    self.target_critic_image_encoder,
                    self.target_critic_state_encoder,
                    self.target_critic_future_encoder,
                    self.target_q_head,
                ],
                self.tau,
            )

            self.global_step += 1
            metrics = {
                "train/critic_loss": float(critic_loss.detach().cpu()),
                "train/actor_loss": float(actor_loss.detach().cpu()),
                "train/alpha_loss": float(alpha_loss.detach().cpu()),
                "train/alpha": float(self.alpha_temperature.alpha.detach().cpu()),
                "train/replay_size": len(self.replay_buffer),
                "train/global_step": self.global_step,
            }

        return metrics
