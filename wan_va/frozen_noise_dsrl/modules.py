import torch
import torch.nn as nn

from wan_va.action_only_dsrl.modules import (
    CompactFutureEncoder,
    CompactMultiQHead,
    CompactStateEncoder,
    GaussianPolicy,
    LightweightImageEncoder64,
)


class FutureLatentAdapter(nn.Module):
    """Compress frozen LingBot future latents into a stable policy condition."""

    def __init__(self, input_dim=48, hidden_dim=64, output_dim=64):
        super().__init__()
        self.summary_encoder = CompactFutureEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )

    def forward(self, future_summary):
        return self.projection(self.summary_encoder(future_summary))


class FutureConditionedNoisePolicy(nn.Module):
    """Lightweight actor that steers LingBot action noise."""

    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128, 128)):
        super().__init__()
        self.policy = GaussianPolicy(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
        )

    def sample(self, features, deterministic=False):
        return self.policy.sample(features, deterministic=deterministic)

    def forward(self, features):
        return self.policy(features)


class FutureConditionedCritic(nn.Module):
    """Multi-head critic for frozen-noise action steering."""

    def __init__(
        self,
        state_dim=64,
        image_dim=64,
        future_dim=64,
        action_dim=32,
        hidden_dims=(128, 128, 128),
        num_q_heads=10,
    ):
        super().__init__()
        self.q_head = CompactMultiQHead(
            state_dim=state_dim,
            image_dim=image_dim,
            future_dim=future_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            num_q_heads=num_q_heads,
        )

    def forward(self, state_features, image_features, future_features, actions):
        return self.q_head(
            state_features,
            image_features,
            future_features,
            actions,
        )


__all__ = [
    "CompactStateEncoder",
    "FutureConditionedCritic",
    "FutureConditionedNoisePolicy",
    "FutureLatentAdapter",
    "LightweightImageEncoder64",
]
