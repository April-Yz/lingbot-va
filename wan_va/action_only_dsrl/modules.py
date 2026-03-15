import math
from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightImageEncoder64(nn.Module):
    """Compact image encoder aligned with RLinf Pi0 DSRL's 64x64 path."""

    def __init__(self, num_images=3, latent_dim=64, image_size=64):
        super().__init__()
        final_h = image_size // 2
        final_w = image_size // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(num_images * 3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * final_h * final_w, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images):
        batch, num_images, channels, height, width = images.shape
        x = images.reshape(batch, num_images * channels, height, width)
        x = x.to(dtype=self.encoder[0].weight.dtype)
        return self.bottleneck(self.encoder(x))


class CompactStateEncoder(nn.Module):
    def __init__(self, state_dim=16, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        if state.dim() > 2:
            state = state.reshape(state.shape[0], -1)
        return self.encoder(state.to(dtype=self.encoder[0].weight.dtype))


class CompactFutureEncoder(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

    def forward(self, future_summary):
        if future_summary.dim() > 2:
            future_summary = future_summary.reshape(future_summary.shape[0], -1)
        return self.encoder(
            future_summary.to(dtype=self.encoder[0].weight.dtype)
        )


class CompactQHead(nn.Module):
    def __init__(
        self,
        state_dim=64,
        image_dim=64,
        future_dim=64,
        action_dim=32,
        hidden_dims=(128, 128, 128),
    ):
        super().__init__()
        input_dim = state_dim + image_dim + future_dim + action_dim
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.extend(
                [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU()]
            )
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for i, module in enumerate(self.net):
            if not isinstance(module, nn.Linear):
                continue
            if i == len(self.net) - 1:
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            else:
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state_features, image_features, future_features, actions):
        x = torch.cat(
            [state_features, image_features, future_features, actions],
            dim=-1,
        )
        return self.net(x.to(dtype=self.net[0].weight.dtype))


class CompactMultiQHead(nn.Module):
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
        self.q_heads = nn.ModuleList(
            [
                CompactQHead(
                    state_dim=state_dim,
                    image_dim=image_dim,
                    future_dim=future_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                )
                for _ in range(num_q_heads)
            ]
        )

    def forward(self, state_features, image_features, future_features, actions):
        return torch.cat(
            [
                q_head(state_features, image_features, future_features, actions)
                for q_head in self.q_heads
            ],
            dim=-1,
        )


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=(128, 128, 128),
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.extend(
                [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU()]
            )
            in_dim = out_dim
        self.shared_net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(in_dim, output_dim)
        self.log_std_layer = nn.Linear(in_dim, output_dim)
        nn.init.xavier_uniform_(self.mean_layer.weight, gain=0.01)
        nn.init.zeros_(self.mean_layer.bias)
        nn.init.xavier_uniform_(self.log_std_layer.weight, gain=0.01)
        nn.init.zeros_(self.log_std_layer.bias)

    def forward(self, features):
        hidden = self.shared_net(features)
        mean = self.mean_layer(hidden)
        log_std = torch.clamp(self.log_std_layer(hidden), -20, 2)
        return mean, log_std

    def sample(self, features, deterministic=False):
        mean, log_std = self.forward(features)
        std = log_std.exp()
        if deterministic:
            squashed = torch.tanh(mean)
            log_prob = torch.zeros(features.shape[0], device=features.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            sample = dist.rsample()
            squashed = torch.tanh(sample)
            log_prob = dist.log_prob(sample) - torch.log(
                1 - squashed.pow(2) + 1e-7
            )
            log_prob = log_prob.sum(dim=-1)
        return squashed, log_prob


class AlphaTemperature(nn.Module):
    def __init__(self, initial_alpha=0.01):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(float(initial_alpha))))

    @property
    def alpha(self):
        return self.log_alpha.exp()


@dataclass
class ReplayBatch:
    images: torch.Tensor
    states: torch.Tensor
    future_summary: torch.Tensor
    steer_noise: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    next_images: torch.Tensor
    next_states: torch.Tensor
    next_future_summary: torch.Tensor


class SimpleReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self._buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self._buffer)

    def add(self, transition):
        self._buffer.append(transition)

    def sample(self, batch_size, device):
        indices = np.random.randint(0, len(self._buffer), size=batch_size)
        batch = [self._buffer[idx] for idx in indices]

        def stack(key, dtype=torch.float32):
            return torch.stack(
                [torch.as_tensor(item[key], dtype=dtype) for item in batch],
                dim=0,
            ).to(device)

        return ReplayBatch(
            images=stack("images"),
            states=stack("states"),
            future_summary=stack("future_summary"),
            steer_noise=stack("steer_noise"),
            reward=stack("reward"),
            done=stack("done"),
            next_images=stack("next_images"),
            next_states=stack("next_states"),
            next_future_summary=stack("next_future_summary"),
        )


def soft_update(source_modules, target_modules, tau):
    for source_module, target_module in zip(source_modules, target_modules):
        for source_param, target_param in zip(
            source_module.parameters(),
            target_module.parameters(),
        ):
            target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def clone_module(module):
    clone = deepcopy(module)
    clone.requires_grad_(False)
    clone.eval()
    return clone
