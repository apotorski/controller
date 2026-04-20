from typing import Optional

from torch import Tensor, nn
from torch.distributions.normal import Normal

MAX_LOG_STD_DEV = 2.0
MIN_LOG_STD_DEV = -20.0


class Policy(nn.Module):

    def __init__(
            self,
            observation_shape: tuple[int, ...],
            action_shape: tuple[int, ...]
            ) -> None:
        super().__init__()

        self.backbone = nn.LSTM(observation_shape[0], 64, batch_first=True)

        self.action_mean_head = nn.Linear(64, action_shape[0])
        self.log_action_std_dev_head = nn.Linear(64, action_shape[0])

    def forward(
            self,
            observation: Tensor,
            memory: Optional[tuple[Tensor, Tensor]] = None
            ) -> tuple[Normal, tuple[Tensor, Tensor]]:
        latent_state, memory = self.backbone(observation, memory)

        action_mean = self.action_mean_head(latent_state)

        log_action_std_dev = self.log_action_std_dev_head(latent_state)
        clamped_log_action_std_dev = log_action_std_dev \
            .clamp(MIN_LOG_STD_DEV, MAX_LOG_STD_DEV)
        action_std_dev = clamped_log_action_std_dev.exp()

        action_distribution = Normal(action_mean, action_std_dev)

        return action_distribution, memory


class ValueFunction(nn.Module):

    def __init__(
            self,
            observation_shape: tuple[int, ...]
            ) -> None:
        super().__init__()

        self.backbone = nn.LSTM(observation_shape[0], 64, batch_first=True)

        self.value_head = nn.Linear(64, 1)

    def forward(
            self,
            observation: Tensor,
            memory: Optional[tuple[Tensor, Tensor]] = None
            ) -> tuple[Normal, tuple[Tensor, Tensor]]:
        latent_state, memory = self.backbone(observation, memory)

        value = self.value_head(latent_state)

        return value, memory