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

        self.backbone = nn.Sequential(
            nn.Linear(observation_shape[0], 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU()
        )

        self.action_mean_head = nn.Linear(256, action_shape[0])
        self.log_action_std_dev_head = nn.Linear(256, action_shape[0])

    def forward(self, observation: Tensor) -> Normal:
        latent_observation = self.backbone(observation)

        action_mean = self.action_mean_head(latent_observation)

        log_action_std_dev = self.log_action_std_dev_head(latent_observation)
        clamped_log_action_std_dev = log_action_std_dev \
            .clamp(MIN_LOG_STD_DEV, MAX_LOG_STD_DEV)
        action_std_dev = clamped_log_action_std_dev.exp()

        action_distribution = Normal(action_mean, action_std_dev)

        return action_distribution


class ValueFunction(nn.Module):

    def __init__(
            self,
            observation_shape: tuple[int, ...]
            ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(observation_shape[0], 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, observation: Tensor) -> Tensor:
        value = self.model(observation)

        return value