import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from agent import Policy, ValueFunction


class ExperienceBuffer:

    def __init__(
            self,
            observation_shape: tuple[int, ...],
            action_shape: tuple[int, ...],
            actor_number: int,
            episode_number: int,
            episode_length: int,
            gamma: float,
            lambda_: float,
            device: torch.device
            ) -> None:
        self._observation_shape = observation_shape
        self._action_shape = action_shape

        self._actor_number = actor_number

        buffer_shape = (actor_number*episode_number, episode_length)

        self._observation_collections = torch.empty(
            buffer_shape + observation_shape,
            device=device
        )
        self._action_collections = torch.empty(
            buffer_shape + action_shape,
            device=device
        )
        self._reward_collections = torch.empty(
            buffer_shape + (1,),
            device=device
        )
        self._action_logprob_collections = torch.empty(
            buffer_shape + (1,),
            device=device
        )
        self._reward_to_go_collections = torch.empty(
            buffer_shape + (1,),
            device=device
        )
        self._advantage_collections = torch.empty(
            buffer_shape + (1,),
            device=device
        )

        self._gamma = gamma
        self._lambda_ = lambda_

        i, j = torch.triu_indices(
            episode_length, episode_length,
            device=device
        )

        self._reward_discount_matrix = torch.zeros(
            size=(episode_length, episode_length),
            device=device
        )
        self._reward_discount_matrix[i, j] = gamma**(j - i)

        self._td_residual_discount_matrix = torch.zeros(
            size=(episode_length, episode_length),
            device=device
        )
        self._td_residual_discount_matrix[i, j] = (gamma*lambda_)**(j - i)

    def insert(
            self,
            episode_index: int,
            time_step: int,
            observations: Tensor,
            actions: Tensor,
            rewards: Tensor
            ) -> None:
        episode_slice = slice(
            self._actor_number*episode_index,
            self._actor_number*(episode_index + 1)
        )

        self._observation_collections[episode_slice, time_step] = observations
        self._action_collections[episode_slice, time_step] = actions
        self._reward_collections[episode_slice, time_step] = rewards

    def backpropagate(
            self,
            policy: Policy,
            value_function: ValueFunction
            ) -> None:
        with torch.no_grad():
            self._action_logprob_collections = \
                policy(self._observation_collections) \
                    .log_prob(self._action_collections)

            value_collections = value_function(self._observation_collections)

        self._reward_to_go_collections = torch.matmul(
            self._reward_discount_matrix, self._reward_collections
        )

        td_residual_collections = self._reward_collections - value_collections
        td_residual_collections[:, :-1] += self._gamma*value_collections[:, 1:]

        self._advantage_collections = torch.matmul(
            self._td_residual_discount_matrix, td_residual_collections
        )

    def to_dataset(self) -> TensorDataset:
        observations = self._observation_collections \
            .flatten(start_dim=0, end_dim=1)
        actions = self._action_collections \
            .flatten(start_dim=0, end_dim=1)
        action_logprobs = self._action_logprob_collections \
            .flatten(start_dim=0, end_dim=1)
        advantages = self._advantage_collections \
            .flatten(start_dim=0, end_dim=1)
        rewards_to_go = self._reward_to_go_collections \
            .flatten(start_dim=0, end_dim=1)

        dataset = TensorDataset(
            observations,
            actions,
            action_logprobs,
            advantages,
            rewards_to_go
        )

        return dataset