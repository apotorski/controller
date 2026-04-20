#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch
from torch import Tensor

from agent import Policy
from environment import Environment


def log_observation(observation: Tensor) -> None:
    x, theta = observation.squeeze()
    logging.info(f'Cart position = {x:.3f}, pole angle = {theta:.3f}')


def main(agent_load_path: Path, episode_length: int) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Evaluation device = \'{device}\'')

    environment = Environment(actor_number=1, device=device)

    policy = Policy(
        environment.observation_shape,
        environment.action_shape
    ).to(device)
    policy.load_state_dict(torch.load(agent_load_path, weights_only=True))
    policy.eval()
    logging.info('Agent loaded from \'{agent_load_path}\'')

    reference_observation = torch.tensor([1.0, 0.0], device=device)

    observation = environment.reset()

    log_observation(observation)

    with torch.inference_mode():
        memory = None
        for _ in range(episode_length):
            observation_error = observation - reference_observation

            unsqueezed_observation_error = observation_error.unsqueeze(1)

            unsqueezed_action_distribution, memory = \
                policy(unsqueezed_observation_error, memory)
            unsqueezed_action = unsqueezed_action_distribution.mean

            action = unsqueezed_action.squeeze(1)

            observation = environment(action)[0]

    log_observation(observation)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    parser = argparse.ArgumentParser(description='Evaluate the agent.')
    parser.add_argument('--agent_load_path', type=Path)
    parser.add_argument('--episode_length', type=int)

    args = parser.parse_args()

    main(**vars(args))