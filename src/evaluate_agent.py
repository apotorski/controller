#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch
from torch import Tensor

from agent import Policy
from environment import Environment


def log_observation(observation: Tensor) -> None:
    x, x_dot, theta, theta_dot = observation.squeeze()
    logging.info(
        f'Cart position = {x:.3f}, '
        f'cart velocity = {x_dot:.3f}, '
        f'pole angle = {theta:.3f}, '
        f'pole angular velocity = {theta_dot:.3f}'
    )


def main(agent_save_path: Path, episode_length: int) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Evaluation device = \'{device}\'')

    environment = Environment(actor_number=1, device=device)

    policy = Policy(
        environment.observation_shape,
        environment.action_shape
    ).to(device)

    policy.load_state_dict(torch.load(agent_save_path, weights_only=True))

    policy.eval()

    reference_observation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    observation = environment.reset()
    log_observation(observation)

    for _ in range(episode_length):
        with torch.inference_mode():
            action = policy(observation - reference_observation).mean

        observation = environment(action)[0]

    log_observation(observation)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    parser = argparse.ArgumentParser(description='Evaluate the agent.')
    parser.add_argument('--agent_save_path', type=Path)
    parser.add_argument('--episode_length', type=int)

    args = parser.parse_args()

    main(**vars(args))