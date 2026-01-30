#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader

from agent import Policy, ValueFunction
from environment import Environment
from experience_buffer import ExperienceBuffer


def collect_episodes(
        policy: Policy,
        value_function: ValueFunction,
        experience_buffer: ExperienceBuffer,
        environment: Environment,
        episode_number: int,
        episode_length: int
        ) -> None:
    for episode_index in range(episode_number):
        observations = environment.reset()

        for time_step in range(episode_length):
            with torch.no_grad():
                actions = policy(observations).sample()

            next_observations, rewards = environment(actions)

            experience_buffer.insert(
                episode_index,
                time_step,
                observations,
                actions,
                rewards
            )

            observations = next_observations

    experience_buffer.backpropagate(policy, value_function)


def update_agent(
        policy: Policy,
        value_function: ValueFunction,
        experience_buffer: ExperienceBuffer,
        batch_size: int,
        epoch_number: int,
        policy_optimizer: optim.AdamW,
        beta: float,
        value_function_optimizer: optim.AdamW
        ) -> None:
    policy.train()
    value_function.train()

    dataset = experience_buffer.to_dataset()

    data_loader = DataLoader(dataset, batch_size)

    for _ in range(epoch_number):
        for observations, actions, previous_action_logprobs, \
                advantages, rewards_to_go in data_loader:
            policy_optimizer.zero_grad()

            action_logprobs = policy(observations).log_prob(actions)

            action_logprob_ratios = action_logprobs - previous_action_logprobs
            action_prob_ratios = action_logprob_ratios.exp()

            loss = torch.mean(
                action_prob_ratios*advantages
                - beta*(action_prob_ratios - 1.0 - action_logprob_ratios)
            ).negative()

            loss.backward()
            policy_optimizer.step()

            value_function_optimizer.zero_grad()

            values = value_function(observations)

            loss = (rewards_to_go - values).square().mean()

            loss.backward()
            value_function_optimizer.step()

    policy.eval()
    value_function.eval()


def evaluate_agent(
        policy: Policy,
        environment: Environment,
        episode_length: int
        ) -> float:
    returns = 0.0

    observations = environment.reset()
    for _ in range(episode_length):
        with torch.no_grad():
            actions = policy(observations).mean

        observations, rewards = environment(actions)
        returns += rewards

    average_return = returns.mean().item()

    return average_return


def main(
        actor_number: int,
        episode_number: int,
        episode_length: int,
        gamma: float,
        lambda_: float,
        beta: float,
        policy_learning_rate: float,
        value_function_learning_rate: float,
        batch_size: int,
        epoch_number: int,
        iteration_number: int,
        agent_save_path: Path
        ) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Training device = \'{device}\'')

    environment = Environment(actor_number, device)

    experience_buffer = ExperienceBuffer(
        environment.observation_shape,
        environment.action_shape,
        actor_number,
        episode_number,
        episode_length,
        gamma,
        lambda_,
        device
    )

    policy = Policy(
        environment.observation_shape,
        environment.action_shape
    ).to(device)

    value_function = ValueFunction(
        environment.observation_shape
    ).to(device)

    policy_optimizer = optim.AdamW(
        policy.parameters(),
        policy_learning_rate
    )

    value_function_optimizer = optim.AdamW(
        value_function.parameters(),
        value_function_learning_rate
    )

    policy.eval()
    value_function.eval()

    for iteration in range(iteration_number):
        collect_episodes(
            policy,
            value_function,
            experience_buffer,
            environment,
            episode_number,
            episode_length
        )

        update_agent(
            policy,
            value_function,
            experience_buffer,
            batch_size,
            epoch_number,
            policy_optimizer,
            beta,
            value_function_optimizer
        )

        average_return = evaluate_agent(
            policy,
            environment,
            episode_length
        )

        logging.info(
            f'Training agent '
            f'- iteration: {iteration + 1:,}/{iteration_number:,} '
            f'- average return = {average_return:.3f}'
        )

    torch.save(policy.state_dict(), agent_save_path)
    logging.info(f'Agent is saved to \'{agent_save_path}\'')


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    parser = argparse.ArgumentParser(description='Train the agent.')
    parser.add_argument('--actor_number', type=int)
    parser.add_argument('--episode_number', type=int)
    parser.add_argument('--episode_length', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--lambda_', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--policy_learning_rate', type=float)
    parser.add_argument('--value_function_learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epoch_number', type=int)
    parser.add_argument('--iteration_number', type=int)
    parser.add_argument('--agent_save_path', type=Path)

    args = parser.parse_args()

    main(**vars(args))