# rl_agent/train_dqn.py
"""
Entry point for training a DQN agent on OpenAI Gym environments.

This script sets up:
- Environment
- Replay buffer
- DQN agent
- Training loop with epsilon-greedy action selection and TD updates
"""

import argparse
import gymnasium as gym
import torch

from rl_agent.utils import set_seed, get_device
from rl_agent.replay_buffer import ReplayBuffer
from rl_agent.dqn import DQNAgent, DQNConfig


def train(args):
    """Run training loop for DQN on CartPole.

    Args:
        args (argparse.Namespace): Command-line arguments specifying training setup.
    """
    # Reproducibility
    set_seed(args.seed)

    # Device setup
    device = get_device()

    # Environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Agent and replay buffer
    cfg = DQNConfig()
    agent = DQNAgent(state_dim, action_dim, device, cfg)
    buffer = ReplayBuffer(args.buffer_size, state_dim, device)

    # Initialize state
    state, _ = env.reset(seed=args.seed)
    episode_return = 0.0

    for step in range(1, args.train_steps + 1):
        # --- Collect transition ---
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, float(done))
        episode_return += reward

        # --- Learn (if buffer has enough samples) ---
        if len(buffer) >= cfg.batch_size:
            batch = buffer.sample(cfg.batch_size)
            loss = agent.update(batch)
            agent.maybe_update_target()
        else:
            loss = None

        state = next_state

        # --- Handle episode end ---
        if done:
            print(f"Step {step} | Episode return: {episode_return:.1f}")
            episode_return = 0.0
            state, _ = env.reset()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--buffer_size", type=int, default=50_000, help="Replay buffer capacity")
    parser.add_argument("--train_steps", type=int, default=10_000, help="Number of environment steps")
    args = parser.parse_args()

    train(args)
