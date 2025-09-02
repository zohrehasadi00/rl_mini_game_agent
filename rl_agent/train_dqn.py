"""
Train a DQN agent on CartPole with TensorBoard logging and checkpointing.
"""

import argparse
import os
from pathlib import Path
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from rl_agent.utils import set_seed, get_device
from rl_agent.replay_buffer import ReplayBuffer
from rl_agent.dqn import DQNAgent, DQNConfig


def moving_average(values, k=10):
    """Compute simple moving average over the last k values.

    Args:
        values (list[float]): Sequence of numbers.
        k (int): Window size.

    Returns:
        float: Moving average of the last k elements, or mean of all if len < k.
    """
    if not values:
        return 0.0
    w = min(k, len(values))
    return sum(values[-w:]) / float(w)


def train(args):
    """Run DQN training with logging and checkpoints.

    Args:
        args (argparse.Namespace): CLI arguments.
    """
    # Reproducibility and device
    set_seed(args.seed)
    device = get_device()

    # Output dirs
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # Environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Agent and buffer
    cfg = DQNConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        target_update_every=1000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=20_000,
        gradient_clip=10.0,
    )
    agent = DQNAgent(state_dim, action_dim, device, cfg)
    buffer = ReplayBuffer(args.buffer_size, state_dim, device)

    # Warmup fills the buffer with random actions for stability
    state, _ = env.reset(seed=args.seed)
    for _ in range(args.warmup_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, float(done))
        state = next_state if not done else env.reset()[0]

    # Training loop
    state, _ = env.reset(seed=args.seed + 1)
    episode_return = 0.0
    returns = []
    best_ma10 = float("-inf")

    for step in range(1, args.train_steps + 1):
        # Act and step
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, float(done))
        episode_return += reward

        # Learn
        if len(buffer) >= cfg.batch_size:
            batch = buffer.sample(cfg.batch_size)
            loss = agent.update(batch)
            agent.maybe_update_target()
            writer.add_scalar("train/loss", loss, global_step=step)

            # Log Q value magnitude occasionally
            if step % 500 == 0:
                with torch.no_grad():
                    s_sample = batch[0][:32]  # 32 states
                    q_vals = agent.q(s_sample)
                    writer.add_scalar("train/q_mean", float(q_vals.mean().item()), step)
        else:
            loss = None

        # Log epsilon every step
        writer.add_scalar("train/epsilon", agent.epsilon(), global_step=step)

        # Episode end
        if done:
            returns.append(episode_return)
            ma10 = moving_average(returns, k=10)
            writer.add_scalar("episodic/return", episode_return, global_step=step)
            writer.add_scalar("episodic/return_ma10", ma10, global_step=step)

            # Save best checkpoint by MA10
            if ma10 > best_ma10:
                best_ma10 = ma10
                ckpt_path = os.path.join(args.ckpt_dir, "best_cartpole_dqn.pt")
                torch.save(agent.q.state_dict(), ckpt_path)

            print(
                f"step {step:6d}  return {episode_return:6.1f}  "
                f"ma10 {ma10:6.1f}  eps {agent.epsilon():.3f}  "
                f"loss {0.0 if loss is None else loss:.4f}"
            )
            episode_return = 0.0
            state, _ = env.reset()

        # Info print
        if step % 5000 == 0:
            print(f"[info] step {step} | buffer {len(buffer)} | best_ma10 {best_ma10:.1f}")

        state = next_state

    env.close()
    writer.close()
    print(f"Training complete. Best moving average (last 10): {best_ma10:.1f}")


def make_arg_parser():
    """Create an argument parser for training configuration.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--buffer_size", type=int, default=50_000, help="Replay buffer capacity")
    p.add_argument("--warmup_steps", type=int, default=1_000, help="Random steps before learning")
    p.add_argument("--train_steps", type=int, default=100_000, help="Environment steps for training")
    p.add_argument("--log_dir", type=str, default="runs/cartpole_dqn", help="TensorBoard log directory")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    return p


if __name__ == "__main__":
    parser = make_arg_parser()
    args = parser.parse_args()
    train(args)
