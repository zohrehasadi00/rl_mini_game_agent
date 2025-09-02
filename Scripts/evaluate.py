"""
Evaluate a trained DQN checkpoint on CartPole.
"""

import argparse
from pathlib import Path
import statistics as stats
import gymnasium as gym
import torch

from rl_agent.dqn import QNetwork
from rl_agent.utils import get_device, set_seed


def evaluate(ckpt_path: str, episodes: int = 10, record_dir: str | None = None, seed: int = 123):
    """Evaluate a saved Q-network checkpoint.

    Args:
        ckpt_path (str): Path to model state_dict.
        episodes (int): Number of evaluation episodes.
        record_dir (str | None): If set, directory to save episode videos.
        seed (int): Random seed for env reset.

    Returns:
        tuple[float, float]: Mean and standard deviation of returns.
    """
    device = get_device()

    # Build env (optionally with recorder)
    if record_dir:
        Path(record_dir).mkdir(parents=True, exist_ok=True)
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder=record_dir, episode_trigger=lambda e: True)
    else:
        env = gym.make("CartPole-v1")

    # Infer dims from a dummy reset
    obs, _ = env.reset(seed=seed)
    state_dim = obs.shape[0]
    action_dim = env.action_space.n

    # Load net
    q = QNetwork(state_dim, action_dim).to(device)
    q.load_state_dict(torch.load(ckpt_path, map_location=device))
    q.eval()

    def act_greedy(state):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            return int(q(s).argmax(dim=1).item())

    returns = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            a = act_greedy(s)
            s, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        returns.append(total)
        print(f"Episode {ep+1}/{episodes} return: {total:.1f}")

    env.close()
    mean_r = float(stats.mean(returns))
    std_r = float(stats.pstdev(returns)) if len(returns) > 1 else 0.0
    print(f"Mean return: {mean_r:.1f} ± {std_r:.1f} over {episodes} episodes")
    return mean_r, std_r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/best_cartpole_dqn.pt", help="Path to checkpoint")
    ap.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    ap.add_argument("--record_dir", type=str, default=None, help="Directory to save videos")
    ap.add_argument("--seed", type=int, default=123, help="Seed for evaluation environment")
    args = ap.parse_args()

    evaluate(args.ckpt, args.episodes, args.record_dir, args.seed)
