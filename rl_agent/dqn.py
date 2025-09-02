# rl_agent/dqn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Multi-layer perceptron that estimates Q-values Q(s, a).

    This network maps a continuous state vector to a vector of Q-values,
    one per discrete action.

    Args:
        state_dim (int): Dimension of the input state vector.
        action_dim (int): Number of discrete actions.
        hidden (Tuple[int, int], optional): Sizes of the two hidden layers.
            Defaults to (128, 128).

    Shape:
        - Input: (N, state_dim) float32
        - Output: (N, action_dim) float32

    Example:
        >>> net = QNetwork(4, 2)
        >>> x = torch.randn(5, 4)
        >>> q = net(x)
        >>> q.shape
        torch.Size([5, 2])
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of states.

        Args:
            x (torch.Tensor): Batch of states, shape (N, state_dim), dtype float32.

        Returns:
            torch.Tensor: Batch of Q-value vectors, shape (N, action_dim).
        """
        return self.net(x)


@dataclass
class DQNConfig:
    """Configuration hyperparameters for DQNAgent.

    Attributes:
        gamma (float): Discount factor in [0, 1]. Defaults to 0.99.
        lr (float): Learning rate for Adam optimizer. Defaults to 1e-3.
        batch_size (int): Minibatch size used during updates. Defaults to 64.
        target_update_every (int): Frequency (in environment steps) to copy
            online network weights into the target network. Defaults to 1000.
        eps_start (float): Initial epsilon for epsilon-greedy policy. Defaults to 1.0.
        eps_end (float): Final epsilon after decay finishes. Defaults to 0.05.
        eps_decay_steps (int): Number of steps over which epsilon decays linearly.
            Defaults to 20_000.
        gradient_clip (Optional[float]): If set, global norm clip value. Defaults to None.
    """

    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    target_update_every: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    gradient_clip: Optional[float] = None


class DQNAgent:
    """Deep Q-Network agent with a target network and epsilon-greedy policy.

    The agent maintains two networks:
    - Online network (trained every step).
    - Target network (a delayed copy for stable bootstrap targets).

    It selects actions using an epsilon-greedy policy and performs TD learning
    with mean-squared error loss between predicted Q(s, a) and the target
    r + gamma * max_a' Q_target(s', a') * (1 - done).

    Args:
        state_dim (int): Dimension of the environment's state space.
        action_dim (int): Number of discrete actions.
        device (torch.device): Torch device ('cpu' or 'cuda').
        cfg (DQNConfig): Hyperparameter configuration.

    Attributes:
        q (QNetwork): Online Q-network.
        q_target (QNetwork): Target Q-network (periodically updated).
        optimizer (torch.optim.Optimizer): Adam optimizer over q parameters.
        action_dim (int): Cached number of actions.
        total_steps (int): Total environment steps seen by the agent, used for epsilon decay.
        device (torch.device): Torch device where computations occur.
        cfg (DQNConfig): Hyperparameter configuration.
    """

    def __init__(self, state_dim: int, action_dim: int, device: torch.device, cfg: DQNConfig):
        self.device = device
        self.cfg = cfg

        self.q = QNetwork(state_dim, action_dim).to(device)
        self.q_target = QNetwork(state_dim, action_dim).to(device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.action_dim = action_dim
        self.total_steps = 0

    def epsilon(self) -> float:
        """Compute the current epsilon value for epsilon-greedy exploration.

        Epsilon decays linearly from `eps_start` to `eps_end` over `eps_decay_steps`.
        After that, it remains at `eps_end`.

        Returns:
            float: Current epsilon in [eps_end, eps_start].
        """
        frac = max(0.0, 1.0 - self.total_steps / float(self.cfg.eps_decay_steps))
        return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * frac

    def act(self, state) -> int:
        """Select an action using an epsilon-greedy policy.

        With probability epsilon, a random action is chosen (exploration).
        Otherwise, the action that maximizes the current Q-value is chosen (exploitation).

        Args:
            state: Environment state. Can be a 1D array-like (state_dim,) or tensor.

        Returns:
            int: Selected action index in [0, action_dim).
        """
        self.total_steps += 1
        if torch.rand(1).item() < self.epsilon():
            return int(torch.randint(0, self.action_dim, (1,)).item())

        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q(s)
            return int(q_values.argmax(dim=1).item())

    def update(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """Perform a single DQN update step from a replay buffer batch.

        The batch must contain tensors already placed on the correct device:
        (states, actions, rewards, next_states, dones).

        Args:
            batch (Tuple[torch.Tensor, ...]): A 5-tuple containing:
                - states: (B, state_dim) float32
                - actions: (B, 1) int64
                - rewards: (B, 1) float32
                - next_states: (B, state_dim) float32
                - dones: (B, 1) float32 in {0.0, 1.0}

        Returns:
            float: The scalar training loss (MSE) for logging.
        """
        states, actions, rewards, next_states, dones = batch

        # Q(s, a) for the chosen actions
        q_sa = self.q(states).gather(1, actions)  # shape: (B, 1)

        with torch.no_grad():
            # max_a' Q_target(s', a')
            q_next_max, _ = self.q_target(next_states).max(dim=1, keepdim=True)
            targets = rewards + self.cfg.gamma * q_next_max * (1.0 - dones)

        loss = F.mse_loss(q_sa, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.gradient_clip)
        self.optimizer.step()

        return float(loss.item())

    def maybe_update_target(self) -> None:
        """Copy online network weights to the target network at fixed intervals.

        The copy occurs when `total_steps % target_update_every == 0`.
        This stabilizes training by decoupling target computation from rapid
        changes in the online network.
        """
        if self.total_steps % self.cfg.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
