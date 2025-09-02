# Reinforcement Learning Mini-Game Agent

## 🎯 Project Idea
The goal of this project was to build a **Reinforcement Learning (RL) agent from scratch** using **PyTorch** and apply it to a simple environment: **CartPole-v1** from OpenAI Gym.

CartPole is a classic control problem: a pole is attached to a cart that moves left or right. The agent must learn to balance the pole by choosing discrete actions (`left`, `right`).

This project was designed as a **first PyTorch project**:
- To learn how to implement and train neural networks in PyTorch.
- To understand the full pipeline of RL: environment interaction, replay buffer, policy updates, logging, and evaluation.
- To create a clean, GitHub-worthy project with reproducibility and clarity.


## 🧠 Techniques Used
### Core Method: **Deep Q-Network (DQN)**
- **Q-Learning** with a neural network approximating action-value function.
- **Replay Buffer** to store transitions `(s, a, r, s', done)` and break correlation.
- **Target Network** to stabilize updates (periodically synced with online net).
- **Epsilon-Greedy Exploration** (ε decays from 1.0 → 0.05).

### Implementation Details
- Neural network: 2 hidden layers (128 units each, ReLU activations).
- Loss: **MSE / Huber** between predicted Q(s,a) and TD target.
- Optimizer: Adam.
- Gradient clipping for stability.
- Logging with **TensorBoard**.
- Evaluation script with option to record **videos** of trained policy.


## ⚙️ Installation
```bash
# clone repo
git clone https://github.com/<your-username>/mini-rl-agent.git
cd mini-rl-agent

# create and activate venv
python -m venv .venv
.venv\Scripts\activate  # (Windows PowerShell)

# install dependencies
pip install -r requirements.txt



