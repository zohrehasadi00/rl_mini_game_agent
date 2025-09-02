import torch

from rl_agent.dqn import QNetwork, DQNAgent, DQNConfig


def test_qnetwork_shapes():
    state_dim, action_dim = 4, 2
    net = QNetwork(state_dim, action_dim)
    x = torch.randn(8, state_dim)
    q = net(x)
    assert q.shape == (8, action_dim), f"Unexpected shape: {q.shape}"


def test_agent_update_step():
    device = torch.device("cpu")
    cfg = DQNConfig(gamma=0.99, lr=1e-3, batch_size=4, target_update_every=5)
    agent = DQNAgent(state_dim=4, action_dim=2, device=device, cfg=cfg)

    B, state_dim = 8, 4
    # Fake batch on correct device
    states = torch.randn(B, state_dim, device=device)
    actions = torch.randint(0, 2, (B, 1), device=device, dtype=torch.long)
    rewards = torch.randn(B, 1, device=device)
    next_states = torch.randn(B, state_dim, device=device)
    dones = torch.randint(0, 2, (B, 1), device=device, dtype=torch.float32)

    loss = agent.update((states, actions, rewards, next_states, dones))
    assert isinstance(loss, float), "Loss should be a float"
    assert loss >= 0.0, "Loss should be non-negative"

    # Step epsilon/target update behavior
    eps_before = agent.epsilon()
    _ = agent.act(states[0].cpu().numpy())  # increments total_steps
    eps_after = agent.epsilon()
    assert eps_after <= eps_before + 1e-6, "Epsilon should not increase during decay"

    # Ensure target network copy runs without error
    for _ in range(cfg.target_update_every - (agent.total_steps % cfg.target_update_every)):
        agent.total_steps += 1
    agent.maybe_update_target()  # should trigger a copy
