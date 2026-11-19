from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from components.batch_schema import build_batch_schema
from components.episode_buffer import EpisodeBatch
from controllers.basic_controller import BasicMAC
from learners.ppo_learner import PPOLearner


def make_tabx_ppo(**overrides):
    config_path = Path(__file__).resolve().parents[1] / "src/config/algs/mappo_tabx.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.update(gamma=0.5, epochs=1, entropy_coef=0., use_gae=False)
    config.update(overrides)
    args = SimpleNamespace(
        **config, common_reward=True, use_cuda=False, n_agents=2, n_actions=2, learner_log_interval=1
    )
    scheme, groups, preprocess = build_batch_schema(
        dict(n_agents=2, n_actions=2, obs_shape=3, state_shape=4), include_agent_alive=True
    )
    batch = EpisodeBatch(scheme, groups, 2, 7, preprocess=preprocess)
    stats = {}
    logger = SimpleNamespace(log_stat=lambda name, value, step: stats.update({name: value}))
    learner = PPOLearner(BasicMAC(batch.scheme, groups, args), batch.scheme, logger, args)
    return learner, batch, stats


class FixedCritic(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(values)

    def forward(self, batch):
        return self.values.unsqueeze(-1)


def test_tabx_ppo_uses_team_targets_and_masks_actor_slots():
    learner, batch, stats = make_tabx_ppo()
    # One unit dies after the second transition. The other trajectory ends
    # after two transitions and leaves four padded frames.
    batch["filled"][0, :, 0] = 1
    batch["filled"][1, :3, 0] = 1
    batch["avail_actions"].fill_(1)
    batch["terminated"][0, 5, 0] = batch["terminated"][1, 1, 0] = 1
    batch["reward"][0, :6, 0] = torch.tensor([1., 2., 3., 4., 5., 20.])
    batch["reward"][1, :2, 0] = torch.tensor([-1., 8.])
    alive = batch["agent_alive"].squeeze(-1)
    alive[0, :6, 0] = alive[0, :2, 1] = alive[1, :3] = True

    learner.critic = FixedCritic(torch.tensor([1., 3.]).repeat(2, 7, 1))
    learner.target_critic = FixedCritic(torch.tensor([100., 900.]).repeat(2, 7, 1))
    learner.critic_params = list(learner.critic.parameters())
    learner.critic_optimiser = torch.optim.Adam(learner.critic_params, lr=learner.args.lr)
    team_values = learner._tabx_team_values(learner.target_critic.values, alive)
    assert team_values[0, -1, 0].item() == 0  # mutual elimination
    filled = batch["filled"].bool()
    targets = learner.tabx_nstep_returns(
        batch["reward"][:, :-1],
        filled[:, :-1] & filled[:, 1:],
        batch["terminated"][:, :-1].bool(),
        team_values,
    )
    first_returns = torch.tensor([6.6875, 6.375, 8.75, 11.5, 15., 20.])
    second_returns = torch.tensor([3., 8., 0., 0., 0., 0.])
    torch.testing.assert_close(targets[:, :, 0], torch.stack((first_returns, second_returns)))

    learner.train(batch, t_env=8, episode_num=2)

    expected_td = torch.cat((first_returns, second_returns[:2])) - torch.tensor([2., 2., 1., 1., 1., 1., 2., 2.])
    assert stats["critic_loss"] == pytest.approx(expected_td.square().mean().item())
    # Both units receive the same team advantage before the casualty. Only
    # the surviving unit contributes to the actor loss after the casualty.
    live_counts = torch.tensor([2., 2., 1., 1., 1., 1., 2., 2.])
    expected_policy_loss = -(expected_td * live_counts).sum() / live_counts.sum()
    assert stats["pg_loss"] == pytest.approx(expected_policy_loss.item())
    assert not learner.critic.values.grad[0, 2:, 1].any()
    assert not learner.critic.values.grad[1, 2:].any()


def test_tabx_ppo_ignores_nonzero_terminal_value_at_bootstrap_horizon():
    learner, _, _ = make_tabx_ppo()
    rewards = torch.zeros(1, 5, 1)
    rewards[0, -1, 0] = 7.
    valid = torch.ones_like(rewards, dtype=torch.bool)
    terminated = torch.zeros_like(valid)
    terminated[0, -1, 0] = True
    team_values = torch.zeros(1, 6, 1)
    team_values[0, -1, 0] = 1000.

    targets = learner.tabx_nstep_returns(rewards, valid, terminated, team_values)

    torch.testing.assert_close(targets.flatten(), torch.tensor([0.4375, 0.875, 1.75, 3.5, 7.]))


def test_tabx_gae_freezes_team_advantages_and_targets_for_four_epochs(monkeypatch):
    learner, batch, stats = make_tabx_ppo(use_gae=True, gamma=0.99, epochs=4)
    batch["filled"][0, :4, 0] = 1
    batch["filled"][1, :3, 0] = 1
    batch["avail_actions"].fill_(1)
    batch["terminated"][0, 2, 0] = batch["terminated"][1, 1, 0] = 1
    batch["reward"].fill_(100.)
    batch["reward"][0, :3, 0] = torch.tensor([0., 0., 1.])
    batch["reward"][1, :2, 0] = torch.tensor([0., 1.])
    batch["agent_alive"][0, :4, 0] = batch["agent_alive"][0, 0, 1] = True
    batch["agent_alive"][1, :3] = True
    initial_values = torch.full((2, 7, 2), 500.)
    initial_values[0, :4, 0] = torch.tensor([2., 1., 2., 999.])
    initial_values[0, 0, 1] = 4.
    initial_values[1, :2] = 0.
    initial_values[1, 2] = 999.
    learner.critic = FixedCritic(initial_values.clone())
    learner.critic_params = list(learner.critic.parameters())
    learner.critic_optimiser = torch.optim.Adam(learner.critic_params, lr=learner.args.lr)

    def unexpected_target_critic(_batch):
        pytest.fail("GAE must use the pre-update online critic")

    monkeypatch.setattr(learner.target_critic, "forward", unexpected_target_critic)
    computed_advantages, epoch_targets, actor_gradients = [], [], []
    compute_gae = learner._tabx_gae
    train_critic = learner.train_critic_sequential

    def capture_advantages(*args):
        assert not torch.is_grad_enabled()
        advantages, targets = compute_gae(*args)
        computed_advantages.append(advantages.clone())
        return advantages, targets

    def capture_targets(*args, target_returns):
        assert not target_returns.requires_grad
        epoch_targets.append(target_returns.clone())
        return train_critic(*args, target_returns=target_returns)

    def capture_actor_gradient():
        # Hold the actor fixed so changes in the critic cannot be hidden by a policy update.
        actor_gradients.append(torch.cat([param.grad.flatten() for param in learner.agent_params]))

    monkeypatch.setattr(learner, "_tabx_gae", capture_advantages)
    monkeypatch.setattr(learner, "train_critic_sequential", capture_targets)
    monkeypatch.setattr(learner.agent_optimiser, "step", capture_actor_gradient)

    learner.train(batch, t_env=5, episode_num=2)

    expected_advantages = torch.tensor([
        [-1.99108209, 0.0197, -1., 0., 0., 0.],
        [0.9603, 1., 0., 0., 0., 0.],
    ])
    expected_targets = expected_advantages + torch.tensor([
        [3., 1., 2., 0., 0., 0.], [0., 0., 0., 0., 0., 0.],
    ])
    assert len(computed_advantages) == 1
    torch.testing.assert_close(computed_advantages[0].squeeze(-1), expected_advantages)
    assert len(epoch_targets) == len(actor_gradients) == 4
    for target, gradient in zip(epoch_targets, actor_gradients):
        torch.testing.assert_close(target.squeeze(-1), expected_targets)
        torch.testing.assert_close(gradient, actor_gradients[0])
    assert not torch.equal(learner.critic.values, initial_values)
    live_counts = torch.tensor([[2., 1., 1., 0., 0., 0.], [2., 2., 0., 0., 0., 0.]])
    expected_policy_loss = -(expected_advantages * live_counts).sum() / live_counts.sum()
    assert stats["pg_loss"] == pytest.approx(expected_policy_loss.item(), abs=1e-6)
