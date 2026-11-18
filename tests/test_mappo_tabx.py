from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from components.batch_schema import build_batch_schema
from components.episode_buffer import EpisodeBatch
from controllers.basic_controller import BasicMAC
from learners.ppo_learner import PPOLearner


@pytest.fixture
def tabx_ppo():
    config_path = Path(__file__).resolve().parents[1] / "src/config/algs/mappo_tabx.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.update(gamma=0.5, epochs=1, entropy_coef=0.)
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


def test_tabx_ppo_uses_team_targets_and_masks_actor_slots(tabx_ppo):
    learner, batch, stats = tabx_ppo
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


def test_tabx_ppo_ignores_nonzero_terminal_value_at_bootstrap_horizon(tabx_ppo):
    learner, _, _ = tabx_ppo
    rewards = torch.zeros(1, 5, 1)
    rewards[0, -1, 0] = 7.
    valid = torch.ones_like(rewards, dtype=torch.bool)
    terminated = torch.zeros_like(valid)
    terminated[0, -1, 0] = True
    team_values = torch.zeros(1, 6, 1)
    team_values[0, -1, 0] = 1000.

    targets = learner.tabx_nstep_returns(rewards, valid, terminated, team_values)

    torch.testing.assert_close(targets.flatten(), torch.tensor([0.4375, 0.875, 1.75, 3.5, 7.]))
