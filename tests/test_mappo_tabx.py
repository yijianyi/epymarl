from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from components.batch_schema import build_batch_schema
from components.episode_buffer import EpisodeBatch
from controllers.basic_controller import BasicMAC
from learners.ppo_learner import PPOLearner


def make_tabx_ppo(*, recorded_behavior=False, **overrides):
    config_path = Path(__file__).resolve().parents[1] / "src/config/algs/mappo_tabx.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.update(gamma=0.5, epochs=1, entropy_coef=0., use_gae=False)
    config.update(overrides)
    args = SimpleNamespace(
        **config, common_reward=True, use_cuda=False, n_agents=2, n_actions=2, learner_log_interval=1
    )
    scheme, groups, preprocess = build_batch_schema(
        dict(n_agents=2, n_actions=2, obs_shape=3, state_shape=4), include_agent_alive=True,
        include_ppo_behavior=recorded_behavior,
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


def test_recorded_temperature_and_teacher_control_in_ppo():
    learner, batch, stats = make_tabx_ppo(recorded_behavior=True, use_gae=True, obs_last_action=True)
    batch["filled"][:, :4] = 1
    batch["avail_actions"].fill_(1)
    batch["agent_alive"][0, :3] = True
    batch["agent_alive"][1, :3, 0] = True
    batch["reward"][:, :3] = 1
    batch["terminated"][:, 2] = 1
    batch["teacher_control"][:, 1] = True
    batch["control_remaining"][:, :3, 0] = torch.tensor([2, 1, 2])
    batch["policy_temperature"][:, :3] = 1.2
    with torch.no_grad():
        recorded = learner._log_policy(learner.mac, batch).gather(-1, batch["actions"][:, :-1])
        batch["behavior_log_prob"][:, :-1] = recorded
        batch["behavior_log_prob"][:, 1] = -1000

    learner.train(batch, t_env=6, episode_num=2)

    assert stats["first_ratio_max_error"] < 1e-6
    assert stats["teacher_fraction"] == pytest.approx(1 / 3)
    assert stats["actor_actions"] == 6
    actor = [parameter.detach().clone() for parameter in learner.agent_params]
    critic = [parameter.detach().clone() for parameter in learner.critic_params]
    optimizer_steps = [state["step"].item() for state in learner.agent_optimiser.state.values()]

    batch["teacher_control"][:, :3] = True
    learner.train(batch, t_env=12, episode_num=4)

    assert stats["teacher_fraction"] == 1
    assert stats["actor_actions"] == stats["pg_loss"] == stats["entropy"] == 0
    assert all(torch.equal(before, after) for before, after in zip(actor, learner.agent_params))
    assert optimizer_steps == [state["step"].item() for state in learner.agent_optimiser.state.values()]
    assert any(not torch.equal(before, after) for before, after in zip(critic, learner.critic_params))


def test_medium_bc_schedule_and_batch_overrides(tmp_path):
    from tabx_config import rollout_schedule
    from tabx_scheduler import training_config

    config, overrides = training_config(
        ["tabx_bc=True", f"tabx_demo_root={tmp_path}", "tabx_dagger_rounds=0",
         "tabx_mix_start=0.5", "tabx_mappo_temp_peak=1.2"],
        algorithm="mappo_tabx",
    )
    args = SimpleNamespace(**config)
    assert overrides["tabx_dagger_rounds"] == 0
    assert rollout_schedule(args, 0) == (0.5, 1.0)
    assert rollout_schedule(args, 20000) == pytest.approx((0.3, 1.2))
    assert rollout_schedule(args, 50000) == pytest.approx((0.0, 1.125))
    assert rollout_schedule(args, 100000) == (0.0, 1.0)


def test_bc_uses_legal_teacher_labels_on_live_transitions():
    from tabx_imitation import behavior_cloning_loss

    batch = {
        "teacher_actions": torch.ones((1, 3, 3, 1), dtype=torch.long),
        "actions": torch.zeros((1, 3, 3, 1), dtype=torch.long),
        "transition_mask": torch.tensor([[[1], [1], [0]]]),
        "filled": torch.ones((1, 3, 1)),
        "present": torch.tensor([1, 1, 0]).reshape(1, 1, 3, 1).expand(1, 3, 3, 1),
        "alive": torch.tensor([1, 0, 1]).reshape(1, 1, 3, 1).expand(1, 3, 3, 1),
        "teacher_mask": torch.tensor([1, 0, 0]).reshape(1, 3, 1, 1).expand(1, 3, 3, 1),
        "avail_actions": torch.tensor([1, 1, 0]).expand(1, 3, 3, 3),
    }
    logits = torch.tensor([0., 1., 100.]).expand(1, 2, 3, 3).clone().requires_grad_()
    loss = behavior_cloning_loss(logits, batch)
    loss.backward()

    assert loss.item() == pytest.approx(torch.log1p(torch.exp(torch.tensor(-1.))).item())
    assert logits.grad[0, 0, 0, 1] < 0
    assert logits.grad[0, 0, 0, 2] == 0
    assert not logits.grad[0, 0, 1:].any()
    assert not logits.grad[0, 1:].any()


def test_bc_uses_manifest_bytes_that_passed_the_hash_check(tmp_path, monkeypatch):
    import tabx_imitation
    from tabx_storage import atomic_json, file_sha256

    directory = tmp_path / "hp/23/medium"
    manifest = directory / "manifest.json"
    shard = directory / "0000000001.h5"
    metadata = dict(task_id="23", timeout_rule="hp", episode_limit=8, collection_phase="expert_collect")
    directory.mkdir(parents=True)
    shard.write_bytes(b"verified shard")
    atomic_json(manifest, dict(metadata=metadata, shards=[dict(path=shard.name, sha256=file_sha256(shard))]))
    pinned = file_sha256(manifest)
    args = SimpleNamespace(tabx_demo_root=str(tmp_path), tabx_bc_teacher="medium",
                           tabx_expected_demo_manifest_sha256=pinned, seed=1)
    env = SimpleNamespace(manifest={"task_id": "23"}, rule="hp", episode_limit=8)
    monkeypatch.setattr(tabx_imitation, "HeuristicTeacher", lambda *_: SimpleNamespace(metadata=lambda: {}))
    monkeypatch.setattr(tabx_imitation, "simulator_identity", lambda: {})
    monkeypatch.setattr(tabx_imitation, "verify_shard", lambda *_: dict(
        metadata=metadata, schema_version=tabx_imitation.SCHEMA_VERSION,
        label_source="deterministic_teacher", episodes=1, seeds=[1],
    ))
    read_bytes = Path.read_bytes

    def read_then_publish(path):
        data = read_bytes(path)
        if path == manifest:
            (directory / "0000000002.h5").write_bytes(b"new shard")
            atomic_json(manifest, dict(metadata=metadata, shards=[]))
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_publish)
    pool = tabx_imitation.DemonstrationPool(env, args)
    assert pool.manifest_sha256 == pinned
    assert pool.entries == [(shard, 0)]
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch for task 23"):
        tabx_imitation.DemonstrationPool(env, args)
