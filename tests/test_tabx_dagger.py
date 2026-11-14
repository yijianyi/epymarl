"""Deterministic labels, executed histories, weighted BC and DAgger provenance."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tabx_config import bc_stage_budgets, configure_bc
from tabx_dataset import (episodes_to_batch, read_episode, read_episode_batch, trajectory_hash,
                          validate_episode, verify_shard, write_shard)
from tabx_imitation import behavior_cloning_loss, DemonstrationPool
from tabx_scheduler import training_config
from tabx_storage import atomic_json, file_sha256, read_json
from test_tabx import controlled_rollout
from test_tabx_official import tiny_task


def labeled_episode(monkeypatch, seed=11, length=3):
    _, episode = controlled_rollout(monkeypatch, length, seed)
    labels = episode["transitions"]["actions"].copy()
    labels[:, 0] = 4
    labels[:, 1] = 5
    episode["transitions"].update(teacher_actions=labels,
        teacher_mask=episode["frames"]["alive"][:-1, :, None].copy())
    return episode


def test_labels_schema_hash_and_executed_rnn_history(monkeypatch, tmp_path):
    from controllers.basic_controller import BasicMAC
    from envs.tabx_layout import TABXLayout
    episode = labeled_episode(monkeypatch)
    path = write_shard(tmp_path / "labeled.h5", [episode], TABXLayout(9, 1, 0).metadata())
    assert verify_shard(path)["schema_version"] == "tabx-hdf5-3"
    batch, _ = read_episode_batch(path)
    assert batch["teacher_actions"].dtype == torch.int64
    assert batch["teacher_mask"].shape == (1, 4, 10, 1)
    assert batch["teacher_mask"][:, -1].sum() == 0
    assert batch["teacher_mask"][:, :, -1].sum() == 0
    config, _ = training_config([])
    config.update(n_agents=10, n_actions=8, device="cpu", use_cuda=False)
    mac = BasicMAC(batch.scheme, batch.groups, NS(**config))
    previous = mac._build_inputs(batch, 1).reshape(1, 10, -1)[:, :, 378:386]
    assert previous[0, 0].argmax() == 0  # executed UP, though advice was ATTACK
    logits = torch.zeros(1, 4, 10, 8)
    logits[:, :, 0, 4] = 10
    clean, _ = behavior_cloning_loss(logits, batch, target="teacher_actions")
    legacy, _ = behavior_cloning_loss(logits, batch, target="actions")
    assert clean < legacy
    changed = deepcopy(episode)
    changed["transitions"]["teacher_actions"][0, 0] = 5
    assert trajectory_hash(changed) != trajectory_hash(episode)
    with h5py.File(path, "r+") as data:
        data["transitions/teacher_actions"][0, 0, 0] = 5
    with pytest.raises(ValueError, match="Trajectory checksum"):
        verify_shard(path)
    changed["frames"]["avail_actions"][0, 0, 5] = 0
    with pytest.raises(ValueError, match="Illegal teacher action"):
        validate_episode(changed)
    _, old = controlled_rollout(monkeypatch, 2, 12)
    old_path = write_shard(tmp_path / "legacy.h5", [old], TABXLayout(9, 1, 0).metadata())
    assert verify_shard(old_path)["schema_version"] == "tabx-hdf5-2"
    assert read_episode(old_path, 0)["label_source"] == "executed_actions"
    old_batch, _ = read_episode_batch(old_path)
    with pytest.raises(ValueError, match="required"):
        behavior_cloning_loss(torch.zeros(1, 3, 10, 8), old_batch, target="teacher_actions")


def test_weighted_loss_target_weights_and_masked_metrics(monkeypatch):
    episode = labeled_episode(monkeypatch)
    # Ignore a living unit's label; death, short final frames and padding are also masked.
    episode["transitions"]["teacher_mask"][1:, 2] = 0
    episode["frames"]["alive"][1:, 3] = 0
    episode["frames"]["avail_actions"][1:, 3] = 0
    episode["frames"]["avail_actions"][1:, 3, 7] = 1
    episode["transitions"]["teacher_mask"][1:, 3] = 0
    episode["transitions"]["actions"][1:, 3] = 7
    episode["transitions"]["teacher_actions"][1:, 3] = 7
    short = labeled_episode(monkeypatch, seed=12, length=1)
    batch, _ = episodes_to_batch([episode, short])
    torch.manual_seed(5)
    logits = torch.randn(2, 4, 10, 8, requires_grad=True)
    weights = [1, 1, 1, 1, 3, 3, 3, 1]
    loss, metrics = behavior_cloning_loss(logits, batch, target="teacher_actions", action_weights=weights)
    valid = batch["teacher_mask"][:, :-1, :, 0].bool()
    selected = logits[:, :-1][valid]
    labels = batch["teacher_actions"][:, :-1, :, 0][valid]
    ce = F.cross_entropy(selected, labels, reduction="none")
    w = torch.tensor(weights)[labels]
    torch.testing.assert_close(loss, (ce * w).sum() / w.sum())
    unit_loss, _ = behavior_cloning_loss(logits, batch, target="teacher_actions", action_weights=[1]*8)
    torch.testing.assert_close(unit_loss, ce.mean())
    assert metrics["unweighted_ce"] == pytest.approx(float(ce.mean()))
    assert sum(metrics["action_support"]) == metrics["valid_actions"] == int(valid.sum())
    loss.backward()
    assert logits.grad[0, 1:, 2].count_nonzero() == 0
    assert logits.grad[0, 1:, 3].count_nonzero() == 0
    assert logits.grad[:, :, -1].count_nonzero() == 0
    assert logits.grad[0, -1].count_nonzero() == 0
    assert logits.grad[1, 1:].count_nonzero() == 0


def test_fixed_budget_defaults_and_explicit_ablation_configs():
    config, _ = training_config(["tabx_bc=True", "tabx_demo_root=demos"])
    assert (config["tabx_bc_steps"], config["tabx_bc_lr"], config["hidden_dim"], config["tabx_bc_batch_size"]) == (1000, .0005, 128, 32)
    assert bc_stage_budgets(config) == [400, 200, 200, 200]
    assert config["tabx_dagger_episodes"] == 32
    assert config["tabx_bc_target"] == "teacher_actions"
    assert config["tabx_bc_action_weights"] == [1, 1, 1, 1, 3, 3, 3, 1]
    config["tabx_bc_steps"] = 5000
    configure_bc(config)
    assert bc_stage_budgets(config) == [2000, 1000, 1000, 1000]
    legacy, _ = training_config(["tabx_bc=True", "tabx_demo_root=demos", "tabx_dagger_rounds=0",
        "tabx_bc_target=actions", "tabx_bc_action_weights=[1,1,1,1,1,1,1,1]"])
    assert bc_stage_budgets(legacy) == [1000]
    with pytest.raises(ValueError, match="DAgger requires"):
        configure_bc(dict(config, tabx_bc_target="actions"))
    with pytest.raises(ValueError, match="eight finite positive"):
        configure_bc(dict(config, tabx_bc_action_weights=[0]*8))


def test_teacher_pair_uses_same_previous_memory_and_epsilon_zero(tmp_path):
    from envs.tabx_wrapper import TABXWrapper
    from tabx_expert import HeuristicTeacher
    import jax.numpy as jnp
    bank = tmp_path / "bank.json"
    atomic_json(bank, dict(schema_version="1.0", tasks=[tiny_task()]))
    env = TABXWrapper("tiny", task_bank=bank, max_episode_steps=4)
    try:
        env.reset(seed=7)
        teacher = HeuristicTeacher(env, "advanced", .1)
        teacher.reset(7)
        teacher.memory = {key: 0 for key in env.base_env.ally_keys}
        calls = []
        def policy(key, obs, previous, params, physics):
            calls.append(previous)
            return jnp.array(0 if float(params.epsilon.item()) else 1), previous + 1
        teacher.policy = policy
        actions, labels, mask = teacher.actions_and_labels()
        assert actions[0] == 0 and labels[0] == 1 and mask[0]
        assert calls == [0, 0] and list(teacher.memory.values()) == [1]
        teacher.actions_and_labels()
        assert calls == [0, 0, 1, 1] and list(teacher.memory.values()) == [2]
        # Real official policy: epsilon zero has identical execution and advice.
        zero = HeuristicTeacher(env, "advanced", 0.)
        zero.reset(7)
        action, advice, _ = zero.actions_and_labels()
        np.testing.assert_array_equal(action, advice)
        avail = env.get_avail_actions()
        assert avail[np.arange(10), action].all()
        assert avail[np.arange(10), advice].all()
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get("TABX_BC_SMOKE") != "1", reason="Opt in with TABX_BC_SMOKE=1")
def test_official_relabel_and_three_round_dagger_handoff(tmp_path):
    bank = tmp_path / "bank.json"
    atomic_json(bank, dict(schema_version="1.0", tasks=[tiny_task()]))
    demos, output = tmp_path / "demos", tmp_path / "output"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def execute(arguments, name):
        log = tmp_path / f"{name}.log"
        with log.open("w") as stream:
            result = subprocess.run([sys.executable, *arguments], env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == 0, log.read_text()[-8000:]
    execute(["src/tabx_pipeline.py", "collect-expert", "--bank", str(bank), "--root", str(demos), "--task", "tiny",
             "--episodes", "2", "--episode-limit", "4", "--shard-episodes", "1"], "collect")
    source = demos / "hp/tiny/advanced"
    # Reconstruct an actual v2 source from the collected executed trajectories.
    legacy = tmp_path / "legacy/hp/tiny/advanced"
    legacy.mkdir(parents=True)
    manifest = read_json(source / "manifest.json")
    metadata = {key:value for key,value in manifest["metadata"].items() if not key.startswith("label_")}
    metadata["label_source"] = "executed_actions"
    entries = []
    for entry in manifest["shards"]:
        ep = read_episode(source / entry["path"], 0)
        ep["transitions"].pop("teacher_actions"); ep["transitions"].pop("teacher_mask")
        path = write_shard(legacy / entry["path"], [ep], metadata)
        entries.append(dict(path=path.name, sha256=file_sha256(path)))
    atomic_json(legacy / "manifest.json", dict(manifest, metadata=metadata, shards=entries))
    before = {path.name:path.read_bytes() for path in legacy.iterdir()}
    relabeled = tmp_path / "relabeled/hp/tiny/advanced"
    execute(["src/tabx_pipeline.py", "label-expert", "--bank", str(bank), "--task", "tiny",
             "--source", str(legacy), "--output", str(relabeled)], "relabel")
    assert before == {path.name:path.read_bytes() for path in legacy.iterdir()}
    for entry in entries:
        old, new = read_episode(legacy / entry["path"], 0), read_episode(relabeled / entry["path"], 0)
        for group in ("frames", "transitions"):
            for key, value in old[group].items():
                np.testing.assert_array_equal(value, new[group][key])
        fresh = read_episode(source / entry["path"], 0)
        np.testing.assert_array_equal(new["transitions"]["teacher_actions"], fresh["transitions"]["teacher_actions"])
    execute(["src/main.py", "--config=qmix_tabx", "--env-config=tabx", "with", "env_args.key=tiny",
        f"env_args.task_bank={bank}", f"tabx_output_root={output}", "tabx_run_id=dagger-smoke", "seed=7", "use_cuda=False",
        "tabx_bc=True", f"tabx_demo_root={tmp_path / 'relabeled'}", "tabx_bc_steps=4", "tabx_bc_batch_size=2",
        "tabx_bc_eval_episodes=2", "tabx_bc_require_wins=False", "tabx_dagger_episodes=2",
        "tabx_demo_decay_steps=8", "tabx_demo_fraction=0.5", "t_max=12", "batch_size=2", "buffer_size=4",
        "env_args.max_episode_steps=4", "save_model_interval=8", "tabx_screen_episodes=2", "learner_log_interval=4"], "train")
    run = output / "hp/tiny/7/dagger-smoke"
    summary = read_json(run / "bc_summary.json")
    assert summary["bc_updates"] == 4 and summary["stage_updates"] == [1,1,1,1]
    assert summary["dagger_episodes"] == 6 and summary["dagger_environment_steps"] == 24
    assert summary["demonstration_environment_steps"] == 8 and summary["online_t_env"] == 0
    optimizer = torch.load(run / "bc_optimizer.th", weights_only=True)
    assert optimizer["state"] and all(float(state["step"]) == 4 for state in optimizer["state"].values())
    from tabx_storage import stable_seed
    all_seeds = set()
    failures, collector_hashes = 0, []
    for round_index in (1,2,3):
        directory = run / "dagger" / f"round_{round_index}"
        report = read_json(directory / "manifest.json")
        assert report["metadata"]["dagger_round"] == round_index
        assert file_sha256(directory / "collector/agent.th") == report["metadata"]["collector_checkpoint_sha256"]
        collector_hashes.append(report["metadata"]["collector_checkpoint_sha256"])
        assert not all_seeds.intersection(report["seeds"])
        all_seeds.update(report["seeds"])
        failures += report["episodes"] - report["wins"]
        for entry in report["shards"]:
            assert file_sha256(directory / entry["path"]) == entry["sha256"]
    assert failures > 0
    assert len(set(collector_hashes)) == 3
    evaluation = read_json(run / "bc_evaluations/after_bc_online_step_0.json")
    assert not all_seeds.intersection(evaluation["seeds"])
    assert evaluation["online_t_env"] == 0 and evaluation["model_teacher_agreement"]["valid_actions"] > 0
    train = [json.loads(line) for line in (run / "training_episodes.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in train] == [0,1,2]
    assert read_json(run / "training_summary.json")["t_env"] == 12
    assert (run / "online_bc_metrics.jsonl").exists()


def test_aggregation_half_original_and_round_uniform_including_failures(monkeypatch):
    from tabx_imitation import AggregatedDemonstrationPool
    import tabx_imitation
    episode = labeled_episode(monkeypatch)
    failed = deepcopy(episode)
    failed["episode"].update(is_win=0, winner=1)
    requested, selected = [], []
    class Original:
        entries, wins, metadata, directory, collection_steps, seeds = [0], [0], {}, Path("unused"), 3, {1}
        def sample_episodes(self, count):
            requested.append(count)
            return [deepcopy(episode) for _ in range(count)]
    pool = AggregatedDemonstrationPool(Original(), 7)
    pool.rounds = [[("round1", i) for i in range(1)], [("round2", i) for i in range(7)], [("round3", i) for i in range(30)]]
    def read(path, index):
        selected.append(path)
        return failed
    monkeypatch.setattr(tabx_imitation, "read_episode", read)
    for _ in range(60):
        batch = pool.sample(32)
        assert batch.batch_size == 32 and batch["has_teacher_labels"].sum() == 32
    assert requested == [16] * 60
    assert len(selected) == 16 * 60
    for name in ("round1", "round2", "round3"):
        assert .27 < selected.count(name) / len(selected) < .40


def test_td_keeps_executed_action_when_teacher_advice_differs(monkeypatch, tmp_path):
    from learners.q_learner import QLearner
    episode = labeled_episode(monkeypatch)
    batch, _ = episodes_to_batch([episode])
    class MAC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
        def init_hidden(self, batch_size):
            pass
        def forward(self, batch, t):
            return self.q.expand(batch.batch_size, 10, 8)
    args = NS(n_agents=10, mixer=None, lr=.01, learner_log_interval=1, use_cuda=False,
        standardise_returns=False, standardise_rewards=False, common_reward=True, double_q=True,
        gamma=0., grad_norm_clip=10, target_update_interval_or_tau=200, env="tabx", tabx_run_dir=str(tmp_path),
        tabx_bc_target="teacher_actions", tabx_bc_action_weights=[1,1,1,1,3,3,3,1])
    logged = {}
    learner = QLearner(MAC(), batch.scheme, NS(log_stat=lambda k,v,t: logged.update({k:v})), args)
    learner.train(batch, 1, 0, demo_batch=batch, bc_weight=0.)
    # Only the padding slot executed IDLE; real units executed UP, not advice ATTACK/turn.
    assert logged["q_taken_mean"] == pytest.approx(.7)


def test_dagger_freezes_collector_and_resets_each_episode(monkeypatch, tmp_path):
    from envs import REGISTRY
    from envs.tabx_layout import TABXLayout
    from components.batch_schema import build_batch_schema
    from controllers.basic_controller import BasicMAC
    from runners.episode_runner import EpisodeRunner
    from tabx_pipeline import QuietLogger
    from tabx_dagger import collect_dagger_round
    import tabx_dagger
    from test_tabx import ControlledEnv
    metadata = dict(task_id="fixture", task_hash="a"*64, bank_hash="b"*64, **TABXLayout(9, 1, 0).metadata())
    class Env(ControlledEnv):
        def __init__(self, **kwargs):
            super().__init__(length=3)
            self.manifest, self.rule = metadata, "hp"
        def step(self, actions):
            obs, reward, end, truncated, info = super().step(actions)
            info["is_win"] = 0
            return obs, reward, end, truncated, info
    resets = []
    class Teacher:
        def __init__(self, env, *args):
            self.env = env
        def metadata(self):
            return dict(teacher="advanced", teacher_epsilon=0., label_source="deterministic_teacher", label_epsilon=0.)
        def reset(self, seed):
            resets.append(seed)
            self.t = 0
        def actions_and_labels(self):
            assert self.t == self.env.t
            self.t += 1
            a = self.env.get_avail_actions().argmax(-1)
            return a, a.copy(), self.env.get_frame()["alive"].reshape(10, 1)
    monkeypatch.setitem(REGISTRY, "controlled", Env)
    monkeypatch.setattr(tabx_dagger, "HeuristicTeacher", Teacher)
    config, _ = training_config([])
    config.update(env="controlled", env_args={}, device="cpu", use_cuda=False, batch_size_run=1, n_agents=10,
        n_actions=8, state_shape=340, tabx_run_dir=str(tmp_path), tabx_dagger_episodes=2, tabx_bc_eval_episodes=2, seed=7)
    args = NS(**config)
    runner = EpisodeRunner(args, QuietLogger())
    scheme, groups, preprocess = build_batch_schema(runner.get_env_info())
    mac = BasicMAC(scheme | {"actions_onehot": {"vshape": (8,)}}, groups, args)
    runner.setup(scheme, groups, preprocess, mac)
    seen, gradients = [], []
    def observe(module, inputs):
        seen.append(torch.cat([p.detach().flatten() for p in module.parameters()]).clone())
        gradients.append(any(p.requires_grad for p in module.parameters()))
    mac.agent.register_forward_pre_hook(observe)
    before = deepcopy(mac.agent.state_dict())
    paths = collect_dagger_round(NS(mac=mac), runner, NS(seeds=set()), args, 1)
    assert len(resets) == 2 and len(set(resets)) == 2
    assert seen and not any(gradients) and all(torch.equal(seen[0], item) for item in seen)
    assert runner.mac is mac and runner.t_env == 0
    for name, parameter in mac.agent.state_dict().items():
        torch.testing.assert_close(parameter, before[name])
    report = read_json(paths[0].parent / "manifest.json")
    assert report["wins"] == 0 and report["episodes"] == 2


@pytest.mark.skipif(os.environ.get("TABX_BC_SMOKE") != "1", reason="Opt in with TABX_BC_SMOKE=1")
def test_three_ablation_modes_share_budget_and_agreement(tmp_path):
    bank = tmp_path / "bank.json"
    atomic_json(bank, dict(schema_version="1.0", tasks=[tiny_task()]))
    demos, output = tmp_path / "demos", tmp_path / "output"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    def execute(command, name):
        log = tmp_path / f"{name}.log"
        with log.open("w") as stream:
            result = subprocess.run([sys.executable, *command], env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == 0, log.read_text()[-5000:]
    execute(["src/tabx_pipeline.py", "collect-expert", "--bank", str(bank), "--root", str(demos),
             "--task", "tiny", "--episodes", "2", "--episode-limit", "2"], "collect")
    for name, target, rounds in (("original", "actions", 0), ("clean", "teacher_actions", 0), ("dagger", "teacher_actions", 3)):
        command = ["src/main.py", "--config=qmix_tabx", "--env-config=tabx", "with", "env_args.key=tiny",
            f"env_args.task_bank={bank}", f"tabx_output_root={output}", f"tabx_run_id={name}", "seed=7", "use_cuda=False",
            "tabx_bc=True", f"tabx_demo_root={demos}", "tabx_bc_steps=4", "tabx_bc_eval_episodes=2",
            "tabx_bc_require_wins=False", f"tabx_bc_target={target}", f"tabx_dagger_rounds={rounds}",
            "tabx_dagger_episodes=1", "env_args.max_episode_steps=2", "t_max=0"]
        if name == "original":
            command.append("tabx_bc_action_weights=[1,1,1,1,1,1,1,1]")
        execute(command, name)
        run = output / "hp/tiny/7" / name
        summary = read_json(run / "bc_summary.json")
        evaluation = read_json(run / "bc_evaluations/after_bc_online_step_0.json")
        assert summary["bc_updates"] == 4 and summary["bc_target"] == target
        assert summary["dagger_environment_steps"] == (6 if rounds else 0)
        assert evaluation["online_t_env"] == 0
        assert evaluation["model_teacher_agreement"]["valid_actions"] > 0
        assert read_json(run / "training_summary.json")["t_env"] == 0
