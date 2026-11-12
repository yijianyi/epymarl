"""Executed-action BC masks, independent replay and the official training handoff."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from components.batch_schema import build_batch_schema
from components.episode_buffer import EpisodeBatch
from controllers.basic_controller import BasicMAC
from envs.tabx_layout import TABXLayout
from learners.q_learner import QLearner
from tabx_dataset import episodes_to_batch, read_episode_batch, write_shard
from tabx_imitation import (DemonstrationPool, behavior_cloning_loss, configure_bc,
                           demonstration_schedule, mix_batches, pretrain)
from tabx_pipeline import QuietLogger, parser
from tabx_scheduler import training_config
from tabx_storage import atomic_json, file_sha256, provenance, read_json
from test_tabx import controlled_rollout
from test_tabx_official import tiny_task


def config(**overrides):
    values, _ = training_config([])
    values.update(use_cuda=False, device="cpu", n_agents=10, n_actions=8, state_shape=340, **overrides)
    return values


def test_opt_in_config_preserves_baseline_and_sets_online_exploration():
    baseline = config()
    assert configure_bc(baseline)["epsilon_start"] == 1.
    bc = config(tabx_bc=True, tabx_demo_root="demos")
    configure_bc(bc)
    assert (bc["epsilon_start"], bc["epsilon_finish"], bc["epsilon_anneal_time"]) == (.1, .05, 100000)
    args = NS(**bc)
    assert demonstration_schedule(args, 0) == (8, 1.)
    assert demonstration_schedule(args, 50000) == (4, .5)
    assert demonstration_schedule(args, 100000) == (0, 0.)
    assert demonstration_schedule(args, 100001) == (0, 0.)
    for kwargs in ({"tabx_demo_fraction": 1.}, {"tabx_bc_steps": 0}, {"evaluation_epsilon": .05},
                   {"checkpoint_path": "old"}, {"tabx_bc_lr": float("nan")}):
        with pytest.raises(ValueError):
            configure_bc(config(tabx_bc=True, tabx_demo_root="demos", **kwargs))
    cli = parser().parse_args(["collect-expert", "--bank", "x", "--root", "y", "--task", "647"])
    assert cli.teacher == "advanced" and cli.episodes == 128 and cli.epsilon is None and cli.episode_limit == 512


def test_readback_bc_masks_dead_absent_terminal_padding_and_legal_logits(tmp_path, monkeypatch):
    _, first = controlled_rollout(monkeypatch, 3, 11)
    _, second = controlled_rollout(monkeypatch, 1, 12)
    # Real agent 0 dies after the first action.
    first["frames"]["alive"][1:, 0] = 0
    first["frames"]["avail_actions"][1:, 0] = 0
    first["frames"]["avail_actions"][1:, 0, 7] = 1
    first["transitions"]["actions"][1:, 0] = 7
    path = write_shard(tmp_path / "data.h5", [first, second], TABXLayout(9, 1, 0).metadata())
    batch, lengths = read_episode_batch(path)
    assert lengths.tolist() == [3, 1]
    assert batch["present"].shape == (2, 4, 10, 1)
    logits = torch.zeros(2, 4, 10, 8, requires_grad=True)
    loss, metrics = behavior_cloning_loss(logits, batch)
    assert metrics["valid_actions"] == 9 * 4 - 2
    assert metrics["accuracy"] == 1.
    assert loss.item() == pytest.approx(np.log(8))
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 1:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, :, -1]) == 0
    assert torch.count_nonzero(logits.grad[0, 3:]) == 0
    assert torch.count_nonzero(logits.grad[1, 1:]) == 0
    # A huge illegal action score must have no effect on loss or accuracy.
    batch["avail_actions"][:, :, 1, 6] = 0
    altered = logits.detach().clone()
    altered[:, :, 1, 6] = 1e10
    masked_loss, _ = behavior_cloning_loss(altered, batch)
    assert masked_loss < loss
    mac = BasicMAC(batch.scheme, batch.groups, NS(**config()))
    previous = mac._build_inputs(batch, 1).reshape(2, 10, -1)[:, :, 378:386]
    torch.testing.assert_close(previous, batch["actions_onehot"][:, 0])


def test_separate_pool_oversamples_wins_validates_identity_and_trains_qmix(tmp_path, monkeypatch):
    _, first = controlled_rollout(monkeypatch, 3, 11)
    _, second = controlled_rollout(monkeypatch, 1, 12)
    second["episode"].update(is_win=0, winner=1)
    layout = TABXLayout(9, 1, 0).metadata()
    metadata = dict(layout, task_id="fixture", task_hash="a"*64, bank_hash="b"*64, timeout_rule="hp",
                    opponent="advanced", episode_limit=512, collection_phase="expert_collect",
                    provenance=provenance(), teacher="advanced", teacher_epsilon=.1)
    shard = write_shard(tmp_path / "0000000011.h5", [first, second], metadata)
    atomic_json(tmp_path / "manifest.json", dict(metadata=metadata, shards=[dict(path=shard.name, sha256=file_sha256(shard))]))
    env = NS(manifest=metadata, rule="hp", episode_limit=512, get_layout_info=lambda: layout)
    pool = DemonstrationPool(tmp_path, env, 7, .75)
    sampled = pool.sample(100)
    assert .65 < float((sampled["transition_mask"].sum(1) == 3).float().mean()) < .85
    assert pool.collection_steps == 4
    demos = pool.sample(1)
    scheme, groups, preprocess = build_batch_schema(dict(n_agents=10, n_actions=8, state_shape=340, obs_shape=378))
    online = EpisodeBatch(scheme, groups, 1, 5, preprocess=preprocess)
    source, _ = episodes_to_batch([second])
    online.update({k: source[k] for k in online.data.transition_data}, ts=slice(0, 2), mark_filled=False)
    mixed = mix_batches(demos, online)
    assert mixed.batch_size == 2 and mixed.max_seq_length == 5
    torch.testing.assert_close(mixed["actions_onehot"][:1, :demos.max_seq_length], demos["actions_onehot"])
    args = NS(**config(tabx_run_dir=str(tmp_path), tabx_bc_steps=2, tabx_bc_batch_size=2))
    mac = BasicMAC(mixed.scheme, mixed.groups, args)
    learner = QLearner(mac, mixed.scheme, QuietLogger(), args)
    before = [p.detach().clone() for p in mac.parameters()]
    pretrain(learner, pool, args, QuietLogger())
    assert learner.training_steps == 0
    assert any(not torch.equal(a, b) for a, b in zip(before, mac.parameters()))
    for actual, target in zip(mac.parameters(), learner.target_mac.parameters()):
        torch.testing.assert_close(actual, target)
    learner.train(mixed, 4, 1, demo_batch=demos, bc_weight=.5)
    assert learner.training_steps == 1
    assert all(torch.isfinite(p).all() for p in learner.params)
    bad_env = NS(**vars(env))
    bad_env.rule = "loss"
    with pytest.raises(ValueError, match="timeout_rule"):
        DemonstrationPool(tmp_path, bad_env, 7)
    manifest = read_json(tmp_path / "manifest.json")
    legacy = dict(manifest, metadata=dict(metadata, opponent="medium"))
    atomic_json(tmp_path / "manifest.json", legacy)
    with pytest.raises(ValueError, match="opponent"):
        DemonstrationPool(tmp_path, env, 7)
    atomic_json(tmp_path / "manifest.json", manifest)
    losing = tmp_path / "losing"
    losing_shard = write_shard(losing / "0000000012.h5", [second], metadata)
    atomic_json(losing / "manifest.json", dict(metadata=metadata, shards=[
        dict(path=losing_shard.name, sha256=file_sha256(losing_shard))]))
    with pytest.raises(ValueError, match="zero wins"):
        DemonstrationPool(losing, env, 7)
    assert not DemonstrationPool(losing, env, 7, require_wins=False).wins
    with shard.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        DemonstrationPool(tmp_path, env, 7)


@pytest.mark.skipif(os.environ.get("TABX_BC_SMOKE") != "1", reason="Opt in with TABX_BC_SMOKE=1")
def test_official_teacher_bc_and_online_handoff(tmp_path):
    task = tiny_task()
    task["scenario"]["healths"] = [[10.], [5.]]
    bank = tmp_path / "bank.json"
    atomic_json(bank, dict(schema_version="1.0", tasks=[task]))
    demos, output = tmp_path / "demos", tmp_path / "output"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def execute(arguments, name):
        log = tmp_path / f"{name}.log"
        with log.open("w") as stream:
            result = subprocess.run([sys.executable, *arguments], env=environment, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == 0, log.read_text()[-6000:]
    command = ["src/tabx_pipeline.py", "collect-expert", "--bank", str(bank), "--root", str(demos),
               "--task", "tiny", "--episodes", "2", "--episode-limit", "4", "--shard-episodes", "1"]
    execute(command, "collect")
    execute(command, "collect-resume")
    summary = read_json(demos / "hp/tiny/advanced/teacher_summary.json")
    assert summary["episodes"] == 2 and summary["environment_steps"] == 8
    assert summary["metadata"]["opponent"] == "advanced"
    assert summary["metadata"]["teacher_epsilon"] == pytest.approx(.1)
    execute(["src/main.py", "--config=qmix_tabx", "--env-config=tabx", "with", "env_args.key=tiny",
        f"env_args.task_bank={bank}", f"tabx_output_root={output}", "tabx_run_id=bc-smoke", "seed=7", "use_cuda=False",
        "tabx_bc=True", f"tabx_demo_root={demos}", "tabx_bc_steps=2", "tabx_bc_batch_size=2",
        "tabx_bc_eval_episodes=2", "tabx_bc_require_wins=False", "tabx_demo_decay_steps=8", "tabx_demo_fraction=0.5",
        "t_max=12", "batch_size=2", "buffer_size=4", "env_args.max_episode_steps=4",
        "save_model_interval=8", "tabx_screen_episodes=2", "learner_log_interval=4"], "train")
    run = output / "hp/tiny/7/bc-smoke"
    bc = read_json(run / "bc_evaluations/after_bc_online_step_0.json")
    assert bc["online_t_env"] == 0 and len(bc["teacher_episodes"]) == len(bc["episodes"]) == 2
    assert bc["seeds"] == [r["seed"] for r in bc["teacher_episodes"]]
    assert bc["action_selection"] == "greedy"
    collection_seeds = set()
    for shard in (demos / "hp/tiny/advanced").glob("*.h5"):
        import h5py
        with h5py.File(shard) as data:
            collection_seeds.update(data["episodes/seed"][:])
    assert not collection_seeds.intersection(bc["seeds"])
    assert read_json(run / "bc_evaluations/online_decay_complete.json")["online_t_env"] == 8
    assert read_json(run / "bc_evaluations/online_final.json")["online_t_env"] == 12
    records = [json.loads(line) for line in (run / "training_episodes.jsonl").read_text().splitlines()]
    assert [r["episode_index"] for r in records] == [0, 1, 2]
    assert records[0]["epsilon"] == .1
    assert read_json(run / "checkpoints/0/manifest.json")["policy_stage"] == "after_bc_before_online"
    assert read_json(run / "training_summary.json")["learner_updates"] == 3
    assert read_json(run / "bc_summary.json")["demonstration_environment_steps"] == 8


def test_learning_report_uses_online_budget_and_leaves_missing_metrics_empty(tmp_path):
    from tabx_imitation_reporting import summarize_run
    from tabx_reporting import EVAL_FIELDS, write_csv
    atomic_json(tmp_path / "manifest.json", dict(task_id="647", train_seed=1, run_id="test"))
    atomic_json(tmp_path / "config.json", dict(env_args={"max_episode_steps": 512}, tabx_bc=True))
    result = summarize_run(tmp_path, 1500000)
    assert result["bc_win_rate"] is None and result["first_training_win_step"] is None
    assert result["last_5_eval_win_rate"] is None and result["screen_evaluations"] == 0
    rows = [dict(checkpoint_step=t, phase="screen", action_selection="greedy", wins=1 if t else 0,
                 win_rate_pct=10 + i) for i, t in enumerate((0, 25000, 50000, 75000, 100123, 125000, 1500000, 1525000))]
    write_csv(tmp_path / "eval_summary.csv", EVAL_FIELDS, rows)
    result = summarize_run(tmp_path, 1500000)
    assert result["first_greedy_win_step"] == 25000 and result["final_eval_step"] == 1500000
    assert result["early_5_eval_win_rate"] == pytest.approx(.13)
    assert result["last_5_eval_win_rate"] == pytest.approx(.14)
    assert result["at_100k_eval_step"] is None and result["screen_evaluations"] == 7
    assert result["cutoff_step"] == 1500000 and result["budget_reached"] is None


def test_greedy_evaluation_preserves_exploration_rng_and_online_counter(tmp_path, monkeypatch):
    from envs import REGISTRY
    from runners.episode_runner import EpisodeRunner
    from tabx_imitation import evaluate_policy
    from test_tabx import ControlledEnv, ControlledMAC
    monkeypatch.setitem(REGISTRY, "controlled", ControlledEnv)
    args = NS(env="controlled", env_args={"length": 2}, common_reward=True, reward_scalarisation="sum",
              batch_size_run=1, n_agents=10, device="cpu", render=False, test_nepisode=1000,
              runner_log_interval=10, tabx_master_seed=0, tabx_bc_eval_episodes=2, use_cuda=False,
              tabx_run_dir=str(tmp_path))
    runner = EpisodeRunner(args, QuietLogger())
    runner.env.manifest = dict(task_hash="fixture")
    scheme, groups, preprocess = build_batch_schema(runner.get_env_info())
    mac = ControlledMAC()
    mac.action_selector = NS(epsilon=.1)
    mac.agent = torch.nn.Linear(1, 1)
    original_select = mac.select_actions
    def select(*positional, **keywords):
        torch.rand(3)
        np.random.rand(3)
        mac.action_selector.epsilon = 0.
        return original_select(*positional, **keywords)
    mac.select_actions = select
    runner.setup(scheme, groups, preprocess, mac)
    runner.t_env = 123
    torch.manual_seed(7)
    np.random.seed(7)
    torch_before, numpy_before = torch.get_rng_state(), np.random.get_state()
    mac.action_selector.epsilon = .1
    result = evaluate_policy(runner, NS(seeds=set()), args, stage="test")
    torch.testing.assert_close(torch.get_rng_state(), torch_before)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
    assert runner.t_env == 123 and runner.env.episode_seed == result["seeds"][-1]
    assert mac.action_selector.epsilon == .1 and mac.agent.training
