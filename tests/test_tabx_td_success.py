"""Successful original-teacher episodes supply TD only after legacy sampling exits."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tabx_config import configure_bc, success_td_count
from tabx_dataset import episodes_to_batch, read_episode, write_shard
from tabx_imitation import AggregatedDemonstrationPool, DemonstrationPool, demonstration_schedule, mix_batches
from tabx_reporting import read_csv
from tabx_scheduler import training_config
from tabx_storage import atomic_json, file_sha256, provenance, read_json
from tabx_td_success import SuccessTDPool, td_batch_counts, td_source_metrics
from test_tabx import controlled_rollout
from test_tabx_imitation import config


@pytest.mark.parametrize("fraction", [-.1, 1., float("nan"), float("inf"), True, "0.25"])
def test_invalid_success_fraction(fraction):
    with pytest.raises(ValueError, match="tabx_td_success_fraction"):
        configure_bc(config(tabx_td_success_fraction=fraction))


def test_counts_gate_and_stdlib_config():
    for fraction, expected in ((0., 0), (.25, 8), (.0001, 1), (.999, 31)):
        values = config(tabx_bc=True, tabx_demo_root="demos", tabx_td_success_fraction=fraction)
        assert success_td_count(values) == expected
        configure_bc(values)
        assert configure_bc(deepcopy(values)) == values
    with pytest.raises(ValueError, match="tabx_bc=True"):
        configure_bc(config(tabx_td_success_fraction=.25))
    with pytest.raises(ValueError, match="batch_size"):
        configure_bc(config(tabx_bc=True, tabx_demo_root="demos", batch_size=1, tabx_td_success_fraction=.25))
    enabled = NS(**config(tabx_bc=True, tabx_demo_root="demos", tabx_td_success_fraction=.25))
    control = NS(**dict(vars(enabled), tabx_td_success_fraction=0.))
    for step in (0, 50000, 93749, 93750, 100000, 300000):
        count, weight = demonstration_schedule(enabled, step)
        legacy, success, online = td_batch_counts(enabled, count)
        assert legacy + success + online == 32 and online >= 1
        if count:
            assert td_batch_counts(control, count) == (legacy, success, online)
        else:
            assert (legacy, success, online) == (0, 8, 24)
            assert td_batch_counts(control, count) == (0, 0, 32)
        assert demonstration_schedule(control, step) == (count, weight)
    # The pure validation path must also work without ML/dataset libraries.
    code = """
import builtins,sys
sys.path.insert(0,'src')
old=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','jax','h5py','numpy'}: raise AssertionError(name)
    return old(name,*args,**kwargs)
builtins.__import__=guard
from tabx_config import success_td_count
assert success_td_count(dict(tabx_bc=True,batch_size=32,tabx_td_success_fraction=.25))==8
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.fixture
def teacher_pool(tmp_path, monkeypatch):
    from envs.tabx_layout import TABXLayout
    episodes = [controlled_rollout(monkeypatch, length, 100+i)[1] for i, length in enumerate((1, 2, 4))]
    episodes[0]["episode"]["end_reason"] = 1  # Elimination and timeout wins both count.
    episodes[-1]["episode"].update(is_win=0, winner=1)
    for episode in episodes:
        actual = episode["transitions"]["actions"]
        labels = actual.copy()
        labels[:, :9] = 5
        episode["transitions"].update(teacher_actions=labels,
            teacher_mask=episode["frames"]["alive"][:-1, :, None].copy())
    metadata = dict(TABXLayout(9, 1, 0).metadata(), task_id="fixture", task_hash="a"*64, bank_hash="b"*64,
        timeout_rule="hp", opponent="advanced", episode_limit=512, collection_phase="expert_collect",
        provenance=provenance(), teacher="advanced", teacher_epsilon=.1, label_source="deterministic_teacher", label_epsilon=0.)
    path = write_shard(tmp_path / "0000000100.h5", episodes, metadata)
    atomic_json(tmp_path / "manifest.json", dict(metadata=metadata, shards=[dict(path=path.name, sha256=file_sha256(path))]))
    env = NS(manifest=metadata, rule="hp", episode_limit=512, get_layout_info=lambda: TABXLayout(9, 1, 0).metadata())
    return DemonstrationPool(tmp_path, env, 7, target="teacher_actions"), episodes


def test_success_sampling_preserves_legacy_rng_and_actual_complete_episodes(teacher_pool, tmp_path):
    pool, episodes = teacher_pool
    aggregate = AggregatedDemonstrationPool(pool, 7)
    original_state, aggregate_state = deepcopy(pool.rng.bit_generator.state), deepcopy(aggregate.rng.bit_generator.state)
    numpy_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
    success = SuccessTDPool(aggregate, "a"*64, 7)
    mirror = SuccessTDPool(aggregate, "a"*64, 7)
    batch = success.sample(100)
    torch.testing.assert_close(batch["actions"], mirror.sample(100)["actions"])
    assert pool.rng.bit_generator.state == original_state and aggregate.rng.bit_generator.state == aggregate_state
    assert torch.equal(torch_state, torch.get_rng_state())
    np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
    # Short one-step elimination and two-step timeout wins; no four-step losing trajectory.
    lengths = batch["transition_mask"].sum(1).flatten()
    assert set(lengths.tolist()) == {1, 2}
    for i, length in enumerate(lengths.tolist()):
        actual = episodes[int(length)-1]
        np.testing.assert_array_equal(batch["actions"][i, :int(length)], actual["transitions"]["actions"])
        np.testing.assert_array_equal(batch["reward"][i, :int(length)], actual["transitions"]["reward"])
        assert batch["filled"][i].sum() == length+1
        assert (batch["teacher_actions"][i, :int(length), :9] == 5).all()
    success.publish(tmp_path)
    success.activate(93801)
    success.activate(100003)
    saved = read_json(tmp_path / "td_success_pool.json")
    assert saved["pool_size"] == 2 and saved["activation_step"] == 93801
    assert saved["manifest_sha256"] == file_sha256(pool.directory / "manifest.json")
    # Legacy stream is byte-for-byte unchanged after independent sampling.
    after = pool.sample(8)
    pool.rng.bit_generator.state = original_state
    before = pool.sample(8)
    for key in after.data.transition_data:
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
    empty = NS(original=NS(wins=[]))
    with pytest.raises(ValueError, match="original teacher pool"):
        SuccessTDPool(empty, "a"*64, 7)


def test_group_diagnostics_use_one_transition_average_and_exclude_padding():
    error = torch.tensor([[[2.], [100.], [100.]], [[3.], [3.], [3.]]], requires_grad=True)
    mask = torch.tensor([[[1.], [0.], [0.]], [[1.], [1.], [1.]]])
    result = td_source_metrics(error, error+1, error-1, mask, (0, 1, 1))
    assert result["success_td_loss"] == 4 and result["online_loss"] == 9
    assert result["success_td_episode_fraction"] == .5 and result["success_td_transition_fraction"] == .25
    assert result["success_td_q_taken_mean"] == 3 and result["online_target_mean"] == 2
    assert result["legacy_demo_loss"] is None
    assert (result["success_td_loss"]*1 + result["online_loss"]*3)/4 == 7.75
    assert error.grad is None


@pytest.mark.parametrize("with_bc", [False, True])
def test_source_diagnostics_do_not_change_updates_and_success_labels_never_enter_bc(teacher_pool, tmp_path, with_bc):
    from controllers.basic_controller import BasicMAC
    from learners.q_learner import QLearner
    from tabx_pipeline import QuietLogger
    pool, episodes = teacher_pool
    success = SuccessTDPool(pool, "a"*64, 7).sample(8)
    online = episodes_to_batch([episodes[-1]] * 24)[0]
    batch = mix_batches(success, online)
    args = NS(**config(tabx_run_dir=str(tmp_path), tabx_bc_target="teacher_actions"))
    learner = QLearner(BasicMAC(batch.scheme, batch.groups, args), batch.scheme, QuietLogger(), args)
    # The next-step RNN history uses executed movement; labels suggest turning.
    inputs = learner.mac._build_inputs(batch, 1).reshape(32, 10, -1)
    torch.testing.assert_close(inputs[:, :, 378:386], batch["actions_onehot"][:, 0])
    assert (batch["actions"][:8, 0, :9] != success["teacher_actions"][:, 0, :9]).all()
    reference = deepcopy(learner)
    # A separate legacy prefix is the only batch allowed to receive BC.
    if with_bc:
        legacy = pool.sample(2)
        batch = mix_batches(legacy, batch)
        counts = (2, 8, 24)
    else:
        legacy, counts = None, (0, 8, 24)
    learner.train(batch, 100001, 32, demo_batch=legacy, bc_weight=.5, td_counts=counts)
    altered = deepcopy(batch)
    start = counts[0]
    altered["teacher_actions"][start:start+8, :, :9] = 6
    reference.train(altered, 100001, 32, demo_batch=legacy, bc_weight=.5)
    for actual, expected in zip(learner.params, reference.params):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    row = read_csv(tmp_path / "train_metrics.csv")[0]
    metrics = json.loads((tmp_path / "td_source_metrics.jsonl").read_text())
    assert metrics["success_td_episodes"] == 8 and metrics["online_episodes"] == 24
    assert metrics["bc_episodes"] == (2 if with_bc else 0)
    assert metrics["bc_weight"] == (.5 if with_bc else 0)
    if not with_bc:
        assert row["bc_loss"] == "" and row["loss"] == row["total_loss"]
    reconstructed = sum(metrics[f"{name}_loss"] * metrics[f"{name}_transitions"]
                        for name in ("legacy_demo", "success_td", "online") if metrics[f"{name}_transitions"])
    assert float(row["loss"]) == pytest.approx(reconstructed / sum(metrics[f"{name}_transitions"]
        for name in ("legacy_demo", "success_td", "online")), rel=1e-6)


@pytest.mark.skipif(os.environ.get("TABX_TD_SUCCESS_SMOKE") != "1", reason="Opt in with TABX_TD_SUCCESS_SMOKE=1")
def test_real_paired_training_sealing_recovery_and_pre_activation_equivalence(tmp_path, monkeypatch):
    from tabx_pipeline import parser
    from tabx_gamma_experiment import gamma_pair, make_pair_plan
    import tabx_scheduler
    from test_tabx_official import tiny_task
    task = tiny_task()
    task["scenario"].update(healths=[[100.], [1.]], attack_damages=[[100.], [0.]],
        sight_angles=[[3.14], [3.14]], attack_cooldowns=[[.1], [.1]])
    bank, demos, output = tmp_path / "bank.json", tmp_path / "demos", tmp_path / "source"
    atomic_json(bank, dict(schema_version="1.0", tasks=[task]))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    def run(command, name):
        log = tmp_path / f"{name}.log"
        with log.open("w") as stream:
            result = subprocess.run([sys.executable, *command], env=environment, stdout=stream,
                stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == 0, log.read_text()[-6000:]
    run(["src/tabx_pipeline.py", "collect-expert", "--bank", str(bank), "--root", str(demos), "--task", "tiny",
         "--episodes", "4", "--episode-limit", "4", "--epsilon", "0"], "collect")
    assert read_json(demos / "hp/tiny/advanced/teacher_summary.json")["wins"] > 0
    run(["src/main.py", "--config=qmix_tabx", "--env-config=tabx", "with", "env_args.key=tiny",
        f"env_args.task_bank={bank}", f"tabx_output_root={output}", "tabx_run_id=source", "seed=7", "use_cuda=False",
        "tabx_bc=True", f"tabx_demo_root={demos}", "tabx_bc_steps=4", "tabx_bc_batch_size=2", "tabx_bc_eval_episodes=2",
        "tabx_dagger_episodes=1", "env_args.max_episode_steps=4", "batch_size=32", "buffer_size=64",
        "tabx_demo_decay_steps=192", "save_model_interval=64", "learner_log_interval=4", "t_max=0"], "pretrain")
    source = output / "hp/tiny/7/source/checkpoints/0"
    arguments = ["td-success-pair", "--source", str(source), "--bank", str(bank), "--task", "tiny", "--seed", "7",
        "--root", str(tmp_path / "pair"), "--pair-id", "test", "--budget", "224", "--eval-workers", "1"]
    args = parser().parse_args(arguments)
    plan = make_pair_plan(args)
    first, second = plan["training_configs"]
    assert [k for k in first if first[k] != second[k]] == ["tabx_td_success_fraction"]
    assert first["gamma"] == second["gamma"] == .99
    assert (first["tabx_td_success_fraction"], second["tabx_td_success_fraction"]) == (0., .25)
    assert second["batch_size"] == 32 and second["tabx_dagger_rounds"] == 3
    assert len(plan["source_pool_inventory"]["pools"]) == 4
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(tabx_scheduler, "probe_gpus", lambda gpus: {gpu: dict(name="CPU-smoke-only") for gpu in gpus})
    original = tabx_scheduler.BatchScheduler.training_command
    def cpu_command(self, job, attempt):
        return ["use_cuda=False" if x == "use_cuda=True" else "tabx_require_cuda=False" if x == "tabx_require_cuda=True" else x
                for x in original(self, job, attempt)]
    monkeypatch.setattr(tabx_scheduler.BatchScheduler, "training_command", cpu_command)
    gamma_pair(parser().parse_args(arguments + ["--plan-only"]))
    directory = tmp_path / "pair/_td_success_pairs/test"
    sealed = (directory / "pair.json").read_bytes()
    gamma_pair(args)
    report = read_json(directory / "summary.json")
    assert report["treatment"] == "tabx_td_success_fraction_only"
    runs = [Path(result["run_directory"]) for result in report["results"]]
    starts = [read_json(run / "online_start.json") for run in runs]
    for key in ("agent_parameter_sha256", "mixer_parameter_sha256", "source_pool_sha256"):
        assert starts[0][key] == starts[1][key]
    for run in runs:
        assert not (run / "dagger").exists() and not (run / "bc_metrics.csv").exists()
        assert read_json(run / "bc_summary.json")["bc_updates_this_run"] == 0
        for path in (run / "evaluations").glob("*.json"):
            assert read_json(path)["request"]["seeds"] == plan["evaluation_seeds"]
    metrics = [[json.loads(line) for line in (run / "td_source_metrics.jsonl").read_text().splitlines()] for run in runs]
    activated = read_json(runs[1] / "td_success_pool.json")["activation_step"]
    assert activated >= 180
    assert [row for row in metrics[0] if row["t_env"] < activated] == [row for row in metrics[1] if row["t_env"] < activated]
    assert any(row["t_env"] < activated for row in metrics[1])
    assert all(row["success_td_episodes"] == 0 for row in metrics[0])
    later = [row for row in metrics[1] if row["t_env"] >= activated]
    assert later and all((row["legacy_demo_episodes"], row["success_td_episodes"], row["online_episodes"],
        row["bc_episodes"], row["bc_weight"]) == (0, 8, 24, 0, 0.) for row in later)
    assert all(row["loss"] == row["total_loss"] for row in later)
    # Actual model/optimizer paths and exploration trajectories match up to the gate.
    checkpoints = [{p.name: p for p in (run / "checkpoints").iterdir() if int(p.name) < activated} for run in runs]
    assert checkpoints[0].keys() == checkpoints[1].keys()
    for step in checkpoints[0]:
        for name in ("agent.th", "mixer.th", "opt.th"):
            assert file_sha256(checkpoints[0][step] / name) == file_sha256(checkpoints[1][step] / name)
    episodes = [[json.loads(line) for line in (run / "training_episodes.jsonl").read_text().splitlines()] for run in runs]
    for left, right in zip(episodes[0], episodes[1]):
        if left["t_env"] >= activated:
            break
        for key in ("episode_index", "episode_seed", "t_env", "epsilon", "is_win", "return_value", "length"):
            assert left[key] == right[key]
    expected_counts = sum(row["success_td_episodes"] for row in metrics[1])
    assert report["results"][1]["td_sampling"]["sampled_episodes"]["success_td"] == expected_counts
    assert (directory / "pair.json").read_bytes() == sealed
    gamma_pair(parser().parse_args(arguments + ["--report-only"]))
    gamma_pair(parser().parse_args(arguments + ["--resume"]))
    assert (directory / "pair.json").read_bytes() == sealed
    assert read_json(directory / "summary.json")["results"][1]["td_sampling"]["sampled_episodes"]["success_td"] == expected_counts
    manifest_path = demos / "hp/tiny/advanced/manifest.json"
    contents = manifest_path.read_bytes()
    manifest_path.write_bytes(contents + b" ")
    with pytest.raises(ValueError, match="manifest differs"):
        make_pair_plan(args)
    manifest_path.write_bytes(contents)
