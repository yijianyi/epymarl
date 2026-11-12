"""Budget boundaries, atomic expert recovery, loss semantics and effective configs."""
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

from tabx_config import configure_bc
from tabx_imitation_reporting import summarize_run
from tabx_reporting import EVAL_FIELDS, TRAIN_FIELDS, append_train_metrics, read_csv, train_line, write_csv
from tabx_storage import atomic_json, read_json


@pytest.mark.parametrize("endpoints,wins,cutoff,reached,first_win", [
    ([99900, 100000, 100150], [100000], 100000, True, 100000),
    ([99900, 100020, 100150], [100020], 100020, True, 100020),
    ([99900, 100020, 100150], [100150], 100020, True, None),
    ([99000, 99900], [99000], 99900, False, 99000),
    (None, [], 100000, None, None),
    ([], [], 0, False, None),
])
def test_budget_cutoff_is_the_first_reaching_episode(tmp_path, endpoints, wins, cutoff, reached, first_win):
    atomic_json(tmp_path / "manifest.json", dict(task_id="647", train_seed=1, run_id="test"))
    atomic_json(tmp_path / "config.json", dict(env_args=dict(max_episode_steps=512), tabx_bc=True))
    if endpoints is not None:
        (tmp_path / "training_episodes.jsonl").write_text(''.join(
            json.dumps(dict(t_env=step, is_win=int(step in wins)))+'\n' for step in endpoints))
    # Include evaluations on the boundary, after it, and beyond training progress.
    rows = [dict(checkpoint_step=step, phase="screen", action_selection="greedy", wins=int(step in wins),
                 win_rate_pct=100 if step in wins else 0) for step in (0, 99000, 99900, 100000, 100020, 100150)]
    write_csv(tmp_path / "eval_summary.csv", EVAL_FIELDS, rows)
    result = summarize_run(tmp_path, 100000)
    assert result["online_budget"] == 100000
    assert result["cutoff_step"] == cutoff and result["budget_reached"] is reached
    assert result["first_training_win_step"] == first_win
    included = [row for row in rows if row["checkpoint_step"] <= cutoff]
    assert result["screen_evaluations"] == len(included)
    online = [row["checkpoint_step"] for row in included if row["checkpoint_step"] > 0]
    assert result["final_eval_step"] == (max(online) if online else None)
    if wins == [100150]:
        assert result["first_greedy_win_step"] is None
        assert result["at_100k_eval_step"] <= 100020
    # A pending cutoff evaluation is absent, not a zero-win observation.
    write_csv(tmp_path / "eval_summary.csv", EVAL_FIELDS, [row for row in rows if row["checkpoint_step"] < 99900])
    pending = summarize_run(tmp_path, 100000)
    assert pending["last_5_eval_win_rate"] is None
    assert pending["at_100k_eval_win_rate"] is None


def test_effective_config_is_stdlib_only_idempotent_and_preserves_overrides():
    from tabx_scheduler import training_config
    settings = ["tabx_bc=True", "tabx_demo_root=demos", "tabx_bc_epsilon_start=0.2",
                "tabx_bc_epsilon_finish=0.03", "tabx_demo_decay_steps=54321", "epsilon_start=0.9"]
    config, overrides = training_config(settings)
    assert [config[k] for k in ("epsilon_start", "epsilon_finish", "epsilon_anneal_time")] == [.2, .03, 54321]
    assert overrides["epsilon_start"] == .9
    before = deepcopy(config)
    assert configure_bc(config) == configure_bc(config) == before
    # Import both the shared function and scheduler without optional ML libraries.
    code = """
import builtins,sys
sys.path.insert(0,'src')
original = builtins.__import__
blocked = {'torch','jax','h5py','numpy'}
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('heavy dependency: '+name)
    return original(name,*args,**kwargs)
builtins.__import__ = guarded
from tabx_config import configure_bc
blocked.remove('numpy')  # The scheduler already uses NumPy for TABX layout metadata.
from tabx_scheduler import training_config
config,_ = training_config(['tabx_bc=True','tabx_demo_root=demos'])
assert config['epsilon_start']==.1 and config['epsilon_anneal_time']==100000
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.fixture
def expert_collection(tmp_path, monkeypatch):
    import tabx_expert
    import envs.tabx_wrapper
    from envs.tabx_layout import TABXLayout
    from tabx_pipeline import parser
    from tabx_storage import stable_seed
    from test_tabx import ControlledEnv
    calls = []
    class Env(ControlledEnv):
        def __init__(self, key, timeout_rule, **kwargs):
            super().__init__(length=3)
            self.rule = timeout_rule
            self.manifest = dict(task_id=key, task_hash="a"*64, bank_hash="b"*64, **TABXLayout(9, 1, 0).metadata())
        def get_layout_info(self):
            return TABXLayout(9, 1, 0).metadata()
        def step(self, actions):
            obs, reward, terminated, truncated, info = super().step(actions)
            if self.episode_seed == stable_seed(self.manifest["task_hash"], 0, "expert_collect", 0):
                info["is_win"] = 0
            return obs, reward, terminated, truncated, info
    class Teacher:
        epsilon = .1
        def __init__(self, env, *args):
            self.env = env
        def reset(self, seed):
            calls.append(seed)
        def actions(self):
            return self.env.get_avail_actions().argmax(-1)
        def metadata(self):
            return dict(teacher="advanced", teacher_epsilon=.1, opponent="advanced")
    monkeypatch.setattr(envs.tabx_wrapper, "TABXWrapper", Env)
    monkeypatch.setattr(tabx_expert, "HeuristicTeacher", Teacher)
    args = parser().parse_args(["collect-expert", "--bank", "unused", "--root", str(tmp_path), "--task", "647",
                                "--episodes", "2", "--shard-episodes", "1"])
    tabx_expert.collect_expert(args)
    target = tmp_path / "hp/647/advanced"
    return NS(args=args, target=target, calls=calls, env=Env("647", "hp"), teacher=Teacher)


def test_expert_resume_ignores_temporary_recovers_unlisted_and_rejects_corruption(expert_collection):
    import tabx_expert
    from tabx_dataset import EpisodeRecorder, write_shard
    from tabx_expert_shards import published_expert_shards
    from tabx_imitation import DemonstrationPool
    from tabx_storage import stable_seed
    fixture = expert_collection
    args, target, calls, env = fixture.args, fixture.target, fixture.calls, fixture.env
    original_entries = read_json(target / "manifest.json")["shards"]
    (target / ".shard-interrupted.h5").touch()
    (target / "unpublished.h5").touch()
    calls.clear()
    tabx_expert.collect_expert(args)
    assert calls == []
    assert len(DemonstrationPool(target, env, 7).entries) == 2
    manifest = read_json(target / "manifest.json")
    seed = stable_seed(env.manifest["task_hash"], args.seed, "expert_collect", 2)
    recorder = EpisodeRecorder()
    tabx_expert.teacher_rollout(env, fixture.teacher(env), seed, recorder)
    unlisted = write_shard(target / f"{seed:010d}.h5", [recorder.episode], manifest["metadata"])
    contents = unlisted.read_bytes()
    with pytest.raises(ValueError, match="rerun collect-expert"):
        DemonstrationPool(target, env, 7)
    args.episodes = 3
    calls.clear()
    tabx_expert.collect_expert(args)
    assert calls == [] and unlisted.read_bytes() == contents
    recovered_entries = read_json(target / "manifest.json")["shards"]
    assert len(recovered_entries) == 3
    assert all(entry in recovered_entries for entry in original_entries)
    assert len(DemonstrationPool(target, env, 7).entries) == 3
    tabx_expert.collect_expert(args)
    assert calls == [] and read_json(target / "manifest.json")["shards"] == recovered_entries
    assert all(len(p.name) == 13 for p in published_expert_shards(target))
    # A formally published name is verified even when absent from the manifest.
    (target / "0000000000.h5").write_bytes(b"broken")
    with pytest.raises((OSError, ValueError)):
        tabx_expert.collect_expert(args)


@pytest.mark.parametrize("damage", ["readable_content_change", "missing_registered_file"])
def test_expert_resume_preserves_checksum_authority_and_json_on_failure(expert_collection, damage):
    import h5py
    import tabx_expert
    from tabx_dataset import read_episode, verify_shard
    from tabx_storage import file_sha256
    fixture = expert_collection
    target = fixture.target
    manifest_path, summary_path = target / "manifest.json", target / "teacher_summary.json"
    original_json = {path: path.read_bytes() for path in (manifest_path, summary_path)}
    entry = next(entry for entry in read_json(manifest_path)["shards"]
                 if not read_episode(target / entry["path"], 0)["episode"]["is_win"])
    shard = target / entry["path"]
    if damage == "readable_content_change":
        # Alter only outcome metadata: the HDF5 and trajectory validation still pass.
        with h5py.File(shard, "r+") as data:
            data["episodes/is_win"][0] = 1
            data["episodes/winner"][0] = 0
        verify_shard(shard)
        assert file_sha256(shard) != entry["sha256"]
        message = "SHA-256 mismatch"
    else:
        shard.rename(target / (shard.name + ".moved"))
        message = "Missing registered expert shard"
    # There is also a requested new episode: validation must fail before sampling it.
    fixture.args.episodes = 3
    fixture.calls.clear()
    with pytest.raises(ValueError, match=message) as error:
        tabx_expert.collect_expert(fixture.args)
    assert entry["path"] in str(error.value)
    assert fixture.calls == []
    assert {path: path.read_bytes() for path in original_json} == original_json


@pytest.mark.parametrize("use_bc", [False, True])
def test_loss_columns_and_parameter_updates_match_previous_combined_loss(tmp_path, use_bc):
    from learners.q_learner import QLearner
    from tabx_imitation import behavior_cloning_loss
    # Give one action a distinct score so BC has a nonzero, measurable gradient.
    class MAC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.3))
        def init_hidden(self, batch_size):
            pass
        def forward(self, batch, t):
            return self.weight * torch.arange(8.).expand(batch.batch_size, 10, 8)
    values = dict(reward=torch.ones(1, 3, 1), actions=torch.zeros(1, 3, 10, 1, dtype=torch.long),
                  terminated=torch.tensor([[[0],[1],[0]]]), filled=torch.ones(1, 3, 1),
                  avail_actions=torch.ones(1, 3, 10, 8), transition_mask=torch.tensor([[[1],[1],[0]]]),
                  present=torch.ones(1, 3, 10, 1), alive=torch.ones(1, 3, 10, 1))
    values["actions"][:] = 1
    class Batch:
        batch_size, max_seq_length = 1, 3
        def __getitem__(self, key):
            return values[key]
    args = NS(n_agents=10, mixer=None, lr=.01, learner_log_interval=1, use_cuda=False,
              standardise_returns=False, standardise_rewards=False, common_reward=True, double_q=True,
              gamma=0., grad_norm_clip=10, target_update_interval_or_tau=200, env="tabx", tabx_run_dir=str(tmp_path))
    log = {}
    learner = QLearner(MAC(), {}, NS(log_stat=lambda k,v,t: log.update({k:v})), args)
    reference = MAC()
    optimiser = torch.optim.Adam(reference.parameters(), lr=args.lr)
    outputs = torch.stack([reference.forward(Batch(), t) for t in range(3)], dim=1)
    chosen = outputs[:, :-1].gather(3, values["actions"][:, :-1]).squeeze(3)
    mask = values["filled"][:, :-1].expand_as(chosen)
    old_loss = (((chosen - values["reward"][:, :-1]) * mask)**2).sum() / mask.sum()
    weight = .4 if use_bc else 0.
    if use_bc:
        old_bc, _ = behavior_cloning_loss(outputs, Batch())
        old_loss = old_loss + weight * old_bc
    old_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(reference.parameters()), args.grad_norm_clip)
    optimiser.step()
    learner.train(Batch(), 2, 1, demo_batch=Batch() if use_bc else None, bc_weight=weight)
    row = read_csv(tmp_path / "train_metrics.csv")[0]
    assert float(row["total_loss"]) == pytest.approx(float(row["loss"]) + weight * float(row["bc_loss"] or 0))
    assert float(row["total_loss"]) == pytest.approx(old_loss.item())
    assert float(row["bc_weight"]) == weight
    assert bool(row["bc_loss"]) is use_bc
    if not use_bc:
        assert row["total_loss"] == row["loss"]
    torch.testing.assert_close(learner.mac.weight, reference.weight, rtol=0, atol=0)
    torch.testing.assert_close(learner.mac.weight.grad, reference.weight.grad, rtol=0, atol=0)
    line = train_line("hp", "647", 1, "0", row)
    assert "total_loss=" in line and "bc_weight=" in line


def test_historical_loss_is_not_reinterpreted_or_rewritten(tmp_path):
    legacy = dict(t_env=4, learner_updates=1, loss=3., td_error_abs=1., grad_norm_before_clip=2.,
                  grad_norm_clip=10, q_taken_mean=.1, target_mean=.2, lr=.001)
    path = tmp_path / "train_metrics.csv"
    write_csv(path, TRAIN_FIELDS[:9], [legacy])
    before = path.read_bytes()
    assert read_csv(path)[0]["loss"] == "3.0"
    assert "total_loss=" not in train_line("hp", "647", 1, "0", read_csv(path)[0])
    with pytest.raises(ValueError, match="historical CSV"):
        append_train_metrics(path, legacy)
    assert path.read_bytes() == before


@pytest.fixture(scope="module")
def official_bc_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("bc-config-inputs")
    from test_tabx_official import tiny_task
    bank = root / "bank.json"
    atomic_json(bank, dict(schema_version="1.0", tasks=[tiny_task()]))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log = root / "collect.log"
    with log.open("w") as stream:
        result = subprocess.run([sys.executable, "src/tabx_pipeline.py", "collect-expert", "--bank", str(bank),
            "--root", str(root / "demos"), "--task", "tiny", "--episodes", "2", "--episode-limit", "4"],
            env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, log.read_text()[-5000:]
    return root, bank


@pytest.mark.skipif(os.environ.get("TABX_BC_SMOKE") != "1", reason="Opt in with TABX_BC_SMOKE=1")
@pytest.mark.parametrize("variant,expected", [
    ("baseline", (1., .05, 80000)), ("bc-default", (.1, .05, 100000)), ("bc-custom", (.2, .03, 54321))])
def test_real_batch_run_and_selector_share_effective_config(tmp_path, monkeypatch, official_bc_inputs, variant, expected):
    import tabx_scheduler
    from tabx_pipeline import parser
    from components.action_selectors import EpsilonGreedyActionSelector
    inputs, bank = official_bc_inputs
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(tabx_scheduler, "probe_gpus", lambda gpus: {gpu: dict(name="CPU-smoke-only") for gpu in gpus})
    original = tabx_scheduler.BatchScheduler.training_command
    def cpu_command(self, job, attempt):
        return ["use_cuda=False" if part == "use_cuda=True" else
                "tabx_require_cuda=False" if part == "tabx_require_cuda=True" else part
                for part in original(self, job, attempt)]
    monkeypatch.setattr(tabx_scheduler.BatchScheduler, "training_command", cpu_command)
    arguments = ["batch-train", "--bank", str(bank), "--root", str(tmp_path / "output"), "--rule", "hp",
                  "--tasks", "tiny", "--seeds", "7", "--gpus", "0", "--eval-workers", "1", "--batch-id", variant,
                  "--poll-seconds", "0.05"]
    settings = ["t_max=16", "batch_size=4", "buffer_size=4", "env_args.max_episode_steps=4",
                "save_model_interval=16", "tabx_screen_episodes=2", "learner_log_interval=4"]
    if variant != "baseline":
        settings += ["tabx_bc=True", f"tabx_demo_root={inputs / 'demos'}", "tabx_bc_steps=1",
                     "tabx_bc_batch_size=2", "tabx_bc_eval_episodes=2", "tabx_bc_require_wins=False"]
    if variant == "bc-custom":
        settings += ["tabx_bc_epsilon_start=0.2", "tabx_bc_epsilon_finish=0.03", "tabx_demo_decay_steps=54321",
                     "epsilon_start=0.9"]
    for setting in settings:
        arguments += ["--set", setting]
    assert tabx_scheduler.batch_train(parser().parse_args(arguments)) == 0
    batch_dir = tmp_path / "output/_batches/hp" / variant
    batch = read_json(batch_dir / "batch.json")
    state = read_json(batch_dir / "state.json")
    run = tmp_path / "output" / state["jobs"][0]["attempts"][0]["run"]
    run_config = read_json(run / "config.json")
    keys = ("epsilon_start", "epsilon_finish", "epsilon_anneal_time")
    assert tuple(batch["training_config"][key] for key in keys) == expected
    assert tuple(batch["identity"]["training_config"][key] for key in keys) == expected
    assert tuple(run_config[key] for key in keys) == expected
    selector = EpsilonGreedyActionSelector(NS(**run_config))
    assert selector.schedule.eval(0) == expected[0]
    assert selector.schedule.eval(expected[2]) == pytest.approx(expected[1])
    rows = [json.loads(line) for line in (run / "training_episodes.jsonl").read_text().splitlines()]
    for row in rows:
        assert row["epsilon"] == pytest.approx(selector.schedule.eval(row["t_env"] - row["length"]))
    if variant == "bc-custom":
        assert batch["overrides"]["epsilon_start"] == .9
    metrics = read_csv(run / "train_metrics.csv")
    assert metrics
    for row in metrics:
        assert float(row["total_loss"]) == pytest.approx(
            float(row["loss"]) + float(row["bc_weight"]) * float(row["bc_loss"] or 0))
