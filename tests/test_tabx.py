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

from components.batch_schema import build_batch_schema
from envs.tabx_outcome import adjudicate, replace_terminal_reward
from envs.tabx_tasks import load_task, task_hash
from envs.tabx_layout import TABXLayout, LAYOUT_VERSION, SIMULATOR_CONTRACT
from tabx_dataset import EpisodeRecorder, read_episode, read_episode_batch, verify_shard, write_shard
from tabx_storage import EvalQueue, SnapshotPublisher, atomic_json, read_json, stable_seed


@pytest.mark.parametrize("rule,hp,expected", [(r, hp, win) for r in ("hp", "loss")
                                              for hp, win in ((0.8, r == "hp"), (0.5, False), (0.2, False))])
def test_timeout_rules(rule, hp, expected):
    result = adjudicate(True, True, hp, 0.5, 512, rule=rule)
    assert result.is_win == expected
    assert result.ended and result.reason == 2 and result.hit_time_limit
    assert replace_terminal_reward(0.25 - 1, -1, result) == (0.25 + (1 if expected else -1), 0.25)


@pytest.mark.parametrize("rule", ["hp", "loss"])
@pytest.mark.parametrize("ally,enemy,t,reason,win", [(True, False, 512, 1, True), (True, False, 7, 1, True),
                                                     (False, True, 7, 1, False), (False, False, 512, 3, False),
                                                     (True, True, 511, 0, False)])
def test_adjudication_priority(rule, ally, enemy, t, reason, win):
    result = adjudicate(ally, enemy, 0.5, 0.9, t, rule=rule)
    assert result.reason == reason and result.is_win == win
    assert result.ended == (reason != 0)


def test_task_identity_and_integrity(tmp_path):
    task = {"task_id": "23", "scenario": {"body_radiuss": [[1]]}, "zone_scenario": {"n_zone": 0}}
    task["metadata"] = {"canonical_hash": "a" * 64}
    path = tmp_path / "bank.json"
    atomic_json(path, dict(schema_version="1.0", tasks=[task]))
    loaded, manifest = load_task(path, 23)
    assert manifest["task_hash"] == task_hash(task)
    assert not manifest["canonical_hash_matches"]
    loaded["scenario"]["body_radiuss"][0][0] = 5
    assert load_task(path, "23")[0] == task
    with pytest.raises(ValueError, match="SHA-256"):
        load_task(path, 23, expected_hash="b" * 64)
    with pytest.raises(KeyError):
        load_task(path, "24")
    atomic_json(path, dict(schema_version="1.0", tasks=[task, task]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_task(path, 23)


def test_shared_pilot_locks_relatives_and_external_split(tmp_path):
    from tabx_pilot import build_pilot
    tasks = [dict(task_id=str(i), scenario=dict(teams=[[0], [1]], marker=i),
                  zone_scenario=dict(n_zone=i % 11), metadata={}) for i in range(40)]
    for task in tasks:
        task["metadata"]["source_namespace"] = "fixture-bank"
    tasks[0]["metadata"].update(parent_ids=["1", "missing-parent"])
    bank = tmp_path / "bank.json"
    atomic_json(bank, dict(schema_version="1.0", tasks=tasks))
    path = tmp_path / "pilot.json"
    pilot = build_pilot(bank, path, seed=1, count=8)
    assert len(pilot["pilot"]) == 8 and set(pilot["pilot"]).issubset(pilot["splits"]["train"])
    assert any(set(group) == {"0", "1"} for group in pilot["lineage_groups"])
    assert pilot["unresolved_lineage"][0]["reference"] == "missing-parent"
    assert pilot["unresolved_lineage"][0]["children"] == ["0"]
    assert build_pilot(bank, path, seed=1, count=8) == pilot
    with pytest.raises(ValueError, match="different inputs"):
        build_pilot(bank, path, seed=2, count=8)
    external = tmp_path / "split.json"
    atomic_json(external, dict(train=[str(i) for i in range(40) if i != 1], validation=[], test=["1"]))
    with pytest.raises(ValueError, match="relatives"):
        build_pilot(bank, tmp_path / "other.json", count=8, split_file=external)
    assert read_json(external)["test"] == ["1"]
    assert read_json(tmp_path / "other.conflicts.json")["conflicts"][0]["task_ids"] == ["0", "1"]
    colliding_input = tmp_path / "keep.conflicts.json"
    atomic_json(colliding_input, read_json(external))
    original_bytes = colliding_input.read_bytes()
    with pytest.raises(ValueError, match="relatives"):
        build_pilot(bank, tmp_path / "keep.json", count=8, split_file=colliding_input)
    assert colliding_input.read_bytes() == original_bytes
    assert (tmp_path / "keep.conflicts-1.json").is_file()


def test_absent_parents_crossover_and_scoped_legacy_names():
    from tabx_pilot import build_families
    def task(key, **metadata):
        return dict(task_id=key, scenario={"marker": key}, zone_scenario={}, metadata=metadata)
    first, second = "a" * 64, "b" * 64
    tasks = {key: task(key) for key in ("a", "b", "c", "d", "e", "f", "g", "h", "parent")}
    tasks["a"]["metadata"] = {"parent_hash": first}
    tasks["b"]["metadata"] = {"parent_hashes": [first, second]}
    tasks["c"]["metadata"] = {"parent_hash": second}
    tasks["d"]["metadata"] = {"source": {"file": "source-A", "parent": "old-1"}}
    tasks["e"]["metadata"] = {"source": {"file": "source-A", "parent": "old-1"}}
    tasks["f"]["metadata"] = {"source": {"file": "source-B", "parent": "old-1"}}
    tasks["g"]["metadata"] = {"parent": "old-1"}
    tasks["h"]["metadata"] = {"parent": {"canonical_hash": first, "task_id": "parent"}}
    result = build_families(tasks)
    assert ["a", "b", "c", "h"] in result["groups"]
    assert ["d", "e"] in result["groups"]
    for key in ("f", "g", "parent"):
        assert [key] in result["groups"]
    assert result["untrusted_lineage"][0]["task_id"] == "g"
    # Adding the missing parent includes it in the already connected family.
    tasks["parent"]["metadata"]["canonical_hash"] = first
    assert ["a", "b", "c", "h", "parent"] in build_families(tasks)["groups"]
    for key, origin in (("x", "bank-A"), ("y", "bank-B")):
        tasks[key] = task(key, source_namespace=origin, merged_from={"file": "tasks.json"}, parent="old-1")
    result = build_families(tasks)
    assert ["x"] in result["groups"] and ["y"] in result["groups"]


def test_timeout_tertiles_use_only_training_and_keep_ties():
    from tabx_pilot import timeout_strata
    rates = [0, 0.01, 0.01, 0.02, 0.03, 0.03, 0.04, 1.0]
    tasks = {str(i): dict(task_id=str(i), metadata={"confirmation": {"original": {"truncation_rate": rate}}})
             for i, rate in enumerate(rates)}
    splits = dict(train=[str(i) for i in range(7)], validation=["7"], test=[])
    groups, strata = timeout_strata(tasks, splits)
    assert strata["positive_tertiles"] == pytest.approx([0.016666666666666666, 0.03])
    assert groups["0"] == "zero" and groups["1"] == groups["2"] == "low"
    assert groups["4"] == groups["5"] == "medium" and groups["6"] == "high"
    tasks["7"]["metadata"]["confirmation"]["original"]["truncation_rate"] = 0.0001
    assert timeout_strata(tasks, splits)[1]["positive_tertiles"] == strata["positive_tertiles"]


@pytest.mark.parametrize("rule,enemy_alive,raw_done,hp,expected", [
    ("hp", True, -1, 0.8, 1.25), ("loss", True, -1, 0.8, -0.75),
    ("hp", False, 0, 0.8, 1.25), ("loss", False, 0, 0.8, 1.25)])
def test_wrapper_uses_new_state_and_replaces_terminal_once(rule, enemy_alive, raw_done, hp, expected):
    from envs.tabx_wrapper import TABXWrapper
    env = TABXWrapper.__new__(TABXWrapper)
    keys = [f"unit_{i:02d}" for i in range(20)]
    env.base_env = NS(unit_keys=keys, ally_keys=keys[:10], win_reward=1, lose_reward=-1)
    env.layout = TABXLayout(10, 10, 10)
    env.simulator = NS(get_avail_actions=lambda state: {k: np.ones(8) for k in keys[:10]})
    env.jax = NS(numpy=np, random=NS(split=lambda key: (key, key)))
    env._episode_key = 0
    env.rule, env.episode_limit, env.t, env.first_kill, env.ended = rule, 512, 511, 0, False
    env.manifest = {"task_id": "23"}
    state = {k: NS(status=NS(is_disabled=np.array([i not in (0, 10)]),
                             is_alive=np.array([i in (0, 10)]), health=np.array([1.0]))) for i, k in enumerate(keys)}
    state["game_manager"] = NS(timestep=np.array([512]), team_hp_ratio=np.array([hp, 0.5]))
    obs = {k: np.zeros(378, dtype=np.float32) for k in keys[:10]}
    obs["world_state"] = np.zeros(340, dtype=np.float32)
    env.env_state = {"state": state}
    env._cache(obs)
    env._previous_alive = env._all_alive.copy()
    next_state = deepcopy(state)
    next_state[keys[10]].status.is_alive = np.array([enemy_alive])
    next_state[keys[10]].status.health = np.array([float(enemy_alive)])
    env._step = lambda *args: (obs, {"state": next_state}, {"__all__": 0.25 + raw_done},
                               {"__all__": False}, {"done_reward": np.array([raw_done, 0])})
    actions = np.full(10, 7)
    _, reward, terminated, truncated, info = env.step(torch.tensor(actions))
    assert reward == expected and terminated and not truncated
    assert info["hit_time_limit"] and not info["episode_limit"]
    assert info["end_reason"] == (2 if enemy_alive else 1)
    assert all(np.isscalar(value) for value in info.values())
    with pytest.raises(RuntimeError):
        env.step(actions)


class ControlledEnv:
    def __init__(self, length=3, **kwargs):
        self.episode_limit = length
        self.n_agents = 10

    def reset(self, seed=None):
        self.t = 0
        self.episode_seed = seed
        return self.get_obs(), {}

    def get_env_info(self):
        return dict(n_agents=10, n_actions=8, state_shape=340, obs_shape=378, episode_limit=self.episode_limit)

    def get_frame(self):
        frame = dict(obs=self.get_obs(), state=self.get_state(), avail_actions=self.get_avail_actions(),
                     present=np.ones(10, dtype=np.uint8), alive=np.ones(10, dtype=np.uint8))
        frame["present"][-1] = frame["alive"][-1] = 0
        return frame

    def get_obs(self):
        obs = np.full((10, 378), self.t / 512, dtype=np.float32)
        obs[-1] = 0
        return obs

    def get_state(self):
        return np.full(340, self.t / 512, dtype=np.float32)

    def get_avail_actions(self):
        avail = np.ones((10, 8), dtype=np.uint8)
        avail[-1] = 0
        avail[-1, 7] = 1
        return avail

    def step(self, actions):
        self.t += 1
        end = self.t == self.episode_limit
        return self.get_obs(), 0.125, end, False, dict(episode_limit=False, end_reason=2 if end else 0,
                                                     hit_time_limit=end, is_win=1, ally_hp_ratio=0.8, enemy_hp_ratio=0.3)

    def close(self):
        pass


class ControlledMAC:
    action_selector = NS(epsilon=0.0)

    def init_hidden(self, batch_size):
        pass

    def select_actions(self, batch, t_ep, t_env, **kwargs):
        return batch["avail_actions"][:, t_ep].argmax(dim=-1)


def controlled_rollout(monkeypatch, length, seed):
    from envs import REGISTRY
    from runners.episode_runner import EpisodeRunner
    from tabx_pipeline import QuietLogger
    monkeypatch.setitem(REGISTRY, "controlled", ControlledEnv)
    args = NS(env="controlled", env_args={"length": length}, common_reward=True, reward_scalarisation="sum",
              batch_size_run=1, n_agents=10, device="cpu", render=False, test_nepisode=1, runner_log_interval=10)
    runner = EpisodeRunner(args, QuietLogger())
    scheme, groups, preprocess = build_batch_schema(runner.get_env_info())
    runner.setup(scheme, groups, preprocess, ControlledMAC())
    recorder = EpisodeRecorder()
    batch = runner.run(test_mode=True, episode_seed=seed, recorder=recorder)
    assert runner.env.episode_seed == seed and runner.t == length
    assert batch["terminated"][0, length - 1, 0] == 1
    assert batch["terminated"][0, :length - 1].sum() == 0
    assert batch["filled"].sum() == length + 1
    return batch, recorder.episode


@pytest.mark.parametrize("length", [3, 512])
def test_real_runner_final_frame_and_hdf5_round_trip(monkeypatch, tmp_path, length):
    original, first = controlled_rollout(monkeypatch, length, 11)
    _, second = controlled_rollout(monkeypatch, 2, 12)
    metadata = dict(episode_limit=512, **TABXLayout(9, 1, 0).metadata())
    path = write_shard(tmp_path / "shard.h5", [first, second], metadata)
    result = verify_shard(path)
    assert result["frames"] == length + 4 and result["steps"] == length + 2
    batch, lengths = read_episode_batch(path)
    assert lengths.tolist() == [length, 2]
    for name in ("obs", "state", "avail_actions"):
        torch.testing.assert_close(batch[name][0, :length + 1], original[name][0, :length + 1])
    for name in ("actions", "reward", "terminated", "actions_onehot"):
        torch.testing.assert_close(batch[name][0, :length], original[name][0, :length])
    assert batch["transition_mask"][0].sum() == length
    assert batch["transition_mask"][1].sum() == 2
    assert batch["filled"][1].sum() == 3
    assert read_episode(path, 1)["frames"]["obs"].shape[0] == 3
    with pytest.raises(FileExistsError):
        write_shard(path, [first], metadata)
    first["transitions"]["actions"][0, -1, 0] = 0
    with pytest.raises(ValueError, match="Illegal"):
        write_shard(tmp_path / "illegal.h5", [first], metadata)


def test_queue_retry_and_owner_fencing(tmp_path):
    queue, other = EvalQueue(tmp_path), EvalQueue(tmp_path)
    payload = dict(rule="hp", seeds=[1, 2], checkpoint_hash="abc")
    key = queue.enqueue(payload, 1)
    assert queue.enqueue(payload, 1) == key
    assert queue.enqueue(dict(payload, seeds=[3]), 1) is None
    assert other.claim("loss") is None
    _, owner, _ = queue.claim("hp")
    assert other.claim("hp") is None
    queue.record(key, owner, dict(seed=1, return_value=3))
    with queue.connect() as db:
        db.execute("UPDATE requests SET updated=0 WHERE id=?", (key,))
    _, next_owner, _ = other.claim("hp", 1)
    with pytest.raises(RuntimeError, match="lease"):
        queue.record(key, owner, dict(seed=2))
    assert queue.results(key) == [dict(seed=1, return_value=3)]
    other.record(key, next_owner, dict(seed=2, return_value=4))
    other.finish(key, next_owner)
    assert queue.status(key)["status"] == "complete"


def test_snapshot_schedule_and_hashes(tmp_path):
    run_dir = tmp_path / "hp/23/1/run"
    run = dict(task_id="23", task_hash="a" * 64, bank_hash="b" * 64, timeout_rule="hp",
               train_seed=1, master_seed=0, run_id="run", provenance={})
    atomic_json(run_dir / "manifest.json", run)
    args = NS(tabx_run_dir=str(run_dir), tabx_output_root=str(tmp_path), save_model_interval=10,
              tabx_screen_episodes=2, tabx_max_pending_evals=1)
    class Learner:
        training_steps = 1
        def save_models(self, path):
            for name in ("agent.th", "mixer.th", "opt.th"):
                (Path(path) / name).write_bytes(b"weights")
    publisher = SnapshotPublisher(args, None)
    publisher.publish(Learner(), 0, force_eval=True)
    assert publisher.next_save == 10
    publisher.publish(Learner(), 12)
    assert publisher.next_save == 20
    assert read_json(run_dir / "evaluation_requests/12.json")["status"] == "skipped_backpressure"
    publisher.publish(Learner(), 12, force_eval=True)
    assert read_json(run_dir / "evaluation_requests/12.json")["status"] == "queued"
    assert set(p.name for p in (run_dir / "checkpoints").iterdir()) == {"0", "12"}
    assert read_json(run_dir / "checkpoints/0/manifest.json")["t_env"] == 0
    assert not list((run_dir / "checkpoints").glob(".snapshot-*"))
    completion = publisher.complete(Learner(), 20, dict(t_env=20))
    assert [entry["t_env"] for entry in read_json(completion)["checkpoints"]] == [0, 12, 20]
    from tabx_storage import completed_checkpoints
    assert [int(path.name) for path in completed_checkpoints(run_dir)[2]] == [0, 12, 20]
    with pytest.raises(FileExistsError, match="immutable"):
        publisher.complete(Learner(), 20, dict(t_env=20))


def test_training_seed_pairing_ignores_run_rule_and_previous_lengths():
    from envs.tabx_tasks import training_episode_seed
    from envs.tabx_wrapper import TABXWrapper
    def sequence(run_id, rule, seed, lengths):
        env = TABXWrapper.__new__(TABXWrapper)
        env.run_id, env.rule = run_id, rule
        env.manifest, env.params = {"task_hash": "a" * 64}, {}
        env.jax = NS(random=NS(PRNGKey=lambda seed: seed, split=lambda key: (key + 1, key + 2)))
        env._reset = lambda key, params: ({}, {})
        env._cache = lambda obs: None
        env._obs = np.zeros((10, 378), dtype=np.float32)
        env._all_alive = np.ones(20, dtype=bool)
        env._native_alive = np.ones(20, dtype=bool)
        env.seed(seed)
        seeds = []
        for index, length in enumerate(lengths):
            env.reset()
            assert env.last_train_episode_index == index
            seeds.append(env.last_episode_seed)
            for _ in range(length):
                env._next_key()
        # Explicit evaluator resets do not consume a training episode index.
        env.reset(seed=123)
        env.reset()
        assert env.last_episode_seed == training_episode_seed("a" * 64, seed, len(lengths))
        return seeds
    first = sequence("attempt-A", "hp", 1, [1] * 10)
    assert first == sequence("attempt-B", "hp", 1, [1] * 10)
    assert first == sequence("attempt-C", "loss", 1, list(range(1, 11)))
    assert first != sequence("attempt-A", "hp", 2, [1] * 10)
    assert first == [training_episode_seed("a" * 64, 1, i) for i in range(10)]


def test_select_waits_for_completion_and_uses_all_checkpoints(tmp_path):
    from envs.tabx_tasks import file_sha256
    from tabx_selection import select_tiers
    run_dir = tmp_path / "hp/23/1/run"
    atomic_json(run_dir / "manifest.json", dict(task_id="23", task_hash="a" * 64, bank_hash="b" * 64,
                timeout_rule="hp", train_seed=1, master_seed=0, run_id="run", seed_protocol="test", provenance={}))
    args = NS(tabx_run_dir=str(run_dir), tabx_output_root=str(tmp_path), save_model_interval=25000,
              tabx_screen_episodes=2, tabx_max_pending_evals=16)
    class Learner:
        training_steps = 0
        def save_models(self, path):
            (Path(path) / "agent.th").write_bytes(str(self.training_steps).encode())
    publisher, learner = SnapshotPublisher(args, None), Learner()
    publisher.publish(learner, 0)
    learner.training_steps = 1
    publisher.publish(learner, 25000)
    queue = EvalQueue(tmp_path)
    while claim := queue.claim("hp"):
        key, owner, request = claim
        for seed in request["seeds"]:
            queue.record(key, owner, dict(seed=seed, return_value=1))
        queue.finish(key, owner)
    assert select_tiers(tmp_path, run_dir, 2, 2, 2)["status"] == "training_in_progress"
    assert not (run_dir / "selection").exists()
    learner.training_steps = 2
    complete = publisher.complete(learner, 50000, dict(t_env=50000))
    result = select_tiers(tmp_path, run_dir, 2, 2, 2)
    assert result["status"] == "pending"
    state = read_json(result["state"])
    assert [int(Path(p).name) for p in state["snapshots"]] == [0, 25000, 50000]
    assert state["training_complete_sha256"] == file_sha256(complete)
    with queue.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 4
    (run_dir / "checkpoints/50000/agent.th").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        select_tiers(tmp_path, run_dir, 2, 2, 2)


def test_seed_stability_and_phase_isolation():
    seed = stable_seed("hash", 0, "screen", 1)
    assert seed == stable_seed("hash", 0, "screen", 1)
    assert len({stable_seed("hash", 0, phase, 1) for phase in ("train", "baseline", "screen", "confirm", "final", "collect")}) == 6
    code = "import sys;sys.path.insert(0,'src');from tabx_storage import stable_seed;print(stable_seed('hash',0,'screen',1))"
    assert int(subprocess.check_output([sys.executable, "-c", code], env=dict(os.environ, PYTHONHASHSEED="1999"))) == seed


def test_snapshot_identity_hashes_and_portable_paths(tmp_path):
    from envs.tabx_tasks import file_sha256
    from tabx_storage import resolve_under, validate_snapshot
    task = dict(task_id="23", task_hash="a", bank_hash="b")
    path = tmp_path / "snapshot"
    path.mkdir()
    (path / "agent.th").write_bytes(b"test")
    atomic_json(path / "config.json", dict(env="tabx"))
    atomic_json(path / "manifest.json", dict(**task, timeout_rule="hp", simulator_contract=SIMULATOR_CONTRACT, layout_version=LAYOUT_VERSION,
                                            agent_sha256=file_sha256(path / "agent.th"),
                                            config_sha256=file_sha256(path / "config.json")))
    assert validate_snapshot(path, task, "hp", check_runtime=False)["task_id"] == "23"
    with pytest.raises(ValueError, match="timeout_rule"):
        validate_snapshot(path, task, "loss", check_runtime=False)
    (path / "agent.th").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_snapshot(path, task, "hp", check_runtime=False)
    with pytest.raises(ValueError, match="escapes"):
        resolve_under(tmp_path, "../outside")


def test_collect_resume_extend_and_verify(monkeypatch, tmp_path):
    from envs import REGISTRY
    from envs.tabx_tasks import file_sha256
    from runners.episode_runner import EpisodeRunner
    import tabx_pipeline
    monkeypatch.setitem(REGISTRY, "controlled", ControlledEnv)
    def factory(*args, **kwargs):
        options = NS(env="controlled", env_args={"length": 3}, common_reward=True, reward_scalarisation="sum",
                     batch_size_run=1, n_agents=10, device="cpu", render=False, test_nepisode=1, runner_log_interval=10)
        runner = EpisodeRunner(options, tabx_pipeline.QuietLogger())
        scheme, groups, preprocess = build_batch_schema(runner.get_env_info())
        runner.setup(scheme, groups, preprocess, ControlledMAC())
        return runner
    monkeypatch.setattr(tabx_pipeline, "build_rollout", factory)
    run_relative = "hp/23/1/run"
    identity = dict(task_id="23", task_hash="a", bank_hash="b", timeout_rule="hp", run_id="run",
                    master_seed=0, seed_protocol="test", provenance={}, **TABXLayout(9, 1, 0).metadata())
    atomic_json(tmp_path / run_relative / "manifest.json", identity)
    tiers = dict(identity, schema_version="tabx-tiers-1", run=run_relative, action_selection="greedy",
                 evaluation_epsilon=0, tiers={}, R0={}, RH={}, paired_difference={})
    for tier in ("medium", "high"):
        snapshot = tmp_path / run_relative / "checkpoints" / tier
        snapshot.mkdir(parents=True)
        (snapshot / "agent.th").write_bytes(tier.encode())
        tiers["tiers"][tier] = dict(snapshot=str(snapshot.relative_to(tmp_path)),
                                    checkpoint_hash=file_sha256(snapshot / "agent.th"), q=0.5 if tier == "medium" else 1)
    tiers_path = tmp_path / "tiers.json"
    atomic_json(tiers_path, tiers)
    args = NS(root=str(tmp_path), rule="hp", tiers=str(tiers_path), task="23", bank="runtime-bank",
              tabx_root=None, threads=1, cpu_affinity=None, episodes_per_tier=2, shard_episodes=2, expected_episodes=2)
    tabx_pipeline.collect(args)
    tabx_pipeline.verify(args)
    tabx_pipeline.collect(args)  # No completed seed is replayed.
    args.episodes_per_tier = args.expected_episodes = 5
    tabx_pipeline.collect(args)
    tabx_pipeline.verify(args)
    for tier in ("medium", "high"):
        files = list((tmp_path / run_relative / "data").glob(f"*/{tier}/*.h5"))
        assert len(files) == 3
        assert sum(verify_shard(path)["episodes"] for path in files) == 5


@pytest.mark.parametrize("values,expected_status", [({0: 0, 10: 5, 20: 10, 30: 10}, "complete"),
                                                     ({0: 0, 10: 0, 20: 10, 30: 0}, "insufficient_quality_gap")])
def test_selection_reentrant_and_held_out_phases(tmp_path, values, expected_status):
    from tabx_selection import select_tiers
    root, run_dir = tmp_path, tmp_path / "hp/23/1/run"
    atomic_json(run_dir / "manifest.json", dict(task_id="23", task_hash="a" * 64, bank_hash="b" * 64,
                timeout_rule="hp", train_seed=1, master_seed=0, run_id="run", seed_protocol="test", provenance={}))
    args = NS(tabx_run_dir=str(run_dir), tabx_output_root=str(root), save_model_interval=10,
              tabx_screen_episodes=4, tabx_max_pending_evals=16)
    class Learner:
        training_steps = 0
        def save_models(self, path):
            (Path(path) / "agent.th").write_bytes(f"weights-{self.training_steps}".encode())
    publisher, learner = SnapshotPublisher(args, None), Learner()
    for step in (0, 10, 20, 30):
        learner.training_steps = step
        publisher.publish(learner, step)
    publisher.complete(learner, 30, dict(t_env=30))
    queue = EvalQueue(root)
    phases = set()
    for _ in range(5):
        result = select_tiers(root, run_dir, 4, 4, 4)
        if result["status"] != "pending":
            break
        assert result["status"] == "pending"
        while claim := queue.claim("hp"):
            key, owner, request = claim
            phases.add(request["phase"])
            step = int(Path(request["snapshot"]).name)
            value = 0 if request["random_policy"] else values[step]
            for seed in request["seeds"]:
                queue.record(key, owner, dict(seed=seed, return_value=value))
            queue.finish(key, owner)
    assert result["status"] == expected_status
    if expected_status != "complete":
        assert result["detail"] == "no_adjacent_stable_high_candidate"
        assert phases == {"screen", "baseline"}
        return
    tiers = read_json(result["tiers"])
    assert tiers["tiers"]["medium"]["q"] == 0.5
    assert tiers["paired_difference"]["ci95"][0] > 0
    assert phases == {"screen", "baseline", "confirm", "final"}
    assert select_tiers(root, run_dir, 4, 4, 4)["status"] == "complete"


@pytest.mark.skipif(not os.environ.get("TABX_BANK"), reason="Set TABX_BANK for official simulator tests")
def test_actual_tabx_shapes_masks_and_native_physics():
    # Mirror deployment: fresh interpreter, EPyMARL/src as script root. pytest
    # may have already imported the checkout's unrelated `src` package.
    code = ("import sys,runpy;sys.path.insert(0,'src');"
            "runpy.run_path('tests/test_tabx_official.py')['native_checks']()")
    subprocess.run([sys.executable, "-c", code], check=True)
