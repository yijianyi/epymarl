"""Shared admission and CLI regression tests for parallel Medium MAPPO."""
from collections import Counter
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from envs.tabx_layout import LAYOUT_VERSION, SIMULATOR_CONTRACT, TABXLayout
from tabx_dataset import write_shard
from tabx_gpu_pool import SharedGpuPool
from tabx_medium_cli import parser, training_args
from tabx_medium_collection import collection_status_path
import tabx_medium_scheduler as medium_scheduler
from tabx_medium_scheduler import MediumScheduler
from tabx_online_start import BC_STAGE, publish_step0_record, write_step0_index
from tabx_scheduler import BatchScheduler, training_config
from tabx_storage import atomic_json, file_sha256, read_json, simulator_identity
from test_tabx import controlled_rollout


def test_medium_script_defaults_match_the_experiment(tmp_path):
    common = ["--bank", str(tmp_path / "bank.json"), "--tasks", "23,647"]
    demos = parser(None).parse_args(common)
    assert (demos.rule, demos.workers, demos.episodes, demos.episode_limit, demos.seed) == ("hp", 4, 128, 512, 0)
    for group in "ABCD":
        options = parser(group).parse_args([*common, "--gpus", "0"])
        args = training_args(group, options)
        config, _ = training_config(args.overrides, algorithm=args.alg)
        assert (args.seeds, args.jobs_per_gpu, args.eval_workers) == ("1,2,3", 12, 1)
        assert config["t_max"] == 300000
        assert config["batch_size"] == config["buffer_size"] == 8
        assert config["tabx_bc"] == (group != "A")
        assert config["tabx_mix_start"] == (0.5 if group in "CD" else 0.0)
        assert config["tabx_mappo_temp_peak"] == (1.2 if group == "D" else 1.0)
        if group == "B":
            assert args.collection_id == demos.collection_id == "default"


def test_pool_caps_both_gpus_and_rotates_ready_groups(tmp_path):
    pools = [SharedGpuPool(tmp_path, "hp", ["0", "1"], 12, group, f"batch-{group}") for group in "ABCD"]
    leases = []
    for pool in pools:
        pool.register()
        pool.ready(True)
    try:
        for i in range(24):
            winner = i % 4
            assert all(pool.acquire(more_ready=True) is None for index, pool in enumerate(pools) if index != winner)
            gpu, slot, fd = pools[winner].acquire(more_ready=True)
            leases.append((gpu, fd))
        assert Counter(gpu for gpu, _ in leases) == {"0": 12, "1": 12}
        assert all(pool.acquire(more_ready=True) is None for pool in pools)
        assert SharedGpuPool(tmp_path, "hp", ["1", "0"], 12, "A", "test").config == pools[0].config
        with pytest.raises(ValueError, match="share --gpus and --jobs-per-gpu"):
            SharedGpuPool(tmp_path, "hp", ["0"], 12, "A", "test").register()
        os.close(leases.pop()[1])
        gpu, _, fd = pools[0].acquire(more_ready=True)
        leases.append((gpu, fd))
        assert Counter(gpu for gpu, _ in leases) == {"0": 12, "1": 12}
    finally:
        for _, fd in leases:
            os.close(fd)
        for pool in pools:
            pool.unregister()


def test_running_child_retains_capacity_across_scheduler_restart(tmp_path):
    a = SharedGpuPool(tmp_path, "hp", ["0"], 1, "A", "batch-A")
    b = SharedGpuPool(tmp_path, "hp", ["0"], 1, "B", "batch-B")
    a.register()
    a.ready(True)
    _, slot, fd = a.acquire(more_ready=False)
    a.record_launch(slot, {"task_id": "23", "train_seed": 1}, {"run_id": "attempt-A"})
    child = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE, pass_fds=(fd,))
    os.close(fd)
    a.unregister()
    b.register()
    b.ready(True)
    try:
        assert list(a.active_attempts()) == ["attempt-A"]
        assert b.acquire(more_ready=True) is None
        child.communicate(b"\n", timeout=5)
        assert list(a.active_attempts()) == []
        _, _, next_fd = b.acquire(more_ready=False)
        os.close(next_fd)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        b.unregister()


def test_parallel_index_only_adds_entries_after_resume(tmp_path):
    batch = dict(batch_id="B", rule="hp", bank_hash="bank")
    first = {"647/1": {"agent_sha256": "original"}}
    write_step0_index(tmp_path, batch, first, extend=True)
    both = dict(first, **{"23/1": {"agent_sha256": "retried"}})
    write_step0_index(tmp_path, batch, both, extend=True)
    with pytest.raises(ValueError, match="Refusing to replace"):
        write_step0_index(tmp_path, batch, {"647/1": {"agent_sha256": "changed"}}, extend=True)
    assert read_json(tmp_path / "step0_index.json")["entries"] == both


def make_medium_scheduler(root, group, config, *, tasks=("23",), collection_id="default", resume=False):
    args = SimpleNamespace(root=str(root), rule="hp", gpus=["0"], jobs_per_gpu=1, batch_id=group,
                           medium_group=group, resume=resume)
    batch = dict(bank_hash="bank", task_hashes={task: f"task-{task}" for task in tasks},
                 identity=dict(collection_id=collection_id, demo_manifest_sha256={}), training_config=config)
    jobs = [dict(task_id=task, train_seed=1, status="pending", needs_training=True, attempts=[]) for task in tasks]
    return MediumScheduler(args, root / "_batches/hp" / group, batch, dict(jobs=jobs))


@pytest.fixture
def demo_dependency(tmp_path, monkeypatch):
    demo_root = tmp_path / "demos"
    status_path = collection_status_path(demo_root, "hp", "23")
    scheduler = make_medium_scheduler(tmp_path / "runs", "B",
                                      dict(tabx_demo_root=str(demo_root), env_args={"max_episode_steps": 8}),
                                      collection_id="updated")
    directory = status_path.parent
    manifest_path = directory / "manifest.json"
    metadata = dict(task_id="23", task_hash="task", bank_hash="bank", timeout_rule="hp", episode_limit=8,
                    teacher="medium", opponent="medium", label_teacher="medium", label_epsilon=0.0,
                    collection_phase="expert_collect", label_source="deterministic_teacher",
                    provenance={"tabx": simulator_identity()}, **TABXLayout(9, 1, 0).metadata())
    metadata["task_hash"] = scheduler.task_identity("23")["task_hash"]
    directory.mkdir(parents=True)
    shards = []

    def publish_episodes(start, count):
        episodes = []
        for seed in range(start, start + count):
            _, episode = controlled_rollout(monkeypatch, 2, seed)
            episode["transitions"]["teacher_actions"] = episode["transitions"]["actions"].copy()
            episode["transitions"]["teacher_mask"] = episode["frames"]["alive"][:-1, :, None].copy()
            episodes.append(episode)
        shard = write_shard(directory / f"{start:010d}.h5", episodes, metadata)
        shards.append(dict(path=shard.name, sha256=file_sha256(shard)))
        atomic_json(manifest_path, dict(episodes=start + count - 1, metadata=metadata, shards=shards))
        return file_sha256(manifest_path)

    def publish_ready(collection_id, count, digest, **extra):
        atomic_json(status_path, dict(status="ready", collection_id=collection_id, task_id="23",
                    task_hash=metadata["task_hash"], bank_hash="bank", rule="hp", episode_limit=8,
                    requested_episodes=count, actual_episodes=count, manifest_sha256=digest, **extra))

    return SimpleNamespace(scheduler=scheduler, status_path=status_path, manifest_path=manifest_path,
                           metadata=metadata, frozen=scheduler.manifest["identity"]["demo_manifest_sha256"],
                           publish_episodes=publish_episodes, publish_ready=publish_ready)


def test_bc_waits_for_current_collection_with_older_valid_shards(demo_dependency):
    data = demo_dependency
    old_hash = data.publish_episodes(1, 3)
    assert data.scheduler.check_dependency(data.scheduler.jobs[0]) == "waiting_demo"
    data.publish_ready("old", 3, old_hash)
    new_hash = data.publish_episodes(4, 1)
    assert data.scheduler.check_dependency(data.scheduler.jobs[0]) == "waiting_demo"
    assert data.frozen == {}
    data.publish_ready("updated", 4, new_hash)
    assert data.scheduler.check_dependency(data.scheduler.jobs[0]) is None
    assert data.frozen == {"23": new_hash}


def test_bc_rejects_a_stale_ready_hash_without_pinning(demo_dependency):
    data = demo_dependency
    digest = data.publish_episodes(1, 3)
    data.publish_ready("updated", 3, digest)
    data.publish_episodes(4, 1)
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch for task 23"):
        data.scheduler.demo_ready("23")
    assert data.frozen == {}


def test_bc_retries_when_ready_changes_during_manifest_read(demo_dependency, monkeypatch):
    data = demo_dependency
    digest = data.publish_episodes(1, 3)
    data.publish_ready("updated", 3, digest)
    new_hash = data.publish_episodes(4, 1)
    original = medium_scheduler.read_expert_manifest

    def switch_ready(directory, expected):
        data.publish_ready("updated", 4, new_hash)
        return original(directory, expected)

    monkeypatch.setattr(medium_scheduler, "read_expert_manifest", switch_ready)
    assert not data.scheduler.demo_ready("23")
    assert data.frozen == {}
    assert data.scheduler.demo_ready("23")
    assert data.frozen == {"23": new_hash}


def test_bc_checks_hdf5_structure_before_pinning_demonstrations(demo_dependency):
    data = demo_dependency
    shard = data.status_path.parent / "0000000001.h5"
    shard.write_bytes(b"this shard has the expected checksum but is not valid HDF5")
    atomic_json(data.manifest_path, dict(episodes=1, metadata=data.metadata,
                shards=[dict(path=shard.name, sha256=file_sha256(shard))]))
    data.publish_ready("updated", 1, file_sha256(data.manifest_path))
    with pytest.raises(OSError, match="file signature not found"):
        data.scheduler.demo_ready("23")
    assert data.frozen == {}


@pytest.fixture
def failed_source(tmp_path):
    root = tmp_path / "runs"
    source = root / "_batches/hp/B"
    runtime = source / "runtime.json"
    job_state = source / "jobs/000000.json"
    env_args = dict(max_episode_steps=8, world_state_type="global", timeout_rule="hp")
    atomic_json(source / "batch.json", dict(bank_hash="bank", tasks=["23", "647"], seeds=[1],
                identity=dict(medium_group="B", demo_manifest_sha256={"23": "demo", "647": "demo-647"})))
    atomic_json(job_state, dict(scheduler_attempt_id="second", status="dependency_failed"))
    atomic_json(runtime, dict(scheduler_attempt_id="second", status="failed"))

    def resume(group):
        scheduler = make_medium_scheduler(root, group, dict(tabx_online_source=str(source), env_args=env_args),
                                          tasks=("23", "647"), resume=True)
        scheduler.jobs[0]["failed_source_attempt_id"] = "first"
        scheduler.remember_source_failures()
        return scheduler

    def publish_weights(task_id="23"):
        task = dict(task_id=task_id, task_hash=f"task-{task_id}", bank_hash="bank")
        demo_hash = "demo" if task_id == "23" else "demo-647"
        snapshot = root / "hp" / task_id / "1/B-attempt/checkpoints/0"
        snapshot.mkdir(parents=True)
        (snapshot / "agent.th").write_bytes(b"actor")
        (snapshot / "critic.th").write_bytes(b"critic")
        atomic_json(snapshot / "config.json", dict(env_args=env_args, tabx_bc=True))
        atomic_json(snapshot / "manifest.json", dict(
            **task, simulator_contract=SIMULATOR_CONTRACT, layout_version=LAYOUT_VERSION, timeout_rule="hp",
            t_env=0, policy_stage=BC_STAGE, train_seed=1, batch_id="B",
            config_sha256=file_sha256(snapshot / "config.json"), agent_sha256=file_sha256(snapshot / "agent.th"),
            critic_sha256=file_sha256(snapshot / "critic.th"),
        ))
        atomic_json(snapshot.parents[1] / "bc_summary.json", dict(dataset_manifest_sha256=demo_hash))
        args = SimpleNamespace(tabx_output_root=str(root), env_args=env_args, tabx_batch_id="B", seed=1,
                               tabx_expected_demo_manifest_sha256=demo_hash)
        publish_step0_record(args, task, snapshot)
        return snapshot

    return SimpleNamespace(runtime=runtime, job_state=job_state, resume=resume, publish_weights=publish_weights)


def test_c_and_d_wait_for_recovered_source_and_share_sealed_weights(failed_source):
    source = failed_source
    schedulers = [source.resume(group) for group in "CD"]
    for scheduler in schedulers:
        assert scheduler.historical_source_failures[scheduler.jobs[0]["job_id"]] == "second"
        assert scheduler.check_dependency(scheduler.jobs[0]) == "waiting_step0"
    for state in ("preparing", "running"):
        atomic_json(source.runtime, dict(scheduler_attempt_id="third", status=state))
        for scheduler in schedulers:
            assert scheduler.check_dependency(scheduler.jobs[0]) == "waiting_step0"
    source.publish_weights()
    atomic_json(source.runtime, dict(scheduler_attempt_id="third", status="failed"))
    for scheduler in schedulers:
        assert scheduler.check_dependency(scheduler.jobs[0]) is None
    assert schedulers[0].jobs[0]["source_step0_sha256"] == schedulers[1].jobs[0]["source_step0_sha256"]
    assert schedulers[0].sources == schedulers[1].sources


@pytest.mark.parametrize("runtime_status", ["failed", "running"])
def test_c_and_d_report_only_matching_failed_source_attempt(failed_source, runtime_status, capsys):
    source = failed_source
    schedulers = [source.resume(group) for group in "CD"]
    for scheduler in schedulers:
        assert scheduler.check_dependency(scheduler.jobs[0]) == "waiting_step0"
    # The next attempt may reach failure between two dependency polls.
    source.publish_weights("647")
    atomic_json(source.job_state, dict(scheduler_attempt_id="third", status="failed"))
    atomic_json(source.runtime, dict(scheduler_attempt_id="third", status=runtime_status))
    for scheduler in schedulers:
        failed, eligible = scheduler.jobs
        scheduler.pending_train.extend(scheduler.jobs_by_id)
        scheduler.pool = SimpleNamespace(ready=lambda value: None, acquire=lambda **options: None)
        scheduler.fill_slots()
        assert failed["dependency_status"] == "dependency_failed"
        assert failed["failed_source_attempt_id"] == "third"
        assert "Source B attempt third failed before sealing step 0: 23/1" in capsys.readouterr().out
        assert eligible["dependency_status"] == "waiting_gpu"
        assert list(scheduler.pending_train) == [eligible["job_id"]]


def test_b_publishes_preparing_with_new_attempt_before_recovery(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    scheduler = make_medium_scheduler(root, "B", {}, resume=True)
    runtime_path = scheduler.directory / "runtime.json"
    previous = dict(scheduler_attempt_id="old", status="failed", active_trainers=[{"run_id": "prior"}])
    atomic_json(runtime_path, previous)

    def check_preparing(manager):
        status = read_json(runtime_path)
        assert status["status"] == "preparing"
        assert status["active_trainers"] == previous["active_trainers"]
        assert status["scheduler_attempt_id"] == manager.scheduler_attempt_id != "old"
        manager.authorized = True
        manager.state["status"] = "running"
        manager._write_runtime()
        manager.set_dependency(manager.jobs[0], "dependency_failed", "test failure")
        assert read_json(runtime_path)["scheduler_attempt_id"] == manager.jobs[0]["scheduler_attempt_id"]
        return 0

    monkeypatch.setattr(BatchScheduler, "run", check_preparing)
    assert scheduler.run() == 0
