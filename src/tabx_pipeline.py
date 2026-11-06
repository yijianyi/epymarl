"""Portable TABX CPU evaluation/collection CLI. Optional imports stay in commands."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


class QuietLogger:
    def __init__(self):
        import logging
        self.console_logger = logging.getLogger("tabx_pipeline")

    def log_stat(self, *args, **kwargs):
        pass


def configure_cpu(threads=1, affinity=None):
    # Called BEFORE importing torch/JAX or launching a new worker interpreter.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    if affinity:
        cores = set()
        for item in affinity.split(","):
            bounds = item.split("-")
            cores.update(range(int(bounds[0]), int(bounds[-1]) + 1))
        os.sched_setaffinity(0, cores)


def build_rollout(snapshot, bank, rule, tabx_root=None, random_policy=False):
    import numpy as np
    import torch
    from components.batch_schema import build_batch_schema
    from components.episode_buffer import EpisodeBatch
    from controllers import REGISTRY
    from envs.tabx_tasks import load_task, import_tabx
    from runners.episode_runner import EpisodeRunner
    from tabx_storage import read_json, validate_snapshot

    config = read_json(Path(snapshot) / "config.json")
    if config["env"] != "tabx" or config["runner"] != "episode":
        raise ValueError("CPU rollout requires a TABX episode snapshot")
    if config["env_args"]["timeout_rule"] != rule:
        raise ValueError("Worker rule differs from snapshot config")
    # The snapshot can be moved to another machine; paths are supplied anew.
    import_tabx(tabx_root)
    _, task_manifest = load_task(bank, config["env_args"]["key"])
    validate_snapshot(snapshot, task_manifest, rule)
    config.update(device="cpu", use_cuda=False, batch_size_run=1, render=False,
                  evaluation_epsilon=0.0, test_greedy=True, test_nepisode=10**9)
    config["env_args"].update(task_bank=str(bank), tabx_root=tabx_root,
                              expected_task_hash=task_manifest["task_hash"], expected_bank_hash=task_manifest["bank_hash"])
    args = SimpleNamespace(**config)
    runner = EpisodeRunner(args, QuietLogger())
    info = runner.get_env_info()
    args.n_agents, args.n_actions, args.state_shape = info["n_agents"], info["n_actions"], info["state_shape"]
    scheme, groups, preprocess = build_batch_schema(info)
    derived = EpisodeBatch(scheme, groups, 1, 1, preprocess=preprocess)
    if random_policy:
        class RandomMAC:
            action_selector = SimpleNamespace(epsilon=1.0)

            def init_hidden(self, batch_size):
                pass

            def select_actions(self, batch, t_ep, t_env, **kwargs):
                return torch.distributions.Categorical(batch["avail_actions"][:, t_ep].float()).sample().long()
        mac = RandomMAC()
    else:
        mac = REGISTRY[args.mac](derived.scheme, groups, args)
        mac.load_models(str(snapshot))
        mac.agent.eval()
    runner.setup(scheme, groups, preprocess, mac)
    return runner


def rollout(runner, seed, recorder=None):
    import numpy as np
    import torch
    from envs.tabx_outcome import END_REASONS
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    started = time.monotonic()
    with torch.inference_mode():
        batch = runner.run(test_mode=True, episode_seed=int(seed), recorder=recorder)
    info = runner.last_episode_info
    if runner.t < 1 or not batch["terminated"][0, runner.t - 1, 0].item():
        raise ValueError("Terminal transition is missing in runner batch")
    return dict(seed=int(seed), **info, end_reason_name=END_REASONS[int(info["end_reason"])],
                elapsed_seconds=time.monotonic() - started)


def process_request(queue, claim, bank, tabx_root=None):
    import jax
    import resource
    import torch
    from tabx_storage import atomic_json, read_json, resolve_under
    key, owner, request = claim
    runner = None
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or jax.default_backend() != "cpu":
            raise RuntimeError("Evaluator must run with no visible GPU and JAX CPU")
        snapshot = resolve_under(queue.root, request["snapshot"])
        manifest = read_json(snapshot / "manifest.json")
        for field, snapshot_field in (("checkpoint_hash", "agent_sha256"), ("task_hash", "task_hash"),
                                      ("bank_hash", "bank_hash"), ("task_id", "task_id"),
                                      ("run_id", "run_id"), ("rule", "timeout_rule")):
            if request[field] != manifest[snapshot_field]:
                raise ValueError(f"Request/snapshot mismatch: {field}")
        runner = build_rollout(snapshot, bank, request["rule"], tabx_root, request["random_policy"])
        finished = {row["seed"] for row in queue.results(key)}
        for seed in request["seeds"]:
            queue.heartbeat(key, owner)
            if seed not in finished:
                queue.record(key, owner, rollout(runner, seed))
        rows = queue.results(key)
        total_seconds = sum(row["elapsed_seconds"] for row in rows)
        report = dict(request=request, episodes=rows, worker=dict(
            jax_backend=jax.default_backend(), cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
            torch_device="cpu", torch_threads=torch.get_num_threads(),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            steps_per_second=sum(row["length"] for row in rows) / max(total_seconds, 1e-9)))
        atomic_json(resolve_under(queue.root, request["run"]) / "evaluations" / f"{key}.json", report)
        queue.finish(key, owner)
        print(json.dumps(dict(request_id=key, status="complete", episodes=len(rows))), flush=True)
        return True
    except Exception:
        error = traceback.format_exc()
        queue.finish(key, owner, error)
        print(json.dumps(dict(request_id=key, status="failed", error=error)), file=sys.stderr, flush=True)
        return False
    finally:
        if runner is not None:
            runner.close_env()


def eval_worker(args):
    configure_cpu(args.threads, args.cpu_affinity)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(args.root).expanduser().resolve() / "_jax_cache"))
    if args.workers > 1:
        children = []
        command = [sys.executable, str(Path(__file__).resolve()), "eval-worker", "--root", args.root,
                   "--bank", args.bank, "--rule", args.rule, "--workers", "1", "--threads", str(args.threads),
                   "--poll-seconds", str(args.poll_seconds), "--lease-seconds", str(args.lease_seconds)]
        if args.tabx_root:
            command.extend(["--tabx-root", args.tabx_root])
        if args.once:
            command.append("--once")
        if args.retry_failed:
            from tabx_storage import EvalQueue
            EvalQueue(args.root).retry_failed(args.rule)
        try:
            core_groups = [[] for _ in range(args.workers)]
            if args.cpu_affinity:
                cores = sorted(os.sched_getaffinity(0))
                if len(cores) < args.workers:
                    raise ValueError("CPU affinity needs at least one core per worker")
                for index, core in enumerate(cores):
                    core_groups[index % args.workers].append(core)
            for group in core_groups:
                child_command = command + (["--cpu-affinity", ",".join(map(str, group))] if group else [])
                children.append(subprocess.Popen(child_command, env=os.environ.copy()))
            while any(child.poll() is None for child in children):
                time.sleep(0.5)
            if any(child.returncode for child in children):
                raise RuntimeError("An evaluation worker process exited unsuccessfully")
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                child.wait()
        return
    import torch
    from tabx_storage import EvalQueue
    torch.set_num_threads(args.threads)
    queue = EvalQueue(args.root)
    if args.retry_failed:
        queue.retry_failed(args.rule)
    failed = False
    while True:
        claim = queue.claim(args.rule, args.lease_seconds)
        if claim is None:
            if args.once:
                if failed:
                    raise RuntimeError("One or more evaluation requests failed; inspect the queue errors")
                return
            time.sleep(args.poll_seconds)
        else:
            failed = not process_request(queue, claim, args.bank, args.tabx_root) or failed


@contextmanager
def collection_lock(directory):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another collector owns {directory}") from exc
        yield


def load_tiers(root, path, rule):
    from tabx_storage import file_sha256, read_json, resolve_under
    tiers = read_json(path)
    if tiers.get("schema_version") != "tabx-tiers-1" or tiers["timeout_rule"] != rule:
        raise ValueError("tiers.json schema or rule mismatch")
    if tiers["action_selection"] != "greedy" or tiers["evaluation_epsilon"] != 0:
        raise ValueError("Unsupported collection action protocol")
    run_dir = resolve_under(root, tiers["run"])
    run = read_json(run_dir / "manifest.json")
    if "training_complete_sha256" in tiers:
        from tabx_storage import completed_checkpoints
        inventory = completed_checkpoints(run_dir)
        if inventory is None or inventory[1] != tiers["training_complete_sha256"]:
            raise ValueError("tiers.json is bound to a different training completion inventory")
    for key in ("task_id", "task_hash", "bank_hash", "timeout_rule", "run_id", "master_seed", "seed_protocol"):
        if tiers[key] != run[key]:
            raise ValueError(f"Run/tiers mismatch: {key}")
    for tier in ("medium", "high"):
        snapshot = resolve_under(root, tiers["tiers"][tier]["snapshot"])
        if file_sha256(snapshot / "agent.th") != tiers["tiers"][tier]["checkpoint_hash"]:
            raise ValueError(f"{tier} checkpoint hash mismatch")
    return tiers, run_dir


def collect(args):
    configure_cpu(args.threads, args.cpu_affinity)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(args.root).expanduser().resolve() / "_jax_cache"))
    import torch
    from tabx_dataset import EpisodeRecorder, verify_shard, write_shard
    from tabx_storage import atomic_json, file_sha256, resolve_under, stable_seed
    torch.set_num_threads(args.threads)
    root = Path(args.root).expanduser().resolve()
    tiers, run_dir = load_tiers(root, args.tiers, args.rule)
    tiers_hash = file_sha256(args.tiers)
    for tier in ("medium", "high"):
        target = run_dir / "data" / tiers_hash[:16] / tier
        with collection_lock(target):
            metadata = dict(task_id=tiers["task_id"], task_hash=tiers["task_hash"], bank_hash=tiers["bank_hash"],
                            timeout_rule=args.rule, tier=tier, run_id=tiers["run_id"],
                            checkpoint_hash=tiers["tiers"][tier]["checkpoint_hash"], tiers_hash=tiers_hash,
                            seed_protocol=tiers["seed_protocol"], action_selection="greedy", provenance=tiers["provenance"],
                            tier_metrics=tiers["tiers"][tier], selection_metrics={k: tiers[k] for k in ("R0", "RH", "paired_difference")})
            complete = set()
            for shard in sorted(target.glob("*.h5")):
                verified = verify_shard(shard, {k: metadata[k] for k in ("task_hash", "timeout_rule", "tier", "checkpoint_hash", "tiers_hash")})
                if complete.intersection(verified["seeds"]):
                    raise ValueError("Duplicate collection seeds across shards")
                complete.update(verified["seeds"])
            desired = [stable_seed(tiers["task_hash"], tiers["master_seed"], "collect", i) for i in range(args.episodes_per_tier)]
            missing = [seed for seed in desired if seed not in complete]
            runner = None
            started = time.monotonic()
            steps = 0
            try:
                if missing:
                    runner = build_rollout(resolve_under(root, tiers["tiers"][tier]["snapshot"]), args.bank,
                                           args.rule, args.tabx_root)
                    metadata["episode_limit"] = runner.episode_limit
                for start in range(0, len(missing), args.shard_episodes):
                    episodes = []
                    seeds = missing[start:start + args.shard_episodes]
                    for seed in seeds:
                        recorder = EpisodeRecorder()
                        result = rollout(runner, seed, recorder)
                        steps += result["length"]
                        episodes.append(recorder.episode)
                    import hashlib
                    name = hashlib.sha256(json.dumps(seeds).encode()).hexdigest()[:20] + ".h5"
                    path = write_shard(target / name, episodes, metadata)
                    complete.update(seeds)
                    print(json.dumps(dict(tier=tier, shard=str(path), episodes=len(seeds))), flush=True)
                shards = [dict(path=p.name, sha256=file_sha256(p), bytes=p.stat().st_size) for p in sorted(target.glob("*.h5"))]
                atomic_json(target / "manifest.json", dict(metadata=metadata, requested_episodes=args.episodes_per_tier,
                                                           published_episodes=len(complete), seeds=sorted(complete), shards=shards,
                                                           last_session_steps_per_second=steps / max(time.monotonic() - started, 1e-9)))
            finally:
                if runner is not None:
                    runner.close_env()


def verify(args):
    import numpy as np
    from tabx_dataset import read_episode, read_episode_batch, verify_shard
    from tabx_storage import atomic_json, file_sha256, stable_seed
    root = Path(args.root).expanduser().resolve()
    tiers, run_dir = load_tiers(root, args.tiers, args.rule)
    if str(args.task) != tiers["task_id"]:
        raise ValueError("Requested task differs from tiers.json")
    tiers_hash = file_sha256(args.tiers)
    reports = {}
    for tier in ("medium", "high"):
        directory = run_dir / "data" / tiers_hash[:16] / tier
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"No collection manifest for {tier}")
        from tabx_storage import read_json
        collection = read_json(manifest_path)
        seeds, hashes, steps, size = set(), [], 0, 0
        shards = sorted(directory.glob("*.h5"))
        published = {entry["path"]: entry for entry in collection["shards"]}
        if set(published) != {p.name for p in shards}:
            raise ValueError("Shard list differs from published manifest; rerun collect to reconcile")
        for shard in shards:
            if file_sha256(shard) != published[shard.name]["sha256"]:
                raise ValueError("Published shard SHA-256 mismatch")
            result = verify_shard(shard, dict(task_hash=tiers["task_hash"], timeout_rule=args.rule,
                                              checkpoint_hash=tiers["tiers"][tier]["checkpoint_hash"], tier=tier, tiers_hash=tiers_hash))
            if seeds.intersection(result["seeds"]):
                raise ValueError("Duplicate seeds across shards")
            seeds.update(result["seeds"])
            hashes.extend(result["trajectory_hashes"])
            steps += result["steps"]
            size += result["bytes"]
            # Every published shard is exercised through the downstream reader.
            batch, lengths = read_episode_batch(shard, [0])
            ep = read_episode(shard, 0)
            length = int(lengths[0])
            for name in ("obs", "state", "avail_actions"):
                np.testing.assert_array_equal(batch[name][0, :length + 1].cpu().numpy(), ep["frames"][name])
            for name in ("actions", "reward", "terminated"):
                np.testing.assert_array_equal(batch[name][0, :length].cpu().numpy(), ep["transitions"][name])
        desired = {stable_seed(tiers["task_hash"], tiers["master_seed"], "collect", i)
                   for i in range(collection["requested_episodes"])}
        if not desired.issubset(seeds) or seeds != set(collection["seeds"]) or len(seeds) != collection["published_episodes"]:
            raise ValueError("Missing or mismatched collection seeds")
        if args.expected_episodes is not None and len(seeds) != args.expected_episodes:
            raise ValueError(f"Expected {args.expected_episodes} episodes per tier; found {len(seeds)}")
        reports[tier] = dict(episodes=len(seeds), steps=steps, bytes=size,
                             bytes_per_episode=size / max(len(seeds), 1),
                             duplicate_trajectory_fraction=1 - len(set(hashes)) / max(len(hashes), 1),
                             q=tiers["tiers"][tier]["q"])
    report = dict(status="verified", task_id=tiers["task_id"], timeout_rule=args.rule, tiers=reports)
    atomic_json(run_dir / "data" / tiers_hash[:16] / "verification.json", report)
    print(json.dumps(report, indent=2))


def check_env(args):
    configure_cpu(args.threads, args.cpu_affinity)
    import numpy as np
    from envs.tabx_wrapper import TABXWrapper
    from tabx_storage import provenance
    for task_id in args.tasks.split(","):
        env = TABXWrapper(task_id, task_bank=args.bank, timeout_rule=args.rule, tabx_root=args.tabx_root)
        try:
            actions, frames, rewards = [], [], []
            env.reset(seed=args.seed)
            initial = env.get_frame()
            for _ in range(args.steps):
                available = env.get_avail_actions()
                action = np.array([np.flatnonzero(row)[0] for row in available])
                _, reward, ended, _, _ = env.step(action)
                actions.append(action)
                frames.append(env.get_frame())
                rewards.append(reward)
                if ended:
                    break
            env.reset(seed=args.seed)
            for name, value in initial.items():
                np.testing.assert_array_equal(value, env.get_frame()[name])
            for action, frame, reward in zip(actions, frames, rewards):
                _, replay_reward, _, _, _ = env.step(action)
                for name, value in frame.items():
                    np.testing.assert_array_equal(value, env.get_frame()[name])
                np.testing.assert_equal(reward, replay_reward)
            teams = np.asarray(env.task["scenario"]["teams"]).reshape(-1)
            print(json.dumps(dict(task_id=task_id, info=env.get_env_info(), allies=int((teams == 0).sum()),
                                  enemies=int((teams == 1).sum()), zones=env.task["zone_scenario"]["n_zone"],
                                  replay_steps=len(actions), task_hash=env.manifest["task_hash"],
                                  bank_hash=env.manifest["bank_hash"],
                                  canonical_hash_matches=env.manifest["canonical_hash_matches"])), flush=True)
        finally:
            env.close()
    print(json.dumps(provenance(), indent=2))


def report(args):
    from envs.tabx_tasks import index_bank
    from tabx_storage import atomic_json, read_json
    _, tasks, bank_hash = index_bank(args.bank)
    selected = set(tasks)
    if args.pilot:
        pilot = read_json(args.pilot)
        if pilot["bank_hash"] != bank_hash:
            raise ValueError("Pilot uses a different task bank")
        selected = set(pilot["pilot"])
    root = Path(args.root).expanduser().resolve()
    rules = args.rules.split(",")
    if not set(rules).issubset({"hp", "loss"}):
        raise ValueError("rules must contain hp and/or loss")
    rows, achieved = [], {rule: set() for rule in rules}
    for rule in rules:
        for path in sorted((root / rule).glob("*/*/*/manifest.json")):
            run = read_json(path)
            if run["bank_hash"] != bank_hash or run["task_id"] not in selected:
                continue
            directory = path.parent
            summary = directory / "training_summary.json"
            tiers = [read_json(p) for p in directory.glob("selection/*/tiers.json")]
            failures = [read_json(p) for p in directory.glob("selection/*/needs_training.json")]
            collected = [read_json(p) for p in directory.glob("data/*/*/manifest.json")]
            if tiers:
                achieved[rule].add(run["task_id"])
            rows.append(dict(task_id=run["task_id"], timeout_rule=rule, run_id=run["run_id"],
                             training_status="complete" if (directory / "training_complete.json").is_file() else "training_in_progress",
                             training=read_json(summary) if summary.exists() else None,
                             selected_pairs=len(tiers), medium_q=[t["tiers"]["medium"]["q"] for t in tiers],
                             failed_selections=[f["reason"] for f in failures],
                             collections=[dict(tier=c["metadata"]["tier"], episodes=c["published_episodes"],
                                               checkpoint_hash=c["metadata"]["checkpoint_hash"]) for c in collected]))
    result = dict(bank_hash=bank_hash, tasks_requested=len(selected), runs=len(rows),
                  per_rule={rule: dict(qualified=len(achieved[rule]), missing_or_unqualified=sorted(selected - achieved[rule]))
                            for rule in rules}, details=rows)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    def common(name, help_text, bank=False, root=False):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--rule", choices=("hp", "loss"), default="hp")
        if bank:
            command.add_argument("--bank", required=True, help="Task-bank JSON on this machine")
            command.add_argument("--tabx-root", default=os.environ.get("TABX_ROOT"), help="AnyTABX checkout root")
            command.add_argument("--threads", type=int, default=1)
            command.add_argument("--cpu-affinity", help="Optional allocated CPU cores, e.g. 0-7,16-23")
        if root:
            command.add_argument("--root", required=True, help="Output root used during training")
        return command
    check = common("check-env", "Check shapes, masks and same-seed replay", bank=True)
    check.add_argument("--tasks", default="23,651,657,647")
    check.add_argument("--seed", type=int, default=1)
    check.add_argument("--steps", type=int, default=8)
    worker = common("eval-worker", "Run a bounded pool of independent CPU workers", bank=True, root=True)
    worker.add_argument("--workers", type=int, default=1)
    worker.add_argument("--once", action="store_true", help="Drain the current queue, then exit")
    worker.add_argument("--retry-failed", action="store_true")
    worker.add_argument("--poll-seconds", type=float, default=2.0)
    worker.add_argument("--lease-seconds", type=float, default=3600)
    select = common("select", "Queue missing evaluations or publish tiers.json", root=True)
    select.add_argument("--task", required=True)
    select.add_argument("--run-id", required=True)
    select.add_argument("--baseline-episodes", type=int, default=64)
    select.add_argument("--screen-episodes", type=int, default=32)
    select.add_argument("--confirm-episodes", type=int, default=128)
    select.add_argument("--medium-range", default="0.4,0.6")
    select.add_argument("--selection-round", type=int, default=0, help="New held-out seeds after a failed selection attempt")
    collect_command = common("collect", "Collect or extend immutable HDF5 shards", bank=True, root=True)
    collect_command.add_argument("--tiers", required=True)
    collect_command.add_argument("--episodes-per-tier", type=int, default=500)
    collect_command.add_argument("--shard-episodes", type=int, default=10)
    verify_command = common("verify", "Validate shards and EpisodeBatch readback", root=True)
    verify_command.add_argument("--tiers", required=True)
    verify_command.add_argument("--task", required=True)
    verify_command.add_argument("--expected-episodes", type=int)
    pilot = commands.add_parser("pilot", help="Lock the shared split and 32 training-task pilot list")
    pilot.add_argument("--bank", required=True)
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--split-file", help="Existing JSON with train/validation/test task ID lists")
    pilot.add_argument("--seed", type=int, default=0)
    pilot.add_argument("--count", type=int, default=32)
    report_command = commands.add_parser("report", help="Report runs, unqualified tasks and collection counts")
    report_command.add_argument("--bank", required=True)
    report_command.add_argument("--root", required=True)
    report_command.add_argument("--pilot")
    report_command.add_argument("--rules", default="hp,loss")
    report_command.add_argument("--output")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("threads", "workers", "steps", "episodes_per_tier", "shard_episodes", "poll_seconds", "lease_seconds"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.command == "check-env":
        check_env(args)
    elif args.command == "eval-worker":
        eval_worker(args)
    elif args.command == "select":
        from tabx_selection import find_run, select_tiers
        run_dir = find_run(args.root, args.rule, args.task, args.run_id)
        print(json.dumps(select_tiers(args.root, run_dir, args.baseline_episodes, args.screen_episodes,
                                     args.confirm_episodes, tuple(map(float, args.medium_range.split(","))),
                                     args.selection_round), indent=2))
    elif args.command == "collect":
        collect(args)
    elif args.command == "verify":
        verify(args)
    elif args.command == "pilot":
        from tabx_pilot import build_pilot
        manifest = build_pilot(args.bank, args.output, args.seed, args.count, args.split_file)
        print(json.dumps(dict(output=args.output, pilot=manifest["pilot"],
                              split_counts={k: len(v) for k, v in manifest["splits"].items()},
                              unresolved_lineage=len(manifest["unresolved_lineage"]),
                              tasks_without_lineage=len(manifest["tasks_without_lineage_metadata"])), indent=2))
        print("\nHistorical timeout group | Training tasks | Pilot tasks")
        print("--- | ---: | ---:")
        for group, counts in manifest["timeout_stratification"]["counts"].items():
            print(f"{group} | {counts['train']} | {counts['pilot']}")
    elif args.command == "report":
        report(args)


if __name__ == "__main__":
    main()
