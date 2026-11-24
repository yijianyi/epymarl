"""Bounded collection with per-task readiness for concurrent MAPPO batches."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tabx_expert_shards import demonstration_directory, read_expert_manifest
from tabx_scheduler import REPOSITORY, batch_lock, process_identity, stop_process, task_selection
from tabx_storage import atomic_json, ensure_output_contract, safe_component


def collection_status_path(root, rule, task):
    return demonstration_directory(root, rule, task, "medium") / "collection.json"


def collect_parallel(args):
    collection_id = safe_component(args.collection_id)
    bank, tasks, bank_hash, hashes = task_selection(args)
    root = Path(args.root)
    ensure_output_contract(root)
    owner = dict(pid=os.getpid(), process_start=process_identity(os.getpid()))
    with batch_lock(root / "_collection" / args.rule):
        pending = iter(tasks)
        active = {}
        finished = set()
        failed = False
        stopping = False

        def stop(signum, frame):
            nonlocal stopping
            stopping = True

        def publish(task, status, **extra):
            atomic_json(collection_status_path(root, args.rule, task), dict(
                status=status, task_id=task, task_hash=hashes[task], bank_hash=bank_hash, rule=args.rule,
                owner=owner, collection_id=collection_id, requested_episodes=args.episodes,
                episode_limit=args.episode_limit, seed=args.seed, **extra))

        handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for task in tasks:
                publish(task, "queued")
            while len(finished) < len(tasks) and not stopping:
                while len(active) < args.workers and not stopping:
                    task = next(pending, None)
                    if task is None:
                        break
                    directory = collection_status_path(root, args.rule, task).parent
                    log = directory / "collect.log"
                    command = [sys.executable, str(REPOSITORY / "src/tabx_pipeline.py"), "collect-expert",
                               "--bank", str(bank), "--bank-sha256", bank_hash, "--root", str(root),
                               "--rule", args.rule, "--task", task, "--teacher", "medium", "--opponent", "medium",
                               "--seed", str(args.seed), "--episodes", str(args.episodes),
                               "--episode-limit", str(args.episode_limit), "--shard-episodes", str(args.shard_episodes),
                               "--threads", str(args.threads)]
                    publish(task, "collecting", log=str(log))
                    try:
                        with log.open("a", buffering=1) as stream:
                            child = subprocess.Popen(command, cwd=REPOSITORY, start_new_session=True,
                                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
                                                              PYTHONUNBUFFERED="1"),
                                                     stdout=stream, stderr=subprocess.STDOUT)
                    except OSError as error:
                        publish(task, "failed", error=str(error), log=str(log))
                        finished.add(task)
                        failed = True
                        print(f"[COLLECT-FAILED] task={task} log={log} error={error}", flush=True)
                    else:
                        active[task] = child, log
                        print(f"[COLLECT] task={task} pid={child.pid} log={log}", flush=True)
                for task, (child, log) in list(active.items()):
                    code = child.poll()
                    if code is None:
                        continue
                    del active[task]
                    finished.add(task)
                    try:
                        if code:
                            raise RuntimeError(f"collect-expert exited with code {code}")
                        manifest, digest = read_expert_manifest(log.parent)
                        if manifest["episodes"] < args.episodes:
                            raise ValueError("Collector published fewer than the requested episodes")
                        publish(task, "ready", actual_episodes=manifest["episodes"], manifest_sha256=digest, log=str(log))
                        print(f"[DEMO-READY] task={task} episodes={manifest['episodes']} sha256={digest}", flush=True)
                    except (OSError, ValueError, KeyError, RuntimeError) as error:
                        publish(task, "failed", error=str(error), log=str(log))
                        print(f"[COLLECT-FAILED] task={task} log={log} error={error}", flush=True)
                        failed = True
                if len(finished) < len(tasks) and not stopping:
                    time.sleep(args.poll_seconds)
            return 130 if stopping else int(failed)
        finally:
            for child, _ in active.values():
                stop_process(child)
            for task in tasks:
                if task not in finished:
                    publish(task, "interrupted")
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
