"""Collect and verify greedy HDF5 trajectories from the selected coach tiers."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
from pathlib import Path

from tabx_coach_selection import (MEDIUM_PARAMETERS, SCHEMA, eligible_snapshot, in_half_range, phase_seeds,
                                 selection_path)
from tabx_medium_cli import positive_int, repo_path
from tabx_pipeline import collect_tier, collection_lock, configure_collection, verify_collection_tier
from tabx_reporting import evaluation_summary
from tabx_scheduler import task_selection
from tabx_storage import (EvalQueue, SEED_PROTOCOL, atomic_json, completed_checkpoints, ensure_output_contract,
                          file_sha256, read_json, resolve_under, safe_component, validate_snapshot)


def load_coach_tiers(root, directory, batch, task, queue, official):
    path = selection_path(directory, task)
    digest = file_sha256(path)
    selection = read_json(path)
    if file_sha256(path) != digest:
        raise ValueError(f"Coach selection changed while reading: {path}")
    expected = dict(schema_version=SCHEMA, batch_id=batch["batch_id"], task_id=task,
                    task_hash=batch["task_hashes"][task], bank_hash=batch["bank_hash"], rule=batch["rule"],
                    episode_limit=batch["training_config"]["env_args"]["max_episode_steps"],
                    medium_parameter_sha256=official["parameter_json_sha256"][MEDIUM_PARAMETERS],
                    official_tabx=official, layout_version=official["layout_version"],
                    simulator_contract=official["simulator_contract"], seed_protocol=SEED_PROTOCOL,
                    action_selection="greedy", evaluation_epsilon=0.0)
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"Coach selection {key} mismatch: {path}")
    if any(info["incomplete_runs"] for info in selection["groups"].values()):
        return path, digest, [], dict(high="training_incomplete", half="training_incomplete")
    selected = selection["selected"]
    if set(selected) != {"high", "half"} or selected["half"] is not None and selected["high"] is None:
        raise ValueError(f"Invalid coach tier set: {path}")
    if selected["half"] is not None and (
            selected["high"]["checkpoint_hash"] == selected["half"]["checkpoint_hash"]
            or selected["half"]["confirm"]["mean_return"] >= selected["high"]["confirm"]["mean_return"]
            or not in_half_range(selected["half"]["q_confirm"])):
        raise ValueError(f"Invalid half coach: {path}")

    def confirm_report(key, policy, step, **fields):
        report = queue.completed_result(key)
        if report is None:
            raise ValueError(f"Missing confirmation result: {key}")
        protocol = {field: selection[field] for field in ("task_id", "task_hash", "bank_hash", "rule",
                                                          "episode_limit", "official_tabx", "seed_protocol")}
        protocol.update(schema_version="tabx-coach-eval-1", phase="coach_confirm", request_type=policy,
                        seeds=selection["confirm_seeds"], **fields)
        if any(report["request"].get(field) != value for field, value in protocol.items()):
            raise ValueError(f"Wrong confirmation protocol: {key}")
        return evaluation_summary(key, report, step)

    medium = None
    if any(selected.values()):
        medium = confirm_report(selection["confirm_medium_request_id"], "medium", 0, action_selection="medium")
        if medium != selection["confirm_reference"]["medium"]:
            raise ValueError(f"Coach Medium baseline differs from the confirmation result: {path}")
    task_identity = {key: selection[key] for key in ("task_id", "task_hash", "bank_hash")}
    jobs = []
    for tier, item in selected.items():
        if item is None:
            continue
        if (item not in selection["candidates"] or item["selected_tier"] != tier
                or item["medium_comparison"] != "above_medium" or not item["collectable"]):
            raise ValueError(f"Unqualified {tier} coach: {path}")
        run_dir = resolve_under(root, item["run"])
        snapshot = resolve_under(root, item["snapshot"])
        inventory = completed_checkpoints(run_dir)
        if inventory is None or inventory[1] != item["training_complete_sha256"] or snapshot not in inventory[2]:
            raise ValueError(f"{tier} coach does not belong to its sealed training completion: {snapshot}")
        run = read_json(run_dir / "manifest.json")
        for key, value in dict(task_identity, timeout_rule=batch["rule"], batch_id=batch["batch_id"],
                               run_id=item["run_id"], train_seed=item["train_seed"], group=item["group"]).items():
            if run.get(key) != value:
                raise ValueError(f"{tier} coach run identity mismatch for {key}: {run_dir}")
        saved = validate_snapshot(snapshot, task_identity, batch["rule"], check_runtime=False)
        if (saved["t_env"] != item["t_env"] or saved["agent_sha256"] != item["checkpoint_hash"]
                or saved["critic_sha256"] != item["critic_sha256"]
                or file_sha256(snapshot / "manifest.json") != item["snapshot_manifest_sha256"]
                or not eligible_snapshot(saved, selection["pure_ppo_start"])):
            raise ValueError(f"{tier} coach snapshot does not match its selection: {snapshot}")
        stats = confirm_report(item["confirm_request_id"], "coach_checkpoint", item["t_env"],
                               action_selection="greedy", **{key: item[key] for key in (
                                   "run_id", "snapshot", "snapshot_manifest_sha256", "checkpoint_hash")})
        if stats != item["confirm"] or stats["wins"] <= medium["wins"] or stats["episodes"] != medium["episodes"]:
            raise ValueError(f"{tier} coach fails the paired Medium win threshold: {snapshot}")
        metadata = dict(task_identity, timeout_rule=batch["rule"], tier=tier, run_id=item["run_id"],
                        group=item["group"], train_seed=item["train_seed"], checkpoint_hash=item["checkpoint_hash"],
                        tiers_hash=digest, coach_selection_sha256=digest, coach_selection=str(path.relative_to(root)),
                        training_complete_sha256=inventory[1], snapshot_manifest_sha256=item["snapshot_manifest_sha256"],
                        episode_limit=selection["episode_limit"], action_selection="greedy",
                        evaluation_epsilon=0.0, seed_protocol=SEED_PROTOCOL, collection_phase="coach_collect",
                        provenance=run["provenance"], native_shape=run["native_shape"], official_tabx=official,
                        layout_version=selection["layout_version"], simulator_contract=selection["simulator_contract"],
                        tier_metrics=item, medium_reference=medium)
        jobs.append(dict(snapshot=snapshot, metadata=metadata, master_seed=selection["master_seed"],
                         excluded_seeds=selection["screen_seeds"] + selection["confirm_seeds"]))
    return path, digest, jobs, selection["missing_reasons"]


def run_collection_job(args, target, job):
    configure_collection(args, Path(args.data_root))
    metadata = job["metadata"]

    def desired(count):
        return phase_seeds(metadata["task_hash"], metadata["collection_phase"], count,
                           master_seed=job["master_seed"], exclude=job["excluded_seeds"])

    if args.verify:
        fields = ("task_id", "task_hash", "bank_hash", "timeout_rule", "tier", "run_id", "episode_limit",
                  "checkpoint_hash", "tiers_hash", "training_complete_sha256", "snapshot_manifest_sha256")
        with collection_lock(target):
            report = verify_collection_tier(target, {key: metadata[key] for key in fields}, desired,
                                            expected_episodes=args.episodes_per_tier, outcome_metrics=True)
            atomic_json(target / "verification.json", report)
        return report
    return collect_tier(args, target, metadata, job["snapshot"], desired(args.episodes_per_tier), outcome_metrics=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bank", required=True, help="TABX task bank JSON")
    parser.add_argument("--bank-sha256", help="Expected task bank SHA-256")
    parser.add_argument("--tasks", required=True, help="Comma-separated task IDs")
    parser.add_argument("--rule", choices=("hp", "loss"), default="hp", help="Timeout rule")
    parser.add_argument("--root", type=repo_path, default="results/tabx-mappo-medium-coach", help="Training and selection root")
    parser.add_argument("--data-root", type=repo_path, required=True, help="Formal trajectory output directory")
    parser.add_argument("--batch-id", default="coach-medium-v1", help="Training batch ID")
    parser.add_argument("--episodes-per-tier", type=positive_int, default=500, help="Episodes per selected tier")
    parser.add_argument("--workers", type=positive_int, default=4, help="Concurrent CPU collection jobs")
    parser.add_argument("--threads", type=positive_int, default=1, help="CPU threads per worker")
    parser.add_argument("--shard-episodes", type=positive_int, default=8, help="Maximum new episodes per HDF5 shard")
    parser.add_argument("--verify", action="store_true", help="Verify existing HDF5 shards and read them as EpisodeBatch")
    args = parser.parse_args(argv)
    args.tabx_root = args.cpu_affinity = None
    bank, tasks, bank_hash, hashes = task_selection(args)
    args.bank = str(bank)
    root = Path(args.root)
    directory = root / "_batches" / args.rule / safe_component(args.batch_id)
    batch = read_json(directory / "batch.json")
    if (batch.get("batch_id"), batch.get("rule"), batch.get("bank_hash")) != (args.batch_id, args.rule, bank_hash):
        raise ValueError("Training batch ID, rule or task bank differs")
    if any(hashes[task] != batch["task_hashes"].get(task) for task in tasks):
        raise ValueError("Requested tasks differ from the training batch")
    official = ensure_output_contract(args.data_root)
    queue = EvalQueue(root)
    pending = []
    for task in tasks:
        path, digest, jobs, reasons = load_coach_tiers(root, directory, batch, task, queue, official)
        print(json.dumps(dict(task_id=task, selection=str(path), tiers=[job["metadata"]["tier"] for job in jobs],
                              missing_reasons=reasons)), flush=True)
        for job in jobs:
            target = Path(args.data_root) / args.rule / args.batch_id / task / digest[:16] / job["metadata"]["tier"]
            pending.append((target, job))
    failed = False
    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending)), mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(run_collection_job, args, target, job): target for target, job in pending}
            for future in as_completed(futures):
                target = futures[future]
                try:
                    report = future.result()
                    print(json.dumps(dict(directory=str(target), outcomes=report["outcomes"])), flush=True)
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    failed = True
                    print(f"[COLLECTION-FAILED] {target}: {error}", flush=True)
    return int(failed)
