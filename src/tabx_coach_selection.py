"""Compare B/C/D on paired held-out seeds and publish collection manifests."""
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import signal
import tempfile
import time

from tabx_coach_scheduler import GROUPS
from tabx_reporting import evaluation_summary, write_csv
from tabx_scheduler import process_alive, process_identity, stop_process
from tabx_storage import SEED_PROTOCOL, atomic_json, file_sha256, read_json, resolve_under, simulator_identity, stable_seed

SCHEMA = "tabx-coach-selection-1"
QUALITY_RANGE = (0.4, 0.6)
MEDIUM_PARAMETERS = "src/tabx/heuristic_policy/parameters/medium.json"
CANDIDATE_FIELDS = ("task_id", "group", "role", "train_seed", "run_id", "t_env", "snapshot",
                    "checkpoint_hash", "critic_sha256", "snapshot_manifest_sha256", "training_complete_sha256",
                    "screen_request_id", "confirm_request_id", "screen_mean", "confirm_mean", "q_screen",
                    "q_confirm", "wins", "medium_wins", "episodes", "win_rate_diff_pp", "medium_comparison",
                    "collectable", "selected_tier", "screen_half_reason", "confirmation_error")


def selection_path(directory, task):
    return directory / "coaches" / task / "coach_selection.json"


def pure_ppo_start(configs):
    return max((step for config in configs for step in (
        config["tabx_mix_decay_steps"] if config["tabx_mix_start"] > 0 else 0,
        config["tabx_mappo_temp_end_step"] if config["tabx_mappo_temp_peak"] > 1 else 0)), default=0)


def eligible_snapshot(manifest, cutoff):
    start = manifest.get("batch_start_t_env")
    return (manifest.get("policy_stage") == "online_ppo" and manifest["learner_updates"] > 0
            and type(start) is int and cutoff <= start < manifest["t_env"]
            and manifest.get("teacher_probability") == 0 and manifest.get("policy_temperature") == 1.0)


def phase_seeds(task_hash, phase, count, *, master_seed=0, exclude=()):
    used, selected = set(exclude), []
    index = 0
    while len(selected) < count:
        seed = stable_seed(task_hash, master_seed, phase, index)
        index += 1
        if seed not in used:
            selected.append(seed)
            used.add(seed)
    return selected


def by_return(candidate, phase):
    stats = candidate[phase]
    return (-stats["mean_return"], -stats["wins"], candidate["t_env"], candidate["train_seed"], candidate["run_id"])


def by_half(candidate, phase):
    return (abs(candidate[f"q_{phase}"] - 0.5), *by_return(candidate, phase)[1:])


def in_half_range(q):
    return q is not None and QUALITY_RANGE[0] <= q <= QUALITY_RANGE[1]


def screen_candidates(screens, groups, random_stats):
    best_mean = max((item["screen"]["mean_return"] for item in screens), default=None)
    gap = None if random_stats is None or best_mean is None else best_mean - random_stats["mean_return"]
    reference = dict(random=random_stats, top_mean_return=best_mean, gap=gap)
    candidates = []
    for group, info in groups.items():
        unique = {}
        for item in sorted((item for item in screens if item["group"] == group),
                           key=lambda item: by_return(item, "screen")):
            unique.setdefault(item["checkpoint_hash"], item)
        items = list(unique.values())
        info["screened_distinct_weights"] = len(items)
        if len(items) < 2:
            info["missing_half_reason"] = "fewer_than_two_distinct_screened_weights"
        if not items:
            info["missing_best_reason"] = "no_screened_pure_ppo_weights"
            continue
        for item in items:
            item["q_screen"] = (item["screen"]["mean_return"] - random_stats["mean_return"]) / gap if gap and gap > 0 else None
        candidates.append(dict(items[0], role="best"))
        if len(items) > 1 and gap is not None and gap > 0:
            half = min(items[1:], key=lambda item: by_half(item, "screen"))
            reason = None if in_half_range(half["q_screen"]) else "outside_screen_half_range"
            candidates.append(dict(half, role="half", screen_half_reason=reason))
            info["missing_half_reason"] = reason
        elif len(items) > 1:
            info["missing_half_reason"] = "missing_positive_screen_gap"
    return candidates, reference


def confirm_selection(candidates, medium, random_stats, *, screen_issue=None):
    measured = [item for item in candidates if item.get("confirm")]
    top = max((item["confirm"]["mean_return"] for item in measured), default=None)
    gap = None if random_stats is None or top is None else top - random_stats["mean_return"]
    for item in candidates:
        item.update(q_confirm=None, medium_comparison=None, win_rate_diff_pp=None, collectable=False, selected_tier=None)
    for item in measured:
        stats = item["confirm"]
        if gap is not None and gap > 0:
            item["q_confirm"] = (stats["mean_return"] - random_stats["mean_return"]) / gap
        if medium is not None:
            if stats["episodes"] != medium["episodes"]:
                raise ValueError("Checkpoint and Medium must use the same number of paired episodes")
            difference = stats["wins"] - medium["wins"]
            comparison = "above_medium" if difference > 0 else "tied_medium" if difference == 0 else "below_medium"
            item.update(medium_comparison=comparison, win_rate_diff_pp=100 * difference / medium["episodes"])
    eligible = [item for item in measured if item["medium_comparison"] == "above_medium"]
    selected = dict(high=None, half=None)
    if screen_issue:
        reason = screen_issue
    elif not candidates:
        reason = "no_screened_pure_ppo_candidates"
    elif medium is None or random_stats is None or len(measured) != len(candidates):
        reason = "confirm_evaluation_incomplete"
    elif not eligible:
        reason = "no_candidate_above_medium"
    else:
        reason = None
    reasons = dict(high=reason, half=reason)
    if reason is None:
        high = min(eligible, key=lambda item: by_return(item, "confirm"))
        selected["high"] = high
        halves = [item for item in eligible if item["checkpoint_hash"] != high["checkpoint_hash"]
                  and item["confirm"]["mean_return"] < high["confirm"]["mean_return"]
                  and in_half_range(item["q_confirm"])]
        if halves:
            selected["half"] = min(halves, key=lambda item: by_half(item, "confirm"))
        else:
            reasons["half"] = ("no_positive_confirm_gap" if gap is None or gap <= 0
                               else "no_distinct_above_medium_candidate_with_lower_return_and_half_quality")
        for tier, item in selected.items():
            if item is not None:
                item.update(collectable=True, selected_tier=tier)
    return selected, reasons, dict(medium=medium, random=random_stats, top_mean_return=top)


def screen_snapshots(scheduler, jobs, cutoff):
    screens = []
    groups = {group: dict(eligible_snapshots=0, distinct_eligible_weights=0, failed_screens=0, incomplete_runs=[])
              for group in GROUPS}
    weights = defaultdict(set)
    root = scheduler.root
    for job in jobs:
        group, seed = job["group"], job["train_seed"]
        info = groups[group]
        attempt = job["attempts"][-1] if job["attempts"] else {}
        run_id = attempt.get("run_id")
        if not attempt.get("inventory_verified"):
            info["incomplete_runs"].append(dict(train_seed=seed, run_id=run_id, status=job["status"],
                                                error=job.get("dependency_error") or attempt.get("error")))
            continue
        _, completion_hash, snapshots = scheduler.inventories[run_id]
        for snapshot in snapshots:
            manifest = read_json(snapshot / "manifest.json")
            if not eligible_snapshot(manifest, cutoff):
                continue
            info["eligible_snapshots"] += 1
            checkpoint_hash = manifest["agent_sha256"]
            weights[group].add(checkpoint_hash)
            key = attempt["screen_requests"][snapshot.name]
            evaluation = attempt["evaluations"].get(key, {})
            if not evaluation.get("valid") or evaluation.get("status") != "complete":
                info["failed_screens"] += 1
                continue
            screens.append(dict(task_id=job["task_id"], group=group, train_seed=seed, run_id=run_id,
                                t_env=manifest["t_env"], learner_updates=manifest["learner_updates"],
                                batch_start_t_env=manifest["batch_start_t_env"],
                                teacher_probability=manifest["teacher_probability"],
                                policy_temperature=manifest["policy_temperature"],
                                snapshot=str(snapshot.relative_to(root)), run=attempt["run"],
                                checkpoint_hash=checkpoint_hash, critic_sha256=manifest["critic_sha256"],
                                snapshot_manifest_sha256=file_sha256(snapshot / "manifest.json"),
                                training_complete_sha256=completion_hash, screen=evaluation["summary"],
                                screen_request_id=key))
    for group, info in groups.items():
        info["distinct_eligible_weights"] = len(weights[group])
    return screens, groups


class CoachEvaluation:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.queue = scheduler.queue
        self.rule = scheduler.args.rule
        self.scope = ".coach-eval-" + hashlib.sha256(scheduler.args.batch_id.encode()).hexdigest()
        self.runtime = scheduler.directory / "coach_evaluation_runtime.json"
        self.worker = None
        self.stopping = False

    def evaluate(self, requests):
        pending, results, errors = {}, {}, {}
        retry = set()
        for request in requests:
            key = self.queue.request_id(request)
            recovered = self.queue.recover(request, retry_failed=self.scheduler.args.resume)
            if recovered["report"] is None:
                pending[key] = request
            else:
                results[key] = recovered["report"]
        if pending and self.worker is None:
            self.worker = self.scheduler.spawn_eval_workers(self.scope, "coach-eval-workers.log")
            atomic_json(self.runtime, dict(pid=self.worker.pid, process_start=process_identity(self.worker.pid)))
        while pending and not self.stopping:
            for notice in self.queue.notifications(self.rule, self.scope):
                key = notice["request_id"]
                if key in pending:
                    if notice["status"] == "complete":
                        results[key] = self.queue.completed_result(key)
                        del pending[key]
                    elif key not in retry:
                        retry.add(key)
                        self.queue.recover(pending[key], retry_failed=True)
                        continue
                    else:
                        errors[key] = notice["error"]
                        del pending[key]
                self.queue.acknowledge(notice)
            if not pending:
                break
            if self.worker.poll() is not None:
                errors.update((key, "Coach evaluation worker exited; see coach-eval-workers.log") for key in pending)
                break
            time.sleep(self.scheduler.args.poll_seconds)
        return results, errors

    def prepare(self):
        old = read_json(self.runtime) if self.runtime.exists() else {}
        while process_alive(old) and not self.stopping:
            time.sleep(self.scheduler.args.poll_seconds)
        if not self.stopping:
            self.queue.release_batch(self.rule, self.scope)

    def close(self):
        if self.worker is not None:
            stop_process(self.worker)
            self.queue.release_batch(self.rule, self.scope)
            atomic_json(self.runtime, {})


def select_coaches(scheduler):
    evaluation = CoachEvaluation(scheduler)
    queue, manifest = scheduler.queue, scheduler.manifest
    episode_limit = scheduler.config["env_args"]["max_episode_steps"]
    screen_count = scheduler.config["tabx_screen_episodes"]
    confirm_count = manifest["identity"]["confirm_episodes"]
    master_seed = scheduler.config["tabx_master_seed"]
    official = simulator_identity()
    cutoff = pure_ppo_start(scheduler.training_parameters(job)[0] for job in scheduler.jobs)

    def request(protocol, phase, policy, **extra):
        return dict(protocol, schema_version="tabx-coach-eval-1", batch_id=evaluation.scope,
                    request_type=policy, phase=phase, action_selection="greedy" if policy == "coach_checkpoint" else policy,
                    random_policy=policy == "uniform_legal", **extra)

    def stop(signum, frame):
        evaluation.stopping = True

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        evaluation.prepare()
        selections, screens, screen_requests = {}, {}, {}
        for task in manifest["tasks"]:
            task_hash = manifest["task_hashes"][task]
            screen_seeds = [stable_seed(task_hash, master_seed, "screen", i) for i in range(screen_count)]
            confirm_seeds = phase_seeds(task_hash, "coach_confirm", confirm_count, master_seed=master_seed,
                                       exclude=screen_seeds)
            protocol = dict(task_id=task, task_hash=task_hash, bank_hash=manifest["bank_hash"],
                            rule=manifest["rule"], episode_limit=episode_limit, official_tabx=official,
                            seed_protocol=SEED_PROTOCOL)
            jobs = [job for job in scheduler.jobs if job["task_id"] == task]
            screens[task], groups = screen_snapshots(scheduler, jobs, cutoff)
            selections[task] = dict(schema_version=SCHEMA, batch_id=manifest["batch_id"],
                                    **protocol, medium_parameter_sha256=official["parameter_json_sha256"][MEDIUM_PARAMETERS],
                                    layout_version=official["layout_version"], simulator_contract=official["simulator_contract"],
                                    master_seed=master_seed, action_selection="greedy", evaluation_epsilon=0.0,
                                    pure_ppo_start=cutoff, half_range=list(QUALITY_RANGE), groups=groups,
                                    screen_seeds=screen_seeds, confirm_seeds=confirm_seeds)
            screen_requests[task] = request(dict(protocol, seeds=screen_seeds), "coach_screen", "uniform_legal")
        print(f"[COACH] Evaluating random screen baselines for {len(selections)} tasks", flush=True)
        screen_results, errors = evaluation.evaluate(screen_requests.values())
        if evaluation.stopping:
            return 130
        confirm_requests = {}
        for task, selection in selections.items():
            key = queue.request_id(screen_requests[task])
            screen_stats = summary(key, screen_results)
            candidates, reference = screen_candidates(screens[task], selection["groups"], screen_stats)
            selection.update(candidates=candidates, screen_reference=dict(reference, random_request_id=key))
            protocol = {key: selection[key] for key in ("task_id", "task_hash", "bank_hash", "rule", "episode_limit",
                                                        "official_tabx", "seed_protocol")}
            protocol["seeds"] = selection["confirm_seeds"]
            for policy in ("medium", "uniform_legal"):
                payload = request(protocol, "coach_confirm", policy)
                key = queue.request_id(payload)
                confirm_requests[key] = payload
                selection[f"confirm_{policy}_request_id"] = key
            for item in candidates:
                payload = request(protocol, "coach_confirm", "coach_checkpoint", **{
                    key: item[key] for key in ("run_id", "snapshot", "checkpoint_hash", "snapshot_manifest_sha256")})
                key = queue.request_id(payload)
                confirm_requests[key] = payload
                item["confirm_request_id"] = key
        print(f"[COACH] Evaluating {len(confirm_requests)} paired candidate and baseline requests", flush=True)
        confirm_results, confirm_errors = evaluation.evaluate(confirm_requests.values())
        errors.update(confirm_errors)
        if evaluation.stopping:
            return 130
        for task, selection in selections.items():
            task_keys = [selection["screen_reference"]["random_request_id"], selection["confirm_medium_request_id"],
                         selection["confirm_uniform_legal_request_id"]]
            for item in selection["candidates"]:
                key = item["confirm_request_id"]
                task_keys.append(key)
                item["confirm"] = summary(key, confirm_results, item["t_env"])
                item["confirmation_error"] = errors.get(key)
            medium = summary(task_keys[1], confirm_results)
            random_stats = summary(task_keys[2], confirm_results)
            screen_ref = selection["screen_reference"]
            groups = selection["groups"].values()
            screen_issue = None
            if any(info["incomplete_runs"] for info in groups):
                screen_issue = "training_incomplete"
            elif screen_ref["random"] is None or any(info["failed_screens"] for info in groups):
                screen_issue = "screen_evaluation_incomplete"
            elif screen_ref["gap"] is not None and screen_ref["gap"] <= 0:
                screen_issue = "missing_positive_screen_gap"
            selected, reasons, reference = confirm_selection(selection["candidates"], medium, random_stats,
                                                             screen_issue=screen_issue)
            selection.update(selected=selected, missing_reasons=reasons, confirm_reference=reference,
                             evaluation_errors={key: errors[key] for key in task_keys if key in errors})
            atomic_json(selection_path(scheduler.directory, task), selection)
        write_coach_reports(scheduler.directory, manifest["tasks"])
        print(f"[COACH] Report: {scheduler.directory / 'COACH_REPORT.md'}", flush=True)
        return int(bool(errors))
    finally:
        try:
            evaluation.close()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def summary(key, reports, step=0):
    return evaluation_summary(key, reports[key], step) if key in reports else None


def write_coach_reports(directory, tasks):
    rows, lines = [], ["# MAPPO Medium coach selection", ""]
    for task in tasks:
        selection = read_json(selection_path(directory, task))
        lines.extend([f"## Task {task}", "",
                      f"Pure PPO batches start at step {selection['pure_ppo_start']:,} or later.", ""])
        medium = selection["confirm_reference"]["medium"]
        if medium:
            lines.extend([f"Medium vs Medium won {medium['wins']}/{medium['episodes']} held-out episodes.", ""])
        for group, info in selection["groups"].items():
            lines.append(f"- Group {group} had {info['distinct_eligible_weights']} distinct eligible weights and "
                         f"{info['screened_distinct_weights']} with screen results.")
            for key in ("missing_best_reason", "missing_half_reason"):
                if info.get(key):
                    lines.append(f"  - {key.removesuffix('_reason').replace('_', ' ')} "
                                 f"({info[key].replace('_', ' ')})")
            for run in info["incomplete_runs"]:
                error = f" ({run['error']})" if run["error"] else ""
                lines.append(f"  - Seed {run['train_seed']} ended with {run['status']}{error}.")
        lines.extend(["", "| Group | Candidate | Step | Screen q | Confirm q | Wins | Δ vs Medium (pp) | Result |",
                      "|---|---|---:|---:|---:|---:|---:|---|"])
        for item in selection["candidates"]:
            confirm = item.get("confirm") or {}
            rows.append(dict(item, screen_mean=item["screen"]["mean_return"], confirm_mean=confirm.get("mean_return"),
                             wins=confirm.get("wins"), episodes=confirm.get("episodes"),
                             medium_wins=medium["wins"] if medium else None))
            wins = f"{confirm['wins']}/{confirm['episodes']}" if confirm else "—"
            screen_q, confirm_q, diff = (f"{item[key]:.3f}" if item[key] is not None else "—"
                                         for key in ("q_screen", "q_confirm", "win_rate_diff_pp"))
            result = item["selected_tier"] or (item["medium_comparison"] or "evaluation unavailable").replace("_", " ")
            lines.append(f"| {item['group']} | {item['role']} | {item['t_env']} | {screen_q} | {confirm_q} | "
                         f"{wins} | {diff} | {result} |")
        lines.append("")
        for tier, reason in selection["missing_reasons"].items():
            if reason:
                lines.append(f"- {tier.capitalize()} unavailable ({reason.replace('_', ' ')}).")
        lines.append("")
    write_csv(directory / "coach_candidates.csv", CANDIDATE_FIELDS, rows)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as stream:
        try:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.replace(stream.name, directory / "COACH_REPORT.md")
        finally:
            Path(stream.name).unlink(missing_ok=True)
