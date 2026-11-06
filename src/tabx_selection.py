"""Reentrant evaluation -> candidate confirmation -> held-out tier selection."""
import hashlib
import json
import math
from pathlib import Path
import statistics

from tabx_storage import EvalQueue, atomic_json, completed_checkpoints, evaluation_request, read_json, resolve_under


def summarize(rows):
    values = [float(row["return_value"]) for row in rows]
    if len(values) < 2:
        raise ValueError("At least two episodes are needed for uncertainty estimates")
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    return dict(n=len(values), mean=mean, standard_error=se, ci95=[mean - 1.96 * se, mean + 1.96 * se])


def paired_difference(high, low):
    high, low = {r["seed"]: r for r in high}, {r["seed"]: r for r in low}
    if set(high) != set(low):
        raise ValueError("Paired comparison requires identical episode seeds")
    return summarize([dict(return_value=high[s]["return_value"] - low[s]["return_value"]) for s in sorted(high)])


def find_run(root, rule, task, run_id):
    from tabx_storage import safe_component
    for part in (rule, task, run_id):
        safe_component(part)
    matches = list((Path(root).expanduser().resolve() / rule / str(task)).glob(f"*/{run_id}/manifest.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one run for {rule}/{task}/{run_id}; found {len(matches)}")
    run_dir = matches[0].parent
    run = read_json(matches[0])
    if run["timeout_rule"] != rule or run["task_id"] != str(task) or run["run_id"] != run_id:
        raise ValueError("Run manifest identity mismatch")
    return run_dir


def select_tiers(root, run_dir, baseline_episodes=64, screen_episodes=32, confirm_episodes=128,
                 medium_range=(0.4, 0.6), selection_round=0):
    root, run_dir = Path(root).expanduser().resolve(), Path(run_dir).resolve()
    if min(baseline_episodes, screen_episodes, confirm_episodes) < 2:
        raise ValueError("Selection episode counts must be >= 2")
    lower, upper = medium_range
    if not 0 < lower < upper < 1:
        raise ValueError("medium-range must satisfy 0 < lower < upper < 1")
    inventory = completed_checkpoints(run_dir)
    if inventory is None:
        return dict(status="training_in_progress", run_id=read_json(run_dir / "manifest.json")["run_id"])
    complete, completion_hash, snapshots = inventory
    options = dict(baseline=baseline_episodes, screen=screen_episodes, confirm=confirm_episodes,
                   medium_range=list(medium_range), selection_round=selection_round,
                   training_complete_sha256=completion_hash)
    selection_id = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:16]
    directory = run_dir / "selection" / selection_id
    tiers_path = directory / "tiers.json"
    if tiers_path.exists():
        return dict(status="complete", tiers=str(tiers_path))
    queue = EvalQueue(root)
    state_path = directory / "state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["training_complete_sha256"] != completion_hash:
            raise ValueError("Selection state is bound to a different training completion inventory")
    else:
        if len(snapshots) < 2:
            return dict(status="insufficient_checkpoints", required=2, found=len(snapshots))
        state = dict(status="screen", options=options,
                     training_complete_sha256=completion_hash, final_t_env=complete["final_t_env"],
                     snapshots=[str(p.relative_to(root)) for p in snapshots])
        atomic_json(state_path, state)

    def evaluate(snapshot, phase, count, random_policy=False):
        request = evaluation_request(root, run_dir, resolve_under(root, snapshot),
                                     phase if selection_round == 0 else f"{phase}:{selection_round}", count,
                                     random_policy=random_policy)
        key = queue.enqueue(request)
        status = queue.status(key)
        if status["status"] == "failed":
            raise RuntimeError(f"Evaluation {key} failed: {status['error']}; fix and use eval-worker --retry-failed")
        rows = queue.results(key)
        if status["status"] != "complete" or len(rows) != count:
            pending.append(key)
            return None
        return rows

    def wait_for_evaluations():
        atomic_json(state_path, state)
        return dict(status="pending", phase=state["status"], requests=pending, state=str(state_path))

    def fail(reason, **details):
        state.update(status="failed", reason=reason, **details)
        atomic_json(state_path, state)
        atomic_json(directory / "needs_training.json", dict(task_id=read_json(run_dir / "manifest.json")["task_id"],
                                                           reason=reason, **details))
        return dict(status=reason, state=str(state_path), **details)

    if state["status"] == "failed":
        return dict(status=state["reason"], state=str(state_path))

    pending = []
    baseline = evaluate(state["snapshots"][0], "baseline", baseline_episodes, True)
    screens = {p: evaluate(p, "screen", screen_episodes) for p in state["snapshots"]}
    if pending:
        return wait_for_evaluations()
    r0 = summarize(baseline)
    screen_stats = {p: summarize(rows) for p, rows in screens.items()}
    if state["status"] == "screen":
        paths = state["snapshots"]
        best = max(stats["mean"] for stats in screen_stats.values())
        gap = best - r0["mean"]
        best_stats = max(screen_stats.values(), key=lambda stats: stats["mean"])
        noise = 1.96 * math.hypot(best_stats["standard_error"], r0["standard_error"])
        if gap <= max(noise, 1e-8):
            return fail("insufficient_quality_gap", baseline=r0, screen=screen_stats)
        stable = []
        for index, path in enumerate(paths):
            neighbors = paths[max(0, index - 1):index] + paths[index + 1:index + 2]
            current = screen_stats[path]
            supported = [screen_stats[n] for n in neighbors
                         if abs(current["mean"] - screen_stats[n]["mean"]) <=
                         max(0.15 * gap, 1.96 * math.hypot(current["standard_error"], screen_stats[n]["standard_error"]))]
            if supported and current["mean"] > r0["mean"]:
                score = max(min(current["mean"], neighbor["mean"]) for neighbor in supported)
                stable.append((score, path))
        if not stable:
            return fail("insufficient_quality_gap", detail="no_adjacent_stable_high_candidate", screen=screen_stats)
        stable.sort(reverse=True)
        high_paths = [p for _, p in stable[:3]]
        best = max(screen_stats[p]["mean"] for p in high_paths)
        gap = best - r0["mean"]
        medium_paths = sorted(paths, key=lambda p: abs((screen_stats[p]["mean"] - r0["mean"]) / gap - 0.5))[:4]
        state.update(status="confirm", candidates=list(dict.fromkeys(high_paths + medium_paths)),
                     high_candidates=high_paths, baseline=r0, screen=screen_stats)
        atomic_json(state_path, state)

    confirmations = {p: evaluate(p, "confirm", confirm_episodes) for p in state["candidates"]}
    if pending:
        return wait_for_evaluations()
    if state["status"] == "confirm":
        stats = {p: summarize(rows) for p, rows in confirmations.items()}
        high = max(state["high_candidates"], key=lambda p: stats[p]["mean"])
        gap = stats[high]["mean"] - r0["mean"]
        if gap <= max(1e-8, 1.96 * math.hypot(stats[high]["standard_error"], r0["standard_error"])):
            return fail("insufficient_quality_gap", confirmation=stats)
        mediums = [p for p in stats if p != high and lower <= (stats[p]["mean"] - r0["mean"]) / gap <= upper]
        if not mediums:
            return fail("missing_medium", confirmation=stats,
                        checkpoint_steps_to_densify=[int(Path(p).name) for p in sorted(stats, key=lambda p: abs((stats[p]["mean"] - r0["mean"]) / gap - 0.5))[:2]])
        medium = min(mediums, key=lambda p: abs((stats[p]["mean"] - r0["mean"]) / gap - 0.5))
        difference = paired_difference(confirmations[high], confirmations[medium])
        if difference["ci95"][0] <= 0:
            return fail("insufficient_quality_gap", paired_difference=difference)
        state.update(status="final", high=high, medium=medium, confirmation=stats)
        atomic_json(state_path, state)

    # Candidate choice used confirmation seeds. A separate held-out phase is
    # mandatory before publishing the ONE chosen pair. Never rank on these seeds.
    final_high = evaluate(state["high"], "final", confirm_episodes)
    final_medium = evaluate(state["medium"], "final", confirm_episodes)
    final_baseline = evaluate(state["snapshots"][0], "final", confirm_episodes, True)
    if pending:
        return wait_for_evaluations()
    high_stats, medium_stats, final_r0 = summarize(final_high), summarize(final_medium), summarize(final_baseline)
    gain = paired_difference(final_high, final_baseline)
    difference = paired_difference(final_high, final_medium)
    gap = high_stats["mean"] - final_r0["mean"]
    if gap <= 1e-8 or gain["ci95"][0] <= 0 or difference["ci95"][0] <= 0:
        return fail("insufficient_quality_gap", held_out_gain=gain, paired_difference=difference)
    q = (medium_stats["mean"] - final_r0["mean"]) / gap
    if not lower <= q <= upper:
        return fail("missing_medium", held_out_medium_q=q)
    run = read_json(run_dir / "manifest.json")
    tiers = dict(schema_version="tabx-tiers-1", task_id=run["task_id"], task_hash=run["task_hash"],
                 bank_hash=run["bank_hash"], timeout_rule=run["timeout_rule"], run_id=run["run_id"],
                 train_seed=run["train_seed"], master_seed=run["master_seed"], run=str(run_dir.relative_to(root)),
                 seed_protocol=run["seed_protocol"], provenance=run["provenance"],
                 training_complete=str((run_dir / "training_complete.json").relative_to(root)),
                 training_complete_sha256=completion_hash,
                 selection=options, action_selection="greedy", evaluation_epsilon=0.0,
                 baseline_screen=r0, R0=final_r0, RH=high_stats, paired_difference=difference,
                 high_baseline_difference=gain, uncertainty="normal-approximation 95% CI of per-seed returns/differences",
                 reasons=dict(high="high return with adjacent-stage support and held-out improvement",
                              medium="q within configured range and positive held-out paired gap"),
                 tiers={})
    for name, path, stats, quality in (("medium", state["medium"], medium_stats, q), ("high", state["high"], high_stats, 1.0)):
        manifest = read_json(resolve_under(root, path) / "manifest.json")
        tiers["tiers"][name] = dict(snapshot=path, checkpoint_hash=manifest["agent_sha256"], q=quality, returns=stats)
    atomic_json(tiers_path, tiers)
    state["status"] = "complete"
    atomic_json(state_path, state)
    return dict(status="complete", tiers=str(tiers_path), medium_q=q)
