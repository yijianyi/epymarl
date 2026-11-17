"""Portable state panels: replay executed prefixes, never another model's hidden state."""
from pathlib import Path
import hashlib

import numpy as np
import torch as th

from tabx_expert import HeuristicTeacher
from tabx_storage import atomic_json, file_sha256, read_json, simulator_identity, stable_seed
from tabx_value_diagnostic import _frame, _greedy, collect_states, evaluate_forks, joint_q


def state_checksum(env):
    import jax
    digest = hashlib.sha256()
    leaves, structure = jax.tree_util.tree_flatten((env.env_state, env._episode_key, env.t,
                                                   env.first_kill, env._previous_alive))
    digest.update(str(structure).encode())
    for value in leaves:
        value = np.asarray(value)
        digest.update(f"{value.dtype}:{value.shape}".encode())
        digest.update(value.tobytes())
    for name, value in sorted(env.get_frame().items()):
        digest.update(name.encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def action_diagnostics(outputs, available, actions, teacher_actions, frame):
    legal = outputs.masked_fill(~available.bool(), -th.inf)[0].detach().cpu()
    alive = np.asarray(frame["present"]).astype(bool) & np.asarray(frame["alive"]).astype(bool)
    rows = []
    for i in np.flatnonzero(alive):
        ordered = legal[i][available[0, i].cpu().bool()].sort(descending=True).values
        rows.append(dict(agent=int(i), model_action=int(actions[i]), teacher_action=int(teacher_actions[i]),
            differs=bool(actions[i] != teacher_actions[i]),
            greedy_vs_second_q_gap=float(ordered[0]-ordered[1]) if len(ordered) > 1 else None,
            teacher_minus_model_q=float(legal[i, teacher_actions[i]]-legal[i, actions[i]])))
    category = lambda a: "movement" if a < 4 else "attack" if a == 4 else "turn" if a in (5, 6) else "idle"
    return dict(valid_alive_agents=len(rows), differing_agents=sum(row["differs"] for row in rows), agents=rows,
        teacher_corrections={name: sum(row["differs"] and category(row["teacher_action"]) == name for row in rows)
                             for name in ("movement", "attack", "turn", "idle")})


def replay_state(runner, mixer, teacher, entry):
    runner.reset(episode_seed=entry["episode_seed"])
    runner.mac.init_hidden(1)
    teacher.reset(entry["episode_seed"])
    prefix = entry["prefix_actions"]
    if len(prefix) != entry["state_t"]:
        raise ValueError("State panel prefix length differs from state_t")
    for t in range(len(prefix)+1):
        _frame(runner.batch, runner.env, t)
        outputs = runner.mac.forward(runner.batch, t=t, test_mode=True)
        _, advice, _ = teacher.actions_and_labels()
        if t == len(prefix):
            break
        action = np.asarray(prefix[t], dtype=np.int64)
        available = runner.env.get_avail_actions()
        if action.shape != (10,) or np.any((action < 0) | (action >= 8)) or not available[np.arange(10), action].all():
            raise ValueError("Illegal fixed-prefix executed action")
        runner.batch.update({"actions": th.as_tensor(action).reshape(1, -1)}, ts=t)
        _, _, terminated, truncated, _ = runner.env.step(action)
        if terminated or truncated:
            raise ValueError("State panel prefix terminates before the selected decision")
    actual = state_checksum(runner.env)
    if actual != entry["state_sha256"]:
        raise ValueError("Fixed panel simulator/RNG/state checksum mismatch")
    if advice.tolist() != entry["teacher_actions"]:
        raise ValueError("Rebuilt teacher history differs from fixed panel advice")
    actions = _greedy(outputs, runner.batch["avail_actions"][:, t])[0].cpu().numpy()
    hidden = runner.mac.hidden_states.detach().clone()
    return dict(episode_seed=entry["episode_seed"], state_t=t, prefix_actions=prefix,
        panel_state_id=entry["state_id"], state_sha256=actual, source_snapshot=entry["source_snapshot"],
        model_actions=actions, teacher_actions=advice, hidden=hidden, snapshot=runner.env.capture_state(),
        hidden_sha256=hashlib.sha256(hidden.cpu().numpy().tobytes()).hexdigest(),
        action_diagnostics=action_diagnostics(outputs, runner.batch["avail_actions"][:, t], actions, advice, runner.env.get_frame()),
        q_model=joint_q(mixer, outputs, runner.batch["state"][:, t], actions),
        q_teacher=joint_q(mixer, outputs, runner.batch["state"][:, t], advice))


def cluster_summary(rows, gammas, seed, bootstrap_samples=1000):
    """Equal-weight episode clusters; resample whole episodes, not their states."""
    if not rows:
        return dict(episodes=0, states=0, metrics={}, limitation="No sampled states")
    groups = {}
    for row in rows:
        groups.setdefault(row["episode_seed"], []).append(row)
    result, rng = {}, np.random.default_rng(seed)
    functions = {f"return_delta_gamma_{gamma}": lambda row, g=str(gamma): row["paired_returns"][g]["delta_mean"] for gamma in gammas}
    functions.update(q_gap_teacher_minus_model=lambda row: row["q_gap_teacher_minus_model"],
        win_delta=lambda row: np.mean([b["is_win"]-a["is_win"] for a, b in zip(row["model_rollouts"], row["teacher_rollouts"])]))
    for name, value in functions.items():
        means = np.array([np.mean([value(row) for row in episode]) for episode in groups.values()])
        interval = None
        if len(means) > 1:
            resampled = means[rng.integers(len(means), size=(bootstrap_samples, len(means)))].mean(1)
            interval = np.percentile(resampled, [2.5, 97.5]).tolist()
        result[name] = dict(episode_weighted_mean=float(means.mean()), ci95=interval, episode_means=means.tolist())
    return dict(episodes=len(groups), states=len(rows), bootstrap_samples=bootstrap_samples, metrics=result,
        method="Episode-cluster percentile bootstrap; CI unavailable for a single episode cluster.")


def create_value_panel(args):
    from types import SimpleNamespace
    from tabx_pipeline import configure_cpu
    from tabx_propagation_diagnostic import load_diagnostic_models
    configure_cpu(args.threads, args.cpu_affinity)
    panel_path = Path(args.panel).expanduser().resolve()
    if panel_path.exists():
        raise FileExistsError("Refusing to overwrite immutable state panel")
    output = panel_path.parent / (panel_path.stem + "_sources")
    output.mkdir(parents=True, exist_ok=False)
    entries, identity, sources = [], None, []
    for index, snapshot in enumerate(args.snapshots):
        runner, mixer, _, _, metadata = load_diagnostic_models(snapshot, args.bank, args.rule, args.tabx_root)
        try:
            current = dict(task_hash=runner.env.manifest["task_hash"], bank_hash=runner.env.manifest["bank_hash"],
                task_id=runner.env.manifest["task_id"], rule=args.rule, episode_limit=runner.episode_limit,
                simulator=simulator_identity())
            if identity is not None and current != identity:
                raise ValueError("Panel sources have different task/rule/simulator/episode limit")
            identity = current
            collect_args = SimpleNamespace(**dict(vars(args), snapshot=str(snapshot)))
            teacher = HeuristicTeacher(runner.env, "advanced", 0.)
            with th.inference_mode():
                candidates, _, count = collect_states(runner, mixer, teacher, collect_args, output / str(index))
                for candidate in candidates:
                    # Reconstruct the exact collected simulator state to seal it; hidden vectors are never serialized.
                    runner.env.restore_state(candidate["snapshot"])
                    entries.append(dict(state_id=len(entries), episode_seed=candidate["episode_seed"],
                        state_t=candidate["state_t"], context=candidate["context"], prefix_actions=candidate["prefix_actions"],
                        teacher_actions=candidate["teacher_actions"].tolist(), state_sha256=state_checksum(runner.env),
                        source_snapshot=str(Path(snapshot).resolve())))
            sources.append(dict(**metadata, candidate_states=count, selected_states=len(candidates)))
        finally:
            runner.close_env()
    atomic_json(panel_path, dict(schema_version="tabx-value-panel-1", identity=identity, sources=sources,
        collection_seed=args.seed, states=entries, requested_states=args.states*len(args.snapshots),
        status="complete" if len(entries) == args.states*len(args.snapshots) else "insufficient_candidate_states"))
    print(f"fixed value panel: {panel_path} ({len(entries)} states)", flush=True)


def evaluate_value_panel(args):
    from tabx_pipeline import configure_cpu
    from tabx_propagation_diagnostic import load_diagnostic_models
    configure_cpu(args.threads, args.cpu_affinity)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    panel_path = Path(args.panel).expanduser().resolve()
    panel = read_json(panel_path)
    if panel.get("schema_version") != "tabx-value-panel-1":
        raise ValueError("Unsupported state panel schema")
    results = []
    for snapshot in args.snapshots:
        runner, mixer, _, _, metadata = load_diagnostic_models(snapshot, args.bank, args.rule, args.tabx_root)
        try:
            identity = dict(task_hash=runner.env.manifest["task_hash"], bank_hash=runner.env.manifest["bank_hash"],
                task_id=runner.env.manifest["task_id"], rule=args.rule, episode_limit=runner.episode_limit, simulator=simulator_identity())
            if identity != panel["identity"]:
                raise ValueError("Fixed state panel task/rule/simulator/episode limit mismatch")
            teacher = HeuristicTeacher(runner.env, "advanced", 0.)
            gammas = list(dict.fromkeys([.99, .995, float(runner.args.gamma)]))
            fork_args = type(args)(**dict(vars(args), seed=panel["collection_seed"]))
            with th.inference_mode():
                candidates = [replay_state(runner, mixer, teacher, entry) for entry in panel["states"]]
                rows = evaluate_forks(runner, candidates, fork_args, gammas)
            results.append(dict(**metadata, gamma=runner.args.gamma, states=rows,
                cluster_summary=cluster_summary(rows, gammas,
                    stable_seed(identity["task_hash"], panel["collection_seed"], "panel_cluster_bootstrap", 0), args.bootstrap_samples)))
        finally:
            runner.close_env()
    atomic_json(output / "report.json", dict(schema_version="tabx-value-panel-report-1", panel_sha256=file_sha256(panel_path),
        repeats=args.repeats, checkpoints=results, difference_direction="teacher minus model",
        interpretation="Monotone QMIX makes greedy joint Q maximal. Paired gains and Q gaps over time are diagnostic, not proof of a cause; model actions may differ from teacher."))
