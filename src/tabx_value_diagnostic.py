"""Paired one-action forks: predicted QMIX values versus greedy continuation returns."""
from pathlib import Path
import copy
import hashlib
import itertools

import numpy as np
import torch as th

from tabx_dataset import EpisodeRecorder, read_episode, verify_shard, write_shard
from tabx_expert import HeuristicTeacher
from tabx_storage import atomic_json, file_sha256, provenance, read_json, simulator_identity, stable_seed


def discounted_return(rewards, gamma):
    total = 0.
    for reward in reversed(rewards):
        total = float(reward) + gamma * total
    return total


def compare_return_rankings(trajectories, gammas=(.99, .995), tolerance=1e-9):
    rows = [dict(trajectory=row["trajectory"], length=len(row["rewards"]), is_win=row.get("is_win"),
                 returns={str(gamma): discounted_return(row["rewards"], gamma) for gamma in gammas}) for row in trajectories]
    comparisons = []
    for first, second in itertools.combinations(gammas, 2):
        reversals, ties_changed, pairs = [], [], 0
        for a, b in itertools.combinations(rows, 2):
            before = a["returns"][str(first)] - b["returns"][str(first)]
            after = a["returns"][str(second)] - b["returns"][str(second)]
            sign = lambda value: 0 if abs(value) <= tolerance else (1 if value > 0 else -1)
            left, right = sign(before), sign(after)
            pairs += 1
            if left * right < 0:
                reversals.append([a["trajectory"], b["trajectory"]])
            elif left != right:
                ties_changed.append([a["trajectory"], b["trajectory"]])
        comparisons.append(dict(gammas=[first, second], pairs=pairs, reversals=reversals,
                                ties_changed=ties_changed, reversed_pairs=len(reversals)))
    return dict(trajectories=rows, comparisons=comparisons,
                note="Same observed trajectories, different discounted targets; this does not establish a training benefit.")


def _frame(batch, env, timestep):
    batch.update(dict(obs=[env.get_obs()], state=[env.get_state()], avail_actions=[env.get_avail_actions()]), ts=timestep)


def _greedy(outputs, available):
    return outputs.masked_fill(~available.bool(), -th.inf).argmax(-1)


def joint_q(mixer, outputs, state, actions):
    indices = th.as_tensor(actions, device=outputs.device, dtype=th.long).reshape(1, -1, 1)
    chosen = outputs.gather(-1, indices).squeeze(-1)
    return float(mixer(chosen[:, None], state.reshape(1, 1, -1)).item())


def continuation(runner, snapshot, hidden_after_observation, forced_actions, seed, gammas):
    env, mac = runner.env, runner.mac
    env.restore_state(snapshot, branch_seed=seed)
    mac.hidden_states = hidden_after_observation.detach().clone()
    batch = runner.new_batch()
    _frame(batch, env, 0)
    action = th.as_tensor(forced_actions, device=batch.device, dtype=th.long).reshape(1, -1)
    rewards, timestep, ended = [], 0, False
    while not ended:
        batch.update({"actions": action}, ts=timestep)
        _, reward, terminated, truncated, info = env.step(action[0])
        rewards.append(reward)
        ended = terminated or truncated
        timestep += 1
        if not ended:
            _frame(batch, env, timestep)
            outputs = mac.forward(batch, t=timestep, test_mode=True)
            action = _greedy(outputs, batch["avail_actions"][:, timestep])
    return dict(seed=seed, rewards=rewards, remaining_steps=len(rewards), is_win=int(info["is_win"]),
        end_reason=int(info["end_reason"]), elimination_win=int(info["elimination_win"]), timeout_win=int(info["timeout_win"]),
        returns={str(gamma): discounted_return(rewards, gamma) for gamma in gammas})


def candidate_kind(env, actions, advice, previous_contact, include_search=False):
    alive = env.get_frame()["alive"].astype(bool)
    contact = False
    for i, key in enumerate(env.base_env.ally_keys):
        if not alive[i]:
            continue
        other = env.get_native_observations()[key][14:14+(env.base_env.num_agents-1)*16].reshape(-1, 16)
        contact |= bool(np.any((other[:, 0] > 0) & (other[:, 13] == 0)))
    selected = np.r_[np.asarray(actions)[alive], np.asarray(advice)[alive]]
    if np.any(selected == 4):
        return "pre_attack", contact
    if np.isin(selected, [5, 6]).any() and (contact or previous_contact or include_search):
        return "pre_turn" if contact or previous_contact else "search_turn", contact
    if contact and not previous_contact:
        return "contact", contact
    return None, contact


def collect_states(runner, mixer, teacher, args, output):
    reservoir, candidate_count, trajectories = [], 0, []
    rng = np.random.default_rng(args.seed)
    runtime = provenance()
    for episode_index in range(args.episodes):
        seed = stable_seed(runner.env.manifest["task_hash"], args.seed, "value_diagnostic_collect", episode_index)
        runner.reset(episode_seed=seed)
        runner.mac.init_hidden(1)
        teacher.reset(seed)
        recorder = EpisodeRecorder()
        recorder.start(runner.env.get_frame(), seed)
        prefix, total, ended, contact, last_candidate = [], 0., False, False, -args.spacing
        timestep = 0
        while not ended:
            _frame(runner.batch, runner.env, timestep)
            outputs = runner.mac.forward(runner.batch, t=timestep, test_mode=True)
            action = _greedy(outputs, runner.batch["avail_actions"][:, timestep])
            actual = action[0].cpu().numpy()
            _, advice, mask = teacher.actions_and_labels()
            kind, contact = candidate_kind(runner.env, actual, advice, contact, args.include_search_turns)
            if kind and timestep - last_candidate >= args.spacing:
                last_candidate = timestep
                candidate_count += 1
                slot = len(reservoir) if len(reservoir) < args.states else int(rng.integers(candidate_count))
                if slot < args.states:
                    hidden = runner.mac.hidden_states.detach().clone()
                    candidate = dict(episode_seed=seed, state_t=timestep, context=kind, prefix_actions=copy.deepcopy(prefix),
                        model_actions=actual.copy(), teacher_actions=advice.copy(),
                        joint_actions_differ=bool(np.any(actual != advice)),
                        q_model=joint_q(mixer, outputs, runner.batch["state"][:, timestep], actual),
                        q_teacher=joint_q(mixer, outputs, runner.batch["state"][:, timestep], advice),
                        hidden_sha256=hashlib.sha256(hidden.cpu().numpy().tobytes()).hexdigest(),
                        snapshot=runner.env.capture_state(), hidden=hidden)
                    if slot == len(reservoir):
                        reservoir.append(candidate)
                    else:
                        reservoir[slot] = candidate
            runner.batch.update({"actions": action}, ts=timestep)
            _, reward, terminated, truncated, info = runner.env.step(actual)
            ended = terminated or truncated
            recorder.step(actual, reward, ended, runner.env.get_frame(), info, teacher_actions=advice, teacher_mask=mask)
            total += reward
            timestep += 1
            prefix.append(actual.tolist())
        recorder.finish(dict(info, return_value=total))
        metadata = dict(runner.env.manifest, **teacher.metadata(), timeout_rule=runner.env.rule,
                        episode_limit=runner.episode_limit, collection_phase="value_diagnostic_collect",
                        provenance=runtime,
                        source_checkpoint_sha256=file_sha256(Path(args.snapshot) / "agent.th"))
        write_shard(output / "trajectories" / f"{seed:010d}.h5", [recorder.episode], metadata)
        trajectories.append(dict(trajectory=f"diagnostic_episode_{episode_index}", rewards=recorder.episode["transitions"]["reward"][:, 0].tolist(), is_win=int(info["is_win"])))
        print(f"diagnostic collection {episode_index+1}/{args.episodes}, candidate states={candidate_count}", flush=True)
    reservoir.sort(key=lambda row: (row["episode_seed"], row["state_t"]))
    return reservoir, trajectories, candidate_count


def evaluate_forks(runner, candidates, args, gammas):
    rows = []
    for index, candidate in enumerate(candidates):
        model, teacher = [], []
        for repeat in range(args.repeats):
            seed = stable_seed(runner.env.manifest["task_hash"], args.seed, "value_diagnostic_branch", repeat,
                               f"{candidate['episode_seed']}:{candidate['state_t']}")
            for actions, target in ((candidate["model_actions"], model), (candidate["teacher_actions"], teacher)):
                target.append(continuation(runner, candidate["snapshot"], candidate["hidden"], actions, seed, gammas))
        row = {key: value for key, value in candidate.items() if key not in ("snapshot", "hidden", "model_actions", "teacher_actions")}
        row.update(state_id=index, model_actions=candidate["model_actions"].tolist(), teacher_actions=candidate["teacher_actions"].tolist(),
                   model_rollouts=model, teacher_rollouts=teacher, q_gap_teacher_minus_model=candidate["q_teacher"]-candidate["q_model"])
        row["paired_returns"] = {}
        for gamma in gammas:
            delta = np.array([b["returns"][str(gamma)]-a["returns"][str(gamma)] for a,b in zip(model, teacher)])
            row["paired_returns"][str(gamma)] = dict(model_mean=float(np.mean([a["returns"][str(gamma)] for a in model])),
                teacher_mean=float(np.mean([b["returns"][str(gamma)] for b in teacher])), delta_mean=float(delta.mean()),
                delta_std=float(delta.std(ddof=1)) if len(delta)>1 else 0.,
                delta_standard_error=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None,
                deltas=delta.tolist())
        row["model_outcomes"] = dict(wins=sum(r["is_win"] for r in model), elimination_wins=sum(r["elimination_win"] for r in model),
                                    timeout_wins=sum(r["timeout_win"] for r in model))
        row["teacher_outcomes"] = dict(wins=sum(r["is_win"] for r in teacher), elimination_wins=sum(r["elimination_win"] for r in teacher),
                                      timeout_wins=sum(r["timeout_win"] for r in teacher))
        changes = [b["is_win"]-a["is_win"] for a, b in zip(model, teacher)]
        row["paired_outcomes"] = dict(win_deltas=changes, win_delta_mean=float(np.mean(changes)),
            model_loss_teacher_win=changes.count(1), model_win_teacher_loss=changes.count(-1),
            model_remaining_steps_mean=float(np.mean([r["remaining_steps"] for r in model])),
            teacher_remaining_steps_mean=float(np.mean([r["remaining_steps"] for r in teacher])))
        rows.append(row)
        print(f"diagnostic fork {index+1}/{len(candidates)} complete", flush=True)
    return rows


def _existing_trajectories(paths, env):
    trajectories = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            manifest = read_json(path / "manifest.json")
            files = []
            for entry in manifest["shards"]:
                shard = path / entry["path"]
                if shard.parent != path or file_sha256(shard) != entry["sha256"]:
                    raise ValueError("Existing trajectory checksum/path mismatch")
                files.append(shard)
        else:
            files = [path]
        for shard in files:
            report = verify_shard(shard, dict(task_hash=env.manifest["task_hash"], bank_hash=env.manifest["bank_hash"],
                                            timeout_rule=env.rule, episode_limit=env.episode_limit))
            for key, value in simulator_identity().items():
                if report["metadata"].get("provenance", {}).get("tabx", {}).get(key) != value:
                    raise ValueError(f"Existing trajectory simulator mismatch: {key}")
            for index in range(report["episodes"]):
                episode = read_episode(shard, index)
                trajectories.append(dict(trajectory=f"{shard}:{index}", rewards=episode["transitions"]["reward"][:, 0].tolist(),
                                         is_win=episode["episode"]["is_win"]))
    return trajectories


def diagnose_value(args):
    from tabx_pipeline import configure_cpu, build_rollout
    from modules.mixers.qmix import QMixer
    from tabx_online_start import source_inventory, parameter_hash
    configure_cpu(args.threads, args.cpu_affinity)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    path, manifest, config, hashes = source_inventory(args.snapshot)
    if config.get("standardise_returns") or config.get("standardise_rewards"):
        raise ValueError("Value diagnostic currently requires unstandardized rewards/returns")
    if not np.isfinite(args.return_tolerance) or args.return_tolerance < 0:
        raise ValueError("return-tolerance must be finite and nonnegative")
    args.snapshot = str(path)
    gammas = list(dict.fromkeys([.99, .995, float(config["gamma"])]))
    runner = build_rollout(path, args.bank, args.rule, args.tabx_root, compatible_source=True)
    if runner.args.mac != "basic_mac" or runner.args.agent_output_type != "q":
        raise ValueError("Value diagnostic requires a shared basic Q controller")
    mixer = QMixer(runner.args)
    mixer.load_state_dict(th.load(path / "mixer.th", map_location="cpu", weights_only=True))
    mixer.eval()
    teacher = HeuristicTeacher(runner.env, "advanced", 0.)
    try:
        with th.inference_mode():
            candidates, trajectories, count = collect_states(runner, mixer, teacher, args, output)
            forks = evaluate_forks(runner, candidates, args, gammas)
        trajectories.extend(_existing_trajectories(args.trajectories, runner.env))
        trained_gamma = str(float(config["gamma"]))
        better = [row for row in forks if row["paired_returns"][trained_gamma]["delta_mean"] > args.return_tolerance]
        misses = [row for row in better if row["q_gap_teacher_minus_model"] <= args.return_tolerance]
        report = dict(schema_version="tabx-value-diagnostic-1", source_snapshot=str(path), source_manifest=manifest,
            source_hashes=hashes, agent_parameter_sha256=parameter_hash(runner.mac.agent), mixer_parameter_sha256=parameter_hash(mixer),
            trained_gamma=config["gamma"], return_gammas=gammas, requested_states=args.states, sampled_states=len(forks),
            candidate_states=count, branch_repeats=args.repeats, teacher_actual_better_states=len(better),
            teacher_better_but_not_ranked_higher=len(misses), mismatch_fraction=len(misses)/len(better) if better else None,
            states=forks, status="complete" if len(forks)==args.states else "insufficient_candidate_states",
            trajectory_rankings=compare_return_rankings(trajectories, gammas),
            interpretation="The monotone QMIX mixer makes greedy joint Q maximal by construction. "
                "A better teacher first action under paired greedy continuations is evidence of ranking/greedy-policy error, "
                "not proof of a specific cause. Returns estimate this policy's continuation, not optimal Q. "
                "Small repeated samples are noisy. Discount re-ranking alone does not establish training benefit.")
        from tabx_state_panel import cluster_summary
        report["episode_cluster_summary"] = cluster_summary(forks, gammas, args.seed)
        atomic_json(output / "report.json", report)
        print(f"diagnostic report: {output / 'report.json'}", flush=True)
    finally:
        runner.close_env()
