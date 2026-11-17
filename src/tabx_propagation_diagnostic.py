"""Fixed original-win trajectories, replayed from t=0 with each checkpoint's RNN."""
from pathlib import Path
import copy
import hashlib
import json

import torch as th

from tabx_dataset import episodes_to_batch, read_episode
from tabx_imitation import DemonstrationPool
from tabx_storage import atomic_json, file_sha256, read_json, target_snapshot_info
from tabx_td_success import distance_metrics


def episode_digest(episode):
    digest = hashlib.sha256()
    for group in ("frames", "transitions"):
        for key, value in sorted(episode[group].items()):
            digest.update(f"{group}:{key}:{value.dtype}:{value.shape}".encode())
            digest.update(value.tobytes())
    digest.update(json.dumps(episode["episode"], sort_keys=True).encode())
    return digest.hexdigest()


def fixed_trajectories(pool):
    rows, hashes = [], {}
    for index in pool.wins:
        shard, episode_index, _ = pool.entries[index]
        episode = read_episode(shard, episode_index)
        if shard not in hashes:
            hashes[shard] = file_sha256(shard)
        rows.append(dict(shard=shard.name, episode_index=episode_index, seed=int(episode["episode"]["seed"]),
            length=int(episode["episode"]["length"]), trajectory_sha256=episode_digest(episode),
            shard_sha256=hashes[shard]))
    return dict(schema_version="tabx-propagation-panel-1", source="all_original_teacher_wins",
        manifest_sha256=file_sha256(pool.directory / "manifest.json"), task_hash=pool.metadata["task_hash"],
        trajectories=rows, episodes=len(rows))


def load_diagnostic_models(snapshot, bank, rule, tabx_root=None):
    from tabx_pipeline import build_rollout
    from modules.mixers.qmix import QMixer
    from tabx_online_start import source_inventory
    path, manifest, config, hashes = source_inventory(snapshot)
    if config.get("standardise_returns") or config.get("standardise_rewards"):
        raise ValueError("Propagation diagnostic requires unstandardized rewards/returns")
    runner = build_rollout(path, bank, rule, tabx_root, compatible_source=True)
    mixer = QMixer(runner.args)
    mixer.load_state_dict(th.load(path / "mixer.th", map_location="cpu", weights_only=True))
    mixer.eval()
    status = target_snapshot_info(path, manifest)
    target_mac, target_mixer = None, None
    if status["available"]:
        target_mac, target_mixer = copy.deepcopy(runner.mac), copy.deepcopy(mixer)
        target_mac.agent.load_state_dict(th.load(path / "target_agent.th", map_location="cpu", weights_only=True))
        target_mixer.load_state_dict(th.load(path / "target_mixer.th", map_location="cpu", weights_only=True))
        target_mac.agent.eval()
        target_mixer.eval()
    return runner, mixer, target_mac, target_mixer, dict(snapshot=str(path), manifest=manifest,
        source_hashes=hashes, target_status=status)


def evaluate_trajectory_batch(mac, mixer, target_mac, target_mixer, batch, args):
    mac.init_hidden(batch.batch_size)
    outputs = th.stack([mac.forward(batch, t=t, test_mode=True) for t in range(batch.max_seq_length)], dim=1)
    chosen = outputs[:, :-1].gather(-1, batch["actions"][:, :-1]).squeeze(-1)
    q = mixer(chosen, batch["state"][:, :-1])
    targets = None
    if target_mac is not None:
        target_mac.init_hidden(batch.batch_size)
        future = th.stack([target_mac.forward(batch, t=t, test_mode=True) for t in range(batch.max_seq_length)], dim=1)[:, 1:]
        # Match QLearner's finite sentinel: padded frames can have no legal actions.
        future = future.masked_fill(~batch["avail_actions"][:, 1:].bool(), -9999999)
        if args.double_q:
            greedy = outputs[:, 1:].masked_fill(~batch["avail_actions"][:, 1:].bool(), -th.inf).argmax(-1, keepdim=True)
            values = future.gather(-1, greedy).squeeze(-1)
        else:
            values = future.max(-1).values
        next_q = target_mixer(values, batch["state"][:, 1:])
        targets = batch["reward"][:, :-1] + args.gamma * (1-batch["terminated"][:, :-1].float()) * next_q
    return q, targets


def combine_statistics(rows):
    """Each trajectory appears once, so both step and trajectory counts add."""
    result = {}
    for name in rows[0]:
        steps = sum(row[name]["valid_steps"] for row in rows)
        result[name] = dict(valid_steps=steps, trajectories=sum(row[name]["trajectories"] for row in rows))
        for metric in rows[0][name]:
            if metric in ("valid_steps", "trajectories"):
                continue
            values = [(row[name][metric], row[name]["valid_steps"]) for row in rows if row[name]["valid_steps"]]
            result[name][metric] = (sum(value * count for value, count in values) / steps
                if values and all(value is not None for value, _ in values) else None)
    return result


def selected_checkpoints(args):
    if args.snapshots:
        return [(None, Path(path).expanduser().resolve()) for path in args.snapshots]
    snapshots = sorted((p for p in (Path(args.run).expanduser().resolve() / "checkpoints").iterdir()
                        if p.name.isdigit() and (p / "manifest.json").is_file()), key=lambda p: int(p.name))
    if not snapshots:
        raise ValueError("No published checkpoints found")
    steps = [int(step) for step in args.steps.split(",")]
    if not steps or any(step < 0 for step in steps):
        raise ValueError("Checkpoint steps must be nonnegative")
    return [(step, min(snapshots, key=lambda p: (abs(int(p.name)-step), int(p.name)))) for step in steps]


def diagnose_propagation(args):
    from tabx_pipeline import configure_cpu
    configure_cpu(args.threads, args.cpu_affinity)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoints, results, panel = selected_checkpoints(args), [], None
    panel_path = Path(args.panel).expanduser().resolve() if args.panel else output / "trajectories.json"
    for requested, path in checkpoints:
        runner, mixer, target_mac, target_mixer, metadata = load_diagnostic_models(path, args.bank, args.rule, args.tabx_root)
        try:
            pool = DemonstrationPool(args.demo_dir, runner.env, 0, require_wins=True, target="actions")
            current = fixed_trajectories(pool)
            if panel is not None and current != panel:
                raise ValueError("Propagation checkpoints do not share the same fixed trajectories")
            panel = current
            if panel_path.exists():
                if read_json(panel_path) != panel:
                    raise ValueError("Fixed propagation panel changed: dataset/trajectory checksum mismatch")
            else:
                atomic_json(panel_path, panel)
            per_trajectory = []
            with th.inference_mode():
                for offset in range(0, len(panel["trajectories"]), args.batch_size):
                    entries = panel["trajectories"][offset:offset+args.batch_size]
                    episodes = [read_episode(pool.directory / row["shard"], row["episode_index"]) for row in entries]
                    batch = episodes_to_batch(episodes)[0]
                    q, targets = evaluate_trajectory_batch(runner.mac, mixer, target_mac, target_mixer, batch, runner.args)
                    for i, entry in enumerate(entries):
                        groups = distance_metrics(q[i:i+1], targets[i:i+1] if targets is not None else None,
                            batch["reward"][i:i+1, :-1], batch["transition_mask"][i:i+1, :-1],
                            batch["terminated"][i:i+1, :-1], runner.args.gamma, (0, 1, 0), batch["episode_win"][i:i+1])
                        per_trajectory.append(dict(**entry, statistics=groups["success_td"]))
            results.append(dict(**metadata, requested_step=requested, checkpoint_step=metadata["manifest"]["t_env"],
                gamma=runner.args.gamma, groups=combine_statistics([row["statistics"] for row in per_trajectory]),
                trajectories=per_trajectory, timing="saved_post_optimizer_state; RNN replayed from trajectory start"))
            print(f"propagation checkpoint {path}: {len(per_trajectory)} fixed wins", flush=True)
        finally:
            runner.close_env()
    atomic_json(output / "report.json", dict(schema_version="tabx-propagation-diagnostic-1",
        panel=str(panel_path), panel_sha256=file_sha256(panel_path), checkpoints=results,
        interpretation="Training trajectories measure fit/propagation, not generalization. Behavior returns are not exact current-policy Q targets."))
