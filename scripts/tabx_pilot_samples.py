"""Collect paired, greedy final-checkpoint trajectories from two completed TABX batches."""
import argparse
import multiprocessing
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabx_pipeline import build_rollout, collection_lock, configure_cpu, rollout
from tabx_reporting import write_csv
from tabx_storage import completed_checkpoints, file_sha256, read_json, resolve_under, safe_component, stable_seed


def collect_run(args):
    run, bank, count = args
    configure_cpu()
    import h5py
    import torch
    from tabx_dataset import EpisodeRecorder, verify_shard, write_shard

    torch.set_num_threads(1)
    inventory = completed_checkpoints(run)
    if inventory is None or inventory[0]["learner_updates"] < 1:
        raise ValueError(f"No completed learner updates in {run}")
    completion, _, snapshots = inventory
    snapshot = snapshots[-1]
    manifest = read_json(run / "manifest.json")
    rule = manifest["timeout_rule"]
    phase = "pilot_final_sample"
    seeds = [stable_seed(manifest["task_hash"], manifest["master_seed"], phase, i) for i in range(count)]
    metadata = {k: manifest[k] for k in ("task_id", "train_seed", "timeout_rule", "task_hash", "bank_hash",
                                           "run_id", "master_seed", "seed_protocol", "provenance", "native_shape",
                                           "layout_version", "simulator_contract")}
    metadata.update(checkpoint_hash=completion["checkpoints"][-1]["agent_sha256"],
                    checkpoint_step=completion["final_t_env"], phase=phase, action_selection="greedy")
    directory = run / "pilot_samples"
    shard = directory / f"final-{count}.h5"
    with collection_lock(directory):
        if not shard.exists():
            runner = build_rollout(snapshot, bank, rule)
            episodes = []
            try:
                for seed in seeds:
                    recorder = EpisodeRecorder()
                    rollout(runner, seed, recorder)
                    episodes.append(recorder.episode)
                write_shard(shard, episodes, dict(metadata, episode_limit=runner.episode_limit))
            finally:
                runner.close_env()
        if verify_shard(shard, metadata)["seeds"] != seeds:
            raise ValueError(f"Unexpected episode seeds in {shard}")
    with h5py.File(shard) as data:
        rows = data["episodes"]
        timeout = rows["end_reason"][:] == 2
        elimination = (rows["end_reason"][:] == 1) & (rows["is_win"][:] == 1)
        hp_win = elimination | (timeout & (rows["ally_hp_ratio"][:] > rows["enemy_hp_ratio"][:]))
        row = dict(task_id=manifest["task_id"], seed=manifest["train_seed"], trained_with=rule,
                   checkpoint_step=completion["final_t_env"], learner_updates=completion["learner_updates"],
                   episodes=count, loss_win_pct=100 * elimination.mean(), hp_win_pct=100 * hp_win.mean(),
                   pure_timeout_pct=100 * timeout.mean(), shard=str(shard))
    print(f"{rule} task {row['task_id']} seed {row['seed']}  {shard}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if min(args.workers, args.episodes) < 1:
        parser.error("--episodes and --workers must be positive")
    root, bank = args.root.resolve(), args.bank.resolve(strict=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(root / "_jax_cache"))
    batch_id = safe_component(args.batch_id)
    digest = file_sha256(bank)
    jobs, identities = [], []
    for rule in ("hp", "loss"):
        batch = root / "_batches" / rule / batch_id
        identity = read_json(batch / "batch.json")["identity"]
        if identity.pop("rule") != rule or identity["bank_hash"] != digest:
            raise ValueError(f"Wrong task bank or rule for {batch}")
        identities.append(identity)
        state = read_json(batch / "state.json")
        if state["status"] != "complete":
            raise ValueError(f"Finish or resume the batch before sampling: {batch}")
        for job in state["jobs"]:
            run = resolve_under(root, job["attempts"][-1]["run"])
            jobs.append((run, str(bank), args.episodes))
    if identities[0] != identities[1]:
        raise ValueError("Paired batches must match in tasks, seeds, source and training hyperparameters")

    # Recycle the interpreter after each model to release native JAX compilation memory.
    with multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        rows = list(pool.imap(collect_run, jobs))
    output = root / "pilot_comparison" / batch_id / f"final-{args.episodes}.csv"
    write_csv(output, list(rows[0]), rows)
    for rule in ("hp", "loss"):
        subset = [row for row in rows if row["trained_with"] == rule]
        means = [statistics.mean(row[k] for row in subset) for k in ("loss_win_pct", "hp_win_pct", "pure_timeout_pct")]
        print(f"trained_with={rule}  loss_win={means[0]:.2f}%  hp_win={means[1]:.2f}%  pure_timeout={means[2]:.2f}%")
    print(output)


if __name__ == "__main__":
    main()
