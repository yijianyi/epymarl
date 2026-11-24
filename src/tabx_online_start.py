"""Seal and resolve MAPPO actor/critic weights at online step zero."""
from pathlib import Path

from tabx_storage import atomic_json, file_sha256, read_json, resolve_under, validate_snapshot

INDEX_NAME = "step0_index.json"
WEIGHTS = {"agent.th": "agent_sha256", "critic.th": "critic_sha256"}
BC_STAGE = "after_bc_before_online"
ARCHITECTURE = ("env", "learner", "mac", "agent", "agent_output_type", "critic_type", "use_rnn",
                "hidden_dim", "obs_last_action", "obs_agent_id", "obs_individual_obs")


def _snapshot_entry(path, task, rule, seed):
    manifest = validate_snapshot(path, task, rule, check_runtime=False)
    if (manifest.get("t_env"), manifest.get("policy_stage"), manifest.get("train_seed")) != (0, BC_STAGE, seed):
        raise ValueError(f"Expected the BC step-0 checkpoint for task {task['task_id']}, seed {seed}: {path}")
    if not manifest.get("critic_sha256"):
        raise ValueError(f"Source step-0 critic is unsealed: {path}")
    return manifest


def seal_step_zero(directory, root, jobs, batch):
    entries = {}
    rule = batch["rule"]
    demo_hashes = batch["identity"]["demo_manifest_sha256"]
    for job in jobs:
        task, seed = job["task_id"], job["train_seed"]
        run = resolve_under(root, job["attempts"][-1]["run"])
        path = run / "checkpoints" / "0"
        manifest = _snapshot_entry(path, dict(task_id=task, task_hash=batch["task_hashes"][task],
                                              bank_hash=batch["bank_hash"]), rule, seed)
        if manifest.get("batch_id") != batch["batch_id"]:
            raise ValueError(f"Step-0 snapshot belongs to a different batch: {path}")
        used_hash = read_json(run / "bc_summary.json").get("dataset_manifest_sha256")
        if used_hash != demo_hashes[task]:
            raise ValueError(f"BC summary demonstration manifest SHA-256 mismatch for task {task}, seed {seed}")
        entries[f"{task}/{seed}"] = dict(snapshot=str(path.relative_to(root)),
                                         manifest_sha256=file_sha256(path / "manifest.json"),
                                         dataset_manifest_sha256=used_hash,
                                         **{key: manifest[key] for key in WEIGHTS.values()})
    index = dict(schema_version="tabx-mappo-step0-1", batch_id=batch["batch_id"], rule=rule,
                 bank_hash=batch["bank_hash"], entries=entries)
    target = directory / INDEX_NAME
    if target.exists():
        if read_json(target) != index:
            raise ValueError(f"Refusing to replace a sealed step-0 index: {target}")
    else:
        atomic_json(target, index)


def read_source_index(source, rule, bank_hash, expected_sha256=None):
    path = Path(source) / INDEX_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Source BC batch has no sealed step-0 index: {path}")
    digest = file_sha256(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"Sealed source index SHA-256 mismatch: {path}")
    index = read_json(path)
    if (index.get("schema_version"), index.get("rule"), index.get("bank_hash")) != (
            "tabx-mappo-step0-1", rule, bank_hash):
        raise ValueError(f"Incompatible source batch index: {path}")
    return index, digest


def resolve_source(config, task, rule, seed, *, index=None):
    source = Path(config["tabx_online_source"])
    entry = None
    if (index is not None or config.get("tabx_source_index_sha256")
            or (source / INDEX_NAME).is_file() or not (source / "agent.th").is_file()):
        if index is None:
            index, _ = read_source_index(source, rule, task["bank_hash"], config.get("tabx_source_index_sha256"))
        key = f"{task['task_id']}/{seed}"
        if key not in index["entries"]:
            raise ValueError(f"Step-0 index has no checkpoint for {key}: {source}")
        entry = index["entries"][key]
        path = resolve_under(source.parents[2], entry["snapshot"])
        if file_sha256(path / "manifest.json") != entry["manifest_sha256"]:
            raise ValueError(f"Sealed source manifest SHA-256 mismatch: {path}")
    else:
        path = source
    manifest = _snapshot_entry(path, task, rule, seed)
    if entry is not None:
        if manifest.get("batch_id") != index["batch_id"] or any(manifest[key] != entry[key] for key in WEIGHTS.values()):
            raise ValueError(f"Source snapshot differs from the sealed index: {path}")
    saved = read_json(path / "config.json")
    if not saved.get("tabx_bc") or saved.get("tabx_online_source") or saved.get("tabx_mix_start"):
        raise ValueError(f"The source must be the original medium BC run: {path}")
    for key in ARCHITECTURE:
        if config.get(key) != saved.get(key):
            raise ValueError(f"Source/current architecture differs: {key}")
    for key in ("max_episode_steps", "world_state_type"):
        if config["env_args"][key] != saved["env_args"][key]:
            raise ValueError(f"Source/current environment differs: {key}")
    return path, manifest


def initialize_online(learner, runner, args):
    path, manifest = resolve_source(vars(args), runner.env.manifest, runner.env.rule, args.seed)
    validate_snapshot(path, runner.env.manifest, runner.env.rule)
    learner.load_initial_weights(path)
    atomic_json(Path(args.tabx_run_dir) / "online_start.json", dict(
        source_snapshot=str(path), source_manifest_sha256=file_sha256(path / "manifest.json"),
        **{key: manifest[key] for key in WEIGHTS.values()}, online_t_env=runner.t_env, bc_updates_this_run=0,
    ))
    return path
