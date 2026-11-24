"""Seal and resolve MAPPO actor/critic weights at online step zero."""
import hashlib
import json
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
    write_step0_index(directory, batch, entries)


def write_step0_index(directory, batch, entries, *, extend=False):
    index = dict(schema_version="tabx-mappo-step0-1", batch_id=batch["batch_id"], rule=batch["rule"],
                 bank_hash=batch["bank_hash"], entries=entries)
    target = directory / INDEX_NAME
    if target.exists():
        previous = read_json(target)
        if previous == index:
            return
        if (not extend or {**previous, "entries": entries} != index
                or any(entries.get(key) != entry for key, entry in previous["entries"].items())):
            raise ValueError(f"Refusing to replace a sealed step-0 index: {target}")
    atomic_json(target, index)


def step0_record_path(directory, task, seed):
    return directory / "step0" / task / f"{seed}.json"


def publish_step0_record(args, task, path):
    """Publish the original BC weights before the first online rollout."""
    root = Path(args.tabx_output_root)
    rule = args.env_args["timeout_rule"]
    directory = root / "_batches" / rule / args.tabx_batch_id
    target = step0_record_path(directory, task["task_id"], args.seed)
    manifest = _snapshot_entry(path, task, rule, args.seed)
    used_hash = read_json(path.parents[1] / "bc_summary.json")["dataset_manifest_sha256"]
    if used_hash != args.tabx_expected_demo_manifest_sha256 or not used_hash:
        raise ValueError(f"BC manifest SHA-256 mismatch at step 0: {task['task_id']}/{args.seed}")
    if manifest.get("batch_id") != args.tabx_batch_id:
        raise ValueError(f"Step-0 snapshot belongs to a different batch: {path}")
    record = dict(schema_version="tabx-mappo-ready-1", batch_id=args.tabx_batch_id, rule=rule,
                  train_seed=args.seed, **{key: task[key] for key in ("task_id", "task_hash", "bank_hash")},
                  entry=dict(snapshot=str(path.relative_to(root)), dataset_manifest_sha256=used_hash,
                             manifest_sha256=file_sha256(path / "manifest.json"),
                             **{key: manifest[key] for key in WEIGHTS.values()}))
    atomic_json(target, record, replace=False)


def read_step0_record(directory, task, rule, seed, expected_sha256=None):
    path = step0_record_path(directory, task["task_id"], seed)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"Sealed step-0 record SHA-256 mismatch: {path}")
    record = json.loads(data)
    expected = dict(schema_version="tabx-mappo-ready-1", batch_id=directory.name, rule=rule, train_seed=seed,
                    **{key: task[key] for key in ("task_id", "task_hash", "bank_hash")})
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Incompatible step-0 record {key}: {path}")
    return record, digest


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
    record_hash = config.get("tabx_source_step0_sha256")
    if record_hash:
        record, _ = read_step0_record(source, task, rule, seed, record_hash)
        entry = record["entry"]
        source_batch = record["batch_id"]
    elif (index is not None or config.get("tabx_source_index_sha256")
            or (source / INDEX_NAME).is_file() or not (source / "agent.th").is_file()):
        if index is None:
            index, _ = read_source_index(source, rule, task["bank_hash"], config.get("tabx_source_index_sha256"))
        key = f"{task['task_id']}/{seed}"
        if key not in index["entries"]:
            raise ValueError(f"Step-0 index has no checkpoint for {key}: {source}")
        entry = index["entries"][key]
        source_batch = index["batch_id"]
    if entry is not None:
        path = resolve_under(source.parents[2], entry["snapshot"])
        if file_sha256(path / "manifest.json") != entry["manifest_sha256"]:
            raise ValueError(f"Sealed source manifest SHA-256 mismatch: {path}")
    else:
        path = source
    manifest = _snapshot_entry(path, task, rule, seed)
    if entry is not None:
        if manifest.get("batch_id") != source_batch or any(manifest[key] != entry[key] for key in WEIGHTS.values()):
            raise ValueError(f"Source snapshot differs from the sealed index: {path}")
        if record_hash:
            summary = read_json(path.parents[1] / "bc_summary.json")
            if entry["dataset_manifest_sha256"] != summary.get("dataset_manifest_sha256"):
                raise ValueError(f"Source BC summary differs from sealed step-0 record: {path}")
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
    if getattr(args, "tabx_publish_bc_step0", False):
        from tabx_expert_shards import demonstration_directory, read_expert_manifest
        directory = demonstration_directory(args.tabx_demo_root, runner.env.rule,
                                            runner.env.manifest["task_id"], args.tabx_bc_teacher)
        read_expert_manifest(directory, args.tabx_expected_demo_manifest_sha256)
        original = read_json(path.parents[1] / "bc_summary.json")
        if original["dataset_manifest_sha256"] != args.tabx_expected_demo_manifest_sha256:
            raise ValueError("Retried B job demonstration SHA-256 differs from the sealed weights")
        atomic_json(Path(args.tabx_run_dir) / "bc_summary.json",
                    dict(original, bc_updates_this_attempt=0, source_snapshot=str(path)))
    learner.load_initial_weights(path)
    atomic_json(Path(args.tabx_run_dir) / "online_start.json", dict(
        source_snapshot=str(path), source_manifest_sha256=file_sha256(path / "manifest.json"),
        **{key: manifest[key] for key in WEIGHTS.values()}, online_t_env=runner.t_env, bc_updates_this_run=0,
    ))
    return path
