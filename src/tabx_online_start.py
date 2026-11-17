"""Start fresh online QMIX from a sealed BC checkpoint and its original demo pool."""
from pathlib import Path
import hashlib
import json

from tabx_storage import atomic_json, file_sha256, provenance, read_json, simulator_identity, validate_snapshot

ARCHITECTURE = ("mac", "agent", "agent_output_type", "use_rnn", "hidden_dim", "obs_last_action", "obs_agent_id",
                "obs_individual_obs", "mixer", "mixing_embed_dim", "hypernet_layers", "hypernet_embed",
                "mixer_layer_norm", "mixer_layer_norm_eps")


def source_inventory(snapshot):
    path = Path(snapshot).expanduser().resolve()
    manifest, config = read_json(path / "manifest.json"), read_json(path / "config.json")
    hashes = {name: file_sha256(path / name) for name in ("agent.th", "mixer.th", "config.json")}
    for name, key in (("agent.th", "agent_sha256"), ("config.json", "config_sha256"), ("mixer.th", "mixer_sha256")):
        if key in manifest and manifest[key] != hashes[name]:
            raise ValueError(f"Source checkpoint checksum mismatch: {name}")
    from tabx_storage import TARGET_FILES, target_snapshot_info
    target_snapshot_info(path, manifest)
    hashes.update({name: manifest[key] for name, key in TARGET_FILES if manifest.get(key)})
    return path, manifest, config, hashes


def validate_compatible_source(snapshot, task_manifest, rule):
    path, manifest, config, hashes = source_inventory(snapshot)
    validate_snapshot(path, task_manifest, rule, check_runtime=False)
    # EPyMARL authoring code can evolve; simulator semantics and weights cannot drift.
    for key, value in simulator_identity().items():
        if manifest.get("provenance", {}).get("tabx", {}).get(key) != value:
            raise ValueError(f"Source simulator incompatibility: {key}")
    from tabx_storage import runtime_versions
    current = runtime_versions()
    for key in ("torch", "jax", "jaxlib"):
        saved = manifest["provenance"]["versions"][key]
        if str(saved).split("+")[0] != str(current[key]).split("+")[0]:
            raise ValueError(f"Source runtime incompatibility: {key}")
    if config["env"] != "tabx" or config["learner"] != "q_learner" or config["mixer"] != "qmix":
        raise ValueError("Source must be a TABX QMIX checkpoint")
    return path, manifest, config, hashes


def configure_online_start(config):
    source = config.get("tabx_online_source")
    if not source:
        return
    path, manifest, saved, hashes = source_inventory(source)
    if not config.get("tabx_bc") or config.get("checkpoint_path") or config.get("evaluate") or config.get("save_replay"):
        raise ValueError("tabx_online_source requires tabx_bc=True and a fresh online attempt")
    if manifest.get("t_env") != 0 or manifest.get("policy_stage") != "after_bc_before_online":
        raise ValueError("Online source must be the BC/DAgger online-step-0 checkpoint")
    for key in ARCHITECTURE:
        if config.get(key) != saved.get(key):
            raise ValueError(f"Source/current architecture differs: {key}")
    for key, value in saved.items():
        if key.startswith(("tabx_bc_", "tabx_dagger_", "tabx_demo_")) and key not in ("tabx_demo_root", "tabx_bc_eval_episodes"):
            config[key] = value
    # Legacy BC checkpoints explicitly use executed-action targets and no DAgger.
    config["tabx_bc_target"] = saved.get("tabx_bc_target", "actions")
    config["tabx_bc_action_weights"] = saved.get("tabx_bc_action_weights", [1]*8)
    config["tabx_dagger_rounds"] = saved.get("tabx_dagger_rounds", 0)
    if not config.get("tabx_demo_root"):
        config["tabx_demo_root"] = saved.get("tabx_demo_root")
    config["tabx_online_source"] = str(path)
    for name, key in (("agent.th", "tabx_source_agent_sha256"), ("mixer.th", "tabx_source_mixer_sha256")):
        if config.get(key) and config[key] != hashes[name]:
            raise ValueError(f"Sealed online source changed: {name}")
        config[key] = hashes[name]


def pool_inventory(snapshot, demo_root):
    path, manifest, config, _ = source_inventory(snapshot)
    run = path.parent.parent
    summary_path = run / "bc_summary.json"
    summary = read_json(summary_path)
    original = Path(demo_root).expanduser().resolve() / manifest["timeout_rule"] / manifest["task_id"] / config["tabx_bc_teacher"]
    pretraining_run = Path(summary.get("source_pretraining_run", run))
    directories = [original] + [pretraining_run / "dagger" / f"round_{i}" for i in range(1, len(summary.get("dagger_round_manifests", [])) + 1)]
    inventory = []
    for directory in directories:
        data = read_json(directory / "manifest.json")
        digest = file_sha256(directory / "manifest.json")
        if directory == original and digest != summary["dataset_manifest_sha256"]:
            raise ValueError("Original demonstration manifest differs from BC source")
        files = []
        for entry in data["shards"]:
            actual = file_sha256(directory / entry["path"])
            if actual != entry["sha256"]:
                raise ValueError(f"Source pool shard checksum mismatch: {entry['path']}")
            files.append([entry["path"], actual])
        if directory != original:
            metadata = data["metadata"]
            if file_sha256(directory / metadata["collector_checkpoint"]) != metadata["collector_checkpoint_sha256"]:
                raise ValueError("Source DAgger collector changed")
        inventory.append(dict(manifest_sha256=digest, shards=sorted(files)))
    return dict(bc_summary_sha256=file_sha256(summary_path), pools=inventory)


def pool_fingerprint(snapshot, demo_root):
    payload = pool_inventory(snapshot, demo_root)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parameter_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def initialize_online(learner, runner, pool, args):
    import torch as th
    from tabx_imitation import AggregatedDemonstrationPool
    path, manifest, config, hashes = validate_compatible_source(args.tabx_online_source, runner.env.manifest, runner.env.rule)
    for key, name in (("tabx_source_agent_sha256", "agent.th"), ("tabx_source_mixer_sha256", "mixer.th")):
        if hashes[name] != getattr(args, key):
            raise ValueError(f"Online source changed after configuration: {name}")
    if config["env_args"]["max_episode_steps"] != runner.episode_limit:
        raise ValueError("Online source episode limit differs")
    source_run = path.parent.parent
    fingerprint = pool_fingerprint(path, args.tabx_demo_root)
    if args.tabx_source_pool_sha256 and args.tabx_source_pool_sha256 != fingerprint:
        raise ValueError("Sealed source demonstration/DAgger pool changed")
    summary = read_json(source_run / "bc_summary.json")
    pretraining_run = Path(summary.get("source_pretraining_run", source_run))
    if file_sha256(pool.directory / "manifest.json") != summary["dataset_manifest_sha256"]:
        raise ValueError("Original demonstration manifest differs from BC source")
    rounds = len(summary.get("dagger_round_manifests", []))
    if rounds:
        pool = AggregatedDemonstrationPool(pool, args.seed, args.tabx_dagger_fraction)
        for index in range(1, rounds + 1):
            directory = pretraining_run / "dagger" / f"round_{index}"
            data = read_json(directory / "manifest.json")
            pool.add_round([directory / entry["path"] for entry in data["shards"]])
        if pool.dagger_steps != summary["dagger_environment_steps"]:
            raise ValueError("Restored DAgger interaction count differs")
    learner.mac.load_models(str(path))
    learner.mixer.load_state_dict(th.load(path / "mixer.th", map_location=args.device, weights_only=True))
    learner._update_targets_hard()
    if learner.optimiser.state or learner.training_steps or runner.t_env or runner.env._episode_index:
        raise RuntimeError("Online start must have fresh optimizer/replay/counters")
    summary.update(bc_updates_this_run=0, source_pretraining_run=str(pretraining_run),
                   stage="pretrained_weights_before_online", online_t_env=0)
    atomic_json(Path(args.tabx_run_dir) / "bc_summary.json", summary)
    atomic_json(Path(args.tabx_run_dir) / "online_start.json", dict(source_snapshot=str(path),
        source_hashes=hashes, agent_parameter_sha256=parameter_hash(learner.mac.agent),
        source_pool_sha256=fingerprint,
        mixer_parameter_sha256=parameter_hash(learner.mixer), original_manifest_sha256=summary["dataset_manifest_sha256"],
        restored_dagger_rounds=rounds, bc_updates_this_run=0, online_t_env=0,
        optimizer="fresh", replay="fresh", target="hard_synchronized", source_code_match=(
            manifest["provenance"]["epymarl"]["source_hash"] == provenance()["epymarl"]["source_hash"])))
    return pool
