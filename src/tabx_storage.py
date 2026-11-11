"""Atomic snapshots and a bounded, transactional single-machine CPU queue."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time
import uuid

from envs.tabx_tasks import TRAIN_SEED_PROTOCOL, OFFICIAL_TABX_URL, file_sha256, import_tabx, official_tabx_root
from envs.tabx_layout import LAYOUT_VERSION, SIMULATOR_CONTRACT

SEED_PROTOCOL = "sha256-json-v1:uint32(task_hash,master_seed,phase,index,stream)"


def stable_seed(task_hash, master_seed, phase, index, stream=""):
    payload = [task_hash, int(master_seed), str(phase), int(index), str(stream)]
    return int.from_bytes(hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).digest()[:4], "big")


def safe_component(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", ".."):
        raise ValueError(f"Unsafe path component: {value!r}")
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def resolve_under(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Path escapes output root: {relative}")
    return target


def code_hash(root, pattern="src/**/*.py"):
    digest = hashlib.sha256()
    root = Path(root)
    for path in sorted(root.glob(pattern)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def git_version(root):
    def call(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
                                           text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return dict(commit=call("rev-parse", "HEAD"), dirty=bool(call("status", "--porcelain")))


def runtime_versions():
    versions = {"python": platform.python_version()}
    for name in ("jax", "jaxlib", "torch", "numpy", "h5py", "flax", "chex"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def simulator_identity():
    root = official_tabx_root().resolve()
    files = ("src/tabx/physics/parameters/default.json",
             "src/tabx/heuristic_policy/parameters/medium.json",
             "src/tabx/heuristic_policy/parameters/advanced.json")
    if not all((root / name).is_file() for name in files):
        raise ImportError("Official TABX submodule missing; run git submodule update --init --recursive")
    return dict(simulator_contract=SIMULATOR_CONTRACT, layout_version=LAYOUT_VERSION,
                repository_url=OFFICIAL_TABX_URL, submodule_commit=git_version(root)["commit"],
                source_hash=code_hash(root, "src/tabx/**/*.py"),
                parameter_json_sha256={name: file_sha256(root / name) for name in files},
                outcome_definition=dict(hp="raw ally_hp_ratio > enemy_hp_ratio; exact tie loses; no rounding",
                                        loss="ally loses pure timeout", priority="post-action elimination before timeout",
                                        official_difference="official TABX rounds HP ratios to 4 decimal places before deciding"))


def ensure_output_contract(root):
    """A migrated simulator starts a fresh output root; old runs stay read-only."""
    root = Path(root).expanduser().resolve()
    expected = simulator_identity()
    marker = root / "tabx_semantics.json"
    if marker.exists():
        if read_json(marker) != expected:
            raise ValueError("Output root uses different TABX semantics/parameters; specify a new output directory")
        return
    if (root / "eval_queue.sqlite3").exists() or any(root.glob("hp/*/*/*/manifest.json")) or any(root.glob("loss/*/*/*/manifest.json")):
        if marker.exists() and read_json(marker) == expected:
            return
        raise ValueError("Existing output contains legacy TABX runs; migration requires a new output directory")
    # Exclusive publish avoids two initial training jobs accepting different
    # simulator identities into the same root.
    root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".semantics-", dir=root)
    os.close(descriptor)
    try:
        atomic_json(temporary, expected)
        try:
            os.link(temporary, marker)
        except FileExistsError:
            if read_json(marker) != expected:
                raise ValueError("Concurrent run initialized a different TABX output contract")
    finally:
        os.unlink(temporary)


def provenance():
    module = import_tabx()
    root = Path(__file__).resolve().parent.parent
    tabx_root = Path(module.__file__).resolve().parents[2]
    return dict(epymarl={**git_version(root), "source_hash": code_hash(root)},
                tabx={**git_version(tabx_root), "source_hash": code_hash(tabx_root, "src/tabx/**/*.py"),
                      **simulator_identity()},
                simulator_contract=SIMULATOR_CONTRACT, layout_version=LAYOUT_VERSION,
                versions=runtime_versions(), imports={"src.tabx": module.__file__,
                                                       "envs": str(root / "src/envs/__init__.py")})


def validate_snapshot(path, task_manifest, rule, *, check_runtime=True):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if manifest.get("simulator_contract") != SIMULATOR_CONTRACT or manifest.get("layout_version") != LAYOUT_VERSION:
        raise ValueError("Snapshot uses legacy TABX simulator/layout semantics; start a new run")
    for key, expected in (("task_id", task_manifest["task_id"]), ("task_hash", task_manifest["task_hash"]),
                          ("bank_hash", task_manifest["bank_hash"]), ("timeout_rule", rule)):
        if manifest[key] != expected:
            raise ValueError(f"Snapshot {key} mismatch: {manifest[key]} != {expected}")
    if file_sha256(path / "agent.th") != manifest["agent_sha256"]:
        raise ValueError("Snapshot agent SHA-256 mismatch")
    if file_sha256(path / "config.json") != manifest["config_sha256"]:
        raise ValueError("Snapshot config SHA-256 mismatch")
    for filename, digest in manifest.get("model_sha256", {}).items():
        if filename not in ("agent.th", "mixer.th", "opt.th") or file_sha256(path / filename) != digest:
            raise ValueError(f"Snapshot model checksum mismatch: {path / filename}")
    from tabx_difficulty import checkpoint_difficulty
    checkpoint_difficulty(path, manifest=manifest)
    if check_runtime:
        current = provenance()
        for library in ("epymarl", "tabx"):
            if manifest["provenance"][library]["source_hash"] != current[library]["source_hash"]:
                raise ValueError(f"{library} source differs from the snapshot")
        for key in ("repository_url", "submodule_commit", "parameter_json_sha256", "outcome_definition"):
            if manifest["provenance"]["tabx"].get(key) != current["tabx"][key]:
                raise ValueError(f"Official TABX provenance differs from snapshot: {key}")
        for library in ("jax", "jaxlib", "torch"):
            saved, local = manifest["provenance"]["versions"][library], current["versions"][library]
            if str(saved).split("+")[0] != str(local).split("+")[0]:
                raise ValueError(f"{library} version differs from the snapshot: {saved} vs {local}")
    return manifest


def prepare_tabx_run(config, task_manifest):
    """Resolve the selected snapshot and effective opponent before any run/env."""
    from tabx_difficulty import DIFFICULTIES, checkpoint_difficulty, requested_difficulty, select_checkpoint
    requested = config["env_args"].get("opponent_difficulty")
    if requested is not None and requested not in DIFFICULTIES:
        raise ValueError("opponent_difficulty must be medium or advanced")
    parent = config.get("tabx_parent_checkpoint")
    if isinstance(parent, str):
        parent = json.loads(parent)
        config["tabx_parent_checkpoint"] = parent
    checkpoint = config.get("checkpoint_path")
    if checkpoint:
        snapshot = select_checkpoint(checkpoint, config.get("load_step", 0))
        config["checkpoint_path"] = str(snapshot)
        if parent:
            identified = checkpoint_difficulty(snapshot)
            if (identified["difficulty"] != "medium" or requested != "advanced"
                    or config.get("evaluate") or config.get("save_replay")):
                raise ValueError(f"checkpoint={snapshot}: curriculum requires medium -> advanced training; "
                                 f"source={identified['difficulty']}, requested={requested}")
            validate_parent_checkpoint(config["tabx_output_root"], parent, task_manifest,
                                       int(config["seed"]), config["env_args"]["timeout_rule"])
            if snapshot != resolve_under(config["tabx_output_root"], parent["snapshot"]):
                raise ValueError(f"checkpoint={snapshot}: curriculum parent path mismatch")
            resolution = dict(difficulty="advanced", source="curriculum_medium_to_advanced",
                              checkpoint_difficulty="medium", evidence=identified["evidence"])
        else:
            resolution = requested_difficulty(snapshot, requested)
        resolution["checkpoint"] = str(snapshot)
    else:
        if parent:
            raise ValueError("Curriculum continuation requires checkpoint_path")
        resolution = dict(difficulty=requested or "medium",
                          source="launch_argument" if requested is not None else "new_run_default", evidence={})
    config["env_args"]["opponent_difficulty"] = resolution["difficulty"]
    config["tabx_difficulty_resolution"] = resolution
    return config


def create_run(config, task_manifest):
    from tabx_reporting import TRAIN_METRICS_VERSION
    root_value = config.get("tabx_output_root")
    if not root_value:
        raise ValueError("Set tabx_output_root to your server's output directory")
    root = Path(root_value).expanduser().resolve()
    ensure_output_contract(root)
    rule = config["env_args"]["timeout_rule"]
    if rule not in ("hp", "loss"):
        raise ValueError("Invalid timeout rule")
    difficulty = config["env_args"].get("opponent_difficulty", "medium")
    if difficulty not in ("medium", "advanced"):
        raise ValueError("opponent_difficulty must be medium or advanced")
    parent = config.get("tabx_parent_checkpoint")
    # Sacred rejects new nested dictionary keys on its CLI; transport the exact
    # identity as JSON, then persist it as structured metadata in the run.
    if isinstance(parent, str):
        parent = json.loads(parent)
        config["tabx_parent_checkpoint"] = parent
    if parent:
        validate_parent_checkpoint(root, parent, task_manifest, int(config["seed"]), rule)
        if difficulty != "advanced" or Path(config["checkpoint_path"]).resolve() != resolve_under(root, parent["snapshot"]):
            raise ValueError("Advanced continuation must load its recorded parent checkpoint")
        if config["t_max"] <= parent["t_env"]:
            raise ValueError("Advanced t_max must exceed the handoff step")
    run_id = config.get("tabx_run_id") or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    relative = Path(rule) / safe_component(task_manifest["task_id"]) / str(int(config["seed"])) / safe_component(run_id)
    path = root / relative
    path.mkdir(parents=True, exist_ok=False)
    config["tabx_run_id"] = run_id
    config["tabx_output_root"] = str(root)
    config["tabx_run_dir"] = str(path)
    config["local_results_path"] = str(path)
    manifest = dict(schema_version="tabx-run-1", run_id=run_id, timeout_rule=rule,
                    opponent_difficulty=difficulty, parent_checkpoint=parent,
                    difficulty_resolution=config.get("tabx_difficulty_resolution"),
                    handoff_t_env=parent["t_env"] if parent else None,
                    training_metrics_version=TRAIN_METRICS_VERSION,
                    task_id=task_manifest["task_id"], task_hash=task_manifest["task_hash"],
                    bank_hash=task_manifest["bank_hash"], task_manifest=task_manifest,
                    train_seed=int(config["seed"]), master_seed=int(config.get("tabx_master_seed", 0)),
                    seed_protocol=SEED_PROTOCOL, provenance=provenance(),
                    train_seed_protocol=TRAIN_SEED_PROTOCOL,
                    simulator_contract=SIMULATOR_CONTRACT, layout_version=LAYOUT_VERSION,
                    native_shape=task_manifest["native_shape"],
                    layout={key: task_manifest[key] for key in ("native_shape", "n_ally", "n_enemy", "n_zone", "unit_slots", "native_obs_size", "native_state_size")},
                    restored_from=config.get("checkpoint_path") or None,
                    restore_semantics="agent_mixer_optimizer_and_t_env; synced_targets; fresh_replay_rng_update_count",
                    deployment={"torch_device": config["device"], "jax_platforms": os.environ.get("JAX_PLATFORMS"),
                                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "pid": os.getpid(),
                                "external_gpu": config.get("tabx_assigned_gpu"),
                                "gpu_name": config.get("tabx_cuda_device_name")})
    if config.get("tabx_batch_id"):
        manifest["batch_id"] = safe_component(config["tabx_batch_id"])
    if config.get("tabx_train_log"):
        log = resolve_under(root, config["tabx_train_log"])
        if not log.is_file():
            raise ValueError("The scheduler must open the training log before launching a job")
        (path / "train.log").symlink_to(os.path.relpath(log, path))
    atomic_json(path / "manifest.json", manifest)
    atomic_json(path / "config.json", config)
    return path


from tabx_queue import EvalQueue


def evaluation_request(root, run_dir, snapshot, phase, count, *, random_policy=False):
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    run = read_json(run_dir / "manifest.json")
    manifest = read_json(Path(snapshot) / "manifest.json")
    request = dict(schema_version="tabx-eval-1", rule=run["timeout_rule"], task_id=run["task_id"],
                task_hash=run["task_hash"], bank_hash=run["bank_hash"], train_seed=run["train_seed"],
                run_id=run["run_id"], run=str(run_dir.relative_to(root)),
                snapshot=str(Path(snapshot).resolve().relative_to(root)),
                checkpoint_hash=manifest["agent_sha256"], phase=phase,
                random_policy=bool(random_policy), action_selection="uniform_legal" if random_policy else "greedy",
                seed_protocol=SEED_PROTOCOL,
                simulator_contract=SIMULATOR_CONTRACT, layout_version=LAYOUT_VERSION,
                official_tabx=run.get("provenance", {}).get("tabx", {}), native_shape=run.get("native_shape"),
                seeds=[stable_seed(run["task_hash"], run["master_seed"], phase, i) for i in range(count)])
    if "opponent_difficulty" in manifest:
        request["opponent_difficulty"] = manifest["opponent_difficulty"]
    if run.get("batch_id"):
        request["batch_id"] = run["batch_id"]
    return request


class SnapshotPublisher:
    def __init__(self, args, env):
        self.args, self.env = args, env
        self.run_dir = Path(args.tabx_run_dir)
        self.root = Path(args.tabx_output_root)
        self.queue = EvalQueue(self.root)
        self.interval = int(args.save_model_interval)
        if self.interval <= 0:
            raise ValueError("save_model_interval must be positive")
        self.last_step = None
        self.next_save = self.interval

    def publish(self, learner, step, *, force_eval=False):
        step = int(step)
        if self.last_step == step:
            return self.run_dir / "checkpoints" / str(step)
        parent = self.run_dir / "checkpoints"
        parent.mkdir(exist_ok=True)
        path = parent / str(step)
        tmp = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=parent))
        try:
            learner.save_models(str(tmp))
            atomic_json(tmp / "config.json", vars(self.args))
            run = read_json(self.run_dir / "manifest.json")
            manifest = dict(schema_version="tabx-snapshot-1", t_env=step,
                            learner_updates=int(learner.training_steps), run_id=run["run_id"],
                            task_id=run["task_id"], task_hash=run["task_hash"], bank_hash=run["bank_hash"],
                            timeout_rule=run["timeout_rule"], train_seed=run["train_seed"],
                            provenance=run["provenance"], agent_sha256=file_sha256(tmp / "agent.th"),
                            config_sha256=file_sha256(tmp / "config.json"))
            manifest["model_sha256"] = {name: file_sha256(tmp / name)
                                         for name in ("agent.th", "mixer.th", "opt.th") if (tmp / name).is_file()}
            if "opponent_difficulty" in run:
                manifest["opponent_difficulty"] = run["opponent_difficulty"]
            manifest["parent_checkpoint"] = run.get("parent_checkpoint")
            manifest["train_seed_protocol"] = TRAIN_SEED_PROTOCOL
            manifest["training_metrics_version"] = run.get("training_metrics_version")
            if run.get("batch_id"):
                manifest["batch_id"] = run["batch_id"]
            manifest["training_episodes"] = getattr(self.env, "_episode_index", None)
            manifest.update(simulator_contract=SIMULATOR_CONTRACT, layout_version=LAYOUT_VERSION,
                            native_shape=run.get("native_shape"), official_tabx=run.get("provenance", {}).get("tabx", {}))
            atomic_json(tmp / "manifest.json", manifest)
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite immutable snapshot {path}")
            os.rename(tmp, path)
        finally:
            if tmp.exists():
                import shutil
                shutil.rmtree(tmp)
        request = evaluation_request(self.root, self.run_dir, path, "screen", self.args.tabx_screen_episodes)
        key = self.queue.enqueue(request)
        atomic_json(self.run_dir / "evaluation_requests" / f"{step}.json",
                    dict(status="queued", request_id=key, checkpoint_step=step))
        self.last_step = step
        self.next_save = (step // self.interval + 1) * self.interval
        return path

    def complete(self, learner, step, summary):
        """Final snapshot first, then one immutable, atomically published inventory."""
        complete_path = self.run_dir / "training_complete.json"
        if complete_path.exists():
            raise FileExistsError(f"Training completion is immutable: {complete_path}")
        self.publish(learner, step, force_eval=True)
        checkpoints = []
        for path in sorted((p for p in (self.run_dir / "checkpoints").iterdir() if p.name.isdigit()),
                           key=lambda p: int(p.name)):
            manifest = read_json(path / "manifest.json")
            if manifest["t_env"] != int(path.name):
                raise ValueError(f"Checkpoint step mismatch: {path}")
            for filename, key in (("agent.th", "agent_sha256"), ("config.json", "config_sha256")):
                if file_sha256(path / filename) != manifest[key]:
                    raise ValueError(f"Checkpoint checksum mismatch: {path / filename}")
            checkpoints.append(dict(t_env=manifest["t_env"], path=str(path.relative_to(self.run_dir)),
                                    agent_sha256=manifest["agent_sha256"], config_sha256=manifest["config_sha256"],
                                    manifest_sha256=file_sha256(path / "manifest.json")))
        if not checkpoints or checkpoints[-1]["t_env"] != int(step):
            raise ValueError("Final checkpoint is missing from the completion inventory")
        run = read_json(self.run_dir / "manifest.json")
        atomic_json(self.run_dir / "training_summary.json", summary)
        complete = {key: run[key] for key in ("task_id", "task_hash", "bank_hash", "timeout_rule", "run_id", "train_seed")}
        complete["opponent_difficulty"] = run.get("opponent_difficulty", "medium")
        complete["parent_checkpoint"] = run.get("parent_checkpoint")
        complete.update(schema_version="tabx-training-complete-1", status="complete", final_t_env=int(step),
                        checkpoints=checkpoints, learner_updates=int(learner.training_steps),
                        train_seed_protocol=TRAIN_SEED_PROTOCOL,
                        summary_sha256=file_sha256(self.run_dir / "training_summary.json"))
        atomic_json(complete_path, complete)
        return complete_path


def completed_checkpoints(run_dir):
    """Validate the producer's sealed checkpoint inventory; never infer candidates."""
    run_dir = Path(run_dir).resolve()
    path = run_dir / "training_complete.json"
    if not path.is_file():
        return None
    complete = read_json(path)
    if complete.get("schema_version") != "tabx-training-complete-1" or complete.get("status") != "complete":
        raise ValueError("Invalid training completion manifest")
    run = read_json(run_dir / "manifest.json")
    for key in ("task_id", "task_hash", "bank_hash", "timeout_rule", "run_id", "train_seed"):
        if complete[key] != run[key]:
            raise ValueError(f"Training completion identity mismatch: {key}")
    if file_sha256(run_dir / "training_summary.json") != complete["summary_sha256"]:
        raise ValueError("Training summary checksum differs from the completion inventory")
    if complete.get("opponent_difficulty", "medium") != run.get("opponent_difficulty", "medium"):
        raise ValueError("Completion opponent difficulty mismatch")
    if complete.get("parent_checkpoint") != run.get("parent_checkpoint"):
        raise ValueError("Completion parent identity mismatch")
    entries = complete["checkpoints"]
    steps = [entry["t_env"] for entry in entries]
    if not steps or steps != sorted(set(steps)) or steps[-1] != complete["final_t_env"]:
        raise ValueError("Invalid completed checkpoint step list")
    snapshots = []
    for entry in entries:
        snapshot = resolve_under(run_dir, entry["path"])
        if snapshot != run_dir / "checkpoints" / str(entry["t_env"]):
            raise ValueError("Completion inventory contains an unexpected checkpoint path")
        for filename, key in (("agent.th", "agent_sha256"), ("config.json", "config_sha256"), ("manifest.json", "manifest_sha256")):
            if file_sha256(snapshot / filename) != entry[key]:
                raise ValueError(f"Completed checkpoint checksum mismatch: {snapshot / filename}")
        manifest = read_json(snapshot / "manifest.json")
        for key in ("task_id", "task_hash", "bank_hash", "timeout_rule", "run_id", "train_seed"):
            if manifest[key] != run[key]:
                raise ValueError(f"Completed checkpoint identity mismatch: {key}")
        if manifest["t_env"] != entry["t_env"] or manifest["agent_sha256"] != entry["agent_sha256"] or manifest["config_sha256"] != entry["config_sha256"]:
            raise ValueError("Checkpoint metadata differs from the completion inventory")
        for filename, digest in manifest.get("model_sha256", {}).items():
            if filename not in ("agent.th", "mixer.th", "opt.th") or file_sha256(snapshot / filename) != digest:
                raise ValueError(f"Completed checkpoint model checksum mismatch: {snapshot / filename}")
        if manifest.get("opponent_difficulty", "medium") != run.get("opponent_difficulty", "medium"):
            raise ValueError("Completed checkpoint opponent difficulty mismatch")
        if manifest.get("parent_checkpoint") != run.get("parent_checkpoint"):
            raise ValueError("Completed checkpoint parent identity mismatch")
        snapshots.append(snapshot)
    actual = {p for p in (run_dir / "checkpoints").iterdir() if p.name.isdigit() and p.is_dir()}
    if actual != set(snapshots):
        raise ValueError("Published checkpoint set differs from the completion inventory")
    return complete, file_sha256(path), snapshots


def final_checkpoint_identity(root, run_dir, inventory=None):
    """Identify a completed run's final snapshot, never directory recency or quality."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    inventory = inventory if inventory is not None else completed_checkpoints(run_dir)
    if inventory is None:
        raise ValueError(f"Missing training completion: {run_dir / 'training_complete.json'}")
    complete, completion_hash, snapshots = inventory
    snapshot = snapshots[-1]
    run = read_json(run_dir / "manifest.json")
    manifest = validate_snapshot(snapshot, run, run["timeout_rule"], check_runtime=False)
    # A continuation needs the whole QMIX state, not only evaluation weights.
    for name in ("agent.th", "mixer.th", "opt.th"):
        digest = manifest.get("model_sha256", {}).get(name)
        if not digest or file_sha256(snapshot / name) != digest:
            raise ValueError(f"Incomplete continuation checkpoint: {snapshot / name}")
    return dict(run=str(run_dir.relative_to(root)), snapshot=str(snapshot.relative_to(root)),
                training_complete_sha256=completion_hash,
                manifest_sha256=file_sha256(snapshot / "manifest.json"), t_env=complete["final_t_env"],
                **{key: manifest[key] for key in ("run_id", "task_id", "task_hash", "bank_hash", "train_seed", "timeout_rule",
                                                  "agent_sha256", "config_sha256", "opponent_difficulty", "model_sha256")})


def validate_parent_checkpoint(root, parent, task_manifest, seed, rule, *, check_runtime=True):
    run = resolve_under(root, parent["run"])
    if final_checkpoint_identity(root, run) != parent:
        raise ValueError(f"Parent final checkpoint identity changed: {run}")
    if parent["train_seed"] != seed or parent["opponent_difficulty"] != "medium":
        raise ValueError("Parent checkpoint must belong to this seed and medium stage")
    return validate_snapshot(resolve_under(root, parent["snapshot"]), task_manifest, rule, check_runtime=check_runtime)
