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
import sqlite3
import subprocess
import tempfile
import time
import uuid

from envs.tabx_tasks import TRAIN_SEED_PROTOCOL, file_sha256

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


def provenance():
    import src.tabx
    root = Path(__file__).resolve().parent.parent
    tabx_root = Path(src.tabx.__file__).resolve().parents[2]
    return dict(epymarl={**git_version(root), "source_hash": code_hash(root)},
                tabx={**git_version(tabx_root), "source_hash": code_hash(tabx_root, "src/tabx/**/*.py")},
                versions=runtime_versions(), imports={"src.tabx": src.tabx.__file__,
                                                       "envs": str(root / "src/envs/__init__.py")})


def validate_snapshot(path, task_manifest, rule, *, check_runtime=True):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    for key, expected in (("task_id", task_manifest["task_id"]), ("task_hash", task_manifest["task_hash"]),
                          ("bank_hash", task_manifest["bank_hash"]), ("timeout_rule", rule)):
        if manifest[key] != expected:
            raise ValueError(f"Snapshot {key} mismatch: {manifest[key]} != {expected}")
    if file_sha256(path / "agent.th") != manifest["agent_sha256"]:
        raise ValueError("Snapshot agent SHA-256 mismatch")
    if file_sha256(path / "config.json") != manifest["config_sha256"]:
        raise ValueError("Snapshot config SHA-256 mismatch")
    if check_runtime:
        current = provenance()
        for library in ("epymarl", "tabx"):
            if manifest["provenance"][library]["source_hash"] != current[library]["source_hash"]:
                raise ValueError(f"{library} source differs from the snapshot")
        for library in ("jax", "jaxlib", "torch"):
            saved, local = manifest["provenance"]["versions"][library], current["versions"][library]
            if str(saved).split("+")[0] != str(local).split("+")[0]:
                raise ValueError(f"{library} version differs from the snapshot: {saved} vs {local}")
    return manifest


def create_run(config, task_manifest):
    root_value = config.get("tabx_output_root")
    if not root_value:
        raise ValueError("Set tabx_output_root to your server's output directory")
    root = Path(root_value).expanduser().resolve()
    rule = config["env_args"]["timeout_rule"]
    if rule not in ("hp", "loss"):
        raise ValueError("Invalid timeout rule")
    run_id = config.get("tabx_run_id") or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    relative = Path(rule) / safe_component(task_manifest["task_id"]) / str(int(config["seed"])) / safe_component(run_id)
    path = root / relative
    path.mkdir(parents=True, exist_ok=False)
    config["tabx_run_id"] = run_id
    config["tabx_output_root"] = str(root)
    config["tabx_run_dir"] = str(path)
    config["local_results_path"] = str(path)
    manifest = dict(schema_version="tabx-run-1", run_id=run_id, timeout_rule=rule,
                    task_id=task_manifest["task_id"], task_hash=task_manifest["task_hash"],
                    bank_hash=task_manifest["bank_hash"], task_manifest=task_manifest,
                    train_seed=int(config["seed"]), master_seed=int(config.get("tabx_master_seed", 0)),
                    seed_protocol=SEED_PROTOCOL, provenance=provenance(),
                    train_seed_protocol=TRAIN_SEED_PROTOCOL,
                    restored_from=config.get("checkpoint_path") or None,
                    restore_semantics="new_attempt_with_fresh_replay_targets_rng_and_update_count",
                    deployment={"torch_device": config["device"], "jax_platforms": os.environ.get("JAX_PLATFORMS"),
                                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                "omp_num_threads": os.environ.get("OMP_NUM_THREADS")})
    atomic_json(path / "manifest.json", manifest)
    atomic_json(path / "config.json", config)
    return path


class EvalQueue:
    """SQLite queue, intended for one host/local disk (no NFS lock assumptions).

    Expired leases can be reclaimed. An owner token fences late writers; results
    are committed per episode, so retrying a request replays only missing seeds.
    """
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "eval_queue.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, rule TEXT NOT NULL, payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', owner TEXT, updated REAL,
                    attempts INTEGER NOT NULL DEFAULT 0, error TEXT);
                CREATE TABLE IF NOT EXISTS episodes (
                    request_id TEXT NOT NULL, seed INTEGER NOT NULL, result TEXT NOT NULL,
                    PRIMARY KEY(request_id, seed));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(str(self.path), timeout=60)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def request_id(payload):
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def enqueue(self, payload, max_pending=None):
        key = self.request_id(payload)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT id FROM requests WHERE id=?", (key,)).fetchone():
                return key
            if max_pending is not None:
                count = db.execute("SELECT COUNT(*) FROM requests WHERE rule=? AND status IN ('pending','running')",
                                   (payload["rule"],)).fetchone()[0]
                if count >= max_pending:
                    return None
            db.execute("INSERT INTO requests(id, rule, payload, updated) VALUES(?,?,?,?)",
                       (key, payload["rule"], json.dumps(payload, sort_keys=True), time.time()))
        return key

    def claim(self, rule, lease_seconds=3600):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT * FROM requests WHERE rule=? AND
                (status='pending' OR (status='running' AND updated<?)) ORDER BY rowid LIMIT 1""",
                             (rule, time.time() - lease_seconds)).fetchone()
            if row is None:
                return None
            owner = uuid.uuid4().hex
            db.execute("UPDATE requests SET status='running', owner=?, updated=?, attempts=attempts+1, error=NULL WHERE id=?",
                       (owner, time.time(), row["id"]))
            return row["id"], owner, json.loads(row["payload"])

    def heartbeat(self, key, owner):
        with self.connect() as db:
            changed = db.execute("UPDATE requests SET updated=? WHERE id=? AND owner=? AND status='running'",
                                 (time.time(), key, owner)).rowcount
        if not changed:
            raise RuntimeError("Evaluation lease lost")

    def record(self, key, owner, result):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT id FROM requests WHERE id=? AND owner=? AND status='running'",
                              (key, owner)).fetchone():
                raise RuntimeError("Evaluation lease lost")
            db.execute("INSERT OR IGNORE INTO episodes VALUES(?,?,?)",
                       (key, int(result["seed"]), json.dumps(result, allow_nan=False)))
            db.execute("UPDATE requests SET updated=? WHERE id=?", (time.time(), key))

    def finish(self, key, owner, error=None):
        with self.connect() as db:
            db.execute("UPDATE requests SET status=?, error=?, updated=? WHERE id=? AND owner=?",
                       ("failed" if error else "complete", error, time.time(), key, owner))

    def results(self, key):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT result FROM episodes WHERE request_id=? ORDER BY seed", (key,))]

    def status(self, key):
        with self.connect() as db:
            row = db.execute("SELECT status, error FROM requests WHERE id=?", (key,)).fetchone()
            return dict(row) if row else None

    def retry_failed(self, rule):
        with self.connect() as db:
            return db.execute("UPDATE requests SET status='pending', owner=NULL WHERE rule=? AND status='failed'", (rule,)).rowcount


def evaluation_request(root, run_dir, snapshot, phase, count, *, random_policy=False):
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    run = read_json(run_dir / "manifest.json")
    manifest = read_json(Path(snapshot) / "manifest.json")
    return dict(schema_version="tabx-eval-1", rule=run["timeout_rule"], task_id=run["task_id"],
                task_hash=run["task_hash"], bank_hash=run["bank_hash"], train_seed=run["train_seed"],
                run_id=run["run_id"], run=str(run_dir.relative_to(root)),
                snapshot=str(Path(snapshot).resolve().relative_to(root)),
                checkpoint_hash=manifest["agent_sha256"], phase=phase,
                random_policy=bool(random_policy), action_selection="uniform_legal" if random_policy else "greedy",
                seed_protocol=SEED_PROTOCOL,
                seeds=[stable_seed(run["task_hash"], run["master_seed"], phase, i) for i in range(count)])


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
            if force_eval:
                path = self.run_dir / "checkpoints" / str(step)
                request = evaluation_request(self.root, self.run_dir, path, "screen", self.args.tabx_screen_episodes)
                key = self.queue.enqueue(request)
                atomic_json(self.run_dir / "evaluation_requests" / f"{step}.json", dict(status="queued", request_id=key))
            return
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
            manifest["train_seed_protocol"] = TRAIN_SEED_PROTOCOL
            manifest["training_episodes"] = getattr(self.env, "_episode_index", None)
            atomic_json(tmp / "manifest.json", manifest)
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite immutable snapshot {path}")
            os.rename(tmp, path)
        finally:
            if tmp.exists():
                import shutil
                shutil.rmtree(tmp)
        request = evaluation_request(self.root, self.run_dir, path, "screen", self.args.tabx_screen_episodes)
        key = self.queue.enqueue(request, None if force_eval else int(self.args.tabx_max_pending_evals))
        atomic_json(self.run_dir / "evaluation_requests" / f"{step}.json",
                    dict(status="queued" if key else "skipped_backpressure", request_id=key))
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
        snapshots.append(snapshot)
    actual = {p for p in (run_dir / "checkpoints").iterdir() if p.name.isdigit() and p.is_dir()}
    if actual != set(snapshots):
        raise ValueError("Published checkpoint set differs from the completion inventory")
    return complete, file_sha256(path), snapshots
