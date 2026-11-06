"""Immutable task-bank indexing; no machine-specific paths or TABX import on load."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

TRAIN_SEED_PROTOCOL = "sha256-json-v2:uint32(task_content_hash,train_seed,'train',episode_index)"


def training_episode_seed(content_hash, train_seed, episode_index):
    if int(episode_index) < 0:
        raise ValueError("Training episode_index must be nonnegative")
    payload = [str(content_hash), int(train_seed), "train", int(episode_index)]
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_hash(task):
    payload = {key: task[key] for key in ("scenario", "zone_scenario")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=4)
def _read_bank(path, size, mtime_ns):
    with open(path, encoding="utf-8") as stream:
        bank = json.load(stream)
    if bank.get("schema_version") != "1.0":
        raise ValueError("Expected task-bank schema_version='1.0'")
    tasks = bank.get("tasks")
    if not isinstance(tasks, (list, dict)) or not tasks:
        raise ValueError("Task bank must have nonempty tasks")
    index = {}
    pairs = tasks.items() if isinstance(tasks, dict) else ((t.get("task_id"), t) for t in tasks)
    for key, task in pairs:
        if key is None or str(task.get("task_id", key)) != str(key):
            raise ValueError(f"Missing or inconsistent task identity: {key}")
        key = str(key)
        if key in index:
            raise ValueError(f"Duplicate task_id: {key}")
        index[key] = task
    return bank.get("manifest", {}), index, file_sha256(path)


def index_bank(bank_path):
    if not bank_path:
        raise ValueError("Specify env_args.task_bank (or --bank) at runtime")
    path = Path(bank_path).expanduser().resolve(strict=True)
    stat = path.stat()
    return _read_bank(str(path), stat.st_size, stat.st_mtime_ns)


def load_task(bank_path, task_id, expected_hash=None, expected_bank_hash=None):
    bank_manifest, index, bank_hash = index_bank(bank_path)
    if expected_bank_hash and bank_hash != expected_bank_hash:
        raise ValueError("Task bank SHA-256 mismatch")
    key = str(task_id)
    if key not in index:
        raise KeyError(f"Task {key!r} is not in the task bank")
    task = index[key]
    digest = task_hash(task)
    if expected_hash and digest != expected_hash:
        raise ValueError(f"Task {key} content SHA-256 mismatch")
    # Some merged banks preserve a legacy canonical_hash that differs from the
    # final fields. Keep the source identity AND a checksum of the actual input.
    source_hash = task.get("metadata", {}).get("canonical_hash")
    if source_hash is not None and (len(source_hash) != 64 or
                                   any(c not in "0123456789abcdef" for c in source_hash)):
        raise ValueError(f"Invalid canonical_hash in task {key}")
    manifest = dict(task_id=key, task_hash=digest, canonical_hash=source_hash,
                    canonical_hash_matches=source_hash == digest, bank_hash=bank_hash,
                    bank_manifest=bank_manifest, grid_info=task.get("grid_info", {}))
    return deepcopy(task), manifest


def import_tabx(tabx_root=None):
    """Resolve AnyTABX's `src` before import and refuse a conflicting src package."""
    epymarl_root = Path(__file__).resolve().parents[2]
    # AnyTABX/src is a namespace package, while EPyMARL/src is a regular
    # package: putting AnyTABX first alone does NOT prevent shadowing.
    loaded_src = sys.modules.get("src")
    if loaded_src and getattr(loaded_src, "__file__", None) == str(epymarl_root / "src/__init__.py"):
        raise ImportError("EPyMARL's src package is already loaded; launch the absolute src/main.py or src/tabx_pipeline.py script")
    sys.path[:] = [entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != epymarl_root]
    root = tabx_root or os.environ.get("TABX_ROOT")
    if root:
        root = Path(root).expanduser().resolve(strict=True)
        if not (root / "src/tabx/tabx.py").is_file():
            raise ValueError(f"Not an AnyTABX checkout: {root}")
        sys.path.insert(0, str(root)) if str(root) not in sys.path else None
    try:
        module = importlib.import_module("src.tabx")
    except ImportError as exc:
        raise ImportError("AnyTABX unavailable: set env_args.tabx_root, --tabx-root, "
                          "TABX_ROOT or PYTHONPATH to its checkout root") from exc
    if root and Path(module.__file__).resolve().parent != root / "src/tabx":
        raise ImportError(f"Conflicting src package: loaded {module.__file__}; expected {root}")
    return module


def build_task_params(task, tabx_root=None):
    import_tabx(tabx_root)
    import jax
    import numpy as np
    from src.tabx.sample_task import build_batched_env_params_from_tasks

    converted = deepcopy(task)
    scenario = converted["scenario"]
    teams = np.asarray(scenario["teams"]).reshape(-1)
    if not np.array_equal(teams, np.sort(teams)) or not set(teams).issubset({0, 1}):
        raise ValueError("TABX padding requires team 0 then team 1, with both teams present")
    if int(converted["zone_scenario"]["n_zone"]) != 0:
        params, cfg = build_batched_env_params_from_tasks(
            [converted], physics="default", heuristic="expert",
            max_n_ally=10, max_n_enemy=10, max_n_zone=10)
    else:
        # 70a97a1's JSON converter turns [] into rank one. Use the same
        # constructor helpers with correctly ranked EMPTY regions in this case.
        import jax.numpy as jnp
        from src.tabx.sample_task import validate_task, task_to_structs
        from src.tabx.config import TABXConfig
        from src.tabx.scenarios.utils import generate_padded_unit_scenario, generate_padded_zone_scenario
        from src.tabx.physics.utils import build_batched_physics_params
        from src.tabx.heuristic_policy.utils import build_batched_heuristic_params
        validate_task(converted)
        scenario, zones = task_to_structs(converted)
        zones = zones.replace(position=jnp.empty((0, 2)), axes=jnp.empty((0, 2)),
                              zone_type=jnp.empty((0, 1), dtype=jnp.int32), effect_value=jnp.empty((0, 1)))
        params = {
            "scenario": jax.tree.map(lambda x: x[None], generate_padded_unit_scenario(scenario, 10, 10)),
            "zone_scenario": jax.tree.map(lambda x: x[None], generate_padded_zone_scenario(zones, 10)),
            "physics_params": build_batched_physics_params(["default"], squeeze_when_single_physics=False),
            "heuristic_params": build_batched_heuristic_params(["expert"], squeeze_when_single_heuristic=False),
        }
        cfg = TABXConfig(max_n_ally=10, max_n_enemy=10, max_n_zone=10)
    return jax.tree.map(lambda leaf: leaf[0], params), cfg
