"""Immutable task-bank indexing; no machine-specific paths or TABX import on load."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib
import importlib.machinery
import importlib.util
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


def task_bank_path(bank_path):
    if not bank_path:
        raise ValueError("Specify env_args.task_bank (or --bank) at runtime")
    path = Path(bank_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve(strict=True)


def index_bank(bank_path):
    path = task_bank_path(bank_path)
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


def official_tabx_root():
    return Path(__file__).resolve().parents[2] / "third_party" / "TABX"


OFFICIAL_TABX_URL = "https://github.com/ku-dmlab/TABX.git"
UNIT_FIELDS = {
    "positions": 2, "rotations": 1, "body_weights": 1, "body_radiuss": 1, "teams": 1,
    "pos_min": 2, "pos_max": 2, "unit_ids": 1, "healths": 1, "attack_damages": 1,
    "attack_ranges": 1, "attack_cooldowns": 1, "sight_angles": 1, "is_alive": 1,
    "attack_types": 1, "is_disabled": 1, "speeds": 1,
}
ZONE_FIELDS = {"position": 2, "axes": 2, "zone_type": 1, "effect_value": 1}


def import_tabx(tabx_root=None):
    """Pin the src namespace to the vendored official submodule, ignoring env overrides."""
    root = official_tabx_root().resolve()
    if tabx_root and Path(tabx_root).expanduser().resolve() != root:
        raise ValueError(f"TABX must use the official submodule at {root}; external checkouts are unsupported")
    if not (root / "src/tabx/tabx.py").is_file():
        raise ImportError(f"TABX submodule is not initialized at {root}. Run: git submodule update --init --recursive")
    expected = root / "src/tabx"
    loaded = sys.modules.get("src.tabx")
    if loaded is not None and Path(loaded.__file__).resolve().parent != expected:
        raise ImportError(f"Conflicting src.tabx already loaded: {loaded.__file__}; expected {expected}")
    loaded_src = sys.modules.get("src")
    if loaded_src is not None:
        locations = [Path(p).resolve() for p in getattr(loaded_src, "__path__", [])]
        if locations != [root / "src"]:
            raise ImportError(f"A conflicting src package is already loaded: {locations}. Launch a fresh process with src/main.py or src/tabx_pipeline.py")
    else:
        # The official src directory is a namespace package. Binding its search
        # path explicitly prevents a regular 'src' package elsewhere on sys.path
        # from taking precedence over the pinned checkout.
        spec = importlib.machinery.PathFinder.find_spec("src", [str(root)])
        spec.submodule_search_locations = [str(root / "src")]
        sys.modules["src"] = importlib.util.module_from_spec(spec)
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    module = importlib.import_module("src.tabx")
    if Path(module.__file__).resolve().parent != expected:
        raise ImportError(f"Unexpected src.tabx import: {module.__file__}; expected {expected}")
    return module


def validate_task_structure(task):
    """Validate the unmodified JSON and return its real (ally, enemy, zone) sizes."""
    import numpy as np
    scenario, zones = task.get("scenario"), task.get("zone_scenario")
    if not isinstance(scenario, dict) or set(scenario) != set(UNIT_FIELDS):
        raise ValueError("scenario fields differ from official VectorizedScenario")
    if not isinstance(zones, dict) or set(zones) != set(ZONE_FIELDS) | {"n_zone"}:
        raise ValueError("zone_scenario fields differ from official ZoneScenario")
    teams = np.asarray(scenario["teams"])
    if teams.ndim != 2 or teams.shape[1] != 1 or not np.isin(teams, (0, 1)).all():
        raise ValueError("teams must be [N,1] with values 0 or 1")
    n_ally, n_enemy = int((teams == 0).sum()), int((teams == 1).sum())
    if not (1 <= n_ally <= 10 and 1 <= n_enemy <= 10):
        raise ValueError("Each team must contain between 1 and 10 real units")
    if not np.array_equal(teams[:, 0], np.r_[np.zeros(n_ally), np.ones(n_enemy)]):
        raise ValueError("Preserve original order: all team 0 units must precede team 1 units")
    count = n_ally + n_enemy
    arrays = {}
    for name, width in UNIT_FIELDS.items():
        value = np.asarray(scenario[name])
        if value.shape != (count, width) or not np.isfinite(value).all():
            raise ValueError(f"Invalid finite shape for scenario.{name}: {value.shape}")
        arrays[name] = value
    for name in ("is_alive", "is_disabled"):
        if not np.isin(arrays[name], (0, 1)).all():
            raise ValueError(f"Nonbinary {name}")
    for name, upper in (("unit_ids", 9), ("attack_types", 1)):
        value = arrays[name]
        if not ((value == np.floor(value)) & (value >= 0) & (value <= upper)).all():
            raise ValueError(f"Invalid integer range for {name}")
    for name in ("body_weights", "body_radiuss", "attack_cooldowns"):
        if not (arrays[name] > 0).all():
            raise ValueError(f"{name} must be positive")
    for name in ("healths", "attack_ranges", "speeds"):
        if not (arrays[name] >= 0).all():
            raise ValueError(f"{name} must be nonnegative")
    if not ((arrays["sight_angles"] > 0) & (arrays["sight_angles"] <= 2 * np.pi)).all():
        raise ValueError("sight_angles must be in (0,2pi]")
    if not (arrays["pos_min"] <= arrays["pos_max"]).all():
        raise ValueError("Invalid movement bounds")
    if not ((arrays["positions"] >= arrays["pos_min"]) & (arrays["positions"] <= arrays["pos_max"])).all():
        raise ValueError("Spawn position lies outside movement bounds")
    active = arrays["is_alive"][:, 0].astype(bool) & ~arrays["is_disabled"][:, 0].astype(bool) & (arrays["healths"][:, 0] > 0)
    if not active[:n_ally].any() or not active[n_ally:].any():
        raise ValueError("Both teams require an active unit at reset")
    n_zone = zones["n_zone"]
    if isinstance(n_zone, bool) or not isinstance(n_zone, int) or not 0 <= n_zone <= 10:
        raise ValueError("n_zone must be an integer in [0,10]")
    for name, width in ZONE_FIELDS.items():
        value = np.asarray(zones[name])
        if n_zone == 0 and value.size == 0:
            value = value.reshape(0, width)
        if value.shape != (n_zone, width) or not np.isfinite(value).all():
            raise ValueError(f"Invalid finite shape for zone_scenario.{name}: {value.shape}")
        if name == "axes" and not (value > 0).all():
            raise ValueError("Region axes must be positive")
        if name == "zone_type" and not (np.isin(value, (0, 1, 2, 3))).all():
            raise ValueError("Invalid zone_type")
    # Negative attack damage is healing. Do not clamp or normalize task attributes.
    return n_ally, n_enemy, n_zone


def build_task_params(task, tabx_root=None):
    sizes = validate_task_structure(task)
    import_tabx(tabx_root)
    import jax.numpy as jnp
    from src.tabx.config import TABXConfig
    from src.tabx.scenarios.scenario import VectorizedScenario, ZoneScenario
    from src.tabx.physics.utils import build_batched_physics_params
    from src.tabx.heuristic_policy.utils import build_batched_heuristic_params

    if set(VectorizedScenario.__dataclass_fields__) != set(UNIT_FIELDS):
        raise ValueError("Official scenario schema changed")
    int_fields = {"teams", "unit_ids", "attack_types"}
    bool_fields = {"is_alive", "is_disabled"}
    scenario = VectorizedScenario(**{
        name: jnp.asarray(task["scenario"][name], dtype=jnp.bool_ if name in bool_fields else
                          (jnp.int32 if name in int_fields else jnp.float32))
        for name in UNIT_FIELDS
    })
    n_ally, n_enemy, n_zone = sizes
    zones = ZoneScenario(n_zone=jnp.asarray([[n_zone]], dtype=jnp.int32), **{
        name: jnp.asarray(task["zone_scenario"][name], dtype=jnp.int32 if name == "zone_type" else jnp.float32).reshape(n_zone, width)
        for name, width in ZONE_FIELDS.items()
    })
    return {
        "scenario": scenario, "zone_scenario": zones,
        "physics_params": build_batched_physics_params("default"),
        "heuristic_params": build_batched_heuristic_params("medium"),
    }, TABXConfig(max_n_ally=n_ally, max_n_enemy=n_enemy, max_n_zone=n_zone)
