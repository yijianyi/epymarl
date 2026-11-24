"""Published expert shards have exactly a ten-digit seed filename."""
import hashlib
import json
from pathlib import Path
import re


def demonstration_directory(root, rule, task, teacher):
    return Path(root) / rule / task / teacher


def published_expert_shards(directory):
    return sorted(path for path in Path(directory).iterdir()
                  if re.fullmatch(r"[0-9]{10}\.h5", path.name) and path.is_file())


def read_expert_manifest(directory, expected_sha256=None):
    path = directory / "manifest.json"
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Demonstration manifest SHA-256 mismatch for task {directory.parent.name}: "
                         f"expected {expected_sha256}, got {digest} ({path})")
    return json.loads(data), digest


def verified_expert_shards(directory, manifest):
    from tabx_storage import file_sha256

    shards = manifest["shards"]
    files = {path.name: path for path in published_expert_shards(directory)}
    registered = {entry["path"] for entry in shards}
    # A collector can publish more shards while a reader uses a captured manifest.
    if len(registered) != len(shards) or not registered.issubset(files):
        raise ValueError("Demonstration manifest has duplicate or missing shards; rerun collect-expert")
    for entry in shards:
        path = files[entry["path"]]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Demonstration shard SHA-256 mismatch: {path}")
        yield path
