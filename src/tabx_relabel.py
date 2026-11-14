"""Create an immutable deterministic-label version without replaying/mutating source data."""
from pathlib import Path
import copy

import numpy as np

from tabx_dataset import read_episode, verify_shard, write_shard, validate_episode
from tabx_expert import HeuristicTeacher
from tabx_expert_shards import published_expert_shards
from tabx_storage import atomic_json, file_sha256, read_json


def relabel_expert(args):
    from tabx_pipeline import collection_lock, configure_cpu
    configure_cpu(args.threads, args.cpu_affinity)
    from envs.tabx_wrapper import TABXWrapper
    from tabx_storage import simulator_identity, provenance
    source, output = Path(args.source).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Relabel output must be a separate data version")
    manifest_path = source / "manifest.json"
    manifest = read_json(manifest_path)
    env = TABXWrapper(args.task, task_bank=args.bank, timeout_rule=args.rule,
                      tabx_root=args.tabx_root, expected_bank_hash=args.bank_sha256,
                      max_episode_steps=manifest["metadata"]["episode_limit"])
    try:
        for key in ("task_id", "task_hash", "bank_hash"):
            if manifest["metadata"].get(key) != env.manifest[key]:
                raise ValueError(f"Relabel source task mismatch: {key}")
        if manifest["metadata"].get("timeout_rule") != args.rule:
            raise ValueError("Relabel source rule mismatch")
        for key, value in simulator_identity().items():
            if manifest["metadata"].get("provenance", {}).get("tabx", {}).get(key) != value:
                raise ValueError(f"Relabel source simulator mismatch: {key}")
        files = {path.name: path for path in published_expert_shards(source)}
        if set(files) != {entry["path"] for entry in manifest["shards"]}:
            raise ValueError("Source shards differ from manifest; resume source collection first")
        # Verify all source files against the existing authority before publishing any output.
        seen = set()
        for entry in manifest["shards"]:
            if file_sha256(files[entry["path"]]) != entry["sha256"]:
                raise ValueError(f"Source SHA-256 mismatch: {entry['path']}")
            report = verify_shard(files[entry["path"]], {
                key: manifest["metadata"][key] for key in
                ("task_id", "task_hash", "bank_hash", "timeout_rule", "episode_limit", "native_shape", "opponent")})
            for key, value in simulator_identity().items():
                if report["metadata"].get("provenance", {}).get("tabx", {}).get(key) != value:
                    raise ValueError(f"Relabel source shard simulator mismatch: {key}")
            if seen.intersection(report["seeds"]):
                raise ValueError("Duplicate source seeds")
            seen.update(report["seeds"])
        behavior = HeuristicTeacher(env, manifest["metadata"]["teacher"], manifest["metadata"]["teacher_epsilon"])
        teacher = HeuristicTeacher(env, "advanced", 0.)
        metadata = dict(manifest["metadata"])
        metadata.update(behavior.metadata(), label_teacher="advanced", label_epsilon=0., provenance=provenance(),
                        label_teacher_metadata=teacher.metadata(), source_manifest_sha256=file_sha256(manifest_path),
                        source_directory=str(source))
        output.mkdir(parents=True, exist_ok=False)
        with collection_lock(output):
            shards = []
            for entry in manifest["shards"]:
                report = verify_shard(files[entry["path"]])
                episodes = []
                for index in range(report["episodes"]):
                    original = read_episode(files[entry["path"]], index)
                    episode = copy.deepcopy(original)
                    teacher.reset(original["episode"]["seed"])
                    labels, masks = [], []
                    for t in range(original["episode"]["length"]):
                        frame = original["frames"]
                        observations = {key: env.layout.unpack_local(frame["obs"][t, i], i)
                                        for i, key in enumerate(env.base_env.ally_keys)}
                        _, advice, mask = teacher.actions_and_labels(observations=observations,
                            alive=frame["alive"][t], avail_actions=frame["avail_actions"][t])
                        labels.append(advice.reshape(10, 1))
                        masks.append(mask)
                    episode["transitions"].update(teacher_actions=np.asarray(labels, dtype=np.int64),
                                                  teacher_mask=np.asarray(masks, dtype=np.uint8))
                    validate_episode(episode)
                    episodes.append(episode)
                shard_metadata = dict(metadata, source_shard_sha256=entry["sha256"], source_shard=entry["path"])
                path = write_shard(output / entry["path"], episodes, shard_metadata)
                shards.append(dict(path=path.name, sha256=file_sha256(path), source_sha256=entry["sha256"]))
            result = {k: v for k, v in manifest.items() if k not in ("metadata", "shards")}
            atomic_json(output / "manifest.json", dict(result, metadata=metadata, shards=shards))
    finally:
        env.close()
