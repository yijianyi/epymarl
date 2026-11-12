"""TABX HDF5 v1: independent frame/transition offsets and EpisodeBatch readback."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from envs.tabx_outcome import END_REASONS
from envs.tabx_layout import TABXLayout, LAYOUT_VERSION, SIMULATOR_CONTRACT

SCHEMA_VERSION = "tabx-hdf5-2"
FRAME_FIELDS = {"obs": (np.float32, (10, 378)), "state": (np.float32, (340,)),
                "avail_actions": (np.uint8, (10, 8)), "present": (np.uint8, (10,)),
                "alive": (np.uint8, (10,))}
STEP_FIELDS = {"actions": (np.int64, (10, 1)), "reward": (np.float32, (1,)),
               "terminated": (np.uint8, (1,))}
EPISODE_FIELDS = {"seed": np.int64, "length": np.int64, "return": np.float64,
                  "is_win": np.uint8, "winner": np.int8, "end_reason": np.uint8,
                  "hit_time_limit": np.uint8, "ally_hp_ratio": np.float32, "enemy_hp_ratio": np.float32}


class EpisodeRecorder:
    def start(self, frame, seed):
        self.seed = int(seed)
        self.frames = {name: [np.asarray(frame[name]).copy()] for name in FRAME_FIELDS}
        self.transitions = {name: [] for name in STEP_FIELDS}
        self.episode = None

    def step(self, actions, reward, terminated, next_frame, info):
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        self.transitions["actions"].append(np.asarray(actions, dtype=np.int64).reshape(10, 1))
        self.transitions["reward"].append([reward])
        self.transitions["terminated"].append([terminated])
        for name in FRAME_FIELDS:
            self.frames[name].append(np.asarray(next_frame[name]).copy())

    def finish(self, info):
        length = len(self.transitions["actions"])
        reason = int(info["end_reason"])
        self.episode = {
            "frames": {name: np.asarray(values, dtype=FRAME_FIELDS[name][0]) for name, values in self.frames.items()},
            "transitions": {name: np.asarray(values, dtype=STEP_FIELDS[name][0]) for name, values in self.transitions.items()},
            "episode": dict(seed=self.seed, length=length, **{"return": info["return_value"]},
                            is_win=int(info["is_win"]), winner=0 if info["is_win"] else (-1 if reason == 3 else 1),
                            end_reason=reason, hit_time_limit=int(info["hit_time_limit"]),
                            ally_hp_ratio=info["ally_hp_ratio"], enemy_hp_ratio=info["enemy_hp_ratio"]),
        }
        validate_episode(self.episode)


def validate_episode(episode, limit=512):
    frames, steps, stats = episode["frames"], episode["transitions"], episode["episode"]
    length = int(stats["length"])
    if not 1 <= length <= limit:
        raise ValueError(f"Invalid episode length {length}")
    for fields, specification, count in ((frames, FRAME_FIELDS, length + 1), (steps, STEP_FIELDS, length)):
        for name, (dtype, shape) in specification.items():
            value = fields[name]
            if value.shape != (count, *shape) or value.dtype != np.dtype(dtype):
                raise ValueError(f"Invalid {name}: {value.shape}/{value.dtype}")
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite {name}")
    for name in ("avail_actions", "present", "alive"):
        if not np.isin(frames[name], [0, 1]).all():
            raise ValueError(f"Nonbinary {name}")
    if np.any(frames["alive"] > frames["present"]):
        raise ValueError("A placeholder cannot be alive")
    if not np.all(frames["avail_actions"].sum(axis=-1) > 0):
        raise ValueError("Empty legal action set")
    inactive = ~frames["alive"].astype(bool)
    inactive_actions = frames["avail_actions"][inactive]
    if not np.all(inactive_actions[:, 7] == 1) or not np.all(inactive_actions.sum(axis=-1) == 1):
        raise ValueError("Inactive slots must allow only IDLE")
    actions = steps["actions"][..., 0]
    if np.any((actions < 0) | (actions >= 8)):
        raise ValueError("Action out of range")
    if not np.take_along_axis(frames["avail_actions"][:-1], actions[..., None], axis=-1).all():
        raise ValueError("Illegal recorded action")
    if not np.all(steps["terminated"][:-1] == 0) or steps["terminated"][-1, 0] != 1:
        raise ValueError("Exactly the final transition must be terminal")
    if not np.isclose(steps["reward"].sum(dtype=np.float64), stats["return"], rtol=1e-6, atol=2e-5):
        raise ValueError("Return does not equal the sum of rewards")
    if int(stats["end_reason"]) not in (1, 2, 3):
        raise ValueError("Unknown terminal end reason")
    if int(stats["winner"]) != (0 if stats["is_win"] else (-1 if stats["end_reason"] == 3 else 1)):
        raise ValueError("Inconsistent winner")
    return length


def trajectory_hash(episode):
    digest = hashlib.sha256()
    for name in ("obs", "state"):
        digest.update(episode["frames"][name].tobytes())
    digest.update(episode["transitions"]["actions"].tobytes())
    return digest.hexdigest()


def shard_layout(metadata):
    if metadata.get("layout_version") != LAYOUT_VERSION or metadata.get("simulator_contract") != SIMULATOR_CONTRACT:
        raise ValueError("HDF5 uses legacy or unspecified TABX layout/simulator semantics")
    if "native_shape" not in metadata:
        raise ValueError("HDF5 must record the real native_shape")
    layout = TABXLayout(*metadata["native_shape"])
    for name in ("n_ally", "n_enemy", "n_zone"):
        if name in metadata and metadata[name] != getattr(layout, name):
            raise ValueError(f"Inconsistent shard metadata: {name}")
    return layout


def write_shard(path, episodes, metadata):
    path = Path(path)
    metadata = dict(metadata)
    metadata.update(shard_layout(metadata).metadata())
    if not episodes:
        raise ValueError("Cannot publish an empty shard")
    lengths = [validate_episode(ep, metadata.get("episode_limit", 512)) for ep in episodes]
    if len({ep["episode"]["seed"] for ep in episodes}) != len(episodes):
        raise ValueError("Duplicate seeds in shard")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, tmp = tempfile.mkstemp(prefix=".shard-", suffix=".h5", dir=path.parent)
    os.close(descriptor)
    try:
        with h5py.File(tmp, "w") as data:
            data.attrs["schema_version"] = SCHEMA_VERSION
            data.attrs["metadata"] = json.dumps(metadata, sort_keys=True, allow_nan=False)
            for group, fields in (("frames", FRAME_FIELDS), ("transitions", STEP_FIELDS)):
                for name in fields:
                    values = np.concatenate([ep[group][name] for ep in episodes], axis=0)
                    data.create_dataset(f"{group}/{name}", data=values, chunks=True, compression="gzip", compression_opts=4,
                                        shuffle=True, fletcher32=True)
            data.create_dataset("episodes/frame_offsets", data=np.r_[0, np.cumsum(np.array(lengths) + 1)].astype(np.int64))
            data.create_dataset("episodes/step_offsets", data=np.r_[0, np.cumsum(lengths)].astype(np.int64))
            for name, dtype in EPISODE_FIELDS.items():
                data.create_dataset(f"episodes/{name}", data=np.asarray([ep["episode"][name] for ep in episodes], dtype=dtype))
            data.create_dataset("episodes/trajectory_hash", data=np.asarray([trajectory_hash(ep).encode() for ep in episodes], dtype="S64"))
            data.flush()
        verify_shard(tmp)
        # A hard link publishes without overwriting a file from another writer.
        os.link(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def read_episode(path, index):
    with h5py.File(path, "r") as data:
        fo, so = data["episodes/frame_offsets"], data["episodes/step_offsets"]
        if not 0 <= index < len(fo) - 1:
            raise IndexError(index)
        result = {
            "frames": {name: data[f"frames/{name}"][fo[index]:fo[index + 1]] for name in FRAME_FIELDS},
            "transitions": {name: data[f"transitions/{name}"][so[index]:so[index + 1]] for name in STEP_FIELDS},
            "episode": {name: data[f"episodes/{name}"][index].item() for name in EPISODE_FIELDS},
        }
    return result


def verify_shard(path, expected=None):
    with h5py.File(path, "r") as data:
        if data.attrs["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unknown HDF5 schema")
        metadata = json.loads(data.attrs["metadata"])
        layout = shard_layout(metadata)
        if expected:
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(f"Shard metadata mismatch: {key}")
        fo, so = data["episodes/frame_offsets"][:], data["episodes/step_offsets"][:]
        lengths = data["episodes/length"][:]
        if (fo.dtype != np.int64 or so.dtype != np.int64 or len(fo) != len(lengths) + 1 or
                len(so) != len(fo) or fo[0] != 0 or so[0] != 0 or
                not np.array_equal(np.diff(fo), lengths + 1) or not np.array_equal(np.diff(so), lengths)):
            raise ValueError("Invalid independent frame/step offsets")
        for group, fields, count in (("frames", FRAME_FIELDS, fo[-1]), ("transitions", STEP_FIELDS, so[-1])):
            for name, (dtype, shape) in fields.items():
                dataset = data[f"{group}/{name}"]
                if dataset.shape != (count, *shape) or dataset.dtype != np.dtype(dtype):
                    raise ValueError(f"Invalid dataset {group}/{name}")
        seeds = data["episodes/seed"][:].tolist()
        hashes = data["episodes/trajectory_hash"][:].astype(str).tolist()
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seeds")
        for name, dtype in EPISODE_FIELDS.items():
            if data[f"episodes/{name}"].shape != (len(lengths),) or data[f"episodes/{name}"].dtype != np.dtype(dtype):
                raise ValueError(f"Invalid episode dataset {name}")
    for index in range(len(lengths)):
        episode = read_episode(path, index)
        validate_episode(episode, metadata.get("episode_limit", 512))
        expected_present = np.broadcast_to(layout.present[:10], episode["frames"]["present"].shape)
        if not np.array_equal(episode["frames"]["present"], expected_present):
            raise ValueError("Present masks differ from the native roster")
        if trajectory_hash(episode) != hashes[index]:
            raise ValueError("Trajectory checksum mismatch")
    return dict(episodes=len(lengths), steps=int(so[-1]), frames=int(fo[-1]), seeds=seeds,
                trajectory_hashes=hashes, metadata=metadata, bytes=Path(path).stat().st_size)


def read_episode_batch(path, indices=None, device="cpu"):
    """Return (batch, lengths). Only the first lengths[i] actions are valid.

    `filled` includes the terminal FRAME, as in EpisodeRunner. The additional
    transition_mask makes it impossible to mistake its unexecuted action for data.
    """
    with h5py.File(path, "r") as data:
        if data.attrs["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Legacy TABX HDF5 requires its original layout reader")
        shard_layout(json.loads(data.attrs["metadata"]))
        count = len(data["episodes/length"])
    indices = list(range(count)) if indices is None else list(indices)
    episodes = [read_episode(path, index) for index in indices]
    return episodes_to_batch(episodes, device)


def episodes_to_batch(episodes, device="cpu"):
    """Build a recurrent batch from episodes, including executed-action masks."""
    import torch as th
    from components.episode_buffer import EpisodeBatch
    from components.batch_schema import build_batch_schema
    if not episodes:
        raise ValueError("No episodes selected")
    lengths = np.array([ep["episode"]["length"] for ep in episodes], dtype=np.int64)
    env_info = dict(n_agents=10, n_actions=8, obs_shape=378, state_shape=340)
    scheme, groups, preprocess = build_batch_schema(env_info)
    scheme["transition_mask"] = {"vshape": (1,), "dtype": th.uint8}
    for name in ("present", "alive"):
        scheme[name] = {"vshape": (1,), "group": "agents", "dtype": th.uint8}
    batch = EpisodeBatch(scheme, groups, len(episodes), int(max(lengths)) + 1, preprocess=preprocess, device=device)
    for index, ep in enumerate(episodes):
        length = int(lengths[index])
        batch.update({name: th.as_tensor(ep["frames"][name][None], device=device,
                                         dtype=batch.scheme[name].get("dtype", th.float32))
                      for name in FRAME_FIELDS},
                     bs=slice(index, index + 1), ts=slice(0, length + 1))
        values = {name: ep["transitions"][name][None] for name in STEP_FIELDS}
        values["transition_mask"] = np.ones((1, length, 1), dtype=np.uint8)
        values = {name: th.as_tensor(value, device=device, dtype=batch.scheme[name].get("dtype", th.float32))
                  for name, value in values.items()}
        batch.update(values, bs=slice(index, index + 1), ts=slice(0, length), mark_filled=False)
    return batch, lengths
