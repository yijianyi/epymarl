"""Official ally heuristic rollouts, using the training wrapper's opponent and rules."""
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
from tabx_expert_shards import published_expert_shards


@lru_cache(maxsize=32)
def _policy(n_units, n_zones):
    import jax
    from src.tabx.heuristic_policy import heuristic_policy
    return jax.jit(lambda key, obs, memory, params, physics:
                   heuristic_policy(key, obs, memory, n_units, n_zones, params, physics))


class HeuristicTeacher:
    def __init__(self, env, name="advanced", epsilon=None):
        import jax.numpy as jnp
        from src.tabx.heuristic_policy.utils import build_batched_heuristic_params
        self.env = env
        self.name = name
        self.params = build_batched_heuristic_params(name)
        if epsilon is not None:
            if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
                raise ValueError("Teacher epsilon must be in [0,1]")
            self.params = self.params.replace(epsilon=jnp.full_like(self.params.epsilon, epsilon))
        self.epsilon = float(np.asarray(self.params.epsilon).item())
        self.policy = _policy(env.base_env.num_agents, env.base_env.max_n_zone)

    def reset(self, seed):
        import jax
        from src.tabx.heuristic_policy import LastVisibleTarget
        # Keep teacher randomness independent of simulator randomness.
        self.key = jax.random.fold_in(jax.random.PRNGKey(seed), 0x54454143)
        self.memory = {key: LastVisibleTarget() for key in self.env.base_env.ally_keys}

    def actions(self):
        import jax
        obs = self.env.get_native_observations()
        alive = self.env.get_frame()["alive"]
        actions = np.full(10, self.env.IDLE, dtype=np.int64)
        for i, key in enumerate(self.env.base_env.ally_keys):
            if not alive[i]:
                continue
            self.key, subkey = jax.random.split(self.key)
            action, self.memory[key] = self.policy(subkey, obs[key], self.memory[key], self.params,
                                                  self.env.params["physics_params"])
            actions[i] = int(np.asarray(action).item())
        if not self.env.get_avail_actions()[np.arange(10), actions].all():
            raise ValueError("Official teacher selected an illegal action")
        return actions

    def metadata(self):
        from envs.tabx_tasks import official_tabx_root
        from tabx_storage import file_sha256
        return dict(teacher=self.name, teacher_epsilon=self.epsilon,
                    teacher_parameter_sha256=file_sha256(official_tabx_root() / "src/tabx/heuristic_policy/parameters" / f"{self.name}.json"),
                    teacher_parameters={key: np.asarray(value).tolist()
                                        for key, value in vars(self.params).items()},
                    opponent="medium", teacher_seed_protocol="fold_in_episode_seed_0x54454143")


def teacher_rollout(env, teacher, seed, recorder=None):
    from envs.tabx_outcome import END_REASONS
    started = time.monotonic()
    env.reset(seed=int(seed))
    teacher.reset(int(seed))
    if recorder is not None:
        recorder.start(env.get_frame(), seed)
    total, length, ended = 0., 0, False
    while not ended:
        actions = teacher.actions()
        _, reward, terminated, truncated, info = env.step(actions)
        ended = terminated or truncated
        total += reward
        length += 1
        if recorder is not None:
            recorder.step(actions, reward, ended, env.get_frame(), info)
    result = dict(seed=int(seed), **info, return_value=total, length=length,
                  end_reason_name=END_REASONS[int(info["end_reason"])],
                  elapsed_seconds=time.monotonic() - started)
    if recorder is not None:
        recorder.finish(result)
    return result


def outcome_summary(rows):
    return dict(episodes=len(rows), wins=sum(int(r["is_win"]) for r in rows),
                win_rate=sum(int(r["is_win"]) for r in rows) / len(rows),
                elimination_wins=sum(int(r["is_win"] and r["end_reason"] == 1) for r in rows),
                timeout_wins=sum(int(r["is_win"] and r["end_reason"] == 2) for r in rows),
                environment_steps=sum(int(r["length"]) for r in rows),
                mean_return=float(np.mean([r["return_value"] for r in rows])))


def collect_expert(args):
    from tabx_pipeline import collection_lock, configure_cpu
    configure_cpu(args.threads, args.cpu_affinity)
    from envs.tabx_wrapper import TABXWrapper
    from tabx_dataset import EpisodeRecorder, read_episode, verify_shard, write_shard
    from tabx_storage import atomic_json, file_sha256, provenance, read_json, safe_component, stable_seed
    import os
    root = Path(args.root).expanduser().resolve()
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(root / "_jax_cache"))
    target = root / args.rule / safe_component(args.task) / args.teacher
    env = TABXWrapper(args.task, task_bank=args.bank, timeout_rule=args.rule, tabx_root=args.tabx_root,
                      max_episode_steps=args.episode_limit, expected_bank_hash=args.bank_sha256)
    try:
        teacher = HeuristicTeacher(env, args.teacher, args.epsilon)
        metadata = dict(env.manifest, **teacher.metadata(), timeout_rule=args.rule,
                        episode_limit=env.episode_limit, master_seed=args.seed,
                        collection_phase="expert_collect", provenance=provenance())
        expected = {key: value for key, value in metadata.items() if key != "provenance"}
        with collection_lock(target):
            complete, rows, shards = set(), [], []
            published = {shard.name: shard for shard in published_expert_shards(target)}
            registered = {}
            manifest_path = target / "manifest.json"
            if manifest_path.exists():
                # The published manifest is the checksum authority. Check every
                # registered file before collecting or rebuilding either JSON.
                for entry in read_json(manifest_path)["shards"]:
                    name = entry["path"]
                    if name in registered:
                        raise ValueError(f"Duplicate registered expert shard: {name}")
                    if name not in published:
                        raise ValueError(f"Missing registered expert shard: {name}")
                    if file_sha256(published[name]) != entry["sha256"]:
                        raise ValueError(f"SHA-256 mismatch for registered expert shard: {name}")
                    registered[name] = entry
                    shards.append(entry)
            for shard in published.values():
                try:
                    verified = verify_shard(shard, expected)
                except (OSError, ValueError) as error:
                    raise ValueError(f"Invalid published expert shard {shard}: {error}") from error
                if complete.intersection(verified["seeds"]):
                    raise ValueError("Duplicate expert seeds across shards")
                complete.update(verified["seeds"])
                for i in range(verified["episodes"]):
                    ep = read_episode(shard, i)["episode"]
                    rows.append(dict(ep, return_value=ep["return"]))
                if shard.name not in registered:
                    shards.append(dict(path=shard.name, sha256=file_sha256(shard)))
            desired = [stable_seed(env.manifest["task_hash"], args.seed, "expert_collect", i)
                       for i in range(args.episodes)]
            missing = [seed for seed in desired if seed not in complete]
            for start in range(0, len(missing), args.shard_episodes):
                episodes = []
                for seed in missing[start:start + args.shard_episodes]:
                    recorder = EpisodeRecorder()
                    row = teacher_rollout(env, teacher, seed, recorder)
                    episodes.append(recorder.episode)
                    rows.append(row)
                    complete.add(seed)
                    print(json.dumps(dict(task=args.task, teacher=args.teacher, epsilon=teacher.epsilon,
                                          collected=len(rows), **row)), flush=True)
                shard = write_shard(target / f"{episodes[0]['episode']['seed']:010d}.h5", episodes, metadata)
                shards.append(dict(path=shard.name, sha256=file_sha256(shard)))
            summary = outcome_summary(rows)
            atomic_json(manifest_path, dict(metadata=metadata, shards=sorted(shards, key=lambda entry: entry["path"]), **summary))
            atomic_json(target / "teacher_summary.json", dict(metadata=teacher.metadata(), **summary,
                        status="ready" if summary["wins"] else "review_teacher_before_bc",
                        next_candidate="expert" if not summary["wins"] and args.teacher == "advanced" else None))
            print(json.dumps(dict(directory=str(target), **summary)), flush=True)
    finally:
        env.close()
