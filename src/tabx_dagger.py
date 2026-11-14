"""Greedy learner rollouts with epsilon-zero advice and immutable round provenance."""
import copy
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch as th

from tabx_dataset import EpisodeRecorder, write_shard
from tabx_expert import HeuristicTeacher, outcome_summary
from tabx_storage import atomic_json, file_sha256, provenance, stable_seed


@contextmanager
def isolated_rollouts(runner):
    """Evaluation/DAgger must not advance online RNG, counters or logging state."""
    numpy_rng, torch_rng = np.random.get_state(), th.random.get_rng_state()
    cuda_rng = th.cuda.get_rng_state_all() if th.cuda.is_initialized() else None
    returns, stats, log_t = runner.test_returns, runner.test_stats, runner.log_train_stats_t
    mac, t_env = runner.mac, runner.t_env
    epsilon, training = mac.action_selector.epsilon, mac.agent.training
    runner.test_returns, runner.test_stats = [], {}
    try:
        yield
    finally:
        np.random.set_state(numpy_rng)
        th.random.set_rng_state(torch_rng)
        if cuda_rng is not None:
            th.cuda.set_rng_state_all(cuda_rng)
        runner.mac, runner.t_env = mac, t_env
        mac.agent.train(training)
        mac.action_selector.epsilon = epsilon
        runner.test_returns, runner.test_stats, runner.log_train_stats_t = returns, stats, log_t


def collect_dagger_round(learner, runner, pool, args, round_index):
    target = Path(args.tabx_run_dir) / "dagger" / f"round_{round_index}"
    target.mkdir(parents=True, exist_ok=False)
    checkpoint = target / "collector"
    checkpoint.mkdir()
    learner.mac.save_models(str(checkpoint))
    checkpoint_hash = file_sha256(checkpoint / "agent.th")
    seeds = [stable_seed(runner.env.manifest["task_hash"], args.seed, "dagger_collect", i, str(round_index))
             for i in range(args.tabx_dagger_episodes)]
    holdout = {stable_seed(runner.env.manifest["task_hash"], args.tabx_master_seed, "bc_holdout", i)
               for i in range(args.tabx_bc_eval_episodes)}
    if len(set(seeds)) != len(seeds) or (pool.seeds | holdout).intersection(seeds):
        raise ValueError("DAgger seeds overlap existing collection/evaluation seeds")
    teacher = HeuristicTeacher(runner.env, "advanced", 0.)
    metadata = dict(runner.env.manifest, **teacher.metadata(), episode_limit=runner.episode_limit,
                    timeout_rule=runner.env.rule, collection_phase="dagger_collect", dagger_round=round_index,
                    collector_checkpoint_sha256=checkpoint_hash, collector_checkpoint="collector/agent.th",
                    action_selection="greedy", seed_phase="dagger_collect", train_seed=args.seed,
                    provenance=provenance())
    frozen = copy.copy(learner.mac)
    frozen.agent = copy.deepcopy(learner.mac.agent).eval()
    frozen.action_selector = copy.deepcopy(learner.mac.action_selector)
    frozen.hidden_states = None
    for parameter in frozen.agent.parameters():
        parameter.requires_grad_(False)
    # Force greedy independently of the online exploration schedule.
    frozen.args = copy.copy(learner.mac.args)
    frozen.args.evaluation_epsilon = 0.
    frozen.action_selector.args = frozen.args
    rows, shards = [], []
    with isolated_rollouts(runner), th.inference_mode():
        runner.mac = frozen
        for start in range(0, len(seeds), 8):
            episodes = []
            for seed in seeds[start:start + 8]:
                recorder = EpisodeRecorder()
                runner.run(test_mode=True, episode_seed=seed, recorder=recorder, label_teacher=teacher)
                rows.append(dict(seed=seed, **runner.last_episode_info))
                episodes.append(recorder.episode)
            path = write_shard(target / f"{seeds[start]:010d}.h5", episodes, metadata)
            shards.append(dict(path=path.name, sha256=file_sha256(path)))
    # Verify both the saved collector and live learner stayed unchanged.
    for name, tensor in learner.mac.agent.state_dict().items():
        if not th.equal(tensor, frozen.agent.state_dict()[name]):
            raise RuntimeError("Collector weights changed during DAgger rollouts")
    if file_sha256(checkpoint / "agent.th") != checkpoint_hash:
        raise RuntimeError("Collector checkpoint changed during DAgger rollouts")
    atomic_json(target / "manifest.json", dict(metadata=metadata, seeds=seeds, shards=shards,
                episodes_detail=rows, **outcome_summary(rows)))
    return [target / entry["path"] for entry in shards]
