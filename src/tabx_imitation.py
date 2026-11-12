"""Recurrent behavior cloning and a separate, immutable demonstration pool."""
import csv
from pathlib import Path

import h5py
import numpy as np
import torch as th
from torch.nn import functional as F

from components.episode_buffer import EpisodeBatch
from tabx_dataset import episodes_to_batch, read_episode, verify_shard
from tabx_config import configure_bc
from tabx_expert_shards import published_expert_shards
from tabx_storage import atomic_json, file_sha256, read_json, stable_seed


def behavior_cloning_loss(logits, batch):
    """Score only executed actions of real, alive allies; never the terminal frame."""
    logits = logits[:, :-1]
    actions = batch["actions"][:, :-1, :, 0]
    mask = (batch["transition_mask"][:, :-1].bool()
            & batch["filled"][:, :-1].bool()).expand_as(actions)
    mask = mask & batch["present"][:, :-1, :, 0].bool() & batch["alive"][:, :-1, :, 0].bool()
    legal = batch["avail_actions"][:, :-1].bool()
    if not mask.any():
        raise ValueError("No real, alive demonstrated actions to imitate")
    if not legal.gather(-1, actions.unsqueeze(-1)).squeeze(-1)[mask].all():
        raise ValueError("Illegal demonstration target")
    selected = logits[mask].masked_fill(~legal[mask], -th.inf)
    labels = actions[mask]
    loss = F.cross_entropy(selected, labels)
    return loss, dict(loss=float(loss.detach()), accuracy=float((selected.argmax(-1) == labels).float().mean()),
                      valid_actions=int(mask.sum()))


class DemonstrationPool:
    """Keep lightweight episode indices on CPU; load complete episodes on sampling."""
    def __init__(self, directory, env, seed, win_fraction=.5, require_wins=True):
        self.directory = Path(directory).expanduser().resolve()
        self.manifest = read_json(self.directory / "manifest.json")
        self.metadata = self.manifest["metadata"]
        expected = {key: env.manifest[key] for key in ("task_id", "task_hash", "bank_hash")}
        expected.update(env.get_layout_info(), timeout_rule=env.rule, episode_limit=env.episode_limit, opponent="medium")
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ValueError(f"Demonstration metadata mismatch: {key}")
        if self.metadata.get("collection_phase") != "expert_collect":
            raise ValueError("Use an independent collect-expert dataset")
        from tabx_storage import simulator_identity
        identity = simulator_identity()
        official = self.metadata.get("provenance", {}).get("tabx", {})
        for key, value in identity.items():
            if official.get(key) != value:
                raise ValueError(f"Demonstration simulator provenance mismatch: {key}")
        self.entries, self.seeds = [], set()
        self.collection_steps = 0
        shards = self.manifest["shards"]
        if {entry["path"] for entry in shards} != {p.name for p in published_expert_shards(self.directory)}:
            raise ValueError("Demonstration shards differ from manifest; rerun collect-expert")
        for entry in shards:
            path = self.directory / entry["path"]
            if path.parent != self.directory or file_sha256(path) != entry["sha256"]:
                raise ValueError("Demonstration shard checksum/path mismatch")
            report = verify_shard(path, {k: v for k, v in self.metadata.items() if k != "provenance"})
            for key, value in identity.items():
                if report["metadata"].get("provenance", {}).get("tabx", {}).get(key) != value:
                    raise ValueError(f"Demonstration shard simulator provenance mismatch: {key}")
            if self.seeds.intersection(report["seeds"]):
                raise ValueError("Duplicate demonstration episode seeds")
            self.seeds.update(report["seeds"])
            self.collection_steps += report["steps"]
            with h5py.File(path, "r") as data:
                self.entries.extend((path, i, bool(win)) for i, win in enumerate(data["episodes/is_win"][:]))
        if not self.entries:
            raise ValueError("Empty demonstration pool")
        self.wins = [i for i, entry in enumerate(self.entries) if entry[2]]
        self.losses = [i for i, entry in enumerate(self.entries) if not entry[2]]
        if require_wins and not self.wins:
            raise ValueError("Teacher has zero wins: review teacher_summary.json or collect with --teacher expert")
        self.rng = np.random.default_rng(seed)
        self.win_fraction = max(win_fraction, len(self.wins) / len(self.entries))

    def sample(self, count, device="cpu"):
        if self.wins and self.losses:
            # A target fraction oversamples wins while retaining losing trajectories.
            n_win = int(self.rng.binomial(count, self.win_fraction))
            indices = np.r_[self.rng.choice(self.wins, n_win), self.rng.choice(self.losses, count - n_win)]
            self.rng.shuffle(indices)
        else:
            indices = self.rng.choice(len(self.entries), count)
        episodes = [read_episode(*self.entries[int(i)][:2]) for i in indices]
        return episodes_to_batch(episodes, device)[0]


def mix_batches(demos, online):
    """Demos come first so their logits can also be used for the auxiliary loss."""
    scheme = {k: v for k, v in online.scheme.items() if k != "filled"}
    result = EpisodeBatch(scheme, online.groups, demos.batch_size + online.batch_size,
                          max(demos.max_seq_length, online.max_seq_length), device=online.device)
    for offset, batch in ((0, demos), (demos.batch_size, online)):
        values = {key: batch[key].to(online.device) for key in result.data.transition_data}
        result.update(values, bs=slice(offset, offset + batch.batch_size), ts=slice(0, batch.max_seq_length),
                      mark_filled=False)
    return result


def demonstration_schedule(args, online_steps):
    remaining = max(0., 1. - online_steps / args.tabx_demo_decay_steps)
    count = int(args.batch_size * args.tabx_demo_fraction * remaining + .5)
    return min(count, args.batch_size - 1), args.tabx_demo_bc_weight * remaining


def pretrain(learner, pool, args, logger):
    optimiser = th.optim.Adam(learner.mac.parameters(), lr=args.tabx_bc_lr)
    path = Path(args.tabx_run_dir) / "bc_metrics.csv"
    learner.mac.agent.train()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("update", "loss", "accuracy", "valid_actions", "grad_norm"))
        writer.writeheader()
        for step in range(1, args.tabx_bc_steps + 1):
            batch = pool.sample(args.tabx_bc_batch_size, args.device)
            learner.mac.init_hidden(batch.batch_size)
            outputs = th.stack([learner.mac.forward(batch, t=t) for t in range(batch.max_seq_length)], dim=1)
            loss, metrics = behavior_cloning_loss(outputs, batch)
            optimiser.zero_grad()
            loss.backward()
            norm = th.nn.utils.clip_grad_norm_(list(learner.mac.parameters()), args.grad_norm_clip)
            optimiser.step()
            writer.writerow(dict(update=step, **metrics, grad_norm=float(norm)))
            if step == 1 or step % 100 == 0 or step == args.tabx_bc_steps:
                stream.flush()
                logger.console_logger.info("BC %d/%d loss=%.5f accuracy=%.3f", step, args.tabx_bc_steps,
                                           metrics["loss"], metrics["accuracy"])
    learner._update_targets_hard()
    atomic_json(Path(args.tabx_run_dir) / "bc_summary.json", dict(
        stage="after_bc_before_online", online_t_env=0, bc_updates=args.tabx_bc_steps,
        demonstration_episodes=len(pool.entries), demonstration_wins=len(pool.wins),
        demonstration_environment_steps=pool.collection_steps, dataset_manifest_sha256=file_sha256(pool.directory / "manifest.json"),
        teacher=pool.metadata["teacher"], teacher_epsilon=pool.metadata["teacher_epsilon"]))


def evaluate_policy(runner, pool, args, *, stage, compare_teacher=False):
    """Independent paired seeds; evaluation must not perturb online exploration RNG."""
    from tabx_expert import HeuristicTeacher, outcome_summary, teacher_rollout
    seeds = [stable_seed(runner.env.manifest["task_hash"], args.tabx_master_seed, "bc_holdout", i)
             for i in range(args.tabx_bc_eval_episodes)]
    if pool.seeds.intersection(seeds):
        raise ValueError("BC evaluation seeds overlap demonstration collection")
    numpy_rng, torch_rng = np.random.get_state(), th.random.get_rng_state()
    cuda_rng = th.cuda.get_rng_state_all() if args.use_cuda else None
    was_training = runner.mac.agent.training
    test_returns, test_stats, log_t = runner.test_returns, runner.test_stats, runner.log_train_stats_t
    runner.test_returns, runner.test_stats = [], {}
    epsilon = runner.mac.action_selector.epsilon
    rows, teacher_rows = [], []
    try:
        runner.mac.agent.eval()
        with th.inference_mode():
            for seed in seeds:
                runner.run(test_mode=True, episode_seed=seed)
                rows.append(dict(seed=seed, **runner.last_episode_info))
        if compare_teacher:
            teacher = HeuristicTeacher(runner.env, pool.metadata["teacher"], pool.metadata["teacher_epsilon"])
            for seed in seeds:
                teacher_rows.append(teacher_rollout(runner.env, teacher, seed))
    finally:
        np.random.set_state(numpy_rng)
        th.random.set_rng_state(torch_rng)
        if cuda_rng is not None:
            th.cuda.set_rng_state_all(cuda_rng)
        runner.mac.agent.train(was_training)
        runner.test_returns, runner.test_stats, runner.log_train_stats_t = test_returns, test_stats, log_t
        runner.mac.action_selector.epsilon = epsilon
    result = dict(stage=stage, online_t_env=runner.t_env, action_selection="greedy", evaluation_epsilon=0.,
                  seed_phase="bc_holdout", seeds=seeds, model=outcome_summary(rows), episodes=rows)
    if teacher_rows:
        result.update(teacher=outcome_summary(teacher_rows), teacher_episodes=teacher_rows,
                      teacher_metadata=teacher.metadata())
    atomic_json(Path(args.tabx_run_dir) / "bc_evaluations" / f"{stage}.json", result)
    runner.logger.console_logger.info("%s online_t_env=%d: model %d/%d wins%s", stage, runner.t_env,
        result["model"]["wins"], len(rows),
        f", teacher {result['teacher']['wins']}/{len(teacher_rows)} wins" if teacher_rows else "")
    return result


def prepare_imitation(args, learner, runner, logger):
    directory = Path(args.tabx_demo_root).expanduser() / runner.env.rule / str(runner.env.manifest["task_id"]) / args.tabx_bc_teacher
    pool = DemonstrationPool(directory, runner.env, args.seed, args.tabx_bc_win_fraction, args.tabx_bc_require_wins)
    from tabx_expert import HeuristicTeacher
    teacher_metadata = HeuristicTeacher(runner.env, pool.metadata["teacher"], pool.metadata["teacher_epsilon"]).metadata()
    for key, value in teacher_metadata.items():
        if pool.metadata.get(key) != value:
            raise ValueError(f"Demonstration teacher differs from current official teacher: {key}")
    pretrain(learner, pool, args, logger)
    evaluate_policy(runner, pool, args, stage="after_bc_online_step_0", compare_teacher=True)
    return pool
