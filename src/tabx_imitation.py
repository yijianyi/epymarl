"""Recurrent behavior cloning and a separate, immutable demonstration pool."""
import csv
from pathlib import Path

import h5py
import numpy as np
import torch as th
from torch.nn import functional as F

from components.episode_buffer import EpisodeBatch
from envs.tabx_tasks import ENEMY_HEURISTIC
from tabx_dataset import episodes_to_batch, read_episode, verify_shard
from tabx_config import configure_bc
from tabx_expert_shards import published_expert_shards
from tabx_storage import atomic_json, file_sha256, read_json, stable_seed


def classification_metrics(predictions, labels, n_actions=8):
    matrix = th.bincount(labels * n_actions + predictions, minlength=n_actions**2).reshape(n_actions, n_actions)
    support, predicted, correct = matrix.sum(1), matrix.sum(0), matrix.diag()
    return dict(accuracy=float(correct.sum() / support.sum().clamp_min(1)), valid_actions=int(support.sum()),
                action_support=support.tolist(), action_precision=(correct / predicted.clamp_min(1)).tolist(),
                action_recall=(correct / support.clamp_min(1)).tolist(), confusion_matrix=matrix.tolist())


def behavior_cloning_loss(logits, batch, *, target="actions", action_weights=None):
    """Use executed history, but supervise with the explicitly selected target."""
    if target not in ("actions", "teacher_actions"):
        raise ValueError("Unknown BC target")
    logits = logits[:, :-1]
    actions = batch[target][:, :-1, :, 0]
    mask = (batch["transition_mask"][:, :-1].bool()
            & batch["filled"][:, :-1].bool()).expand_as(actions)
    mask = mask & batch["present"][:, :-1, :, 0].bool() & batch["alive"][:, :-1, :, 0].bool()
    if target == "teacher_actions":
        if not batch["has_teacher_labels"].bool().all():
            raise ValueError("Deterministic teacher_actions required; relabel legacy data explicitly")
        mask = mask & batch["teacher_mask"][:, :-1, :, 0].bool()
    legal = batch["avail_actions"][:, :-1].bool()
    if not mask.any():
        raise ValueError("No real, alive demonstrated actions to imitate")
    if not legal.gather(-1, actions.unsqueeze(-1)).squeeze(-1)[mask].all():
        raise ValueError("Illegal demonstration target")
    selected, labels = logits[mask].masked_fill(~legal[mask], -th.inf), actions[mask]
    ce = F.cross_entropy(selected, labels, reduction="none")
    weights = th.as_tensor([1.] * logits.shape[-1] if action_weights is None else action_weights,
                           device=logits.device, dtype=logits.dtype)
    if weights.shape != (logits.shape[-1],) or not th.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("BC action weights must be finite and positive, one per action")
    sample_weights = weights[labels]
    loss = (ce * sample_weights).sum() / sample_weights.sum()
    metrics = classification_metrics(selected.argmax(-1), labels, logits.shape[-1])
    metrics.update(loss=float(loss.detach()), unweighted_ce=float(ce.detach().mean()))
    return loss, metrics


class DemonstrationPool:
    """Keep lightweight episode indices on CPU; load complete episodes on sampling."""
    def __init__(self, directory, env, seed, win_fraction=.5, require_wins=True, target="actions"):
        self.directory = Path(directory).expanduser().resolve()
        self.manifest = read_json(self.directory / "manifest.json")
        self.metadata = self.manifest["metadata"]
        self.target = target
        expected = {key: env.manifest[key] for key in ("task_id", "task_hash", "bank_hash")}
        expected.update(env.get_layout_info(), timeout_rule=env.rule, episode_limit=env.episode_limit, opponent=ENEMY_HEURISTIC)
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
            if target == "teacher_actions" and (report["schema_version"] != "tabx-hdf5-3"
                    or report["label_source"] != "deterministic_teacher" or report["metadata"].get("label_epsilon") != 0):
                raise ValueError("Deterministic teacher_actions required; use label-expert to create a new data version")
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

    def sample_episodes(self, count):
        if self.wins and self.losses:
            # A target fraction oversamples wins while retaining losing trajectories.
            n_win = int(self.rng.binomial(count, self.win_fraction))
            indices = np.r_[self.rng.choice(self.wins, n_win), self.rng.choice(self.losses, count - n_win)]
            self.rng.shuffle(indices)
        else:
            indices = self.rng.choice(len(self.entries), count)
        episodes = [read_episode(*self.entries[int(i)][:2]) for i in indices]
        return episodes

    def sample(self, count, device="cpu"):
        return episodes_to_batch(self.sample_episodes(count), device)[0]


class AggregatedDemonstrationPool:
    """Original teacher wins are oversampled; DAgger rounds/episodes are uniform."""
    def __init__(self, original, seed, dagger_fraction=.5):
        self.original = original
        self.metadata, self.directory = original.metadata, original.directory
        self.entries, self.wins = original.entries, original.wins
        self.collection_steps = original.collection_steps
        self.seeds = set(original.seeds)
        self.rounds = []
        self.rng = np.random.default_rng(seed)
        self.dagger_fraction = dagger_fraction
        self.dagger_steps = 0

    def add_round(self, paths):
        entries = []
        if not paths:
            raise ValueError("Empty DAgger round")
        manifest = read_json(Path(paths[0]).parent / "manifest.json")
        expected_paths = {entry["path"]: entry["sha256"] for entry in manifest["shards"]}
        if {Path(p).name for p in paths} != set(expected_paths):
            raise ValueError("DAgger paths differ from round manifest")
        checkpoint = Path(paths[0]).parent / manifest["metadata"]["collector_checkpoint"]
        if file_sha256(checkpoint) != manifest["metadata"]["collector_checkpoint_sha256"]:
            raise ValueError("DAgger collector checkpoint checksum mismatch")
        round_seeds, round_steps = set(), 0
        for path in paths:
            if Path(path).parent != Path(paths[0]).parent or file_sha256(path) != expected_paths[Path(path).name]:
                raise ValueError("DAgger shard checksum/path mismatch")
            report = verify_shard(path, manifest["metadata"])
            for key in ("task_id", "task_hash", "bank_hash", "timeout_rule", "episode_limit", "opponent"):
                if report["metadata"].get(key) != self.metadata.get(key):
                    raise ValueError(f"DAgger/original dataset mismatch: {key}")
            if report["schema_version"] != "tabx-hdf5-3" or report["label_source"] != "deterministic_teacher":
                raise ValueError("DAgger requires deterministic teacher labels")
            if (self.seeds | round_seeds).intersection(report["seeds"]):
                raise ValueError("Duplicate aggregation seed")
            round_seeds.update(report["seeds"])
            round_steps += report["steps"]
            entries.extend((path, i) for i in range(report["episodes"]))
        if not entries:
            raise ValueError("Empty DAgger round")
        self.seeds.update(round_seeds)
        self.dagger_steps += round_steps
        self.rounds.append(entries)

    def sample(self, count, device="cpu"):
        dagger_count = int(count * self.dagger_fraction + .5) if self.rounds else 0
        episodes = self.original.sample_episodes(count - dagger_count)
        for _ in range(dagger_count):
            entries = self.rounds[int(self.rng.integers(len(self.rounds)))]
            episodes.append(read_episode(*entries[int(self.rng.integers(len(entries)))]))
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


def pretrain(learner, pool, args, logger, runner=None):
    from tabx_config import bc_stage_budgets
    from tabx_dagger import collect_dagger_round
    optimiser = th.optim.Adam(learner.mac.parameters(), lr=args.tabx_bc_lr)
    budgets = bc_stage_budgets(vars(args))
    target = getattr(args, "tabx_bc_target", "actions")
    if len(budgets) > 1 and runner is None:
        raise ValueError("DAgger requires a rollout runner")
    scalar_fields = ("update", "stage", "loss", "unweighted_ce", "accuracy", "valid_actions", "grad_norm", "label_source")
    action_fields = tuple(f"action_{i}_{name}" for i in range(8) for name in ("support", "precision", "recall"))
    learner.mac.agent.train()
    step, round_reports = 0, []
    with (Path(args.tabx_run_dir) / "bc_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields + action_fields)
        writer.writeheader()
        for stage, updates in enumerate(budgets):
            if stage:
                paths = collect_dagger_round(learner, runner, pool, args, stage)
                pool.add_round(paths)
                round_reports.append(str(paths[0].parent / "manifest.json"))
                logger.console_logger.info("DAgger round %d collected; cumulative interaction steps=%d", stage, pool.dagger_steps)
            for _ in range(updates):
                step += 1
                batch = pool.sample(args.tabx_bc_batch_size, args.device)
                learner.mac.init_hidden(batch.batch_size)
                outputs = th.stack([learner.mac.forward(batch, t=t) for t in range(batch.max_seq_length)], dim=1)
                loss, metrics = behavior_cloning_loss(outputs, batch, target=target,
                    action_weights=getattr(args, "tabx_bc_action_weights", None))
                optimiser.zero_grad()
                loss.backward()
                norm = th.nn.utils.clip_grad_norm_(list(learner.mac.parameters()), args.grad_norm_clip)
                optimiser.step()
                row = {name: metrics[name] for name in ("loss", "unweighted_ce", "accuracy", "valid_actions")}
                row.update(update=step, stage=stage, grad_norm=float(norm),
                           label_source="deterministic_teacher" if target == "teacher_actions" else "executed_actions")
                for i in range(8):
                    for name in ("support", "precision", "recall"):
                        row[f"action_{i}_{name}"] = metrics[f"action_{name}"][i]
                writer.writerow(row)
                if step == 1 or step % 100 == 0 or step == args.tabx_bc_steps:
                    stream.flush()
                    logger.console_logger.info("BC %d/%d stage=%d loss=%.5f unweighted_ce=%.5f accuracy=%.3f", step,
                        args.tabx_bc_steps, stage, metrics["loss"], metrics["unweighted_ce"], metrics["accuracy"])
    th.save(optimiser.state_dict(), Path(args.tabx_run_dir) / "bc_optimizer.th")
    learner._update_targets_hard()
    atomic_json(Path(args.tabx_run_dir) / "bc_summary.json", dict(
        stage="after_bc_before_online", online_t_env=0, bc_updates=step, stage_updates=budgets,
        bc_target=target, action_weights=getattr(args, "tabx_bc_action_weights", [1.] * 8),
        label_source="deterministic_teacher" if target == "teacher_actions" else "executed_actions",
        demonstration_episodes=len(pool.entries), demonstration_wins=len(pool.wins),
        demonstration_environment_steps=pool.collection_steps,
        dagger_environment_steps=getattr(pool, "dagger_steps", 0),
        dagger_episodes=sum(len(r) for r in getattr(pool, "rounds", [])), dagger_round_manifests=round_reports,
        dataset_manifest_sha256=file_sha256(pool.directory / "manifest.json"),
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
    confusion = th.zeros(8, 8, dtype=th.long)
    try:
        runner.mac.agent.eval()
        with th.inference_mode():
            label_teacher = HeuristicTeacher(runner.env, "advanced", 0.) if getattr(pool, "metadata", {}).get("teacher") else None
            for seed in seeds:
                if label_teacher is None:
                    runner.run(test_mode=True, episode_seed=seed)
                else:
                    from tabx_dataset import EpisodeRecorder
                    recorder = EpisodeRecorder()
                    runner.run(test_mode=True, episode_seed=seed, recorder=recorder, label_teacher=label_teacher)
                    steps = recorder.episode["transitions"]
                    valid = steps["teacher_mask"][..., 0].astype(bool)
                    labels = th.as_tensor(steps["teacher_actions"][..., 0][valid])
                    actual = th.as_tensor(steps["actions"][..., 0][valid])
                    confusion += th.bincount(labels * 8 + actual, minlength=64).reshape(8, 8)
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
    if confusion.sum():
        support, predicted, correct = confusion.sum(1), confusion.sum(0), confusion.diag()
        result["model_teacher_agreement"] = dict(accuracy=float(correct.sum() / support.sum()),
            valid_actions=int(support.sum()), action_support=support.tolist(),
            action_precision=(correct / predicted.clamp_min(1)).tolist(),
            action_recall=(correct / support.clamp_min(1)).tolist(), confusion_matrix=confusion.tolist())
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
    pool = DemonstrationPool(directory, runner.env, args.seed, args.tabx_bc_win_fraction, args.tabx_bc_require_wins, args.tabx_bc_target)
    from tabx_expert import HeuristicTeacher
    teacher_metadata = HeuristicTeacher(runner.env, pool.metadata["teacher"], pool.metadata["teacher_epsilon"]).metadata()
    for key, value in teacher_metadata.items():
        if key in ("label_source", "label_teacher", "label_epsilon"):
            continue
        if pool.metadata.get(key) != value:
            raise ValueError(f"Demonstration teacher differs from current official teacher: {key}")
    if args.tabx_bc_target == "teacher_actions":
        if pool.metadata.get("label_teacher") != "advanced":
            raise ValueError("teacher_actions training requires epsilon-zero advanced labels; use label-expert")
        if "label_teacher_metadata" in pool.metadata:
            actual = HeuristicTeacher(runner.env, "advanced", 0.).metadata()
            if pool.metadata["label_teacher_metadata"] != actual:
                raise ValueError("Relabel teacher parameters/version differ from the current advanced teacher")
    if args.tabx_dagger_rounds:
        pool = AggregatedDemonstrationPool(pool, args.seed, args.tabx_dagger_fraction)
    pretrain(learner, pool, args, logger, runner=runner)
    evaluate_policy(runner, pool, args, stage="after_bc_online_step_0", compare_teacher=True)
    return pool
