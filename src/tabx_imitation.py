"""Recurrent MAPPO behavior cloning from verified official TABX demonstrations."""
from pathlib import Path

import numpy as np
import torch as th
from torch.nn import functional as F

from tabx_dataset import SCHEMA_VERSION, episodes_to_batch, read_episode, verify_shard
from tabx_expert import HeuristicTeacher
from tabx_expert_shards import demonstration_directory, read_expert_manifest, verified_expert_shards
from tabx_storage import atomic_json, simulator_identity


class DemonstrationPool:
    def __init__(self, env, args):
        task = env.manifest["task_id"]
        directory = demonstration_directory(args.tabx_demo_root, env.rule, task, args.tabx_bc_teacher)
        expected_sha256 = getattr(args, "tabx_expected_demo_manifest_sha256", None)
        self.manifest, self.manifest_sha256 = read_expert_manifest(directory, expected_sha256)
        metadata = self.manifest["metadata"]
        expected = dict(env.manifest, **HeuristicTeacher(env, args.tabx_bc_teacher).metadata(),
                        timeout_rule=env.rule, episode_limit=env.episode_limit, collection_phase="expert_collect")
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Demonstration metadata mismatch: {key} in {directory}")
        identity = simulator_identity()
        self._check_identity(metadata, identity)
        self.entries = []
        seeds = set()
        for path in verified_expert_shards(directory, self.manifest):
            verified = verify_shard(path, {key: value for key, value in metadata.items() if key != "provenance"})
            self._check_identity(verified["metadata"], identity)
            if verified["schema_version"] != SCHEMA_VERSION or verified["label_source"] != "deterministic_teacher":
                raise ValueError("MAPPO BC requires deterministic teacher_actions from collect-expert")
            if seeds.intersection(verified["seeds"]):
                raise ValueError("Duplicate demonstration episode seeds")
            seeds.update(verified["seeds"])
            self.entries.extend((path, index) for index in range(verified["episodes"]))
        if not self.entries:
            raise ValueError("Empty demonstration pool")
        self.rng = np.random.default_rng(args.seed)

    @staticmethod
    def _check_identity(metadata, identity):
        official = metadata.get("provenance", {}).get("tabx", {})
        for key, value in identity.items():
            if official.get(key) != value:
                raise ValueError(f"Demonstration simulator provenance mismatch: {key}")

    def sample(self, count, device):
        indices = self.rng.choice(len(self.entries), count, replace=count > len(self.entries))
        batch, _ = episodes_to_batch([read_episode(*self.entries[index]) for index in indices])
        batch.to(device)
        return batch


def behavior_cloning_loss(logits, batch):
    """Mask padding, inactive units, missing labels, and illegal logits."""
    labels = batch["teacher_actions"][:, :-1, :, 0]
    mask = batch["transition_mask"][:, :-1].bool() & batch["filled"][:, :-1].bool()
    for field in ("present", "alive", "teacher_mask"):
        mask = mask & batch[field][:, :-1, :, 0].bool()
    if not mask.any():
        raise ValueError("No valid, live units with deterministic demonstration labels")
    legal = batch["avail_actions"][:, :-1].bool()[mask]
    labels = labels[mask]
    if not legal.gather(-1, labels.unsqueeze(-1)).all():
        raise ValueError("Illegal demonstrated teacher action")
    return F.cross_entropy(logits[mask].masked_fill(~legal, -th.inf), labels)


def pretrain_actor(learner, runner, args, logger):
    pool = DemonstrationPool(runner.env, args)
    if pool.manifest["zero_wins"]:
        logger.console_logger.warning("TABX task %s has zero demonstration wins; review teacher_summary.json",
                                      runner.env.manifest["task_id"])
    optimizer = th.optim.Adam(learner.mac.parameters(), lr=args.tabx_bc_lr)
    for update in range(1, args.tabx_bc_steps + 1):
        batch = pool.sample(args.tabx_bc_batch_size, args.device)
        learner.mac.init_hidden(batch.batch_size)
        logits = th.stack([learner.mac.forward_logits(batch, t) for t in range(batch.max_seq_length - 1)], dim=1)
        loss = behavior_cloning_loss(logits, batch)
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(learner.agent_params, args.grad_norm_clip)
        optimizer.step()
        if update == 1 or update % 100 == 0 or update == args.tabx_bc_steps:
            logger.console_logger.info("TABX BC update %d/%d, loss=%.6f", update, args.tabx_bc_steps, loss.item())
    learner.old_mac.load_state(learner.mac)
    atomic_json(Path(args.tabx_run_dir) / "bc_summary.json", dict(
        updates=args.tabx_bc_steps, batch_size=args.tabx_bc_batch_size, lr=args.tabx_bc_lr,
        final_loss=loss.item(), dataset_manifest_sha256=pool.manifest_sha256,
        demonstration_episodes=len(pool.entries), demonstration_wins=pool.manifest["wins"],
        stage="after_bc_before_online", online_t_env=runner.t_env,
    ))
