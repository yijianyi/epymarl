"""Independent original-teacher success replay; only executed trajectories enter TD."""
from pathlib import Path

import numpy as np

from tabx_config import success_td_count
from tabx_dataset import episodes_to_batch, read_episode
from tabx_storage import atomic_json, file_sha256, stable_seed


class SuccessTDPool:
    def __init__(self, pool, task_hash, seed):
        self.original = getattr(pool, "original", pool)
        if not self.original.wins:
            raise ValueError("Success TD replay requires winning episodes in the original teacher pool")
        # Construction and sampling never touch the legacy BC/DAgger RNGs.
        self.seed = stable_seed(task_hash, seed, "td_success_replay", 0)
        self.rng = np.random.default_rng(self.seed)
        self.metadata = dict(schema_version="tabx-td-success-pool-1", source="original_teacher_wins",
            directory=str(self.original.directory), pool_size=len(self.original.wins),
            original_episodes=len(self.original.entries), sampling_seed=self.seed,
            manifest_sha256=file_sha256(self.original.directory / "manifest.json"), activation_step=None)
        self.run_dir = None

    def publish(self, run_dir):
        self.run_dir = Path(run_dir)
        atomic_json(self.run_dir / "td_success_pool.json", self.metadata)

    def activate(self, step):
        if self.metadata["activation_step"] is None:
            self.metadata["activation_step"] = int(step)
            if self.run_dir is not None:
                atomic_json(self.run_dir / "td_success_pool.json", self.metadata)

    def sample(self, count, device="cpu"):
        indices = self.rng.choice(self.original.wins, count, replace=True)
        episodes = [read_episode(*self.original.entries[int(i)][:2]) for i in indices]
        return episodes_to_batch(episodes, device)[0]


def td_batch_counts(args, demo_count):
    """Use the legacy integer count (including its rounding) as the sole gate."""
    success_count = success_td_count(vars(args)) if demo_count == 0 else 0
    return demo_count, success_count, args.batch_size - demo_count - success_count


def td_source_metrics(td_error, q_taken, targets, mask, counts):
    """Detached diagnostics; optimization still averages the complete batch once."""
    error, q_taken, targets = td_error.detach(), q_taken.detach(), targets.detach()
    result, offset = {}, 0
    total_transitions = float(mask[:, :, :1].sum())
    for name, count in zip(("legacy_demo", "success_td", "online"), counts):
        selected = slice(offset, offset + count)
        valid = mask[selected]
        denominator = float(valid.sum())
        transitions = int(valid[:, :, :1].sum())
        result.update({f"{name}_episodes": count, f"{name}_transitions": transitions})
        for key, values in (("loss", error.square()), ("q_taken_mean", q_taken), ("target_mean", targets)):
            result[f"{name}_{key}"] = float((values[selected] * valid).sum() / denominator) if denominator else None
        offset += count
    if offset != td_error.shape[0]:
        raise ValueError("TD source counts do not match batch size")
    result["success_td_episode_fraction"] = counts[1] / offset
    result["success_td_transition_fraction"] = result["success_td_transitions"] / total_transitions if total_transitions else 0.
    return result
