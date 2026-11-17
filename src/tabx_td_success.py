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


DISTANCE_BINS = (("terminal", 1, 1), ("near", 2, 5), ("middle", 6, 20),
                 ("earlier", 21, 50), ("far", 51, None))


def distance_metrics(q_taken, targets, rewards, mask, terminated, gamma, counts, wins):
    """Complete executed histories only. G is the behavior return, not policy Q*."""
    def array(value):
        return value.detach().cpu().double().numpy()
    valid = array(mask)[:, :, 0] > 0
    lengths = valid.sum(1)
    if not np.array_equal(valid, np.arange(valid.shape[1])[None] < lengths[:, None]):
        raise ValueError("Distance diagnostics require contiguous complete trajectories")
    ends = array(terminated)[:, :, 0] > 0
    if any(length < 1 or not ends[i, length-1] for i, length in enumerate(lengths)):
        raise ValueError("Distance diagnostics require actual terminal transitions")
    reward = array(rewards).mean(-1)
    q = array(q_taken).mean(-1)
    target = array(targets).mean(-1) if targets is not None else None
    returns = np.zeros_like(reward)
    for i, length in enumerate(lengths):
        total = 0.
        for t in range(int(length)-1, -1, -1):
            total = reward[i, t] + gamma * total
            returns[i, t] = total
    values = dict(q_taken_mean=q, reward_mean=reward, trajectory_return_mean=returns)
    if target is not None:
        error = q - target
        values.update(target_mean=target, td_error_mean=error, td_error_abs_mean=np.abs(error), td_loss=error**2)
    else:
        values.update({key: None for key in ("target_mean", "td_error_mean", "td_error_abs_mean", "td_loss")})
    distance = lengths[:, None] - np.arange(valid.shape[1])[None]
    groups, offset = {}, 0
    for name, count in zip(("legacy_demo", "success_td", "online"), counts):
        selected = np.zeros(len(lengths), dtype=bool)
        selected[offset:offset+count] = True
        groups[name] = selected
        if name == "online":
            outcomes = array(wins).reshape(-1)
            if len(outcomes) != len(lengths) or not np.isin(outcomes[selected], [0, 1]).all():
                raise ValueError("Online diagnostic outcomes must be recorded, not inferred from rewards")
            groups.update(online_win=selected & (outcomes == 1), online_loss=selected & (outcomes == 0))
        offset += count
    if offset != len(lengths):
        raise ValueError("Distance source counts do not match batch size")
    result = {}
    for source, episodes in groups.items():
        result[source] = {}
        for name, lo, hi in (("all", 1, None), *DISTANCE_BINS):
            selection = valid & episodes[:, None] & (distance >= lo)
            if hi is not None:
                selection &= distance <= hi
            steps = int(selection.sum())
            result[source][name] = dict(valid_steps=steps, trajectories=int(selection.any(1).sum()),
                **{key: float(value[selection].mean()) if steps and value is not None else None for key, value in values.items()})
    return result


def gradient_norm(parameters):
    """Read pre-clip gradients without changing them or consuming RNG."""
    import torch as th
    terms = [p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
    return float(th.stack(terms).sum().sqrt()) if terms else 0.
