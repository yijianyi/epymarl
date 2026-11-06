"""One adjudication point for TABX, independent of the simulator's done cache."""
from dataclasses import dataclass


END_REASONS = {0: "ongoing", 1: "elimination", 2: "timeout", 3: "mutual_elimination"}


@dataclass(frozen=True)
class Outcome:
    ended: bool
    is_win: bool
    reason: int
    hit_time_limit: bool
    terminal_reward: float

    @property
    def end_reason(self):
        return END_REASONS[self.reason]


def adjudicate(ally_alive, enemy_alive, ally_hp_ratio, enemy_hp_ratio,
               timestep, limit=512, rule="hp", win_reward=1.0, lose_reward=-1.0):
    if rule not in ("hp", "loss"):
        raise ValueError(f"Unknown timeout rule: {rule!r}")
    hit_limit = timestep >= limit
    if not ally_alive and not enemy_alive:
        return Outcome(True, False, 3, hit_limit, float(lose_reward))
    if not ally_alive or not enemy_alive:
        win = bool(ally_alive)
        return Outcome(True, win, 1, hit_limit, float(win_reward if win else lose_reward))
    if hit_limit:
        win = rule == "hp" and ally_hp_ratio > enemy_hp_ratio
        return Outcome(True, win, 2, True, float(win_reward if win else lose_reward))
    return Outcome(False, False, 0, False, 0.0)


def replace_terminal_reward(raw_team_reward, raw_done_reward, outcome):
    dense = float(raw_team_reward) - float(raw_done_reward)
    return dense + outcome.terminal_reward, dense
