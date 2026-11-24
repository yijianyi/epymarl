"""Shared training, CPU rollout and dataset readback schema."""
import torch as th
from components.transforms import OneHot


def build_batch_schema(env_info, common_reward=True, *, include_agent_alive=False, include_ppo_behavior=False):
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "reward": {"vshape": (1,) if common_reward else (env_info["n_agents"],)},
    }
    if include_agent_alive:
        scheme["agent_alive"] = {"vshape": (1,), "group": "agents", "dtype": th.bool}
    if include_ppo_behavior:
        scheme.update(
            teacher_control={"vshape": (1,), "dtype": th.bool},
            control_remaining={"vshape": (1,), "dtype": th.int},
            policy_temperature={"vshape": (1,), "dtype": th.float},
            behavior_log_prob={"vshape": (1,), "group": "agents", "dtype": th.float},
        )
    groups = {"agents": env_info["n_agents"]}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=env_info["n_actions"])])}
    return scheme, groups, preprocess
