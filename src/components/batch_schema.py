"""Shared training, CPU rollout and dataset readback schema."""
import torch as th
from components.transforms import OneHot


def build_batch_schema(env_info, common_reward=True):
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "reward": {"vshape": (1,) if common_reward else (env_info["n_agents"],)},
    }
    groups = {"agents": env_info["n_agents"]}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=env_info["n_actions"])])}
    return scheme, groups, preprocess
