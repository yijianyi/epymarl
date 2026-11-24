"""MAPPO experiment settings shared by direct training and the batch scheduler."""
import math
from pathlib import Path

from envs.tabx_tasks import ENEMY_HEURISTIC


def configure_mappo(config):
    if "tabx_mix_start" not in config:
        return
    if (config["env"], config["learner"], config["mac"], config["agent_output_type"],
            config["action_selector"], config["critic_type"]) != (
            "tabx", "ppo_learner", "basic_mac", "pi_logits", "multinomial", "cv_critic"):
        raise ValueError("TABX MAPPO requires a shared logits actor, multinomial sampling and a central critic")
    if not config["test_greedy"] or not config["mask_before_softmax"]:
        raise ValueError("TABX MAPPO requires legal-action masking and greedy evaluation")
    if config["buffer_size"] != config["batch_size"] or config["batch_size"] < 1:
        raise ValueError("TABX PPO requires buffer_size=batch_size>=1")
    for key in ("tabx_bc_steps", "tabx_bc_batch_size", "tabx_mix_decay_steps", "tabx_mix_block_steps",
                "tabx_mappo_temp_peak_step", "tabx_mappo_temp_end_step"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("tabx_bc_lr", "tabx_mix_start", "tabx_mappo_temp_peak"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    if config["tabx_bc_lr"] <= 0 or not 0 <= config["tabx_mix_start"] <= 1:
        raise ValueError("Require tabx_bc_lr>0 and tabx_mix_start in [0,1]")
    if config["tabx_mappo_temp_peak"] < 1 or config["tabx_mappo_temp_end_step"] <= config["tabx_mappo_temp_peak_step"]:
        raise ValueError("Require temperature peak>=1 and end step after peak step")
    if type(config["tabx_dagger_rounds"]) is not int or config["tabx_dagger_rounds"] != 0:
        raise ValueError("This MAPPO experiment requires tabx_dagger_rounds=0")
    if config["tabx_bc_teacher"] != "medium" or config["tabx_demo_opponent"] != ENEMY_HEURISTIC:
        raise ValueError("The experiment requires official medium on both sides of demonstration games")
    for key in ("tabx_demo_root", "tabx_online_source"):
        if config[key] is not None:
            if not isinstance(config[key], str) or not config[key].strip():
                raise ValueError(f"{key} must be a directory path")
            config[key] = str(Path(config[key]).expanduser().resolve())
    if not isinstance(config["tabx_bc"], bool):
        raise ValueError("tabx_bc must be a boolean")
    source = config["tabx_online_source"]
    if source and not config["tabx_bc"]:
        raise ValueError("tabx_online_source requires tabx_bc=True")
    if config["tabx_bc"]:
        if not source and not config["tabx_demo_root"]:
            raise ValueError("BC pretraining requires tabx_demo_root from collect-expert")
        if (config["checkpoint_path"] or config["evaluate"] or config["save_replay"]
                or not config["tabx_async_eval"] or not config["save_model"]):
            raise ValueError("TABX BC requires a fresh training attempt with async checkpoint publishing")


def rollout_schedule(args, step):
    beta = args.tabx_mix_start * max(0.0, 1.0 - step / args.tabx_mix_decay_steps)
    peak, end = args.tabx_mappo_temp_peak_step, args.tabx_mappo_temp_end_step
    if step <= peak:
        warmup = step / peak
    else:
        warmup = max(0.0, (end - step) / (end - peak))
    return beta, 1.0 + warmup * (args.tabx_mappo_temp_peak - 1.0)
