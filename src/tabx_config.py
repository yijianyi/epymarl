"""Effective TABX BC configuration shared by scheduling and training; stdlib only."""
import math


def configure_bc(config):
    if not config.get("tabx_bc", False):
        return config
    if (config["env"] != "tabx" or config["learner"] != "q_learner" or config["mixer"] != "qmix"
            or config["mac"] != "basic_mac" or config["agent_output_type"] != "q"):
        raise ValueError("TABX BC requires the shared basic_mac QLearner/QMIX agent")
    if config.get("checkpoint_path") or config.get("evaluate") or config.get("save_replay"):
        raise ValueError("TABX BC starts a fresh training attempt; checkpoint evaluation uses tabx_bc=False")
    if not config.get("tabx_demo_root"):
        raise ValueError("tabx_bc=True requires tabx_demo_root from collect-expert")
    if config["tabx_bc_teacher"] not in ("advanced", "expert"):
        raise ValueError("tabx_bc_teacher must be advanced or expert")
    for key in ("tabx_bc_steps", "tabx_bc_batch_size", "tabx_bc_eval_episodes", "tabx_demo_decay_steps"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("tabx_demo_fraction", "tabx_bc_win_fraction", "tabx_bc_epsilon_start", "tabx_bc_epsilon_finish"):
        if not math.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if config["tabx_demo_fraction"] >= 1:
        raise ValueError("Online batches must include online episodes")
    for key in ("tabx_bc_lr", "tabx_demo_bc_weight"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["evaluation_epsilon"] != 0:
        raise ValueError("BC comparisons require evaluation_epsilon=0")
    if config["tabx_bc_target"] not in ("actions", "teacher_actions"):
        raise ValueError("tabx_bc_target must be actions or teacher_actions")
    weights = config["tabx_bc_action_weights"]
    if not isinstance(weights, (list, tuple)) or len(weights) != 8 or any(
            isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("tabx_bc_action_weights requires eight finite positive weights")
    rounds = config["tabx_dagger_rounds"]
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
        raise ValueError("tabx_dagger_rounds must be a nonnegative integer")
    if rounds:
        if config["tabx_bc_target"] != "teacher_actions":
            raise ValueError("DAgger requires teacher_actions targets")
        episodes = config["tabx_dagger_episodes"]
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
            raise ValueError("tabx_dagger_episodes must be a positive integer")
        if not 0 < config["tabx_dagger_initial_fraction"] < 1 or not 0 < config["tabx_dagger_fraction"] < 1:
            raise ValueError("DAgger fractions must be between zero and one")
        initial = max(1, int(config["tabx_bc_steps"] * config["tabx_dagger_initial_fraction"]))
        if config["tabx_bc_steps"] - initial < rounds:
            raise ValueError("BC budget must provide at least one update per DAgger round")
    config.update(epsilon_start=config["tabx_bc_epsilon_start"], epsilon_finish=config["tabx_bc_epsilon_finish"],
                  epsilon_anneal_time=config["tabx_demo_decay_steps"])
    return config


def bc_stage_budgets(config):
    """Keep the requested total budget, including when users increase BC steps."""
    total, rounds = config["tabx_bc_steps"], config["tabx_dagger_rounds"]
    if not rounds:
        return [total]
    initial = max(1, int(total * config["tabx_dagger_initial_fraction"]))
    each, remainder = divmod(total - initial, rounds)
    return [initial] + [each + int(i < remainder) for i in range(rounds)]
