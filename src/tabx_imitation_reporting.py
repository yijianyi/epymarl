"""Compare baseline and BC runs at a fixed online environment-step budget."""
from pathlib import Path
import statistics

from tabx_reporting import read_csv, read_episodes, write_csv
from tabx_storage import atomic_json, read_json


FIELDS = ("task_id", "train_seed", "variant", "run_id", "status", "online_budget", "cutoff_step", "budget_reached", "final_eval_step",
          "demonstration_steps", "teacher_collection_win_rate", "teacher_holdout_win_rate", "bc_win_rate",
          "first_training_win_step", "first_greedy_win_step", "early_5_eval_win_rate", "at_100k_eval_step",
          "at_100k_eval_win_rate", "last_5_eval_win_rate", "screen_evaluations", "run_directory")


def summarize_run(directory, budget):
    directory = Path(directory)
    run, config = read_json(directory / "manifest.json"), read_json(directory / "config.json")
    episode_path = directory / "training_episodes.jsonl"
    episodes = sorted(read_episodes(episode_path), key=lambda row: int(row["t_env"]))
    if episode_path.exists():
        crossing = next((int(row["t_env"]) for row in episodes if int(row["t_env"]) >= budget), None)
        cutoff_step = crossing if crossing is not None else (int(episodes[-1]["t_env"]) if episodes else 0)
        budget_reached = crossing is not None
    else:
        # A historical run cannot prove how far the crossing episode extended.
        cutoff_step, budget_reached = budget, None
    training = [row for row in episodes if int(row["t_env"]) <= cutoff_step]
    evaluations = read_csv(directory / "eval_summary.csv") if (directory / "eval_summary.csv").exists() else []
    screens = [row for row in evaluations
               if row["phase"] == "screen" and row["action_selection"] == "greedy"
               and int(row["checkpoint_step"]) <= cutoff_step]
    screens.sort(key=lambda r: int(r["checkpoint_step"]))
    online = [row for row in screens if int(row["checkpoint_step"]) > 0]
    bc_path = directory / "bc_evaluations/after_bc_online_step_0.json"
    bc = read_json(bc_path) if bc_path.exists() else {}
    milestone_cutoff = next((int(row["t_env"]) for row in training if int(row["t_env"]) >= 100000), 100000)
    milestone = next((row for row in online if 100000 <= int(row["checkpoint_step"]) <= milestone_cutoff), None)
    rate = lambda row: float(row["win_rate_pct"]) / 100
    average = lambda rows: statistics.mean(rate(row) for row in rows) if rows else None
    result = dict(task_id=run["task_id"], train_seed=run["train_seed"], run_id=run["run_id"],
        variant="bc_qmix" if config.get("tabx_bc") else "qmix", online_budget=budget,
        cutoff_step=cutoff_step, budget_reached=budget_reached,
        status="training_complete" if (directory / "training_complete.json").exists() else "training_in_progress",
        final_eval_step=int(online[-1]["checkpoint_step"]) if online else None,
        demonstration_steps=0, teacher_collection_win_rate=None,
        teacher_holdout_win_rate=bc.get("teacher", {}).get("win_rate"), bc_win_rate=bc.get("model", {}).get("win_rate"),
        first_training_win_step=next((row["t_env"] for row in training if row["is_win"]), None),
        first_greedy_win_step=next((int(row["checkpoint_step"]) for row in screens if int(row["wins"]) > 0), None),
        early_5_eval_win_rate=average(online[:5]) if len(online) >= 5 else None,
        at_100k_eval_step=int(milestone["checkpoint_step"]) if milestone else None,
        at_100k_eval_win_rate=rate(milestone) if milestone else None,
        last_5_eval_win_rate=average(online[-5:]) if len(online) >= 5 else None,
        screen_evaluations=len(screens), run_directory=str(directory.resolve()))
    if (directory / "bc_summary.json").exists():
        summary = read_json(directory / "bc_summary.json")
        result["demonstration_steps"] = summary["demonstration_environment_steps"]
        result["teacher_collection_win_rate"] = summary["demonstration_wins"] / summary["demonstration_episodes"]
    return result


def learning_report(args):
    import json
    root = Path(args.root).expanduser().resolve()
    tasks, seeds = set(args.tasks.split(",")), set(map(int, args.seeds.split(",")))
    rows = []
    for path in sorted((root / args.rule).glob("*/*/*/manifest.json")):
        run = read_json(path)
        if run["task_id"] in tasks and run["train_seed"] in seeds:
            rows.append(summarize_run(path.parent, args.online_budget))
    output = Path(args.output).expanduser().resolve()
    write_csv(output.with_suffix(".csv"), FIELDS, rows)
    atomic_json(output.with_suffix(".json"), dict(online_budget=args.online_budget, runs=rows,
        notes="Rates are fractions. Empty metrics are unobserved. Runs/attempts remain separate. "
              "cutoff_step is the first episode endpoint reaching budget, or the latest endpoint while training. "
              "Without an episode log, cutoff_step=budget and budget_reached is unknown (null). "
              "Compare equal cutoff_step/budget and screen counts; incomplete evaluation is not zero wins."))
    print(json.dumps(dict(runs=len(rows), csv=str(output.with_suffix('.csv')), json=str(output.with_suffix('.json')))))
