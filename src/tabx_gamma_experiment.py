"""Seal single-factor online QMIX comparisons from identical BC/DAgger weights."""
from pathlib import Path
import json

from tabx_storage import atomic_json, file_sha256, read_json, stable_seed


def _flatten(config, prefix=""):
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten(value, name)
        else:
            yield name, value


def make_pair_plan(args):
    from tabx_online_start import source_inventory, pool_fingerprint, pool_inventory
    from tabx_scheduler import PROTECTED, training_config
    from envs.tabx_tasks import load_task
    path, manifest, saved, hashes = source_inventory(args.source)
    if not 0 <= args.seed < 2**32:
        raise ValueError("Training seed must be an integer in [0,2**32)")
    if manifest.get("t_env") != 0 or manifest.get("policy_stage") != "after_bc_before_online":
        raise ValueError("Gamma pair requires the BC/DAgger online-step-0 snapshot")
    if manifest["task_id"] != args.task or manifest["timeout_rule"] != args.rule:
        raise ValueError("Gamma pair source task/rule mismatch")
    _, task = load_task(args.bank, args.task, manifest["task_hash"], manifest["bank_hash"])
    root = Path(args.root).expanduser().resolve()
    demo_root = str(Path(args.demo_root or saved["tabx_demo_root"]).expanduser().resolve())
    fingerprint = pool_fingerprint(path, demo_root)
    defaults, _ = training_config([])
    known = dict(_flatten(defaults))
    overrides = {}
    for key, value in _flatten(saved):
        if key in known and value != known[key] and key not in PROTECTED and not any(part.startswith("tabx_") for part in key.split(".")[1:]):
            overrides[key] = value
    # Every source algorithm/architecture hyperparameter must either be inherited or match a protected default.
    for key in PROTECTED.intersection(known):
        if key in saved and key in ("agent", "mac", "mixer", "learner", "use_rnn", "runner", "env", "batch_size_run"):
            if saved[key] != known[key]:
                raise ValueError(f"Source algorithm cannot be reproduced by batch-train: {key}")
    overrides.update(tabx_bc=True, tabx_online_source=str(path), tabx_demo_root=demo_root,
        tabx_source_agent_sha256=hashes["agent.th"], tabx_source_mixer_sha256=hashes["mixer.th"],
        tabx_source_pool_sha256=fingerprint, tabx_screen_episodes=128, tabx_bc_eval_episodes=128,
        tabx_eval_seed_phase="bc_holdout", t_max=args.budget)
    success_pair = args.command == "td-success-pair"
    treatment = "tabx_td_success_fraction" if success_pair else "gamma"
    arms = ((0., "td0"), (.25, "td025")) if success_pair else ((.99, "g099"), (.995, "g0995"))
    if success_pair:
        overrides["gamma"] = .99
    configs, commands = [], []
    for value, suffix in arms:
        settings = [f"{key}={value!r}" for key, value in sorted(dict(overrides, **{treatment: value}).items())]
        config, _ = training_config(settings)
        configs.append(config)
        commands.append(["batch-train", "--bank", str(Path(args.bank).expanduser().resolve()), "--root", str(root),
            "--rule", args.rule, "--tasks", args.task, "--seeds", str(args.seed), "--gpus", args.gpus,
            "--jobs-per-gpu", str(args.jobs_per_gpu), "--eval-workers", str(args.eval_workers),
            "--batch-id", f"{args.pair_id}-{suffix}", *sum((["--set", setting] for setting in settings), [])])
    differences = [key for key in configs[0] if configs[0][key] != configs[1][key]]
    if differences != [treatment]:
        raise RuntimeError(f"Single-factor treatment violated: {differences}")
    return dict(schema_version="tabx-td-success-pair-1" if success_pair else "tabx-gamma-pair-1",
        treatment_key=treatment, arms=[dict(value=value, suffix=suffix) for value, suffix in arms],
        pair_id=args.pair_id, task_id=args.task, rule=args.rule,
        seed=args.seed, budget=args.budget, source_snapshot=str(path), source_hashes=hashes,
        source_pool_sha256=fingerprint, source_pool_inventory=pool_inventory(path, demo_root),
        evaluation_episodes=128, seed_phase="bc_holdout",
        evaluation_seeds=[stable_seed(task["task_hash"], saved.get("tabx_master_seed", 0), "bc_holdout", i) for i in range(128)],
        treatment_differences=differences, training_configs=configs, commands=commands,
        shared_protocol_overrides=dict(t_max=args.budget, screen_episodes=128, bc_eval_episodes=128))


def summarize_pair(root, plan):
    from tabx_reporting import read_csv
    results = []
    treatment = plan.get("treatment_key", "gamma")
    arms = plan.get("arms", [dict(value=.99, suffix="g099"), dict(value=.995, suffix="g0995")])
    for arm in arms:
        suffix, value = arm["suffix"], arm["value"]
        state_path = Path(root) / "_batches" / plan["rule"] / f"{plan['pair_id']}-{suffix}" / "state.json"
        if not state_path.exists():
            results.append({treatment: value, "status": "not_started"})
            continue
        state = read_json(state_path)
        job = state["jobs"][0]
        run = Path(root) / job["attempts"][-1]["run"]
        from tabx_imitation_reporting import summarize_run
        row = summarize_run(run, plan["budget"])
        evaluations = [entry for entry in read_csv(run / "eval_summary.csv") if entry["phase"] == "screen"
                       and int(entry["checkpoint_step"]) <= row["cutoff_step"]]
        row.update({treatment: value}, evaluations=[{key: entry[key] for key in
            ("checkpoint_step", "episodes", "wins", "win_rate_pct", "elimination_wins", "timeout_wins", "mean_return")}
            for entry in sorted(evaluations, key=lambda x: int(x["checkpoint_step"]))])
        if (run / "online_start.json").exists():
            row["online_start"] = read_json(run / "online_start.json")
        if (run / "td_success_pool.json").exists():
            row["success_td_pool"] = read_json(run / "td_success_pool.json")
        metrics_path = run / "td_source_metrics.jsonl"
        if metrics_path.exists():
            from tabx_reporting import read_episodes
            metrics = [m for m in read_episodes(metrics_path) if m["t_env"] <= row["cutoff_step"]]
            row["td_sampling"] = dict(updates=len(metrics),
                sampled_episodes={source: sum(m[f"{source}_episodes"] for m in metrics)
                                  for source in ("legacy_demo", "success_td", "online")},
                latest=metrics[-1] if metrics else None)
        results.append(row)
    available = [row["online_start"] for row in results if "online_start" in row]
    if len(available) == 2:
        for key in ("agent_parameter_sha256", "mixer_parameter_sha256", "source_pool_sha256"):
            if available[0][key] != available[1][key]:
                raise RuntimeError(f"Paired initial models/pools differ: {key}")
    return dict(pair_id=plan["pair_id"], treatment=f"{treatment}_only", results=results,
                notes="Compare identical observed endpoint budgets and completed fixed-seed evaluations; no missing eval is counted as zero.")


def gamma_pair(args):
    from tabx_pipeline import parser
    from tabx_scheduler import batch_train
    from tabx_storage import safe_component
    safe_component(args.pair_id)
    namespace = "_td_success_pairs" if args.command == "td-success-pair" else "_gamma_pairs"
    directory = Path(args.root).expanduser().resolve() / namespace / args.pair_id
    path = directory / "pair.json"
    if args.report_only:
        atomic_json(directory / "summary.json", summarize_pair(args.root, read_json(path)))
        return
    plan = make_pair_plan(args)
    if path.exists():
        if read_json(path) != plan:
            raise ValueError("Existing pair differs from requested frozen protocol")
    else:
        atomic_json(path, plan)
    if args.plan_only:
        print(json.dumps(dict(plan=str(path), commands=plan["commands"]), indent=2))
        return
    for command in plan["commands"]:
        if args.resume:
            command = command + ["--resume"]
        code = batch_train(parser().parse_args(command))
        atomic_json(directory / "summary.json", summarize_pair(args.root, plan))
        if code:
            raise SystemExit(code)
