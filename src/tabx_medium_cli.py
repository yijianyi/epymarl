"""Command line settings for the five parallel MAPPO Medium entry points."""
import argparse
import math
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DEMO_ROOT = "results/tabx-demos-hp-medium-vs-medium"
TRAIN_ROOT = "results/tabx-mappo-hp-all-medium"
BATCH_NAMES = dict(A="mappo-baseline", B="mappo-bc-medium", C="mappo-mix", D="mappo-mix-warmup")
DEFAULT_COLLECTION_ID = "default"


def positive_int(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def positive_float(raw):
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def repo_path(raw):
    return str((REPOSITORY / Path(raw).expanduser()).resolve())


def parser(group):
    description = {None: "Collect official medium vs medium demonstrations",
                   "coach": "Train and select B/C/D MAPPO coaches against Medium"}.get(group, f"Run MAPPO Medium group {group}")
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bank", required=True, help="TABX task bank JSON")
    parser.add_argument("--tasks", required=True, help="Comma-separated task IDs")
    parser.add_argument("--bank-sha256", help="Expected task bank SHA-256")
    parser.add_argument("--rule", choices=("hp", "loss"), default="hp", help="Timeout rule")
    parser.add_argument("--episode-limit", type=positive_int, default=512, help="Maximum steps per episode, up to 512")
    parser.add_argument("--poll-seconds", type=positive_float, default=1.0, help="Scheduler polling interval")
    parser.add_argument("--root", type=repo_path, default=DEMO_ROOT if group is None else TRAIN_ROOT,
                        help="Output directory, resolved relative to the repository")
    if group in (None, "B", "coach"):
        parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID,
                            help="Collection version shared by the collector and group B")
    if group is None:
        parser.add_argument("--workers", type=positive_int, default=4, help="Concurrent CPU collection tasks")
        parser.add_argument("--episodes", type=positive_int, default=128, help="Minimum episodes per task")
        parser.add_argument("--seed", type=int, default=0, help="Demonstration master seed")
        parser.add_argument("--threads", type=positive_int, default=1, help="CPU threads per collection process")
        parser.add_argument("--shard-episodes", type=positive_int, default=8, help="Maximum new episodes per HDF5 shard")
        return parser
    parser.add_argument("--gpus", required=True, help="Comma-separated external GPU IDs, shared by A through D")
    parser.add_argument("--seeds", default="1,2,3", help="Comma-separated training seeds")
    parser.add_argument("--jobs-per-gpu", type=positive_int, default=12, help="Maximum training processes per GPU across all groups")
    parser.add_argument("--t-max", type=positive_int, default=300000, help="Online environment step budget per training job")
    parser.add_argument("--batch-size", type=positive_int, default=8, help="Episodes per PPO batch and buffer")
    parser.add_argument("--eval-workers", type=positive_int, default=1, help="Concurrent CPU evaluation workers for this group")
    parser.add_argument("--screen-episodes", type=positive_int, default=32, help="Greedy evaluation episodes per checkpoint")
    parser.add_argument("--save-model-interval", type=positive_int, default=25000, help="Environment step interval for checkpoint publication")
    parser.add_argument("--status-interval", type=positive_float, default=30.0, help="Seconds between progress messages")
    parser.add_argument("--batch-id", default=BATCH_NAMES.get(group, "coach-medium-v1"),
                        help="Batch name within the output root and rule")
    parser.add_argument("--resume", action="store_true", help="Reuse completed work and retry interrupted or failed jobs")
    if group in ("B", "coach"):
        parser.add_argument("--demo-root", type=repo_path, default=DEMO_ROOT, help="Demonstration output directory")
        parser.add_argument("--bc-steps", type=positive_int, default=1000, help="Actor behavior cloning updates")
        parser.add_argument("--bc-batch-size", type=positive_int, default=32, help="Demonstrated episodes per BC update")
        parser.add_argument("--bc-lr", type=positive_float, default=0.0005, help="Behavior cloning learning rate")
    if group in ("C", "D"):
        parser.add_argument("--source-batch-id", default=BATCH_NAMES["B"], help="B batch in the same output root and rule")
    if group in ("C", "D", "coach"):
        parser.add_argument("--mix-start", type=float, default=0.5, help="Initial medium control probability")
        parser.add_argument("--mix-decay-steps", type=positive_int, default=50000, help="Online step at which teacher control reaches zero")
        parser.add_argument("--mix-block-steps", type=positive_int, default=8, help="Environment steps between team control draws")
    if group in ("D", "coach"):
        parser.add_argument("--temp-peak", type=float, default=1.2, help="Peak MAPPO action temperature")
        parser.add_argument("--temp-peak-step", type=positive_int, default=20000, help="Online step at peak temperature")
        parser.add_argument("--temp-end-step", type=positive_int, default=100000, help="Online step at which temperature returns to 1")
    if group == "coach":
        parser.set_defaults(root="results/tabx-mappo-medium-coach", seeds="1", t_max=3000000,
                            batch_size=32, bc_steps=10000, eval_workers=4)
        parser.add_argument("--confirm-episodes", type=positive_int, default=128,
                            help="Paired held-out evaluation episodes per candidate and baseline")
        parser.add_argument("--log-root", type=repo_path, help="Separate root for scheduler, training and evaluation logs")
    return parser


def training_args(group, args):
    from tabx_pipeline import parser as pipeline_parser
    from tabx_storage import safe_component

    pairs = dict(bank=args.bank, tasks=args.tasks, rule=args.rule, root=args.root, alg="mappo_tabx", gpus=args.gpus,
                 seeds=args.seeds, jobs_per_gpu=args.jobs_per_gpu, eval_workers=args.eval_workers,
                 batch_id=args.batch_id, poll_seconds=args.poll_seconds, status_interval=args.status_interval)
    if args.bank_sha256:
        pairs["bank_sha256"] = args.bank_sha256
    argv = ["batch-train", *(f"--{key.replace('_', '-')}={value}" for key, value in pairs.items())]
    if args.resume:
        argv.append("--resume")
    settings = dict(t_max=args.t_max, batch_size=args.batch_size, buffer_size=args.batch_size,
                    tabx_screen_episodes=args.screen_episodes, save_model_interval=args.save_model_interval)
    settings["env_args.max_episode_steps"] = args.episode_limit
    if group == "B":
        settings.update(tabx_bc=True, tabx_demo_root=args.demo_root, tabx_bc_steps=args.bc_steps,
                        tabx_bc_batch_size=args.bc_batch_size, tabx_bc_lr=args.bc_lr)
    if group in ("C", "D"):
        source = safe_component(args.source_batch_id)
        if source == args.batch_id:
            raise ValueError("--source-batch-id must be different from --batch-id")
        settings.update(tabx_bc=True, tabx_online_source=str(Path(args.root) / "_batches" / args.rule / source),
                        tabx_mix_start=args.mix_start, tabx_mix_decay_steps=args.mix_decay_steps,
                        tabx_mix_block_steps=args.mix_block_steps)
    if group == "D":
        settings.update(tabx_mappo_temp_peak=args.temp_peak, tabx_mappo_temp_peak_step=args.temp_peak_step,
                        tabx_mappo_temp_end_step=args.temp_end_step)
    for key, value in settings.items():
        # repr preserves string paths through the shared --set parser.
        argv.extend(["--set", f"{key}={value!r}"])
    options = pipeline_parser().parse_args(argv)
    options.medium_group = group
    if group == "B":
        options.collection_id = safe_component(args.collection_id)
    return options


def coach_train(args):
    from tabx_coach_scheduler import CoachScheduler, coach_jobs
    from tabx_scheduler import batch_train
    from tabx_storage import safe_component

    options = training_args("B", args)
    options.medium_group = None
    options.log_root = args.log_root
    safe_component(args.batch_id)
    source = str(Path(args.root) / "_batches" / args.rule / args.batch_id)
    mix = [f"tabx_online_source={source!r}", f"tabx_mix_start={args.mix_start!r}",
           f"tabx_mix_decay_steps={args.mix_decay_steps}", f"tabx_mix_block_steps={args.mix_block_steps}"]
    warmup = [f"tabx_mappo_temp_peak={args.temp_peak!r}", f"tabx_mappo_temp_peak_step={args.temp_peak_step}",
              f"tabx_mappo_temp_end_step={args.temp_end_step}"]
    return batch_train(options, scheduler_class=CoachScheduler, job_specs=coach_jobs,
                       job_overrides={"C": mix, "D": mix + warmup},
                       identity_fields=dict(confirm_episodes=args.confirm_episodes, log_root=args.log_root))


def main(group=None, argv=None):
    cli = parser(group)
    args = cli.parse_args(argv)
    if args.episode_limit > 512:
        cli.error("--episode-limit must be at most 512")
    if group is None:
        if not 0 <= args.seed < 2**32:
            cli.error("--seed must be a uint32")
        from tabx_medium_collection import collect_parallel
        args.pilot = args.task_list = None
        return collect_parallel(args)
    if group == "coach":
        return coach_train(args)
    from tabx_medium_scheduler import MediumScheduler
    from tabx_scheduler import batch_train
    return batch_train(training_args(group, args), scheduler_class=MediumScheduler)
