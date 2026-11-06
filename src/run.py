import datetime
import json
import os
from os.path import dirname, abspath
import pprint
import shutil
import time
import threading
from types import SimpleNamespace as SN

import torch as th

from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.batch_schema import build_batch_schema
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.general_reward_support import test_alg_config_supports_reward
from utils.logging import Logger
from utils.timehelper import time_left, time_str


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    if args.env == "tabx":
        if args.runner != "episode" or args.batch_size_run != 1:
            raise ValueError("TABX currently requires runner=episode and batch_size_run=1")
        if getattr(args, "tabx_async_eval", False) and not args.save_model:
            raise ValueError("TABX asynchronous evaluation requires save_model=True")
        if getattr(args, "tabx_async_eval", False) and (args.tabx_screen_episodes < 2 or args.tabx_max_pending_evals < 1):
            raise ValueError("TABX requires tabx_screen_episodes>=2 and tabx_max_pending_evals>=1")
        import logging
        logging.getLogger("jax").setLevel(logging.WARNING)
        logging.getLogger("jaxlib").setLevel(logging.WARNING)
        from envs.tabx_tasks import import_tabx, load_task, validate_task_structure
        from envs.tabx_layout import TABXLayout
        from tabx_storage import create_run
        import_tabx(args.env_args.get("tabx_root"))
        task, task_manifest = load_task(args.env_args.get("task_bank"), args.env_args["key"],
                                       args.env_args.get("expected_task_hash"), args.env_args.get("expected_bank_hash"))
        task_manifest.update(TABXLayout(*validate_task_structure(task)).metadata())
        _config["device"] = args.device
        run_dir = create_run(_config, task_manifest)
        import jax
        jax.config.update("jax_compilation_cache_dir", os.environ.get(
            "JAX_COMPILATION_CACHE_DIR", str(run_dir.parents[3] / "_jax_cache")))
        _config["env_args"]["seed"] = int(args.seed)
        from tabx_storage import atomic_json
        atomic_json(run_dir / "config.json", _config)
        args = SN(**_config)
    assert test_alg_config_supports_reward(
        args
    ), "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    unique_token = (
        f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"
    )

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(args.local_results_path, "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    if args.use_wandb:
        logger.setup_wandb(
            _config, args.wandb_team, args.wandb_project, args.wandb_mode
        )

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Finish logging
    logger.finish()

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    scheme, groups, preprocess = build_batch_schema(env_info, args.common_reward)

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    async_tabx = getattr(args, "tabx_async_eval", False)
    if async_tabx and (args.env != "tabx" or args.runner != "episode" or args.batch_size_run != 1 or not args.save_model):
        raise ValueError("tabx_async_eval requires TABX, episode runner, batch_size_run=1 and save_model=True")

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Accept either a concrete snapshot or an existing numbered parent.
        if os.path.isfile(os.path.join(args.checkpoint_path, "agent.th")):
            model_path = args.checkpoint_path
            from pathlib import Path
            timestep_to_load = int(Path(model_path).name) if Path(model_path).name.isdigit() else 0
        else:
            model_path = None
        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if model_path is None and not timesteps:
            raise ValueError(f"No checkpoints found in {args.checkpoint_path}")
        if model_path is None and args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        elif model_path is None:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = model_path or os.path.join(args.checkpoint_path, str(timestep_to_load))

        if args.env == "tabx":
            from tabx_storage import validate_snapshot
            validate_snapshot(model_path, runner.env.manifest, runner.env.rule)

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    publisher = None
    if async_tabx:
        from tabx_storage import SnapshotPublisher
        publisher = SnapshotPublisher(args, runner.env)
        # Restored weights start a NEW attempt. Never label them random step 0.
        if not args.checkpoint_path:
            publisher.publish(learner, 0, force_eval=True)
        else:
            publisher.next_save = (runner.t_env // publisher.interval + 1) * publisher.interval

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env < args.t_max or (not async_tabx and runner.t_env == args.t_max):
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        if args.env == "tabx":
            from pathlib import Path
            record = dict(episode_index=runner.env.last_train_episode_index,
                          episode_seed=runner.env.last_episode_seed, t_env=runner.t_env,
                          **runner.last_episode_info)
            with (Path(args.tabx_run_dir) / "training_episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            if async_tabx:
                logger.log_stat("learner_updates", learner.training_steps, runner.t_env)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if not async_tabx and (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if publisher is not None and runner.t_env >= publisher.next_save:
            publisher.publish(learner, runner.t_env)

        if not async_tabx and args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

            if args.use_wandb and args.wandb_save_model:
                wandb_save_dir = os.path.join(
                    logger.wandb.dir, "models", args.unique_token, str(runner.t_env)
                )
                os.makedirs(wandb_save_dir, exist_ok=True)
                for f in os.listdir(save_path):
                    shutil.copyfile(
                        os.path.join(save_path, f), os.path.join(wandb_save_dir, f)
                    )

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    if publisher is not None:
        import resource
        publisher.complete(learner, runner.t_env, dict(
            t_env=runner.t_env, learner_updates=learner.training_steps, episodes=episode,
            elapsed_seconds=time.time() - start_time,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_cuda_bytes=th.cuda.max_memory_allocated() if args.use_cuda else 0))
    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
