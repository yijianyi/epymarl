"""Branch-state replay, one-action Q diagnosis and gamma-only warm-start experiments."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tabx_storage import atomic_json, file_sha256, read_json
from tabx_value_diagnostic import discounted_return, compare_return_rankings, continuation, joint_q
from test_tabx_official import tiny_task


def test_discounted_targets_and_ranking_reversal():
    assert discounted_return([1,2,3], .5) == 2.75
    late = [0.] * 376 + [1.]
    result = compare_return_rankings([dict(trajectory="early", rewards=[.1]), dict(trajectory="late", rewards=late)])
    rows = {row["trajectory"]:row for row in result["trajectories"]}
    assert rows["late"]["returns"]["0.99"] == pytest.approx(.99**376)
    assert rows["late"]["returns"]["0.995"] == pytest.approx(.995**376)
    assert rows["late"]["returns"]["0.99"] < .1 < rows["late"]["returns"]["0.995"]
    assert result["comparisons"][0]["reversed_pairs"] == 1


def test_monotone_q_greedy_joint_action_is_maximal():
    from modules.mixers.qmix import QMixer
    from tabx_scheduler import training_config
    config, _ = training_config([])
    config.update(n_agents=10, state_shape=340)
    mixer = QMixer(NS(**config))
    outputs, state = torch.randn(1,10,8), torch.randn(1,340)
    greedy = outputs.argmax(-1).numpy()[0]
    alternative = (greedy + 1) % 8
    with torch.no_grad():
        assert joint_q(mixer, outputs, state, greedy) >= joint_q(mixer, outputs, state, alternative) - 1e-7


def test_branch_restore_keeps_enemy_memory_and_future_rng(tmp_path):
    from envs.tabx_wrapper import TABXWrapper
    bank = tmp_path / "bank.json"
    atomic_json(bank, dict(schema_version="1.0", tasks=[tiny_task()]))
    env = TABXWrapper("tiny", task_bank=bank, max_episode_steps=4)
    env.reset(seed=7)
    action = env.get_avail_actions().argmax(-1)
    env.step(action)
    snapshot = env.capture_state()
    def execute(branch_seed):
        env.restore_state(snapshot, branch_seed=branch_seed)
        records = []
        while not env.ended:
            _, reward, _, _, info = env.step(env.get_avail_actions().argmax(-1))
            records.append((reward, info, env.get_obs(), env.get_state(), np.array(env._episode_key)))
        return records
    for first, second in zip(execute(91), execute(91)):
        assert first[:2] == second[:2]
        for a,b in zip(first[2:], second[2:]):
            np.testing.assert_array_equal(a,b)
    env.restore_state(snapshot)
    assert env.t == 1 and env._episode_index == 0 and not env.ended
    np.testing.assert_array_equal(env._episode_key, snapshot["fields"]["_episode_key"])
    env.close()


def test_continuation_uses_forced_action_as_next_rnn_input():
    from components.batch_schema import build_batch_schema
    from components.episode_buffer import EpisodeBatch
    from test_tabx import ControlledEnv
    env = ControlledEnv(length=3)
    env.reset(seed=1)
    env.last = None
    env.restore_state = lambda snapshot, branch_seed=None: setattr(env, "t", 0)
    schema, groups, preprocess = build_batch_schema(env.get_env_info())
    seen = []
    class MAC:
        hidden_states = None
        def forward(self, batch, t, test_mode=False):
            seen.append((t, batch["actions_onehot"][:,t-1].argmax(-1).clone(), self.hidden_states.clone()))
            self.hidden_states += 1
            return torch.zeros(1,10,8)
    # ControlledEnv omits the real wrapper's terminal convenience fields.
    original = env.step
    def step(actions):
        obs, reward, terminated, truncated, info = original(actions)
        info.update(elimination_win=0, timeout_win=int(terminated))
        return obs, reward, terminated, truncated, info
    env.step = step
    runner = NS(env=env, mac=MAC(), new_batch=lambda: EpisodeBatch(schema, groups, 1,4,preprocess=preprocess))
    hidden = torch.full((1,10,2), 7.)
    forced = np.full(10,5)
    forced[-1] = 7
    result = continuation(runner, {}, hidden, forced, 3, [.99,.995])
    assert result["remaining_steps"] == 3
    assert seen[0][0] == 1 and (seen[0][1][:,:9] == 5).all() and (seen[0][2] == 7).all()
    assert (seen[1][1][:,:9] == 0).all() and seen[1][1][0,-1] == 7 and (seen[1][2] == 8).all()


@pytest.fixture(scope="module")
def pretrained_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("value-source")
    bank, demos, output = root / "bank.json", root / "demos", root / "output"
    atomic_json(bank, dict(schema_version="1.0", tasks=[tiny_task()]))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    def run(command, name):
        log = root / f"{name}.log"
        with log.open("w") as stream:
            result = subprocess.run([sys.executable,*command], env=environment, stdout=stream, stderr=subprocess.STDOUT,timeout=180)
        assert result.returncode == 0, log.read_text()[-6000:]
    run(["src/tabx_pipeline.py","collect-expert","--bank",str(bank),"--root",str(demos),"--task","tiny",
         "--episodes","2","--episode-limit","2"], "collect")
    run(["src/main.py","--config=qmix_tabx","--env-config=tabx","with","env_args.key=tiny",f"env_args.task_bank={bank}",
        f"tabx_output_root={output}","tabx_run_id=source","seed=7","use_cuda=False","tabx_bc=True",f"tabx_demo_root={demos}",
        "tabx_bc_steps=4","tabx_bc_batch_size=2","tabx_bc_eval_episodes=2","tabx_bc_require_wins=False","tabx_dagger_episodes=1",
        "env_args.max_episode_steps=2","batch_size=2","buffer_size=4","tabx_demo_fraction=0.5","t_max=0"],"pretrain")
    return NS(root=root,bank=bank,demos=demos,source=output/"hp/tiny/7/source/checkpoints/0",env=environment)


@pytest.mark.skipif(os.environ.get("TABX_VALUE_SMOKE") != "1", reason="Opt in with TABX_VALUE_SMOKE=1")
def test_source_sealing_restoration_and_gamma_pair_plan(pretrained_source,tmp_path):
    from tabx_pipeline import parser
    from tabx_gamma_experiment import make_pair_plan
    from tabx_online_start import configure_online_start, pool_fingerprint
    from tabx_scheduler import training_config
    source = pretrained_source
    args = parser().parse_args(["gamma-pair","--source",str(source.source),"--bank",str(source.bank),"--task","tiny",
        "--seed","7","--root",str(tmp_path),"--pair-id","test","--budget","6","--plan-only"])
    plan = make_pair_plan(args)
    first,second=plan["training_configs"]
    assert [k for k in first if first[k]!=second[k]] == ["gamma"]
    assert (first["gamma"],second["gamma"]) == (.99,.995)
    assert first["tabx_source_agent_sha256"] == second["tabx_source_agent_sha256"]
    assert first["tabx_source_mixer_sha256"] == second["tabx_source_mixer_sha256"]
    assert len(plan["evaluation_seeds"]) == len(set(plan["evaluation_seeds"])) == 128
    assert first["tabx_screen_episodes"] == first["tabx_bc_eval_episodes"] == 128
    assert first["tabx_eval_seed_phase"] == "bc_holdout" and first["t_max"] == 6
    assert first["tabx_bc_steps"] == 4 and first["batch_size"] == 2
    copied = tmp_path / "copy/checkpoints/0"
    shutil.copytree(source.source, copied)
    config,_=training_config([])
    config.update(tabx_bc=True,tabx_online_source=str(copied))
    with (copied / "mixer.th").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError,match="mixer.th"):
        configure_online_start(config)
    assert pool_fingerprint(source.source,source.demos) == plan["source_pool_sha256"]


@pytest.mark.skipif(os.environ.get("TABX_VALUE_SMOKE") != "1", reason="Opt in with TABX_VALUE_SMOKE=1")
def test_real_gamma_pair_and_value_diagnostic(pretrained_source,tmp_path,monkeypatch):
    from tabx_pipeline import parser
    import tabx_scheduler
    from tabx_gamma_experiment import gamma_pair
    from tabx_value_diagnostic import diagnose_value
    source=pretrained_source
    for name in ("CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name,raising=False)
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name,"1")
    monkeypatch.setenv("JAX_PLATFORMS","cpu")
    monkeypatch.setattr(tabx_scheduler,"probe_gpus",lambda gpus:{gpu:dict(name="CPU-smoke-only") for gpu in gpus})
    original=tabx_scheduler.BatchScheduler.training_command
    def cpu_command(self,job,attempt):
        return ["use_cuda=False" if x=="use_cuda=True" else "tabx_require_cuda=False" if x=="tabx_require_cuda=True" else x
                for x in original(self,job,attempt)]
    monkeypatch.setattr(tabx_scheduler.BatchScheduler,"training_command",cpu_command)
    args=parser().parse_args(["gamma-pair","--source",str(source.source),"--bank",str(source.bank),"--task","tiny","--seed","7",
        "--root",str(tmp_path/"pair"),"--pair-id","test","--budget","6","--eval-workers","1"])
    gamma_pair(args)
    report=read_json(tmp_path/"pair/_gamma_pairs/test/summary.json")
    starts=[]
    for suffix,result in zip(("g099","g0995"),report["results"]):
        assert result["cutoff_step"]==6 and result["budget_reached"]
        run=Path(result["run_directory"])
        starts.append(read_json(run/"online_start.json"))
        assert not (run/"bc_metrics.csv").exists() and not (run/"dagger").exists()
        summary=read_json(run/"bc_summary.json")
        assert summary["bc_updates_this_run"]==0 and summary["dagger_episodes"]==3
        assert starts[-1]["restored_dagger_rounds"]==3 and starts[-1]["optimizer"]=="fresh"
        assert read_json(run/"training_summary.json")["learner_updates"]>0
        assert (run/"online_bc_metrics.jsonl").exists()
        records=[json.loads(line) for line in (run/"training_episodes.jsonl").read_text().splitlines()]
        assert [row["episode_index"] for row in records]==[0,1,2]
        evaluations=[read_json(path) for path in (run/"evaluations").glob("*.json")]
        assert evaluations and all(len(row["episodes"])==128 for row in evaluations)
        plan=read_json(tmp_path/"pair/_gamma_pairs/test/pair.json")
        assert all(row["request"]["seeds"]==plan["evaluation_seeds"] for row in evaluations)
        assert read_json(run/"bc_evaluations/after_bc_online_step_0.json")["seeds"]==plan["evaluation_seeds"]
    for key in ("agent_parameter_sha256","mixer_parameter_sha256","source_pool_sha256"):
        assert starts[0][key]==starts[1][key]
    from tabx_pipeline import build_rollout
    from tabx_online_start import parameter_hash
    from modules.mixers.qmix import QMixer
    original_runner=build_rollout(source.source,source.bank,"hp",compatible_source=True)
    expected_mixer=QMixer(original_runner.args)
    expected_mixer.load_state_dict(torch.load(source.source/"mixer.th",weights_only=True))
    assert starts[0]["agent_parameter_sha256"]==parameter_hash(original_runner.mac.agent)
    assert starts[0]["mixer_parameter_sha256"]==parameter_hash(expected_mixer)
    original_runner.close_env()
    final=Path(report["results"][1]["run_directory"])/"checkpoints/6"
    diagnostic=parser().parse_args(["diagnose-value","--snapshot",str(final),"--bank",str(source.bank),"--output",str(tmp_path/"diagnostic"),
        "--states","2","--episodes","2","--repeats","2","--spacing","1","--include-search-turns",
        "--trajectories",str(source.demos/"hp/tiny/advanced")])
    diagnose_value(diagnostic)
    d=read_json(tmp_path/"diagnostic/report.json")
    assert d["sampled_states"]==2 and d["trained_gamma"]==.995
    assert len(d["trajectory_rankings"]["trajectories"])==4
    for row in d["states"]:
        a,b=row["model_rollouts"],row["teacher_rollouts"]
        assert [x["seed"] for x in a]==[x["seed"] for x in b]
        if row["model_actions"]==row["teacher_actions"]:
            assert [x["rewards"] for x in a]==[x["rewards"] for x in b]
        assert "0.99" in row["paired_returns"] and "0.995" in row["paired_returns"]


@pytest.mark.skipif(os.environ.get("TABX_VALUE_SMOKE") != "1", reason="Opt in with TABX_VALUE_SMOKE=1")
def test_restored_targets_optimizer_and_aggregated_sampling(pretrained_source,tmp_path):
    from tabx_pipeline import build_rollout
    from tabx_online_start import configure_online_start, initialize_online, pool_fingerprint
    from tabx_imitation import DemonstrationPool
    from learners.q_learner import QLearner
    from tabx_pipeline import QuietLogger
    source=pretrained_source
    runner=build_rollout(source.source,source.bank,"hp",compatible_source=True)
    config=vars(runner.args).copy()
    config.update(tabx_online_source=str(source.source),tabx_run_dir=str(tmp_path),tabx_bc_eval_episodes=128)
    configure_online_start(config)
    args=NS(**config)
    learner=QLearner(runner.mac,runner.new_batch().scheme,QuietLogger(),args)
    original=DemonstrationPool(source.demos/"hp/tiny/advanced",runner.env,7,require_wins=False,target="teacher_actions")
    pool=initialize_online(learner,runner,original,args)
    assert len(pool.rounds)==3 and learner.training_steps==runner.t_env==runner.env._episode_index==0
    assert not learner.optimiser.state
    for live,target in zip(learner.mac.parameters(),learner.target_mac.parameters()):
        torch.testing.assert_close(live,target,rtol=0,atol=0)
    for live,target in zip(learner.mixer.parameters(),learner.target_mixer.parameters()):
        torch.testing.assert_close(live,target,rtol=0,atol=0)
    batch=pool.sample(32)
    assert batch["has_teacher_labels"].sum()==32 and batch["teacher_mask"].sum()>0
    copied_demos=tmp_path/"copied-demos"
    shutil.copytree(source.demos,copied_demos)
    manifest=read_json(copied_demos/"hp/tiny/advanced/manifest.json")
    with (copied_demos/"hp/tiny/advanced"/manifest["shards"][0]["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError,match="checksum"):
        pool_fingerprint(source.source,copied_demos)
    runner.close_env()
