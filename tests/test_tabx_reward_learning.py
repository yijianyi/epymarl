"""Learning-rate and terminal-regression single-factor protocols and objectives."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from tabx_config import configure_bc
from tabx_reporting import read_csv
from tabx_storage import atomic_json, read_json
from tabx_td_loss import terminal_td_objective
from test_tabx_imitation import config
from test_tabx_td_success import teacher_pool


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), True, "3"])
def test_invalid_terminal_weight(weight):
    with pytest.raises(ValueError, match="tabx_td_terminal_weight"):
        configure_bc(config(tabx_td_terminal_weight=weight))


def test_terminal_weights_normalize_both_outcomes_and_ignore_padding():
    # A 3-step winning episode and a 2-step losing episode, plus padded/frame positions.
    error = torch.tensor([[[1.], [2.], [3.], [999.]], [[4.], [5.], [999.], [999.]]], requires_grad=True)
    mask = torch.tensor([[[1.], [1.], [1.], [0.]], [[1.], [1.], [0.], [0.]]])
    terminal = torch.tensor([[[0.], [0.], [1.], [1.]], [[0.], [1.], [1.], [1.]]])
    masked = error*mask
    original = masked.square().sum()/mask.sum()
    baseline, baseline_metrics = terminal_td_objective(masked, mask, terminal, 1., original)
    assert baseline is original and baseline_metrics["weighted_td_loss"] == 11.
    weighted, metrics = terminal_td_objective(masked, mask, terminal, 3., original)
    assert weighted.item() == pytest.approx((1+4+3*9+16+3*25)/(1+1+3+1+3))
    assert metrics["td_terminal_transitions"] == 2 and metrics["td_weight_sum"] == 9
    assert metrics["td_terminal_weight_fraction"] == pytest.approx(6/9)
    weighted.backward()
    torch.testing.assert_close(error.grad, 2*error.detach()*mask*(1+2*terminal)/9)
    assert (error.grad[mask == 0] == 0).all()


@pytest.mark.parametrize("weight", [1., 3.])
def test_weighted_learner_updates_match_manual_objective_and_keep_unweighted_logs(teacher_pool, tmp_path, weight):
    from controllers.basic_controller import BasicMAC
    from learners.q_learner import QLearner
    from tabx_dataset import episodes_to_batch
    from tabx_pipeline import QuietLogger
    from tabx_propagation_diagnostic import evaluate_trajectory_batch
    _, episodes = teacher_pool
    batch = episodes_to_batch(episodes)[0]
    args = NS(**config(tabx_td_terminal_weight=weight, tabx_run_dir=str(tmp_path), tabx_td_diagnostics=True,
        tabx_td_diagnostic_interval=1, tabx_bc_target="teacher_actions"))
    learner = QLearner(BasicMAC(batch.scheme, batch.groups,args),batch.scheme,QuietLogger(),args)
    reference = deepcopy(learner)
    q, targets = evaluate_trajectory_batch(reference.mac, reference.mixer, reference.target_mac, reference.target_mixer, batch,args)
    targets = targets.detach()
    mask = batch["transition_mask"][:, :-1].float()
    factors = 1+(weight-1)*batch["terminated"][:, :-1]
    loss = (((q-targets)*mask).square()*factors).sum()/(mask*factors).sum()
    # BC retains its original objective and independent schedule.
    from tabx_imitation import behavior_cloning_loss
    reference.mac.init_hidden(batch.batch_size)
    logits = torch.stack([reference.mac.forward(batch,t=t) for t in range(batch.max_seq_length)],dim=1)
    bc,_ = behavior_cloning_loss(logits,batch,target="teacher_actions",action_weights=args.tabx_bc_action_weights)
    expected_total = loss+.4*bc
    reference.optimiser.zero_grad()
    expected_total.backward()
    torch.nn.utils.clip_grad_norm_(reference.params,args.grad_norm_clip)
    reference.optimiser.step()
    learner.train(batch,100005,32,demo_batch=batch,bc_weight=.4,td_counts=(3,0,0))
    for actual,expected in zip(learner.params,reference.params):
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-8)
    row = read_csv(tmp_path/"train_metrics.csv")[0]
    assert float(row["weighted_td_loss"]) == pytest.approx(loss.item(),rel=1e-6)
    assert float(row["total_loss"]) == pytest.approx(float(row["weighted_td_loss"])+.4*float(row["bc_loss"]))
    assert float(row["td_terminal_weight"]) == weight
    assert float(row["td_terminal_transitions"]) == 3
    if weight == 1.:
        assert row["weighted_td_loss"] == row["loss"]
    unweighted = (((q-targets)*mask).square()).sum()/mask.sum()
    assert float(row["loss"]) == pytest.approx(unweighted.item(),rel=1e-6)
    record = json.loads((tmp_path/"td_distance_metrics.jsonl").read_text())
    assert record["td_terminal_weight"] == weight
    assert record["groups"]["legacy_demo"]["all"]["td_loss"] == pytest.approx(unweighted.item(),rel=1e-6)


@pytest.mark.skipif(os.environ.get("TABX_REWARD_SMOKE") != "1",reason="Opt in with TABX_REWARD_SMOKE=1")
def test_real_learning_rate_and_terminal_weight_pairs(tmp_path,monkeypatch):
    from tabx_pipeline import parser
    from tabx_gamma_experiment import gamma_pair,make_pair_plan,PAIR_SPECS
    import tabx_scheduler
    from test_tabx_official import tiny_task
    task = tiny_task()
    task["scenario"].update(healths=[[100.],[1.]],attack_damages=[[100.],[0.]],
        sight_angles=[[3.14],[3.14]],attack_cooldowns=[[.1],[.1]])
    bank,demos,output = tmp_path/"bank.json",tmp_path/"demos",tmp_path/"source"
    atomic_json(bank,dict(schema_version="1.0",tasks=[task]))
    environment = dict(os.environ,CUDA_VISIBLE_DEVICES="",JAX_PLATFORMS="cpu",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    def run(command,name):
        log = tmp_path/f"{name}.log"
        with log.open("w") as stream:
            result = subprocess.run([sys.executable,*command],env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        assert result.returncode == 0,log.read_text()[-6000:]
    run(["src/tabx_pipeline.py","collect-expert","--bank",str(bank),"--root",str(demos),"--task","tiny",
        "--episodes","4","--episode-limit","4","--epsilon","0"],"collect")
    run(["src/main.py","--config=qmix_tabx","--env-config=tabx","with","env_args.key=tiny",f"env_args.task_bank={bank}",
        f"tabx_output_root={output}","tabx_run_id=source","seed=7","use_cuda=False","tabx_bc=True",f"tabx_demo_root={demos}",
        "tabx_bc_steps=4","tabx_bc_batch_size=2","tabx_bc_eval_episodes=2","tabx_dagger_episodes=1",
        "env_args.max_episode_steps=4","batch_size=32","buffer_size=64","tabx_demo_decay_steps=192",
        "save_model_interval=128","learner_log_interval=4","t_max=0"],"pretrain")
    source = output/"hp/tiny/7/source/checkpoints/0"
    for name in ("CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name,raising=False)
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name,"1")
    monkeypatch.setenv("JAX_PLATFORMS","cpu")
    monkeypatch.setattr(tabx_scheduler,"probe_gpus",lambda gpus:{gpu:dict(name="CPU-smoke-only") for gpu in gpus})
    original = tabx_scheduler.BatchScheduler.training_command
    def cpu_command(self,job,attempt):
        return ["use_cuda=False" if x=="use_cuda=True" else "tabx_require_cuda=False" if x=="tabx_require_cuda=True" else x
                for x in original(self,job,attempt)]
    monkeypatch.setattr(tabx_scheduler.BatchScheduler,"training_command",cpu_command)
    for command in ("lr-pair","terminal-weight-pair"):
        args = parser().parse_args([command,"--source",str(source),"--bank",str(bank),"--task","tiny","--seed","7",
            "--root",str(tmp_path/command),"--pair-id","test","--budget","240","--eval-workers","1"])
        plan = make_pair_plan(args)
        a,b = plan["training_configs"]
        key = PAIR_SPECS[command][1]
        assert [k for k in a if a[k]!=b[k]] == [key]
        assert a["tabx_td_diagnostics"] and b["tabx_td_diagnostic_interval"] == 50
        assert a["tabx_td_success_fraction"] == b["tabx_td_success_fraction"] == .25
        assert a["batch_size"] == b["batch_size"] == 32
        assert a["target_update_interval_or_tau"] == b["target_update_interval_or_tau"] == 200
        assert a["gamma"] == b["gamma"] == .99
        assert key not in plan["shared_protocol_overrides"]
        if command == "lr-pair":
            assert (a["lr"],b["lr"]) == (.00005,.0001)
            assert a["tabx_td_terminal_weight"] == b["tabx_td_terminal_weight"] == 1.
        else:
            assert a["lr"] == b["lr"] == .00005
            assert (a["tabx_td_terminal_weight"],b["tabx_td_terminal_weight"]) == (1.,3.)
        frozen = deepcopy(args)
        frozen.plan_only = True
        gamma_pair(frozen)
        directory = Path(args.root)/PAIR_SPECS[command][0]/"test"
        sealed = (directory/"pair.json").read_bytes()
        gamma_pair(args)
        report = read_json(directory/"summary.json")
        assert len(report["results"]) == 2
        starts = []
        for config,result in zip((a,b),report["results"]):
            run_dir = Path(result["run_directory"])
            actual = read_json(run_dir/"config.json")
            for key in ("lr","batch_size","gamma","target_update_interval_or_tau","tabx_td_success_fraction",
                        "tabx_td_terminal_weight","tabx_td_diagnostics","tabx_td_diagnostic_interval"):
                assert actual[key] == config[key]
            starts.append(read_json(run_dir/"online_start.json"))
            assert result["terminal_fit_history"] and result["checkpoint_learning_state"]["learner_updates"] >= 50
            assert all(entry["request"]["seeds"]==plan["evaluation_seeds"]
                       for entry in [read_json(path) for path in (run_dir/"evaluations").glob("*.json")])
            for row in read_csv(run_dir/"train_metrics.csv"):
                assert float(row["total_loss"]) == pytest.approx(float(row["weighted_td_loss"])
                    +float(row["bc_weight"])*float(row["bc_loss"] or 0),rel=1e-5)
            assert read_json(run_dir/"bc_summary.json")["bc_updates_this_run"] == 0
        for key in ("agent_parameter_sha256","mixer_parameter_sha256","source_pool_sha256"):
            assert starts[0][key] == starts[1][key]
        refresh = deepcopy(args)
        refresh.report_only = True
        gamma_pair(refresh)
        assert (directory/"pair.json").read_bytes() == sealed
